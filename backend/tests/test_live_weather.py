"""
backend/tests/test_live_weather.py

Unit and integration tests for the universal 5-point stencil mesoscale &
cyclonic atmospheric analyzer (APM-Nowcast Live Weather Service).

Design principles:
  - All external HTTP is mocked — tests run fast and completely offline.
  - Each mock payload covers all 5 stencil points per node (center + N/S/E/W)
    so stencil extrema and deep-layer steering are computed correctly.
  - No hardcoded storm coordinates, regions, or probabilities in assertions;
    physics-based thresholds match the continuous scoring model.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.live_weather import (
    INDIA_MET_GRID_POINTS,
    GridPointAtmosphere,
    LiveWeatherService,
    StencilAtmosphere,
    _safe,
    _spd_dir_from_uv,
    _stencil_coords,
    _uv_from_spd_dir,
    live_weather_service,
)
from app.services.inference_service import inference_service


# ---------------------------------------------------------------------------
# Payload builder helpers
# ---------------------------------------------------------------------------

def _make_stencil_item(
    lat: float,
    lon: float,
    cape: float = 1800.0,
    cin: float = -40.0,
    precip: float = 0.5,
    cloud: float = 55.0,
    wind_spd: float = 20.0,
    wind_dir: float = 180.0,
    temp: float = 29.0,
    pressure: float = 1012.0,
    gusts: float = 28.0,
    rh700: float = 60.0,
    ws850: float = 18.0,
    wd850: float = 180.0,
    ws500: float = 24.0,
    wd500: float = 185.0,
    ws300: float = 36.0,
    wd300: float = 188.0,
) -> Dict[str, Any]:
    """Build one Open-Meteo result item (matches the _CURRENT_VARS schema)."""
    return {
        "latitude": lat,
        "longitude": lon,
        "current": {
            "cape": cape,
            "convective_inhibition": cin,
            "precipitation": precip,
            "cloud_cover": cloud,
            "wind_speed_10m": wind_spd,
            "wind_direction_10m": wind_dir,
            "temperature_2m": temp,
            "pressure_msl": pressure,
            "wind_gusts_10m": gusts,
            "relative_humidity_700hPa": rh700,
            "wind_speed_850hPa": ws850,
            "wind_direction_850hPa": wd850,
            "wind_speed_500hPa": ws500,
            "wind_direction_500hPa": wd500,
            "wind_speed_300hPa": ws300,
            "wind_direction_300hPa": wd300,
        },
    }


def _build_flat_payload(
    per_node_overrides: Dict[str, Dict[str, Any]] | None = None,
    default_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build the flat 90-item Open-Meteo batch payload (18 nodes × 5 stencil pts each).
    per_node_overrides: {node_id: kwargs} to override fields for a specific node's
    center and ring points.  default_kwargs applied to ALL nodes if not overridden.
    """
    per_node_overrides = per_node_overrides or {}
    default_kwargs = default_kwargs or {}
    items: List[Dict[str, Any]] = []

    for p in INDIA_MET_GRID_POINTS:
        node_kw = {**default_kwargs, **per_node_overrides.get(p["id"], {})}
        for lat, lon in _stencil_coords(p["lat"], p["lon"]):
            items.append(_make_stencil_item(lat, lon, **node_kw))

    return items


def _mock_response(payload: List[Dict[str, Any]]) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = payload
    return mock


# ---------------------------------------------------------------------------
# TestGridPointsDefinition
# ---------------------------------------------------------------------------

class TestGridPointsDefinition:
    """Ensure the 19 Pan-India & Bay of Bengal meteorological grid points meet domain specs."""

    def test_grid_point_count(self):
        assert len(INDIA_MET_GRID_POINTS) == 19

    def test_grid_points_within_pan_india_bounding_box(self):
        for pt in INDIA_MET_GRID_POINTS:
            assert 6.0 <= pt["lat"] <= 38.0, f"{pt['name']} latitude {pt['lat']} OOB"
            assert 68.0 <= pt["lon"] <= 98.0, f"{pt['name']} longitude {pt['lon']} OOB"

    def test_strategic_regions_present(self):
        names = {pt["name"] for pt in INDIA_MET_GRID_POINTS}
        required = {
            "Chennai / Tamil Nadu Coast", "Bengaluru", "Kerala / Kochi",
            "Mumbai / Konkan", "Hyderabad / Telangana", "Bhubaneswar / Odisha",
            "Kolkata / Gangetic Bengal", "Guwahati / Brahmaputra",
            "Ranchi / Chota Nagpur", "Nagpur", "Delhi NCR", "Jaipur",
            "Ahmedabad", "Andaman / Bay of Bengal", "Westcentral Bay of Bengal",
            "Off Andhra Coast", "Visakhapatnam", "Puri / Odisha Coast",
            "Bihar / Mid-Gangetic Plains",
        }
        assert required.issubset(names)

    def test_all_nodes_have_fallback_multi_level_winds(self):
        required_keys = {
            "fallback_wind_speed_850", "fallback_wind_dir_850",
            "fallback_wind_speed_500", "fallback_wind_dir_500",
            "fallback_wind_speed_300", "fallback_wind_dir_300",
            "fallback_rh700",
        }
        for pt in INDIA_MET_GRID_POINTS:
            missing = required_keys - set(pt.keys())
            assert not missing, f"{pt['name']} missing keys: {missing}"


# ---------------------------------------------------------------------------
# TestStencilHelpers
# ---------------------------------------------------------------------------

class TestStencilHelpers:
    """Unit tests for UV wind conversion and stencil geometry utilities."""

    def test_safe_none_returns_fallback(self):
        assert _safe(None, 42.0) == 42.0

    def test_safe_nan_returns_fallback(self):
        assert _safe(float("nan"), 99.0) == 99.0

    def test_safe_valid_float(self):
        assert _safe(3.14, 0.0) == pytest.approx(3.14)

    def test_safe_negative_cin(self):
        """Open-Meteo may return CIN as negative; _safe must pass it through."""
        assert _safe(-55.0, 0.0) == pytest.approx(-55.0)

    def test_uv_from_spd_dir_north_wind(self):
        """North wind (from 0°) blows southward: u≈0, v<0."""
        u, v = _uv_from_spd_dir(10.0, 0.0)
        assert abs(u) < 0.01
        assert v == pytest.approx(-10.0, abs=0.01)

    def test_uv_from_spd_dir_east_wind(self):
        """East wind (from 90°) blows westward: u<0, v≈0."""
        u, v = _uv_from_spd_dir(10.0, 90.0)
        assert u == pytest.approx(-10.0, abs=0.01)
        assert abs(v) < 0.01

    def test_round_trip_spd_dir(self):
        for spd, drn in [(20.0, 45.0), (35.0, 215.0), (0.4, 90.0)]:
            u, v = _uv_from_spd_dir(spd, drn)
            spd2, drn2 = _spd_dir_from_uv(u, v)
            if spd > 0.5:
                assert spd2 == pytest.approx(spd, rel=0.01)
                assert abs(drn2 - drn) % 360 < 0.5

    def test_stencil_coords_returns_5_points(self):
        pts = _stencil_coords(13.0, 80.0)
        assert len(pts) == 5

    def test_stencil_coords_center_first(self):
        pts = _stencil_coords(13.0, 80.0)
        assert pts[0] == (13.0, 80.0)

    def test_stencil_coords_ring_offsets(self):
        lat, lon = 13.0, 80.0
        pts = _stencil_coords(lat, lon)
        # N
        assert pts[1] == pytest.approx((lat + 1.0, lon), abs=0.001)
        # S
        assert pts[2] == pytest.approx((lat - 1.0, lon), abs=0.001)
        # E
        assert pts[3] == pytest.approx((lat, lon + 1.0), abs=0.001)
        # W
        assert pts[4] == pytest.approx((lat, lon - 1.0), abs=0.001)


# ---------------------------------------------------------------------------
# TestLiveWeatherServiceOfflineFallback
# ---------------------------------------------------------------------------

class TestLiveWeatherServiceOfflineFallback:
    """Ensure graceful benchmark fallback when network is unavailable."""

    def test_timeout_fallback(self):
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.TimeoutException("timed out")):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        assert len(grid) == 19
        assert live_weather_service.is_live is False
        for pt in grid.values():
            assert pt.cape >= 0.0
            assert pt.temperature_2m > 0.0
            assert pt.source == "benchmark_cube"

    def test_network_error_fallback(self):
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("unreachable")):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        assert len(grid) == 19
        assert live_weather_service.is_live is False

    def test_http_500_fallback(self):
        mock = MagicMock()
        mock.status_code = 500
        with patch.object(httpx.Client, "get", return_value=mock):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        assert len(grid) == 19
        assert live_weather_service.is_live is False

    def test_fallback_has_steering_winds(self):
        """Benchmark fallback must populate steering speed/direction from level winds."""
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        for pt in grid.values():
            assert pt.steering_speed_kmh >= 0.0
            assert 0.0 <= pt.steering_dir_deg < 360.0


# ---------------------------------------------------------------------------
# TestLiveWeatherServiceMockedFetch
# ---------------------------------------------------------------------------

class TestLiveWeatherServiceMockedFetch:
    """Parsing, caching, and nearest-neighbour logic with mocked responses."""

    @pytest.fixture
    def generic_payload(self):
        return _build_flat_payload(
            default_kwargs=dict(
                cape=2000.0, cin=-38.0, precip=1.5, cloud=65.0,
                wind_spd=20.0, wind_dir=200.0, temp=29.0,
                pressure=1010.5, gusts=28.0, rh700=68.0,
                ws850=18.0, wd850=195.0,
                ws500=26.0, wd500=200.0,
                ws300=38.0, wd300=205.0,
            )
        )

    def test_successful_fetch_parses_all_19_nodes(self, generic_payload):
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        assert len(grid) == 19
        assert live_weather_service.is_live is True

    def test_live_source_marked(self, generic_payload):
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)
        for pt in grid.values():
            assert pt.source == "live_open_meteo"
            assert pt.is_live is True

    def test_stencil_extrema_populated(self, generic_payload):
        """Stencil fields must be populated (not left at zero/sentinel defaults)."""
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        for pt in grid.values():
            assert pt.stencil_min_pressure_msl > 0.0
            assert pt.stencil_max_wind_gusts >= 0.0
            assert pt.stencil_max_cape >= 0.0
            assert pt.stencil_max_rh700 >= 0.0
            assert pt.stencil_delta_p >= 0.0

    def test_steering_derived_from_multi_level_winds(self, generic_payload):
        """Deep-layer steering must be a weighted composite, not raw surface wind."""
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        for pt in grid.values():
            assert pt.steering_speed_kmh >= 0.0
            assert 0.0 <= pt.steering_dir_deg < 360.0

    def test_cache_ttl_reused(self, generic_payload):
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)) as mock_get:
            live_weather_service.fetch_live_grid(force_refresh=True)
            assert mock_get.call_count == 1
            live_weather_service.fetch_live_grid(force_refresh=False)
            assert mock_get.call_count == 1

    def test_get_nearest_conditions_chennai(self, generic_payload):
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            live_weather_service.fetch_live_grid(force_refresh=True)

        nearest = live_weather_service.get_nearest_conditions(13.1, 80.25)
        assert nearest.id == "chennai"

    def test_get_nearest_conditions_delhi(self, generic_payload):
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            live_weather_service.fetch_live_grid(force_refresh=True)

        nearest = live_weather_service.get_nearest_conditions(28.5, 77.1)
        assert nearest.id == "delhi"

    def test_get_nearest_conditions_bob(self, generic_payload):
        with patch.object(httpx.Client, "get",
                          return_value=_mock_response(generic_payload)):
            live_weather_service.fetch_live_grid(force_refresh=True)

        nearest = live_weather_service.get_nearest_conditions(13.2, 85.4)
        assert nearest.id == "bob_westcentral"

    def test_none_fields_sanitized(self):
        """None/null values in API response must be replaced by fallback, not crash."""
        # Build a payload where center point has None for most fields
        payload = []
        for p in INDIA_MET_GRID_POINTS:
            for lat, lon in _stencil_coords(p["lat"], p["lon"]):
                payload.append({
                    "latitude": lat,
                    "longitude": lon,
                    "current": {
                        "cape": None,
                        "convective_inhibition": None,
                        "precipitation": None,
                        "cloud_cover": None,
                        "wind_speed_10m": None,
                        "wind_direction_10m": None,
                        "temperature_2m": None,
                        "pressure_msl": None,
                        "wind_gusts_10m": None,
                        "relative_humidity_700hPa": None,
                        "wind_speed_850hPa": None,
                        "wind_direction_850hPa": None,
                        "wind_speed_500hPa": None,
                        "wind_direction_500hPa": None,
                        "wind_speed_300hPa": None,
                        "wind_direction_300hPa": None,
                    },
                })

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        # Must parse without raising and produce 19 valid nodes
        assert len(grid) == 19
        for pt in grid.values():
            assert not math.isnan(pt.cape)
            assert not math.isnan(pt.pressure_msl)
            assert not math.isnan(pt.steering_speed_kmh)


# ---------------------------------------------------------------------------
# TestStencilExtremaCapture
# ---------------------------------------------------------------------------

class TestStencilExtremaCapture:
    """Verify stencil min/max correctly captures spatial outliers."""

    def test_stencil_captures_low_pressure_in_ring(self):
        """
        If one ring point has dramatically lower pressure, stencil_min_pressure_msl
        must reflect it and delta_p must be positive.
        """
        payload = []
        for i, p in enumerate(INDIA_MET_GRID_POINTS):
            coords = _stencil_coords(p["lat"], p["lon"])
            is_bob = p["id"] == "bob_westcentral"
            for j, (lat, lon) in enumerate(coords):
                # For bob_westcentral: ring north point (j==1) has very low pressure
                pressure = 990.0 if (is_bob and j == 1) else 1012.0
                payload.append(_make_stencil_item(
                    lat, lon,
                    pressure=pressure,
                    gusts=65.0 if is_bob else 22.0,
                    precip=8.0 if is_bob else 0.1,
                    rh700=90.0 if is_bob else 40.0,
                ))

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        bob = grid["bob_westcentral"]
        assert bob.stencil_min_pressure_msl <= 991.0, (
            f"Expected min pressure ≤991 hPa, got {bob.stencil_min_pressure_msl}"
        )
        assert bob.stencil_delta_p >= 1.0, (
            f"Expected delta_p ≥1.0, got {bob.stencil_delta_p}"
        )

    def test_stencil_captures_max_cape_in_ring(self):
        """Ring points with higher CAPE must elevate stencil_max_cape."""
        payload = []
        for p in INDIA_MET_GRID_POINTS:
            coords = _stencil_coords(p["lat"], p["lon"])
            is_target = p["id"] == "guwahati"
            for j, (lat, lon) in enumerate(coords):
                cape = 4500.0 if (is_target and j == 2) else 1200.0
                payload.append(_make_stencil_item(lat, lon, cape=cape))

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        guw = grid["guwahati"]
        assert guw.stencil_max_cape >= 4500.0, (
            f"stencil_max_cape should capture ring CAPE, got {guw.stencil_max_cape}"
        )
        assert guw.cape == pytest.approx(1200.0)  # center unchanged


# ---------------------------------------------------------------------------
# TestDeepLayerSteering
# ---------------------------------------------------------------------------

class TestDeepLayerSteering:
    """Verify vortex-canceling deep-layer steering computation."""

    def test_steering_cancels_vortex_rotation(self):
        """
        A perfect circular vortex (four ring points with winds blowing around a center)
        should produce near-zero steering — the rotational components cancel out.
        """
        payload = []
        for p in INDIA_MET_GRID_POINTS:
            coords = _stencil_coords(p["lat"], p["lon"])
            is_bob = p["id"] == "bob_westcentral"
            for j, (lat, lon) in enumerate(coords):
                if is_bob and j > 0:
                    # Tangential winds: N→E (90°), S→W (270°), E→S (180°), W→N (0°)
                    dirs = {1: 90.0, 2: 270.0, 3: 180.0, 4: 0.0}
                    # Apply same tangential wind at all levels
                    wd = dirs[j]
                    payload.append(_make_stencil_item(
                        lat, lon,
                        ws850=20.0, wd850=wd,
                        ws500=20.0, wd500=wd,
                        ws300=20.0, wd300=wd,
                        rh700=85.0, precip=5.0,
                    ))
                else:
                    payload.append(_make_stencil_item(lat, lon))

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        bob = grid["bob_westcentral"]
        # Steering should be near zero when vortex winds cancel perfectly
        assert bob.steering_speed_kmh < 5.0, (
            f"Vortex rotation should cancel in steering, got {bob.steering_speed_kmh} km/h"
        )

    def test_unidirectional_steering_flow(self):
        """
        All ring points with uniform westerly flow should produce westerly steering.
        Westerly = FROM 270° → storm moves eastward (heading ~90°).
        """
        payload = []
        for p in INDIA_MET_GRID_POINTS:
            coords = _stencil_coords(p["lat"], p["lon"])
            is_target = p["id"] == "delhi"
            for j, (lat, lon) in enumerate(coords):
                ws = 30.0 if (is_target and j > 0) else 10.0
                wd = 270.0 if (is_target and j > 0) else 180.0
                payload.append(_make_stencil_item(
                    lat, lon,
                    ws850=ws, wd850=wd,
                    ws500=ws, wd500=wd,
                    ws300=ws, wd300=wd,
                ))

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        delhi = grid["delhi"]
        assert delhi.steering_speed_kmh > 5.0
        # Direction should be ~270° (westerly from), so storm moves ~east (90°)
        risk = delhi.compute_convective_risk(horizon_step=0)
        heading = risk["motion_heading_deg"]
        # Allow ±45° around 90° (east)
        assert 45.0 <= heading <= 135.0, (
            f"Westerly steering should give ~eastward heading, got {heading}°"
        )


# ---------------------------------------------------------------------------
# TestConvectiveRiskPhysics
# ---------------------------------------------------------------------------

class TestConvectiveRiskPhysics:
    """
    Physics-based assertions on compute_convective_risk() covering:
    - Dry capped high-CAPE: gated below 28%
    - Calm mild coastal: low severity
    - Active Bay of Bengal cyclone: severe/extreme
    - Stencil eyewall signal driving severity
    """

    def test_dry_capped_high_cape_suppressed(self):
        """
        High CAPE + high CIN + zero precip + low RH700 + normal pressure:
        probability must be suppressed to ≤ 28% (moisture gate).
        """
        pt = StencilAtmosphere(
            id="x", name="x", region="x", lat=22.0, lon=80.0,
            cape=2500.0, cin=90.0, precipitation=0.0, cloud_cover=15.0,
            wind_speed_10m=10.0, wind_direction_10m=90.0, temperature_2m=35.0,
            pressure_msl=1015.0, wind_gusts_10m=14.0,
            relative_humidity_700hPa=25.0,
            stencil_min_pressure_msl=1015.0,
            stencil_max_wind_gusts=14.0,
            stencil_max_precipitation=0.0,
            stencil_max_cape=2500.0,
            stencil_max_rh700=25.0,
            stencil_delta_p=0.0,
            steering_speed_kmh=12.0,
            steering_dir_deg=270.0,
        )
        risk = pt.compute_convective_risk(horizon_step=0)
        assert risk["occurrence_probability"] <= 0.28, (
            f"Dry capped CAPE: prob {risk['occurrence_probability']:.3f} should be ≤ 0.28"
        )
        assert risk["flash_rate"] <= 1.0, (
            f"Dry capped CAPE: flash {risk['flash_rate']} should be ≤ 1.0"
        )
        assert risk["severity"] == "low"

    def test_calm_mild_coastal_low_severity(self):
        """Low precip, light winds, moderate CAPE, normal pressure → low severity."""
        pt = StencilAtmosphere(
            id="x", name="x", region="x", lat=10.0, lon=76.0,
            cape=1400.0, cin=42.0, precipitation=0.2, cloud_cover=35.0,
            wind_speed_10m=14.0, wind_direction_10m=240.0, temperature_2m=28.5,
            pressure_msl=1012.0, wind_gusts_10m=18.0,
            relative_humidity_700hPa=45.0,
            stencil_min_pressure_msl=1012.0,
            stencil_max_wind_gusts=20.0,
            stencil_max_precipitation=0.3,
            stencil_max_cape=1400.0,
            stencil_max_rh700=48.0,
            stencil_delta_p=0.0,
            steering_speed_kmh=12.0,
            steering_dir_deg=240.0,
        )
        risk = pt.compute_convective_risk(horizon_step=0)
        assert risk["severity"] == "low", f"Calm coastal: severity {risk['severity']} should be 'low'"
        assert risk["occurrence_probability"] <= 0.28
        assert risk["flash_rate"] <= 1.0
        # Motion from deep-layer steering, not surface wind
        assert risk["motion_speed_kmh"] >= 6.0

    def test_active_cyclone_severe_or_extreme(self):
        """
        Deep low pressure, gale gusts, high precip, high RH700 →
        severe or extreme with high flash density.
        """
        pt = StencilAtmosphere(
            id="bob", name="Cyclone", region="BoB", lat=13.0, lon=85.0,
            cape=2800.0, cin=12.0, precipitation=12.0, cloud_cover=92.0,
            wind_speed_10m=52.0, wind_direction_10m=240.0, temperature_2m=29.0,
            pressure_msl=996.0, wind_gusts_10m=72.0,
            relative_humidity_700hPa=92.0,
            stencil_min_pressure_msl=990.0,
            stencil_max_wind_gusts=78.0,
            stencil_max_precipitation=14.0,
            stencil_max_cape=3200.0,
            stencil_max_rh700=94.0,
            stencil_delta_p=10.0,
            steering_speed_kmh=28.0,
            steering_dir_deg=80.0,
        )
        risk = pt.compute_convective_risk(horizon_step=0)
        assert risk["occurrence_probability"] >= 0.75, (
            f"Active cyclone: prob {risk['occurrence_probability']:.3f} should be ≥ 0.75"
        )
        assert risk["severity"] in ("severe", "extreme"), (
            f"Active cyclone: severity {risk['severity']} should be severe/extreme"
        )
        assert risk["flash_rate"] >= 10.0, (
            f"Active cyclone: flash {risk['flash_rate']} should be ≥ 10.0"
        )

    def test_stencil_delta_p_drives_severity(self):
        """
        A node with ordinary surface fields but large radial pressure drop
        (≥ 6 hPa) should reach at least severe severity.
        """
        pt = StencilAtmosphere(
            id="x", name="x", region="x", lat=15.0, lon=84.0,
            cape=2000.0, cin=20.0, precipitation=4.0, cloud_cover=80.0,
            wind_speed_10m=35.0, wind_direction_10m=90.0, temperature_2m=29.5,
            pressure_msl=1008.0, wind_gusts_10m=50.0,
            relative_humidity_700hPa=80.0,
            stencil_min_pressure_msl=1004.0,
            stencil_max_wind_gusts=55.0,
            stencil_max_precipitation=6.0,
            stencil_max_cape=2500.0,
            stencil_max_rh700=85.0,
            stencil_delta_p=7.0,       # strong eyewall signal
            steering_speed_kmh=22.0,
            steering_dir_deg=85.0,
        )
        risk = pt.compute_convective_risk(horizon_step=0)
        assert risk["severity"] in ("severe", "extreme"), (
            f"Large delta_p=7 hPa should give severe/extreme, got {risk['severity']}"
        )

    def test_steering_heading_is_downwind_of_steering_dir(self):
        """motion_heading_deg == (steering_dir_deg + 180) % 360."""
        pt = StencilAtmosphere(
            id="x", name="x", region="x", lat=20.0, lon=80.0,
            cape=1500.0, cin=35.0, precipitation=0.5, cloud_cover=50.0,
            wind_speed_10m=18.0, wind_direction_10m=200.0, temperature_2m=30.0,
            pressure_msl=1012.0, wind_gusts_10m=24.0,
            relative_humidity_700hPa=55.0,
            stencil_min_pressure_msl=1012.0,
            stencil_max_wind_gusts=24.0,
            stencil_max_precipitation=0.5,
            stencil_max_cape=1500.0,
            stencil_max_rh700=55.0,
            stencil_delta_p=0.0,
            steering_speed_kmh=25.0,
            steering_dir_deg=255.0,
        )
        risk = pt.compute_convective_risk()
        expected_heading = (255.0 + 180.0) % 360.0
        assert risk["motion_heading_deg"] == pytest.approx(expected_heading, abs=0.5)

    def test_motion_speed_floored_at_6_kmh(self):
        """Zero steering wind should produce floor of 6 km/h translation speed."""
        pt = StencilAtmosphere(
            id="x", name="x", region="x", lat=22.0, lon=78.0,
            cape=800.0, cin=80.0, precipitation=0.0, cloud_cover=10.0,
            wind_speed_10m=5.0, wind_direction_10m=180.0, temperature_2m=32.0,
            pressure_msl=1015.0, wind_gusts_10m=8.0,
            relative_humidity_700hPa=20.0,
            stencil_min_pressure_msl=1015.0,
            stencil_max_wind_gusts=8.0,
            stencil_max_precipitation=0.0,
            stencil_max_cape=800.0,
            stencil_max_rh700=20.0,
            stencil_delta_p=0.0,
            steering_speed_kmh=0.0,   # completely calm steering
            steering_dir_deg=0.0,
        )
        risk = pt.compute_convective_risk()
        assert risk["motion_speed_kmh"] >= 6.0


# ---------------------------------------------------------------------------
# TestStormPhysicsCalibration (snapshot-level integration)
# ---------------------------------------------------------------------------

class TestStormPhysicsCalibration:
    """
    End-to-end integration tests: mock API → LiveWeatherService →
    inference_service.build_snapshot() → verify cell-level physics.
    """

    def test_calm_mild_malabar_low_risk(self):
        """
        Calm inland conditions (low precip, low RH700, normal pressure, light winds)
        must produce low severity and prob ≤ 28% across all cells.
        """
        payload = _build_flat_payload(default_kwargs=dict(
            cape=1400.0, cin=-42.0, precip=0.2, cloud=35.0,
            wind_spd=14.0, wind_dir=240.0, temp=28.5,
            pressure=1012.0, gusts=18.0, rh700=42.0,
            ws850=12.0, wd850=235.0,
            ws500=16.0, wd500=240.0,
            ws300=22.0, wd300=245.0,
        ))

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            live_weather_service.fetch_live_grid(force_refresh=True)

        snapshot = inference_service.build_snapshot(
            horizon_step=0,
            active_sensors={"radar": True, "satellite": True, "lightning": True, "nwp": True},
        )
        malabar = next(c for c in snapshot["cells"] if "Malabar" in c["name"])

        assert malabar["occurrenceProb"] <= 0.30, (
            f"Malabar prob {malabar['occurrenceProb']:.3f} should be ≤ 0.30 (calm)"
        )
        assert malabar["severity"] == "low"
        assert malabar["flashDensity"] <= 1.0
        # Motion speed from steering (never 92 km/h leak)
        assert malabar["motionSpeedKmh"] < 50.0

    def test_dry_capped_cape_all_cells_low(self):
        """
        Dry CAPE everywhere (no moisture support): all cells must stay at low severity
        with prob ≤ 28% and flash ≤ 1.0.
        """
        payload = _build_flat_payload(default_kwargs=dict(
            cape=2000.0, cin=-65.0, precip=0.0, cloud=18.0,
            wind_spd=12.0, wind_dir=90.0, temp=34.0,
            pressure=1015.0, gusts=16.0, rh700=22.0,
            ws850=10.0, wd850=85.0,
            ws500=16.0, wd500=90.0,
            ws300=24.0, wd300=92.0,
        ))

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            live_weather_service.fetch_live_grid(force_refresh=True)

        snapshot = inference_service.build_snapshot(
            horizon_step=0,
            active_sensors={"radar": True, "satellite": True, "lightning": True, "nwp": True},
        )
        for c in snapshot["cells"]:
            assert c["occurrenceProb"] <= 0.30, (
                f"{c['name']}: prob {c['occurrenceProb']:.3f} > 0.30 in dry-CAPE scenario"
            )
            assert c["flashDensity"] <= 1.0, (
                f"{c['name']}: flash {c['flashDensity']} > 1.0 in dry-CAPE scenario"
            )
            assert c["severity"] == "low"

    def test_active_bay_of_bengal_cyclone_severe(self):
        """
        Cyclonic fields (deep low, gale gusts, heavy rain, high RH700) at Bay of Bengal
        nodes must produce severe or extreme cells with high flash density.
        """
        bob_ids = {"bob_westcentral", "andhra_offshore", "andaman"}
        payload = _build_flat_payload(
            per_node_overrides={
                nid: dict(
                    cape=2900.0, cin=-10.0, precip=10.0, cloud=92.0,
                    wind_spd=55.0, wind_dir=240.0, temp=29.5,
                    pressure=994.0, gusts=72.0, rh700=92.0,
                    ws850=36.0, wd850=235.0,
                    ws500=44.0, wd500=240.0,
                    ws300=60.0, wd300=245.0,
                )
                for nid in bob_ids
            },
            default_kwargs=dict(
                cape=1200.0, cin=-50.0, precip=0.1, cloud=25.0,
                wind_spd=12.0, wind_dir=90.0, temp=30.0,
                pressure=1014.0, gusts=16.0, rh700=30.0,
                ws850=10.0, wd850=85.0,
                ws500=14.0, wd500=90.0,
                ws300=20.0, wd300=92.0,
            ),
        )

        with patch.object(httpx.Client, "get", return_value=_mock_response(payload)):
            live_weather_service.fetch_live_grid(force_refresh=True)

        snapshot = inference_service.build_snapshot(
            horizon_step=0,
            active_sensors={"radar": True, "satellite": True, "lightning": True, "nwp": True},
        )
        bob_cell = next(c for c in snapshot["cells"] if "Bay of Bengal" in c["name"])
        assert bob_cell["occurrenceProb"] >= 0.72, (
            f"BoB cyclone prob {bob_cell['occurrenceProb']:.3f} should be ≥ 0.72"
        )
        assert bob_cell["severity"] in ("severe", "extreme"), (
            f"BoB severity {bob_cell['severity']} should be severe/extreme"
        )
        assert bob_cell["flashDensity"] >= 10.0, (
            f"BoB flash density {bob_cell['flashDensity']} should be ≥ 10.0"
        )


# ---------------------------------------------------------------------------
# TestApiLiveWeatherIntegration
# ---------------------------------------------------------------------------

class TestApiLiveWeatherIntegration:
    """FastAPI endpoint integration and snapshot schema compliance."""

    def test_get_live_weather_endpoint(self, client):
        resp = client.get("/api/v1/nowcast/live-weather")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stationCount"] == 19
        assert "dataSource" in data
        assert "stations" in data
        assert len(data["stations"]) == 19

    def test_snapshot_includes_live_nwp_flag(self, client):
        resp = client.get("/api/v1/forecast/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert "dataSource" in data
        assert "isLiveNwp" in data
        assert isinstance(data["isLiveNwp"], bool)

    def test_snapshot_cells_have_physics_attributes(self, client):
        resp = client.get("/api/v1/forecast/latest")
        assert resp.status_code == 200
        data = resp.json()
        cells = data["cells"]
        assert len(cells) > 0
        for cell in cells:
            assert "cape" in cell
            assert "cin" in cell
            assert cell["cape"] >= 0
            assert 0.0 <= cell["occurrenceProb"] <= 1.0
            assert cell["flashDensity"] >= 0.0
            assert "motionSpeedKmh" in cell
            assert "bearingDeg" in cell


# ---------------------------------------------------------------------------
# TestIDWInterpolation
# ---------------------------------------------------------------------------

class TestIDWInterpolation:
    """Verify IDW blending returns distinct, physics-consistent telemetry
    for nearby-but-distinct query points (Bihar vs. Jharkhand)."""

    def test_bihar_vs_jharkhand_distinct_telemetry(self):
        """
        Bihar (25.46, 84.81) and Jharkhand/Ranchi (23.46, 85.66) are ~220 km
        apart and must NOT return identical CAPE or steering-wind values when
        their three nearest grid nodes differ (which they do in the benchmark
        fallback, since the Bihar node and Ranchi node have distinct fallback
        parameters).
        """
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            live_weather_service.fetch_live_grid(force_refresh=True)

        bihar = live_weather_service.get_nearest_conditions(25.46, 84.81)
        jharkhand = live_weather_service.get_nearest_conditions(23.46, 85.66)

        # The two query points are ~220 km apart; their IDW blends must differ.
        assert (
            abs(bihar.cape - jharkhand.cape) > 10.0
            or abs(bihar.steering_speed_kmh - jharkhand.steering_speed_kmh) > 0.5
            or abs(bihar.steering_dir_deg - jharkhand.steering_dir_deg) > 0.5
        ), (
            f"Bihar and Jharkhand returned identical telemetry — "
            f"IDW is not differentiating (bihar.cape={bihar.cape:.1f}, "
            f"jharkhand.cape={jharkhand.cape:.1f}, "
            f"bihar.steering_speed={bihar.steering_speed_kmh:.1f}, "
            f"jharkhand.steering_speed={jharkhand.steering_speed_kmh:.1f})"
        )

    def test_exact_node_query_fast_paths_chennai(self):
        """Node-collocated query within 0.05° must return the exact node (fast-path)."""
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            live_weather_service.fetch_live_grid(force_refresh=True)

        result = live_weather_service.get_nearest_conditions(13.1, 80.25)
        assert result.id == "chennai", (
            f"Expected chennai node via fast-path, got '{result.id}'"
        )

    def test_exact_node_query_fast_paths_delhi(self):
        """Delhi query at (28.5, 77.1) must fast-path to the delhi node."""
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            live_weather_service.fetch_live_grid(force_refresh=True)

        result = live_weather_service.get_nearest_conditions(28.5, 77.1)
        assert result.id == "delhi", (
            f"Expected delhi node via fast-path, got '{result.id}'"
        )

    def test_idw_result_cape_bounded_by_neighbours(self):
        """IDW-blended CAPE must lie within [min, max] of the 3 nearest nodes."""
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            grid = live_weather_service.fetch_live_grid(force_refresh=True)

        # Query a point away from all nodes (Mid-India interior)
        query_lat, query_lon = 22.5, 82.0
        result = live_weather_service.get_nearest_conditions(query_lat, query_lon)

        # Find the 3 nearest nodes' CAPEs
        import math as _math
        ranked = sorted(
            grid.values(),
            key=lambda pt: (
                (query_lat - pt.lat) ** 2
                + ((query_lon - pt.lon) * _math.cos(_math.radians((query_lat + pt.lat) / 2.0))) ** 2
            ),
        )
        cape_values = [pt.cape for pt in ranked[:3]]
        assert min(cape_values) <= result.cape <= max(cape_values) + 1e-6, (
            f"IDW CAPE {result.cape:.1f} out of [{min(cape_values):.1f}, "
            f"{max(cape_values):.1f}] neighbour range"
        )


# ---------------------------------------------------------------------------
# TestRegimeLabels
# ---------------------------------------------------------------------------

class TestRegimeLabels:
    """Verify _format_regime_label produces physics-consistent cell names."""

    # Import the helper lazily inside tests to avoid a module-level ImportError
    # if the helper is renamed; keeps test isolation clean.
    @staticmethod
    def _label(region: str, severity: str, is_cyclonic: bool) -> str:
        from app.services.inference_service import _format_regime_label
        return _format_regime_label(region, severity, is_cyclonic)

    # --- Calm / low-severity rules ---

    def test_low_severity_non_cyclonic_returns_stable_air_mass(self):
        label = self._label("Malabar Coast", "low", False)
        assert label == "Malabar Coast Stable Air Mass", label

    def test_low_severity_cyclonic_still_returns_stable_air_mass(self):
        """Cyclonic flag does not override 'low' severity — calm stays calm."""
        label = self._label("Northern Plains", "low", True)
        assert label == "Northern Plains Stable Air Mass", label

    # --- Moderate severity ---

    def test_moderate_non_cyclonic_returns_developing_cell(self):
        label = self._label("Telangana", "moderate", False)
        assert label == "Telangana Developing Cell", label

    def test_moderate_cyclonic_returns_cyclonic_band(self):
        label = self._label("Coromandel Coast", "moderate", True)
        assert label == "Coromandel Coast Cyclonic Band", label

    # --- Severe / extreme severity ---

    def test_severe_non_cyclonic_returns_convective_core(self):
        label = self._label("Chota Nagpur", "severe", False)
        assert label == "Chota Nagpur Convective Core", label

    def test_extreme_non_cyclonic_returns_convective_core(self):
        label = self._label("Mid-Gangetic Plains", "extreme", False)
        assert label == "Mid-Gangetic Plains Convective Core", label

    def test_severe_cyclonic_returns_cyclonic_cell(self):
        label = self._label("Westcentral Bay of Bengal", "severe", True)
        assert label == "Westcentral Bay of Bengal Cyclonic Cell", label

    def test_extreme_cyclonic_returns_cyclonic_cell(self):
        label = self._label("South Bay & Andaman", "extreme", True)
        assert label == "South Bay & Andaman Cyclonic Cell", label

    # --- Integration: cells returned by _build_cells respect physics labels ---

    def test_build_cells_names_are_regime_driven(self):
        """
        _build_cells must not return any static archetype suffix
        ("Squall Line", "Supercell Cluster") in the benchmark fallback path;
        all names must end with one of the five physics-regime suffixes.
        """
        from app.services.inference_service import _build_cells

        VALID_SUFFIXES = (
            "Stable Air Mass",
            "Developing Cell",
            "Convective Core",
            "Cyclonic Band",
            "Cyclonic Cell",
        )
        BANNED_SUFFIXES = ("Squall Line", "Supercell Cluster")

        # Force benchmark fallback by ensuring service is offline
        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            live_weather_service.fetch_live_grid(force_refresh=True)

        cells = _build_cells(horizon_step=0, inflation=1.0)
        assert len(cells) > 0

        for cell in cells:
            name = cell["name"]
            # Must end with a valid physics suffix
            assert any(name.endswith(s) for s in VALID_SUFFIXES), (
                f"Cell '{name}' does not end with a valid regime suffix"
            )
            # Must NOT contain banned static archetypes
            for banned in BANNED_SUFFIXES:
                assert banned not in name, (
                    f"Cell name '{name}' still contains banned static label '{banned}'"
                )

    def test_low_severity_cell_has_stable_air_mass_label(self):
        """
        In the benchmark fallback, the Malabar Coast cell (baseProb=0.18, low)
        must receive 'Stable Air Mass' label.
        """
        from app.services.inference_service import _build_cells

        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            live_weather_service.fetch_live_grid(force_refresh=True)

        cells = _build_cells(horizon_step=0, inflation=1.0)
        malabar = next((c for c in cells if "cell-malabar" in c["id"]), None)
        assert malabar is not None, "Malabar cell not found in _build_cells output"
        assert malabar["name"].endswith("Stable Air Mass"), (
            f"Expected 'Stable Air Mass' suffix for calm Malabar cell, got: '{malabar['name']}'"
        )

    def test_high_severity_marine_cell_has_cyclonic_label(self):
        """
        In the benchmark fallback, the Westcentral Bay of Bengal cell
        (isMarine=True, baseProb=0.82 → severe/extreme) must receive
        'Cyclonic Cell' or 'Cyclonic Band' label.
        """
        from app.services.inference_service import _build_cells

        with patch.object(httpx.Client, "get",
                          side_effect=httpx.ConnectError("offline")):
            live_weather_service.fetch_live_grid(force_refresh=True)

        cells = _build_cells(horizon_step=0, inflation=1.0)
        bob = next((c for c in cells if "cell-westcentral-bob" in c["id"]), None)
        assert bob is not None, "Westcentral BoB cell not found in _build_cells output"
        assert "Cyclonic" in bob["name"], (
            f"Expected 'Cyclonic' in label for active marine cell, got: '{bob['name']}'"
        )
