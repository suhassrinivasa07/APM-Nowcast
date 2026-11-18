"""
live_weather.py – Universal mesoscale & cyclonic atmospheric analyzer for APM-Nowcast.

Architecture:
  - 5-Point Spatial Stencil (center ± 1.0° N/S/E/W) per node: extracts stencil-
    core extrema (min pressure, max gusts, max precip, max cape, max RH700) and
    local radial pressure drop (delta_p) so coarse NWP grids never miss a cyclone
    eyewall or tight convective core.
  - Multi-level query: cape, convective_inhibition, precipitation, cloud_cover,
    pressure_msl, wind_speed_10m, wind_gusts_10m, wind_direction_10m,
    relative_humidity_700hPa, wind_speed_850hPa, wind_direction_850hPa,
    wind_speed_500hPa, wind_direction_500hPa, wind_speed_300hPa, wind_direction_300hPa.
  - Deep-layer steering (850 × 0.25 + 500 × 0.50 + 300 × 0.25 hPa) computed
    over the 4 outer ring points cancels internal vortex spin and yields true
    environmental steering drift for cell translation.
  - Zero hardcoded storm probabilities, coordinates, or severity rules: continuous
    physical sigmoid from thermodynamic, moisture/cloud, and cyclonic pressure/gust
    forcing. Requires moisture/dynamic support before elevating above 28%.
  - Sanitizes None/NaN fields safely. Graceful 4-second timeout with benchmark
    atmospheric cube fallback.
  - In-memory cache with 10-minute TTL (600 s).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from common.logging_config import get_logger

log = get_logger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 600.0       # 10-minute TTL
HTTP_TIMEOUT_SECONDS = 4.0      # 4-second hard timeout for live calls

# Stencil offset: ±1.0° cardinal cardinal neighbours
_STENCIL_DELTA = 1.0

# Deep-layer steering level weights (must sum to 1.0)
_STEERING_WEIGHTS = {
    "850hPa": 0.25,
    "500hPa": 0.50,
    "300hPa": 0.25,
}

# ---------------------------------------------------------------------------
# 18 Strategic Indian Meteorological Grid Points (Pan-India & Bay of Bengal)
# ---------------------------------------------------------------------------
INDIA_MET_GRID_POINTS: List[Dict[str, Any]] = [
    {
        "id": "chennai",
        "name": "Chennai / Tamil Nadu Coast",
        "lat": 13.1,
        "lon": 80.3,
        "region": "Coromandel Coast",
        "fallback_cape": 2450.0,
        "fallback_cin": 28.0,
        "fallback_precip": 1.2,
        "fallback_cloud": 65.0,
        "fallback_wind_speed": 24.0,
        "fallback_wind_dir": 65.0,
        "fallback_temp": 29.5,
        "fallback_pressure_msl": 1010.5,
        "fallback_wind_gusts": 32.0,
        "fallback_rh700": 72.0,
        "fallback_wind_speed_850": 20.0,
        "fallback_wind_dir_850": 60.0,
        "fallback_wind_speed_500": 28.0,
        "fallback_wind_dir_500": 70.0,
        "fallback_wind_speed_300": 42.0,
        "fallback_wind_dir_300": 75.0,
    },
    {
        "id": "kerala",
        "name": "Kerala / Kochi",
        "lat": 9.9312,
        "lon": 76.2673,
        "region": "Malabar Coast",
        "fallback_cape": 1380.0,
        "fallback_cin": 42.0,
        "fallback_precip": 0.2,
        "fallback_cloud": 35.0,
        "fallback_wind_speed": 14.0,
        "fallback_wind_dir": 240.0,
        "fallback_temp": 28.5,
        "fallback_pressure_msl": 1012.0,
        "fallback_wind_gusts": 18.0,
        "fallback_rh700": 45.0,
        "fallback_wind_speed_850": 12.0,
        "fallback_wind_dir_850": 235.0,
        "fallback_wind_speed_500": 16.0,
        "fallback_wind_dir_500": 240.0,
        "fallback_wind_speed_300": 22.0,
        "fallback_wind_dir_300": 250.0,
    },
    {
        "id": "bengaluru",
        "name": "Bengaluru",
        "lat": 12.9716,
        "lon": 77.5946,
        "region": "South Interior Karnataka",
        "fallback_cape": 1850.0,
        "fallback_cin": 38.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 45.0,
        "fallback_wind_speed": 18.0,
        "fallback_wind_dir": 85.0,
        "fallback_temp": 27.0,
        "fallback_pressure_msl": 1013.0,
        "fallback_wind_gusts": 24.0,
        "fallback_rh700": 50.0,
        "fallback_wind_speed_850": 16.0,
        "fallback_wind_dir_850": 80.0,
        "fallback_wind_speed_500": 22.0,
        "fallback_wind_dir_500": 85.0,
        "fallback_wind_speed_300": 32.0,
        "fallback_wind_dir_300": 90.0,
    },
    {
        "id": "mumbai",
        "name": "Mumbai / Konkan",
        "lat": 19.0760,
        "lon": 72.8777,
        "region": "Konkan & Western Ghats",
        "fallback_cape": 2100.0,
        "fallback_cin": 42.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 35.0,
        "fallback_wind_speed": 28.0,
        "fallback_wind_dir": 320.0,
        "fallback_temp": 31.0,
        "fallback_pressure_msl": 1012.5,
        "fallback_wind_gusts": 34.0,
        "fallback_rh700": 48.0,
        "fallback_wind_speed_850": 24.0,
        "fallback_wind_dir_850": 315.0,
        "fallback_wind_speed_500": 32.0,
        "fallback_wind_dir_500": 320.0,
        "fallback_wind_speed_300": 44.0,
        "fallback_wind_dir_300": 325.0,
    },
    {
        "id": "hyderabad",
        "name": "Hyderabad / Telangana",
        "lat": 17.3850,
        "lon": 78.4867,
        "region": "Telangana Plateau",
        "fallback_cape": 1850.0,
        "fallback_cin": 55.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 30.0,
        "fallback_wind_speed": 22.0,
        "fallback_wind_dir": 110.0,
        "fallback_temp": 32.5,
        "fallback_pressure_msl": 1013.0,
        "fallback_wind_gusts": 28.0,
        "fallback_rh700": 42.0,
        "fallback_wind_speed_850": 18.0,
        "fallback_wind_dir_850": 105.0,
        "fallback_wind_speed_500": 24.0,
        "fallback_wind_dir_500": 110.0,
        "fallback_wind_speed_300": 36.0,
        "fallback_wind_dir_300": 115.0,
    },
    {
        "id": "bhubaneswar",
        "name": "Bhubaneswar / Odisha",
        "lat": 20.2961,
        "lon": 85.8245,
        "region": "Mahanadi Basin & Odisha Coast",
        "fallback_cape": 3050.0,
        "fallback_cin": 28.0,
        "fallback_precip": 0.6,
        "fallback_cloud": 60.0,
        "fallback_wind_speed": 34.0,
        "fallback_wind_dir": 215.0,
        "fallback_temp": 31.0,
        "fallback_pressure_msl": 1009.5,
        "fallback_wind_gusts": 42.0,
        "fallback_rh700": 68.0,
        "fallback_wind_speed_850": 28.0,
        "fallback_wind_dir_850": 210.0,
        "fallback_wind_speed_500": 36.0,
        "fallback_wind_dir_500": 215.0,
        "fallback_wind_speed_300": 50.0,
        "fallback_wind_dir_300": 220.0,
    },
    {
        "id": "kolkata",
        "name": "Kolkata / Gangetic Bengal",
        "lat": 22.5726,
        "lon": 88.3639,
        "region": "Lower Gangetic Plain",
        "fallback_cape": 2850.0,
        "fallback_cin": 35.0,
        "fallback_precip": 0.4,
        "fallback_cloud": 55.0,
        "fallback_wind_speed": 32.0,
        "fallback_wind_dir": 210.0,
        "fallback_temp": 32.0,
        "fallback_pressure_msl": 1010.0,
        "fallback_wind_gusts": 40.0,
        "fallback_rh700": 65.0,
        "fallback_wind_speed_850": 26.0,
        "fallback_wind_dir_850": 205.0,
        "fallback_wind_speed_500": 34.0,
        "fallback_wind_dir_500": 210.0,
        "fallback_wind_speed_300": 48.0,
        "fallback_wind_dir_300": 215.0,
    },
    {
        "id": "guwahati",
        "name": "Guwahati / Brahmaputra",
        "lat": 26.1445,
        "lon": 91.7362,
        "region": "Brahmaputra Valley & Northeast",
        "fallback_cape": 3450.0,
        "fallback_cin": 15.0,
        "fallback_precip": 3.8,
        "fallback_cloud": 82.0,
        "fallback_wind_speed": 40.0,
        "fallback_wind_dir": 230.0,
        "fallback_temp": 28.0,
        "fallback_pressure_msl": 1008.0,
        "fallback_wind_gusts": 48.0,
        "fallback_rh700": 80.0,
        "fallback_wind_speed_850": 32.0,
        "fallback_wind_dir_850": 225.0,
        "fallback_wind_speed_500": 40.0,
        "fallback_wind_dir_500": 230.0,
        "fallback_wind_speed_300": 56.0,
        "fallback_wind_dir_300": 235.0,
    },
    {
        "id": "ranchi",
        "name": "Ranchi / Chota Nagpur",
        "lat": 23.3441,
        "lon": 85.3096,
        "region": "Chota Nagpur Plateau",
        "fallback_cape": 3200.0,
        "fallback_cin": 22.0,
        "fallback_precip": 1.8,
        "fallback_cloud": 72.0,
        "fallback_wind_speed": 38.0,
        "fallback_wind_dir": 248.0,
        "fallback_temp": 30.5,
        "fallback_pressure_msl": 1009.0,
        "fallback_wind_gusts": 46.0,
        "fallback_rh700": 74.0,
        "fallback_wind_speed_850": 30.0,
        "fallback_wind_dir_850": 243.0,
        "fallback_wind_speed_500": 38.0,
        "fallback_wind_dir_500": 248.0,
        "fallback_wind_speed_300": 52.0,
        "fallback_wind_dir_300": 252.0,
    },
    {
        "id": "nagpur",
        "name": "Nagpur",
        "lat": 21.1458,
        "lon": 79.0882,
        "region": "Central India & Vidarbha",
        "fallback_cape": 1600.0,
        "fallback_cin": 65.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 20.0,
        "fallback_wind_speed": 18.0,
        "fallback_wind_dir": 70.0,
        "fallback_temp": 33.0,
        "fallback_pressure_msl": 1013.5,
        "fallback_wind_gusts": 22.0,
        "fallback_rh700": 38.0,
        "fallback_wind_speed_850": 14.0,
        "fallback_wind_dir_850": 65.0,
        "fallback_wind_speed_500": 20.0,
        "fallback_wind_dir_500": 70.0,
        "fallback_wind_speed_300": 30.0,
        "fallback_wind_dir_300": 75.0,
    },
    {
        "id": "delhi",
        "name": "Delhi NCR",
        "lat": 28.6139,
        "lon": 77.2090,
        "region": "Northern Plains / NCR",
        "fallback_cape": 950.0,
        "fallback_cin": 85.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 25.0,
        "fallback_wind_speed": 16.0,
        "fallback_wind_dir": 290.0,
        "fallback_temp": 28.0,
        "fallback_pressure_msl": 1014.0,
        "fallback_wind_gusts": 20.0,
        "fallback_rh700": 30.0,
        "fallback_wind_speed_850": 12.0,
        "fallback_wind_dir_850": 285.0,
        "fallback_wind_speed_500": 18.0,
        "fallback_wind_dir_500": 290.0,
        "fallback_wind_speed_300": 28.0,
        "fallback_wind_dir_300": 295.0,
    },
    {
        "id": "jaipur",
        "name": "Jaipur",
        "lat": 26.9124,
        "lon": 75.7873,
        "region": "East Rajasthan Semi-Arid",
        "fallback_cape": 650.0,
        "fallback_cin": 110.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 15.0,
        "fallback_wind_speed": 19.0,
        "fallback_wind_dir": 280.0,
        "fallback_temp": 30.0,
        "fallback_pressure_msl": 1014.5,
        "fallback_wind_gusts": 24.0,
        "fallback_rh700": 25.0,
        "fallback_wind_speed_850": 14.0,
        "fallback_wind_dir_850": 275.0,
        "fallback_wind_speed_500": 20.0,
        "fallback_wind_dir_500": 280.0,
        "fallback_wind_speed_300": 30.0,
        "fallback_wind_dir_300": 285.0,
    },
    {
        "id": "ahmedabad",
        "name": "Ahmedabad",
        "lat": 23.0225,
        "lon": 72.5714,
        "region": "Gujarat Coastal Plain",
        "fallback_cape": 1400.0,
        "fallback_cin": 75.0,
        "fallback_precip": 0.0,
        "fallback_cloud": 20.0,
        "fallback_wind_speed": 21.0,
        "fallback_wind_dir": 310.0,
        "fallback_temp": 33.0,
        "fallback_pressure_msl": 1013.5,
        "fallback_wind_gusts": 26.0,
        "fallback_rh700": 32.0,
        "fallback_wind_speed_850": 16.0,
        "fallback_wind_dir_850": 305.0,
        "fallback_wind_speed_500": 22.0,
        "fallback_wind_dir_500": 310.0,
        "fallback_wind_speed_300": 32.0,
        "fallback_wind_dir_300": 315.0,
    },
    {
        "id": "andaman",
        "name": "Andaman / Bay of Bengal",
        "lat": 11.6234,
        "lon": 92.7265,
        "region": "Bay of Bengal & Andaman Sea",
        "fallback_cape": 2750.0,
        "fallback_cin": 18.0,
        "fallback_precip": 4.5,
        "fallback_cloud": 85.0,
        "fallback_wind_speed": 35.0,
        "fallback_wind_dir": 235.0,
        "fallback_temp": 28.5,
        "fallback_pressure_msl": 1006.5,
        "fallback_wind_gusts": 48.0,
        "fallback_rh700": 82.0,
        "fallback_wind_speed_850": 30.0,
        "fallback_wind_dir_850": 230.0,
        "fallback_wind_speed_500": 38.0,
        "fallback_wind_dir_500": 235.0,
        "fallback_wind_speed_300": 52.0,
        "fallback_wind_dir_300": 240.0,
    },
    {
        "id": "bob_westcentral",
        "name": "Westcentral Bay of Bengal",
        "lat": 13.1,
        "lon": 85.3,
        "region": "Bay of Bengal Marine",
        "fallback_cape": 2600.0,
        "fallback_cin": 15.0,
        "fallback_precip": 3.2,
        "fallback_cloud": 80.0,
        "fallback_wind_speed": 36.0,
        "fallback_wind_dir": 80.0,
        "fallback_temp": 29.0,
        "fallback_pressure_msl": 1004.0,
        "fallback_wind_gusts": 52.0,
        "fallback_rh700": 85.0,
        "fallback_wind_speed_850": 32.0,
        "fallback_wind_dir_850": 75.0,
        "fallback_wind_speed_500": 42.0,
        "fallback_wind_dir_500": 80.0,
        "fallback_wind_speed_300": 58.0,
        "fallback_wind_dir_300": 85.0,
    },
    {
        "id": "andhra_offshore",
        "name": "Off Andhra Coast",
        "lat": 15.5,
        "lon": 84.0,
        "region": "Bay of Bengal Marine",
        "fallback_cape": 2500.0,
        "fallback_cin": 18.0,
        "fallback_precip": 2.8,
        "fallback_cloud": 75.0,
        "fallback_wind_speed": 32.0,
        "fallback_wind_dir": 85.0,
        "fallback_temp": 29.2,
        "fallback_pressure_msl": 1005.5,
        "fallback_wind_gusts": 46.0,
        "fallback_rh700": 80.0,
        "fallback_wind_speed_850": 28.0,
        "fallback_wind_dir_850": 80.0,
        "fallback_wind_speed_500": 36.0,
        "fallback_wind_dir_500": 85.0,
        "fallback_wind_speed_300": 50.0,
        "fallback_wind_dir_300": 88.0,
    },
    {
        "id": "visakhapatnam",
        "name": "Visakhapatnam",
        "lat": 17.7,
        "lon": 83.3,
        "region": "Andhra Coast",
        "fallback_cape": 2350.0,
        "fallback_cin": 25.0,
        "fallback_precip": 1.5,
        "fallback_cloud": 70.0,
        "fallback_wind_speed": 26.0,
        "fallback_wind_dir": 90.0,
        "fallback_temp": 29.8,
        "fallback_pressure_msl": 1008.0,
        "fallback_wind_gusts": 38.0,
        "fallback_rh700": 75.0,
        "fallback_wind_speed_850": 22.0,
        "fallback_wind_dir_850": 85.0,
        "fallback_wind_speed_500": 30.0,
        "fallback_wind_dir_500": 90.0,
        "fallback_wind_speed_300": 44.0,
        "fallback_wind_dir_300": 92.0,
    },
    {
        "id": "puri_odisha",
        "name": "Puri / Odisha Coast",
        "lat": 19.8,
        "lon": 85.8,
        "region": "Odisha Coast",
        "fallback_cape": 2700.0,
        "fallback_cin": 22.0,
        "fallback_precip": 2.0,
        "fallback_cloud": 72.0,
        "fallback_wind_speed": 28.0,
        "fallback_wind_dir": 100.0,
        "fallback_temp": 30.0,
        "fallback_pressure_msl": 1008.5,
        "fallback_wind_gusts": 40.0,
        "fallback_rh700": 76.0,
        "fallback_wind_speed_850": 24.0,
        "fallback_wind_dir_850": 95.0,
        "fallback_wind_speed_500": 32.0,
        "fallback_wind_dir_500": 100.0,
        "fallback_wind_speed_300": 46.0,
        "fallback_wind_dir_300": 102.0,
    },
]

# Variables queried per stencil point (center + 4 ring points)
_CURRENT_VARS: List[str] = [
    "cape",
    "convective_inhibition",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "relative_humidity_700hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "wind_speed_500hPa",
    "wind_direction_500hPa",
    "wind_speed_300hPa",
    "wind_direction_300hPa",
]


# ---------------------------------------------------------------------------
# Helper: safe float extraction (None / NaN → fallback)
# ---------------------------------------------------------------------------
def _safe(value: Any, fallback: float) -> float:
    """Return float(value) if value is not None/NaN, else fallback."""
    if value is None:
        return fallback
    try:
        v = float(value)
        return fallback if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Helper: wind (u, v) ↔ speed/direction conversions
# ---------------------------------------------------------------------------
def _uv_from_spd_dir(speed: float, direction_deg: float) -> Tuple[float, float]:
    """Convert meteorological wind speed + direction to (u, v) components.
    Meteorological convention: direction is FROM which wind blows (0° = N).
    u = eastward component, v = northward component.
    """
    rad = math.radians(direction_deg)
    u = -speed * math.sin(rad)   # eastward
    v = -speed * math.cos(rad)   # northward
    return u, v


def _spd_dir_from_uv(u: float, v: float) -> Tuple[float, float]:
    """Convert (u, v) wind components back to speed and meteorological direction."""
    speed = math.sqrt(u * u + v * v)
    if speed < 0.5:
        return speed, 0.0
    # Math direction (blowing TO), convert to met direction (blowing FROM)
    dir_math = math.degrees(math.atan2(u, v))
    met_dir = (dir_math + 180.0) % 360.0
    return speed, met_dir


# ---------------------------------------------------------------------------
# Stencil geometry builder
# ---------------------------------------------------------------------------
def _stencil_coords(lat: float, lon: float) -> List[Tuple[float, float]]:
    """Return center + 4 cardinal ±1.0° neighbours: [C, N, S, E, W]."""
    d = _STENCIL_DELTA
    return [
        (lat,     lon),         # center (index 0)
        (lat + d, lon),         # north  (index 1)
        (lat - d, lon),         # south  (index 2)
        (lat,     lon + d),     # east   (index 3)
        (lat,     lon - d),     # west   (index 4)
    ]


# ---------------------------------------------------------------------------
# StencilAtmosphere – 5-point stencil aggregated per grid node
# ---------------------------------------------------------------------------
@dataclass
class StencilAtmosphere:
    """
    5-point stencil atmospheric aggregate for one Indian meteorological grid node.

    Center fields (index 0) provide the basic thermodynamic state.
    Stencil extrema (min/max over all 5 points) capture sub-grid cyclone eyes
    and convective cores that a single-point NWP grid cell may miss.
    Deep-layer steering is derived from the 4 outer ring points (vortex-canceling).
    """
    # --- Identity ---
    id: str
    name: str
    region: str
    lat: float
    lon: float

    # --- Center-point fields ---
    cape: float               # J/kg
    cin: float                # J/kg (CIN magnitude; always stored as positive)
    precipitation: float      # mm/h
    cloud_cover: float        # %
    wind_speed_10m: float     # km/h
    wind_direction_10m: float # degrees (met, from-north)
    temperature_2m: float     # °C
    pressure_msl: float       # hPa
    wind_gusts_10m: float     # km/h
    relative_humidity_700hPa: float  # %

    # --- Stencil extrema (over center + 4 ring points) ---
    stencil_min_pressure_msl: float    # hPa  – tracks cyclone eye
    stencil_max_wind_gusts: float      # km/h – tracks eyewall gusts
    stencil_max_precipitation: float   # mm/h – tracks convective core
    stencil_max_cape: float            # J/kg – tracks maximum instability
    stencil_max_rh700: float           # %    – tracks mid-level moisture
    stencil_delta_p: float             # hPa  – radial pressure drop (eyewall signal)

    # --- Deep-layer steering (vortex-canceling ring average) ---
    steering_speed_kmh: float    # km/h (850×0.25 + 500×0.50 + 300×0.25)
    steering_dir_deg: float      # degrees met (storm translation FROM direction)

    # --- Source metadata ---
    is_live: bool = False
    source: str = "benchmark_cube"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Backward-compat alias: expose 'rain' == 'precipitation' so callers
        # that relied on the pre-stencil GridPointAtmosphere schema still work.
        d.setdefault("rain", d.get("precipitation", 0.0))
        return d

    # -----------------------------------------------------------------------
    # Core physics: compute_convective_risk()
    # -----------------------------------------------------------------------
    def compute_convective_risk(
        self,
        horizon_step: int = 0,
    ) -> Dict[str, Any]:
        """
        Universal mesoscale & cyclonic risk scorer — zero hardcoded coordinates.

        1. Motion vector: steering_speed_kmh & steering_dir_deg from deep-layer
           ring average. Storm translation heading is downwind from the steering
           flow direction: (steering_dir_deg + 180) % 360.

        2. Physical continuous probability from:
           - Thermodynamic: CAPE / CIN
           - Moisture / cloud: RH700, precipitation, cloud cover
           - Cyclonic forcing: stencil_delta_p, stencil_min_pressure_msl, gusts
           Requires moisture/dynamic support before exceeding 28%.

        3. Severity purely from continuous physical fields — no region names.
        """
        time_evo = [1.0, 1.18, 1.28, 1.12][horizon_step] if horizon_step in (0, 1, 2, 3) else 1.0

        # --- Storm translation from deep-layer steering ---
        motion_heading_deg = round((self.steering_dir_deg + 180.0) % 360.0, 1)
        motion_speed_kmh = round(max(6.0, self.steering_speed_kmh), 1)
        sustained_wind_kmh = round(max(self.wind_speed_10m, self.stencil_max_wind_gusts * 0.8), 1)

        # --- Thermodynamic forcing ---
        cin_val = abs(self.cin)
        cin_cap_factor = max(0.04, 1.0 - min(cin_val, 120.0) / 95.0)
        thermo_lift = (self.stencil_max_cape / 1600.0) * cin_cap_factor

        # --- Moisture / cloud triggering ---
        max_precip = self.stencil_max_precipitation
        cloud_frac = self.cloud_cover / 100.0
        rh700_frac = self.stencil_max_rh700 / 100.0
        moist_trigger = (rh700_frac * 0.45) + (cloud_frac * 0.25) + (min(max_precip / 4.0, 1.5) * 0.70)

        # --- Cyclonic / dynamic forcing ---
        # Pressure anomaly uses stencil minimum (tracks the actual cyclone eye)
        pressure_anomaly = max(0.0, 1012.0 - self.stencil_min_pressure_msl)
        pressure_forcing = min(pressure_anomaly / 8.0, 2.5)
        # Radial pressure drop: strong signal for eyewall vs. environment
        delta_p_forcing = min(self.stencil_delta_p / 4.0, 2.0)
        gust_forcing = max(0.0, (self.stencil_max_wind_gusts - 25.0) / 35.0)
        surface_wind_forcing = max(0.0, (self.wind_speed_10m - 25.0) / 35.0)
        dynamic_forcing = (
            pressure_forcing * 0.55
            + delta_p_forcing * 0.25
            + gust_forcing * 0.40
            + surface_wind_forcing * 0.25
        )

        # --- Moisture/dynamic gate: suppress false alerts in dry capped CAPE ---
        # Support requires: RH700 ≥ 65%, OR precip > 0.5 mm, OR cyclonic vortex
        has_moisture_support = self.stencil_max_rh700 >= 65.0
        has_precip_trigger = max_precip > 0.5
        has_cyclonic_vortex = (
            self.stencil_delta_p >= 2.5
            or self.stencil_min_pressure_msl <= 1004.0
        )
        has_support = has_moisture_support or has_precip_trigger or has_cyclonic_vortex

        # --- Continuous physical CI index ---
        trigger = moist_trigger + dynamic_forcing
        ci_index = thermo_lift * min(trigger, 2.2) + dynamic_forcing * 0.55

        # --- Sigmoid probability ---
        prob_raw = 0.08 + 0.88 / (1.0 + math.exp(-2.5 * (ci_index - 0.90)))

        # Suppress to ≤ 28% if moisture/dynamic support is absent
        if not has_support:
            prob_raw = min(prob_raw, 0.28)

        prob = round(min(0.96, max(0.06, prob_raw * (1.0 + horizon_step * 0.02))), 3)

        # --- Flash rate ---
        base_flash = (
            (thermo_lift * 6.0 + dynamic_forcing * 9.0) * min(1.2, moist_trigger)
            + max_precip * 0.75
        )
        flash_rate = round(max(0.15, base_flash * time_evo), 2)
        if not has_support:
            flash_rate = round(min(flash_rate, 1.0), 2)

        # --- Continuous severity classification ---
        is_extreme = (
            prob >= 0.85
            or flash_rate >= 20.0
            or (self.stencil_min_pressure_msl <= 994.0 and self.stencil_max_wind_gusts >= 60.0)
            or self.stencil_max_wind_gusts >= 70.0
        )
        is_severe = (
            prob >= 0.72
            or (prob >= 0.60 and flash_rate >= 12.0)
            or (self.stencil_min_pressure_msl <= 1004.0 and self.stencil_max_wind_gusts >= 45.0)
            or (self.stencil_max_cape >= 2400.0 and max_precip >= 6.0)
            or self.stencil_delta_p >= 6.0
        )

        if is_extreme:
            severity = "extreme"
            growth = "intensifying"
        elif is_severe:
            severity = "severe"
            growth = "intensifying"
        elif prob < 0.26 and flash_rate <= 1.0:
            severity = "low"
            growth = "decaying" if cin_val > 35 else "steady"
        else:
            severity = "moderate"
            growth = "steady"

        return {
            "occurrence_probability": prob,
            "flash_rate": flash_rate,
            "severity": severity,
            "growth": growth,
            "motion_speed_kmh": motion_speed_kmh,
            "motion_heading_deg": motion_heading_deg,
            "sustained_wind_kmh": sustained_wind_kmh,
            "advect_bearing": motion_heading_deg,
        }


# ---------------------------------------------------------------------------
# Backward-compat alias: GridPointAtmosphere → StencilAtmosphere
# ---------------------------------------------------------------------------
GridPointAtmosphere = StencilAtmosphere


# ---------------------------------------------------------------------------
# Stencil parsing helpers
# ---------------------------------------------------------------------------
def _parse_stencil_items(
    items: List[Dict[str, Any]],
    start_idx: int,
    p: Dict[str, Any],
) -> StencilAtmosphere:
    """
    Parse 5 consecutive Open-Meteo result items (center + N/S/E/W) into a
    StencilAtmosphere, computing extrema and deep-layer steering.
    If fewer than 5 items are available the missing ones use fallback values.
    """

    def _get(item: Dict[str, Any], key: str, fb: float) -> float:
        curr = item.get("current", {}) if item else {}
        return _safe(curr.get(key), fb)

    # -- Extract all 5 stencil points --
    stencil_data: List[Dict[str, Any]] = []
    for k in range(5):
        idx = start_idx + k
        stencil_data.append(items[idx] if idx < len(items) else {})

    center = stencil_data[0]
    ring   = stencil_data[1:]  # N, S, E, W

    # Center fields
    cape        = _get(center, "cape",                   p["fallback_cape"])
    cin_raw     = _get(center, "convective_inhibition",  p["fallback_cin"])
    cin         = abs(cin_raw)  # always store magnitude
    precip      = _get(center, "precipitation",          p["fallback_precip"])
    cloud       = _get(center, "cloud_cover",            p["fallback_cloud"])
    wind_spd    = _get(center, "wind_speed_10m",         p["fallback_wind_speed"])
    wind_dir    = _get(center, "wind_direction_10m",     p["fallback_wind_dir"])
    temp        = _get(center, "temperature_2m",         p["fallback_temp"])
    pressure    = _get(center, "pressure_msl",           p["fallback_pressure_msl"])
    gusts       = _get(center, "wind_gusts_10m",         p["fallback_wind_gusts"])
    rh700       = _get(center, "relative_humidity_700hPa", p["fallback_rh700"])

    # Stencil extrema (all 5 points)
    all_pressures = [pressure]
    all_gusts     = [gusts]
    all_precips   = [precip]
    all_capes     = [cape]
    all_rh700     = [rh700]

    for r in ring:
        all_pressures.append(_get(r, "pressure_msl",             p["fallback_pressure_msl"]))
        all_gusts.append    (_get(r, "wind_gusts_10m",           p["fallback_wind_gusts"]))
        all_precips.append  (_get(r, "precipitation",            p["fallback_precip"]))
        all_capes.append    (_get(r, "cape",                     p["fallback_cape"]))
        all_rh700.append    (_get(r, "relative_humidity_700hPa", p["fallback_rh700"]))

    min_pressure = min(all_pressures)
    max_gusts    = max(all_gusts)
    max_precip   = max(all_precips)
    max_cape     = max(all_capes)
    max_rh700    = max(all_rh700)
    # Radial pressure drop: mean-of-ring minus center minimum
    ring_mean_p  = sum(all_pressures[1:]) / 4.0 if ring else pressure
    delta_p      = max(0.0, ring_mean_p - min_pressure)

    # Deep-layer steering from the 4 outer ring points (vortex-canceling)
    # Weighted composite: 850 × 0.25 + 500 × 0.50 + 300 × 0.25
    u_sum, v_sum = 0.0, 0.0
    for r in ring:
        u850, v850 = _uv_from_spd_dir(
            _get(r, "wind_speed_850hPa",    p["fallback_wind_speed_850"]),
            _get(r, "wind_direction_850hPa", p["fallback_wind_dir_850"]),
        )
        u500, v500 = _uv_from_spd_dir(
            _get(r, "wind_speed_500hPa",    p["fallback_wind_speed_500"]),
            _get(r, "wind_direction_500hPa", p["fallback_wind_dir_500"]),
        )
        u300, v300 = _uv_from_spd_dir(
            _get(r, "wind_speed_300hPa",    p["fallback_wind_speed_300"]),
            _get(r, "wind_direction_300hPa", p["fallback_wind_dir_300"]),
        )
        u_layer = u850 * 0.25 + u500 * 0.50 + u300 * 0.25
        v_layer = v850 * 0.25 + v500 * 0.50 + v300 * 0.25
        u_sum += u_layer
        v_sum += v_layer

    n_ring = max(1, len(ring))
    steering_speed, steering_dir = _spd_dir_from_uv(u_sum / n_ring, v_sum / n_ring)

    return StencilAtmosphere(
        id=p["id"],
        name=p["name"],
        region=p["region"],
        lat=p["lat"],
        lon=p["lon"],
        cape=cape,
        cin=cin,
        precipitation=precip,
        cloud_cover=cloud,
        wind_speed_10m=wind_spd,
        wind_direction_10m=wind_dir,
        temperature_2m=temp,
        pressure_msl=pressure,
        wind_gusts_10m=gusts,
        relative_humidity_700hPa=rh700,
        stencil_min_pressure_msl=min_pressure,
        stencil_max_wind_gusts=max_gusts,
        stencil_max_precipitation=max_precip,
        stencil_max_cape=max_cape,
        stencil_max_rh700=max_rh700,
        stencil_delta_p=delta_p,
        steering_speed_kmh=round(steering_speed, 1),
        steering_dir_deg=round(steering_dir, 1),
        is_live=True,
        source="live_open_meteo",
    )


def _populate_fallback_stencil(p: Dict[str, Any]) -> StencilAtmosphere:
    """Build a StencilAtmosphere from benchmark cube fallback values."""
    # For fallback, derive steering from center-level fallbacks
    u850, v850 = _uv_from_spd_dir(p["fallback_wind_speed_850"], p["fallback_wind_dir_850"])
    u500, v500 = _uv_from_spd_dir(p["fallback_wind_speed_500"], p["fallback_wind_dir_500"])
    u300, v300 = _uv_from_spd_dir(p["fallback_wind_speed_300"], p["fallback_wind_dir_300"])
    u_steer = u850 * 0.25 + u500 * 0.50 + u300 * 0.25
    v_steer = v850 * 0.25 + v500 * 0.50 + v300 * 0.25
    steer_speed, steer_dir = _spd_dir_from_uv(u_steer, v_steer)

    return StencilAtmosphere(
        id=p["id"],
        name=p["name"],
        region=p["region"],
        lat=p["lat"],
        lon=p["lon"],
        cape=float(p["fallback_cape"]),
        cin=float(p["fallback_cin"]),
        precipitation=float(p["fallback_precip"]),
        cloud_cover=float(p["fallback_cloud"]),
        wind_speed_10m=float(p["fallback_wind_speed"]),
        wind_direction_10m=float(p["fallback_wind_dir"]),
        temperature_2m=float(p["fallback_temp"]),
        pressure_msl=float(p["fallback_pressure_msl"]),
        wind_gusts_10m=float(p["fallback_wind_gusts"]),
        relative_humidity_700hPa=float(p["fallback_rh700"]),
        # Fallback stencil extrema = center values (no spatial variability info)
        stencil_min_pressure_msl=float(p["fallback_pressure_msl"]),
        stencil_max_wind_gusts=float(p["fallback_wind_gusts"]),
        stencil_max_precipitation=float(p["fallback_precip"]),
        stencil_max_cape=float(p["fallback_cape"]),
        stencil_max_rh700=float(p["fallback_rh700"]),
        stencil_delta_p=0.0,  # no spatial gradient info in fallback
        steering_speed_kmh=round(steer_speed, 1),
        steering_dir_deg=round(steer_dir, 1),
        is_live=False,
        source="benchmark_cube",
    )


# ---------------------------------------------------------------------------
# LiveWeatherService — Singleton with 5-point stencil batch fetch
# ---------------------------------------------------------------------------
class LiveWeatherService:
    """
    Singleton managing real-time NWP atmospheric conditions across India via
    5-point stencil batch queries to Open-Meteo (no API key required).

    Each of the 18 grid nodes expands to 5 coordinate pairs (center + ±1.0° ring),
    yielding up to 90 coordinate sets per batch request. Stencil extrema capture
    cyclone eyes and convective cores that single-point queries miss.
    Deep-layer steering (850/500/300 hPa weighted average over the 4 ring points)
    cancels internal vortex rotation and gives true environmental storm drift.

    Cache TTL: 10 minutes. Fallback: benchmark atmospheric cube on any error.
    """

    _instance: Optional["LiveWeatherService"] = None

    def __new__(cls) -> "LiveWeatherService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self) -> None:
        self._cache: Dict[str, StencilAtmosphere] = {}
        self._cache_time: float = 0.0
        self._is_live: bool = False
        self._last_error: Optional[str] = None
        self._populate_fallback()

    def _populate_fallback(self) -> None:
        """Loads benchmark cube values for all 18 grid points."""
        self._cache = {p["id"]: _populate_fallback_stencil(p) for p in INDIA_MET_GRID_POINTS}
        self._is_live = False

    def _is_cache_valid(self) -> bool:
        return bool(
            self._cache
            and (time.time() - self._cache_time < CACHE_TTL_SECONDS)
            and self._is_live
        )

    def _build_batch_coords(self) -> Tuple[List[float], List[float]]:
        """
        Build flattened latitude/longitude lists for the full 5-point stencil batch.
        Order: for each of the 18 nodes → [center, N, S, E, W] = 90 total rows.
        """
        lats, lons = [], []
        for p in INDIA_MET_GRID_POINTS:
            for lat, lon in _stencil_coords(p["lat"], p["lon"]):
                lats.append(lat)
                lons.append(lon)
        return lats, lons

    def fetch_live_grid(self, force_refresh: bool = False) -> Dict[str, StencilAtmosphere]:
        """
        Synchronously fetch and cache the 5-point stencil NWP grid.
        Returns cached data within TTL, or benchmark fallback on any error.
        """
        if not force_refresh and self._is_cache_valid():
            return self._cache

        lats, lons = self._build_batch_coords()
        params = {
            "latitude": lats,
            "longitude": lons,
            "current": _CURRENT_VARS,
        }

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(OPEN_METEO_FORECAST_URL, params=params)
            if response.status_code == 200:
                self._parse_and_cache(response.json())
                log.info("Fetched & cached 5-point stencil NWP grid (%d nodes)", len(INDIA_MET_GRID_POINTS))
                return self._cache
            else:
                self._last_error = f"Open-Meteo HTTP {response.status_code}"
                log.warning("Open-Meteo %s — benchmark fallback", response.status_code)
        except Exception as exc:
            self._last_error = str(exc)
            log.warning("Stencil fetch failed (%s) — benchmark fallback", exc)

        if not self._cache or force_refresh:
            self._populate_fallback()
        elif time.time() - self._cache_time >= CACHE_TTL_SECONDS:
            self._is_live = False
        return self._cache

    async def fetch_live_grid_async(self, force_refresh: bool = False) -> Dict[str, StencilAtmosphere]:
        """Async version of fetch_live_grid."""
        if not force_refresh and self._is_cache_valid():
            return self._cache

        lats, lons = self._build_batch_coords()
        params = {
            "latitude": lats,
            "longitude": lons,
            "current": _CURRENT_VARS,
        }

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(OPEN_METEO_FORECAST_URL, params=params)
            if response.status_code == 200:
                self._parse_and_cache(response.json())
                log.info("Async fetched 5-point stencil NWP grid")
                return self._cache
            else:
                self._last_error = f"Open-Meteo HTTP {response.status_code}"
                log.warning("Open-Meteo async %s — benchmark fallback", response.status_code)
        except Exception as exc:
            self._last_error = str(exc)
            log.warning("Async stencil fetch failed (%s) — benchmark fallback", exc)

        if not self._cache or force_refresh:
            self._populate_fallback()
        elif time.time() - self._cache_time >= CACHE_TTL_SECONDS:
            self._is_live = False
        return self._cache

    def _parse_and_cache(self, raw_data: Any) -> None:
        """
        Parse flat Open-Meteo multi-location JSON response into StencilAtmosphere objects.
        Each node occupies 5 consecutive items in the response list (center + N/S/E/W).
        """
        items: List[Dict[str, Any]] = raw_data if isinstance(raw_data, list) else [raw_data]

        new_grid: Dict[str, StencilAtmosphere] = {}
        for i, p in enumerate(INDIA_MET_GRID_POINTS):
            start = i * 5
            new_grid[p["id"]] = _parse_stencil_items(items, start, p)

        self._cache = new_grid
        self._cache_time = time.time()
        self._is_live = True
        self._last_error = None

    def get_nearest_conditions(self, lat: float, lon: float) -> StencilAtmosphere:
        """Find the closest grid node to (lat, lon)."""
        grid = self._cache or self.fetch_live_grid()
        best: Optional[StencilAtmosphere] = None
        min_dist_sq = float("inf")

        for point in grid.values():
            d_lat = lat - point.lat
            d_lon = (lon - point.lon) * math.cos(math.radians((lat + point.lat) / 2.0))
            dist_sq = d_lat * d_lat + d_lon * d_lon
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best = point

        if best is None:
            self._populate_fallback()
            best = next(iter(self._cache.values()))
        return best

    @property
    def is_live(self) -> bool:
        return self._is_live and (time.time() - self._cache_time < CACHE_TTL_SECONDS)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def get_all_points(self) -> List[Dict[str, Any]]:
        """Return all grid nodes as dicts."""
        grid = self._cache or self.fetch_live_grid()
        return [pt.to_dict() for pt in grid.values()]


# Global singleton
live_weather_service = LiveWeatherService()
