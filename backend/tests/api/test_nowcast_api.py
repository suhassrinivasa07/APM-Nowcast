"""
backend/tests/api/test_nowcast_api.py

Integration tests for Module 10 – Real-Time Inference, Explainability &
Scenario Simulation API endpoints.

All tests use the shared ``client`` fixture from conftest.py which triggers
the FastAPI lifespan (initialises NowcastInferenceService) automatically.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper: assert a dict has all required keys
# ---------------------------------------------------------------------------

def _has_keys(d: dict, *keys: str) -> None:
    for k in keys:
        assert k in d, f"Key {k!r} missing from response: {list(d.keys())}"


# ---------------------------------------------------------------------------
# Existing endpoints must still pass (regression)
# ---------------------------------------------------------------------------

class TestHealthRegression:
    def test_root_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_api_v1_health(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# GET /api/v1/forecast/latest
# ---------------------------------------------------------------------------

class TestForecastLatest:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/forecast/latest")
        assert resp.status_code == 200

    def test_snapshot_shape(self, client):
        resp = client.get("/api/v1/forecast/latest")
        data = resp.json()
        _has_keys(data, "horizon", "cells", "attentionWeights", "thermodynamics",
                  "stations", "alerts", "timestamp", "uncertaintyInflationFactor")

    def test_horizon_step_zero(self, client):
        resp = client.get("/api/v1/forecast/latest")
        data = resp.json()
        assert data["horizon"]["step"] == 0
        assert data["horizon"]["minutes"] == 15

    def test_cells_non_empty(self, client):
        resp = client.get("/api/v1/forecast/latest")
        data = resp.json()
        assert len(data["cells"]) > 0

    def test_cell_occurrence_prob_in_range(self, client):
        resp = client.get("/api/v1/forecast/latest")
        data = resp.json()
        for cell in data["cells"]:
            assert 0.0 <= cell["occurrenceProb"] <= 1.0, (
                f"occurrenceProb={cell['occurrenceProb']} out of range"
            )

    def test_attention_weights_sum_to_one(self, client):
        resp = client.get("/api/v1/forecast/latest")
        data = resp.json()
        aw = data["attentionWeights"]
        total = aw["satellite"] + aw["radar"] + aw["lightning"] + aw["nwp"]
        assert abs(total - 1.0) < 1e-3, f"Attention weights sum={total:.6f} != 1.0"

    def test_thermodynamics_shape(self, client):
        resp = client.get("/api/v1/forecast/latest")
        data = resp.json()
        thermo = data["thermodynamics"]
        _has_keys(thermo, "cape", "cin", "moistureConvergence",
                  "filmGateActivation", "liftedIndex", "shear06km", "convectiveRegime")


# ---------------------------------------------------------------------------
# POST /api/v1/predict
# ---------------------------------------------------------------------------

class TestPredict:
    _PAYLOAD_ALL_ON = {
        "horizon": 0,
        "satellite": True,
        "radar": True,
        "lightning": True,
        "nwp": True,
    }

    def test_returns_200(self, client):
        resp = client.post("/api/v1/predict", json=self._PAYLOAD_ALL_ON)
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        resp = client.post("/api/v1/predict", json=self._PAYLOAD_ALL_ON)
        data = resp.json()
        _has_keys(data,
                  "lightning_probs_mean", "lightning_probs_max",
                  "flash_rate_mean", "flash_rate_max",
                  "aleatoric_uncertainty_mean", "epistemic_uncertainty_mean",
                  "total_uncertainty_mean", "attention_weights",
                  "physics_gate_stats", "uncertainty_inflation_factor",
                  "inference_device", "inference_ms",
                  "horizon", "timestamp")

    def test_lightning_probs_mean_length_is_t_out(self, client):
        resp = client.post("/api/v1/predict", json=self._PAYLOAD_ALL_ON)
        data = resp.json()
        # T_out=4 (default model config)
        assert len(data["lightning_probs_mean"]) == 4

    def test_inference_ms_positive(self, client):
        resp = client.post("/api/v1/predict", json=self._PAYLOAD_ALL_ON)
        data = resp.json()
        assert data["inference_ms"] > 0.0

    def test_attention_weights_present(self, client):
        resp = client.post("/api/v1/predict", json=self._PAYLOAD_ALL_ON)
        data = resp.json()
        aw = data["attention_weights"]
        _has_keys(aw, "satellite", "radar", "lightning", "nwp")

    def test_physics_gate_stats_present(self, client):
        resp = client.post("/api/v1/predict", json=self._PAYLOAD_ALL_ON)
        data = resp.json()
        phys = data["physics_gate_stats"]
        _has_keys(phys, "film_scale_mean", "film_bias_mean", "cape_activation")

    def test_horizon_in_response(self, client):
        resp = client.post("/api/v1/predict", json={**self._PAYLOAD_ALL_ON, "horizon": 2})
        data = resp.json()
        assert data["horizon"]["step"] == 2
        assert data["horizon"]["minutes"] == 45


# ---------------------------------------------------------------------------
# POST /api/v1/scenarios/dropout  — radar=False must zero radar weight
# ---------------------------------------------------------------------------

class TestScenariosDropout:
    def test_returns_200(self, client):
        resp = client.post("/api/v1/scenarios/dropout", json={
            "horizon": 0,
            "satellite": True,
            "radar": False,
            "lightning": True,
            "nwp": True,
        })
        assert resp.status_code == 200

    def test_radar_dropout_zeros_radar_attention(self, client):
        resp = client.post("/api/v1/scenarios/dropout", json={
            "horizon": 0,
            "satellite": True,
            "radar": False,
            "lightning": True,
            "nwp": True,
        })
        data = resp.json()
        aw = data["attention_weights"]
        assert aw["radar"] == 0.0, (
            f"Expected radar=0.0 after dropout, got {aw['radar']}"
        )

    def test_remaining_weights_sum_to_one(self, client):
        resp = client.post("/api/v1/scenarios/dropout", json={
            "horizon": 0,
            "satellite": True,
            "radar": False,
            "lightning": True,
            "nwp": True,
        })
        data = resp.json()
        aw = data["attention_weights"]
        total = aw["satellite"] + aw["radar"] + aw["lightning"] + aw["nwp"]
        assert abs(total - 1.0) < 1e-3, (
            f"Attention weights sum={total:.6f} != 1.0 after dropout"
        )

    def test_response_has_cells_and_alerts(self, client):
        resp = client.post("/api/v1/scenarios/dropout", json={
            "horizon": 1,
            "satellite": True,
            "radar": False,
            "lightning": True,
            "nwp": True,
        })
        data = resp.json()
        _has_keys(data, "cells", "alerts", "scenario", "horizon",
                  "attention_weights", "uncertainty_inflation_factor", "timestamp")

    def test_all_modalities_dropped_fallback(self, client):
        """When all sensors are off the service falls back to equal weights."""
        resp = client.post("/api/v1/scenarios/dropout", json={
            "horizon": 0,
            "satellite": False,
            "radar": False,
            "lightning": False,
            "nwp": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        aw = data["attention_weights"]
        total = aw["satellite"] + aw["radar"] + aw["lightning"] + aw["nwp"]
        assert abs(total - 1.0) < 1e-3, (
            f"All-off fallback weights sum={total:.6f} != 1.0"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks/metrics
# ---------------------------------------------------------------------------

class TestBenchmarksMetrics:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/benchmarks/metrics")
        assert resp.status_code == 200

    def test_at_least_four_models(self, client):
        resp = client.get("/api/v1/benchmarks/metrics")
        data = resp.json()
        assert len(data) >= 4, f"Expected >=4 benchmark entries, got {len(data)}"

    def test_first_entry_schema(self, client):
        resp = client.get("/api/v1/benchmarks/metrics")
        data = resp.json()
        entry = data[0]
        _has_keys(entry, "modelName", "csi", "pod", "far", "fss",
                  "brier", "ece", "flashRmse")

    def test_csi_has_four_horizons(self, client):
        resp = client.get("/api/v1/benchmarks/metrics")
        data = resp.json()
        for entry in data:
            assert len(entry["csi"]) == 4, (
                f"CSI for {entry['modelName']} has {len(entry['csi'])} steps, expected 4"
            )


# ---------------------------------------------------------------------------
# GET /api/v1/verification/benchmarks  (frontend alias)
# ---------------------------------------------------------------------------

class TestVerificationBenchmarks:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/verification/benchmarks")
        assert resp.status_code == 200

    def test_same_data_as_benchmarks_metrics(self, client):
        resp_a = client.get("/api/v1/benchmarks/metrics")
        resp_b = client.get("/api/v1/verification/benchmarks")
        assert resp_a.json() == resp_b.json()


# ---------------------------------------------------------------------------
# GET /api/v1/nowcast/snapshot
# ---------------------------------------------------------------------------

class TestNowcastSnapshot:
    def test_returns_200_all_sensors(self, client):
        resp = client.get(
            "/api/v1/nowcast/snapshot",
            params={"horizon": 0, "satellite": True, "radar": True,
                    "lightning": True, "nwp": True},
        )
        assert resp.status_code == 200

    def test_snapshot_shape(self, client):
        resp = client.get("/api/v1/nowcast/snapshot", params={"horizon": 0})
        data = resp.json()
        _has_keys(data, "horizon", "cells", "attentionWeights", "thermodynamics",
                  "stations", "alerts", "timestamp", "uncertaintyInflationFactor")

    def test_horizon_step_matches_query(self, client):
        for step in range(4):
            resp = client.get("/api/v1/nowcast/snapshot", params={"horizon": step})
            assert resp.status_code == 200
            data = resp.json()
            assert data["horizon"]["step"] == step

    def test_attention_weights_sum_to_one(self, client):
        resp = client.get("/api/v1/nowcast/snapshot",
                          params={"horizon": 0, "satellite": True, "radar": True,
                                  "lightning": True, "nwp": True})
        data = resp.json()
        aw = data["attentionWeights"]
        total = aw["satellite"] + aw["radar"] + aw["lightning"] + aw["nwp"]
        assert abs(total - 1.0) < 1e-3, f"Attention weights sum={total:.6f} != 1.0"

    def test_radar_dropout_zeros_radar_weight(self, client):
        resp = client.get("/api/v1/nowcast/snapshot",
                          params={"horizon": 0, "satellite": True, "radar": False,
                                  "lightning": True, "nwp": True})
        data = resp.json()
        aw = data["attentionWeights"]
        assert aw["radar"] == 0.0, (
            f"Expected radar=0.0 when radar=False, got {aw['radar']}"
        )

    def test_cells_non_empty(self, client):
        resp = client.get("/api/v1/nowcast/snapshot", params={"horizon": 1})
        data = resp.json()
        assert len(data["cells"]) > 0


# ---------------------------------------------------------------------------
# POST /api/v1/explain/modalities
# ---------------------------------------------------------------------------

class TestExplainModalities:
    _PAYLOAD = {
        "horizon": 1,
        "satellite": True,
        "radar": True,
        "lightning": True,
        "nwp": True,
    }

    def test_returns_200(self, client):
        resp = client.post("/api/v1/explain/modalities", json=self._PAYLOAD)
        assert resp.status_code == 200

    def test_attention_weights_present(self, client):
        resp = client.post("/api/v1/explain/modalities", json=self._PAYLOAD)
        data = resp.json()
        _has_keys(data, "attention_weights", "physics_gate_stats",
                  "horizon", "timestamp", "uncertainty_inflation_factor",
                  "modality_descriptions")

    def test_attention_weights_keys(self, client):
        resp = client.post("/api/v1/explain/modalities", json=self._PAYLOAD)
        data = resp.json()
        aw = data["attention_weights"]
        _has_keys(aw, "satellite", "radar", "lightning", "nwp")

    def test_modality_descriptions_dict(self, client):
        resp = client.post("/api/v1/explain/modalities", json=self._PAYLOAD)
        data = resp.json()
        desc = data["modality_descriptions"]
        assert isinstance(desc, dict)
        assert len(desc) >= 4
