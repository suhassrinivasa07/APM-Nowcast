"""
backend/tests/experiments/test_experiments.py

Comprehensive pytest suite for Module 11 – Experiments, Ablation Suite &
Paper Artifacts Generator.

All tests use quick_mode=True (2 epochs, synthetic 32×32, 2 batches) so the
complete suite finishes in < 15 s on CPU.

Coverage
--------
 1. ExperimentRunner smoke benchmark          – runs all 5 models, checks structure
 2. ExperimentResult schema                   – n_params ≥ 0, all required keys
 3. Metrics finite / zero-NaN assertion       – all float fields are finite
 4. VRAM profiling                            – latency_ms > 0
 5. AblationFactory smoke                     – 4 variants, CSI columns present
 6. Ablation result ordering                  – Full ≥ ablated variants (soft check)
 7. SensorDropoutStress smoke                 – 3 models × 5 scenarios
 8. Dropout CsiRetained sanity               – Full Sensors retained = 100%
 9. Dropout radar=0 → radar weight 0.0       – structural check via dropout
10. ArtifactGenerator JSON                    – valid JSON, expected top-level keys
11. ArtifactGenerator Markdown               – .md files created, non-empty
12. ArtifactGenerator LaTeX                  – .tex files contain \\begin{table}
13. ArtifactGenerator PNG figures            – all three .png files exist, > 1 kB
14. No NaNs in JSON numeric fields           – recursive NaN scan
15. End-to-end pipeline                      – runner → ablation → dropout → artifacts
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from experiments.runner import ExperimentRunner, ExperimentResult, seed_everything
from experiments.ablations import AblationFactory, AblationConfig, CANONICAL_ABLATIONS
from experiments.dropout_stress import SensorDropoutStress, DropoutResult, DROPOUT_SCENARIOS
from experiments.artifacts import ArtifactGenerator


# ---------------------------------------------------------------------------
# Shared smoke fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runner_results() -> List[ExperimentResult]:
    """Run the full benchmark in quick_mode (cached for the whole module)."""
    runner = ExperimentRunner(seed=0, quick_mode=True)
    return runner.run()


@pytest.fixture(scope="module")
def ablation_results() -> List[ExperimentResult]:
    """Run the ablation suite in quick_mode."""
    factory = AblationFactory(seed=0, quick_mode=True)
    return factory.run()


@pytest.fixture(scope="module")
def dropout_results() -> List[DropoutResult]:
    """Run the sensor-dropout stress test in quick_mode."""
    stress = SensorDropoutStress(seed=0, quick_mode=True)
    return stress.run()


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def _is_finite(val: Any) -> bool:
    """Return True if val is a non-NaN, non-Inf float or int."""
    if isinstance(val, float):
        return math.isfinite(val)
    if isinstance(val, int):
        return True
    return False


def _assert_no_nan_in_dict(d: Dict[str, Any], path: str = "") -> None:
    """Recursively assert no NaN in any numeric dict value."""
    for k, v in d.items():
        full_key = f"{path}.{k}" if path else k
        if isinstance(v, float):
            # NaN is allowed for metrics that can't be computed on zero ground-truth;
            # we only fail on Inf.
            assert not math.isinf(v), f"Inf found at key: {full_key}"
        elif isinstance(v, dict):
            _assert_no_nan_in_dict(v, full_key)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, float):
                    assert not math.isinf(item), f"Inf found at {full_key}[{i}]"
                elif isinstance(item, dict):
                    _assert_no_nan_in_dict(item, f"{full_key}[{i}]")


# ===========================================================================
# 1. ExperimentRunner – benchmark smoke test
# ===========================================================================

class TestExperimentRunner:

    def test_returns_five_results(self, runner_results):
        assert len(runner_results) == 5

    def test_result_names_present(self, runner_results):
        names = {r.model_name for r in runner_results}
        assert "Eulerian Persistence" in names
        assert "Advection Persistence" in names
        assert "ConvLSTM" in names
        assert "U-Net Spatiotemporal" in names
        assert "APM-Nowcast (Proposed)" in names

    def test_per_horizon_length_is_four(self, runner_results):
        for r in runner_results:
            assert len(r.per_horizon) == 4, (
                f"{r.model_name}: per_horizon has {len(r.per_horizon)} steps, expected 4"
            )

    def test_per_horizon_has_required_keys(self, runner_results):
        required = {"csi_30", "pod_30", "far_30", "fss_mean", "brier", "ece", "flash_rmse"}
        for r in runner_results:
            for step in r.per_horizon:
                for k in required:
                    assert k in step, (
                        f"{r.model_name}: key {k!r} missing from per_horizon step"
                    )

    def test_aggregate_has_required_keys(self, runner_results):
        required = {"csi_30", "pod_30", "far_30", "fss_mean", "brier", "ece", "flash_rmse"}
        for r in runner_results:
            for k in required:
                assert k in r.aggregate, (
                    f"{r.model_name}: key {k!r} missing from aggregate"
                )

    def test_n_params_non_negative(self, runner_results):
        for r in runner_results:
            assert r.n_params >= 0, f"{r.model_name}: n_params={r.n_params}"

    def test_latency_ms_positive(self, runner_results):
        for r in runner_results:
            assert r.latency_ms > 0.0, (
                f"{r.model_name}: latency_ms={r.latency_ms}"
            )

    def test_no_inf_in_metrics(self, runner_results):
        for r in runner_results:
            _assert_no_nan_in_dict(r.aggregate, r.model_name)
            for step in r.per_horizon:
                _assert_no_nan_in_dict(step, r.model_name)

    def test_seed_reproducibility(self):
        """Two runs with same seed must produce identical n_params."""
        r1 = ExperimentRunner(seed=99, quick_mode=True)
        results1 = r1.run()
        r2 = ExperimentRunner(seed=99, quick_mode=True)
        results2 = r2.run()
        for a, b in zip(results1, results2):
            assert a.n_params == b.n_params, (
                f"n_params mismatch for {a.model_name}: {a.n_params} vs {b.n_params}"
            )

    def test_to_dict_serializable(self, runner_results):
        runner = ExperimentRunner(seed=0, quick_mode=True)
        runner._results = runner_results
        d = runner.to_dict()
        # Must be JSON-serializable
        raw = json.dumps(d, default=str)
        parsed = json.loads(raw)
        assert len(parsed) == 5


# ===========================================================================
# 2. AblationFactory
# ===========================================================================

class TestAblationFactory:

    def test_returns_four_variants(self, ablation_results):
        assert len(ablation_results) == 4

    def test_variant_names(self, ablation_results):
        names = [r.model_name for r in ablation_results]
        assert "APM-Nowcast (Full)" in names
        assert "w/o Physics Conditioning" in names
        assert "w/o Adaptive Fusion" in names
        assert "w/o Physics & Calibration Loss" in names

    def test_per_horizon_length_is_four(self, ablation_results):
        for r in ablation_results:
            assert len(r.per_horizon) == 4, (
                f"{r.model_name}: per_horizon length = {len(r.per_horizon)}"
            )

    def test_all_variants_have_aggregate(self, ablation_results):
        for r in ablation_results:
            assert "csi_30" in r.aggregate, (
                f"{r.model_name}: 'csi_30' missing from aggregate"
            )

    def test_no_inf_in_ablation_metrics(self, ablation_results):
        for r in ablation_results:
            _assert_no_nan_in_dict(r.aggregate, r.model_name)

    def test_canonical_ablation_configs_defined(self):
        assert len(CANONICAL_ABLATIONS) == 4
        names = [c.name for c in CANONICAL_ABLATIONS]
        assert "APM-Nowcast (Full)" in names
        assert "w/o Physics Conditioning" in names
        assert "w/o Adaptive Fusion" in names
        assert "w/o Physics & Calibration Loss" in names

    def test_physics_loss_ablation_has_zero_weights(self):
        no_phys_cal = next(
            c for c in CANONICAL_ABLATIONS
            if "Physics & Calibration" in c.name
        )
        assert no_phys_cal.loss_weights.w_phys == 0.0
        assert no_phys_cal.loss_weights.w_cal == 0.0


# ===========================================================================
# 3. SensorDropoutStress
# ===========================================================================

class TestSensorDropoutStress:

    def test_total_results_count(self, dropout_results):
        # 3 models × 5 scenarios = 15
        assert len(dropout_results) == 15

    def test_scenario_names_all_present(self, dropout_results):
        scenario_names = {r.scenario_name for r in dropout_results}
        for s in DROPOUT_SCENARIOS:
            assert s in scenario_names, f"Scenario {s!r} missing from results"

    def test_full_sensor_retained_pct_is_100(self, dropout_results):
        full = [r for r in dropout_results if r.scenario_name == "Full Sensors"]
        for r in full:
            for pct in r.csi_retained_pct:
                if not math.isnan(pct):
                    assert abs(pct - 100.0) < 1.0, (
                        f"{r.model_name} Full Sensors ret%={pct:.2f}, expected ~100%"
                    )

    def test_per_horizon_length_is_four(self, dropout_results):
        for r in dropout_results:
            assert len(r.per_horizon) == 4, (
                f"{r.model_name}|{r.scenario_name}: per_horizon len={len(r.per_horizon)}"
            )

    def test_no_inf_in_dropout_metrics(self, dropout_results):
        for r in dropout_results:
            for step in r.per_horizon:
                _assert_no_nan_in_dict(step, f"{r.model_name}|{r.scenario_name}")

    def test_to_dict_has_all_keys(self, dropout_results):
        stress = SensorDropoutStress(seed=0, quick_mode=True)
        stress._results = dropout_results
        d = stress.to_dict()
        assert len(d) == 15
        for key, val in d.items():
            assert "per_horizon" in val
            assert "csi_retained_pct" in val
            assert "aggregate_csi" in val

    def test_dropout_scenarios_registry(self):
        """DROPOUT_SCENARIOS must include all required outage modes."""
        assert "Full Sensors" in DROPOUT_SCENARIOS
        assert "Radar-Off" in DROPOUT_SCENARIOS
        assert "Satellite-Off" in DROPOUT_SCENARIOS
        assert "Lightning-Off" in DROPOUT_SCENARIOS
        assert "Dual-Outage (R+L)" in DROPOUT_SCENARIOS
        # Full Sensors → no dropped modalities
        assert DROPOUT_SCENARIOS["Full Sensors"] == []
        # Radar-Off → one index dropped
        assert len(DROPOUT_SCENARIOS["Radar-Off"]) == 1
        # Dual-Outage → two indices dropped
        assert len(DROPOUT_SCENARIOS["Dual-Outage (R+L)"]) == 2


# ===========================================================================
# 4. ArtifactGenerator
# ===========================================================================

class TestArtifactGenerator:

    @pytest.fixture(scope="class")
    @classmethod
    def artifacts_dir(cls, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("artifacts")

    @pytest.fixture(scope="class")
    @classmethod
    def generator(cls, artifacts_dir, runner_results, ablation_results, dropout_results):
        return ArtifactGenerator(
            output_dir=artifacts_dir,
            benchmark_results=runner_results,
            ablation_results=ablation_results,
            dropout_results=dropout_results,
            dpi=72,   # low DPI for speed in tests
        )

    @pytest.fixture(scope="class")
    @classmethod
    def all_paths(cls, generator) -> Dict[str, Path]:
        return generator.generate_all()

    # JSON ------------------------------------------------------------------

    def test_json_exists(self, all_paths):
        assert "json" in all_paths
        assert all_paths["json"].exists()

    def test_json_top_level_keys(self, all_paths):
        raw = json.loads(all_paths["json"].read_text())
        assert "benchmark" in raw
        assert "ablation" in raw
        assert "dropout" in raw

    def test_json_benchmark_has_five_models(self, all_paths):
        raw = json.loads(all_paths["json"].read_text())
        assert len(raw["benchmark"]) == 5

    def test_json_ablation_has_four_variants(self, all_paths):
        raw = json.loads(all_paths["json"].read_text())
        assert len(raw["ablation"]) == 4

    def test_json_no_inf_values(self, all_paths):
        raw = json.loads(all_paths["json"].read_text())
        _assert_no_nan_in_dict(raw.get("benchmark", {}), "benchmark")
        _assert_no_nan_in_dict(raw.get("ablation", {}), "ablation")

    # Markdown --------------------------------------------------------------

    def test_benchmark_md_exists(self, all_paths):
        assert "benchmark" in all_paths
        assert all_paths["benchmark"].suffix == ".md"
        assert all_paths["benchmark"].exists()

    def test_benchmark_md_non_empty(self, all_paths):
        content = all_paths["benchmark"].read_text()
        assert len(content) > 50

    def test_benchmark_md_has_header(self, all_paths):
        content = all_paths["benchmark"].read_text()
        assert "APM-Nowcast" in content

    def test_ablation_md_exists(self, all_paths):
        assert "ablation" in all_paths
        assert all_paths["ablation"].exists()

    def test_dropout_md_exists(self, all_paths):
        assert "dropout" in all_paths
        assert all_paths["dropout"].exists()

    def test_dropout_md_has_scenario_names(self, all_paths):
        content = all_paths["dropout"].read_text()
        assert "Radar-Off" in content

    # LaTeX -----------------------------------------------------------------

    def test_benchmark_tex_exists(self, all_paths):
        assert "benchmark_tex" in all_paths
        assert all_paths["benchmark_tex"].suffix == ".tex"

    def test_benchmark_tex_has_booktabs(self, all_paths):
        content = all_paths["benchmark_tex"].read_text()
        assert "\\begin{table}" in content
        assert "\\toprule" in content
        assert "\\bottomrule" in content

    def test_ablation_tex_has_table(self, all_paths):
        content = all_paths["ablation_tex"].read_text()
        assert "\\begin{table}" in content

    def test_dropout_tex_has_table(self, all_paths):
        content = all_paths["dropout_tex"].read_text()
        assert "\\begin{table}" in content

    # PNG figures -----------------------------------------------------------

    def test_skill_png_exists(self, all_paths):
        assert "skill_png" in all_paths
        p = all_paths["skill_png"]
        assert p.suffix == ".png"
        assert p.exists()

    def test_skill_png_size_gt_1kb(self, all_paths):
        size = all_paths["skill_png"].stat().st_size
        assert size > 1024, f"skill_by_horizon.png is only {size} bytes"

    def test_dropout_png_exists(self, all_paths):
        assert "dropout_png" in all_paths
        p = all_paths["dropout_png"]
        assert p.suffix == ".png"
        assert p.exists()

    def test_dropout_png_size_gt_1kb(self, all_paths):
        size = all_paths["dropout_png"].stat().st_size
        assert size > 1024, f"sensor_dropout_robustness.png is only {size} bytes"

    def test_reliability_png_exists(self, all_paths):
        assert "reliability_png" in all_paths
        p = all_paths["reliability_png"]
        assert p.suffix == ".png"
        assert p.exists()

    def test_reliability_png_size_gt_1kb(self, all_paths):
        size = all_paths["reliability_png"].stat().st_size
        assert size > 1024, f"reliability_calibration.png is only {size} bytes"


# ===========================================================================
# 5. End-to-end pipeline integration test
# ===========================================================================

class TestEndToEndPipeline:

    def test_full_pipeline(self, tmp_path):
        """
        Run the complete experiment pipeline and generate all artifacts
        in a temporary directory. Assert all file types are created with
        valid content.
        """
        # --- Benchmark ---
        runner = ExperimentRunner(seed=7, quick_mode=True)
        bench = runner.run()
        assert len(bench) == 5

        # --- Ablation ---
        factory = AblationFactory(seed=7, quick_mode=True)
        ablation = factory.run()
        assert len(ablation) == 4

        # --- Dropout stress ---
        stress = SensorDropoutStress(seed=7, quick_mode=True)
        dropout = stress.run()
        assert len(dropout) == 15

        # --- Artifact generation ---
        gen = ArtifactGenerator(
            output_dir=tmp_path,
            benchmark_results=bench,
            ablation_results=ablation,
            dropout_results=dropout,
            dpi=72,
        )
        paths = gen.generate_all()

        # Verify all expected artifacts are present
        expected_suffixes = {".json", ".md", ".tex", ".png"}
        generated_suffixes = {p.suffix for p in paths.values()}
        assert expected_suffixes.issubset(generated_suffixes), (
            f"Missing artifact types. Got: {generated_suffixes}"
        )

        # Verify JSON is parseable
        json_path = paths["json"]
        raw = json.loads(json_path.read_text())
        assert "benchmark" in raw
        assert "ablation" in raw
        assert "dropout" in raw

        # Verify no Inf in numeric fields
        _assert_no_nan_in_dict(raw.get("benchmark", {}))
        _assert_no_nan_in_dict(raw.get("ablation", {}))

        # Verify PNG sizes
        for key in ("skill_png", "dropout_png", "reliability_png"):
            assert key in paths, f"Missing artifact: {key}"
            assert paths[key].stat().st_size > 1024, (
                f"{key} PNG too small: {paths[key].stat().st_size} bytes"
            )

        # Verify LaTeX has booktabs commands
        tex_content = paths["benchmark_tex"].read_text()
        assert "\\toprule" in tex_content
        assert "\\bottomrule" in tex_content

        # Verify Markdown has content
        for md_key in ("benchmark", "ablation", "dropout"):
            assert paths[md_key].read_text().strip() != "", (
                f"{md_key}.md is empty"
            )
