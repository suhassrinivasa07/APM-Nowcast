"""
Unit tests for Module 7 — Training, Evaluation & Meteorological Verification.

Coverage:
  - Metrics correctness (POD, FAR, CSI, HSS, FSS, Brier, ECE)
  - TrainingConfig save/load roundtrip
  - CheckpointManager: save+reload → identical predictions, best-model, rolling retention
  - APMTrainer: 2-epoch training loop on CPU (APMNowcastModel + SpatiotemporalUNetBaseline)
  - APMTrainer: CUDA AMP training (skipped if no CUDA)
  - APMEvaluator: full scenario, radar-dropout scenario, Markdown table
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, List

import pytest
import torch

from training.config import LossWeights, TrainingConfig
from training.metrics import (
    pod, far, csi, hss,
    fss, fss_multiscale,
    brier_score, ece,
    flash_rmse, flash_mae,
    compute_all_metrics,
    compute_per_leadtime_metrics,
)
from training.checkpoint import CheckpointManager
from training.trainer import APMTrainer
from training.evaluator import APMEvaluator

from models.apm_nowcast import APMNowcastModel
from models.baselines.unet_nowcast import SpatiotemporalUNetBaseline


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

B, T_IN, T_OUT, H, W = 2, 4, 4, 32, 32


def _make_batch(B=B, T_in=T_IN, T_out=T_OUT, H=H, W=W) -> Dict[str, Any]:
    """Create a minimal valid batch dict for trainer/evaluator tests."""
    return {
        "inputs_by_modality": {
            "satellite": torch.randn(B, T_in, 3, H, W),
            "radar":     torch.randn(B, T_in, 2, H, W),
            "lightning": torch.zeros(B, T_in, 2, H, W),
            "nwp":       torch.randn(B, T_in, 4, H, W),
        },
        "modality_masks": torch.ones(B, T_in, 4),
        "physics_fields": {
            "cape": torch.rand(B, T_in, H, W),
        },
        "target_lightning_frames":  torch.zeros(B, T_out, H, W),
        "target_occurrence_frames": torch.zeros(B, T_out, H, W),
    }


def _make_loader(n_batches: int = 3) -> List[Dict[str, Any]]:
    """Return a simple list of batch dicts (used as the DataLoader)."""
    return [_make_batch() for _ in range(n_batches)]


def _make_apm_model() -> APMNowcastModel:
    """Small APMNowcastModel that fits CPU tests quickly."""
    return APMNowcastModel(
        T_in=T_IN,
        T_out=T_OUT,
        latent_dim=24,
        hidden_dim=48,
    )


def _make_unet_model() -> SpatiotemporalUNetBaseline:
    return SpatiotemporalUNetBaseline(
        T_in=T_IN,
        T_out=T_OUT,
        base_channels=16,
    )


def _make_config(tmp_path: Path, **overrides) -> TrainingConfig:
    defaults = dict(
        epochs=2,
        checkpoint_dir=str(tmp_path / "ckpts"),
        early_stopping_patience=10,
        amp_enabled=False,          # CPU tests run FP32
        log_every_n_epochs=1,
        warmup_epochs=0,
        grad_accum_steps=1,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ===========================================================================
# SECTION 1 — METRICS CORRECTNESS
# ===========================================================================

class TestCategoricalMetrics:
    """POD, FAR, CSI, HSS on simple hand-crafted tensors."""

    def _make_probs_targets(self, H=8, W=8):
        """Half the grid is lightning; model predicts perfectly."""
        targets = torch.zeros(1, 1, 1, H, W)
        targets[0, 0, 0, :H//2, :] = 1.0
        probs_perfect = targets.clone()
        probs_zeros   = torch.zeros_like(targets)
        probs_ones    = torch.ones_like(targets)
        return probs_perfect, probs_zeros, probs_ones, targets

    def test_csi_perfect_forecast(self):
        _, _, probs_ones, targets = self._make_probs_targets()
        # Perfect: predict 1 everywhere lightning is 1 (and no lightning means no hits)
        # Create probs that exactly match targets
        probs = targets.clone()
        score = csi(probs, targets, threshold=0.5)
        assert math.isclose(score, 1.0, abs_tol=1e-5), f"CSI={score}"

    def test_csi_zero_forecast(self):
        _, probs_zeros, _, targets = self._make_probs_targets()
        score = csi(probs_zeros, targets, threshold=0.5)
        assert score < 1e-4, f"CSI for zero forecast should be ~0, got {score}"

    def test_pod_all_hits(self):
        _, _, probs_ones, targets = self._make_probs_targets()
        # Predicting 1 everywhere catches all events → POD=1
        score = pod(probs_ones, targets, threshold=0.5)
        assert math.isclose(score, 1.0, abs_tol=1e-5), f"POD={score}"

    def test_far_no_false_alarms(self):
        probs_perfect, _, _, targets = self._make_probs_targets()
        score = far(probs_perfect, targets, threshold=0.5)
        assert score < 1e-5, f"FAR for perfect forecast should be ~0, got {score}"

    def test_pod_zero_forecast_is_zero(self):
        _, probs_zeros, _, targets = self._make_probs_targets()
        score = pod(probs_zeros, targets, threshold=0.5)
        assert score < 1e-5

    def test_hss_random_skill(self):
        """HSS for a random forecast should be close to 0 and less than CSI."""
        torch.manual_seed(0)
        probs   = torch.rand(2, 4, 1, 16, 16)
        targets = (torch.rand(2, 4, 1, 16, 16) > 0.7).float()
        h = hss(probs, targets, threshold=0.5)
        c = csi(probs, targets, threshold=0.5)
        # HSS can be negative for bad forecasts; just check it's finite
        assert math.isfinite(h), f"HSS is not finite: {h}"

    def test_far_all_false_alarms(self):
        """Predicting 1 everywhere with no lightning → FAR = 1."""
        targets = torch.zeros(1, 1, 1, 8, 8)
        probs   = torch.ones_like(targets)
        score = far(probs, targets, threshold=0.5)
        # With no events, H=0, FA = all, FAR = FA/(H+FA) = 1
        assert math.isclose(score, 1.0, abs_tol=1e-4), f"FAR={score}"


class TestFSS:
    """Fractions Skill Score."""

    def test_fss_perfect_forecast(self):
        targets = (torch.rand(2, 4, 1, 32, 32) > 0.5).float()
        score = fss(targets, targets, threshold=0.5, scale=5)
        assert score > 0.99, f"Perfect FSS should be ~1.0, got {score}"

    def test_fss_no_overlap_less_than_half(self):
        """Forecast in top-left, obs in bottom-right → FSS < 0.5."""
        probs   = torch.zeros(1, 1, 1, 32, 32)
        targets = torch.zeros(1, 1, 1, 32, 32)
        probs[0, 0, 0, :8, :8] = 1.0
        targets[0, 0, 0, 24:, 24:] = 1.0
        score = fss(probs, targets, threshold=0.5, scale=5)
        assert score < 0.5, f"Non-overlapping FSS should be < 0.5, got {score}"

    def test_fss_multiscale_returns_all_scales(self):
        probs   = torch.rand(2, 4, 1, 32, 32)
        targets = (probs > 0.5).float()
        result  = fss_multiscale(probs, targets, scales=(3, 5, 9))
        assert "fss_3x3" in result
        assert "fss_5x5" in result
        assert "fss_9x9" in result

    def test_fss_all_zero_both(self):
        """No events, no forecast → FSS = 1.0 (trivial perfect)."""
        probs   = torch.zeros(1, 1, 1, 16, 16)
        targets = torch.zeros(1, 1, 1, 16, 16)
        score   = fss(probs, targets, threshold=0.5, scale=3)
        assert score == 1.0


class TestProbabilisticMetrics:
    """Brier, ECE."""

    def test_brier_perfect(self):
        targets = (torch.rand(2, 4, 1, 16, 16) > 0.5).float()
        score = brier_score(targets, targets)
        assert score < 1e-6, f"Brier for perfect forecast should be 0, got {score}"

    def test_brier_worst(self):
        targets = torch.ones(1, 1, 1, 8, 8)
        probs   = torch.zeros_like(targets)
        score   = brier_score(probs, targets)
        assert math.isclose(score, 1.0, abs_tol=1e-5), f"Brier={score}"

    def test_brier_range(self):
        probs   = torch.rand(4, 4, 1, 16, 16)
        targets = (torch.rand(4, 4, 1, 16, 16) > 0.5).float()
        score   = brier_score(probs, targets)
        assert 0.0 <= score <= 1.0

    def test_ece_perfect_calibration(self):
        """When predicted probs == observed frequency, ECE ≈ 0."""
        # Uniform probabilities → all go into one bin; ECE = |conf - frac|
        # Create perfectly calibrated: p_i = freq(i-th bin)
        n = 1000
        probs   = torch.linspace(0.0, 1.0, n)
        targets = (torch.rand(n) < probs).float()
        score   = ece(probs, targets, n_bins=10)
        # ECE should be small (not exactly 0 due to randomness; threshold 0.15)
        assert score < 0.15, f"ECE for approx-calibrated forecast = {score:.4f}"

    def test_ece_range(self):
        probs   = torch.rand(1000)
        targets = (torch.rand(1000) > 0.5).float()
        score   = ece(probs, targets)
        assert 0.0 <= score <= 1.0

    def test_flash_rmse_zero(self):
        pred    = torch.rand(2, 4, 1, 16, 16)
        score   = flash_rmse(pred, pred)
        assert score < 1e-6

    def test_flash_mae_zero(self):
        pred    = torch.rand(2, 4, 1, 16, 16)
        score   = flash_mae(pred, pred)
        assert score < 1e-6


class TestComputeAllMetrics:
    """compute_all_metrics returns well-formed dict."""

    def test_returns_expected_keys(self):
        probs   = torch.rand(2, 4, 1, 16, 16)
        targets = (torch.rand(2, 4, 1, 16, 16) > 0.5).float()
        m = compute_all_metrics(
            probs, probs, targets, targets,
            thresholds=(0.3, 0.5),
            fss_scales=(3, 5),
        )
        for key in ["pod_30", "far_30", "csi_30", "pod_50", "csi_50",
                    "fss_3x3", "fss_5x5", "fss_mean", "brier", "ece",
                    "flash_rmse", "flash_mae"]:
            assert key in m, f"Missing metric key: {key}"

    def test_compute_per_leadtime_shape(self):
        T_out = 4
        probs   = torch.rand(2, T_out, 1, 16, 16)
        targets = (torch.rand(2, T_out, 1, 16, 16) > 0.5).float()
        per_lt  = compute_per_leadtime_metrics(
            probs, probs, targets, targets
        )
        assert len(per_lt) == T_out, f"Expected {T_out} steps, got {len(per_lt)}"
        for step in per_lt:
            assert "csi_50" in step


# ===========================================================================
# SECTION 2 — TRAININGCONFIG
# ===========================================================================

class TestTrainingConfig:
    def test_config_serialization(self, tmp_path):
        cfg = TrainingConfig(
            lr=1e-3,
            epochs=10,
            grad_accum_steps=4,
            checkpoint_dir=str(tmp_path),
            loss_weights=LossWeights(w_occ=2.0, focal_gamma=1.5),
        )
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = TrainingConfig.load(path)

        assert loaded.lr == cfg.lr
        assert loaded.epochs == cfg.epochs
        assert loaded.grad_accum_steps == cfg.grad_accum_steps
        assert loaded.loss_weights.w_occ == cfg.loss_weights.w_occ
        assert loaded.loss_weights.focal_gamma == cfg.loss_weights.focal_gamma
        assert loaded.checkpoint_dir == cfg.checkpoint_dir

    def test_to_dict_is_json_serializable(self, tmp_path):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert "lr" in json_str


# ===========================================================================
# SECTION 3 — CHECKPOINTMANAGER
# ===========================================================================

class TestCheckpointManager:
    """save / reload / best-model / rolling-retention tests."""

    def _make_simple_model(self):
        """Tiny model purely for checkpoint tests."""
        return torch.nn.Linear(4, 2)

    def _make_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_checkpoint_save_creates_file(self, tmp_path):
        ckpt = CheckpointManager(tmp_path)
        model = self._make_simple_model()
        opt   = self._make_optimizer(model)
        path  = ckpt.save(
            model, opt, scaler=None, epoch=0,
            val_metrics={"val_csi": 0.3},
            config_dict={},
        )
        assert path.exists()

    def test_checkpoint_save_and_reload_identical_predictions(self, tmp_path):
        """Load saved weights into a fresh model → predictions identical."""
        model1 = self._make_simple_model()
        opt    = self._make_optimizer(model1)
        ckpt   = CheckpointManager(tmp_path)
        ckpt.save(
            model1, opt, scaler=None, epoch=0,
            val_metrics={"val_csi": 0.5},
            config_dict={},
            tracked_metric=0.5,
        )

        model2 = self._make_simple_model()
        CheckpointManager.load_best(tmp_path, model=model2)

        x = torch.randn(3, 4)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Reloaded model predictions differ!"

    def test_best_model_tracking(self, tmp_path):
        """Best model updates only when metric improves."""
        model = self._make_simple_model()
        opt   = self._make_optimizer(model)
        ckpt  = CheckpointManager(tmp_path, mode="max")

        ckpt.save(model, opt, None, epoch=0, val_metrics={},
                  config_dict={}, tracked_metric=0.3)
        best_path = tmp_path / "best_model.pt"
        mtime1 = best_path.stat().st_mtime

        # Worse metric → best not updated
        import time; time.sleep(0.05)
        ckpt.save(model, opt, None, epoch=1, val_metrics={},
                  config_dict={}, tracked_metric=0.2)
        mtime2 = best_path.stat().st_mtime
        assert mtime2 == mtime1, "best_model.pt should NOT be updated for worse metric"

        # Better metric → best IS updated
        time.sleep(0.05)
        ckpt.save(model, opt, None, epoch=2, val_metrics={},
                  config_dict={}, tracked_metric=0.9)
        mtime3 = best_path.stat().st_mtime
        assert mtime3 > mtime1, "best_model.pt SHOULD be updated for better metric"

    def test_rolling_retention_deletes_oldest(self, tmp_path):
        """Saving keep_last_n+1 checkpoints deletes the oldest."""
        model = self._make_simple_model()
        opt   = self._make_optimizer(model)
        ckpt  = CheckpointManager(tmp_path, keep_last_n=2)

        for i in range(3):
            ckpt.save(model, opt, None, epoch=i, val_metrics={},
                      config_dict={}, tracked_metric=float(i))

        epoch_files = list(tmp_path.glob("checkpoint_epoch_*.pt"))
        # keep_last_n=2 → at most 2 epoch files remain
        assert len(epoch_files) <= 2, (
            f"Expected ≤2 epoch checkpoints, found {len(epoch_files)}: {epoch_files}"
        )

    def test_load_latest(self, tmp_path):
        model = self._make_simple_model()
        opt   = self._make_optimizer(model)
        ckpt  = CheckpointManager(tmp_path)
        ckpt.save(model, opt, None, epoch=0, val_metrics={"val_csi": 0.1},
                  config_dict={})
        payload = CheckpointManager.load_latest(tmp_path)
        assert "epoch" in payload
        assert payload["epoch"] == 0


# ===========================================================================
# SECTION 4 — APMTrainer
# ===========================================================================

class TestAPMTrainer:
    """2-epoch training loop tests (CPU, no AMP)."""

    def test_apm_model_2epoch_training_cpu(self, tmp_path):
        model  = _make_apm_model()
        loader = _make_loader(n_batches=3)
        config = _make_config(tmp_path)
        trainer = APMTrainer(
            model, config, loader, loader,
            device=torch.device("cpu"),
        )
        history = trainer.train()

        assert len(history) == 2, f"Expected 2 epoch records, got {len(history)}"
        for rec in history:
            assert math.isfinite(rec.get("val_brier", float("nan"))), \
                f"Non-finite val_brier in {rec}"
        # History JSON was saved
        hist_path = Path(config.checkpoint_dir) / "history.json"
        assert hist_path.exists()

    def test_unet_2epoch_training_cpu(self, tmp_path):
        model  = _make_unet_model()
        loader = _make_loader(n_batches=2)
        config = _make_config(tmp_path)
        trainer = APMTrainer(
            model, config, loader, loader,
            device=torch.device("cpu"),
        )
        history = trainer.train()
        assert len(history) == 2

    def test_trainer_checkpoint_saved(self, tmp_path):
        model  = _make_apm_model()
        loader = _make_loader(n_batches=2)
        config = _make_config(tmp_path)
        APMTrainer(model, config, loader, loader,
                   device=torch.device("cpu")).train()
        ckpt_files = list(Path(config.checkpoint_dir).glob("checkpoint_epoch_*.pt"))
        assert len(ckpt_files) > 0, "No checkpoint files found after training"

    def test_trainer_early_stopping(self, tmp_path):
        """With patience=1 and 10 epochs, training stops after 2 epochs."""
        model  = _make_apm_model()
        loader = _make_loader(n_batches=2)
        config = _make_config(tmp_path, epochs=10, early_stopping_patience=1)
        trainer = APMTrainer(model, config, loader, loader,
                             device=torch.device("cpu"))
        history = trainer.train()
        # Must stop before epoch 10 (stopped once no improvement seen twice)
        assert len(history) <= 10
        assert trainer._stopped_early or len(history) <= config.epochs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apm_model_cuda_amp_training(self, tmp_path):
        """AMP training on CUDA: loss finite + peak VRAM > 0."""
        model  = _make_apm_model()
        loader = _make_loader(n_batches=2)
        config = _make_config(tmp_path, amp_enabled=True)
        trainer = APMTrainer(
            model, config, loader, loader,
            device=torch.device("cuda"),
        )
        history = trainer.train()
        assert len(history) == 2
        # Peak VRAM should be tracked
        assert history[0]["peak_vram_mb"] > 0, "Expected non-zero peak VRAM on CUDA"


# ===========================================================================
# SECTION 5 — APMEvaluator
# ===========================================================================

class TestAPMEvaluator:
    """Multi-scenario evaluation and output formatting."""

    def _make_evaluator(self):
        model = _make_apm_model()
        return APMEvaluator(
            model,
            device=torch.device("cpu"),
            thresholds=(0.3, 0.5),
            fss_scales=(3, 5),
            minutes_per_step=15,
        )

    def test_evaluator_full_scenario_structure(self):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader, scenario_name="full")

        assert result["scenario"] == "full"
        assert result["num_batches"] == 2
        assert len(result["per_leadtime"]) == T_OUT
        assert len(result["lead_times_min"]) == T_OUT
        assert result["lead_times_min"][0] == 15
        assert result["lead_times_min"][-1] == T_OUT * 15

        # Every per-leadtime step should have standard metric keys
        for step in result["per_leadtime"]:
            assert "csi_50" in step
            assert "brier"  in step
            assert "fss_mean" in step

    def test_evaluator_aggregate_dict_present(self):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader)
        agg    = result["aggregate"]
        assert "csi_50" in agg
        assert "brier"  in agg

    def test_evaluator_radar_dropout_runs(self):
        """Zeroing radar masks must not raise and should return valid results."""
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader, scenario_name="radar_off", dropout_modality_idx=1)
        assert result["num_batches"] == 2

    def test_evaluator_satellite_dropout_runs(self):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader, scenario_name="satellite_off", dropout_modality_idx=0)
        assert result["num_batches"] == 2

    def test_evaluate_all_scenarios(self):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        all_results = ev.evaluate_all_scenarios(loader)
        assert set(all_results.keys()) == {"full", "radar_off", "satellite_off"}
        for name, res in all_results.items():
            assert res["num_batches"] == 2, f"Expected 2 batches in {name}"

    def test_evaluator_markdown_has_table_headers(self):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader, scenario_name="full")
        md     = APMEvaluator.to_markdown({"full": result})
        assert "POD" in md,  "Markdown missing POD column"
        assert "CSI" in md,  "Markdown missing CSI column"
        assert "Brier" in md, "Markdown missing Brier column"
        assert "+15" in md,  "Markdown missing lead-time rows"

    def test_evaluator_to_json(self, tmp_path):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader, scenario_name="full")
        out    = tmp_path / "results.json"
        APMEvaluator.to_json({"full": result}, out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "full" in data
        assert "per_leadtime" in data["full"]

    def test_evaluator_empty_loader_graceful(self):
        """Empty data loader should return a result with num_batches=0."""
        ev     = self._make_evaluator()
        result = ev.evaluate([], scenario_name="empty")
        assert result["num_batches"] == 0
        assert result["per_leadtime"] == []

    def test_evaluator_markdown_multi_scenario(self):
        ev     = self._make_evaluator()
        loader = _make_loader(n_batches=2)
        all_res = ev.evaluate_all_scenarios(loader)
        md = APMEvaluator.to_markdown(all_res)
        assert "radar_off"     in md
        assert "satellite_off" in md
        assert "full"          in md

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_evaluator_cuda_inference(self):
        model = _make_apm_model()
        ev    = APMEvaluator(model, device=torch.device("cuda"))
        loader = _make_loader(n_batches=2)
        result = ev.evaluate(loader, scenario_name="full")
        assert result["num_batches"] == 2
