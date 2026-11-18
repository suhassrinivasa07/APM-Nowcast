"""
experiments/runner.py

Reproducible Benchmark & Ablation Orchestrator for APM-Nowcast Module 11.

Responsibilities
----------------
1. Seeds Python / NumPy / PyTorch for full reproducibility.
2. Trains and evaluates every model in the benchmark suite on identical
   synthetic or real Zarr datacube splits.
3. Profiles peak RTX 3050 VRAM (MB) and wall-clock inference latency
   (ms/sample) for each model.
4. Collects multi-horizon (T+15, T+30, T+45, T+60 min) CSI, POD, FAR,
   FSS, Brier Score and ECE.
5. Returns structured ExperimentResult dataclasses ready for
   ArtifactGenerator to export as LaTeX / Markdown / JSON / PNG.

Smoke / CI mode (quick_mode=True)
----------------------------------
  - 2 epochs, synthetic 32×32 patches, 2 batches per split.
  - Entire pipeline completes in < 15 s on CPU.

Full-scale mode (quick_mode=False)
------------------------------------
  - User-supplied DataLoaders over real INSAT-3D / DWR / ERA5 Zarr cubes.
"""
from __future__ import annotations

import math
import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from common.logging_config import get_logger
from datacube.schema import NUM_MODALITIES, MODALITY_TO_INDEX
from models.base import BaseNowcastModel
from models.apm_nowcast import APMNowcastModel
from models.baselines import (
    EulerianPersistenceModel,
    AdvectionPersistenceModel,
    ConvLSTMNowcastModel,
    SpatiotemporalUNetBaseline,
)
from training.config import TrainingConfig, LossWeights
from training.trainer import APMTrainer
from training.evaluator import APMEvaluator

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical per-lead-time labels
# ---------------------------------------------------------------------------
LEAD_TIME_LABELS = ["+15 min", "+30 min", "+45 min", "+60 min"]

# ---------------------------------------------------------------------------
# Synthetic batch factory (used in quick_mode)
# ---------------------------------------------------------------------------

def _make_batch(
    B: int = 2,
    T_in: int = 4,
    T_out: int = 4,
    H: int = 32,
    W: int = 32,
    n_mods: int = 4,
) -> Dict[str, Any]:
    """Return a minimal valid batch dict matching DataLoader output."""
    return {
        "inputs_by_modality": {
            "satellite": torch.randn(B, T_in, 3, H, W),
            "radar":     torch.randn(B, T_in, 2, H, W),
            "lightning": torch.clamp(torch.randn(B, T_in, 2, H, W), 0.0),
            "nwp":       torch.randn(B, T_in, 4, H, W),
        },
        "modality_masks": torch.ones(B, T_in, n_mods),
        "physics_fields": {
            "cape": torch.rand(B, T_in, H, W) * 3000,
        },
        "target_lightning_frames":  torch.zeros(B, T_out, H, W),
        "target_occurrence_frames": torch.zeros(B, T_out, H, W),
    }


def _make_loader(n_batches: int = 2, **kw) -> List[Dict[str, Any]]:
    return [_make_batch(**kw) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    log.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# VRAM & latency profiler
# ---------------------------------------------------------------------------

def profile_inference(
    model: BaseNowcastModel,
    device: torch.device,
    H: int = 64,
    W: int = 64,
    B: int = 1,
    T_in: int = 4,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> Tuple[float, float]:
    """
    Profile peak VRAM (MB) and mean wall-clock latency (ms/sample).

    Returns
    -------
    (latency_ms, peak_vram_mb)
    """
    model.eval()
    batch = _make_batch(B=B, T_in=T_in, H=H, W=W)
    inputs = {k: v.to(device) for k, v in batch["inputs_by_modality"].items()}
    masks  = batch["modality_masks"].to(device)
    phys   = {k: v.to(device) for k, v in batch["physics_fields"].items()}

    use_amp = device.type == "cuda"

    # Reset VRAM stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    @torch.inference_mode()
    def _run():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model(inputs, masks)
        else:
            model(inputs, masks)

    # Warmup
    for _ in range(n_warmup):
        _run()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _run()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return round(elapsed_ms, 2), round(peak_vram_mb, 1)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Structured result for one model in the benchmark table."""
    model_name: str
    n_params: int
    # Per-horizon metrics (list of 4 dicts: T+15, T+30, T+45, T+60)
    per_horizon: List[Dict[str, float]] = field(default_factory=list)
    # Aggregate (mean across horizons)
    aggregate: Dict[str, float] = field(default_factory=dict)
    # Hardware profile
    latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    inference_device: str = "cpu"
    # Raw evaluator output (kept for downstream artifact generation)
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default benchmark model registry
# ---------------------------------------------------------------------------

def _build_default_models(
    T_in: int,
    T_out: int,
    quick_mode: bool,
) -> Dict[str, BaseNowcastModel]:
    """
    Instantiate all benchmark models.

    quick_mode → smallest channel widths to minimize CPU time.
    """
    if quick_mode:
        # Ultra-compact for smoke tests
        convlstm = ConvLSTMNowcastModel(T_in=T_in, T_out=T_out,
                                         total_in_channels=11,
                                         hidden_dims=(8, 16))
        unet     = SpatiotemporalUNetBaseline(T_in=T_in, T_out=T_out,
                                               total_in_channels=11,
                                               base_channels=8)
        apm      = APMNowcastModel(T_in=T_in, T_out=T_out,
                                    latent_dim=16, hidden_dim=24)
    else:
        convlstm = ConvLSTMNowcastModel(T_in=T_in, T_out=T_out,
                                         total_in_channels=11,
                                         hidden_dims=(32, 64))
        unet     = SpatiotemporalUNetBaseline(T_in=T_in, T_out=T_out,
                                               total_in_channels=11,
                                               base_channels=32)
        apm      = APMNowcastModel(T_in=T_in, T_out=T_out,
                                    latent_dim=24, hidden_dim=48)

    return {
        "Eulerian Persistence":    EulerianPersistenceModel(T_in=T_in, T_out=T_out),
        "Advection Persistence":   AdvectionPersistenceModel(T_in=T_in, T_out=T_out),
        "ConvLSTM":                convlstm,
        "U-Net Spatiotemporal":    unet,
        "APM-Nowcast (Proposed)":  apm,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Reproducible benchmark & ablation orchestrator.

    Parameters
    ----------
    seed : int
        Global random seed for reproducibility.
    quick_mode : bool
        If True, uses synthetic data and 2 epochs (CI/smoke test).
    train_loader, val_loader : Iterable or None
        Real DataLoaders (required when quick_mode=False).
    device : torch.device or None
        Defaults to CUDA if available, else CPU.
    models : dict or None
        Custom model registry.  If None, uses _build_default_models().
    training_epochs : int
        Epochs per model when quick_mode=False.
    H, W : int
        Spatial patch size (default 64×64).
    T_in, T_out : int
        Temporal input/output steps (default 4/4).
    """

    def __init__(
        self,
        seed: int = 42,
        quick_mode: bool = False,
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
        device: Optional[torch.device] = None,
        models: Optional[Dict[str, BaseNowcastModel]] = None,
        training_epochs: int = 50,
        H: int = 64,
        W: int = 64,
        T_in: int = 4,
        T_out: int = 4,
    ) -> None:
        seed_everything(seed)
        self.seed = seed
        self.quick_mode = quick_mode
        self.H, self.W = (32, 32) if quick_mode else (H, W)
        self.T_in = T_in
        self.T_out = T_out
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Smoke config: 2 epochs, 2 batches/split
        if quick_mode:
            self._n_batches = 2
            self._epochs = 2
            self._train_loader = _make_loader(self._n_batches, H=self.H, W=self.W,
                                              T_in=T_in, T_out=T_out)
            self._val_loader   = _make_loader(self._n_batches, H=self.H, W=self.W,
                                              T_in=T_in, T_out=T_out)
        else:
            assert train_loader is not None and val_loader is not None, (
                "train_loader and val_loader are required when quick_mode=False"
            )
            self._train_loader = train_loader
            self._val_loader   = val_loader
            self._epochs = training_epochs

        self._models = models or _build_default_models(T_in, T_out, quick_mode)
        self._results: List[ExperimentResult] = []

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def _make_config(self, model_name: str) -> TrainingConfig:
        ckpt_dir = tempfile.mkdtemp(prefix="apm_smoke_") if self.quick_mode else "checkpoints"
        return TrainingConfig(
            epochs=self._epochs,
            lr=3e-4,
            batch_size=2 if self.quick_mode else 4,
            grad_accum_steps=1,
            amp_enabled=(self.device.type == "cuda"),
            checkpoint_dir=ckpt_dir,
            early_stopping_patience=self._epochs + 1,  # disable early-stop in smoke
            loss_weights=LossWeights(w_occ=1.0, w_int=0.5, w_phys=0.1, w_cal=0.05),
        )

    def _train_model(
        self,
        name: str,
        model: BaseNowcastModel,
    ) -> None:
        """
        Train `model` for self._epochs on self._train_loader.
        Zero-parameter baselines are skipped (no training needed).
        """
        if model.count_parameters() == 0:
            log.info(f"[Runner] Skipping training for zero-param model: {name}")
            return

        log.info(f"[Runner] Training {name}  ({model.count_parameters():,} params)")
        config = self._make_config(name)
        trainer = APMTrainer(
            model=model,
            config=config,
            train_loader=self._train_loader,
            val_loader=self._val_loader,
            device=self.device,
        )
        trainer.train()

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        name: str,
        model: BaseNowcastModel,
    ) -> ExperimentResult:
        log.info(f"[Runner] Evaluating {name}")
        evaluator = APMEvaluator(model=model, device=self.device)
        raw = evaluator.evaluate(self._val_loader, scenario_name="full")

        per_lt = raw.get("per_leadtime", [])
        agg    = raw.get("aggregate", {})

        # Build per-horizon summary (keep only key metrics)
        per_horizon = []
        for step_metrics in per_lt:
            per_horizon.append({
                "csi_30":    step_metrics.get("csi_30", float("nan")),
                "pod_30":    step_metrics.get("pod_30", float("nan")),
                "far_30":    step_metrics.get("far_30", float("nan")),
                "fss_mean":  step_metrics.get("fss_mean", float("nan")),
                "brier":     step_metrics.get("brier", float("nan")),
                "ece":       step_metrics.get("ece", float("nan")),
                "flash_rmse": step_metrics.get("flash_rmse", float("nan")),
            })

        # VRAM / latency profiling
        lat_ms, vram_mb = profile_inference(
            model, self.device, H=self.H, W=self.W
        )

        return ExperimentResult(
            model_name=name,
            n_params=model.count_parameters(),
            per_horizon=per_horizon,
            aggregate={
                "csi_30":    agg.get("csi_30", float("nan")),
                "pod_30":    agg.get("pod_30", float("nan")),
                "far_30":    agg.get("far_30", float("nan")),
                "fss_mean":  agg.get("fss_mean", float("nan")),
                "brier":     agg.get("brier", float("nan")),
                "ece":       agg.get("ece", float("nan")),
                "flash_rmse": agg.get("flash_rmse", float("nan")),
            },
            latency_ms=lat_ms,
            peak_vram_mb=vram_mb,
            inference_device=str(self.device),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[ExperimentResult]:
        """
        Train and evaluate all models. Returns a list of ExperimentResult.

        In quick_mode this runs in < 15 s on CPU.
        """
        self._results = []
        for name, model in self._models.items():
            model.to(self.device)
            self._train_model(name, model)
            result = self._evaluate_model(name, model)
            self._results.append(result)
            log.info(
                f"[Runner] {name}: CSI={result.aggregate.get('csi_30', float('nan')):.4f} | "
                f"lat={result.latency_ms:.1f}ms | VRAM={result.peak_vram_mb:.0f}MB"
            )
        return self._results

    def results(self) -> List[ExperimentResult]:
        return self._results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all results to a JSON-compatible dict."""
        import dataclasses
        out = {}
        for r in self._results:
            out[r.model_name] = {
                "n_params": r.n_params,
                "latency_ms": r.latency_ms,
                "peak_vram_mb": r.peak_vram_mb,
                "inference_device": r.inference_device,
                "per_horizon": r.per_horizon,
                "aggregate": r.aggregate,
            }
        return out
