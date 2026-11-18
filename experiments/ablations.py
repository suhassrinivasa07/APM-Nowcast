"""
experiments/ablations.py

Systematic Ablation Factory for APM-Nowcast Module 11.

Isolates each research contribution by constructing clean model/config
variants that differ from the full APM-Nowcast in exactly one axis:

  1. APM-Nowcast (Full)                   – all components enabled
  2. w/o Physics Conditioning             – FiLM modulation bypassed
  3. w/o Adaptive Fusion                  – uniform unweighted modality mean
  4. w/o Physics & Calibration Loss       – occurrence + intensity loss only

Each ablation is evaluated with the same ExperimentRunner infrastructure
so results are directly comparable.
"""
from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from common.logging_config import get_logger
from datacube.schema import NUM_MODALITIES, MODALITY_TO_INDEX
from models.apm_nowcast import APMNowcastModel
from models.base import BaseNowcastModel
from training.config import TrainingConfig, LossWeights
from training.trainer import APMTrainer
from training.evaluator import APMEvaluator
from experiments.runner import (
    ExperimentResult,
    _make_loader,
    profile_inference,
    seed_everything,
    LEAD_TIME_LABELS,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Specification for a single ablation variant.

    Attributes
    ----------
    name : str
        Human-readable label used in tables and figures.
    disable_physics_conditioning : bool
        If True the FiLM physics conditioning gate is bypassed (γ=1, β=0).
    disable_adaptive_fusion : bool
        If True cross-modal attention is replaced by a uniform mean-pool.
    loss_weights : LossWeights
        Loss term weights. Set w_phys=0 and w_cal=0 to ablate physics/cal loss.
    model_kwargs : dict
        Extra kwargs forwarded to APMNowcastModel.__init__.
    description : str
        One-sentence description for the paper.
    """
    name: str
    disable_physics_conditioning: bool = False
    disable_adaptive_fusion: bool = False
    loss_weights: LossWeights = field(default_factory=LossWeights)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# No-physics wrapper (bypasses FiLM gate)
# ---------------------------------------------------------------------------

class _NoPhysicsAPM(nn.Module):
    """
    Thin wrapper around APMNowcastModel that neutralises the FiLM gate
    by zeroing the physics_fields input at forward time, forcing γ=1, β=0.
    """

    def __init__(self, inner: APMNowcastModel) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Pass empty physics fields → FiLM conditioning sees zero CAPE → γ≈1, β≈0
        return self.inner(inputs_by_modality, modality_masks,
                          physics_fields=None, **kwargs)

    # Delegate BaseNowcastModel API
    def count_parameters(self) -> int:
        return self.inner.count_parameters()

    def vram_footprint_mb(self, *a, **kw) -> float:
        return self.inner.vram_footprint_mb(*a, **kw)

    # Make APMTrainer / APMEvaluator work
    def eval(self):
        self.inner.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.inner.train(mode)
        return super().train(mode)

    def parameters(self, recurse=True):
        return self.inner.parameters(recurse)

    def named_parameters(self, *a, **kw):
        return self.inner.named_parameters(*a, **kw)

    def state_dict(self, *a, **kw):
        return self.inner.state_dict(*a, **kw)

    def load_state_dict(self, *a, **kw):
        return self.inner.load_state_dict(*a, **kw)

    def to(self, *a, **kw):
        self.inner.to(*a, **kw)
        return super().to(*a, **kw)


# ---------------------------------------------------------------------------
# Uniform-fusion wrapper (bypasses cross-modal attention)
# ---------------------------------------------------------------------------

class _UniformFusionAPM(nn.Module):
    """
    Wraps APMNowcastModel and monkey-patches the adaptive fusion module
    to return a uniform average over active modalities (no learned weights).
    """

    def __init__(self, inner: APMNowcastModel) -> None:
        super().__init__()
        self.inner = inner
        self._patch_fusion()

    def _patch_fusion(self) -> None:
        """Replace AdaptiveModalityFusion.forward with a simple mean."""
        fusion_mod = self.inner.fusion  # AdaptiveModalityFusion instance

        def _uniform_forward(
            encoded_modalities: Dict[str, torch.Tensor],
            modality_masks: torch.Tensor,
            *args,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # encoded_modalities: {mod: [B, T_in, D, Hh, Wh]}
            # modality_masks: [B, T_in, M]
            tensors = list(encoded_modalities.values())
            if not tensors:
                raise ValueError("No encoded modalities to fuse")
            fused = torch.stack(tensors, dim=0).mean(dim=0)  # [B, T_in, D, Hh, Wh]
            B, T_in, D, Hh, Wh = tensors[0].shape
            M = len(tensors)
            uniform_weights = torch.ones(
                B, M, Hh, Wh,
                device=tensors[0].device,
                dtype=tensors[0].dtype,
            ) / M
            return fused, uniform_weights

        fusion_mod.forward = _uniform_forward

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def count_parameters(self) -> int:
        return self.inner.count_parameters()

    def vram_footprint_mb(self, *a, **kw) -> float:
        return self.inner.vram_footprint_mb(*a, **kw)

    def eval(self):
        self.inner.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.inner.train(mode)
        return super().train(mode)

    def parameters(self, recurse=True):
        return self.inner.parameters(recurse)

    def named_parameters(self, *a, **kw):
        return self.inner.named_parameters(*a, **kw)

    def state_dict(self, *a, **kw):
        return self.inner.state_dict(*a, **kw)

    def load_state_dict(self, *a, **kw):
        return self.inner.load_state_dict(*a, **kw)

    def to(self, *a, **kw):
        self.inner.to(*a, **kw)
        return super().to(*a, **kw)


# ---------------------------------------------------------------------------
# Canonical ablation suite definition
# ---------------------------------------------------------------------------

CANONICAL_ABLATIONS: List[AblationConfig] = [
    AblationConfig(
        name="APM-Nowcast (Full)",
        description="Full proposed architecture with all components enabled.",
    ),
    AblationConfig(
        name="w/o Physics Conditioning",
        disable_physics_conditioning=True,
        description="FiLM physics conditioning gate bypassed (γ=1, β=0).",
    ),
    AblationConfig(
        name="w/o Adaptive Fusion",
        disable_adaptive_fusion=True,
        description="Cross-modal attention replaced by uniform modality averaging.",
    ),
    AblationConfig(
        name="w/o Physics & Calibration Loss",
        loss_weights=LossWeights(w_occ=1.0, w_int=0.5, w_phys=0.0, w_cal=0.0),
        description="Trained with occurrence + intensity losses only (no physics or ECE calibration).",
    ),
]


# ---------------------------------------------------------------------------
# AblationFactory
# ---------------------------------------------------------------------------

class AblationFactory:
    """
    Builds and evaluates the canonical APM-Nowcast ablation suite.

    Parameters
    ----------
    seed : int
        Global random seed.
    quick_mode : bool
        If True, runs 2 synthetic epochs on 32×32 patches (< 15 s CI).
    train_loader, val_loader : Iterable or None
        Required when quick_mode=False.
    device : torch.device or None
    base_model_kwargs : dict
        Extra kwargs for APMNowcastModel (e.g. latent_dim, hidden_dim).
    ablation_configs : list of AblationConfig or None
        Defaults to CANONICAL_ABLATIONS.
    """

    def __init__(
        self,
        seed: int = 42,
        quick_mode: bool = False,
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
        device: Optional[torch.device] = None,
        base_model_kwargs: Optional[Dict[str, Any]] = None,
        ablation_configs: Optional[List[AblationConfig]] = None,
        T_in: int = 4,
        T_out: int = 4,
    ) -> None:
        seed_everything(seed)
        self.seed = seed
        self.quick_mode = quick_mode
        self.T_in = T_in
        self.T_out = T_out
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ablation_configs = ablation_configs or CANONICAL_ABLATIONS

        H = W = 32 if quick_mode else 64
        n_batches = 2 if quick_mode else 4
        self._epochs = 2 if quick_mode else 50

        if quick_mode:
            self._train_loader = _make_loader(n_batches, H=H, W=W, T_in=T_in, T_out=T_out)
            self._val_loader   = _make_loader(n_batches, H=H, W=W, T_in=T_in, T_out=T_out)
        else:
            assert train_loader is not None and val_loader is not None
            self._train_loader = train_loader
            self._val_loader   = val_loader

        # Compact dims in quick_mode
        if quick_mode:
            self._base_kwargs = dict(latent_dim=16, hidden_dim=24, T_in=T_in, T_out=T_out)
        else:
            self._base_kwargs = dict(latent_dim=24, hidden_dim=48, T_in=T_in, T_out=T_out)
        if base_model_kwargs:
            self._base_kwargs.update(base_model_kwargs)

        self.H = H
        self._results: List[ExperimentResult] = []

    # ------------------------------------------------------------------

    def _build_model(self, cfg: AblationConfig) -> nn.Module:
        kw = {**self._base_kwargs, **cfg.model_kwargs}
        base = APMNowcastModel(**kw)
        if cfg.disable_physics_conditioning:
            return _NoPhysicsAPM(base)
        if cfg.disable_adaptive_fusion:
            return _UniformFusionAPM(base)
        return base

    def _make_training_config(self, cfg: AblationConfig) -> TrainingConfig:
        ckpt_dir = tempfile.mkdtemp(prefix="apm_ablation_") if self.quick_mode else "checkpoints"
        return TrainingConfig(
            epochs=self._epochs,
            lr=3e-4,
            batch_size=2 if self.quick_mode else 4,
            grad_accum_steps=1,
            amp_enabled=(self.device.type == "cuda"),
            checkpoint_dir=ckpt_dir,
            early_stopping_patience=self._epochs + 1,
            loss_weights=cfg.loss_weights,
        )

    def _train_and_eval(self, cfg: AblationConfig) -> ExperimentResult:
        log.info(f"[Ablation] Running: {cfg.name}")
        model = self._build_model(cfg)
        model.to(self.device)

        # Training (skip for 0-param models, though all ablations are parametric)
        n_params = (
            model.inner.count_parameters()
            if hasattr(model, "inner")
            else model.count_parameters()
        )
        if n_params > 0:
            train_cfg = self._make_training_config(cfg)
            trainer = APMTrainer(
                model=model,
                config=train_cfg,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                device=self.device,
            )
            trainer.train()

        # Evaluation
        evaluator = APMEvaluator(model=model, device=self.device)
        raw = evaluator.evaluate(self._val_loader, scenario_name="full")

        per_lt  = raw.get("per_leadtime", [])
        agg     = raw.get("aggregate", {})
        per_horizon = [
            {
                "csi_30":    m.get("csi_30", float("nan")),
                "pod_30":    m.get("pod_30", float("nan")),
                "far_30":    m.get("far_30", float("nan")),
                "fss_mean":  m.get("fss_mean", float("nan")),
                "brier":     m.get("brier", float("nan")),
                "ece":       m.get("ece", float("nan")),
                "flash_rmse": m.get("flash_rmse", float("nan")),
            }
            for m in per_lt
        ]

        lat_ms, vram_mb = profile_inference(
            model, self.device, H=self.H, W=self.H
        )

        return ExperimentResult(
            model_name=cfg.name,
            n_params=n_params,
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
        """Run all ablation variants and return results list."""
        self._results = []
        for cfg in self.ablation_configs:
            result = self._train_and_eval(cfg)
            self._results.append(result)
        return self._results

    def results(self) -> List[ExperimentResult]:
        return self._results

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for r in self._results:
            out[r.model_name] = {
                "n_params": r.n_params,
                "latency_ms": r.latency_ms,
                "peak_vram_mb": r.peak_vram_mb,
                "per_horizon": r.per_horizon,
                "aggregate": r.aggregate,
            }
        return out
