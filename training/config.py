"""
Typed configuration dataclass for APM-Nowcast training runs.

Centralizes all hyperparameters so experiments are fully reproducible
from a single serializable config object.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LossWeights:
    """Configurable weights for each APMCompositeLoss sub-term."""
    w_occ: float = 1.0
    w_int: float = 0.5
    w_phys: float = 0.1
    w_cal: float = 0.05
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75


@dataclass
class TrainingConfig:
    """
    Complete training hyperparameter specification for APM-Nowcast.

    All fields have sensible defaults for RTX 3050 (4 GB VRAM) training
    on 64×64 patches at batch size 4.
    """
    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # Schedule
    epochs: int = 50
    warmup_epochs: int = 2
    cosine_eta_min: float = 1e-6

    # Batch / accumulation
    batch_size: int = 4
    grad_accum_steps: int = 2          # effective batch = batch_size * grad_accum_steps

    # AMP / precision
    amp_enabled: bool = True           # False → pure FP32 (CPU fallback)

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_csi"   # maximize
    early_stopping_mode: str = "max"         # 'max' or 'min'

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_keep_last_n: int = 3

    # Loss weights
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # Verification thresholds
    prob_thresholds: tuple = (0.3, 0.5)
    fss_scales: tuple = (3, 5, 9)

    # Misc
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    log_every_n_epochs: int = 1

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["loss_weights"] = asdict(self.loss_weights)
        return d

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        lw_dict = raw.pop("loss_weights", {})
        raw["loss_weights"] = LossWeights(**lw_dict)
        raw["betas"] = tuple(raw["betas"])
        raw["prob_thresholds"] = tuple(raw["prob_thresholds"])
        raw["fss_scales"] = tuple(raw["fss_scales"])
        return cls(**raw)
