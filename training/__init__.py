"""
training — APM-Nowcast Training, Evaluation & Verification Package.

Public API
----------
Configuration:
    TrainingConfig      — Full hyperparameter specification (save/load JSON)
    LossWeights         — Sub-term weight knobs for APMCompositeLoss

Training:
    APMTrainer          — AMP + gradient-accumulation + cosine LR training loop

Checkpointing:
    CheckpointManager   — Best-model tracking + rolling epoch retention

Evaluation:
    APMEvaluator        — Multi-horizon, multi-scenario skill verification

Metrics:
    pod, far, csi, hss                  — Categorical contingency metrics
    fss, fss_multiscale                 — Spatial Fractions Skill Score
    brier_score, ece                    — Probabilistic calibration
    flash_rmse, flash_mae               — Intensity regression
    compute_all_metrics                 — Aggregated batch metric dict
    compute_per_leadtime_metrics        — Per-lead-time metric list
"""

from training.config import LossWeights, TrainingConfig
from training.checkpoint import CheckpointManager
from training.trainer import APMTrainer
from training.evaluator import APMEvaluator
from training.metrics import (
    pod,
    far,
    csi,
    hss,
    fss,
    fss_multiscale,
    brier_score,
    ece,
    flash_rmse,
    flash_mae,
    compute_all_metrics,
    compute_per_leadtime_metrics,
)

__all__ = [
    # Config
    "TrainingConfig",
    "LossWeights",
    # Training
    "APMTrainer",
    # Checkpoint
    "CheckpointManager",
    # Evaluation
    "APMEvaluator",
    # Metrics
    "pod",
    "far",
    "csi",
    "hss",
    "fss",
    "fss_multiscale",
    "brier_score",
    "ece",
    "flash_rmse",
    "flash_mae",
    "compute_all_metrics",
    "compute_per_leadtime_metrics",
]
