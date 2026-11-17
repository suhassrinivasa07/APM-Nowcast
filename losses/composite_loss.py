"""
Unified APM-Nowcast Composite Loss Function.

Combines occurrence, intensity, physics consistency, and uncertainty calibration
sub-losses with configurable weights. Returns a detached metrics dictionary
logging every sub-loss for experiment tracking (TensorBoard / W&B / MLflow).

Compatible with both:
  - APMNowcastModel outputs (full uncertainty + physics fields)
  - Baseline model outputs (no uncertainty keys, physics_fields=None)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from losses.calibration_loss import UncertaintyCalibrationLoss
from losses.intensity_loss import HeteroscedasticIntensityLoss
from losses.occurrence_loss import OccurrenceLoss
from losses.physics_loss import PhysicsConsistencyLoss


class APMCompositeLoss(nn.Module):
    """
    Multi-objective loss combining meteorological, probabilistic, and
    physics-guided penalty terms for lightning nowcasting.

    Default Weights (empirically balanced):
      - occurrence  (w_occ):   1.0  (primary classification objective)
      - intensity   (w_int):   0.5  (flash rate regression)
      - physics     (w_phys):  0.1  (soft physics constraint regularizer)
      - calibration (w_cal):   0.05 (uncertainty calibration regularizer)
    """

    def __init__(
        self,
        w_occ: float = 1.0,
        w_int: float = 0.5,
        w_phys: float = 0.1,
        w_cal: float = 0.05,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ) -> None:
        super().__init__()
        self.w_occ = w_occ
        self.w_int = w_int
        self.w_phys = w_phys
        self.w_cal = w_cal

        self.occurrence_loss = OccurrenceLoss(
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        self.intensity_loss = HeteroscedasticIntensityLoss()
        self.physics_loss = PhysicsConsistencyLoss()
        self.calibration_loss = UncertaintyCalibrationLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        physics_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        outputs : model prediction dict containing:
            Required: 'lightning_logits', 'lightning_probs', 'flash_rate_pred'
            Optional: 'aleatoric_uncertainty', 'epistemic_uncertainty'
        targets : dict containing:
            'target_occurrence': [B, T_out, 1, H, W] binary occurrence
            'target_flash_rate': [B, T_out, 1, H, W] observed flash counts
        physics_fields : dict of physics guidance tensors or None

        Returns
        -------
        total_loss : scalar loss tensor (differentiable)
        metrics    : dict of detached scalar floats for logging
        """
        logits = outputs["lightning_logits"]
        probs = outputs["lightning_probs"]
        flash_pred = outputs["flash_rate_pred"]

        target_occ = targets["target_occurrence"]
        target_flash = targets["target_flash_rate"]

        # --- 1. Occurrence Loss (Focal + Dice) ---
        loss_occ = self.occurrence_loss(logits, probs, target_occ)

        # --- 2. Intensity Loss (Heteroscedastic or L1 fallback) ---
        aleatoric = outputs.get("aleatoric_uncertainty", None)
        loss_int = self.intensity_loss(flash_pred, target_flash, aleatoric)

        # --- 3. Physics Consistency Loss (graceful None fallback) ---
        loss_phys = self.physics_loss(probs, physics_fields)

        # --- 4. Uncertainty Calibration Loss ---
        epistemic = outputs.get("epistemic_uncertainty", None)
        loss_cal = self.calibration_loss(probs, target_occ, aleatoric, epistemic)

        # --- Weighted Combination ---
        total = (
            self.w_occ * loss_occ
            + self.w_int * loss_int
            + self.w_phys * loss_phys
            + self.w_cal * loss_cal
        )

        # --- Metrics Dictionary (all detached scalars) ---
        metrics = {
            "loss_total": total.detach().item(),
            "loss_occ": loss_occ.detach().item(),
            "loss_int": loss_int.detach().item(),
            "loss_phys": loss_phys.detach().item(),
            "loss_cal": loss_cal.detach().item(),
        }

        return total, metrics
