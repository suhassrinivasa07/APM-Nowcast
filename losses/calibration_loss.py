"""
Uncertainty Calibration Regularizer for APM-Nowcast.

Aligns the total predictive uncertainty (aleatoric + epistemic) with the actual
spatial prediction error so that:
  - Regions with large errors produce correspondingly large uncertainty estimates.
  - Overconfident errors (high error + low uncertainty) are penalized.
  - Under-confident predictions (low error + high uncertainty) are mildly penalized.

Calibration metric:
    L_cal = | |y - p_hat| - sqrt(sigma_a^2 + sigma_e^2) |

This encourages the learned uncertainty to track the true error magnitude,
yielding well-calibrated probabilistic forecasts suitable for operational
decision-making in severe weather warnings.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyCalibrationLoss(nn.Module):
    """
    Calibration regularizer aligning total predictive uncertainty with
    actual prediction error magnitude.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        lightning_probs: torch.Tensor,
        targets: torch.Tensor,
        aleatoric_uncertainty: torch.Tensor | None = None,
        epistemic_uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        lightning_probs      : [B, T_out, 1, H, W] predicted probabilities
        targets              : [B, T_out, 1, H, W] binary occurrence ground truth
        aleatoric_uncertainty: [B, T_out, 1, H, W] predicted aleatoric variance
        epistemic_uncertainty: [B, T_out, 1, H, W] estimated epistemic variance

        Returns
        -------
        Scalar calibration loss. Returns 0.0 if no uncertainty is provided.
        """
        if aleatoric_uncertainty is None and epistemic_uncertainty is None:
            return torch.tensor(0.0, device=lightning_probs.device,
                                dtype=lightning_probs.dtype)

        # Actual prediction error magnitude
        abs_error = torch.abs(lightning_probs - targets)

        # Total predictive uncertainty: sqrt(sigma_a^2 + sigma_e^2)
        total_var = torch.zeros_like(abs_error)
        if aleatoric_uncertainty is not None:
            total_var = total_var + aleatoric_uncertainty
        if epistemic_uncertainty is not None:
            total_var = total_var + epistemic_uncertainty

        total_std = torch.sqrt(total_var + self.eps)

        # Calibration: uncertainty should match error magnitude
        calibration_error = torch.abs(abs_error - total_std)

        return calibration_error.mean()
