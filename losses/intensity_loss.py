"""
Heteroscedastic Gaussian Intensity Loss for Flash Rate Nowcasting.

Penalizes flash rate prediction errors weighted by predicted aleatoric
uncertainty (heteroscedastic noise variance), implementing the negative
log-likelihood of a Gaussian observation model:

    NLL = |y - y_hat| / sigma_a + log(sigma_a)

where sigma_a is the predicted per-pixel aleatoric standard deviation.

This formulation:
  - Incentivizes the model to predict higher uncertainty in difficult regions
    (storm edges, initiation zones) rather than making overconfident errors.
  - Uses L1 error (Laplacian NLL) for robustness to heavy-tailed flash count
    distributions typical of Indian monsoon thunderstorms.

FP16 safety: sigma_a is clamped to [1e-3, 10.0] to prevent log(0) or
division-by-zero under half precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HeteroscedasticIntensityLoss(nn.Module):
    """
    Aleatoric-uncertainty-weighted flash rate intensity loss.
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

    def forward(
        self,
        flash_rate_pred: torch.Tensor,
        targets: torch.Tensor,
        aleatoric_uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        flash_rate_pred       : [B, T_out, 1, H, W] non-negative predicted flash density
        targets               : [B, T_out, 1, H, W] observed flash counts (ground truth)
        aleatoric_uncertainty : [B, T_out, 1, H, W] predicted variance (sigma_a^2)
                                If None, falls back to simple L1 loss.

        Returns
        -------
        Scalar loss tensor
        """
        # L1 error (robust to heavy-tailed flash counts)
        abs_error = torch.abs(flash_rate_pred - targets)

        if aleatoric_uncertainty is not None:
            # sigma_a = sqrt(predicted_variance), clamped for FP16 safety
            sigma_a = torch.clamp(
                torch.sqrt(aleatoric_uncertainty + self.eps),
                min=self.sigma_min,
                max=self.sigma_max,
            )
            # Laplacian NLL: |y - y_hat| / sigma + log(sigma)
            nll = abs_error / sigma_a + torch.log(sigma_a + self.eps)
            return nll.mean()
        else:
            # Fallback: simple L1 loss (for baseline models without uncertainty)
            return abs_error.mean()
