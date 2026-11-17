"""
Dual Probabilistic & Uncertainty-Aware Forecast Heads for APM-Nowcast.

Produces:
  Standard BaseNowcastModel outputs:
    - lightning_logits:  [B, T_out, 1, H, W]
    - lightning_probs:   [B, T_out, 1, H, W]
    - flash_rate_pred:   [B, T_out, 1, H, W]

  Research uncertainty outputs:
    - aleatoric_uncertainty:  [B, T_out, 1, H, W] (heteroscedastic predicted variance)
    - epistemic_uncertainty:  [B, T_out, 1, H, W] (MC-Dropout ensemble variance)

Architecture:
  Upsamples latent [B, T_out, D, H/2, W/2] back to full resolution [B, T_out, *, H, W]
  using learned transpose convolution, then splits into:
    1. Occurrence head: logits + sigmoid -> binary lightning probability
    2. Flash rate head: non-negative density via Softplus
    3. Aleatoric uncertainty head: log-variance -> Softplus spread
  Epistemic uncertainty is computed at inference via MC-Dropout forward passes.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyForecastHeads(nn.Module):
    """
    Multi-task forecast decoder producing calibrated probabilities, flash density,
    and dual uncertainty estimates from spatiotemporal latent forecast states.
    """

    def __init__(
        self,
        latent_dim: int = 24,
        upsample_scale: int = 2,
        mc_dropout_rate: float = 0.1,
        mc_samples: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mc_dropout_rate = mc_dropout_rate
        self.mc_samples = mc_samples

        # Spatial upsampling back to full resolution
        g = _safe_groups(4, latent_dim)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=4, stride=upsample_scale, padding=1, bias=False),
            nn.GroupNorm(g, latent_dim),
            nn.GELU(),
        )

        # Shared feature refinement at full resolution
        self.shared_refine = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, latent_dim),
            nn.GELU(),
        )

        # MC-Dropout layer for epistemic uncertainty (applied during MC inference)
        self.mc_dropout = nn.Dropout2d(p=mc_dropout_rate)

        # Head 1: Binary occurrence logits
        self.occurrence_head = nn.Conv2d(latent_dim, 1, kernel_size=1, bias=True)

        # Head 2: Flash rate density (non-negative via Softplus)
        self.flash_rate_head = nn.Conv2d(latent_dim, 1, kernel_size=1, bias=True)

        # Head 3: Aleatoric uncertainty (log-variance -> Softplus)
        self.aleatoric_head = nn.Conv2d(latent_dim, 1, kernel_size=1, bias=True)

    def forward(
        self,
        forecast_latents: torch.Tensor,
        estimate_epistemic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        forecast_latents : [B, T_out, D, Hh, Wh]
        estimate_epistemic : if True, runs MC-Dropout forward passes (training or eval)

        Returns
        -------
        dict with standard + uncertainty outputs at full resolution [B, T_out, 1, H, W]
        """
        B, T_out, D, Hh, Wh = forecast_latents.shape

        # Process each forecast horizon
        flat = forecast_latents.reshape(B * T_out, D, Hh, Wh)

        # Upsample to full resolution
        up = self.upsample(flat)   # [B*T_out, D, H, W]
        feat = self.shared_refine(up) + up   # residual refinement

        # Deterministic predictions
        logits = self.occurrence_head(feat)                      # [B*T_out, 1, H, W]
        probs = torch.sigmoid(logits)
        flash_rate = F.softplus(self.flash_rate_head(feat))      # non-negative
        aleatoric = F.softplus(self.aleatoric_head(feat)) + 1e-6 # positive variance

        _, _, H, W = logits.shape

        # Reshape to [B, T_out, 1, H, W]
        logits = logits.reshape(B, T_out, 1, H, W)
        probs = probs.reshape(B, T_out, 1, H, W)
        flash_rate = flash_rate.reshape(B, T_out, 1, H, W)
        aleatoric = aleatoric.reshape(B, T_out, 1, H, W)

        outputs = {
            "lightning_logits": logits,
            "lightning_probs": probs,
            "flash_rate_pred": flash_rate,
            "aleatoric_uncertainty": aleatoric,
        }

        # Epistemic uncertainty via MC-Dropout
        if estimate_epistemic:
            outputs["epistemic_uncertainty"] = self._mc_dropout_uncertainty(
                forecast_latents, B, T_out, H, W
            )
        else:
            # Placeholder zeros during training (epistemic computed at eval time)
            outputs["epistemic_uncertainty"] = torch.zeros_like(aleatoric)

        return outputs

    def _mc_dropout_uncertainty(
        self,
        forecast_latents: torch.Tensor,
        B: int,
        T_out: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Estimate epistemic uncertainty by running MC-Dropout forward passes
        and computing the variance of predicted probabilities.
        """
        _, _, D, Hh, Wh = forecast_latents.shape
        flat = forecast_latents.reshape(B * T_out, D, Hh, Wh)

        mc_probs = []
        for _ in range(self.mc_samples):
            up = self.upsample(flat)
            feat = self.shared_refine(up) + up
            # Apply dropout even in eval mode for MC estimation
            feat_dropped = F.dropout2d(feat, p=self.mc_dropout_rate, training=True)
            logits_mc = self.occurrence_head(feat_dropped)
            probs_mc = torch.sigmoid(logits_mc)
            mc_probs.append(probs_mc)

        # Stack MC samples and compute variance
        mc_stack = torch.stack(mc_probs, dim=0)  # [mc_samples, B*T_out, 1, H, W]
        epistemic = mc_stack.var(dim=0)  # [B*T_out, 1, H, W]
        return epistemic.reshape(B, T_out, 1, H, W)


def _safe_groups(desired: int, channels: int) -> int:
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g
