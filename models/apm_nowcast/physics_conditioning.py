"""
Feature-wise Linear Modulation (FiLM) Physics-Guided Conditioning for APM-Nowcast.

Modulates fused convective latent features using thermodynamic stability and
shear fields (CAPE, CIN, bulk wind shear, moisture convergence) extracted from
the NWP encoder output:

    conditioned = gamma(physics) * fused + beta(physics) + advection_bias

where gamma and beta are spatially-varying affine parameters predicted from
ERA5 thermodynamic fields, and advection_bias is a learned residual correction
encoding physical prior knowledge about convective evolution.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsFiLMConditioning(nn.Module):
    """
    FiLM conditioning block that modulates fused atmospheric features using
    ERA5 thermodynamic guidance.

    Physics encoder projects NWP latent -> spatially varying (gamma, beta) modulation
    parameters + residual advection bias.
    """

    def __init__(self, latent_dim: int = 24, physics_dim: int = 24) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Physics-to-modulation parameter prediction
        # Takes the NWP encoder output and predicts per-pixel FiLM parameters
        self.film_generator = nn.Sequential(
            nn.Conv2d(physics_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(4, latent_dim), latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim * 2, kernel_size=1, bias=True),
        )

        # Advection bias: residual physical prior encoding expected convective motion
        self.advection_bias = nn.Sequential(
            nn.Conv2d(physics_dim, latent_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=1, bias=True),
        )

        # Initialize gamma close to 1 and beta close to 0 for near-identity start
        nn.init.ones_(self.film_generator[-1].weight[:latent_dim])
        nn.init.zeros_(self.film_generator[-1].weight[latent_dim:])
        nn.init.zeros_(self.film_generator[-1].bias)

    def forward(
        self,
        fused_features: torch.Tensor,
        nwp_encoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        fused_features : [B, T_in, D, Hh, Wh] — adaptively fused multimodal latent
        nwp_encoded    : [B, T_in, D_nwp, Hh, Wh] — NWP encoder output

        Returns
        -------
        conditioned : [B, T_in, D, Hh, Wh]
        physics_stats : dict with diagnostic FiLM modulation statistics
        """
        B, T, D, Hh, Wh = fused_features.shape
        _, _, D_nwp, _, _ = nwp_encoded.shape

        # Merge batch and time
        fused_flat = fused_features.reshape(B * T, D, Hh, Wh)
        nwp_flat = nwp_encoded.reshape(B * T, D_nwp, Hh, Wh)

        # Generate FiLM parameters: [B*T, 2*D, Hh, Wh]
        film_params = self.film_generator(nwp_flat)
        gamma = film_params[:, :D]       # multiplicative modulation
        beta = film_params[:, D:]        # additive modulation

        # Advection bias from physics fields
        adv_bias = self.advection_bias(nwp_flat)  # [B*T, D, Hh, Wh]

        # FiLM conditioning: gamma * feat + beta + advection_bias
        conditioned = gamma * fused_flat + beta + adv_bias

        # Reshape back
        conditioned = conditioned.reshape(B, T, D, Hh, Wh)

        # Diagnostic statistics for interpretability
        physics_stats = {
            "gamma_mean": gamma.mean().detach(),
            "gamma_std": gamma.std().detach(),
            "beta_mean": beta.mean().detach(),
            "beta_std": beta.std().detach(),
            "advection_bias_norm": adv_bias.norm(dim=1).mean().detach(),
        }

        return conditioned, physics_stats


def _safe_groups(desired: int, channels: int) -> int:
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g
