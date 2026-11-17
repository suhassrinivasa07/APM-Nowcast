"""
Modality-Specific Spatiotemporal Encoders for APM-Nowcast.

Each encoder projects a single sensor modality [B, T_in, C_m, H, W] into a shared
latent representation [B, T_in, D, H/2, W/2] using lightweight spatial convolutions
with temporal weight-sharing (same Conv2D applied per-timestep).

Encoders:
- SatelliteEncoder:  TIR1, WV, MIR brightness temperature channels -> latent D.
- RadarEncoder:      Composite reflectivity + CAPPI dBZ -> latent D.
- LightningEncoder:  Flash count + occurrence mask -> latent D.
- NWPEncoder:        CAPE, CIN, humidity, wind speed -> latent D.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    Generic lightweight spatiotemporal encoder for a single sensor modality.

    Architecture:
      Conv2D(C_in -> D/2, 3x3, stride=2) -> GroupNorm -> GELU
      Conv2D(D/2 -> D,   3x3, stride=1)  -> GroupNorm -> GELU + residual projection

    Output: [B, T_in, D, H/2, W/2]
    """

    def __init__(self, in_channels: int, latent_dim: int, num_groups: int = 4) -> None:
        super().__init__()
        mid = max(latent_dim // 2, 8)

        # Ensure group counts divide channel dims
        g_mid = _safe_groups(num_groups, mid)
        g_out = _safe_groups(num_groups, latent_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(g_mid, mid),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(mid, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(g_out, latent_dim),
        )
        self.residual = nn.Conv2d(mid, latent_dim, kernel_size=1, bias=False) if mid != latent_dim else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T_in, C_in, H, W]

        Returns
        -------
        [B, T_in, D, H/2, W/2]
        """
        B, T, C, H, W = x.shape
        # Merge batch and time for weight-shared per-timestep encoding
        x_flat = x.reshape(B * T, C, H, W)
        h = self.stem(x_flat)
        out = self.act(self.refine(h) + self.residual(h))
        _, D, Hh, Wh = out.shape
        return out.reshape(B, T, D, Hh, Wh)


class SatelliteEncoder(ModalityEncoder):
    """Encoder for INSAT-3D/3DR TIR1 + WV + MIR brightness temperatures (3 channels)."""
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__(in_channels=3, latent_dim=latent_dim)


class RadarEncoder(ModalityEncoder):
    """Encoder for Indian DWR composite reflectivity + CAPPI (2 channels)."""
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__(in_channels=2, latent_dim=latent_dim)


class LightningEncoder(ModalityEncoder):
    """Encoder for NRSC flash count + binary occurrence mask (2 channels)."""
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__(in_channels=2, latent_dim=latent_dim)


class NWPEncoder(ModalityEncoder):
    """Encoder for ERA5 thermodynamic fields: CAPE, CIN, humidity, wind (4 channels)."""
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__(in_channels=4, latent_dim=latent_dim)


def _safe_groups(desired: int, channels: int) -> int:
    """Find the largest group count <= desired that divides channels evenly."""
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g
