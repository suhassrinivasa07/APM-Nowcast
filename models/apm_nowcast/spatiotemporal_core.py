"""
Multi-Scale Residual ConvGRU Spatiotemporal Evolution Core for APM-Nowcast.

Recurrently evolves physics-conditioned fused atmospheric latent states from
T_in observed frames into T_out forecast horizon latent states using a
lightweight ConvGRU with multi-scale residual skip connections.

Architecture:
  ConvGRU (D -> D) processes T_in conditioned frames to build a hidden state,
  then auto-regressively generates T_out latent forecast frames.
  Multi-scale context from encoder skip connection is added via residual fusion.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    2D Convolutional Gated Recurrent Unit cell.

    Uses combined gate convolution for efficiency (fewer kernel launches):
      gates = Conv2d([x; h], 2*hidden)   -> split into reset, update
      candidate = Conv2d([x; r*h], hidden)
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.gate_conv = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.candidate_conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C_in, H, W]
        h : [B, hidden, H, W] or None

        Returns
        -------
        h_next : [B, hidden, H, W]
        """
        B, _, Hh, Wh = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_channels, Hh, Wh, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h], dim=1)
        gates = self.gate_conv(combined)
        r, z = torch.split(gates, self.hidden_channels, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        candidate_input = torch.cat([x, r * h], dim=1)
        candidate = torch.tanh(self.candidate_conv(candidate_input))

        h_next = (1 - z) * h + z * candidate
        return h_next


class SpatiotemporalEvolutionCore(nn.Module):
    """
    Multi-scale residual ConvGRU core that evolves atmospheric latent states
    from T_in observed frames into T_out forecast frames.

    Encoder phase:  processes T_in conditioned frames to build hidden state.
    Forecaster phase: auto-regressively generates T_out latent forecast states.
    """

    def __init__(
        self,
        latent_dim: int = 24,
        hidden_dim: int = 48,
        T_out: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.T_out = T_out

        # Input projection: latent_dim -> hidden_dim
        g = _safe_groups(4, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
        )

        # Encoder ConvGRU
        self.encoder_gru = ConvGRUCell(hidden_dim, hidden_dim)

        # Forecaster ConvGRU (auto-regressive)
        self.forecaster_gru = ConvGRUCell(hidden_dim, hidden_dim)

        # Output projection back to latent_dim
        g_out = _safe_groups(4, latent_dim)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g_out, latent_dim),
            nn.GELU(),
        )

        # Learned forecast seed: projects last encoder hidden to forecaster input
        self.forecast_seed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)

    def forward(self, conditioned: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        conditioned : [B, T_in, D, Hh, Wh] — physics-conditioned fused latent

        Returns
        -------
        forecast_latents : [B, T_out, D, Hh, Wh] — predicted future latent states
        """
        B, T_in, D, Hh, Wh = conditioned.shape

        # --- Encoder phase ---
        h_enc = None
        for t in range(T_in):
            x_t = conditioned[:, t]  # [B, D, Hh, Wh]
            x_proj = self.input_proj(x_t)  # [B, hidden, Hh, Wh]
            h_enc = self.encoder_gru(x_proj, h_enc)

        # --- Forecaster phase ---
        h_fore = h_enc
        forecast_input = self.forecast_seed(h_enc)
        forecast_latents = []

        for t in range(self.T_out):
            h_fore = self.forecaster_gru(forecast_input, h_fore)
            out_t = self.output_proj(h_fore)  # [B, D, Hh, Wh]
            forecast_latents.append(out_t)
            # Feed hidden state through projection as next input (auto-regressive)
            forecast_input = h_fore

        # Stack: [B, T_out, D, Hh, Wh]
        return torch.stack(forecast_latents, dim=1)


def _safe_groups(desired: int, channels: int) -> int:
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g
