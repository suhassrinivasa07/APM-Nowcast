"""
Encoder-Forecaster Convolutional LSTM (ConvLSTM) Baseline for Spatiotemporal Nowcasting.

Architectural Highlights:
- Fused multimodal input interface with adaptive sensor dropout protection.
- Optimized 2-layer ConvLSTM Encoder and Forecaster with combined gate convolutions.
- Dual prediction heads:
  1. Binary occurrence logits & probabilities [B, T_out, 1, H, W]
  2. Non-negative flash rate intensity [B, T_out, 1, H, W] via Softplus
- Parameter budget < 1.5M parameters; VRAM consumption < 1.0 GB on [4, 4, C, 64, 64] batches.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger
from models.base import BaseNowcastModel

log = get_logger(__name__)


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM cell with combined 4-gate convolution for computational efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Combined convolution computing input, forget, cell, and output gates in a single pass
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMNowcastModel(BaseNowcastModel):
    """
    Spatiotemporal ConvLSTM Encoder-Forecaster for Multimodal Lightning Nowcasting.
    """

    def __init__(
        self,
        T_in: int = 4,
        T_out: int = 4,
        total_in_channels: Optional[int] = None,
        hidden_dims: Tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__(T_in=T_in, T_out=T_out)
        self.in_channels = total_in_channels or sum(self.modality_channel_counts.values())
        self.h1_dim, self.h2_dim = hidden_dims

        # Initial feature projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.h1_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, self.h1_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder ConvLSTM layers
        self.enc_cell1 = ConvLSTMCell(self.h1_dim, self.h1_dim)
        self.downsample = nn.Conv2d(self.h1_dim, self.h2_dim, kernel_size=3, stride=2, padding=1)
        self.enc_cell2 = ConvLSTMCell(self.h2_dim, self.h2_dim)

        # Forecaster ConvLSTM layer
        self.forecaster_cell = ConvLSTMCell(self.h2_dim, self.h2_dim)

        # Decoder / Upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.h2_dim, self.h1_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, self.h1_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Dual prediction heads
        self.occurrence_head = nn.Sequential(
            nn.Conv2d(self.h1_dim, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.flash_rate_head = nn.Sequential(
            nn.Conv2d(self.h1_dim, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 1. Fuse multimodal inputs with sensor dropout protection: [B, T_in, C_total, H, W]
        fused_inputs, _ = self.fuse_multimodal_inputs(inputs_by_modality, modality_masks)
        B, T_in, _, H, W = fused_inputs.shape

        # 2. Encoder pass
        s1 = None
        s2 = None

        for t in range(T_in):
            x_t = fused_inputs[:, t]  # [B, C_total, H, W]
            feat1 = self.input_conv(x_t)
            h1, c1 = self.enc_cell1(feat1, s1)
            s1 = (h1, c1)

            feat2 = self.downsample(h1)
            h2, c2 = self.enc_cell2(feat2, s2)
            s2 = (h2, c2)

        # 3. Forecaster unrolling across T_out horizons
        f_state = s2
        curr_input = s2[0]  # Initialize with last encoder state

        pred_logits_list = []
        pred_rates_list = []

        for step in range(self.T_out):
            h_f, c_f = self.forecaster_cell(curr_input, f_state)
            f_state = (h_f, c_f)
            curr_input = h_f

            # Upsample back to original spatial resolution (H, W)
            dec_feat = self.upsample(h_f)

            # Occurrence logits: [B, 1, H, W]
            step_logits = self.occurrence_head(dec_feat)
            # Flash rate: [B, 1, H, W] non-negative
            step_rates = F.softplus(self.flash_rate_head(dec_feat))

            pred_logits_list.append(step_logits)
            pred_rates_list.append(step_rates)

        # Stack over forecast horizon: [B, T_out, 1, H, W]
        lightning_logits = torch.stack(pred_logits_list, dim=1)
        flash_rate_pred = torch.stack(pred_rates_list, dim=1)
        lightning_probs = torch.sigmoid(lightning_logits)

        return {
            "lightning_logits": lightning_logits,
            "lightning_probs": lightning_probs,
            "flash_rate_pred": flash_rate_pred,
        }
