"""
Spatiotemporal U-Net Baseline for Multimodal Lightning Nowcasting.

Architectural Highlights:
- Early-fusion temporal-channel projection: [B, T_in * C_total, H, W].
- Residual Encoder-Decoder with skip connections and GroupNorm for stable small-batch training.
- Multi-horizon dual heads producing simultaneous occurrence probabilities and flash rates:
  [B, T_out, 1, H, W].
- Parameter budget < 0.9M parameters; VRAM consumption < 0.8 GB on [4, 4, C, 64, 64] batches.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger
from models.base import BaseNowcastModel

log = get_logger(__name__)


class DoubleConvBlock(nn.Module):
    """Residual double convolution block with GroupNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 4) -> None:
        super().__init__()
        # Ensure num_groups divides channels cleanly
        g1 = min(num_groups, out_channels)
        while out_channels % g1 != 0 and g1 > 1:
            g1 -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_channels),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.residual(x))


class SpatiotemporalUNetBaseline(BaseNowcastModel):
    """
    Spatiotemporal U-Net baseline predicting multi-horizon lightning occurrence and rate.
    """

    def __init__(
        self,
        T_in: int = 4,
        T_out: int = 4,
        total_in_channels: Optional[int] = None,
        base_channels: int = 16,
    ) -> None:
        super().__init__(T_in=T_in, T_out=T_out)
        self.c_in = total_in_channels or sum(self.modality_channel_counts.values())
        self.combined_in_channels = self.c_in * T_in

        c1 = base_channels       # 16
        c2 = base_channels * 2   # 32
        c3 = base_channels * 4   # 64

        # Encoder
        self.enc1 = DoubleConvBlock(self.combined_in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with dilated receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(4, c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, c3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(c3, c2)  # c2 (from up3) + c2 (from enc2) = c3

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(c2, c1)  # c1 (from up2) + c1 (from enc1) = c2

        self.up1 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)  # up from bottleneck to enc3
        self.dec_bottleneck = DoubleConvBlock(c3 + c3, c3)

        # Dual multi-horizon prediction heads
        # Outputs T_out forecast horizons directly: [B, T_out, H, W]
        self.occurrence_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, self.T_out, kernel_size=1),
        )

        self.flash_rate_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, self.T_out, kernel_size=1),
        )

    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 1. Fuse multimodal inputs: [B, T_in, C_total, H, W]
        fused, _ = self.fuse_multimodal_inputs(inputs_by_modality, modality_masks)
        B, T_in, C, H, W = fused.shape

        # 2. Reshape to temporal-channel stack: [B, T_in * C_total, H, W]
        x = fused.view(B, T_in * C, H, W)

        # 3. Encoder path
        e1 = self.enc1(x)         # [B, c1, H, W]
        p1 = self.pool1(e1)       # [B, c1, H/2, W/2]

        e2 = self.enc2(p1)        # [B, c2, H/2, W/2]
        p2 = self.pool2(e2)       # [B, c2, H/4, W/4]

        e3 = self.enc3(p2)        # [B, c3, H/4, W/4]
        p3 = self.pool3(e3)       # [B, c3, H/8, W/8]

        # 4. Bottleneck
        b = self.bottleneck(p3)   # [B, c3, H/8, W/8]

        # 5. Decoder path with skip connections
        u3 = self.up1(b)                                # [B, c3, H/4, W/4]
        d3 = self.dec_bottleneck(torch.cat([u3, e3], dim=1))

        u2 = self.up3(d3)                               # [B, c2, H/2, W/2]
        d2 = self.dec3(torch.cat([u2, e2], dim=1))

        u1 = self.up2(d2)                               # [B, c1, H, W]
        d1 = self.dec2(torch.cat([u1, e1], dim=1))

        # 6. Multi-horizon dual heads
        occ_raw = self.occurrence_head(d1)   # [B, T_out, H, W]
        rate_raw = self.flash_rate_head(d1)  # [B, T_out, H, W]

        # Shape to standard contract: [B, T_out, 1, H, W]
        lightning_logits = occ_raw.unsqueeze(2)
        lightning_probs = torch.sigmoid(lightning_logits)
        flash_rate_pred = F.softplus(rate_raw.unsqueeze(2))

        return {
            "lightning_logits": lightning_logits,
            "lightning_probs": lightning_probs,
            "flash_rate_pred": flash_rate_pred,
        }
