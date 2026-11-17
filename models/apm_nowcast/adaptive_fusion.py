"""
Mask-Aware Adaptive Cross-Modal Attention & Reliability Gating for APM-Nowcast.

Dynamically re-weights available sensor modalities using learned attention while
strictly zeroing out dropped/missing sensors. When a modality mask is 0, its
attention weight is set to -inf before softmax so the resulting weight is exactly 0.0.

Output:
  fused: [B, T_in, D, H/2, W/2] — adaptively fused multimodal latent
  modality_weights: [B, M, Hh, Wh] — per-pixel sensor reliability weights for explainability
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datacube.schema import NUM_MODALITIES


class AdaptiveModalityFusion(nn.Module):
    """
    Mask-aware cross-modal spatial attention fusion.

    For each spatial position, computes per-modality importance scores from the
    encoded latent representations, masks out unavailable sensors with -inf,
    and applies softmax to produce normalized reliability weights.
    """

    def __init__(self, latent_dim: int = 24, num_modalities: int = NUM_MODALITIES) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.M = num_modalities

        # Per-modality importance scoring: D -> 1 per spatial position
        self.importance_proj = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_dim // 2, 1, kernel_size=1, bias=True),
        )

        # Post-fusion refinement
        self.refine = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(4, latent_dim), latent_dim),
            nn.GELU(),
        )

    def forward(
        self,
        encoded_modalities: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        modality_order: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoded_modalities : dict mapping modality name -> [B, T_in, D, Hh, Wh]
        modality_masks : [B, T_in, M]
        modality_order : list of modality names matching mask index order

        Returns
        -------
        fused : [B, T_in, D, Hh, Wh]
        modality_weights : [B, M, Hh, Wh] (time-averaged attention map for explainability)
        """
        # Collect latent tensors in canonical modality order
        latents = []
        for mod_name in modality_order:
            latents.append(encoded_modalities[mod_name])

        # Stack: [B, T_in, M, D, Hh, Wh]
        stacked = torch.stack(latents, dim=2)
        B, T, M, D, Hh, Wh = stacked.shape

        # Flatten B*T*M for importance scoring
        flat = stacked.reshape(B * T * M, D, Hh, Wh)
        scores = self.importance_proj(flat)  # [B*T*M, 1, Hh, Wh]
        scores = scores.reshape(B, T, M, 1, Hh, Wh)

        # Apply modality masks: set unavailable sensors to -inf
        # modality_masks: [B, T, M] -> [B, T, M, 1, 1, 1]
        mask_expanded = modality_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Where mask is 0, set score to -inf to produce exact 0 after softmax
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        # Softmax across modality dimension: [B, T, M, 1, Hh, Wh]
        # Handle edge case where ALL modalities are masked for a sample (avoid NaN from all-inf softmax)
        all_masked = (modality_masks.sum(dim=-1, keepdim=True) == 0).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1, 1]
        weights = F.softmax(scores, dim=2)  # [B, T, M, 1, Hh, Wh]

        # If all modalities are masked, softmax produces NaN — replace with 0
        weights = weights.masked_fill(all_masked.unsqueeze(2), 0.0)

        # Weighted sum: [B, T, D, Hh, Wh]
        fused = (weights * stacked).sum(dim=2)

        # Refinement pass (temporal weight-sharing)
        fused_flat = fused.reshape(B * T, D, Hh, Wh)
        fused_flat = self.refine(fused_flat) + fused_flat  # residual
        fused = fused_flat.reshape(B, T, D, Hh, Wh)

        # Time-averaged attention weights for explainability: [B, M, Hh, Wh]
        modality_weights = weights.squeeze(3).mean(dim=1)  # mean over T_in

        return fused, modality_weights


def _safe_groups(desired: int, channels: int) -> int:
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g
