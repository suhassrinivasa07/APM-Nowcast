"""
Class-Imbalance-Aware Occurrence Loss for Sparse Lightning Nowcasting.

Combines:
  1. Focal Loss (gamma=2.0, alpha=0.75): Down-weights easy negatives (empty sky)
     while focusing gradient signal on hard misclassified convective cells.
  2. Differentiable Soft Dice Loss: Measures spatial overlap between predicted
     and observed binary lightning occurrence masks, robust to extreme class
     imbalance characteristic of Indian monsoon lightning patterns.

All computations are logit-based via F.binary_cross_entropy_with_logits for
FP16 numerical stability under torch.amp.autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Logit-based Focal Loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Uses F.binary_cross_entropy_with_logits internally for FP16 safety.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, T_out, 1, H, W] raw occurrence logits
        targets : [B, T_out, 1, H, W] binary occurrence (0 or 1)
        """
        # BCE with logits (numerically stable in FP16)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Predicted probability for focal modulation
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)

        # Focal modulation factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Class-balance alpha weighting
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class SoftDiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for binary segmentation.

    Dice = 2 * |P ∩ G| / (|P| + |G| + eps)
    Loss = 1 - Dice
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        probs   : [B, T_out, 1, H, W] sigmoid probabilities in [0, 1]
        targets : [B, T_out, 1, H, W] binary occurrence
        """
        # Flatten spatial + temporal dims for Dice computation
        p_flat = probs.reshape(probs.shape[0], -1)
        t_flat = targets.reshape(targets.shape[0], -1)

        intersection = (p_flat * t_flat).sum(dim=1)
        cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return (1.0 - dice).mean()


class OccurrenceLoss(nn.Module):
    """
    Combined Focal + Soft Dice occurrence loss for sparse lightning detection.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        dice_weight: float = 0.3,
        focal_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, T_out, 1, H, W] raw logits
        probs   : [B, T_out, 1, H, W] sigmoid probabilities
        targets : [B, T_out, 1, H, W] binary occurrence ground truth
        """
        return self.focal_weight * self.focal(logits, targets) + \
               self.dice_weight * self.dice(probs, targets)
