"""
Differentiable Atmospheric Physics Consistency Penalties for APM-Nowcast.

Encodes domain-expert meteorological knowledge as soft constraints:

1. **Convective Inhibition Suppression:**  Penalizes high lightning probability
   predictions in strongly stable regimes (low CAPE / high CIN), where deep
   convective initiation is physically unlikely.

2. **Moisture-Convergence Alignment:**  Low-level moisture convergence is a
   necessary ingredient for thunderstorm sustenance; penalizes lightning
   predictions in regions of moisture divergence (dry subsiding air).

3. **Temporal Smoothness:**  Penalizes excessive frame-to-frame flicker in
   predicted lightning probability fields, encoding the physical reality that
   convective systems evolve smoothly over 15–30 minute nowcasting horizons.

Gracefully returns 0.0 if `physics_fields` is None so baseline models
(persistence, ConvLSTM, U-Net) can also use APMCompositeLoss.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class PhysicsConsistencyLoss(nn.Module):
    """
    Differentiable physics-guided penalty encouraging meteorologically
    consistent lightning predictions.
    """

    def __init__(
        self,
        cape_suppression_weight: float = 1.0,
        cin_suppression_weight: float = 1.0,
        moisture_weight: float = 0.5,
        temporal_smoothness_weight: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cape_w = cape_suppression_weight
        self.cin_w = cin_suppression_weight
        self.moisture_w = moisture_weight
        self.temporal_w = temporal_smoothness_weight
        self.eps = eps

    def forward(
        self,
        lightning_probs: torch.Tensor,
        physics_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        lightning_probs : [B, T_out, 1, H, W] predicted occurrence probability
        physics_fields  : dict mapping variable names to tensors.
                          Expected keys (optional): 'cape', 'cin',
                          'relative_humidity_2m', 'wind_speed_10m'.
                          Each tensor has shape [B, T_in, H, W] (last observed frame used).
                          If None, returns 0.0 (graceful fallback for baselines).

        Returns
        -------
        Scalar physics consistency penalty
        """
        if physics_fields is None or len(physics_fields) == 0:
            return torch.zeros((), device=lightning_probs.device, dtype=lightning_probs.dtype)

        total_penalty = torch.tensor(0.0, device=lightning_probs.device, dtype=lightning_probs.dtype)

        # Use average of predicted lightning probability across forecast horizon
        # [B, 1, H, W]
        mean_probs = lightning_probs.mean(dim=1, keepdim=False).squeeze(1)  # [B, H, W]

        # --- 1. CAPE Suppression: penalize high p(lightning) where CAPE is low ---
        if "cape" in physics_fields:
            cape = physics_fields["cape"]
            # Use last observed timestep: [B, H, W]
            cape_last = cape[:, -1] if cape.ndim == 4 else cape
            # Suppress indicator: low CAPE -> suppress = 1, high CAPE -> suppress ~ 0
            # Sigmoid centered at 0 (normalized CAPE), scale controls sharpness
            cape_suppress = torch.sigmoid(-cape_last * 3.0)  # ~1 when CAPE << 0 (below mean)
            penalty_cape = (mean_probs * cape_suppress).mean()
            total_penalty = total_penalty + self.cape_w * penalty_cape

        # --- 2. CIN Suppression: penalize high p(lightning) where CIN is high ---
        if "cin" in physics_fields:
            cin = physics_fields["cin"]
            cin_last = cin[:, -1] if cin.ndim == 4 else cin
            # High CIN (positive after normalization) suppresses convection
            cin_suppress = torch.sigmoid(cin_last * 3.0)  # ~1 when CIN >> 0 (above mean)
            penalty_cin = (mean_probs * cin_suppress).mean()
            total_penalty = total_penalty + self.cin_w * penalty_cin

        # --- 3. Moisture-Convergence Alignment ---
        if "relative_humidity_2m" in physics_fields:
            rh = physics_fields["relative_humidity_2m"]
            rh_last = rh[:, -1] if rh.ndim == 4 else rh
            # Low humidity -> moisture divergence / dry air -> suppress lightning
            dry_suppress = torch.sigmoid(-rh_last * 2.0)  # ~1 when RH << mean
            penalty_moisture = (mean_probs * dry_suppress).mean()
            total_penalty = total_penalty + self.moisture_w * penalty_moisture

        # --- 4. Temporal Smoothness: penalize frame-to-frame flicker ---
        if lightning_probs.shape[1] > 1:
            # Temporal differences across forecast horizons: [B, T_out-1, 1, H, W]
            temporal_diff = lightning_probs[:, 1:] - lightning_probs[:, :-1]
            penalty_smooth = (temporal_diff ** 2).mean()
            total_penalty = total_penalty + self.temporal_w * penalty_smooth

        return total_penalty
