"""
End-to-End APMNowcastModel — Uncertainty-Aware Physics-Guided Adaptive Multimodal
Deep Learning Architecture for Thunderstorm Initiation & Lightning Nowcasting.

Assembles:
  1. Modality-Specific Encoders (satellite, radar, lightning, NWP)
  2. Adaptive Cross-Modal Attention Fusion with sensor dropout protection
  3. FiLM Physics-Guided Conditioning from ERA5 thermodynamic fields
  4. Multi-Scale ConvGRU Spatiotemporal Evolution Core
  5. Dual Probabilistic + Uncertainty-Aware Forecast Heads

Inherits from BaseNowcastModel for seamless integration with the datacube
DataLoader and baseline comparison harness.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from common.logging_config import get_logger
from datacube.schema import (
    MODALITY_ORDER,
    MODALITY_TO_INDEX,
    NUM_MODALITIES,
    ModalityType,
)
from models.base import BaseNowcastModel

from .adaptive_fusion import AdaptiveModalityFusion
from .encoders import (
    LightningEncoder,
    NWPEncoder,
    RadarEncoder,
    SatelliteEncoder,
)
from .physics_conditioning import PhysicsFiLMConditioning
from .spatiotemporal_core import SpatiotemporalEvolutionCore
from .uncertainty_heads import UncertaintyForecastHeads

log = get_logger(__name__)


class APMNowcastModel(BaseNowcastModel):
    """
    Uncertainty-Aware Physics-Guided Adaptive Multimodal Nowcasting Network.

    Parameters
    ----------
    T_in : int
        Number of observed input time steps (default: 4).
    T_out : int
        Number of forecast horizon time steps (default: 4).
    latent_dim : int
        Shared latent spatial feature dimension per modality encoder (default: 24).
    hidden_dim : int
        Hidden dimension for the ConvGRU evolution core (default: 48).
    mc_dropout_rate : float
        Dropout probability for MC-Dropout epistemic uncertainty (default: 0.1).
    mc_samples : int
        Number of MC forward passes for epistemic estimation (default: 8).
    """

    def __init__(
        self,
        T_in: int = 4,
        T_out: int = 4,
        latent_dim: int = 24,
        hidden_dim: int = 48,
        mc_dropout_rate: float = 0.1,
        mc_samples: int = 8,
    ) -> None:
        super().__init__(T_in=T_in, T_out=T_out)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Modality name ordering (matches MODALITY_ORDER indices in masks)
        self._modality_names = [m.value for m in MODALITY_ORDER]

        # --- 1. Modality-Specific Encoders ---
        self.encoders = nn.ModuleDict({
            "satellite": SatelliteEncoder(latent_dim=latent_dim),
            "radar": RadarEncoder(latent_dim=latent_dim),
            "lightning": LightningEncoder(latent_dim=latent_dim),
            "nwp": NWPEncoder(latent_dim=latent_dim),
        })

        # --- 2. Adaptive Cross-Modal Fusion ---
        self.fusion = AdaptiveModalityFusion(
            latent_dim=latent_dim,
            num_modalities=NUM_MODALITIES,
        )

        # --- 3. Physics-Guided FiLM Conditioning ---
        self.physics_conditioning = PhysicsFiLMConditioning(
            latent_dim=latent_dim,
            physics_dim=latent_dim,  # NWP encoder output dim
        )

        # --- 4. Spatiotemporal Evolution Core ---
        self.evolution_core = SpatiotemporalEvolutionCore(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            T_out=T_out,
        )

        # --- 5. Uncertainty-Aware Forecast Heads ---
        self.forecast_heads = UncertaintyForecastHeads(
            latent_dim=latent_dim,
            upsample_scale=2,
            mc_dropout_rate=mc_dropout_rate,
            mc_samples=mc_samples,
        )

    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        estimate_epistemic: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through APM-Nowcast architecture.

        Parameters
        ----------
        inputs_by_modality : Dict[str, Tensor]
            Per-modality input tensors [B, T_in, C_m, H, W].
        modality_masks : Tensor
            Binary sensor availability [B, T_in, NUM_MODALITIES].
        estimate_epistemic : bool
            If True, compute MC-Dropout epistemic uncertainty (expensive).

        Returns
        -------
        Dict[str, Tensor] containing:
            Standard: lightning_logits, lightning_probs, flash_rate_pred
            Research: aleatoric_uncertainty, epistemic_uncertainty,
                      modality_attention_weights, physics_gate_stats
        """
        # Determine reference shape
        ref = None
        for t in inputs_by_modality.values():
            if t is not None:
                ref = t
                break
        if ref is None:
            raise ValueError("All input modalities are None.")

        B, T_in, _, H, W = ref.shape
        device = ref.device
        dtype = ref.dtype

        # --- 1. Encode each modality independently ---
        encoded = {}
        for mod_name in self._modality_names:
            mod_idx = MODALITY_TO_INDEX[ModalityType(mod_name)]
            if mod_name in inputs_by_modality and inputs_by_modality[mod_name] is not None:
                raw = inputs_by_modality[mod_name].to(device, dtype)
                # Apply modality mask: zero out dropped modalities before encoding
                mask_m = modality_masks[:, :, mod_idx].view(B, T_in, 1, 1, 1).to(device, dtype)
                raw = raw * mask_m
                encoded[mod_name] = self.encoders[mod_name](raw)
            else:
                # Missing modality -> zero latent (encoder input is all zeros)
                C_m = self.modality_channel_counts[mod_name]
                zero_input = torch.zeros(B, T_in, C_m, H, W, device=device, dtype=dtype)
                encoded[mod_name] = self.encoders[mod_name](zero_input)

        # --- 2. Adaptive cross-modal fusion ---
        fused, modality_weights = self.fusion(
            encoded_modalities=encoded,
            modality_masks=modality_masks,
            modality_order=self._modality_names,
        )
        # fused: [B, T_in, D, H/2, W/2]
        # modality_weights: [B, M, H/2, W/2]

        # --- 3. Physics-guided conditioning ---
        nwp_encoded = encoded["nwp"]
        conditioned, physics_stats = self.physics_conditioning(fused, nwp_encoded)
        # conditioned: [B, T_in, D, H/2, W/2]

        # --- 4. Spatiotemporal evolution ---
        forecast_latents = self.evolution_core(conditioned)
        # forecast_latents: [B, T_out, D, H/2, W/2]

        # --- 5. Forecast heads with uncertainty ---
        outputs = self.forecast_heads(
            forecast_latents,
            estimate_epistemic=estimate_epistemic,
        )
        # Adds: lightning_logits, lightning_probs, flash_rate_pred,
        #        aleatoric_uncertainty, epistemic_uncertainty

        # --- 6. Attach interpretability outputs ---
        outputs["modality_attention_weights"] = modality_weights
        outputs["physics_gate_stats"] = physics_stats

        return outputs
