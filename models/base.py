"""
Abstract Base Model Interface for APM-Nowcast Spatiotemporal Forecasting.

Defines:
- Standardized forward pass signature consuming multimodal inputs and availability masks.
- Output contract: lightning_logits, lightning_probs, and flash_rate_pred of shape [B, T_out, 1, H, W].
- Parameter counting and VRAM footprint estimation.
- Robust multimodal fusion with sensor dropout protection.
"""
from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger
from datacube.schema import (
    MODALITY_ORDER,
    MODALITY_TO_INDEX,
    NUM_MODALITIES,
    ModalityType,
    get_channels_for_modality,
    get_modality_channel_counts,
)

log = get_logger(__name__)


class BaseNowcastModel(nn.Module, abc.ABC):
    """
    Abstract Base Class for all APM-Nowcast baseline and deep learning models.
    """

    def __init__(self, T_in: int = 4, T_out: int = 4) -> None:
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.modality_channel_counts = get_modality_channel_counts()

    @abc.abstractmethod
    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Standardized forward pass.

        Parameters
        ----------
        inputs_by_modality : Dict[str, torch.Tensor]
            Dictionary containing tensors per sensor modality:
            - 'satellite': [B, T_in, C_sat, H, W]
            - 'radar':     [B, T_in, C_rad, H, W]
            - 'lightning': [B, T_in, C_lit, H, W]
            - 'nwp':       [B, T_in, C_nwp, H, W]
        modality_masks : torch.Tensor
            Binary tensor of shape [B, T_in, NUM_MODALITIES] indicating sensor availability
            (1 = present, 0 = dropped/missing).

        Returns
        -------
        Dict[str, torch.Tensor] containing:
            - 'lightning_logits': [B, T_out, 1, H, W] (raw occurrence logits)
            - 'lightning_probs':  [B, T_out, 1, H, W] (sigmoid probabilities in [0, 1])
            - 'flash_rate_pred':  [B, T_out, 1, H, W] (non-negative flash density >= 0 via Softplus)
        """
        pass

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count total or trainable parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def estimate_vram_mb(
        self,
        batch_size: int = 4,
        T_in: int = 4,
        T_out: int = 4,
        patch_size: Tuple[int, int] = (64, 64),
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """
        Estimate parameter and activation memory footprint in Megabytes.
        """
        bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        param_bytes = sum(p.numel() for p in self.parameters()) * bytes_per_elem

        # Estimate forward activation tensor volume (inputs + hidden activations + outputs)
        H, W = patch_size
        total_in_channels = sum(self.modality_channel_counts.values())
        input_elements = batch_size * T_in * total_in_channels * H * W
        output_elements = batch_size * T_out * 2 * H * W
        # Rough heuristic: activations ~ 4x input+output elements
        activation_bytes = (input_elements + output_elements) * 4 * bytes_per_elem

        total_mb = (param_bytes + activation_bytes) / (1024 * 1024)
        return round(total_mb, 2)

    def fuse_multimodal_inputs(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Fuse multimodal inputs into a single contiguous tensor [B, T_in, C_total, H, W]
        with sensor dropout protection.

        If a modality is missing from inputs_by_modality or flagged as 0 in modality_masks,
        its channels are safely zeroed out so unmasked missing values never leak into the model.
        """
        # Determine reference shape from any available modality
        ref_tensor = None
        for t in inputs_by_modality.values():
            if t is not None:
                ref_tensor = t
                break

        if ref_tensor is None:
            raise ValueError("All modality input tensors are None.")

        B, T_in, _, H, W = ref_tensor.shape
        target_device = device or ref_tensor.device
        target_dtype = dtype or ref_tensor.dtype

        modality_slices: List[torch.Tensor] = []

        for mod_type in MODALITY_ORDER:
            mod_name = mod_type.value
            expected_c = self.modality_channel_counts[mod_name]
            mod_idx = MODALITY_TO_INDEX[mod_type]

            # Binary mask for this modality: [B, T_in, 1, 1, 1]
            if modality_masks.ndim == 3 and modality_masks.shape[-1] == NUM_MODALITIES:
                m_mask = modality_masks[:, :, mod_idx].view(B, T_in, 1, 1, 1).to(target_device, target_dtype)
            else:
                m_mask = torch.ones((B, T_in, 1, 1, 1), device=target_device, dtype=target_dtype)

            if mod_name in inputs_by_modality and inputs_by_modality[mod_name] is not None:
                m_tensor = inputs_by_modality[mod_name].to(target_device, target_dtype)
                # Apply sensor availability mask: zero out dropped sensors
                masked_tensor = m_tensor * m_mask
            else:
                # Modality completely missing in inputs dict -> zero placeholder
                masked_tensor = torch.zeros((B, T_in, expected_c, H, W), device=target_device, dtype=target_dtype)

            modality_slices.append(masked_tensor)

        # Concatenate along channel dimension: [B, T_in, C_total, H, W]
        fused = torch.cat(modality_slices, dim=2)
        total_c = fused.shape[2]
        return fused, total_c

    @staticmethod
    def validate_output_contract(
        outputs: Dict[str, torch.Tensor],
        expected_B: int,
        expected_T_out: int,
        expected_H: int,
        expected_W: int,
    ) -> bool:
        """Verify that model output conforms to the APM-Nowcast prediction contract."""
        required_keys = {"lightning_logits", "lightning_probs", "flash_rate_pred"}
        if not required_keys.issubset(outputs.keys()):
            return False

        expected_shape = (expected_B, expected_T_out, 1, expected_H, expected_W)

        for k in required_keys:
            t = outputs[k]
            if t.shape != expected_shape:
                return False
            if torch.isnan(t).any().item():
                return False

        # Validate probability bounds [0, 1]
        probs = outputs["lightning_probs"]
        if (probs < 0.0).any().item() or (probs > 1.0).any().item():
            return False

        # Validate non-negative flash rate >= 0
        rates = outputs["flash_rate_pred"]
        if (rates < 0.0).any().item():
            return False

        return True
