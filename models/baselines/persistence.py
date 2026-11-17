"""
Physical Meteorological Baseline Models: Eulerian & Advection Persistence.

Zero-parameter benchmarks standard in weather nowcasting:
1. Eulerian Persistence: Assumes stationary atmospheric conditions: X(t + tau) = X(t).
2. Advection Persistence: Estimates Lagrangian motion fields (optical flow displacement)
   from consecutive radar/satellite scans and advects lightning fields forward in time
   using differentiable spatial grid sampling.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger
from datacube.schema import MODALITY_TO_INDEX, ModalityType
from models.base import BaseNowcastModel

log = get_logger(__name__)


class EulerianPersistenceModel(BaseNowcastModel):
    """
    Zero-parameter Eulerian Persistence benchmark.
    Repeats the latest observed lightning state across all future forecast horizons.
    """

    def __init__(self, T_in: int = 4, T_out: int = 4) -> None:
        super().__init__(T_in=T_in, T_out=T_out)
        # No learnable parameters

    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Determine reference shape and device
        ref_tensor = None
        for t in inputs_by_modality.values():
            if t is not None:
                ref_tensor = t
                break

        if ref_tensor is None:
            raise ValueError("All inputs_by_modality tensors are None.")

        B, T_in, _, H, W = ref_tensor.shape
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        lit_idx = MODALITY_TO_INDEX[ModalityType.LIGHTNING]

        # Extract last frame of lightning if available and not dropped
        lit_present = (
            "lightning" in inputs_by_modality
            and inputs_by_modality["lightning"] is not None
            and (modality_masks[:, -1, lit_idx] > 0).any()
        )

        if lit_present:
            lit_tensor = inputs_by_modality["lightning"]
            # channels: 0 = flash_count, 1 = occurrence_mask
            last_flash = lit_tensor[:, -1:, 0:1, :, :]  # [B, 1, 1, H, W]
            last_occ = lit_tensor[:, -1:, 1:2, :, :]    # [B, 1, 1, H, W]

            # Zero out any sample where lightning mask was 0 at last step
            m_mask = modality_masks[:, -1:, lit_idx].view(B, 1, 1, 1, 1)
            last_flash = last_flash * m_mask
            last_occ = last_occ * m_mask
        else:
            # Completely missing lightning -> 0 probability
            last_flash = torch.zeros((B, 1, 1, H, W), device=device, dtype=dtype)
            last_occ = torch.zeros((B, 1, 1, H, W), device=device, dtype=dtype)

        # Broadcast across T_out forecast horizons: [B, T_out, 1, H, W]
        flash_rate_pred = torch.clamp(last_flash.repeat(1, self.T_out, 1, 1, 1), min=0.0)
        lightning_probs = torch.clamp(last_occ.repeat(1, self.T_out, 1, 1, 1), 0.0, 1.0)

        # Invert probability to logits: logit(p) with epsilon stabilization
        eps = 1e-4
        clamped_probs = torch.clamp(lightning_probs, eps, 1.0 - eps)
        lightning_logits = torch.log(clamped_probs / (1.0 - clamped_probs))

        return {
            "lightning_logits": lightning_logits,
            "lightning_probs": lightning_probs,
            "flash_rate_pred": flash_rate_pred,
        }


class AdvectionPersistenceModel(BaseNowcastModel):
    """
    Zero-parameter Lagrangian Advection benchmark.
    Computes apparent motion field from consecutive radar or satellite scans,
    then recursively advects the last observed lightning field forward in time.
    """

    def __init__(self, T_in: int = 4, T_out: int = 4, smoothness: float = 1.0) -> None:
        super().__init__(T_in=T_in, T_out=T_out)
        self.smoothness = smoothness

    def forward(
        self,
        inputs_by_modality: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        ref_tensor = next(t for t in inputs_by_modality.values() if t is not None)
        B, T_in, _, H, W = ref_tensor.shape
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        # 1. Estimate motion vectors (u, v) from the best available tracker field (radar > satellite > lightning)
        motion_field = self._estimate_motion(inputs_by_modality, modality_masks, B, H, W, device, dtype)

        # 2. Extract last observed lightning state
        lit_idx = MODALITY_TO_INDEX[ModalityType.LIGHTNING]
        lit_present = (
            "lightning" in inputs_by_modality
            and inputs_by_modality["lightning"] is not None
            and (modality_masks[:, -1, lit_idx] > 0).any()
        )

        if lit_present:
            lit_tensor = inputs_by_modality["lightning"]
            m_mask = modality_masks[:, -1:, lit_idx].view(B, 1, 1, 1, 1)
            curr_flash = (lit_tensor[:, -1, 0:1, :, :] * m_mask[:, 0]).to(dtype)  # [B, 1, H, W]
            curr_occ = (lit_tensor[:, -1, 1:2, :, :] * m_mask[:, 0]).to(dtype)    # [B, 1, H, W]
        else:
            curr_flash = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
            curr_occ = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        # 3. Base coordinate grid for sampling in [-1, 1] normalized space
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # Convert pixel motion to normalized grid displacement [-2/W, 2/W]
        norm_motion = motion_field.clone()
        norm_motion[:, 0] = norm_motion[:, 0] * (2.0 / max(1, W))
        norm_motion[:, 1] = norm_motion[:, 1] * (2.0 / max(1, H))
        norm_motion = norm_motion.permute(0, 2, 3, 1)  # [B, H, W, 2]

        pred_flash_slices = []
        pred_occ_slices = []

        advected_flash = curr_flash
        advected_occ = curr_occ

        # Recursive forward advection across T_out horizons
        for step in range(1, self.T_out + 1):
            # Backward sampling grid: grid_t = base_grid - step * norm_motion
            sampling_grid = base_grid - (step * norm_motion)
            sampling_grid = torch.clamp(sampling_grid, -1.0, 1.0)

            # Bilinear interpolation warp
            warped_flash = F.grid_sample(curr_flash, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            warped_occ = F.grid_sample(curr_occ, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

            pred_flash_slices.append(warped_flash)
            pred_occ_slices.append(warped_occ)

        # Stack to [B, T_out, 1, H, W]
        flash_pred = torch.stack(pred_flash_slices, dim=1)
        occ_pred = torch.stack(pred_occ_slices, dim=1)

        flash_rate_pred = torch.clamp(flash_pred, min=0.0)
        lightning_probs = torch.clamp(occ_pred, 0.0, 1.0)

        eps = 1e-4
        clamped_probs = torch.clamp(lightning_probs, eps, 1.0 - eps)
        lightning_logits = torch.log(clamped_probs / (1.0 - clamped_probs))

        return {
            "lightning_logits": lightning_logits,
            "lightning_probs": lightning_probs,
            "flash_rate_pred": flash_rate_pred,
        }

    def _estimate_motion(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Estimate 2D apparent motion velocity (u, v) using spatiotemporal gradients.
        Returns tensor of shape [B, 2, H, W].
        """
        # Select best available tracking field
        tracker = None
        if "radar" in inputs and inputs["radar"] is not None:
            rad_idx = MODALITY_TO_INDEX[ModalityType.RADAR]
            if (masks[:, -2:, rad_idx] > 0).all():
                tracker = inputs["radar"][:, :, 0, :, :]  # composite reflectivity
        if tracker is None and "satellite" in inputs and inputs["satellite"] is not None:
            sat_idx = MODALITY_TO_INDEX[ModalityType.SATELLITE]
            if (masks[:, -2:, sat_idx] > 0).all():
                tracker = -inputs["satellite"][:, :, 0, :, :]  # -TIR1 (clouds are bright)
        if tracker is None and "lightning" in inputs and inputs["lightning"] is not None:
            tracker = inputs["lightning"][:, :, 0, :, :]

        if tracker is None or tracker.shape[1] < 2:
            return torch.zeros((B, 2, H, W), device=device, dtype=dtype)

        # Consecutive frames t-2 and t-1
        f1 = tracker[:, -2]  # [B, H, W]
        f2 = tracker[:, -1]  # [B, H, W]

        # Temporal difference
        It = (f2 - f1).unsqueeze(1)  # [B, 1, H, W]

        # Spatial gradients using Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0

        f2_pad = F.pad(f2.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        Ix = F.conv2d(f2_pad, sobel_x)
        Iy = F.conv2d(f2_pad, sobel_y)

        # Optical flow normal flow approximation: v = - (It / (||grad||^2 + eps)) * grad
        grad_mag_sq = Ix ** 2 + Iy ** 2 + self.smoothness
        u = - (It * Ix) / grad_mag_sq
        v = - (It * Iy) / grad_mag_sq

        # Clip excessive velocities to realistic storm advection speeds (max 8 pixels per frame)
        max_vel = 8.0
        u = torch.clamp(u, -max_vel, max_vel)
        v = torch.clamp(v, -max_vel, max_vel)

        motion = torch.cat([u, v], dim=1)  # [B, 2, H, W]
        return motion
