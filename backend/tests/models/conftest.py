"""
Pytest fixtures for APM-Nowcast models testing.
"""
from typing import Dict

import pytest
import torch

from datacube.schema import MODALITY_ORDER, MODALITY_TO_INDEX, NUM_MODALITIES, ModalityType


@pytest.fixture
def sample_batch() -> Dict[str, object]:
    """
    Standard multimodal test batch:
    - B=2, T_in=4, H=64, W=64
    - Satellite: 3 channels (tir1, wv, mir)
    - Radar: 2 channels (reflectivity, echo_top)
    - Lightning: 2 channels (flash_count, occurrence_mask)
    - NWP: 4 channels (cape, cin, total_totals, wind_shear)
    - Modality masks: [2, 4, 4] with all sensors active (1.0)
    """
    B, T_in, H, W = 2, 4, 64, 64

    # Synthetic realistic distributions
    satellite = torch.randn(B, T_in, 3, H, W)
    radar = torch.clamp(torch.randn(B, T_in, 2, H, W) * 10.0 + 20.0, 0.0, 70.0)

    # Lightning: sparse flashes in the center
    lightning = torch.zeros(B, T_in, 2, H, W)
    lightning[:, :, 0, 28:36, 28:36] = torch.poisson(torch.ones(B, T_in, 8, 8) * 3.0)
    lightning[:, :, 1, 28:36, 28:36] = (lightning[:, :, 0, 28:36, 28:36] > 0).float()

    nwp = torch.randn(B, T_in, 4, H, W)

    inputs_by_modality = {
        "satellite": satellite,
        "radar": radar,
        "lightning": lightning,
        "nwp": nwp,
    }

    # All sensors present: [B, T_in, NUM_MODALITIES]
    modality_masks = torch.ones(B, T_in, NUM_MODALITIES, dtype=torch.float32)

    return {
        "inputs_by_modality": inputs_by_modality,
        "modality_masks": modality_masks,
        "batch_size": B,
        "T_in": T_in,
        "T_out": 4,
        "height": H,
        "width": W,
    }


@pytest.fixture
def sample_batch_missing_radar(sample_batch: Dict[str, object]) -> Dict[str, object]:
    """
    Batch where radar modality is missing / dropped (e.g., radar downtime).
    """
    inputs = dict(sample_batch["inputs_by_modality"])
    inputs["radar"] = None

    masks = sample_batch["modality_masks"].clone()
    rad_idx = MODALITY_TO_INDEX[ModalityType.RADAR]
    masks[:, :, rad_idx] = 0.0

    return {
        "inputs_by_modality": inputs,
        "modality_masks": masks,
        "batch_size": sample_batch["batch_size"],
        "T_in": sample_batch["T_in"],
        "T_out": sample_batch["T_out"],
        "height": sample_batch["height"],
        "width": sample_batch["width"],
    }
