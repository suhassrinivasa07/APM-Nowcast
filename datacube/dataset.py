"""
PyTorch Dataset and DataLoader Pipeline for Atmospheric Multimodal Data Cubes.

Optimized for:
- NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) via compact 64x64 / 96x96 spatial patches.
- Zero-NaN tensor guarantees: missing observations and sensor dropouts are sentinel-imputed
  alongside explicit binary modality presence masks [T_in, M].
- Low memory footprint: float32 / float16 precision and on-the-fly patch slicing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import xarray as xr

from common.logging_config import get_logger
from datacube.normalizer import ModalityNormalizer
from datacube.patch_sampler import ConvectivePatchSampler, PatchCoordinate
from datacube.schema import (
    CHANNEL_REGISTRY,
    MODALITY_ORDER,
    MODALITY_TO_INDEX,
    NUM_MODALITIES,
    ModalityType,
    get_channels_for_modality,
)

log = get_logger(__name__)


class AtmosphericDataCubeDataset(Dataset):
    """
    PyTorch Dataset providing windowed spatiotemporal multimodal patches.
    """

    def __init__(
        self,
        cube: xr.Dataset,
        patch_coords: List[PatchCoordinate],
        normalizer: ModalityNormalizer,
        T_in: int = 4,
        T_out: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cube = cube
        self.patch_coords = patch_coords
        self.normalizer = normalizer
        self.T_in = T_in
        self.T_out = T_out
        self.T_total = T_in + T_out
        self.dtype = dtype

        # Pre-cache channel names per modality
        self.channels_by_modality: Dict[str, List[str]] = {
            mod.value: get_channels_for_modality(mod) for mod in MODALITY_ORDER
        }

    def __len__(self) -> int:
        return len(self.patch_coords)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        coord = self.patch_coords[idx]
        t0 = coord.t_start_idx
        r0 = coord.lat_start_idx
        c0 = coord.lon_start_idx
        H = coord.patch_height
        W = coord.patch_width

        # Time index slices
        hist_slice = slice(t0, t0 + self.T_in)
        fut_slice = slice(t0 + self.T_in, t0 + self.T_total)
        lat_slice = slice(r0, r0 + H)
        lon_slice = slice(c0, c0 + W)

        # 1. Modality Availability Mask: [T_in, NUM_MODALITIES]
        if "modality_mask" in self.cube:
            raw_mod_mask = self.cube["modality_mask"].isel(time=hist_slice).values.astype(np.float32)
        else:
            raw_mod_mask = np.ones((self.T_in, NUM_MODALITIES), dtype=np.float32)
        modality_masks = torch.from_numpy(raw_mod_mask).to(self.dtype)

        # 2. Extract and Normalize inputs per modality: [T_in, C_mod, H, W]
        inputs_by_modality: Dict[str, torch.Tensor] = {}

        for mod_type in MODALITY_ORDER:
            mod_name = mod_type.value
            channel_names = self.channels_by_modality[mod_name]
            channel_tensors = []

            for var_name in channel_names:
                if var_name in self.cube:
                    raw_patch = self.cube[var_name].isel(time=hist_slice, latitude=lat_slice, longitude=lon_slice).values
                    # Extract validity mask
                    vmask_key = f"{var_name}_valid_mask"
                    if vmask_key in self.cube:
                        vmask_patch = self.cube[vmask_key].isel(time=hist_slice, latitude=lat_slice, longitude=lon_slice).values
                    else:
                        vmask_patch = ~np.isnan(raw_patch)

                    # Transform with ModalityNormalizer and impute sentinels
                    norm_patch = self.normalizer.transform(var_name, raw_patch, valid_mask=vmask_patch)
                else:
                    # Variable not present in cube (e.g. partial sensor suite)
                    norm_patch = np.zeros((self.T_in, H, W), dtype=np.float32)

                # Ensure non-NaN guarantee
                if np.any(np.isnan(norm_patch)):
                    norm_patch = np.nan_to_num(norm_patch, nan=0.0)

                channel_tensors.append(norm_patch)

            # Stack along channel dimension: [T_in, C_mod, H, W]
            mod_array = np.stack(channel_tensors, axis=1).astype(np.float32)
            inputs_by_modality[mod_name] = torch.from_numpy(mod_array).to(self.dtype)

        # 3. Physics guidance fields (e.g. CAPE, CIN, relative humidity)
        physics_fields: Dict[str, torch.Tensor] = {}
        for p_var in ("cape", "cin", "relative_humidity_2m", "wind_speed_10m"):
            if p_var in self.cube:
                p_patch = self.cube[p_var].isel(time=hist_slice, latitude=lat_slice, longitude=lon_slice).values
                vmask_key = f"{p_var}_valid_mask"
                vmask = self.cube[vmask_key].isel(time=hist_slice, latitude=lat_slice, longitude=lon_slice).values if vmask_key in self.cube else ~np.isnan(p_patch)
                norm_p = self.normalizer.transform(p_var, p_patch, valid_mask=vmask)
                physics_fields[p_var] = torch.from_numpy(np.nan_to_num(norm_p, nan=0.0)).to(self.dtype)

        # 4. Target lightning forecast frames: [T_out, H, W]
        if "flash_count" in self.cube:
            fut_flash_raw = self.cube["flash_count"].isel(time=fut_slice, latitude=lat_slice, longitude=lon_slice).values
            fut_flash = np.nan_to_num(fut_flash_raw, nan=0.0).astype(np.float32)
        else:
            fut_flash = np.zeros((self.T_out, H, W), dtype=np.float32)

        if "occurrence_mask" in self.cube:
            fut_occ_raw = self.cube["occurrence_mask"].isel(time=fut_slice, latitude=lat_slice, longitude=lon_slice).values
            fut_occ = np.nan_to_num(fut_occ_raw, nan=0.0).astype(np.float32)
        else:
            fut_occ = (fut_flash > 0).astype(np.float32)

        return {
            "inputs_by_modality": inputs_by_modality,
            "modality_masks": modality_masks,
            "physics_fields": physics_fields,
            "target_lightning_frames": torch.from_numpy(fut_flash).to(self.dtype),
            "target_occurrence_frames": torch.from_numpy(fut_occ).to(self.dtype),
            "is_convective": coord.is_convective,
            "t_start_idx": coord.t_start_idx,
            "spatial_crop": (r0, c0, H, W),
        }


def create_dataloaders(
    cube: xr.Dataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    normalizer: Optional[ModalityNormalizer] = None,
    patch_size: Tuple[int, int] = (64, 64),
    T_in: int = 4,
    T_out: int = 4,
    spatial_stride: int = 32,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, ModalityNormalizer]:
    """
    Factory creating PyTorch DataLoaders for train, val, and test splits.
    Fits the normalizer ONLY on train_indices.
    """
    # 1. Fit Normalizer exclusively on training indices
    if normalizer is None or not normalizer.is_fitted():
        normalizer = ModalityNormalizer().fit(cube, train_indices=train_indices)

    sampler = ConvectivePatchSampler(
        patch_size=patch_size,
        T_in=T_in,
        T_out=T_out,
        spatial_stride=spatial_stride,
    )

    # 2. Extract patch coordinates per partition
    train_patches = sampler.extract_patch_coordinates(cube, time_indices=train_indices)
    val_patches = sampler.extract_patch_coordinates(cube, time_indices=val_indices)
    test_patches = sampler.extract_patch_coordinates(cube, time_indices=test_indices)

    # 3. Create Datasets
    train_ds = AtmosphericDataCubeDataset(cube, train_patches, normalizer, T_in=T_in, T_out=T_out)
    val_ds = AtmosphericDataCubeDataset(cube, val_patches, normalizer, T_in=T_in, T_out=T_out)
    test_ds = AtmosphericDataCubeDataset(cube, test_patches, normalizer, T_in=T_in, T_out=T_out)

    # 4. Construct DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(len(train_ds) > 0),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    log.info(
        "DataLoaders constructed",
        extra={
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "test_batches": len(test_loader),
            "batch_size": batch_size,
            "patch_size": patch_size,
        },
    )

    return train_loader, val_loader, test_loader, normalizer
