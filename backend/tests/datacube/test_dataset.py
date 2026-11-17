"""
Unit tests for PyTorch AtmosphericDataCubeDataset and DataLoader pipeline.
"""
from __future__ import annotations

import pytest
import torch

from datacube.dataset import AtmosphericDataCubeDataset, create_dataloaders
from datacube.normalizer import ModalityNormalizer
from datacube.patch_sampler import ConvectivePatchSampler
from datacube.splitter import SpatiotemporalSplitter


class TestAtmosphericDataset:
    def test_dataset_item_shapes_and_types(self, synthetic_unified_cube):
        normalizer = ModalityNormalizer().fit(synthetic_unified_cube)
        sampler = ConvectivePatchSampler(patch_size=(32, 32), T_in=4, T_out=4, spatial_stride=16)
        patches = sampler.extract_patch_coordinates(synthetic_unified_cube)

        assert len(patches) > 0
        dataset = AtmosphericDataCubeDataset(
            cube=synthetic_unified_cube,
            patch_coords=patches,
            normalizer=normalizer,
            T_in=4,
            T_out=4,
        )

        item = dataset[0]

        assert "inputs_by_modality" in item
        assert "modality_masks" in item
        assert "physics_fields" in item
        assert "target_lightning_frames" in item
        assert "target_occurrence_frames" in item

        inputs = item["inputs_by_modality"]
        assert "satellite" in inputs
        assert "radar" in inputs
        assert "lightning" in inputs
        assert "nwp" in inputs

        # Shapes: [T_in, C_mod, H, W]
        assert inputs["satellite"].shape == (4, 3, 32, 32)
        assert inputs["radar"].shape == (4, 2, 32, 32)
        assert inputs["lightning"].shape == (4, 2, 32, 32)
        assert inputs["nwp"].shape == (4, 4, 32, 32)

        # Modality mask shape: [T_in, NUM_MODALITIES]
        assert item["modality_masks"].shape == (4, 4)

        # Targets: [T_out, H, W]
        assert item["target_lightning_frames"].shape == (4, 32, 32)
        assert item["target_occurrence_frames"].shape == (4, 32, 32)

        # CRITICAL TEST: ZERO NaNs in any tensor
        for mod_name, tensor in inputs.items():
            assert not torch.isnan(tensor).any().item(), f"NaNs found in {mod_name} tensor!"

        assert not torch.isnan(item["target_lightning_frames"]).any().item()
        assert not torch.isnan(item["target_occurrence_frames"]).any().item()

    def test_dataloader_batch_iteration_zero_nans(self, synthetic_unified_cube):
        times = synthetic_unified_cube["time"].values
        split = SpatiotemporalSplitter.chronological_split(
            times,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            buffer_frames=1,
        )

        train_loader, val_loader, test_loader, normalizer = create_dataloaders(
            cube=synthetic_unified_cube,
            train_indices=split.train_indices,
            val_indices=split.val_indices,
            test_indices=split.test_indices,
            patch_size=(32, 32),
            T_in=4,
            T_out=4,
            batch_size=2,
            num_workers=0,
        )

        assert len(train_loader) > 0
        batch = next(iter(train_loader))

        sat_batch = batch["inputs_by_modality"]["satellite"]
        # Batch shape: [B, T_in, C, H, W] -> [2, 4, 3, 32, 32]
        assert sat_batch.shape == (2, 4, 3, 32, 32)
        assert not torch.isnan(sat_batch).any().item()

        targets = batch["target_lightning_frames"]
        assert targets.shape == (2, 4, 32, 32)
        assert not torch.isnan(targets).any().item()

        mod_masks = batch["modality_masks"]
        assert mod_masks.shape == (2, 4, 4)

    def test_cuda_transfer_compatibility(self, synthetic_unified_cube):
        """Verify tensor transfer to CUDA without type or memory exceptions."""
        normalizer = ModalityNormalizer().fit(synthetic_unified_cube)
        sampler = ConvectivePatchSampler(patch_size=(32, 32), T_in=4, T_out=4, spatial_stride=32)
        patches = sampler.extract_patch_coordinates(synthetic_unified_cube)
        dataset = AtmosphericDataCubeDataset(synthetic_unified_cube, patches, normalizer, T_in=4, T_out=4)

        item = dataset[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer tensors to device
        gpu_sat = item["inputs_by_modality"]["satellite"].to(device)
        gpu_mask = item["modality_masks"].to(device)
        gpu_target = item["target_lightning_frames"].to(device)

        assert gpu_sat.device.type == device.type
        assert gpu_mask.device.type == device.type
        assert gpu_target.device.type == device.type

        if device.type == "cuda":
            # Verify memory consumption is minimal
            allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            assert allocated_mb < 200.0, f"VRAM allocation ({allocated_mb} MB) unexpectedly high"
