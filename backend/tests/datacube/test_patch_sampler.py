"""
Unit tests for ConvectivePatchSampler.
"""
from __future__ import annotations

import pytest

from datacube.patch_sampler import ConvectivePatchSampler, PatchCoordinate


class TestConvectivePatchSampler:
    def test_extract_patch_coordinates_bounds(self, synthetic_unified_cube):
        sampler = ConvectivePatchSampler(
            patch_size=(32, 32),
            T_in=4,
            T_out=4,
            spatial_stride=16,
        )

        patches = sampler.extract_patch_coordinates(synthetic_unified_cube)
        assert len(patches) > 0

        n_lat = len(synthetic_unified_cube["latitude"])
        n_lon = len(synthetic_unified_cube["longitude"])

        for p in patches:
            assert p.patch_height == 32
            assert p.patch_width == 32
            assert 0 <= p.lat_start_idx <= n_lat - 32
            assert 0 <= p.lon_start_idx <= n_lon - 32
            assert 0 <= p.t_start_idx <= len(synthetic_unified_cube["time"]) - 8

    def test_convective_patches_flagged(self, synthetic_unified_cube):
        sampler = ConvectivePatchSampler(
            patch_size=(32, 32),
            T_in=4,
            T_out=4,
            spatial_stride=16,
            min_reflectivity_dbz=35.0,
        )

        patches = sampler.extract_patch_coordinates(synthetic_unified_cube)

        convective = [p for p in patches if p.is_convective]
        assert len(convective) > 0, "Sampler should identify convective burst steps 4-7"

        for p in convective:
            # Must have either lightning or high reflectivity
            assert p.target_flash_count > 0 or p.max_reflectivity_dbz >= 35.0
