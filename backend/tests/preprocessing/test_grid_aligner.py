"""
Unit tests for GridAligner and CommonGrid.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from common.schemas import GeoBoundingBox
from preprocessing.common.grid_aligner import CommonGrid, GridAligner


class TestCommonGrid:
    def test_grid_initialization_and_properties(self):
        grid = CommonGrid(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, resolution_deg=0.5)
        assert grid.lat_min == 10.0
        assert grid.lat_max == 20.0
        assert len(grid.lats) == 21
        assert len(grid.lons) == 21
        assert grid.shape == (21, 21)
        assert grid.lat_edges.shape == (22,)
        assert grid.lon_edges.shape == (22,)

    def test_from_bbox(self):
        bbox = GeoBoundingBox(lat_min=5.0, lat_max=35.0, lon_min=65.0, lon_max=95.0)
        grid = CommonGrid.from_bbox(bbox, resolution_deg=1.0)
        assert grid.lat_min == 5.0
        assert grid.lat_max == 35.0
        assert len(grid.lats) == 31


class TestGridAligner:
    def test_resample_2d_regular_preserves_nan(self):
        grid = CommonGrid(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)
        aligner = GridAligner(grid)

        src_lats = np.linspace(10.0, 20.0, 11, dtype=np.float32)
        src_lons = np.linspace(70.0, 80.0, 11, dtype=np.float32)
        src_data = np.ones((11, 11), dtype=np.float32) * 50.0

        # Inject NaNs
        src_data[3:6, 3:6] = np.nan

        resampled, valid_mask = aligner.resample_2d_regular(src_lats, src_lons, src_data, method="nearest")

        assert resampled.shape == grid.shape
        # Crucial Requirement: Missing observations MUST be NaN, NEVER silent 0
        assert np.any(np.isnan(resampled)), "NaNs were not preserved in resampled grid!"
        assert np.any(valid_mask), "Valid pixels missing in mask"
        assert not np.all(valid_mask), "Invalid mask did not flag NaN region"

        # Check that where valid_mask is False, resampled is NaN (not 0.0)
        invalid_locs = ~valid_mask
        assert np.all(np.isnan(resampled[invalid_locs])), "Invalid locations must be NaN!"

    def test_resample_out_of_bounds_is_nan(self):
        # Target grid is larger than source coverage
        target_grid = CommonGrid(lat_min=5.0, lat_max=25.0, lon_min=65.0, lon_max=85.0, resolution_deg=1.0)
        aligner = GridAligner(target_grid)

        # Source data only covers 10 - 20 N
        src_lats = np.linspace(10.0, 20.0, 5, dtype=np.float32)
        src_lons = np.linspace(70.0, 80.0, 5, dtype=np.float32)
        src_data = np.full((5, 5), 25.0, dtype=np.float32)

        resampled, valid_mask = aligner.resample_2d_regular(src_lats, src_lons, src_data)

        # Areas outside 10-20 N, 70-80 E must be NaN
        out_of_bounds = (target_grid.lats < 10.0) | (target_grid.lats > 20.0)
        tgt_lons_2d, tgt_lats_2d = target_grid.meshgrid()
        oob_mask = (tgt_lats_2d < 10.0) | (tgt_lats_2d > 20.0)

        assert np.all(np.isnan(resampled[oob_mask])), "Out-of-bounds coordinates must be filled with NaN"

    def test_align_xarray_dataset(self):
        grid = CommonGrid(lat_min=12.0, lat_max=18.0, lon_min=72.0, lon_max=78.0, resolution_deg=0.5)
        aligner = GridAligner(grid)

        src_lats = np.linspace(10.0, 20.0, 11)
        src_lons = np.linspace(70.0, 80.0, 11)
        ds = xr.Dataset(
            data_vars={
                "temperature": (["latitude", "longitude"], np.full((11, 11), 300.0, dtype=np.float32)),
                "humidity": (["latitude", "longitude"], np.full((11, 11), 75.0, dtype=np.float32)),
            },
            coords={"latitude": src_lats, "longitude": src_lons},
        )

        aligned_ds = aligner.align_xarray_dataset(ds)

        assert "temperature" in aligned_ds
        assert "temperature_valid_mask" in aligned_ds
        assert "humidity" in aligned_ds
        assert "humidity_valid_mask" in aligned_ds
        assert aligned_ds["temperature"].shape == grid.shape
        assert len(aligned_ds["latitude"]) == len(grid.lats)
        assert len(aligned_ds["longitude"]) == len(grid.lons)
