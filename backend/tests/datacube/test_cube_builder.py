"""
Unit tests for DataCubeBuilder and Zarr storage.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from datacube.cube_builder import DataCubeBuilder
from datacube.schema import MODALITY_TO_INDEX, ModalityType


class TestDataCubeBuilder:
    def test_build_cube_dimensions(self, synthetic_unified_cube, cube_grid, synthetic_time_steps):
        cube = synthetic_unified_cube
        assert "time" in cube.dims
        assert "latitude" in cube.dims
        assert "longitude" in cube.dims

        assert len(cube["time"]) == len(synthetic_time_steps)
        assert len(cube["latitude"]) == cube_grid.shape[0]
        assert len(cube["longitude"]) == cube_grid.shape[1]

        # Verify channels from each modality are present
        assert "tir1" in cube
        assert "composite_reflectivity" in cube
        assert "flash_count" in cube
        assert "cape" in cube
        assert "modality_mask" in cube

    def test_modality_availability_mask_tracks_dropout(self, synthetic_unified_cube):
        cube = synthetic_unified_cube
        mod_mask = cube["modality_mask"].values
        n_times = len(cube["time"])

        assert mod_mask.shape == (n_times, 4)

        rad_idx = MODALITY_TO_INDEX[ModalityType.RADAR]
        sat_idx = MODALITY_TO_INDEX[ModalityType.SATELLITE]

        # Satellite was present across all steps
        assert np.all(mod_mask[:, sat_idx] == 1)

        # Radar was intentionally dropped out at index 3 and index 7
        assert mod_mask[3, rad_idx] == 0
        assert mod_mask[7, rad_idx] == 0
        # Radar was present at other steps
        assert mod_mask[0, rad_idx] == 1
        assert mod_mask[4, rad_idx] == 1

        # Check that where radar was dropped, radar values are filled with NaN
        rad_arr = cube["composite_reflectivity"].values
        assert np.all(np.isnan(rad_arr[3])), "Dropped radar step must contain NaNs"
        assert np.all(np.isnan(rad_arr[7])), "Dropped radar step must contain NaNs"
        assert not np.all(np.isnan(rad_arr[4])), "Active radar step must contain valid values"

    def test_zarr_save_and_reload(self, synthetic_unified_cube, datacube_tmp_dir, cube_grid):
        builder = DataCubeBuilder(grid=cube_grid)
        zarr_path = datacube_tmp_dir / "test_unified_cube.zarr"

        # Persist with zarr_format=2, consolidated=False
        saved_path = builder.save_cube_zarr(synthetic_unified_cube, zarr_path)
        assert saved_path.exists()

        # Reload
        reloaded = builder.load_cube_zarr(saved_path)
        assert "tir1" in reloaded
        assert "modality_mask" in reloaded
        assert reloaded["modality_mask"].shape == (len(synthetic_unified_cube["time"]), 4)
        assert len(reloaded["time"]) == len(synthetic_unified_cube["time"])
        reloaded.close()
