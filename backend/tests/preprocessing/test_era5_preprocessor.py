"""
Unit tests for ERA5Preprocessor.
"""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing.nwp.era5_preprocessor import ERA5Preprocessor
from backend.tests.preprocessing.conftest_preprocessing import (
    synthetic_era5_file,
    test_grid,
    prep_tmp_dir,
)


class TestERA5Preprocessor:
    def test_process_file_and_derived_features(self, synthetic_era5_file, test_grid):
        preprocessor = ERA5Preprocessor(target_grid=test_grid)
        # Select 06:00 UTC slice
        ds = preprocessor.process_file(synthetic_era5_file, target_time="2024-06-01T06:00:00Z")

        # Basic extracted variables
        assert "t2m" in ds
        assert "d2m" in ds
        assert "cape" in ds
        assert "u10" in ds
        assert "v10" in ds

        # Derived thermodynamic parameters
        assert "dewpoint_depression_2m" in ds
        assert "relative_humidity_2m" in ds
        assert "wind_speed_10m" in ds

        # Verify numerical values
        # T2m = 305 K, D2m = 295 K -> Dewpoint depression ~ 10 K
        dp_dep = ds["dewpoint_depression_2m"].values
        valid_dp = dp_dep[~np.isnan(dp_dep)]
        assert np.allclose(valid_dp, 10.0, atol=0.5)

        # 10m Wind: u=4, v=3 -> speed = sqrt(16 + 9) = 5 m/s
        wspd = ds["wind_speed_10m"].values
        valid_wspd = wspd[~np.isnan(wspd)]
        assert np.allclose(valid_wspd, 5.0, atol=0.2)

        # Relative humidity must be in range [0, 100]
        rh = ds["relative_humidity_2m"].values
        valid_rh = rh[~np.isnan(rh)]
        assert np.all(valid_rh >= 0.0)
        assert np.all(valid_rh <= 100.0)

        # Regridding shape check
        assert ds["t2m"].shape == test_grid.shape
        assert "t2m_valid_mask" in ds
