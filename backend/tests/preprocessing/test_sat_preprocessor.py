"""
Unit tests for SatellitePreprocessor (INSAT-3D/3DR).
"""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing.satellite.sat_preprocessor import SatellitePreprocessor
from backend.tests.preprocessing.conftest_preprocessing import (
    synthetic_sat_file,
    test_grid,
    prep_tmp_dir,
)


class TestSatellitePreprocessor:
    def test_process_file_basic(self, synthetic_sat_file, test_grid):
        preprocessor = SatellitePreprocessor(target_grid=test_grid)
        ds = preprocessor.process_file(synthetic_sat_file, target_time="2024-06-01T06:00:00Z")

        assert "tir1" in ds
        assert "tir1_valid_mask" in ds
        assert ds["tir1"].shape == test_grid.shape
        assert ds.attrs["satellite_name"] == "INSAT-3D"
        assert "temporal_offset_minutes" in ds.attrs
        assert ds.attrs["sync_status"] == "synchronized"

    def test_physical_masking_and_nan_preservation(self, synthetic_sat_file, test_grid):
        preprocessor = SatellitePreprocessor(target_grid=test_grid)
        ds = preprocessor.process_file(synthetic_sat_file)

        tir1 = ds["tir1"].values
        mask = ds["tir1_valid_mask"].values

        # Must have valid pixels
        assert np.any(mask)
        # Must have masked/NaN pixels from the injected 0.0, 400K, and -999K
        assert np.any(~mask)

        # CRITICAL TEST: Missing/unphysical pixels MUST be NaN, NEVER 0.0
        invalid_pixels = tir1[~mask]
        assert np.all(np.isnan(invalid_pixels)), "Masked pixels must be NaN and NEVER 0.0"

        # Valid pixels must lie within physical tropical convective limits (160 - 340 K)
        valid_pixels = tir1[mask]
        assert np.all(valid_pixels >= 160.0)
        assert np.all(valid_pixels <= 340.0)
