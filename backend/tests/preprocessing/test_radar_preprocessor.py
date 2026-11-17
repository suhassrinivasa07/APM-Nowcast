"""
Unit tests for RadarPreprocessor (Indian DWR).
"""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing.radar.radar_preprocessor import RadarPreprocessor
from backend.tests.preprocessing.conftest_preprocessing import (
    synthetic_dwr_file,
    test_grid,
    prep_tmp_dir,
)


class TestRadarPreprocessor:
    def test_process_file_basic(self, synthetic_dwr_file, test_grid):
        preprocessor = RadarPreprocessor(target_grid=test_grid, station_id="VABB")
        ds = preprocessor.process_file(synthetic_dwr_file, target_time="2024-06-01T06:00:00Z")

        assert "composite_reflectivity" in ds
        assert "composite_reflectivity_valid_mask" in ds
        assert "reflectivity_cappi" in ds
        assert "radial_velocity" in ds

        assert ds["composite_reflectivity"].shape == test_grid.shape
        assert ds.attrs["station_id"] == "VABB"

    def test_radar_clutter_filter(self, synthetic_dwr_file, test_grid):
        # When clutter filter is enabled
        preprocessor = RadarPreprocessor(target_grid=test_grid, clutter_filter=True)
        ds = preprocessor.process_file(synthetic_dwr_file)

        comp = ds["composite_reflectivity"].values
        # At least one region contains valid echoes (~30 dBZ)
        valid_echoes = comp[~np.isnan(comp)]
        assert len(valid_echoes) > 0

        # Areas outside radar observation radius or filtered for clutter MUST be NaN
        mask = ds["composite_reflectivity_valid_mask"].values
        assert np.any(~mask)
        assert np.all(np.isnan(comp[~mask])), "Missing or clutter pixels must be NaN and NEVER 0.0"
