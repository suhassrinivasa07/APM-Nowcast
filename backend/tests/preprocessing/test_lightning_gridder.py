"""
Unit tests for LightningGridder (discrete point binning).
"""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing.lightning.lightning_gridder import LightningGridder
from backend.tests.preprocessing.conftest_preprocessing import (
    synthetic_lightning_csv_file,
    test_grid,
    prep_tmp_dir,
)


class TestLightningGridder:
    def test_discrete_binning_integrity(self, synthetic_lightning_csv_file, test_grid):
        gridder = LightningGridder(target_grid=test_grid)
        # Target time is 06:00:00 UTC with 30 min accumulation window [05:30:00, 06:00:00]
        ds = gridder.process_file(
            synthetic_lightning_csv_file,
            target_time="2024-06-01T06:00:00Z",
            accumulation_minutes=30.0,
        )

        assert "flash_count" in ds
        assert "occurrence_mask" in ds
        assert "peak_current_max" in ds
        assert "detection_coverage_mask" in ds

        flash_count = ds["flash_count"].values
        occurrence = ds["occurrence_mask"].values
        peak_curr = ds["peak_current_max"].values

        # Out of 6 synthetic events, exactly 4 were inside the 30-min window
        total_flashes = int(np.sum(flash_count))
        assert total_flashes == 4, f"Expected 4 accumulated flashes, got {total_flashes}"

        # Occurrence mask must be strictly binary: {0, 1}
        unique_occ = np.unique(occurrence)
        assert set(unique_occ).issubset({0, 1})
        assert np.array_equal(occurrence == 1, flash_count > 0)

        # Discrete Cell Check: Two strokes were at (20.2, 80.2) and (20.3, 80.4)
        # Under 1.0 degree grid, they land in the exact same cell [20-21 N, 80-81 E]
        assert np.max(flash_count) == 2, "Cell with 2 strokes must have count=2"

        # NaN Preservation for Peak Current: Cells with 0 flashes must have peak_current_max == NaN (not 0.0)
        zero_cells = flash_count == 0
        assert np.all(np.isnan(peak_curr[zero_cells])), "Peak current for 0-count cells must be NaN, not 0.0"

        # Cells with flashes must have valid non-NaN peak current
        active_cells = flash_count > 0
        assert np.all(~np.isnan(peak_curr[active_cells]))

    def test_cg_and_ic_counts(self, synthetic_lightning_csv_file, test_grid):
        gridder = LightningGridder(target_grid=test_grid)
        ds = gridder.process_file(
            synthetic_lightning_csv_file,
            target_time="2024-06-01T06:00:00Z",
            accumulation_minutes=30.0,
        )

        cg_total = int(np.sum(ds["cg_flash_count"].values))
        ic_total = int(np.sum(ds["ic_flash_count"].values))

        # In conftest: inside window, 3 strokes were CG and 1 was IC
        assert cg_total == 3
        assert ic_total == 1
        assert cg_total + ic_total == int(np.sum(ds["flash_count"].values))
