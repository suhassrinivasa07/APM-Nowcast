"""
Integration tests for the end-to-end PreprocessingPipeline and PreprocessingReport.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from preprocessing.pipeline import PreprocessingPipeline, PreprocessingReport
from backend.tests.preprocessing.conftest_preprocessing import (
    synthetic_sat_file,
    synthetic_dwr_file,
    synthetic_lightning_csv_file,
    synthetic_era5_file,
    test_grid,
    prep_tmp_dir,
)


class TestPreprocessingPipeline:
    def test_process_all_modalities(
        self,
        synthetic_sat_file,
        synthetic_dwr_file,
        synthetic_lightning_csv_file,
        synthetic_era5_file,
        test_grid,
        prep_tmp_dir,
    ):
        output_dir = prep_tmp_dir / "pipeline_outputs"
        pipeline = PreprocessingPipeline(
            target_grid=test_grid,
            output_dir=output_dir,
            enable_plots=True,
        )

        target_time = "2024-06-01T06:00:00Z"
        datasets, report = pipeline.process_all(
            target_time=target_time,
            satellite_file=synthetic_sat_file,
            radar_file=synthetic_dwr_file,
            lightning_file=synthetic_lightning_csv_file,
            era5_file=synthetic_era5_file,
            save_format="netcdf",
        )

        # 1. Check all modalities processed
        assert "satellite" in datasets
        assert "radar" in datasets
        assert "lightning" in datasets
        assert "nwp" in datasets

        # 2. Check grid shape alignment across all modalities
        for name, ds in datasets.items():
            first_var = next(v for v in ds.data_vars if not v.endswith("_valid_mask"))
            assert ds[first_var].shape == test_grid.shape, f"{name} shape does not match target grid!"

        # 3. Check reports
        assert report.overall_status in ("PASSED", "WARNING")
        assert len(report.modalities) == 4
        assert "satellite" in report.modalities
        assert "radar" in report.modalities
        assert "lightning" in report.modalities
        assert "nwp" in report.modalities

        # 4. Check JSON report serialization
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["target_time_utc"] == "2024-06-01T06:00:00+00:00"
        assert len(data["modalities"]) == 4

        # 5. Check Markdown report generation
        md_str = report.to_markdown()
        assert "# APM-Nowcast Preprocessing Report" in md_str
        assert "SATELLITE" in md_str
        assert "RADAR" in md_str
        assert "LIGHTNING" in md_str
        assert "NWP" in md_str
        assert "Zero-Corruption Guard" in md_str

        # 6. Check report files on disk
        reports_on_disk = list(output_dir.glob("preprocessing_report_*.json"))
        assert len(reports_on_disk) >= 1
        md_reports_on_disk = list(output_dir.glob("preprocessing_report_*.md"))
        assert len(md_reports_on_disk) >= 1

        # 7. Check QC Plots generated
        qc_plots = list((output_dir / "qc_plots").glob("*.png"))
        assert len(qc_plots) >= 4, f"Expected 4 QC plots, found {len(qc_plots)}"
        for plot in qc_plots:
            assert plot.stat().st_size > 1000, f"QC plot {plot.name} appears empty or corrupted"

    def test_zarr_persistence_and_reload(
        self,
        synthetic_sat_file,
        test_grid,
        prep_tmp_dir,
    ):
        output_dir = prep_tmp_dir / "zarr_outputs"
        pipeline = PreprocessingPipeline(
            target_grid=test_grid,
            output_dir=output_dir,
            enable_plots=False,
        )

        datasets, report = pipeline.process_all(
            target_time="2024-06-01T06:00:00Z",
            satellite_file=synthetic_sat_file,
            save_format="zarr",
        )

        zarr_path = output_dir / "insat_20240601_060000.zarr"
        assert zarr_path.exists()

        # Reload with xarray
        reloaded = xr.open_zarr(str(zarr_path), consolidated=False)
        assert "tir1" in reloaded
        assert reloaded["tir1"].shape == test_grid.shape
        reloaded.close()
