"""Tests for data_ingestion/common/file_validator.py using synthetic files."""
from __future__ import annotations

import pytest
from pathlib import Path

from data_ingestion.common.file_validator import (
    compute_checksum,
    validate_file_exists,
    validate_format,
    validate_hdf5_datasets,
    validate_netcdf_variables,
    validate_netcdf_bounds,
)
from common.schemas import GeoBoundingBox

from backend.tests.ingestion.conftest_ingestion import (
    synthetic_insat_hdf5,
    synthetic_era5_netcdf,
    synthetic_dwr_netcdf,
    synthetic_imerg_hdf5,
    synthetic_lightning_csv,
    tmp_data_root,
)


class TestFileExists:
    def test_existing_file_passes(self, synthetic_insat_hdf5):
        result = validate_file_exists(synthetic_insat_hdf5)
        assert result.ok
        assert result.file_size_bytes > 0

    def test_missing_file_fails(self, tmp_path):
        result = validate_file_exists(tmp_path / "nonexistent.h5")
        assert not result.ok
        assert "not found" in result.reason.lower()

    def test_empty_file_fails(self, tmp_path):
        p = tmp_path / "empty.h5"
        p.touch()
        result = validate_file_exists(p)
        assert not result.ok


class TestFormatValidation:
    def test_valid_hdf5(self, synthetic_insat_hdf5):
        assert validate_format(synthetic_insat_hdf5, "hdf5").ok

    def test_valid_netcdf4(self, synthetic_era5_netcdf):
        # ERA5 NC4 files use HDF5 container — same magic bytes
        assert validate_format(synthetic_era5_netcdf, "netcdf4").ok

    def test_valid_csv(self, synthetic_lightning_csv):
        assert validate_format(synthetic_lightning_csv, "csv").ok

    def test_csv_as_hdf5_fails(self, synthetic_lightning_csv):
        assert not validate_format(synthetic_lightning_csv, "hdf5").ok


class TestNetCDFVariables:
    def test_required_variables_present(self, synthetic_era5_netcdf):
        result = validate_netcdf_variables(synthetic_era5_netcdf, ["t2m", "cape"])
        assert result.ok

    def test_missing_variable_detected(self, synthetic_era5_netcdf):
        result = validate_netcdf_variables(synthetic_era5_netcdf, ["nonexistent_var"])
        assert not result.ok
        assert "nonexistent_var" in result.reason


class TestHDF5Datasets:
    def test_required_datasets_present(self, synthetic_insat_hdf5):
        assert validate_hdf5_datasets(synthetic_insat_hdf5, ["IMG_TIR1", "IMG_MIR"]).ok

    def test_missing_dataset_detected(self, synthetic_insat_hdf5):
        result = validate_hdf5_datasets(synthetic_insat_hdf5, ["NONEXISTENT_DATASET"])
        assert not result.ok


class TestNetCDFBounds:
    def test_valid_bounds_intersect(self, synthetic_era5_netcdf):
        bbox = GeoBoundingBox(lat_min=5.0, lat_max=35.0, lon_min=65.0, lon_max=95.0)
        assert validate_netcdf_bounds(synthetic_era5_netcdf, bbox).ok

    def test_disjoint_bounds_fail(self, synthetic_era5_netcdf):
        bbox = GeoBoundingBox(lat_min=50.0, lat_max=80.0, lon_min=0.0, lon_max=30.0)
        assert not validate_netcdf_bounds(synthetic_era5_netcdf, bbox).ok


class TestChecksum:
    def test_md5_deterministic(self, synthetic_insat_hdf5):
        c1 = compute_checksum(synthetic_insat_hdf5)
        c2 = compute_checksum(synthetic_insat_hdf5)
        assert c1 == c2
        assert len(c1) == 32  # MD5 hex length
