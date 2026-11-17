"""Tests for ERA5 NWP ingestion pipeline."""
from __future__ import annotations

import os
import pytest

from common.schemas import IngestionStatus
from data_ingestion.nwp.era5_reader import read_era5_metadata
from data_ingestion.nwp.era5_downloader import ERA5Downloader, ConfigurationError

from backend.tests.ingestion.conftest_ingestion import (
    synthetic_era5_netcdf,
    tmp_data_root,
)


class TestERA5Reader:
    def test_reads_variables(self, synthetic_era5_netcdf):
        meta = read_era5_metadata(synthetic_era5_netcdf)
        assert "t2m"  in meta["variables"]
        assert "cape" in meta["variables"]

    def test_reads_lat_bounds(self, synthetic_era5_netcdf):
        meta = read_era5_metadata(synthetic_era5_netcdf)
        assert meta["lat_min"] == pytest.approx(10.0, abs=0.1)
        assert meta["lat_max"] == pytest.approx(30.0, abs=0.1)

    def test_reads_lon_bounds(self, synthetic_era5_netcdf):
        meta = read_era5_metadata(synthetic_era5_netcdf)
        assert meta["lon_min"] == pytest.approx(70.0, abs=0.1)
        assert meta["lon_max"] == pytest.approx(90.0, abs=0.1)

    def test_reads_time_range(self, synthetic_era5_netcdf):
        meta = read_era5_metadata(synthetic_era5_netcdf)
        assert meta["time_start"] is not None


class TestERA5DownloaderLocalMode:
    def test_offline_mode_flag(self, tmp_path):
        backup = {k: os.environ.pop(k, None) for k in ["CDSAPI_URL", "CDSAPI_KEY"]}
        try:
            dl = ERA5Downloader(
                local_dir=tmp_path / "era5",
                inventory_dir=tmp_path / "inv",
            )
            assert not dl.is_online
        finally:
            for k, v in backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_download_raises_config_error_when_offline(self, tmp_path):
        backup = {k: os.environ.pop(k, None) for k in ["CDSAPI_URL", "CDSAPI_KEY"]}
        try:
            dl = ERA5Downloader(
                local_dir=tmp_path / "era5",
                inventory_dir=tmp_path / "inv",
            )
            with pytest.raises(ConfigurationError):
                dl.download_single_levels(["2m_temperature"], 2024, 6)
        finally:
            for k, v in backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_ingest_local_file(self, synthetic_era5_netcdf, tmp_path):
        dl = ERA5Downloader(
            local_dir=synthetic_era5_netcdf.parent,
            inventory_dir=tmp_path / "inv",
        )
        rec = dl.ingest_local(synthetic_era5_netcdf)
        assert rec is not None
        assert rec.dataset_type == "nwp"
        assert rec.status       == IngestionStatus.LOCAL
        assert "t2m" in rec.variables
