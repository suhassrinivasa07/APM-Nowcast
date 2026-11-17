"""Tests for IMD DWR radar ingestion pipeline."""
from __future__ import annotations

import pytest

from common.schemas import IngestionStatus
from data_ingestion.radar.radar_downloader import RadarDownloader
from data_ingestion.radar.radar_metadata import read_radar_metadata

from backend.tests.ingestion.conftest_ingestion import (
    synthetic_dwr_netcdf,
    tmp_data_root,
)


class TestRadarMetadataReader:
    def test_reads_variables(self, synthetic_dwr_netcdf):
        meta = read_radar_metadata(synthetic_dwr_netcdf)
        assert "DBZ"  in meta["variables"]
        assert "time" in meta["variables"]

    def test_reads_valid_time(self, synthetic_dwr_netcdf):
        meta = read_radar_metadata(synthetic_dwr_netcdf)
        assert meta["valid_time"] is not None
        assert meta["valid_time"].year == 2024

    def test_reads_lat_bounds(self, synthetic_dwr_netcdf):
        meta = read_radar_metadata(synthetic_dwr_netcdf)
        assert meta["lat_min"] == pytest.approx(18.0, abs=0.1)
        assert meta["lat_max"] == pytest.approx(22.0, abs=0.1)


class TestRadarDownloader:
    def test_ingest_local_directory(self, synthetic_dwr_netcdf, tmp_path):
        downloader = RadarDownloader(
            local_dir=synthetic_dwr_netcdf.parent,
            inventory_dir=tmp_path / "radar_inv",
            station_id="VABB",
        )
        records = downloader.ingest_local_directory()
        assert len(records) >= 1
        rec = records[0]
        assert rec.source == "IMD-DWR-VABB"
        assert rec.dataset_type == "radar"
        assert rec.file_size_bytes > 0

    def test_empty_directory_returns_empty(self, tmp_path):
        downloader = RadarDownloader(
            local_dir=tmp_path / "empty_radar",
            inventory_dir=tmp_path / "inv",
        )
        records = downloader.ingest_local_directory()
        assert records == []
