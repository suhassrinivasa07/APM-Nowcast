"""Tests for INSAT-3D satellite ingestion pipeline."""
from __future__ import annotations

import os
import pytest

from common.schemas import IngestionStatus
from data_ingestion.satellite.metadata_reader import read_insat_metadata
from data_ingestion.satellite.satellite_downloader import SatelliteDownloader
from data_ingestion.satellite.mosdac_client import MOSDACClient

from backend.tests.ingestion.conftest_ingestion import (
    synthetic_insat_hdf5,
    tmp_data_root,
)


class TestINSATMetadataReader:
    def test_reads_satellite_name(self, synthetic_insat_hdf5):
        meta = read_insat_metadata(synthetic_insat_hdf5)
        assert meta["satellite"] == "INSAT-3D"

    def test_reads_valid_time(self, synthetic_insat_hdf5):
        meta = read_insat_metadata(synthetic_insat_hdf5)
        assert meta["valid_time"] is not None
        assert meta["valid_time"].year == 2024

    def test_reads_dataset_list(self, synthetic_insat_hdf5):
        meta = read_insat_metadata(synthetic_insat_hdf5)
        assert "IMG_TIR1" in meta["datasets"]
        assert "IMG_MIR"  in meta["datasets"]

    def test_reads_center_coordinates(self, synthetic_insat_hdf5):
        meta = read_insat_metadata(synthetic_insat_hdf5)
        assert meta["lat_center"] == pytest.approx(22.5)
        assert meta["lon_center"] == pytest.approx(82.0)


class TestSatelliteDownloader:
    def test_ingest_local_file(self, synthetic_insat_hdf5, tmp_path):
        downloader = SatelliteDownloader(
            local_dir=synthetic_insat_hdf5.parent,
            inventory_dir=tmp_path / "inventory",
        )
        records = downloader.ingest_local_directory(source_dir=synthetic_insat_hdf5.parent)
        assert len(records) >= 1
        rec = records[0]
        assert rec.dataset_type == "satellite"
        assert rec.file_size_bytes > 0
        assert rec.checksum_md5 is not None
        assert len(rec.checksum_md5) == 32

    def test_unauthenticated_download_returns_failed_record(self, tmp_path):
        downloader = SatelliteDownloader(
            local_dir=tmp_path / "sat",
            inventory_dir=tmp_path / "inv",
        )
        rec = downloader.download_product("gallery/fake/product.h5")
        assert rec is not None
        assert rec.status == IngestionStatus.FAILED

    def test_mosdac_client_local_mode_without_credentials(self):
        backup = {}
        for k in ("MOSDAC_USER", "MOSDAC_PASSWORD"):
            backup[k] = os.environ.pop(k, None)
        try:
            client = MOSDACClient()
            assert not client.is_authenticated
            assert client.list_products("some/path") == []
        finally:
            for k, v in backup.items():
                if v is not None:
                    os.environ[k] = v
