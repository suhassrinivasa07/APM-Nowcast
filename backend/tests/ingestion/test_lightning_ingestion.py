"""Tests for NRSC Lightning ingestion pipeline."""
from __future__ import annotations

from datetime import timezone
import pytest

from common.schemas import IngestionStatus
from data_ingestion.lightning.nrsc_loader import NRSCLoader
from data_ingestion.lightning.lightning_reader import read_lightning_csv

from backend.tests.ingestion.conftest_ingestion import (
    synthetic_lightning_csv,
    tmp_data_root,
)


class TestLightningCSVReader:
    def test_reads_records(self, synthetic_lightning_csv):
        df, meta = read_lightning_csv(synthetic_lightning_csv)
        assert len(df) == 20
        assert meta["n_records"] == 20

    def test_time_range_is_utc(self, synthetic_lightning_csv):
        df, meta = read_lightning_csv(synthetic_lightning_csv)
        assert meta["time_start"] is not None
        assert meta["time_start"].tzinfo == timezone.utc

    def test_bounding_box_in_india(self, synthetic_lightning_csv):
        _, meta = read_lightning_csv(synthetic_lightning_csv)
        bbox = meta["bbox"]
        assert 5.0 <= bbox["lat_min"] <= 40.0
        assert 65.0 <= bbox["lon_min"] <= 100.0

    def test_standardised_lat_column(self, synthetic_lightning_csv):
        df, _ = read_lightning_csv(synthetic_lightning_csv)
        assert "lat" in df.columns or "latitude" in df.columns


class TestNRSCLoader:
    def test_ingest_csv(self, synthetic_lightning_csv, tmp_path):
        loader = NRSCLoader(
            local_dir=synthetic_lightning_csv.parent,
            inventory_dir=tmp_path / "lightning_inv",
        )
        records = loader.ingest_csv(path=synthetic_lightning_csv)
        assert len(records) == 1
        rec = records[0]
        assert rec.source      == "NRSC-Lightning"
        assert rec.dataset_type == "lightning"
        assert rec.status      == IngestionStatus.VALIDATED
        assert rec.metadata["n_records"] == 20
