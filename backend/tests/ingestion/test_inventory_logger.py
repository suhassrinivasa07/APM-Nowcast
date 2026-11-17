"""Tests for InventoryLogger (JSON-lines + SQLite)."""
from __future__ import annotations

import pytest

from common.schemas import DatasetInventoryRecord, IngestionStatus
from data_ingestion.common.download_logger import InventoryLogger


@pytest.fixture
def inventory(tmp_path) -> InventoryLogger:
    return InventoryLogger(tmp_path / "inventory")


class TestInventoryLogger:
    def test_record_and_query_by_source(self, inventory):
        rec = DatasetInventoryRecord(
            record_id="test-001",
            source="ERA5-CDS",
            dataset_type="nwp",
            file_path="/data/era5.nc",
            status=IngestionStatus.VALIDATED,
            variables=["t2m", "cape"],
        )
        inventory.record(rec)
        results = inventory.query(source="ERA5-CDS")
        assert len(results) == 1
        assert results[0]["record_id"] == "test-001"

    def test_jsonl_file_is_created(self, inventory, tmp_path):
        inventory.record(DatasetInventoryRecord(
            record_id="test-002",
            source="MOSDAC",
            dataset_type="satellite",
            file_path="/tmp/insat.h5",
        ))
        jsonl = tmp_path / "inventory" / "inventory.jsonl"
        assert jsonl.exists()
        assert "MOSDAC" in jsonl.read_text()

    def test_update_status(self, inventory):
        inventory.record(DatasetInventoryRecord(
            record_id="test-003",
            source="NRSC-Lightning",
            dataset_type="lightning",
            file_path="/tmp/lightning.csv",
            status=IngestionStatus.PENDING,
        ))
        inventory.update_status("test-003", IngestionStatus.FAILED, error="Validation failed")
        results = inventory.query(status=IngestionStatus.FAILED)
        assert any(r["record_id"] == "test-003" for r in results)

    def test_summary_groups_by_source_and_status(self, inventory):
        for i, src in enumerate(["ERA5-CDS", "ERA5-CDS", "MOSDAC"]):
            inventory.record(DatasetInventoryRecord(
                record_id=f"s-{i}",
                source=src,
                dataset_type="nwp" if "ERA5" in src else "satellite",
                file_path=f"/tmp/file{i}.nc",
                status=IngestionStatus.VALIDATED,
            ))
        summary = inventory.summary()
        assert "ERA5-CDS" in summary
        assert summary["ERA5-CDS"]["validated"] == 2
