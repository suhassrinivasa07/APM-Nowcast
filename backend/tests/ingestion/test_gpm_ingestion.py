"""Tests for NASA GPM IMERG ingestion pipeline."""
from __future__ import annotations

import os
import pytest

from common.schemas import IngestionStatus
from data_ingestion.gpm.imerg_loader import IMERGLoader, _read_imerg_metadata

from backend.tests.ingestion.conftest_ingestion import (
    synthetic_imerg_hdf5,
    tmp_data_root,
)


class TestIMERGMetadata:
    def test_reads_grid_group(self, synthetic_imerg_hdf5):
        meta = _read_imerg_metadata(synthetic_imerg_hdf5)
        assert "Grid" in meta["datasets"]

    def test_reads_file_header(self, synthetic_imerg_hdf5):
        meta = _read_imerg_metadata(synthetic_imerg_hdf5)
        assert "2024-06-01" in meta.get("file_header_raw", "")


class TestIMERGLoader:
    def test_ingest_local_hdf5(self, synthetic_imerg_hdf5, tmp_path):
        loader = IMERGLoader(
            local_dir=synthetic_imerg_hdf5.parent,
            inventory_dir=tmp_path / "gpm_inv",
        )
        records = loader.ingest_local(path=synthetic_imerg_hdf5)
        assert len(records) == 1
        rec = records[0]
        assert rec.source      == "NASA-GPM-IMERG"
        assert rec.dataset_type == "gpm"
        assert rec.file_size_bytes > 0

    def test_no_earthdata_token_means_local_mode(self, tmp_path):
        backup = os.environ.pop("EARTHDATA_TOKEN", None)
        try:
            loader = IMERGLoader(
                local_dir=tmp_path / "gpm",
                inventory_dir=tmp_path / "inv",
            )
            assert not loader.is_authenticated
        finally:
            if backup is not None:
                os.environ["EARTHDATA_TOKEN"] = backup
