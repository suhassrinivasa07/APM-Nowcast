"""
NRSC Lightning ECV (Essential Climate Variable) data loader.

NRSC Lightning data is distributed as:
  - CSV files: datetime_utc, latitude, longitude, peak_current_kA, type
  - NetCDF annual climatology files

The NRSC portal requires institutional access; automated download is not
supported. This module ingests locally-stored files seamlessly.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from common.logging_config import get_logger
from common.schemas import DatasetInventoryRecord, IngestionStatus
from data_ingestion.common.download_logger import InventoryLogger
from data_ingestion.common.file_validator import compute_checksum, validate_file_exists
from data_ingestion.lightning.lightning_reader import read_lightning_csv, read_lightning_netcdf

log = get_logger(__name__)


class NRSCLoader:
    """
    Ingest NRSC Lightning ECV data from local CSV or NetCDF files.

    Parameters
    ----------
    local_dir     : Directory containing NRSC lightning files.
    inventory_dir : Inventory storage directory.
    """

    def __init__(self, local_dir: Path, inventory_dir: Path) -> None:
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._inv = InventoryLogger(inventory_dir)

    def ingest_csv(self, path: Optional[Path] = None) -> list[DatasetInventoryRecord]:
        """Ingest all CSV lightning files in local_dir (or a specific path)."""
        paths = [path] if path else sorted(self._local_dir.glob("*.csv"))
        return [r for p in paths if (r := self._process_csv(p)) is not None]

    def ingest_netcdf(self, path: Optional[Path] = None) -> list[DatasetInventoryRecord]:
        """Ingest all NetCDF lightning files."""
        paths = [path] if path else sorted(self._local_dir.glob("*.nc"))
        return [r for p in paths if (r := self._process_netcdf(p)) is not None]

    def _process_csv(self, path: Path) -> Optional[DatasetInventoryRecord]:
        res = validate_file_exists(path)
        if not res:
            return None

        try:
            df, meta = read_lightning_csv(path)
            status = IngestionStatus.VALIDATED
            error  = None
        except Exception as exc:
            meta   = {}
            df     = None
            status = IngestionStatus.FAILED
            error  = str(exc)
            log.error("Lightning CSV read failed", extra={"path": str(path), "exc": str(exc)})

        checksum = compute_checksum(path)
        rec = DatasetInventoryRecord(
            record_id=str(uuid.uuid4()),
            source="NRSC-Lightning",
            dataset_type="lightning",
            file_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            checksum_md5=checksum,
            status=status,
            valid_time_start=meta.get("time_start"),
            valid_time_end=meta.get("time_end"),
            variables=meta.get("columns", []),
            error_message=error,
            metadata={
                "n_records": int(meta.get("n_records", 0)),
                "bbox":      meta.get("bbox", {}),
            },
        )
        self._inv.record(rec)
        return rec

    def _process_netcdf(self, path: Path) -> Optional[DatasetInventoryRecord]:
        res = validate_file_exists(path)
        if not res:
            return None

        try:
            meta   = read_lightning_netcdf(path)
            status = IngestionStatus.VALIDATED
            error  = None
        except Exception as exc:
            meta   = {}
            status = IngestionStatus.FAILED
            error  = str(exc)

        checksum = compute_checksum(path)
        rec = DatasetInventoryRecord(
            record_id=str(uuid.uuid4()),
            source="NRSC-Lightning",
            dataset_type="lightning",
            file_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            checksum_md5=checksum,
            status=status,
            valid_time_start=meta.get("valid_time"),
            variables=meta.get("variables", []),
            error_message=error,
            metadata=meta,
        )
        self._inv.record(rec)
        return rec
