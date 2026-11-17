"""
High-level INSAT-3D/3DR/3DS downloader orchestrator.

Supports:
  - Automated download via MOSDACClient when credentials are present
  - Local directory scan for pre-downloaded HDF5 files
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from common.logging_config import get_logger
from common.schemas import DatasetInventoryRecord, IngestionStatus
from data_ingestion.common.download_logger import InventoryLogger
from data_ingestion.common.file_validator import (
    compute_checksum,
    validate_format,
)
from data_ingestion.satellite.mosdac_client import MOSDACClient
from data_ingestion.satellite.metadata_reader import read_insat_metadata

log = get_logger(__name__)


class SatelliteDownloader:
    """
    Download and ingest INSAT-3D/3DR/3DS Level-1B imagery.

    Parameters
    ----------
    local_dir       : Directory to store / scan for HDF5 satellite files.
    inventory_dir   : Directory for JSON-lines and SQLite inventory.
    client          : MOSDACClient instance; a default unauthenticated one
                      is created if not provided.
    """

    def __init__(
        self,
        local_dir: Path,
        inventory_dir: Path,
        client: Optional[MOSDACClient] = None,
    ) -> None:
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._inv    = InventoryLogger(inventory_dir)
        self._client = client or MOSDACClient()

    # ------------------------------------------------------------------
    def ingest_local_directory(
        self, source_dir: Optional[Path] = None, pattern: str = "*.h5"
    ) -> list[DatasetInventoryRecord]:
        """
        Scan a local directory for satellite HDF5 files, validate, and
        log each into the inventory.
        """
        scan_dir = source_dir or self._local_dir
        files = sorted(scan_dir.glob(pattern))
        log.info(
            "Scanning local satellite directory",
            extra={"dir": str(scan_dir), "found": len(files)},
        )
        return [r for f in files if (r := self._process_local_file(f)) is not None]

    def download_product(
        self,
        remote_path: str,
        filename: Optional[str] = None,
    ) -> Optional[DatasetInventoryRecord]:
        """
        Download a single INSAT product from MOSDAC.

        Returns DatasetInventoryRecord (status=FAILED if download fails).
        """
        name = filename or Path(remote_path).name
        dest = self._local_dir / name

        success = self._client.download(remote_path, dest)
        if not success:
            rec = DatasetInventoryRecord(
                record_id=str(uuid.uuid4()),
                source="MOSDAC",
                dataset_type="satellite",
                file_path=str(dest),
                status=IngestionStatus.FAILED,
                error_message="Download failed or credentials not configured",
            )
            self._inv.record(rec)
            return rec

        return self._process_local_file(dest)

    # ------------------------------------------------------------------
    def _process_local_file(self, path: Path) -> Optional[DatasetInventoryRecord]:
        """Validate, extract metadata, and register a satellite file."""
        record_id = str(uuid.uuid4())

        fmt_res = validate_format(path, "hdf5")
        if not fmt_res:
            log.warning(
                "Satellite file format invalid",
                extra={"path": str(path), "reason": fmt_res.reason},
            )
            rec = DatasetInventoryRecord(
                record_id=record_id,
                source="MOSDAC",
                dataset_type="satellite",
                file_path=str(path),
                status=IngestionStatus.FAILED,
                error_message=fmt_res.reason,
            )
            self._inv.record(rec)
            return rec

        try:
            meta = read_insat_metadata(path)
        except Exception as exc:
            log.warning("INSAT metadata extraction failed", extra={"path": str(path), "exc": str(exc)})
            meta = {}

        checksum = compute_checksum(path)
        status   = IngestionStatus.LOCAL if not self._client.is_authenticated else IngestionStatus.VALIDATED

        rec = DatasetInventoryRecord(
            record_id=record_id,
            source="MOSDAC",
            dataset_type="satellite",
            file_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            checksum_md5=checksum,
            status=status,
            valid_time_start=meta.get("valid_time"),
            variables=meta.get("datasets", []),
            metadata=meta,
        )
        self._inv.record(rec)
        log.info("Satellite file ingested", extra={"path": str(path), "size": rec.file_size_bytes})
        return rec
