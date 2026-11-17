"""
IMD Doppler Weather Radar (DWR) file ingestion.

IMD DWR data is typically provided as NetCDF / HDF5 exports from MOSDAC or
direct IMD workstation outputs. Since IMD DWR has no public API, this module
supports LOCAL directory ingestion of files obtained from IMD/MOSDAC portals.
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
    validate_file_exists,
    validate_format,
)
from data_ingestion.radar.radar_metadata import read_radar_metadata

log = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".nc", ".h5", ".hdf5", ".hdf", ".nc4"}


class RadarDownloader:
    """
    Ingests IMD DWR files from a local directory.

    Parameters
    ----------
    local_dir    : Directory containing DWR files.
    inventory_dir: Directory for inventory DB and JSONL.
    station_id   : ICAO/IMD station code, e.g. 'VABB' for Mumbai.
    """

    def __init__(
        self,
        local_dir: Path,
        inventory_dir: Path,
        station_id: str = "UNKNOWN",
    ) -> None:
        self._local_dir  = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._inv        = InventoryLogger(inventory_dir)
        self._station_id = station_id

    def ingest_local_directory(self) -> list[DatasetInventoryRecord]:
        """Scan local_dir for supported DWR files and register them."""
        files = [
            p for p in sorted(self._local_dir.iterdir())
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
        log.info(
            "Scanning DWR directory",
            extra={"dir": str(self._local_dir), "files": len(files), "station": self._station_id},
        )
        return [r for f in files if (r := self._process_file(f)) is not None]

    def _process_file(self, path: Path) -> Optional[DatasetInventoryRecord]:
        record_id = str(uuid.uuid4())
        res = validate_file_exists(path)
        if not res:
            log.warning("DWR file missing", extra={"path": str(path)})
            return None

        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5", ".hdf"}:
            fmt = "hdf5"
        elif suffix in {".nc", ".nc4"}:
            fmt = "netcdf4"
        else:
            fmt = "unknown"

        fmt_res = validate_format(path, fmt) if fmt != "unknown" else res

        try:
            meta = read_radar_metadata(path)
        except Exception as exc:
            meta = {}
            log.warning("Radar metadata extraction failed", extra={"path": str(path), "exc": str(exc)})

        checksum = compute_checksum(path)
        rec = DatasetInventoryRecord(
            record_id=record_id,
            source=f"IMD-DWR-{self._station_id}",
            dataset_type="radar",
            file_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            checksum_md5=checksum,
            status=IngestionStatus.VALIDATED if fmt_res.ok else IngestionStatus.FAILED,
            valid_time_start=meta.get("valid_time"),
            variables=meta.get("variables", []),
            error_message=None if fmt_res.ok else fmt_res.reason,
            metadata={**meta, "station_id": self._station_id},
        )
        self._inv.record(rec)
        return rec
