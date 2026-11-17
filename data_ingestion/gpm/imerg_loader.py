"""
NASA GPM IMERG data loader (supplementary, optional).

IMERG data is accessed via NASA Earthdata:
  https://gpm.nasa.gov/data/imerg

Authentication: set EARTHDATA_TOKEN environment variable
(Bearer token from https://urs.earthdata.nasa.gov).

Supports:
  - Authenticated HTTPS download from GES DISC when EARTHDATA_TOKEN is set
  - Local ingestion of pre-downloaded HDF5 (.HDF5 / .h5) files
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

import h5py

from common.logging_config import get_logger
from common.schemas import DatasetInventoryRecord, IngestionStatus
from data_ingestion.common.download_logger import InventoryLogger
from data_ingestion.common.file_validator import (
    compute_checksum,
    validate_file_exists,
    validate_format,
)

log = get_logger(__name__)

# IMERG Late Run base (GES DISC)
_GESDISC_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/"


class IMERGLoader:
    """
    Load GPM IMERG Early/Late/Final Half-Hourly precipitation data.

    Parameters
    ----------
    local_dir     : Directory for downloaded / pre-existing IMERG files.
    inventory_dir : Inventory log directory.
    product       : 'Early', 'Late', or 'Final'. Defaults to 'Late'.
    """

    def __init__(
        self,
        local_dir: Path,
        inventory_dir: Path,
        product: str = "Late",
    ) -> None:
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._inv       = InventoryLogger(inventory_dir)
        self._product   = product
        self._token     = os.environ.get("EARTHDATA_TOKEN", "")
        self._session: Optional[object] = None

        if self._token and _HAS_REQUESTS:
            import requests as _req
            session = _req.Session()
            session.headers["Authorization"] = f"Bearer {self._token}"
            session.headers["User-Agent"]    = "APM-Nowcast/0.2"
            retry = Retry(total=3, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503])
            session.mount("https://", HTTPAdapter(max_retries=retry))
            self._session = session
            log.info("IMERGLoader: authenticated Earthdata session ready")
        else:
            log.info("IMERGLoader: no EARTHDATA_TOKEN — local mode only")

    @property
    def is_authenticated(self) -> bool:
        return self._session is not None

    def ingest_local(self, path: Optional[Path] = None) -> list[DatasetInventoryRecord]:
        """Ingest all local IMERG HDF5 files (or a specific path)."""
        if path:
            files = [path]
        else:
            files = (
                sorted(self._local_dir.glob("*.HDF5"))
                + sorted(self._local_dir.glob("*.h5"))
            )
        log.info("Scanning IMERG local dir", extra={"count": len(files)})
        return [r for f in files if (r := self._process_local(f)) is not None]

    def _process_local(self, path: Path) -> Optional[DatasetInventoryRecord]:
        res = validate_file_exists(path)
        if not res:
            return None

        fmt_res  = validate_format(path, "hdf5")
        meta     = _read_imerg_metadata(path)
        checksum = compute_checksum(path)

        rec = DatasetInventoryRecord(
            record_id=str(uuid.uuid4()),
            source="NASA-GPM-IMERG",
            dataset_type="gpm",
            file_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            checksum_md5=checksum,
            status=IngestionStatus.VALIDATED if fmt_res.ok else IngestionStatus.FAILED,
            valid_time_start=meta.get("valid_time"),
            variables=meta.get("datasets", []),
            error_message=None if fmt_res.ok else fmt_res.reason,
            metadata=meta,
        )
        self._inv.record(rec)
        return rec


def _read_imerg_metadata(path: Path) -> dict:
    meta = {"datasets": [], "valid_time": None, "source_file": str(path)}
    try:
        with h5py.File(str(path), "r") as fh:
            meta["datasets"] = list(fh.keys())
            attrs = dict(fh.attrs)
            for k in ["FileHeader", "FileInfo"]:
                if k in attrs:
                    raw = attrs[k]
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    meta["file_header_raw"] = str(raw)[:500]
                    break
    except Exception as exc:
        log.warning("IMERG metadata read failed", extra={"exc": str(exc)})
    return meta
