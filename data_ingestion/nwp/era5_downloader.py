"""
ERA5 downloader using the Copernicus CDS API (cdsapi).

Authentication: set CDSAPI_URL and CDSAPI_KEY environment variables,
or place a ~/.cdsapirc file (standard CDS setup).

If credentials are absent, the downloader operates in OFFLINE mode and
raises ConfigurationError for any download attempt.

Official CDS API docs: https://cds.climate.copernicus.eu/api-how-to
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Union

from common.logging_config import get_logger
from common.schemas import DatasetInventoryRecord, GeoBoundingBox, IngestionStatus
from data_ingestion.common.download_logger import InventoryLogger
from data_ingestion.common.file_validator import compute_checksum, validate_file_exists
from data_ingestion.nwp.era5_reader import read_era5_metadata

log = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when required credentials / libraries are not available."""


def _build_cds_client():
    """Return a cdsapi.Client instance; raise ConfigurationError if unavailable."""
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise ConfigurationError("cdsapi not installed. pip install cdsapi") from exc

    url = os.environ.get("CDSAPI_URL", "")
    key = os.environ.get("CDSAPI_KEY", "")

    if url and key:
        return cdsapi.Client(url=url, key=key, quiet=True, verify=True)
    try:
        return cdsapi.Client(quiet=True, verify=True)
    except Exception as exc:
        raise ConfigurationError(
            "CDS credentials not found. Set CDSAPI_URL and CDSAPI_KEY "
            "or create ~/.cdsapirc. See https://cds.climate.copernicus.eu/api-how-to"
        ) from exc


class ERA5Downloader:
    """
    Download ERA5 single-level and pressure-level reanalysis data from Copernicus CDS.

    Parameters
    ----------
    local_dir     : Directory where downloaded files will be stored.
    inventory_dir : Inventory log directory.
    bbox          : Spatial domain. Defaults to Indian sub-continental domain.
    """

    _SINGLE_LEVEL_DATASET   = "reanalysis-era5-single-levels"
    _PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"

    def __init__(
        self,
        local_dir: Path,
        inventory_dir: Path,
        bbox: Optional[GeoBoundingBox] = None,
    ) -> None:
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._inv   = InventoryLogger(inventory_dir)
        self._bbox  = bbox or GeoBoundingBox.india_domain()
        self._client           = None
        self._cds_available    = False

        try:
            self._client        = _build_cds_client()
            self._cds_available = True
            log.info("ERA5Downloader: CDS client ready")
        except ConfigurationError as exc:
            log.info(f"ERA5Downloader: offline mode — {exc}")

    @property
    def is_online(self) -> bool:
        return self._cds_available

    def download_single_levels(
        self,
        variables: list[str],
        year: Union[int, str],
        month: Union[int, str],
        days: Optional[list[str]] = None,
        hours: Optional[list[str]] = None,
        output_name: Optional[str] = None,
    ) -> Optional[DatasetInventoryRecord]:
        """
        Download ERA5 single-level fields to NetCDF.

        Raises ConfigurationError when offline.
        """
        if not self._cds_available:
            raise ConfigurationError(
                "CDS client not available. Configure CDSAPI_URL and CDSAPI_KEY."
            )
        days  = days  or [f"{d:02d}" for d in range(1, 32)]
        hours = hours or ["00:00", "06:00", "12:00", "18:00"]
        output_name = output_name or f"era5_sl_{year}{int(month):02d}.nc"
        dest = self._local_dir / output_name

        request = {
            "product_type": "reanalysis",
            "variable":     variables,
            "year":         str(year),
            "month":        f"{int(month):02d}",
            "day":          days,
            "time":         hours,
            "area":         self._bbox.as_cds_area(),
            "format":       "netcdf",
        }
        return self._submit_request(self._SINGLE_LEVEL_DATASET, request, dest, variables)

    def download_pressure_levels(
        self,
        variables: list[str],
        pressure_levels: list[str],
        year: Union[int, str],
        month: Union[int, str],
        days: Optional[list[str]] = None,
        hours: Optional[list[str]] = None,
        output_name: Optional[str] = None,
    ) -> Optional[DatasetInventoryRecord]:
        """Download ERA5 pressure-level fields to NetCDF. Raises ConfigurationError offline."""
        if not self._cds_available:
            raise ConfigurationError("CDS client not available.")

        days  = days  or [f"{d:02d}" for d in range(1, 32)]
        hours = hours or ["00:00", "06:00", "12:00", "18:00"]
        output_name = output_name or f"era5_pl_{year}{int(month):02d}.nc"
        dest = self._local_dir / output_name

        request = {
            "product_type":   "reanalysis",
            "variable":       variables,
            "pressure_level": pressure_levels,
            "year":           str(year),
            "month":          f"{int(month):02d}",
            "day":            days,
            "time":           hours,
            "area":           self._bbox.as_cds_area(),
            "format":         "netcdf",
        }
        return self._submit_request(self._PRESSURE_LEVEL_DATASET, request, dest, variables)

    def ingest_local(self, path: Path) -> Optional[DatasetInventoryRecord]:
        """Register a locally-available ERA5 NetCDF file into the inventory."""
        res = validate_file_exists(path)
        if not res:
            log.warning("ERA5 local file not found", extra={"path": str(path)})
            return None

        meta     = read_era5_metadata(path)
        checksum = compute_checksum(path)

        rec = DatasetInventoryRecord(
            record_id=str(uuid.uuid4()),
            source="ERA5-CDS",
            dataset_type="nwp",
            file_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            checksum_md5=checksum,
            status=IngestionStatus.LOCAL,
            valid_time_start=meta.get("time_start"),
            valid_time_end=meta.get("time_end"),
            variables=meta.get("variables", []),
            metadata=meta,
        )
        self._inv.record(rec)
        return rec

    def _submit_request(
        self,
        dataset: str,
        request: dict,
        dest: Path,
        variables: list[str],
    ) -> Optional[DatasetInventoryRecord]:
        record_id = str(uuid.uuid4())
        log.info("Submitting CDS request", extra={"dataset": dataset, "dest": str(dest)})
        try:
            assert self._client is not None
            self._client.retrieve(dataset, request, str(dest))
        except Exception as exc:
            log.error("CDS download failed", extra={"exc": str(exc)})
            rec = DatasetInventoryRecord(
                record_id=record_id,
                source="ERA5-CDS",
                dataset_type="nwp",
                file_path=str(dest),
                status=IngestionStatus.FAILED,
                error_message=str(exc),
                variables=variables,
            )
            self._inv.record(rec)
            return rec

        meta     = read_era5_metadata(dest)
        checksum = compute_checksum(dest)
        rec = DatasetInventoryRecord(
            record_id=record_id,
            source="ERA5-CDS",
            dataset_type="nwp",
            file_path=str(dest.resolve()),
            file_size_bytes=dest.stat().st_size,
            checksum_md5=checksum,
            status=IngestionStatus.VALIDATED,
            valid_time_start=meta.get("time_start"),
            valid_time_end=meta.get("time_end"),
            variables=variables,
            metadata=meta,
        )
        self._inv.record(rec)
        return rec
