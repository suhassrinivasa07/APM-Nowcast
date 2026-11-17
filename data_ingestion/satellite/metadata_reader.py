"""
Extract metadata from INSAT-3D/3DR/3DS HDF5 Level-1B files.

The standard INSAT L1B HDF5 layout stores:
  - Global attributes: Satellite_Name, Nominal_Central_Point_Coordinates_(lat,lon),
    Image_Date, Image_Time_in_UTC
  - Dataset groups: IMG_TIR1, IMG_TIR2, IMG_MIR, IMG_SWIR, IMG_WV, IMG_VIS, IMG_SSIR
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

from common.logging_config import get_logger

log = get_logger(__name__)

_DATE_KEYS = ["Image_Date", "Acquisition_Date", "Start_Date"]
_TIME_KEYS = ["Image_Time_in_UTC", "Start_Time", "Acquisition_Time_in_UTC"]
_SAT_KEYS  = ["Satellite_Name", "Sensor_Name", "Platform_Name"]
_LAT_KEYS  = ["Nominal_Central_Point_Coordinates_(lat,lon)"]


def read_insat_metadata(path: Path) -> dict[str, Any]:
    """
    Open an INSAT L1B HDF5 file and extract key metadata.

    Returns dict with keys:
      satellite, valid_time (datetime UTC), datasets (list[str]),
      lat_center, lon_center, attributes (dict)
    """
    meta: dict[str, Any] = {
        "satellite":   "UNKNOWN",
        "valid_time":  None,
        "datasets":    [],
        "lat_center":  None,
        "lon_center":  None,
        "attributes":  {},
        "source_file": str(path),
    }

    try:
        with h5py.File(str(path), "r") as fh:
            attrs = dict(fh.attrs)
            meta["attributes"] = {
                k: (v.decode() if isinstance(v, bytes) else str(v))
                for k, v in attrs.items()
            }

            for key in _SAT_KEYS:
                if key in attrs:
                    meta["satellite"] = _decode(attrs[key])
                    break

            date_str = _extract_first(attrs, _DATE_KEYS)
            time_str = _extract_first(attrs, _TIME_KEYS)
            meta["valid_time"] = _parse_insat_datetime(date_str, time_str)

            meta["datasets"] = list(fh.keys())

            for key in _LAT_KEYS:
                if key in attrs:
                    val = _decode(attrs[key])
                    try:
                        parts = val.strip("()").split(",")
                        meta["lat_center"] = float(parts[0])
                        meta["lon_center"] = float(parts[1])
                    except Exception:
                        pass
                    break

    except Exception as exc:
        log.warning("Could not read INSAT HDF5 metadata", extra={"path": str(path), "exc": str(exc)})

    return meta


def _decode(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _extract_first(attrs: dict, keys: list[str]) -> str:
    for key in keys:
        if key in attrs:
            return _decode(attrs[key])
    return ""


def _parse_insat_datetime(date_str: str, time_str: str) -> datetime | None:
    """Parse INSAT date/time attribute strings into a UTC-aware datetime."""
    combined = f"{date_str} {time_str}".strip()
    for fmt in [
        "%d-%b-%Y %H:%M:%S",
        "%Y%m%d %H%M%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            dt = datetime.strptime(combined, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
