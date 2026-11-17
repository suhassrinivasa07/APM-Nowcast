"""
File validation utilities for all ingested datasets.

Validates:
  - File existence and non-zero size
  - File format magic bytes (HDF5, NetCDF, GRIB2, CSV, GeoTIFF)
  - Variable presence inside NetCDF / HDF5 files
  - Coordinate bounds against expected bounding box
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Optional, Sequence

import h5py
import netCDF4 as nc4
import numpy as np

from common.logging_config import get_logger
from common.schemas import GeoBoundingBox

log = get_logger(__name__)

# Magic-byte signatures (first 4 bytes)
_MAGIC: dict[str, bytes] = {
    "hdf5":    b"\x89HDF",
    "netcdf3": b"CDF\x01",
    "netcdf4": b"\x89HDF",   # NC4 is an HDF5 container
    "grib2":   b"GRIB",
    "geotiff": b"II*\x00",   # little-endian TIFF
    "bigtiff": b"MM\x00*",   # big-endian TIFF
}


class ValidationResult:
    __slots__ = ("ok", "reason", "file_size_bytes", "checksum_md5")

    def __init__(
        self,
        ok: bool,
        reason: str = "",
        file_size_bytes: int = 0,
        checksum_md5: str = "",
    ) -> None:
        self.ok = ok
        self.reason = reason
        self.file_size_bytes = file_size_bytes
        self.checksum_md5 = checksum_md5

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"ValidationResult(ok={self.ok}, reason={self.reason!r})"


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def _read_magic(path: Path, n: int = 4) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)


def validate_file_exists(path: Path) -> ValidationResult:
    """Check file exists and is non-empty."""
    if not path.exists():
        return ValidationResult(False, f"File not found: {path}")
    size = path.stat().st_size
    if size == 0:
        return ValidationResult(False, f"File is empty: {path}")
    return ValidationResult(True, file_size_bytes=size)


def validate_format(path: Path, expected_format: str) -> ValidationResult:
    """
    Validate file magic bytes against expected_format.
    expected_format: one of 'hdf5', 'netcdf3', 'netcdf4', 'grib2',
                     'geotiff', 'csv' (no magic — checked by extension).
    """
    res = validate_file_exists(path)
    if not res:
        return res

    if expected_format == "csv":
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.reader(fh)
                next(reader)
        except StopIteration:
            return ValidationResult(False, "CSV file has no rows", res.file_size_bytes)
        except Exception as exc:
            return ValidationResult(False, f"CSV parse error: {exc}", res.file_size_bytes)
        return ValidationResult(True, file_size_bytes=res.file_size_bytes)

    magic = _read_magic(path)
    expected = _MAGIC.get(expected_format)
    if expected is None:
        return ValidationResult(False, f"Unknown format: {expected_format}")
    if not magic.startswith(expected):
        return ValidationResult(
            False,
            f"Magic mismatch for {expected_format}: got {magic!r}, want {expected!r}",
            res.file_size_bytes,
        )
    return ValidationResult(True, file_size_bytes=res.file_size_bytes)


def validate_netcdf_variables(
    path: Path,
    required_variables: Sequence[str],
) -> ValidationResult:
    """Confirm required variable names exist inside a NetCDF3/4 file."""
    res = validate_file_exists(path)
    if not res:
        return res
    try:
        with nc4.Dataset(str(path), "r") as ds:
            missing = [v for v in required_variables if v not in ds.variables]
        if missing:
            return ValidationResult(False, f"Missing variables: {missing}", res.file_size_bytes)
    except Exception as exc:
        return ValidationResult(False, f"NetCDF open error: {exc}", res.file_size_bytes)
    return ValidationResult(True, file_size_bytes=res.file_size_bytes)


def validate_hdf5_datasets(
    path: Path,
    required_datasets: Sequence[str],
) -> ValidationResult:
    """Confirm required dataset paths exist inside an HDF5 file."""
    res = validate_file_exists(path)
    if not res:
        return res
    try:
        with h5py.File(str(path), "r") as fh:
            missing = [d for d in required_datasets if d not in fh]
        if missing:
            return ValidationResult(False, f"Missing HDF5 datasets: {missing}", res.file_size_bytes)
    except Exception as exc:
        return ValidationResult(False, f"HDF5 open error: {exc}", res.file_size_bytes)
    return ValidationResult(True, file_size_bytes=res.file_size_bytes)


def validate_netcdf_bounds(
    path: Path,
    bbox: GeoBoundingBox,
    lat_var: str = "latitude",
    lon_var: str = "longitude",
) -> ValidationResult:
    """Verify lat/lon ranges in a NetCDF file intersect the expected bounding box."""
    res = validate_file_exists(path)
    if not res:
        return res
    try:
        with nc4.Dataset(str(path), "r") as ds:
            if lat_var not in ds.variables or lon_var not in ds.variables:
                return ValidationResult(False, f"Coordinate variables {lat_var}/{lon_var} not found")
            lats = ds.variables[lat_var][:]
            lons = ds.variables[lon_var][:]
            lat_ok = float(np.min(lats)) <= bbox.lat_max and float(np.max(lats)) >= bbox.lat_min
            lon_ok = float(np.min(lons)) <= bbox.lon_max and float(np.max(lons)) >= bbox.lon_min
            if not (lat_ok and lon_ok):
                return ValidationResult(
                    False,
                    f"Grid [{float(np.min(lats)):.2f}-{float(np.max(lats)):.2f}N, "
                    f"{float(np.min(lons)):.2f}-{float(np.max(lons)):.2f}E] "
                    f"does not intersect bbox",
                    res.file_size_bytes,
                )
    except Exception as exc:
        return ValidationResult(False, f"Bounds check error: {exc}", res.file_size_bytes)
    return ValidationResult(True, file_size_bytes=res.file_size_bytes)


def compute_checksum(path: Path) -> str:
    """Return MD5 hex digest of a file."""
    return _md5(path)
