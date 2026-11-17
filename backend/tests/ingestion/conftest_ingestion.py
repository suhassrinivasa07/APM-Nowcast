"""
Synthetic sample generators for offline pytest verification.

Creates minimal valid HDF5, NetCDF, and CSV files that satisfy
the ingestion pipeline validators — without any external credentials.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# HDF5 synthetic INSAT-3D L1B sample
# ---------------------------------------------------------------------------

def make_synthetic_insat_hdf5(path: Path) -> Path:
    """Write a minimal valid INSAT-3D-like HDF5 file."""
    with h5py.File(str(path), "w") as fh:
        fh.attrs["Satellite_Name"] = "INSAT-3D"
        fh.attrs["Image_Date"]     = "01-Jun-2024"
        fh.attrs["Image_Time_in_UTC"] = "06:00:00"
        fh.attrs["Nominal_Central_Point_Coordinates_(lat,lon)"] = "(22.5, 82.0)"
        for name in ["IMG_TIR1", "IMG_MIR", "IMG_WV", "IMG_TIR2", "IMG_VIS", "IMG_SSIR"]:
            ds = fh.create_dataset(name, data=np.random.uniform(200, 320, (4, 4)).astype(np.float32))
            ds.attrs["units"] = "K"
    return path


# ---------------------------------------------------------------------------
# NetCDF synthetic ERA5-like sample
# ---------------------------------------------------------------------------

def make_synthetic_era5_netcdf(path: Path) -> Path:
    """Write a minimal valid ERA5-like NetCDF4 file over India."""
    with nc4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.6"
        ds.createDimension("time",      1)
        ds.createDimension("latitude",  4)
        ds.createDimension("longitude", 4)

        t_var          = ds.createVariable("time", "f8", ("time",))
        t_var.units    = "hours since 2024-06-01 00:00:00"
        t_var.calendar = "standard"
        t_var[:]       = [0]

        lat       = ds.createVariable("latitude",  "f4", ("latitude",))
        lon       = ds.createVariable("longitude", "f4", ("longitude",))
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        lat[:]    = np.linspace(10.0, 30.0, 4, dtype=np.float32)
        lon[:]    = np.linspace(70.0, 90.0, 4, dtype=np.float32)

        t2m          = ds.createVariable("t2m",  "f4", ("time", "latitude", "longitude"))
        t2m.units    = "K"
        t2m.long_name = "2 metre temperature"
        t2m[:]       = np.random.uniform(290, 310, (1, 4, 4)).astype(np.float32)

        cape       = ds.createVariable("cape", "f4", ("time", "latitude", "longitude"))
        cape.units = "J kg**-1"
        cape[:]    = np.random.uniform(0, 2000, (1, 4, 4)).astype(np.float32)
    return path


# ---------------------------------------------------------------------------
# NetCDF synthetic DWR sample
# ---------------------------------------------------------------------------

def make_synthetic_dwr_netcdf(path: Path) -> Path:
    """Write a minimal CF-compliant DWR reflectivity NetCDF4 file."""
    with nc4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time",      1)
        ds.createDimension("latitude",  4)
        ds.createDimension("longitude", 4)

        t          = ds.createVariable("time", "f8", ("time",))
        t.units    = "hours since 2024-06-01 06:00:00"
        t.calendar = "standard"
        t[:]       = [0]

        lat       = ds.createVariable("latitude",  "f4", ("latitude",))
        lon       = ds.createVariable("longitude", "f4", ("longitude",))
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        lat[:]    = np.linspace(18.0, 22.0, 4, dtype=np.float32)
        lon[:]    = np.linspace(72.0, 76.0, 4, dtype=np.float32)

        dbz           = ds.createVariable("DBZ", "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
        dbz.units     = "dBZ"
        dbz.long_name = "Reflectivity"
        dbz[:]        = np.random.uniform(-10, 65, (1, 4, 4)).astype(np.float32)
    return path


# ---------------------------------------------------------------------------
# HDF5 synthetic IMERG sample
# ---------------------------------------------------------------------------

def make_synthetic_imerg_hdf5(path: Path) -> Path:
    """Write a minimal IMERG-like HDF5 file."""
    with h5py.File(str(path), "w") as fh:
        fh.attrs["FileHeader"] = b"StartGranuleDateTime=2024-06-01T06:00:00.000Z;"
        grid = fh.create_group("Grid")
        grid.create_dataset("lat",  data=np.linspace(5.0,  40.0, 10, dtype=np.float32))
        grid.create_dataset("lon",  data=np.linspace(65.0, 100.0, 10, dtype=np.float32))
        grid.create_dataset("time", data=np.array([0.0], dtype=np.float64))
        precip = grid.create_dataset(
            "precipitationCal",
            data=np.random.uniform(0, 20, (1, 10, 10)).astype(np.float32),
        )
        precip.attrs["units"]      = "mm/hr"
        precip.attrs["_FillValue"] = -9999.9
    return path


# ---------------------------------------------------------------------------
# CSV synthetic lightning sample
# ---------------------------------------------------------------------------

def make_synthetic_lightning_csv(path: Path, n: int = 20) -> Path:
    """Write a minimal NRSC-style lightning CSV."""
    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    rng  = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append({
            "datetime_utc":    base.replace(second=i * 3).isoformat(),
            "latitude":        float(rng.uniform(10.0, 30.0)),
            "longitude":       float(rng.uniform(70.0, 90.0)),
            "peak_current_kA": float(rng.uniform(-100, 100)),
            "type":            rng.choice(["CG", "CC", "IC"]),
        })
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory root for all ingestion test data."""
    return tmp_path_factory.mktemp("apm_test_data")


@pytest.fixture(scope="session")
def synthetic_insat_hdf5(tmp_data_root: Path) -> Path:
    p = tmp_data_root / "satellite" / "INSAT3D_IMG_L1B_20240601_0600.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    return make_synthetic_insat_hdf5(p)


@pytest.fixture(scope="session")
def synthetic_era5_netcdf(tmp_data_root: Path) -> Path:
    p = tmp_data_root / "nwp" / "era5_sl_202406.nc"
    p.parent.mkdir(parents=True, exist_ok=True)
    return make_synthetic_era5_netcdf(p)


@pytest.fixture(scope="session")
def synthetic_dwr_netcdf(tmp_data_root: Path) -> Path:
    p = tmp_data_root / "radar" / "DWR_VABB_20240601_0600.nc"
    p.parent.mkdir(parents=True, exist_ok=True)
    return make_synthetic_dwr_netcdf(p)


@pytest.fixture(scope="session")
def synthetic_imerg_hdf5(tmp_data_root: Path) -> Path:
    p = tmp_data_root / "gpm" / "3B-HHR.MS.MRG.3IMERG.20240601.HDF5"
    p.parent.mkdir(parents=True, exist_ok=True)
    return make_synthetic_imerg_hdf5(p)


@pytest.fixture(scope="session")
def synthetic_lightning_csv(tmp_data_root: Path) -> Path:
    p = tmp_data_root / "lightning" / "NRSC_lightning_20240601.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return make_synthetic_lightning_csv(p)
