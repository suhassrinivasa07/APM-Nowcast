"""
Pytest fixtures and synthetic dataset generators for Module 3 Preprocessing tests.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np
import pytest

from preprocessing.common.grid_aligner import CommonGrid


@pytest.fixture(scope="session")
def test_grid() -> CommonGrid:
    """Coarser resolution grid for fast unit test execution (1.0 degree)."""
    return CommonGrid(lat_min=10.0, lat_max=30.0, lon_min=70.0, lon_max=90.0, resolution_deg=1.0)


@pytest.fixture(scope="session")
def prep_tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("prep_tests")


@pytest.fixture(scope="session")
def synthetic_sat_file(prep_tmp_dir: Path) -> Path:
    """
    Creates an INSAT-3D HDF5 file containing:
    - Normal convective cloud-top brightness temps (200 - 290 K)
    - Space pixels (0 / NaN)
    - Unphysical values (> 350 K) to verify physical QC masking
    """
    path = prep_tmp_dir / "INSAT3D_L1B_TEST.h5"
    with h5py.File(str(path), "w") as fh:
        fh.attrs["Satellite_Name"] = "INSAT-3D"
        fh.attrs["Image_Date"] = "01-Jun-2024"
        fh.attrs["Image_Time_in_UTC"] = "06:00:00"
        fh.attrs["Nominal_Central_Point_Coordinates_(lat,lon)"] = "(20.0, 80.0)"

        shape = (10, 10)
        # Normal physical values: 220 to 280 K
        tir1 = np.linspace(220, 280, 100, dtype=np.float32).reshape(shape)
        # Inject space pixels (0.0) and unphysically high temperature (400.0) in the central cells
        tir1[4:6, 4:6] = 400.0
        tir1[4, 4] = 0.0
        tir1[5, 5] = -999.0

        ds_tir1 = fh.create_dataset("IMG_TIR1", data=tir1)
        ds_tir1.attrs["units"] = "K"

        # Water vapor channel
        wv = np.linspace(230, 260, 100, dtype=np.float32).reshape(shape)
        fh.create_dataset("IMG_WV", data=wv)

    return path


@pytest.fixture(scope="session")
def synthetic_dwr_file(prep_tmp_dir: Path) -> Path:
    """
    Creates a DWR NetCDF file containing:
    - Realistic meteorological echoes (15 - 55 dBZ)
    - High dBZ + near-zero velocity ground clutter
    - Fill values (-9999.0)
    """
    path = prep_tmp_dir / "DWR_QC_TEST.nc"
    with nc4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.station_id = "VABB"
        ds.latitude = 19.0
        ds.longitude = 73.0

        ds.createDimension("time", 1)
        ds.createDimension("latitude", 8)
        ds.createDimension("longitude", 8)

        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2024-06-01 06:00:00"
        t[:] = [0]

        lats = ds.createVariable("latitude", "f4", ("latitude",))
        lons = ds.createVariable("longitude", "f4", ("longitude",))
        lats[:] = np.linspace(17.0, 21.0, 8, dtype=np.float32)
        lons[:] = np.linspace(71.0, 75.0, 8, dtype=np.float32)

        # Reflectivity
        dbz = ds.createVariable("DBZ", "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
        dbz_data = np.full((1, 8, 8), 30.0, dtype=np.float32)
        # Add fill value
        dbz_data[0, 0, 0] = -9999.0
        # Add ground clutter candidate: 50 dBZ
        dbz_data[0, 2, 2] = 50.0
        dbz[:] = dbz_data

        # Velocity
        vel = ds.createVariable("VEL", "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
        vel_data = np.full((1, 8, 8), 12.0, dtype=np.float32)
        # Ground clutter velocity: 0.2 m/s (near zero)
        vel_data[0, 2, 2] = 0.2
        vel[:] = vel_data

    return path


@pytest.fixture(scope="session")
def synthetic_lightning_csv_file(prep_tmp_dir: Path) -> Path:
    """
    Creates an NRSC Lightning CSV with events:
    - 4 events within window [05:30:00, 06:00:00]
    - 2 events outside window (e.g. 05:15:00 and 06:10:00)
    - Specific lat/lon to test discrete bin placement
    """
    path = prep_tmp_dir / "NRSC_LIGHTNING_TEST.csv"
    rows = [
        # Inside window: 05:45:00 UTC at (20.2, 80.2)
        {"datetime_utc": "2024-06-01T05:45:00Z", "latitude": 20.2, "longitude": 80.2, "peak_current_kA": -45.0, "type": "CG"},
        # Inside window: 05:50:00 UTC in same cell (20.3, 80.4) -> cell should have count 2
        {"datetime_utc": "2024-06-01T05:50:00Z", "latitude": 20.3, "longitude": 80.4, "peak_current_kA": -60.0, "type": "CG"},
        # Inside window: 05:55:00 UTC in distinct cell (25.5, 85.5)
        {"datetime_utc": "2024-06-01T05:55:00Z", "latitude": 25.5, "longitude": 85.5, "peak_current_kA": 22.0, "type": "IC"},
        # Inside window: 05:59:00 UTC in distinct cell (15.0, 75.0)
        {"datetime_utc": "2024-06-01T05:59:00Z", "latitude": 15.0, "longitude": 75.0, "peak_current_kA": 30.0, "type": "CG"},
        # Outside window (too early: 05:15:00)
        {"datetime_utc": "2024-06-01T05:15:00Z", "latitude": 20.2, "longitude": 80.2, "peak_current_kA": -15.0, "type": "CG"},
        # Outside window (too late: 06:10:00)
        {"datetime_utc": "2024-06-01T06:10:00Z", "latitude": 20.2, "longitude": 80.2, "peak_current_kA": -18.0, "type": "CG"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


@pytest.fixture(scope="session")
def synthetic_era5_file(prep_tmp_dir: Path) -> Path:
    """
    Creates an ERA5 NetCDF with single-level and dynamic variables:
    t2m, d2m, cape, cin, u10, v10.
    """
    path = prep_tmp_dir / "ERA5_PREP_TEST.nc"
    with nc4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("valid_time", 2)
        ds.createDimension("latitude", 6)
        ds.createDimension("longitude", 6)

        vt = ds.createVariable("valid_time", "f8", ("valid_time",))
        vt.units = "hours since 2024-06-01 00:00:00"
        vt[:] = [0, 6]  # 00:00 and 06:00 UTC

        lat = ds.createVariable("latitude", "f4", ("latitude",))
        lon = ds.createVariable("longitude", "f4", ("longitude",))
        lat[:] = np.linspace(10.0, 30.0, 6, dtype=np.float32)
        lon[:] = np.linspace(70.0, 90.0, 6, dtype=np.float32)

        t2m = ds.createVariable("t2m", "f4", ("valid_time", "latitude", "longitude"))
        t2m.units = "K"
        t2m[:] = np.full((2, 6, 6), 305.0, dtype=np.float32)

        d2m = ds.createVariable("d2m", "f4", ("valid_time", "latitude", "longitude"))
        d2m.units = "K"
        d2m[:] = np.full((2, 6, 6), 295.0, dtype=np.float32)

        cape = ds.createVariable("cape", "f4", ("valid_time", "latitude", "longitude"))
        cape.units = "J kg**-1"
        cape[:] = np.full((2, 6, 6), 1800.0, dtype=np.float32)

        u10 = ds.createVariable("u10", "f4", ("valid_time", "latitude", "longitude"))
        u10.units = "m s**-1"
        u10[:] = np.full((2, 6, 6), 4.0, dtype=np.float32)

        v10 = ds.createVariable("v10", "f4", ("valid_time", "latitude", "longitude"))
        v10.units = "m s**-1"
        v10[:] = np.full((2, 6, 6), 3.0, dtype=np.float32)

    return path
