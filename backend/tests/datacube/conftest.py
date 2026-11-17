"""
Pytest fixtures and synthetic multimodal time sequences for Datacube tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import xarray as xr

from datacube.cube_builder import DataCubeBuilder
from datacube.schema import MODALITY_ORDER, ModalityType
from preprocessing.common.grid_aligner import CommonGrid


@pytest.fixture(scope="session")
def datacube_tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("datacube_tests")


@pytest.fixture(scope="session")
def cube_grid() -> CommonGrid:
    """Compact 64x64 grid for low VRAM & fast unit testing."""
    # 64 points at 0.25 deg = 15.75 deg span
    return CommonGrid(lat_min=15.0, lat_max=30.75, lon_min=70.0, lon_max=85.75, resolution_deg=0.25)


@pytest.fixture(scope="session")
def synthetic_time_steps() -> List[datetime]:
    """24 consecutive 15-minute time steps covering a 6-hour convective sequence."""
    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [base + timedelta(minutes=15 * i) for i in range(24)]


@pytest.fixture(scope="session")
def synthetic_time_step_datasets(
    cube_grid: CommonGrid,
    synthetic_time_steps: List[datetime],
) -> List[Dict[str, xr.Dataset]]:
    """
    Constructs a list of per-timestep modality dictionaries:
    Simulates:
    - Step 0-13: Satellite, Lightning, NWP present.
    - Step 3 & Step 7: Radar DROPOUT (simulating radar missingness).
    - Step 4-7: Convective burst with lightning flashes and high dBZ.
    - Partial NaNs inside satellite and radar (valid_mask == False).
    """
    datasets = []
    n_lat, n_lon = cube_grid.shape
    lats = cube_grid.lats
    lons = cube_grid.lons

    for idx, t in enumerate(synthetic_time_steps):
        t_dict: Dict[str, xr.Dataset] = {}

        # 1. Satellite: TIR1 (210-280K), WV (230-260K), MIR (240-300K)
        sat_tir1 = np.full((n_lat, n_lon), 260.0 - (idx * 2.0), dtype=np.float32)
        sat_tir1[0:5, 0:5] = np.nan  # Partial space/missing pixels
        sat_vmask = ~np.isnan(sat_tir1)

        sat_ds = xr.Dataset(
            data_vars={
                "tir1": (["latitude", "longitude"], sat_tir1),
                "tir1_valid_mask": (["latitude", "longitude"], sat_vmask),
                "wv": (["latitude", "longitude"], np.full((n_lat, n_lon), 245.0, dtype=np.float32)),
                "mir": (["latitude", "longitude"], np.full((n_lat, n_lon), 270.0, dtype=np.float32)),
            },
            coords={"latitude": lats, "longitude": lons},
        )
        t_dict["satellite"] = sat_ds

        # 2. Radar: Dropout at index 3 and 7
        if idx not in (3, 7):
            radar_dbz = np.full((n_lat, n_lon), 15.0, dtype=np.float32)
            # Convective core in center for convective steps
            if 4 <= idx <= 7:
                radar_dbz[25:35, 25:35] = 52.0
            radar_dbz[0:3, 0:3] = np.nan
            rad_vmask = ~np.isnan(radar_dbz)

            radar_ds = xr.Dataset(
                data_vars={
                    "composite_reflectivity": (["latitude", "longitude"], radar_dbz),
                    "composite_reflectivity_valid_mask": (["latitude", "longitude"], rad_vmask),
                    "reflectivity_cappi": (["latitude", "longitude"], radar_dbz * 0.9),
                },
                coords={"latitude": lats, "longitude": lons},
            )
            t_dict["radar"] = radar_ds
        else:
            # Dropped out sensor
            t_dict["radar"] = None

        # 3. Lightning: Flashes during convective burst
        flash_cnt = np.zeros((n_lat, n_lon), dtype=np.float32)
        if 4 <= idx <= 7:
            flash_cnt[28:32, 28:32] = 8.0  # Convective lightning core
        occ = (flash_cnt > 0).astype(np.float32)

        lit_ds = xr.Dataset(
            data_vars={
                "flash_count": (["latitude", "longitude"], flash_cnt),
                "occurrence_mask": (["latitude", "longitude"], occ),
            },
            coords={"latitude": lats, "longitude": lons},
        )
        t_dict["lightning"] = lit_ds

        # 4. NWP: ERA5 CAPE, CIN, RH, Wind
        nwp_ds = xr.Dataset(
            data_vars={
                "cape": (["latitude", "longitude"], np.full((n_lat, n_lon), 2200.0, dtype=np.float32)),
                "cin": (["latitude", "longitude"], np.full((n_lat, n_lon), 45.0, dtype=np.float32)),
                "relative_humidity_2m": (["latitude", "longitude"], np.full((n_lat, n_lon), 82.0, dtype=np.float32)),
                "wind_speed_10m": (["latitude", "longitude"], np.full((n_lat, n_lon), 7.5, dtype=np.float32)),
            },
            coords={"latitude": lats, "longitude": lons},
        )
        t_dict["nwp"] = nwp_ds

        datasets.append(t_dict)

    return datasets


@pytest.fixture(scope="session")
def synthetic_unified_cube(
    cube_grid: CommonGrid,
    synthetic_time_steps: List[datetime],
    synthetic_time_step_datasets: List[Dict[str, xr.Dataset]],
) -> xr.Dataset:
    """Pre-built unified xarray data cube."""
    builder = DataCubeBuilder(grid=cube_grid)
    return builder.build_cube_from_time_steps(synthetic_time_steps, synthetic_time_step_datasets)
