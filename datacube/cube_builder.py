"""
Unified Atmospheric Data Cube Builder for Multimodal Spatiotemporal Earth Observation.

Merges time sequences of preprocessed datasets (satellite, radar, lightning, ERA5)
into a unified multi-dimensional xarray Dataset / Zarr cube with explicit
modality availability masks tracking sensor dropout.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from common.logging_config import get_logger
from common.schemas import UTCTimestamp
from datacube.schema import (
    CHANNEL_REGISTRY,
    MODALITY_ORDER,
    MODALITY_TO_INDEX,
    NUM_MODALITIES,
    ModalityType,
    get_channels_for_modality,
)
from preprocessing.common.grid_aligner import CommonGrid

log = get_logger(__name__)


class DataCubeBuilder:
    """
    Constructs and persists unified spatiotemporal atmospheric data cubes.
    """

    def __init__(self, grid: Optional[CommonGrid] = None) -> None:
        self.grid = grid or CommonGrid()

    def build_cube_from_time_steps(
        self,
        time_steps: List[Union[datetime, str, np.datetime64, pd.Timestamp]],
        time_step_datasets: List[Dict[str, xr.Dataset]],
    ) -> xr.Dataset:
        """
        Merge per-timestamp modality datasets into a single 4D/3D space-time cube.

        Parameters
        ----------
        time_steps : List of N timestamps.
        time_step_datasets : List of N dicts mapping modality name -> xr.Dataset.
            e.g. [{"satellite": ds_sat, "radar": ds_rad, "lightning": ds_lit, "nwp": ds_nwp}, ...]
            If a modality is missing for a timestamp, it can be omitted from the dict or set to None.

        Returns
        -------
        xr.Dataset containing:
            - Data variables for each registered channel: (time, latitude, longitude)
            - Channel validity masks: (time, latitude, longitude)
            - Binary modality availability mask: (time, modality)
        """
        n_times = len(time_steps)
        if n_times == 0:
            raise ValueError("time_steps list cannot be empty.")

        # Convert timestamps to uniform UTC datetime64
        utc_times = [
            np.datetime64(UTCTimestamp(dt=t).dt.astimezone(timezone.utc).replace(tzinfo=None), "ns")
            for t in time_steps
        ]

        n_lat, n_lon = self.grid.shape
        lats = self.grid.lats
        lons = self.grid.lons

        # Pre-allocate data structures
        data_arrays: Dict[str, np.ndarray] = {
            var_name: np.full((n_times, n_lat, n_lon), np.nan, dtype=np.float32)
            for var_name in CHANNEL_REGISTRY.keys()
        }
        valid_masks: Dict[str, np.ndarray] = {
            f"{var_name}_valid_mask": np.zeros((n_times, n_lat, n_lon), dtype=bool)
            for var_name in CHANNEL_REGISTRY.keys()
        }

        # Modality availability matrix: shape (N_times, N_modalities)
        modality_mask = np.zeros((n_times, NUM_MODALITIES), dtype=np.uint8)

        # Populate slices
        for t_idx, mod_dict in enumerate(time_step_datasets):
            if not mod_dict:
                continue

            for mod_type in MODALITY_ORDER:
                mod_name = mod_type.value
                ds_mod = mod_dict.get(mod_name)
                m_idx = MODALITY_TO_INDEX[mod_type]

                if ds_mod is None:
                    # Sensor missing / dropped out at this timestamp
                    modality_mask[t_idx, m_idx] = 0
                    continue

                # Modality is present
                modality_mask[t_idx, m_idx] = 1

                # Extract registered variables belonging to this modality
                for var_name in get_channels_for_modality(mod_type):
                    if var_name in ds_mod:
                        arr = ds_mod[var_name].values
                        # Squeeze leading singleton dimension if present
                        if arr.ndim == 3 and arr.shape[0] == 1:
                            arr = arr[0]

                        # Verify dimensions match grid shape
                        if arr.shape == (n_lat, n_lon):
                            data_arrays[var_name][t_idx] = arr.astype(np.float32)

                            # Validity mask
                            vmask_key = f"{var_name}_valid_mask"
                            if vmask_key in ds_mod:
                                vm = ds_mod[vmask_key].values
                                if vm.ndim == 3 and vm.shape[0] == 1:
                                    vm = vm[0]
                                valid_masks[vmask_key][t_idx] = vm.astype(bool)
                            else:
                                valid_masks[vmask_key][t_idx] = ~np.isnan(arr)

        # Assemble unified xarray Dataset
        coords = {
            "time": (["time"], utc_times, {"standard_name": "time"}),
            "latitude": (["latitude"], lats, {"units": "degrees_north"}),
            "longitude": (["longitude"], lons, {"units": "degrees_east"}),
            "modality": (["modality"], [m.value for m in MODALITY_ORDER]),
        }

        data_vars = {}
        for var_name, arr in data_arrays.items():
            spec = CHANNEL_REGISTRY[var_name]
            data_vars[var_name] = (
                ["time", "latitude", "longitude"],
                arr,
                {"units": spec.units, "description": spec.description, "modality": spec.modality.value},
            )
            vmask_key = f"{var_name}_valid_mask"
            data_vars[vmask_key] = (
                ["time", "latitude", "longitude"],
                valid_masks[vmask_key],
                {"description": f"True where {var_name} observation is valid and unmasked"},
            )

        data_vars["modality_mask"] = (
            ["time", "modality"],
            modality_mask,
            {"description": "Binary presence mask (1=sensor available, 0=sensor dropout)"},
        )

        cube = xr.Dataset(data_vars=data_vars, coords=coords)
        cube.attrs.update({
            "title": "APM-Nowcast Unified Multimodal Atmospheric Data Cube",
            "spatial_resolution_deg": self.grid.resolution_deg,
            "domain_coverage": f"Lat [{self.grid.lat_min}, {self.grid.lat_max}], Lon [{self.grid.lon_min}, {self.grid.lon_max}]",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

        return cube

    def save_cube_zarr(
        self,
        cube: xr.Dataset,
        output_path: Union[str, Path],
        overwrite: bool = True,
    ) -> Path:
        """
        Persist unified cube to a Zarr store with zarr_format=2 and consolidated=False
        for Windows & Zarr v3 compatibility.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "w-"

        cube.to_zarr(str(path), mode=mode, zarr_format=2, consolidated=False)
        log.info("Unified data cube persisted to Zarr", extra={"path": str(path)})
        return path

    @staticmethod
    def load_cube_zarr(path: Union[str, Path]) -> xr.Dataset:
        """Load unified data cube from a Zarr store."""
        return xr.open_zarr(str(path), consolidated=False)
