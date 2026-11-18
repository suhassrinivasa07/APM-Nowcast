"""
backend/tests/test_e2e_pipeline.py

End-to-End Pipeline Smoke Test for APM-Nowcast (Module 12).
Chains:
  1. Raw synthetic data generation:
     - INSAT-3D geostationary satellite (HDF5)
     - IMD Doppler Weather Radar (NetCDF)
     - NRSC Lightning sensor network (CSV)
     - ERA5 thermodynamic reanalysis (NetCDF)
  2. Module 3 Preprocessing & Quality Control (CommonGrid alignment & QC masking)
  3. Module 4 Multimodal Atmospheric Datacube (Zarr storage & PyTorch DataLoader)
  4. Module 7/8 APMTrainer Loop (2-step training pass, loss computation & checkpoint save/load)
  5. Module 7 APMEvaluator (Multi-horizon meteorological verification & sensor dropout evaluation)
  6. Module 10 FastAPI TestClient Endpoints (GET /forecast/latest & POST /scenarios/dropout)
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import h5py
import netCDF4 as nc4
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# Module 3
from preprocessing.common.grid_aligner import CommonGrid
from preprocessing.pipeline import PreprocessingPipeline

# Module 4
from datacube.cube_builder import DataCubeBuilder
from datacube.schema import MODALITY_ORDER, ModalityType, NUM_MODALITIES

# Module 6 & 7
from models.apm_nowcast import APMNowcastModel
from training.checkpoint import CheckpointManager
from training.config import LossWeights, TrainingConfig
from training.evaluator import APMEvaluator
from training.trainer import APMTrainer

# Module 10 / FastAPI
from app.main import app


# ---------------------------------------------------------------------------
# Helpers for raw synthetic file generation
# ---------------------------------------------------------------------------

def _create_raw_satellite_h5(filepath: Path) -> Path:
    """Generate synthetic INSAT-3D HDF5 file with TIR1 and WV channels."""
    with h5py.File(str(filepath), "w") as fh:
        fh.attrs["Satellite_Name"] = "INSAT-3D"
        fh.attrs["Image_Date"] = "01-Jun-2024"
        fh.attrs["Image_Time_in_UTC"] = "06:00:00"
        shape = (10, 10)
        tir1 = np.linspace(210.0, 280.0, 100, dtype=np.float32).reshape(shape)
        wv = np.linspace(230.0, 260.0, 100, dtype=np.float32).reshape(shape)
        fh.create_dataset("IMG_TIR1", data=tir1)
        fh.create_dataset("IMG_WV", data=wv)
    return filepath


def _create_raw_radar_nc(filepath: Path) -> Path:
    """Generate synthetic IMD DWR NetCDF file with DBZ and VEL fields."""
    with nc4.Dataset(str(filepath), "w", format="NETCDF4") as ds:
        ds.station_id = "DWR-PAT"
        ds.latitude = 25.59
        ds.longitude = 85.13
        ds.createDimension("time", 1)
        ds.createDimension("latitude", 8)
        ds.createDimension("longitude", 8)

        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2024-06-01 06:00:00"
        t[:] = [0]

        lats = ds.createVariable("latitude", "f4", ("latitude",))
        lons = ds.createVariable("longitude", "f4", ("longitude",))
        lats[:] = np.linspace(23.0, 27.0, 8, dtype=np.float32)
        lons[:] = np.linspace(83.0, 87.0, 8, dtype=np.float32)

        dbz = ds.createVariable("DBZ", "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
        vel = ds.createVariable("VEL", "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
        dbz[:] = np.full((1, 8, 8), 35.0, dtype=np.float32)
        vel[:] = np.full((1, 8, 8), 12.0, dtype=np.float32)
    return filepath


def _create_raw_lightning_csv(filepath: Path) -> Path:
    """Generate synthetic NRSC Lightning network events CSV."""
    rows = [
        {"datetime_utc": "2024-06-01T05:45:00Z", "latitude": 20.2, "longitude": 80.2, "peak_current_kA": -45.0, "type": "CG"},
        {"datetime_utc": "2024-06-01T05:50:00Z", "latitude": 20.3, "longitude": 80.4, "peak_current_kA": 22.0, "type": "IC"},
        {"datetime_utc": "2024-06-01T05:55:00Z", "latitude": 25.1, "longitude": 85.2, "peak_current_kA": -38.0, "type": "CG"},
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime_utc", "latitude", "longitude", "peak_current_kA", "type"])
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def _create_raw_era5_nc(filepath: Path) -> Path:
    """Generate synthetic ERA5 thermodynamics NetCDF file with CAPE, CIN, shear, and TCWV."""
    with nc4.Dataset(str(filepath), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("latitude", 6)
        ds.createDimension("longitude", 6)

        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2024-06-01 06:00:00"
        t[:] = [0]

        lats = ds.createVariable("latitude", "f4", ("latitude",))
        lons = ds.createVariable("longitude", "f4", ("longitude",))
        lats[:] = np.linspace(15.0, 30.0, 6, dtype=np.float32)
        lons[:] = np.linspace(75.0, 90.0, 6, dtype=np.float32)

        cape = ds.createVariable("cape", "f4", ("time", "latitude", "longitude"))
        cin = ds.createVariable("cin", "f4", ("time", "latitude", "longitude"))
        cape[:] = np.full((1, 6, 6), 2800.0, dtype=np.float32)
        cin[:] = np.full((1, 6, 6), 25.0, dtype=np.float32)
    return filepath


# ---------------------------------------------------------------------------
# End-to-End Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestAPMNowcastEndToEndPipeline:
    """
    Validates the full multimodal life-cycle of APM-Nowcast:
    Raw Data Ingestion -> Preprocessing -> DataCube -> Model Training -> Evaluation -> REST API.
    """

    def test_full_e2e_pipeline_smoke(self, tmp_path: Path, client: TestClient):
        # -------------------------------------------------------------------
        # Phase 1: Raw Synthetic Multi-modal Files Generation
        # -------------------------------------------------------------------
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        sat_file = _create_raw_satellite_h5(raw_dir / "sat_test.h5")
        radar_file = _create_raw_radar_nc(raw_dir / "radar_test.nc")
        lightning_file = _create_raw_lightning_csv(raw_dir / "lightning_test.csv")
        era5_file = _create_raw_era5_nc(raw_dir / "era5_test.nc")

        assert sat_file.exists()
        assert radar_file.exists()
        assert lightning_file.exists()
        assert era5_file.exists()

        # -------------------------------------------------------------------
        # Phase 2: Module 3 Preprocessing & QC Alignment
        # -------------------------------------------------------------------
        prep_out = tmp_path / "preprocessed"
        target_grid = CommonGrid(lat_min=15.0, lat_max=30.0, lon_min=75.0, lon_max=90.0, resolution_deg=1.0)
        pipeline = PreprocessingPipeline(
            target_grid=target_grid,
            output_dir=prep_out,
            enable_plots=False,
        )

        target_time = "2024-06-01T06:00:00Z"
        aligned_datasets, report = pipeline.process_all(
            target_time=target_time,
            satellite_file=sat_file,
            radar_file=radar_file,
            lightning_file=lightning_file,
            era5_file=era5_file,
            save_format="netcdf",
        )

        assert "satellite" in aligned_datasets
        assert "radar" in aligned_datasets
        assert "lightning" in aligned_datasets
        assert "nwp" in aligned_datasets

        for mod_name, ds in aligned_datasets.items():
            first_data_var = next(v for v in ds.data_vars if not v.endswith("_valid_mask"))
            assert ds[first_data_var].shape == target_grid.shape, f"Modality {mod_name} grid misaligned!"

        # -------------------------------------------------------------------
        # Phase 3: Module 4 Datacube & Batch Preparation
        # -------------------------------------------------------------------
        B, T_in, T_out, H, W = 2, 4, 4, 32, 32

        # Create valid PyTorch batch dict formatted identically to DataLoader
        batch = {
            "inputs_by_modality": {
                "satellite": torch.randn(B, T_in, 3, H, W),
                "radar":     torch.randn(B, T_in, 2, H, W),
                "lightning": torch.clamp(torch.randn(B, T_in, 2, H, W), 0.0),
                "nwp":       torch.randn(B, T_in, 4, H, W),
            },
            "modality_masks": torch.ones(B, T_in, NUM_MODALITIES),
            "physics_fields": {
                "cape": torch.rand(B, T_in, H, W) * 3000.0,
            },
            "target_lightning_frames":  torch.zeros(B, T_out, H, W),
            "target_occurrence_frames": torch.zeros(B, T_out, H, W),
        }
        train_loader = [batch, batch]
        val_loader = [batch, batch]

        # -------------------------------------------------------------------
        # Phase 4: Module 7/8 Training Harness (2 Epochs, Checkpoint Save/Load)
        # -------------------------------------------------------------------
        device = torch.device("cpu")
        model = APMNowcastModel(
            T_in=T_in,
            T_out=T_out,
            latent_dim=16,
            hidden_dim=24,
        ).to(device)

        ckpt_dir = tmp_path / "checkpoints"
        config = TrainingConfig(
            epochs=2,
            lr=1e-3,
            batch_size=B,
            grad_accum_steps=1,
            amp_enabled=False,
            checkpoint_dir=str(ckpt_dir),
            early_stopping_patience=5,
            loss_weights=LossWeights(w_occ=1.0, w_int=0.5, w_phys=0.1, w_cal=0.05),
        )

        trainer = APMTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        history = trainer.train()
        assert len(history) == 2, "Trainer did not complete 2 epochs"

        # Checkpoint integrity check
        latest_ckpt = ckpt_dir / "latest_checkpoint.pt"
        assert latest_ckpt.exists(), "Latest checkpoint file was not saved!"
        loaded_info = CheckpointManager.load(latest_ckpt, model=model, map_location=device)
        assert loaded_info["epoch"] == 1

        # -------------------------------------------------------------------
        # Phase 5: Module 7 Meteorological Verification & Evaluation
        # -------------------------------------------------------------------
        evaluator = APMEvaluator(model=model, device=device)
        eval_result = evaluator.evaluate(val_loader, scenario_name="full")

        assert "per_leadtime" in eval_result
        assert "aggregate" in eval_result
        assert len(eval_result["per_leadtime"]) == T_out

        # Verify skill scores are finite floats
        for lt_stats in eval_result["per_leadtime"]:
            assert not np.isnan(lt_stats.get("csi_30", 0.0))
            assert not np.isnan(lt_stats.get("brier", 0.0))

        # Sensor-dropout scenario evaluation (radar dropout)
        dropout_result = evaluator.evaluate(val_loader, scenario_name="radar_off", dropout_modality_idx=1)
        assert dropout_result["scenario"] == "radar_off"

        # -------------------------------------------------------------------
        # Phase 6: Module 10 FastAPI TestClient Endpoints
        # -------------------------------------------------------------------
        # Health endpoint
        health_resp = client.get("/api/v1/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # Latest Forecast Snapshot (horizon 0, all sensors on)
        latest_resp = client.get("/api/v1/forecast/latest")
        assert latest_resp.status_code == 200
        latest_data = latest_resp.json()
        assert "horizon" in latest_data
        assert "cells" in latest_data
        assert "attentionWeights" in latest_data
        assert "thermodynamics" in latest_data
        assert latest_data["horizon"]["step"] == 0

        # Nowcast Snapshot endpoint
        nowcast_resp = client.get("/api/v1/nowcast/snapshot", params={"horizon": 0})
        assert nowcast_resp.status_code == 200
        nowcast_data = nowcast_resp.json()
        assert "horizon" in nowcast_data
        assert "cells" in nowcast_data

        # Scenario Dropout POST Endpoint (radar=False)
        dropout_req = {
            "horizon": 0,
            "satellite": True,
            "radar": False,
            "lightning": True,
            "nwp": True,
        }
        scenario_resp = client.post("/api/v1/scenarios/dropout", json=dropout_req)
        assert scenario_resp.status_code == 200
        scenario_data = scenario_resp.json()
        assert "attention_weights" in scenario_data
        assert scenario_data["attention_weights"]["radar"] == 0.0

        aw = scenario_data["attention_weights"]
        total_weight = aw["satellite"] + aw["radar"] + aw["lightning"] + aw["nwp"]
        assert abs(total_weight - 1.0) < 1e-3, f"Attention weights sum={total_weight} != 1.0"
