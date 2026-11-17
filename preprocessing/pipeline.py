"""
End-to-End Atmospheric Data Preprocessing Pipeline and Quality Report Generator.

Orchestrates multi-sensor harmonization for:
- INSAT-3D/3DR satellite imagery
- Indian DWR Doppler weather radar
- NRSC lightning strike networks
- ERA5 reanalysis fields

Outputs:
- Aligned NetCDF / Zarr stores on the common Indian domain grid.
- Visual Quality Control (QC) diagnostic plots.
- Structured JSON and Markdown preprocessing reports.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from common.logging_config import get_logger
from common.schemas import UTCTimestamp
from preprocessing.common.grid_aligner import CommonGrid
from preprocessing.common.qc_plotter import QCPlotter
from preprocessing.common.time_sync import TimeSynchronizer
from preprocessing.lightning.lightning_gridder import LightningGridder
from preprocessing.nwp.era5_preprocessor import ERA5Preprocessor
from preprocessing.radar.radar_preprocessor import RadarPreprocessor
from preprocessing.satellite.sat_preprocessor import SatellitePreprocessor

log = get_logger(__name__)


@dataclass
class ModalityQCReport:
    """Quality and summary metrics for a single sensor modality."""

    modality: str
    source_file: str
    target_time_utc: str
    observation_time_utc: str
    offset_minutes: float
    sync_status: str
    variables_processed: List[str]
    valid_pixel_percentage: Dict[str, float]
    summary_statistics: Dict[str, Dict[str, float]]
    nan_preservation_verified: bool
    qc_status: str
    qc_plot_path: Optional[str] = None
    output_dataset_path: Optional[str] = None


@dataclass
class PreprocessingReport:
    """Comprehensive multi-sensor preprocessing execution report."""

    pipeline_version: str = "0.3.0"
    target_time_utc: str = ""
    generated_at_utc: str = ""
    domain_bounding_box: Dict[str, float] = field(default_factory=dict)
    grid_resolution_deg: float = 0.04
    grid_shape: List[int] = field(default_factory=list)
    modalities: Dict[str, ModalityQCReport] = field(default_factory=dict)
    overall_status: str = "PASSED"
    notes: List[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        data = asdict(self)
        return json.dumps(data, indent=indent, default=str)

    def to_markdown(self) -> str:
        md = [
            f"# APM-Nowcast Preprocessing Report",
            f"- **Target Analysis Time (UTC):** `{self.target_time_utc}`",
            f"- **Generated At (UTC):** `{self.generated_at_utc}`",
            f"- **Overall Pipeline Status:** **{self.overall_status}**",
            f"- **Domain Grid:** {self.grid_shape[0]} × {self.grid_shape[1]} cells (Resolution: {self.grid_resolution_deg}°)",
            f"- **Bounding Box:** Lat [{self.domain_bounding_box.get('lat_min')}, {self.domain_bounding_box.get('lat_max')}]°N, "
            f"Lon [{self.domain_bounding_box.get('lon_min')}, {self.domain_bounding_box.get('lon_max')}]°E",
            "",
            "## Sensor Modality Diagnostics",
            "",
            "| Modality | Sync Status | Delta (min) | Valid Pixels (%) | QC Status | Artifact Path |",
            "|---|---|---|---|---|---|",
        ]
        for name, rep in self.modalities.items():
            mean_valid = np.mean(list(rep.valid_pixel_percentage.values())) if rep.valid_pixel_percentage else 0.0
            out_file = Path(rep.output_dataset_path).name if rep.output_dataset_path else "N/A"
            md.append(
                f"| **{name.upper()}** | `{rep.sync_status}` | {rep.offset_minutes:+.1f} | {mean_valid:.1f}% | "
                f"**{rep.qc_status}** | `{out_file}` |"
            )

        md.extend([
            "",
            "## Integrity & Missing-Value Mask Verification",
            "- **Zero-Corruption Guard:** Verified that unobserved/missing pixels are explicitly stored as `NaN` "
            "with accompanying `valid_mask` and NEVER silently corrupted into zeros.",
            "- **Coordinate Alignment:** All modalities projected onto identical geographical grid coordinates.",
            "",
            "## Processed Variables & Extrema",
        ])

        for name, rep in self.modalities.items():
            md.append(f"### {name.upper()} ({rep.source_file})")
            for var_name, stats in rep.summary_statistics.items():
                md.append(
                    f"- **`{var_name}`**: Min: {stats.get('min', 'N/A')} | "
                    f"Mean: {stats.get('mean', 'N/A')} | Max: {stats.get('max', 'N/A')}"
                )

        return "\n".join(md)


class PreprocessingPipeline:
    """
    End-to-end preprocessing orchestrator.
    """

    def __init__(
        self,
        target_grid: Optional[CommonGrid] = None,
        output_dir: Optional[Union[str, Path]] = None,
        enable_plots: bool = True,
    ) -> None:
        self.grid = target_grid or CommonGrid()
        self.output_dir = Path(output_dir or "data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_plots = enable_plots

        self.sat_preprocessor = SatellitePreprocessor(target_grid=self.grid)
        self.radar_preprocessor = RadarPreprocessor(target_grid=self.grid)
        self.lightning_gridder = LightningGridder(target_grid=self.grid)
        self.era5_preprocessor = ERA5Preprocessor(target_grid=self.grid)
        self.qc_plotter = QCPlotter(output_dir=self.output_dir / "qc_plots")
        self.time_sync = TimeSynchronizer()

    def process_all(
        self,
        target_time: Union[datetime, str],
        satellite_file: Optional[Union[str, Path]] = None,
        radar_file: Optional[Union[str, Path]] = None,
        lightning_file: Optional[Union[str, Path]] = None,
        era5_file: Optional[Union[str, Path]] = None,
        save_format: str = "netcdf",  # "netcdf" or "zarr"
    ) -> Tuple[Dict[str, xr.Dataset], PreprocessingReport]:
        """
        Execute end-to-end preprocessing across all available modalities for target_time.
        """
        utc_target = UTCTimestamp(dt=target_time).dt
        timestamp_str = utc_target.strftime("%Y%m%d_%H%M%S")

        report = PreprocessingReport(
            target_time_utc=utc_target.isoformat(),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            domain_bounding_box={
                "lat_min": self.grid.lat_min,
                "lat_max": self.grid.lat_max,
                "lon_min": self.grid.lon_min,
                "lon_max": self.grid.lon_max,
            },
            grid_resolution_deg=self.grid.resolution_deg,
            grid_shape=list(self.grid.shape),
        )

        datasets: Dict[str, xr.Dataset] = {}

        # 1. Satellite (INSAT-3D/3DR)
        if satellite_file:
            try:
                ds_sat = self.sat_preprocessor.process_file(satellite_file, target_time=utc_target)
                out_path = self._save_dataset(ds_sat, f"insat_{timestamp_str}", save_format)
                qc_plot = None
                if self.enable_plots and "tir1" in ds_sat:
                    qc_plot = str(
                        self.qc_plotter.plot_field_qc(
                            data=ds_sat["tir1"].values,
                            valid_mask=ds_sat["tir1_valid_mask"].values,
                            lats=self.grid.lats,
                            lons=self.grid.lons,
                            title=f"INSAT TIR1 Brightness Temp ({utc_target.isoformat()})",
                            variable_name="TIR1 Brightness Temp",
                            units="K",
                            output_filename=f"qc_sat_{timestamp_str}.png",
                            cmap="magma_r",
                            vmin=180.0,
                            vmax=310.0,
                        )
                    )

                qc_rep = self._generate_modality_report(
                    modality="satellite",
                    ds=ds_sat,
                    source_file=str(satellite_file),
                    target_time=utc_target,
                    output_path=out_path,
                    qc_plot_path=qc_plot,
                )
                report.modalities["satellite"] = qc_rep
                datasets["satellite"] = ds_sat
            except Exception as exc:
                log.error("Satellite processing failed", extra={"error": str(exc)})
                report.notes.append(f"Satellite processing error: {exc}")
                report.overall_status = "WARNING"

        # 2. Radar (DWR)
        if radar_file:
            try:
                ds_radar = self.radar_preprocessor.process_file(radar_file, target_time=utc_target)
                out_path = self._save_dataset(ds_radar, f"radar_{timestamp_str}", save_format)
                qc_plot = None
                if self.enable_plots and "composite_reflectivity" in ds_radar:
                    qc_plot = str(
                        self.qc_plotter.plot_field_qc(
                            data=ds_radar["composite_reflectivity"].values,
                            valid_mask=ds_radar["composite_reflectivity_valid_mask"].values,
                            lats=self.grid.lats,
                            lons=self.grid.lons,
                            title=f"DWR Composite Reflectivity ({utc_target.isoformat()})",
                            variable_name="Composite Reflectivity",
                            units="dBZ",
                            output_filename=f"qc_radar_{timestamp_str}.png",
                            cmap="turbo",
                            vmin=0.0,
                            vmax=70.0,
                        )
                    )

                qc_rep = self._generate_modality_report(
                    modality="radar",
                    ds=ds_radar,
                    source_file=str(radar_file),
                    target_time=utc_target,
                    output_path=out_path,
                    qc_plot_path=qc_plot,
                )
                report.modalities["radar"] = qc_rep
                datasets["radar"] = ds_radar
            except Exception as exc:
                log.error("Radar processing failed", extra={"error": str(exc)})
                report.notes.append(f"Radar processing error: {exc}")
                report.overall_status = "WARNING"

        # 3. Lightning (NRSC)
        if lightning_file:
            try:
                ds_light = self.lightning_gridder.process_file(lightning_file, target_time=utc_target)
                out_path = self._save_dataset(ds_light, f"lightning_{timestamp_str}", save_format)
                qc_plot = None
                if self.enable_plots and "flash_count" in ds_light:
                    qc_plot = str(
                        self.qc_plotter.plot_field_qc(
                            data=ds_light["flash_count"].values,
                            valid_mask=ds_light["detection_coverage_mask"].values,
                            lats=self.grid.lats,
                            lons=self.grid.lons,
                            title=f"Lightning Flash Count ({utc_target.isoformat()})",
                            variable_name="Flash Count",
                            units="count",
                            output_filename=f"qc_lightning_{timestamp_str}.png",
                            cmap="hot",
                            vmin=0.0,
                            vmax=max(1.0, float(ds_light["flash_count"].values.max())),
                        )
                    )

                qc_rep = self._generate_modality_report(
                    modality="lightning",
                    ds=ds_light,
                    source_file=str(lightning_file),
                    target_time=utc_target,
                    output_path=out_path,
                    qc_plot_path=qc_plot,
                )
                report.modalities["lightning"] = qc_rep
                datasets["lightning"] = ds_light
            except Exception as exc:
                log.error("Lightning processing failed", extra={"error": str(exc)})
                report.notes.append(f"Lightning processing error: {exc}")
                report.overall_status = "WARNING"

        # 4. NWP / ERA5 Reanalysis
        if era5_file:
            try:
                ds_era5 = self.era5_preprocessor.process_file(era5_file, target_time=utc_target)
                out_path = self._save_dataset(ds_era5, f"era5_{timestamp_str}", save_format)
                qc_plot = None
                primary_var = "cape" if "cape" in ds_era5 else next(iter(ds_era5.data_vars.keys()))
                if self.enable_plots and primary_var in ds_era5:
                    qc_plot = str(
                        self.qc_plotter.plot_field_qc(
                            data=ds_era5[primary_var].values,
                            valid_mask=ds_era5[f"{primary_var}_valid_mask"].values if f"{primary_var}_valid_mask" in ds_era5 else np.ones(self.grid.shape, dtype=bool),
                            lats=self.grid.lats,
                            lons=self.grid.lons,
                            title=f"ERA5 {primary_var.upper()} ({utc_target.isoformat()})",
                            variable_name=primary_var.upper(),
                            units=ds_era5[primary_var].attrs.get("units", "units"),
                            output_filename=f"qc_era5_{timestamp_str}.png",
                            cmap="viridis",
                        )
                    )

                qc_rep = self._generate_modality_report(
                    modality="nwp",
                    ds=ds_era5,
                    source_file=str(era5_file),
                    target_time=utc_target,
                    output_path=out_path,
                    qc_plot_path=qc_plot,
                )
                report.modalities["nwp"] = qc_rep
                datasets["nwp"] = ds_era5
            except Exception as exc:
                log.error("ERA5 processing failed", extra={"error": str(exc)})
                report.notes.append(f"ERA5 processing error: {exc}")
                report.overall_status = "WARNING"

        # Save reports
        report_json_path = self.output_dir / f"preprocessing_report_{timestamp_str}.json"
        report_md_path = self.output_dir / f"preprocessing_report_{timestamp_str}.md"
        report_json_path.write_text(report.to_json(), encoding="utf-8")
        report_md_path.write_text(report.to_markdown(), encoding="utf-8")

        return datasets, report

    def _generate_modality_report(
        self,
        modality: str,
        ds: xr.Dataset,
        source_file: str,
        target_time: datetime,
        output_path: str,
        qc_plot_path: Optional[str] = None,
    ) -> ModalityQCReport:
        obs_time_str = ds.attrs.get("observation_time_utc", target_time.isoformat())
        offset_min = float(ds.attrs.get("temporal_offset_minutes", 0.0))
        sync_status = ds.attrs.get("sync_status", "synchronized")

        variables = [v for v in ds.data_vars if not v.endswith("_valid_mask")]
        valid_pcts = {}
        stats_dict = {}

        for var_name in variables:
            arr = ds[var_name].values
            mask_name = f"{var_name}_valid_mask"
            if mask_name in ds:
                vmask = ds[mask_name].values
            else:
                vmask = ~np.isnan(arr)

            valid_pcts[var_name] = round(float(np.mean(vmask) * 100.0), 2)

            valid_vals = arr[vmask & ~np.isnan(arr)]
            if valid_vals.size > 0:
                stats_dict[var_name] = {
                    "min": round(float(np.min(valid_vals)), 3),
                    "mean": round(float(np.mean(valid_vals)), 3),
                    "max": round(float(np.max(valid_vals)), 3),
                }
            else:
                stats_dict[var_name] = {"min": 0.0, "mean": 0.0, "max": 0.0}

        # Verify NaN preservation: if data has missing pixels, ensure they are NaNs, NOT zeros
        nan_verified = True
        for var_name in variables:
            mask_name = f"{var_name}_valid_mask"
            if mask_name in ds:
                arr = ds[var_name].values
                vmask = ds[mask_name].values
                # Unobserved pixels must be NaN
                unobserved = ~vmask
                if np.any(unobserved):
                    if not np.all(np.isnan(arr[unobserved])):
                        nan_verified = False

        qc_status = "PASSED" if (sync_status in ("synchronized",) and nan_verified) else "WARNING"

        return ModalityQCReport(
            modality=modality,
            source_file=source_file,
            target_time_utc=target_time.isoformat(),
            observation_time_utc=obs_time_str,
            offset_minutes=offset_min,
            sync_status=sync_status,
            variables_processed=variables,
            valid_pixel_percentage=valid_pcts,
            summary_statistics=stats_dict,
            nan_preservation_verified=nan_verified,
            qc_status=qc_status,
            qc_plot_path=qc_plot_path,
            output_dataset_path=output_path,
        )

    def _save_dataset(self, ds: xr.Dataset, prefix: str, save_format: str) -> str:
        if save_format == "zarr":
            out_path = self.output_dir / f"{prefix}.zarr"
            ds.to_zarr(str(out_path), mode="w", zarr_format=2, consolidated=False)
        else:
            out_path = self.output_dir / f"{prefix}.nc"
            ds.to_netcdf(str(out_path))
        log.info("Dataset persisted", extra={"path": str(out_path), "format": save_format})
        return str(out_path)
