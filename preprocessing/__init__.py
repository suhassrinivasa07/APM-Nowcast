"""
APM-Nowcast Preprocessing Package.

Module 3: Atmospheric data preprocessing for satellite, radar, lightning,
and numerical weather prediction (NWP) datasets.
"""

from preprocessing.common.grid_aligner import CommonGrid, GridAligner
from preprocessing.common.time_sync import TimeSynchronizer, TimeWindow, TimeSyncResult
from preprocessing.satellite.sat_preprocessor import SatellitePreprocessor
from preprocessing.radar.radar_preprocessor import RadarPreprocessor
from preprocessing.lightning.lightning_gridder import LightningGridder
from preprocessing.nwp.era5_preprocessor import ERA5Preprocessor
from preprocessing.pipeline import PreprocessingPipeline, PreprocessingReport

__all__ = [
    "CommonGrid",
    "GridAligner",
    "TimeSynchronizer",
    "TimeWindow",
    "TimeSyncResult",
    "SatellitePreprocessor",
    "RadarPreprocessor",
    "LightningGridder",
    "ERA5Preprocessor",
    "PreprocessingPipeline",
    "PreprocessingReport",
]
