"""
Unified Multimodal Cube Schema, Channel Registry, and Modality Definitions.

Defines:
- Standardized sensor modalities (satellite, radar, lightning, NWP).
- Channel order and channel metadata for deep learning tensors.
- Scaling methods (Z-score, Min-Max, Log1p, Identity).
- Binary modality availability masks for adaptive multimodal dropout.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ModalityType(str, Enum):
    """Sensor modalities ingested into the unified atmospheric data cube."""
    SATELLITE = "satellite"
    RADAR = "radar"
    LIGHTNING = "lightning"
    NWP = "nwp"


class ScalingType(str, Enum):
    """Normalization strategy per channel."""
    ZSCORE = "zscore"          # (x - mu) / sigma
    MINMAX = "minmax"          # (x - min) / (max - min)
    LOG1P = "log1p"            # log(1 + max(0, x)) then scaled
    IDENTITY = "identity"      # Passthrough (e.g. binary occurrence mask)


@dataclass(frozen=True)
class ChannelSpec:
    """Specification for a single variable within the multimodal data cube."""
    name: str
    modality: ModalityType
    scaling: ScalingType
    units: str
    sentinel_value: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    description: str = ""


# Modality ordering for multi-sensor tensor packing and modality availability masks
MODALITY_ORDER: List[ModalityType] = [
    ModalityType.SATELLITE,
    ModalityType.RADAR,
    ModalityType.LIGHTNING,
    ModalityType.NWP,
]

MODALITY_TO_INDEX: Dict[ModalityType, int] = {
    mod: idx for idx, mod in enumerate(MODALITY_ORDER)
}

NUM_MODALITIES: int = len(MODALITY_ORDER)


# Canonical registry of channels included in APM-Nowcast training cubes
CHANNEL_REGISTRY: Dict[str, ChannelSpec] = {
    # Satellite: INSAT-3D/3DR Brightness Temperatures (K)
    "tir1": ChannelSpec(
        name="tir1",
        modality=ModalityType.SATELLITE,
        scaling=ScalingType.ZSCORE,
        units="K",
        clip_min=160.0,
        clip_max=330.0,
        description="Thermal Infrared 1 (10.8 um) Brightness Temperature",
    ),
    "wv": ChannelSpec(
        name="wv",
        modality=ModalityType.SATELLITE,
        scaling=ScalingType.ZSCORE,
        units="K",
        clip_min=180.0,
        clip_max=300.0,
        description="Water Vapor (6.8 um) Brightness Temperature",
    ),
    "mir": ChannelSpec(
        name="mir",
        modality=ModalityType.SATELLITE,
        scaling=ScalingType.ZSCORE,
        units="K",
        clip_min=200.0,
        clip_max=340.0,
        description="Middle Infrared (3.9 um) Brightness Temperature",
    ),
    # Radar: DWR Reflectivity (dBZ)
    "composite_reflectivity": ChannelSpec(
        name="composite_reflectivity",
        modality=ModalityType.RADAR,
        scaling=ScalingType.LOG1P,
        units="dBZ",
        clip_min=0.0,
        clip_max=75.0,
        description="Column-Maximum Composite Reflectivity",
    ),
    "reflectivity_cappi": ChannelSpec(
        name="reflectivity_cappi",
        modality=ModalityType.RADAR,
        scaling=ScalingType.LOG1P,
        units="dBZ",
        clip_min=0.0,
        clip_max=75.0,
        description="Lowest Elevation / Surface CAPPI Reflectivity",
    ),
    # Lightning: NRSC Event Aggregations
    "flash_count": ChannelSpec(
        name="flash_count",
        modality=ModalityType.LIGHTNING,
        scaling=ScalingType.LOG1P,
        units="count",
        clip_min=0.0,
        clip_max=500.0,
        description="Total Accumulated Flash Count per Grid Cell",
    ),
    "occurrence_mask": ChannelSpec(
        name="occurrence_mask",
        modality=ModalityType.LIGHTNING,
        scaling=ScalingType.IDENTITY,
        units="binary",
        clip_min=0.0,
        clip_max=1.0,
        description="Binary Lightning Strike Occurrence (1 if flashes > 0 else 0)",
    ),
    # NWP: ERA5 Convective & Thermodynamic Indicators
    "cape": ChannelSpec(
        name="cape",
        modality=ModalityType.NWP,
        scaling=ScalingType.LOG1P,
        units="J kg**-1",
        clip_min=0.0,
        clip_max=6000.0,
        description="Convective Available Potential Energy",
    ),
    "cin": ChannelSpec(
        name="cin",
        modality=ModalityType.NWP,
        scaling=ScalingType.LOG1P,
        units="J kg**-1",
        clip_min=0.0,
        clip_max=1000.0,
        description="Convective Inhibition",
    ),
    "relative_humidity_2m": ChannelSpec(
        name="relative_humidity_2m",
        modality=ModalityType.NWP,
        scaling=ScalingType.MINMAX,
        units="%",
        clip_min=0.0,
        clip_max=100.0,
        description="2m Relative Humidity Proxy",
    ),
    "wind_speed_10m": ChannelSpec(
        name="wind_speed_10m",
        modality=ModalityType.NWP,
        scaling=ScalingType.ZSCORE,
        units="m s**-1",
        clip_min=0.0,
        clip_max=60.0,
        description="10m Horizontal Wind Speed Magnitude",
    ),
}


def get_channels_for_modality(modality: ModalityType) -> List[str]:
    """Return ordered channel variable names belonging to the specified modality."""
    return [name for name, spec in CHANNEL_REGISTRY.items() if spec.modality == modality]


def get_modality_channel_counts() -> Dict[str, int]:
    """Return count of channels per modality."""
    counts = {}
    for mod in MODALITY_ORDER:
        counts[mod.value] = len(get_channels_for_modality(mod))
    return counts
