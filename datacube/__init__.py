"""
APM-Nowcast Multimodal Data Cube & PyTorch Dataset Pipeline.

Module 4: Spatiotemporal cube construction, modality availability tracking,
modality-specific normalization, convective-weighted patch sampling,
and PyTorch DataLoader interfaces.
"""

from datacube.cube_builder import DataCubeBuilder
from datacube.dataset import AtmosphericDataCubeDataset, create_dataloaders
from datacube.normalizer import ModalityNormalizer
from datacube.patch_sampler import ConvectivePatchSampler, PatchCoordinate
from datacube.schema import (
    CHANNEL_REGISTRY,
    MODALITY_ORDER,
    MODALITY_TO_INDEX,
    NUM_MODALITIES,
    ChannelSpec,
    ModalityType,
    ScalingType,
    get_channels_for_modality,
    get_modality_channel_counts,
)
from datacube.splitter import SpatiotemporalSplitter, SplitResult

__all__ = [
    "DataCubeBuilder",
    "AtmosphericDataCubeDataset",
    "create_dataloaders",
    "ModalityNormalizer",
    "ConvectivePatchSampler",
    "PatchCoordinate",
    "SpatiotemporalSplitter",
    "SplitResult",
    "ModalityType",
    "ScalingType",
    "ChannelSpec",
    "CHANNEL_REGISTRY",
    "MODALITY_ORDER",
    "MODALITY_TO_INDEX",
    "NUM_MODALITIES",
    "get_channels_for_modality",
    "get_modality_channel_counts",
]
