"""
Unit tests for unified datacube schema and channel registry.
"""
from __future__ import annotations

import pytest

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


class TestDatacubeSchema:
    def test_modality_order_and_count(self):
        assert NUM_MODALITIES == 4
        assert len(MODALITY_ORDER) == 4
        assert ModalityType.SATELLITE in MODALITY_ORDER
        assert ModalityType.RADAR in MODALITY_ORDER
        assert ModalityType.LIGHTNING in MODALITY_ORDER
        assert ModalityType.NWP in MODALITY_ORDER

        for i, mod in enumerate(MODALITY_ORDER):
            assert MODALITY_TO_INDEX[mod] == i

    def test_channel_registry_completeness(self):
        assert len(CHANNEL_REGISTRY) >= 10
        for name, spec in CHANNEL_REGISTRY.items():
            assert isinstance(spec, ChannelSpec)
            assert spec.name == name
            assert spec.modality in MODALITY_ORDER
            assert isinstance(spec.scaling, ScalingType)
            assert spec.sentinel_value == 0.0

    def test_get_channels_for_modality(self):
        sat_channels = get_channels_for_modality(ModalityType.SATELLITE)
        assert "tir1" in sat_channels
        assert "wv" in sat_channels
        assert "mir" in sat_channels

        rad_channels = get_channels_for_modality(ModalityType.RADAR)
        assert "composite_reflectivity" in rad_channels
        assert "reflectivity_cappi" in rad_channels

        lit_channels = get_channels_for_modality(ModalityType.LIGHTNING)
        assert "flash_count" in lit_channels
        assert "occurrence_mask" in lit_channels

        nwp_channels = get_channels_for_modality(ModalityType.NWP)
        assert "cape" in nwp_channels
        assert "cin" in nwp_channels
        assert "wind_speed_10m" in nwp_channels

    def test_get_modality_channel_counts(self):
        counts = get_modality_channel_counts()
        assert counts["satellite"] >= 3
        assert counts["radar"] >= 2
        assert counts["lightning"] >= 2
        assert counts["nwp"] >= 4
