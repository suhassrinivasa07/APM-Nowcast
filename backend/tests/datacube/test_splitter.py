"""
Unit tests for SpatiotemporalSplitter and zero-leakage verification.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from datacube.splitter import SpatiotemporalSplitter, SplitResult


class TestSpatiotemporalSplitter:
    @pytest.fixture
    def consecutive_times(self) -> list[datetime]:
        base = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(40)]

    def test_chronological_split_disjoint_and_ordered(self, consecutive_times):
        split = SpatiotemporalSplitter.chronological_split(
            consecutive_times,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            buffer_frames=2,
        )

        assert len(split.train_indices) > 0
        assert len(split.val_indices) > 0
        assert len(split.test_indices) > 0

        # Sets must be strictly disjoint
        train_s = set(split.train_indices)
        val_s = set(split.val_indices)
        test_s = set(split.test_indices)

        assert train_s.isdisjoint(val_s)
        assert train_s.isdisjoint(test_s)
        assert val_s.isdisjoint(test_s)

        # Chronological ordering: train strictly precedes val, val strictly precedes test
        assert max(split.train_times) < min(split.val_times)
        assert max(split.val_times) < min(split.test_times)

        # Mathematical verification
        assert SpatiotemporalSplitter.verify_zero_leakage(split) is True

    def test_storm_event_split(self):
        # 3 distinct storm episodes separated by 10-hour gaps
        base = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
        times = []
        # Episode 1 (4 frames)
        for i in range(4):
            times.append(base + timedelta(minutes=15 * i))
        # Episode 2 (4 frames after 10 hr gap)
        base2 = base + timedelta(hours=10)
        for i in range(4):
            times.append(base2 + timedelta(minutes=15 * i))
        # Episode 3 (4 frames after another 10 hr gap)
        base3 = base2 + timedelta(hours=10)
        for i in range(4):
            times.append(base3 + timedelta(minutes=15 * i))

        split = SpatiotemporalSplitter.storm_event_split(
            times,
            quiescent_gap_hours=6.0,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )

        assert len(split.train_indices) > 0
        assert len(split.val_indices) > 0
        assert len(split.test_indices) > 0
        assert SpatiotemporalSplitter.verify_zero_leakage(split) is True

    def test_verify_zero_leakage_detects_leak(self, consecutive_times):
        clean_split = SpatiotemporalSplitter.chronological_split(consecutive_times, buffer_frames=2)
        assert SpatiotemporalSplitter.verify_zero_leakage(clean_split) is True

        # Artificially inject a leakage (index in both train and val)
        leaky_split = SplitResult(
            train_indices=clean_split.train_indices + [clean_split.val_indices[0]],
            val_indices=clean_split.val_indices,
            test_indices=clean_split.test_indices,
            train_times=clean_split.train_times,
            val_times=clean_split.val_times,
            test_times=clean_split.test_times,
        )
        assert SpatiotemporalSplitter.verify_zero_leakage(leaky_split) is False
