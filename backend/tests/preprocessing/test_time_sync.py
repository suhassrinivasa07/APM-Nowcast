"""
Unit tests for TimeSynchronizer and TimeWindow.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from preprocessing.common.time_sync import TimeSynchronizer, TimeWindow


class TestTimeSynchronizer:
    def setup_method(self):
        self.sync = TimeSynchronizer()

    def test_ensure_utc(self):
        naive_dt = datetime(2024, 6, 1, 12, 0, 0)
        utc_dt = self.sync.ensure_utc(naive_dt)
        assert utc_dt.tzinfo == timezone.utc

        str_dt = "2024-06-01T12:00:00Z"
        utc_dt_str = self.sync.ensure_utc(str_dt)
        assert utc_dt_str.tzinfo == timezone.utc

        pd_ts = pd.Timestamp("2024-06-01 12:00:00")
        utc_ts = self.sync.ensure_utc(pd_ts)
        assert utc_ts.tzinfo == timezone.utc

    def test_evaluate_alignment_within_tolerance(self):
        target = "2024-06-01T06:00:00Z"
        obs = "2024-06-01T06:07:00Z"  # 7 min offset

        res = self.sync.evaluate_alignment(obs, target, tolerance_minutes=15.0)
        assert res.is_matched is True
        assert res.status == "synchronized"
        assert res.offset_minutes == pytest.approx(7.0)

    def test_evaluate_alignment_exceeding_tolerance(self):
        target = "2024-06-01T06:00:00Z"
        obs = "2024-06-01T06:25:00Z"  # 25 min offset

        res = self.sync.evaluate_alignment(obs, target, tolerance_minutes=15.0)
        assert res.is_matched is False
        assert res.status == "temporal_offset_exceeded"
        assert res.warning is not None

    def test_filter_dataframe_retrospective_window(self):
        target = "2024-06-01T06:00:00Z"
        df = pd.DataFrame({
            "timestamp": [
                "2024-06-01T05:15:00Z",  # too early (45 min before)
                "2024-06-01T05:40:00Z",  # inside (20 min before)
                "2024-06-01T05:59:00Z",  # inside (1 min before)
                "2024-06-01T06:05:00Z",  # too late (5 min after)
            ],
            "val": [1, 2, 3, 4],
        })

        filtered, window = self.sync.filter_dataframe_window(
            df,
            time_column="timestamp",
            target_time=target,
            retrospective_minutes=30.0,
        )

        assert len(filtered) == 2
        assert list(filtered["val"]) == [2, 3]

    def test_find_nearest_slice_index(self):
        times = [
            datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 6, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        ]
        target = datetime(2024, 6, 1, 5, 45, tzinfo=timezone.utc)

        idx, actual, diff_min = self.sync.find_nearest_slice_index(times, target, max_offset_minutes=60.0)
        assert idx == 1  # 06:00 is closest (15 min difference)
        assert diff_min == pytest.approx(15.0)
