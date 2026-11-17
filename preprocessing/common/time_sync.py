"""
Strict UTC Timestamp Synchronization and Temporal Windowing Engine.

Handles multi-source temporal alignment:
- Enforces UTC normalization across all modalities.
- Validates observations within sensor-specific tolerance windows.
- Accumulates retrospective event streams (e.g., lightning past 15/30/60 min).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple, Union

import pandas as pd

from common.logging_config import get_logger
from common.schemas import UTCTimestamp

log = get_logger(__name__)


@dataclass(frozen=True)
class TimeSyncResult:
    """Outcome of a temporal synchronization evaluation."""

    is_matched: bool
    target_time: datetime
    actual_time: datetime
    offset_seconds: float
    offset_minutes: float
    status: str
    warning: Optional[str] = None


@dataclass(frozen=True)
class TimeWindow:
    """Configurable UTC temporal analysis window."""

    target_time: datetime
    tolerance_minutes: float = 15.0
    retrospective_minutes: Optional[float] = None

    @classmethod
    def create(
        cls,
        target_time: Union[datetime, str],
        tolerance_minutes: float = 15.0,
        retrospective_minutes: Optional[float] = None,
    ) -> "TimeWindow":
        utc_target = UTCTimestamp(dt=target_time).dt
        return cls(
            target_time=utc_target,
            tolerance_minutes=tolerance_minutes,
            retrospective_minutes=retrospective_minutes,
        )

    @property
    def start_time(self) -> datetime:
        """Start of the acceptance or accumulation window."""
        if self.retrospective_minutes is not None:
            return self.target_time - timedelta(minutes=self.retrospective_minutes)
        return self.target_time - timedelta(minutes=self.tolerance_minutes)

    @property
    def end_time(self) -> datetime:
        """End of the acceptance window."""
        if self.retrospective_minutes is not None:
            return self.target_time
        return self.target_time + timedelta(minutes=self.tolerance_minutes)

    def contains(self, dt: Union[datetime, str]) -> bool:
        """Check if an observation timestamp falls inside the window."""
        utc_dt = UTCTimestamp(dt=dt).dt
        return self.start_time <= utc_dt <= self.end_time


class TimeSynchronizer:
    """
    Evaluates temporal proximity and enforces sensor alignment windows.
    """

    @staticmethod
    def ensure_utc(dt: Union[datetime, str, pd.Timestamp]) -> datetime:
        """Coerce any datetime/string to a UTC-aware datetime."""
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is None:
                return dt.tz_localize("UTC").to_pydatetime()
            return dt.tz_convert("UTC").to_pydatetime()
        return UTCTimestamp(dt=dt).dt

    def evaluate_alignment(
        self,
        actual_time: Union[datetime, str],
        target_time: Union[datetime, str],
        tolerance_minutes: float = 15.0,
    ) -> TimeSyncResult:
        """
        Evaluate whether actual observation time matches target_time within tolerance.
        """
        t_actual = self.ensure_utc(actual_time)
        t_target = self.ensure_utc(target_time)

        offset_sec = (t_actual - t_target).total_seconds()
        abs_offset_min = abs(offset_sec) / 60.0

        is_matched = abs_offset_min <= tolerance_minutes
        if is_matched:
            status = "synchronized"
            warning = None
        else:
            status = "temporal_offset_exceeded"
            warning = (
                f"Observation time {t_actual.isoformat()} deviates from target "
                f"{t_target.isoformat()} by {abs_offset_min:.1f} min (tolerance: {tolerance_minutes} min)"
            )
            log.warning(warning)

        return TimeSyncResult(
            is_matched=is_matched,
            target_time=t_target,
            actual_time=t_actual,
            offset_seconds=offset_sec,
            offset_minutes=offset_sec / 60.0,
            status=status,
            warning=warning,
        )

    def filter_dataframe_window(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_time: Union[datetime, str],
        retrospective_minutes: float = 30.0,
    ) -> Tuple[pd.DataFrame, TimeWindow]:
        """
        Filter an event DataFrame (e.g. lightning) to a retrospective time window [T - delta, T].
        """
        window = TimeWindow.create(
            target_time=target_time,
            retrospective_minutes=retrospective_minutes,
        )

        if df.empty or time_column not in df.columns:
            return df.iloc[0:0].copy(), window

        # Ensure UTC datetime series
        utc_series = pd.to_datetime(df[time_column], utc=True)
        mask = (utc_series >= window.start_time) & (utc_series <= window.end_time)
        filtered = df.loc[mask].copy()
        filtered[time_column] = utc_series.loc[mask]

        return filtered, window

    def find_nearest_slice_index(
        self,
        available_times: Sequence[Union[datetime, str, np.datetime64, pd.Timestamp]],
        target_time: Union[datetime, str],
        max_offset_minutes: float = 60.0,
    ) -> Tuple[int, datetime, float]:
        """
        Find the index of the closest timestamp in a sequence.

        Returns
        -------
        (best_idx, actual_time_utc, offset_minutes)

        Raises
        ------
        ValueError if sequence is empty or closest timestamp exceeds max_offset_minutes.
        """
        if len(available_times) == 0:
            raise ValueError("No available timestamps provided.")

        t_target = self.ensure_utc(target_time)
        best_idx = 0
        best_diff = float("inf")
        best_time = t_target

        for i, t in enumerate(available_times):
            utc_t = self.ensure_utc(t)
            diff = abs((utc_t - t_target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx = i
                best_time = utc_t

        offset_min = best_diff / 60.0
        if offset_min > max_offset_minutes:
            raise ValueError(
                f"Closest timestamp {best_time.isoformat()} is {offset_min:.1f} min away, "
                f"exceeding maximum permitted offset of {max_offset_minutes} min."
            )

        return best_idx, best_time, offset_min
