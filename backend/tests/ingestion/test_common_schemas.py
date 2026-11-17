"""Unit tests for common/schemas.py"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from common.schemas import (
    UTCTimestamp,
    GeoCoordinate,
    GeoBoundingBox,
    GridMetadata,
    DatasetInventoryRecord,
    IngestionStatus,
)


class TestUTCTimestamp:
    def test_naive_becomes_utc(self):
        ts = UTCTimestamp(dt=datetime(2024, 6, 1, 12, 0))
        assert ts.dt.tzinfo is not None
        assert ts.dt.tzinfo == timezone.utc

    def test_string_input(self):
        ts = UTCTimestamp(dt="2024-06-01T06:00:00")
        assert isinstance(ts.dt, datetime)

    def test_isoformat_property(self):
        ts = UTCTimestamp(dt=datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert "2024-06-01" in ts.isoformat


class TestGeoBoundingBox:
    def test_valid_bbox(self):
        bbox = GeoBoundingBox(lat_min=5, lat_max=40, lon_min=65, lon_max=100)
        assert bbox.lat_min == 5.0

    def test_invalid_ordering_lat(self):
        with pytest.raises(ValidationError):
            GeoBoundingBox(lat_min=40, lat_max=5, lon_min=65, lon_max=100)

    def test_invalid_ordering_lon(self):
        with pytest.raises(ValidationError):
            GeoBoundingBox(lat_min=5, lat_max=40, lon_min=100, lon_max=65)

    def test_india_domain(self):
        bbox = GeoBoundingBox.india_domain()
        assert bbox.lat_min == 5.0 and bbox.lat_max == 40.0

    def test_contains_inside(self):
        bbox   = GeoBoundingBox.india_domain()
        inside = GeoCoordinate(latitude=22.5, longitude=82.0)
        assert bbox.contains(inside)

    def test_contains_outside(self):
        bbox    = GeoBoundingBox.india_domain()
        outside = GeoCoordinate(latitude=0.0, longitude=0.0)
        assert not bbox.contains(outside)

    def test_cds_area(self):
        bbox = GeoBoundingBox.india_domain()
        assert bbox.as_cds_area() == [40.0, 65.0, 5.0, 100.0]


class TestGridMetadata:
    def test_valid_grid(self):
        bbox = GeoBoundingBox.india_domain()
        grid = GridMetadata(
            source="ERA5", variable="t2m", units="K",
            bbox=bbox, n_lat=140, n_lon=140,
            lat_resolution_deg=0.25, lon_resolution_deg=0.25,
            valid_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        assert grid.source == "ERA5"

    def test_naive_valid_time_becomes_utc(self):
        bbox = GeoBoundingBox.india_domain()
        grid = GridMetadata(
            source="INSAT3D", variable="TIR1",
            bbox=bbox, n_lat=100, n_lon=100,
            lat_resolution_deg=0.04, lon_resolution_deg=0.04,
            valid_time=datetime(2024, 6, 1, 6, 0),
        )
        assert grid.valid_time.tzinfo == timezone.utc


class TestDatasetInventoryRecord:
    def test_basic_creation(self):
        rec = DatasetInventoryRecord(
            record_id="abc-123",
            source="ERA5-CDS",
            dataset_type="nwp",
            file_path="/data/era5.nc",
            status=IngestionStatus.VALIDATED,
        )
        assert rec.status == IngestionStatus.VALIDATED
        assert rec.variables == []

    def test_default_status_is_pending(self):
        rec = DatasetInventoryRecord(
            record_id="xyz",
            source="MOSDAC",
            dataset_type="satellite",
            file_path="/tmp/file.h5",
        )
        assert rec.status == IngestionStatus.PENDING
