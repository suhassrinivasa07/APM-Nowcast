"""
Shared Pydantic schemas used across data_ingestion, backend, and research modules.
All timestamps are enforced as UTC-aware datetimes.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class UTCTimestamp(BaseModel):
    """A UTC-normalised datetime wrapper."""

    dt: datetime = Field(..., description="UTC-aware datetime")

    @field_validator("dt", mode="before")
    @classmethod
    def ensure_utc(cls, v: object) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse datetime from {v!r}")

    @property
    def isoformat(self) -> str:
        return self.dt.isoformat()


class GeoCoordinate(BaseModel):
    """A WGS-84 geographic point."""

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Decimal degrees North")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Decimal degrees East")

    def __str__(self) -> str:
        return f"({self.latitude:.4f}°N, {self.longitude:.4f}°E)"


class GeoBoundingBox(BaseModel):
    """
    Rectangular lat/lon bounding box in WGS-84.
    Default covers the Indian sub-continental domain used in APM-Nowcast.
    """

    lat_min: float = Field(..., ge=-90.0, le=90.0)
    lat_max: float = Field(..., ge=-90.0, le=90.0)
    lon_min: float = Field(..., ge=-180.0, le=180.0)
    lon_max: float = Field(..., ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "GeoBoundingBox":
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be strictly less than lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be strictly less than lon_max")
        return self

    @classmethod
    def india_domain(cls) -> "GeoBoundingBox":
        """Standard Indian sub-continental analysis domain."""
        return cls(lat_min=5.0, lat_max=40.0, lon_min=65.0, lon_max=100.0)

    def contains(self, coord: GeoCoordinate) -> bool:
        return (
            self.lat_min <= coord.latitude <= self.lat_max
            and self.lon_min <= coord.longitude <= self.lon_max
        )

    def as_cds_area(self) -> list[float]:
        """Return [N, W, S, E] for Copernicus CDS API calls."""
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]


class GridMetadata(BaseModel):
    """Metadata describing a regular lat/lon grid tile."""

    source: str = Field(..., description="Data source identifier, e.g. INSAT3D, ERA5")
    variable: str = Field(..., description="Geophysical variable name")
    units: str = Field(default="", description="SI or CF-compliant units string")
    bbox: GeoBoundingBox
    n_lat: int = Field(..., gt=0)
    n_lon: int = Field(..., gt=0)
    lat_resolution_deg: float = Field(..., gt=0.0)
    lon_resolution_deg: float = Field(..., gt=0.0)
    valid_time: datetime
    level_hpa: Optional[float] = Field(default=None, description="Pressure level in hPa, None for surface")
    crs: str = Field(default="EPSG:4326", description="Coordinate reference system")

    @field_validator("valid_time", mode="before")
    @classmethod
    def ensure_utc(cls, v: object) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse datetime from {v!r}")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class IngestionStatus(str, enum.Enum):
    PENDING    = "pending"
    DOWNLOADED = "downloaded"
    VALIDATED  = "validated"
    FAILED     = "failed"
    LOCAL      = "local"      # file provided manually, not downloaded


class DatasetInventoryRecord(BaseModel):
    """
    One entry in the dataset inventory (persisted as JSON-lines or SQLite).
    """

    record_id: str = Field(..., description="Unique record identifier (UUID or hash)")
    source: str = Field(..., description="Source system identifier")
    dataset_type: str = Field(..., description="satellite | radar | lightning | nwp | gpm")
    file_path: str = Field(..., description="Absolute path to the ingested file")
    file_size_bytes: int = Field(default=0, ge=0)
    checksum_md5: Optional[str] = Field(default=None)
    bbox: Optional[GeoBoundingBox] = None
    valid_time_start: Optional[datetime] = None
    valid_time_end: Optional[datetime] = None
    variables: list[str] = Field(default_factory=list)
    status: IngestionStatus = IngestionStatus.PENDING
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
