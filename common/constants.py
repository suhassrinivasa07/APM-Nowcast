"""
Domain constants for APM-Nowcast.
"""
from common.schemas import GeoBoundingBox

# ------------------------------------------------------------------
# Indian domain defaults
# ------------------------------------------------------------------
INDIA_BBOX = GeoBoundingBox(lat_min=5.0, lat_max=40.0, lon_min=65.0, lon_max=100.0)

# Standard spatial resolutions (degrees)
INSAT3D_TIR_RESOLUTION_DEG = 0.04     # ~4 km
DWR_NATIVE_RESOLUTION_DEG   = 0.01    # ~1 km (after mosaic)
ERA5_RESOLUTION_DEG         = 0.25
GPM_IMERG_RESOLUTION_DEG    = 0.1

# Standard time-step in minutes
INSAT3D_CADENCE_MIN    = 30
DWR_VOLUME_CADENCE_MIN = 10
GPM_IMERG_CADENCE_MIN  = 30

# File format magic bytes (first 4 bytes)
MAGIC_HDF5    = b"\x89HDF"
MAGIC_NETCDF  = b"CDF\x01"
MAGIC_NETCDF4 = b"\x89HDF"   # NC4 uses HDF5 container
MAGIC_GRIB2   = b"GRIB"
