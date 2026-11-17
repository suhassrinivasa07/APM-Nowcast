# Data Management for APM-Nowcast

This directory stores datasets for Thunderstorm Initiation (TI) and Lightning Nowcasting over India.

## Subdirectories
- `raw/`: Raw observations including INSAT-3D/3DR satellite imagery (TIR, VIS, WV channels), Doppler Weather Radar (DWR) reflectivity/velocity volumes, IITM / Lightning Location Network data, and numerical weather model (NWP/GFS/NCMRWF) analyses.
- `processed/`: Collocated spatiotemporal grids, normalized tensor datasets, and processed NetCDF / Zarr stores.

*Note: Large raw datasets and processed tensors are excluded from git version control via `.gitignore`.*
