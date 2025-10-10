# Zarr Data Tree Structure for CropFM Dataset

This document describes the recommended zarr data tree structure for storing the CropFM European crop monitoring dataset with 1 million samples and 6 modalities.

## Main Zarr Data Tree Structure

```
dataset.zarr/
├── metadata/                          # Sample-level metadata
│   ├── sample_info                    # shape: (1000000,) - structured array
│   │   # Contains: sample_id, grid_id, country, continent, classification
│   │   #          tempCropArea, cellArea, coordinates (lon, lat)
│   └── spatial_index                  # shape: (1000000, 2) - spatial index [lon, lat]
│
├── static_modalities/                 # Static data (time-invariant)
│   ├── soil/                         
│   │   ├── clay                      # shape: (1000000, 3) - 3 depth layers
│   │   ├── nitrogen                  # shape: (1000000, 3)
│   │   ├── phh2o                     # shape: (1000000, 3)
│   │   └── soc                       # shape: (1000000, 3)
│   └── elevation/
│       ├── elevation                 # shape: (1000000,)
│       └── slope                     # shape: (1000000,)
│
├── temporal_modalities/              # Time series data
│   ├── agera5/
│   │   ├── data                      # shape: (1000000, max_days, 9)
│   │   ├── timestamps               # shape: (1000000, max_days) - dates
│   │   ├── valid_mask              # shape: (1000000, max_days) - valid data mask
│   │   └── variable_names          # shape: (9,) - variable names
│   │
│   ├── sentinel1/
│   │   ├── data                      # shape: (1000000, max_observations, 3)
│   │   ├── timestamps               # shape: (1000000, max_observations)
│   │   ├── valid_mask              # shape: (1000000, max_observations)
│   │   └── band_names              # shape: (3,) - band names
│   │
│   ├── sentinel2/
│   │   ├── data                      # shape: (1000000, max_observations, 15)
│   │   ├── timestamps               # shape: (1000000, max_observations)
│   │   ├── valid_mask              # shape: (1000000, max_observations)
│   │   └── band_names              # shape: (15,) - band names
│   │
│   └── fapar/
│       ├── data                      # shape: (1000000, max_observations, 2)
│       ├── timestamps               # shape: (1000000, max_observations)
│       ├── valid_mask              # shape: (1000000, max_observations)
│       └── variable_names          # shape: (2,) - variable names
│
└── indices/                          # Indices and lookup tables
    ├── country_index                 # Country to sample ID mapping
    ├── grid_index                   # Grid to sample ID mapping
    └── temporal_coverage           # Temporal coverage statistics per sample
```

## Data Organization Details

### Metadata Group
- **sample_info**: Structured array containing all sample-level metadata including geographic coordinates, administrative boundaries, and land use information
- **spatial_index**: Optimized spatial index for geographic queries

### Static Modalities Group
- **soil**: Soil properties at 3 depth layers (0-5cm, 5-15cm, 15-30cm)
  - Clay content, nitrogen, pH, and soil organic carbon
- **elevation**: Topographic information including elevation and slope

### Temporal Modalities Group
- **agera5**: Daily meteorological data with 9 variables
- **sentinel1**: SAR data with VV, VH polarizations and incidence angle
- **sentinel2**: Multispectral optical data with 15 bands
- **fapar**: Vegetation indices (FAPAR and LAI)

Each temporal modality includes:
- **data**: The actual observations
- **timestamps**: Temporal information for each observation
- **valid_mask**: Boolean mask indicating valid observations
- **variable/band_names**: Descriptive names for data dimensions

### Indices Group
- **country_index**: Efficient lookup for samples by country
- **grid_index**: Spatial grid-based indexing
- **temporal_coverage**: Statistics on temporal data availability per sample

## Storage Considerations

- Uses zarr's Blosc compression for efficient storage
- Chunked arrays optimized for both spatial and temporal access patterns
- Pre-allocated arrays with valid_mask to handle irregular time series
- Estimated compressed storage: 3-6 GB for the complete dataset
