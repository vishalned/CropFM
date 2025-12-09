# Zarr Data Tree Structure for CropFM Dataset

This document describes the zarr data tree structure for storing the CropFM European crop monitoring dataset. There are two versions: raw (from data download) and cleaned/aggregated (after processing).

---

## 1. Raw Zarr Structure (from `save2zarr.py`)

This is the structure created during initial data download from Google Earth Engine.

```
raw_dataset.zarr/
├── metadata/                          # Sample-level metadata
│   ├── sample_info                    # shape: (n_samples,) - structured array
│   │   # dtype: [('sample_id', 'U20'), ('grid_id', 'i4'), ('country', 'U50'), 
│   │   #         ('continent', 'U20'), ('classification', 'i4'), 
│   │   #         ('tempCropArea', 'f8'), ('cellArea', 'f8'),
│   │   #         ('longitude', 'f8'), ('latitude', 'f8')]
│   │   # chunks: (10000,)
│   │   # fill_value: None
│   │
│   └── coordinates                    # shape: (n_samples, 2) - [longitude, latitude]
│       # dtype: float64
│       # chunks: (10000, 2)
│       # fill_value: None
│
├── static_modalities/                 # Static data (time-invariant)
│   ├── soil/                         
│   │   ├── clay                      # shape: (n_samples, 3) - 3 depth layers
│   │   │   # dtype: float32
│   │   │   # chunks: (10000, 3)
│   │   │   # fill_value: np.nan
│   │   ├── nitrogen                  # shape: (n_samples, 3)
│   │   │   # dtype: float32, chunks: (10000, 3), fill_value: np.nan
│   │   ├── phh2o                     # shape: (n_samples, 3)
│   │   │   # dtype: float32, chunks: (10000, 3), fill_value: np.nan
│   │   └── soc                       # shape: (n_samples, 3)
│   │       # dtype: float32, chunks: (10000, 3), fill_value: np.nan
│   │
│   └── elevation/
│       ├── elevation                 # shape: (n_samples,)
│       │   # dtype: float32, chunks: (10000,), fill_value: np.nan
│       └── slope                     # shape: (n_samples,)
│           # dtype: float32, chunks: (10000,), fill_value: np.nan
│
└── temporal_modalities/              # Time series data (irregular observations)
    ├── agera5/
    │   ├── data                      # shape: (n_samples, 365, 8)
    │   │   # dtype: float32
    │   │   # chunks: (1000, 365, 8)
    │   │   # fill_value: np.nan (trailing NaNs indicate unused time slots)
    │   ├── timestamps               # shape: (n_samples, 365)
    │   │   # dtype: str (date strings like "2021-01-15")
    │   │   # chunks: (1000, 365)
    │   │   # fill_value: 'NA' (for unused time slots)
    │   ├── valid_mask              # shape: (n_samples, 365)
    │   │   # dtype: bool
    │   │   # chunks: (1000, 365)
    │   │   # fill_value: False (for unused time slots)
    │   └── variable_names          # shape: (8,)
    │       # dtype: str, chunks: (8,), fill_value: None
    │
    ├── sentinel1/
    │   ├── data                      # shape: (n_samples, 200, 3)
    │   │   # dtype: float32, chunks: (1000, 200, 3), fill_value: np.nan
    │   ├── timestamps               # shape: (n_samples, 200)
    │   │   # dtype: str, chunks: (1000, 200), fill_value: 'NA'
    │   ├── valid_mask              # shape: (n_samples, 200)
    │   │   # dtype: bool, chunks: (1000, 200), fill_value: False
    │   └── variable_names          # shape: (3,)
    │
    ├── sentinel2/
    │   ├── data                      # shape: (n_samples, 100, 15)
    │   │   # dtype: float32, chunks: (1000, 100, 15), fill_value: np.nan
    │   ├── timestamps               # shape: (n_samples, 100)
    │   │   # dtype: str, chunks: (1000, 100), fill_value: 'NA'
    │   ├── valid_mask              # shape: (n_samples, 100)
    │   │   # dtype: bool, chunks: (1000, 100), fill_value: False
    │   └── variable_names          # shape: (15,)
    │
    └── fapar/
        ├── data                      # shape: (n_samples, 110, 2)
        │   # dtype: float32, chunks: (1000, 110, 2), fill_value: np.nan
        ├── timestamps               # shape: (n_samples, 110)
        │   # dtype: str, chunks: (1000, 110), fill_value: 'NA'
        ├── valid_mask              # shape: (n_samples, 110)
        │   # dtype: bool, chunks: (1000, 110), fill_value: False
        └── variable_names          # shape: (2,)
```

### Raw Structure Notes:
- **fill_value**: Used for pre-allocated arrays. Trailing positions are filled with `np.nan` (data), `'NA'` (timestamps), or `False` (valid_mask)
- **valid_mask**: Indicates which time slots have actual data (True) vs padding (False)
- **Timestamps**: Date strings in format "YYYY-MM-DD"
- **Variable names**: Descriptive names for each variable/band in the data array

---

## 2. Cleaned/Aggregated Zarr Structure (from `clean_data.py`)

This is the structure after cleaning and weekly aggregation. Trailing NaNs are removed, data is aggregated to weekly resolution, and encoded features are added.

```
cleaned_dataset.zarr/
├── metadata/                          # Sample-level metadata
│   ├── sample_info                    # shape: (n_samples,) - structured array
│   │   # Same as raw structure
│   │   # chunks: (10000,)
│   │
│   ├── coordinates                    # shape: (n_samples, 2) - [longitude, latitude]
│   │   # dtype: float64, chunks: (10000, 2)
│   │
│   └── encoded_coordinates            # shape: (n_samples, 4) - NEW
│       # dtype: float32, chunks: (10000, 4)
│       # Cyclic encoding: [lon_sin, lon_cos, lat_sin, lat_cos]
│       # Encoding: sin(2π*coord/360), cos(2π*coord/360)
│       # fill_value: None (no padding needed)
│
├── static_modalities/                 # Static data (copied as-is)
│   ├── soil/                         
│   │   ├── clay                      # shape: (n_samples, 3)
│   │   │   # dtype: float32, chunks: (10000, 3) or (10000,) if 1D
│   │   ├── nitrogen                  # Same structure
│   │   ├── phh2o                     # Same structure
│   │   └── soc                       # Same structure
│   │
│   └── elevation/
│       ├── elevation                 # shape: (n_samples,)
│       │   # dtype: float32, chunks: (10000,)
│       └── slope                     # shape: (n_samples,)
│           # dtype: float32, chunks: (10000,)
│
└── temporal_modalities/              # Time series data (weekly aggregated)
    ├── agera5/
    │   ├── data                      # shape: (n_samples, 52, 8)
    │   │   # dtype: float32
    │   │   # chunks: (1000, 52, 8)
    │   │   # fill_value: None (pre-filled with np.nan, no fill_value set)
    │   │   # Weekly aggregated using mean/median/max
    │   │   # Trailing NaNs removed, interpolated no_data values (0s)
    │   │
    │   ├── timestamps               # shape: (n_samples, 52)
    │   │   # dtype: str (week keys like "2021-W15")
    │   │   # chunks: (1000, 52)
    │   │   # fill_value: None (pre-filled with 'NA')
    │   │
    │   ├── valid_mask              # shape: (n_samples, 52)
    │   │   # dtype: bool
    │   │   # chunks: (1000, 52)
    │   │   # fill_value: None (pre-filled with False)
    │   │
    │   ├── variable_names          # shape: (8,)
    │   │   # dtype: str, chunks: (8,)
    │   │
    │   └── week_encoding            # shape: (n_samples, 52, 2) - NEW
    │       # dtype: float32, chunks: (1000, 52, 2)
    │       # Cyclic encoding: [week_sin, week_cos]
    │       # Encoding: sin(2π*week_num/52), cos(2π*week_num/52)
    │       # Week number extracted from ISO week format "YYYY-W##"
    │       # fill_value: None (zeros for invalid positions)
    │
    ├── sentinel1/
    │   ├── data                      # shape: (n_samples, 52, 3)
    │   │   # dtype: float32, chunks: (1000, 52, 3)
    │   ├── timestamps               # shape: (n_samples, 52)
    │   │   # dtype: str, chunks: (1000, 52)
    │   ├── valid_mask              # shape: (n_samples, 52)
    │   │   # dtype: bool, chunks: (1000, 52)
    │   ├── variable_names          # shape: (3,)
    │   └── week_encoding            # shape: (n_samples, 52, 2)
    │       # Same encoding as agera5
    │
    ├── sentinel2/
    │   ├── data                      # shape: (n_samples, 52, 15)
    │   │   # dtype: float32, chunks: (1000, 52, 15)
    │   ├── timestamps               # shape: (n_samples, 52)
    │   │   # dtype: str, chunks: (1000, 52)
    │   ├── valid_mask              # shape: (n_samples, 52)
    │   │   # dtype: bool, chunks: (1000, 52)
    │   ├── variable_names          # shape: (15,)
    │   └── week_encoding            # shape: (n_samples, 52, 2)
    │
    └── fapar/
        ├── data                      # shape: (n_samples, 52, 2)
        │   # dtype: float32, chunks: (1000, 52, 2)
        ├── timestamps               # shape: (n_samples, 52)
        │   # dtype: str, chunks: (1000, 52)
        ├── valid_mask              # shape: (n_samples, 52)
        │   # dtype: bool, chunks: (1000, 52)
        ├── variable_names          # shape: (2,)
        └── week_encoding            # shape: (n_samples, 52, 2)
```

### Cleaned Structure Notes:
- **No fill_value**: Arrays are pre-filled with values (NaN, 'NA', False) but no fill_value is set
- **Weekly aggregation**: All temporal data aggregated to weekly resolution (max 52 weeks)
- **Trailing NaNs removed**: Only valid data positions remain (no padding)
- **No_data interpolation**: Zero values from GEE API are interpolated
- **Encoded features**: 
  - `encoded_coordinates`: Cyclic encoding of lat/lon for spatial features
  - `week_encoding`: Cyclic encoding of week numbers for temporal features
- **Timestamps**: ISO week format "YYYY-W##" (e.g., "2021-W15")
- **Uniform shape**: All samples have the same max_weeks dimension (52)

---

## Encoding Details

### Coordinate Encoding (`metadata/encoded_coordinates`)
- **Purpose**: Cyclic encoding of geographic coordinates for neural network input
- **Method**: 
  - `lon_sin = sin(2π × longitude / 360)`
  - `lon_cos = cos(2π × longitude / 360)`
  - `lat_sin = sin(2π × latitude / 360)`
  - `lat_cos = cos(2π × latitude / 360)`
- **Shape**: `(n_samples, 4)`
- **Range**: [-1, 1] for sin/cos values

### Week Encoding (`temporal_modalities/{modality}/week_encoding`)
- **Purpose**: Cyclic encoding of week numbers for temporal features
- **Method**:
  - Extract week number from ISO week format "YYYY-W##" (1-52)
  - `week_sin = sin(2π × week_num / 52)`
  - `week_cos = cos(2π × week_num / 52)`
- **Shape**: `(n_samples, max_weeks, 2)`
- **Range**: [-1, 1] for sin/cos values
- **Invalid positions**: Set to 0.0 (where valid_mask is False)

---

## Root Attributes

Both structures include root-level attributes:
- `total_samples`: Number of samples in the dataset
- `created_date`: ISO timestamp of creation
- `description`: Dataset description
- Additional metadata as needed

---

## Storage Considerations

- **Raw dataset**: Uses zarr's Blosc compression, estimated 3-6 GB compressed
- **Cleaned dataset**: Similar compression, slightly smaller due to aggregation
- **Chunking**: Optimized for batch access patterns (1000-10000 samples per chunk)
- **Access patterns**: 
  - Raw: Sequential sample access with variable-length time series
  - Cleaned: Uniform shape enables efficient batch processing

---
