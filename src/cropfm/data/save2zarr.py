import zarr
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json

def create_zarr_dataset(store_path: str, total_samples: int = 1000000):
    """
    Create zarr dataset structure for CropFM European crop monitoring data.
    
    Args:
        store_path: Path where the zarr dataset will be stored
        total_samples: Total number of samples to allocate space for
        
    Returns:
        zarr.Group: Root zarr group with complete dataset structure
    """
    
    # Create root group
    root = zarr.open_group(store_path, mode='w')
    
    # 1. Metadata group
    metadata_group = root.create_group('metadata')
    
    # Sample information - using structured array
    sample_info_dtype = np.dtype([
        ('sample_id', 'U20'),
        ('grid_id', np.int32),
        ('country', 'U50'),
        ('continent', 'U20'),
        ('classification', np.int32),
        ('tempCropArea', np.float64),
        ('cellArea', np.float64),
        ('longitude', np.float64),
        ('latitude', np.float64)
    ])
    
    sample_info = metadata_group.create_dataset(
        'sample_info',
        shape=(total_samples,),
        dtype=sample_info_dtype,
        chunks=(10000,)
    )
    
    # Spatial index
    spatial_index = metadata_group.create_dataset(
        'spatial_index',
        shape=(total_samples, 2),
        dtype=np.float64,
        chunks=(10000, 2)
    )
    
    # 2. Static modalities group
    static_group = root.create_group('static_modalities')
    
    # Soil data
    soil_group = static_group.create_group('soil')
    soil_variables = ['clay', 'nitrogen', 'phh2o', 'soc']
    for var in soil_variables:
        soil_group.create_dataset(
            var,
            shape=(total_samples, 3),  # 3 depth layers
            dtype=np.float32,
            chunks=(10000, 3),
            fill_value=np.nan
        )
    
    # Elevation data
    elevation_group = static_group.create_group('elevation')
    elevation_group.create_dataset(
        'elevation',
        shape=(total_samples,),
        dtype=np.float32,
        chunks=(10000,),
        fill_value=np.nan
    )
    elevation_group.create_dataset(
        'slope',
        shape=(total_samples,),
        dtype=np.float32,
        chunks=(10000,),
        fill_value=np.nan
    )
    
    # 3. Temporal modalities group
    temporal_group = root.create_group('temporal_modalities')
    
    # Temporal data configuration
    temporal_configs = {
        'agera5': {'max_time': 365, 'n_variables': 9},  # Daily data, max 365 days
        'sentinel1': {'max_time': 100, 'n_variables': 3},  # Irregular, estimated max 100 observations
        'sentinel2': {'max_time': 100, 'n_variables': 15}, # Irregular, estimated max 100 observations
        'fapar': {'max_time': 50, 'n_variables': 2}        # Irregular, estimated max 50 observations
    }
    
    for modality, config in temporal_configs.items():
        mod_group = temporal_group.create_group(modality)
        
        # Data array
        mod_group.create_dataset(
            'data',
            shape=(total_samples, config['max_time'], config['n_variables']),
            dtype=np.float32,
            chunks=(1000, config['max_time'], config['n_variables']),
            fill_value=np.nan
        )
        
        # Timestamps (stored as days relative to reference date)
        mod_group.create_dataset(
            'timestamps',
            shape=(total_samples, config['max_time']),
            dtype=np.int32,  # Days
            chunks=(1000, config['max_time']),
            fill_value=-1
        )
        
        # Valid data mask
        mod_group.create_dataset(
            'valid_mask',
            shape=(total_samples, config['max_time']),
            dtype=bool,
            chunks=(1000, config['max_time']),
            fill_value=False
        )
    
    # 4. Indices group
    indices_group = root.create_group('indices')
    
    # Add attribute information
    root.attrs['created_date'] = pd.Timestamp.now().isoformat()
    root.attrs['total_samples'] = total_samples
    root.attrs['reference_date'] = '2020-01-01'  # Reference date for timestamps
    root.attrs['description'] = 'CropFM European crop monitoring dataset'
    
    return root

def add_sample_data(zarr_root, sample_idx: int, sample_data: Dict[str, Any]):
    """
    Add data for a single sample to the zarr dataset.
    
    Args:
        zarr_root: Root zarr group
        sample_idx: Index of the sample to add
        sample_data: Dictionary containing all modality data for the sample
        
    Returns:
        None
    """
    
    # 1. Add metadata
    metadata = sample_data['metadata']
    zarr_root['metadata/sample_info'][sample_idx] = (
        metadata['sample_id'],
        metadata['grid_id'],
        metadata['country'],
        metadata['continent'],
        metadata['classification'],
        metadata['tempCropArea'],
        metadata['cellArea'],
        metadata['coordinates'][0],  # longitude
        metadata['coordinates'][1]   # latitude
    )
    
    zarr_root['metadata/spatial_index'][sample_idx] = metadata['coordinates']
    
    # 2. Add static data
    if 'soil' in sample_data:
        soil_data = sample_data['soil']
        for var in ['clay', 'nitrogen', 'phh2o', 'soc']:
            if var in soil_data:
                zarr_root[f'static_modalities/soil/{var}'][sample_idx] = soil_data[var]
    
    if 'elevation' in sample_data:
        elev_data = sample_data['elevation']
        zarr_root['static_modalities/elevation/elevation'][sample_idx] = elev_data['elevation']
        zarr_root['static_modalities/elevation/slope'][sample_idx] = elev_data['slope']
    
    # 3. Add temporal data
    reference_date = pd.Timestamp('2020-01-01')
    
    for modality in ['agera5', 'sentinel1', 'sentinel2', 'fapar']:
        if modality in sample_data:
            mod_data = sample_data[modality]
            
            # Convert timestamps to days
            timestamps = pd.to_datetime(mod_data['dates'])
            days_since_ref = (timestamps - reference_date).days.values
            
            n_obs = len(mod_data['data'])
            
            # Store data
            zarr_root[f'temporal_modalities/{modality}/data'][sample_idx, :n_obs] = mod_data['data']
            zarr_root[f'temporal_modalities/{modality}/timestamps'][sample_idx, :n_obs] = days_since_ref
            zarr_root[f'temporal_modalities/{modality}/valid_mask'][sample_idx, :n_obs] = True