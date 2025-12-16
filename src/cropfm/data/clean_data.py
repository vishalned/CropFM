'''
Clean and aggregate temporal data from zarr files.

This script:
1. Removes trailing NaN values (padding from zarr pre-allocation)
2. Aggregates temporal modalities to weekly resolution
3. Interpolates no_data values (obtained from GEE api) in the aggregated data
4. Saves cleaned and aggregated data to a new zarr file



possible issue, need to seperate nan checks for vars? - for fapar i.e 2 vars
check sentinel2 -- why are there so many nans?
'''

import zarr
import numpy as np
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime

NO_DATA_VALUE = {
    'sentinel2': 0,
}
MAX_TIMESTEPS = {
    'weekly': 52,
    'daily': 365,
    'monthly': 12,
}

def encode_coordinates(coordinates):
    """
    Encode latitude and longitude using cyclic encoding (sin/cos).
    
    Args:
        coordinates: (n_samples, 2) array with [longitude, latitude]
    
    Returns:
        encoded_coords: (n_samples, 4) array with [lon_sin, lon_cos, lat_sin, lat_cos]
    """
    lon = coordinates[:, 0]  # longitude
    lat = coordinates[:, 1]  # latitude
    
    # Cyclic encoding: sin(2π*value/360) and cos(2π*value/360)
    lon_sin = np.sin(2 * np.pi * lon / 360)
    lon_cos = np.cos(2 * np.pi * lon / 360)
    lat_sin = np.sin(2 * np.pi * lat / 360)
    lat_cos = np.cos(2 * np.pi * lat / 360)
    
    encoded = np.column_stack([lon_sin, lon_cos, lat_sin, lat_cos])
    return encoded.astype(np.float32)

def encode_weeks(timestamps, max_weeks=52):
    """
    Encode week numbers using cyclic encoding (sin/cos).
    Extracts week number from ISO week format "YYYY-W##" or datetime.
    
    Args:
        timestamps: (n_samples, max_weeks) array of week keys like "2021-W15" or datetime
        max_weeks: Maximum number of weeks in a year (52 or 53)
    
    Returns:
        week_encoding: (n_samples, max_weeks, 2) array with [week_sin, week_cos]
    """
    n_samples, max_weeks_dim = timestamps.shape
    week_encoding = np.zeros((n_samples, max_weeks_dim, 2), dtype=np.float32)
    
    for i in range(n_samples):
        for j in range(max_weeks_dim):
            ts_str = timestamps[i, j]
            
            # Skip invalid timestamps
            if ts_str == 'NA' or ts_str == '' or pd.isna(ts_str):
                continue
            
            # Extract week number from "YYYY-W##" format
            try:
                if isinstance(ts_str, str) and 'W' in ts_str:
                    # Format: "2021-W15"
                    week_num = int(ts_str.split('-W')[1])
                else:
                    # Try parsing as datetime and get ISO week
                    dt = pd.to_datetime(ts_str)
                    week_num = dt.isocalendar()[1]  # ISO week number (1-53)
                
                # Cyclic encoding: sin(2π*week/52) and cos(2π*week/52)
                week_sin = np.sin(2 * np.pi * week_num / max_weeks)
                week_cos = np.cos(2 * np.pi * week_num / max_weeks)
                
                week_encoding[i, j, 0] = week_sin
                week_encoding[i, j, 1] = week_cos
            except:
                # If parsing fails, leave as zeros
                pass
    
    return week_encoding

def remove_trailing_nans(data, timestamps, valid_mask):
    """
    Remove trailing NaNs from data, timestamps, and valid_mask.
    Uses the valid_mask to determine which indices to keep.
    
    Args:
        data: (n_samples, max_time, n_vars) array
        timestamps: (n_samples, max_time) array
        valid_mask: (n_samples, max_time) array
    
    Returns:
        trimmed_data, trimmed_timestamps, trimmed_valid_mask
    """
    n_samples, max_time, n_vars = data.shape
    trimmed_data_list = []
    trimmed_timestamps_list = []
    trimmed_valid_mask_list = []
    
    for i in range(n_samples):
        sample_data = data[i]  # (max_time, n_vars)
        sample_timestamps = timestamps[i]  # (max_time,)
        sample_valid_mask = valid_mask[i]  # (max_time,)
        
        # Use valid_mask to find valid indices
        valid_indices = np.where(sample_valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No valid data, return empty arrays
            trimmed_data_list.append(np.array([]).reshape(0, n_vars))
            trimmed_timestamps_list.append(np.array([]))
            trimmed_valid_mask_list.append(np.array([]))
        else:
            # Use only the indices where valid_mask is True
            trimmed_data_list.append(sample_data[valid_indices])
            trimmed_timestamps_list.append(sample_timestamps[valid_indices])
            trimmed_valid_mask_list.append(sample_valid_mask[valid_indices])
    
    return trimmed_data_list, trimmed_timestamps_list, trimmed_valid_mask_list


def get_week_key(dt):
    """Get year-week key for grouping"""
    if dt is None:
        return None
    return f"{dt.year}-W{dt.isocalendar()[1]:02d}"

def aggregate_to_weekly(data, timestamps, modality_name, aggregation_method='mean'):
    """
    Aggregate to weekly, handle year boundaries/ISO week 53, then clip to 52 weeks.
    Returns (52, n_vars) data, 52 week keys, 52-length valid mask.
    """
    if len(data) == 0:
        n_vars = data.shape[1] if len(data.shape) > 1 else 0
        return (np.full((52, n_vars), np.nan, dtype=np.float32),
                np.array(['NA'] * 52, dtype='U20'),
                np.zeros(52, dtype=bool))

    ts = pd.to_datetime(timestamps, errors='coerce')
    valid_ts = ~ts.isna()
    if not np.any(valid_ts):
        n_vars = data.shape[1] if len(data.shape) > 1 else 0
        return (np.full((52, n_vars), np.nan, dtype=np.float32),
                np.array(['NA'] * 52, dtype='U20'),
                np.zeros(52, dtype=bool))

    d = data[valid_ts]
    if modality_name == 'sentinel2':
        d = np.where(d == NO_DATA_VALUE.get('sentinel2', 0), np.nan, d)

    df = pd.DataFrame(d, index=ts[valid_ts])

    if aggregation_method == 'median':
        weekly = df.resample('W').median()
    elif aggregation_method == 'max':
        weekly = df.resample('W').max()
    else:
        weekly = df.resample('W').mean()

    # Reindex across full week span
    full_idx = pd.date_range(weekly.index.min(), weekly.index.max(), freq='W')
    weekly = weekly.reindex(full_idx)

    # Build keys and valid mask
    week_keys_full = np.array([f"{dt.year}-W{dt.isocalendar()[1]:02d}" for dt in weekly.index])
    weekly_valid_full = ~weekly.isna().all(axis=1)

    # Clip/pad to exactly 52 weeks: keep earliest 52
    n_vars = weekly.shape[1]
    out_data = np.full((52, n_vars), np.nan, dtype=np.float32)
    out_keys = np.array(['NA'] * 52, dtype='U20')
    out_valid = np.zeros(52, dtype=bool)

    use_len = min(len(weekly), 52)
    out_data[:use_len] = weekly.values[:use_len]
    out_keys[:use_len] = week_keys_full[:use_len]
    out_valid[:use_len] = weekly_valid_full[:use_len]

    return out_data, out_keys, out_valid

def aggregate_to_daily(data, timestamps, valid_mask, aggregation_method='mean'):
    raise NotImplementedError("Not implemented yet")

def aggregate_to_monthly(data, timestamps, valid_mask, aggregation_method='mean'):
    raise NotImplementedError("Not implemented yet")


def main(args):
    root = zarr.open(args.input_zarr_path, mode='r')
    total_samples = root.attrs.get('total_samples', 0)
    print(f"Total samples: {total_samples:,}")

    # Create output zarr
    root_out = zarr.open_group(args.output_zarr_path, mode='w')

    # Copy metadata
    print("Copying metadata...")
    if 'metadata' in root:
        metadata_group = root_out.create_group('metadata')
        
        # Copy sample_info
        if 'sample_info' in root['metadata']:
            sample_info = root['metadata']['sample_info'][:]
            metadata_group.create_array('sample_info', data=sample_info, chunks=(10000,))
        
        # Copy coordinates
        if 'coordinates' in root['metadata']:
            coordinates = root['metadata']['coordinates'][:]
            metadata_group.create_array('coordinates', data=coordinates, chunks=(10000, 2))
            
            # Encode coordinates (sin/cos encoding)
            print("  Encoding coordinates...")
            encoded_coords = encode_coordinates(coordinates)
            metadata_group.create_array('encoded_coordinates', data=encoded_coords, chunks=(10000, 4))
            print(f"  ✓ Encoded coordinates: {encoded_coords.shape}")
    
    # Copy root attributes
    root_out.attrs.update(root.attrs)

    # copy static modalities
    static_group = root_out.create_group('static_modalities')
    for modality in root['static_modalities'].keys():
        print(f"Modality: {modality}")
        root_out['static_modalities'].create_group(modality)
        for var in root['static_modalities'][modality].keys():
            print(f"    Variable: {var}")
            var_data = root['static_modalities'][modality][var][:]
            chunks = (10000, var_data.shape[1]) if len(var_data.shape) > 1 else (10000,)
            root_out['static_modalities'][modality].create_array(
                var, 
                data=var_data, 
                chunks = chunks
            )

    # temporal modalities
    temporal_group = root_out.create_group('temporal_modalities')
    for modality_name in root['temporal_modalities'].keys():
    # for modality_name in ['sentinel2']:
        root_out['temporal_modalities'].create_group(modality_name)
        variable_names = root['temporal_modalities'][modality_name]['variable_names'][:]


        data = root['temporal_modalities'][modality_name]['data'][:]  # (n_samples, max_time, n_vars)
        timestamps = root['temporal_modalities'][modality_name]['timestamps'][:]  # (n_samples, max_time)
        valid_mask = root['temporal_modalities'][modality_name]['valid_mask'][:]  # (n_samples, max_time)

        print(f"Modality: {modality_name}")
        print(f"    Data shape: {data.shape}")
        print(f"    Timestamps shape: {timestamps.shape}")
        print(f"    Valid mask shape: {valid_mask.shape}")
        print(f"    Variable names: {variable_names}")
        
        n_samples = data.shape[0]
        n_vars = len(variable_names)
        
        # First, remove trailing NaNs for each sample
        trimmed_data, trimmed_timestamps, trimmed_valid_mask = remove_trailing_nans(
            data, timestamps, valid_mask
        )
        # Use hardcoded max timesteps
        max_timesteps = MAX_TIMESTEPS['weekly'] # hardcoded for now
        
        # Create output arrays with hardcoded max_timesteps
        aggregated_data_out = np.full((n_samples, max_timesteps, n_vars), np.nan, dtype=np.float32)
        aggregated_timestamps_out = np.full((n_samples, max_timesteps), 'NA', dtype='U20')
        aggregated_valid_mask_out = np.zeros((n_samples, max_timesteps), dtype=bool)
        
        # Step 1: Aggregate all samples first
        print(f"  Aggregating {n_samples} samples...")
        for i in range(n_samples):
            sample_data = trimmed_data[i]
            sample_timestamps = trimmed_timestamps[i]
            sample_valid_mask = trimmed_valid_mask[i]
            
            # Aggregate to weekly
            weekly_data, weekly_timestamps, weekly_valid_mask = aggregate_to_weekly(
                sample_data, 
                sample_timestamps,
                modality_name,
                aggregation_method='mean' # hardcoded for now
            )
            
            # Fill in the aggregated data (truncate to max_timesteps if needed)
            n_weeks = len(weekly_timestamps)
            if n_weeks > 0:
                # Truncate to max_timesteps if sample has more weeks
                n_weeks_to_use = min(n_weeks, max_timesteps)
                aggregated_data_out[i, :n_weeks_to_use] = weekly_data[:n_weeks_to_use]
                aggregated_timestamps_out[i, :n_weeks_to_use] = weekly_timestamps[:n_weeks_to_use]
                aggregated_valid_mask_out[i, :n_weeks_to_use] = weekly_valid_mask[:n_weeks_to_use]

        # Step 2 & 3 removed: we keep NaNs for missing/no_data and rely on valid_mask to indicate usable entries
        root_out['temporal_modalities'][modality_name].create_array(
            'data', 
            data=aggregated_data_out, 
            chunks=(10000, max_timesteps, n_vars)
        )
        root_out['temporal_modalities'][modality_name].create_array(
            'timestamps', 
            data=aggregated_timestamps_out, 
            chunks=(10000, max_timesteps)
        )
        root_out['temporal_modalities'][modality_name].create_array(
            'valid_mask', 
            data=aggregated_valid_mask_out, 
            chunks=(10000, max_timesteps)
        )
        root_out['temporal_modalities'][modality_name].create_array(
            'variable_names', 
            data=variable_names, 
            chunks=(n_vars,)
        )
        
        # Encode weeks (sin/cos encoding of week numbers)
        print(f"  Encoding weeks for {modality_name}...")
        week_encoding = encode_weeks(aggregated_timestamps_out, max_weeks=max_timesteps)
        root_out['temporal_modalities'][modality_name].create_array(
            'week_encoding',
            data=week_encoding,
            chunks=(10000, max_timesteps, 2)
        )
        print(f"  ✓ Week encoding shape: {week_encoding.shape}")

    print("Data cleaned and aggregated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_zarr_path', type=str, default='/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged.zarr')
    parser.add_argument('--output_zarr_path', type=str, default='/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged_cleaned.zarr')
    args = parser.parse_args()
    main(args)