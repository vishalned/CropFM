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
    'fapar': 0,
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

def interpolate_no_data_values(data, no_data_value, interpolation_method='linear'):
    """
    Interpolate no_data values (typically 0) in the data.
    
    Args:
        data: (max_time, n_vars) array
        no_data_value: value to replace (e.g., 0)
        interpolation_method: 'linear' or 'cubic'
    
    Returns:
        data with no_data values interpolated
    """
    if len(data) == 0:
        return data
    
    interpolated_data = data.copy()
    n_time, n_vars = data.shape
    
    for var_idx in range(n_vars):
        var_data = data[:, var_idx]
        
        # Find indices where data is valid (not NaN and not no_data_value)
        valid_mask = (~np.isnan(var_data)) & (var_data != no_data_value)
        valid_indices = np.where(valid_mask)[0]
        
        # Find indices that need interpolation
        invalid_mask = (np.isnan(var_data)) | (var_data == no_data_value)
        invalid_indices = np.where(invalid_mask)[0]
        
        if len(invalid_indices) == 0:
            continue
        
        # Create interpolation function
        try:
            if interpolation_method == 'cubic' and len(valid_indices) >= 4:
                interp_func = interp1d(
                    valid_indices, 
                    var_data[valid_indices], 
                    kind='cubic',
                    fill_value='extrapolate',
                    bounds_error=False
                )
            else:
                interp_func = interp1d(
                    valid_indices, 
                    var_data[valid_indices], 
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
            
            # Interpolate only for invalid indices
            interpolated_values = interp_func(invalid_indices)
            interpolated_data[invalid_indices, var_idx] = interpolated_values
        except:
            # If interpolation fails, leave as is
            pass
    
    return interpolated_data

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime"""
    if ts_str == 'NA' or ts_str == '' or pd.isna(ts_str):
        return None
    try:
        return pd.to_datetime(ts_str)
    except:
        return None

def get_week_key(dt):
    """Get year-week key for grouping"""
    if dt is None:
        return None
    return f"{dt.year}-W{dt.isocalendar()[1]:02d}"

def aggregate_to_weekly(data, timestamps, aggregation_method='mean'):
    """
    Aggregate temporal data to weekly resolution using pandas resample (much faster).
    Always returns exactly 52 weeks, interpolating missing weeks.
    
    Args:
        data: (max_time, n_vars) array
        timestamps: (max_time,) array of timestamp strings
        aggregation_method: 'mean', 'median', or 'max'
    
    Returns:
        weekly_data: (52, n_vars) array - always 52 weeks
        weekly_timestamps: (52,) array of week keys - always 52 weeks
        weekly_valid_mask: (52,) boolean array - True for weeks with original data, False for interpolated
    """
    if len(data) == 0:
        try:
            n_vars = data.shape[1] if len(data.shape) > 1 else 0
        except (IndexError, AttributeError):
            n_vars = 0
        return (np.full((52, n_vars), np.nan, dtype=np.float32), 
                np.array(['NA'] * 52, dtype='U20'), 
                np.zeros(52, dtype=bool))
    
    # Convert timestamps to pandas DatetimeIndex
    timestamps_series = pd.to_datetime(timestamps, errors='coerce')
    valid_mask = ~timestamps_series.isna()
    
    if not np.any(valid_mask):
        n_vars = data.shape[1] if len(data.shape) > 1 else 0
        return (np.full((52, n_vars), np.nan, dtype=np.float32), 
                np.array(['NA'] * 52, dtype='U20'), 
                np.zeros(52, dtype=bool))
    
    # Filter to valid timestamps and data
    valid_timestamps = timestamps_series[valid_mask]
    valid_data = data[valid_mask]
    
    # Determine the year from the first timestamp
    year = valid_timestamps[0].year
    
    # Create DataFrame with DatetimeIndex
    df = pd.DataFrame(valid_data, index=valid_timestamps)
    
    # Resample to weekly and aggregate
    if aggregation_method == 'mean':
        weekly_df = df.resample('W').mean()
    elif aggregation_method == 'median':
        weekly_df = df.resample('W').median()
    elif aggregation_method == 'max':
        weekly_df = df.resample('W').max()
    else:
        weekly_df = df.resample('W').mean()
    
    if len(weekly_df) == 0:
        n_vars = data.shape[1] if len(data.shape) > 1 else 0
        return (np.full((52, n_vars), np.nan, dtype=np.float32), 
                np.array(['NA'] * 52, dtype='U20'), 
                np.zeros(52, dtype=bool))
    
    # Create week keys for all 52 weeks: ["2021-W01", "2021-W02", ..., "2021-W52"]
    all_week_keys = [f"{year}-W{i:02d}" for i in range(1, 53)]
    
    # Get week keys from aggregated data
    aggregated_week_keys = []
    for dt in weekly_df.index:
        week_key = get_week_key(dt)
        aggregated_week_keys.append(week_key)
    
    # Create full 52-week arrays
    n_vars = weekly_df.shape[1]
    weekly_data_full = np.full((52, n_vars), np.nan, dtype=np.float32)
    weekly_valid_mask = np.zeros(52, dtype=bool)
    
    # Map aggregated data to correct week positions
    for week_key, row_data in zip(aggregated_week_keys, weekly_df.values):
        if week_key in all_week_keys:
            week_idx = all_week_keys.index(week_key)
            weekly_data_full[week_idx] = row_data
            weekly_valid_mask[week_idx] = True
    
    # Interpolate missing weeks for each variable
    for var_idx in range(n_vars):
        var_data = weekly_data_full[:, var_idx]
        if np.any(np.isnan(var_data)):
            # Use linear interpolation
            valid_indices = np.where(~np.isnan(var_data))[0]
            if len(valid_indices) > 1:
                interp_func = interp1d(valid_indices, var_data[valid_indices], 
                                     kind='cubic', fill_value='extrapolate', 
                                     bounds_error=False)
                all_indices = np.arange(52)
                var_data_interp = interp_func(all_indices)
                weekly_data_full[:, var_idx] = var_data_interp
            elif len(valid_indices) == 1:
                # Only one data point. This should not happen.
                raise ValueError("only one valid data point. check the data.")
    
    weekly_timestamps = np.array(all_week_keys)
    
    return weekly_data_full, weekly_timestamps, weekly_valid_mask

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

        
        # Step 2: Check for NaN values in aggregated data (only in valid positions)
        print(f"  Checking for NaN values after aggregation...")
        # Only check positions where valid_mask is True
        valid_positions = aggregated_valid_mask_out
        if np.any(valid_positions):
            valid_data = aggregated_data_out[valid_positions]
            nan_count = np.sum(np.isnan(valid_data))
            if nan_count > 0:
                total_valid = valid_data.size
                nan_percentage = nan_count / total_valid * 100
                
                print(f"\n  ⚠ NaN DETECTION DETAILS for {modality_name}:")
                print(f"    Total NaN values: {nan_count:,} / {total_valid:,} ({nan_percentage:.2f}%)")
                
                # Check NaN distribution across variables
                print(f"\n    NaN distribution by variable:")
                for var_idx, var_name in enumerate(variable_names):
                    var_data = aggregated_data_out[:, :, var_idx]  # (n_samples, max_weeks)
                    var_valid_data = var_data[valid_positions]
                    var_nan_count = np.sum(np.isnan(var_valid_data))
                    if var_nan_count > 0:
                        var_nan_pct = var_nan_count / len(var_valid_data) * 100
                        print(f"      {var_name}: {var_nan_count:,} NaNs ({var_nan_pct:.2f}%)")
                
                # Find samples with NaNs
                samples_with_nans = []
                for i in range(n_samples):
                    sample_data = aggregated_data_out[i]  # (max_weeks, n_vars)
                    sample_mask = aggregated_valid_mask_out[i]  # (max_weeks,)
                    if np.any(sample_mask):
                        sample_valid_data = sample_data[sample_mask]
                        if np.any(np.isnan(sample_valid_data)):
                            nan_in_sample = np.sum(np.isnan(sample_valid_data))
                            total_in_sample = sample_valid_data.size
                            samples_with_nans.append((i, nan_in_sample, total_in_sample))
                
                print(f"\n    Samples with NaNs: {len(samples_with_nans):,} / {n_samples:,} ({len(samples_with_nans)/n_samples*100:.2f}%)")
                
                # Show first 10 samples with NaNs
                if len(samples_with_nans) > 0:
                    print(f"\n    First 10 samples with NaNs:")
                    for sample_idx, nan_count_sample, total_count_sample in samples_with_nans[:10]:
                        nan_pct_sample = nan_count_sample / total_count_sample * 100
                        print(f"      Sample {sample_idx}: {nan_count_sample}/{total_count_sample} NaNs ({nan_pct_sample:.2f}%)")
                    
                    # Show detailed example of first sample with NaNs
                    example_idx = samples_with_nans[0][0]
                    example_data = aggregated_data_out[example_idx]  # (max_weeks, n_vars)
                    example_mask = aggregated_valid_mask_out[example_idx]  # (max_weeks,)
                    example_timestamps = aggregated_timestamps_out[example_idx]  # (max_weeks,)
                    
                    print(f"\n    Example sample {example_idx} details:")
                    print(f"      Valid weeks: {np.sum(example_mask)}")
                    valid_weeks = np.where(example_mask)[0]
                    
                    # Check which weeks have NaNs
                    weeks_with_nans = []
                    for week_idx in valid_weeks:
                        week_data = example_data[week_idx]  # (n_vars,)
                        if np.any(np.isnan(week_data)):
                            nan_var_indices = np.where(np.isnan(week_data))[0]
                            nan_var_names = [variable_names[i] for i in nan_var_indices]
                            weeks_with_nans.append((week_idx, example_timestamps[week_idx], nan_var_names))
                    
                    if len(weeks_with_nans) > 0:
                        print(f"      Weeks with NaNs: {len(weeks_with_nans)}")
                        print(f"      First 5 weeks with NaNs:")
                        for week_idx, ts, nan_var_names in weeks_with_nans[:5]:
                            print(f"        Week {week_idx} ({ts}): NaNs in {nan_var_names}")
                    
                    # Show full data for first few valid weeks
                    print(f"\n      Data for first 5 valid weeks:")
                    for week_idx in valid_weeks[:5]:
                        week_data = example_data[week_idx]
                        ts = example_timestamps[week_idx]
                        nan_mask = np.isnan(week_data)
                        print(f"        Week {week_idx} ({ts}): {week_data} (NaNs: {np.sum(nan_mask)})")
                
                # Check NaN distribution across weeks
                print(f"\n    NaN distribution across weeks:")
                for week_idx in range(max_timesteps):
                    week_data = aggregated_data_out[:, week_idx, :]  # (n_samples, n_vars)
                    week_mask = aggregated_valid_mask_out[:, week_idx]  # (n_samples,)
                    if np.any(week_mask):
                        week_valid_data = week_data[week_mask]
                        week_nan_count = np.sum(np.isnan(week_valid_data))
                        if week_nan_count > 0:
                            week_nan_pct = week_nan_count / week_valid_data.size * 100
                            print(f"      Week {week_idx}: {week_nan_count:,} NaNs ({week_nan_pct:.2f}%)")
                
                raise ValueError(
                    f"\nFound {nan_count:,}/{total_valid:,} NaN values ({nan_percentage:.2f}%) "
                    f"in aggregated data for {modality_name}. This should not happen after aggregation.\n"
                    f"See detailed statistics above for more information."
                )
            else:
                print(f"  ✓ No NaN values found in aggregated data")
        else:
            print(f"  ⚠ No valid data positions found")
        
        # Step 3: Clean the aggregated data (interpolate no_data values)
        print(f"  Cleaning aggregated data (interpolating no_data values)...")
        no_data_value = NO_DATA_VALUE.get(modality_name, 0)
        for i in range(n_samples):
            sample_weekly_data = aggregated_data_out[i]  # (max_timesteps, n_vars)
            sample_valid_mask = aggregated_valid_mask_out[i]  # (max_timesteps,)
            
            # Only clean valid positions
            if np.any(sample_valid_mask):
                valid_indices = np.where(sample_valid_mask)[0]
                if len(valid_indices) > 0:
                    valid_data = sample_weekly_data[valid_indices]  # (n_valid_weeks, n_vars)
                    cleaned_data = interpolate_no_data_values(
                        valid_data,
                        no_data_value,
                        interpolation_method='linear'
                    )
                    aggregated_data_out[i, valid_indices] = cleaned_data
        
        root_out['temporal_modalities'][modality_name].create_array(
            'data', 
            data=aggregated_data_out, 
            chunks=(1000, max_timesteps, n_vars)
        )
        root_out['temporal_modalities'][modality_name].create_array(
            'timestamps', 
            data=aggregated_timestamps_out, 
            chunks=(1000, max_timesteps)
        )
        root_out['temporal_modalities'][modality_name].create_array(
            'valid_mask', 
            data=aggregated_valid_mask_out, 
            chunks=(1000, max_timesteps)
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
            chunks=(1000, max_timesteps, 2)
        )
        print(f"  ✓ Week encoding shape: {week_encoding.shape}")

    print("Data cleaned and aggregated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_zarr_path', type=str, default='/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged.zarr')
    parser.add_argument('--output_zarr_path', type=str, default='/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged_cleaned.zarr')
    args = parser.parse_args()
    main(args)