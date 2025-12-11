"""
Compute normalization statistics (mean, std) on aggregated weekly data.
Saves statistics to JSON file.
"""
import zarr
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def compute_stats_for_array(data, valid_mask=None):
    """Compute mean and std for an array, considering valid_mask if provided"""
    if valid_mask is not None:
        valid_data = data[valid_mask]
    else:
        valid_data = data.flatten()
    
    # Remove NaN values
    valid_data = valid_data[~np.isnan(valid_data)]
    
    if len(valid_data) == 0:
        return {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0, 'count': 0}
    
    return {
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'count': int(len(valid_data))
    }

def compute_metadata_stats(root):
    """Compute statistics for encoded coordinates in metadata"""
    stats = {}
    
    print("Computing statistics for metadata...")
    
    if 'metadata' in root and 'encoded_coordinates' in root['metadata']:
        encoded_coords = root['metadata']['encoded_coordinates'][:]  # (n_samples, 4)
        
        # Compute stats for each encoded coordinate component
        coord_names = ['lon_sin', 'lon_cos', 'lat_sin', 'lat_cos']
        stats['encoded_coordinates'] = {}
        
        for idx, name in enumerate(coord_names):
            coord_data = encoded_coords[:, idx]  # (n_samples,)
            stats['encoded_coordinates'][name] = compute_stats_for_array(coord_data)
            print(f"  ✓ encoded_coordinates/{name}: mean={stats['encoded_coordinates'][name]['mean']:.4f}, std={stats['encoded_coordinates'][name]['std']:.4f}")
    
    return stats

def compute_static_stats(root):
    """Compute statistics for static modalities"""
    stats = {}
    
    print("\nComputing statistics for static modalities...")
    
    # Soil data
    if 'soil' in root['static_modalities']:
        soil_group = root['static_modalities/soil']
        stats['soil'] = {}
        
        for var in ['clay', 'nitrogen', 'phh2o', 'soc']:
            if var not in soil_group:
                continue
            
            data = soil_group[var][:]  # (n_samples, 3)
            # Flatten across samples and depth layers
            stats['soil'][var] = compute_stats_for_array(data)
            print(f"  ✓ soil/{var}: mean={stats['soil'][var]['mean']:.4f}, std={stats['soil'][var]['std']:.4f}")
    
    # Elevation data
    if 'elevation' in root['static_modalities']:
        elev_group = root['static_modalities/elevation']
        stats['elevation'] = {}
        
        for var in ['elevation', 'slope']:
            if var not in elev_group:
                continue
            
            data = elev_group[var][:]  # (n_samples,)
            stats['elevation'][var] = compute_stats_for_array(data)
            print(f"  ✓ elevation/{var}: mean={stats['elevation'][var]['mean']:.4f}, std={stats['elevation'][var]['std']:.4f}")
    
    return stats

def compute_temporal_stats(root):
    """Compute statistics for temporal modalities (on aggregated weekly data)"""
    stats = {}
    
    print("\nComputing statistics for temporal modalities...")
    
    temporal_modalities = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    
    for modality in temporal_modalities:
        if modality not in root['temporal_modalities']:
            continue
        
        print(f"\nProcessing {modality}...")
        mod_group = root[f'temporal_modalities/{modality}']
        
        var_names = mod_group['variable_names'][:]
        data = mod_group['data'][:]  # (n_samples, max_weeks, n_vars)
        valid_mask = mod_group['valid_mask'][:]  # (n_samples, max_weeks)
        
        stats[modality] = {}
        
        # Compute stats for data variables
        for var_idx, var_name in enumerate(var_names):
            var_data = data[:, :, var_idx]  # (n_samples, max_weeks)
            
            # Only use valid observations
            valid_data = var_data[valid_mask]
            
            stats[modality][var_name] = compute_stats_for_array(valid_data)
            print(f"  ✓ {modality}/{var_name}: mean={stats[modality][var_name]['mean']:.4f}, std={stats[modality][var_name]['std']:.4f}")
        
        # Compute stats for week encoding if it exists
        if 'week_encoding' in mod_group:
            week_encoding = mod_group['week_encoding'][:]  # (n_samples, max_weeks, 2)
            
            # Compute stats for week_sin and week_cos
            week_names = ['week_sin', 'week_cos']
            stats[modality]['week_encoding'] = {}
            
            for idx, week_name in enumerate(week_names):
                week_data = week_encoding[:, :, idx]  # (n_samples, max_weeks)
                
                # Only use valid observations (same mask as data)
                valid_week_data = week_data[valid_mask]
                
                stats[modality]['week_encoding'][week_name] = compute_stats_for_array(valid_week_data)
                print(f"  ✓ {modality}/week_encoding/{week_name}: mean={stats[modality]['week_encoding'][week_name]['mean']:.4f}, std={stats[modality]['week_encoding'][week_name]['std']:.4f}")
    
    return stats

def main():
    input_zarr_path = '/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged_cleaned.zarr'
    stats_output_path = Path('/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/normalization_stats.json')
    
    print("=" * 80)
    print("COMPUTE NORMALIZATION STATISTICS")
    print("=" * 80)
    print(f"Input:  {input_zarr_path}")
    print(f"Output: {stats_output_path}")
    print()
    
    # Open zarr
    root = zarr.open(input_zarr_path, mode='r')
    total_samples = root.attrs.get('total_samples', 0)
    print(f"Total samples: {total_samples:,}")
    
    # Compute statistics
    metadata_stats = compute_metadata_stats(root)
    static_stats = compute_static_stats(root)
    temporal_stats = compute_temporal_stats(root)
    
    all_stats = {
        'metadata': {
            **metadata_stats,
            'info': {
                'total_samples': int(total_samples),
                'computed_date': str(pd.Timestamp.now()),
                'data_source': input_zarr_path
            }
        },
        'static_modalities': static_stats,
        'temporal_modalities': temporal_stats
    }
    
    # Save to JSON
    with open(stats_output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("STATISTICS COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"Statistics saved to: {stats_output_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Static modalities: {len(all_stats['static_modalities'])} groups")
    print(f"  Temporal modalities: {len(all_stats['temporal_modalities'])} groups")

if __name__ == '__main__':
    import pandas as pd
    main()