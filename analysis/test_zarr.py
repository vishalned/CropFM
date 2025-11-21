import zarr
import numpy as np

# Open the zarr dataset
zarr_path = '/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files_1200/europe_data_1200_merged.zarr'
root = zarr.open(zarr_path, mode='r')

print("=" * 80)
print("ZARR DATASET OVERVIEW")
print("=" * 80)
print(f"Total samples: {root.attrs.get('total_samples', 'N/A')}")
print(f"Created: {root.attrs.get('created_date', 'N/A')}")
print()

# Print for first sample (index 0) - change this to any sample index
sample_idx = 1200

print(f"\n{'=' * 80}")
print(f"SAMPLE {sample_idx}")
print(f"{'=' * 80}")

# 1. Metadata
print("\n--- METADATA ---")
sample_info = root['metadata/sample_info'][sample_idx]
print(f"Sample ID: {sample_info['sample_id']}")
print(f"Grid ID: {sample_info['grid_id']}")
print(f"Country: {sample_info['country']}")
print(f"Continent: {sample_info['continent']}")
print(f"Classification: {sample_info['classification']}")
print(f"Coordinates: {root['metadata/coordinates'][sample_idx]}")

# 2. Static modalities - Soil
print("\n--- SOIL (3 depth layers) ---")
for var in ['clay', 'nitrogen', 'phh2o', 'soc']:
    data = root[f'static_modalities/soil/{var}'][sample_idx]
    print(f"{var}: {data}")

# 3. Static modalities - Elevation
print("\n--- ELEVATION ---")
elevation = root['static_modalities/elevation/elevation'][sample_idx]
slope = root['static_modalities/elevation/slope'][sample_idx]
print(f"Elevation: {elevation}")
print(f"Slope: {slope}")

# 4. Static modalities - WorldCereal
if 'worldcereal_cropmask' in root['static_modalities']:
    print("\n--- WORLDCEREAL CROP MASK ---")
    aez_id = root['static_modalities/worldcereal_cropmask/aez_id'][sample_idx]
    crop_mask = root['static_modalities/worldcereal_cropmask/crop_mask'][sample_idx]
    print(f"AEZ ID: {aez_id}")
    print(f"Crop Mask: {crop_mask}")

if 'worldcereal_cropcalendar' in root['static_modalities']:
    print("\n--- WORLDCEREAL CROP CALENDAR ---")
    aez_id = root['static_modalities/worldcereal_cropcalendar/aez_id'][sample_idx]
    crop_calendar = root['static_modalities/worldcereal_cropcalendar/crop_calendar'][sample_idx]
    var_names = root['static_modalities/worldcereal_cropcalendar/variable_names'][:]
    print(f"AEZ ID: {aez_id}")
    print(f"Variable names: {var_names}")
    print(f"Crop Calendar: {crop_calendar}")

# 5. Temporal modalities
print("\n--- TEMPORAL MODALITIES ---")
for modality in ['agera5', 'sentinel1', 'sentinel2', 'fapar']:
    if modality not in root['temporal_modalities']:
        continue
    
    print(f"\n{modality.upper()}:")
    mod_group = root[f'temporal_modalities/{modality}']
    
    # Get variable names
    var_names = mod_group['variable_names'][:]
    print(f"  Variables: {var_names}")
    
    # Get data and mask
    data = mod_group['data'][sample_idx]
    timestamps = mod_group['timestamps'][sample_idx]
    valid_mask = mod_group['valid_mask'][sample_idx]
    
    # Find valid observations
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        print(f"  Number of valid observations: {len(valid_indices)}")
        print(f"  First few timestamps: {timestamps[valid_indices[:5]]}")
        print(f"  First few data points:")
        for i in valid_indices[:3]:
            print(f"    {timestamps[i]}: {data[i]}")
    else:
        print(f"  No valid observations")