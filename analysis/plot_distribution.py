"""
Visualize distributions of all modalities in the zarr dataset.
"""
import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_static_modality_distributions(root, output_dir):
    """Plot distributions for static modalities"""
    
    # 1. Soil properties
    if 'soil' in root['static_modalities']:
        soil_vars = ['clay', 'nitrogen', 'phh2o', 'soc']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(soil_vars):
            soil_data = root[f'static_modalities/soil/{var}'][:]
            # Flatten all depth layers
            soil_flat = soil_data.flatten()
            # Remove NaN values
            soil_clean = soil_flat[~np.isnan(soil_flat)]
            
            axes[i].hist(soil_clean, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(f'{var.capitalize()} Value')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{var.capitalize()} Distribution\n(n={len(soil_clean):,} values)')
            axes[i].axvline(np.mean(soil_clean), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(soil_clean):.2f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'soil_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved soil_distributions.png")
    
    # 2. Elevation
    if 'elevation' in root['static_modalities']:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        elevation = root['static_modalities/elevation/elevation'][:]
        slope = root['static_modalities/elevation/slope'][:]
        
        elev_clean = elevation[~np.isnan(elevation)]
        slope_clean = slope[~np.isnan(slope)]
        
        axes[0].hist(elev_clean, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Elevation (m)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Elevation Distribution\n(n={len(elev_clean):,} samples)')
        axes[0].axvline(np.mean(elev_clean), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(elev_clean):.1f}m')
        axes[0].legend()
        
        axes[1].hist(slope_clean, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Slope (degrees)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Slope Distribution\n(n={len(slope_clean):,} samples)')
        axes[1].axvline(np.mean(slope_clean), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(slope_clean):.2f}°')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'elevation_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved elevation_distributions.png")
    
    # 3. WorldCereal Crop Mask
    if 'worldcereal_cropmask' in root['static_modalities']:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        crop_mask = root['static_modalities/worldcereal_cropmask/crop_mask'][:]
        aez_ids = root['static_modalities/worldcereal_cropmask/aez_id'][:]
        
        # Remove invalid values
        valid_mask = crop_mask != -1
        crop_mask_clean = crop_mask[valid_mask]
        aez_clean = aez_ids[aez_ids != -1]
        
        # Crop type distribution
        crop_labels = {0: 'No crop', 1: 'Maize', 2: 'Winter Cereals', 3: 'Spring Cereals'}
        crop_counts = np.bincount(crop_mask_clean.astype(int))
        
        bars = axes[0].bar(range(len(crop_counts)), crop_counts)
        axes[0].set_xlabel('Crop Type')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Crop Mask Distribution')
        axes[0].set_xticks(range(len(crop_counts)))
        axes[0].set_xticklabels([crop_labels.get(i, f'Type {i}') for i in range(len(crop_counts))])
        
        # Add value labels on bars
        for bar, count in zip(bars, crop_counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(count), ha='center', va='bottom')
        
        # AEZ distribution (top 20)
        unique_aez, counts = np.unique(aez_clean, return_counts=True)
        top_indices = np.argsort(counts)[-20:][::-1]
        top_aez = unique_aez[top_indices]
        top_counts = counts[top_indices]
        
        axes[1].bar(range(len(top_aez)), top_counts)
        axes[1].set_xlabel('AEZ ID (Top 20)')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Top 20 Agro-Ecological Zones')
        axes[1].set_xticks(range(len(top_aez)))
        axes[1].set_xticklabels(top_aez, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'worldcereal_cropmask_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved worldcereal_cropmask_distributions.png")
    
    # 4. WorldCereal Crop Calendar
    if 'worldcereal_cropcalendar' in root['static_modalities']:
        crop_calendar = root['static_modalities/worldcereal_cropcalendar/crop_calendar'][:]
        var_names = root['static_modalities/worldcereal_cropcalendar/variable_names'][:]
        
        # Filter out invalid values (-1)
        valid_mask = (crop_calendar != -1).any(axis=1)
        crop_calendar_clean = crop_calendar[valid_mask]
        
        if len(crop_calendar_clean) > 0:
            n_vars = crop_calendar_clean.shape[1]
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i in range(min(n_vars, 6)):
                var_data = crop_calendar_clean[:, i]
                var_data_clean = var_data[var_data != -1]
                
                if len(var_data_clean) > 0:
                    var_name = var_names[i] if i < len(var_names) and var_names[i] else f'Variable {i}'
                    axes[i].hist(var_data_clean, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_xlabel('Day of Year')
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{var_name}\n(n={len(var_data_clean):,} valid values)')
                    axes[i].axvline(np.mean(var_data_clean), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(var_data_clean):.1f}')
                    axes[i].legend()
            
            # Hide unused subplots
            for i in range(n_vars, 6):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'worldcereal_cropcalendar_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved worldcereal_cropcalendar_distributions.png")


def plot_temporal_modality_distributions(root, output_dir):
    """Plot distributions for temporal modalities (aggregated across all time steps)"""
    
    temporal_modalities = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    
    for modality in temporal_modalities:
        if modality not in root['temporal_modalities']:
            continue
        
        print(f"\nProcessing {modality}...")
        mod_group = root[f'temporal_modalities/{modality}']
        
        # Get variable names
        var_names = mod_group['variable_names'][:]
        n_vars = len(var_names)
        
        # Get all data and valid mask
        data = mod_group['data'][:]  # Shape: (n_samples, max_time, n_vars)
        valid_mask = mod_group['valid_mask'][:]  # Shape: (n_samples, max_time)
        
        # Collect all valid values for each variable
        all_values = []
        for var_idx in range(n_vars):
            # Extract valid values for this variable across all samples and time steps
            var_data = data[:, :, var_idx]  # Shape: (n_samples, max_time)
            valid_values = var_data[valid_mask]  # Flatten and mask
            valid_values_clean = valid_values[~np.isnan(valid_values)]
            all_values.append(valid_values_clean)
        
        # Create subplots
        n_cols = min(4, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_vars == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for var_idx in range(n_vars):
            var_values = all_values[var_idx]
            var_name = var_names[var_idx] if var_idx < len(var_names) else f'Variable {var_idx}'
            
            if len(var_values) > 0:
                axes[var_idx].hist(var_values, bins=50, alpha=0.7, edgecolor='black')
                axes[var_idx].set_xlabel('Value')
                axes[var_idx].set_ylabel('Frequency')
                axes[var_idx].set_title(f'{var_name}\n(n={len(var_values):,} observations)')
                axes[var_idx].axvline(np.mean(var_values), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(var_values):.2f}')
                axes[var_idx].legend()
            else:
                axes[var_idx].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                                  transform=axes[var_idx].transAxes)
                axes[var_idx].set_title(f'{var_name} - No Data')
        
        # Hide unused subplots
        for var_idx in range(n_vars, len(axes)):
            axes[var_idx].axis('off')
        
        plt.suptitle(f'{modality.upper()} - Value Distributions (All Time Steps)', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'{modality}_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {modality}_distributions.png")


def main():
    # Path to zarr dataset
    zarr_path = '/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files_1200/europe_data_1200_merged.zarr'
    
    # Create output directory
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading zarr dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    total_samples = root.attrs.get('total_samples', 0)
    print(f"Total samples: {total_samples:,}")
    
    print("\n" + "="*80)
    print("PLOTTING STATIC MODALITIES")
    print("="*80)
    plot_static_modality_distributions(root, output_dir)
    
    print("\n" + "="*80)
    print("PLOTTING TEMPORAL MODALITIES")
    print("="*80)
    plot_temporal_modality_distributions(root, output_dir)
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir.absolute()}")
    print("="*80)


if __name__ == '__main__':
    main()