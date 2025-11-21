"""
Analyze Sentinel-2 zero values and their correlation with quality flags.
"""
import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_sentinel2_zeros(root, output_dir):
    """Analyze zero values in Sentinel-2 data and correlate with quality flags"""
    
    if 'sentinel2' not in root['temporal_modalities']:
        print("Sentinel-2 data not found")
        return
    
    mod_group = root['temporal_modalities/sentinel2']
    var_names = mod_group['variable_names'][:]
    data = mod_group['data'][:]  # Shape: (n_samples, max_time, n_vars)
    valid_mask = mod_group['valid_mask'][:]  # Shape: (n_samples, max_time)
    
    # Find indices for spectral bands and quality flags
    spectral_bands = [i for i, name in enumerate(var_names) if name.startswith('B')]
    scl_idx = var_names.tolist().index('SCL') if 'SCL' in var_names else None
    qa60_idx = var_names.tolist().index('QA60') if 'QA60' in var_names else None
    msk_cldprb_idx = var_names.tolist().index('MSK_CLDPRB') if 'MSK_CLDPRB' in var_names else None
    
    print("=" * 80)
    print("SENTINEL-2 ZERO VALUE ANALYSIS")
    print("=" * 80)
    
    # Analyze each spectral band
    print("\nSPECTRAL BANDS (B1-B12) - Zero Value Analysis:")
    print("-" * 80)
    
    zero_stats = {}
    for band_idx in spectral_bands:
        band_name = var_names[band_idx]
        band_data = data[:, :, band_idx]
        
        # Get all valid values
        valid_values = band_data[valid_mask]
        valid_clean = valid_values[~np.isnan(valid_values)]
        
        # Count zeros
        zero_count = np.sum(valid_clean == 0)
        zero_pct = (zero_count / len(valid_clean)) * 100 if len(valid_clean) > 0 else 0
        
        # Count very small values (< 10, which is < 0.001 reflectance)
        very_small_count = np.sum(valid_clean < 10)
        very_small_pct = (very_small_count / len(valid_clean)) * 100 if len(valid_clean) > 0 else 0
        
        zero_stats[band_name] = {
            'zero_count': zero_count,
            'zero_pct': zero_pct,
            'very_small_count': very_small_count,
            'very_small_pct': very_small_pct,
            'total_valid': len(valid_clean),
            'min': np.min(valid_clean) if len(valid_clean) > 0 else 0,
            'mean': np.mean(valid_clean) if len(valid_clean) > 0 else 0
        }
        
        print(f"{band_name}:")
        print(f"  Total valid observations: {len(valid_clean):,}")
        print(f"  Zero values: {zero_count:,} ({zero_pct:.2f}%)")
        print(f"  Values < 10 (<0.001 reflectance): {very_small_count:,} ({very_small_pct:.2f}%)")
        print(f"  Min value: {np.min(valid_clean):.2f}, Mean: {np.mean(valid_clean):.2f}")
        
        if zero_pct > 1:
            print(f"  ⚠️  WARNING: {zero_pct:.2f}% zeros - likely no-data values")
    
    # Correlate zeros with SCL (Scene Classification Layer)
    if scl_idx is not None:
        print("\n" + "=" * 80)
        print("CORRELATION WITH SCL (Scene Classification Layer)")
        print("=" * 80)
        
        scl_data = data[:, :, scl_idx]
        scl_valid = scl_data[valid_mask]
        scl_clean = scl_valid[~np.isnan(scl_valid)]
        
        # Count SCL=0 (No Data)
        scl_zero_count = np.sum(scl_clean == 0)
        scl_zero_pct = (scl_zero_count / len(scl_clean)) * 100 if len(scl_clean) > 0 else 0
        
        print(f"\nSCL=0 (No Data) occurrences: {scl_zero_count:,} ({scl_zero_pct:.2f}%)")
        
        # For each spectral band, check if zeros correlate with SCL=0
        print("\nZero values in spectral bands vs SCL=0:")
        print("-" * 80)
        
        for band_idx in spectral_bands[:3]:  # Check first 3 bands as example
            band_name = var_names[band_idx]
            band_data = data[:, :, band_idx]
            
            # Find observations where both band and SCL are valid
            both_valid = valid_mask & (~np.isnan(band_data)) & (~np.isnan(scl_data))
            
            # Count cases where band=0 AND SCL=0
            band_zero_and_scl_zero = np.sum((band_data[both_valid] == 0) & (scl_data[both_valid] == 0))
            band_zero_total = np.sum(band_data[both_valid] == 0)
            
            if band_zero_total > 0:
                correlation_pct = (band_zero_and_scl_zero / band_zero_total) * 100
                print(f"{band_name}: {band_zero_and_scl_zero:,}/{band_zero_total:,} zeros occur when SCL=0 ({correlation_pct:.1f}%)")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("Creating enhanced plots...")
    print("=" * 80)
    
    # Plot 1: Zero value percentages for each band
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Zero percentages
    band_names = [var_names[i] for i in spectral_bands]
    zero_pcts = [zero_stats[b]['zero_pct'] for b in band_names]
    
    axes[0, 0].bar(range(len(band_names)), zero_pcts, color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Band')
    axes[0, 0].set_ylabel('Percentage of Zero Values (%)')
    axes[0, 0].set_title('Zero Value Percentage by Band\n(0 = likely no-data)')
    axes[0, 0].set_xticks(range(len(band_names)))
    axes[0, 0].set_xticklabels(band_names, rotation=45)
    axes[0, 0].axhline(y=1, color='orange', linestyle='--', label='1% threshold')
    axes[0, 0].legend()
    
    # Very small values (< 10)
    very_small_pcts = [zero_stats[b]['very_small_pct'] for b in band_names]
    axes[0, 1].bar(range(len(band_names)), very_small_pcts, color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Band')
    axes[0, 1].set_ylabel('Percentage of Values < 10 (%)')
    axes[0, 1].set_title('Very Small Values (< 0.001 reflectance)\nby Band')
    axes[0, 1].set_xticks(range(len(band_names)))
    axes[0, 1].set_xticklabels(band_names, rotation=45)
    
    # Distribution excluding zeros (for one example band)
    example_band_idx = spectral_bands[0]
    example_band_name = var_names[example_band_idx]
    example_data = data[:, :, example_band_idx]
    example_valid = example_data[valid_mask]
    example_clean = example_valid[~np.isnan(example_valid)]
    example_nonzero = example_clean[example_clean > 0]
    
    axes[1, 0].hist(example_clean, bins=100, alpha=0.5, label='All values', edgecolor='black')
    axes[1, 0].hist(example_nonzero, bins=100, alpha=0.7, label='Non-zero values', edgecolor='black', color='green')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (no-data)')
    axes[1, 0].set_xlabel('Reflectance Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{example_band_name} Distribution\n(All vs Non-zero)')
    axes[1, 0].set_xlim(-100, 2000)  # Focus on lower range
    axes[1, 0].legend()
    
    # Summary statistics table
    axes[1, 1].axis('off')
    summary_text = "SUMMARY:\n\n"
    summary_text += f"Total spectral bands analyzed: {len(spectral_bands)}\n"
    summary_text += f"Bands with >1% zeros: {sum(1 for b in band_names if zero_stats[b]['zero_pct'] > 1)}\n"
    summary_text += f"Bands with >5% zeros: {sum(1 for b in band_names if zero_stats[b]['zero_pct'] > 5)}\n\n"
    summary_text += "RECOMMENDATION:\n"
    summary_text += "Zero values in Sentinel-2 reflectance\n"
    summary_text += "bands are likely NO-DATA and should\n"
    summary_text += "be filtered or masked.\n\n"
    summary_text += "Consider filtering:\n"
    summary_text += "- All zeros in spectral bands\n"
    summary_text += "- Values where SCL=0\n"
    summary_text += "- Values where QA60 indicates\n"
    summary_text += "  clouds/cirrus"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Sentinel-2 Zero Value Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'sentinel2_zero_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved sentinel2_zero_analysis.png")
    
    # Save detailed report
    report_path = output_dir / 'sentinel2_zero_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("SENTINEL-2 ZERO VALUE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONCLUSION: Zero values in Sentinel-2 reflectance bands are NO-DATA\n\n")
        f.write("DETAILED STATISTICS:\n")
        f.write("-" * 80 + "\n")
        for band_name in band_names:
            stats = zero_stats[band_name]
            f.write(f"\n{band_name}:\n")
            f.write(f"  Total valid observations: {stats['total_valid']:,}\n")
            f.write(f"  Zero values: {stats['zero_count']:,} ({stats['zero_pct']:.2f}%)\n")
            f.write(f"  Values < 10: {stats['very_small_count']:,} ({stats['very_small_pct']:.2f}%)\n")
            f.write(f"  Min: {stats['min']:.2f}, Mean: {stats['mean']:.2f}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Filter zero values in spectral bands (B1-B12) as no-data\n")
        f.write("2. Use SCL=0 to identify no-data pixels\n")
        f.write("3. Check QA60 flags for cloud/cirrus detection\n")
        f.write("4. Consider filtering values < 10 (<0.001 reflectance) as suspicious\n")
    
    print(f"✓ Saved detailed report to: {report_path}")


def main():
    zarr_path = '/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files_1200/europe_data_1200_merged.zarr'
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading zarr dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    analyze_sentinel2_zeros(root, output_dir)


if __name__ == '__main__':
    main()