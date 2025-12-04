"""
Merge multiple zarr batch files into a single zarr dataset.
"""
import zarr
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import warnings

# Suppress asyncio warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def merge_zarr_batches(base_zarr_path, total_batches, output_zarr_path=None):
    """
    Merge multiple zarr batch files into a single zarr dataset.
    
    Args:
        base_zarr_path: Base path of zarr files (e.g., /path/to/data.zarr)
                       Batch files should be named: data_batch_000.zarr, data_batch_001.zarr, etc.
        total_batches: Total number of batch files to merge
        output_zarr_path: Output path for merged zarr. If None, uses base_zarr_path
    """
    if output_zarr_path is None:
        output_zarr_path = base_zarr_path
    
    print(f"Merging {total_batches} zarr batch files...")
    print(f"Base path: {base_zarr_path}")
    print(f"Output path: {output_zarr_path}")
    
    # Collect all batch files
    batch_files = []
    total_samples = 0
    
    zarr_dir = os.path.dirname(base_zarr_path)
    zarr_basename = os.path.basename(base_zarr_path).replace('.zarr', '')
    
    for batch_idx in range(total_batches):
        batch_path = os.path.join(zarr_dir, f"{zarr_basename}_batch_{batch_idx:03d}.zarr")
        if os.path.exists(batch_path):
            batch_root = zarr.open(batch_path, mode='r')
            batch_samples = batch_root['metadata/sample_info'].shape[0]
            batch_files.append((batch_path, batch_samples))
            total_samples += batch_samples
            print(f"  Batch {batch_idx}: {batch_path} ({batch_samples} samples)")
        else:
            print(f"  WARNING: Batch {batch_idx} file not found: {batch_path}")
    
    if not batch_files:
        print("ERROR: No batch files found!")
        return
    
    print(f"\nTotal samples to merge: {total_samples}")
    
    # Create output zarr dataset
    print(f"\nCreating merged zarr dataset at {output_zarr_path}...")
    
    # Use the first batch as template
    first_batch_path, first_batch_samples = batch_files[0]
    first_batch_root = zarr.open(first_batch_path, mode='r')
    
    # Create output zarr with same structure
    output_root = zarr.open_group(output_zarr_path, mode='w')
    
    # Track which variable_names arrays we've already copied
    variable_names_copied = set()
    
    # Copy structure from first batch
    def copy_structure(source_group, target_group, total_samples, current_path=""):
        """Recursively copy zarr structure"""
        for key in source_group.keys():
            item = source_group[key]
            full_path = f"{current_path}/{key}" if current_path else key
            if isinstance(item, zarr.Group):
                target_subgroup = target_group.create_group(key)
                copy_structure(item, target_subgroup, total_samples, full_path)
            elif isinstance(item, zarr.Array):
                # Create array with same shape but adjusted first dimension
                source_shape = item.shape
                if len(source_shape) > 0:
                    # For variable_names, keep original shape (not per-sample)
                    if 'variable_names' in full_path:
                        new_shape = source_shape
                    else:
                        # Adjust first dimension to total_samples
                        new_shape = (total_samples,) + source_shape[1:]
                else:
                    new_shape = source_shape
                
                target_array = target_group.create_array(
                    key,
                    shape=new_shape,
                    dtype=item.dtype,
                    chunks=item.chunks,
                    fill_value=item.fill_value
                )
    
    copy_structure(first_batch_root, output_root, total_samples)
    
    # Copy attributes
    output_root.attrs.update(first_batch_root.attrs)
    output_root.attrs['total_samples'] = total_samples
    output_root.attrs['merged_from_batches'] = total_batches
    
    # Merge data from all batches
    print("\nMerging data from batches...")
    sample_offset = 0
    
    for batch_idx, (batch_path, batch_samples) in enumerate(batch_files):
        print(f"  Merging batch {batch_idx + 1}/{len(batch_files)} ({batch_samples} samples)...")
        batch_root = zarr.open(batch_path, mode='r')
        
        def merge_data(source_group, target_group, sample_offset, batch_samples, current_path=""):
            """Recursively merge data"""
            for key in source_group.keys():
                item = source_group[key]
                full_path = f"{current_path}/{key}" if current_path else key
                
                if isinstance(item, zarr.Group):
                    merge_data(item, target_group[key], sample_offset, batch_samples, full_path)
                elif isinstance(item, zarr.Array):
                    target_array = target_group[key]
                    
                    # Handle variable_names arrays specially (shared, not per-sample)
                    if 'variable_names' in full_path:
                        if full_path not in variable_names_copied:
                            # Copy from first batch only
                            try:
                                if item.size > 0:
                                    target_array[:] = item[:]
                                    variable_names_copied.add(full_path)
                                    print(f"    Copied variable_names: {full_path}")
                                else:
                                    print(f"    WARNING: Empty variable_names array: {full_path}")
                            except Exception as e:
                                print(f"    ERROR copying variable_names {full_path}: {e}")
                        # Skip for all other batches
                        continue
                    
                    # Copy data for regular arrays
                    try:
                        if len(item.shape) > 0:
                            # Multi-dimensional array
                            if len(item.shape) == 1:
                                # Check if this is a per-sample 1D array or a shared array
                                if item.shape[0] == batch_samples:
                                    # Per-sample array
                                    target_array[sample_offset:sample_offset + batch_samples] = item[:]
                                else:
                                    # Shared array (like variable_names that wasn't caught above)
                                    if sample_offset == 0:
                                        target_array[:] = item[:]
                            elif len(item.shape) == 2:
                                target_array[sample_offset:sample_offset + batch_samples, :] = item[:, :]
                            elif len(item.shape) == 3:
                                target_array[sample_offset:sample_offset + batch_samples, :, :] = item[:, :, :]
                            else:
                                # For higher dimensions, use ellipsis
                                slices = [slice(sample_offset, sample_offset + batch_samples)] + [slice(None)] * (len(item.shape) - 1)
                                target_array[tuple(slices)] = item[:]
                        else:
                            # Scalar array, copy once
                            if sample_offset == 0:
                                target_array[:] = item[:]
                    except Exception as e:
                        print(f"    ERROR copying {full_path}: {e}")
                        raise
        
        try:
            merge_data(batch_root, output_root, sample_offset, batch_samples)
            sample_offset += batch_samples
        except Exception as e:
            print(f"ERROR merging batch {batch_idx}: {e}")
            raise
        finally:
            # Close batch root to free resources
            batch_root = None
    
    print(f"\nMerge complete! Merged {total_samples} samples into {output_zarr_path}")
    print(f"Total batches merged: {len(batch_files)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge zarr batch files')
    parser.add_argument('--base-zarr-path', type=str, required=True,
                       help='Base path of zarr batch files')
    parser.add_argument('--total-batches', type=int, required=True,
                       help='Total number of batch files')
    parser.add_argument('--output-zarr-path', type=str, default=None,
                       help='Output path for merged zarr (default: same as base path)')
    
    args = parser.parse_args()
    
    try:
        merge_zarr_batches(args.base_zarr_path, args.total_batches, args.output_zarr_path)
    except KeyboardInterrupt:
        print("\nMerge interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)