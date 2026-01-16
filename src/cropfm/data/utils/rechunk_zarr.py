"""
Script to rechunk an existing zarr file for better random access performance.

This script creates a new zarr file with smaller chunks optimized for random access
patterns (shuffled training) instead of sequential access.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def rechunk_zarr(
    input_zarr_path: str,
    output_zarr_path: str,
    temporal_chunk_size: int = 10,
    static_chunk_size: int = 1000,
    metadata_chunk_size: int = 1000,
    batch_size: int = 1000
):
    """
    Rechunk a zarr file with new chunk sizes.
    
    Args:
        input_zarr_path: Path to input zarr file
        output_zarr_path: Path to output zarr file (will be created)
        temporal_chunk_size: Chunk size for sample dimension in temporal modalities (default: 10)
        static_chunk_size: Chunk size for sample dimension in static modalities (default: 1000)
        metadata_chunk_size: Chunk size for sample dimension in metadata (default: 1000)
        batch_size: Number of samples to process at once (default: 1000)
    """
    print(f"Opening input zarr: {input_zarr_path}")
    input_root = zarr.open(input_zarr_path, mode='r')
    
    # Get total samples
    total_samples = input_root.attrs['total_samples']
    print(f"Total samples: {total_samples}")
    
    # Create output zarr
    print(f"Creating output zarr: {output_zarr_path}")
    output_root = zarr.open_group(output_zarr_path, mode='w')
    
    # Copy root attributes
    output_root.attrs.update(input_root.attrs)
    output_root.attrs['rechunked'] = True
    output_root.attrs['temporal_chunk_size'] = temporal_chunk_size
    output_root.attrs['static_chunk_size'] = static_chunk_size
    
    # 1. Rechunk metadata
    print("\n=== Processing metadata ===")
    if 'metadata' in input_root:
        input_meta = input_root['metadata']
        output_meta = output_root.create_group('metadata')
        
        # Copy sample_info
        if 'sample_info' in input_meta:
            print("  Copying sample_info...")
            input_arr = input_meta['sample_info']
            output_arr = output_meta.create_array(
                'sample_info',
                shape=input_arr.shape,
                dtype=input_arr.dtype,
                chunks=(metadata_chunk_size,)
            )
            # Copy in batches
            for i in tqdm(range(0, total_samples, batch_size), desc="    sample_info"):
                end_idx = min(i + batch_size, total_samples)
                output_arr[i:end_idx] = input_arr[i:end_idx]
        
        # Copy coordinates
        if 'coordinates' in input_meta:
            print("  Copying coordinates...")
            input_arr = input_meta['coordinates']
            output_arr = output_meta.create_array(
                'coordinates',
                shape=input_arr.shape,
                dtype=input_arr.dtype,
                chunks=(metadata_chunk_size, input_arr.shape[1] if len(input_arr.shape) > 1 else None)
            )
            for i in tqdm(range(0, total_samples, batch_size), desc="    coordinates"):
                end_idx = min(i + batch_size, total_samples)
                output_arr[i:end_idx] = input_arr[i:end_idx]
        
        # Copy encoded_coordinates if it exists
        if 'encoded_coordinates' in input_meta:
            print("  Copying encoded_coordinates...")
            input_arr = input_meta['encoded_coordinates']
            output_arr = output_meta.create_array(
                'encoded_coordinates',
                shape=input_arr.shape,
                dtype=input_arr.dtype,
                chunks=(metadata_chunk_size, input_arr.shape[1] if len(input_arr.shape) > 1 else None)
            )
            for i in tqdm(range(0, total_samples, batch_size), desc="    encoded_coordinates"):
                end_idx = min(i + batch_size, total_samples)
                output_arr[i:end_idx] = input_arr[i:end_idx]
    
    # 2. Rechunk static modalities
    print("\n=== Processing static modalities ===")
    if 'static_modalities' in input_root:
        input_static = input_root['static_modalities']
        output_static = output_root.create_group('static_modalities')
        
        for group_name in input_static.keys():
            print(f"  Processing {group_name}...")
            input_group = input_static[group_name]
            output_group = output_static.create_group(group_name)
            
            for array_name in input_group.keys():
                input_arr = input_group[array_name]
                print(f"    Copying {array_name}...")
                
                # Determine chunk size based on array shape
                if len(input_arr.shape) == 1:
                    new_chunks = (static_chunk_size,)
                elif len(input_arr.shape) == 2:
                    new_chunks = (static_chunk_size, input_arr.shape[1])
                else:
                    new_chunks = (static_chunk_size,) + input_arr.shape[1:]
                
                output_arr = output_group.create_array(
                    array_name,
                    shape=input_arr.shape,
                    dtype=input_arr.dtype,
                    chunks=new_chunks,
                    fill_value=input_arr.fill_value if hasattr(input_arr, 'fill_value') else None
                )
                
                # Copy in batches
                for i in tqdm(range(0, total_samples, batch_size), desc=f"      {array_name}"):
                    end_idx = min(i + batch_size, total_samples)
                    if len(input_arr.shape) == 1:
                        output_arr[i:end_idx] = input_arr[i:end_idx]
                    elif len(input_arr.shape) == 2:
                        output_arr[i:end_idx, :] = input_arr[i:end_idx, :]
                    else:
                        # For higher dimensions, use ellipsis
                        output_arr[i:end_idx] = input_arr[i:end_idx]
    
    # 3. Rechunk temporal modalities (this is the main optimization)
    print("\n=== Processing temporal modalities ===")
    if 'temporal_modalities' in input_root:
        input_temporal = input_root['temporal_modalities']
        output_temporal = output_root.create_group('temporal_modalities')
        
        for modality_name in input_temporal.keys():
            print(f"  Processing {modality_name}...")
            input_mod = input_temporal[modality_name]
            output_mod = output_temporal.create_group(modality_name)
            
            # Process each array in the modality
            for array_name in input_mod.keys():
                input_arr = input_mod[array_name]
                print(f"    Copying {array_name}...")
                
                # Determine new chunk size based on array shape
                if len(input_arr.shape) == 1:
                    # Variable names - keep as is
                    new_chunks = input_arr.shape
                elif len(input_arr.shape) == 2:
                    # Timestamps, valid_mask: (n_samples, max_time)
                    new_chunks = (temporal_chunk_size, input_arr.shape[1])
                elif len(input_arr.shape) == 3:
                    # Data, week_encoding: (n_samples, max_time, n_variables)
                    new_chunks = (temporal_chunk_size, input_arr.shape[1], input_arr.shape[2])
                else:
                    # Fallback: use original chunks
                    new_chunks = input_arr.chunks
                
                output_arr = output_mod.create_array(
                    array_name,
                    shape=input_arr.shape,
                    dtype=input_arr.dtype,
                    chunks=new_chunks,
                    fill_value=input_arr.fill_value if hasattr(input_arr, 'fill_value') else None
                )
                
                # Copy in batches
                if len(input_arr.shape) == 1:
                    # Small array, copy all at once
                    output_arr[:] = input_arr[:]
                elif len(input_arr.shape) == 2:
                    # 2D: (n_samples, max_time)
                    for i in tqdm(range(0, total_samples, batch_size), desc=f"      {array_name}"):
                        end_idx = min(i + batch_size, total_samples)
                        output_arr[i:end_idx, :] = input_arr[i:end_idx, :]
                elif len(input_arr.shape) == 3:
                    # 3D: (n_samples, max_time, n_variables)
                    for i in tqdm(range(0, total_samples, batch_size), desc=f"      {array_name}"):
                        end_idx = min(i + batch_size, total_samples)
                        output_arr[i:end_idx, :, :] = input_arr[i:end_idx, :, :]
                else:
                    # Higher dimensions - copy all at once (shouldn't happen)
                    output_arr[:] = input_arr[:]
    
    print("\n=== Rechunking complete! ===")
    print(f"Input:  {input_zarr_path}")
    print(f"Output: {output_zarr_path}")
    print(f"\nNew chunk sizes:")
    print(f"  Temporal modalities: {temporal_chunk_size} samples per chunk")
    print(f"  Static modalities: {static_chunk_size} samples per chunk")
    print(f"  Metadata: {metadata_chunk_size} samples per chunk")


def main():
    parser = argparse.ArgumentParser(description='Rechunk a zarr file for better random access performance')
    parser.add_argument('input_zarr', type=str, help='Path to input zarr file')
    parser.add_argument('output_zarr', type=str, help='Path to output zarr file')
    parser.add_argument('--temporal-chunk-size', type=int, default=10,
                        help='Chunk size for sample dimension in temporal modalities (default: 10)')
    parser.add_argument('--static-chunk-size', type=int, default=1000,
                        help='Chunk size for sample dimension in static modalities (default: 1000)')
    parser.add_argument('--metadata-chunk-size', type=int, default=1000,
                        help='Chunk size for sample dimension in metadata (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of samples to process at once (default: 1000)')
    
    args = parser.parse_args()
    
    rechunk_zarr(
        input_zarr_path=args.input_zarr,
        output_zarr_path=args.output_zarr,
        temporal_chunk_size=args.temporal_chunk_size,
        static_chunk_size=args.static_chunk_size,
        metadata_chunk_size=args.metadata_chunk_size,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

