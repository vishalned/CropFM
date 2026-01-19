"""
Script to rechunk an existing zarr file for better random access performance (Zarr 3.0).

Configuration:
- Shard size: 1,024 points (physical file size)
- Chunk size: 128 points (logical unit, matches batch size)
- Compression: Blosc LZ4 (fast compression/decompression)
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
# Zarr 3.0 Codecs
from zarr.codecs import BloscCodec

def rechunk_zarr(
    input_zarr_path: str,
    output_zarr_path: str,
    chunk_size: int = 128,
    shard_size: int = 1024,
    batch_size: int = 1000,
    use_sharding: bool = True,
    use_blosc: bool = True
):
    print(f"Opening input zarr: {input_zarr_path}")
    input_root = zarr.open(input_zarr_path, mode='r')
    
    # Get total samples from attributes
    if 'total_samples' not in input_root.attrs:
        # Fallback if attribute is missing: check first array found
        total_samples = 0
        print("Warning: 'total_samples' attr not found. Attempting to infer...")
    else:
        total_samples = input_root.attrs['total_samples']
    
    print(f"Total samples: {total_samples}")
    
    # Create output zarr (Zarr 3.0 style)
    print(f"Creating output zarr: {output_zarr_path}")
    output_root = zarr.create_group(output_zarr_path, overwrite=True)
    
    # Copy root attributes
    output_root.attrs.update(input_root.attrs)
    output_root.attrs['rechunked_v3'] = True
    output_root.attrs['chunk_size'] = chunk_size
    output_root.attrs['shard_size'] = shard_size
    
    # Configure Blosc Compressor for Zarr 3.0
    compressor_list = []
    if use_blosc:
        # 'shuffle' in Zarr 3.0 is a string: 'bitshuffle', 'shuffle', or 'noshuffle'
        compressor_list = [BloscCodec(
            cname='lz4',
            clevel=5,
            shuffle='bitshuffle'
        )]
        print(f"Using Blosc compression (lz4, level 5, bitshuffle)")
    
    def get_shards(chunks_tuple):
        """Calculate shards parameter. Shard size must be a multiple of chunk size."""
        if not use_sharding:
            return None
        # Shard size applies only to first dimension, other dimensions match chunks
        return (shard_size,) + chunks_tuple[1:]

    def copy_array(input_arr, output_group, name, is_metadata=False):
        """Helper to create and copy array data."""
        # 1. Determine Chunks
        if is_metadata and input_arr.ndim == 1:
            # For small 1D metadata, don't rechunk or shard
            new_chunks = input_arr.shape
            new_shards = None
            current_compressors = None
        else:
            # For data, use the sample-based chunking
            new_chunks = (chunk_size,) + input_arr.shape[1:]
            new_shards = get_shards(new_chunks)
            current_compressors = compressor_list

        # 2. Create Array (Zarr 3.0 API)
        out_arr = output_group.create_array(
            name,
            shape=input_arr.shape,
            dtype=input_arr.dtype,
            chunks=new_chunks,
            shards=new_shards,
            compressors=current_compressors,
            fill_value=input_arr.fill_value if hasattr(input_arr, 'fill_value') else None,
            overwrite=True
        )

        # 3. Copy Data
        if input_arr.size == 0:
            return
            
        # If very small, copy at once; otherwise batch
        if input_arr.shape[0] <= batch_size:
            out_arr[:] = input_arr[:]
        else:
            for i in tqdm(range(0, input_arr.shape[0], batch_size), desc=f"    {name}", leave=False):
                end_idx = min(i + batch_size, input_arr.shape[0])
                # Zarr 3.0 handles the rechunking logic during the assignment
                out_arr[i:end_idx] = input_arr[i:end_idx]

    # --- 1. Processing metadata ---
    if 'metadata' in input_root:
        print("\n=== Processing metadata ===")
        out_meta = output_root.create_group('metadata')
        for arr_name in input_root['metadata'].keys():
            copy_array(input_root['metadata'][arr_name], out_meta, arr_name, is_metadata=True)

    # --- 2. Processing static modalities ---
    if 'static_modalities' in input_root:
        print("\n=== Processing static modalities ===")
        out_static = output_root.create_group('static_modalities')
        for group_name in input_root['static_modalities'].keys():
            print(f"  Group: {group_name}")
            in_grp = input_root['static_modalities'][group_name]
            out_grp = out_static.create_group(group_name)
            for arr_name in in_grp.keys():
                copy_array(in_grp[arr_name], out_grp, arr_name)

    # --- 3. Processing temporal modalities ---
    if 'temporal_modalities' in input_root:
        print("\n=== Processing temporal modalities ===")
        out_temp = output_root.create_group('temporal_modalities')
        for mod_name in input_root['temporal_modalities'].keys():
            print(f"  Modality: {mod_name}")
            in_mod = input_root['temporal_modalities'][mod_name]
            out_mod = out_temp.create_group(mod_name)
            for arr_name in in_mod.keys():
                # Check if it's a variable names array (usually 1D strings)
                is_meta = (arr_name == 'variable_names' or in_mod[arr_name].ndim == 1)
                copy_array(in_mod[arr_name], out_mod, arr_name, is_metadata=is_meta)

    print("\n=== Rechunking complete! ===")
    print(f"Configuration: Chunk={chunk_size}, Shard={shard_size}")

def main():
    parser = argparse.ArgumentParser(description='Rechunk Zarr with Zarr 3.0 Sharding')
    parser.add_argument('input_zarr', type=str)
    parser.add_argument('output_zarr', type=str)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--shard-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--no-sharding', action='store_true')
    parser.add_argument('--no-blosc', action='store_true')
    
    args = parser.parse_args()
    
    rechunk_zarr(
        input_zarr_path=args.input_zarr,
        output_zarr_path=args.output_zarr,
        chunk_size=args.chunk_size,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        use_sharding=not args.no_sharding,
        use_blosc=not args.no_blosc
    )

if __name__ == '__main__':
    main()