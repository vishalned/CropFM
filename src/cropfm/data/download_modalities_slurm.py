import hydra
import ee
import json
import numpy as np
import sys
import os
from omegaconf import DictConfig
from cropfm.data.modalities import (
    agera5,
    sentinel1,
    sentinel2,
    fapar,
    soil,
    elevation,
    worldcereal_cropmask,
    worldcereal_cropcalender
)
import logging
import time
from cropfm.data.utils.save2zarr import create_zarr_dataset, add_sample_data

log = logging.getLogger(__name__)

def log_failed_point(point_id, error_file_path, error_msg=None):
    """Log failed point ID to file"""
    with open(error_file_path, 'a') as f:
        if error_msg:
            f.write(f"{point_id}\t{error_msg}\n")
        else:
            f.write(f"{point_id}\n")

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml', version_base=None)
def main(cfg: DictConfig):
    ee.Initialize()
    log.setLevel(cfg.log_level)
    log.debug(f'CFG: {cfg}')
    log.info(f'Downloading modalities: {cfg.modalities}')
    
    # Get batch index from environment variable (set by SLURM)
    batch_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', cfg.get('batch_idx', 0)))
    total_batches = cfg.get('total_batches', 40)
    
    log.info(f"Processing batch {batch_idx + 1}/{total_batches}")
    
    # Load the GeoJSON file
    geojson_path = cfg.points_geojson_file
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    log.info(f"Loaded GeoJSON with {len(geojson_data['features'])} total features")
    

    ## for testing only
    # geojson_data['features'] = geojson_data['features'][:1200]

    
    # Calculate batch boundaries
    total_points = len(geojson_data['features'])
    points_per_batch = total_points // total_batches
    batch_start = batch_idx * points_per_batch
    batch_end = batch_start + points_per_batch
    
    # Last batch gets remaining points
    if batch_idx == total_batches - 1:
        batch_end = total_points
    
    selected_features = geojson_data['features'][batch_start:batch_end]
    
    log.info(f"Batch {batch_idx + 1}: Processing points {batch_start} to {batch_end-1} ({len(selected_features)} points)")
    
    # Create zarr dataset for this batch
    # Use batch index in the zarr path
    base_zarr_path = cfg.zarr_path
    zarr_dir = os.path.dirname(base_zarr_path)
    zarr_basename = os.path.basename(base_zarr_path).replace('.zarr', '')
    zarr_path = os.path.join(zarr_dir, f"{zarr_basename}_batch_{batch_idx:03d}.zarr")
    
    # Create error log file path (similar naming to zarr)
    error_log_path = cfg.get('error_log_path', os.path.join(zarr_dir, 'failed_points.txt'))
    error_log_dir = os.path.dirname(error_log_path) if os.path.dirname(error_log_path) else '.'
    error_log_basename = os.path.basename(error_log_path).replace('.txt', '')
    error_file_path = os.path.join(error_log_dir, f"{error_log_basename}_batch_{batch_idx:03d}.txt")
    
    # Clear error log file if it exists (for fresh start)
    if os.path.exists(error_file_path):
        os.remove(error_file_path)
    
    total_samples = len(selected_features)
    zarr_root = create_zarr_dataset(zarr_path, total_samples)
    log.info(f"Created zarr dataset at {zarr_path} with {total_samples} samples")
    log.info(f"Error log will be written to: {error_file_path}")
    
    start_time_total = time.time()
    failed_count = 0
    
    # Process each point sequentially
    for i, feature in enumerate(selected_features):
        point_id = feature['id']
        coordinates = feature['geometry']['coordinates']
        point = ee.Geometry.Point(coordinates)
        
        log.info(f"Processing point {i+1}/{len(selected_features)} (global index {batch_start + i}) - ID: {point_id}, Coordinates: {coordinates}")
        
        # Initialize sample data dictionary with feature properties
        properties = feature.get('properties', {})
        sample_data = {
            'metadata': {
                'sample_id': point_id,
                'grid_id': properties.get('grid_id', int(point_id.split('_')[0]) if '_' in point_id else 0),
                'country': properties.get('country', 'Unknown'),
                'continent': properties.get('continent', 'Europe'),
                'classification': properties.get('classification', 0),
                'tempCropArea': properties.get('tempCropArea', 0.0),
                'cellArea': properties.get('cellArea', 0.0),
                'coordinates': coordinates
            }
        }
        
        # Track failed modalities for this point
        failed_modalities = []
        
        # Run all modalities for this point
        for modality in cfg.modalities:
            start_time = time.time()
            try:
                modality_df = eval(modality)(cfg[modality], point, log_level=cfg.log_level)
                sample_data[modality] = modality_df
                
                log.info(f'  {modality} for point {point_id}: {time.time() - start_time:.2f} seconds')
            except Exception as e:
                error_msg = f"{modality}: {str(e)}"
                log.error(f'  Error in {modality} for point {point_id}: {e}')
                failed_modalities.append(error_msg)
        
        # Save to zarr if at least some data was collected
        try:
            # Check if we have any modality data
            has_data = any(mod in sample_data for mod in cfg.modalities if mod != 'metadata')
            
            if has_data:
                add_sample_data(zarr_root, i, sample_data)
                log.info(f"Saved data for point {point_id} to zarr (batch index {i})")
                
                # If some modalities failed, log the point with partial failure
                if failed_modalities:
                    error_msg = "Partial failure: " + "; ".join(failed_modalities)
                    log_failed_point(point_id, error_file_path, error_msg)
                    log.warning(f"Point {point_id} saved with partial failures: {error_msg}")
            else:
                # No data collected at all - complete failure
                error_msg = "All modalities failed: " + "; ".join(failed_modalities) if failed_modalities else "No data collected"
                log_failed_point(point_id, error_file_path, error_msg)
                log.error(f"Point {point_id} failed completely: {error_msg}")
                failed_count += 1
                
        except Exception as e:
            error_msg = f"Zarr save error: {str(e)}"
            log.error(f"Error saving data for point {point_id} to zarr: {e}")
            log_failed_point(point_id, error_file_path, error_msg)
            failed_count += 1
        
        log.info(f"Completed point {point_id}")

    log.info(f'Total time taken for batch {batch_idx + 1}: {time.time() - start_time_total:.2f} seconds')
    log.info(f'Data saved to zarr dataset at {zarr_path}')
    log.info(f'Batch {batch_idx + 1} complete: {len(selected_features)} points processed')
    
    # Summary of failures
    if os.path.exists(error_file_path):
        with open(error_file_path, 'r') as f:
            failed_points_count = len([line for line in f if line.strip()])
        log.info(f'Failed points logged to: {error_file_path} ({failed_points_count} total failures)')
    else:
        log.info(f'No failed points - all {len(selected_features)} points succeeded!')


if __name__ == '__main__':
    main()