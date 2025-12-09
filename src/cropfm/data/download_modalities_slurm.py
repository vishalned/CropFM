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
    
    # Batch size for GEE processing
    gee_batch_size = cfg.get('batch_size', 50)
    log.info(f"Using GEE batch size: {gee_batch_size} points per GEE batch")
    
    # Process points in GEE batches
    for gee_batch_start in range(0, len(selected_features), gee_batch_size):
        gee_batch_end = min(gee_batch_start + gee_batch_size, len(selected_features))
        gee_batch_features = selected_features[gee_batch_start:gee_batch_end]
        
        log.info(f"Processing GEE batch {gee_batch_start//gee_batch_size + 1}/{(len(selected_features)-1)//gee_batch_size + 1} "
                 f"(points {gee_batch_start} to {gee_batch_end-1})")
        
        # Create FeatureCollection for this batch
        point_features = []
        for feature in gee_batch_features:
            point_id = feature['id']
            coordinates = feature['geometry']['coordinates']
            point = ee.Geometry.Point(coordinates)
            # Add point_id as a property for later matching
            point_feature = ee.Feature(point, {'point_id': point_id})
            point_features.append(point_feature)
        
        point_fc = ee.FeatureCollection(point_features)
        
        # Process all modalities for this batch
        batch_results = {}
        for modality in cfg.modalities:
            start_time = time.time()
            try:
                modality_func = eval(modality)
                modality_results = modality_func(
                    cfg[modality], 
                    point_fc, 
                    gee_batch_features, 
                    log_level=cfg.log_level
                )
                batch_results[modality] = modality_results
                log.info(f'  {modality} batch ({len(gee_batch_features)} points): {time.time() - start_time:.2f} seconds')
            except Exception as e:
                log.error(f'  Error in {modality} batch: {e}')
                raise  # Re-raise to fail fast
        
        # Save batch results to zarr
        for i, feature in enumerate(gee_batch_features):
            point_id = feature['id']
            sample_idx = gee_batch_start + i
            global_idx = batch_start + sample_idx
            
            # Initialize sample data dictionary
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
                    'coordinates': feature['geometry']['coordinates']
                }
            }
            
            # Track failed modalities for this point
            failed_modalities = []
            
            # Add modality data from batch results
            for modality in cfg.modalities:
                if modality in batch_results and point_id in batch_results[modality]:
                    sample_data[modality] = batch_results[modality][point_id]
                else:
                    failed_modalities.append(f"{modality}: No data")
            
            # Save to zarr if at least some data was collected
            try:
                # Check if we have any modality data
                has_data = any(mod in sample_data for mod in cfg.modalities if mod != 'metadata')
                
                if has_data:
                    add_sample_data(zarr_root, sample_idx, sample_data)
                    log.info(f"Saved data for point {point_id} to zarr (batch index {sample_idx}, global {global_idx})")
                    
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