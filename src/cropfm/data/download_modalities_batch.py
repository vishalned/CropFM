import hydra
import ee
import json
import numpy as np
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
from cropfm.data.utils import save2zarr
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import urllib3
import zarr

log = logging.getLogger(__name__)

# Suppress urllib3 connection pool warnings
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# Thread-safe file writing lock
failed_points_lock = threading.Lock()

def log_failed_point(point_id, error_file_path, error_msg=None):
    """Thread-safe function to log failed point IDs"""
    with failed_points_lock:
        with open(error_file_path, 'a') as f:
            if error_msg:
                f.write(f"{point_id}\t{error_msg}\n")
            else:
                f.write(f"{point_id}\n")

def process_single_point(args):
    """Process ALL modalities for a single point"""
    feature, cfg, zarr_root, sample_idx, error_file_path = args
    
    point_id = feature['id']
    coordinates = feature['geometry']['coordinates']
    point = ee.Geometry.Point(coordinates)
    
    # Get metadata
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
    
    # Process all modalities for this point sequentially
    failed_modalities = []
    for modality in cfg.modalities:
        start_time = time.time()
        try:
            modality_result = eval(modality)(cfg[modality], point, log_level=cfg.log_level)
            sample_data[modality] = modality_result
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
            save2zarr.add_sample_data(zarr_root, sample_idx, sample_data)
            log.info(f"Saved data for point {point_id} to zarr")
            
            # If some modalities failed, still log the point with partial failure
            if failed_modalities:
                error_msg = "Partial failure: " + "; ".join(failed_modalities)
                log_failed_point(point_id, error_file_path, error_msg)
            
            return True, point_id
        else:
            # No data collected at all - complete failure
            error_msg = "All modalities failed: " + "; ".join(failed_modalities) if failed_modalities else "No data collected"
            log_failed_point(point_id, error_file_path, error_msg)
            return False, point_id
            
    except Exception as e:
        error_msg = f"Zarr save error: {str(e)}"
        log.error(f"Error saving data for point {point_id} to zarr: {e}")
        log_failed_point(point_id, error_file_path, error_msg)
        return False, point_id

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml', version_base=None)
def main(cfg: DictConfig):
    ee.Initialize()
    log.setLevel(cfg.log_level)
    log.debug(f'CFG: {cfg}')
    log.info(f'Downloading modalities: {cfg.modalities}')
    
    # Load the GeoJSON file
    geojson_path = cfg.points_geojson_file
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    log.info(f"Loaded GeoJSON with {len(geojson_data['features'])} features")
    
    # Select features for processing
    selected_features = geojson_data['features']  # Or use slicing like [:1000] for testing
    
    log.info(f"Selected {len(selected_features)} points for processing")
    
    # Setup paths
    zarr_path = cfg.zarr_path
    error_file_path = cfg.get('error_log_path', os.path.join(os.path.dirname(zarr_path), 'failed_points.txt'))
    
    # Resume functionality
    start_batch = cfg.get('resume_from_batch', 0)  # Default: start from beginning
    batch_size = cfg.get('batch_size', 40)
    
    if start_batch > 0:
        log.info(f"RESUMING from batch {start_batch} (batch_size={batch_size})")
        log.info(f"This will start at point index {start_batch * batch_size}")
    
    # Create or open zarr dataset
    total_samples = len(selected_features)
    
    # Check if zarr already exists (for resume mode)
    if os.path.exists(zarr_path) and start_batch > 0:
        log.info(f"Opening existing zarr dataset at {zarr_path} (resume mode)")
        zarr_root = zarr.open_group(zarr_path, mode='r+')  # Read-write mode
        
        # Verify the size matches
        existing_size = zarr_root.attrs.get('total_samples', 0)
        if existing_size != total_samples:
            log.warning(f"Existing zarr has {existing_size} samples, but we have {total_samples} points. "
                       f"Continuing anyway (will overwrite from batch {start_batch})")
    else:
        if os.path.exists(zarr_path) and start_batch == 0:
            log.warning(f"Zarr dataset already exists at {zarr_path}. "
                       f"Set resume_from_batch=0 to recreate, or use resume_from_batch>0 to resume")
        log.info(f"Creating new zarr dataset at {zarr_path} with {total_samples} samples")
        zarr_root = save2zarr.create_zarr_dataset(zarr_path, total_samples)
    
    # Clear error log file if starting fresh (batch 0)
    if start_batch == 0 and os.path.exists(error_file_path):
        log.info(f"Clearing existing error log: {error_file_path}")
        os.remove(error_file_path)
    
    # Create error log directory if it doesn't exist
    os.makedirs(os.path.dirname(error_file_path) if os.path.dirname(error_file_path) else '.', exist_ok=True)
    
    start_time_total = time.time()
    
    # Process points in batches
    total_batches = (len(selected_features) + batch_size - 1) // batch_size
    
    for batch_num in range(start_batch, total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(selected_features))
        features_batch = selected_features[batch_start:batch_end]
        
        log.info(f"Processing batch {batch_num + 1}/{total_batches}: points {batch_start+1} to {batch_end} (indices {batch_start} to {batch_end-1})")
        
        # Create tasks: one task per point
        tasks = [
            (feature, cfg, zarr_root, batch_start + i, error_file_path)
            for i, feature in enumerate(features_batch)
        ]
        
        # Process points in parallel (max 40 concurrent points due to GEE limit)
        max_workers = min(40, len(tasks))
        
        failed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_point, task) for task in tasks]
            
            # Wait for all to complete and track failures
            for future in concurrent.futures.as_completed(futures):
                try:
                    success, point_id = future.result()
                    if not success:
                        failed_count += 1
                except Exception as e:
                    log.error(f"Unexpected error in future: {e}")
                    failed_count += 1
        
        log.info(f"Batch {batch_num + 1} complete: {len(features_batch) - failed_count}/{len(features_batch)} points succeeded, "
                 f"{failed_count} failed")
        
        # Optional: break for testing
        # break
    
    # Summary
    failed_points_count = 0
    if os.path.exists(error_file_path):
        with open(error_file_path, 'r') as f:
            failed_points_count = len([line for line in f if line.strip()])
    
    log.info(f'Total time taken: {time.time() - start_time_total:.2f} seconds')
    log.info(f'Data saved to zarr dataset at {zarr_path}')
    log.info(f'Failed points logged to: {error_file_path} ({failed_points_count} total failures)')

if __name__ == '__main__':
    main()