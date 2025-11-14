import hydra
import ee
import json
import numpy as np
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
import urllib3

log = logging.getLogger(__name__)

# Suppress urllib3 connection pool warnings
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

def process_single_point(args):
    """Process ALL modalities for a single point"""
    feature, cfg, zarr_root, sample_idx = args
    
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
    for modality in cfg.modalities:
        start_time = time.time()
        try:
            modality_result = eval(modality)(cfg[modality], point, log_level=cfg.log_level)
            sample_data[modality] = modality_result
            log.info(f'  {modality} for point {point_id}: {time.time() - start_time:.2f} seconds')
        except Exception as e:
            log.error(f'  Error in {modality} for point {point_id}: {e}')
    
    # Save to zarr
    try:
        save2zarr.add_sample_data(zarr_root, sample_idx, sample_data)
        log.info(f"Saved data for point {point_id} to zarr")
        return True
    except Exception as e:
        log.error(f"Error saving data for point {point_id} to zarr: {e}")
        return False

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml')
def main(cfg: DictConfig):
    # Increase urllib3 connection pool size (must be before ee.Initialize())
    # urllib3.poolmanager.PoolManager.maxsize = 50
    
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
    
    # Create zarr dataset
    zarr_path = cfg.zarr_path
    total_samples = len(selected_features)
    zarr_root = save2zarr.create_zarr_dataset(zarr_path, total_samples)
    log.info(f"Created zarr dataset at {zarr_path} with {total_samples} samples")
    
    start_time_total = time.time()
    
    # Process points in batches to manage memory
    batch_size = 40

    for batch_start in range(0, len(selected_features), batch_size):
        batch_end = min(batch_start + batch_size, len(selected_features))
        features_batch = selected_features[batch_start:batch_end]
        
        log.info(f"Processing batch {batch_start//batch_size + 1}: points {batch_start+1} to {batch_end}")
        
        # Create tasks: one task per point
        tasks = [
            (feature, cfg, zarr_root, batch_start + i)
            for i, feature in enumerate(features_batch)
        ]
        
        # Process points in parallel (max 40 concurrent points due to GEE limit)
        # Each thread processes all modalities for one point sequentially
        max_workers = min(40, len(tasks))  # 40 points in parallel
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_point, task) for task in tasks]
            
            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Wait for completion, handle errors if needed

        break ######## For testing only #########
    
    log.info(f'Total time taken for all modalities on all points: {time.time() - start_time_total:.2f} seconds')
    log.info(f'Data saved to zarr dataset at {zarr_path}')

if __name__ == '__main__':
    main()