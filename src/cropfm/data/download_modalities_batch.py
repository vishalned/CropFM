import hydra
import ee
import json
import numpy as np
from omegaconf import DictConfig
from modalities import (
    agera5,
    sentinel1,
    sentinel2,
    fapar,
    soil,
    elevation
)
import logging
import time
from save2zarr import create_zarr_dataset, add_sample_data
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

log = logging.getLogger(__name__)

def process_single_point_modality(args):
    """Process a single modality for a single point"""
    modality_name, cfg_modality, point, point_id, log_level = args
    
    try:
        start_time = time.time()
        modality_df = eval(modality_name)(cfg_modality, point, log_level=log_level)
        
        # Convert DataFrame to format expected by zarr
        if modality_name in ['agera5', 'sentinel1', 'sentinel2', 'fapar']:
            # Temporal modalities
            result = {
                'data': modality_df.drop(columns=['date','modality']).values if 'date' in modality_df.columns else modality_df.values,
                'dates': modality_df['date'].tolist() if 'date' in modality_df.columns else []
            }
        elif modality_name in ['soil', 'elevation']:
            # Static modalities - convert DataFrame to proper dictionary format
            if modality_name == 'elevation':
                # Handle elevation data structure
                elev_dict = {}
                for _, row in modality_df.iterrows():
                    elev_dict[row['variable']] = row['value']
                result = elev_dict
            elif modality_name == 'soil':
                # Handle soil data structure - group by variable and collect values by depth
                soil_dict = {}
                for var in ['clay', 'nitrogen', 'phh2o', 'soc']:
                    var_data = modality_df[modality_df['variable'] == var]
                    if len(var_data) > 0:
                        # Sort by depth layer order and extract values
                        depth_order = ['0-5cm', '5-15cm', '15-30cm']
                        values = []
                        for depth in depth_order:
                            depth_data = var_data[var_data['depth_layer'] == depth]
                            if len(depth_data) > 0:
                                values.append(depth_data.iloc[0]['value'])
                            else:
                                values.append(np.nan)  # Fill missing depths with NaN
                        soil_dict[var] = values
                result = soil_dict
        
        processing_time = time.time() - start_time
        log.info(f'  {modality_name} for point {point_id}: {processing_time:.2f} seconds')
        
        return modality_name, result, None
        
    except Exception as e:
        log.error(f'  Error in {modality_name} for point {point_id}: {e}')
        return modality_name, None, str(e)

def process_point_batch(args):
    """Process all modalities for a batch of points in parallel"""
    features_batch, cfg, zarr_root, batch_start_idx = args
    
    # Create all point-modality combinations for this batch
    tasks = []
    point_metadata = {}
    
    for i, feature in enumerate(features_batch):
        point_id = feature['id']
        coordinates = feature['geometry']['coordinates']
        point = ee.Geometry.Point(coordinates)
        
        # Store metadata for later use
        properties = feature.get('properties', {})
        point_metadata[i] = {
            'sample_id': point_id,
            'grid_id': properties.get('grid_id', int(point_id.split('_')[0]) if '_' in point_id else 0),
            'country': properties.get('country', 'Unknown'),
            'continent': properties.get('continent', 'Europe'),
            'classification': properties.get('classification', 0),
            'tempCropArea': properties.get('tempCropArea', 0.0),
            'cellArea': properties.get('cellArea', 0.0),
            'coordinates': coordinates
        }
        
        # Create tasks for all modalities for this point
        for modality in cfg.modalities:
            tasks.append((modality, cfg[modality], point, point_id, cfg.log_level))
    
    # Process all point-modality combinations in parallel
    results = {}
    max_workers = min(100, len(tasks))  # EE can handle ~100 concurrent requests
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_point_modality, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            modality_name, point_id = task[0], task[3]
            
            try:
                modality_name, result, error = future.result()
                
                if point_id not in results:
                    results[point_id] = {}
                
                if error is None:
                    results[point_id][modality_name] = result
                else:
                    log.error(f"Failed to process {modality_name} for {point_id}: {error}")
                    
            except Exception as e:
                log.error(f"Task failed: {e}")
    
    # Save results to zarr
    for i, feature in enumerate(features_batch):
        point_id = feature['id']
        
        if point_id in results:
            sample_data = {
                'metadata': point_metadata[i]
            }
            sample_data.update(results[point_id])
            
            try:
                add_sample_data(zarr_root, batch_start_idx + i, sample_data)
                log.info(f"Saved data for point {point_id} to zarr")
            except Exception as e:
                log.error(f"Error saving data for point {point_id} to zarr: {e}")

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml')
def main(cfg: DictConfig):
    ee.Initialize()
    log.setLevel(cfg.log_level)
    log.debug(f'CFG: {cfg}')
    log.info(f'Downloading modalities: {cfg.modalities}')
    
    # Load the GeoJSON file
    geojson_path = '/home/WUR/xiong015/xxglt/CropFM/data/grids/europe_sampled_points_merged.geojson'
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    log.info(f"Loaded GeoJSON with {len(geojson_data['features'])} features")
    
    # Select first 15 features for testing
    selected_features = geojson_data['features'][:60]
    
    log.info(f"Selected {len(selected_features)} points for testing")
    
    # Create zarr dataset
    zarr_path = '/home/WUR/xiong015/xxglt/CropFM/data/modalities/test_dataset.zarr'
    total_samples = len(selected_features)
    zarr_root = create_zarr_dataset(zarr_path, total_samples)
    log.info(f"Created zarr dataset at {zarr_path} with {total_samples} samples")
    
    start_time_total = time.time()
    
    # Process points in batches to manage memory and API limits
    batch_size = 60  # Adjust based on your system capacity and EE limits
    
    for batch_start in range(0, len(selected_features), batch_size):
        batch_end = min(batch_start + batch_size, len(selected_features))
        features_batch = selected_features[batch_start:batch_end]
        
        log.info(f"Processing batch {batch_start//batch_size + 1}: points {batch_start+1} to {batch_end}")
        
        # Process this batch
        process_point_batch((features_batch, cfg, zarr_root, batch_start))
    
    log.info(f'Total time taken for all modalities on all points: {time.time() - start_time_total:.2f} seconds')
    log.info(f'Data saved to zarr dataset at {zarr_path}')

if __name__ == '__main__':
    main()