import hydra
import ee
import json
import numpy as np
from omegaconf import DictConfig
from cropfm.data.modalities import ( # these are called using the eval function
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

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml')
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
    
    # Select first 1 features for testing
    selected_features = geojson_data['features'][:200]
    
    log.info(f"Selected {len(selected_features)} points for testing")
    
    # Create zarr dataset
    zarr_path = cfg.zarr_path
    total_samples = len(selected_features)
    zarr_root = create_zarr_dataset(zarr_path, total_samples)
    log.info(f"Created zarr dataset at {zarr_path} with {total_samples} samples")
    
    # Batch size for GEE processing
    batch_size = cfg.get('batch_size', 50)
    log.info(f"Using batch size: {batch_size} points per GEE batch")
    
    start_time_total = time.time()
    
    # Process points in batches
    for batch_start in range(0, len(selected_features), batch_size):
        batch_end = min(batch_start + batch_size, len(selected_features))
        batch_features = selected_features[batch_start:batch_end]
        
        log.info(f"Processing GEE batch {batch_start//batch_size + 1}/{(len(selected_features)-1)//batch_size + 1} "
                 f"(points {batch_start} to {batch_end-1})")
        
        # Create FeatureCollection for this batch
        point_features = []
        for feature in batch_features:
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
                    batch_features, 
                    log_level=cfg.log_level
                )
                batch_results[modality] = modality_results
                log.info(f'  {modality} batch ({len(batch_features)} points): {time.time() - start_time:.2f} seconds')
            except Exception as e:
                log.error(f'  Error in {modality} batch: {e}')
                raise  # Re-raise to fail fast
        
        # Save batch results to zarr
        for i, feature in enumerate(batch_features):
            point_id = feature['id']
            sample_idx = batch_start + i
            
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
            
            # Add modality data from batch results
            for modality in cfg.modalities:
                if modality in batch_results and point_id in batch_results[modality]:
                    sample_data[modality] = batch_results[modality][point_id]
            
            # Save to zarr
            try:
                add_sample_data(zarr_root, sample_idx, sample_data)
                log.info(f"Saved data for point {point_id} to zarr (index {sample_idx})")
            except Exception as e:
                log.error(f"Error saving point {point_id}: {e}")

    log.info(f'Total time taken for all modalities on all points: {time.time() - start_time_total:.2f} seconds')
    log.info(f'Data saved to zarr dataset at {zarr_path}')


if __name__ == '__main__':
    main()