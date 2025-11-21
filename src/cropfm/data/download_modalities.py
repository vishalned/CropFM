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
    selected_features = geojson_data['features'][:2]
    
    log.info(f"Selected {len(selected_features)} points for testing")
    
    # Create zarr dataset
    zarr_path = cfg.zarr_path
    total_samples = len(selected_features)
    zarr_root = create_zarr_dataset(zarr_path, total_samples)
    log.info(f"Created zarr dataset at {zarr_path} with {total_samples} samples")
    
    start_time_total = time.time()
    # Process each point
    for i, feature in enumerate(selected_features):
        point_id = feature['id']
        coordinates = feature['geometry']['coordinates']
        point = ee.Geometry.Point(coordinates)
        
        log.info(f"Processing point {i+1}/{len(selected_features)} - ID: {point_id}, Coordinates: {coordinates}")
        
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
        
        # Run all modalities for this point
        for modality in cfg.modalities:
            start_time = time.time()
            try:
                modality_df = eval(modality)(cfg[modality], point, log_level=cfg.log_level)
                sample_data[modality] = modality_df
                
                log.info(f'  {modality} for point {point_id}: {time.time() - start_time:.2f} seconds')
            except Exception as e:
                log.error(f'  Error in {modality} for point {point_id}: {e}')
        add_sample_data(zarr_root, i, sample_data)
        log.info(f"Saved data for point {point_id} to zarr")
        log.info(f"Completed point {point_id}")

    log.info(f'Total time taken for all modalities on all points: {time.time() - start_time_total:.2f} seconds')
    log.info(f'Data saved to zarr dataset at {zarr_path}')


if __name__ == '__main__':
    main()