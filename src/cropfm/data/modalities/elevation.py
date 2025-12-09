import ee
import pandas as pd
import logging
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def elevation(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    """
    Extract elevation and slope data for multiple points in batch using ASTER GDEM.
    ASTER GDEM - 30m resolution
    
    Args:
        cfg: Hydra config containing elevation parameters
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional parameters (can override config)
        
    Returns:
        dict: Dictionary mapping point_id to data dict
    """
    log.setLevel(cfg.log_level)
    bands = list(cfg.bands)
    
    log.info(f"Starting batch elevation extraction for {point_fc.size().getInfo()} points")
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # Load ASTER GDEM
    elevation_image = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').select(bands).float()
    
    # Calculate slope
    slope_image = ee.Terrain.slope(elevation_image)
    
    # Combine elevation and slope into one image
    combined_image = elevation_image.addBands(slope_image)
    
    # Sample all points at once
    samples = combined_image.sampleRegions(
        collection=point_fc,
        scale=10,
        geometries=False
    )
    
    # Get all features
    all_features = samples.getInfo()['features']
    
    # Group by point_id
    batch_results = {}
    for point_id in point_id_map.keys():
        batch_results[point_id] = {
            'elevation': np.nan,
            'slope': np.nan
        }
    
    for feature in all_features:
        point_id = feature['properties'].get('point_id')
        if point_id and point_id in batch_results:
            props = feature['properties']
            elevation_value = props.get(bands[0])
            slope_value = props.get('slope')
            
            if elevation_value is not None:
                batch_results[point_id]['elevation'] = elevation_value
            if slope_value is not None:
                batch_results[point_id]['slope'] = slope_value
    
    # Convert to expected format
    for point_id, data in batch_results.items():
        df = pd.DataFrame({
            'elevation': [data['elevation']],
            'slope': [data['slope']],
        }, index=[0])
        
        batch_results[point_id] = {
            'modality': cfg.name,
            'data': df,
            'variable_names': ['elevation', 'slope'],
        }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results