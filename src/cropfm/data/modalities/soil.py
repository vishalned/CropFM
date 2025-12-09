import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def soil(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    """
    Extract SoilGrids soil properties data for multiple points in batch.
    SoilGrids - 250m resolution

    Args:
        cfg: Hydra config containing soil parameters
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional parameters (can override config)
        
    Returns:
        dict: Dictionary mapping point_id to data dict
    """
    log.setLevel(cfg.log_level)
    variables = list(cfg.variables)
    depth_layers = list(cfg.depth_layers)
    
    log.info(f"Starting batch SoilGrids extraction for {point_fc.size().getInfo()} points")
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # SoilGrids asset mapping
    soil_assets = {
        'clay': 'projects/soilgrids-isric/clay_mean',
        'nitrogen': 'projects/soilgrids-isric/nitrogen_mean', 
        'phh2o': 'projects/soilgrids-isric/phh2o_mean',
        'soc': 'projects/soilgrids-isric/soc_mean'
    }
    
    # Collect all band names we need
    all_band_names = []
    for var in variables:
        band_names = [f"{var}_{depth}_mean" for depth in depth_layers]
        all_band_names.extend(band_names)
    
    # Create a composite image with all needed bands
    soil_images = []
    for var in variables:
        soil_image = ee.Image(soil_assets[var])
        band_names = [f"{var}_{depth}_mean" for depth in depth_layers]
        soil_images.append(soil_image.select(band_names))
    
    # Combine all soil images
    composite_image = ee.Image.cat(soil_images)
    
    # Sample all points at once
    samples = composite_image.sampleRegions(
        collection=point_fc,
        scale=250,
        geometries=False
    )
    
    # Get all features
    all_features = samples.getInfo()['features']
    
    # Group by point_id and organize data
    batch_results = {}
    for point_id in point_id_map.keys():
        batch_results[point_id] = {'soil_data': {}}
    
    for feature in all_features:
        point_id = feature['properties'].get('point_id')
        if point_id and point_id in batch_results:
            props = feature['properties']
            # Organize by variable
            for var in variables:
                band_names = [f"{var}_{depth}_mean" for depth in depth_layers]
                batch_results[point_id]['soil_data'][var] = [
                    props.get(band_name) for band_name in band_names
                ]
    
    # Convert to expected format
    for point_id, data in batch_results.items():
        if data['soil_data']:
            df = pd.DataFrame(data['soil_data'])
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': df,
                'variable_names': variables,
            }
        else:
            # No data for this point
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': pd.DataFrame(columns=variables),
                'variable_names': variables,
            }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results
