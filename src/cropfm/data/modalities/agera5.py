import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def agera5(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    """
    Extract AgERA5 meteorological data for multiple points in batch.
    AgERA5 - 0.1° x 0.1°
    
    Args:
        cfg: Hydra config containing agera5 parameters
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional parameters (can override config)
        
    Returns:
        dict: Dictionary mapping point_id to data dict
    """
    log.setLevel(cfg.log_level)
    
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date
    variables = list(cfg.variables)
    
    log.info(f"Starting batch {cfg.name} extraction for {point_fc.size().getInfo()} points")
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # Read in Image Collection and filter by date and location
    agera5_collection = (ee.ImageCollection('projects/climate-engine-pro/assets/ce-ag-era5-v2/daily')
                         .filterDate(start_date, end_date)
                         .filterBounds(point_fc.geometry().bounds()))  # Filter by bounding box
    
    # Sample pixel values from the image collection
    def sample_agera5_image(image):
        selected_image = image.select(variables)
        # Use sampleRegions for multiple points
        pixel_values = selected_image.sampleRegions(
            collection=point_fc,
            scale=10000,  # AgERA5 native resolution ~10km
            geometries=False  # Don't include geometries in output
        )
        
        def add_metadata(feature):
            return feature.set('date', image.date().format('YYYY-MM-dd'))
        
        return pixel_values.map(add_metadata)
    
    agera5_samples = agera5_collection.map(sample_agera5_image).flatten()
    
    log.debug(f'Number of AgERA5 samples: {agera5_samples.size().getInfo()}')
    
    # Get all features
    all_features = agera5_samples.getInfo()['features']
    
    # Group by point_id
    results = {point_id: {'data_rows': [], 'timestamps': []} for point_id in point_id_map.keys()}
    
    for feature in all_features:
        point_id = feature['properties'].get('point_id')
        if point_id and point_id in results:
            row = {k: v for k, v in feature['properties'].items() 
                   if k not in ['point_id']}  # Remove point_id from data
            results[point_id]['data_rows'].append(row)
            results[point_id]['timestamps'].append(feature['properties'].get('date'))
    
    # Convert to expected format
    batch_results = {}
    for point_id, data in results.items():
        if data['data_rows']:
            df = pd.DataFrame(data['data_rows'])
            data_columns = [col for col in variables if col in df.columns]
            
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': df[data_columns] if data_columns else df,
                'variable_names': variables,
                'timestamps': data['timestamps']
            }
        else:
            # No data for this point
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': pd.DataFrame(columns=variables),
                'variable_names': variables,
                'timestamps': []
            }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results
