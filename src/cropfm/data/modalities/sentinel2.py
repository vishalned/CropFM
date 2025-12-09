import ee
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

def sentinel2(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    '''
    Extract Sentinel-2 data for multiple points in batch.
    S2 - 10m resolution

    Args:
        cfg: The configuration for the Sentinel-2 data
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional arguments
    
    Returns:
        dict: Dictionary mapping point_id to data dict
    '''
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting batch {cfg.name} extraction for {point_fc.size().getInfo()} points")
    
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date
    s2_band_names = list(cfg.bands)
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # Sentinel-2 collection
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filterBounds(point_fc.geometry().bounds())  # Filter by bounding box
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cfg.cloud_cover_threshold)))
    
    def sample_s2_image(image):
        """Sample all points from a single image"""
        selected_image = image.select(s2_band_names)
        # Use sampleRegions for multiple points
        pixel_values = selected_image.sampleRegions(
            collection=point_fc,
            scale=10,
            geometries=False  # Don't include geometries in output
        )
        
        def add_metadata(feature):
            return feature.set('date', image.date().format('YYYY-MM-dd'))
        
        return pixel_values.map(add_metadata)
    
    # Process all images
    s2_samples = s2_collection.map(sample_s2_image).flatten()
    
    log.debug(f'Number of Sentinel-2 samples: {s2_samples.size().getInfo()}')
    
    # Get all features
    all_features = s2_samples.getInfo()['features']
    
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
            data_columns = [col for col in s2_band_names if col in df.columns]
            
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': df[data_columns] if data_columns else df,
                'variable_names': s2_band_names,
                'timestamps': data['timestamps']
            }
        else:
            # No data for this point
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': pd.DataFrame(columns=s2_band_names),
                'variable_names': s2_band_names,
                'timestamps': []
            }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results
