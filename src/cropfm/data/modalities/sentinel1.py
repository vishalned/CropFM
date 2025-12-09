import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def sentinel1(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    '''
    Extract Sentinel-1 data for multiple points in batch.
    S1 - 10m resolution

    Args:
        cfg: The configuration for the Sentinel-1 data
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional arguments (can override config).
    
    Returns:
        dict: Dictionary mapping point_id to data dict
    '''
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting batch {cfg.name} extraction for {point_fc.size().getInfo()} points")
    
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date
    s1_band_names = list(cfg.bands)
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # Sentinel-1 collection
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterDate(start_date, end_date)
                     .filterBounds(point_fc.geometry().bounds())  # Filter by bounding box
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW')))
    
    def sample_s1_image(image):
        """Sample all points from a single image"""
        selected_image = image.select(s1_band_names)
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
    s1_samples = s1_collection.map(sample_s1_image).flatten()
    
    log.debug(f'Number of Sentinel-1 samples: {s1_samples.size().getInfo()}')
    
    # Get all features
    all_features = s1_samples.getInfo()['features']
    
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
            data_columns = [col for col in s1_band_names if col in df.columns]
            
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': df[data_columns] if data_columns else df,
                'variable_names': s1_band_names,
                'timestamps': data['timestamps']
            }
        else:
            # No data for this point
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': pd.DataFrame(columns=s1_band_names),
                'variable_names': s1_band_names,
                'timestamps': []
            }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results
