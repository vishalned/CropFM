import pandas as pd
import ee
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

def fapar(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    """
    Extract FAPAR (Fraction of Absorbed Photosynthetically Active Radiation) data for multiple points in batch.
    FAPAR - 5.6km resolution
    
    Args:
        cfg: Hydra config containing fapar parameters
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
    
    log.info(f"Starting batch FAPAR extraction for {point_fc.size().getInfo()} points")
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # FAPAR/LAI collection
    collection = (ee.ImageCollection('NOAA/CDR/VIIRS/LAI_FAPAR/V1')
                 .filterDate(start_date, end_date)
                 .filterBounds(point_fc.geometry().bounds())  # Filter by bounding box
                 .select(variables))
    
    def sample_fapar_image(image):
        """Sample all points from a single FAPAR image"""
        # Use sampleRegions for multiple points
        pixel_values = image.sampleRegions(
            collection=point_fc,
            scale=5566,  # VIIRS LAI/FAPAR native resolution ~5.6km
            geometries=False  # Don't include geometries in output
        )
        
        def add_metadata(feature):
            return feature.set('date', image.date().format('YYYY-MM-dd'))
        
        return pixel_values.map(add_metadata)
    
    # Process all images in collection
    fapar_samples = collection.map(sample_fapar_image).flatten()
    
    log.debug(f'FAPAR pixel samples count: {fapar_samples.size().getInfo()}')
    
    # Get all features
    all_features = fapar_samples.getInfo()['features']
    
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
