import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def worldcereal_cropcalender(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    '''
    Extract WorldCereal crop calendar data for multiple points in batch from AEZ polygons.
        - 2021 model products
    
    Extracts crop calendar properties including start of season (SOS) and 
    end of season (EOS) dates for:
    - Maize (main season)
    - Winter cereals
    - Spring cereals
    
    Args:
        cfg: Hydra config containing worldcereal_cropcalender parameters
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional parameters (can override config)
        
    Returns:
        dict: Dictionary mapping point_id to data dict
    '''
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting batch {cfg.name} extraction for {point_fc.size().getInfo()} points")
    
    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
    # Load AEZ table (FeatureCollection)
    aez_table = ee.FeatureCollection('ESA/WorldCereal/AEZ/v100')
    
    # Define crop calendar properties of interest
    calendar_properties = [
        'aez_id',
        'tc-maize-main_sos', 'tc-maize-main_eos',
        'tc-wintercereals_sos', 'tc-wintercereals_eos',
        'tc-springcereals_sos', 'tc-springcereals_eos'
    ]
    
    # Get bounding box of all points for efficient filtering
    bounds = point_fc.geometry().bounds()
    aez_in_bounds = aez_table.filterBounds(bounds)
    
    # For each point, find intersecting AEZ and extract calendar properties
    def add_calendar_properties(feature):
        point_geom = feature.geometry()
        # Find AEZ polygons that intersect this point
        intersecting_aez = aez_in_bounds.filterBounds(point_geom)
        
        # Get the first intersecting AEZ
        aez_feature = intersecting_aez.first()
        
        # Extract each property individually using .get() instead of .select()
        result_feature = feature
        for prop in calendar_properties:
            prop_value = ee.Algorithms.If(
                intersecting_aez.size().gt(0),
                aez_feature.get(prop),
                None
            )
            result_feature = result_feature.set(prop, prop_value)
        
        return result_feature
    
    point_fc_with_calendar = point_fc.map(add_calendar_properties)
    
    # Get all features with calendar data
    all_features = point_fc_with_calendar.getInfo()['features']
    
    # Group by point_id
    batch_results = {}
    for point_id in point_id_map.keys():
        batch_results[point_id] = {}
    
    for feature in all_features:
        point_id = feature['properties'].get('point_id')
        if point_id and point_id in batch_results:
            props = feature['properties']
            # Extract calendar properties
            calendar_dict = {prop: props.get(prop) for prop in calendar_properties if prop in props}
            batch_results[point_id] = calendar_dict
    
    # Convert to expected format
    for point_id, calendar_dict in batch_results.items():
        if calendar_dict:
            df = pd.DataFrame(calendar_dict, index=[0])
            data_columns = [col for col in calendar_properties if col in df.columns]
            
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': df[data_columns] if data_columns else df,
                'variable_names': calendar_properties,
            }
        else:
            # No data for this point
            batch_results[point_id] = {
                'modality': cfg.name,
                'data': pd.DataFrame(columns=calendar_properties),
                'variable_names': calendar_properties,
            }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results