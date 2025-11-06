import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def worldcereal_cropcalender(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    '''
    Extract WorldCereal crop calendar data for a point from AEZ polygons.
        - 2021 model products
    
    Extracts crop calendar properties including start of season (SOS) and 
    end of season (EOS) dates for:
    - Maize (main season)
    - Winter cereals
    - Spring cereals
    
    Args:
        cfg: Hydra config containing worldcereal_cropcalender parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted crop calendar data with AEZ ID and seasonal dates
    '''
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting {cfg.name} extraction")
    
    # Load AEZ table (FeatureCollection) - same as used in worldcereal cropmask
    aez_table = ee.FeatureCollection('ESA/WorldCereal/AEZ/v100')
    
    # Filter to get the AEZ containing the point
    aez_for_point = aez_table.filterBounds(point)
    
    # Define crop calendar properties of interest
    calendar_properties = [
        'aez_id',
        'tc-maize-main_sos', 'tc-maize-main_eos',
        'tc-wintercereals_sos', 'tc-wintercereals_eos',
        'tc-springcereals_sos', 'tc-springcereals_eos'
    ]
    
    # Get the crop calendar feature for the AEZ polygon containing the point
    # Select only the properties we need
    crop_calendar_feature = ee.Feature(aez_for_point.first()).select(calendar_properties)
    
    # Get properties as a dictionary
    calendar_dict = crop_calendar_feature.toDictionary().getInfo()
    
    log.debug(f'Crop calendar for AEZ containing point: {calendar_dict}')
    
    # Prepare data for DataFrame
    all_data = []
    
    # Add the calendar data with modality information
    calendar_data = calendar_dict.copy()
    calendar_data['modality'] = cfg.name
    all_data.append(calendar_data)
    
    df = pd.DataFrame(all_data)
    
    log.info(f"Successfully extracted {len(df)} WorldCereal crop calendar observations")
    
    return df