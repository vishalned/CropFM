import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def agera5(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract AgERA5 meteorological data for a point.
    AgERA5 - 0.1° x 0.1°
    
    Args:
        cfg: Hydra config containing agera5 parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted AgERA5 data with metadata
    """
    log.setLevel(cfg.log_level)
    
    # Get config values with optional overrides from kwargs
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date
    variables = list(cfg.variables)
    
    log.info(f"Starting {cfg.name} extraction")
    
    # Read in Image Collection and filter by date and location
    agera5_collection = ee.ImageCollection('projects/climate-engine-pro/assets/ce-ag-era5-v2/daily') \
        .filterDate(start_date, end_date) \
        .filterBounds(point)
    
    # Sample pixel values from the image collection
    def sample_agera5_image(image):
        selected_image = image.select(variables)
        pixel_value = selected_image.sample(
            region=point,
            scale=10000,  # AgERA5 native resolution ~10km
            numPixels=1
        )
        
        def add_metadata(feature):
            return feature.set('date', image.date().format('YYYY-MM-dd'))
        
        return pixel_value.map(add_metadata)
    
    agera5_samples = agera5_collection.map(sample_agera5_image).flatten()
    
    # Print sample information
    log.debug(f'Number of AgERA5 samples: {agera5_samples.size().getInfo()}')
    
    # Convert to pandas DataFrame
    agera5_samples_data = agera5_samples.getInfo()
    agera5_features = agera5_samples_data['features']
    
    agera5_data_rows = []
    for feature in agera5_features:
        row = feature['properties'].copy()
        agera5_data_rows.append(row)
    

    agera5_data_dict = {
        'modality': cfg.name,
        'data': pd.DataFrame(agera5_data_rows),
        'variable_names': variables,
        'timestamps': [feature['properties']['date'] for feature in agera5_features]
    }

    data_columns = [col for col in variables if col in agera5_data_dict['data'].columns]
    agera5_data_dict['data'] = agera5_data_dict['data'][data_columns]

    log.info(f"Successfully extracted {len(agera5_data_dict['data'])} AgERA5 observations")

    return agera5_data_dict
