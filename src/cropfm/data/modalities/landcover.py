import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def landcover(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract ESA WorldCover land cover data for a point.
    ESA WorldCover - 10m resolution

    Args:
        cfg: Hydra config containing landcover parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted land cover data with metadata
    """
    
    # Get config values with optional overrides from kwargs
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting {cfg.name} extraction")
    
    # ESA WorldCover collections for both years
    landcover_v100 = ee.ImageCollection("ESA/WorldCover/v100")  # 2020
    landcover_v200 = ee.ImageCollection("ESA/WorldCover/v200")  # 2021
    
    all_data = []
    
    # Get variables and years from config
    variables = list(cfg.variables)
    years = list(cfg.years)
    
    # Process each year
    for year in years:
        if year == 2020:
            landcover_image = landcover_v100.first()
        elif year == 2021:
            landcover_image = landcover_v200.first()
        else:
            log.warning(f"Year {year} not supported, skipping")
            continue
            
        # Process each variable (typically just 'Map')
        for variable in variables:
            pixel_value = landcover_image.select(variable).sample(
                region=point,
                scale=10,
                numPixels=1
            )
            
            sample_data = pixel_value.getInfo()
            
            if sample_data['features'] and sample_data['features'][0]['properties'][variable] is not None:
                value = sample_data['features'][0]['properties'][variable]
                
                all_data.append({
                    'variable': f'landcover_{variable.lower()}',
                    'value': value,
                    'year': year,
                    'modality': cfg.name
                })
    
    df = pd.DataFrame(all_data)
    log.info(f"Successfully extracted {len(df)} land cover observations")
    
    return df