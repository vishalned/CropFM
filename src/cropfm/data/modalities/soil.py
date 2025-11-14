import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def soil(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract SoilGrids soil properties data for a point.
    SoilGrids - 250m resolution

    Args:
        cfg: Hydra config containing soil parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted soil properties data with metadata
    """
    
    # Get config values with optional overrides from kwargs
    log.setLevel(cfg.log_level)
    variables = list(cfg.variables)
    depth_layers = list(cfg.depth_layers)
    
    log.info(f"Starting SoilGrids extraction")
    
    # SoilGrids asset mapping
    soil_assets = {
        'clay': 'projects/soilgrids-isric/clay_mean',
        'nitrogen': 'projects/soilgrids-isric/nitrogen_mean', 
        'phh2o': 'projects/soilgrids-isric/phh2o_mean',
        'soc': 'projects/soilgrids-isric/soc_mean'
    }
    
    soil_data = {}
    
    for var in variables:      
        log.debug(f"Processing soil property: {var}")
        
        # Load the soil property image
        soil_image = ee.Image(soil_assets[var])

        band_names = [f"{var}_{depth}_mean" for depth in depth_layers]

        pixel_value = soil_image.select(band_names).sample(
            region=point,
            scale=250,
            numPixels=1
        )

        sample_data = pixel_value.getInfo()

        if sample_data['features']:
            soil_data[var] = [sample_data['features'][0]['properties'][band_name] for band_name in band_names]
            
    df = pd.DataFrame(soil_data)
    soil_data_dict = {
        'modality': cfg.name,
        'data': df,
        'variable_names': variables,
    }

    log.info(f"Successfully extracted {len(soil_data_dict['data'])} soil property observations")
    
    return soil_data_dict
