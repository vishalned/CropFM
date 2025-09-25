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
    
    all_data = []
    
    for var in variables:      
        log.debug(f"Processing soil property: {var}")
        
        # Load the soil property image
        soil_image = ee.Image(soil_assets[var])
        
        for depth in depth_layers:
            band_name = f"{var}_{depth}_mean"
            
            # Sample the soil property at the point
            pixel_value = soil_image.select(band_name).sample(
                region=point,
                scale=250,  # SoilGrids native resolution is 250m
                numPixels=1
            )
            
            # Extract the value
            sample_data = pixel_value.getInfo()
            
            if sample_data['features'] and sample_data['features'][0]['properties'][band_name] is not None:
                value = sample_data['features'][0]['properties'][band_name]
                
                all_data.append({
                    'property': var,
                    'depth_layer': depth,
                    'value': value,
                    'modality': cfg.name
                })
    
    df = pd.DataFrame(all_data)
    log.info(f"Successfully extracted {len(df)} soil property observations")
    
    return df
