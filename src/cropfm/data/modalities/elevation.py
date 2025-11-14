import ee
import pandas as pd
import logging
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def elevation(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract elevation and slope data for a point using ASTER GDEM.
    ASTER GDEM - 30m resolution
    
    Args:
        cfg: Hydra config containing elevation parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted elevation and slope data with metadata
    """
    
    # Get config values with optional overrides from kwargs
    bands = list(cfg.bands)
    
    log.info(f"Starting elevation extraction")
    
    # Load ASTER GDEM
    elevation_image = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').select(bands).float()
    
    # Sample elevation at the point
    elevation_sample = elevation_image.sample(
        region=point,
        scale=10,
        numPixels=1
    )
    
    # Get elevation data
    elevation_data = elevation_sample.getInfo()
    
    elevation_value = np.nan
    slope_value = np.nan
    
    if elevation_data['features']:
        elevation_value = elevation_data['features'][0]['properties'].get(bands[0])
        
        if elevation_value is not None:
            # Calculate and sample slope
            slope_image = ee.Terrain.slope(elevation_image)
            
            slope_sample = slope_image.sample(
                region=point,
                scale=10,
                numPixels=1
            )
            
            slope_data = slope_sample.getInfo()
            
            if slope_data['features']:
                slope_value = slope_data['features'][0]['properties'].get('slope')
                
                if slope_value is not None:
                    slope_value = slope_data['features'][0]['properties'].get('slope')

    df = pd.DataFrame({
        'elevation': elevation_value,
        'slope': slope_value,
    }, index=[0])

    elevation_data_dict = {
        'modality': cfg.name,
        'data': df,
        'variable_names': ['elevation', 'slope'],
    }

    log.info(f"Successfully extracted {len(elevation_data_dict['data'])} elevation observations")
    
    return elevation_data_dict