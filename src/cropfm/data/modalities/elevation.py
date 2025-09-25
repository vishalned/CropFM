import ee
import pandas as pd
import logging
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
        scale=30,
        numPixels=1
    )
    
    # Get elevation data
    elevation_data = elevation_sample.getInfo()
    
    all_data = []
    
    if elevation_data['features']:
        elevation_value = elevation_data['features'][0]['properties'].get(bands[0])
        
        if elevation_value is not None:
            all_data.append({
                'variable': 'elevation',
                'value': elevation_value,
                'modality': cfg.name,
            })
            
            log.debug(f"Elevation: {elevation_value} meters")
            
            # Calculate and sample slope
            slope_image = ee.Terrain.slope(elevation_image)
            
            slope_sample = slope_image.sample(
                region=point,
                scale=30,
                numPixels=1
            )
            
            slope_data = slope_sample.getInfo()
            
            if slope_data['features']:
                slope_value = slope_data['features'][0]['properties'].get('slope')
                
                if slope_value is not None:
                    all_data.append({
                        'variable': 'slope',
                        'value': slope_value,
                        'modality': cfg.name,
                    })
    
    df = pd.DataFrame(all_data)
    log.info(f"Successfully extracted {len(df)} elevation observations")
    
    return df