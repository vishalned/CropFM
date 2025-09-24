import pandas as pd
import ee
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

def fapar(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract FAPAR (Fraction of Absorbed Photosynthetically Active Radiation) data for a point.
    FAPAR - 5.6km resolution
    
    Args:
        cfg: Hydra config containing fapar parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted FAPAR data with metadata
    """
    log.setLevel(cfg.log_level)
    
    # Get config values with optional overrides from kwargs
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date
    variables = list(cfg.variables)
    
    log.info(f"Starting FAPAR extraction")
    
    # FAPAR/LAI collection
    collection = (ee.ImageCollection('NOAA/CDR/VIIRS/LAI_FAPAR/V1')
                  .filterDate(start_date, end_date)
                  .filterBounds(point)
                  .select(variables))
    
    def sample_fapar_image(image):
        """Sample a single FAPAR image at the point"""
        pixel_value = image.sample(
            region=point,
            scale=5566,  # VIIRS LAI/FAPAR native resolution ~5.6km
            numPixels=1
        )
        
        def add_metadata(feature):
            return (feature.set('date', image.date().format('YYYY-MM-dd'))
                           .set('modality', cfg.name))
        
        return pixel_value.map(add_metadata)
    
    # Process all images in collection
    fapar_samples = collection.map(sample_fapar_image).flatten()
    
    # Get sample information
    log.debug(f'FAPAR pixel samples count: {fapar_samples.size().getInfo()}')
    
    features = fapar_samples.getInfo()['features']


    # Convert to pandas DataFrame
    data_rows = []
    for feature in features:
        row = feature['properties'].copy()
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    log.info(f"Successfully extracted {len(df)} FAPAR observations")
    
    return df
