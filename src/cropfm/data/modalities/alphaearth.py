import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def alphaearth(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract Google Satellite Embedding data for a point.
    Google Satellite Embedding - Annual embeddings from satellite imagery

    Args:
        cfg: Hydra config containing alphaearth parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted Google Satellite Embedding data with metadata
    """
    
    # Get config values with optional overrides from kwargs
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting {cfg.name} extraction")
    
    # Load Google Satellite Embedding collection
    dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    
    # Get variables (embedding bands) and years from config
    variables = list(cfg.variables)
    years = list(cfg.years)
    
    # Create collection for all years
    year_images = []
    for year in years:
        start_date = f'{year}-01-01'
        end_date = f'{year + 1}-01-01'
        
        yearly_image = dataset.filterDate(start_date, end_date).filterBounds(point).first()
        year_images.append(yearly_image)
    
    # Convert to ImageCollection
    alphaearth_collection = ee.ImageCollection(year_images)
    
    # Sample alphaearth image with filtered bands
    def sample_alphaearth_image(image):
        selected_image = image.select(variables)
        pixel_value = selected_image.sample(
            region=point,
            scale=1000,  # Google Satellite Embedding resolution (1km)
            numPixels=1
        )
        
        def add_metadata(feature):
            return feature.set('year', image.date().format('YYYY')) \
                        .set('modality', cfg.name)
        
        return pixel_value.map(add_metadata)

    alphaearth_samples = ee.ImageCollection(alphaearth_collection).map(sample_alphaearth_image).flatten()

    # Print alphaearth sample information
    log.debug('Number of Google Satellite Embedding samples: %s', alphaearth_samples.size().getInfo())

    # Convert alphaearth samples to pandas DataFrame
    alphaearth_samples_data = alphaearth_samples.getInfo()
    alphaearth_features = alphaearth_samples_data['features']

    alphaearth_data_rows = []
    for feature in alphaearth_features:
        row = feature['properties'].copy()
        alphaearth_data_rows.append(row)

    alphaearth_df = pd.DataFrame(alphaearth_data_rows)

    log.info(f"Successfully extracted {len(alphaearth_df)} Google Satellite Embedding observations")

    return alphaearth_df