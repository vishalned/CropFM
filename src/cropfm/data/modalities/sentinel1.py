import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def sentinel1(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    '''
    Extract Sentinel-1 data for a point.
    S1 - 10m resolution

    Args:
        cfg: The configuration for the Sentinel-1 data. (DictConfig)
        point: The point to extract data from. (ee.Geometry.Point)
        **kwargs: Additional arguments (can override config).

    Returns:
        pd.DataFrame: Extracted Sentinel-1 bands with metadata
    '''
    log.setLevel(cfg.log_level)

    log.info(f"Starting {cfg.name} extraction")
    # Define date range
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date

    # Sentinel-1 data processing
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterDate(start_date, end_date) \
    .filterBounds(point) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Define S1 band names to filter
    s1_band_names = list(cfg.bands)

    # Sample S1 image with filtered bands
    def sample_s1_image(image):
        selected_image = image.select(s1_band_names)
        pixel_value = selected_image.sample(
            region=point,
            scale=10,
            numPixels=1
        )
        
        def add_metadata(feature):
            return feature.set('date', image.date().format('YYYY-MM-dd')) \
                        .set('modality', cfg.name)
        
        return pixel_value.map(add_metadata)

    s1_samples = ee.ImageCollection(s1_collection).map(sample_s1_image).flatten()

    # Print S1 sample information
    log.debug('Number of Sentinel-1 samples: %s', s1_samples.size().getInfo())

    # Convert S1 samples to pandas DataFrame
    s1_samples_data = s1_samples.getInfo()
    s1_features = s1_samples_data['features']

    s1_data_rows = []
    for feature in s1_features:
        row = feature['properties'].copy()
        s1_data_rows.append(row)

    s1_df = pd.DataFrame(s1_data_rows)

    log.info(f"Successfully extracted {len(s1_df)} Sentinel-1 observations")

    return s1_df
