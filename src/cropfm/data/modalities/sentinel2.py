import ee
import pandas as pd
from omegaconf import DictConfig, OmegaConf

def sentinel2(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    '''
    A function to extract Sentinel-2 data from the Earth Engine API.

    Args:
        cfg: The configuration for the Sentinel-2 data. (DictConfig)
        point: The point to extract data from. (ee.Geometry.Point)
        **kwargs: Additional arguments.

    Returns:
        pd.DataFrame: The extracted data.
    '''

    # Define date range
    start_date = cfg.date_range.start_date
    end_date = cfg.date_range.end_date

    # Sentinel-2 data processing

    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterDate(start_date, end_date) \
    .filterBounds(point) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cfg.cloud_cover_threshold))

    # Define S2 band names to filter
    s2_band_names = list(cfg.bands)

    # Sample only first S2 image with filtered bands
    def sample_s2_image(image):
        selected_image = image.select(s2_band_names)
        pixel_value = selected_image.sample(
            region=point,
            scale=10,
            numPixels=1
        )
        
        def add_metadata(feature):
            return feature.set('date', image.date().format('YYYY-MM-dd')) \
                        .set('satellite', 'S2')
        
        return pixel_value.map(add_metadata)

    s2_samples = ee.ImageCollection(s2_collection).map(sample_s2_image).flatten()

    # Print S2 sample information
    # print(f'S2 pixel samples: {s2_samples.getInfo()}')

    # Convert S2 samples to pandas DataFrame
    s2_samples_data = s2_samples.getInfo()
    s2_features = s2_samples_data['features']

    s2_data_rows = []
    for feature in s2_features:
        row = feature['properties'].copy()
        s2_data_rows.append(row)

    s2_df = pd.DataFrame(s2_data_rows)

    return s2_df


# if __name__ == '__main__':
#     import ee
#     ee.Initialize()
    
#     cfg = DictConfig(OmegaConf.load('configs/data/modalities.yaml'))
#     point = ee.Geometry.Point([10.659969917504554, 50.2844142988367])
#     s2_df = sentinel2(cfg.sentinel2, point)
#     print(s2_df)