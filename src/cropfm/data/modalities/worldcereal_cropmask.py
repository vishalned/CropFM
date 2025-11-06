import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def worldcereal_cropmask(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    '''
    Extract WorldCereal data for a point.
        - 2021 model products
    Key for the crop mask is:
    - 0: No crop
    - 1: Maize
    - 2: Winter Cereals
    - 3: Spring Cereals

    Args:
        cfg: Hydra config containing worldcereal parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted WorldCereal data with metadata
    '''
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting {cfg.name} extraction")

    # Load WorldCereal 2021 model products image collection
    dataset = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100')
    # Load AEZ table (FeatureCollection)
    aez_table = ee.FeatureCollection('ESA/WorldCereal/AEZ/v100')

    # Mask pixels with no crop (value 0)
    def mask_other(img):
        return img.updateMask(img.neq(0))

    masked_dataset = dataset.map(mask_other)

    maize = masked_dataset.filter(ee.Filter.eq('product', 'maize')) \
                        .filter(ee.Filter.eq('season', 'tc-maize-main')) \
                        .mosaic()

    winter_cereals = masked_dataset.filter(ee.Filter.eq('product', 'wintercereals')) \
                                .filter(ee.Filter.eq('season', 'tc-wintercereals')) \
                                .mosaic()

    spring_cereals = masked_dataset.filter(ee.Filter.eq('product', 'springcereals')) \
                                .filter(ee.Filter.eq('season', 'tc-springcereals')) \
                                .mosaic()

    maize_class = maize.select('classification')
    winter_class = winter_cereals.select('classification')
    spring_class = spring_cereals.select('classification')

    crop_mask = ee.Image(0).rename('crop_mask')

    crop_mask = crop_mask.where(maize_class.gt(0), 1)
    crop_mask = crop_mask.where(winter_class.gt(0).And(crop_mask.eq(0)), 2)
    crop_mask = crop_mask.where(spring_class.gt(0).And(crop_mask.eq(0)), 3)

    # Sample the crop mask at the point
    sample_fc = crop_mask.sample(region=point, scale=10, numPixels=1)

    # Get the crop mask value
    first_feature = sample_fc.first()
    crop_value = first_feature.get('crop_mask').getInfo()

    ###### AEZ extraction ######
    # Filter AEZ polygons intersecting the point
    aez_for_point = aez_table.filterBounds(point)

    # Get AEZ ID property for the first feature (assuming one polygon)
    aez_list = aez_for_point.aggregate_array('aez_id').getInfo()
    aez_id = aez_list[0] if aez_list else None

    all_data = []
    all_data.append({
        'aez_id': aez_id,
        'crop_mask': crop_value,
        'modality': cfg.name
    })

    df = pd.DataFrame(all_data)
    log.info(f"Successfully extracted {len(df)} WorldCereal observations")

    return df