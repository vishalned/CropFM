import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def worldcereal_cropmask(
    cfg: DictConfig,
    point_fc: ee.FeatureCollection,
    batch_features: list,
    **kwargs
) -> dict:
    '''
    Extract WorldCereal data for multiple points in batch.
        - 2021 model products
    Key for the crop mask is:
    - 0: No crop
    - 1: Maize
    - 2: Winter Cereals
    - 3: Spring Cereals

    Args:
        cfg: Hydra config containing worldcereal parameters
        point_fc: FeatureCollection of points to extract data from
        batch_features: List of GeoJSON features (for point_id mapping)
        **kwargs: Additional parameters (can override config)
        
    Returns:
        dict: Dictionary mapping point_id to data dict
    '''
    log.setLevel(cfg.log_level)
    
    log.info(f"Starting batch {cfg.name} extraction for {point_fc.size().getInfo()} points")

    # Create point_id lookup
    point_id_map = {feat['id']: i for i, feat in enumerate(batch_features)}
    
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

    # Batch sample the crop mask at all points
    samples = crop_mask.sampleRegions(
        collection=point_fc,
        scale=10,
        geometries=False
    )
    
    # Get all features
    all_features = samples.getInfo()['features']
    
    # Group crop_mask values by point_id
    crop_mask_results = {}
    for point_id in point_id_map.keys():
        crop_mask_results[point_id] = None
    
    for feature in all_features:
        point_id = feature['properties'].get('point_id')
        if point_id and point_id in crop_mask_results:
            crop_value = feature['properties'].get('crop_mask')
            crop_mask_results[point_id] = crop_value
    
    # For AEZ, we need to do spatial join - batch process by filtering AEZ table with bounds
    # Get bounding box of all points
    bounds = point_fc.geometry().bounds()
    aez_in_bounds = aez_table.filterBounds(bounds)
    
    # For each point, find intersecting AEZ
    # We'll use a spatial join approach
    def add_aez_id(feature):
        point_geom = feature.geometry()
        # Find AEZ polygons that intersect this point
        intersecting_aez = aez_in_bounds.filterBounds(point_geom)
        aez_id = ee.Algorithms.If(
            intersecting_aez.size().gt(0),
            intersecting_aez.first().get('aez_id'),
            None
        )
        return feature.set('aez_id', aez_id)
    
    point_fc_with_aez = point_fc.map(add_aez_id)
    
    # Get AEZ IDs
    aez_features = point_fc_with_aez.getInfo()['features']
    aez_results = {}
    for feature in aez_features:
        point_id = feature['properties'].get('point_id')
        aez_id = feature['properties'].get('aez_id')
        if point_id:
            aez_results[point_id] = aez_id
    
    # Combine results
    batch_results = {}
    for point_id in point_id_map.keys():
        df = pd.DataFrame({
            'aez_id': [aez_results.get(point_id)],
            'crop_mask': [crop_mask_results.get(point_id)],
        }, index=[0])
        
        batch_results[point_id] = {
            'modality': cfg.name,
            'data': df,
            'variable_names': ['aez_id', 'crop_mask'],
        }
    
    log.info(f"Successfully extracted data for {len(batch_results)} points")
    return batch_results