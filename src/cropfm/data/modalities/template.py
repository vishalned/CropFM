'''
This is a template for a modality.
'''

import pandas as pd
import ee
from omegaconf import DictConfig

def modality_name(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    '''
    This is a template for a modality.

    Args:
        cfg: The configuration for the modality. (DictConfig)
        point: The point to extract data from. (ee.Geometry.Point)
        **kwargs: Additional arguments.

    Returns:
        pd.DataFrame: The extracted data.
    '''
    pass
        
