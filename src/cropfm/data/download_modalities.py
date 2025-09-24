import hydra
import ee
from omegaconf import DictConfig
from cropfm.data.modalities import (
    sentinel2,
    fapar
)
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml')
def main(cfg: DictConfig):
    ee.Initialize()
    log.setLevel(cfg.log_level)
    log.debug(f'CFG: {cfg}')

    # sample point
    point = ee.Geometry.Point([10.659969917504554, 50.2844142988367])

    for modality in cfg.modalities:
        modality_df = eval(modality)(cfg[modality], point, log_level=cfg.log_level)
        print(modality_df)



if __name__ == '__main__':
    main()