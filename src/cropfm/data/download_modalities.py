import hydra
import ee
from omegaconf import DictConfig
from cropfm.data.modalities import (
    sentinel2
)

@hydra.main(config_path='../../../configs/data', config_name='modalities.yaml')
def main(cfg: DictConfig):
    ee.Initialize()
    point = ee.Geometry.Point([10.659969917504554, 50.2844142988367])

    sentinel2_df = sentinel2(cfg.sentinel2, point)
    print(sentinel2_df)



if __name__ == '__main__':
    main()