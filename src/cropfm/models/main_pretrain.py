"""Main pretraining script for CropMAE with Hydra"""

import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from cropfm.models.arch.mae import CropMAE
from cropfm.models.module import CropMAEModule
from cropfm.models.datamodule import CropFMDataModule
from cropfm.models.utils.callbacks import configure_callbacks


@hydra.main(version_base=None, config_path="../../../configs/models", config_name="base.pretrain")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""
    # print("Configuration:")
    # print(OmegaConf.to_yaml(cfg))

    arch = CropMAE(
        modalities=cfg.data.modalities,
        **cfg.model
    )

    module = CropMAEModule(
        model=arch,
        **cfg.model
    )

    datamodule = CropFMDataModule(
        **cfg.data
    )

    callbacks = configure_callbacks(cfg, arch, pretraining=True) # returns a dictionary with "callbacks" and "loggers"

    # Initialize trainer
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"],
    )

    # Train model
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

