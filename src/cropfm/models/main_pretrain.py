"""Main pretraining script for CropMAE with Hydra"""

import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from cropfm.models.module import CropMAEModule
from cropfm.models.datamodule import CropFMDataModule


@hydra.main(version_base=None, config_path="../../configs/models", config_name="mae")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize model
    model = CropMAEModule(
        modality_dims=cfg.model.modality_dims,
        embedding_dim=cfg.model.embedding_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        mask_ratio=cfg.model.mask_ratio,
        max_sequence_length=cfg.model.max_sequence_length,
        use_pos_embedding=cfg.model.use_pos_embedding,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )

    # Initialize datamodule
    datamodule = CropFMDataModule(
        batch_size=cfg.data.get("batch_size", 32),
        num_workers=cfg.data.get("num_workers", 4),
        train_split=cfg.data.get("train_split", 0.8),
        val_split=cfg.data.get("val_split", 0.1),
        test_split=cfg.data.get("test_split", 0.1),
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=3,
            filename="epoch_{epoch:02d}-train_loss_{train_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Setup logger
    logger = None
    if cfg.get("logger", {}).get("use_wandb", False):
        logger = WandbLogger(
            project=cfg.logger.project,
            name=cfg.logger.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Initialize trainer
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("devices", "auto"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

