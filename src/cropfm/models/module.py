"""PyTorch Lightning module for CropMAE"""

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig, OmegaConf

from cropfm.models.arch.mae import CropMAE
from cropfm.models.utils.losses import compute_losses, aggregate_losses


class CropMAEModule(LightningModule):
    """
    PyTorch Lightning module for training CropMAE

    Args:
        model: CropMAE model
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        max_epochs: Maximum epochs

    """

    def __init__(
        self,
        model: CropMAE,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        loss_aggregation: dict | None = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Loss aggregation configuration
        if loss_aggregation is None:
            loss_aggregation = {'equal_weighting': True}
        elif isinstance(loss_aggregation, DictConfig):
            loss_aggregation = OmegaConf.to_container(loss_aggregation, resolve=True)
        
        self.equal_weighting = loss_aggregation.get('equal_weighting', True)
        self.loss_weights = loss_aggregation.get('weights', None)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for CropMAE
        Args:
            x: Input data
        Returns:
            Reconstructions dict
        """
        output = self.model(x)
        return output.get('reconstructions', {})

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for CropMAE
        Args:
            batch: Batch of data
            batch_idx: Batch index
        Returns:
            Loss
        """
        output = self.model(batch)
        
        # Compute all losses from output
        losses = compute_losses(output, batch)
        
        # Log individual losses (grouped by loss type)
        for loss_type, loss_dict in losses.items():
            for loss_name, loss_value in loss_dict.items():
                self.log(
                    f"train_loss_{loss_type}_{loss_name}",
                    loss_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        
        # Aggregate losses
        total_loss = aggregate_losses(
            losses=losses,
            equal_weighting=self.equal_weighting,
            weights=self.loss_weights,
        )
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        """
        Configure optimizers for CropMAE
        Returns:
            Optimizer and scheduler
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

