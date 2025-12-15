"""PyTorch Lightning module for CropMAE"""

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cropfm.models.arch.mae import CropMAE


class CropMAEModule(LightningModule):
    """PyTorch Lightning module for training CropMAE"""

    def __init__(
        self,
        model: CropMAE,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.criterion = nn.MSELoss()

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        reconstructions = self.model(batch)
        total_loss = 0.0
        num_modalities = 0

        for modality_name in batch.keys():
            if modality_name in reconstructions:
                loss = self.criterion(reconstructions[modality_name], batch[modality_name])
                total_loss += loss
                num_modalities += 1
                self.log(
                    f"train_loss_{modality_name}",
                    loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        avg_loss = total_loss / num_modalities if num_modalities > 0 else total_loss
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return avg_loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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

