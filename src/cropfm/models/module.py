"""PyTorch Lightning module for CropMAE"""

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cropfm.models.arch.mae import CropMAE


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
        """
        Forward pass for CropMAE
        Args:
            x: Input data
        Returns:
            Reconstructions
        """
        reconstructions, _ = self.model(x)
        return reconstructions

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for CropMAE
        Args:
            batch: Batch of data
            batch_idx: Batch index
        Returns:
            Loss
        """
        reconstructions, modality_masks = self.model(batch)
        total_loss = 0.0
        num_modalities = 0

        for modality_name in batch.keys():
            if modality_name in reconstructions:
                pred = reconstructions[modality_name]  # (B, T, D)
                target = batch[modality_name]['data']  # (B, T, D)
                mask = modality_masks[modality_name]  # (B, T) - True for masked tokens
                
                # Compute MSE only on masked tokens
                # Expand mask to match feature dimension: (B, T) -> (B, T, D)
                mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # (B, T, D)
                
                # Compute loss only where:
                #  - token is masked by MAE (mask_expanded == True)
                #  - and target is not NaN (to avoid NaN losses from missing data)
                # IMPORTANT: we must index with the mask instead of multiplying by it,
                #            because 0 * NaN is still NaN in floating-point arithmetic.
                diff = pred - target
                valid_target_mask = ~torch.isnan(target)
                combined_mask = mask_expanded & valid_target_mask  # (B, T, D)

                denom = combined_mask.sum()
                if denom == 0:
                    continue

                # Select only masked & valid positions, then compute MSE over them
                selected_diff = diff[combined_mask]  # (denom,)
                loss = (selected_diff ** 2).mean()

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

