"""PyTorch Lightning module for CropMAE"""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig, OmegaConf

from cropfm.models.arch.mae import CropMAE
from cropfm.models.utils.losses import (
    _satclip_embedding_from_batch,
    aggregate_losses,
    compute_losses,
    get_eam_loss_module,
)


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
        self.save_hyperparameters(ignore=['model'])

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

    def _log_auxiliary_metrics(
        self,
        output: dict,
        batch: dict,
        batch_idx: int,
        losses: dict,
    ) -> None:
        """Whether contrastive / EAM contributed to this step; optional EAM align vs shatter."""
        if batch_idx % 50 != 0:
            return
        c_active = float(
            'contrastive' in losses and len(losses.get('contrastive', {})) > 0
        )
        self.log("diag/contrastive_active", c_active, on_step=True, on_epoch=False, logger=True)

        s_eam = _satclip_embedding_from_batch(batch)
        eam_ok = (
            s_eam is not None
            and 'visual_embedding' in output
            and 'worldcereal_cropmask' in batch
            and 'eam' in losses
            and len(losses.get('eam', {})) > 0
        )
        self.log("diag/eam_active", float(eam_ok), on_step=True, on_epoch=False, logger=True)

        # Extra O(B²) pass for interpretability only (rarely)
        if batch_idx % 200 == 0 and eam_ok:
            with torch.no_grad():
                _, align, shatter = get_eam_loss_module().compute_loss_components(
                    output['visual_embedding'].detach(),
                    s_eam.detach(),
                    batch['worldcereal_cropmask']['data'],
                )
            self.log("eam/align", align, on_step=True, on_epoch=False, logger=True)
            self.log("eam/shatter", shatter, on_step=True, on_epoch=False, logger=True)

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
        self._log_auxiliary_metrics(output, batch, batch_idx, losses)
        
        # Log individual losses (grouped by loss type)
        for loss_type, loss_dict in losses.items():
            for loss_name, loss_value in loss_dict.items():
                self.log(
                    f"train_loss_{loss_type}_{loss_name}",
                    loss_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
        
        # Aggregate losses
        total_loss = aggregate_losses(
            losses=losses,
            equal_weighting=self.equal_weighting,
            weights=self.loss_weights,
        )
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log contrastive collapse metrics (every 50 steps to limit overhead)
        if (
            batch_idx % 50 == 0
            and "projected_representation" in output
        ):
            pr = output["projected_representation"].detach()
            B, D = pr.shape
            if B >= 2:
                repr_std = pr.std(dim=0).mean()
                repr_norm = pr.norm(dim=1).mean()
                pr_norm = F.normalize(pr, dim=1)
                sim = pr_norm @ pr_norm.T
                mask = ~torch.eye(B, dtype=torch.bool, device=pr.device)
                max_offdiag_sim = sim[mask].reshape(B, B - 1).max(dim=1)[0].mean()
                self.log("contrastive/repr_std", repr_std, on_step=True, on_epoch=False, logger=True)
                self.log("contrastive/repr_norm", repr_norm, on_step=True, on_epoch=False, logger=True)
                self.log("contrastive/max_offdiag_sim", max_offdiag_sim, on_step=True, on_epoch=False, logger=True)

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

