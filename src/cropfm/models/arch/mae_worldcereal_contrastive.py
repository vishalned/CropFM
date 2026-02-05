"""MAE model for CropFM with WorldCereal contrastive prediction (Option 2)

This implementation adds separate decoder heads for predicting worldcereal_cropmask 
(classification) and worldcereal_cropcalendar (regression), plus contrastive learning
for crop_mask using InfoNCE loss. Worldcereal modalities are excluded from inputs 
and treated as prediction targets only.
"""

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from cropfm.models.arch.mae_worldcereal_dual import (
    CropMAEWorldcerealDual,
    EncoderWithCLS,
)


class CropMAEWorldcerealContrastive(CropMAEWorldcerealDual):
    """CropMAE with WorldCereal contrastive prediction (Option 2)
    
    Predicts worldcereal_cropmask (classification) and worldcereal_cropcalendar (regression)
    using separate decoder heads, plus contrastive learning for crop_mask.
    Worldcereal modalities are excluded from inputs.
    
    Loss aggregation: reconstruction + crop_mask_classification + crop_calendar_regression + crop_mask_contrastive
    """
    
    def __init__(
        self,
        modalities: dict,
        embedding_dim: int = 128,
        encoder_depth: int = 6,
        encoder_num_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        max_sequence_length: int = 1000,
        use_pos_embedding: bool = True,
        weight_decay: float = 0.05,
        encoder_attention: DictConfig | None = None,
        decoder_attention: DictConfig | None = None,
        masking: DictConfig | None = None,
        use_cls_token: bool = True,
        projection_dim: int = 128,  # Dimension for contrastive projection
        **kwargs
    ):
        super().__init__(
            modalities=modalities,
            embedding_dim=embedding_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            weight_decay=weight_decay,
            encoder_attention=encoder_attention,
            decoder_attention=decoder_attention,
            masking=masking,
            use_cls_token=use_cls_token,
            **kwargs
        )
        
        # Projection MLP for contrastive learning
        self.projection_dim = projection_dim
        if 'worldcereal_cropmask' in self.modality_dims:
            self.projection_mlp = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            self.projection_mlp = None
    
    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass for CropMAE with WorldCereal contrastive prediction
        
        Args:
            x: Input tensor dict (worldcereal modalities filtered out before tokenization)
            mask: Mask tensor
            
        Returns:
            Dictionary with keys:
            - 'reconstructions': dict of reconstruction per input modality
            - 'modality_masks': dict of mask per modality
            - 'worldcereal_predictions': dict with 'worldcereal_cropmask' and/or 'worldcereal_cropcalendar'
            - 'projected_representation': projected representation for contrastive learning (B, projection_dim)
        """
        # Get base predictions from parent (includes global_repr from dual's encode)
        output = super().forward(x, mask)
        
        # Reuse global_repr from parent to avoid redundant tokenize+mask+encode
        global_repr = output.get('global_repr')
        if global_repr is None:
            raise RuntimeError("CropMAEWorldcerealDual must provide global_repr in output")
        
        # Project for contrastive learning
        if self.projection_mlp is not None:
            projected_representation = self.projection_mlp(global_repr)  # (B, projection_dim)
            output['projected_representation'] = projected_representation
        
        return output
