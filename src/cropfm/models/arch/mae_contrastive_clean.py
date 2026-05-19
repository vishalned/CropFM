"""MAE model for CropFM with WorldCereal contrastive and cropcalendar prediction (Clean Version)

This implementation adds:
1. Contrastive learning for crop_mask using InfoNCE loss (applied to CLS token).
2. MLP head for predicting worldcereal_cropcalendar (regression) (applied to CLS token).
3. No cropmask classification head.
4. Bypasses the reconstruction decoder for auxiliary tasks (fixing the dual decoder bug).
"""

import torch
from torch import nn
from omegaconf import DictConfig

from cropfm.models.arch.mae import (
    CropMAE,
    Encoder,
    AUXILIARY_BATCH_KEYS,
)


class EncoderWithCLS(Encoder):
    """Encoder with optional CLS token support"""
    
    def __init__(
        self,
        embedding_dim: int,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        max_sequence_length: int = 1000,
        use_pos_embedding: bool = True,
        num_modalities: int = 8,
        attention_config: DictConfig | None = None,
        use_cls_token: bool = False,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=num_modalities,
            attention_config=attention_config,
        )
        self.use_cls_token = use_cls_token
        
        if use_cls_token:
            # CLS token: learnable parameter
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        week_indices: torch.Tensor | None = None,
        is_temporal: torch.Tensor | None = None,
        modality_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with optional CLS token"""
        B = x.shape[0]
        
        # Prepend CLS token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
            
            # Expand week_indices, is_temporal, modality_indices for CLS token
            if week_indices is not None:
                cls_week = torch.zeros(B, 1, dtype=week_indices.dtype, device=week_indices.device)
                week_indices = torch.cat([cls_week, week_indices], dim=1)
            if is_temporal is not None:
                cls_temporal = torch.zeros(B, 1, dtype=is_temporal.dtype, device=is_temporal.device)
                is_temporal = torch.cat([cls_temporal, is_temporal], dim=1)
            if modality_indices is not None:
                cls_mod = torch.zeros(B, 1, dtype=modality_indices.dtype, device=modality_indices.device)
                modality_indices = torch.cat([cls_mod, modality_indices], dim=1)
        
        # Call parent forward
        return super().forward(x, week_indices, is_temporal, modality_indices)


class CropMAEContrastiveClean(CropMAE):
    """Clean CropMAE with Contrastive learning for cropmask and regression for cropcalendar.
    
    Bypasses the decoder bug by applying MLPs directly to the global representation (CLS token).
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
        projection_dim: int = 128,
        **kwargs
    ):
        # Get all modality dims first
        modality_dims = {modality: len(modalities['modality_list'][modality]['variables']) 
                        for modality in modalities['modality_list']}
        
        # Filter worldcereal from input modalities (they are targets, not inputs)
        worldcereal_modalities = ['worldcereal_cropmask', 'worldcereal_cropcalendar']
        input_modality_dims = {k: v for k, v in modality_dims.items() 
                              if k not in worldcereal_modalities and k not in AUXILIARY_BATCH_KEYS}
        
        # Temporarily modify modalities dict for parent init
        temp_modalities = modalities.copy()
        temp_modalities['modality_list'] = {k: v for k, v in modalities['modality_list'].items() 
                                            if k not in worldcereal_modalities and k not in AUXILIARY_BATCH_KEYS}
        
        # Initialize parent with filtered modalities
        super().__init__(
            modalities=temp_modalities,
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
            **kwargs
        )
        
        self.modality_dims = modality_dims
        self.use_cls_token = use_cls_token
        
        # Replace encoder with CLS-enabled version
        num_modalities = len(input_modality_dims)
        self.encoder = EncoderWithCLS(
            embedding_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=num_modalities,
            attention_config=encoder_attention,
            use_cls_token=use_cls_token,
        )
        
        # Projection MLP for contrastive learning (Cropmask)
        self.projection_dim = projection_dim
        if 'worldcereal_cropmask' in modality_dims:
            self.projection_mlp = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            self.projection_mlp = None

        # MLP for Crop Calendar prediction
        if 'worldcereal_cropcalendar' in modality_dims:
            # Predict 2 values (sos, season_length)
            self.cropcalendar_mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 2)
            )
        else:
            self.cropcalendar_mlp = None

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass for CropMAE Contrastive Clean
        """
        worldcereal_modalities = ['worldcereal_cropmask', 'worldcereal_cropcalendar']
        input_x = {k: v for k, v in x.items() if k not in worldcereal_modalities and k not in AUXILIARY_BATCH_KEYS}
        
        # Call parent forward to get reconstructions
        output = super().forward(input_x, mask)
        
        # --- Recompute encoder forward to extract global_repr ---
        # (This keeps the code modular without overriding the entire CropMAE forward loop)
        data_dict = {}
        for modality, modality_dict in input_x.items():
            modality_data = modality_dict['data']
            modality_data = torch.nan_to_num(modality_data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = modality_data
        
        valid_mask_dict = {}
        for modality, modality_dict in input_x.items():
            valid_mask_dict[modality] = modality_dict.get('valid_mask', None)
        
        tokens = self.tokenizer(data_dict, valid_mask_dict)
        
        B = tokens.shape[0]
        N_total = tokens.shape[1]
        device = tokens.device
        
        week_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        is_temporal = torch.zeros(B, N_total, dtype=torch.bool, device=device)
        modality_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        
        modality_boundaries = {}
        start_idx = 0
        for modality_name, modality_data in data_dict.items():
            B_mod, T, D = modality_data.shape
            modality_boundaries[modality_name] = (start_idx, start_idx + T)
            if modality_name in self.modality_to_idx:
                mod_idx = self.modality_to_idx[modality_name]
                modality_indices[:, start_idx:start_idx + T] = mod_idx
            if modality_name in self.temporal_modalities:
                is_temporal[:, start_idx:start_idx + T] = True
                if modality_name in input_x and 'week_indices' in input_x[modality_name]:
                    week_idx = input_x[modality_name]['week_indices']
                    if week_idx.dim() == 1:
                        week_idx = week_idx.unsqueeze(0).expand(B, -1)
                    elif week_idx.shape[0] != B:
                        week_idx = week_idx[:B]
                    week_indices[:, start_idx:start_idx + T] = week_idx.to(device)
            else:
                is_temporal[:, start_idx:start_idx + T] = False
            start_idx += T
            
        try:
            masked_tokens, _, kept_indices, _ = self.masking(
                tokens,
                mask=mask,
                modality_boundaries=modality_boundaries,
                week_indices=week_indices,
                is_temporal=is_temporal,
                modality_indices=modality_indices,
            )
        except TypeError:
            masked_tokens, _, kept_indices, _ = self.masking(tokens, mask)
            
        kept_week_indices = None
        kept_is_temporal = None
        kept_modality_indices = None
        if week_indices.numel() > 0:
            kept_week_indices = torch.gather(week_indices, 1, kept_indices)
            kept_is_temporal = torch.gather(is_temporal, 1, kept_indices)
            kept_modality_indices = torch.gather(modality_indices, 1, kept_indices)
            
        encoded_tokens = self.encoder(
            masked_tokens,
            week_indices=kept_week_indices,
            is_temporal=kept_is_temporal,
            modality_indices=kept_modality_indices
        )
        
        # Extract global representation (CLS token)
        if self.use_cls_token:
            global_repr = encoded_tokens[:, 0, :]  # (B, D)
        else:
            global_repr = encoded_tokens.mean(dim=1)  # (B, D)
        
        output['global_repr'] = global_repr
        
        # --- Auxiliary Tasks directly on global_repr (FIXING THE BUG) ---
        worldcereal_predictions = {}
        
        # 1. Crop Calendar Regression
        if self.cropcalendar_mlp is not None:
            worldcereal_predictions['worldcereal_cropcalendar'] = self.cropcalendar_mlp(global_repr)
            
        if len(worldcereal_predictions) > 0:
            output['worldcereal_predictions'] = worldcereal_predictions
            
        # 2. Contrastive Projection for Crop Mask
        if self.projection_mlp is not None:
            output['projected_representation'] = self.projection_mlp(global_repr)
            
        return output
