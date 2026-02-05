"""MAE model for CropFM with WorldCereal dual decoder prediction (Option 1)

This implementation adds separate decoder heads for predicting worldcereal_cropmask 
(classification) and worldcereal_cropcalendar (regression) as downstream tasks.
Worldcereal modalities are excluded from inputs and treated as prediction targets only.
"""

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from cropfm.models.arch.mae import (
    CropMAE,
    Encoder,
    Decoder,
    ModalityTokenizer,
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


class CropMAEWorldcerealDual(CropMAE):
    """CropMAE with WorldCereal dual decoder prediction (Option 1)
    
    Predicts worldcereal_cropmask (classification) and worldcereal_cropcalendar (regression)
    using separate decoder heads. Worldcereal modalities are excluded from inputs.
    
    Loss aggregation: reconstruction + crop_mask_classification + crop_calendar_regression
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
        use_cls_token: bool = True,  # Use CLS token or mean pooling for global representation
        **kwargs
    ):
        # Get all modality dims first
        modality_dims = {modality: len(modalities['modality_list'][modality]['variables']) 
                        for modality in modalities['modality_list']}
        
        # Filter worldcereal from input modalities (they are targets, not inputs)
        worldcereal_modalities = ['worldcereal_cropmask', 'worldcereal_cropcalendar']
        input_modality_dims = {k: v for k, v in modality_dims.items() 
                              if k not in worldcereal_modalities}
        
        # Temporarily modify modalities dict for parent init
        temp_modalities = modalities.copy()
        temp_modalities['modality_list'] = {k: v for k, v in modalities['modality_list'].items() 
                                            if k not in worldcereal_modalities}
        
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
        
        # Store original modality dims for worldcereal heads
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
        
        # Worldcereal decoder heads
        self.worldcereal_heads = nn.ModuleDict()
        if 'worldcereal_cropmask' in modality_dims:
            # Crop mask: classification head (4 classes: 0=no crop, 1=maize, 2=winter cereals, 3=spring cereals)
            self.worldcereal_heads['worldcereal_cropmask'] = nn.Linear(decoder_embed_dim, 4)
        
        if 'worldcereal_cropcalendar' in modality_dims:
            # Crop calendar: regression head - predict 2 values (sos, season_length) for the crop type
            # Season length (not EOS) handles year-crossing crops where EOS < SOS
            self.worldcereal_heads['worldcereal_cropcalendar'] = nn.Linear(decoder_embed_dim, 2)
    
    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass for CropMAE with WorldCereal prediction
        
        Args:
            x: Input tensor dict (worldcereal modalities filtered out before tokenization)
            mask: Mask tensor
            
        Returns:
            Dictionary with keys:
            - 'reconstructions': dict of reconstruction per input modality
            - 'modality_masks': dict of mask per modality
            - 'worldcereal_predictions': dict with 'worldcereal_cropmask' and/or 'worldcereal_cropcalendar'
        """

        # Filter worldcereal from inputs (they are targets, not inputs)
        worldcereal_modalities = ['worldcereal_cropmask', 'worldcereal_cropcalendar']
        input_x = {k: v for k, v in x.items() if k not in worldcereal_modalities}
        
        # Call parent forward with filtered inputs to get base outputs
        output = super().forward(input_x, mask)
        
        # Extract global representation for worldcereal prediction
        # We need to recompute encoder output to get global representation
        # (In practice, you might want to cache this, but for simplicity we recompute)
        
        # Re-tokenize and encode to get global representation
        data_dict = {}
        for modality, modality_dict in input_x.items():
            modality_data = modality_dict['data']
            modality_data = torch.nan_to_num(modality_data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = modality_data
        
        valid_mask_dict = {}
        for modality, modality_dict in input_x.items():
            valid_mask_dict[modality] = modality_dict.get('valid_mask', None)
        
        tokens = self.tokenizer(data_dict, valid_mask_dict)
        
        # Build structure info
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
        
        # Mask and encode
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
        
        # Extract global representation
        if self.use_cls_token:
            global_repr = encoded_tokens[:, 0, :]  # (B, D) - CLS token
        else:
            global_repr = encoded_tokens.mean(dim=1)  # (B, D) - mean pooling
        
        # Decode global representation for worldcereal prediction
        dummy_kept_indices = torch.zeros(B, 1, dtype=torch.long, device=device)
        dummy_removed_indices = torch.zeros(B, 0, dtype=torch.long, device=device)
        
        decoded_global_repr = self.decoder(
            global_repr.unsqueeze(1),  # (B, 1, D)
            dummy_kept_indices,
            dummy_removed_indices,
            week_indices=torch.zeros(B, 1, dtype=torch.long, device=device),
            is_temporal=torch.zeros(B, 1, dtype=torch.bool, device=device),
            modality_indices=torch.zeros(B, 1, dtype=torch.long, device=device)
        )
        decoded_global_repr = decoded_global_repr.squeeze(1)  # (B, decoder_embed_dim)
        
        # Predict worldcereal
        worldcereal_predictions = {}
        if 'worldcereal_cropmask' in self.worldcereal_heads:
            worldcereal_predictions['worldcereal_cropmask'] = self.worldcereal_heads['worldcereal_cropmask'](decoded_global_repr)
        if 'worldcereal_cropcalendar' in self.worldcereal_heads:
            worldcereal_predictions['worldcereal_cropcalendar'] = self.worldcereal_heads['worldcereal_cropcalendar'](decoded_global_repr)
        
        # Add worldcereal predictions to output
        output['worldcereal_predictions'] = worldcereal_predictions
        # Expose global_repr for child classes (e.g. contrastive) to avoid redundant encode
        output['global_repr'] = global_repr
        
        return output
