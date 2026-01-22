"""Main MAE model for CropFM"""

import numpy as np
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from cropfm.models.arch.transformer import TransformerBlock
from cropfm.models.arch.masking import RandomMasking


def get_sinusoid_encoding_table(positions: int | list[int], d_hid: int, T: int = 10000) -> torch.Tensor:
    """Sinusoid position encoding table
    Args:
        positions: Number of positions
        d_hid: Dimension of the hidden features
        T: Maximum sequence length
    Returns:
        Sinusoid encoding table
    """
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position: int, hid_idx: int) -> float:
        return position / (T ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position: int) -> list[float]:
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


def get_week_encoding_table(weeks: int | list[int], d_hid: int, num_weeks: int = 52) -> torch.Tensor:
    """Sinusoid week encoding table
    Args:
        weeks: Number of weeks or list of week indices (0-51)
        d_hid: Dimension of the hidden features
        num_weeks: Total number of weeks in a year (default: 52)
    Returns:
        Sinusoid week encoding table
    Formula: pweek,2i = sin((2π × week) / 52), pweek,2i+1 = cos((2π × week) / 52)
    """
    if isinstance(weeks, int):
        weeks = list(range(weeks))

    def get_week_angle_vec(week: int) -> list[float]:
        # Formula: sin((2π × week) / 52) and cos((2π × week) / 52) for each dimension pair
        angle = 2 * np.pi * week / num_weeks
        vec = []
        for hid_j in range(d_hid):
            if hid_j % 2 == 0:
                # Even indices: sin((2π × week) / 52)
                vec.append(np.sin(angle))
            else:
                # Odd indices: cos((2π × week) / 52)
                vec.append(np.cos(angle))
        return vec

    week_table = np.array([get_week_angle_vec(week_i) for week_i in weeks])
    return torch.FloatTensor(week_table)


class ModalityTokenizer(nn.Module):
    """Tokenization module that creates tokens for each modality and timestep
    Args:
        modality_dims: Dictionary of modality dimensions
        embedding_dim: Dimension of the embedding
    Returns:
        Tokenized input
    """

    def __init__(self, modality_dims: dict[str, int], embedding_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.embedding_dim = embedding_dim
        self.tokenizers = nn.ModuleDict(
            {
                modality_name: nn.Linear(mod_dim, embedding_dim)
                for modality_name, mod_dim in modality_dims.items()
            }
        )

        self.non_valid_mask_tokens = nn.ParameterDict(
            {
                modality_name: nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
                for modality_name in modality_dims.keys()
            }
        )

    def forward(self, x: dict[str, torch.Tensor], valid_mask_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for modality_name, modality_data in x.items():
            if modality_name not in self.tokenizers:
                raise ValueError(f"Modality {modality_name} not found in tokenizers")
            B, T, D = modality_data.shape
            modality_flat = modality_data.reshape(B * T, D) 
            modality_tokens = self.tokenizers[modality_name](modality_flat)
            modality_tokens = modality_tokens.reshape(B, T, self.embedding_dim)
            
            if valid_mask_dict[modality_name] is not None and modality_name in self.non_valid_mask_tokens:
                valid_mask = valid_mask_dict[modality_name] # (B, T)

                if isinstance(valid_mask, np.ndarray):
                        valid_mask = torch.from_numpy(valid_mask).to(
                            device=modality_tokens.device, 
                            dtype=torch.bool
                        )
                if valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0).expand(B, -1)
                
                # Get invalid mask (where valid_mask is False)
                invalid_mask = ~valid_mask  # (B, T)
                
                # Expand invalid_mask to match token dimensions: (B, T) -> (B, T, 1)
                invalid_mask_expanded = invalid_mask.unsqueeze(-1)  # (B, T, 1)
                
                # Get mask token for this modality: (1, embedding_dim)
                mask_token = self.non_valid_mask_tokens[modality_name]  # (1, embedding_dim)
                
                # Expand mask_token to match modality_tokens: (1, embedding_dim) -> (B, T, embedding_dim)
                mask_token_expanded = mask_token.view(1, 1, -1).expand(B, T, -1)  # (B, T, embedding_dim)
                
                # Replace invalid tokens with mask tokens using torch.where (differentiable)
                modality_tokens = torch.where(
                    invalid_mask_expanded,
                    mask_token_expanded,
                    modality_tokens
                )

            tokens.append(modality_tokens)
        return torch.cat(tokens, dim=1)


class Encoder(nn.Module):
    """Encoder module for MAE
    Args:
        embedding_dim: Dimension of the embedding
        depth: Number of transformer blocks
        num_heads: Number of heads
        mlp_ratio: Ratio of hidden features to input features
        qkv_bias: Whether to use bias in the linear projections
        drop: Dropout rate
        attn_drop: Dropout rate for the attention
        max_sequence_length: Maximum sequence length
        use_pos_embedding: Whether to use positional encoding
    Returns:
        Encoded input
    """

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
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_pos_embedding = use_pos_embedding

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    attention_config=attention_config,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)

        if use_pos_embedding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_sequence_length, embedding_dim), requires_grad=False
            )
            pos_embed = get_sinusoid_encoding_table(
                self.pos_embed.shape[1], self.pos_embed.shape[-1]
            )
            self.pos_embed.data.copy_(pos_embed)
        
        # Week encoding table: 52 weeks, embedding_dim
        self.week_embed = nn.Parameter(
            torch.zeros(52, embedding_dim), requires_grad=False
        )
        week_embed = get_week_encoding_table(52, embedding_dim, num_weeks=52)
        self.week_embed.data.copy_(week_embed)
        
        # Modality-specific learnable embeddings
        self.modality_embed = nn.Embedding(num_modalities, embedding_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        week_indices: torch.Tensor | None = None,
        is_temporal: torch.Tensor | None = None,
        modality_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for Encoder
        Args:
            x: Input tensor (B, N, D)
            week_indices: Week indices (0-51) for each token (B, N), None for static modalities
            is_temporal: Boolean mask (B, N) indicating temporal tokens (True) vs static (False)
            modality_indices: Modality indices for each token (B, N)
        Returns:
            Encoded input
        """
        if self.use_pos_embedding:
            seq_len = x.shape[1]
            if seq_len <= self.pos_embed.shape[1]:
                pos_emb = self.pos_embed[:, :seq_len, :]  # (1, N, D)
                # Zero out positional encoding for static modalities
                if is_temporal is not None:
                    pos_emb = pos_emb * is_temporal.unsqueeze(-1).float()  # (B, N, D)
                x = x + pos_emb
        
        if week_indices is not None:
            # Get week encoding: (B, N) -> (B, N, D)
            week_emb = self.week_embed[week_indices]  # (B, N, D)
            # Zero out week encoding for static modalities
            if is_temporal is not None:
                week_emb = week_emb * is_temporal.unsqueeze(-1).float()  # (B, N, D)
            x = x + week_emb
        
        if modality_indices is not None:
            # Add modality-specific embedding: (B, N) -> (B, N, D)
            modality_emb = self.modality_embed(modality_indices)  # (B, N, D)
            x = x + modality_emb
        
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class Decoder(nn.Module):
    """Decoder module for MAE
    Args:
        encoder_embed_dim: Dimension of the encoder embedding
        decoder_embed_dim: Dimension of the decoder embedding
        decoder_depth: Number of transformer blocks
        decoder_num_heads: Number of heads
        mlp_ratio: Ratio of hidden features to input features
        qkv_bias: Whether to use bias in the linear projections
        drop: Dropout rate
        attn_drop: Dropout rate for the attention
        max_sequence_length: Maximum sequence length
        use_pos_embedding: Whether to use positional encoding
    Returns:
        Decoded input
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        max_sequence_length: int = 1000,
        use_pos_embedding: bool = True,
        num_modalities: int = 8,
        attention_config: DictConfig | None = None,
    ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.use_pos_embedding = use_pos_embedding

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    attention_config=attention_config,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        if use_pos_embedding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_sequence_length, decoder_embed_dim), requires_grad=False
            )
            pos_embed = get_sinusoid_encoding_table(
                self.pos_embed.shape[1], self.pos_embed.shape[-1]
            )
            self.pos_embed.data.copy_(pos_embed)
        
        # Week encoding table: 52 weeks, decoder_embed_dim
        self.week_embed = nn.Parameter(
            torch.zeros(52, decoder_embed_dim), requires_grad=False
        )
        week_embed = get_week_encoding_table(52, decoder_embed_dim, num_weeks=52)
        self.week_embed.data.copy_(week_embed)
        
        # Modality-specific learnable embeddings
        self.modality_embed = nn.Embedding(num_modalities, decoder_embed_dim)

    def add_masked_tokens(
        self, x: torch.Tensor, kept_indices: torch.Tensor, removed_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Decoder
        Args:
            x: Input tensor
            kept_indices: Kept indices
            removed_indices: Removed indices
        Returns:
            Decoded input
        """
        B = x.shape[0]
        device = x.device
        num_masked = removed_indices.shape[1]
        num_kept = kept_indices.shape[1]
        N_total = num_kept + num_masked
        D = x.shape[-1]

        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, num_masked, D)
        x_full = torch.cat([x, mask_tokens], dim=1)
        combined_indices = torch.cat([kept_indices, removed_indices], dim=1)

        output = torch.zeros(B, N_total, D, device=device, dtype=x.dtype)
        for b in range(B):
            batch_indices = combined_indices[b]
            output[b, batch_indices] = x_full[b]

        return output

    def forward(
        self, 
        x: torch.Tensor, 
        kept_indices: torch.Tensor, 
        removed_indices: torch.Tensor,
        week_indices: torch.Tensor | None = None,
        is_temporal: torch.Tensor | None = None,
        modality_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for Decoder
        Args:
            x: Input tensor (B, num_kept, D)
            kept_indices: Kept indices (B, num_kept)
            removed_indices: Removed indices (B, num_masked)
            week_indices: Week indices (0-51) for full sequence (B, N_total), None for static
            is_temporal: Boolean mask (B, N_total) indicating temporal tokens (True) vs static (False)
            modality_indices: Modality indices for full sequence (B, N_total)
        Returns:
            Decoded input
        """
        x = self.decoder_embed(x)
        x = self.add_masked_tokens(x, kept_indices, removed_indices)

        if self.use_pos_embedding:
            seq_len = x.shape[1]
            if seq_len <= self.pos_embed.shape[1]:
                pos_emb = self.pos_embed[:, :seq_len, :]  # (1, N_total, D)
                # Zero out positional encoding for static modalities
                if is_temporal is not None:
                    pos_emb = pos_emb * is_temporal.unsqueeze(-1).float()  # (B, N_total, D)
                x = x + pos_emb
        
        if week_indices is not None:
            # Get week encoding: (B, N_total) -> (B, N_total, D)
            week_emb = self.week_embed[week_indices]  # (B, N_total, D)
            # Zero out week encoding for static modalities
            if is_temporal is not None:
                week_emb = week_emb * is_temporal.unsqueeze(-1).float()  # (B, N_total, D)
            x = x + week_emb
        
        if modality_indices is not None:
            # Add modality-specific embedding: (B, N_total) -> (B, N_total, D)
            modality_emb = self.modality_embed(modality_indices)  # (B, N_total, D)
            x = x + modality_emb

        for block in self.decoder_blocks:
            x = block(x)

        return self.decoder_norm(x)


class CropMAE(nn.Module):
    """Masked Autoencoder (MAE) for crop foundational models
    Args:
        modalities: Dictionary of modalities
        embedding_dim: Dimension of the embedding
        encoder_depth: Number of encoder transformer blocks
        encoder_num_heads: Number of encoder heads
        decoder_embed_dim: Dimension of the decoder embedding
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of decoder heads
        mlp_ratio: Ratio of hidden features to input features
        mask_ratio: Ratio of tokens to mask
        max_sequence_length: Maximum sequence length
        use_pos_embedding: Whether to use positional encoding
        weight_decay: Weight decay
    Returns:
        Reconstructions
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
        masking_config: DictConfig | None = None,
        **kwargs
    ):
        super().__init__()
        """
        Initialize CropMAE
        Args:
            modalities: Dictionary of modalities
            embedding_dim: Dimension of the embedding
            encoder_depth: Number of encoder transformer blocks
            encoder_num_heads: Number of encoder heads
            decoder_embed_dim: Dimension of the decoder embedding
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of decoder heads
            mlp_ratio: Ratio of hidden features to input features
            mask_ratio: Ratio of tokens to mask
            max_sequence_length: Maximum sequence length
            use_pos_embedding: Whether to use positional encoding
            weight_decay: Weight decay
        """
        modality_dims = {modality: len(modalities['modality_list'][modality]['variables']) for modality in modalities['modality_list']}
        
        # Store modality types for forward pass
        TEMPORAL_MODALITIES = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
        STATIC_MODALITIES = ['soil', 'elevation', 'worldcereal_cropmask', 
                             'worldcereal_cropcalender', 'encoded_coordinates']
        self.temporal_modalities = TEMPORAL_MODALITIES
        self.static_modalities = STATIC_MODALITIES
        
        # Create modality name to index mapping for learnable embeddings
        all_modalities = list(modality_dims.keys())
        self.modality_to_idx = {mod_name: idx for idx, mod_name in enumerate(all_modalities)}
        num_modalities = len(all_modalities)

        self.tokenizer = ModalityTokenizer(
            modality_dims=modality_dims,
            embedding_dim=embedding_dim,
        )
        
        # Handle masking instantiation
        if masking_config is not None:
            from hydra.utils import instantiate
            # Override mask_ratio if not set in config
            masking_cfg = OmegaConf.create(OmegaConf.to_container(masking_config, resolve=True))
            if 'mask_ratio' not in masking_cfg:
                masking_cfg.mask_ratio = mask_ratio
            self.masking = instantiate(masking_cfg)
        else:
            # Fallback to default RandomMasking
            self.masking = RandomMasking(mask_ratio=mask_ratio)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=num_modalities,
            attention_config=encoder_attention,
        )
        self.decoder = Decoder(
            encoder_embed_dim=embedding_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=num_modalities,
            attention_config=decoder_attention,
        )
        self.reconstruction_heads = nn.ModuleDict(
            {
                modality_name: nn.Linear(decoder_embed_dim, mod_dim)
                for modality_name, mod_dim in modality_dims.items()
            }
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights
        Args:
            m: Module
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Forward pass for CropMAE
        Args:
            x: Input tensor
            mask: Mask tensor
        Returns:
            Tuple of (reconstructions per modality, mask per modality)
        """
        # x is expected to be a dict[modality] -> {'data': tensor, 'valid_mask': tensor or None}
        # For now we ignore valid_mask here – masking is handled at token level.
        data_dict = {}
        for modality, modality_dict in x.items():
            modality_data = modality_dict['data']
            # Replace NaN values with 0 to prevent NaN propagation through tokenizer
            # This is safe because invalid timesteps will be masked out later
            modality_data = torch.nan_to_num(modality_data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = modality_data

        valid_mask_dict = {}
        for modality, modality_dict in x.items():
            if 'valid_mask' in modality_dict:
                valid_mask_dict[modality] = modality_dict['valid_mask']
            else:
                valid_mask_dict[modality] = None

        tokens = self.tokenizer(data_dict, valid_mask_dict)
        masked_tokens, mask, kept_indices, removed_indices = self.masking(tokens, mask)
        
        # Build week_indices, is_temporal, and modality_indices tensors for positional encoding
        B = tokens.shape[0]
        N_total = tokens.shape[1]
        device = tokens.device
        
        # Initialize week_indices (B, N_total), is_temporal (B, N_total), and modality_indices (B, N_total)
        week_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        is_temporal = torch.zeros(B, N_total, dtype=torch.bool, device=device)
        modality_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        
        start_idx = 0
        for modality_name, modality_data in data_dict.items():
            B_mod, T, D = modality_data.shape
            
            # Set modality index for all tokens of this modality
            if modality_name in self.modality_to_idx:
                mod_idx = self.modality_to_idx[modality_name]
                modality_indices[:, start_idx:start_idx + T] = mod_idx
            
            if modality_name in self.temporal_modalities:
                # Temporal modalities: set is_temporal=True and load week_indices
                is_temporal[:, start_idx:start_idx + T] = True
                
                # Extract week indices for temporal modalities
                if modality_name in x and 'week_indices' in x[modality_name]:
                    week_idx = x[modality_name]['week_indices']  # (B, T) after batching or (T,) per sample
                    if week_idx.dim() == 1:
                        # Single sample case: expand to batch
                        week_idx = week_idx.unsqueeze(0).expand(B, -1)  # (B, T)
                    elif week_idx.shape[0] != B:
                        # Handle batch size mismatch
                        week_idx = week_idx[:B]
                    week_indices[:, start_idx:start_idx + T] = week_idx.to(device)
                # If week_indices not provided, leave as zeros (will be zeroed out by is_temporal mask)
            else:
                # Static modalities: set is_temporal=False, week_indices stays zero
                is_temporal[:, start_idx:start_idx + T] = False
            
            start_idx += T
        
        # Extract week_indices, is_temporal, and modality_indices for kept tokens only (for encoder)
        kept_week_indices = None
        kept_is_temporal = None
        kept_modality_indices = None
        if week_indices.numel() > 0:
            kept_week_indices = torch.gather(week_indices, 1, kept_indices)  # (B, num_kept)
            kept_is_temporal = torch.gather(is_temporal, 1, kept_indices)  # (B, num_kept)
            kept_modality_indices = torch.gather(modality_indices, 1, kept_indices)  # (B, num_kept)
        
        encoded_tokens = self.encoder(
            masked_tokens, 
            week_indices=kept_week_indices,
            is_temporal=kept_is_temporal,
            modality_indices=kept_modality_indices
        )
        decoded_tokens = self.decoder(
            encoded_tokens, 
            kept_indices, 
            removed_indices,
            week_indices=week_indices,  # Full sequence for decoder
            is_temporal=is_temporal,  # Full sequence for decoder
            modality_indices=modality_indices  # Full sequence for decoder
        )

        reconstructions: dict[str, torch.Tensor] = {}
        modality_masks: dict[str, torch.Tensor] = {}  # Track mask per modality
        start_idx = 0
        for modality_name, modality_data in data_dict.items():
            B, T, D = modality_data.shape
            # Extract mask for this modality
            modality_mask = mask[:, start_idx : start_idx + T]  # (B, T)
            modality_masks[modality_name] = modality_mask
            
            modality_tokens = decoded_tokens[:, start_idx : start_idx + T, :]
            modality_recon = self.reconstruction_heads[modality_name](modality_tokens)
            reconstructions[modality_name] = modality_recon
            start_idx += T

        return reconstructions, modality_masks
