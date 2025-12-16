"""Main MAE model for CropFM"""

import numpy as np
import torch
from torch import nn

from cropfm.models.arch.transformer import TransformerBlock


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

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for modality_name, modality_data in x.items():
            if modality_name not in self.tokenizers:
                raise ValueError(f"Modality {modality_name} not found in tokenizers")
            print(f"Modality {modality_name} shape: {modality_data.shape}")
            B, T, D = modality_data.shape
            modality_flat = modality_data.reshape(B * T, D) 
            modality_tokens = self.tokenizers[modality_name](modality_flat)
            modality_tokens = modality_tokens.reshape(B, T, self.embedding_dim)
            tokens.append(modality_tokens)
        return torch.cat(tokens, dim=1)


class RandomMasking(nn.Module):
    """Random masking module for MAE
    Args:
        mask_ratio: Ratio of tokens to mask
    Returns:
        Masked input
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self, x: torch.Tensor, mask_ratio: float = 0.75
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for RandomMasking
        Args:
            x: Input tensor
            mask_ratio: Ratio of tokens to mask
        Returns:
            Masked input
            Masked input
            Kept indices
            Removed indices
        """
        B, N, C = x.shape
        device = x.device

        num_masked = int(N * self.mask_ratio)
        num_kept = N - num_masked
        indices = torch.rand(B, N, device=device).argsort(dim=1)
        mask = indices < num_masked

        kept_indices = torch.zeros(B, num_kept, dtype=torch.long, device=device)
        removed_indices = torch.zeros(B, num_masked, dtype=torch.long, device=device)
        masked_x = torch.zeros(B, num_kept, C, device=device)

        for b in range(B):
            batch_mask = mask[b]
            batch_kept_idx = torch.where(~batch_mask)[0]
            batch_removed_idx = torch.where(batch_mask)[0]
            kept_len = len(batch_kept_idx)
            removed_len = len(batch_removed_idx)
            kept_indices[b, :kept_len] = batch_kept_idx
            removed_indices[b, :removed_len] = batch_removed_idx
            masked_x[b, :kept_len] = x[b][~batch_mask] # store the actual token embeddings for the kept tokens

        return masked_x, mask, kept_indices, removed_indices


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Encoder
        Args:
            x: Input tensor
        Returns:
            Encoded input
        """
        if self.use_pos_embedding:
            seq_len = x.shape[1]
            if seq_len <= self.pos_embed.shape[1]:
                x = x + self.pos_embed[:, :seq_len, :]
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
        x = self.decoder_embed(x)
        x = self.add_masked_tokens(x, kept_indices, removed_indices)

        if self.use_pos_embedding:
            seq_len = x.shape[1]
            if seq_len <= self.pos_embed.shape[1]:
                x = x + self.pos_embed[:, :seq_len, :]

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

        self.tokenizer = ModalityTokenizer(
            modality_dims=modality_dims,
            embedding_dim=embedding_dim,
        )
        self.masking = RandomMasking(mask_ratio=mask_ratio)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
        )
        self.decoder = Decoder(
            encoder_embed_dim=embedding_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
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
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for CropMAE
        Args:
            x: Input tensor
            mask: Mask tensor
        Returns:
            Reconstructions
        """
        x, valid_mask = x['data'], x['valid_mask']
        raise NotImplementedError("complete the valid mask handling")
        tokens = self.tokenizer(x)
        masked_tokens, mask, kept_indices, removed_indices = self.masking(tokens, mask)
        encoded_tokens = self.encoder(masked_tokens)
        decoded_tokens = self.decoder(encoded_tokens, kept_indices, removed_indices)

        reconstructions = {}
        start_idx = 0
        for modality_name, modality_data in x.items():
            B, T, D = modality_data.shape
            modality_tokens = decoded_tokens[:, start_idx : start_idx + T, :]
            modality_recon = self.reconstruction_heads[modality_name](modality_tokens)
            reconstructions[modality_name] = modality_recon
            start_idx += T

        return reconstructions
