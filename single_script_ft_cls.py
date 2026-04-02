"""
CropFM encoder backbone with CLS token.

This file is self-contained: it defines the tokenizer, transformer encoder,
and a `CropFMBackboneCLS` module that:
  - takes a CropFM-style input batch,
  - runs it through tokenizer + encoder,
  - returns a global feature vector from a CLS token.

Intended usage (inside another repo, e.g. CropBench):

    backbone = CropFMBackboneCLS(
        modalities_config=modalities_cfg,   # CropFM-style modalities dict
        embedding_dim=256,
        encoder_depth=8,
        encoder_num_heads=8,
    )
    backbone.load_from_cropfm_checkpoint("path/to/cropmae_checkpoint.ckpt")

    features = backbone(batch)   # (B, embedding_dim)

You can then plug `backbone` into your LightningModule as a feature extractor.
"""

import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Positional and week encodings
# ---------------------------------------------------------------------------

def get_sinusoid_encoding_table(
    positions: int | List[int],
    d_hid: int,
    T: int = 10000,
) -> torch.Tensor:
    """
    Sinusoidal position encoding table (standard Transformer style).

    positions: int or list of position indices
    d_hid: embedding dimension
    T: scale (default 1e4)
    """
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position: int, hid_idx: int) -> float:
        return position / (T ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position: int) -> List[float]:
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def get_week_encoding_table(
    weeks: int | List[int],
    d_hid: int,
    num_weeks: int = 52,
) -> torch.Tensor:
    """
    Sinusoidal week encoding table.

    weeks: int or list of week indices (0..num_weeks-1)
    d_hid: embedding dimension
    num_weeks: number of weeks in a year (default 52)
    """
    if isinstance(weeks, int):
        weeks = list(range(weeks))

    def get_week_angle_vec(week: int) -> List[float]:
        angle = 2 * math.pi * week / num_weeks
        vec: List[float] = []
        for hid_j in range(d_hid):
            if hid_j % 2 == 0:
                vec.append(math.sin(angle))
            else:
                vec.append(math.cos(angle))
        return vec

    week_table = np.array([get_week_angle_vec(week_i) for week_i in weeks])
    return torch.FloatTensor(week_table)


# ---------------------------------------------------------------------------
# Tokenizer: modality -> tokens
# ---------------------------------------------------------------------------

class ModalityTokenizer(nn.Module):
    """
    Tokenization module that creates tokens for each modality and timestep.

    Expects input as:
        x: dict[modality_name] -> Tensor[B, T, D_mod]
        valid_mask_dict: dict[modality_name] -> Tensor[B, T] or None

    Returns:
        Tensor[B, N_total, embedding_dim], where N_total is sum over modalities T.
    """

    def __init__(self, modality_dims: Dict[str, int], embedding_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.embedding_dim = embedding_dim

        # Per-modality linear projection to embedding_dim
        self.tokenizers = nn.ModuleDict(
            {
                modality_name: nn.Linear(mod_dim, embedding_dim)
                for modality_name, mod_dim in modality_dims.items()
            }
        )

        # Learnable mask token for invalid timesteps per modality
        self.non_valid_mask_tokens = nn.ParameterDict(
            {
                modality_name: nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
                for modality_name in modality_dims.keys()
            }
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        valid_mask_dict: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for modality_name, modality_data in x.items():
            if modality_name not in self.tokenizers:
                raise ValueError(f"Modality {modality_name} not found in tokenizers")

            B, T, D = modality_data.shape
            modality_flat = modality_data.reshape(B * T, D)
            modality_tokens = self.tokenizers[modality_name](modality_flat)
            modality_tokens = modality_tokens.reshape(B, T, self.embedding_dim)

            valid_mask = valid_mask_dict.get(modality_name, None)
            if valid_mask is not None and modality_name in self.non_valid_mask_tokens:
                # valid_mask: (B, T) -> invalid positions become a learned mask token
                if isinstance(valid_mask, np.ndarray):
                    valid_mask = torch.from_numpy(valid_mask).to(
                        device=modality_tokens.device,
                        dtype=torch.bool,
                    )
                if valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0).expand(B, -1)

                invalid_mask = ~valid_mask  # (B, T)
                invalid_mask_expanded = invalid_mask.unsqueeze(-1)  # (B, T, 1)
                mask_token = self.non_valid_mask_tokens[modality_name]  # (1, E)
                mask_token_expanded = mask_token.view(1, 1, -1).expand(B, T, -1)

                modality_tokens = torch.where(
                    invalid_mask_expanded,
                    mask_token_expanded,
                    modality_tokens,
                )

            tokens.append(modality_tokens)

        return torch.cat(tokens, dim=1)


# ---------------------------------------------------------------------------
# Transformer blocks (Attention + MLP)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """
    Standard multi-head self-attention, matching CropFM's implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(F, "scaled_dot_product_attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)  # each: [B, num_heads, N, head_dim]

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with self-attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder with positional, week and modality embeddings
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encoder module for CropFM MAE backbone (without CLS).

    Adds:
      - Sinusoidal positional encodings (optionally)
      - Sinusoidal week encodings (temporal tokens only)
      - Learnable modality embeddings
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
                torch.zeros(1, max_sequence_length, embedding_dim),
                requires_grad=False,
            )
            pos_embed = get_sinusoid_encoding_table(
                self.pos_embed.shape[1],
                self.pos_embed.shape[-1],
            )
            self.pos_embed.data.copy_(pos_embed)

        # Week encoding: 52 x D, non-trainable
        self.week_embed = nn.Parameter(
            torch.zeros(52, embedding_dim),
            requires_grad=False,
        )
        week_embed = get_week_encoding_table(52, embedding_dim, num_weeks=52)
        self.week_embed.data.copy_(week_embed)

        # Learnable modality embeddings
        self.modality_embed = nn.Embedding(num_modalities, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None,
        is_temporal: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:            (B, N, D)
        week_indices: (B, N) or None
        is_temporal:  (B, N) bool or None (True = temporal, False = static)
        modality_indices: (B, N) or None
        """
        B, N, D = x.shape

        if self.use_pos_embedding:
            seq_len = N
            if seq_len <= self.pos_embed.shape[1]:
                pos_emb = self.pos_embed[:, :seq_len, :]  # (1, N, D)
                if is_temporal is not None:
                    # Broadcast over batch and zero-out for static tokens
                    pos_emb = pos_emb * is_temporal.unsqueeze(-1).float()
                x = x + pos_emb

        if week_indices is not None:
            # (B, N) -> (B, N, D)
            week_emb = self.week_embed[week_indices]  # (B, N, D)
            if is_temporal is not None:
                week_emb = week_emb * is_temporal.unsqueeze(-1).float()
            x = x + week_emb

        if modality_indices is not None:
            modality_emb = self.modality_embed(modality_indices)  # (B, N, D)
            x = x + modality_emb

        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class EncoderWithCLS(Encoder):
    """
    Encoder that prepends a learnable CLS token and uses it as global representation.
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
        use_cls_token: bool = True,
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
        )
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None,
        is_temporal: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, 1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

            # Expand metadata for CLS token (week=0, is_temporal=False, modality=0)
            if week_indices is not None:
                cls_week = torch.zeros(B, 1, dtype=week_indices.dtype, device=week_indices.device)
                week_indices = torch.cat([cls_week, week_indices], dim=1)
            if is_temporal is not None:
                cls_temporal = torch.zeros(B, 1, dtype=is_temporal.dtype, device=is_temporal.device)
                is_temporal = torch.cat([cls_temporal, is_temporal], dim=1)
            if modality_indices is not None:
                cls_mod = torch.zeros(B, 1, dtype=modality_indices.dtype, device=modality_indices.device)
                modality_indices = torch.cat([cls_mod, modality_indices], dim=1)

        return super().forward(x, week_indices, is_temporal, modality_indices)


# ---------------------------------------------------------------------------
# High-level backbone: tokenizer + EncoderWithCLS
# ---------------------------------------------------------------------------

class CropFMBackboneCLS(nn.Module):
    """
    High-level backbone that:
      - builds ModalityTokenizer + EncoderWithCLS from a modalities config,
      - optionally loads weights from a CropFM pretraining checkpoint,
      - takes a CropFM-style batch and returns (B, embedding_dim) CLS features.

    Expected input batch format (per sample, already collated to tensors):
        batch[modality]['data']         # Tensor[B, T, D_mod] or [B, D_mod] for static -> will be expanded
        batch[modality].get('valid_mask')  # Optional Tensor[B, T] bool
        batch[modality].get('week_indices') # Optional Tensor[B, T] for temporal modalities
    """

    DEFAULT_TEMPORAL_MODALITIES = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    DEFAULT_STATIC_MODALITIES = [
        'soil',
        'elevation',
        'worldcereal_cropmask',
        'worldcereal_cropcalendar',
        'encoded_coordinates',
    ]

    def __init__(
        self,
        modalities_config: Dict[str, Any],
        embedding_dim: int = 128,
        encoder_depth: int = 6,
        encoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_sequence_length: int = 1000,
        use_pos_embedding: bool = True,
        temporal_modalities: Optional[List[str]] = None,
        static_modalities: Optional[List[str]] = None,
    ):
        super().__init__()

        # Allow either:
        #   modalities_config["modality_list"][name]["variables"]
        # or:
        #   modalities_config[name] = {"variables": [...]} or int dimension
        if "modality_list" in modalities_config:
            mod_cfg = modalities_config["modality_list"]
        else:
            mod_cfg = modalities_config

        modality_dims: Dict[str, int] = {}
        for mod_name, cfg in mod_cfg.items():
            if isinstance(cfg, dict) and "variables" in cfg:
                modality_dims[mod_name] = len(cfg["variables"])
            elif isinstance(cfg, int):
                modality_dims[mod_name] = cfg
            else:
                raise ValueError(
                    f"Unsupported modality config for {mod_name}: {cfg}"
                )

        all_modalities = list(modality_dims.keys())
        self.modality_to_idx = {m: i for i, m in enumerate(all_modalities)}

        self.temporal_modalities = temporal_modalities or self.DEFAULT_TEMPORAL_MODALITIES
        self.static_modalities = static_modalities or self.DEFAULT_STATIC_MODALITIES

        self.tokenizer = ModalityTokenizer(
            modality_dims=modality_dims,
            embedding_dim=embedding_dim,
        )
        self.encoder = EncoderWithCLS(
            embedding_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=len(all_modalities),
            use_cls_token=True,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

    @torch.no_grad()
    def load_from_cropfm_checkpoint(
        self,
        ckpt_path: str,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
    ) -> None:
        """
        Load tokenizer + encoder weights from a Lightning checkpoint produced by CropMAE.

        Expects checkpoint with keys like:
          'model.tokenizer.*'
          'model.encoder.*'
        (i.e. saved from CropMAEModule(model=arch, ...)).
        """
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state_dict = ckpt.get("state_dict", ckpt)

        backbone_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("model.tokenizer.") or k.startswith("model.encoder."):
                new_k = k[len("model.") :]  # strip 'model.'
                backbone_state[new_k] = v

        missing, unexpected = self.load_state_dict(backbone_state, strict=strict)
        if missing:
            print(f"[CropFMBackboneCLS] Missing keys when loading: {missing}")
        if unexpected:
            print(f"[CropFMBackboneCLS] Unexpected keys when loading: {unexpected}")

    def _build_sequence_metadata(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build week_indices, is_temporal, modality_indices tensors for the full sequence.

        data_dict: modality -> Tensor[B, T_mod, D]
        batch_dict: original batch (contains 'week_indices' etc.)
        Returns:
            week_indices:   (B, N_total) long
            is_temporal:    (B, N_total) bool
            modality_indices: (B, N_total) long
        """
        # Compute total sequence length
        B = next(iter(data_dict.values())).shape[0]
        lengths: List[int] = []
        for modality, tensor in data_dict.items():
            lengths.append(tensor.shape[1])
        N_total = sum(lengths)
        device = next(iter(data_dict.values())).device

        week_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        is_temporal = torch.zeros(B, N_total, dtype=torch.bool, device=device)
        modality_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)

        start = 0
        for modality, modality_data in data_dict.items():
            T = modality_data.shape[1]
            end = start + T

            # modality index
            if modality in self.modality_to_idx:
                mod_idx = self.modality_to_idx[modality]
            else:
                mod_idx = 0
            modality_indices[:, start:end] = mod_idx

            # temporal or static
            temporal = modality in self.temporal_modalities
            if temporal:
                is_temporal[:, start:end] = True

                # look for week_indices in batch
                week_info = batch_dict.get(modality, {}).get("week_indices", None)
                if week_info is not None:
                    if isinstance(week_info, torch.Tensor):
                        w = week_info
                    else:
                        w = torch.as_tensor(week_info, device=device)
                    if w.dim() == 1:
                        # (T,) -> (B, T)
                        w = w.unsqueeze(0).expand(B, -1)
                    elif w.shape[0] != B:
                        w = w[:B]
                    week_indices[:, start:end] = w.to(device)
                # else: leave as zeros
            else:
                # static: is_temporal stays False; week_indices remain 0
                is_temporal[:, start:end] = False

            start = end

        return week_indices, is_temporal, modality_indices

    def forward(self, batch: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass.

        batch: dict[modality] -> {
            'data': Tensor[B, T, D]  or Tensor[B, D] for static
            'valid_mask': Optional[Tensor[B, T]]  (temporal)
            'week_indices': Optional[Tensor[B, T]] (temporal)
        }

        Returns:
            global_repr: Tensor[B, embedding_dim] (CLS token)
        """
        # 1) Prepare data_dict and valid_mask_dict for tokenizer
        data_dict: Dict[str, torch.Tensor] = {}
        valid_mask_dict: Dict[str, Optional[torch.Tensor]] = {}

        for modality, m_dict in batch.items():
            data = m_dict["data"]
            if data.dim() == 2:
                # static [B, D] -> [B, 1, D]
                data = data.unsqueeze(1)

            # Replace NaNs to avoid propagation (invalid timesteps will be masked)
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = data

            valid_mask = m_dict.get("valid_mask", None)
            if valid_mask is not None:
                if valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0).expand(data.shape[0], -1)
            valid_mask_dict[modality] = valid_mask

        tokens = self.tokenizer(data_dict, valid_mask_dict)  # (B, N, D)

        # 2) Build sequence metadata
        week_indices, is_temporal, modality_indices = self._build_sequence_metadata(
            data_dict=data_dict,
            batch_dict=batch,
        )

        # 3) Encode with CLS
        encoded = self.encoder(
            tokens,
            week_indices=week_indices,
            is_temporal=is_temporal,
            modality_indices=modality_indices,
        )  # (B, N+1, D)

        # Global representation = CLS token at position 0
        global_repr = encoded[:, 0, :]  # (B, D)
        return global_repr