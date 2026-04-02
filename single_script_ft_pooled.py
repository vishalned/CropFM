"""
CropFM encoder backbone with mean pooling over tokens.

This file is self-contained: it defines the tokenizer, transformer encoder,
and a `CropFMBackbonePooled` module that:
  - takes a CropFM-style input batch,
  - runs it through tokenizer + encoder,
  - returns a global feature vector via mean pooling across tokens.

Usage:

    backbone = CropFMBackbonePooled(
        modalities_config=modalities_cfg,
        embedding_dim=256,
        encoder_depth=8,
        encoder_num_heads=8,
    )
    backbone.load_from_cropfm_checkpoint("path/to/cropmae_checkpoint.ckpt")

    features = backbone(batch)   # (B, embedding_dim)
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
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position: int, hid_idx: int) -> float:
        return position / (T ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position: int) -> List[float]:
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


def get_week_encoding_table(
    weeks: int | List[int],
    d_hid: int,
    num_weeks: int = 52,
) -> torch.Tensor:
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
# Tokenizer
# ---------------------------------------------------------------------------

class ModalityTokenizer(nn.Module):
    def __init__(self, modality_dims: Dict[str, int], embedding_dim: int):
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
                if isinstance(valid_mask, np.ndarray):
                    valid_mask = torch.from_numpy(valid_mask).to(
                        device=modality_tokens.device,
                        dtype=torch.bool,
                    )
                if valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0).expand(B, -1)

                invalid_mask = ~valid_mask
                invalid_mask_expanded = invalid_mask.unsqueeze(-1)
                mask_token = self.non_valid_mask_tokens[modality_name]
                mask_token_expanded = mask_token.view(1, 1, -1).expand(B, T, -1)

                modality_tokens = torch.where(
                    invalid_mask_expanded,
                    mask_token_expanded,
                    modality_tokens,
                )

            tokens.append(modality_tokens)

        return torch.cat(tokens, dim=1)


# ---------------------------------------------------------------------------
# Transformer components
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
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv =         qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

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
# Encoder and pooled backbone
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
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

        self.week_embed = nn.Parameter(
            torch.zeros(52, embedding_dim),
            requires_grad=False,
        )
        week_embed = get_week_encoding_table(52, embedding_dim, num_weeks=52)
        self.week_embed.data.copy_(week_embed)

        self.modality_embed = nn.Embedding(num_modalities, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None,
        is_temporal: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        if self.use_pos_embedding:
            seq_len = N
            if seq_len <= self.pos_embed.shape[1]:
                pos_emb = self.pos_embed[:, :seq_len, :]
                if is_temporal is not None:
                    pos_emb = pos_emb * is_temporal.unsqueeze(-1).float()
                x = x + pos_emb

        if week_indices is not None:
            week_emb = self.week_embed[week_indices]
            if is_temporal is not None:
                week_emb = week_emb * is_temporal.unsqueeze(-1).float()
            x = x + week_emb

        if modality_indices is not None:
            modality_emb = self.modality_embed(modality_indices)
            x = x + modality_emb

        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class CropFMBackbonePooled(nn.Module):
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
                raise ValueError(f"Unsupported modality config for {mod_name}: {cfg}")

        all_modalities = list(modality_dims.keys())
        self.modality_to_idx = {m: i for i, m in enumerate(all_modalities)}

        self.temporal_modalities = temporal_modalities or self.DEFAULT_TEMPORAL_MODALITIES
        self.static_modalities = static_modalities or self.DEFAULT_STATIC_MODALITIES

        self.tokenizer = ModalityTokenizer(
            modality_dims=modality_dims,
            embedding_dim=embedding_dim,
        )
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=len(all_modalities),
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
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # 1) Build backbone_state from checkpoint
        backbone_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("model.tokenizer.") or k.startswith("model.encoder."):
                new_k = k[len("model.") :]
                backbone_state[new_k] = v

        # 2) Compare to current model state
        model_state = self.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped_shape: Dict[str, tuple[torch.Size, torch.Size]] = {}
        skipped_missing_in_model: list[str] = []

        for k, v in backbone_state.items():
            if k not in model_state:
                skipped_missing_in_model.append(k)
                continue
            if v.shape != model_state[k].shape:
                skipped_shape[k] = (v.shape, model_state[k].shape)
                continue
            filtered[k] = v

        # 3) Load only compatible keys; ignore PyTorch's "missing" report (we log our own)
        self.load_state_dict(filtered, strict=strict)
        
        if skipped_missing_in_model:
            print(f"--- Skipped {len(skipped_missing_in_model)} keys "
                f"not present in model (e.g. {skipped_missing_in_model[:5]})")
        if skipped_shape:
            examples = list(skipped_shape.items())[:5]
            msg_lines = [
                f"  -- {k}: ckpt {ckpt_shape} vs model {model_shape}"
                for k, (ckpt_shape, model_shape) in examples
            ]
            print(f"--- Skipped {len(skipped_shape)} keys due to shape mismatch:\n"
                + "\n".join(msg_lines))

    # --------------------------- freezing API -----------------------------

    def freeze_all(self) -> None:
        """Freeze tokenizer + encoder (useful for head-only warmup)."""
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze tokenizer + encoder."""
        for p in self.tokenizer.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_tokenizer(self) -> None:
        for p in self.tokenizer.parameters():
            p.requires_grad = False

    def unfreeze_tokenizer(self) -> None:
        for p in self.tokenizer.parameters():
            p.requires_grad = True

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_encoder_except_last_k(self, k: int) -> None:
        """
        Freeze encoder blocks except the last k blocks.

        Example: k=2 → last 2 blocks trainable, earlier ones frozen.
        """
        blocks = self.encoder.blocks
        # Freeze all blocks
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
        # Unfreeze last k
        if k > 0:
            for block in blocks[-k:]:
                for p in block.parameters():
                    p.requires_grad = True

    def _build_sequence_metadata(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = next(iter(data_dict.values())).shape[0]
        lengths: List[int] = [t.shape[1] for t in data_dict.values()]
        N_total = sum(lengths)
        device = next(iter(data_dict.values())).device

        week_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        is_temporal = torch.zeros(B, N_total, dtype=torch.bool, device=device)
        modality_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)

        start = 0
        for modality, modality_data in data_dict.items():
            T = modality_data.shape[1]
            end = start + T

            mod_idx = self.modality_to_idx.get(modality, 0)
            modality_indices[:, start:end] = mod_idx

            temporal = modality in self.temporal_modalities
            if temporal:
                is_temporal[:, start:end] = True
                week_info = batch_dict.get(modality, {}).get("week_indices", None)
                if week_info is not None:
                    if isinstance(week_info, torch.Tensor):
                        w = week_info
                    else:
                        w = torch.as_tensor(week_info, device=device)
                    if w.dim() == 1:
                        w = w.unsqueeze(0).expand(B, -1)
                    elif w.shape[0] != B:
                        w = w[:B]
                    week_indices[:, start:end] = w.to(device)
            else:
                is_temporal[:, start:end] = False

            start = end

        return week_indices, is_temporal, modality_indices

    def forward(self, batch: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        data_dict: Dict[str, torch.Tensor] = {}
        valid_mask_dict: Dict[str, Optional[torch.Tensor]] = {}

        for modality, m_dict in batch.items():
            data = m_dict["data"]
            if data.dim() == 2:
                data = data.unsqueeze(1)

            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = data

            valid_mask = m_dict.get("valid_mask", None)
            if valid_mask is not None and valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0).expand(data.shape[0], -1)
            valid_mask_dict[modality] = valid_mask

        tokens = self.tokenizer(data_dict, valid_mask_dict)  # (B, N, D)

        week_indices, is_temporal, modality_indices = self._build_sequence_metadata(
            data_dict=data_dict,
            batch_dict=batch,
        )

        encoded = self.encoder(
            tokens,
            week_indices=week_indices,
            is_temporal=is_temporal,
            modality_indices=modality_indices,
        )  # (B, N, D)

        # Mean pooling over all tokens; if you prefer temporal-only, you can mask here
        global_repr = encoded.mean(dim=1)  # (B, D)
        return global_repr