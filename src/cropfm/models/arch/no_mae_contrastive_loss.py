"""Encoder-only CropFM for contrastive / EAM pretraining (no MAE masking, no decoder).

Tokenization matches the MAE encoder path (temporal ``week_indices``, ``is_temporal``,
``modality_indices``) but **excludes** ``worldcereal_cropmask`` and ``worldcereal_cropcalendar``
from the input—they are targets / supervision only and may still be present in the batch
for losses. There is no masking and no reconstruction head.
"""

from __future__ import annotations

import torch
from torch import nn
from omegaconf import DictConfig

from cropfm.models.arch.mae import AUXILIARY_BATCH_KEYS, ModalityTokenizer
from cropfm.models.arch.mae_worldcereal_dual import EncoderWithCLS


class CropFMNoMAEContrastiveLoss(nn.Module):
    """Encoder-only backbone: same tokenizer + structure as CropMAE, without MAE."""

    primary_metric_name = "loss"

    def __init__(
        self,
        modalities: dict,
        embedding_dim: int = 256,
        encoder_depth: int = 6,
        encoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_sequence_length: int = 1000,
        use_pos_embedding: bool = True,
        encoder_attention: DictConfig | None = None,
        use_cls_token: bool = True,
        visual_dim: int = 256,
        **kwargs,
    ):
        super().__init__()
        del kwargs  # Hydra may pass unused keys

        modality_dims = {
            modality: len(modalities['modality_list'][modality]['variables'])
            for modality in modalities['modality_list']
        }
        worldcereal_modalities = ['worldcereal_cropmask', 'worldcereal_cropcalendar']
        input_modality_dims = {
            k: v for k, v in modality_dims.items() if k not in worldcereal_modalities
        }

        # Match CropMAE.forward temporal vs static (mae.py)
        self.temporal_modalities = ['agera5', 'sentinel1', 'sentinel2', 'fapar']

        all_modalities = list(input_modality_dims.keys())
        self.modality_to_idx = {m: i for i, m in enumerate(all_modalities)}
        num_modalities = len(all_modalities)

        self.embedding_dim = embedding_dim
        self.visual_dim = visual_dim

        self.tokenizer = ModalityTokenizer(
            modality_dims=input_modality_dims,
            embedding_dim=embedding_dim,
        )
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
        self.visual_proj = nn.Linear(embedding_dim, visual_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

    def forward(self, x: dict[str, dict]) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Batch dict; may include worldcereal targets and ``satclip_embedding``
            (loss-only), which are **ignored** here.
        Returns:
            ``visual_embedding`` of shape ``(B, visual_dim)``.
        """
        worldcereal_modalities = ['worldcereal_cropmask', 'worldcereal_cropcalendar']
        input_x = {
            k: v
            for k, v in x.items()
            if k not in worldcereal_modalities and k not in AUXILIARY_BATCH_KEYS
        }

        data_dict: dict[str, torch.Tensor] = {}
        for modality, modality_dict in input_x.items():
            modality_data = modality_dict['data']
            modality_data = torch.nan_to_num(modality_data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = modality_data

        valid_mask_dict: dict[str, torch.Tensor | None] = {}
        for modality, modality_dict in input_x.items():
            if 'valid_mask' in modality_dict:
                valid_mask_dict[modality] = modality_dict['valid_mask']
            else:
                valid_mask_dict[modality] = None

        tokens = self.tokenizer(data_dict, valid_mask_dict)
        B = tokens.shape[0]
        N_total = tokens.shape[1]
        device = tokens.device

        week_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)
        is_temporal = torch.zeros(B, N_total, dtype=torch.bool, device=device)
        modality_indices = torch.zeros(B, N_total, dtype=torch.long, device=device)

        start_idx = 0
        for modality_name, modality_data in data_dict.items():
            B_mod, T, D = modality_data.shape
            if modality_name in self.modality_to_idx:
                mod_idx = self.modality_to_idx[modality_name]
                modality_indices[:, start_idx : start_idx + T] = mod_idx
            if modality_name in self.temporal_modalities:
                is_temporal[:, start_idx : start_idx + T] = True
                if modality_name in input_x and 'week_indices' in input_x[modality_name]:
                    week_idx = input_x[modality_name]['week_indices']
                    if week_idx.dim() == 1:
                        week_idx = week_idx.unsqueeze(0).expand(B, -1)
                    elif week_idx.shape[0] != B:
                        week_idx = week_idx[:B]
                    week_indices[:, start_idx : start_idx + T] = week_idx.to(device)
            else:
                is_temporal[:, start_idx : start_idx + T] = False
            start_idx += T

        encoded = self.encoder(
            tokens,
            week_indices=week_indices,
            is_temporal=is_temporal,
            modality_indices=modality_indices,
        )
        if self.encoder.use_cls_token:
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)

        z = self.visual_proj(pooled)
        return {'visual_embedding': z}
