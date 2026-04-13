"""
Unified CropFM finetuning backbone.

This file is intentionally decoder-free for downstream finetuning:
- It loads tokenizer + encoder weights from pretraining checkpoints.
- It supports MAE-family and encoder-only contrastive-family checkpoints.
- It supports CLS or mean pooling.

Notes:
- Pretraining masking style (random vs structured) does not change finetuning forward.
- Decoder/reconstruction heads are ignored on purpose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from omegaconf import DictConfig

from cropfm.models.arch.mae import ModalityTokenizer
from cropfm.models.arch.mae_worldcereal_dual import EncoderWithCLS


class CropFMBackboneFT(nn.Module):
    """
    Single finetuning backbone for multiple pretraining variants.

    Args:
        modalities_config: CropFM modalities config dict.
        model_family:
            - "mae": standard MAE-family checkpoints.
            - "no_mae_contrastive": encoder-only contrastive/EAM checkpoints.
        pooling:
            - "cls": use CLS token embedding.
            - "mean": mean-pool all encoded tokens.
        exclude_modalities: optional list of modalities to exclude from encoder input.
            If None and model_family == "no_mae_contrastive", defaults to
            ["worldcereal_cropmask", "worldcereal_cropcalendar"] to match pretraining.
    """

    DEFAULT_TEMPORAL_MODALITIES = ["agera5", "sentinel1", "sentinel2", "fapar"]
    DEFAULT_STATIC_MODALITIES = [
        "soil",
        "elevation",
        "worldcereal_cropmask",
        "worldcereal_cropcalendar",
        "encoded_coordinates",
    ]

    def __init__(
        self,
        modalities_config: Dict[str, Any],
        model_family: str = "mae",
        pooling: str = "cls",
        embedding_dim: int = 256,
        encoder_depth: int = 6,
        encoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_sequence_length: int = 1000,
        use_pos_embedding: bool = True,
        encoder_attention: DictConfig | None = None,
        temporal_modalities: Optional[List[str]] = None,
        static_modalities: Optional[List[str]] = None,
        exclude_modalities: Optional[List[str]] = None,
        include_projection_head: bool = False,
        projection_dim: int = 256,
    ):
        super().__init__()

        self.model_family = model_family
        self.pooling = pooling
        self.include_projection_head = include_projection_head

        if self.pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling '{self.pooling}'. Use 'cls' or 'mean'.")
        if self.model_family not in {"mae", "no_mae_contrastive"}:
            raise ValueError(
                f"Unsupported model_family '{self.model_family}'. "
                "Use 'mae' or 'no_mae_contrastive'."
            )

        if "modality_list" in modalities_config:
            raw_modalities = modalities_config["modality_list"]
        else:
            raw_modalities = modalities_config

        if exclude_modalities is None and self.model_family == "no_mae_contrastive":
            exclude_modalities = ["worldcereal_cropmask", "worldcereal_cropcalendar"]
        exclude_set = set(exclude_modalities or [])

        modality_dims: Dict[str, int] = {}
        for mod_name, cfg in raw_modalities.items():
            if mod_name in exclude_set:
                continue
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
        self.encoder = EncoderWithCLS(
            embedding_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            use_pos_embedding=use_pos_embedding,
            num_modalities=len(all_modalities),
            attention_config=encoder_attention,
            use_cls_token=(self.pooling == "cls"),
        )

        # Optional projection head for compatibility with encoder-only contrastive checkpoints.
        self.visual_proj = (
            nn.Linear(embedding_dim, projection_dim) if self.include_projection_head else None
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

    @staticmethod
    def _get_state_dict(ckpt_path: str, map_location: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        return ckpt.get("state_dict", ckpt)

    @torch.no_grad()
    def load_from_cropfm_checkpoint(
        self,
        ckpt_path: str,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
        load_projection_head: bool = False,
    ) -> None:
        """
        Load tokenizer/encoder (and optional projection head) from pretraining ckpt.

        Supports keys with prefixes:
        - model.tokenizer.*, model.encoder.*, model.visual_proj.*
        - tokenizer.*, encoder.*, visual_proj.*
        """
        state_dict = self._get_state_dict(ckpt_path=ckpt_path, map_location=map_location)

        allowed_prefixes = ["tokenizer.", "encoder."]
        if load_projection_head and self.visual_proj is not None:
            allowed_prefixes.append("visual_proj.")

        candidate_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            if any(k.startswith(prefix) for prefix in allowed_prefixes):
                candidate_state[k] = v

        model_state = self.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped_missing: List[str] = []
        skipped_shape: Dict[str, Tuple[torch.Size, torch.Size]] = {}

        for k, v in candidate_state.items():
            if k not in model_state:
                skipped_missing.append(k)
                continue
            if v.shape != model_state[k].shape:
                skipped_shape[k] = (v.shape, model_state[k].shape)
                continue
            filtered[k] = v

        self.load_state_dict(filtered, strict=strict)

        print(
            f"[CropFMBackboneFT] Loaded {len(filtered)} keys from checkpoint "
            f"({ckpt_path})."
        )
        if skipped_missing:
            print(
                f"[CropFMBackboneFT] Skipped {len(skipped_missing)} keys not present "
                f"in model (example: {skipped_missing[:5]})."
            )
        if skipped_shape:
            examples = list(skipped_shape.items())[:5]
            detail = "\n".join(
                f"  - {k}: ckpt {s_ckpt} vs model {s_model}"
                for k, (s_ckpt, s_model) in examples
            )
            print(
                f"[CropFMBackboneFT] Skipped {len(skipped_shape)} keys due to shape mismatch:\n"
                f"{detail}"
            )

    def _build_sequence_metadata(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = next(iter(data_dict.values())).shape[0]
        lengths = [t.shape[1] for t in data_dict.values()]
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

            if modality in self.temporal_modalities:
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

            start = end

        return week_indices, is_temporal, modality_indices

    def _prepare_inputs(
        self, batch: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        data_dict: Dict[str, torch.Tensor] = {}
        valid_mask_dict: Dict[str, Optional[torch.Tensor]] = {}

        for modality, m_dict in batch.items():
            if modality not in self.modality_to_idx:
                continue

            data = m_dict["data"]
            if data.dim() == 2:
                data = data.unsqueeze(1)

            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[modality] = data

            valid_mask = m_dict.get("valid_mask", None)
            if valid_mask is not None and valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0).expand(data.shape[0], -1)
            valid_mask_dict[modality] = valid_mask

        if len(data_dict) == 0:
            raise ValueError("No usable modalities found in batch for this backbone.")

        return data_dict, valid_mask_dict

    def forward_features(self, batch: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        data_dict, valid_mask_dict = self._prepare_inputs(batch)
        tokens = self.tokenizer(data_dict, valid_mask_dict)

        week_indices, is_temporal, modality_indices = self._build_sequence_metadata(
            data_dict=data_dict,
            batch_dict=batch,
        )

        encoded = self.encoder(
            tokens,
            week_indices=week_indices,
            is_temporal=is_temporal,
            modality_indices=modality_indices,
        )

        if self.pooling == "cls":
            return encoded[:, 0, :]
        return encoded.mean(dim=1)

    def forward(self, batch: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """Return pooled feature representation for downstream heads."""
        return self.forward_features(batch)

    def forward_projected(self, batch: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Return projected embedding (for contrastive-style downstream heads).
        Requires include_projection_head=True at initialization.
        """
        if self.visual_proj is None:
            raise RuntimeError(
                "Projection head is disabled. Set include_projection_head=True."
            )
        return self.visual_proj(self.forward_features(batch))

    # ---------- freezing helpers ----------
    def freeze_all(self) -> None:
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.tokenizer.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = True
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = True

    def freeze_tokenizer(self) -> None:
        for p in self.tokenizer.parameters():
            p.requires_grad = False

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_encoder_except_last_k(self, k: int) -> None:
        blocks = self.encoder.blocks
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
        if k > 0:
            for block in blocks[-k:]:
                for p in block.parameters():
                    p.requires_grad = True
