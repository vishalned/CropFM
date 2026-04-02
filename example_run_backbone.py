"""
Example: load CropFMBackbonePooled and run on sample data.

Assumes `single_script_ft_pooled.py` is in the same directory.
"""

import torch
from torch import nn

from single_script_ft_pooled import CropFMBackbonePooled


def build_example_modalities_config():
    """
    Minimal example of a CropFM-style modalities config.
    Here we keep it small, but structure mimics the real one.
    """
    return {
        "modality_list": {
            # temporal: 10 timesteps, 3 variables (e.g. bands)
            "sentinel2": {"variables": ["B02", "B03", "B04"]},
            # temporal: 10 timesteps, 2 variables (dummy)
            "agera5": {"variables": ["temp", "precip"]},
            # static: 2 variables (e.g. elevation + slope)
            "elevation": {"variables": ["elev", "slope"]},
            # static: coordinates, 4 variables
            "encoded_coordinates": {"variables": ["x_sin", "x_cos", "y_sin", "y_cos"]},
        }
    }


def build_example_batch(batch_size: int = 4, timesteps: int = 10):
    """
    Create fake data shaped like the real pretraining batch.

    Returns:
        batch: dict[modality] -> {
            'data': Tensor,
            'valid_mask': Optional[Tensor],
            'week_indices': Optional[Tensor],
        }
    """
    device = torch.device("cpu")

    B, T = batch_size, timesteps

    # sentinel2: temporal [B, T, 3]
    sentinel2_data = torch.randn(B, T, 3, device=device)
    sentinel2_valid = torch.ones(B, T, dtype=torch.bool, device=device)
    sentinel2_week = torch.arange(T, device=device) % 52  # (T,)
    # agera5: temporal [B, T, 2]
    agera5_data = torch.randn(B, T, 2, device=device)
    agera5_valid = torch.ones(B, T, dtype=torch.bool, device=device)
    agera5_week = torch.arange(T, device=device) % 52

    # elevation: static [B, 2] (will be expanded to [B, 1, 2] inside backbone)
    elevation_data = torch.randn(B, 2, device=device)
    # encoded_coordinates: static [B, 4]
    coords_data = torch.randn(B, 4, device=device)

    batch = {
        "sentinel2": {
            "data": sentinel2_data,
            "valid_mask": sentinel2_valid,
            "week_indices": sentinel2_week,
        },
        "agera5": {
            "data": agera5_data,
            "valid_mask": agera5_valid,
            "week_indices": agera5_week,
        },
        "elevation": {
            "data": elevation_data,
        },
        "encoded_coordinates": {
            "data": coords_data,
        },
    }
    return batch


def main():
    modalities_cfg = build_example_modalities_config()

    backbone = CropFMBackbonePooled(
        modalities_config=modalities_cfg,
        embedding_dim=32,
        encoder_depth=2,
        encoder_num_heads=8,
        mlp_ratio=4.0,
        max_sequence_length=1000,
    )

    # Optionally load pretrained weights (uncomment and point to your ckpt)
    backbone.load_from_cropfm_checkpoint("/home/vnedungadi/CropFM/experiments/pretrain/base_pretrain_run/model_checkpoints/last.ckpt")

    # Example: head on top (for a classification task with 10 classes)
    head = nn.Linear(32, 10)

    # Example freezing patterns:
    # 1) Freeze everything, train head only (warmup)
    backbone.freeze_all()

    # 2) Later, unfreeze encoder but keep tokenizer frozen
    # backbone.unfreeze_encoder()
    # backbone.freeze_tokenizer()

    # 3) Or unfreeze only last 2 encoder blocks
    # backbone.freeze_encoder()
    # backbone.freeze_encoder_except_last_k(k=2)

    batch = build_example_batch(batch_size=4, timesteps=10)

    with torch.no_grad():
        feats = backbone(batch)      # (B, 128)
        logits = head(feats)         # (B, 10)

    print("Features shape:", feats.shape)
    print("Logits shape:  ", logits.shape)


if __name__ == "__main__":
    main()