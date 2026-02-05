"""Loss computation functions for CropFM training"""

import torch
import torch.nn.functional as F
from typing import Dict, Any


def compute_losses(
    output: Dict[str, Any],
    batch: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute losses from model output.
    
    Args:
        output: Dictionary from model forward pass (contains 'reconstructions', 'modality_masks', etc.)
        batch: Batch of data with targets
        
    Returns:
        Dictionary mapping loss type to dictionary of losses:
        {
            'reconstruction': {'modality1': loss1, 'modality2': loss2, ...},
            # Other loss types can be added here in the future
        }
    """
    losses = {}
    
    # Handle reconstruction loss
    if 'reconstructions' in output:
        reconstruction_losses = compute_reconstruction_loss(
            reconstructions=output['reconstructions'],
            targets=batch,
            modality_masks=output.get('modality_masks', {}),
        )
        losses['reconstruction'] = reconstruction_losses
    
    # Handle worldcereal prediction losses
    if 'worldcereal_predictions' in output:
        worldcereal_losses = compute_worldcereal_loss(
            worldcereal_predictions=output['worldcereal_predictions'],
            targets=batch,
        )
        if len(worldcereal_losses) > 0:
            losses['worldcereal'] = worldcereal_losses
    
    # Handle contrastive loss
    if 'projected_representation' in output and 'worldcereal_cropmask' in batch:
        contrastive_losses = compute_contrastive_loss(
            projected_representation=output['projected_representation'],
            crop_mask_labels=batch['worldcereal_cropmask']['data'],
            temperature=0.07,  # Could be made configurable
        )
        if len(contrastive_losses) > 0:
            losses['contrastive'] = contrastive_losses
    
    return losses


def compute_reconstruction_loss(
    reconstructions: Dict[str, torch.Tensor],
    targets: Dict[str, Dict[str, torch.Tensor]],
    modality_masks: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute reconstruction loss (MSE) for each modality on masked tokens only.
    
    Args:
        reconstructions: Dict mapping modality name to predicted reconstruction (B, T, D)
        targets: Dict mapping modality name to {'data': tensor (B, T, D), ...}
        modality_masks: Dict mapping modality name to mask tensor (B, T) where True = masked token
        
    Returns:
        Dict mapping modality name to loss scalar tensor
    """
    losses = {}
    
    for modality_name in reconstructions.keys():
        if modality_name not in targets:
            continue
            
        pred = reconstructions[modality_name]  # (B, T, D)
        target = targets[modality_name]['data']  # (B, T, D)
        mask = modality_masks[modality_name]  # (B, T) - True for masked tokens
        
        # Compute MSE only on masked tokens
        # Expand mask to match feature dimension: (B, T) -> (B, T, D)
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # (B, T, D)
        
        # Compute loss only where:
        #  - token is masked by MAE (mask_expanded == True)
        #  - and target is not NaN (to avoid NaN losses from missing data)
        # IMPORTANT: we must index with the mask instead of multiplying by it,
        #            because 0 * NaN is still NaN in floating-point arithmetic.
        diff = pred - target
        valid_target_mask = ~torch.isnan(target)
        combined_mask = mask_expanded & valid_target_mask  # (B, T, D)

        denom = combined_mask.sum()
        if denom == 0:
            continue

        # Select only masked & valid positions, then compute MSE over them
        selected_diff = diff[combined_mask]  # (denom,)
        loss = (selected_diff ** 2).mean()
        
        losses[modality_name] = loss
    
    return losses


def compute_worldcereal_loss(
    worldcereal_predictions: Dict[str, torch.Tensor],
    targets: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Compute worldcereal prediction losses (classification for crop_mask, regression for crop_calendar).
    
    Args:
        worldcereal_predictions: Dict with 'worldcereal_cropmask' (B, 4) and/or 'worldcereal_cropcalendar' (B, 2) [sos, season_length]
        targets: Batch dict containing worldcereal targets
        
    Returns:
        Dict mapping prediction type to loss scalar tensor
    """
    losses = {}
    
    # Crop mask classification loss
    if 'worldcereal_cropmask' in worldcereal_predictions and 'worldcereal_cropmask' in targets:
        crop_mask_pred = worldcereal_predictions['worldcereal_cropmask']  # (B, 4)
        crop_mask_target = targets['worldcereal_cropmask']['data']  # (B, 2) [aez_id, crop_mask] or (B, 1)
        
        # Extract crop_mask (second element) if shape is (B, 2)
        if crop_mask_target.shape[-1] == 2:
            crop_mask_target = crop_mask_target[:, 1]  # (B,) - extract crop_mask column
        elif crop_mask_target.dim() > 1:
            crop_mask_target = crop_mask_target.squeeze(-1)  # (B,)
        
        # Convert to long for CrossEntropyLoss
        crop_mask_target = crop_mask_target.long()
        
        # Only compute loss where target is valid (not NaN)
        valid_mask = ~torch.isnan(crop_mask_target.float())
        if valid_mask.sum() > 0:
            loss = F.cross_entropy(
                crop_mask_pred[valid_mask],
                crop_mask_target[valid_mask]
            )
            losses['crop_mask'] = loss
    
    # Crop calendar regression loss
    # Predict SOS and season_length (not EOS) because EOS < SOS indicates year-crossing seasons
    if 'worldcereal_cropcalendar' in worldcereal_predictions and 'worldcereal_cropcalendar' in targets:
        crop_cal_pred = worldcereal_predictions['worldcereal_cropcalendar']  # (B, 2) - sos, season_length
        crop_cal_target_full = targets['worldcereal_cropcalendar']['data']  # (B, 6) - [maize_sos, maize_eos, winter_sos, winter_eos, spring_sos, spring_eos]
        
        # Get crop mask to determine which calendar values to use
        crop_mask_target = None
        if 'worldcereal_cropmask' in targets:
            crop_mask_data = targets['worldcereal_cropmask']['data']
            if crop_mask_data.dim() > 1:
                crop_mask_target = crop_mask_data.squeeze(-1)
            else:
                crop_mask_target = crop_mask_data
            crop_mask_target = crop_mask_target.long()
        
        # Extract (sos, eos) and convert to (sos, season_length) based on crop mask
        # Crop mask: 0=no crop, 1=maize, 2=winter cereals, 3=spring cereals
        # Calendar indices: [maize_sos, maize_eos, winter_sos, winter_eos, spring_sos, spring_eos]
        # Season length: when EOS < SOS (year-crossing), length = (365 - SOS) + EOS; else length = EOS - SOS
        B = crop_cal_pred.shape[0]
        crop_cal_target = torch.zeros(B, 2, device=crop_cal_pred.device, dtype=crop_cal_pred.dtype)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=crop_cal_pred.device)
        
        def _sos_eos_to_sos_seasonlen(sos: torch.Tensor, eos: torch.Tensor, days_per_year: int = 365) -> torch.Tensor:
            """Convert (sos, eos) to (sos, season_length). Handles year-crossing (eos < sos)."""
            season_length = torch.where(
                eos >= sos,
                eos - sos,
                (days_per_year - sos) + eos
            )
            return torch.stack([sos.float(), season_length.float()], dim=-1)
        
        if crop_mask_target is not None:
            # Maize (class 1): use indices [0, 1]
            maize_mask = (crop_mask_target == 1)
            if maize_mask.any():
                sos, eos = crop_cal_target_full[maize_mask, 0], crop_cal_target_full[maize_mask, 1]
                crop_cal_target[maize_mask] = _sos_eos_to_sos_seasonlen(sos, eos)
                valid_mask[maize_mask] = ~torch.isnan(crop_cal_target_full[maize_mask, 0:2]).any(dim=-1)
            
            # Winter cereals (class 2): use indices [2, 3]
            winter_mask = (crop_mask_target == 2)
            if winter_mask.any():
                sos, eos = crop_cal_target_full[winter_mask, 2], crop_cal_target_full[winter_mask, 3]
                crop_cal_target[winter_mask] = _sos_eos_to_sos_seasonlen(sos, eos)
                valid_mask[winter_mask] = ~torch.isnan(crop_cal_target_full[winter_mask, 2:4]).any(dim=-1)
            
            # Spring cereals (class 3): use indices [4, 5]
            spring_mask = (crop_mask_target == 3)
            if spring_mask.any():
                sos, eos = crop_cal_target_full[spring_mask, 4], crop_cal_target_full[spring_mask, 5]
                crop_cal_target[spring_mask] = _sos_eos_to_sos_seasonlen(sos, eos)
                valid_mask[spring_mask] = ~torch.isnan(crop_cal_target_full[spring_mask, 4:6]).any(dim=-1)
            
            # Class 0 (no crop/"other") is excluded from valid_mask (stays False)
        else:
            # Fallback: if no crop mask, check for NaN in any calendar values
            valid_mask = ~torch.isnan(crop_cal_target_full).any(dim=-1)
            sos, eos = crop_cal_target_full[:, 0], crop_cal_target_full[:, 1]
            crop_cal_target = _sos_eos_to_sos_seasonlen(sos, eos)
        
        # Compute MSE only for valid samples (excludes "other"/no crop)
        if valid_mask.sum() > 0:
            loss = F.mse_loss(
                crop_cal_pred[valid_mask],
                crop_cal_target[valid_mask]
            )
            losses['crop_calendar'] = loss
    
    return losses


def compute_contrastive_loss(
    projected_representation: torch.Tensor,  # (B, projection_dim)
    crop_mask_labels: torch.Tensor,  # (B, 2) [aez_id, crop_mask] or (B, 1)
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """
    Compute InfoNCE contrastive loss for crop mask prediction.
    
    Positive pairs: samples with same crop type
    Negative pairs: samples with different crop types
    
    Args:
        projected_representation: Projected representations (B, projection_dim)
        crop_mask_labels: Crop type labels (B, 2) or (B, 1) with values 0-3
        temperature: Temperature scaling for softmax (default: 0.07)
        
    Returns:
        Dict with single key 'contrastive': loss scalar tensor
    """
    B = projected_representation.shape[0]
    device = projected_representation.device
    
    # Extract crop_mask (second element) if shape is (B, 2)
    if crop_mask_labels.shape[-1] == 2:
        crop_mask_target = crop_mask_labels[:, 1]  # (B,) - extract crop_mask column
    elif crop_mask_labels.dim() > 1:
        crop_mask_target = crop_mask_labels.squeeze(-1)  # (B,)
    else:
        crop_mask_target = crop_mask_labels
    
    # Convert to long
    crop_mask_target = crop_mask_target.long()
    
    # Only compute loss where target is valid (not NaN)
    valid_mask = ~torch.isnan(crop_mask_target.float())
    if valid_mask.sum() < 2:
        # Need at least 2 samples for contrastive loss
        return {}
    
    # Filter to valid samples
    valid_repr = projected_representation[valid_mask]  # (valid_B, projection_dim)
    valid_labels = crop_mask_target[valid_mask]  # (valid_B,)
    valid_B = valid_repr.shape[0]
    
    # Normalize representations
    valid_repr = F.normalize(valid_repr, p=2, dim=1)
    
    # Compute similarity matrix: (valid_B, valid_B)
    sim_matrix = torch.matmul(valid_repr, valid_repr.T) / temperature
    
    # Create mask for positive pairs (same crop type, excluding self)
    labels_expanded = valid_labels.unsqueeze(1)  # (valid_B, 1)
    labels_expanded_T = valid_labels.unsqueeze(0)  # (1, valid_B)
    positive_mask = (labels_expanded == labels_expanded_T).float()  # (valid_B, valid_B)
    
    # Remove diagonal (self-similarity)
    eye_mask = torch.eye(valid_B, device=device)
    positive_mask = positive_mask * (1 - eye_mask)
    
    # Vectorized InfoNCE: -log( sum_p exp(sim_i,p) / sum_{j!=i} exp(sim_i,j) )
    exp_sim = torch.exp(sim_matrix)
    denominator = (exp_sim * (1 - eye_mask)).sum(dim=1)  # (valid_B,)
    numerator = (exp_sim * positive_mask).sum(dim=1)  # (valid_B,)
    loss_per_sample = -torch.log(numerator / denominator.clamp(min=1e-8) + 1e-8)
    
    # Only average over samples that have at least one positive
    has_positives = positive_mask.sum(dim=1) > 0
    if has_positives.sum() == 0:
        return {}
    
    contrastive_loss = loss_per_sample[has_positives].mean()
    return {'contrastive': contrastive_loss}


def aggregate_losses(
    losses: Dict[str, Dict[str, torch.Tensor]],
    equal_weighting: bool = True,
    weights: Dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Aggregate multiple losses into a single scalar.
    
    Args:
        losses: Dictionary mapping loss type to dictionary of losses:
            {
                'reconstruction': {'modality1': loss1, 'modality2': loss2, ...},
                # Other loss types...
            }
        equal_weighting: If True, average all losses equally. If False, use weights dict.
        weights: Dict mapping loss name to weight (only used if equal_weighting=False)
            Can specify weights for loss types (e.g., 'reconstruction': 1.0) or
            individual losses (e.g., 'modality1': 0.5)
        
    Returns:
        Aggregated loss scalar tensor
    """
    if len(losses) == 0:
        # Return zero loss if no losses
        # Try to get device from any loss value
        device = torch.device('cpu')
        for loss_type_dict in losses.values():
            if len(loss_type_dict) > 0:
                device = next(iter(loss_type_dict.values())).device
                break
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Flatten all losses into a single dict for aggregation
    flattened_losses = {}
    for loss_type, loss_dict in losses.items():
        for loss_name, loss_value in loss_dict.items():
            # Use format: 'loss_type:loss_name' for flattened keys
            flattened_key = f"{loss_type}:{loss_name}"
            flattened_losses[flattened_key] = loss_value
    
    if len(flattened_losses) == 0:
        device = torch.device('cpu')
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    if equal_weighting:
        # Simple average
        loss_values = list(flattened_losses.values())
        return sum(loss_values) / len(loss_values)
    else:
        # Weighted sum
        if weights is None:
            raise ValueError("weights must be provided when equal_weighting=False")
        
        total_loss = 0.0
        for flattened_key, loss_value in flattened_losses.items():
            # Try to find weight for flattened key first, then try loss_type, then default to 1.0
            weight = weights.get(flattened_key, 1.0)
            if weight == 1.0:
                # Try to get weight by loss type (e.g., 'reconstruction')
                loss_type = flattened_key.split(':')[0]
                weight = weights.get(loss_type, 1.0)
            total_loss += weight * loss_value
        
        return total_loss
