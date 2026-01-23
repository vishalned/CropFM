"""Loss computation functions for CropFM training"""

import torch
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
    
    # Other loss types can be added here in the future
    # For example:
    # if 'worldcereal_predictions' in output:
    #     losses['worldcereal'] = compute_worldcereal_loss(...)
    
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
