"""Masking strategies for MAE pretraining"""

import torch
from torch import nn
from typing import Optional


class RandomMasking(nn.Module):
    """Random masking module for MAE
    
    Randomly masks tokens across the entire sequence without considering
    modality or temporal structure.
    
    Args:
        mask_ratio: Ratio of tokens to mask (default: 0.75)
    """
    
    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for RandomMasking
        
        Args:
            x: Input tensor [B, N, C]
            mask: Optional pre-computed mask (ignored, will compute new)
            
        Returns:
            masked_x: Kept tokens [B, num_kept, C]
            mask: Boolean mask [B, N] where True means masked
            kept_indices: Indices of kept tokens [B, num_kept]
            removed_indices: Indices of masked tokens [B, num_masked]
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
            masked_x[b, :kept_len] = x[b][~batch_mask]

        return masked_x, mask, kept_indices, removed_indices
