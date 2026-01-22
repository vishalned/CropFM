"""Masking strategies for MAE pretraining"""

import torch
from torch import nn
from typing import Optional, Union
import random


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


class StructuredMasking(nn.Module):
    """Structured masking that randomly selects one of 4 masking strategies
    
    Strategies:
    0: RandomMasking - Random tokens across entire sequence
    1: ModalityMasking - Mask entire modalities
    2: RandomTimestepsMasking - Mask random timesteps per modality
    3: ContiguousTimestepsMasking - Mask contiguous temporal patterns
    
    Args:
        mask_ratio: Overall mask ratio for random masking (default: 0.75)
        modality_mask_ratio: Ratio of modalities to mask (default: 0.45)
        timestep_mask_ratio: Ratio of timesteps to mask per modality (default: 0.5)
        contiguous_pattern: List of patterns for contiguous masking. Randomly selects one per forward pass.
            Patterns: "monthly", "first_half", "second_half"
            (default: ["monthly"])
        strategy_weights: Weights for each strategy [random, modality, timesteps, contiguous] (default: equal)
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.75,
        modality_mask_ratio: float = 0.45,
        timestep_mask_ratio: float = 0.5,
        contiguous_pattern: list[str] = None,
        strategy_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.modality_mask_ratio = modality_mask_ratio
        self.timestep_mask_ratio = timestep_mask_ratio
        
        # Always use a list - convert single string if provided, default to ["monthly"]
        if contiguous_pattern is None:
            self.contiguous_patterns = ["monthly"]
        elif isinstance(contiguous_pattern, str):
            # Backward compatibility: convert single string to list
            self.contiguous_patterns = [contiguous_pattern]
        else:
            self.contiguous_patterns = contiguous_pattern
        
        
        # Normalize strategy weights (default: equal probability)
        if strategy_weights is None:
            self.strategy_weights = [0.25, 0.25, 0.25, 0.25]
        else:
            total = sum(strategy_weights)
            self.strategy_weights = [w / total for w in strategy_weights]
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        modality_boundaries: Optional[dict[str, tuple[int, int]]] = None,
        week_indices: Optional[torch.Tensor] = None,
        is_temporal: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that randomly selects one of 4 masking strategies
        
        Args:
            x: Input tensor [B, N, C]
            mask: Optional pre-computed mask (ignored)
            modality_boundaries: Dict mapping modality_name -> (start_idx, end_idx)
            week_indices: Tensor [B, N] with week indices (0-51) for temporal tokens
            is_temporal: Boolean tensor [B, N] indicating temporal vs static tokens
            modality_indices: Tensor [B, N] with modality indices
            
        Returns:
            masked_x: Kept tokens [B, num_kept, C]
            mask: Boolean mask [B, N] where True means masked
            kept_indices: Indices of kept tokens [B, num_kept]
            removed_indices: Indices of masked tokens [B, num_masked]
        """
        # Randomly select strategy (per forward call, same for entire batch)
        strategy_idx = self._select_strategy()
        
        if strategy_idx == 0:
            return self._random_mask(x, mask)
        elif strategy_idx == 1:
            return self._modality_mask(x, mask, modality_boundaries)
        elif strategy_idx == 2:
            return self._random_timesteps_mask(x, mask, modality_boundaries)
        elif strategy_idx == 3:
            return self._contiguous_timesteps_mask(x, mask, modality_boundaries, week_indices, is_temporal)
        else:
            # Fallback to random masking
            return self._random_mask(x, mask)
    
    def _select_strategy(self) -> int:
        """Randomly select strategy index based on weights"""
        r = random.random()
        cumsum = 0.0
        for i, weight in enumerate(self.strategy_weights):
            cumsum += weight
            if r < cumsum:
                return i
        return 0  # Fallback to first strategy
    
    def _build_mask_outputs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to convert boolean mask to mask outputs
        
        Args:
            x: Input tensor [B, N, C]
            mask: Boolean mask [B, N] where True means masked
            
        Returns:
            masked_x, mask, kept_indices, removed_indices
        """
        B, N, C = x.shape
        device = x.device
        
        # Count kept and masked tokens per batch
        num_kept_per_batch = (~mask).sum(dim=1)  # [B]
        num_masked_per_batch = mask.sum(dim=1)  # [B]
        max_kept = num_kept_per_batch.max().item()
        max_masked = num_masked_per_batch.max().item()
        
        kept_indices = torch.zeros(B, max_kept, dtype=torch.long, device=device)
        removed_indices = torch.zeros(B, max_masked, dtype=torch.long, device=device)
        masked_x = torch.zeros(B, max_kept, C, device=device)
        
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
    
    def _random_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strategy 0: Random masking (uses RandomMasking logic)"""
        B, N, C = x.shape
        device = x.device
        
        num_masked = int(N * self.mask_ratio)
        num_kept = N - num_masked
        indices = torch.rand(B, N, device=device).argsort(dim=1)
        mask = indices < num_masked
        
        return self._build_mask_outputs(x, mask)
    
    def _modality_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        modality_boundaries: Optional[dict[str, tuple[int, int]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strategy 1: Mask entire modalities"""
        B, N, C = x.shape
        device = x.device
        
        # Fallback to random masking if modality_boundaries not provided
        if modality_boundaries is None or len(modality_boundaries) == 0:
            return self._random_mask(x, mask)
        
        # Initialize mask (all False = keep all)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # Get list of modalities
        modalities = list(modality_boundaries.keys())
        num_modalities = len(modalities)
        
        # Calculate how many modalities to mask (ensure at least 1 remains)
        num_modalities_to_mask = max(1, int(num_modalities * self.modality_mask_ratio))
        num_modalities_to_mask = min(num_modalities_to_mask, num_modalities - 1)
        
        # Randomly select modalities to mask
        modalities_to_mask = random.sample(modalities, num_modalities_to_mask)
        
        # Mask all tokens for selected modalities
        for mod_name in modalities_to_mask:
            start_idx, end_idx = modality_boundaries[mod_name]
            mask[:, start_idx:end_idx] = True
        
        return self._build_mask_outputs(x, mask)
    
    def _random_timesteps_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        modality_boundaries: Optional[dict[str, tuple[int, int]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strategy 2: Mask random timesteps within each modality"""
        B, N, C = x.shape
        device = x.device
        
        # Fallback to random masking if modality_boundaries not provided
        if modality_boundaries is None or len(modality_boundaries) == 0:
            return self._random_mask(x, mask)
        
        # Initialize mask (all False = keep all)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # For each modality, randomly mask timesteps
        for mod_name, (start_idx, end_idx) in modality_boundaries.items():
            T = end_idx - start_idx
            
            # Skip static modalities (T=1)
            if T <= 1:
                continue
            
            # Calculate how many timesteps to mask
            num_timesteps_to_mask = int(T * self.timestep_mask_ratio)
            num_timesteps_to_mask = max(1, min(num_timesteps_to_mask, T - 1))
            
            # For each batch, randomly select timesteps to mask
            for b in range(B):
                timesteps_to_mask = torch.randperm(T, device=device)[:num_timesteps_to_mask]
                mask[b, start_idx + timesteps_to_mask] = True
        
        return self._build_mask_outputs(x, mask)
    
    def _pattern_to_weeks(self, pattern: str) -> Optional[list[int]]:
        """Convert pattern string to list of kept weeks"""
        if pattern == "monthly":
            return [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        elif pattern == "first_half":
            return list(range(26, 52))  # Weeks 26-51
        elif pattern == "second_half":
            return list(range(0, 26))   # Weeks 0-25
        else:
            return None  # Unknown pattern
    
    def _contiguous_timesteps_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        modality_boundaries: Optional[dict[str, tuple[int, int]]] = None,
        week_indices: Optional[torch.Tensor] = None,
        is_temporal: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strategy 3: Mask contiguous timesteps based on temporal patterns"""
        B, N, C = x.shape
        device = x.device
        
        # Fallback to random masking if required info not provided
        if modality_boundaries is None or week_indices is None or is_temporal is None:
            return self._random_mask(x, mask)
        
        # Randomly select one pattern from the list (or use single pattern)
        selected_pattern = random.choice(self.contiguous_patterns)
        
        # Convert pattern to list of kept weeks
        kept_weeks = self._pattern_to_weeks(selected_pattern)
        
        if kept_weeks is None:
            # Unknown pattern, fallback to random
            return self._random_mask(x, mask)
        
        kept_weeks_set = set(kept_weeks)
        
        # Initialize mask (all False = keep all)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # For temporal modalities, mask tokens that don't match kept weeks
        for mod_name, (start_idx, end_idx) in modality_boundaries.items():
            # Only apply to temporal modalities
            # Check if this modality has temporal tokens (use is_temporal)
            mod_is_temporal = is_temporal[:, start_idx:end_idx].any().item()
            
            if not mod_is_temporal:
                # Static modality: keep all
                continue
            
            # Get week indices for this modality
            mod_week_indices = week_indices[:, start_idx:end_idx]  # [B, T]
            
            # Mask tokens where week index is NOT in kept_weeks
            for b in range(B):
                for t_idx in range(end_idx - start_idx):
                    week_idx = mod_week_indices[b, t_idx].item()
                    if week_idx not in kept_weeks_set:
                        mask[b, start_idx + t_idx] = True
        
        return self._build_mask_outputs(x, mask)
