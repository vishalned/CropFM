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
            _, strategy_mask, _, _ = self._random_mask(x, mask)
        elif strategy_idx == 1:
            _, strategy_mask, _, _ = self._modality_mask(x, mask, modality_boundaries)
        elif strategy_idx == 2:
            _, strategy_mask, _, _ = self._random_timesteps_mask(x, mask, modality_boundaries)
        elif strategy_idx == 3:
            _, strategy_mask, _, _ = self._contiguous_timesteps_mask(
                x, mask, modality_boundaries, week_indices, is_temporal
            )
        else:
            # Fallback to random masking
            _, strategy_mask, _, _ = self._random_mask(x, mask)

        # Enforce configured global mask ratio for every strategy:
        # if a structured strategy under/over-masks, top up or trim at random.
        strategy_mask = self._enforce_mask_ratio(strategy_mask)
        return self._build_mask_outputs(x, strategy_mask)
    
    def _select_strategy(self) -> int:
        """Randomly select strategy index based on weights"""
        r = random.random()
        cumsum = 0.0
        for i, weight in enumerate(self.strategy_weights):
            cumsum += weight
            if r < cumsum:
                return i
        return 0  # Fallback to first strategy

    def _enforce_mask_ratio(self, mask: torch.Tensor) -> torch.Tensor:
        """Adjust per-sample mask to exactly match configured ``mask_ratio``."""
        B, N = mask.shape
        target_masked = int(N * self.mask_ratio)

        if target_masked <= 0:
            return torch.zeros_like(mask, dtype=torch.bool)
        if target_masked >= N:
            return torch.ones_like(mask, dtype=torch.bool)

        adjusted = mask.clone()
        for b in range(B):
            current_masked = int(adjusted[b].sum().item())
            delta = target_masked - current_masked
            if delta == 0:
                continue

            if delta > 0:
                # Under-masked: add random masked tokens from currently unmasked positions.
                candidates = torch.where(~adjusted[b])[0]
                if candidates.numel() == 0:
                    continue
                add_k = min(delta, candidates.numel())
                perm = torch.randperm(candidates.numel(), device=mask.device)[:add_k]
                adjusted[b, candidates[perm]] = True
            else:
                # Over-masked: unmask random tokens from currently masked positions.
                candidates = torch.where(adjusted[b])[0]
                if candidates.numel() == 0:
                    continue
                remove_k = min(-delta, candidates.numel())
                perm = torch.randperm(candidates.numel(), device=mask.device)[:remove_k]
                adjusted[b, candidates[perm]] = False

        return adjusted
    
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
        max_kept = int(num_kept_per_batch.max().item())
        max_masked = int(num_masked_per_batch.max().item())

        # Sort indices so kept (False) come first, masked (True) last.
        # This avoids per-sample Python loops and keeps everything on-device.
        order = torch.argsort(mask.to(torch.int8), dim=1)  # (B, N)

        kept_indices = order[:, :max_kept].contiguous()
        removed_indices = order[:, N - max_masked :].contiguous() if max_masked > 0 else order[:, :0].contiguous()

        # Zero out padded indices (to match prior behavior) and zero out gathered tokens for pads.
        kept_valid = (torch.arange(max_kept, device=device).unsqueeze(0) < num_kept_per_batch.unsqueeze(1))  # (B, max_kept)
        removed_valid = (torch.arange(max_masked, device=device).unsqueeze(0) < num_masked_per_batch.unsqueeze(1)) if max_masked > 0 else None

        if max_kept > 0:
            kept_indices = torch.where(kept_valid, kept_indices, torch.zeros((), dtype=torch.long, device=device))
            masked_x = x.gather(1, kept_indices.unsqueeze(-1).expand(-1, -1, C))
            masked_x = masked_x * kept_valid.unsqueeze(-1).to(dtype=masked_x.dtype)
        else:
            masked_x = x[:, :0, :]

        if max_masked > 0 and removed_valid is not None:
            removed_indices = torch.where(removed_valid, removed_indices, torch.zeros((), dtype=torch.long, device=device))

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
        
        # For each modality, randomly mask timesteps (vectorized over batch)
        for _, (start_idx, end_idx) in modality_boundaries.items():
            T = end_idx - start_idx
            if T <= 1:
                continue

            k = int(T * self.timestep_mask_ratio)
            k = max(1, min(k, T - 1))

            scores = torch.rand(B, T, device=device)
            idx = scores.topk(k, largest=False, dim=1).indices  # (B, k)
            mod_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            mod_mask.scatter_(1, idx, True)
            mask[:, start_idx:end_idx] = mod_mask
        
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
        
        keep = torch.zeros(52, dtype=torch.bool, device=device)
        keep[torch.tensor(kept_weeks, dtype=torch.long, device=device)] = True
        
        # Initialize mask (all False = keep all)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # For temporal modalities, mask tokens that don't match kept weeks (vectorized)
        for _, (start_idx, end_idx) in modality_boundaries.items():
            # Only apply to temporal modalities
            if not is_temporal[:, start_idx:end_idx].any().item():
                continue
            mod_week_indices = week_indices[:, start_idx:end_idx].long()  # (B, T)
            mask[:, start_idx:end_idx] = ~keep[mod_week_indices]
        
        return self._build_mask_outputs(x, mask)
