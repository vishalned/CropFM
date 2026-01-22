"""Attention modules: Standard and Sliding Window Attention"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.jit import Final


class Attention(nn.Module):
    """Multi-head self-attention module
    Args:
        dim: Dimension of the input
        num_heads: Number of heads
        qkv_bias: Whether to use bias in the linear projections
        attn_drop: Dropout rate for the attention
        proj_drop: Dropout rate for the projection
    Returns:
        Output tensor
    """

    fast_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Attention
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        B, N, C = x.shape
        qkv = self.qkv(x) # [B, N, 3*dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim) # [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0) # each [B, num_heads, N, head_dim]

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
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


class SlidingWindowAttention(nn.Module):
    """Multi-head self-attention with sliding window (local attention)
    
    Each token can only attend to tokens within a local window around it.
    Uses position-based masking for flexible window control.
    
    Args:
        dim: Dimension of the input (same as output)
        num_heads: Number of heads
        window_size: Size of the attention window
        qkv_bias: Whether to use bias in the linear projections
        attn_drop: Dropout rate for the attention
        proj_drop: Dropout rate for the projection
        causal: Whether to use causal masking (default: False for bidirectional MAE)
    Returns:
        Output tensor of same shape as input [B, N, C]
    """

    fast_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 64,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert window_size > 0, "window_size must be positive"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.causal = causal
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Same as Attention class: single qkv projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # d_in = d_out = dim
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SlidingWindowAttention
        Args:
            x: Input tensor of shape [B, N, C] where C = dim
        Returns:
            Output tensor of shape [B, N, C] where C = dim
        """
        B, N, C = x.shape  # C = dim
        
        # Compute QKV (same as Attention class)
        qkv = self.qkv(x)  # [B, N, 3*dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)  # each [B, num_heads, N, head_dim]

        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        
        # Create sliding window mask using position-based approach
        device = x.device
        q_positions = torch.arange(N, device=device, dtype=torch.long)  # [N]
        k_positions = torch.arange(N, device=device, dtype=torch.long)  # [N]
        
        # Compute position differences: [N, N]
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)  # [N, N]
        
        # Create sliding window mask
        # Token at position i can attend to tokens in [i - window_size//2, i + window_size//2]
        half_window = self.window_size // 2
        window_mask = (diff >= -half_window) & (diff < half_window + 1)  # [N, N]
        
        # Add causal mask if needed (for MAE, usually False)
        if self.causal:
            causal_mask = diff >= 0  # Can't attend to future tokens
            window_mask = window_mask & causal_mask
        
        # Expand mask for multi-head: [1, 1, N, N]
        window_mask_expanded = window_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply mask: set masked positions to -inf before softmax
        attn_scores = attn_scores.masked_fill(~window_mask_expanded, float('-inf'))
        
        if self.fast_attn:
            # Use fast attention path when available
            attn_mask_inverted = ~window_mask_expanded
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask_inverted,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            # Manual attention computation
            attn_weights = attn_scores.softmax(dim=-1)  # [B, num_heads, N, N]
            attn_weights = self.attn_drop(attn_weights)
            x = attn_weights @ v  # [B, num_heads, N, head_dim]

        # Reshape: [B, num_heads, N, head_dim] -> [B, N, C] where C = dim
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # [B, N, dim] -> [B, N, dim]
        x = self.proj_drop(x)
        return x
