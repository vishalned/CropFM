"""Transformer architecture components: Attention, MLP, and Transformer Block"""

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


class Mlp(nn.Module):
    """MLP as used in Vision Transformer and related networks
    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features
        out_features: Number of output features
        act_layer: Activation layer
        bias: Whether to use bias in the linear projections
        drop: Dropout rate
    Returns:
        Output tensor
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
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
        """
        Forward pass for Mlp
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and MLP
    Args:
        dim: Dimension of the input
        num_heads: Number of heads
        mlp_ratio: Ratio of hidden features to input features
        qkv_bias: Whether to use bias in the linear projections
        drop: Dropout rate
        attn_drop: Dropout rate for the attention
        act_layer: Activation layer
        norm_layer: Normalization layer
    Returns:
        Output tensor
    """

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
            dim,
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
        """
        Forward pass for TransformerBlock
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

