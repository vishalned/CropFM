"""Transformer architecture components: MLP and Transformer Block"""

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from cropfm.models.arch.attention import Attention


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
        attention: Pre-instantiated attention module (optional)
        attention_config: Hydra config for attention instantiation (optional)
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
        attention: nn.Module | None = None,
        attention_config: DictConfig | None = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Handle attention instantiation
        if attention is not None:
            # Use provided attention instance
            self.attn = attention
        elif attention_config is not None:
            # Instantiate from Hydra config
            # Override dim and num_heads from config if not already set
            attn_cfg = OmegaConf.create(OmegaConf.to_container(attention_config, resolve=True))
            if 'dim' not in attn_cfg or attn_cfg.dim is None:
                attn_cfg.dim = dim
            if 'num_heads' not in attn_cfg or attn_cfg.num_heads is None:
                attn_cfg.num_heads = num_heads
            if 'qkv_bias' not in attn_cfg:
                attn_cfg.qkv_bias = qkv_bias
            if 'attn_drop' not in attn_cfg:
                attn_cfg.attn_drop = attn_drop
            if 'proj_drop' not in attn_cfg:
                attn_cfg.proj_drop = drop
            
            self.attn = instantiate(attn_cfg)
        else:
            # Fallback to default Attention
            self.attn = Attention(
                dim=dim,
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

