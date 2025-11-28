"""Positional encoding utilities"""

import numpy as np
import torch


def get_sinusoid_encoding_table(positions: int | list[int], d_hid: int, T: int = 10000) -> torch.Tensor:
    """
    Sinusoid position encoding table
    
    Args:
        positions: int or list of integers, if int uses range(positions)
        d_hid: Hidden dimension size (embedding dimension)
        T: Temperature parameter for encoding
        
    Returns:
        Position encoding table of shape (positions, d_hid)
    """
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position: int, hid_idx: int) -> float:
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position: int) -> list[float]:
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)
