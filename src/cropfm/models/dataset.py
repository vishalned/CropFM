"""Dataset placeholder for CropMAE"""

import torch
from torch.utils.data import Dataset


class CropFMDataset(Dataset):
    """Placeholder dataset for CropFM MAE training"""

    def __init__(self):
        """Initialize dataset - placeholder for now"""
        pass

    def __len__(self) -> int:
        """Return dataset size - placeholder"""
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary mapping modality names to tensors of shape (T, D)
            where T is number of timesteps and D is feature dimension
        """
        # Placeholder - implement actual data loading here
        raise NotImplementedError("Dataset not yet implemented")

