"""Dataset of multi-modal agricultural data for CropMAE"""

import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
from typing import Callable, List, Dict, Optional


class CropFMDataset(Dataset):
    """
    PyTorch Dataset for loading multi-modal agricultural time series data from Zarr format.
    
    Supports both temporal modalities (satellite imagery, weather data) and static modalities
    (soil properties, elevation, crop masks).
    
    Args:
        zarr_path: Path to the Zarr dataset file
        split: Dataset split, one of 'train', 'val', or 'test'
        modalities: List of modality names to load. If None, loads all available modalities
        transform: Optional transform to apply to samples
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.1)
        seed: Random seed for reproducible splits (default: 42)
    
    Example:
        >>> dataset = CropFMDataset(
        ...     zarr_path='/path/to/data.zarr',
        ...     split='train',
        ...     modalities=['sentinel2', 'agera5', 'soil']
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['sentinel2', 'agera5', 'soil'])
    """
    # Class-level configuration: define modality categories
    TEMPORAL_MODALITIES = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    STATIC_MODALITIES = ['soil', 'elevation', 'worldcereal_cropmask', 
                         'worldcereal_cropcalender']
    
    # Configuration for soil variables
    SOIL_VARIABLES = ['clay', 'nitrogen', 'phh2o', 'soc']

    def __init__(
        self, 
        zarr_path: str,
        split: str = 'train',
        modalities: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42
    ):
        """Initialize dataset"""

        # Validate split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")

        # Open Zarr dataset in read-only mode
        self.zarr_path = zarr_path
        self.split = split
        self.transform = transform

        # Set default modalities if not provided
        if modalities is None:
            self.modalities = self.TEMPORAL_MODALITIES + self.STATIC_MODALITIES
        else:
            self.modalities = modalities
            # Validate that all requested modalities are supported
            self._validate_modalities()

        # get total samples
        self.zarr_store = zarr.open(self.zarr_path, mode='r')
        total_samples = self.zarr_store.attrs['total_samples']
        # Create reproducible train/val/test splits
        self.indices = self._create_split_indices(
            total_samples, train_split, val_split, seed
        )
        

    def _validate_modalities(self):
            """Validate that requested modalities are supported"""
            all_modalities = self.TEMPORAL_MODALITIES + self.STATIC_MODALITIES
            for modality in self.modalities:
                if modality not in all_modalities:
                    raise ValueError(
                        f"Unknown modality '{modality}'. "
                        f"Supported modalities: {all_modalities}"
                    )
        
    def _create_split_indices(
        self, 
        total_samples: int, 
        train_split: float, 
        val_split: float,
        seed: int
    ) -> List[int]:
        """
        Create train/val/test split indices with optional shuffling.
        
        Args:
            total_samples: Total number of samples in dataset
            train_split: Fraction for training
            val_split: Fraction for validation
            seed: Random seed for reproducibility
        
        Returns:
            List of indices for the current split
        """
        # Calculate split boundaries
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)
        
        # Create shuffled indices for reproducibility
        np.random.seed(seed)
        all_indices = np.random.permutation(total_samples)
        
        # Select indices based on split
        if self.split == 'train':
            indices = all_indices[:train_end]
        elif self.split == 'val':
            indices = all_indices[train_end:val_end]
        elif self.split == 'test':
            indices = all_indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        return indices.tolist()

    def __len__(self) -> int:
        """Return dataset size - placeholder"""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary mapping modality names to tensors of shape (T, D)
            where T is number of timesteps and D is feature dimension
        """

         # CRITICAL: idx is relative to split, actual_idx is absolute position in Zarr
        actual_idx = self.indices[idx]
        
        sample = {}
        
        # Load each requested modality
        for modality in self.modalities:
            if modality in self.TEMPORAL_MODALITIES:
                sample[modality] = self._load_temporal_modality(modality, actual_idx)
            elif modality in self.STATIC_MODALITIES:
                sample[modality] = self._load_static_modality(modality, actual_idx)
        
        # Apply optional transform
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_temporal_modality(self, modality: str, actual_idx: int) -> torch.Tensor:
        """
        Load temporal (time-series) modality data.
        
        Args:
            modality: Name of temporal modality (agera5, sentinel1, sentinel2, fapar)
            actual_idx: Absolute index in Zarr array
        
        Returns:
            Tensor of shape (T, D) where T is number of valid timesteps, D is feature dimension
        """
        # Load data and validity mask
        data = self.zarr_root[f'temporal_modalities/{modality}/data'][actual_idx]
        valid_mask = self.zarr_root[f'temporal_modalities/{modality}/valid_mask'][actual_idx]
        
        # Filter to only valid timesteps (removes padding/missing data)
        valid_data = data[valid_mask]
        
        # Convert to PyTorch tensor
        return torch.from_numpy(valid_data).float()
    
    def _load_static_modality(self, modality: str, actual_idx: int) -> torch.Tensor:
        """
        Load static (non-temporal) modality data.
        
        Args:
            modality: Name of static modality (soil, elevation, etc.)
            actual_idx: Absolute index in Zarr array
        
        Returns:
            Tensor with shape depending on the specific modality
        """
        if modality == 'soil':
            return self._load_soil_data(actual_idx)
        elif modality == 'elevation':
            return self._load_elevation_data(actual_idx)
        elif modality == 'worldcereal_cropmask':
            return self._load_cropmask_data(actual_idx)
        elif modality == 'worldcereal_cropcalender':
            return self._load_cropcalendar_data(actual_idx)
        else:
            raise ValueError(f"Unknown static modality: {modality}")
    
    def _load_soil_data(self, actual_idx: int) -> torch.Tensor:
        """
        Load soil property data.
        
        Returns:
            Tensor of shape (4, 3) for 4 variables × 3 depth layers
            Variables: clay, nitrogen, phh2o, soc
            Depth layers: 0-5cm, 5-15cm, 15-30cm
        """
        soil_data = []
        for var in self.SOIL_VARIABLES:
            var_data = self.zarr_root[f'static_modalities/soil/{var}'][actual_idx]
            soil_data.append(var_data)
        
        # Stack into (n_variables, n_depth_layers) tensor
        return torch.from_numpy(np.stack(soil_data)).float()
    
    def _load_elevation_data(self, actual_idx: int) -> torch.Tensor:
        """
        Load elevation and slope data.
        
        Returns:
            Tensor of shape (2,) containing [elevation, slope]
        """
        elevation = self.zarr_root['static_modalities/elevation/elevation'][actual_idx]
        slope = self.zarr_root['static_modalities/elevation/slope'][actual_idx]
        
        return torch.tensor([elevation, slope]).float()
    
    def _load_cropmask_data(self, actual_idx: int) -> torch.Tensor:
        """
        Load WorldCereal crop mask data.
        
        Returns:
            Tensor of shape (2,) containing [aez_id, crop_mask]
        """
        aez_id = self.zarr_root['static_modalities/worldcereal_cropmask/aez_id'][actual_idx]
        crop_mask = self.zarr_root['static_modalities/worldcereal_cropmask/crop_mask'][actual_idx]
        
        return torch.tensor([aez_id, crop_mask]).long()
    
    def _load_cropcalendar_data(self, actual_idx: int) -> torch.Tensor:
        """
        Load WorldCereal crop calendar data.
        
        Returns:
            Tensor of shape (7,) containing:
            [aez_id, sos_maize, eos_maize, sos_winter, eos_winter, sos_spring, eos_spring]
        """
        aez_id = self.zarr_root['static_modalities/worldcereal_cropcalendar/aez_id'][actual_idx]
        calendar = self.zarr_root['static_modalities/worldcereal_cropcalendar/crop_calendar'][actual_idx]
        
        # Concatenate aez_id with 6 calendar values
        return torch.from_numpy(np.concatenate([[aez_id], calendar])).long()
    
    def get_modality_info(self, modality: str) -> Dict[str, any]:
        """
        Get metadata information about a specific modality.
        
        Args:
            modality: Name of the modality
        
        Returns:
            Dictionary containing variable names, dimensions, etc.
        """
        if modality in self.TEMPORAL_MODALITIES:
            var_names = self.zarr_root[f'temporal_modalities/{modality}/variable_names'][:]
            max_time = self.zarr_root[f'temporal_modalities/{modality}/data'].shape[1]
            n_vars = self.zarr_root[f'temporal_modalities/{modality}/data'].shape[2]
            
            return {
                'type': 'temporal',
                'variable_names': var_names.tolist(),
                'max_timesteps': max_time,
                'n_variables': n_vars
            }
        elif modality in self.STATIC_MODALITIES:
            return {
                'type': 'static',
                'modality': modality
            }
        else:
            raise ValueError(f"Unknown modality: {modality}")