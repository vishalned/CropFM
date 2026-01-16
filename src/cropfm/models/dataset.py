"""Dataset of multi-modal agricultural data for CropMAE"""

import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
import json
from pathlib import Path
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
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.1)
        seed: Random seed for reproducible splits (default: 42)
    
    Example:
        >>> dataset = CropFMDataset(
        ...     zarr_path='/path/to/data.zarr',
        ...     split='train',
        ...     modalities={
        ...         'sentinel2': {'variables': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']},
        ...         'agera5': {'variables': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']},
        ...         'soil': {'variables': ['clay', 'nitrogen', 'phh2o', 'soc']}
        ...     }
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['sentinel2', 'agera5', 'soil'])
    """
    # Class-level configuration: define modality categories
    TEMPORAL_MODALITIES = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    STATIC_MODALITIES = ['soil', 'elevation', 'worldcereal_cropmask', 
                         'worldcereal_cropcalender', 'encoded_coordinates']
    
    # Configuration for soil variables
    SOIL_VARIABLES = ['clay', 'nitrogen', 'phh2o', 'soc']

    def __init__(
        self, 
        zarr_path: str,
        split: str,
        modalities: dict,
        train_split: float,
        val_split: float,
        seed: int,
    ):
        """Initialize dataset"""

        # Validate split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")

        # Open Zarr dataset in read-only mode
        self.zarr_path = zarr_path
        self.split = split

        self.modalities = modalities
        # Validate that all requested modalities are supported
        self._validate_modalities()

        # get total samples
        self.zarr_root = zarr.open(self.zarr_path, mode='r')
        total_samples = self.zarr_root.attrs['total_samples']
        # Create reproducible train/val/test splits
        self.indices = self._create_split_indices(
            total_samples, train_split, val_split, seed
        )

        # for testing, use all samples
        # self.indices = list(range(total_samples))
        
        # Cache normalization stats to avoid repeated file reads
        norm_stats_path = Path(self.zarr_path).parent / 'normalization_stats2.json'
        with open(norm_stats_path, 'r') as f:
            self.norm_stats = json.load(f)
        # Cache variable names for temporal modalities to avoid repeated zarr reads
        self.temporal_variable_names = {}
        for modality in self.TEMPORAL_MODALITIES:
            if modality in self.modalities:
                self.temporal_variable_names[modality] = self.zarr_root[f'temporal_modalities/{modality}/variable_names'][:]
                if not isinstance(self.temporal_variable_names[modality], list):
                    self.temporal_variable_names[modality] = self.temporal_variable_names[modality].tolist()
        

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

    def _normalize_data(self, data: torch.Tensor, modality: str, variables: list[str]) -> torch.Tensor:
        """
        Normalize the data.
        Args:
            data: The data to normalize
            modality: The modality of the data
            variables: The variables of the data
        Returns:
            The normalized data
        """
        # Use cached normalization stats instead of reading from disk
        norm_stats = self.norm_stats

        # containers for per-variable normalization statistics
        mean: list[float] = []
        std: list[float] = []

        if modality in self.TEMPORAL_MODALITIES:
            modality_stats = norm_stats['temporal_modalities'][modality]
            
            # get the stats for the variables in the same order as the variables list
            for var in variables:
                mean.append(modality_stats[var]['mean'])
                std.append(modality_stats[var]['std'])
            
            data = (data - torch.tensor(mean)) / torch.tensor(std)

        elif modality in self.STATIC_MODALITIES:
            if modality == 'encoded_coordinates':
                modality_stats = norm_stats['metadata']['encoded_coordinates']
            elif modality == 'week_encoding':
                modality_stats = norm_stats['temporal_modalities']['agera5']['week_encoding']
            else:
                modality_stats = norm_stats['static_modalities'][modality]

            for var in variables:
                mean.append(modality_stats[var]['mean'])
                std.append(modality_stats[var]['std'])

            data = (data - torch.tensor(mean)) / torch.tensor(std)

            # For static modalities, ensure no NaNs remain after normalization (H5)
            try:
                data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

        return data
            
    def __len__(self) -> int:
        """Return dataset size - placeholder"""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        For temporal modalities, the valid_mask is also returned.
        We do this to allow the transformer to handle the nan data by replace them 
        with learnable parameters or mask tokens or mask them out.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary mapping modality names to tensors of shape (T, D) and valid_mask (optional)
            where T is number of timesteps and D is feature dimension
        """

         # CRITICAL: idx is relative to split, actual_idx is absolute position in Zarr
        actual_idx = self.indices[idx]
        
        sample = {}
        
        # Cache week indices shape from first temporal modality to avoid reloading
        week_indices_shape = None
        
        # Load each requested modality
        for modality in self.modalities:
            variables = self.modalities[modality]['variables']
            # initialize nested dict for this modality
            sample[modality] = {}
            if modality in self.TEMPORAL_MODALITIES:
                sample_data, mask = self._load_temporal_modality(
                    modality, 
                    variables, 
                    actual_idx
                )
                
                sample[modality]['data'] = self._normalize_data(sample_data, modality, variables)
                sample[modality]['valid_mask'] = mask
                # Load week indices for temporal positional encoding
                # Use cached shape from first temporal modality to avoid reloading data
                if week_indices_shape is None:
                    week_indices_shape = sample_data.shape[0]
                sample[modality]['week_indices'] = self._load_week_indices_from_shape(week_indices_shape)

                if modality == 'sentinel2':
                    # for all valid timesteps, we replace normalized nan data with 0
                    tmp = sample[modality]['data']
                    # convert numpy mask to torch tensor on same device/dtype for safe indexing
                    torch_mask = torch.from_numpy(mask).to(dtype=torch.bool, device=tmp.device)
                    # only choose the valid timesteps
                    tmp_valid = tmp[torch_mask]
                    # replace nan with 0 (torch op)
                    tmp_valid = torch.nan_to_num(tmp_valid, nan=0.0)
                    # write back the updated valid timesteps
                    tmp[torch_mask] = tmp_valid
                    sample[modality]['data'] = tmp

            elif modality in self.STATIC_MODALITIES:
                sample_data = self._load_static_modality(modality, variables, actual_idx)
                sample[modality]['data'] = self._normalize_data(sample_data, modality, variables)

        return sample
    
    def _load_temporal_modality(self, modality: str, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load temporal (time-series) modality data.
        
        Args:
            modality: Name of temporal modality (agera5, sentinel1, sentinel2, fapar)
            variables: List of variables to load
            actual_idx: Absolute index in Zarr array
        
        Returns:
            Tensor of shape (T, D) where T is number of valid timesteps, D is feature dimension
        """
        # Load data and validity mask
        data = self.zarr_root[f'temporal_modalities/{modality}/data'][actual_idx]

        # Use cached variable names instead of loading from zarr every time
        if modality in self.temporal_variable_names:
            actual_variables = self.temporal_variable_names[modality]
        else:
            # Fallback: load if not cached (shouldn't happen normally)
            actual_variables = self.zarr_root[f'temporal_modalities/{modality}/variable_names'][:]
            if not isinstance(actual_variables, list):
                actual_variables = actual_variables.tolist()

        data = data[:, [actual_variables.index(var) for var in variables]]

        
        valid_mask = self.zarr_root[f'temporal_modalities/{modality}/valid_mask'][actual_idx]
        # we not comment the below lines since we keep the nan data to allow the transformer to handle it
        # data = data[valid_mask]
        
        # Convert to PyTorch tensor
        return torch.from_numpy(data).float(), valid_mask
    
    def _load_static_modality(self, modality: str, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load static (non-temporal) modality data.
        
        Args:
            modality: Name of static modality (soil, elevation, etc.)
            variables: List of variables to load
            actual_idx: Absolute index in Zarr array
        
        Returns:
            Tensor with shape depending on the specific modality
        """
        if modality == 'soil':
            return self._load_soil_data(variables, actual_idx)
        elif modality == 'elevation':
            return self._load_elevation_data(variables, actual_idx)
        elif modality == 'worldcereal_cropmask':
            return self._load_cropmask_data(variables, actual_idx)
        elif modality == 'worldcereal_cropcalender':
            return self._load_cropcalendar_data(variables, actual_idx)
        elif modality == 'week_encoding':
            return self._load_week_encoding_data(variables, actual_idx)
        elif modality == 'encoded_coordinates':
            return self._load_encoded_coordinates_data(variables, actual_idx)
        else:
            raise ValueError(f"Unknown static modality: {modality}")
    
    def _load_soil_data(self, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load soil property data.
        
        Returns:
            Tensor of shape (1, 4 * 3) for 4 variables × 3 depth layers
            Variables: clay, nitrogen, phh2o, soc
            Depth layers: 0-5cm, 5-15cm, 15-30cm
        """
        actual_depth_layers = ['0-5cm', '5-15cm', '15-30cm']
        soil_data = []
        for var in variables:
            base_var = var.split('_')[0]
            depth = actual_depth_layers.index(var.split('_')[1])
            var_data = self.zarr_root[f'static_modalities/soil/{base_var}'][actual_idx, depth]
            soil_data.append(var_data)
        
        # Stack into (n_variables, n_depth_layers) tensor
        return torch.from_numpy(np.stack(soil_data).reshape(1, -1)).float()
    
    def _load_elevation_data(self, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load elevation and slope data.
        
        Returns:
            Tensor of shape (2,) containing [elevation, slope]
        """
        for var in variables:
            if var == 'elevation':
                elevation = float(self.zarr_root['static_modalities/elevation/elevation'][actual_idx])
            elif var == 'slope':
                slope = float(self.zarr_root['static_modalities/elevation/slope'][actual_idx])
        
        return torch.tensor([elevation, slope], dtype=torch.float32).reshape(1, -1)
    
    def _load_cropmask_data(self, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load WorldCereal crop mask data.
        
        Returns:
            Tensor of shape (2,) containing [aez_id, crop_mask]
        """
        for var in variables:
            if var == 'aez_id':
                aez_id = self.zarr_root['static_modalities/worldcereal_cropmask/aez_id'][actual_idx]
            elif var == 'crop_mask':
                crop_mask = self.zarr_root['static_modalities/worldcereal_cropmask/crop_mask'][actual_idx]
        
        return torch.tensor([aez_id, crop_mask]).long()
    
    def _load_cropcalendar_data(self, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load WorldCereal crop calendar data.
        
        Returns:
            Tensor of shape (7,) containing:
            [aez_id, sos_maize, eos_maize, sos_winter, eos_winter, sos_spring, eos_spring]
        """
        for var in variables:
            if var == 'aez_id':
                aez_id = self.zarr_root['static_modalities/worldcereal_cropcalendar/aez_id'][actual_idx]
            elif var == 'crop_calendar':
                calendar = self.zarr_root['static_modalities/worldcereal_cropcalendar/crop_calendar'][actual_idx]
        
        # Concatenate aez_id with 6 calendar values
        return torch.from_numpy(np.concatenate([[aez_id], calendar])).long()
    
    def _load_week_indices_from_shape(self, T: int) -> torch.Tensor:
        """
        Generate week indices for temporal positional encoding from shape.
        Uses timestep indices directly as week indices (0-51), since all temporal modalities
        share the same timesteps and week encoding.
        
        Args:
            T: Number of timesteps (from already loaded temporal modality)
            
        Returns:
            Tensor of shape (T,) containing week indices (0-51) for each timestep
        """
        # Use timestep indices directly as week indices (modulo 52)
        week_indices = np.arange(T) % 52
        
        return torch.from_numpy(week_indices).long()
    
    def _load_week_indices(self, actual_idx: int) -> torch.Tensor:
        """
        Load week indices for temporal positional encoding (legacy method).
        Uses timestep indices directly as week indices (0-51), since all temporal modalities
        share the same timesteps and week encoding.
        
        Args:
            actual_idx: Absolute index in Zarr array
            
        Returns:
            Tensor of shape (T,) containing week indices (0-51) for each timestep
        """
        # Load any temporal modality to get the number of timesteps
        # All temporal modalities have the same number of timesteps
        data = self.zarr_root['temporal_modalities/agera5/data'][actual_idx]
        T = data.shape[0]
        
        # Use timestep indices directly as week indices (modulo 52)
        week_indices = np.arange(T) % 52
        
        return torch.from_numpy(week_indices).long()
    
    def _load_week_encoding_data(self, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load week encoding data.
        
        Returns:
            Tensor of shape (2,) containing [sin_week, cos_week]
        """

        # small bug when creating the week_encoding, we did it for every modality. but its all similar
        # so we just load the sentinel2 week_encoding
        actual_variables = ['week_sin', 'week_cos']

        week_encoding = []
        for var in variables:
            week_encoding.append(self.zarr_root['temporal_modalities/sentinel2/week_encoding'][actual_idx][:, actual_variables.index(var)])

        arr = np.stack(week_encoding, axis=1)

        return torch.tensor(arr).float()
    
    def _load_encoded_coordinates_data(self, variables: list[str], actual_idx: int) -> torch.Tensor:
        """
        Load coordinates data.
        
        Returns:
            Tensor of shape (4,) containing [lon_sin, lon_cos, lat_sin, lat_cos]
        """
        actual_variables = ['lon_sin', 'lon_cos', 'lat_sin', 'lat_cos']
        encoded_coordinates = []
        for var in variables:
            if var == 'lon_sin':
                encoded_coordinates.append(self.zarr_root['metadata/encoded_coordinates'][actual_idx][actual_variables.index(var)])
            elif var == 'lon_cos':
                encoded_coordinates.append(self.zarr_root['metadata/encoded_coordinates'][actual_idx][actual_variables.index(var)])
            elif var == 'lat_sin':
                encoded_coordinates.append(self.zarr_root['metadata/encoded_coordinates'][actual_idx][actual_variables.index(var)])
            elif var == 'lat_cos':
                encoded_coordinates.append(self.zarr_root['metadata/encoded_coordinates'][actual_idx][actual_variables.index(var)])
        
        return torch.tensor(encoded_coordinates).float().reshape(1, -1)
    
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



if __name__ == "__main__":
    dataset = CropFMDataset(
        zarr_path='/lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged_cleaned.zarr',
        split='train',
        modalities={'soil': {'variables': ['clay_0-5cm', 'clay_5-15cm', 'clay_15-30cm', 'nitrogen_0-5cm', 'nitrogen_5-15cm', 'nitrogen_15-30cm', 'phh2o_0-5cm', 'phh2o_5-15cm', 'phh2o_15-30cm', 'soc_0-5cm', 'soc_5-15cm', 'soc_15-30cm']}},
        train_split=0.8,
        val_split=0.1,
        seed=42
    )
    print(dataset[0])
    print(dataset[0]['soil'].shape)