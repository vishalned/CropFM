import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class CropFMDataset(Dataset):
    """
    Optimized PyTorch Dataset for Zarr 3.0.
    
    Features:
    - Pre-cached array references (Fast IO)
    - Vectorized normalization (Fast Math)
    - Zero-cost variable indexing (Pre-calculated in __init__)
    - Native Zarr 3.0 slicing

    The old version of the dataset
    """
    
    TEMPORAL_MODALITIES = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    STATIC_MODALITIES = ['soil', 'elevation', 'worldcereal_cropmask', 
                         'worldcereal_cropcalendar', 'encoded_coordinates']

    def __init__(
        self, 
        zarr_path: str,
        split: str,
        modalities: dict,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")

        self.zarr_path = zarr_path
        self.split = split
        self.modalities = modalities
        
        # 1. Open Zarr Root and cache total samples
        self.root = zarr.open(self.zarr_path, mode='r')
        total_samples = self.root.attrs.get('total_samples', 0)
        
        # 2. Setup Split Indices
        self.indices = self._create_split_indices(total_samples, train_split, val_split, seed)

        # 3. Load and Pre-vectorize Normalization Stats
        # We prepare these as tensors now so we don't do it per-sample
        norm_stats_path = Path(self.zarr_path).parent / 'normalization_stats2.json'
        with open(norm_stats_path, 'r') as f:
            raw_stats = json.load(f)
        self.norm_tensors = self._prepare_norm_tensors(raw_stats)

        # 4. Array Cache and Variable Mapping
        # We store direct references to the Zarr arrays to skip group lookups
        self.arr_cache = {}
        self.var_idx_map = {}
        
        self._initialize_resources()

    def _initialize_resources(self):
        """Pre-binds Zarr arrays and calculates variable indices once."""
        for mod, config in self.modalities.items():
            requested_vars = config.get('variables', [])
            
            if mod in self.TEMPORAL_MODALITIES:
                base = f'temporal_modalities/{mod}'
                self.arr_cache[f"{mod}_data"] = self.root[f"{base}/data"]
                self.arr_cache[f"{mod}_mask"] = self.root[f"{base}/valid_mask"]
                
                # Pre-calculate column indices for requested variables
                all_vars = self.root[f"{base}/variable_names"][:].tolist()
                self.var_idx_map[mod] = [all_vars.index(v) for v in requested_vars]
                
            elif mod == 'soil':
                # Soil is a special case: multiple arrays for one modality
                for v in ['clay', 'nitrogen', 'phh2o', 'soc']:
                    self.arr_cache[f"soil_{v}"] = self.root[f"static_modalities/soil/{v}"]
                # Soil depths are usually fixed: 0-5cm, 5-15cm, 15-30cm
                # Mapping logic handled in _load_soil
            
            elif mod == 'encoded_coordinates':
                self.arr_cache[mod] = self.root['metadata/encoded_coordinates']
            
            elif mod in self.STATIC_MODALITIES:
                # Direct group access for elevation, masks, calendars
                self.arr_cache[mod] = self.root[f"static_modalities/{mod}"]

    def _prepare_norm_tensors(self, stats: dict) -> dict:
        """Converts stats to mean/std tensors for the specific requested variables."""
        tensors = {}
        for mod, config in self.modalities.items():
            means, stds = [], []
            for var in config.get('variables', []):
                # Search across possible locations in the JSON
                val = None
                for cat in ['temporal_modalities', 'static_modalities', 'metadata']:
                    if cat in stats and mod in stats[cat] and var in stats[cat][mod]:
                        val = stats[cat][mod][var]
                        break
                
                if val:
                    means.append(val['mean'])
                    stds.append(val['std'])
            else:
                    means.append(0.0); stds.append(1.0)
            
            tensors[mod] = {
                'mean': torch.tensor(means).float().view(1, -1),
                'std': torch.tensor(stds).float().view(1, -1)
            }
        return tensors

    def _create_split_indices(self, total_samples, train_split, val_split, seed):
        np.random.seed(seed)
        all_indices = np.random.permutation(total_samples)
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)
        
        if self.split == 'train': return all_indices[:train_end]
        if self.split == 'val': return all_indices[train_end:val_end]
        return all_indices[val_end:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = int(self.indices[idx])
        sample = {}
        
        for mod in self.modalities:
            if mod in self.TEMPORAL_MODALITIES:
                # 1. IO: Pull raw data and mask (hitting the 128-chunk cache)
                data_raw = self.arr_cache[f"{mod}_data"][actual_idx]
                mask_raw = self.arr_cache[f"{mod}_mask"][actual_idx]
                
                # 2. Slice variables and convert to Torch
                data = torch.from_numpy(data_raw[:, self.var_idx_map[mod]]).float()
                
                # 3. Vectorized Normalization
                data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                
                # 4. Fill NaNs if needed (S2 specific)
                if mod == 'sentinel2':
                    data = torch.nan_to_num(data, nan=0.0)

                sample[mod] = {
                    'data': data,
                    'valid_mask': torch.from_numpy(mask_raw).bool(),
                    'week_indices': torch.arange(data.shape[0]) % 52
                }

            elif mod == 'soil':
                # Soil IO: Stack the requested variables/depths
                soil_data = []
                requested_vars = self.modalities[mod]['variables']
                # Optimization: We assume vars like 'clay_0-5cm'
                # We pull them into a flat vector
                for var_name in requested_vars:
                    base_var, depth_str = var_name.split('_')
                    depth_idx = ['0-5cm', '5-15cm', '15-30cm'].index(depth_str)
                    val = self.arr_cache[f"soil_{base_var}"][actual_idx, depth_idx]
                    soil_data.append(float(val))
                
                data = torch.from_numpy(np.array(soil_data, dtype=np.float32)).float().unsqueeze(0)
                data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                sample[mod] = {'data': torch.nan_to_num(data, nan=0.0)}

            elif mod == 'elevation':
                # elevation group contains 'elevation' and 'slope' arrays
                e = float(self.arr_cache[mod]['elevation'][actual_idx])
                s = float(self.arr_cache[mod]['slope'][actual_idx])
                data = torch.from_numpy(np.array([e, s], dtype=np.float32)).float().unsqueeze(0)
                data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                sample[mod] = {'data': data}

            elif mod == 'encoded_coordinates':
                data = torch.from_numpy(self.arr_cache[mod][actual_idx]).float().unsqueeze(0)
                data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                sample[mod] = {'data': data}
                
            elif mod == 'worldcereal_cropmask':
                aez = int(self.arr_cache[mod]['aez_id'][actual_idx])
                mask = int(self.arr_cache[mod]['crop_mask'][actual_idx])
                sample[mod] = {'data': torch.tensor([aez, mask]).long()}

            elif mod == 'worldcereal_cropcalendar':
                aez = int(self.arr_cache[mod]['aez_id'][actual_idx])
                cal = self.arr_cache[mod]['crop_calendar'][actual_idx] # shape (6,)
                data = np.concatenate([[aez], cal], dtype=np.int32)
                sample[mod] = {'data': torch.from_numpy(data).long()}

        return sample

if __name__ == "__main__":
    # Example usage
    MODS = {
        'sentinel2': {'variables': ['B02', 'B03', 'B04', 'B08']},
        'soil': {'variables': ['clay_0-5cm', 'soc_0-5cm']},
        'elevation': {'variables': ['elevation', 'slope']}
    }
    
    ds = CropFMDataset(
        zarr_path='path/to/your/rechunked.zarr',
        split='train',
        modalities=MODS
    )
    
    print(f"Dataset loaded with {len(ds)} samples.")
    sample = ds[0]
    print("S2 Shape:", sample['sentinel2']['data'].shape)
    print("Soil Data:", sample['soil']['data'])