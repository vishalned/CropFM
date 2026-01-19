import torch
from torch.utils.data import IterableDataset, DataLoader
import zarr
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

class CropFMIterableDataset(IterableDataset):
    """
    Vectorized version of CropFMIterableDataset.
    Processes entire chunks at once using batch operations for maximum performance.
    
    Key optimizations:
    - Batch tensor conversion (entire chunk at once)
    - Batch normalization (entire chunk at once)
    - Batch NaN handling (entire chunk at once)
    - Reusable week_indices (created once per chunk)
    - Pre-loaded static modalities for entire chunk
    """
    
    TEMPORAL_MODALITIES = ['agera5', 'sentinel1', 'sentinel2', 'fapar']
    STATIC_MODALITIES = ['soil', 'elevation', 'worldcereal_cropmask', 
                         'worldcereal_cropcalendar', 'encoded_coordinates']

    def __init__(
        self, 
        zarr_path: str,
        split: str,
        modalities: dict,
        chunk_size: int = 128, # Match your rechunking size
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
        shuffle: bool = True
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.split = split
        self.modalities = modalities
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed

        # Open root once just to get metadata
        root = zarr.open(self.zarr_path, mode='r')
        total_samples = root.attrs.get('total_samples', 0)
        
        # Calculate split boundaries
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)
        
        if split == 'train':
            self.start_idx, self.end_idx = 0, train_end
        elif split == 'val':
            self.start_idx, self.end_idx = train_end, val_end
        else:
            self.start_idx, self.end_idx = val_end, total_samples
        
        # Prepare Normalization Tensors
        norm_stats_path = Path(self.zarr_path).parent / 'normalization_stats2.json'
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        self.norm_tensors = self._prepare_norm_tensors(stats)
    
    def __len__(self):
        """Return the number of samples in this split."""
        return self.end_idx - self.start_idx

    def _prepare_norm_tensors(self, stats: dict) -> dict:
        """Vectorizes normalization stats for every requested modality."""
        tensors = {}
        for mod, config in self.modalities.items():
            means, stds = [], []
            for var in config.get('variables', []):
                val = None
                for cat in ['temporal_modalities', 'static_modalities', 'metadata']:
                    if cat in stats and mod in stats[cat] and var in stats[cat][mod]:
                        val = stats[cat][mod][var]
                        break
                means.append(val['mean'] if val else 0.0)
                stds.append(val['std'] if val else 1.0)
            
            tensors[mod] = {
                'mean': torch.tensor(means).float().view(1, -1),
                'std': torch.tensor(stds).float().view(1, -1)
            }
        return tensors

    def __iter__(self):
        """Worker iteration logic for multi-process DataLoader."""
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Generate absolute indices for this split
        all_indices = list(range(self.start_idx, self.end_idx))
        
        # 2. Divide into chunk-aligned blocks (to avoid redundant decompressions)
        chunk_indices = [all_indices[i:i + self.chunk_size] 
                        for i in range(0, len(all_indices), self.chunk_size)]

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(chunk_indices)

        # 3. Handle Multiprocessing sharding
        if worker_info is not None:
            per_worker = int(np.ceil(len(chunk_indices) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_chunks = chunk_indices[worker_id * per_worker : (worker_id + 1) * per_worker]
        else:
            iter_chunks = chunk_indices

        # 4. Open Zarr Arrays within worker process
        root = zarr.open(self.zarr_path, mode='r')
        arr_cache = {}
        var_idx_map = {}
        
        # Initialize Cache for every possible modality
        for mod, config in self.modalities.items():
            if mod in self.TEMPORAL_MODALITIES:
                arr_cache[f"{mod}_data"] = root[f"temporal_modalities/{mod}/data"]
                arr_cache[f"{mod}_mask"] = root[f"temporal_modalities/{mod}/valid_mask"]
                all_vars = root[f"temporal_modalities/{mod}/variable_names"][:].tolist()
                var_idx_map[mod] = [all_vars.index(v) for v in config['variables']]
            elif mod == 'soil':
                for v in ['clay', 'nitrogen', 'phh2o', 'soc']:
                    arr_cache[f"soil_{v}"] = root[f"static_modalities/soil/{v}"]
            elif mod == 'elevation':
                arr_cache[f"{mod}_elevation"] = root[f"static_modalities/elevation/elevation"]
                arr_cache[f"{mod}_slope"] = root[f"static_modalities/elevation/slope"]
            elif mod == 'worldcereal_cropmask':
                arr_cache[f"{mod}_aez"] = root[f"static_modalities/worldcereal_cropmask/aez_id"]
                arr_cache[f"{mod}_mask"] = root[f"static_modalities/worldcereal_cropmask/crop_mask"]
            elif mod == 'worldcereal_cropcalendar':
                arr_cache[f"{mod}_aez"] = root[f"static_modalities/worldcereal_cropcalendar/aez_id"]
                arr_cache[f"{mod}_cal"] = root[f"static_modalities/worldcereal_cropcalendar/crop_calendar"]
            elif mod == 'encoded_coordinates':
                arr_cache[mod] = root['metadata/encoded_coordinates']

        # 5. Stream through chunks with vectorized processing
        for chunk in iter_chunks:
            # Linear slice fetch (The FAST part)
            start, end = chunk[0], chunk[-1] + 1
            chunk_len = len(chunk)
            
            # Pre-load the entire block into memory
            loaded_chunk = {}
            for mod in self.modalities:
                if mod in self.TEMPORAL_MODALITIES:
                    loaded_chunk[f"{mod}_data"] = arr_cache[f"{mod}_data"][start:end]
                    loaded_chunk[f"{mod}_mask"] = arr_cache[f"{mod}_mask"][start:end]
                elif mod == 'encoded_coordinates':
                    loaded_chunk[mod] = arr_cache[mod][start:end]
            
            # ===== VECTORIZED PROCESSING: Process entire chunk at once =====
            processed_chunk = {}
            
            # Process temporal modalities in batch
            for mod in self.modalities:
                if mod in self.TEMPORAL_MODALITIES:
                    # Load entire chunk: shape (chunk_len, T, D)
                    chunk_data = loaded_chunk[f"{mod}_data"]  # numpy array
                    chunk_mask = loaded_chunk[f"{mod}_mask"]  # numpy array
                    
                    # Vectorized: Convert entire chunk to tensor at once
                    # Select only requested variables: (chunk_len, T, num_vars)
                    chunk_data_tensor = torch.from_numpy(
                        chunk_data[:, :, var_idx_map[mod]]
                    ).float()
                    
                    # Vectorized: Normalize entire chunk at once
                    # norm_tensors[mod]['mean'] shape: (1, num_vars)
                    # Broadcasting: (chunk_len, T, num_vars) - (1, num_vars) -> (chunk_len, T, num_vars)
                    chunk_data_tensor = (
                        chunk_data_tensor - self.norm_tensors[mod]['mean']
                    ) / self.norm_tensors[mod]['std']
                    
                    # Vectorized: Handle NaN for entire chunk (Sentinel2)
                    if mod == 'sentinel2':
                        chunk_data_tensor = torch.nan_to_num(chunk_data_tensor, nan=0.0)
                    
                    # Convert masks to tensor: (chunk_len, T)
                    chunk_mask_tensor = torch.from_numpy(chunk_mask).bool()
                    
                    # Create week_indices once: shape (T,)
                    # This is reused for all samples in the chunk
                    week_indices = torch.arange(chunk_data_tensor.shape[1]) % 52
                    
                    # Store processed chunk data
                    processed_chunk[mod] = {
                        'data': chunk_data_tensor,  # (chunk_len, T, num_vars)
                        'mask': chunk_mask_tensor,  # (chunk_len, T)
                        'week_indices': week_indices  # (T,)
                    }
            
            # Pre-load static modalities for entire chunk
            static_chunk_data = {}
            for mod in self.modalities:
                if mod == 'soil':
                    # Pre-load all soil variables for entire chunk
                    soil_chunk = {}
                    requested_vars = self.modalities[mod]['variables']
                    for var_name in requested_vars:
                        base_var, d_str = var_name.split('_')
                        d_idx = ['0-5cm', '5-15cm', '15-30cm'].index(d_str)
                        # Load entire chunk: (chunk_len,)
                        soil_chunk[var_name] = arr_cache[f"soil_{base_var}"][start:end, d_idx]
                    
                    # Vectorized: Stack and convert to tensor: (chunk_len, num_vars)
                    soil_data_list = [soil_chunk[var_name] for var_name in requested_vars]
                    soil_data_array = np.stack(soil_data_list, axis=1)  # (chunk_len, num_vars)
                    soil_data_tensor = torch.from_numpy(soil_data_array).float()
                    
                    # Vectorized: Normalize entire chunk at once
                    soil_data_tensor = (
                        soil_data_tensor - self.norm_tensors[mod]['mean']
                    ) / self.norm_tensors[mod]['std']
                    soil_data_tensor = torch.nan_to_num(soil_data_tensor, nan=0.0)
                    
                    static_chunk_data[mod] = soil_data_tensor  # (chunk_len, num_vars)
                
                elif mod == 'elevation':
                    # Pre-load entire chunk: (chunk_len,)
                    elev_chunk = arr_cache[f"{mod}_elevation"][start:end]
                    slope_chunk = arr_cache[f"{mod}_slope"][start:end]
                    
                    # Vectorized: Stack and convert: (chunk_len, 2)
                    elev_data_array = np.stack([elev_chunk, slope_chunk], axis=1)
                    elev_data_tensor = torch.from_numpy(elev_data_array).float()
                    
                    # Vectorized: Normalize entire chunk at once
                    elev_data_tensor = (
                        elev_data_tensor - self.norm_tensors[mod]['mean']
                    ) / self.norm_tensors[mod]['std']
                    
                    static_chunk_data[mod] = elev_data_tensor  # (chunk_len, 2)
                
                elif mod == 'worldcereal_cropmask':
                    # Pre-load entire chunk: (chunk_len,)
                    aez_chunk = arr_cache[f"{mod}_aez"][start:end]
                    mask_chunk = arr_cache[f"{mod}_mask"][start:end]
                    
                    # Vectorized: Stack and convert: (chunk_len, 2)
                    mask_data_array = np.stack([aez_chunk, mask_chunk], axis=1)
                    mask_data_tensor = torch.from_numpy(mask_data_array).long()
                    
                    static_chunk_data[mod] = mask_data_tensor  # (chunk_len, 2)
                
                elif mod == 'worldcereal_cropcalendar':
                    # Pre-load entire chunk
                    aez_chunk = arr_cache[f"{mod}_aez"][start:end]  # (chunk_len,)
                    cal_chunk = arr_cache[f"{mod}_cal"][start:end]  # (chunk_len, 6)
                    
                    # Vectorized: Concatenate aez with calendar for each sample
                    # Shape: (chunk_len, 7) where first column is aez, rest is calendar
                    calendar_data_list = []
                    for i in range(chunk_len):
                        calendar_data_list.append(np.concatenate([[aez_chunk[i]], cal_chunk[i]]))
                    calendar_data_array = np.stack(calendar_data_list, axis=0)
                    calendar_data_tensor = torch.from_numpy(calendar_data_array).long()
                    
                    static_chunk_data[mod] = calendar_data_tensor  # (chunk_len, 7)
                
                elif mod == 'encoded_coordinates':
                    # Already loaded in loaded_chunk
                    # Vectorized: Convert entire chunk: (chunk_len, D)
                    coord_data_tensor = torch.from_numpy(loaded_chunk[mod]).float()
                    
                    # Vectorized: Normalize entire chunk at once
                    coord_data_tensor = (
                        coord_data_tensor - self.norm_tensors[mod]['mean']
                    ) / self.norm_tensors[mod]['std']
                    
                    static_chunk_data[mod] = coord_data_tensor  # (chunk_len, D)
            
            # ===== YIELD INDIVIDUAL SAMPLES FROM PREPROCESSED CHUNK =====
            for i in range(chunk_len):
                actual_idx = chunk[i]
                sample = {}
                
                # Slice from preprocessed temporal modalities
                for mod in self.modalities:
                    if mod in self.TEMPORAL_MODALITIES:
                        sample[mod] = {
                            'data': processed_chunk[mod]['data'][i],  # (T, num_vars)
                            'valid_mask': processed_chunk[mod]['mask'][i],  # (T,)
                            'week_indices': processed_chunk[mod]['week_indices']  # (T,) - reused!
                        }
                    
                    elif mod in static_chunk_data:
                        # Slice from preprocessed static modalities
                        if mod == 'soil':
                            sample[mod] = {
                                'data': static_chunk_data[mod][i].unsqueeze(0)  # (1, num_vars)
                            }
                        elif mod == 'elevation':
                            sample[mod] = {
                                'data': static_chunk_data[mod][i].unsqueeze(0)  # (1, 2)
                            }
                        elif mod == 'worldcereal_cropmask':
                            sample[mod] = {
                                'data': static_chunk_data[mod][i]  # (2,)
                            }
                        elif mod == 'worldcereal_cropcalendar':
                            sample[mod] = {
                                'data': static_chunk_data[mod][i]  # (7,)
                            }
                        elif mod == 'encoded_coordinates':
                            sample[mod] = {
                                'data': static_chunk_data[mod][i].unsqueeze(0)  # (1, D)
                            }
                
                yield sample

def cropfm_collate_fn(batch):
    """Stacking logic for multi-modal batches."""
    elem = batch[0]
    collated = {}
    for mod in elem.keys():
        collated[mod] = {}
        for key in elem[mod].keys():
            items = [d[mod][key] for d in batch]
            if isinstance(items[0], torch.Tensor):
                collated[mod][key] = torch.stack(items, dim=0)
            else:
                collated[mod][key] = items
    return collated

if __name__ == "__main__":
    # Test initialization
    MODS = {
        'sentinel2': {'variables': ['B02', 'B03']},
        'soil': {'variables': ['clay_0-5cm', 'soc_0-5cm']},
        'worldcereal_cropmask': {'variables': ['crop_mask']}
    }
    
    dataset = CropFMIterableDataset(
        zarr_path='your_data.zarr',
        split='train',
        modalities=MODS,
        shuffle=True
    )
    
    loader = DataLoader(dataset, batch_size=128, collate_fn=cropfm_collate_fn, num_workers=4)
    for batch in loader:
        print("Batch loaded successfully!")
        break
