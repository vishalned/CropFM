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
    Full-modality high-performance streaming dataset for Zarr 3.0.
    Optimized for sequential reads within chunks to maximize Blosc throughput.

    The old version of the dataset in an iterable dataset.
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
        
        # Store total length for progress bar
        self._len = self.end_idx - self.start_idx

        # Prepare Normalization Tensors (Pre-calculated in __init__)
        norm_stats_path = Path(self.zarr_path).parent / 'normalization_stats2.json'
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        self.norm_tensors = self._prepare_norm_tensors(stats)
    
    def __len__(self):
        """Return the number of samples in this split."""
        return self._len

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

        # 5. Stream through chunks
        for chunk in iter_chunks:
            # Linear slice fetch (The FAST part)
            start, end = chunk[0], chunk[-1] + 1
            
            # Pre-load the entire block into memory
            loaded_chunk = {}
            for mod in self.modalities:
                if mod in self.TEMPORAL_MODALITIES:
                    loaded_chunk[f"{mod}_data"] = arr_cache[f"{mod}_data"][start:end]
                    loaded_chunk[f"{mod}_mask"] = arr_cache[f"{mod}_mask"][start:end]
                elif mod == 'encoded_coordinates':
                    loaded_chunk[mod] = arr_cache[mod][start:end]
                # Other statics are fast enough to index directly, 
                # but we could pre-load them too for extreme speed.

            # Yield individual samples from memory
            for i in range(len(chunk)):
                actual_idx = chunk[i]
                rel_idx = i
                sample = {}

                for mod in self.modalities:
                    if mod in self.TEMPORAL_MODALITIES:
                        raw_data = loaded_chunk[f"{mod}_data"][rel_idx]
                        raw_mask = loaded_chunk[f"{mod}_mask"][rel_idx]
                        data = torch.from_numpy(raw_data[:, var_idx_map[mod]]).float()
                        data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                        if mod == 'sentinel2': data = torch.nan_to_num(data, nan=0.0)
                        
                        sample[mod] = {
                            'data': data,
                            'valid_mask': torch.from_numpy(raw_mask).bool(),
                            'week_indices': torch.arange(data.shape[0]) % 52
                        }
                    
                    elif mod == 'soil':
                        soil_data = []
                        for var_name in self.modalities[mod]['variables']:
                            base_var, d_str = var_name.split('_')
                            d_idx = ['0-5cm', '5-15cm', '15-30cm'].index(d_str)
                            soil_data.append(float(arr_cache[f"soil_{base_var}"][actual_idx, d_idx]))
                        data = torch.tensor(soil_data).float().unsqueeze(0)
                        data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                        sample[mod] = {'data': torch.nan_to_num(data, nan=0.0)}

                    elif mod == 'elevation':
                        e = float(arr_cache[f"{mod}_elevation"][actual_idx])
                        s = float(arr_cache[f"{mod}_slope"][actual_idx])
                        data = torch.tensor([e, s]).float().unsqueeze(0)
                        data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                        sample[mod] = {'data': data}

                    elif mod == 'worldcereal_cropmask':
                        aez = int(arr_cache[f"{mod}_aez"][actual_idx])
                        mask = int(arr_cache[f"{mod}_mask"][actual_idx])
                        sample[mod] = {'data': torch.tensor([aez, mask]).long()}

                    elif mod == 'worldcereal_cropcalendar':
                        aez = int(arr_cache[f"{mod}_aez"][actual_idx])
                        cal = arr_cache[f"{mod}_cal"][actual_idx]
                        data = np.concatenate([[aez], cal])
                        sample[mod] = {'data': torch.from_numpy(data).long()}

                    elif mod == 'encoded_coordinates':
                        data = torch.from_numpy(loaded_chunk[mod][rel_idx]).float().unsqueeze(0)
                        data = (data - self.norm_tensors[mod]['mean']) / self.norm_tensors[mod]['std']
                        sample[mod] = {'data': data}

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