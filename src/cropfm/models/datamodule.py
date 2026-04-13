"""DataModule for CropFM multi-modal agricultural data using IterableDataset"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os

# Import the new iterable dataset and the collate function
from cropfm.models.dataset import CropFMIterableDataset, cropfm_collate_fn


class CropFMDataModule(LightningDataModule):
    def __init__(
        self,
        zarr_path: str,
        modalities: dict,
        batch_size: int,
        num_workers: int,
        train_split: float,
        val_split: float,
        test_split: float,
        seed: int,
        pin_memory: bool,
        chunk_size: int = 128, # Added to match your Zarr rechunking
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.zarr_path = zarr_path
        self.modalities = modalities['modality_list']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.pin_memory = pin_memory
        # IterableDataset workers build full batches locally. Use a chunk size that is
        # larger than the batch and not an exact multiple of it to reduce synchronized
        # worker reload boundaries that can starve the queue intermittently.
        min_chunk = int(batch_size) * 4 + max(1, int(batch_size) // 4)
        self.chunk_size = max(int(chunk_size), min_chunk)
        
        # Placeholders
        self.train_dataset: Optional[CropFMIterableDataset] = None
        self.val_dataset: Optional[CropFMIterableDataset] = None
        self.test_dataset: Optional[CropFMIterableDataset] = None

    def prepare_data(self):
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(f"Zarr dataset not found at: {self.zarr_path}")

    def setup(self, stage: Optional[str] = None):
        """Instantiate the Iterable Datasets"""
        common_kwargs = {
            "zarr_path": self.zarr_path,
            "modalities": self.modalities,
            "chunk_size": self.chunk_size,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "seed": self.seed,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = CropFMIterableDataset(
                split='train', 
                shuffle=True, # Enable chunk-level shuffling for training
                **common_kwargs
            )
            self.val_dataset = CropFMIterableDataset(
                split='val', 
                shuffle=False, 
                **common_kwargs
            )
           
        if stage == "test" or stage is None:
            self.test_dataset = CropFMIterableDataset(
                split='test', 
                shuffle=False, 
                **common_kwargs
            )
                
    def train_dataloader(self) -> DataLoader:
        """
        CRITICAL CHANGES:
        1. shuffle=False (IterableDatasets don't support DataLoader shuffling)
        2. collate_fn=cropfm_collate_fn (Needed for multi-modal dicts)
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False, # HANDLED INTERNALLY IN DATASET
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=cropfm_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=cropfm_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=cropfm_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )