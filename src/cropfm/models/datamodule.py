"""DataModule for CropFM multi-modal agricultural data"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional, Callable, List
import os

from cropfm.models.dataset import CropFMDataset


class CropFMDataModule(LightningDataModule):
    """
    LightningDataModule for CropFM multi-modal agricultural time series data.
    
    This DataModule coordinates dataset creation and dataloader configuration for
    training, validation, and testing phases. The actual data splitting logic is
    handled by the CropFMDataset class.
    
    Args:
        zarr_path (str): 
            Path to the Zarr dataset file containing all modalities.
            Example: '/data/cropfm/processed_data.zarr'
        
        modalities (Optional[List[str]], optional): 
            List of modality names to load. If None, loads all available modalities.
            Available modalities include:
            - Temporal: 'agera5', 'sentinel1', 'sentinel2', 'fapar'
            - Static: 'soil', 'elevation', 'worldcereal_cropmask', 'worldcereal_cropcalender'
            Default: None (loads all modalities)
        
        transform (Optional[Callable], optional): 
            Optional transform function to apply to each sample.
            Should accept a dict of tensors and return a dict of tensors.
            Example: transform=lambda x: {k: normalize(v) for k, v in x.items()}
            Default: None
        
        batch_size (int, optional): 
            Number of samples per batch for all dataloaders.
            Default: 32
        
        num_workers (int, optional): 
            Number of subprocesses to use for data loading.
            Set to 0 for single-process loading (useful for debugging).
            Typically set to number of CPU cores available.
            Default: 4
        
        train_split (float, optional): 
            Fraction of data to use for training.
            Must be in range (0, 1) and train_split + val_split <= 1.0.
            Default: 0.8 (80% of data)
        
        val_split (float, optional): 
            Fraction of data to use for validation.
            Must be in range (0, 1) and train_split + val_split <= 1.0.
            The remaining data (1 - train_split - val_split) is used for testing.
            Default: 0.1 (10% of data)
        
        seed (int, optional): 
            Random seed for reproducible dataset splitting.
            Using the same seed ensures the same train/val/test split across runs.
            Default: 42
        
        pin_memory (bool, optional): 
            Whether to pin memory in DataLoader for faster GPU transfer.
            Set to True when using GPU, False for CPU-only training.
            Default: True
    
    Attributes:
        train_dataset (CropFMDataset): Training dataset (available after setup)
        val_dataset (CropFMDataset): Validation dataset (available after setup)
        test_dataset (CropFMDataset): Test dataset (available after setup)
    
    Example:
        >>> # Create DataModule
        >>> dm = CropFMDataModule(
        ...     zarr_path='/data/cropfm.zarr',
        ...     modalities=['sentinel2', 'agera5', 'soil'],
        ...     batch_size=64,
        ...     num_workers=8,
        ...     train_split=0.8,
        ...     val_split=0.1,
        ...     seed=42
        ... )
        >>> 
        >>> # Use with PyTorch Lightning Trainer
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        zarr_path: str,
        modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        # Validate splits
        if not 0 < train_split < 1:
            raise ValueError(f"train_split must be in (0, 1), got {train_split}")
        if not 0 < val_split < 1:
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")
        if train_split + val_split >= 1.0:
            raise ValueError(
                f"train_split ({train_split}) + val_split ({val_split}) must be < 1.0"
            )
        
        # Save hyperparameters (exclude transform as it's not serializable)
        self.save_hyperparameters(ignore=['transform'])
        
        # Store parameters as instance variables
        self.zarr_path = zarr_path
        self.modalities = modalities
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.pin_memory = pin_memory
        
        # Calculate test split for reference
        self.test_split = 1.0 - train_split - val_split
        
        # Initialize dataset placeholders (will be created in setup())
        self.train_dataset: Optional[CropFMDataset] = None
        self.val_dataset: Optional[CropFMDataset] = None
        self.test_dataset: Optional[CropFMDataset] = None

    def prepare_data(self):
        """
        Prepare data - called only once and on a single process.
        
        This method is used for operations that should only be done once,
        such as downloading data or verifying data integrity.
        In distributed training, this runs on only one process.
        
        Here we verify that the Zarr dataset exists and is accessible.
        
        Raises:
            FileNotFoundError: If the Zarr dataset is not found at zarr_path
        """
        # Verify zarr path exists
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(
                f"Zarr dataset not found at: {self.zarr_path}\n"
                f"Please ensure the data has been processed and saved to this location."
            )

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        
        Called on every process in distributed training. This method creates
        the train/val/test datasets by instantiating CropFMDataset with the
        appropriate split parameter.
        
        Args:
            stage (Optional[str]): 
                Current stage - one of 'fit', 'validate', 'test'.
                if None, creates all datasets.
                - 'fit' or None: Creates train and val datasets
                - 'validate': Creates val dataset only
                - 'test': Creates test dataset only
        """
        # Setup for training and validation
        if stage == "fit" or stage is None:
            # Create training dataset
            self.train_dataset = CropFMDataset(
                zarr_path=self.zarr_path,
                split='train',
                modalities=self.modalities,
                transform=self.transform,
                train_split=self.train_split,
                val_split=self.val_split,
                seed=self.seed
            )
            
            # Create validation dataset
            self.val_dataset = CropFMDataset(
                zarr_path=self.zarr_path,
                split='val',
                modalities=self.modalities,
                transform=self.transform,
                train_split=self.train_split,
                val_split=self.val_split,
                seed=self.seed
            )
           
        # Setup for testing
        if stage == "test" or stage is None:
            self.test_dataset = CropFMDataset(
                zarr_path=self.zarr_path,
                split='test',
                modalities=self.modalities,
                transform=self.transform,
                train_split=self.train_split,
                val_split=self.val_split,
                seed=self.seed
            )
                
    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.
        
        Returns:
            DataLoader: PyTorch DataLoader for training data with shuffling enabled
        
        Configuration:
            - shuffle=True: Shuffles data every epoch for better generalization
            - drop_last=False: Keeps the last incomplete batch
            - pin_memory: Set based on __init__ parameter
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data each epoch
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader.
        
        Returns:
            DataLoader: PyTorch DataLoader for validation data without shuffling
        
        Configuration:
            - shuffle=False: No shuffling for consistent validation
            - drop_last=False: Evaluate on all validation samples
            - pin_memory: Set based on __init__ parameter
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader.
        
        Returns:
            DataLoader: PyTorch DataLoader for test data without shuffling
        
        Configuration:
            - shuffle=False: No shuffling for consistent testing
            - drop_last=False: Evaluate on all test samples
            - pin_memory: Set based on __init__ parameter
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )