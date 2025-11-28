"""DataModule placeholder for CropMAE"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cropfm.models.dataset import CropFMDataset


class CropFMDataModule(LightningDataModule):
    """Placeholder DataModule for CropFM MAE training"""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        """
        Args:
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: str | None = None):
        """Setup datasets - placeholder for now"""
        # TODO: Implement dataset initialization
        pass

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader - placeholder"""
        # TODO: Implement training dataloader
        dataset = CropFMDataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader - placeholder"""
        # TODO: Implement validation dataloader
        dataset = CropFMDataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader - placeholder"""
        # TODO: Implement test dataloader
        dataset = CropFMDataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

