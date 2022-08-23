import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from jsonargparse.typing import PositiveInt, ClosedUnitInterval
from kaggle.api.kaggle_api_extended import KaggleApi

from . import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"


def load_data_from_json(data_dir: Union[str, Path] = DATA_DIR) -> dict:
    """Reads data from json file and returns dict."""
    data_dir = Path(str(data_dir))
    data_json = data_dir / "shipsnet.json"
    with data_json.open("r") as file:
        data_dict = json.load(file)
    return data_dict


class LabelledTensorDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


class ShipsDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: PositiveInt = 32,
        train_frac: ClosedUnitInterval = 0.75,
        data_dir: Union[str, Path] = DATA_DIR,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.data_dir = Path(str(data_dir)).resolve()

    def prepare_data(self) -> None:
        """Downloads and extracts the dataset."""
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()
        kaggle_api.dataset_download_files(
            dataset="rhammell/ships-in-satellite-imagery",
            path=str(self.data_dir),
            force=False,  # skip if already downloaded
            quiet=False,
            unzip=True,
        )

    def setup(self, stage: Union[str, None] = None) -> None:
        """Creates train/val/test datasets."""
        data_dict = load_data_from_json(self.data_dir)

        # Convert to torch.Tensor
        pixels = torch.tensor(data_dict["data"], dtype=float).view(-1, 3, 80, 80)
        labels = torch.tensor(data_dict["labels"], dtype=bool)

        # Rescale pixels to [-0.5, 0.5]
        pixels = pixels / 255 - 0.5

        dataset = LabelledTensorDataset(pixels, labels)

        # Split into train / validation / test
        n_tot = len(dataset)
        n_train = int(n_tot * self.train_frac)
        n_val = (n_tot - n_train) // 2
        n_test = n_tot - n_val - n_train
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers=1)
