import logging

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule

from src.data.original.dataset_soares import TrajectoryDataset


logging.basicConfig()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def filter_data_by_key(pair, pattern: str):
    key, value = pair
    if pattern in key:
        return True
    else:
        return False

def pad_collate(batch):
    """Collate function to return data in a batch when loaded in the dataloader"""
    inputs = []
    outputs = []
    raw = []
    ids = []
    for data in batch:
        ids.append(data["flight_id"])

        in_data = list(dict(filter(lambda pair: filter_data_by_key(pair, "_in") , data.items())).values())
        in_features = []
        for feature in in_data:
            in_feature = torch.tensor(feature, dtype=torch.long).unsqueeze(dim=1)
            in_features.append(in_feature)
        inputs.append(torch.cat(in_features, dim=1).unsqueeze(dim=2))

        out_data = list(dict(filter(lambda pair: filter_data_by_key(pair, "_out") , data.items())).values())
        out_features = []
        for feature in out_data:
            out_feature = torch.tensor(feature, dtype=torch.long).unsqueeze(dim=1)
            out_features.append(out_feature)
        outputs.append(torch.cat(out_features, dim=1).unsqueeze(dim=2))

        raw_data = list(dict(filter(lambda pair: filter_data_by_key(pair, "_raw") , data.items())).values())
        raw_features = []
        for feature in raw_data:
            raw_feature = torch.tensor(feature, dtype=torch.float).unsqueeze(dim=1)
            raw_features.append(raw_feature)
        raw.append(torch.cat(raw_features, dim=1).unsqueeze(dim=2))

        # context_features = []
        # for feature in data["context_data"]:
        #     context_feature = torch.tensor([feature], dtype=torch.long).unsqueeze(dim=1)
        #     context_features.append(context_feature)
        # context.append(torch.cat(context_features, dim=1).unsqueeze(dim=2))
    return torch.cat(inputs, dim=2), torch.cat(outputs, dim=2), ids, torch.cat(raw, dim=2)

    
class TrajectoryDataModule(LightningDataModule):
    """Data Module for holding dataloaders for Trajectory data"""

    def __init__(
        self,
        dataset_kwargs: dict,
        test_size: float = 0.1,
        val_size: float = 0.2,
        batch_size: int = 32,
        collate_fn = pad_collate,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        if "columns" not in dataset_kwargs.keys():
            self.data = TrajectoryDataset(
                destination=dataset_kwargs["airport"],
                start=dataset_kwargs["start"],
                end=dataset_kwargs["end"],
                input_len=dataset_kwargs["input_len"],
                target_len=dataset_kwargs["target_len"],
                data_source=dataset_kwargs["data_source"],
                sampling_time=dataset_kwargs["sampling_time"],
                h3_resolution=dataset_kwargs["h3_resolution"],
                training_columns=dataset_kwargs["training_columns"],
            )
        else:
            self.data = TrajectoryDataset(
                destination=dataset_kwargs["airport"],
                start=dataset_kwargs["start"],
                end=dataset_kwargs["end"],
                input_len=dataset_kwargs["input_len"],
                target_len=dataset_kwargs["target_len"],
                data_source=dataset_kwargs["data_source"],
                sampling_time=dataset_kwargs["sampling_time"],
                h3_resolution=dataset_kwargs["h3_resolution"],
                training_columns=dataset_kwargs["training_columns"],
                columns=dataset_kwargs["columns"]
            )
            
        self.train_data, self.val_data , self.test_data = random_split(
            self.data,
            [1 - (val_size + test_size), val_size, test_size],
            torch.Generator().manual_seed(42)
        )
        _logger.debug(f"Num training flights: {len(self.train_data)}")
        _logger.debug(f"Num validation flights: {len(self.val_data)}")
        _logger.debug(f"Num test flights: {len(self.test_data)}")

    def train_dataloader(self, shuffle: bool = True) -> DataLoader: # noqa: FBT001, FBT002
        """Training data loader"""
        return self._make_dloader(self.train_data, shuffle=shuffle)

    def val_dataloader(self, shuffle: bool = True) -> DataLoader: # noqa: FBT001, FBT002
        """Validation data loader"""
        return self._make_dloader(self.val_data, shuffle=shuffle)

    def test_dataloader(self, shuffle: bool = True) -> DataLoader: # noqa: FBT001, FBT002
        """Test data loader"""
        return self._make_dloader(self.test_data, shuffle=shuffle)

    def _make_dloader(self, dataset_split: Dataset, shuffle: bool = False) -> DataLoader: # noqa: FBT001, FBT002
        """Built the dataloader for a specific data split"""
        return DataLoader(
            dataset_split,
            shuffle=shuffle,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )