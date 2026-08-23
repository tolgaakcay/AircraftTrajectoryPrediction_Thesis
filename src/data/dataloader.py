import logging

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule

from src.data.dataset import TrajectoryDataset


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
    wx_enc = []  #Change to OG Code - weather encoder-input batches ("_wxenc" keys)
    wx_dec = []  #Change to OG Code - weather decoder-input batches ("_wxdec" keys)
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

        #Change to OG Code - separate patterns from "_in"/"_out" so weather never gets mixed into the
        #predicted-feature tensors above (see experiments_weather.ipynb "Plumbing plan" discussion);
        #a no-weather dataset produces no "_wxenc"/"_wxdec" keys at all, so these stay empty lists
        wxenc_data = list(dict(filter(lambda pair: filter_data_by_key(pair, "_wxenc"), data.items())).values())
        if wxenc_data:
            wxenc_features = [torch.tensor(feature, dtype=torch.long).unsqueeze(dim=1) for feature in wxenc_data]
            wx_enc.append(torch.cat(wxenc_features, dim=1).unsqueeze(dim=2))

        wxdec_data = list(dict(filter(lambda pair: filter_data_by_key(pair, "_wxdec"), data.items())).values())
        if wxdec_data:
            wxdec_features = [torch.tensor(feature, dtype=torch.long).unsqueeze(dim=1) for feature in wxdec_data]
            wx_dec.append(torch.cat(wxdec_features, dim=1).unsqueeze(dim=2))

        # context_features = []
        # for feature in data["context_data"]:
        #     context_feature = torch.tensor([feature], dtype=torch.long).unsqueeze(dim=1)
        #     context_features.append(context_feature)
        # context.append(torch.cat(context_features, dim=1).unsqueeze(dim=2))
    wx_enc_tensor = torch.cat(wx_enc, dim=2) if wx_enc else None  #Change to OG Code - None when the dataset has no weather, matching how x_4/y_4 already use None for "not provided"
    wx_dec_tensor = torch.cat(wx_dec, dim=2) if wx_dec else None
    return torch.cat(inputs, dim=2), torch.cat(outputs, dim=2), ids, torch.cat(raw, dim=2), wx_enc_tensor, wx_dec_tensor

    
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
                weather_data_path=dataset_kwargs.get("weather_data_path"),  #Change to OG Code - optional, defaults to None (no weather)
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
                columns=dataset_kwargs["columns"],
                weather_data_path=dataset_kwargs.get("weather_data_path"),  #Change to OG Code - optional, defaults to None (no weather)
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