"""Lightning data module"""

from typing import Optional

import awkward as ak
import lightning as L
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
import tqdm


class JetEvents:
    def __init__(self):
        """Class for jet events.

        This class serves as a type hint for the data structure of jet events.
        """

        self.data = ak.Array([])
        self.num_ptcs = ak.Array([])


class JetTorchDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: int):
        """Torch format dataset.

        Args:
            x : torch.Tensor
                Feature data.
            y : torch.Tensor
                Label of data.
        """

        # Data features
        self.x = x

        # Data label
        # CrossEntropyLoss requires in format of `torch.long` and size of (N,).
        self.y = torch.full((x.shape[0], ), y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class JetLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        events_list: list[JetEvents],
        num_train: int,
        num_valid: int,
        num_test: int,
        batch_size: int,
        pad_num_ptcs: Optional[int] = None,
    ):
        """Pytorch Lightning Data Module for jet.

        Args:
            events_list : list[ak.Array]
                List of FatJetEvents, label will be the sequence order.
            num_train / num_valid / num_test : int
                Number of training / validation / testing data.
            batch_size : int
                Batch size for data loaders.
            pad_num_ptcs : int (default None)
                Pad number of particles within jets.
        """

        super().__init__()

        self.batch_size = batch_size
        self.num_train = num_train
        self.num_valid = num_valid
        self.num_test = num_test

        # Jet and particle features used.
        part_flow = ['log_part_pt', 'part_deta', 'part_dphi', 'log_part_E']
        part_flow = [field for field in part_flow if field in events_list[0].data.fields]
        non_part_flow = [field for field in events_list[0].data.fields if field not in part_flow]

        # Make sure particle flow features are in the first three fields.
        self.fields = part_flow + non_part_flow

        # Determine maximum number of particles within jets.
        if pad_num_ptcs is None:
            pad_num_ptcs = int(max([max(jet_events.num_ptcs) for jet_events in events_list]))

        # Pad each jet events with float('nan') to pad_num_ptcs.
        torch_data: list[torch.Tensor] = []
        for jet_events in tqdm.tqdm(events_list, desc='Creating JetLightningDataModule'):
            data = jet_events.data
            data = ak.pad_none(data, target=pad_num_ptcs, clip=False)
            data = ak.fill_none(data, float('nan'))
            data = [ak.to_torch(data[field]) for field in self.fields]
            data = torch.stack(data, dim=-1).to(torch.float32)
            torch_data.append(data)

        # Assign class labels.
        torch_data = [JetTorchDataset(x=x, y=i) for i, x in enumerate(torch_data)]
        self.data_list: list[JetTorchDataset] = torch_data

    def train_dataloader(self):
        """Training data loader"""

        idx_start = 0
        idx_end = self.num_train
        dataset = ConcatDataset([Subset(data, range(idx_start, idx_end)) for data in self.data_list])

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Validation data loader"""

        idx_start = self.num_train
        idx_end = self.num_train + self.num_valid
        dataset = ConcatDataset([Subset(data, range(idx_start, idx_end)) for data in self.data_list])

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Testing data loader"""

        idx_start = self.num_train + self.num_valid
        idx_end = self.num_train + self.num_valid + self.num_test
        dataset = ConcatDataset([Subset(data, range(idx_start, idx_end)) for data in self.data_list])

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
