import os
from typing import Optional

import awkward as ak
import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
import uproot


# Define root directory relative to this file
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class JetEvents:
    """Base class for jet events."""
    CHANNELS = []  # Channel names for different jet processes.
    INCLUDE_MASS = None  # Whether include mass or energy in the features.
    INPUT_DIM = None  # Number of input features.

    def __init__(self):
        self.channel: str = None
        self.data: ak.Array = None

    def number_of_particles(self) -> ak.Array:
        """Calculate the number of particles per event."""
        raise NotImplementedError


class JetClass(JetEvents):
    CHANNELS = [
        'HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q',
        'TTBar', 'TTBarLep', 'WToQQ', 'ZJetsToNuNu', 'ZToQQ',
    ]

    INPUT_DIM = 17

    # Features used for jet classification
    FEATURES = [
        # Kinematics
        'part_deta', 'part_dphi', 'log_part_pt', 'log_part_E',
        'log_pt_rel', 'log_E_rel', 'part_dR',

        # Particle Identification
        'part_charge', 'part_isElectron', 'part_isMuon', 'part_isPhoton',
        'part_isChargedHadron', 'part_isNeutralHadron',

        # Trajectory Displacement
        'tanh(part_d0val)', 'tanh(part_dzval)', 'part_d0err', 'part_dzerr',
    ]

    ALIASES = {
        # Definition of the features in FEATURES
        'part_pt': 'sqrt(part_px ** 2 + part_py ** 2)',
        'part_dR': 'sqrt(part_deta ** 2 + part_dphi ** 2)',
        'log_part_pt': 'log(part_pt)',
        'log_part_E': 'log(part_energy)',
        'log_pt_rel': 'log(part_pt / jet_pt)',
        'log_E_rel': 'log(part_energy / jet_energy)',
    }

    INCLUDE_MASS = True

    def __init__(self, channel: str, num_root: int = 1, **kwargs) -> None:
        """JetClass https://zenodo.org/records/6619768

        - 10 classes of jet events.
        - Each root file contains 100K events.

        Args:
            num_root : int, optional
                Number of root files to be loaded.
        """

        self.channel = channel

        # Build dataset directory
        dataset_dir = os.path.join(root_dir, 'dataset', 'JetClass')

        # Collect arrays before concatenation for efficiency
        arrays = []
        for i in range(num_root):
            root_file_path = os.path.join(dataset_dir, f"{channel}_00{i}.root")

            # Open and read the root file
            with uproot.open(root_file_path) as root_file:
                tree = root_file['tree;1']
                data = tree.arrays(self.FEATURES, aliases=self.ALIASES)
                arrays.append(data)

        # Concatenate arrays
        self.data = ak.concatenate(arrays)

    def number_of_particles(self) -> ak.Array:
        return ak.num(self.data['part_deta'], axis=1)


class JetNet:
    CHANNELS = ['q', 'g', 't', 'w', 'z']
    INCLUDE_MASS = False
    INPUT_DIM = 3

    def __init__(self, channel: str, **kwargs) -> None:
        """JetNet https://zenodo.org/records/6975118"""

        self.channel = channel

        # Read the hdf5 files.
        dataset_dir = os.path.join(root_dir, 'dataset', 'JetNet')
        hdf5_path = os.path.join(dataset_dir, channel + '.hdf5')
        hdf5_file = h5py.File(hdf5_path, 'r')

        # Data: (N, 30, 4) with 4 features (eta_rel, phi_rel, pt_rel, mask)
        ptc_events = np.array(hdf5_file['particle_features'])
        ptc_events, mask = ptc_events[..., :-1], (ptc_events[..., -1] == 0.)

        # Turn into non-padded awkward array.
        ptc_events = ak.mask(ptc_events, mask == False)
        ptc_events = ak.drop_none(ptc_events)

        # `jet_features` shape (N, 4) with 4 features (pt, eta, mass, # of particles)
        jet_events = np.array(hdf5_file['jet_features'])

        # Fatjet information (R = 0.8).
        jet_pt: np.ndarray = jet_events[:, 0][..., np.newaxis]
        jet_eta: np.ndarray = jet_events[:, 1][..., np.newaxis]
        jet_mass: np.ndarray = jet_events[:, 2][..., np.newaxis]

        # Only retain daughter particle information.
        self.data = ak.Array({
            'log_part_pt': np.log(ptc_events[..., 2] * jet_pt),
            'part_deta': ptc_events[..., 0],
            'part_dphi': ptc_events[..., 1],
        })

    def number_of_particles(self) -> ak.Array:
        return ak.num(self.data['part_deta'], axis=1)


class JetTorchDataset(Dataset):
    def __init__(self, jet_events: JetEvents, label: int, pad_num_ptcs: int):
        """Turn JetEvents into torch format dataset.

        Args:
            jet_events : JetEvents
                Jet events, the data can be fetched by `jet_events.data`.
            label : int
                Label of the jet events.
            pad_num_ptcs : int
                Pad (or discard) particles within jets to certain number.
        """

        self.channel = jet_events.channel

        # Pad the data and turn it into torch tensor.
        data, fields = self.padding(jet_events, pad_num_ptcs)
        self.data = data
        self.fields = fields

        # CrossEntropyLoss requires in format of `torch.long` and size of (N,).
        self.label = torch.full((len(self.data), ), label, dtype=torch.long)

    def padding(self, jet_events: JetEvents, pad_num_ptcs: int) -> torch.Tensor:
        # Get the awkward array data.
        data = jet_events.data

        # Sort the particle respect to pT.
        sorted_indices = ak.argsort(data['log_part_pt'], axis=1, ascending=False)
        data = ak.Array({field: data[field][sorted_indices] for field in data.fields})

        # Pad and clip to exactly pad_num_ptcs.
        data = ak.pad_none(data, target=pad_num_ptcs, clip=True, axis=1)
        data = ak.fill_none(data, float('nan'))

        # Order the fields in ['log_part_pt', 'part_deta', 'part_dphi', 'log_part_E', ...]
        fields = [field for field in ('log_part_pt', 'part_deta', 'part_dphi', 'log_part_E') if field in data.fields]
        fields += [field for field in data.fields if field not in fields]

        # Convert to torch tensor respect to the order of `fields`.
        data = [ak.to_torch(data[field]) for field in fields]
        data = torch.stack(data, dim=-1).to(torch.float32)

        return data, fields

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]


class JetLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        jet_events_list: list[JetEvents],
        num_train: int,
        num_valid: int,
        num_test: int,
        batch_size: int,
        pad_num_ptcs: int,
        **kwargs,
    ):
        """Pytorch Lightning Data Module for jet.

        Args:
            jet_events_list : list[JetEvents]
                List of FatJetEvents, label will be the sequence order.
            num_train / num_valid / num_test : int
                Number of training / validation / testing data.
            batch_size : int
                Batch size for data loaders.
            pad_num_ptcs : int
                Pad number of particles within jets.
        """

        super().__init__()

        self.batch_size = batch_size
        self.num_train = num_train
        self.num_valid = num_valid
        self.num_test = num_test

        self.data_list: list[JetTorchDataset] = [
            JetTorchDataset(
                jet_events=jet_events,
                label=i,
                pad_num_ptcs=pad_num_ptcs,
            ) for i, jet_events in enumerate(jet_events_list)
        ]

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


if __name__ == '__main__':

    def print_jet_info(jet_events: JetEvents, channel: str) -> None:
        num_ptcs = jet_events.number_of_particles()
        print(f"Channel {channel} in {jet_events.__class__.__name__}:")
        print(f"- Max number of particles = {ak.max(num_ptcs)}")
        print(f"- Number of events = {len(num_ptcs)}")
        print('-' * 50)

    # JetClass
    for channel in JetClass.CHANNELS[:2]:
        jet_events = JetClass(channel=channel, num_root=1)
        print_jet_info(jet_events, channel)

    # JetNet
    for channel in JetNet.CHANNELS[:2]:
        jet_events = JetNet(channel=channel)
        print_jet_info(jet_events, channel)

    # JetLightningDataModule
    JetLightningDataModule(
        jet_events_list=[JetNet(channel='q'), JetNet(channel='g')],
        num_train=1000,
        num_valid=100,
        num_test=100,
        batch_size=64,
        pad_num_ptcs=50,
    )
