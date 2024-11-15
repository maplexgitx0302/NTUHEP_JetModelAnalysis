"""Module for dataset JetNet.

- https://zenodo.org/records/6975118
"""

import os

import awkward as ak
import h5py
import numpy as np

root_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

channels = ['q', 'g', 't', 'w', 'z']


class JetEvents:
    def __init__(self, channel: str) -> None:
        """Event class.

        Args:
            channel : str
                Source of the jet, could be ['q', 'g', 't', 'w', 'z'].
        """

        # Read the hdf5 files.
        dataset_dir = os.path.join(root_dir, 'dataset', 'jetnet')
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

        # Convert to awkward array.
        data = ak.Array({
            # # Fatjet information (R = 0.8).
            # 'jet_pt': jet_events[:, 0],
            # 'jet_eta': jet_events[:, 1],
            # 'jet_mass': jet_events[:, 2],

            # Daughter particle information.
            'part_deta': ptc_events[..., 0],
            'part_dphi': ptc_events[..., 1],
            'part_pt': ptc_events[..., 2] * jet_events[:, 0, np.newaxis],
            'pt_rel': ptc_events[..., 2],
        })

        self.data = data
        self.num_ptcs = jet_events[:, 3]


if __name__ == '__main__':
    for channel in ['q', 'g', 't', 'w', 'z']:
        jet_events = JetEvents(channel=channel)
        max_num_ptcs = ak.max(jet_events.num_ptcs)
        print(f"Channel {channel}: Max number of particles = {max_num_ptcs}")
