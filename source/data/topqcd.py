"""Module for dataset TopQCD.

Top Quark Tagging Reference Dataset
- https://zenodo.org/records/2603256
"""

import os
import random

import awkward as ak
import numpy as np
import pandas as pd

root_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

channels = ['top', 'qcd']


class JetEvents:
    def __init__(self, channel: str, mode: str, num_data: int) -> None:
        """Event class.

        Args:
            channel : str
                'top' or 'qcd'.
            mode : str
                'train', 'valid', or 'test'.
        """

        # Read `.h5` file.
        dataset_dir = os.path.join(root_dir, 'dataset', 'top_qcd')
        h5_path = os.path.join(dataset_dir, f"{mode}.h5")
        df = pd.read_hdf(h5_path, key='table')

        # Select top or QCD.
        df = df[df['is_signal_new'].astype(int) == (channel == 'top')]

        # Find fatjet Px, Py, Pz.
        df['Fatjet_PX'] = df[[col for col in df.columns if col[:2] == 'PX']].sum(axis=1)
        df['Fatjet_PY'] = df[[col for col in df.columns if col[:2] == 'PY']].sum(axis=1)
        df['Fatjet_PZ'] = df[[col for col in df.columns if col[:2] == 'PZ']].sum(axis=1)

        # Convert (E, Px, Py, Pz) to (Pt, Eta, Phi)
        df['Fatjet_P'] = np.sqrt(df['Fatjet_PX'] ** 2 + df['Fatjet_PY'] ** 2 + df['Fatjet_PZ'] ** 2)
        df['Fatjet_Pt'] = np.sqrt(df['Fatjet_PX'] ** 2 + df['Fatjet_PY'] ** 2)
        df['Fatjet_Phi'] = np.arctan2(df['Fatjet_PY'], df['Fatjet_PX'])
        df['Fatjet_Phi'] = np.mod(df['Fatjet_Phi'] + np.pi, 2 * np.pi) - np.pi
        df['Fatjet_Eta'] = np.arctanh(df['Fatjet_PZ'] / df['Fatjet_P'])

        # Ignore warnings for calculating Pt, Phi, Eta of padded particles.
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # Get Pt, Phi, Eta of daughter particles.
        # The leading 200 jet constituent four-momenta are stored,
        # with zero-padding for jets with fewer than 200.
        for i in range(200):
            df[f"P_{i}"] = np.sqrt(df[f"PX_{i}"] ** 2 + df[f"PY_{i}"] ** 2 + df[f"PZ_{i}"] ** 2)
            df[f"Pt_{i}"] = np.sqrt(df[f"PX_{i}"] ** 2 + df[f"PY_{i}"] ** 2)
            df[f"Phi_{i}"] = np.arctan2(df[f"PY_{i}"], df[f"PX_{i}"])
            df[f"Phi_{i}"] = np.mod(df[f"Phi_{i}"] + np.pi, 2 * np.pi) - np.pi
            df[f"Eta_{i}"] = np.arctanh(df[f"PZ_{i}"] / df[f"P_{i}"])

        # Turn into numpy array for convenience.
        pt = df[[f"Pt_{i}" for i in range(200)]].to_numpy()
        eta = df[[f"Eta_{i}" for i in range(200)]].to_numpy()
        phi = df[[f"Phi_{i}" for i in range(200)]].to_numpy()

        fatjet_pt = df['Fatjet_Pt'].to_numpy()
        fatjet_eta = df['Fatjet_Eta'].to_numpy()
        fatjet_phi = df['Fatjet_Phi'].to_numpy()

        # Mask for padded particles -> (pt == 0.) | (eta == nan).
        mask = (pt == 0.) | np.isnan(eta)
        pt = ak.drop_none(ak.mask(pt, mask == False))
        eta = ak.drop_none(ak.mask(eta, mask == False))
        phi = ak.drop_none(ak.mask(phi, mask == False))

        # Particle flow features.
        pt_rel = pt / fatjet_pt[:, np.newaxis]
        delta_eta = eta - fatjet_eta[:, np.newaxis]
        delta_phi = phi - fatjet_phi[:, np.newaxis]

        data = ak.Array({
            # # Fatjet information (R = 0.8).
            # 'jet_pt': fatjet_pt,
            # 'jet_eta': fatjet_eta,
            # 'jet_phi': fatjet_phi,

            # Daughter particle information.
            'part_deta': delta_eta,
            'part_dphi': delta_phi,
            'part_pt': pt,
            'pt_rel': pt_rel,
        })

        data = data[random.sample(range(len(data)), num_data)]
        self.data = data
        self.num_ptcs = ak.num(data['part_deta'], axis=1)

    def __add__(self, other: 'JetEvents') -> 'JetEvents':
        self.data = ak.concatenate([self.data, other.data], axis=0)
        self.num_ptcs = ak.concatenate([self.num_ptcs, other.num_ptcs], axis=0)
        return self


if __name__ == '__main__':
    for channel in channels:
        for mode in ['train', 'valid', 'test']:
            jet_events = JetEvents(channel=channel, mode=mode)
            max_num_ptcs = max(jet_events.num_ptcs)
            print(f"Channel {channel} ({mode}): Max number of particles = {max_num_ptcs}")
