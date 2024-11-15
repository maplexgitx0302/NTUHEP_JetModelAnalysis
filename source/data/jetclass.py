"""Module for dataset JetClass.

JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics
- https://zenodo.org/records/6619768
"""

import os

import awkward as ak
import uproot

root_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

channels = [
    'HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q',
    'TTBar', 'TTBarLep', 'WToQQ', 'ZJetsToNuNu', 'ZToQQ',
]

features = [
    # Kinematics
    'part_deta',
    'part_dphi',
    'log_part_pt',
    'log_part_E',
    'log_pt_rel',
    'log_E_rel',
    'part_dR',

    # Particle Identification
    'part_charge',
    'part_isElectron',
    'part_isMuon',
    'part_isPhoton',
    'part_isChargedHadron',
    'part_isNeutralHadron',

    # Trajectory Displacement
    'tanh(part_d0val)',
    'tanh(part_dzval)',
    'part_d0err',
    'part_dzerr',
]

aliases = {
    'part_pt': 'sqrt(part_px ** 2 + part_py ** 2)',
    'part_dR': 'sqrt(part_deta ** 2 + part_dphi ** 2)',
    'log_part_pt': 'log(part_pt)',
    'log_part_E': 'log(part_energy)',
    'log_pt_rel': 'log(part_pt / jet_pt)',
    'log_E_rel': 'log(part_energy / jet_energy)',
}


class JetEvents:
    def __init__(self, channel: str, num_root: int = 1) -> None:
        """Event class.

        Args:
            channel : str
                Channel name.
            num_root : int, optional
                Number of root files to be loaded.
        """

        dataset_dir = os.path.join(root_dir, 'dataset', 'jet_class')

        self.data = None
        for i in range(num_root):
            _file = uproot.open(os.path.join(dataset_dir, f"{channel}_00{i}.root"))
            _tree = _file['tree;1']
            _data = _tree.arrays(features, aliases=aliases)

            if self.data is None:
                self.data = _data
            else:
                self.data = ak.concatenate([self.data, _data])

        self.num_ptcs = ak.num(self.data['part_deta'], axis=1)


if __name__ == '__main__':
    for channel in channels:
        jet_events = JetEvents(channel=channel, num_root=2)
        max_num_ptcs = max(jet_events.num_ptcs)
        print(f"Channel {channel}: Max number of particles = {max_num_ptcs} | #>128 = {ak.sum(jet_events.num_ptcs > 128)}\n")
