import os
import yaml

import numpy as np

from source.dashplot import HeatmapDashboard

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Data configuration.
rnd_seed = config['rnd_seed']
dataset = config['dataset']
model = config['model']

# Load the summary file (the intermediate outputs from inference).
summary_path = os.path.join('intermediate_outputs', f"{dataset}_{model}_{rnd_seed}.npy")
summary = np.load(summary_path, allow_pickle=True).item()

# Create the Dash app.
app = HeatmapDashboard(
    num_blocks=8,
    figsize=(2000, 650),
    **summary
)

if __name__ == '__main__':
    app.run()
