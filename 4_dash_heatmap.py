import os
import yaml

import numpy as np

from source.dashplot import ParticleTransformerHeatmap, HeatmapDashboard

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Data configuration.
rnd_seed = config['rnd_seed']
dataset = config['dataset']
model = config['model']

# Load the summary file (the intermediate outputs from inference).
summary_path = os.path.join('intermediate_outputs', f"{dataset}_{model}_{rnd_seed}.npy")
summary = np.load(summary_path, allow_pickle=True).item()

# Create the particle transformer heatmap object.
particle_features = summary['particle_features']
intermediate_outputs = summary['intermediate_outputs']
heatmap = ParticleTransformerHeatmap(
    particle_features=particle_features,
    intermediate_outputs=intermediate_outputs,
    figsize=(2000, 575)
)

# Create the Dash app.
app = HeatmapDashboard(
    channels=summary['channels'],
    num_data=summary['num_data'],
    num_epochs=summary['num_epochs'],
    heatmap=heatmap,
    io_buttons=[[f"Block {i+1}" for i in range(8)]]
)

if __name__ == '__main__':
    app.run()
