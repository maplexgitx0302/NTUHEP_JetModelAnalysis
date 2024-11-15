import numpy as np

from source.data import jetclass, jetnet, topqcd
from source.utils.dashplot import create_dash_app

dataset = jetclass

summary = np.load(f"{dataset.__name__.split('.')[-1]}.npy", allow_pickle=True).item()
num_data = summary['num_data']
num_epochs = summary['num_epochs']
particle_features = summary['particle_features']
intermediate_outputs = summary['intermediate_outputs']

app = create_dash_app(
    channels=dataset.channels,
    num_data=num_data,
    num_epochs=num_epochs,
    num_blocks=8,
    particle_features=particle_features,
    intermediate_outputs=intermediate_outputs,
    figsize=(2000, 575),
)

if __name__ == '__main__':
    app.run_server(debug=True)
