import itertools

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HeatmapObject:
    def __init__(self, particle_features: pd.DataFrame, figsize: tuple[int, int]):
        """Template class for heatmap dashboard."""

        self.particle_features = particle_features
        self.figsize = figsize

    def plot(self, channel, data_index, epoch_index, io_indices) -> go.Figure:
        """Plot the features and intermediate outputs."""

        # The first row is for particle features.
        particle_features = self.particle_features[
            (self.particle_features['channel'] == channel) &
            (self.particle_features['data_index'] == data_index)
        ]
        features = particle_features['feature'].unique()

        # The rest of rows are for intermediate outputs.
        intermediate_outputs = self.intermediate_outputs_matrices(channel, data_index, epoch_index, io_indices)

        # Determine the number of rows and columns.
        num_row = 1 + len(intermediate_outputs)
        num_col = max(len(features), max(len(outputs) for outputs in intermediate_outputs))

        # Determine the subplot titles.
        subplot_titles = []
        subplot_titles += [f for f in features] + [''] * (num_col - len(features))
        for row_outputs in intermediate_outputs:
            subplot_titles += [row_outputs[i][0] for i in range(len(row_outputs))] + [''] * (num_col - len(row_outputs))

        # Create the figure with plotly.
        fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=subplot_titles)

        # Add the particle features to the first row.
        for i, feature in enumerate(features):
            data = particle_features[particle_features['feature'] == feature]['array'].values[0]
            data = self.feature_to_square_matrix(data)
            fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=1, col=i + 1)

        # Add the intermediate outputs to the rest of rows.
        for i, row_outputs in enumerate(intermediate_outputs):
            for j, (_, data) in enumerate(row_outputs):
                fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=2 + i, col=j + 1)

        # Update the layout.
        fig.update_layout(width=self.figsize[0], height=self.figsize[1])

        return fig

    def feature_to_square_matrix(self, feature) -> np.ndarray:
        """Make 1D feature array into a square matrix."""

        num_ptcs = len(feature)
        min_value = min(feature)
        max_value = max(feature)

        # Pad non-diagonal elements with the min_value - (max_value - min_value).
        pad_value = 2 * min_value - max_value
        diagonal = np.full((num_ptcs, num_ptcs), pad_value)
        np.fill_diagonal(diagonal, feature)

        return diagonal

    def intermediate_outputs_matrices(self, channel: str, data_index: int, epoch_index: int, io_indices: list[int]) -> list[list[tuple[str, np.ndarray]]]:
        """To be implemented in the child class.

        The matrices returned will be plotted row by row.

        Args:
            channel: str
                The channel name in string format.
            data_index: int
                The index of the data.
            epoch_index: int
                The index of the epoch.
            io_indices: list[int]
                The indices of the intermediate outputs.
        Returns:
            In format of:
            [
                [(io_1_1, matrix_1_1), (io_1_2, matrix_1_2), ...],
                [(io_2_1, matrix_2_1), (io_2_2, matrix_2_2), ...],
            ]
        """

        raise NotImplementedError


class ParticleTransformerHeatmap(HeatmapObject):
    def __init__(self, particle_features: pd.DataFrame, intermediate_outputs: pd.DataFrame, figsize: tuple[int, int]):

        super().__init__(particle_features=particle_features, figsize=figsize)

        self.intermediate_outputs = intermediate_outputs

    def intermediate_outputs_matrices(self, channel, data_index, epoch_index, io_indices):
        # Extract the indices in `io_indices`.
        block_index = io_indices[0]

        # Filter the intermediate outputs.
        filtered_intermediate_outputs = self.intermediate_outputs[
            (self.intermediate_outputs['channel'] == channel) &
            (self.intermediate_outputs['data_index'] == data_index) &
            (self.intermediate_outputs['epoch_index'] == epoch_index) &
            (self.intermediate_outputs['block_index'] == block_index)
        ]

        # Intermediate outputs for each head.
        head_outputs = filtered_intermediate_outputs['output'].values[0]

        return [
            [(f"Head {i+1}", head_outputs[i]) for i in range(len(head_outputs))],
        ]


class HeatmapDashboard:
    def __init__(self, channels, num_data, num_epochs, heatmap: HeatmapObject, io_buttons: list[list[str]]):

        self.channels = channels
        self.num_data = num_data
        self.num_epochs = num_epochs
        self.heatmap = heatmap
        self.io_buttons = io_buttons

        self.default_style = {'background-color': 'lightgray'}
        self.active_style = {'background-color': 'lightblue'}

        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def run(self):
        self.app.run_server(debug=True)

    def setup_layout(self):
        # Channel buttons (verticle).
        channel_buttons = html.Div(
            [html.Button(label, id=f'channel_{i}', n_clicks=0, style=self.default_style) for i, label in enumerate(self.channels)],
            style={'display': 'flex', 'gap': '5px', 'flex-direction': 'column', 'margin-right': '10px'},
        )

        # Data buttons (verticle).
        data_buttons = html.Div(
            [html.Button(f"Data {i+1}", id=f'data_{i}', n_clicks=0, style=self.default_style) for i in range(self.num_data)],
            style={'display': 'flex', 'gap': '5px', 'flex-direction': 'column', 'margin-right': '10px'},
        )

        # Epoch buttons (verticle).
        epoch_buttons = html.Div(
            [html.Button(f"Epoch {i+1}", id=f'epoch_{i}', n_clicks=0, style=self.default_style) for i in range(self.num_epochs)],
            style={'display': 'flex', 'gap': '5px', 'flex-direction': 'column', 'margin-right': '10px'},
        )

        # Intermediate output buttons (horizontal).
        io_buttons = []
        for i in range(len(self.io_buttons)):
            io_buttons.append(html.Div(
                [html.Button(label, id=f'io_{i}_{j}', n_clicks=0, style=self.default_style) for j, label in enumerate(self.io_buttons[i])],
                style={'display': 'flex', 'gap': '5px', 'flex-direction': 'row', 'margin-right': '10px'},
            ))

        # Heatmap graph.
        heatmap = dcc.Graph(id='heatmap_grid', config={'displayModeBar': False})

        # Dashboard layout.
        self.app.layout = html.Div(
            [
                html.Div(
                    [channel_buttons, data_buttons, epoch_buttons],
                    style={'display': 'flex', 'gap': '5px', 'flex-direction': 'row', 'margin-right': '10px'},
                ),
                html.Div([
                    *io_buttons,
                    heatmap,
                ]),
                html.Div([
                    html.Div(id='current_channel', style={'display': 'none'}),
                    html.Div(id='current_data', style={'display': 'none'}),
                    html.Div(id='current_epoch', style={'display': 'none'}),
                    *[html.Div(id=f'current_io_{i}', style={'display': 'none'}) for i in range(len(self.io_buttons))],
                ]),
            ],
            style={'display': 'flex', 'align-items': 'flex-start'},
        )

    def setup_callbacks(self):
        # App callback `Output` (related to the order of return values).
        app_output = []
        app_output += [Output('heatmap_grid', 'figure')]
        app_output += [Output('current_channel', 'children')]
        app_output += [Output('current_data', 'children')]
        app_output += [Output('current_epoch', 'children')]
        app_output += [Output(f'current_io_{i}', 'children') for i in range(len(self.io_buttons))]
        app_output += [Output(f'channel_{i}', 'style') for i in range(len(self.channels))]
        app_output += [Output(f'data_{i}', 'style') for i in range(self.num_data)]
        app_output += [Output(f'epoch_{i}', 'style') for i in range(self.num_epochs)]
        app_output += [Output(f'io_{i}_{j}', 'style') for i in range(len(self.io_buttons)) for j in range(len(self.io_buttons[i]))]

        # App callback `Input` (related to clicks).
        app_input = []
        app_input += [Input(f'channel_{i}', 'n_clicks') for i in range(len(self.channels))]
        app_input += [Input(f'data_{i}', 'n_clicks') for i in range(self.num_data)]
        app_input += [Input(f'epoch_{i}', 'n_clicks') for i in range(self.num_epochs)]
        app_input += [Input(f'io_{i}_{j}', 'n_clicks') for i in range(len(self.io_buttons)) for j in range(len(self.io_buttons[i]))]

        # App callback `State` (related to the order of *args of the decorated function).
        app_state = []
        app_state += [State('current_channel', 'children')]
        app_state += [State('current_data', 'children')]
        app_state += [State('current_epoch', 'children')]
        app_state += [State(f'current_io_{i}', 'children') for i in range(len(self.io_buttons))]

        # Callback to update heatmap grid, block switching, and button colors.
        @self.app.callback(app_output, app_input, app_state)
        def update_heatmap(*args):
            """Called when a button is clicked to update the heatmap grid."""

            # Related to the order of `app_state` (default set to the first index, i.e., 0).
            channel_idx = int(args[-3 - len(self.io_buttons)]) if args[-3 - len(self.io_buttons)] else 0
            data_idx = int(args[-2 - len(self.io_buttons)]) if args[-2 - len(self.io_buttons)] else 0
            epoch_idx = int(args[-1 - len(self.io_buttons)]) if args[-1 - len(self.io_buttons)] else 0
            io_indices = [int(args[-len(self.io_buttons) + i]) if args[-len(self.io_buttons) + i] else 0 for i in range(len(self.io_buttons))]

            # Handle the clicked button with button's ID.
            ctx = dash.callback_context
            button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_clicked.startswith('channel_'):
                channel_idx = int(button_clicked.split('_')[1])
            elif button_clicked.startswith('data_'):
                data_idx = int(button_clicked.split('_')[1])
            elif button_clicked.startswith('epoch_'):
                epoch_idx = int(button_clicked.split('_')[1])
            elif button_clicked.startswith('io_'):
                i = int(button_clicked.split('_')[1])
                j = int(button_clicked.split('_')[2])
                io_indices[i] = j

            # The HeatmapObject's `plot` method will return a plotly figure.
            fig = self.heatmap.plot(self.channels[channel_idx], data_idx, epoch_idx, io_indices)

            # Update the button styles, especially the clicked button.
            styles = self.generate_button_styles(channel_idx, data_idx, epoch_idx, io_indices)

            # The return values should be in the order of `app_output`.
            return [fig, channel_idx, data_idx, epoch_idx, *io_indices] + styles

    def generate_button_styles(self, channel_idx, data_idx, epoch_idx, io_indices):
        """Update button colors based on the current indices."""
        styles = []
        styles += [self.active_style if i == channel_idx else self.default_style for i in range(len(self.channels))]
        styles += [self.active_style if i == data_idx else self.default_style for i in range(self.num_data)]
        styles += [self.active_style if i == epoch_idx else self.default_style for i in range(self.num_epochs)]
        for i, io_index in enumerate(io_indices):
            styles += [self.active_style if j == io_index else self.default_style for j in range(len(self.io_buttons[i]))]
        return styles


def generate_sample_data(channels, num_data, num_epochs, num_blocks, num_heads):
    particle_features = []
    intermediate_outputs = []

    for channel, data_idx in itertools.product(channels, range(num_data)):
        num_ptcs = np.random.randint(5, 20)

        for feature in ['pt', 'eta', 'phi']:
            particle_features.append({'channel': channel, 'data_index': data_idx, 'feature': feature, 'array': np.random.rand(num_ptcs)})

        for epoch_idx, block_idx in itertools.product(range(num_epochs), range(num_blocks)):
            intermediate_outputs.append({'channel': channel, 'data_index': data_idx, 'epoch_index': epoch_idx, 'block_index': block_idx, 'output': np.random.rand(num_heads, num_ptcs, num_ptcs)})

    return pd.DataFrame(particle_features), pd.DataFrame(intermediate_outputs)


if __name__ == '__main__':

    channels = ['QCD', 'TOP', 'Higgs']
    num_data, num_epochs, num_blocks, num_heads = 5, 3, 8, 4

    particle_features, intermediate_outputs = generate_sample_data(channels, num_data, num_epochs, num_blocks, num_heads)

    heatmap = ParticleTransformerHeatmap(particle_features, intermediate_outputs, figsize=(1000, 625))
    dashboard = HeatmapDashboard(channels, num_data, num_epochs, heatmap, io_buttons=[[f"Head {i+1}" for i in range(num_heads)]])

    dashboard.run()
