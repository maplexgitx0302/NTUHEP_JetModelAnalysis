import itertools

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalization(matrix: np.ndarray, method: str) -> np.ndarray:
    methods = {
        "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
        "uniform": lambda x: x / np.maximum(np.sum(x, axis=1, keepdims=True), 1),
        "minmax": lambda x: (x - np.min(x, axis=1, keepdims=True)) / np.maximum(
            np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True), 1
        ),
    }
    return methods.get(method, lambda x: x)(matrix)


class HeatmapObject:
    def __init__(
        self,
        particle_features: pd.DataFrame = None,
        intermediate_outputs: pd.DataFrame = None,
        linear_weights: pd.DataFrame = None,
    ):
        """Template class for heatmap dashboard."""
        self.particle_features = particle_features
        self.intermediate_outputs = intermediate_outputs
        self.linear_weights = linear_weights

    def plot(self, channel, data_index, epoch_index, io_indices) -> go.Figure:
        """Plot the features and intermediate outputs."""

        # Plotly subplots configurations.
        num_row = 0
        num_col = 0
        subplot_titles = []

        # Particle features.
        if self.particle_features is not None:
            features = self.particle_features['feature'].unique()
            num_row += 2
            num_col = max(num_col, len(features))

            # Particle features used for training.
            particle_features: pd.DataFrame = self.particle_features[
                (self.particle_features['channel'] == channel) &
                (self.particle_features['data_index'] == data_index)
            ]

            # Correlation matrices of features.
            X = np.stack([particle_features[particle_features['feature'] == feature]['array'].values[0] for feature in features], axis=1)
            X = np.where(np.isfinite(X), X, 0)
            correlations = []
            for norm_method in ['softmax', 'uniform', 'minmax']:
                X_normalized = normalization(X, method=norm_method)
                corr_matrix = np.dot(X_normalized, X_normalized.T)
                correlations.append((norm_method, corr_matrix))

            subplot_titles.append([f for f in features])
            subplot_titles.append([f"{norm_method.upper()} Corr" for norm_method, _ in correlations])

        # Linear weights and biases.
        if self.linear_weights is not None:
            fields = self.linear_weights['field'].unique()
            num_row += 1
            num_col = max(num_col, len(fields))
            subplot_titles.append([f"W {field}" for field in fields])

        # Intermediate outputs.
        if self.intermediate_outputs is not None:
            intermediate_outputs = self.intermediate_outputs_matrices(channel, data_index, epoch_index, io_indices)
            num_row += len(intermediate_outputs)
            num_col = max(num_col, max(len(outputs) for outputs in intermediate_outputs))
            for row_outputs in intermediate_outputs:
                subplot_titles.append([row_outputs[i][0] for i in range(len(row_outputs))])

        # Create the figure with plotly.
        subplot_titles = [row_titles + [''] * (num_col - len(row_titles)) for row_titles in subplot_titles]
        subplot_titles = list(itertools.chain(*subplot_titles))
        fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=subplot_titles)
        current_row = 0

        if self.particle_features is not None:
            # Add the particle features to the fig.
            current_row += 1
            for i, feature in enumerate(features):
                data = particle_features[particle_features['feature'] == feature]['array'].values[0]
                if ('part_pt' in feature) or ('part_E' in feature):
                    data = self.pad_to_square_matrix(data, pad_method='product')
                    data = normalization(data, method='uniform')
                elif ('eta' in feature) or ('phi' in feature) or ('charge' in feature) or ('dR' in feature):
                    data = self.pad_to_square_matrix(data, pad_method='sum')
                    data = normalization(data, method='softmax')
                elif 'is' in feature:
                    data = self.pad_to_square_matrix(data, pad_method='product')
                    data = normalization(data, method='minmax')
                else:
                    data = self.pad_to_square_matrix(data, pad_method='min_symmetric_pad')
                fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=current_row, col=1 + i)

            # Add the correlation matrices to the fig.
            current_row += 1
            for i, (_, data) in enumerate(correlations):
                fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=current_row, col=1 + i)

        if self.linear_weights is not None:
            # Add the linear weights to the fig.
            current_row += 1
            for i, field in enumerate(fields):
                w = self.linear_weights[
                    (self.linear_weights['field'] == field) &
                    (self.linear_weights['epoch_index'] == epoch_index)
                ]['weights'].values[0]
                fig.add_trace(go.Histogram(x=w, showlegend=False), row=current_row, col=i + 1)

        if self.intermediate_outputs is not None:
            # Add the intermediate outputs to the fig.
            for i, row_outputs in enumerate(intermediate_outputs):
                current_row += 1
                for j, (_, data) in enumerate(row_outputs):
                    showscale = (i == 0) and (j == 0)
                    fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=showscale), row=current_row, col=j + 1)

        # Set the width and height of the figure.
        width = 150 * num_col + 500  # Base width + extra space
        height = 225 * num_row + 100  # Base height + extra space
        fig.update_layout(width=width, height=height)

        # Make the font size smaller.
        font_size = int((width / num_col) / 15)
        fig.update_layout(annotations=[dict(font=dict(size=font_size)) for _ in fig['layout']['annotations']])

        return fig

    def pad_to_square_matrix(self, feature, pad_method: str = None) -> np.ndarray:
        """Convert a 1D feature array into a square matrix based on the specified padding method."""

        # Number of particles.
        num_ptcs = len(feature)

        # Pad the feature array to a square matrix.
        if pad_method == 'min_symmetric_pad':
            # Compute padding value as min_value - (max_value - min_value)
            min_value = np.min(feature)
            max_value = np.max(feature)
            pad_value = 2 * min_value - max_value
            # Create a square matrix filled with the pad value and set the diagonal
            matrix = np.full((num_ptcs, num_ptcs), pad_value)
            np.fill_diagonal(matrix, feature)
        elif pad_method == 'product':
            matrix = np.outer(feature, feature)
        elif pad_method == 'sum':
            matrix = np.add.outer(feature, feature)
        else:
            matrix = np.full((num_ptcs, num_ptcs), 0)
            np.fill_diagonal(matrix, feature)

        return matrix

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
            [
                [(io_1_1, matrix_1_1), (io_1_2, matrix_1_2), ...],
                [(io_2_1, matrix_2_1), (io_2_2, matrix_2_2), ...],
            ]
        """

        pass


class ParticleTransformerHeatmap(HeatmapObject):
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
        app_output = [
            Output("heatmap_grid", "figure"),
            Output("current_channel", "children"),
            Output("current_data", "children"),
            Output("current_epoch", "children"),
            *[Output(f"current_io_{i}", "children") for i in range(len(self.io_buttons))],
            *[Output(f"channel_{i}", "style") for i in range(len(self.channels))],
            *[Output(f"data_{i}", "style") for i in range(self.num_data)],
            *[Output(f"epoch_{i}", "style") for i in range(self.num_epochs)],
            *[Output(f"io_{i}_{j}", "style") for i, row in enumerate(self.io_buttons) for j in range(len(row))],
        ]

        # App callback `Input` (related to clicks).
        app_input = [
            *[Input(f"channel_{i}", "n_clicks") for i in range(len(self.channels))],
            *[Input(f"data_{i}", "n_clicks") for i in range(self.num_data)],
            *[Input(f"epoch_{i}", "n_clicks") for i in range(self.num_epochs)],
            *[Input(f"io_{i}_{j}", "n_clicks") for i, row in enumerate(self.io_buttons) for j in range(len(row))],
        ]

        # App callback `State` (related to the order of *args of the decorated function).
        app_state = [
            State('current_channel', 'children'),
            State('current_data', 'children'),
            State('current_epoch', 'children'),
            *[State(f'current_io_{i}', 'children') for i in range(len(self.io_buttons))],
        ]

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
            styles = self._get_button_styles(channel_idx, data_idx, epoch_idx, io_indices)

            # The return values should be in the order of `app_output`.
            return [fig, channel_idx, data_idx, epoch_idx, *io_indices] + styles

    def _get_button_styles(self, channel_idx, data_idx, epoch_idx, io_indices):
        """Update button colors based on the current indices."""
        styles = [
            *[self.active_style if i == channel_idx else self.default_style for i in range(len(self.channels))],
            *[self.active_style if i == data_idx else self.default_style for i in range(self.num_data)],
            *[self.active_style if i == epoch_idx else self.default_style for i in range(self.num_epochs)],
        ]
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

    heatmap = ParticleTransformerHeatmap(
        particle_features=particle_features,
        intermediate_outputs=intermediate_outputs,
    )
    dashboard = HeatmapDashboard(channels, num_data, num_epochs, heatmap, io_buttons=[[f"Head {i+1}" for i in range(num_heads)]])

    dashboard.run()
