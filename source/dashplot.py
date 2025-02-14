import itertools

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HeatmapDashboard:
    def __init__(self, channels, num_data, num_epochs, num_blocks, particle_features, intermediate_outputs, figsize):

        self.channels = channels
        self.num_data = num_data
        self.num_epochs = num_epochs
        self.num_blocks = num_blocks
        self.particle_features = particle_features
        self.intermediate_outputs = intermediate_outputs
        self.figsize = figsize

        self.default_style = {'background-color': 'lightgray'}
        self.active_style = {'background-color': 'lightblue'}

        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def run(self):
        self.app.run_server(debug=True)

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                # Left-hand side buttons (Channels, Datas, Epochs).
                self.create_button_group('channel', self.channels, 'column'),
                self.create_button_group('data', [f"Data {i+1}" for i in range(self.num_data)], 'column'),
                self.create_button_group('epoch', [f"Epoch {i+1}" for i in range(self.num_epochs)], 'column'),
            ], style={'display': 'flex', 'gap': '5px', 'margin-right': '10px'}),

            html.Div([
                # First row buttons (Blocks).
                self.create_button_group('block', [f"Block {i+1}" for i in range(self.num_blocks)], 'row'),

                # Second row buttons (Previous, Next).
                html.Div([
                    html.Button('Previous Block', id='prev_button', n_clicks=0),
                    html.Button('Next Block', id='next_button', n_clicks=0),
                ], style={'display': 'flex', 'gap': '10px', 'margin-top': '10px'}),

                # Hidden divs to store current indices.
                html.Div(id='current_block', style={'display': 'none'}),
                html.Div(id='current_channel', style={'display': 'none'}),
                html.Div(id='current_data', style={'display': 'none'}),
                html.Div(id='current_epoch', style={'display': 'none'}),

                # Heatmap graph.
                dcc.Graph(id='heatmap_grid', config={'displayModeBar': False}),
            ])
        ], style={'display': 'flex', 'align-items': 'flex-start'})

    def create_button_group(self, prefix, labels, direction):
        # `direction` is either 'column' or 'row'.
        return html.Div([
            html.Button(label, id=f'{prefix}_{i}', n_clicks=0, style=self.default_style)
            for i, label in enumerate(labels)
        ], style={'display': 'flex', 'gap': '5px', 'flex-direction': direction, 'margin-right': '10px'})

    def setup_callbacks(self):
        # Callback to update heatmap grid, block switching, and button colors.
        @self.app.callback(
            # Outputs (related to the order of return values)
            [Output('heatmap_grid', 'figure'),
             Output('current_block', 'children'),
             Output('current_channel', 'children'),
             Output('current_data', 'children'),
             Output('current_epoch', 'children')] +
            [Output(f'channel_{i}', 'style') for i in range(len(self.channels))] +
            [Output(f'data_{i}', 'style') for i in range(self.num_data)] +
            [Output(f'epoch_{i}', 'style') for i in range(self.num_epochs)] +
            [Output(f'block_{i}', 'style') for i in range(self.num_blocks)],

            # Inputs (related to the order of *args).
            [Input(f'channel_{i}', 'n_clicks') for i in range(len(self.channels))] +
            [Input(f'data_{i}', 'n_clicks') for i in range(self.num_data)] +
            [Input(f'epoch_{i}', 'n_clicks') for i in range(self.num_epochs)] +
            [Input(f'block_{i}', 'n_clicks') for i in range(self.num_blocks)] +
            [Input('prev_button', 'n_clicks'), Input('next_button', 'n_clicks')],

            # States (related to the order of *args below).
            [State('current_channel', 'children'),
             State('current_data', 'children'),
             State('current_epoch', 'children'),
             State('current_block', 'children')]
        )
        def update_heatmap(*args):
            """Called when a button is clicked to update the heatmap grid."""

            channel_idx = int(args[-4]) if args[-4] is not None else 0
            data_idx = int(args[-3]) if args[-3] is not None else 0
            epoch_idx = int(args[-2]) if args[-2] is not None else 0
            block_idx = int(args[-1]) if args[-1] is not None else 0

            ctx = dash.callback_context
            button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_clicked.startswith('channel_'):
                channel_idx = int(button_clicked.split('_')[1])
            elif button_clicked.startswith('data_'):
                data_idx = int(button_clicked.split('_')[1])
            elif button_clicked.startswith('epoch_'):
                epoch_idx = int(button_clicked.split('_')[1])
            elif button_clicked.startswith('block_'):
                block_idx = int(button_clicked.split('_')[1])

            if button_clicked == 'prev_button':
                block_idx = max(0, block_idx - 1)
            elif button_clicked == 'next_button':
                block_idx = min(self.num_blocks - 1, block_idx + 1)

            channel = self.channels[channel_idx]

            title = f"Channel {channel} - Data {data_idx + 1} - Epoch {epoch_idx + 1} - Block {block_idx + 1}"

            # Particle features.
            current_features = self.particle_features[
                (self.particle_features['channel'] == channel) &
                (self.particle_features['data_index'] == data_idx)
            ]

            # Intermediate outputs.
            current_outputs = self.intermediate_outputs[
                (self.intermediate_outputs['channel'] == channel) &
                (self.intermediate_outputs['data_index'] == data_idx) &
                (self.intermediate_outputs['epoch_index'] == epoch_idx) &
                (self.intermediate_outputs['block_index'] == block_idx)
            ]

            fig = self.create_heatmap_grid(current_features, current_outputs, title)

            styles = self.generate_button_styles(channel_idx, data_idx, epoch_idx, block_idx)

            return [fig, block_idx, channel_idx, data_idx, epoch_idx] + styles

    def create_heatmap_grid(self, particle_features: pd.DataFrame, intermediate_outputs: pd.DataFrame, title: str):
        """Create a heatmap grid with particle features and intermediate outputs."""

        features = particle_features['feature'].unique()
        head_outputs = intermediate_outputs['output'].values[0]

        num_row = 2
        num_col = max(len(features), len(head_outputs))

        subplot_titles = [f for f in features] + [''] * (num_col - len(features)) + [f"Head {i + 1}" for i in range(len(head_outputs))]

        fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=subplot_titles)

        for i, feature in enumerate(features):
            data = particle_features[particle_features['feature'] == feature]['array'].values[0]
            data = self.square_feature(data)
            fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=1, col=i + 1)

        for i, data in enumerate(head_outputs):
            fig.add_trace(go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=2, col=i + 1)

        fig.update_layout(title=title, title_x=0.5, width=self.figsize[0], height=self.figsize[1])

        return fig

    def square_feature(self, feature):
        """Make 1D feature array into a square matrix."""

        num_ptcs = len(feature)
        min_value = min(feature)
        max_value = max(feature)

        # Pad non-diagonal elements with the min_value - (max_value - min_value).
        pad_value = 2 * min_value - max_value
        diagonal = np.full((num_ptcs, num_ptcs), pad_value)
        np.fill_diagonal(diagonal, feature)

        return diagonal

    def generate_button_styles(self, channel_idx, data_idx, epoch_idx, block_idx):
        """Update button colors based on the current indices."""

        styles = []
        styles += [self.active_style if i == channel_idx else self.default_style for i in range(len(self.channels))]
        styles += [self.active_style if i == data_idx else self.default_style for i in range(self.num_data)]
        styles += [self.active_style if i == epoch_idx else self.default_style for i in range(self.num_epochs)]
        styles += [self.active_style if i == block_idx else self.default_style for i in range(self.num_blocks)]
        return styles

    def run(self):
        self.app.run_server(debug=True)


def generate_sample_data(channels, num_data, num_epochs, num_blocks, num_heads):
    particle_features = []
    intermediate_outputs = []

    for channel, data_idx in itertools.product(channels, range(num_data)):
        num_ptcs = np.random.randint(5, 20)

        for feature in ['pt', 'eta', 'phi']:
            particle_features.append({'channel': channel, 'data_index': data_idx, 'feature': feature, 'array': np.random.rand(num_ptcs, num_ptcs)})

        for epoch_idx, block_idx in itertools.product(range(num_epochs), range(num_blocks)):
            intermediate_outputs.append({'channel': channel, 'data_index': data_idx, 'epoch_index': epoch_idx, 'block_index': block_idx, 'output': np.random.rand(num_heads, num_ptcs, num_ptcs)})

    return pd.DataFrame(particle_features), pd.DataFrame(intermediate_outputs)


if __name__ == '__main__':
    channels = ['QCD', 'TOP', 'Higgs']
    num_data, num_epochs, num_blocks, num_heads = 5, 3, 8, 4

    particle_features, intermediate_outputs = generate_sample_data(channels, num_data, num_epochs, num_blocks, num_heads)

    dashboard = HeatmapDashboard(channels, num_data, num_epochs, num_blocks, particle_features, intermediate_outputs, figsize=(1000, 625))
    dashboard.run()
