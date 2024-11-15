import itertools

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_dash_app(
    channels: list[str],
    num_data: int,
    num_epochs: int,
    num_blocks: int,
    particle_features: dict[tuple[str, int], list[tuple[str, np.ndarray]]],
    intermediate_outputs: dict[tuple[str, int, int, int], np.ndarray],
    figsize: tuple[int, int],
) -> dash.Dash:
    """Create a Dash app for visualizing the heatmap data.

        Args:
            channels : list[str]
                List of channel names.
            num_data : int
                Number of data points.
            num_epochs : int
                Number of epochs.
            num_blocks : int
                Number of blocks.
            particle_features : dict[tuple[str, int], list[tuple[str, np.ndarray]]
                Dictionary containing the particle features for each channel.
                The key is a tuple of (channel, data_index), and the value is a list of
                tuples with shape (feature, 2D numpy array of shape (num_ptcs, num_ptcs)).
            intermediate_outputs : dict[tuple[str, int, int, int], np.ndarray]
                Dictionary containing the heatmap data for each channel.
                The key is a tuple of (channel, data_index, epoch_index, block_index),
                and the value is a 3D numpy array of shape (num_heads, num_ptcs, num_ptcs).
            figsize : tuple[int, int]
                Figure size in pixels.

        Returns:
            dash.Dash
                Dash app instance.
    """

    # Create the Dash app
    app = dash.Dash(__name__)

    # Layout of the dash app
    app.layout = html.Div([

        # Left side columns in a single row
        html.Div([
            # First column for channel
            html.Div([
                html.Button(channel, id=f'channel_{i}', n_clicks=0, style={'background-color': 'lightgray'})
                for i, channel in enumerate(channels)
            ], style={'display': 'flex', 'flex-direction': 'column', 'margin-right': '20px'}),

            # Second column for data index
            html.Div([
                html.Button(f"Data {i+1}", id=f'data_{i}', n_clicks=0, style={'background-color': 'lightgray'})
                for i in range(num_data)
            ], style={'display': 'flex', 'flex-direction': 'column', 'margin-right': '20px'}),

            # Third column for epoch
            html.Div([
                html.Button(f"Epoch {i+1}", id=f'epoch_{i}', n_clicks=0, style={'background-color': 'lightgray'})
                for i in range(num_epochs)
            ], style={'display': 'flex', 'flex-direction': 'column'})
        ], style={'display': 'flex', 'gap': '10px', 'margin-right': '20px'}),

        # "Block" buttons and heatmap grid in another section
        html.Div([
            html.Div([
                html.Button(f'Block {i+1}', id=f'block_{i}', n_clicks=0, style={'background-color': 'lightgray'})
                for i in range(num_blocks)
            ], style={'display': 'flex', 'gap': '10px', 'margin-bottom': '20px'}),

            # Previous and Next buttons for block control
            html.Div([
                html.Button('Previous Block', id='prev_button', n_clicks=0),
                html.Button('Next Block', id='next_button', n_clicks=0),
            ], style={'display': 'flex', 'gap': '10px', 'margin-top': '10px'}),

            # Hidden divs to store the current selections
            html.Div(id='current_block', style={'display': 'none'}),
            html.Div(id='current_channel', style={'display': 'none'}),
            html.Div(id='current_data', style={'display': 'none'}),
            html.Div(id='current_epoch', style={'display': 'none'}),

            # Heatmap grid
            dcc.Graph(id='heatmap_grid', config={'displayModeBar': False}),
        ])
    ], style={'display': 'flex', 'align-items': 'flex-start'})

    @app.callback(  # Callback to update heatmap grid, block switching, and button colors
        # Outputs (related to the order of return values)
        [
            Output('heatmap_grid', 'figure'),
            Output('current_block', 'children'),
            Output('current_channel', 'children'),
            Output('current_data', 'children'),
            Output('current_epoch', 'children')
        ] +
        [Output(f'channel_{i}', 'style') for i in range(len(channels))] +
        [Output(f'data_{i}', 'style') for i in range(num_data)] +
        [Output(f'epoch_{i}', 'style') for i in range(num_epochs)] +
        [Output(f'block_{i}', 'style') for i in range(num_blocks)],

        # Inputs (related to the order of *args)
        [Input(f'channel_{i}', 'n_clicks') for i in range(len(channels))] +
        [Input(f'data_{i}', 'n_clicks') for i in range(num_data)] +
        [Input(f'epoch_{i}', 'n_clicks') for i in range(num_epochs)] +
        [Input(f'block_{i}', 'n_clicks') for i in range(num_blocks)] +
        [Input('prev_button', 'n_clicks'), Input('next_button', 'n_clicks')],

        # States (related to the order of *args)
        [
            State('current_channel', 'children'),
            State('current_data', 'children'),
            State('current_epoch', 'children'),
            State('current_block', 'children'),
        ]
    )
    def update_heatmap_grid(*args):
        """Update the heatmap grid based on the selected channel, data, epoch, and block."""

        # Default button styles
        default_style = {'background-color': 'lightgray'}
        active_style = {'background-color': 'lightblue'}

        # Retrieve current selections or initialize them
        channel_index = int(args[-4]) if args[-4] is not None else 0
        data_index = int(args[-3]) if args[-3] is not None else 0
        epoch_index = int(args[-2]) if args[-2] is not None else 0
        block_index = int(args[-1]) if args[-1] is not None else 0

        # Identify which button was clicked
        ctx = dash.callback_context
        button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]

        # Update selections based on button clicks
        if button_clicked.startswith('channel_'):
            channel_index = int(button_clicked.split('_')[1])
        elif button_clicked.startswith('data_'):
            data_index = int(button_clicked.split('_')[1])
        elif button_clicked.startswith('epoch_'):
            epoch_index = int(button_clicked.split('_')[1])
        elif button_clicked.startswith('block_'):
            block_index = int(button_clicked.split('_')[1])

        # Block navigation
        if button_clicked == 'prev_button':
            block_index = max(0, block_index - 1)
        elif button_clicked == 'next_button':
            block_index = min(num_blocks - 1, block_index + 1)

        # Title reflecting the selections
        main_title = (f"Channel {channels[channel_index]} - "
                      f"Data {data_index + 1} - "
                      f"Epoch {epoch_index + 1} - "
                      f"Block {block_index + 1}")

        # Get current data
        channel = channels[channel_index]
        current_particle_features: list[tuple[str, np.ndarray]] = particle_features[(channel, data_index)]
        current_intermediate_outputs: np.ndarray = intermediate_outputs[(channel, data_index, epoch_index, block_index)]

        # Create a grid of heatmaps for the selected block
        num_row = 2
        num_col = max(len(current_particle_features), len(current_intermediate_outputs))
        fig = make_subplots(
            rows=num_row,
            cols=num_col,
            # specs=[
            #     [{"colspan": 3}, None, None, None, None],
            #     [{}, {}, {}, {}, {}]
            # ],
            # subplot_titles=subplot_titles
        )

        # The first row contains the particle features
        for i, (feature, data) in enumerate(current_particle_features):
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    # coloraxis="coloraxis",
                    text=data,
                    hoverinfo="text",
                    colorscale="Viridis",
                    # colorbar=dict(title=feature),  # Add a title for the color bar
                    showscale=False,
                ),
                row=1, col=i+1
            )

        # The second row contains the intermediate output heatmaps
        for i, data in enumerate(current_intermediate_outputs):
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    # coloraxis="coloraxis",
                    text=data,
                    hoverinfo="text",
                    colorscale="Viridis",
                    # colorbar=dict(title=f"Head {i + 1}"),  # Add a title for each color bar
                    showscale=False,
                ),
                row=2, col=i+1
            )

        # Layout update
        fig.update_layout(
            title=main_title,
            title_x=0.5,
            coloraxis={"colorscale": "Viridis"},
            showlegend=False,
            width=figsize[0],
            height=figsize[1],
            margin=dict(t=100, b=50, l=50, r=50)
        )

        # Button styles
        styles = []
        for i in range(len(channels)):  # Channel styles
            styles.append(active_style if i == channel_index else default_style)
        for i in range(num_data):  # Data styles
            styles.append(active_style if i == data_index else default_style)
        for i in range(num_epochs):  # Epoch styles
            styles.append(active_style if i == epoch_index else default_style)
        for i in range(num_blocks):  # Block styles
            styles.append(active_style if i == block_index else default_style)

        # Return the updated figure, selections, and styles
        return [fig, block_index, channel_index, data_index, epoch_index] + styles

    return app


# Run the app
if __name__ == '__main__':
    channels = ['QCD', 'TOP', 'Higgs']
    num_data = 5
    num_epochs = 3
    num_blocks = 8
    num_heads = 8

    particle_features = {}
    intermediate_outputs = {}

    for channel, data_index in itertools.product(channels, range(num_data)):

        num_ptcs = np.random.randint(30) + 1

        particle_features[(channel, data_index)] = [
            ('pt', np.random.rand(num_ptcs, num_ptcs)),
            ('eta', np.random.rand(num_ptcs, num_ptcs)),
            ('phi', np.random.rand(num_ptcs, num_ptcs)),
        ]

        for epoch_index, block_index in itertools.product(range(num_epochs), range(num_blocks)):
            intermediate_outputs[(channel, data_index, epoch_index, block_index)] = np.random.rand(num_heads, num_ptcs, num_ptcs)

    app: dash.Dash = create_dash_app(
        channels=['QCD', 'TOP', 'Higgs'],
        num_data=num_data,
        num_epochs=num_epochs,
        num_blocks=num_blocks,
        particle_features=particle_features,
        intermediate_outputs=intermediate_outputs,
        figsize=(1100, 800),
    )

    app.run_server(debug=True)
