import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template




import numpy as np
import os

grouped_data_path = '/Users/gaurav/UAH/temperature_modelling/data/'


def update_temperature(grid, wind_speed, wind_direction, diffusion_coefficient, damping_factor):
    """
    Update temperature for a single time step.

    Parameters:
    - grid: 2D NumPy array representing the temperature grid
    - wind_speed: Wind speed
    - wind_direction: Wind direction in radians (0 is east, pi/2 is north, pi is west, 3*pi/2 is south)
    - diffusion_coefficient: Diffusion coefficient
    """
    grid_size = grid.shape[0]

    # Compute gradients
    gradient_x, gradient_y = np.gradient(grid)

    # Compute advection term
    advection_term = (1 + wind_speed / 10) * (np.cos(wind_direction)
                                              * gradient_x + np.sin(wind_direction) * gradient_y)

    # Compute diffusion term
    diffusion_term = diffusion_coefficient * \
        (np.gradient(gradient_x)[0] + np.gradient(gradient_y)[1])

    # Update temperature
    updated_grid = grid - damping_factor * (advection_term + diffusion_term)

    return updated_grid


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 1rem",
    "background-color": "#fef5e7",
}


sidebar = html.Div(
    [
        html.H2("Filters"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with filters", className="lead",
        ),
        dbc.Nav(
            [html.Label('Location'),
                dcc.Dropdown(
                id='location-dropdown',
                options=[
                    {'label': 'Madison', 'value': 'Madison'},
                    {'label': 'LasVegas', 'value': 'lasvegas'},
                    # Add more location options as needed
                ],
                value='Madison'  # Set default value
            ),
                html.Br(),
                html.Label('Year'),
                dcc.Dropdown(
                id='year-dropdown',
                options=[
                    {'label': '2022', 'value': 2022},
                    {'label': '2021', 'value': 2021},
                    # Add more year options as needed
                ],
                value=2021  # Set default value
            ),
                html.Br(),
                html.Label('Month'),
                dcc.Dropdown(
                id='month-dropdown',
                options=[
                    {'label': 'June', 'value': '6'},
                    {'label': 'July', 'value': '7'},
                    {'label': 'August', 'value': '8'}
                    # Add more month options as needed
                ],
                value='6'  # Set default value
            ),
                html.Br(),
                html.Label('Hour'),
                dcc.Dropdown(
                id='hour-dropdown',
                options=[
                    {'label': '0', 'value': 0},
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5},
                    {'label': '6', 'value': 6},
                    {'label': '7', 'value': 7},
                    {'label': '8', 'value': 8},
                    {'label': '9', 'value': 9},
                    {'label': '10', 'value': 10},
                    {'label': '11', 'value': 11},
                    {'label': '12', 'value': 12},
                    {'label': '13', 'value': 13},
                    {'label': '14', 'value': 14},
                    {'label': '15', 'value': 15},
                    {'label': '16', 'value': 16},
                    {'label': '17', 'value': 17},
                    {'label': '18', 'value': 18},
                    {'label': '19', 'value': 19},
                    {'label': '20', 'value': 20},
                    {'label': '21', 'value': 21},
                    {'label': '22', 'value': 22},
                    {'label': '23', 'value': 23},

                    # Add more hour options as needed
                ],
                value=0  # Set default value
            ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

# Define the app layout
app.layout = html.Div([

    dbc.Row([
        dbc.Col(),

        dbc.Col(html.H1('Ambient Temperature Modelling'), width=9,
                style={'margin-left': '7px', 'margin-top': '7px','background-color': '#E7F0FE'})
    ]),
    dbc.Row([
        dbc.Col(),
        dbc.Col([
            html.Label('Wind Speed (Mph)'),
            dcc.Slider(
                id='wind-speed-slider',
                min=0,
                max=100,
                step=5,
                value=5,
                # handleLabel="Speed: {value}",
                tooltip={
                        "always_visible": True,
                    # "style": {"color": "LightSteelBlue", "fontSize": "20px"},
                    "placement": "top"},
                marks= None
            ),
        ], width=9, style={'margin-left': '7px', 'margin-top': '30px'}),  # Adjust the width of the column as needed
    ]),

    dbc.Row([
        dbc.Col(),
        dbc.Col([
            html.Label('Wind Direction (in degrees, 0 : North)'),
            dcc.Slider(
                id='wind-direction-slider',
                min=0,
                max=360,
                step=10,
                value=10,
                marks=None,
                tooltip={
                    "always_visible": True,
                    "placement": "top"},
            ),
        ], width=9, style={'margin-left': '7px', 'margin-top': '7px'}),  # Adjust the width of the column as needed
    ]),

    dbc.Row([
            dbc.Col([dbc.Col(sidebar),
                     dbc.Col(dcc.Graph(id='temperature-plot'), width=9, style={
                             'margin-left': '700px', 'margin-top': '75px', 'margin-right': '15px'})
                     ])
            ]),
])

# Define callback to update the temperature plot


@app.callback(
    Output('temperature-plot', 'figure'),
    [Input('location-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('month-dropdown', 'value'),
     Input('hour-dropdown', 'value'),
     Input('wind-speed-slider', 'value'),
     Input('wind-direction-slider', 'value')]
)
def update_plot(location, year, month, hour, wind_speed, wind_direction):
    # Add logic to fetch data based on location, year, month, and hour
    # Assuming hour1 is the fetched data for the selected location, year, month, and hour

    # hour1 = np.load('/Users/gaurav/UAH/temperature_modelling/Analytics/temp_data/hour_1_values.npy')

    img_path = os.path.join(
        grouped_data_path, f'raster_op/{location}/{year}/numpy_images/{month}/temperature_raster_{hour}.npy')
    print(img_path)

    print(f"Location: {location}, Year: {year}, Month: {month}, Hour: {hour}")
    print(f"Temperature data path: {img_path}")

    try:
        # Load temperature data
        hour1 = np.load(img_path)
        # print(hour1.sum())
        diffusion_coefficient = 0.1
        damping_factor = 0.3
        wind_direction_rad = np.deg2rad(wind_direction)
        updated_temperature = update_temperature(
            hour1, wind_speed, wind_direction_rad, diffusion_coefficient, damping_factor)

        # print(updated_temperature.sum())

        fig = px.imshow(updated_temperature, color_continuous_scale='plasma',)
        fig.update_layout(title='Temperature Distribution',
                          width=900, height=1000)
        fig['layout']['uirevision'] = '123'
        return fig

    except Exception as e:
        print(f"Error loading temperature data: {e}")


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')


