# # Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

import satellite as sat
import dataprocess as dp
import numpy as np

# Incorporate data
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
result_df = sat.station_daily_lst_anomaly_means()
stations_ = result_df.station.unique()
# result_df_cp = result_df[result_df.station.isin(random_stations)]

urban_data = pd.read_csv('data/raster_op/urban_surface_properties_values.csv')
urban_data = urban_data[['station','valueImperviousfraction','valueTreefraction',
       'valueBuildingheight', 'valueNearestDistWater', 'valueWaterfraction',
       'valueLandcover', 'valueBuildingfraction']].rename(columns={'valueImperviousfraction':'Imperv_Frac','valueTreefraction':'TreeFraction','valueBuildingheight':'BuildingHeight','valueNearestDistWater':'DistancetoWater','valueWaterfraction':'WaterFrac','valueLandcover':'LandCover','valueBuildingfraction':'BuildingFrac'})
urban_data = urban_data.round(4)

#Plotting the animation
clean_data = pd.read_csv('data/processed_data/Madison_2021/clean_Madison_pws_.csv')
clean_data['beg_time'] = pd.to_datetime(clean_data['beg_time'])
animation_plot = dp.plot_(clean_data,'day_of_year',200)

# # Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# # Dropdown options
dropdown_options = list(range(int(np.ceil(len(result_df.station.unique())/12))))
# print(dropdown_options)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('Temperature Modelling', className="text-primary text-center fs-3")
    ]),

    dbc.Row([
        dcc.Dropdown(options=dropdown_options, value= dropdown_options[0], id='dropdown_final', clearable=False,style={'width': '200px'})
    ]),

    dbc.Row([

        dbc.Col([
            dbc.Row([dcc.Graph(figure={}, id='line_plot')]),
            dcc.Dropdown(options=sorted(stations_), value= sorted(stations_)[0], id='dropdown_urban_data', multi=True,clearable=False,style={'width': '800px'}),
            dbc.Row([dash_table.DataTable(data=urban_data.to_dict('records'), id='vegetation_table')])
        ],width = 6),
        # dbc.Col([
        #     dcc.Graph(figure={}, id='line_plot')
        # ], width=6),

        dbc.Col([
           dcc.Graph(figure={}, id='animation_graph')
            ], width=6),
    ]),


], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='line_plot', component_property='figure'),
    Input(component_id='dropdown_final', component_property='value')
)
def update_graph(col_chosen):
    # fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
    station_slice = stations_[int(col_chosen)*12:(int(col_chosen)+1)*12]
    fig = px.line(result_df[result_df.station.isin(station_slice)], x="hour", y="adjusted_lst", color='station', title='Adjusted LST for 12 random stations')
    return fig

@callback(
    Output(component_id='animation_graph', component_property='figure'),
    Input(component_id='dropdown_final', component_property='value')
)
def update_graph_2(col_chosen):
    # fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
    station_slice = stations_[int(col_chosen)*12:(int(col_chosen)+1)*12]
    return dp.plot_(clean_data[clean_data.station.isin(station_slice)],'day_of_year',200)

@callback(
    Output(component_id='vegetation_table', component_property='data'),
    Input(component_id='dropdown_urban_data', component_property='value')
)
def update_vegetation_table(column_chosen):
    valid_stat = column_chosen
    data_slice = urban_data[urban_data.station.isin(list(column_chosen))]
    data_slice = data_slice.to_dict('records')
    return data_slice

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host = '127.0.0.1')

