import os
import pandas as pd
import numpy as np

import plotly.express as pe
import plotly.offline as pyo
from datetime import datetime


# os.chdir('/Users/gaurav/UAH/temperature_modelling/')
# pyo.init_notebook_mode()

HOME_PATH = os.path.join(os.path.dirname(__file__),'..')


def resample_daily(df):
    """Resamples the dataframe to daily values : Previous animation plot was hourly, need to do it daily now"""

    df = df.sort_values(by=["station", "beg_time"])
    df.index = [df.beg_time, df.station]
    df = df.groupby(
        [pd.Grouper(level="station"), pd.Grouper(level="beg_time", freq="D")]
    ).mean()

    df = df.reset_index()
    df["beg_time"] = pd.to_datetime(df.beg_time)
    df.sort_values(by=["station", "beg_time", "day_of_year"], inplace=True)

    df = df[
        ["station", "beg_time", "temperature",
            "day_of_year", "latitude", "longitude"]
    ]
    df["temperature"] = df.temperature.round(2)

    return df


def combine_dataframes(dframes, source):
    '''Combines the dataframes from different sources into a single dataframe'''

    if source == 'netatmos':
        df = pd.concat(dframes, ignore_index=True)
        df = df.reset_index(drop=True).iloc[:, 1:]
        df.drop_duplicates(
            subset=['key', 'expire_time_gmt', 'valid_time_gmt', 'temp', 'dewPt'])

    if source == 'wunderground':
        df = pd.concat(dframes, ignore_index=True)
        # df = df.reset_index(drop=True).iloc[:,1:]
        df.drop_duplicates(
            subset=['stationID', 'obsTimeLocal', 'temperature', 'epoch'], inplace=True)
        df = df.sort_values(by=['stationID', 'obsTimeLocal'])
        df = df.reset_index(drop=True)
    return df

# Todo: add condition for netatmos


def process_downloaded_data(location_name, source, item_identifier, year):
    '''Processes the downloaded data into a master csv file
    Usage : process_downloaded_data('Madison','wunderground','pws')
    '''

    input_path = os.path.join(HOME_PATH, 'data/raw_data/', location_name, item_identifier+'_data_'+location_name.lower(), year)
    print(f'Searching for files in {input_path}')
    output_path = os.path.join(
        HOME_PATH, 'data/processed_data',location_name+'_'+year)
    if not os.path.exists(output_path):
        print(output_path)
        os.makedirs(output_path)

    # print(input_path)
    # valid_files = [x for x in os.listdir(input_path) if '.csv' in x]
    valid_files = [input_path+'/' +
                   x for x in os.listdir(input_path) if '.csv' in x]
    master_df = combine_dataframes([pd.read_csv(x)
                                   for x in valid_files], source=source)
    filename = 'master_'+location_name+'_'+item_identifier + '_' + '.csv'
    master_df.to_csv(os.path.join(output_path, filename))
    print('Saving master file in : ', output_path)

    return True


def aggregation(df):
    # aggregation of metrics on hourly basis
    df['beg_time'] = pd.to_datetime(df.beg_time)
    df.index = [df.beg_time, df.station]
    df = df.groupby([pd.Grouper(level='station'), pd.Grouper(
        level='beg_time', freq='H')]).mean()

    # resetting the index for daily animations
    # dfx = df.copy()        #Avoiding this to save memory
    df = df.reset_index()
    df['beg_time'] = pd.to_datetime(df.beg_time)
    df['day_of_year'] = df['beg_time'].apply(
        lambda x: datetime.timetuple(x).tm_yday)

    return df


def process_netatmos(df):
    df = df.drop([df.columns[0]], axis=1)
    df = df.groupby("latitude", group_keys=True).apply(lambda x: x)

    # todo : Find a better way to do this ==> df.map()
    df_cpy = df.copy()
    df_cpy['station'] = df_cpy['latitude']
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_1' if x == 43.06395 else x)
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_2' if x == 43.0704332 else x)
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_3' if x == 43.073576 else x)
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_4' if x == 43.0924018 else x)
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_5' if x == 43.0958646 else x)
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_6' if x == 43.1107383 else x)
    df_cpy['station'] = df_cpy['station'].apply(
        lambda x: 'Station_7' if x == 43.127702907735 else x)

    df_cpy = aggregation(df_cpy)

    return df_cpy


def process_lter(lter_combined):
    '''Usage : 
        data1  = pd.read_csv('data/Madison/WSC_Station_locations.csv')
        data2  = pd.read_csv('data/Madison/WSC_TempC2021_UPDATED.csv')

        data2 = data2.merge(data1,on='SID')
        dp.process_lter(data2)

    '''
    lter_combined.columns = [x.lower() for x in lter_combined.columns]
    lter_combined['sample_time'] = lter_combined['sample_time'].apply(
        lambda x: str(x).zfill(4))
    lter_combined['sample_time'] = lter_combined['sample_time'].apply(
        lambda x: str(x)[:2]+':'+str(x)[2:]+':00')
    lter_combined['beg_time'] = lter_combined.sample_date.str.cat(
        lter_combined.sample_time, sep=' ')

    lter_combined['beg_time'] = pd.to_datetime(lter_combined.beg_time)
    lter_combined.index = [lter_combined.beg_time, lter_combined.sid]
    lter_combined = lter_combined.groupby(
        [pd.Grouper(level='sid'), pd.Grouper(level='beg_time', freq='H')]).mean()

    lter_combined = lter_combined.reset_index()
    lter_combined['beg_time'] = pd.to_datetime(lter_combined.beg_time)
    lter_combined['day_of_year'] = lter_combined['beg_time'].apply(
        lambda x: datetime.timetuple(x).tm_yday)
    lter_combined['value'] = (lter_combined['value']-32)*(5/9)

    lter_combined.columns = ['station', 'beg_time',
                             'temperature', 'latitude', 'longitude', 'day_of_year']

    return lter_combined


def process_wunder(pws_df):
    pws_df['beg_time'] = pd.to_datetime(pws_df.obsTimeLocal)
    pws_df = pws_df.rename(
        {"stationID": "station", "lat": "latitude", "lon": "longitude"}, axis=1)

    pws_df = pws_df.sort_values(by=['station', 'beg_time'])
    pws_df.index = [pws_df.beg_time, pws_df.station]
    pws_df = pws_df.groupby(
        [pd.Grouper(level='station'), pd.Grouper(level='beg_time', freq='H')]).mean()

    # resetting the index for daily animations
    # pws_dfx = pws_df.copy()        #Avoiding this to save memory
    pws_df = pws_df.reset_index()
    pws_df['beg_time'] = pd.to_datetime(pws_df.beg_time)
    pws_df['day_of_year'] = pws_df['beg_time'].apply(
        lambda x: datetime.timetuple(x).tm_yday)

    pws_df.sort_values(by=['station', 'beg_time', 'day_of_year'], inplace=True)

    return pws_df


def plot_(dfx, animation_frame_comp = 'day_of_year', frame_duration = 200, station_name=None,resample=True):
    '''
    dfx                     : Source dataframe
    animation_frame_comp    : defines which column to animate upon (eg. year, month)
    frame_duration          : defines how fast the animation should be
    station_name            : Optional argument to plot a single station

    Usage                   : dp.plot_(df_vegas,'day_of_year',180)
    '''
    # fig = pe.scatter_mapbox(dfx,lat='latitude',lon = 'longitude',animation_frame='day_of_year',color = 'temperature',range_color=[-40,40],hover_data=['station'],height = 610)
    # Making the legend dynamic

    if resample:
        dfx = resample_daily(dfx)

    if station_name:
        dfx = dfx[dfx.station == station_name]

    dfx = dfx.sort_values(by=[animation_frame_comp])
    fig = pe.scatter_mapbox(
        dfx, lat='latitude',
        lon='longitude',
        animation_frame=animation_frame_comp,
        color='temperature',
        hover_data=['station'],
        height=710,
        color_continuous_scale='thermal',
        # width=200
    )

    fig.update_layout(mapbox_style='open-street-map',)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker=dict(size=14, color='black'))
    # fig.update_layouta(updatemenus=dict())

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_duration
    # fig.layout.coloraxis.colorbar.title.text = 'Temperature - °C'

    # fig.show()
    return fig
