"""
Created on : Nov 18, 2023

@author: Gaurav
@version: 1.0

Functions to be used in 00 Predictor.ipynb
"""

import os
import time
import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt


from shapely.geometry import Point
from pyproj import CRS

from lib import helper as helper
from lib import visualizer


MODEL_PATH = '/Users/gaurav/UAH/temperature_modelling/Resources/trained_models/'
PREDICTION_DIR = '/Users/gaurav/UAH/temperature_modelling/Resources/predictions/Madison_2021'

TEST_ECOSTRSS_FILE = '/Users/gaurav/UAH/temperature_modelling/data/raster_op/ECOSTRESS_values_testing.csv'
TEST_URBAN_FILE = '/Users/gaurav/UAH/temperature_modelling/data/raster_op/urban_surface_properties_values_testing.csv'

time_adjusted_df = pd.read_csv(
    '/Users/gaurav/UAH/temperature_modelling/Analytics/time_adjusted_df.csv')

# to be used in add_temp_data() function
prediction_timeframe = ['2021-06-01 00:00:00', '2021-06-31 00:00:00']


def get_station_ranges(len_ec_data):
    ''' This function returns the station ranges for sampling of data

        args: len_ec_data: length of the ec_data : len(ec_data.station.unique())
    '''

    stations_ranges = []
    max_range = len_ec_data//30000
    start_range = 1
    end_range = 30001
    for i in range(max_range):
        station_range = np.arange(start_range, end_range)
        start_range = end_range
        end_range = end_range+30000
        stations_ranges.append(station_range)

    # this list to be used for sampling of data
    stations_ranges.append(np.arange(start_range, len_ec_data+1))

    return stations_ranges


def get_ec_urban(ec_data=None,urb_data=None,hour_filter=None):
    ''' This function returns the ECOSTRESS and Urban Surface data for the given hour

    Parameters : [1] or None
    '''

    if ec_data is None:
        ec_data = pd.read_csv(TEST_ECOSTRSS_FILE)

    if urb_data is None:
        urb_data = pd.read_csv(TEST_URBAN_FILE)


    ec_data = ec_data.query(
        f'hour == {list([hour_filter or list(np.arange(0,24,1))][0])}')

    ec_data = gpd.GeoDataFrame(ec_data)
    ec_data['geometry'] = [Point(xy) for xy in zip(
        ec_data['longitude'], ec_data['latitude'])]
    ec_data.crs = CRS.from_epsg(6879).to_wkt()
    ec_data = ec_data.to_crs(epsg=4326)
    ec_data['latitude'] = ec_data.geometry.y
    ec_data['longitude'] = ec_data.geometry.x

    urb_data = gpd.GeoDataFrame(urb_data)
    urb_data['geometry'] = [Point(xy) for xy in zip(
        urb_data['longitude'], urb_data['latitude'])]
    urb_data.crs = CRS.from_epsg(6879).to_wkt()
    urb_data = urb_data.to_crs(epsg=4326)
    urb_data['latitude'] = urb_data.geometry.y
    urb_data['longitude'] = urb_data.geometry.x

    # #Making sure of the precision for proper joins
    ec_data['latitude'] = np.round(ec_data['latitude'], 7)
    ec_data['longitude'] = np.round(ec_data['longitude'], 7)

    urb_data['latitude'] = np.round(urb_data['latitude'], 7)
    urb_data['longitude'] = np.round(urb_data['longitude'], 7)

    ec_data = ec_data[['station', 'latitude',
                       'longitude', 'hour', 'value_LST']]
    urb_data = urb_data[['station', 'latitude', 'longitude', 'valueImperviousfraction',
                         'valueTreefraction', 'valueBuildingheight', 'valueNearestDistWater', 'valueWaterfraction', 'valueBuildingfraction']]

    station_ranges = get_station_ranges(len(ec_data.station.unique()))

    return ec_data, urb_data, station_ranges


def add_temp_data(ec_data_segment,hour_selection=None):
    '''
    This function adds the temperature data to the dataframe

    '''


    time_adjusted_df_eco_merge = time_adjusted_df[[
        'station', 'latitude', 'longitude']].drop_duplicates()
    
    # time_adjusted_df_eco_merge = time_adjusted_df[(
    #     time_adjusted_df.beg_time > prediction_timeframe[0]) & (time_adjusted_df.beg_time < prediction_timeframe[1])]

    # time_adjusted_df_eco_merge = time_adjusted_df_slice[[
    #     'station', 'latitude', 'longitude']].drop_duplicates()

    #No need to calculate this everytime
    closest_stations = helper.find_closest_station(time_adjusted_df_eco_merge, ec_data_segment)

    

    closest_stations_1 = closest_stations[[
        'station', 'closest_station_1', 'closest_1_distance', 'latitude', 'longitude']]
    closest_stations_2 = closest_stations[[
        'station', 'closest_station_2', 'closest_2_distance', 'latitude', 'longitude']]
    # closest_stations_3 = closest_stations[['station', 'closest_station_3', 'closest_3_distance','latitude','longitude']]

    x1 = pd.merge(closest_stations_1, time_adjusted_df_eco_merge,
                  left_on='closest_station_1', right_on='station', how='left')
    x1 = x1.rename({'station_x': 'station', 'temperature': 'closest_station_1_temp'}, axis=1).drop(
        ['station_y'], axis=1)
    x1 = x1.fillna(method='ffill')
    if hour_selection is not None:
        x1['hour'] = hour_selection[0]
    else:
        x1['hour'] = pd.to_datetime(x1['beg_time']).dt.hour

    x1 = x1[['station', 'hour', 'latitude', 'longitude',
             'closest_station_1_temp', 'closest_1_distance', 'beg_time']]
    x1['day_of_year'] = pd.to_datetime(x1['beg_time']).dt.dayofyear

    x2 = pd.merge(closest_stations_2, time_adjusted_df_eco_merge,
                  left_on='closest_station_2', right_on='station', how='left')
    x2 = x2.rename({'station_x': 'station', 'temperature': 'closest_station_2_temp'}, axis=1).drop(
        ['station_y'], axis=1)
    x2 = x2.fillna(method='ffill')
    x2['hour'] = pd.to_datetime(x2['beg_time']).dt.hour
    x2 = x2[['station', 'hour', 'latitude', 'longitude',
             'closest_station_2_temp', 'closest_2_distance', 'beg_time']]
    x2['day_of_year'] = pd.to_datetime(x2['beg_time']).dt.dayofyear

    x1 = x1[['station', 'hour', 'latitude', 'longitude', 'closest_station_1_temp',
             'closest_1_distance', 'beg_time']].drop_duplicates()
    x2 = x2[['station', 'hour', 'latitude', 'longitude', 'closest_station_2_temp',
             'closest_2_distance', 'beg_time']].drop_duplicates()

    final = x1.merge(x2, on=['station', 'hour', 'latitude',
                     'longitude', 'beg_time'], how='left')
    final['closest_station_1_temp'] = final[[
        'closest_station_1_temp', 'closest_station_2_temp']].mean(axis=1)
    final['closest_1_distance'] = final[[
        'closest_1_distance', 'closest_2_distance']].mean(axis=1)
    final = final.drop(
        ['closest_station_2_temp', 'closest_2_distance'], axis=1)
    final['closest_1_distance'] = 1/(1+final['closest_1_distance'])

    # final_x = closest_stations.merge(
    #     x1, on=['station', 'latitude', 'longitude', 'hour'], how='left')
    # final_x = final_x.rename(
    #     {'closest_1_distance_x': 'closest_1_distance'}, axis=1)
    # final_x['closest_1_distance'] = 1 / (1 + final_x['closest_1_distance'])

    return final


def add_urban_data(ec_data_segment, urb_data_segment, closest_columns):
    ''' 
    This function adds the urban data to the dataframe
    '''
    urb_cols = ['valueImperviousfraction', 'valueTreefraction', 'valueBuildingheight',
                'valueNearestDistWater', 'valueWaterfraction', 'valueBuildingfraction']
    lst_cols = ['value_LST']

    urb_merged = ec_data_segment.merge(
        urb_data_segment, on=['station', 'latitude', 'longitude'], how='inner')
    

    urb_merged = urb_merged[['station', 'latitude',
                             'longitude', 'hour'] + lst_cols + urb_cols]

    if closest_columns is not None:
        test_data = closest_columns.merge(
            urb_merged, on=['station', 'latitude', 'longitude', 'hour'], how='inner')
        test_data = test_data.drop('station', axis=1).rename(
            {'value_LST': 'adjusted_lst'}, axis=1)
        # test_data['day_of_year'] = test_data['beg_time'].dt.dayofyear
        test_data['day_of_year'] = pd.to_datetime(test_data['beg_time']).dt.dayofyear

    else:
        test_data = urb_merged.rename({'value_LST': 'adjusted_lst'}, axis=1)
        test_data['day_of_year'] = 1

    if 'value_LST_x' in test_data.columns:
        test_data = test_data.drop('value_LST_y', axis=1).rename(
            {'value_LST_x': 'adjusted_lst'}, axis=1)

    if 'closest_1_distance_y' in test_data.columns:
        test_data = test_data.drop('closest_1_distance_y', axis=1)

    # handling null values
    test_data['valueBuildingheight'] = test_data['valueBuildingheight'].fillna(
        0)
    test_data['adjusted_lst'] = test_data['adjusted_lst'].fillna(
        test_data['adjusted_lst'].mean())
    test_data['valueNearestDistWater'] = 1 / \
        (1 + test_data['valueNearestDistWater'])
    test_data.loc[test_data.valueTreefraction < -10, 'valueTreefraction'] = 0

    return test_data
# test_data['prediction_temp'] = lr_model.predict(test_data)


def merge_predictions(predict_dir):
    ''' It merges all the predictions into one file '''
    files = os.listdir(predict_dir)
    files = sorted([os.path.join(predict_dir, x)
                   for x in files if 'final' in x])

    df1 = pd.read_csv(files[0])

    for file in files[1:]:
        df2 = pd.read_csv(file)
        df1 = pd.concat([df1, df2])

    # df1 = helper.convert_to_gpd(df1, 'epsg:4326', convert_to='epsg:6520')
    df1 = df1.groupby(['latitude', 'longitude']).mean().reset_index()

    # change value less than 0 to 0
    df1['prediction_temp'] = df1['prediction_temp'].apply(
        lambda x: 0 if x < -30 else x)

    return df1


def calculate_predictions(ec_data, urb_data, stations_ranges,closest,debug,hour_selection):
    ''' Function to calculate predictions for all stations in the dataset
       Returns a list of columns that's necessary for creating plots
    '''
    col_file_name = os.path.join(MODEL_PATH, MODEL_CLASS+'_cols_list.csv')
    COL_LIST = pd.read_csv(col_file_name).columns.tolist()

    for index, station_list in enumerate(stations_ranges[:len(stations_ranges)]):
        time1 = time.time()
        ec2_segment = ec_data[ec_data.station.isin(station_list)]
        urb2_segment = urb_data[urb_data.station.isin(station_list)]


        if closest:
            final = add_temp_data(ec2_segment,hour_selection)
            test_data = add_urban_data(ec2_segment, urb2_segment, closest_columns=final)

        else:
            test_data = add_urban_data(ec2_segment, urb2_segment, closest_columns=None)

        test_data = test_data.fillna(method='ffill')
        # print(test_data.shape)
        if debug:
            print(test_data.head())
            print(test_data.columns)
        
        test_data['prediction_temp'] = MODEL.predict(test_data[COL_LIST])

        save_df = test_data[['latitude', 'longitude',
                             'prediction_temp', 'adjusted_lst']]
        save_df = save_df.groupby(
            ['latitude', 'longitude']).mean().reset_index()
        save_df['station'] = station_list

        save_df.to_csv(f'{PREDICT_DIR}/final_preds_{index}.csv', index=False)

        # print(
        #     f'Iteration {index+1}/{len(stations_ranges)} complete in {round(time2-time1,2)} seconds')
        # print("#############################################")

    print(f'Predictions complete, saved in {PREDICT_DIR}')
    print(f'Time Taken : {round(time.time()-time1,2)} seconds)')
    test_col_list = test_data[COL_LIST].columns.to_list()
    return test_col_list


def get_rasters(PREDICT_DIR, ec_data, urb_data, stations_ranges,closest,debug,hour_selection=None):
    ''' Create rasters for the predictions and the adjusted LST
    Returns : Raster[1,2], bounds and test_column_list
              First raster is of predictions and second is of adjusted LST
    '''

    test_column_list = calculate_predictions(ec_data, urb_data, stations_ranges,closest,debug,hour_selection=None)
    df = merge_predictions(PREDICT_DIR)

    raster1, bounds = visualizer.get_raster(df, 'prediction_temp')
    raster2, bounds = visualizer.get_raster(df, 'adjusted_lst')

    return [raster1, raster2], bounds, test_column_list


def runner(model_name, ec_data, urb_data, stations_ranges,closest=None,debug=False,hour_selection=None):
    '''Wrapper for calculate_predictions and get_rasters

    '''

    # MODEL_NAME    = 'GradientBoostingRegressor_2.sav'
    global MODEL_CLASS
    global PREDICT_DIR
    global MODEL
    MODEL_NAME = model_name
    MODEL = pickle.load(open(MODEL_PATH+MODEL_NAME, 'rb'))
    MODEL_CLASS = MODEL_NAME.split('_')[0]
    PREDICT_DIR = os.path.join(PREDICTION_DIR, MODEL_CLASS)

    if not os.path.exists(PREDICT_DIR):
        os.mkdir(PREDICT_DIR)

    return get_rasters(PREDICT_DIR, ec_data, urb_data, stations_ranges,closest,debug,hour_selection=None)
