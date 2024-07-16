
"""
Created on : Nov 18, 2023

@author: Gaurav
@version: 1.0

Functions to be used in 00 Modeller.ipynb
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pickle
import time
import configparser


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib import satellite as sat
from lib import helper as helper
# from lib import predictor
from lib import visualizer
from lib import helper

# Reading the config file
# config = configparser.ConfigParser()
# config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
# config.read(config_path)
# location = config['DEFAULT']['location']
# year = config['DEFAULT']['year']


# Reading the config file

HOME_DIR = os.path.join(os.path.dirname(__file__),'..') 
MODEL_PATH = os.path.join(HOME_DIR, 'Resources/trained_models/')
hour_lookup_file  = os.path.join(HOME_DIR,'temp_files/hour_lookup.csv')

def create_satellite_data(location='Madison', year=2021):
    """
    This function creates the satellite data for the given year and location
    # Usage : create_satellite_data(location = 'Madison',year=2021,urban_data=False)
    """
    sat.create_station_file(location, year, urban_data=False)
    sat.create_station_file(location, year, urban_data=True,)


def process_raster_data(clean_data, create=False, year=2021, location='Madison'):
    """
    This function processes the raster data for the given year and location
    Cleaning the rows and renaming
    """
    if create:
        create_satellite_data(location, year)

    location, year, shapefile = helper.get_configs()
    TRAIN_ECOSTRESS_FILE = os.path.join(HOME_DIR, f'data/raster_op/{location}/{year}/ECOSTRESS_values.csv')
    TRAIN_URBAN_FILE = os.path.join(HOME_DIR, f'data/raster_op/{location}/{year}/urban_surface_properties_values.csv')

    # This data is reads the tiff files created by 5. Combining Ecostress.ipynn
    ecostress_data = pd.read_csv(TRAIN_ECOSTRESS_FILE)
    urban_data = pd.read_csv(TRAIN_URBAN_FILE)

    ecostress_data = ecostress_data[[
        'station', 'latitude', 'longitude', 'value_LST', 'hour']]
    urban_data = (urban_data.iloc[:, 1:]).drop(
        columns=['beg_time', 'geometry'])

    if 'valueBuildingheight' in urban_data.columns:
        urban_data['valueBuildingheight'] = urban_data['valueBuildingheight'].fillna(
            0)

    # result_df = sat.station_daily_lst_anomaly_means()
    result_df = ecostress_data[['station', 'hour', 'value_LST']]

    result_df = result_df.rename(columns={'value_LST': 'adjusted_lst'})
    result_df.station = result_df.station.str.upper()
    
    result_df['hour'] = result_df['hour'].astype('int64')


    updated_data = pd.merge(clean_data, result_df, on=[
                            'station', 'hour'], how='left')
    updated_data = pd.merge(updated_data, urban_data, on=[
                            'station', 'latitude', 'longitude'], how='inner')

    return updated_data


def null_fill_strategy(final_df, strategy='mean'):
    ''' This function fills the null values in the final_df with the given strategy
    '''
    if strategy == 'mean':
        list_of_cols = [x for x in final_df.columns if 'closest_station' in x]
        for x in list_of_cols:
            final_df[x] = final_df[x].interpolate(
                method='linear', limit_direction='both')

    return final_df


def get_final_df(new_data, updated_data, spatial_columns):
    ''' new_data : consists of closest stations and temperatures
        updated_data : consists of all the satellite data
    '''
    new_data = new_data[spatial_columns]
    final_df = pd.merge(new_data, updated_data, on=[
                        'station', 'beg_time', 'temperature', 'latitude', 'longitude'], how='inner')

    final_df = null_fill_strategy(final_df, strategy='mean')

    return final_df


def split_(sequence, window_size):
    ''' sequence : input array of tempearture values : num_sample * 1
        window_size : number of lagged values to be used as features
    '''

    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y = sequence.iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def get_time_adjusted_df(final_df, start_end_date=None, window_size=5, column='temperature'):
    ''' This function creates the time adjusted dataframe with lagged values
    '''
    # window_size            #480 = 24 obs per day * 20 days : data from past 20 days is taken as features
    # number_of_days = 720            #720 = 24 obs per day * 30 days : to predict for next 30 days

    if start_end_date is None:
        start_date = 70
        end_date = 180

    else:
        start_date = start_end_date[0]
        end_date = start_end_date[1]

    # final_df_slice = final_df.query(f'day_of_year > {start_date} and day_of_year < {end_date}')
    # print(final_df.columns)
    final_df_slice = final_df
    series = final_df_slice[column]

    x_train, y_train = split_(series, window_size)

    columns = ['t_'+str(i) for i in range(window_size, 0, -1)]
    final_df_ = final_df_slice[window_size:].reset_index(drop=True)
    temp_ = pd.DataFrame(x_train, columns=columns)
    time_adjusted_df = pd.concat([final_df_, temp_], axis=1)
    time_adjusted_df.sort_values(['station', 'beg_time'], inplace=True)

    # 1+ because distance can be zero
    time_adjusted_df['closest_1_distance'] = 1 / \
        (1+(time_adjusted_df['closest_1_distance']))
    time_adjusted_df['closest_2_distance'] = 1 / \
        (1 + (time_adjusted_df['closest_2_distance']))
    time_adjusted_df['closest_3_distance'] = 1 / \
        (1 + (time_adjusted_df['closest_3_distance']))

    if 'valueNearestDistWater' in time_adjusted_df.columns:
        time_adjusted_df['valueNearestDistWater'] = 1 / \
            (1 + (time_adjusted_df['valueNearestDistWater']))

    return time_adjusted_df


def get_train_test_data(final_df, window_size=5):
    ''' This function creates the train and test data from time adjusted dataframe'''

    time_adjusted_df = get_time_adjusted_df(
        final_df, start_end_date=None, window_size=window_size, column='temperature')

    if 'beg_time' in time_adjusted_df.columns:
        available_columns = ['station', 'temperature', 'beg_time']
    else:
        available_columns = ['station', 'temperature']

    X = time_adjusted_df.drop(available_columns, axis=1)

    y = time_adjusted_df['temperature']

    # will be required later on to find lagged values in prediction
    time_adjusted_path = os.path.join(HOME_DIR, 'temp_files/time_adjusted_df.csv')
    time_adjusted_df.to_csv(time_adjusted_path, index=False)

    # x_train, x_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.15, random_state=42, shuffle=True)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True)

    # will be required later on to align columns between train and test
    # x_test.head().to_csv('Analytics/X_test.csv', index=False)
    return x_train, x_test, y_train, y_test


def plot_op(model, x_test, y_test, hour_filter):
    ''' Plots the predicted and true values : Used by train save function'''

    plot_df = x_test.copy()
    plot_df['predicted_temperature'] = model.predict(plot_df)
    plot_df['true_temperature'] = y_test
    if len(hour_filter) == 1:
        plot_df['hour'] = hour_filter[0]
    error_score = metrics.mean_squared_error(
        plot_df['true_temperature'], plot_df['predicted_temperature'], squared=False,)

    return plot_df, error_score


def train_save(modelx, data, hour_filter, neural_net=None, fit=True):
    ''' Function to train and save the model
        modelx : model object to be trained 
        model_name : name of the model
        neural_net : A dictionary thats passed to the model.fit function if neural net
                    argyments : epochs, batch_size, verbose = 2
      '''
    # last_file = sorted([ x.split('.')[0][-1] for x in os.listdir(MODEL_PATH) if model_name in x ])[-1]
    model_name = modelx.__class__.__name__
    model_path = os.path.join(MODEL_PATH, model_name)
    os.makedirs(model_path, exist_ok=True)

    d_train, d_test, y_train, y_test = data[0], data[1], data[2], data[3]

    # To save up the list of columns used in the model for prediction (closest temps)
    # d_test.head().to_csv('Analytics/X_test.csv', index=False)

    # To save up the list of columns used in the model for prediction (closest temps)
    cols_list = data[0].head()

    # when its bulk mode, we dont want to clean the directory
    # make sure first file is not deleted

    cols_list.to_csv(os.path.join(
        model_path, model_name+'_cols_list.csv'), index=False)

    # To save up the model
    temp = os.listdir(model_path)
    temp = [x for x in temp if model_name in x and 'cols_list' not in x]
    if len(temp) == 0:
        new_file = 0
    else:

        last_file = sorted([int(x.split('.')[0][len(model_name)+1:])
                            for x in os.listdir(model_path) if model_name in x and 'cols_list' not in x])[-1]
        new_file = int(last_file)+1
    file_name = f'{model_path}/{model_name}_{new_file}.sav'

    # train the model
    if fit is True:
        if neural_net is not None:
            modelx.fit(
                d_train, y_train, epochs=neural_net['epochs'], batch_size=neural_net['batch_size'], verbose=1)
        else:
            modelx.fit(d_train, y_train)

    elif fit is False:
        print('Skipping model fit')

    # save the model using pickle
    print(f'Model Saved : {file_name}')
    pickle.dump(modelx, open(file_name, 'wb'))

    predictions_df, error_score = plot_op(modelx, d_test, y_test, hour_filter)
    return predictions_df, error_score


def get_partitions(final_df, col_list, selection_hour, scaler=False):
    ''' Get final split of data based on hour selected
        selection_hour = [1] or None
    '''
    window_size = 5

    final_df_x = final_df.query(
        f'hour == {list([selection_hour or list(np.arange(0,24,1))][0])}')
    X_train, X_test, y_train, y_test = get_train_test_data(
        final_df_x, window_size)
    hour_status = final_df_x.hour.unique()

    d_train, d_test = X_train[col_list], X_test[col_list]

    # if there is no hour column, then its set to none, such that the plotter function behaves accordingly
    # hour_status = True if 'hour' in d_train.columns else False

    if scaler:
        scaler = StandardScaler()

        d_train = scaler.fit_transform(d_train)
        d_train = pd.DataFrame(d_train, columns=col_list)
        d_test = scaler.transform(d_test)
        d_test = pd.DataFrame(d_test, columns=col_list)

    # print(y_train.isna().sum())
    # print(y_test.isna().sum())

    
    return [d_train, d_test, y_train, y_test], hour_status


def bulk_model_runner(model, grouped_data, col_list, delete=True, bulk_mode=True, fit=True, residuals=False):
    ''' This function runs the model for all hours and saves the predictions and error
        grouped_data = data grouped by station
        col_list = columns to be used for modelling
        delete = True if you want to delete the contents of directory
        bulk_mode = True if you want to train one model for each hour
        fit = True if you want to fit the model, False if you want to load the model
            will be used for transfer learning
    '''

    # cleaning the directory first
    model_name = model.__class__.__name__
    model_path = os.path.join(MODEL_PATH, model_name)
    if delete and os.path.exists(model_path):
        helper.clean_directory(model_path)

    new_grouped_data = grouped_data.copy()
    # print(new_grouped_data.isna().sum())
    context_json = helper.get_context(write=False)



    if residuals:
        # group the grouped data by hour to find hourly mean and subtract it from the temperature
        print('Residuals are being subtracted from data')
        new_grouped_data['delta_temp'] = new_grouped_data['temperature'] - \
            new_grouped_data.groupby('hour')['temperature'].transform('mean')
        new_grouped_data['old_temperature'] = new_grouped_data['temperature']
        new_grouped_data['temperature'] = new_grouped_data['delta_temp']
        hour_lookup = new_grouped_data.groupby(
            'hour')['old_temperature'].mean().reset_index()

        hour_lookup.to_csv(hour_lookup_file, index=False)
        print('Hour lookup saved in /temp_files/hour_lookup.csv')

    model_dict = {}
    model_output_dict = {}
    if bulk_mode:
        feature_importances_dict = dict()
        for hour_ in range(24):
            # print(f'Running model for hour {hour_}')
            # print(new_grouped_data.head())
            data, hour_status = get_partitions(
                new_grouped_data, col_list, [hour_])  # or None
            predictions, error = train_save(model, data, hour_status)
            try:
                feature_importances = model.feature_importances_
            except:
                feature_importances = None
            model_dict[hour_] = [predictions, error, feature_importances]

        for hour in model_dict.keys():
            feature_importances_dict[hour] = model_dict[hour][2]
            model_dict[hour][0]['hourly_rms'] = model_dict[hour][1]

        hour0 = model_dict[0][0]

        for hour in model_dict.keys():
            if hour != 0:
                hour0 = pd.concat([hour0, model_dict[hour][0]], axis=0)

        model_output_dict['hourly_values'] = hour0
        model_output_dict['feature_importances'] = pd.DataFrame(
            feature_importances_dict).mean(axis=1).values

    if not bulk_mode:
        data, hour_status = get_partitions(new_grouped_data, col_list, None)
        predictions, error = train_save(model, data, hour_status)
        try:
            feature_importances = model.feature_importances_
        except:
            feature_importances = None
        model_dict['hourly_values'] = [predictions, error, feature_importances]

        # new module to uniformly output data
        combined_df = model_dict['hourly_values'][0]
        hourly_rmse = combined_df.groupby('hour').apply(lambda x: metrics.mean_squared_error(
            x.predicted_temperature, x.true_temperature, squared=False)).reset_index(name='hourly_rms')
        combined_df = pd.merge(combined_df, hourly_rmse, on='hour')

        model_output_dict['hourly_values'] = combined_df
        model_output_dict['feature_importances'] = model_dict['hourly_values'][2]

    if residuals:
        print('Residuals are being added back to the predictions')
        predictions = model_output_dict['hourly_values']

        model_dict_updated = predictions.merge(hour_lookup, on='hour')
        model_dict_updated['predicted_temperature'] = model_dict_updated['predicted_temperature'] + \
            model_dict_updated['old_temperature']
        model_dict_updated['true_temperature'] = model_dict_updated['true_temperature'] + \
            model_dict_updated['old_temperature']

        model_output_dict['hourly_values'] = model_dict_updated

    return model_output_dict


def find_frequent(outliers_dict, number=6):
    hash_map = dict()
    hash_map_tree = dict()

    for hrs in outliers_dict.keys():
        val = outliers_dict[hrs]
        for v in val:
            hash_map[v] = 1 + hash_map.get(v, 0)
            hash_map_tree[v] = 1 + hash_map_tree.get(v, 0)

    hash_map_tree = {k: v for k, v in sorted(
        hash_map_tree.items(), key=lambda item: item[1], reverse=True)}
    top_n = {k: v for k, v in sorted(
        hash_map_tree.items(), key=lambda x: x[1], reverse=True)[:number]}
    return top_n


def combine_hashes(hash1, hash2):
    for k, v in hash2.items():
        hash1[k] = 1 + hash1.get(k, 0)
    return hash1
