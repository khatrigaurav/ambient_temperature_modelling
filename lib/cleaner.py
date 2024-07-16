"""
Created on : Oct 18, 2023

@author: Gaurav
@version: 1.0

Simple script to clean the master_data and save it.
Usage : python cleaner.py <location> <year>
"""

import os
import sys
import pandas as pd
import numpy as np
# os.chdir(HOME_DIR)

os.chdir(os.path.dirname(__file__))
import crowdqc as crowdqc
import dataprocess as dp
import helper


args = sys.argv[1:]
location = args[0]
year = args[1]

ANOMALY_THRESHOLD = 0.4
HOME_DIR = os.path.join(os.path.dirname(__file__),'..')
os.chdir(os.path.dirname(__file__))


SAVE_PATH = os.path.join(HOME_DIR, 'data/processed_data/')


def download_data(location, year, product_type='pws'):
    ''' Combines location data into single df , saves it and process it
        Usage : download_data('NewYork','2022','pws')
    '''

    dp.process_downloaded_data(location, 'wunderground', product_type, year)
    csv_name = location+'_'+year+'/master_'+location+'_'+product_type+'_.csv'
    csv_name = os.path.join(SAVE_PATH, csv_name)
    df_ = pd.read_csv(csv_name)
    df_ = dp.process_wunder(df_)

    # print(f'Data for {location} and {year} has been downloaded and processed')
    # print(df_.shape)

    return df_


def handle_anomalies(df_,product_type='pws'):
    '''
    Uses crowdqc library to clean data
    '''
    df_, indexes = crowdqc.level1_check(df_, 'temperature')
    crowdqc.level2_check(df_, 'temperature', 'beg_time')
    # anamoly = helper.run_mean_comparison(
    #     df_, anomaly_threshold=ANOMALY_THRESHOLD)

    # # deleting stations with more than 90 days of anomalous data
    # temp_ = anamoly.groupby('station').count().sort_values(
    #     by='beg_time', ascending=False).apply(lambda x: x > 85)
    # stations_to_delete = temp_[temp_.beg_time == True].index.values
    # print(
    #     f'Following stations will be deleted: {stations_to_delete} as they have over 3 months of anomalous data')

    # df_ = df_[~df_.station.isin(stations_to_delete)]
    df_['date_'] = pd.to_datetime(df_['beg_time'].dt.date)
    # df_clean = pd.merge(df_, anamoly[['station', 'beg_time', 'average_temperature']], left_on=[
    #                     'date_', 'station'], right_on=['beg_time', 'station'], how='left')

    # df_clean = df_clean[df_clean['average_temperature'].apply(lambda x: np.isnan(x))]
    # df_clean.rename({'beg_time_x': 'beg_time'}, axis=1, inplace=True)

    csv_name = os.path.join(SAVE_PATH,location+'_'+year+'/clean_'+location+'_'+product_type+'_.csv')

    # '/Users/gaurav/UAH/temperature_modelling/data/processed_data/Madison_2021/clean_Madison_pws_.csv'
    df_[['station', 'beg_time', 'latitude', 'longitude', 'humidityAvg', 'temperature', 'windspeed', 'dewpt', 'heatindex',
              'precipRate', 'day_of_year']].to_csv(csv_name, index=False)

    print(f'New clean data saved in {csv_name}')

    return


def main(locations, years):
    ''' Main Call '''

    df_ = download_data(locations, years)
    df_ = crowdqc.clean_missing(df_)
    # df_.to_csv('temp.csv')
    # print(df_.shape)
    handle_anomalies(df_)


main(location, year)
