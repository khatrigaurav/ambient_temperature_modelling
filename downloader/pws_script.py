"""
Created on Jan 8

@author: Gaurav
@version: 2.1
Updated : March 29

Finalized code, that scrapes weather data from pws_stations_{city_name}.csv file.
"""

# curl 'https://api.weather.com/v1/location/KSFO:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20221014' --compressed

import datetime
import time
import os
import logging
import pandas as pd
import numpy as np
import requests
import configparser


#Section to read variables from config file
config_path = os.path.dirname(__file__)
config = configparser.ConfigParser()
config.read(os.path.join(config_path,'wunder_config.ini'))

API_KEY = config['WUNDERGROUND']['API_KEY']
LOCATION_NAME = config['WUNDERGROUND']['LOCATION_NAME']
START_DATE = config['WUNDERGROUND']['START_DATE']
END_DATE = config['WUNDERGROUND']['END_DATE'] 

date_range = [START_DATE,END_DATE]
year = START_DATE.split('-')[0]
# FILE_PATH = 'data/Orlando/pws_data_orlando'

 #path of the directory where the data is to be stored i.e. data/raw_data/location_name
HOME_DIR = os.path.join(os.path.dirname(__file__),'..',f'data/raw_data/{LOCATION_NAME}')    
FILE_PATH = os.path.join(HOME_DIR,'pws_data'+ '_'+LOCATION_NAME.lower())
YEAR_PATH = os.path.join(HOME_DIR,'pws_data'+ '_'+LOCATION_NAME.lower(),year)

# if not os.path.exists(YEAR_PATH):
#         os.mkdir(YEAR_PATH)

def path_maker(file_paths):
    """Makes a /weather_data/ directory to store new data for
    current weather stations given a file path : /weather_data"""

    working_path = os.getcwd()
    file_path = os.path.join(working_path,file_paths)                             
    if not os.path.exists(file_path):
        os.mkdir(file_path)
        return file_path

    return file_path


full_path = path_maker(FILE_PATH)

station_file = os.path.join(HOME_DIR, 'pws_station_lists_'+ LOCATION_NAME.lower()+'.csv')
# print(f'Using station file {station_file}')
stations = pd.read_csv(station_file)
station_list = list(stations.stationId.unique())

error_path = os.path.join(HOME_DIR,'pws_data'+ '_'+LOCATION_NAME.lower(),'errors')
# 'data/Orlando/pws_data_orlando/errors/'
error_paths = path_maker(error_path)
logging.basicConfig(filename=error_paths+'/error.log', encoding='utf-8', level=logging.ERROR)



    
def date_range_generator():
    '''Generates a date range for the given start and end date'''
    extract_range = np.arange(date_range[0],date_range[1], step=1, dtype='datetime64[D]')
    extract_range = [str(x) for x in extract_range]
    return extract_range



date_range = date_range_generator()
date_range = [x.replace('-','') for x in date_range]




def check_response(station_id,start_date,retries=3):
    '''Checks if the response is valid'''

    cmds = 'https://api.weather.com/v2/pws/history/hourly?stationId=_station_id_&format=json&units=m&date=_start_date_&apiKey='+API_KEY

    # cmds = 'https://api.weather.com/v1/location/_station_id_:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=_start_date_&endDate=_end_date_'
    valid_cmd = cmds.replace("_station_id_",station_id).replace('_start_date_',start_date)

    # print(valid_cmd)
    # data = requests.get(valid_cmd,timeout=20).json()
    for attempt in range(retries):
        try:
            data = requests.get(valid_cmd, timeout=60).json()  # Increased timeout
            if len(data.get('observations', [])) == 0:
                print(f'No data found for {station_id} on {start_date}')
                return None
            return data
        except requests.exceptions.Timeout:
            print(f'Request timed out for {station_id} on {start_date}')
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
    return None

    # print (valid_cmd)
    #status_code : 400 , data couldn't be fetched, invalid station_id
#     if len(data.get('observations')) ==0:
#         # print(f'No data found for {station_id} on {start_date}')
#         logging.error(f'No data found for {station_id} on {start_date}')

#         data = False
#         return data

#     if len(data.get('observations')) > 0:
# #         print('success')
# #         print(station_id)
#         return data

#     return data

def get_data(station_id,start_date):
    #data : Valid 
    data = check_response(station_id,start_date)

    if data:
        data = data.get('observations')
        dataframe = pd.DataFrame(data)

        dataframe['temperature'] = dataframe['metric'].apply(lambda x : x['tempAvg'])
        dataframe['windspeed'] = dataframe['metric'].apply(lambda x : x['windspeedAvg'])
        dataframe['dewpt'] = dataframe['metric'].apply(lambda x : x['dewptAvg'])
        dataframe['heatindex'] = dataframe['metric'].apply(lambda x : x['heatindexHigh'])
        dataframe['precipRate'] = dataframe['metric'].apply(lambda x : x['precipRate'])
        dataframe['precipRate'] = dataframe['metric'].apply(lambda x : x['precipTotal'])
        dataframe = dataframe[['stationID', 'tz', 'obsTimeLocal', 'epoch', 'lat', 'lon',
                            'humidityAvg', 'temperature',
                            'windspeed', 'dewpt', 'heatindex', 'precipRate']]

        return dataframe




def csv_maker(station):
    

    for x in date_range:
        file_path = path_maker(FILE_PATH)
        file_name = os.path.join(file_path, station + '_'+ x +'.csv')

        df = get_data(station,x)

        if df is not None:
            df.to_csv(file_name)
            # print(f"Creating file {file_name}")

    # print(f'Finished processing CSV for {station}')
    
    # post_process_directory(station)



def combine_dataframes(dframes):
    if len(dframes)>0:
        df = pd.concat(dframes,ignore_index=True)
        df = df.reset_index(drop=True).iloc[:,1:]
        df = df.drop_duplicates(subset=['stationID', 'obsTimeLocal', 'epoch'])
    
        return df

    else:
        print("No dataframes found")

def post_process_directory(station):

    files = [ x for x in os.listdir(full_path) if ('.csv' in x and 'master' not in x and station in x)]
    valid_list = [pd.read_csv(os.path.join(full_path,k)) for k in files ]
    combined_frame = combine_dataframes(valid_list)
        
    file_name = os.path.join(YEAR_PATH , station + '.csv')
    if combined_frame is not None:        
        combined_frame.to_csv(file_name,index=False)

    for j in files:
        os.remove(os.path.join(full_path,j))

    print(f"Directory clean successfull {station}")


# if __name__ == '__main__':
#     for each_station in station_list:
#         csv_maker(each_station)
