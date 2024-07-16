"""
Created on March 29

@author: Gaurav
@version: 2.0
Updated : March 28

Finalized code, that scrapes weather data from observation_stations_{city_name}.csv file.
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


def path_maker(file_paths):
    """Makes a /weather_data/ directory to store new data for
    current weather stations given a file path : /weather_data"""

    working_path = os.getcwd()
    file_path = os.path.join(working_path,file_paths)                             
    if not os.path.exists(file_path):
        os.mkdir(file_path)
        return file_path

    return file_path



#Section to read variables from config file
config = configparser.ConfigParser()
config.read('wunder_config.ini')
API_KEY = config['WUNDERGROUND']['API_KEY']
LOCATION_NAME = config['WUNDERGROUND']['LOCATION_NAME']
START_DATE = config['WUNDERGROUND']['START_DATE']
END_DATE = config['WUNDERGROUND']['END_DATE'] 
date_range = [START_DATE,END_DATE]
year = START_DATE.split('-')[0]

FILE_PATH = os.path.join('../data',LOCATION_NAME,'observation_data'+ '_'+LOCATION_NAME.lower())
YEAR_PATH = os.path.join('../data',LOCATION_NAME,'observation_data'+ '_'+LOCATION_NAME.lower(),year)

full_path = path_maker(FILE_PATH)
if not os.path.exists(YEAR_PATH):
    os.mkdir(YEAR_PATH)

# FILE_PATH = '../data/Madison_V2/observation_station_lists_madison'


error_path = FILE_PATH + '/errors'
error_path = path_maker(error_path)
# logging.basicConfig(filename='../data/Madison_V2/observation_errors.log', encoding='utf-8', level=logging.ERROR)
logging.basicConfig(filename=error_path+'/error.log', encoding='utf-8', level=logging.ERROR)


station_file = os.path.join('../data',LOCATION_NAME, 'observation_station_lists_'+ LOCATION_NAME.lower()+'.csv')
stations = pd.read_csv(station_file)
station_list = list(stations.stationId.unique())
# print(station_list)


def check_response(station_id,start_date,end_date):
    '''Checks if the response is valid'''

    cmds = 'https://api.weather.com/v1/location/_station_id_:9:US/observations/historical.json?apiKey=_API_KEY&units=e&startDate=_start_date_&endDate=_end_date_'
    valid_cmd = cmds.replace("_station_id_",station_id).replace('_start_date_',start_date).replace('_end_date_',end_date).replace('_API_KEY',API_KEY)
    data = requests.get(valid_cmd,timeout=20).json()

    # print (valid_cmd)
    #status_code : 400 , data couldn't be fetched, invalid station_id
    if data.get('metadata').get('status_code') == 400:
        print(data.get('errors')[0].get('error').get('message')+ ': {}'.format(station_id))
        logging.error(data.get('errors')[0].get('error').get('message')+ ': {}'.format(station_id))

        data = False
        return data

    if data.get('metadata').get('status_code')==200:
#         print('success')
#         print(station_id)
        return data

    return data

def get_data(station_id,start_date,end_date):
    #data : Valid 
    data = check_response(station_id,start_date,end_date)

    if data:
        data = data.get('observations')
        dataframe = pd.DataFrame(data)

        dataframe['valid_time_gmt'] = dataframe['valid_time_gmt'].apply(lambda x:datetime.datetime.fromtimestamp(x))
        dataframe['expire_time_gmt'] = dataframe['expire_time_gmt'].apply(lambda x:datetime.datetime.fromtimestamp(x))

        # print(stations)

        dataframe['latitude'] = stations[stations['stationId']==station_id]['latitude'].iat[0]
        dataframe['longitude'] = stations[stations['stationId']==station_id]['longitude'].iat[0]

        required_cols = ['key', 'expire_time_gmt', 'obs_id','latitude','longitude', 'obs_name','valid_time_gmt', 'day_ind' ,'temp','dewPt','rh','wdir_cardinal','wspd','gust','pressure','precip_total','wx_phrase']

        dataframe = dataframe[required_cols]

        return dataframe






time1 = time.time()

''' Function to generate csv for one particular date range 
    Usage : csv_maker('2022-10-11', '2022-10-25') '''

# def csv_maker(start_date,end_date):
#     for x in station_list:
#         # FILE_PATH = 'weather_data'
#         # full_path = path_maker(FILE_PATH)
#         # full_path = full_path.strip('.csv')

#         file_name = os.path.join(full_path, x + '_'+ end_date +'.csv')

#         weather_dataframe = get_data(x,start_date,end_date)
        
        
#         # file_name.strip('.csv.')

#         if weather_dataframe is not None:
#             # weather_dataframe['latitude'] = stations[stations['stationId']==x]['latitude']
#             # weather_dataframe['longitude'] = stations[stations['stationId']==x]['longitude']
#             weather_dataframe.to_csv(file_name)
#             print(f"Creating file {file_name}")
#             time.sleep(2)


def csv_maker(start_date,end_date,station_list):
    # for x in station_list:
        # FILE_PATH = 'weather_data'
        # full_path = path_maker(FILE_PATH)
        # full_path = full_path.strip('.csv')

        file_name = os.path.join(full_path, station_list + '_'+ end_date +'.csv')

        weather_dataframe = get_data(station_list,start_date,end_date)
        
        
        # file_name.strip('.csv.')

        if weather_dataframe is not None:
            # weather_dataframe['latitude'] = stations[stations['stationId']==x]['latitude']
            # weather_dataframe['longitude'] = stations[stations['stationId']==x]['longitude']
            weather_dataframe.to_csv(file_name)
            print(f"Creating file {file_name}")
            time.sleep(2)

        # print("File creation successful, Total Time Taken : {} seconds".format(round(time2,2)))

        # time.sleep(2)

#Generates a date window of 30 days which is supported by the API
def date_range_generator(start_date,end_date):
#     if isinstance(end_date,datetime.datetime):
#         end_date = end_date + datetime.timedelta(days=-1)
#         print(end_date)
#         end_date = str(end_date).split(' ')[0]
    
    # end_date=datetime.timedelta(days=-1) + datetime.datetime.today()
    # end_date = str(end_date).split(' ')[0]
    date_range = [start_date,end_date]
    extract_range = np.arange(date_range[0],date_range[1], step=30, dtype='datetime64[D]')
    extract_range = [str(x) for x in extract_range]   
    
    # if not(extract_range[-1] == str(datetime.datetime.today()).split(' ')[0]):
    #     extract_range.append((end_date).split(' ')[0])

    if not(extract_range[-1] == end_date):
        extract_range.append(end_date)
    
    return extract_range

# def date_range_generator():
#     '''Generates a date range for the given start and end date'''
#     extract_range = np.arange(date_range[0],date_range[1], step=1, dtype='datetime64[D]')
#     extract_range = [str(x) for x in extract_range]
#     return extract_range



date_range = date_range_generator(START_DATE,END_DATE)
date_range = [x.replace('-','') for x in date_range]

#runs the scraping code on month by month basis
def batch_loader(station_id):
    # date_range = date_range_generator(start_date)
    # date_range = [x.replace('-','') for x in date_range]
    idx = 0
    
    while(idx<len(date_range)-1):
        
        csv_maker(date_range[idx],date_range[idx+1],station_id)
        idx += 1




def combine_dataframes(dframes):
    df = pd.concat(dframes,ignore_index=True)
    df = df.reset_index(drop=True).iloc[:,1:]
    df.drop_duplicates(subset=['key','expire_time_gmt','valid_time_gmt'])
    
    return df

def post_process_directory(station_id):

    files = [ x for x in os.listdir(FILE_PATH) if '.csv' in x and station_id in x  ]
    unique_files = set([x.split('_')[0] for x in files])
    # print(unique_files)
    


    for x in unique_files:
        valid_list = [pd.read_csv(os.path.join(full_path,k)) for k in files if x in k ]
        cf = combine_dataframes(valid_list)

        
        op_name = os.path.join(YEAR_PATH,x)
        op_name = '../'+op_name.strip('.csv') + '.csv'
        cf.to_csv(op_name)
        # cf.to_csv(op_name + '.csv')
    
    files_to_del = set(files) - set(unique_files)
    # print(files_to_del)
    for j in files_to_del:
        os.remove(os.path.join(full_path,j))

    print("Directory clean successfull")

        
        

# time1 = time.time()
for station in station_list:
    batch_loader(station)
    post_process_directory(station)

time2 =  time.time() - time1
print(f'Total time taken : {round(time2,2)} seconds')


# post_process_directory()