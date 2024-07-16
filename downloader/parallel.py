"""
Created on March 1

@author: Gaurav
@version: 2.0
Updated : March 28

Finalized code, that runs the pws_script.py in parallel.
"""
import time
import os
import sys
from multiprocessing import Pool
import pandas as pd
from pws_script import csv_maker, post_process_directory
import configparser



#Section to read variables from config file
config_path = os.path.dirname(__file__)
config = configparser.ConfigParser()
config.read(os.path.join(config_path,'wunder_config.ini'))

LOCATION_NAME = config['WUNDERGROUND']['LOCATION_NAME']
START_DATE = config['WUNDERGROUND']['START_DATE']

year = START_DATE.split('-')[0]

HOME_DIR = os.path.join(os.path.dirname(__file__),'..','data/raw_data/')     #path of the directory where the data is to be stored i.e. data/raw_data/location_name
YEAR_PATH = os.path.join(HOME_DIR,LOCATION_NAME,'pws_data'+ '_'+LOCATION_NAME.lower(),year)

if not os.path.exists(YEAR_PATH):
        print('year path made')
        os.mkdir(YEAR_PATH)


if len(sys.argv)<2:
    print("Usage: python parallel.py <pws/observation/airport>")
    sys.exit(1)

ITEM = sys.argv[1]
time1 = time.time()

FILEPATH = os.path.join(HOME_DIR,LOCATION_NAME,ITEM+'_station_lists_'+ LOCATION_NAME.lower()+'.csv')
stations = pd.read_csv(FILEPATH)
station_list = list(stations.stationId.unique())


if __name__ == '__main__':
    with Pool(40) as pool: # four parallel jobs
        if ITEM == 'pws':
            results = pool.map(csv_maker,station_list)

        # elif ITEM =='observation':
        #     from observation_script import csv_maker
        #     results = pool.map(observation_script.observation_script,station_list)
        # results = pool.map(csv_maker,station_list)

    for station in station_list:
        try:
            post_process_directory(station)
        except Exception as e:
            # logging.error(f'Error in post-processing for station {station}: {e}')
            print(f"Error in post-processing for station {station}: {e}")
        # print(f'Post processing for {station} done')
    print(f'time taken {time.time()-time1}')
