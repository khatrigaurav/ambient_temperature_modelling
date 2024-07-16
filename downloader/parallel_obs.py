"""
Created on March 1

@author: Gaurav
@version: 2.0
Updated : March 28

Finalized code, that runs the observation_script.py in parallel.
"""
import time
import os
import sys
from multiprocessing import Pool
import pandas as pd
from observation_script import batch_loader, post_process_directory

import configparser


#Section to read variables from config file
config = configparser.ConfigParser()
config.read('wunder_config.ini')

LOCATION_NAME = config['WUNDERGROUND']['LOCATION_NAME']
START_DATE = config['WUNDERGROUND']['START_DATE']
YEAR_PATH = os.path.join('../data',LOCATION_NAME,'observation_data'+ '_'+LOCATION_NAME.lower(),START_DATE.split('-')[0])

if not os.path.exists(YEAR_PATH):
        os.mkdir(YEAR_PATH)

time1 = time.time()

station_file = os.path.join('../data',LOCATION_NAME, 'observation_station_lists_'+ LOCATION_NAME.lower()+'.csv')
stations = pd.read_csv(station_file)
station_list = list(stations.stationId.unique())


if __name__ == '__main__':
    with Pool(25) as pool: # four parallel jobs
        results = pool.map(batch_loader,station_list)

post_process_directory()
print(f'time taken {time.time()-time1}')

