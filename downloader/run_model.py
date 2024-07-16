'''
This scripts runs different models.
'''

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import rioxarray as rxr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import polars as pl #libs for faster data processing


from dateutil import tz
import geopandas as gpd

from pyproj import CRS


import sklearn.metrics as metrics
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression

#Custom modules
import lib.satellite as sat
import lib.dataprocess as dp 
import lib.crowdqc as cqc
from lib import helper
from lib import modeller as mod
from lib import visualizer

default_location , default_year = helper.get_config_locations()
print(f"Default location: {default_location}, Default year: {default_year}")

grouped_data_loc = f'data/processed_data/{default_location}_{default_year}/grouped_data/'
os.makedirs(grouped_data_loc, exist_ok=True)

#updating necessary columns from satellite
necessary_col_sat = ['station', 'beg_time', 'latitude', 'longitude', 
                # 'humidityAvg',
                  'temperature',
                  'heatindex',
               # 'windspeed', 'dewpt', 'precipRate',
               'day_of_year',
               'hour', 'adjusted_lst', 'valueImperviousfraction', 'valueTreefraction',
               'valueBuildingheight', 'valueNearestDistWater', 'valueWaterfraction',
              #  'valueLandcover', 
               'valueBuildingfraction',
              #  'valueUSGS'
               ]

#updating temporal columns
spatial_columns = [ 'station', 'beg_time', 'latitude','longitude',
                     'temperature', 
                     'closest_station_1_temp','closest_station_2_temp','closest_station_3_temp',
                     'closest_1_distance','closest_2_distance','closest_3_distance',
                     'closest_station_1_heatindex','closest_station_2_heatindex','closest_station_3_heatindex'
                     ]

clean_data = pd.read_csv(f'data/processed_data/{default_location}_{default_year}/clean_{default_location}_pws_.csv')
# clean_data = pd.read_csv('data/processed_data/Madison_2022/clean_Madison_pws_.csv')
clean_data['beg_time'] = pd.to_datetime(clean_data['beg_time'])
clean_data['hour'] = clean_data['beg_time'].dt.hour

clean_data = clean_data.query("day_of_year >= 120 & day_of_year <= 320")
clean_data.station = clean_data.station.str.upper()


