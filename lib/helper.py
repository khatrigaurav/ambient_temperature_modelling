"""
Created on : Oct 18, 2023

@author: Gaurav
@version: 1.0

Helper functions that are independent of the source and help in handling dataframes.

Rule : They must take dataframe as one input and return a dataframe as output. Exception: calculate_distance

Functions:
		resample_daily(df):

		find_closest(df):
		find_closest_temperatures(df_daily):
		find_anomaly(df_daily, anomaly_threshold=0.6):

		run_mean_comparison(df,anomaly_threshold=0.5) ---> main function



"""
import os
import json
import zipfile

import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import imageio
import configparser

from shapely.geometry import Point
from scipy.spatial import KDTree
from geopy.distance import geodesic
import matplotlib.pyplot as plt


HOME_PATH = os.path.join(os.path.dirname(__file__),'..')

#__contex_location will be later replaced with location we are dealing with
path_ = os.path.join(HOME_PATH,'..','ECOSTRESS_and_urban_surface_dataset/__context_location__/ECOSTRESS/adjusted_output_directory/')
context_file = os.path.join(HOME_PATH,'temp_files/plot_context.json')

def get_recent(path):
    ''' Function to get the most re'cent file in a folder
        Used by predictor to get most recent model 
        
    '''
    files = os.listdir(path)
    files_sorted_by_creation_time = sorted(files, key=lambda f: os.path.getctime(os.path.join(path, f)))
    files_sorted_by_creation_time = [f for f in files_sorted_by_creation_time if 'DS_Store' not in f and 'saved' not in f]
    return files_sorted_by_creation_time[-1]



def get_configs():
    ''' Get the configuration from the config.ini file'''

    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)
    location = config['DEFAULT']['location']
    year = config['DEFAULT']['year']
    shapefile = config['DEFAULT']['shapefile']

    return location, year, shapefile

def get_config_locations():
    ''' Get the configuration from the config.ini file'''

    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)
    year = config['DEFAULT']['year']
    location = config['DEFAULT']['location']

    return location,year

def zoomed_image_maker(ret):
    ''' Given a numpy image array, this function saves the zoomed images in the folder'''
    for hour in ret.keys():
        plt.figure(figsize=(12,12))
        plt.imshow(ret[hour][250:400,280:480],cmap='plasma')
        plt.colorbar(label='Temperature (°C)',shrink=0.8)
        plt.title(f'Hour: {hour}')
        filename = 'hour_'+str(hour).zfill(2)+'.jpeg'

        save_path = os.path.join(HOME_PATH,'temp_files/zoomed_images/')
        os.makedirs(save_path,exist_ok=True)
        plt.savefig(os.path.join(save_path,filename),dpi=400)
        # plt.savefig(f'Analytics/temp_data/zoomed_images/{filename}',dpi=400)


def get_context(write=True):
    ''' This function is used by visualizer.plot_mean to find context of model being used
        i.e. whether it has interpolated hours    
    '''
    context_json = {}
    context_json['location'] = get_configs()[0]
    context_json['year'] = get_configs()[1]

    context_location = context_json['location']

    interpolated_hrs = []
    path = path_.replace('__context_location__',context_location)

    for file in os.listdir(path):
        if 'interpolated' in file:
            hr = file.split('_')[-1][:2]
            interpolated_hrs.append(int(hr))

    context_json['interpolated_hrs'] = interpolated_hrs

    if write:
        with open(context_file, 'a', encoding='utf-8') as f:
            json.dump(context_json, f)

    return context_json
        

def calculate_distance(coord1, coord2):
    """Calculates the distance between two coordinates in km"""
    return geodesic(coord1, coord2).kilometers

def clean_directory(path):
    ''' Cleans the directory of all the files'''
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                file_path = os.path.join(path, entry.name)
                os.remove(file_path)
    print(f'Directory cleaned {path}')

def uniform_files(raster_path_):
    ''' Converts all the files in the folder to the same resolution as the first file 
        Usage Example : raster_path_ = '/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/Madison/ECOSTRESS/geotiff_clipped_stateplane/'

        uniform_files('data')
                        
    '''

    files = os.listdir(raster_path_)
    files = [x for x in files if '.tif' in x]
    other_files = files[1:]

    temperature_data_1 = rxr.open_rasterio(os.path.join(raster_path_,files[0]))

    for file in other_files:
        temperature_data_2 = rxr.open_rasterio(os.path.join(raster_path_,file))
        temperature_data_2 = temperature_data_2.rio.reproject_match(temperature_data_1)

        temperature_data_2 = temperature_data_2.rio.to_raster(os.path.join(raster_path_,file))

    return temperature_data_1

def convert_to_gpd(data,crs,convert_to = False):
    ''' Takes a dataframe that has latitude and longitude and converts it to a geopandas dataframe
        Additionally, it converts the coordinates to the crs provided

        Usage Example :  convert_to_gpd(df_,'epsg:4326',convert_to='epsg:6879')   
    '''
    new_df = gpd.GeoDataFrame(data).reset_index().drop('index',axis=1)
    new_df['geometry'] = [Point(xy) for xy in zip(new_df['longitude'], new_df['latitude'])]
    new_df = gpd.GeoDataFrame(new_df,geometry=new_df['geometry'],crs=crs)

    if convert_to:
        new_df = new_df.to_crs(convert_to)

    new_df['latitude'] = new_df.geometry.y
    new_df['longitude'] = new_df.geometry.x   
    return new_df

def get_list_of_stations(temperature_data_1,data_col_name='data'):
    ''' Get list of stations from the raster file after uniforming the files
        temperature_data_1 = rxr.open_rasterio('data/temperature_data_1.tif')
    '''

    rds = temperature_data_1.squeeze().drop('spatial_ref').drop('band')
    rds.name = data_col_name

    df = rds.to_dataframe().reset_index()
    # print(df.columns)
    df = df.rename({'x':'longitude','y':'latitude'},axis=1)

    return df

def to_gif(gif_directory,location,year,image_fps = 1.5):
    '''Convert .png images from a directory into a gif file
        gif_directory = '/Users/gaurav/UAH/temperature_modelling/Resources/temperature_plots/Madison_2021_LST/'

    '''
    images = os.listdir(gif_directory)
    # print(images)
    image_paths = sorted([os.path.join(gif_directory,x) for x in images])
    images = [imageio.imread(path) for path in image_paths if '.png' in path or 'jpeg' in path]
    imageio.mimsave(os.path.join(gif_directory,f'{location}_{year}.gif'), images,fps = image_fps)

    # print(image_paths)
    # return images


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



def find_closest_temperatures(df_daily):
    '''
        This functions takes a dataframe and finds closest stations.
		Additionally, it adds temperatures for each of the 3 closest stations too.
    '''
    # if test:
    #     closest_ = find_closest_test(df_daily)
    # else:
    closest_ = find_closest_station(df_daily)

    df_slice = pd.merge(df_daily,closest_,on=['station','latitude','longitude'],how='inner')
    print(df_daily.columns)
    df_joined = pd.merge(
        df_slice
        [
            [
                "station",
                "temperature",
                "beg_time",
                "latitude",
                "longitude",
                "heatindex",
                "closest_station_1",
                "closest_station_2",
                "closest_station_3",
                "closest_1_distance",
                "closest_2_distance",
                "closest_3_distance"
            ]
        ]
        ,
        df_daily
        [["station", "beg_time", "temperature","heatindex"]]
        ,
        left_on=["closest_station_1", "beg_time"],
        right_on=["station", "beg_time"],
        how="left",
    )
    df_joined = df_joined.rename(
        columns={
            "station_x": "station",
            "temperature_x": "temperature",
            "temperature_y": "closest_station_1_temp",
            "heatindex_x" : "heatindex",
            "heatindex_y" : "closest_station_1_heatindex"
        }
    ).drop("station_y", axis=1)

    df_joined = pd.merge(
        df_joined
        [
            [
                "station",
                "temperature",
                "beg_time",
                "latitude",
                "longitude",
                "heatindex",
                "closest_station_1_temp",
                "closest_station_1",
                "closest_station_2",
                "closest_station_3",
                "closest_1_distance",
                "closest_2_distance",
                "closest_3_distance",
                "closest_station_1_heatindex"
            ]
        ],
        df_daily[["station", "beg_time", "temperature","heatindex"]],
                        
        left_on=["closest_station_2", "beg_time"],
        right_on=["station", "beg_time"],
        how="left",
    )
    df_joined = df_joined.rename(
        columns={
            "station_x": "station",
            "temperature_x": "temperature",
            "temperature_y": "closest_station_2_temp",
            "heatindex_x" : "heatindex",
            "heatindex_y" : "closest_station_2_heatindex"
        }
    ).drop("station_y", axis=1)

    df_joined = pd.merge(
        df_joined
        [
            [
                "station",
                "temperature",
                "beg_time",
                "latitude",
                "longitude",
                "heatindex",
                "closest_station_1_temp",
                "closest_station_2_temp",
                "closest_station_1",
                "closest_station_2",
                "closest_station_3",
                "closest_1_distance",
                "closest_2_distance",
                "closest_3_distance",
                "closest_station_1_heatindex",
                "closest_station_2_heatindex"
            ]
        ],
         
        df_daily[["station", "beg_time", "temperature","heatindex"]],

        left_on=["closest_station_3", "beg_time"],
        right_on=["station", "beg_time"],
        how="left",
    )
    df_joined = df_joined.rename(
        columns={
            "station_x": "station",
            "temperature_x": "temperature",
            "temperature_y": "closest_station_3_temp",
            "heatindex_x" : "heatindex",
            "heatindex_y" : "closest_station_3_heatindex"
        }
    ).drop("station_y", axis=1)

    df_joined = df_joined.sort_values(by=["station", "beg_time"])


    return df_joined

# Define a function to find the closest stations using KDTree
def find_closest_stations_kdtree(row, kdtree, closest_tadj, num_closest=3, start_index=0):
    '''
    row : each row of the dataframe for which we need to find closest stations
    kdtree : KDTree object created using the coordinates of the stations
    closest_tadj : dataframe that consists of pws stations and their coordinates
    '''
    source_coord = (row['latitude'], row['longitude'])
    # Query the KDTree for the closest stations
    distances, indices = kdtree.query(source_coord, k=num_closest+start_index)

    # Skip the first index, which is the source station itself if searching within itself
    indices = indices[start_index:]
    distances = distances[start_index:]

    # Get the corresponding station names
    closest_station_names = closest_tadj['station'].iloc[indices].tolist()

    return closest_station_names, distances


def find_closest_station(df_daily, reference_df=None):
    '''
        This functions takes a dataframe and finds closest stations.
                Additionally, it adds temperatures for each of the 3 closest stations and distances too.
                
        df_daily : Should always be the dataframe with PWS stations, lat, long 
        reference_df : Dataframe that consists of individual lat-long values of random points
    '''
    df_daily = df_daily[['station', 'latitude', 'longitude']].drop_duplicates()
    station_coords = df_daily[['latitude', 'longitude']].values
    kdtree = KDTree(station_coords)

    if reference_df is None:
        df_daily[['closest_stations', 'closest_distances']] = df_daily.apply(
            lambda row: find_closest_stations_kdtree(
                row, kdtree, df_daily, num_closest=3, start_index=1),
            axis=1,
            result_type='expand'
        )

        df_daily[['closest_station_1', 'closest_station_2', 'closest_station_3']
                 ] = df_daily['closest_stations'].apply(pd.Series)
        df_daily[['closest_1_distance', 'closest_2_distance', 'closest_3_distance']
                 ] = df_daily['closest_distances'].apply(pd.Series)

        # # Drop the temporary columns
        df_daily = df_daily.drop(
            ['closest_stations', 'closest_distances'], axis=1)
        
        return df_daily

    else:
        reference_df[['closest_stations', 'closest_distances']] = reference_df.apply(
            lambda row: find_closest_stations_kdtree(
                row, kdtree, df_daily, num_closest=3, start_index=0),
            axis=1,
            result_type='expand'
        )

        reference_df[['closest_station_1', 'closest_station_2', 'closest_station_3']
                     ] = reference_df['closest_stations'].apply(pd.Series)
        reference_df[['closest_1_distance', 'closest_2_distance', 'closest_3_distance']
                     ] = reference_df['closest_distances'].apply(pd.Series)

        # # Drop the temporary columns
        reference_df = reference_df.drop(
            ['closest_stations', 'closest_distances'], axis=1)

        return reference_df



def find_anomaly(df_daily, anomaly_threshold=0.6):
    '''
        Returns the dataframe that consist of anomalous days
        TODO : Maybe we can replace it with something like dynamic time warping (DTW algorithm) instead of static mean comparison
    '''
    # df_joined = find_closest_station(df_daily)
    df_joined = find_closest_temperatures(df_daily)
    df_joined["average_temperature"] = np.mean(
        df_joined[
            [
                "closest_station_1_temp",
                "closest_station_2_temp",
                "closest_station_3_temp",
            ]
        ],
        axis=1,
    )

    df_joined = df_joined[
        [
            "station",
            "beg_time",
            "temperature",
            "average_temperature",
            "latitude",
            "longitude",
        ]
    ]

    df_joined["difference"] = (
        abs(df_joined["temperature"] - df_joined["average_temperature"])
        / df_joined["temperature"]
    )
    anamoly = df_joined[df_joined["difference"].apply(
        lambda x: x > anomaly_threshold)]
    anamoly = anamoly[["station", "beg_time",
                       "temperature", "average_temperature"]]

    return anamoly


def run_mean_comparison(df, anomaly_threshold=0.5):
    """Runs the mean comparison for the dataframe and returns the anomalous days"""

    df_daily = resample_daily(df)
    anamoly = find_anomaly(df_daily, anomaly_threshold=anomaly_threshold)

    return anamoly



def combine_results(global_dict1,global_dict2):
    ''' Used to combine the results of two models and return the combined results'''

    return_data = {}
    for items in global_dict2.keys():
        # print(f'Keys being combined : {items}')

        if items != 'bound' and items != 'test_column_list' and items != 'model_name':
            # print(f'Keys being approved : {items}')
            data =np.nanmean([global_dict1[items][0],global_dict2[items][0]],axis=0)
            return_data[items] = data
        
        else:
            return_data[items] = global_dict2[items]

    return return_data


def zip_folder(folder_path, output_path):
    ''' Function to zip a folder and save it to the output path
        Usage Example : 
        folder_to_zip = "/Users/gaurav/UAH/temperature_modelling/data/raster_op/Madison/2023/asdad"
        output_zip_file = "/Users/gaurav/UAH/temperature_modelling/data/raster_op/Madison/2023/asdad.zip"
        
        zip_folder(folder_to_zip, output_zip_file)

        Used in visualizer.py to save old images

    '''
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname=arcname)

