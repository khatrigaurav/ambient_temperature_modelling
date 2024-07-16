# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:25:59 2023

@author: tvo

Script to extract values of underlying raster from the points

Step 1:
    
    Reproject the shapefile of polypoints to the same projections
    with raster images of ecostress and nlcd
    
    
    
Step 2:
    
    Using function extract values from points to get the values of
    either surface properties or LST values from ecostress imaginary
    
    Note: make sure the return datetime from ecostress is in LOCAL hours.
    
    
Call back function: station_daily_lst_anomaly_means()
    
"""

# ****** Required packages ************************#

import os
import pytz
import datetime
import rasterio
import configparser
import numpy as np
import pandas as pd
from pyproj import CRS
import geopandas as gpd
import rioxarray as rxr

from dateutil import tz
from functools import reduce
from shapely.geometry import Point
from lib import helper

# ****** Required packages ************************#
# ****** Required configs ************************#
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)
# location = config['DEFAULT']['location']
# year = config['DEFAULT']['year']
# ****** Required configs ************************#




# Homing : Setting correct path
# location, year,shapefile = helper.get_configs()
# SAVE_DIRECTORY = f'/Users/gaurav/UAH/temperature_modelling/data/raster_op/{location}/{year}/'
HOME_DIRECTORY = os.path.join(os.path.dirname(__file__),'..') 
# HOME_DIRECTORY = f'/Users/gaurav/UAH/temperature_modelling/'
# TIF_FOLDER_NAME = "geotiff_clipped_stateplane"
TIF_FOLDER_NAME = "adjusted_output_directory"


# if location == 'Madison':
#     timezone_ = config['Timezone']['timezone_madison']
# elif location == 'LasVegas':
#     timezone_ = config['Timezone']['timezone_lasvegas']
# elif location == 'Orlando':
#     timezone_ = config['Timezone']['timezone_orlando']
# elif location == 'Denton':
#     timezone_ = config['Timezone']['timezone_denton']



def filename_to_date(input,minute=False):
    ''' Convert ecostress filenames to dates'''

    location, year,shapefile = helper.get_configs()
    if config['Timezone']['timezone_'+location.lower()]:
        timezone_ = config['Timezone']['timezone_'+location.lower()]
    else:
        print("Please enter a valid location or update config.ini file with timezone informations")

    date_part = input.split('_')[3][3:]
    year = date_part[:4]
    time = date_part[-6:]
    day_of_year = date_part[4:-6]

    #convert day of year to date
    # date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(day_of_year) - 1) + datetime.timedelta(hours=int(time[:2]), minutes=int(time[2:4]))
    if minute:
        date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(day_of_year) - 1) + datetime.timedelta(hours=int(time[:2]), minutes=int(time[2:4]))
    
    else:
        date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(day_of_year) - 1) + datetime.timedelta(hours=int(time[:2]), minutes=0)
    date = pytz.utc.localize(date)
    #convert  utc to cdt

    # Convert to Central Daylight Time (CDT) --> For Madison
    #Las Vegas,Nevada (Pacific Daylight time, PDT) 
    # and Orlando, Florida (Eastern Daylight Time, EDT).
    cdt = pytz.timezone(timezone_)
    date_cdt = date.astimezone(cdt)

    # format = "%Y-%m-%d %H:%M:%S"

    # return str(date)
    return str(date_cdt)


def reproject(file_raster, file_csv):
    """
    Function to reproject the shapefile to a desired coordinate

    Example:

        file_raster = str('//uahdata/rgroup/urbanrs/'+
                         'TRANG_DATA/NASA_HAQUAST/'+
                         'final_folder_combine/Madison/ECOSTRESS/'+
                         'geotiff_clipped_stateplane/'+
                         'ECO2LSTE.001_SDS_LST_doy2018259095525_aid0001_clipped_stateplane.tif'
                         )

        file_csv = str('//uahdata/rgroup/urbanrs/'+
                       'TRANG_DATA/NASA_HAQUAST/station_list.csv')



        df_station_reproj = reproject(file_raster,file_csv)
    """
    # read an example of raster file with the coordinate
    rs = rxr.open_rasterio(file_raster)
    des_crs = rs.spatial_ref.crs_wkt

    # read .csv file with coordinates (by default in WGS84)
    df_station = pd.read_csv(file_csv)

    # Transform from .csv to geopandas using latitude and longitude coor
    df_station = gpd.GeoDataFrame(df_station)
    df_station["geometry"] = df_station.apply(
        lambda row: Point(row.longitude, row.latitude), axis=1
    )
    crs_wgs84 = CRS.from_epsg(4326)  # epsg code for WGS84
    df_station.crs = crs_wgs84

    # rerpject to the desired coodinate
    df_station_reproj = df_station.to_crs(des_crs)

    return df_station_reproj


def extract_raster_values(location,year,df_station_reproj, folder_raster, local_time_zone,dataframe,independent_validation=False):
    """
    Function to extract the values of raster for each points
    of the reprojected shapefile
    Example:
        df_station_reproj = reproject(file_raster,file_csv)
        # convert to local time
        # check here for list of local time zone naame
        # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        # column 'TZ indentifier'
        # for example, MAdison, WI has 'America/Chicago' (CST time zone)
        local_time_zone = tz.gettz('America/Chicago')
        folder_raster = str('//uahdata/rgroup/urbanrs/TRANG_DATA/'+
                            'NASA_HAQUAST/final_folder_combine/'+
                            'Madison/ECOSTRESS/geotiff_clipped_stateplane/')

        df_output = extract_raster_values(df_station_reproj, folder_raster, local_time_zone)
    """
    # create a empty list to hold the results #
    output_list = list()
    column_name_list = list()

    # location, year,shapefile = helper.get_configs()
    SAVE_DIRECTORY = os.path.join(HOME_DIRECTORY, f'data/raster_op/{location}/{year}/')



    for file_raster in os.listdir(folder_raster):

        if ".DS_Store" in file_raster:
            continue

        if TIF_FOLDER_NAME not in folder_raster:
            df_station_reproj_copy = df_station_reproj.copy()

        # print(file_raster)
        if TIF_FOLDER_NAME in folder_raster:
            column_name = "value_LST"

        else:
            column_name = "value" + file_raster.split("_")[0]

        # read the raster
        rs = rasterio.open(folder_raster + file_raster)
        print(folder_raster + file_raster)

        # read the first layer of the raster file
        band = rs.read(1)

        # create an empty list to store the raster values output
        value_list = list()

        if TIF_FOLDER_NAME in folder_raster:
            df_station_reproj_copy = df_station_reproj.copy()

        for point in df_station_reproj_copy["geometry"]:

            x = point.xy[0][0]
            y = point.xy[1][0]

            # get indexes of row, col that the point falling over
            # the raster layer
            row, col = rs.index(x, y)
            
            # return in the raster values from the row,col indexes
            
            #TODO : Change this segment later 
            #Temp logic to handle out of bounds
            
            # if row > 680:
            #     row = 680
            # if col > 680:
            #     col = 680
            cnt = 0
            try:
                value = band[row, col]
            
            except IndexError as e:
                value = 0
                cnt += 1
            
            # if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
            #     value = band[row, col]

            # else:
            #     value = 0

            # for building height, there are missing values
            # replace with nan
            if "height" in file_raster:
                if value < 0:
                    value = np.nan

            # append the value to the list
            value_list.append(value)

        # create a new column from the new list of raster values
        # and add to the existing dataframe

        df_station_reproj_copy[column_name] = value_list

        # read the file name of the ecostress file and assign the local time
        if TIF_FOLDER_NAME in folder_raster:
            # read time in UTC
            # time_utc = file_raster.split("_")[3].split("doy")[-1]

            # print(time_utc)
            # local_time = datetime.datetime.strptime(time_utc, "%Y%j%H%M%S")

            # local_time = local_time.replace(tzinfo=local_time_zone)

            # print(local_time)
            hour = file_raster.split(".")[0][-2:]
            df_station_reproj_copy["hour"] = [hour] * len(
                df_station_reproj_copy
            )
            df_station_reproj_copy["filename"] = [file_raster] * len(
                df_station_reproj_copy
            )

        output_list.append(df_station_reproj_copy)
        column_name_list.append(column_name)

    # concat the outputlist to a new output dataframe
    if TIF_FOLDER_NAME not in folder_raster:

        #Feb 4 Change : Updated to False to remove duplicates
        # df_output = reduce(
        #     lambda x, y: pd.merge(x, y, left_index=True, right_index=True), output_list
        # )
        df_output = reduce(
            lambda x, y: pd.merge(x, y, left_index=False, right_index=False), output_list
        )

    else:
        df_output = pd.concat(output_list)

    # export to .csv file

    if TIF_FOLDER_NAME in folder_raster:
        column_names = ["station", "latitude", "longitude", "beg_time", "geometry"] + [
            "value_LST",
            # "localtime",
            "hour",
            "filename",
        ]

        df_output = df_output[column_names]

        if dataframe is not None:
            #because multiple years will have same data
            save_loc = os.path.join(SAVE_DIRECTORY,"../", "ECOSTRESS_values_testing.csv")

            if independent_validation:
                save_loc = os.path.join(SAVE_DIRECTORY,"../", "ECOSTRESS_values_independent.csv")

        else:
            save_loc = os.path.join(SAVE_DIRECTORY, "ECOSTRESS_values.csv")
        
        if not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY)

        df_output.to_csv(save_loc)
        print(f"Ecostress file saved in {save_loc}")

    else:
        column_names = [
            "station",
            "latitude",
            "longitude",
            "beg_time",
            "geometry",
        ] + column_name_list

        df_output = df_output[column_names]

        if dataframe is not None:
            save_loc = os.path.join(SAVE_DIRECTORY,"../", "urban_surface_properties_values_testing.csv")
            
            if independent_validation:
                save_loc = os.path.join(SAVE_DIRECTORY,"../", "urban_surface_properties_values_independent.csv")
        else:
            save_loc = os.path.join(SAVE_DIRECTORY, "urban_surface_properties_values.csv")
            


        df_output.to_csv(save_loc)
        print(f"Urban surface properties file saved in {save_loc}")

    return


def create_station_file(location, year, timezone="America/Chicago", urban_data=False,dataframe=None,independent_validation=False):
    """
    Creates ECOSTRESS files.
    Format of dataframe for test data : station, latitude, longitude
    independent_validation : True if you want to create a separate file for testing for DEnton/Madison testin
    """
    print(f'Location is {location}')
    
    #Return if file already exists
    SAVE_DIRECTORY = os.path.join(HOME_DIRECTORY, f'data/raster_op/{location}/{year}/')
    saves_loc = os.path.join(SAVE_DIRECTORY,"../", "urban_surface_properties_values_testing.csv")
    if os.path.exists(saves_loc) and dataframe is not None and independent_validation is False:
        print("Urban Testing File already exists, returning")
        return
    
    path_name = os.path.join(
        HOME_DIRECTORY,
        "data/processed_data/"
        + location
        + "_"
        + str(year)
        + "/clean_"
        + location
        + "_pws_.csv",
    )
    # file_raster = os.path.join(
    #     HOME_DIRECTORY,
    #     "../ECOSTRESS_and_urban_surface_dataset/"
    #     + location
    #     + "/ECOSTRESS/geotiff_clipped_stateplane/ECO2LSTE.001_SDS_LST_doy2022167193705_aid0001_clipped_stateplane.tif",
    # )

    if 'geotiff_clipped_stateplane' not in TIF_FOLDER_NAME:
        path = os.path.join(
            HOME_DIRECTORY,
            "../ECOSTRESS_and_urban_surface_dataset/"
            + location
            + f'/ECOSTRESS/{TIF_FOLDER_NAME}/',
        )
        temp_file_name = [x for x in os.listdir(path) if '.tif' in x][0]
        # temp_file_name = 'temperature_00.tif'
    else:
        temp_file_name = 'ECO2LSTE.001_SDS_LST_doy2022167193705_aid0001_clipped_stateplane.tif'
    
    file_raster = os.path.join(
        HOME_DIRECTORY,
        "../ECOSTRESS_and_urban_surface_dataset/"
        + location
        + f'/ECOSTRESS/{TIF_FOLDER_NAME}/{temp_file_name}',
    )

    folder_raster = os.path.join(
        HOME_DIRECTORY,
        "../ECOSTRESS_and_urban_surface_dataset/"
        + location
        + f'/ECOSTRESS/{TIF_FOLDER_NAME}/',
    )

    if  dataframe is None:
        df = pd.read_csv(path_name)

    else:
        df = dataframe
        df["beg_time"] = pd.to_datetime("2021-01-01 01:00:00")

    df["beg_time"] = pd.to_datetime(df["beg_time"])

    # saving station file
    df_ = (
        df.groupby(["station", "latitude", "longitude"])
        .count()
        .reset_index()[["station", "latitude", "longitude"]]
    )
    df_["beg_time"] = pd.to_datetime("2021-01-01 01:00:00")
    df_.to_csv(
        os.path.join(
            HOME_DIRECTORY,
            "data/processed_data/"
            + location
            + "_"
            + str(year)
            + "/station_lat_long_final.csv",
        ),
        index=False,
    )
    # df_.to_csv('/Users/gaurav/UAH/temperature_modelling/data/processed_data/'+location+'_'+str(year)+ '/station_lat_long_final.csv', index=False)

    local_time_zone = tz.gettz(timezone)

    # file_raster = f'/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/{location}/ECOSTRESS/geotiff_clipped_stateplane/ECO2LSTE.001_SDS_LST_doy2022167193705_aid0001_clipped_stateplane.tif'
    # folder_raster = f'/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/{location}/ECOSTRESS/geotiff_clipped_stateplane/'

    if urban_data:
        folder_raster = os.path.join(
            HOME_DIRECTORY,
            "../ECOSTRESS_and_urban_surface_dataset/"
            + location
            + "/urban_surface_data/",
        )

    reproj_file = os.path.join(
        HOME_DIRECTORY,
        "data/processed_data/"
        + location
        + "_"
        + str(year)
        + "/station_lat_long_final.csv",
    )
    # print(f'File raster is {file_raster}')
    # print(f'reproj_file  is {reproj_file}')
    if dataframe is None:
        df_station_reproj = reproject(file_raster, reproj_file)
    
    else:
        df_station_reproj = dataframe
        df_station = gpd.GeoDataFrame(df_station_reproj)
        df_station["geometry"] = df_station.apply(
        lambda row: Point(row.longitude, row.latitude), axis=1
    )
    
    # print(df_station_reproj.columns)
    extract_raster_values(location,year, df_station_reproj, folder_raster, local_time_zone,dataframe,independent_validation)


def station_daily_lst_anomaly_means(location=None):
    """This function calculates the lst anomaly values i.e. LST - mean(LST)
    for each station . i.e. hour, stattion, adjusted LST
    """
    if location:
        ecostress_path = os.path.join(
            HOME_DIRECTORY, f"data/raster_op/{location}/ECOSTRESS_values.csv"
        )

    ecostress_path = os.path.join(HOME_DIRECTORY, "data/raster_op/ECOSTRESS_values.csv")
    ecostress_data = pd.read_csv(ecostress_path)
    ecostress_data = ecostress_data[
        [
            "station",
            "latitude",
            "longitude",
            "geometry",
            "value_LST",
            # "localtime",
            "hour",
            "filename",
        ]
    ]

    spatial_means = (
        ecostress_data.groupby(["filename"])
        .mean()
        .reset_index()[["filename", "value_LST"]]
        .rename({"value_LST": "mean_LST"}, axis=1)
    )
    new_ecostress_data = ecostress_data.join(
        spatial_means.set_index("filename"), on="filename"
    )

    # new_ecostress_data[(new_ecostress_data.station == 'KWIWINDS10') & (new_ecostress_data.beg_time == '2022-08-18 00:00:00')]

    new_ecostress_data["localtime"] = pd.to_datetime(new_ecostress_data["localtime"])
    new_ecostress_data["hour"] = new_ecostress_data["localtime"].dt.hour

    new_ecostress_data["adjusted_lst"] = (
        new_ecostress_data["value_LST"] - new_ecostress_data["mean_LST"]
    )

    updated_df = (
        new_ecostress_data.groupby(["station", "hour"])
        .mean()
        .reset_index()[["station", "hour", "adjusted_lst"]]
    )
    # plt.plot(updated_df[updated_df.station=='KWIDANE6'].hour, updated_df[updated_df.station=='KWIDANE6'].adjusted_lst)

    # filling of missing hours

    all_hours = pd.DataFrame({"hour": range(24)})
    unique_stations = updated_df["station"].unique()
    # Create an empty list to store DataFrames
    filled_dfs = []

    for station in unique_stations:
        station_data = updated_df[updated_df.station == station]
        merged_df = all_hours.merge(station_data, on="hour", how="left").fillna(
            {"adjusted_lst": 0}
        )

        if merged_df.loc[0].isna().sum():
            merged_df = merged_df.fillna(method="bfill")

        merged_df = merged_df.fillna(method="ffill")
        merged_df["adjusted_lst"] = merged_df["adjusted_lst"].replace(0, np.nan)
        merged_df["adjusted_lst"] = merged_df["adjusted_lst"].interpolate(
            method="polynomial", order=2
        )  # future expansion

        filled_dfs.append(merged_df)

    result_df = pd.concat(filled_dfs)

    return result_df
