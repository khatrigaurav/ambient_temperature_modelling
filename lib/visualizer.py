"""
Created on : Nov 18, 2023

@author: Gaurav
@version: 1.0

Functions to help in plotting outputs

Functions:

"""

import os
import configparser
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr

from shutil import make_archive
import time

from sklearn import metrics
import plotly.express as pe
import plotly.offline as pyo
import matplotlib.pyplot as plt
from shapely.geometry import Point

from lib import helper as helper


HOME_DIR = os.path.join(os.path.dirname(__file__),'..')     #path of the directory where the data is to be stored i.e. data/raw_data/location_name

grouped_data_path = os.path.join(HOME_DIR,'data')


# Constants and variable paths for raster plots

# ****** Required configs ************************#
# config = configparser.ConfigParser()
# config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
# config.read(config_path)
# location = config['DEFAULT']['location']
# year = config['DEFAULT']['year']
# shapefile = config['DEFAULT']['shapefile']
# ****** Required configs ************************#


def get_path(location):
    '''This function returns the path of the raster file and the boundary file'''

    location, year, shapefile = helper.get_configs()
    
    path_ = os.path.join(HOME_DIR,'..', f'ECOSTRESS_and_urban_surface_dataset/{location}/ECOSTRESS/geotiff_clipped_stateplane/')
    # f'/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/{location}/ECOSTRESS/geotiff_clipped_stateplane/'
    file = os.path.join(path_,os.listdir(path_)[0])
    temperature_data = rxr.open_rasterio(file)
    temperature_data_crs = temperature_data.rio.crs

    SHAPEFILE_PATH = os.path.join(HOME_DIR,'..', f'ECOSTRESS_and_urban_surface_dataset/{location}/shpfile/{shapefile}_UA_mer.shp')
    # f'/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/{location}/shpfile/{shapefile}_UA_mer.shp'
    BOUNDARY_GDF = gpd.read_file(SHAPEFILE_PATH)
    # TARGET_CRS = BOUNDARY_GDF.crs
    # BOUNDARY_GDF_EPS = BOUNDARY_GDF.to_crs(epsg=6879)
    BOUNDARY_GDF_EPS = BOUNDARY_GDF.to_crs(temperature_data_crs)
    SAVE_PATH = os.path.join(HOME_DIR, f'Resources/temperature_plots/{location}')
    # f'/Users/gaurav/UAH/temperature_modelling/Resources/temperature_plots/{location}'

    return temperature_data_crs, BOUNDARY_GDF_EPS, SAVE_PATH


# scatter_data_directory = '/Users/gaurav/UAH/temperature_modelling/Analytics/temp'

# path_ = f'/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/{location}/ECOSTRESS/geotiff_clipped_stateplane/ECO2LSTE.001_SDS_LST_doy2022167193705_aid0001_clipped_stateplane.tif'

#Reading from the raster file to get relevant crs and other details



def plot_(dfx, animation_frame_comp='day_of_year', frame_duration=200, station_name=None, resample=True):
    '''
    dfx                     : Source dataframe
    animation_frame_comp    : defines which column to animate upon (eg. year, month)
    frame_duration          : defines how fast the animation should be
    station_name            : Optional argument to plot a single station

    Usage                   : dp.plot_(df_vegas,'day_of_year',180)
    '''
    # fig = pe.scatter_mapbox(dfx,lat='latitude',lon = 'longitude',animation_frame='day_of_year',color = 'temperature',range_color=[-40,40],hover_data=['station'],height = 610)
    # Making the legend dynamic

    if resample:
        dfx = resample_daily(dfx)

    if station_name:
        dfx = dfx[dfx.station == station_name]

    dfx = dfx.sort_values(by=[animation_frame_comp])
    fig = pe.scatter_mapbox(
        dfx, lat='latitude',
        lon='longitude',
        animation_frame=animation_frame_comp,
        color='temperature',
        hover_data=['station'],
        height=710,
        color_continuous_scale='thermal',
        # width=200
    )

    fig.update_layout(mapbox_style='open-street-map',)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker=dict(size=14, color='black'))
    # fig.update_layouta(updatemenus=dict())

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_duration
    # fig.layout.coloraxis.colorbar.title.text = 'Temperature - °C'

    # fig.show()
    return fig


def get_raster(df_, temp_col='prediction_temps', pixel_size=150):
    ''' This function takes in a dataframe of format : lat , long, 
        temperature and returns a raster of the temperature values
    '''
    location_of_concern = helper.get_context(write=False)['location']
    temperature_data_crs, BOUNDARY_GDF_EPS, SAVE_PATH = get_path(location_of_concern)

    gdf = df_[['latitude', 'longitude', temp_col]]
    gdf['geometry'] = [Point(xy)
                       for xy in zip(gdf['longitude'], gdf['latitude'])]
    # gdf = gpd.GeoDataFrame(
    #     gdf, geometry=gdf['geometry'], crs=temperature_data_crs)
    
    #Jan 30 change :: this is set such that everytime, the final crs would be in lat long format
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gdf['geometry'], crs='epsg:4326')
    
    gdf = gdf.to_crs(temperature_data_crs)
    # gdf = gdf.to_crs('epsg:6520')
    gdf.latitude = gdf.geometry.y
    gdf.longitude = gdf.geometry.x

    xmin, ymin, xmax, ymax = gdf.total_bounds
    rows = int((ymax - ymin) / pixel_size)
    cols = int((xmax - xmin) / pixel_size)

    # Create the raster
    temperature_raster = np.zeros((rows, cols), dtype='float32')

    # Assign temperature values to the raster cells
    for index, row in gdf.iterrows():

        col = int((row['longitude'] - xmin) / pixel_size)
        r = int((ymax - row['latitude']) / pixel_size)

        # Check if indices are within bounds
        if 0 <= r < rows and 0 <= col < cols:
            temperature_raster[r, col] = row[temp_col]

    return temperature_raster, (xmin, xmax, ymin, ymax)


def get_plot(temperature_raster, bounds, cmap='plasma', change_null=False, plot_boundary=True,hour=None,model_name = None,dpi=300,save=True,month=None,zoom_bounds=None,heat_index=False):
    ''' This function takes in a raster which is output of previous function 
        and plots it
        Usage : get_plot(raster1, bounds,cmap = 'coolwarm',change_null=True)

        Input arguments: 
            - temperature raster : Array of predictions
            - bounds : Geographical bounds
            - cmap : Matplotlib colormap for visualization
            - change_null : Boolean flag on whether the null values should be converted to 0 (True) or left as it is for visualization (False)
            - plot_boundary : Boolean flag to determine whether the boundary needs  to be plot or not.
            - dpi : Resolution for saved images (default 300)
            - save : Boolean flag to save the output images.
            - month : Used in creating descriptive plot labels
            - zoom_bounds  : Used for plotting zoomed images (for clear images in large domains) 
            - heat_index : Set true if used for plotting heat index predictions, else False for temperature.

        zoom_bounds=[1.5,1.75]  
    '''

    # np.save('temperature_raster.npy',temperature_raster)

    location_of_concern = helper.get_context(write=False)['location']
    year_of_concern = helper.get_context(write=False)['year']

    #saving the raster
    if save:
        numpy_data_path = os.path.join(grouped_data_path,f'raster_op/{location_of_concern}/{year_of_concern}/numpy_images/{month}')
        scatter_image_path = os.path.join(grouped_data_path,f'raster_op/{location_of_concern}/{year_of_concern}/raster_images/{month}')

        # if os.path.exists(scatter_image_path):
        #     timestamp_obj = datetime.datetime.now()
        #     id_ = str(timestamp_obj.year) + str(timestamp_obj.month).zfill(2) + str(timestamp_obj.day).zfill(2) + str(timestamp_obj.hour).zfill(2) + str(timestamp_obj.minute).zfill(2)
        #     helper.zip_folder(scatter_image_path, f'{scatter_image_path}_{id_}.zip')
        #     print(f'Zipped scatter images in {scatter_image_path}')
        # else:
        #     os.makedirs(scatter_image_path,exist_ok=True)

        os.makedirs(numpy_data_path,exist_ok=True)
        os.makedirs(scatter_image_path,exist_ok=True)
        

        np.save(f'{numpy_data_path}/temperature_raster_{hour}.npy',temperature_raster)

    temperature_data_crs, BOUNDARY_GDF_EPS, SAVE_PATH = get_path(location_of_concern)


    xmin, xmax, ymin, ymax = bounds
    
    if model_name:
        op_path =os.path.join(SAVE_PATH,model_name)
        os.makedirs(op_path,exist_ok=True)

    if change_null:
        temperature_raster[temperature_raster == 0] = np.nan

    plt.figure(figsize=(10,10))

    plt.imshow(temperature_raster, extent=(
        xmin, xmax, ymin, ymax), cmap=cmap, origin='upper')

    if plot_boundary:
        if cmap == 'coolwarm':
            edge_col = 'black'
        else:
            edge_col = 'white'
        BOUNDARY_GDF_EPS.boundary.plot(
            ax=plt.gca(), edgecolor=edge_col, linewidth=1)

    # gdf.plot(ax=plt.gca(), color='red', markersize=1)
    # plt.colorbar(label='Temperature (C)', shrink=0.5)
    if not zoom_bounds:
        if not heat_index:
            cb = plt.colorbar(label='Temperature (C)')
        else:
            cb = plt.colorbar(label='Heat Index (C)')
    # cb.ax.tick_params(labelsize=14)
        cb.ax.yaxis.label.set_size(13)

    if zoom_bounds:
        zoom_bounds_x = float(zoom_bounds[0])
        zoom_bounds_y = float(zoom_bounds[1])
        xrange = xmax - xmin
        yrange = ymax - ymin
        zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax = xmin+xrange/zoom_bounds_x, xmax-xrange/zoom_bounds_x, ymin+ yrange/zoom_bounds_y, ymax-yrange/zoom_bounds_y
        plt.xlim(zoom_xmin, zoom_xmax)
        plt.ylim(zoom_ymin, zoom_ymax)
    
    #reading grouped data for scatter plot
    # grouped_data_filename = os.path.join(grouped_data_path,location_of_concern,'grouped_data',year_of_concern)
    grouped_data_filename = os.path.join(grouped_data_path,'processed_data',f'{location_of_concern}_{year_of_concern}','grouped_data',)
    if month:
        data_file_name = os.path.join(grouped_data_filename,f'grouped_data_filtered_{location_of_concern}_{month}.csv')
    else:
        data_file_name = os.path.join(grouped_data_filename,f'grouped_data_filtered_{location_of_concern}_6.csv')
    
    data = pd.read_csv(data_file_name)

    if location_of_concern == "Madison":
        # data = data[(data.latitude >= 32.83346849047531 ) & (data.latitude <= 33.444736677205995) & (data.longitude >= -97.37843937529293) & (data.longitude <= -96.71208217298359)]
        data = data[(data.latitude >= 42.862070863646345 ) & (data.latitude <= 43.29339264057239) & (data.longitude >= -89.71406385584014) & (data.longitude <= -89.12566010822172)]

    if location_of_concern == "Denton":
        data = data[(data.latitude >= 32.83346849047531 ) & (data.latitude <= 33.444736677205995) & (data.longitude >= -97.37843937529293) & (data.longitude <= -96.71208217298359)]

    if location_of_concern == "LasVegas":
        data = data[(data.latitude >= 35.85152385640185 ) & (data.latitude <= 36.401271128178045) & (data.longitude >= -115.48003237164959) & (data.longitude <= -114.80222299409283)]

    
    if hour is not None:
        data = data[data.hour == hour]
        
        #converting data to same crs as original
        data['geometry'] = [Point(xy)
                       for xy in zip(data['longitude'], data['latitude'])]

        data = gpd.GeoDataFrame(
                data, geometry=data['geometry'], crs='epsg:4326')
        data = data.to_crs(temperature_data_crs)
        data.latitude = data.geometry.y
        data.longitude = data.geometry.x
        plt.scatter(data.longitude,data.latitude,c=data.temperature,s=50,cmap=cmap,edgecolors='black',linewidths=1)
        
        if heat_index:
            plt.title(f'Predicted Heat Index for hour {hour}')
            
        else:
            plt.title(f'Predicted Temperature for hour {hour}')
        if save:
            plt.savefig(f'{scatter_image_path}/scatter_hour_{str(hour).zfill(2)}.jpeg',dpi=dpi)
            print(f'Saved scatter plot in {scatter_image_path}')

    return plt.figure



def plot_feature_importances(test_column_list,model=None,hr=None,bulk_importances=None):
    ''' This function plots the feature importances of the model for RF and XGB
        Used in predictor.ipynb
    '''
    if bulk_importances is not None:
        importances = bulk_importances
        # print(importances)
    else:
        importances = model.feature_importances_
    indices = np.argsort(importances)
    # features = test_data[COL_LIST].columns.to_list()
    features = test_column_list
    plt.figure(figsize=(12,6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='b', align='center')
    # plt.barh(rrf.feature_names_in_, rrf.feature_importances_)
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    if hr:
        plt.title(f'Feature Importances for hour {hr}')
    plt.savefig(os.path.join(HOME_DIR,'temp_files/feature_importances.jpeg'),dpi=700)
    plt.show()
    # plt.savefig(os.path.join(HOME_DIR,'temp_files/feature_importances.jpeg'),dpi=700)


def plot_mean(model_dict,model_name='Gradient Boosting',col_list=None):
    ''' This function plots the mean RMSE and mean predicted and actual temperature
        Used in Modeller.ipynb
        model_dict : Dictionary of format {hour : [prediction_data,error_value,feature_importances]}
        bulk_mode : If True, then 24 models are trained in bulk
        '''
    context_json = helper.get_context(write=False)
    interpolated_hrs = context_json['interpolated_hrs']

    print(f'Time at which model ran : {datetime.datetime.now().strftime("%D %H:%M:%S")}')
    data = model_dict['hourly_values']
    mean_error = data.hourly_rms.mean()

    plot_data = data.groupby('hour').mean()['hourly_rms']
    plot_data.plot(xlabel='Hour',ylabel='RMSE',title=f'Hourly RMSE for {model_name}, RMSE mean: {mean_error:.2f}')
    plt.xlabel('Hour',fontsize=14)
    plt.ylabel('RMSE',fontsize=14)

    plt.axhline(np.mean(mean_error), color='r', linestyle='--',label='Mean RMSE')
    
    ##plotting interpolated hours with scatter dots
    if len(interpolated_hrs) > 0:
        plt.scatter(interpolated_hrs[0],plot_data[interpolated_hrs[0]],color='r',label = 'Interpolated')
        
        for i in interpolated_hrs:
            plt.scatter(i,plot_data[i],color='r')
    plt.legend()
    plt.savefig(os.path.join(HOME_DIR,'temp_files/hourly_rmse.jpeg'),dpi=700)

    data.groupby('hour').mean()[['predicted_temperature',
                                 'true_temperature']].plot(xlabel='Hour',
                                                           ylabel='Temperature in C',
                            title='Predicted vs Actual Temperature for '+model_name,fontsize=11)
    plt.xlabel('Hour',fontsize=14)
    plt.ylabel('Temperature in C',fontsize=14)
    plt.savefig(os.path.join(HOME_DIR,'temp_files/variation.jpeg'),dpi=700)

    try:
        plot_feature_importances(test_column_list=col_list,
                                 bulk_importances=model_dict['feature_importances'])
    except Exception:
        # print(e)
        print('Feature importances not available')


def plot_mean_aggregated(aggregated_data,model_name='Gradient Boosting',col_list=None):
    ''' This function plots the mean RMSE and mean predicted and actual temperature
        Used in Modeller.ipynb
        aggregated_data : Dictionary of format {hour : [prediction_data,error_value,feature_importances]}
        bulk_mode : If True, then 24 models are trained in bulk
        '''
    context_json = helper.get_context(write=False)
    interpolated_hrs = context_json['interpolated_hrs']

    print(f'Time at which model ran : {datetime.datetime.now().strftime("%D %H:%M:%S")}')
    data = aggregated_data
    mean_error = data.hourly_rms.mean()

    # plot_data = data.groupby('hour').mean()['hourly_rms']
    plt.plot(data.hourly_rms,xlabel='Hour',ylabel='RMSE',title=f'Hourly RMSE for {model_name}, RMSE mean: {mean_error:.2f}')
    # data.plot(xlabel='Hour',ylabel='RMSE',title=f'Hourly RMSE for {model_name}, RMSE mean: {mean_error:.2f}')
    plt.xlabel('Hour',fontsize=14)
    plt.ylabel('RMSE',fontsize=14)

    plt.axhline(np.mean(mean_error), color='r', linestyle='--',label='Mean RMSE')
    
    ##plotting interpolated hours with scatter dots
    if len(interpolated_hrs) > 0:
        plt.scatter(interpolated_hrs[0],data.loc[interpolated_hrs[0]],color='r',label = 'Interpolated')
        
        for i in interpolated_hrs:
            plt.scatter(i,data.loc[i],color='r')
    plt.legend()

    data.groupby('hour').mean()[['predicted_temperature',
                                 'true_temperature']].plot(xlabel='Hour',
                                                           ylabel='Temperature in C',
                            title='Predicted vs Actual Temperature for '+model_name,fontsize=11)
    plt.xlabel('Hour',fontsize=14)
    plt.ylabel('Temperature in C',fontsize=14)
    try:
        plot_feature_importances(test_column_list=col_list,
                                 bulk_importances=aggregated_data['feature_importances'])
    except Exception:
        # print(e)
        print('Feature importances not available')
