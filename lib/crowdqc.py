"""
Created on Feb 7

@author: Gaurav
@version: 1.0

Implementation of CrowdQC quality checks.
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.utils.fixes import parse_version, sp_version
from sklearn.linear_model import QuantileRegressor

pd.options.mode.chained_assignment = None  # default='warn'

''' Current flow : clean_missing , level1_check , level2_check, mean_comparison (helper.py)'''
##To do : Mean/Std dev comparision with 3 nearest stations

def clean_stations(df):
    """This function adds new month column, removes stations with missing months"""

    df["month"] = df["beg_time"].apply(lambda x: x.month)
    stations = df.station.unique()
    incomplete_stations = []

    for stat in stations:
        months = df[df.station == stat].month.unique()
        if len(months) < 12:
            incomplete_stations.append(stat)

    print(f"Original Stations : {len(stations)}")
    print(f"Incomplete Stations : {len(incomplete_stations)}")

    complete_stations = list(set(stations) - set(incomplete_stations))
    print(f"Final Stations : {len(complete_stations)}")

    return df, complete_stations

def clean_missing(df, ratio=2000):
    ''' Step 1 of QC cleaning
        This function data with more than 4000 missing observations.
        It then goes ahead and fills null values with linear interpolation
    '''
    print(f'Old Stats : Num of stations {len(df.station.unique())}, Total rows {len(df)}')
    # timediff = timedelta()
    # find num of days between max and min date
    difference = (df.beg_time.max() - df.beg_time.min()).days
    
    if difference <= 32:
        ratio = 560
    elif difference > 32 and difference <= 63:
        ratio = 1200
    
    if difference > 63 and difference <= 93:
        ratio = 1900

    # else:
    #     ratio = 2000
    
    print(f'Difference : {difference} days')
    print(f'Ratio : {ratio}')
    df['month']  = df['beg_time'].dt.month
    valid_stations =  df.groupby(['station','month'])['temperature'].apply(lambda x : x.count() > 560).reset_index()
    valid_stations = valid_stations[valid_stations.temperature == True].station.values

    df = df[df.station.isin(valid_stations)]

    df = df[df['station'].isin(valid_stations)]
    df['temperature'] = df['temperature'].interpolate(method='linear',limit_direction='both')

    print(f'New Stats : Num of stations {len(df.station.unique())}, Total rows {len(df)}')

    ## extension : remove stations with consecutive missing values

    return df




def level_check(dataframe, temp_field, time_field):
    ''' This function simply checks if valid columns are present or not'''

    assert temp_field in dataframe.columns, "Temp column doesn't exist in dataframe"
    assert len(dataframe[temp_field]) > 0, "Empty dataframe given"
    assert time_field in dataframe.columns, f"Invalid Time column : {time_field}"

    print('Level 0 check passed : Valid fields present')

    return True


def level1_check(dataframe, temp_field):
    '''Checks for gross errors in the data i.e. temperature values outside the given range of -40 to 60C'''

    indexes = []
    if max(dataframe[temp_field]) > 60 or min(dataframe[temp_field]) < -40:
        print('QC check 1 failed : Temperature exceeds given range')
        print('#########################')
        print(max(dataframe[temp_field]))
        print(min(dataframe[temp_field]))
        indexes = dataframe.query(
            '{} > 60 or {} < -40'.format(temp_field, temp_field)).index

    else:
        print('QC check 1 passed : Gross Error Test')
        print('#########################')

    # removing invalid indexes
    dataframe = dataframe[~dataframe.index.isin(indexes)]


    return dataframe, indexes


def level2_check(dataset, temp_field, time_field):
    '''This measures spatial consistency L2'''
    #     print(dataset.head())

    time1 = dataset.iloc[0][time_field]
    time2 = dataset.iloc[1][time_field]

    time3 = dataset.iloc[len(dataset)//2][time_field]
    time4 = dataset.iloc[len(dataset)//2 + 1][time_field]

    time5 = dataset.iloc[len(dataset)-2][time_field]
    time6 = dataset.iloc[len(dataset)-1][time_field]

    delta1 = ((datetime.strptime(str(time2), "%Y-%m-%d %H:%M:%S") -
              datetime.strptime(str(time1), "%Y-%m-%d %H:%M:%S")).seconds)/60
    delta2 = ((datetime.strptime(str(time4), "%Y-%m-%d %H:%M:%S") -
              datetime.strptime(str(time3), "%Y-%m-%d %H:%M:%S")).seconds)/60
    delta3 = ((datetime.strptime(str(time6), "%Y-%m-%d %H:%M:%S") -
              datetime.strptime(str(time5), "%Y-%m-%d %H:%M:%S")).seconds)/60
    if delta1 == delta2 == delta3:
        #         print('All time ranges equal')
        print("")

    else:
        print("Time ranges not equal, taking average for QC2 check")

    avg = np.average([delta1, delta2, delta3])
#     print(avg)

    # if 'key' in dataset.columns:
    #     station_key = 'key'
    # else:
    #     station_key = 'station'

    station_key = 'station'

    stations = dataset[station_key].unique()
    invalids = []
    for station in stations:
        dataset_cpy = dataset[dataset[station_key] == station]

        differences = dataset_cpy[temp_field].diff()[1:]

        if avg <= 5:
            invalid_ = differences[differences > 6]
            time_resolution = 5
        if avg > 5 and avg <= 15:
            invalid_ = differences[differences > 10]
            time_resolution = 15
        if avg > 15 and avg <= 30:
            invalid_ = differences[differences > 15]
            time_resolution = 30
        if avg > 30 and avg <= 60:
            invalid_ = differences[differences > 20]
            time_resolution = 60

        if avg > 60:
            print("Temporal resolution is greater than 1 hour")
            time_resolution = 1000

        invalids.append(invalid_)
    if len(invalid_) > 0:
        print(
            f"QC Check 2 failed : Temporal inconsistency found, {avg} minutes range , for {len(invalid_)} rows  ")
        print('#########################')
        return False

    if len(invalid_) == 0:
        print("QC Check 2 passed : Spike Test")
        print('#########################')

    return time_resolution

#     return differences

#     print(invalid_)


def level3_check(dataset, temp_field, resolution):
    ''' This function tests for temporal persistance'''

    # need to manually test this function
    if 'key' in dataset.columns:
        station_key = 'key'
    else:
        station_key = 'station'

    stations = dataset[station_key].unique()
    fi_list = []
    for station in stations:
        dataset_cpy = dataset[dataset[station_key] == station]

        if resolution == 5:
            rollings = 24
            dataset_cpy['rolling_mean'] = dataset_cpy[temp_field].iloc[::-
                                                                       1].rolling(rollings).mean().iloc[::-1]

        elif resolution == 15:
            rollings = 36
            dataset_cpy['rolling_mean'] = dataset_cpy[temp_field].iloc[::-
                                                                       1].rolling(rollings).mean().iloc[::-1]

        elif resolution == 30:
            rollings = 54
            dataset_cpy['rolling_mean'] = dataset_cpy[temp_field].iloc[::-
                                                                       1].rolling(rollings).mean().iloc[::-1]

        elif resolution == 60:
            rollings = 72
            dataset_cpy['rolling_mean'] = dataset_cpy[temp_field].iloc[::-
                                                                       1].rolling(rollings).mean().iloc[::-1]

        else:
            print("Invalid Time resolution")

        failed_indexes = dataset_cpy[(dataset_cpy.rolling_mean ==
                                     dataset_cpy[temp_field]) & ()].index
        fi_list.append(failed_indexes)

        if len(failed_indexes) > 0:
            print(
                f"QC Check 3 failed : {len(failed_indexes)} rows for {station}")
            print(f"\t Failed indexes : {failed_indexes}")

        else:
            print("QC Check 3 passed : Spike Test")



def outlier_check(dataframe, temp_field='temp', time_field='expire_time_gmt'):
    '''Parent wrapper that completes the l1, l2, l3 QC check controls from CrowdQC paper
    key : data_source {wunder, netatmos, lter}
    time_field : column to be used as time
    '''

    l0_status = level_check(dataframe, temp_field, time_field)

    if l0_status:
        # converting temperature to C for wunderground data
        if temp_field == 'temp':
            dataframe['temp'] = (dataframe['temp'] - 32) * (5/9)

        level1_check(dataframe, temp_field)
        resolution = level2_check(dataframe, temp_field, time_field)
        level3_check(dataframe, temp_field, resolution)

# def find_consecutive_indexes(df, consecutive_count=6):
#     temperature_values = df['temperature'].values
#     consecutive_indexes = []

#     for i in range(len(temperature_values) - consecutive_count + 1):
#         if np.all(temperature_values[i:i + consecutive_count] == temperature_values[i]):
#             consecutive_indexes.append(i)

#     return consecutive_indexes

# todo: what to do with incomplete station data



## Functions for data cleaning and ingestion

def level_1_plot(grouped_data,plot_col='temperature'):
    ''' This function plots the temperature pattern for all stations for June 2021
        Used to show initial pattern before cleaning
    '''
    grouped_datax = grouped_data.copy()


    grouped_datax['std_dev'] = grouped_datax.groupby('hour')[plot_col].transform('std')
    plt.figure(figsize=(12,8))
    plt.errorbar(grouped_datax.hour, grouped_datax[plot_col], yerr=grouped_datax.std_dev, fmt='o', ecolor='orange', capthick=2)

    plt.xlabel('Hour')
    plt.title(f'{plot_col} Pattern for all station for June 2021 ')

    return plt


def cleaning_outliers(grouped_data,variation=20):
    ''' First function cleans the outliers from the data
        variation = 20 means that if the temperature is more than 20% of the average of closest stations, then its an outlier
    '''
    gd = grouped_data[['station', 'hour','latitude','longitude','temperature','closest_station_1_temp','closest_station_2_temp','closest_station_3_temp']]
    gd.loc[:,'average_temp'] = gd[['closest_station_1_temp','closest_station_2_temp','closest_station_3_temp']].mean(axis=1)

    gd['variation_closest'] = 100*abs(gd['temperature']-gd['average_temp'])/gd['average_temp']
    flags = gd[gd.variation_closest>variation].groupby('station').count().index.values

    return flags

def clean_missing_data(final_df,print_missing=False,missing_count=20):
    ''' Removing those stations which have a lot of missing data for weeks, months
        By default, it removes stations with less than 20 observations per day
    '''
    missing_stations = []
    for station_id in final_df.station.unique():
        data_segment = final_df[final_df.station==station_id]
        obs = data_segment.resample('D',on='beg_time').mean().reset_index().count()['temperature']
        if obs < missing_count:
            missing_stations.append(station_id)
            
            if print_missing:
                print(f'{station_id} has {obs} observations')

    return missing_stations



def filter_quartiles(grouped_data,feature_column='valueTreefraction',quantiles=[0.02, 0.98]):
    ''' This function filters the data based on quantiles, used to filter out values 
        based on valueTreeFraction and valueImperviousFraction

        quantiles : list of quantiles to be used for filtering
        grouped_data : data that is grouped at hourly level 
        feature_column : column to be used for filtering 
                            eg.(valueTreeFraction, valueImperviousFraction)

        Output : predictions : used for plotting quartile lines
                outlier_dict : Contains list of stations that are outliers for each hour
    '''
    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    predictions = {}
    outlier_dict = {}

    grouped_data_copy = deepcopy(grouped_data)

    for hour in range(24):
        data = {}
        outliers = []
        for q in quantiles:
            gds = grouped_data_copy.query(f"hour=={hour}")
            model = QuantileRegressor(quantile=q,solver=solver)
            data_col = gds[feature_column]
            model.fit(data_col.values.reshape(-1,1), gds.temperature.values)
            temp_preds = model.predict(data_col.values.reshape(-1,1))
            # data.append({q:temp_preds})
            data[q] = temp_preds
            # print(temp_preds)

            if q == min(quantiles):
                outliers.append(gds[gds.temperature<temp_preds.flatten()].station.values)

            if q == max(quantiles):
                outliers.append(gds[gds.temperature>temp_preds.flatten()].station.values)
        predictions[hour] = data
        outlier_dict[hour] = [k for row in outliers for k in row]

    #Plotting example : 
        # for hour in predictions.keys():
        #     if hour in [0,8,16]:
        #         figx = pe.scatter(grouped_data_copy.query("hour==@hour"),x = 'valueTreefraction', y='temperature',color='station',title=f'Temperature Pattern for Normal Stations at {hour} hours')
        #         figx.add_trace(go.Scatter(x=grouped_data_copy.query("hour==@hour").valueTreefraction, y=predictions[hour][quantiles[0]], mode='lines', name='5% line'))
        #         figx.add_trace(go.Scatter(x=grouped_data_copy.query("hour==@hour").valueTreefraction, y=predictions[hour][quantiles[1]], mode='lines', name='95% line'))
        #         figx.show()
    
    del grouped_data_copy
    return predictions, outlier_dict