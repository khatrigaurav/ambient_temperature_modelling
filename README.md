### 1. Installation and Setup
    pip install -r /path/to/requirements.txt
### 2. Folder Structure
1. **/Analytics** : Contains list of notebooks for machine learning and processing
    - Singlepipe.ipynb
    - Combining Ecostress.ipynb

2. **/data**:
    - **/data/raw_data** : Consist of raw data for different regions in the format : raw_data/[location]/pws_data_[location]/[year]/[station_name].csv. Also consists of aggregated monthly data used for final scatter plots.
    - **/data/processed_data** : Consists of following three different files
        - **/processed_data/[location]_[year]/master_[location]_pws_.csv** : Raw aggregated data that consists of all the stations for the location and year.
        - **/processed_data/[location]_[year]/clean_[location]_pws_.csv** : Processed data based on QC control steps that removes outliers.
        - **/processed_data/[location]_[year]/station_lat_long_final.csv** : List of stations used for the year/location.

    - **/data/raster_op** : Consists of final images and numpy prediction arrays. Folder structure is as follows : 
        - **data/raster_op/[location]/[year]/raster_images/[month]** : Output prediction images for location,month, year
        - **data/raster_op/[location]/[year]/numpy_images/[month]** : Numpy prediction arrays for location,month, year
        - **data/raster_op/[location]/[year]/ECOSTRESS_values.csv** : LST measurements for given list of training stations
        - **data/raster_op/[location]/[year]/urban_surface_properties_values.csv** : Urban surface properties for training stations
        - **data/raster_op/[location]/ECOSTRESS_values_testing.csv** : LST measurements for prediction domain
        - **data/raster_op/[location]/urban_surface_properties_values_testing.csv** : Urban surface properties for prediction domain

3. **/downloader** : Wrapper scripts used by download.sh to download raw data using wunderground API.
4. **/lib** : List of helper scripts used to download, clean, build ML models and draw visualizations. Extensively used by notebooks in Analytics folder.
5. **/temp_files** : Folder to store temp files created while running scripts.


### 3. Usage

### 3.1 Downloading Data
1. Edit /downloader/wunder_config.ini based on requirements. All the fields should be in the **"KEY : VALUE"** format. Following fields need to be added/modified based on requirement.
    - **API_KEY** : Needs to be updated every once in a while if you face data download error. In order to get a new API_KEY, go to wunderground website, create an account, add a new device and generate an API key.
    - Location for which data is to be downloaded needs to be added in the following format i.e. **KEY : VALUE**: 
    -   
            ; Madison 
            LOCATION_NAME  : Madison
            LAT1 : 42.86149765986688
            LAT2 : 43.29370869059481
            LONG1  : -89.71449342013024
            LONG2  : -89.12518307562468
        
    - **Timeframe** : At the end of the config file, edit the start and ending dates.
        - START_DATE : 2017-01-01
        - END_DATE : 2017-12-31
        **Note** : You can only have one active location and lat-long ranges in this config file. Other locations need to be commented if not being used.

  2. Once the config file has been updated, go back to the main folder and run the download.sh script via terminal as follows:
        ```
        ./download.sh Madison 2017
This downloads the data for Madison for year 2017 based on the timeframe specified in wunder_config.ini file. Note that the location name and year should match exactly with the terminal command.

#### 3.2 Updating configs for modelling
1. The configurations required for all the remaining steps to follow are to be edited in /lib/config.ini file.
2. Before running models for any location, the config file needs to be updated as follows : 

            location = [name_of_location] [[eg.Orlando]]
            year = [year for which predictions are to be run] [[eg. 2023]]
            shapefile = [shapefile_name]  - this is based on the naming convention used by ECOSTRESS in shapefiles, found in ECOSTRESS_and_urban_surface_dataset/[location]/shpfile/ folder. Example : Madison : Madison_WI, LasVegas: Las_Vegas,Denton: Denton_TX, Orlando : Orlando

            [Timezone]
            timezone_lasvegas = America/Los_Angeles  [[Timezone information, follow available examples to add new locations]]
        
    Example:
        location = Madison
        year = 2023
        shapefile = Madison_WI
        [Timezone]
        timezone_madison = America/Chicago

        
        
#### 3.2 Preparing Ecostress Files for new locations (For new locations only)
1. Initiate this step only if you are running models for a new location
2. Go to /Analytics/Combining Ecostress.ipynb notebook and run it. The notebook combines the LST images, aggregates LST images, interpolates missing hours. Follow the notebook for more details on each of the steps.


#### 3.3 Training ML models and preparing output
1. Run the /Analytics/Singlepipe.ipynb notebook. Each section within the notebook has been commented appropriately. The corresponding modules used from the /lib/ folder has been appropriately commented for their functionalities.