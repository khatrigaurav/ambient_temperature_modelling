#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 location year "
    exit 1
fi

# Assign arguments to variables
location="$1"
year="$2"



# Run the Python scripts with provided arguments
echo " ******** Running weather stations listing ******** "
/Users/gaurav/opt/anaconda3/bin/python downloader/list_weather_stations.py pws

echo " ******** Runnning parallel.py ******** "
/Users/gaurav/opt/anaconda3/bin/python downloader/parallel.py pws

echo " ******** Running data aggregator ******** "
/Users/gaurav/opt/anaconda3/bin/python lib/cleaner.py "$location" "$year"

# echo " ******** Running ECOSTRESS/urban downloader ******** "  #works for denton/madison
# /Users/gaurav/opt/anaconda3/bin/python env_builder.py --location "$location" --year "$year" --month "$month"
