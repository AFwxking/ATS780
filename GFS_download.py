#Script for ATS780 Machine Learning for Atmospheric Sciences HW1
#Needed to download GFS data for random forests

#%%
import requests #https://anaconda.org/anaconda/requests
import os
import xarray as xr
import glob
from datetime import datetime, timedelta

# %%

# Specify the local directory where you want to save the file
local_directory = '/mnt/data2/mking/ATS780/GFS_files/'

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)


# %%
start_year = 2022
start_month = 1
start_day = 1
start_hour = 12
start_min = 0

#Set start time with datetime object
sel_time = datetime(start_year, start_month, start_day, start_hour, start_min)

#From datetime object get required strings
sel_year = sel_time.strftime('%Y')
sel_month = sel_time.strftime('%m')
sel_day = sel_time.strftime('%d')
sel_hr = sel_time.strftime('%H')
fcst_time = sel_time + timedelta(hours =3)
fcst_hr = fcst_time.strftime('%H')

#Loop to download files
for idx in range(364):

    #Set url strings
    print('Setting url strings')
    url_1 = 'https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/ds084.1/'
    url_2 = sel_year + '/'
    url_3 = sel_year + sel_month + sel_day + '/'
    url_4 = 'gfs.0p25.'
    url_5 = sel_year + sel_month + sel_day + sel_hr + '.f003.grib2?'
    url_6 = 'var=Temperature_height_above_ground&var=Relative_humidity_height_above_ground&var=Absolute_vorticity_isobaric&var=Geopotential_height_isobaric&var=Relative_humidity_isobaric&var=Temperature_isobaric&var=Vertical_velocity_geometric_isobaric&var=Vertical_velocity_pressure_isobaric&var=Cloud_mixing_ratio_isobaric&var=Graupel_snow_pellets_isobaric&var=Ice_water_mixing_ratio_isobaric&var=Rain_mixing_ratio_isobaric&var=Snow_mixing_ratio_isobaric&var=Total_cloud_cover_isobaric&north=60&west=230&east=300&south=10&horizStride=1&time='
    url_7 = sel_year + '-' + sel_month + '-' + sel_day + 'T' + fcst_hr + ':00:00Z&&accept=netcdf3'

    #Concatenate strings together
    full_url = url_1 + url_2 + url_3 + url_4 + url_5 + url_6 + url_7

    print(full_url)

    #Set file name
    filename_str = 'gfs.0p25.' + sel_year + sel_month + sel_day + sel_hr + '.f003.nc' 
    filename = os.path.join(local_directory, filename_str)

    #Download file
    print('Downloading file...')
    # Send an HTTP GET request to the URL
    response = requests.get(full_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a local file in write binary mode and write the content
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded to {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    #Update time selection
    print('Updating time selection')
    sel_time = sel_time + timedelta(hours=24) 
    fcst_time = sel_time + timedelta(hours =3)
    fcst_hr = fcst_time.strftime('%H')
    sel_year = sel_time.strftime('%Y')
    sel_month = sel_time.strftime('%m')
    sel_day = sel_time.strftime('%d')
    sel_hr = sel_time.strftime('%H')
    print(sel_time, sel_year, sel_month, sel_day, sel_hr)

    print(f'{(idx/360)*100}% complete')

