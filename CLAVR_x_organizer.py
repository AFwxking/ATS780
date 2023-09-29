#Script for ATS780 Machine Learning for Atmospheric Sciences HW1
#Gets data from CLAVR_x files and saves only required variables into directory...
#also determines matching downloaded GFS 3 hour forecast files and saves interpolated model values 

#%%
import os
import xarray as xr
import glob
import numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


#Import functions from mking_fns
sys.path.append('/home/mking/Custom_Functions/')
from mking_fns import GOESlonlat2xy_fn, meshgrid_finder, bilinear_interp_fn

def save_to_netcdf(latitude, longitude, cloud_mask, datetime_obj, output_file):
    """
    Save latitude, longitude, cloud mask, and datetime data into a NetCDF file.

    Parameters:
    - latitude: 1D numpy array containing latitude values.
    - longitude: 1D numpy array containing longitude values.
    - cloud_mask: 2D numpy array containing cloud mask values.
    - datetime_obj: datetime object representing the time of the data.
    - output_file: The name of the output NetCDF file.

    Returns:
    - None
    """
    # Create a new NetCDF file in 'w' (write) mode
    with nc.Dataset(output_file, 'w') as ncfile:
        # Define dimensions for latitude, longitude, and time
        lat_dim = ncfile.createDimension('latitude', latitude.shape[0])
        lon_dim = ncfile.createDimension('longitude', longitude.shape[0])
        time_dim = ncfile.createDimension('time', 1)  # One time instance

        # Create variables for latitude, longitude, cloud_mask, and time
        lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
        lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
        cloud_mask_var = ncfile.createVariable('cloud_mask', 'i2', ('time', 'latitude', 'longitude'))
        time_var = ncfile.createVariable('time', 'f8', ('time',))

        # Set the data values for latitude, longitude, cloud_mask, and time
        lat_var[:] = latitude
        lon_var[:] = longitude
        cloud_mask_var[0, :, :] = cloud_mask  # Assuming a single time step here
        time_var[0] = nc.date2num(datetime_obj, units='hours since 1970-01-01 00:00:00', calendar='gregorian')

    print(f"Data saved to {output_file}")


def save_GFS_to_nc(latitude, longitude, isobaric, relative_humidity, vertical_velocity, temperature, absolute_vorticity, cloud_mixing_ratio, total_cloud_cover, datetime_obj, output_file):
    """
    Save latitude, longitude, GFS_data, and datetime data into a NetCDF file.

    Parameters:
    - latitude: 1D numpy array containing latitude values.
    - longitude: 1D numpy array containing longitude values.
    - isobaric: 1D numpy array containing pressure values
    - relative humidity: relative humidity values at each pressure level
    - vertical velocity: vertical velocity values (Pa/s) at each pressure level
    - temperature: temperature at each pressure level
    - absolute vorticity: absolute vorticity at each pressure level
    - datetime_obj: datetime object representing the time of the data.
    - output_file: The name of the output NetCDF file.

    Returns:
    - None
    """
    # Create a new NetCDF file in 'w' (write) mode
    with nc.Dataset(output_file, 'w') as ncfile:
        # Define dimensions for latitude, longitude, and time
        lat_dim = ncfile.createDimension('latitude', latitude.shape[0])
        lon_dim = ncfile.createDimension('longitude', longitude.shape[0])
        pressure_dim = ncfile.createDimension('isobaric',isobaric.shape[0] )
        time_dim = ncfile.createDimension('time', 1) 

        # Create variables for latitude, longitude, cloud_mask, and time
        lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
        lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
        isobaric_var = ncfile.createVariable('isobaric', 'f4', ('isobaric'))
        relative_humidity_var = ncfile.createVariable('relative_humidity', 'f4', ('time', 'isobaric','latitude', 'longitude'))
        vertical_velocity_var = ncfile.createVariable('vertical_velocity', 'f4', ('time', 'isobaric','latitude', 'longitude'))
        temperature_var = ncfile.createVariable('temperature', 'f4', ('time', 'isobaric','latitude', 'longitude'))
        absolute_vorticity_var = ncfile.createVariable('absolute vorticity', 'f4', ('time', 'isobaric','latitude', 'longitude'))
        cloud_mixing_ratio_var = ncfile.createVariable('cloud_mixing_ratio', 'f4', ('time', 'isobaric','latitude', 'longitude'))
        total_cloud_cover_var =  ncfile.createVariable('total_cloud_cover', 'f4', ('time', 'isobaric','latitude', 'longitude'))
        time_var = ncfile.createVariable('time', 'f8', ('time',))

        # Set the data values for latitude, longitude, cloud_mask, and time
        lat_var[:] = latitude
        lon_var[:] = longitude
        isobaric_var[:] = isobaric
        relative_humidity_var[0, :, :, :] = relative_humidity  
        vertical_velocity_var[0, :, :, :] = vertical_velocity
        temperature_var[0,:,:,:] = temperature
        absolute_vorticity_var[0,:,:,:] = absolute_vorticity
        cloud_mixing_ratio_var[0,:,:,:] = cloud_mixing_ratio
        total_cloud_cover_var[0,:,:,:] = total_cloud_cover
        time_var[0] = nc.date2num(datetime_obj, units='hours since 1970-01-01 00:00:00', calendar='gregorian')

    print(f"Data saved to {output_file}")

#%%
#Run through same timeframes as GFS_download script to get correct closest CLAVR_x data

start_year = 2022
start_month = 1
start_day = 1
start_hour = 12 + 3 #Using 3 hour forecast so lets add 3 
start_min = 0

#Set start time with datetime object
sel_time = datetime(start_year, start_month, start_day, start_hour, start_min)

#From datetime object get required strings
sel_year = sel_time.strftime('%Y')
sel_month = sel_time.strftime('%m')
sel_julian_day = sel_time.strftime('%j')
sel_day = sel_time.strftime('%d')
sel_hr = sel_time.strftime('%H')

#Create empty file list
clavrx_flist = []
time_list = []

#Creating variable list to check
var_list = ['cld_height_base_acha','cld_height_acha', 'cloud_mask' ]

#Loop to download files
for idx in range(365):

    if idx > 0:
        #Update time selection
        sel_time = sel_time + timedelta(hours = 24)
        sel_year = sel_time.strftime('%Y')
        sel_julian_day = sel_time.strftime('%j')
        sel_hr = sel_time.strftime('%H')

    #Set directory string
    dir_1 = '/mnt/multilayer/ynoh/GOES16_ABI/clavrx_run_SLIDER/RadC/output/'
    dir_2 = sel_year + sel_julian_day + '/'
    dir_3 = 'clavrx_goes16_'+ sel_year + '_' + sel_julian_day + '_' + sel_hr + '0117.level2.hdf'

    #Combine strings
    full_dir = dir_1 + dir_2 + dir_3

    try: #File may not exist...loading data within try/except
        #Test file
        data_load = xr.open_dataset(full_dir, engine ='netcdf4')

        #Get variables from dataset
        vars = list(data_load.variables)

        #Initialize test variable
        var_test = np.zeros((len(var_list)))

        #Check for variables
        for var_idx in range(len(var_list)):
            if var_list[var_idx] in vars:
                var_test[var_idx] = 1

        #Go to next iteration if missing a variable
        if np.min(var_test) == 0:
            print('Missing variable...going to next sample iteration')
            continue
        
        #Save file directory & time
        print(f'Saving file directory: {full_dir}')
        clavrx_flist.append(full_dir)
        time_list.append(sel_time)

    except ValueError as e:
        # Print error and repeat loop
        print(e)
        continue

    except OSError as e:
        # Print error and repeat loop
        print(e)
        continue

    except KeyError as e:
        #Print error and repeat loop
        print(e)
        continue

    

#%% 
# Set rectilinear grid to interpolate data on
# res = 0.02
# left_lon = -113
# right_lon = -62
# top_lat = 50
# bottom_lat = 15

res = 0.02
left_lon = -100
right_lon = -65
top_lat = 50
bottom_lat = 25

#One dimensional arrays defining longitude and latitude
len_lon = np.round(np.arange(left_lon,right_lon, res),2)
len_lat = np.round(np.arange(bottom_lat, top_lat, res),2)

#Us numpy meshgrid function to create 2d coordinates using lat/lon values
meshlon, meshlat = np.meshgrid(len_lon, len_lat)

#Get location for conus GOES file to use as reference for interpolation of data on to grid
GOES16_CONUS = '/mnt/data2/mking/ATS780/GOES_files/OR_ABI-L1b-RadC-M6C13_G16_s20230911401170_e20230911403557_c20230911403596.nc'
geo_xarray = xr.open_dataset(GOES16_CONUS)

# Get test cloud mask data
test_dataset = xr.open_dataset(clavrx_flist[0], engine='netcdf4')
cloud_mask = test_dataset['cloud_mask'].data
lat = test_dataset['latitude']
lon = test_dataset['longitude']

#Interpolate values of GOES data to rectilinear grid (calculating nearest grid indexes here)
i,j,reflon,reflat, boolean_outside_values = GOESlonlat2xy_fn(meshlon,meshlat,geo_xarray,method = 'nearest', boolean= True)
m,n,_,_,_ = meshgrid_finder(meshlat, meshlon, res, reflat, reflon,method='nearest')

#Interpolating cloud mask data to rectilinear grid
print('Interpolating Cloud Mask Data to new grid')
new_cld_msk = np.empty(meshlat.shape)
new_cld_msk[m,n] = cloud_mask[j,i]

#Pixels that aren't "seen" by satellite replace with nans
new_cld_msk[boolean_outside_values] = np.nan

# Create a figure and axis for the plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Define extent using the Plate Carree projection
extent = [meshlon.min(), meshlon.max(), meshlat.min(), meshlat.max()]

# Use imshow to plot data with the correct coordinate transformation
image = ax.imshow(new_cld_msk, cmap='jet', vmin=0, vmax=3, origin='lower', extent=extent)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Add Figure Title
plt.title(f"CLAVRX Cloud Mask Test", fontsize=10, pad=15)

# Adding border features last to be on top
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', edgecolor='black')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='black')

# Show Colorbar
cbar = plt.colorbar(image, ax=ax, shrink=0.5)

plt.savefig('cloud_mask_test_plot.png')
plt.show()
plt.close()

# %%
#Save data into netcdf files in ATS780 directory 

# Specify the local directory where you want to save the file
local_directory = '/mnt/data2/mking/ATS780/CLAVRX_data/'

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)


for idx in range(len(clavrx_flist)):

    #From datetime object get required strings
    file_year = time_list[idx].strftime('%Y')
    file_month = time_list[idx].strftime('%m')
    file_day = time_list[idx].strftime('%d')
    file_hr = time_list[idx].strftime('%H')

    #Create filename and concatenate to local_directory
    c_file_name = local_directory + 'clavrx_' + file_year + file_month + file_day + file_hr + '.nc'

    #Load data
    data_load = xr.open_dataset(clavrx_flist[idx], engine ='netcdf4')

    #Get needed variables (using cloud mask and lat/lon)
    lat = data_load['latitude'].data
    lon = data_load['longitude'].data
    cld_msk = data_load['cloud_mask'].data

    #Preallocate array in correct shape
    new_cld_msk = np.empty(meshlat.shape)

    #Interpolate values of GOES data to rectilinear grid (calculating nearest grid indexes here)
    i,j,reflon,reflat = GOESlonlat2xy_fn(meshlon,meshlat,geo_xarray,method = 'nearest')
    m,n,_,_,_= meshgrid_finder(meshlat, meshlon, res, reflat, reflon,method='nearest')

    #Interpolating cloud mask data to rectilinear grid
    print('Interpolating Cloud Mask Data to new grid')
    new_cld_msk = np.empty(meshlat.shape)
    new_cld_msk[m,n] = cld_msk[j,i]

    #Save interpolated data
    save_to_netcdf(len_lat, len_lon, new_cld_msk, time_list[idx], c_file_name)


    
# %%
#Based on files that were available for clavrx, interpolate GFS data and save into directory

#Get file list within the local directory for just saved clavrx files
new_clavrx_flist = sorted(glob.glob('/mnt/data2/mking/ATS780/CLAVRX_data/*'))

#Define directory where GFS files were saved
GFS_directory = '/mnt/data2/mking/ATS780/GFS_files/'

# Specify the local directory where we will place interpolated GFS data
processed_GFS_directory = '/mnt/data2/mking/ATS780/processed_GFS_files/'

# Create the local directory if it doesn't exist
os.makedirs(processed_GFS_directory, exist_ok=True)

#Loop through time_list and find the corresponding GFS files 

#Create empty list for GFS files
GFS_flist = []

#Get list of gfs files
gfs_files = sorted(glob.glob(GFS_directory + '*'))

#Loop through time_list and find the corresponding GFS files
for idx in range (len(time_list)):

    #Generate expected file name
    model_time = time_list[idx] - timedelta(hours=3)
    file_year = model_time.strftime('%Y')
    file_month = model_time.strftime('%m')
    file_day = model_time.strftime('%d')
    file_hr = model_time.strftime('%H')

    expected_file_name = 'gfs.0p25.' + file_year + file_month + file_day + file_hr + '.f003.nc'
    full_path_expected_file_name = os.path.join(GFS_directory, expected_file_name)

    #Check if filename exists in GFS_directory
    if full_path_expected_file_name in gfs_files:
        GFS_flist.append(os.path.join(GFS_directory,expected_file_name))

#Loop through each GFS file, interpolate to match grid, and then save into netcdf
for idx in range (len(time_list)):

    #From datetime object get required strings
    file_year = time_list[idx].strftime('%Y')
    file_month = time_list[idx].strftime('%m')
    file_day = time_list[idx].strftime('%d')
    file_hr = time_list[idx].strftime('%H')

    #Create filename and concatenate to local_directory
    gfs_file_name = processed_GFS_directory + 'GFS_' + file_year + file_month + file_day + file_hr + '.nc'

    #Check if file exists before interpolating...if there already...skip it
    if os.path.exists(gfs_file_name):
        print(f'{gfs_file_name} already exists...skipping and going to next file.')
        continue

    #Load GFS data
    GFS_load = xr.open_dataset(GFS_flist[idx])
    GFS_lon_data = GFS_load['longitude'].data
    GFS_lat_data = GFS_load['latitude'].data
    isobaric_data_full = GFS_load['isobaric'].data/100 #Converting to mb
    isobaric_bool = (isobaric_data_full >= 100) #Create boolean index that has pressure coordinate at or below 100mb in height (anything above probably no cloud)
    isobaric_data = isobaric_data_full[isobaric_bool] #Update isobaric coordinate
    isobaric1 = GFS_load['isobaric1'].data/100 #Converting alternate pressure coordinates to mb
    isobaric1_bool = (isobaric1 >= 100) #Create boolean index that has pressure coordinate at or below 100mb
    vertical_velocity_data = np.squeeze(GFS_load['Vertical_velocity_pressure_isobaric'].data)
    vertical_velocity_data = vertical_velocity_data[isobaric_bool, :, :] #Eliminate values higher than 100mb
    relative_humidity_data = np.squeeze(GFS_load['Relative_humidity_isobaric'].data)
    relative_humidity_data = relative_humidity_data[isobaric_bool, :, :] #Eliminate values higher than 100mb
    temperature_data = np.squeeze(GFS_load['Temperature_isobaric'].data)
    temperature_data = temperature_data[isobaric_bool, :, :] #Eliminate values higher than 100mb
    absolute_vorticity_data = np.squeeze(GFS_load['Absolute_vorticity_isobaric'].data)
    absolute_vorticity_data = absolute_vorticity_data[isobaric_bool, :, :] #Eliminate value higher than 100mb
    cloud_mixing_ratio_data = np.squeeze(GFS_load['Cloud_mixing_ratio_isobaric'].data) * 1000 #Get in grams/kg...uses isobaric1 coordinate...same as updated isobaric_data created above
    cloud_mixing_ratio_data = cloud_mixing_ratio_data[isobaric1_bool, :, :] #Eliminate values above 100mb
    total_cloud_cover_data = np.squeeze(GFS_load['Total_cloud_cover_isobaric'].data)
    total_cloud_cover_data = total_cloud_cover_data[isobaric1_bool, :, :]

    #Subtract 360 from longitudes that should be negative (i.e. west of prime meridian)
    GFS_lon_data[GFS_lon_data > 180] = GFS_lon_data[GFS_lon_data > 180] - 360

    #Isolating GFS coordinates to only lat/lon associated with meshgrid
    GFS_lon_1d_index = (GFS_lon_data >= np.round(np.nanmin(meshlon))-1) & (GFS_lon_data <= np.round(np.nanmax(meshlon))+1)
    GFS_lat_1d_index = (GFS_lat_data >= np.round(np.nanmin(meshlat))-1) & (GFS_lat_data <= np.round(np.nanmax(meshlat))+1)
    GFS_lon_1d = GFS_lon_data[GFS_lon_1d_index]
    GFS_lat_1d = GFS_lat_data[GFS_lat_1d_index]

    #Creating isolated GFS meshgrid
    iso_lon_mesh, iso_lat_mesh = np.meshgrid(GFS_lon_1d, GFS_lat_1d)

    #Creating meshgrid of full GFS coordinates
    GFS_lon_mesh, GFS_lat_mesh = np.meshgrid(GFS_lon_data, GFS_lat_data)

    #Creating boolean index that only has values within isolated meshgrid
    GFS_index = ((GFS_lon_mesh >= np.round(np.nanmin(meshlon))-1) & (GFS_lon_mesh <= np.round(np.nanmax(meshlon))+1) & (GFS_lat_mesh >= np.round(np.nanmin(meshlat))-1) & (GFS_lat_mesh <= np.round(np.nanmax(meshlat))+1))

    #Using that index to create isolated variables from GFS
    vertical_velocity_data = vertical_velocity_data[:, GFS_index]
    relative_humidity_data = relative_humidity_data[:, GFS_index]
    temperature_data = temperature_data[:, GFS_index]
    absolute_vorticity_data = absolute_vorticity_data[:,GFS_index]
    cloud_mixing_ratio_data = cloud_mixing_ratio_data[:,GFS_index]
    total_cloud_cover_data = total_cloud_cover_data[:,GFS_index]

    #Using shape of isolated meshgrid to reshape isolated variables
    vertical_velocity_data = np.resize(vertical_velocity_data,(np.shape(isobaric_data)[0],np.shape(iso_lon_mesh)[0], np.shape(iso_lon_mesh)[1]))
    relative_humidity_data  = np.resize(relative_humidity_data,(np.shape(isobaric_data)[0],np.shape(iso_lon_mesh)[0], np.shape(iso_lon_mesh)[1]))
    temperature_data  = np.resize(temperature_data,(np.shape(isobaric_data)[0],np.shape(iso_lon_mesh)[0], np.shape(iso_lon_mesh)[1]))
    absolute_vorticity_data  = np.resize(absolute_vorticity_data,(np.shape(isobaric_data)[0],np.shape(iso_lon_mesh)[0], np.shape(iso_lon_mesh)[1]))
    cloud_mixing_ratio_data = np.resize(cloud_mixing_ratio_data,(np.shape(isobaric_data)[0],np.shape(iso_lon_mesh)[0], np.shape(iso_lon_mesh)[1]))
    total_cloud_cover_data = np.resize(total_cloud_cover_data,(np.shape(isobaric_data)[0],np.shape(iso_lon_mesh)[0], np.shape(iso_lon_mesh)[1]))


    #Interpolating GFS data to meshgrid
    print('Working on interpolation')
    new_vertical_velocity = np.empty((np.shape(isobaric_data)[0], np.shape(meshlon)[0], np.shape(meshlon)[1]))
    new_relative_humidity = np.empty((np.shape(isobaric_data)[0], np.shape(meshlon)[0], np.shape(meshlon)[1]))
    new_temperature = np.empty((np.shape(isobaric_data)[0], np.shape(meshlon)[0], np.shape(meshlon)[1]))
    new_absolute_vorticity = np.empty((np.shape(isobaric_data)[0], np.shape(meshlon)[0], np.shape(meshlon)[1]))
    new_cloud_mixing_ratio = np.empty((np.shape(isobaric_data)[0], np.shape(meshlon)[0], np.shape(meshlon)[1]))
    new_total_cloud_cover = np.empty((np.shape(isobaric_data)[0], np.shape(meshlon)[0], np.shape(meshlon)[1]))
    for pres_index in range(np.shape(isobaric_data)[0]):
        new_vertical_velocity[pres_index,:,:] = bilinear_interp_fn(iso_lon_mesh,iso_lat_mesh, vertical_velocity_data[pres_index,:,:], meshlon, meshlat)
        new_relative_humidity[pres_index,:,:] = bilinear_interp_fn(iso_lon_mesh,iso_lat_mesh, relative_humidity_data[pres_index,:,:], meshlon, meshlat)
        new_temperature[pres_index,:,:] = bilinear_interp_fn(iso_lon_mesh,iso_lat_mesh, temperature_data[pres_index,:,:], meshlon, meshlat)
        new_absolute_vorticity[pres_index,:,:] = bilinear_interp_fn(iso_lon_mesh,iso_lat_mesh, absolute_vorticity_data[pres_index,:,:], meshlon, meshlat)
        new_cloud_mixing_ratio[pres_index,:,:] = bilinear_interp_fn(iso_lon_mesh,iso_lat_mesh, cloud_mixing_ratio_data[pres_index,:,:], meshlon, meshlat)
        new_total_cloud_cover[pres_index,:,:] = bilinear_interp_fn(iso_lon_mesh,iso_lat_mesh, total_cloud_cover_data[pres_index,:,:], meshlon, meshlat)
        print(str(round((((pres_index + 1)/np.shape(isobaric_data)[0]))*100,0))+'% complete...',end='\r')

    #Save interpolated GFS variables to netcdf
    save_GFS_to_nc(len_lat, len_lon, isobaric_data, new_relative_humidity, new_vertical_velocity, new_temperature, new_absolute_vorticity, new_cloud_mixing_ratio, new_total_cloud_cover, time_list[idx], gfs_file_name)

    
    #Make plot if first iteration
    if idx == 0:

        cld_cover_test = np.max(new_total_cloud_cover, axis = 0)

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

        # Define extent using the Plate Carree projection
        extent = [meshlon.min(), meshlon.max(), meshlat.min(), meshlat.max()]
        # extent = [GFS_lon_1d.min(), GFS_lon_1d.max(), GFS_lat_1d.min(), GFS_lat_1d.max()]

        # Use imshow to plot data with the correct coordinate transformation
        image = ax.imshow(cld_cover_test, cmap='jet', vmin=0, vmax=100, origin='lower', extent=extent)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Add Figure Title
        plt.title(f"GFS Total Cloud Cover Plot Test", fontsize=10, pad=15)

        # Adding border features last to be on top
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='black')

        # Show Colorbar
        cbar = plt.colorbar(image, ax=ax, shrink=0.5)

        plt.savefig('GFS_data_plot.png')
        plt.show()
        plt.close()


# %%

#Get Clavrx files that will be used as the 3hr persistance to use as baseline

#Create empty file list
persistance_clavrx_flist = []

#Creating variable list to check
var_list = ['cld_height_base_acha','cld_height_acha', 'cloud_mask' ]

#Loop to find persistance files
for idx in range (len(time_list)):

    #Update time selection
    sel_time = time_list[idx] - timedelta(hours=3)
    sel_year = sel_time.strftime('%Y')
    sel_julian_day = sel_time.strftime('%j')
    sel_hr = sel_time.strftime('%H')
        
    #Set directory string
    dir_1 = '/mnt/multilayer/ynoh/GOES16_ABI/clavrx_run_SLIDER/RadC/output/'
    dir_2 = sel_year + sel_julian_day + '/'
    dir_3 = 'clavrx_goes16_'+ sel_year + '_' + sel_julian_day + '_' + sel_hr + '0117.level2.hdf'

    #Combine strings
    full_dir = dir_1 + dir_2 + dir_3

    try: #File may not exist...loading data within try/except
        #Test file
        data_load = xr.open_dataset(full_dir, engine ='netcdf4')

        #Get variables from dataset
        vars = list(data_load.variables)

        #Initialize test variable
        var_test = np.zeros((len(var_list)))

        #Check for variables
        for var_idx in range(len(var_list)):
            if var_list[var_idx] in vars:
                var_test[var_idx] = 1

        #Go to next iteration if missing a variable
        if np.min(var_test) == 0:
            print('Missing variable...going to next sample iteration')
            continue
        
        #Save file directory & time
        print(f'Saving file directory: {full_dir}')
        persistance_clavrx_flist.append(full_dir)

    except ValueError as e:
        # Print error and repeat loop
        print(e)
        persistance_clavrx_flist.append('No File')
        continue

    except OSError as e:
        # Print error and repeat loop
        print(e)
        persistance_clavrx_flist.append('No File')
        continue

    except KeyError as e:
        #Print error and repeat loop
        print(e)
        persistance_clavrx_flist.append('No File')
        continue

#Find where there are no files
def indexes(iterable, obj):
    return (index for index, elem in enumerate(iterable) if elem == obj)

nofile_indexes = indexes(persistance_clavrx_flist, 'No File')
nofile_indexes = (list(nofile_indexes))

# Specify the local directory where the interpolated GFS data resides
processed_GFS_directory = '/mnt/data2/mking/ATS780/processed_GFS_files/'

# Specify the local directory where the interpolated CLAVRx data resides
clavrx_directory = '/mnt/data2/mking/ATS780/CLAVRX_data/'

#Get the sorted file list in each directory
final_clavrx_flist = sorted(glob.glob(clavrx_directory + '*'))
final_GFS_flist = sorted(glob.glob(processed_GFS_directory + '*'))

#Delete files that don't have persistance file
for nofile_idx in nofile_indexes:
    file_path = final_clavrx_flist[nofile_idx]
    try:
        os.remove(file_path)
        print(file_path)
    except OSError as e:
        print(f"Error: {e}")

    file_path = final_GFS_flist[nofile_idx]

    try:
        os.remove(file_path)
        print(file_path)
    except OSError as e:
        print(f"Error: {e}")

#Delete "No File" instances from persistance_clavrx_flist
nofile_indexes.sort(reverse=True)  # Sort in reverse order to avoid index issues
for i in nofile_indexes:
    del persistance_clavrx_flist[i]
    del time_list[i]

# %%
#With Persistance Clavrx Files List...save and interpolate data to process in homework script later...

# Specify the local directory where you want to save the file
local_directory = '/mnt/data2/mking/ATS780/Persist_CLAVRX_data_/'

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)


for idx in range(len(persistance_clavrx_flist)):

    #From datetime object get required strings
    sel_time = time_list[idx] - timedelta(hours=3)
    file_year = sel_time.strftime('%Y')
    file_month = sel_time.strftime('%m')
    file_day = sel_time.strftime('%d')
    file_hr = sel_time.strftime('%H')
    
    #Create filename and concatenate to local_directory
    c_file_name = local_directory + 'clavrx_' + file_year + file_month + file_day + file_hr + '.nc'

    #Load data
    data_load = xr.open_dataset(clavrx_flist[idx], engine ='netcdf4')

    #Get needed variables (using cloud mask and lat/lon)
    lat = data_load['latitude'].data
    lon = data_load['longitude'].data
    cld_msk = data_load['cloud_mask'].data

    #Preallocate array in correct shape
    new_cld_msk = np.empty(meshlat.shape)

    #Interpolate values of GOES data to rectilinear grid (calculating nearest grid indexes here)
    i,j,reflon,reflat = GOESlonlat2xy_fn(meshlon,meshlat,geo_xarray,method = 'nearest')
    m,n,_,_,_= meshgrid_finder(meshlat, meshlon, res, reflat, reflon,method='nearest')

    #Interpolating cloud mask data to rectilinear grid
    print('Interpolating Cloud Mask Data to new grid')
    new_cld_msk = np.empty(meshlat.shape)
    new_cld_msk[m,n] = cld_msk[j,i]

    #Save interpolated data
    save_to_netcdf(len_lat, len_lon, new_cld_msk, sel_time, c_file_name)
# %%
