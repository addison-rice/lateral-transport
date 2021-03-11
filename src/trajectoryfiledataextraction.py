#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads trajectory files and gives a csv file of the endpoint data for easy use.

Trajectory files must have been generated using backtrack_sinking_particles.py
using parcels 2.1.6. Other trajectory files might contain different variable 
names and have different file naming conventions, which will not work in this 
script. You can certainly edit these things to make it work!

Right now it also requires a csv file with columns for the site id and speeds,
called "run_loc", "Sp_6" ... "Sp_1000", with a 1 where a file exists. Other 
configurations will not work with the code as is - and it can get complicated 
to change this structure because of the loops in this script.

Created on Tue Jun 30 15:37:09 2020

@author: addison-rice
"""

import pandas as pd
import numpy as np
import math
from netCDF4 import num2date
from netCDF4 import Dataset as netcdf_dataset

# Read the files

# file path where the nc files are 
data_file_path = '../data/raw/'


# csv with the list of batches and sinking speeds
full_run_list = pd.read_csv(data_file_path+'batch_runs.csv')


# list of sinking speeds and dictionary to connect the csv file with the nc
# file names
all_speeds = ['Sp_6', 'Sp_12', 'Sp_25', 'Sp_50', 'Sp_100', 'Sp_250', 
              'Sp_500', 'Sp_1000']
speeds_dict = {'Sp_6': '6', 'Sp_12': '12', 'Sp_25': '25', 'Sp_50': '50', 
               'Sp_100': '100', 'Sp_250': '250', 'Sp_500': '500', 
               'Sp_1000': '1000'}


# the folder where you want to save the file
output_path = '../data/processed/'

# the name of the csv file output
output_file = 'extracted_trajectory_info.csv'


# functions

def extract_site_info(batch_index):
    """
    Extract site info

    This just reads the csv file to later tell the code what nc files to
    open.

    Parameters
    ----------
    batch_index : numeric
        index of the run_loc from the csv file

    Returns
    -------
    batch_id : string
        run_loc from the csv file
    batch_speeds : list 
        sinking speeds associated with the run_loc

    """
    batch_id = full_run_list['run_loc'][batch_index]
    batch_speeds = []
    for speed in all_speeds:
        if full_run_list[speed][batch_index] == 1:
            batch_speeds.append(speed)
            
    return batch_id, batch_speeds


def open_nc_files(batch_id, speed_list):
    """
    Open .nc files
    
    This code opens the trajectory .nc files specified in the csv. It outputs
    a big list of all the trajectories associated with the batch_id.

    Parameters
    ----------
    batch_id : string
        run_loc from the csv file
    speed_list : list
        sinking speeds associated with the run_loc

    Returns
    -------
    batch_trajectories : list
        all of the nc files associated with the run_loc

    """
    batch_trajectories = []
    batch_id_str = str(batch_id)
    
    for speed in speed_list:
        sp = speeds_dict[speed]
        filename = 'site'+batch_id_str+'_grid_dd30_sp'+sp+'.nc'
        nc_read = netcdf_dataset(data_file_path+filename)
        batch_trajectories.append(nc_read)
        
    return batch_trajectories


def distance(origin, destination):
    """
    Calculate distance
    
    calculates the distance between two lat/lon coordinates in km

    Parameters
    ----------
    origin : list
        latitude, longitude
    destination : list
        latitude, longitude

    Returns
    -------
    d : float
        distance between the two coordinates in km

    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


def find_endpoints(batch_trajectories): 
    """
    Find endpoints
    
    Grabs information from the origin, endpoints, and 150m endpoints from each 
    trajectory.

    Parameters
    ----------
    batch_trajectories : list
        this is the list of nc files read into python returned by the
        open_nc_files function.

    Returns
    -------
    site_lats : list
        trajectory origin latitudes
    site_lons : list
        trajectory origin longitudes
    last_lats : list
        trajectory endpoint latitudes
    last_lons : list
        trajectory endpoint longitudes
    lats_150 : list
        150m water depth latitudes
    lons_150 : list
        150m water depth longitudes
    last_times : list
        endpoint time stamps
    times_150 : list
        150m water depth time stamps
    last_sst : list
        endpoint temperature
    sst_150 : list
        temperature recorded at 150m water depth

    """
    # empty lists to fill
    site_lats = []
    site_lons = []
    last_lats = []
    last_lons = []
    lats_150 = []
    lons_150 = [] 
    last_times = []
    times_150 = []
    last_sst = []
    sst_150 = []
    
    # temporary lists as placeholders
    temp_site_lats = []
    temp_site_lons = []
    temp_lats = []
    temp_lons = []
    temp_lats150 = []
    temp_lons150 = []
    temp_times = []
    temp_times150 = []
    temp_sst = []
    temp_sst150 = []

    for speed in range(len(batch_trajectories)):
        # working with one speed at a time means working with one nc file at
        # a time
        
        # reset temporary lists
        temp_site_lats = []
        temp_site_lons = []
        temp_lats = []
        temp_lons = []
        temp_lats150 = []
        temp_lons150 = []
        temp_times = []
        temp_times150 = []
        temp_sst = []
        temp_sst150 = []

        # extract variables into lists
        lats = batch_trajectories[speed].variables['lat'][:]
        lons = batch_trajectories[speed].variables['lon'][:]
        lats150 = batch_trajectories[speed].variables['lat150'][:]
        lons150 = batch_trajectories[speed].variables['lon150'][:]
        times = batch_trajectories[speed].variables['time'][:]
        ssts = batch_trajectories[speed].variables['temp'][:]
        ssts_150 = batch_trajectories[speed].variables['temp150'][:]

        # if a particle is deleted before time is up, values are masked. 
        # We'd like to get the last valid number.
        for trajectory in range(len(lats)):
            i = -1  # index for the last value
            while np.ma.is_masked(lats[trajectory][i]) is True:
                i -= 1  # if the value is masked, go to one value sooner
                
            j = i  # use j for the 150m values
            while lats150[trajectory][j] > 0:
                # we want the first index where the latitude is recorded.
                # j is actually the last one where it's not recorded, so we
                # extract the information at index j+1
                j -= 1

            # once i and j are determined for a trajectory, we can extract the
            # variables and append them to temporary lists.
            temp_site_lats.append(lats[trajectory][0])
            temp_site_lons.append(lons[trajectory][0])
            temp_lats.append(lats[trajectory][i])
            temp_lons.append(lons[trajectory][i])
            temp_lats150.append(lats150[trajectory][j+1])
            temp_lons150.append(lons150[trajectory][j+1])
            temp_times.append(times[trajectory][i])
            temp_sst.append(ssts[trajectory][i])
            temp_sst150.append(ssts_150[trajectory][j+1])
            temp_times150.append(times[trajectory][j+1])
            
        # after the temporary lists are appended by sinking speed, they
        # are appended to the big lists that are returned by the function.
        # this keeps the structure of being separated by sinking speed.
        site_lats.append(temp_site_lats)
        site_lons.append(temp_site_lons)
        last_lats.append(temp_lats)
        last_lons.append(temp_lons)
        lats_150.append(temp_lats150)
        lons_150.append(temp_lons150)
        last_times.append(temp_times)
        times_150.append(temp_times150)
        last_sst.append(temp_sst)
        sst_150.append(temp_sst150)
    
    return site_lats, site_lons, last_lats, last_lons, lats_150, lons_150,\
        last_times, times_150, last_sst, sst_150


# the lists that will hold the information, and specifying some variable types
lats_lons_dist = []
sp = ''
water_depth = 0.
site_lat_lon = []
final_lat_lon = []
deep_lat_lon = []
temp_final = []
temp_deep = []
time_deep = []
time_surface = []
dist_final = 0.
dist_deep = 0.
trajectory = 0

# the big loop that runs everything
for run_loc in full_run_list.index:
    batch_id, batch_speeds = extract_site_info(run_loc)

    batch_trajectories = open_nc_files(batch_id, batch_speeds)

    site_lats, site_lons, last_lats, last_lons, lats_150, lons_150, \
        last_times, times_150, last_sst, sst_150 \
        = find_endpoints(batch_trajectories)
    
    # loop through for distance and water depth - this should really just be 
    # added to the find_endpoints function
    for speed_index in range(len(batch_speeds)):
        
        nc_data = batch_trajectories[speed_index]
        sp = batch_speeds[speed_index]
        
        batch_water_depth = nc_data.variables['depth0'][:]
        lats = nc_data.variables['lat'][:]
        lons = nc_data.variables['lon'][:]
        lats150 = nc_data.variables['lat150'][:]
        lons150 = nc_data.variables['lon150'][:]
           
        
        for trajectory in range(len(lats)):
        
            site_lat_lon = [site_lats[speed_index][trajectory], 
                            site_lons[speed_index][trajectory]]
    
            final_lat_lon = [last_lats[speed_index][trajectory],
                             last_lons[speed_index][trajectory]]
            deep_lat_lon = [lats_150[speed_index][trajectory],
                            lons_150[speed_index][trajectory]]
            
            water_depth = batch_water_depth[trajectory][1]
            
            # calculate the distance from the starting point for each trajectory
            dist_final = distance(site_lat_lon, final_lat_lon)
            dist_deep = distance(site_lat_lon, deep_lat_lon)
            
    
            # now we can append to the big list
            lats_lons_dist.append([sp, trajectory, water_depth,
                                   site_lat_lon[0], site_lat_lon[1],
                                   final_lat_lon[0], final_lat_lon[1],
                                   deep_lat_lon[0], deep_lat_lon[1],
                                   dist_final, dist_deep,
                                   last_times[speed_index][trajectory],
                                   times_150[speed_index][trajectory],
                                   last_sst[speed_index][trajectory],
                                   sst_150[speed_index][trajectory]])


# create a pandas dataframe to hold site ID, speed, original lat and lon, 
# final lat and lon, 150m lat and lon, water depth, distance to final,  
# distance to 150m, monthly temperatures at the origin and endpoints

cols = ["speed", "trajectory", "water_depth", "site_lat", "site_lon",
        'end_lat', 'end_lon', '150m_lat', '150m_lon', 'dist_final',
        'dist_150m', 'end_time', '150m_time', 'endpt_sst', 'deep_sst']

full_DF = pd.DataFrame(lats_lons_dist, columns=cols)


# convert the times to year, month, and date so we can actually use it
end_times = list(full_DF['end_time'])
times_150m = list(full_DF['150m_time'])

days_end = []
months_end = []
years_end = []
days_deep = []
months_deep = []
years_deep = []

for i in range(len(end_times)):
    end_timestamp = num2date(end_times[i],
                             'seconds since 2000-01-03T12:00:00.000000000')

    days_end.append(end_timestamp.timetuple()[2])
    months_end.append(end_timestamp.timetuple()[1])
    years_end.append(end_timestamp.timetuple()[0])

    try:
        deep_timestamp = num2date(times_150m[i],
                                  'seconds since 2000-01-03T12:00:00.000000000')

        days_deep.append(deep_timestamp.timetuple()[2])
        months_deep.append(deep_timestamp.timetuple()[1])
        years_deep.append(deep_timestamp.timetuple()[0])

    except (ValueError, AttributeError):
        # for whatever reason, this raises errors sometimes
        days_deep.append(np.nan)
        months_deep.append(np.nan)
        years_deep.append(np.nan)

full_DF['days_end'] = days_end
full_DF['months_end'] = months_end
full_DF['years_end'] = years_end
full_DF['days_deep'] = days_deep
full_DF['months_deep'] = months_deep
full_DF['years_deep'] = years_deep

full_DF.to_csv(output_path + output_file, index=False)
