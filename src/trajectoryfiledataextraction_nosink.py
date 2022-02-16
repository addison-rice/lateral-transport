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


filename = 'siteall_grid_dd30_spno_sink.nc'


# list of sinking speeds and dictionary to connect the csv file with the nc
# file names
all_speeds = ['no_sink']
batch_speeds = ['no_sink']
speeds_dict = {'no_sink':'no_sink'}

# the folder where you want to save the file
output_path = '../data/processed/'

# the name of the csv file output
output_file = 'extracted_trajectory_info_nosink.csv'


# functions


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
    last_times : list
        endpoint time stamps
    last_sst : list
        endpoint temperature

    """
    # empty lists to fill
    site_lats = []
    site_lons = []
    last_lats = []
    last_lons = []
    last_times = []
    last_sst = []
    
    # temporary lists as placeholders
    temp_site_lats = []
    temp_site_lons = []
    temp_lats = []
    temp_lons = []
    temp_times = []
    temp_sst = []

    for speed in range(len(batch_trajectories)):
        # working with one speed at a time means working with one nc file at
        # a time
        
        # reset temporary lists
        temp_site_lats = []
        temp_site_lons = []
        temp_lats = []
        temp_lons = []
        temp_times = []
        temp_sst = []

        # extract variables into lists
        lats = batch_trajectories[speed].variables['lat'][:]
        lons = batch_trajectories[speed].variables['lon'][:]
        times = batch_trajectories[speed].variables['time'][:]
        ssts = batch_trajectories[speed].variables['temp'][:]

        # if a particle is deleted before time is up, values are masked. 
        # We'd like to get the last valid number.
        for trajectory in range(len(lats)):
            temp_site_lats.append(lats[trajectory][0])
            temp_site_lons.append(lons[trajectory][0])
            temp_lats.append(lats[trajectory][30])
            temp_lons.append(lons[trajectory][30])
            temp_times.append(times[trajectory][30])
            temp_sst.append(ssts[trajectory][30])
            
        # after the temporary lists are appended by sinking speed, they
        # are appended to the big lists that are returned by the function.
        # this keeps the structure of being separated by sinking speed.
        site_lats.append(temp_site_lats)
        site_lons.append(temp_site_lons)
        last_lats.append(temp_lats)
        last_lons.append(temp_lons)
        last_times.append(temp_times)
        last_sst.append(temp_sst)
    
    return site_lats, site_lons, last_lats, last_lons, \
        last_times, last_sst


# the lists that will hold the information, and specifying some variable types
lats_lons_dist = []
sp = ''
water_depth = 0.
site_lat_lon = []
final_lat_lon = []
temp_final = []
time_surface = []
dist_final = 0.
trajectory = 0

# the big loop that runs everything
batch_trajectories = [netcdf_dataset(data_file_path+filename)]

site_lats, site_lons, last_lats, last_lons,\
    last_times, last_sst \
    = find_endpoints(batch_trajectories)

# loop through for distance and water depth - this should really just be 
# added to the find_endpoints function
for speed_index in range(len(batch_speeds)):
    
    nc_data = batch_trajectories[speed_index]
    sp = batch_speeds[speed_index]
    
    batch_water_depth = nc_data.variables['depth0'][:]
    lats = nc_data.variables['lat'][:]
    lons = nc_data.variables['lon'][:]
       
    
    for trajectory in range(len(lats)):
    
        site_lat_lon = [site_lats[speed_index][trajectory], 
                        site_lons[speed_index][trajectory]]

        final_lat_lon = [last_lats[speed_index][trajectory],
                         last_lons[speed_index][trajectory]]
        
        water_depth = batch_water_depth[trajectory][1]
        
        # calculate the distance from the starting point for each trajectory
        dist_final = distance(site_lat_lon, final_lat_lon)
        

        # now we can append to the big list
        lats_lons_dist.append([sp, trajectory, water_depth,
                               site_lat_lon[0], site_lat_lon[1],
                               final_lat_lon[0], final_lat_lon[1],
                               dist_final,
                               last_times[speed_index][trajectory],
                               last_sst[speed_index][trajectory]])


# create a pandas dataframe to hold site ID, speed, original lat and lon, 
# final lat and lon, 150m lat and lon, water depth, distance to final,  
# distance to 150m, monthly temperatures at the origin and endpoints

cols = ["speed", "trajectory", "water_depth", "site_lat", "site_lon",
        'end_lat', 'end_lon', 'dist_final',
        'end_time', 'endpt_sst']

full_DF = pd.DataFrame(lats_lons_dist, columns=cols)


# convert the times to year, month, and date so we can actually use it
end_times = list(full_DF['end_time'])

days_end = []
months_end = []
years_end = []

for i in range(len(end_times)):
    end_timestamp = num2date(end_times[i],
                             'seconds since 2000-01-03T12:00:00.000000000')

    days_end.append(end_timestamp.timetuple()[2])
    months_end.append(end_timestamp.timetuple()[1])
    years_end.append(end_timestamp.timetuple()[0])



full_DF['days_end'] = days_end
full_DF['months_end'] = months_end
full_DF['years_end'] = years_end

full_DF.to_csv(output_path + output_file, index=False)
