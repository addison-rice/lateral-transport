# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom

run with parcels version:  2.1.6.dev156+g0073434
"""

from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,
                     Field, ErrorCode, ParticleFile, Variable)


from datetime import timedelta as delta
from datetime import datetime
import numpy as np
import math
from glob import glob
import sys
import pandas as pd
import dask
import warnings
import xarray as xr
from parcels import version
print('parcels version: ',version)

# To ignore a lot of warnings:
warnings.simplefilter("ignore", category=xr.SerializationWarning)

# Read directories: 
dirread_pal = 'inputs/forAddison/NEMOdata/'
dirread_top = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
dirread_top_bgc = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
# Write directory: 
dirwrite = 'NEMOres/'

# The sinkspeed m/day - run for 6, 12, 25, 50, 100, 250, 500, 1000
sp = 6. 
dd = 30.  # Depth (m) where a particle is stopped
ddd = 150. # Deeper dwelling depth where parameters are recorded


# 1-D arrays with the latitudes and longitudes of the sediment samples: 
latsz = np.array([35.336,36.032,39.726,40.000,40.000,43.545,43.856,43.894,43.811,43.528,43.783,43.767,43.978,44.242,44.381,44.323,44.395,44.184,44.301,43.044,43.821,43.525,43.673,43.796,43.707,38.988,36.746,34.071,42.800,42.900,38.412,35.986,39.800,39.700,39.700,40.000,41.800,42.000,40.400,40.500,40.500,42.100,42.500,42.100,42.200,42.400,42.200,42.700,37.800,43.000,39.800,42.100,42.300,32.800,38.900,39.800,36.700,36.400,38.300,39.100,39.000,36.800,35.800,42.600,39.700,42.600,42.900,42.800,43.000,43.200,42.900,43.000,43.200,43.200,43.200,42.600,43.200,42.700,43.800,43.400,43.000,42.800,43.500,43.200,43.400,43.300,41.380,31.652,38.200,42.480,36.450,35.350,42.380,35.920,42.130,40.220,40.090,38.950,37.900,37.270,37.040,36.140,35.760,35.750,35.230,34.920,34.530,34.470,34.450,33.530,33.450,32.770,32.650,35.700,35.917,36.083,35.320,33.830,34.500,33.417,34.100,33.733,41.505,35.033,33.380,32.040,35.780,36.154,41.310,38.060,36.470,40.350,40.330,39.100,39.400,38.490,42.890,32.733,42.770,42.930,42.400,42.780,36.680,35.730,41.752,34.370,36.500,38.520,44.540,45.090,33.720,35.860,44.870,43.890,39.600,39.910,39.850,39.720,40.230,40.500,40.630,40.760,41.500,41.670,41.500,41.500,41.500,41.500,41.650,41.800,42.000,42.170,42.170,42.170,39.510,39.340,39.640,39.590,39.830,43.470,34.800,36.200,36.190,35.570,38.230,37.580,33.000])
lonsz = np.array([21.660,-01.955,17.862,17.586,17.467,08.838,08.225,08.092,07.874,07.727,07.713,07.586,08.674,08.854,08.904,08.683,08.768,09.146,09.280,09.269,09.584,09.980,10.118,09.312,09.021,04.023,17.718,32.726,05.030,04.970,13.577,-04.750,17.600,18.000,17.000,17.800,16.900,16.700,18.600,18.500,04.000,04.700,05.000,04.000,04.300,04.200,03.800,04.900,26.300,05.200,24.100,03.800,03.600,34.700,02.600,02.200,25.900,27.100,25.100,02.600,02.700,26.600,13.000,03.400,02.200,03.700,04.700,03.700,03.800,04.900,03.500,04.200,04.100,04.700,04.300,03.200,03.700,03.200,13.700,04.200,03.100,03.100,03.900,03.300,04.400,04.800,09.290,34.073,18.030,03.480,-03.889,28.130,03.880,25.270,03.490,25.240,24.610,24.750,26.220,26.190,13.180,-02.620,27.920,27.550,21.470,23.740,31.790,25.670,33.860,32.990,32.580,19.190,34.100,20.717,27.300,26.833,29.020,25.980,23.417,25.017,25.683,27.917,17.971,17.050,24.770,34.210,26.600,-03.261,17.590,17.180,11.490,11.430,13.210,15.050,13.340,14.300,14.790,30.517,04.800,05.100,04.950,04.650,12.280,13.180,11.769,27.150,24.300,04.000,12.530,12.400,23.500,14.110,12.650,13.370,17.180,16.760,18.600,18.780,18.670,18.640,18.330,18.190,17.310,16.240,16.220,16.410,16.660,17.050,17.190,16.620,16.220,16.000,16.500,16.770,17.980,18.280,18.280,17.680,17.830,13.800,27.280,-04.300,-02.850,-03.480,14.060,00.500,23.620])
site_id = 'all'
depth_est = 4500. # Approximate depth of the site, used to calculate maximum time in the water column


assert ~(np.isnan(latsz)).any(), 'locations should not contain any NaN values'
dep = dd * np.ones(latsz.shape)

# 1-D array which contains the release times:
times = np.array([datetime(2009, 12, 25) - delta(days=x) for x in range(0,int(730),5)])

# Here three 1D arrays are created with the same length, which contain the release locations and times of all particles:
time = np.empty(shape=(0)); lons = np.empty(shape=(0)); lats = np.empty(shape=(0))
for i in range(len(times)):
    lons = np.append(lons,lonsz)
    lats = np.append(lats, latsz)
    time = np.append(time, np.full(len(lonsz),times[i])) 
#%%
def set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, bfile, mesh_mask='/scratch/ckehl/experiments/palaeo-parcels/NEMOdata/domain/coordinates.nc'):
    # Set the fieldset
    filenames = { 'U': {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':ufiles},
                'V' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':vfiles},
                'W' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':wfiles},  
                'S' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [tfiles[0]],
                        'data':tfiles},   
                'T' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [tfiles[0]],
                        'data':tfiles}
                }
    if mesh_mask:
        filenames['mesh_mask'] = mesh_mask
    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'T': 'sst',
                 'S': 'sss' }
    # The dimensions that are loaded for all fields. Notice that only U,V,W are in 4D here. 
    # Only the surface values are loaded for the other fields. Bathymetry B is also independent of time
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'T': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'S': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'} } #,
                  # 'B': {'lon': 'glamf', 'lat': 'gphif'}}  # DOESN'T DO CHUNKING CAUSE IT'S A 2D FIELD
    bfiles = {'lon': mesh_mask, 'lat': mesh_mask, 'data': [bfile, ]}
    bvariables = ('B', 'Bathymetry')
    bdimensions = {'lon': 'glamf', 'lat': 'gphif'}
    bchs = False

    # You could use chuncks(see also the mpi tutorial):
    # https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/documentation_MPI.ipynb
    chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 200, 'x': 200}

    # I recommend for this application to use indices instead of chunking, in which you preset the indices that are loaded yourself 
    indices = {'lat':range(1800,2200), 'lon':range(3300, 4000)}

    if mesh_mask:

        fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize=False, indices=indices)
#        fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize=chs)
        # Create the bathymetry field and add it to the fieldset. The B field should not be chuncked.
        Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
        fieldset.add_field(Bfield, 'B')
        # Set the maximum velocities to 10 m/s: 
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset        

#%% The kernels
# The freedom in the kernels is limited, because they are compiled to C in jit mode (to speed up the simulation)
 
def periodicBC(particle, fieldSet, time):
    # Impose the zonal periodic boundary condition  
    if particle.lon > 180:
        particle.lon -= 360        
    if particle.lon < -180:
        particle.lon += 360   
        
#Sink kernel:
def Sink(particle, fieldset, time):
    if(particle.depth>fieldset.dwellingdepth):
         particle.depth = particle.depth + fieldset.sinkspeed * particle.dt # Decrease the particle depth
    else:
        particle.depth = fieldset.surface
        particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.depth150 = fieldset.deepdwellingdepth
        particle.temp150 = fieldset.T[time+particle.dt, fieldset.deepdwellingdepth, particle.lat, particle.lon]
        particle.salin150 = fieldset.S[time+particle.dt, fieldset.deepdwellingdepth, particle.lat, particle.lon]
        particle.lat150 = particle.lat
        particle.lon150 = particle.lon
        particle.delete()

def Age(particle, fieldset, time):
    # Increase the age of the particle
    particle.age = particle.age + math.fabs(particle.dt)  

def DeleteParticle(particle, fieldset, time):
    # Delete the particle (only used when a particle gets out of bounds)
    particle.delete()

def initials(particle, fieldset, time):
    if particle.age==0.: # If the particle is just created (age==0)
        particle.depth = fieldset.B[time, fieldset.surface, particle.lat, particle.lon] # Set the depth at the ocean bottom
        if(particle.depth  > 5800.):
            particle.age = (particle.depth - 5799.)*fieldset.sinkspeed
            particle.depth = 5799.
        # Also keep track of the location where the particle is released:
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth0 = particle.depth

def run_corefootprintparticles(dirwrite,outfile,lonss,latss,dep, depth_est, speed):
    # arrays which contain the paths to the files:
    ufiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05U.nc'))
    vfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05V.nc'))
    wfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05W.nc'))    
    tfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05T.nc'))     
    bfile = dirread_pal + 'domain/bathymetry_ORCA12_V3.3.nc'

    fieldset = set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, bfile, dirread_pal + 'domain/coordinates.nc')    
    #fieldset.add_periodic_halo(zonal=True)  # No halo needed if no periodic boundary conditions are used 
    # Some constants added to the fieldset, which can be used in a kernel
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('deepdwellingdepth',np.float(ddd))
    fieldset.add_constant('sinkspeed', speed/86400.) # m/day is converted to m/s 
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 0.5)

    class Particle(JITParticle): # Contains all properties of the particles
        temp = Variable('temp', dtype=np.float32, initial=np.nan)
        age = Variable('age', dtype=np.float32, initial=0.)
        salin = Variable('salin', dtype=np.float32, initial=np.nan)
        lon0 = Variable('lon0', dtype=np.float32, initial=0.)
        lat0 = Variable('lat0', dtype=np.float32, initial=0.)
        depth0 = Variable('depth0',dtype=np.float32, initial=0.) 
        lon150 = Variable('lon150', dtype=np.float32, initial=0.)
        lat150 = Variable('lat150', dtype=np.float32, initial=0.)
        depth150 = Variable('depth150',dtype=np.float32, initial=0.) 
        temp150 = Variable('temp150', dtype=np.float32, initial=np.nan)
        salin150 = Variable('salin150', dtype=np.float32, initial=np.nan)
       
    # Set the particleset, given the fieldset and the release locations/times 
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=Particle, lon=lonss.tolist(), lat=latss.tolist(), 
                       time = time)

    # Set where the pset is written. Here write_ondelete is used to write particles when they are deleted. Use the outputdt otherwise
    pfile = ParticleFile(dirwrite + outfile, pset, outputdt=delta(days=1))#write_ondelete=True)# outputdt=delta(days=1))

    # These are the kernels that are applied every time step in this order. The AdvectionRK4_3D does the displacement due to 3D flow advection.
    kernels = pset.Kernel(initials) + Sink + Age  + pset.Kernel(AdvectionRK4_3D) + Age
    
    # Set a meximum sinking time based on the depth and speed, add 360 days for the final particle to get to the surface
    maxtime = depth_est/speed + 360.
    
    # Here the actual computation starts. dt<0 here, because the particles are tracked back in time. 
    pset.execute(kernels, runtime=delta(days=maxtime), dt=delta(minutes=-10), output_file=pfile, verbose_progress=True,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

    print('Execution finished sp = '+str(int(speed)))
    
outfile = 'site'+site_id+'_grid_dd'+str(int(dd)) +'_sp'+str(int(sp)) # Name of the output file
run_corefootprintparticles(dirwrite,outfile,lons,lats,dep,depth_est,sp)

