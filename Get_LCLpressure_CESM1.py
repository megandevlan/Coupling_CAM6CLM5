# --- Import libraries --- #  
import numpy as np 
import xarray as xr 
import pickle 

# --- Read in data --- #

# Set directories and file names 
# cesm1start = '/glade/work/mdfowler/data/CESM1/cam5.1_amip_1d_002.cam2.h1.'
cesm1start = '/Users/meganfowler/gp_fuse/cam5.1_amip_1d_002.cam2.h1.'
fileEnd    = '_sfcConditions.nc'
timeName = ['1979-1989', '1990-1999', '2000-2006']


# Read in files and get time as usable format 
file1 = cesm1start+timeName[0]+fileEnd 
sfcDF = xr.open_dataset(file1, decode_times=True)
sfcDF['time'] = sfcDF.indexes['time'].to_datetimeindex()
print('File 1 finished reading in...')

file2  = cesm1start+timeName[1]+fileEnd 
sfcDF2 = xr.open_dataset(file2, decodeTimes=True) 
sfcDF2['time'] = sfcDF2.indexes['time'].to_datetimeindex()
print('File 2 finished reading in...')

file3  = cesm1start+timeName[2]+fileEnd 
sfcDF3 = xr.open_dataset(file3, decodeTimes=True) 
sfcDF3['time'] = sfcDF2.indexes['time'].to_datetimeindex()
print('File 3 finished reading in...')

# Concat in one array 
sfc_full  = xr.concat([sfcDF,  sfcDF2, sfcDF3], dim="time")

# Get lat and lon 
lat = sfc_full.lat.values
lon = sfc_full.lon.values 

# ------- Start computing ----------

# Convert temperature to ˚C 
T_degC = sfc_full.TREFHT.values - 273.15

# Get RH as fraction, not %
RH_frac = sfc_full.RHREFHT.values / 100.0 

# Convert surface pressure to hPa 
PS_hPa = sfc_full.PS.values / 100.0

# Compute dew point and P_LCL 
Td   = np.full([np.shape(T_degC)[0], len(lat), len(lon)], np.nan)  #Saved to be in Kelvin
Plcl = np.full([np.shape(T_degC)[0], len(lat), len(lon)], np.nan)  #Saved to be in hPa 

# Define constants for dew point calculation
c1 = 610.94   # [Pa]
a1 = 7.625
b1 = 243.04   # [degC]

for iT in range(np.shape(T_degC)[0]):
    
    # Compute dewpoint temperature:
    numerator   = b1 * (np.log(RH_frac[iT,:,:]) + ( (a1*T_degC[iT,:,:])/(b1 + T_degC[iT,:,:]) ) )
    denominator = a1 - np.log(RH_frac[iT,:,:]) - ( (a1*T_degC[iT,:,:])/(b1 + T_degC[iT,:,:]) ) 
    
    Td[iT,:,:] = (numerator/denominator) + 273.15   #Convert to K here too 
    
    # Compute pressure level of LCL:
    inner        = ((Tsfc_full.tas[iT,:,:] - Td[iT,:,:]) / 223.15) + 1  # Part inside ()
#    Plcl[iT,:,:] = PS_hPa[iT,:,:]*(inner**(-3.5))
    Plcl[iT,:,:] = PS_hPa[iT,:,:] - (PS_hPa[iT,:,:]*(inner**(-3.5)))    # Want to get distance above sfc in mb 
   
    if (iT % 1000)==0: 
        print('Done with ', (iT/23726)*100, ' % of days')
    
# Looking at Betts (2004), it looks like Plcl is actually the mean depth to cloud base; not just its pressure level
#   So to get the depth of the layer from sfc to cloud bottom in hPa, need to take Plcl - PS. 
# Plcl = PS_hPa - Plcl

# ------- Save out data --------

# Save the array with residaul S.T. and S.M. z-scores in it 
saveDir  = '/Users/meganfowler/Documents/NCAR/Analysis/Coupling_initial/Coupling_CAM6CLM5/processed_data/'
saveFile = 'LCL-pressure-HeightAboveGround_CESM1.p' 

pickle.dump( Plcl, open( saveDir+saveFile, "wb" ), protocol=4 )

print('Finished computing LCL-pressure level successfully and saved pickle file.') 
