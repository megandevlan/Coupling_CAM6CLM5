# --- Import libraries --- #  
import comet as cm 
import numpy as np 
import xarray as xr 
import pickle 

# --- Read in data --- #

# Set directories and file names 
dailyDir   = '/Users/meganfowler/Documents/NCAR/Analysis/Coupling_initial/data/day/'
middleName = '_day_CESM2_amip_r10i1p1f1_gn_'
varNames   = ['hfls','hfss','mrso','mrsos','tas','hurs','ps']

# Set time period portion of fileNames 
timeName_flx  = ['19700101-19791231','19800101-19891231', 
                 '19900101-19991231','20000101-20091231','20100101-20150101']

# Read in test file to get lat/lon 
fileName = dailyDir+varNames[0]+middleName+timeName_flx[0]+'.nc'
testDF   = xr.open_dataset(fileName, decode_times=True)

# Get lat and lon 
lat = testDF.lat.values
lon = testDF.lon.values 

# Read in surface properties (T, RH, P) 

for iT in range(len(timeName_flx)):

    # Sfc Temp
    tFile          = dailyDir+varNames[4]+middleName+timeName_flx[iT]+'.nc' # File name
    TsfcDF         = xr.open_dataset(tFile,decode_times=True) 
    TsfcDF['time'] = TsfcDF.indexes['time'].to_datetimeindex()

    # Sfc RH
    rhFile       = dailyDir+varNames[5]+middleName+timeName_flx[iT]+'.nc' # File name
    rhDF         = xr.open_dataset(rhFile,decode_times=True)
    rhDF['time'] = rhDF.indexes['time'].to_datetimeindex() # Convert from cf time (non-standard calendar) to datetime

    # Sfc pressure
    psFile       = dailyDir+varNames[6]+'_CFday_CESM2_amip_r10i1p1f1_gn_'+timeName_flx[iT]+'.nc' # File name
    psDF         = xr.open_dataset(psFile,decode_times=True)
    psDF['time'] = psDF.indexes['time'].to_datetimeindex() # Convert from cf time (non-standard calendar) to datetime

    if iT==0:
        Tsfc_full  = TsfcDF 
        RHsfc_full = rhDF
        Psfc_full  = psDF
    else: 
        Tsfc_full  = xr.concat([Tsfc_full,  TsfcDF], dim="time")
        RHsfc_full = xr.concat([RHsfc_full, rhDF],   dim="time")
        Psfc_full  = xr.concat([Psfc_full,  psDF],   dim="time")

    print('Done with ', timeName_flx[iT])

# --- Now compute dewpoint temperature and pressure of the LCL --- # 

# Convert temperature to ˚C 
T_degC = Tsfc_full.tas.values - 273.15

# Get RH as fraction, not %
RH_frac = RHsfc_full.hurs.values / 100.0 

# Convert surface pressure to hPa 
PS_hPa = Psfc_full.ps.values / 100.0

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
    Plcl[iT,:,:] = PS_hPa[iT,:,:] - (PS_hPa[iT,:,:]*(inner**(-3.5)))    
   
    if (iT % 1000)==0: 
        print('Done with ', (iT/23726)*100, ' % of days')
        
 # Add check to set Plcl to zero if it's negative (below the surface pressure)
Plcl[Plcl<0] = np.nan
    
# Looking at Betts (2004), it looks like Plcl is actually the mean depth to cloud base; not just its pressure level
#   So to get the depth of the layer from sfc to cloud bottom in hPa, need to take Plcl - PS. 
# Plcl = PS_hPa - Plcl


# Save the array with residaul S.T. and S.M. z-scores in it 
saveDir  = '/Users/meganfowler/Documents/NCAR/Analysis/Coupling_initial/Coupling_CAM6CLM5/processed_data/'
saveFile = 'LCL-pressure-HeightAboveGround_1970-2014.p' 

pickle.dump( Plcl, open( saveDir+saveFile, "wb" ), protocol=4 )

print('Finished computing LCL-pressure level successfully and saved pickle file.') 
