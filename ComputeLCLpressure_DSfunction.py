# Compute pressure level of the LCL (in units of mb above ground) 
# 
# Author: Meg D. Fowler 
# Date:   14 Sept 2020
#
# Inputs: 
#     filesIn --> path to .nc file containing surface information: 
#                Surface pressure (Pa)
#                Surface temperature (typically at reference height, 2m) (K)
#                Surface relative humidity (typically at reference height, 2m) (%)
#                NOTE: If 'filesIn' is an array of strings (paths), data is concatenated 
#                      along time dimension. 
#     PSname --> Name of surface pressure as saved in file
#     TSname --> Name of surface temperature as saved in file 
#     RHname --> Name of RH as saved in file 
#     fileOutName --> file name to save P_lcl into 
# Outputs: 
#     filePath --> Pressure level (in mb) of LCL above the surface 
# 
# ==================================================================================================

def ComputeLCLpressure(DS,PSname,TSname,RHname):
    # --- Import libraries --- #  
    import numpy as np 
    import xarray as xr 
    import pickle 
    
    # Get lat and lon 
    lat = DS.lat.values
    lon = DS.lon.values 
    
    # Print information about time range: 
    print('Data starts at: ', DS.time.values[0])
    print('Data ends at:   ', DS.time.values[-1])
    nT = len(DS.time.values)
          
    # ------- Start computing ----------

    # Convert temperature to ˚C 
    T_degC = DS[TSname].values - 273.15

    # Get RH as fraction, not %
    RH_frac = DS[RHname].values / 100.0 

    # Convert surface pressure to hPa 
    PS_hPa  = DS[PSname].values / 100.0

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
        inner        = ((DS[TSname].values[iT,:,:] - Td[iT,:,:]) / 223.15) + 1  # Part inside ()
    #    Plcl[iT,:,:] = PS_hPa[iT,:,:]*(inner**(-3.5))
        Plcl[iT,:,:] = PS_hPa[iT,:,:] - (PS_hPa[iT,:,:]*(inner**(-3.5)))    # Want to get distance above sfc in mb 
        
        if (iT % round(nT/10))==0: 
            print('Done with ', (iT/nT)*100, ' % of days')
            
    # Add check to set Plcl to zero if it's negative (below the surface pressure)
    Plcl[Plcl<0] = np.nan

    # Looking at Betts (2004), it looks like Plcl is actually the mean depth to cloud base; not just its pressure level
    #   So to get the depth of the layer from sfc to cloud bottom in hPa, need to take Plcl - PS. 
    # Plcl = PS_hPa - Plcl

     
    # ===========================================================================
    
    # Return statement (what to give back to user) 
    return(Plcl)