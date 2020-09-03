# Modeled *strongly* after CoMeT script, ctp_hilow.f90 by Ahmed Tawfik. 
# Only difference is that this is re-written for python. 
# See details of CoMeT: http://www.coupling-metrics.com/ 
#
# Date: 2 Sep 2020 
# Author: Meg D. Fowler 

# Details: 
#   The script is built to work on one sounding - that is, one point in space and one observation time. 
#   All data should be passed as a dataframe, with any missing values set to NaN. 
#   NOTE: for profiles, the data should be ordered to begin at the surface and descend accordingly. 
# Required inputs:
#   Temperature profile [K]
#   Specific humidity profile [kg/kg]
#   Pressure profile [Pa]
#   Surface temperature [K]
#   Surface specific humidity [kg/kg]
#   Surface pressure [Pa] 
# Outputs: 
#   CTP --> Convective triggering potential [J/kg]
#   HI_low --> low-level humidity index     [K] 

# ==================================================================================================

def CTP_hilow(Tlev, Qlev, Plev, Tsfc, Qsfc, Psfc):
    

    # Import libraries
    import numpy as np 
    import pandas as pd

    # ------- Set constants ----------------------------------------------------

    # Define constants for getting CTP and HI_low
    Rd    = 287.04 
    cp    = 1005.7
    R_cp  = Rd/cp
    C2K   = 273.15
    Lv    = 2.5e6
    Lv_cp = Lv/cp
    Rv    = 461.5 
    grav  = 9.81
    ep    = 0.622

    # Define constants for computing dewpoint temperature 
    A = 610.8
    B = 237.3
    C = 17.2693882

    # Define constants for computing saturation specific humidity 
    t0     = 273.15
    ep     = 0.622 
    es0    = 6.11 
    a      = 17.269
    b      = 35.86
    onemep = 1.0 - ep

    # Define number of segments to use in calculation from 100-300mb above ground 
    nsegments = 20

    # ------------------------------------------------------------------------------


    # ----- Run a few checks (but most of this is unfortunately on the user ) ------- 

    # Check that pressure levels aren't greater than the surface pressure 
    iProblem = np.where(Plev > Psfc)[0]

    if len(iProblem)>0:
        print('***** ERROR: lowest pressure level > surface pressure *****')

    # Check ordering of Plev: 
    if Plev[0]<Plev[-1]:
        print('***** ERROR: pressure levels should be reversed *****') 

    # Check units of pressure 
    if Psfc<=2000.0: 
        print('**** ERROR: pressures should be in Pa, not hPa *****') 

    # ---------------------------------------------------------------------------------


    # ------ Get 50, 100, 150, and 300 mb levels above the ground --------------------
    p50   =   Psfc - 5000.0
    p100  =   Psfc - 10000.0
    p150  =   Psfc - 15000.0
    p300  =   Psfc - 30000.0
    # --------------------------------------------------------------------------------
    
    
    # -- Get index above and below each of the desired pressure levels (50,100,150,300) -- 
    # Based on pressure leves being oriented in descending order, the points of interst should be the 
    # last positive value of Plev - pNN; the first negative value (and thus the level above it) should be 
    # the next index. 
    i50  = np.where((~np.isnan(Plev)) & (Plev - p50 >= 0))[0]   
    lo50 = i50[-1]
    up50 = i50[-1]+1

    i100  = np.where((~np.isnan(Plev)) & (Plev - p100 >= 0))[0]
    lo100 = i100[-1]
    up100 = i100[-1]+1

    i150  = np.where((~np.isnan(Plev)) & (Plev - p150 >= 0))[0]
    lo150 = i150[-1]
    up150 = i150[-1]+1

    i300   = np.where((~np.isnan(Plev)) & (Plev - p300 >= 0))[0]
    lo300 = i300[-1]
    up300 = i300[-1]+1
    # ----------------------------------------------------------------------------------
    
    
    # ================= Low-level humidity calculation =================================

    # From fortran code: "Perform linear interpolation to extract each value at desired level. 
    # This is done by finding the y-intercept equal to the pressure level minus the desired 
    # level (basically find the temp and dew point corresponding to the level)

    x_up    =   Plev[up50]-p50
    x_lo    =   Plev[lo50]-p50
    y_up    =   Tlev[up50]
    y_lo    =   Tlev[lo50]
    temp50  =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    y_up    =   Qlev[up50]
    y_lo    =   Qlev[lo50]
    qhum50  =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    x_up    =   Plev[up150]-p150
    x_lo    =   Plev[lo150]-p150
    y_up    =   Tlev[up150]
    y_lo    =   Tlev[lo150]
    temp150 =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    y_up    =   Qlev[up150]
    y_lo    =   Qlev[lo150]
    qhum150 =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up
    
    # ------------- Calculate dew point temperature [K] ------------------------------
    e50    =  (qhum50*(p50/1e2))/(0.622+0.378*qhum50)  
    e50    =  e50*1e2
    tdew50 =  ( (np.log(e50/A)*B) / (C-np.log(e50/A)) ) + 273.15

    e150    =  (qhum150*(p150/1e2))/(0.622+0.378*qhum150)  
    e150    =  e150*1e2
    
    tdew150 =  ( (np.log(e150/A)*B) / (C-np.log(e150/A)) ) + 273.15
    
    # ---------------- Calculate HI_low variable -----------------------------------
    HI_low = (temp50-tdew50) + (temp150 - tdew150)
    
    
    # ============================ CTP calculation ==================================== 
    
    # Interpolation as above 
    x_up    =   Plev[up100]-p100
    x_lo    =   Plev[lo100]-p100
    y_up    =   Tlev[up100]
    y_lo    =   Tlev[lo100]
    temp100 =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    y_up    =   Qlev[up100]
    y_lo    =   Qlev[lo100]
    qhum100 =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    x_up    =   Plev[up300]-p300
    x_lo    =   Plev[lo300]-p300
    y_up    =   Tlev[up300]
    y_lo    =   Tlev[lo300]
    temp300 =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    y_up    =   Qlev[up300]
    y_lo    =   Qlev[lo300]
    qhum300 =   y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

    # ------ Chop up integration into nsegments from 100mb to 300 mb above ground -----
    p_increment = (p100-p300)/nsegments 
    p_old       = p100
    tseg_old    = temp100
    tpar_old    = temp100
    qseg_old    = qhum100
    CTP         = 0.0
    
    for nn in range(nsegments+1): 
        # Pressure increment between defined levels (Pa)
        p_segment = p100 - (p_increment * nn)
        iMatch    = np.where((~np.isnan(Plev)) & (Plev > p_segment))[0]
        ilo       = iMatch[-1]
        iup       = iMatch[-1]+1

        # Perform another linear interpolation to get T and Q at the increment 
        x_up    =   Plev[iup] - p_segment
        x_lo    =   Plev[ilo] - p_segment
        y_up    =   Tlev[iup]
        y_lo    =   Tlev[ilo]
        t_segment = y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

        y_up    =   Qlev[iup]
        y_lo    =   Qlev[ilo]
        q_segment = y_up - ((y_up-y_lo)/(x_up-x_lo))*x_up

        # Get moist adiabatic laps rate (K/m) and depth of layer from lower to upper level
        pmid  =  ( (p_segment*np.log(p_segment)  +  p_old   *np.log(p_old))  /  np.log(p_segment*p_old) )
        tmid  =  ( (t_segment*np.log(p_segment)  +  tseg_old*np.log(p_old))  /  np.log(p_segment*p_old) )
        qmid  =  ( (q_segment*np.log(p_segment)  +  qseg_old*np.log(p_old))  /  np.log(p_segment*p_old) )
        dz    =  (p_old-p_segment) / (grav * pmid /(Rd*tmid*((1.+(qmid/ep)) / (1. + qmid))))

        # Compute saturation specific humidity
        press = pmid/1e2    # Convert to hPa 
        numerator   = ep * (es0*np.exp((a*( tmid-t0))/( tmid-b)))
        denomenator = press-onemep*(es0*np.exp((a*( tmid-t0))/( tmid-b)))
        esat  =  numerator/denomenator 
        qsat  = esat / (1 + esat)

        moist_lapse  =  (grav/cp) * ( (1. + (Lv    * qsat)/(   Rd*tmid   )) / 
                                      (1. + ((Lv**2) * qsat)/(cp*Rv*(tmid**2))) )

        # Get parcel temperature 
        tpar = tpar_old - moist_lapse*dz 

        # Get mid-point temps from environment and parcel
        tpar_mid = 0.5 * (tpar + tpar_old)
        tseg_mid = 0.5 * (t_segment + tseg_old)

        # Integrate from old increment to increment level
        CTP  =  CTP  +  (Rd * (tpar_mid-tseg_mid) * np.log(p_old/p_segment))

        # Update last increment values 
        tpar_old  =  tpar
        tseg_old  =  t_segment
        qseg_old  =  q_segment
        p_old     =  p_segment
    
    
    # ===========================================================================
    
    # Return statement (what to give back to user) 
    return(CTP, HI_low)
    
    
    
