# Compute variables from Heated Condensation Framework 
# MAJOR NOTE: all of this code originates from Ahmed Tawfik's Coupling Metrics Toolbox.
#   I have made only minor changes to convert his script from .F90 to python, and in 
#   most cases have kept in the .F90 code, commented out. 
# 
# Author: Meg D. Fowler 
# Date:   12 Oct 2020
#
# Details: 
#   The script is built to work on one sounding - that is, one point in space and one observation time. 
#   All data should be passed in via a single xarray dataframe, with any missing values set to NaN. 
#   NOTE: for profiles, the data should be ordered to begin at the surface and descend accordingly.
#
# Required inputs:
#   DF containing: 
#      Temperature profile         [K]
#      Geopotential Height profile [m]
#      Specific humidity profile   [kg/kg]
#      Pressure profile            [Pa]
#   Tname   -- Name of temperature data in DF 
#   Qname   -- Name of sp. humidity data in DF
#   Zname   -- Name of geopotential height data in DF
#   Pname   -- Name of pressure data in DF 
#   nlevs   -- Number of levels in profile 
#     ** NOTE ** Currently using bottom level to represent surface characteristics. 
#
# Outputs: 
#   TBM       --   Buoyant mixing potential temperature [K]
#   BCLH      --   Height above ground of convective threshold [m]
#   BCLP      --   Pressure of convective threshold level [Pa]
#   TDEF      --   Potential temperature deficit needed to initiate [K]
#   TRAN_H    --   Energy transition height [m]
#   TRAN_P    --   Energy transition pressure [Pa]
#   TRAN_T    --   Energy transition temperature [K]
#   SHDEF_M   --   Sensible heat deficit of mixed layer [J/m2]
#   LHDEF_M   --   Latent heat deficit of mixed layer [J/m2]
#   EADV_M    --   Energy advantage of mixed layer [-]
# 
# ==================================================================================================
def HCF(DF, Tname, Qname, Zname, Pname, nlevs): 
    # Import libraries 
    import numpy as np 
    
    # -----------------------------------------------
    #  Get profile data into individual arrays
    #   and select "surface" data as first level 
    # -----------------------------------------------
    
    # Profile starting at level above sfc
    tmp_in   = DF[Tname].values[1::]
    qhum_in  = DF[Qname].values[1::]
    hgt_in   = DF[Zname].values[1::]
    press_in = DF[Pname].values[1::]

    # Sfc values set as first level values 
    t2m      = DF[Tname].values[0]
    q2m      = DF[Qname].values[0]
    h2m      = DF[Zname].values[0]
    psfc     = DF[Pname].values[0]

    # Number of levels to worry about in actual "sounding"
    nlev1 = nlevs-1 
    
    # -----------------------------------------------
    #    Set constants 
    # -----------------------------------------------
    p_ref  = 1e5 
    Lv     = 2.5e6 
    cp     = 1005.7
    R_cp   = 287.04/1005.7

    grav   = 9.81 
    Rd     = 287.04
    pi     = np.pi
    cp_g   = cp/grav
    Lv_g   = Lv/grav
    r2d    = 180./pi

    by100  = 1e2
    t0     = 273.15 
    ep     = 0.622
    es0    = 6.11
    a      = 17.269
    b      = 35.86
    onemep = 1.0 - ep

    # -----------------------------------------------
    #    Initiate empty arrays 
    # -----------------------------------------------
    shdef = np.full([nlev1+1], np.nan)
    lhdef = np.full([nlev1+1], np.nan)
    eadv  = np.full([nlev1+1], np.nan)
    rhoh  = np.full([nlev1+1], np.nan)
    pbar  = np.full([nlev1+1], np.nan)
    qdef  = np.full([nlev1+1], np.nan)
    qmix  = np.full([nlev1+1], np.nan)

    qsat   = np.full([nlev1+1], np.nan)
    dpress = np.full([nlev1+1], np.nan) 
    qbar   = np.full([nlev1+1], np.nan)
    logp   = np.full([nlev1+1], np.nan)
    hbar   = np.full([nlev1+1], np.nan)
    tbar   = np.full([nlev1+1], np.nan)
    tmp_k  = np.full([nlev1+1], np.nan)
    press  = np.full([nlev1+1], np.nan)
    pot_k  = np.full([nlev1+1], np.nan)
    hgt    = np.full([nlev1+1], np.nan)
    qhum   = np.full([nlev1+1], np.nan)

    pot_diff = np.full([nlev1+1], np.nan)
    eadv_0   = np.full([nlev1+1], np.nan)
    xaxis    = np.full([nlev1+1], np.nan)
    xaxis1   = np.full([nlev1+1], np.nan)
    yaxis    = np.full([nlev1+1], np.nan)
    yaxis1   = np.full([nlev1+1], np.nan)
    integral = np.full([nlev1+1], np.nan)
    below    = np.full([nlev1+1], np.nan)
    
    # -----------------------------------------------
    #    Store temp working arrays and initialize 
    # -----------------------------------------------
    nlev      = nlev1+1 

    tmp_k[1:] = tmp_in 
    hgt[1:]   = hgt_in
    qhum[1:]  = qhum_in
    press[1:] = press_in 

    tmp_k[0] = t2m 
    hgt[0]   = h2m
    qhum[0]  = q2m 
    press[0] = psfc

    # -----------------------------------------------
    # Run a few checks on pressure
    #    (but most of this is unfortunately on the user)
    # -----------------------------------------------

    # Check that pressure levels aren't greater than the surface pressure 
    iProblem = np.where(press[1:] >= psfc)[0]

    if len(iProblem)>0:
        print('***** ERROR: lowest pressure level > surface pressure *****')

    # Check ordering of Plev: 
    if press[0]<press[-1]:
        print('***** ERROR: pressure levels should be reversed *****') 

    # Check units of pressure 
    if psfc<=2000.0: 
        print('**** ERROR: pressures should be in Pa, not hPa *****') 



    
    # -----------------------------------------------
    #    Compute column potential temperature 
    # -----------------------------------------------
    pot_k = tmp_k * (p_ref/press)**(R_cp)

    # -----------------------------------------------
    #    Ignore missing data levels when calculating midpoints 
    #     (shouldn't be an issue for model data)
    # -----------------------------------------------
    hbar = hgt
    pbar = press
    tbar = tmp_k

    # -----------------------------------------------
    #    Compute middle layer specific humidity average [kg/kg]
    #    1st layer = the 2m sp. humidity above, then layer averages above 
    # -----------------------------------------------
    qbar = qhum 
    qbar[1:nlev] = ( (qhum[1:nlev]*np.log(press[1:nlev]) + 
                      qhum[0:nlev-1]*np.log(press[0:nlev-1]) )  / 
                      np.log(press[1:nlev]*press[0:nlev-1]))
    # qbar(2:nlev)    = ((qhum(2:nlev  )*log(press(2:nlev  ))  + &
    #                                   qhum(1:nlev-1)*log(press(1:nlev-1))) / &
    #                                   log(press(2:nlev)* press(1:nlev-1)))


    # -----------------------------------------------
    #    Compute pressure difference of each layer 
    # -----------------------------------------------
    if dpress[0]<=0: 
        dpress[0] = 1.     # Set to 1 Pa because h2m is likely zero 
    else:
        dpress[0]   = (psfc / (Rd * t2m * ((1. + (q2m/ep)) / (1. + q2m)) )) * grav * h2m 
        #dpress(1)  =  (psfc / (Rd * t2m * ((1. + (q2m/ep)) / (1. + q2m)) )) * grav * h2m

    # Model data shouldn't have any missing, so not using this line:
    #    where( pbar(1:nlev-1).ne.missing  .and.  pbar(2:nlev).ne.missing )
    dpress[1:nlev] = press[0:nlev-1] - press[1:nlev]

    # -----------------------------------------------
    #    Compute log pressure to linearize it for slope calculation 
    # -----------------------------------------------
    logp = np.log(pbar)

    # -----------------------------------------------
    #    Compute mixed layer sp. humidity and column density [kg/kg]
    # -----------------------------------------------
    qmix = qbar * dpress/grav
    rhoh = dpress/grav

    for izz in range(nlev-1):
        zz = izz+1    # .f90 is: do zz = 2, nlev; so going to increase index by one 
        if (np.isfinite(qmix[zz]) & np.isfinite(qmix[zz-1])): 
            qmix[zz] = qmix[zz-1] + qmix[zz]

        if (np.isfinite(rhoh[zz]) & np.isfinite(rhoh[zz-1]) ):
            rhoh[zz] = rhoh[zz-1] + rhoh[zz]

    # -----------------------------------------------
    #    Compute saturation specific humidity at each level
    # -----------------------------------------------
    pbar = pbar/1e2
    qsat = by100*0.01 *(ep* (es0*np.exp((a*( tbar-t0))/( tbar-b))) ) / (pbar-onemep*(es0*np.exp((a*( tbar-t0))/( tbar-b))))
    qsat = qsat/(1.+qsat)
    pbar  =  pbar*1e2

    # -----------------------------------------------
    #    Calculate specific humidity deficit [kg/kg]
    # -----------------------------------------------
    qmix = qmix / rhoh
    qdef = qsat - qmix 

    # -----------------------------------------------
    #    Check that qdef is always negative outside of tropopause.
    #    Assum tropo height of 10 km; so BCL cannot be higher 
    # -----------------------------------------------
    iCheck = np.where(hbar>=10000.0)[0]
    qdef[iCheck] = -1.0

    #***********************************************************
    #***   Calculate slope of each variable to find the      ***
    #***   y-intercept for each variable;                    ***
    #***   Meaning locate the two data points surrounding    ***
    #***   the sign change in qdef and linearly interpolate  ***
    #***   to find the "zero point"                          ***
    #***********************************************************

    # -----------------------------------------------
    #   Find point where sign first turns negative from the ground up
    # -----------------------------------------------

    # Highest unsaturated level 
    num_unsat = len(np.where((np.isfinite(qdef)) & (qdef>0))[0])
    if num_unsat>0: 
        # i_unsat   =   maxloc( hbar, DIM = 1, MASK = qdef.ne.missing .and. qdef.gt.0 )
        iMask       = np.where((~np.isnan(qdef)) & (qdef>0))[0]
        hbar_masked = hbar[iMask]
        i_unsat     = np.where(hbar==np.nanmax(hbar_masked))[0]
    else: 
        i_unsat = 0 

    # Lowest saturated level 
    num_sat  = len(np.where((np.isfinite(qdef)) & (qdef<=0))[0])
    if num_sat>0:
        #i_sat =   minloc( hbar, DIM = 1, MASK = qdef.ne.missing .and. qdef.le.0 )
        iMask  = np.where((~np.isnan(qdef)) & (qdef<=0))[0]
        hbar_masked = hbar[iMask]
        i_sat = np.where(hbar==np.nanmin(hbar_masked))[0]
    else: 
        i_sat = 0 

    # -----------------------------------------------
    #  If all levels are saturated, then put the deficit to zero
    # -----------------------------------------------
    sat_flag = 0
    if num_unsat==0: 
        pot2m = (t2m) * ((p_ref/psfc)**(R_cp))  *  (1. + 0.61*qmix[0])
        BCLP   = psfc 
        BCLH  = h2m
        TBM   = pot2m 
        TDEF  = 0.

        sat_flag = 1 
        # print('ALL LEVELS ARE SATURATED. Returning')

    # Coding now to keep computing things ONLY IF not all levels are saturated 
    if sat_flag==0:
        # -----------------------------------------------
        # Check to see if first level is saturated (Foggy scenario). 
        # If yes, check 2nd and 3rd layers to see if fog will dissipate. 
        # If the 2nd and/or 3rd are not saturated, then recalculate 
        #  CONVECTIVE saturation transition level. 
        # -----------------------------------------------

        if ((i_sat>1) & (i_unsat>i_sat)):
            i_unsat  =  i_sat - 1

        if (i_sat==1): 
            cc = 0

            for izz in range(nlev-2):
                zz = izz+1    # .f90 is: do zz = 2, nlev-1; so going to increase index by one 

                # Make sure initiation level is below 100 hPa above the ground 
                #      to ensure it is actually fog. 
                if ( ((psfc-pbar[zz])/1e2) > 100 ):   # if( (psfc-pbar(zz))/1e2.gt.100 ) exit
                    break 

                # If it *is* within 100 hPa layer above ground, try to erode fog layer first
                #   to determine convective initiation layer 
                # i_sat  =  minloc( hbar(zz:), DIM = 1, MASK = qdef(zz:).ne.missing .and. qdef(zz:).le.0 )
                hbar_mask1  = hbar[zz:]
                iMask       = np.where((np.isfinite(qdef[zz:])) & (qdef[zz:]<=0))[0]
                hbar_masked = hbar_mask1[iMask]
                i_sat       = np.where(hbar_mask1==np.nanmin(hbar_masked))[0] 

                cc = cc + 1

                # If still saturated then cycle 
                if i_sat==1:
                    continue 
                i_sat = i_sat + cc 
                break 
            i_unsat = i_sat - 1 

        # -----------------------------------------------
        # If all layers below 100 hPa above ground are still saturated, 
        # call it all saturated and use 1st level stats and call it "convective"
        # b/c fog is unlikely to be deeper than 100 hPa above ground 
        # -----------------------------------------------

        if i_unsat==0: 
            pot2m = (t2m) * ((p_ref/psfc)**(R_cp))  *  (1. + 0.61*qmix[0])
            BCLP  = psfc
            BCLH  = h2m 
            TBM   = pot2m 
            TDEF  = 0. 

            sat_flag = 1 
            # print('ALL LEVELS ARE SATURATED. Returning')

        # -----------------------------------------------
        # Check to make sure these are adjacent layers 
        # If not, there's a problem. 
        # -----------------------------------------------
        if ((i_unsat==0) | (i_sat==0)):
            print('=========== ERROR  in locating saturation profiles ============')
            sat_flag=1 
            pot2m = np.nan
            BCLP  = np.nan
            BCLH  = np.nan
            TBM   = np.nan
            TDEF  = np.nan
            TRAN_H  = np.nan
            TRAN_P  = np.nan
            TRAN_T  = np.nan
            SHDEF_M = np.nan
            LHDEF_M = np.nan
            EADV_M  = np.nan
           # print('   Terminating program. ')


    # Check again to make sure we shouldn't be stopping here...
    if sat_flag==0: 
        # ----------------------------------------------- 
        # Get upper and lower bounds for each var to be 
        #   computed at the BCL 
        # -----------------------------------------------
        p_up        =  logp[i_sat]
        t_up        =  tbar[i_sat]
        h_up        =  hbar[i_sat]
        q_up        =  qdef[i_sat]
        m_up        =  qmix[i_sat]

        p_lo        =  logp[i_unsat]
        t_lo        =  tbar[i_unsat]
        h_lo        =  hbar[i_unsat]
        q_lo        =  qdef[i_unsat]
        m_lo        =  qmix[i_unsat]


        # -----------------------------------------------
        # Calculate output variables: BCL height, pressure, 
        #   buoyant mixing potential temp, and potential temp 
        #   deficit. 
        # -----------------------------------------------
        BCLP = np.exp( p_up - ((p_up-p_lo)/(q_up-q_lo))*q_up )
        BCLH = ( h_up - ((h_up-h_lo)/(q_up-q_lo))*q_up )
        qbcl = ( m_up - ((m_up-m_lo)/(q_up-q_lo))*q_up )
        TBM  = ( t_up - ((t_up-t_lo)/(q_up-q_lo))*q_up )* ((p_ref/BCLP)**(R_cp))

        # -----------------------------------------------
        # Calculate virtual potential temperature (K) using mixed humidity. 
        # NOTE: This is an assumption; only influences TDEF but 
        #   an important effect because if pot2m is close to TBM, then a slight 
        #   change in qbcl can mean the difference betwen initiation (TDEF=0)
        #   or not. Should only be an issue over very shallow PBLs.
        # -----------------------------------------------
        pot2m  =  (t2m) * ((p_ref/psfc)**(R_cp))  *  (1. + 0.61*qbcl)
        TDEF   =  TBM  - pot2m
        if TDEF<0: 
            TDEF=0 

    #****************************************************
    #***           ENERGY DEFICIT SECTION             ***
    #****************************************************
    # Takes BCL and TBM tthreshold to estimate sensible 
    #   and latent heat energy [J/m2] necessary for 
    #   initiating convection. Does not discriminate 
    #   between shallow or deep convection. Also outputs
    #   potential temperature, pressure, and height of the
    #   transition from latent heat to sensible heat 
    #   advantage. If there is no transition then transition
    #   levels are set to NaN (missing values). 
    # ----------------------------------------------------

    # Handle case where there are no energy deficits, 
    #  because threshold already reached (convection initiated)
    if TDEF<=0: 
        SHDEF_M    =  0.
        LHDEF_M    =  0.
        EADV_M     =  np.nan
        TRAN_T     =  np.nan
        TRAN_P     =  np.nan
        TRAN_H     =  np.nan
        # End program here 
        sat_flag=1 
        # print('Saturation threshold already reached! Ending program.')

    # Check again to make sure we shouldn't be stopping here...
    if sat_flag==0: 
        # -----------------------------------------------
        #  Find pressure level and mixed specific humidity 
        #    deficit given a potential temperature. 
        # -----------------------------------------------
        pbl_pot = pot2m 

        # Difference between reference potential temperature [K]
        # where( pot_k.ne.missing .and. press.ne.missing .and. tmp_k.gt.0 )   pot_diff  =  pbl_pot - pot_k
        iCalc = np.where(tmp_k>0)[0]
        pot_diff[iCalc]  =  pbl_pot  - pot_k[iCalc] 

        #***********************************************************
        #***   Calculate slope of each variable to find the      ***
        #***   y-intercept for each variable;                    ***
        #***   Meaning locate the two data points surrounding    ***
        #***   the sign change in qdef and linearly interpolate  ***
        #***   to find the "zero point"                          ***
        #***********************************************************

        # -----------------------------------------------
        # Find point where sign first turns negative from ground up 
        # -----------------------------------------------

        # Highest buoyant level 
        num_buoy = len(np.where(np.isfinite(pot_diff) & (pot_diff>0) )[0])
        if num_buoy>0: 
            # i_buoy   =   minloc( pbar, DIM = 1, MASK = pot_diff.ne.missing .and. pot_diff.gt.0 )
            iMask       = np.where( (~np.isnan(pot_diff)) & (pot_diff>0) )[0]
            pbar_masked = pbar[iMask]
            i_buoy      = np.where(pbar==np.min(pbar_masked))[0]
        else:
            i_buoy = 0

        # Lowest negatively buoyant level 
        num_nobuoy = len(np.where(np.isfinite(pot_diff) & (pot_diff<=0) )[0])
        if num_nobuoy>0: 
            # i_nobuoy =   maxloc( pbar, DIM = 1, MASK = pot_diff.ne.missing .and. pot_diff.le.0 )
            iMask = np.where( (~np.isnan(pot_diff)) & (pot_diff<=0))[0]
            pbar_masked = pbar[iMask]
            i_nobuoy = np.where(pbar==np.max(pbar_masked))[0]
        else:
            i_nobuoy = -1    # MDF: Set to -1 instead of 0, since 0 is first index in Py (but not F90)

        # -----------------------------------------------
        #  Check to make sure not all layers are buoyant (not physical)
        # -----------------------------------------------
        if i_nobuoy==-1: 
            print('=========== ERROR  in locating saturation profiles ============')
            sat_flag = 1 
            TRAN_H  = np.nan
            TRAN_P  = np.nan
            TRAN_T  = np.nan
            SHDEF_M = np.nan
            LHDEF_M = np.nan
            EADV_M  = np.nan
           # print('   Terminating program. ')

    # Check again that it's safe to keep computing...
    if sat_flag==0:
        # -----------------------------------------------
        #  Check to see if first level is NOT buoyant 
        #  If so, thermally produced PBL is below the first layer
        # -----------------------------------------------
        if i_nobuoy==0: 
            i_nobuoy = 1    # MDF: Set to 1 insetad of 2 (python vs. F90 indexing)
            i_buoy   = 0    # MDF: Set to 0 instead of 1 (python vs. F90 indexing)

        # -----------------------------------------------
        # Get upper/lower bounds for each variable to be 
        #   computed at the BCL
        # -----------------------------------------------
        p_up        =  logp    [i_nobuoy]
        q_up        =  qdef    [i_nobuoy]
        t_up        =  pot_diff[i_nobuoy]
        p_lo        =  logp    [i_buoy]
        q_lo        =  qdef    [i_buoy]
        t_lo        =  pot_diff[i_buoy]

        # -----------------------------------------------
        #  Calculate output variables 
        # -----------------------------------------------
        pbl_p     =  np.exp( p_up - ((p_up-p_lo)/(t_up-t_lo))*t_up )
        pbl_qdef  =        ( q_up - ((q_up-q_lo)/(t_up-t_lo))*t_up )

        # -----------------------------------------------
        #  Initialize energy deficit working variables 
        # -----------------------------------------------
        #shdef  = np.nan
        #lhdef  = np.nan
        #eadv   = np.nan
        #eadv_0 = np.nan

        # -----------------------------------------------
        #  Make sure pressure of PBL is above lowest level 
        #  This can occur for very shallow BL, and is likely
        #    due to mixing assumptions made in the BL calculation,
        #    where it is assumed to be thermally driven. 
        #  In this case, we assume the mixed layer is between
        #    the surface adn the first atmospheric model level
        # -----------------------------------------------
        if pbl_p>psfc: 
            pbl_p = psfc - (psfc-press[1])/2.0    # MDF: Index with 1 instead of 2 (py v. F90)


        #*************************************************
        #********                                 ********
        #********         --Section--             ********
        #********  Sensible Heat Deficit [J/m2]   ********
        #********                                 ********
        #*************************************************
        xaxis   =   press 
        yaxis   =   pot_k
        pthresh =   BCLP
        tthresh =   TBM

        #yaxis1  =   np.nan   # Already initialized to missing 
        #xaxis1  =   np.nan
        yaxis1[:nlev-1]  =  yaxis[1:nlev]
        xaxis1[:nlev-1]  =  xaxis[1:nlev]

        # -----------------------------------------------
        #  Calculate integrals from mixed layer down and up
        # -----------------------------------------------

        # Deficit for each layer 

        # itop    =   minloc( xaxis1,  DIM = 1, MASK = xaxis1.gt.pthresh .and. xaxis1.ne.missing )
        iMask         = np.where((~np.isnan(xaxis1)) & (xaxis1>pthresh))[0]
        xaxis1_masked = xaxis1[iMask]
        itop          = int(np.where(xaxis1==np.nanmin(xaxis1_masked))[0])
        ibot          = 0 
        nbot          = itop - ibot + 1
        if np.isfinite(psfc):
            total = (cp_g)  *  tthresh  *  (psfc - pthresh)

        integral[0] = 0.0
        below[0]    = 0.0 
        if itop==ibot:
            #---- Case where BCL is within the first layer (i.e. between 1st and 2nd index)
            # between =   (cp_g)  *  0.5*(yaxis(1)+tthresh)  *  (xaxis (1)-pthresh)
            between =   (cp_g)  *  0.5*(yaxis[0]+tthresh)  *  (xaxis[0]-pthresh)
        else:
            between =   (cp_g)  *  0.5*(yaxis1[itop]+tthresh)  *  (xaxis1[itop]-pthresh)


            # MDF: Defining array of levels to care about
            zz_levs = np.arange(ibot,itop,1).astype(int)

            # do zz=ibot,itop
            for izz in range(len(zz_levs)):
                zz = zz_levs[izz]
                integral[zz]    =  np.sum(  (cp_g)  *  0.5*(yaxis[zz:itop]+yaxis1[zz:itop])  * 
                                          (xaxis[zz:itop]-xaxis1[zz:itop]) )
                below   [zz+1]  =  (cp_g)  *  yaxis[zz+1]    *  (xaxis[ibot] - xaxis[zz+1])

        # -----------------------------------------------
        #  Deficit for mixed layer only 
        # -----------------------------------------------
        # itop    =   minloc( xaxis1,  DIM = 1,  MASK = xaxis1.gt.pthresh  .and.  xaxis1.ne.missing )
        iMask         = np.where((~np.isnan(xaxis1)) & (xaxis1>pthresh))[0]
        xaxis1_masked = xaxis1[iMask]
        itop          = int(np.where(xaxis1==np.nanmin(xaxis1_masked))[0])

        # if(  all(.not.(xaxis1.gt.pthresh  .and.  xaxis.lt.pbl_p   .and.  xaxis1.ne.missing))  )  then
        if ( (np.all(xaxis1<=pthresh)) & (np.all(xaxis>=pbl_p)) & (np.all(np.isnan(xaxis1))) ):
            ibot = itop 
        else: 
            # ibot = maxloc( xaxis1,  DIM = 1,  MASK = xaxis1.gt.pthresh  .and.  xaxis.lt.pbl_p   .and.  xaxis1.ne.missing )
            iMask       = np.where( (xaxis1>pthresh) & (xaxis<pbl_p) & (~np.isnan(xaxis1)))[0]
            # MDF: Adding catch for if no cases meet mask criteria:
            if len(iMask>0):
                xaxis1_mask = xaxis1[iMask]
                ibot        = int(np.where(xaxis1==np.nanmax(xaxis1_mask))[0])
            else:
                ibot = itop 

        nbot    =   itop - ibot + 1
        itop0   =   int(itop)
        ibot0   =   int(ibot)

        integral0 = 0.0
        below0    = 0.0
        if itop==ibot: 
            # ---- Case where BCL is within the first layer (i.e. between 1st and 2nd index)
            between0  =   (cp_g)  *  0.5*(pbl_pot+tthresh)  *  (pbl_p  -  pthresh)
            below0    =   (cp_g)  *  pbl_pot                *  (psfc   -  pbl_p  )
            if between0<0:
                between0  =  0.0
        else:
            #*** explicit layer and BCL
            between0   =           (cp_g) * 0.5*(yaxis1[itop] + tthresh) *  (xaxis1[itop] - pthresh)
            integral0  =   np.sum( (cp_g) * 0.5*(yaxis[ibot:itop] + yaxis1[ibot:itop])  *  (xaxis[ibot:itop] - xaxis1[ibot:itop]) )
            #*** explicit layer and PBL
            between0   =   between0  +  ((cp_g)  *  0.5*(yaxis[ibot] + pbl_pot) * (pbl_p  - xaxis [ibot]))
            below0     =                 (cp_g)  *  pbl_pot *  (psfc - pbl_p)

        # -----------------------------------------------
        #   Calculate the Sensible Heat Deficit [J/m2]
        #   Equation: 
        #     SHDEF = Energy from BCL to surface (scalar --> Total) MINUS
        #             the progressive integral from mixed layer
        #               to last resolved level directly below the BCL (nlev --> Integral) MINUS
        #             the energy btw last resolved level and BCL (scalar --> Between) MINUS
        #             the energy from the mixed layer to the surface (nlev --> Below)
        #   NOTE: Sensible heat deficit is calculated from the first layer to the BCL 
        # -----------------------------------------------
        shdef   =   total  -  integral  -  between  -  below
        iCheck  = np.where( (press<BCLP) | (np.isnan(press)) )[0]
        shdef[iCheck] = 0.0
        SHDEF_M   =   total  -  integral0  -  between0  -  below0



        #*************************************************
        #********                                 ********
        #********         --Section--             ********
        #********   Latent Heat Deficit [J/m2]    ********
        #********                                 ********
        #*************************************************

        # -----------------------------------------------
        #  Make sure qdef at PBLH > 0 
        #  Occurs when PBL really close (probably too close to be ignored 
        #     as not having convection). 
        #  For practical purposes, if TDEF/=0 then there is no convection, 
        #     so QDEF at PBL as estimated should also be >0. So here, 
        #     we set PBL qdef = some small number > 0 
        # -----------------------------------------------
        if pbl_qdef<0:
            pbl_qdef = 0.00001

        # -----------------------------------------------
        #  Calculate the Latent Heat Deficit [J/m2]
        #  Equation: 
        #    LHdef = latent heat of vaporization/gravity TIMES 
        #            pressure difference mixed layer down TIMES 
        #            Specific Humidity Deficit
        #  NOTE: Latent heat deficit is calculated from 1st layer to BCL 
        # -----------------------------------------------
        iCompute        = np.where((qdef>0) & (~np.isnan(qdef)))[0]
        lhdef[iCompute] = Lv_g  *  qdef[iCompute]  *  dpress[iCompute]

        iZero           = np.where((press<BCLP) | (np.isnan(press)))[0]
        lhdef[iZero]    = 0 

        if psfc-pbl_p<=0: 
            LHDEF_M   =   Lv_g  *  pbl_qdef  *  (dpress[0])
        else: 
            LHDEF_M   =   Lv_g  *  pbl_qdef  *  (psfc - pbl_p)


        #*************************************************
        #********                                 ********
        #********         --Section--             ********
        #********   Energy Advantage and 45deg    ********
        #********                                 ********
        #*************************************************
        iCompute       = np.where((~np.isnan(lhdef)) & (~np.isnan(shdef)) & (lhdef!=0) & (shdef!=0))[0]
        eadv[iCompute] = np.arctan2(lhdef[iCompute], shdef[iCompute]) * r2d

        if ((LHDEF_M>0) & (SHDEF_M>0)): 
            EADV_M = np.arctan2(LHDEF_M, SHDEF_M) * r2d
        else: 
            EADV_M = np.nan

        #*************************************************
        #  Special no transition case 
        #*************************************************
        if ( (np.all(np.isnan(eadv))) | (np.all(eadv<45)) | (np.all(np.isnan(eadv))) | (np.all(eadv>45)) ):
            TRAN_P  =  np.nan
            TRAN_T  =  np.nan
            TRAN_H  =  np.nan

            # print(' ***** NO TRANSITION IS PRESENT. RETURNING. *****')
            sat_flag = 1

    # Continue with calculations if appropriate
    if sat_flag==0:
        # -----------------------------------------------
        #  Find where energy advantage = 45 degrees 
        #  If it doesn't occur anywhere, set all values to mising 
        # -----------------------------------------------
        iCheck         = np.where(~np.isnan(eadv))[0]
        eadv_0[iCheck] = eadv[iCheck] - 45.0

        # ibefore =  maxloc( hgt, DIM = 1,  MASK = eadv_0.le.0  .and.  eadv_0.ne.missing )  !location right before transition
        iMask    = np.where((eadv_0<=0) & (~np.isnan(eadv_0)))[0]
        if len(iMask>0):
            hgt_mask = hgt[iMask]
            ibefore  = np.where(hgt==np.nanmax(hgt_mask))[0]
        else: 
            ibefore = -1

        # iafter  =  minloc( hgt, DIM = 1,  MASK = eadv_0.gt.0  .and.  eadv_0.ne.missing )  !location right after transition
        iMask    = np.where((eadv_0>0) & (~np.isnan(eadv_0)))[0]
        if len(iMask>0):
            hgt_mask = hgt[iMask]
            iafter   = np.where(hgt==np.nanmin(hgt_mask))[0] 
        else:
            iafter = -1 

        #************************************
        #**** special no transition case ****
        #************************************
        if ( (iafter==-1) | (ibefore==-1) ):
            TRAN_P  =  np.nan
            TRAN_T  =  np.nan
            TRAN_H  =  np.nan

            # print(' ***** NO TRANSITION IS PRESENT. RETURNING. *****')
            sat_flag = 1

    # Check that we should still be computing things: 
    if sat_flag==0:

        #*******************************************************************************
        #**** linear interpolation to find temp, height, and pressure of transition ****
        #*******************************************************************************
        x_hi    =  eadv_0[iafter]
        x_lo    =  eadv_0[ibefore]
        y_hi    =  np.log(press[iafter])
        y_lo    =  np.log(press[ibefore])
        TRAN_P  =  np.exp(  y_hi -  (((y_hi-y_lo)/(x_hi-x_lo)) * x_hi) )

        y_hi    =  pot_k[iafter]
        y_lo    =  pot_k[ibefore]
        TRAN_T  =  y_hi -  (((y_hi-y_lo)/(x_hi-x_lo)) * x_hi)

        y_hi    =  hgt[iafter]
        y_lo    =  hgt[ibefore]
        TRAN_H  =  y_hi -  (((y_hi-y_lo)/(x_hi-x_lo)) * x_hi)        
    
    
    return(TBM, BCLH, BCLP, TDEF, TRAN_H, TRAN_P, TRAN_T, SHDEF_M, LHDEF_M, EADV_M)

