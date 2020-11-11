; ------------------------------------------------------------------------
;  Compute potential temperature of surface data. 
;  
;  11 Nov 2020, Meg D. Fowler 
; ------------------------------------------------------------------------

begin 

; -------- Read in data --------

filenamePS = "/Users/mdfowler/Documents/Analysis/Coupling_initial/data/3hrSim_CAM6-CLM45/f.e21.FHIST.f09_f09.cesm2_cam6_clm4p5.001.cam.h4.1979-1989_sfcPS.nc"
filenameT  = "/Users/mdfowler/Documents/Analysis/Coupling_initial/data/3hrSim_CAM6-CLM45/f.e21.FHIST.f09_f09.cesm2_cam6_clm4p5.001.cam.h3.1979-1989_sfcTemps.nc"
fileOut    = "/Users/mdfowler/Documents/Analysis/Coupling_initial/data/3hrSim_CAM6-CLM45/f.e21.FHIST.f09_f09.cesm2_cam6_clm4p5.001.cam.h4.1979-1989_sfcPotentialTemperature.nc"

; Open file with surface pressure
fPS = addfile(filenamePS,"r")

; Open file with temperature 
fTemp = addfile(filenameT,"r")

; Read a few variables
tbot = fTemp->TBOT
tref = fTemp->TREFHT
ps = fPS->PS
p0 = 100000.0
time = fPS->time
lat = fPS->lat
lon = fPS->lon

; -------- Compute potential temperature  --------
;   Using NCL function, pot_temp 
theta_ps_tbot = pot_temp(ps,tbot,-1,0)
theta_ps_tref = pot_temp(ps,tref,-1,0)

; ------ Convert to UTC time ----------
utc_date = cd_calendar(time, 0)

; Also isolate yr/mon/day/hr arrays
UTChr = utc_date(:,3)
UTCday = utc_date(:,2)
UTCmon = utc_date(:,1)
UTCyr = utc_date(:,0)


; ----- Create netCDF file ------
print("Creating netCDF file")

system("/bin/rm -f " + fileOut)   ; remove any pre-existing file
ncdf = addfile(fileOut,"c")  ; open output netCDF file

; Create global attributes of the file (optional)
fAtt               = True            ; assign file attributes
fAtt@title         = "NCL used to get potential temperature at surface."
fAtt@source_file   =  filenamePS
fAtt@Conventions   = "None"
fAtt@creation_date = systemfunc ("date")
fileattdef( ncdf, fAtt )            ; copy file attributes

; Make time an unlimited dimension
filedimdef(ncdf,"time",-1,True)

;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================

theta_ps_tbot!0 = "time"
theta_ps_tbot!1 = "lat"
theta_ps_tbot!2 = "lon"

theta_ps_tbot&time = time
theta_ps_tbot&lat  = lat
theta_ps_tbot&lon  = lon

theta_ps_tbot@long_name = "Potential temperature, computed using PS and TBOT variables"
theta_ps_tbot@units = tbot@units
ncdf->THETA_PS_TBOT = theta_ps_tbot

; -----------------------------
theta_ps_tref!0 = "time"
theta_ps_tref!1 = "lat"
theta_ps_tref!2 = "lon"

theta_ps_tref&time = time
theta_ps_tref&lat  = lat
theta_ps_tref&lon  = lon

theta_ps_tref@long_name = "Potential temperature, computed using PS and TREFHT variables"
theta_ps_tref@units = tref@units
ncdf->THETA_PS_TREF = theta_ps_tref

; -----------------------------
UTChr!0  = "time"
UTCday!0 = "time"
UTCmon!0 = "time"
UTCyr!0  = "time"

UTChr&time  = time
UTCday&time = time
UTCmon&time = time
UTCyr&time  = time

UTChr@long_name = "Hour in UTC"
UTCday@long_name = "Day in UTC"
UTCmon@long_name = "Mon in UTC"
UTCyr@long_name = "Yr in UTC"

ncdf->UTC_hr = UTChr
ncdf->UTC_day = UTCday
ncdf->UTC_mon = UTCmon
ncdf->UTC_yr = UTCyr

print("File created!")

end





