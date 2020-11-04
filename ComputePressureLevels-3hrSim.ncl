; ------------------------------------------------------------------------
;  Compute vertical profiles of pressure based on files. 
;    Write out to netCDF file using slightly improved emthod compared to 
;    version of the script that's saved in NCAR/Analysis/CLUBB_initial/data.
;
;  Date: 1 Sept 2020
;  Author: Meg Fowler
; ------------------------------------------------------------------------

begin

; -------- Read in data --------

filePS = "/Users/mdfowler/Documents/Analysis/Coupling_initial/data/3hrSim_CAM6-CLM45/f.e21.FHIST.f09_f09.cesm2_cam6_clm4p5.001.cam.h4.1982_conusAllTimes-VertProfiles.nc"
fileOut = "/Users/mdfowler/Documents/Analysis/Coupling_initial/data/3hrSim_CAM6-CLM45/f.e21.FHIST.f09_f09.cesm2_cam6_clm4p5.001.cam.h4.1982_conus-Pressure-UTCtimes.nc"

; Open file with surface pressure
fPS = addfile(filePS,"r")

; Read a few variables
hyam = fPS->hyam
hybm = fPS->hybm
ps = fPS->PS
p0 = fPS->P0
time = fPS->time
lat = fPS->lat
lon = fPS->lon

; ------ Compute pressure levels --------

; Compute pressure levels
pm = pres_hybrid_ccm(ps,p0,hyam,hybm)

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
fAtt@title         = "NCL used to get pressure levels."
fAtt@source_file   =  filePS
fAtt@Conventions   = "None"
fAtt@creation_date = systemfunc ("date")
fileattdef( ncdf, fAtt )            ; copy file attributes

; Make time an unlimited dimension
filedimdef(ncdf,"time",-1,True)

;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================
pm!0 = "time"
pm!1 = "lev"
pm!2 = "lat"
pm!3 = "lon"

pm&time = time
pm&lev  = hyam
pm&lat  = lat
pm&lon  = lon

pm@long_name = "Vertical profile of pressure"
pm@units = ps@units
ncdf->PRESSURE = pm

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
