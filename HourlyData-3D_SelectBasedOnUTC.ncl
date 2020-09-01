;--- Goal: Convert model output time to UTC, then save relevant morning times for US.
;    Date: 31 Aug 2020
;    Author: Meg Fowler

begin

; Set filenames
fileName_in = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrPS.nc"
fileName_out = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrPS-UTCsel.nc"


; Read in file with data and time
fIn  = addfile(fileName_in, "r")

; Read in time
print("Reading in time and converting to UTC...\n")
time = fIn->time
lat  = fIn->lat
lon  = fIn->lon

; Convert to UTC  time
utc_date = cd_calendar(time, 0)

; Save 10-14 UTC
iKeep = ind(utc_date(:,3).ge.10 .and. utc_date(:,3).le.14)
utc_keep = utc_date(iKeep,:)

; Limit lat and lon as well... 
iLat = ind(lat.ge.20. .and. lat.le.60.)
iLon = ind(lon.ge.220. .and. lon.le.300.)
latSel = lat(iLat)
lonSel = lon(iLon)

; Isolate only these times in the variable too
print("Read in PS and selecting only 5 hr/day")
PS = fIn->PS
PS_keep = PS(iKeep,iLat,iLon)

; Make 1D dimension of time to use for actually writing out
rawTime_keep = time(iKeep)

; Also isolate yr/mon/day/hr arrays
UTChr = utc_keep(:,3)
UTCday = utc_keep(:,2)
UTCmon = utc_keep(:,1)
UTCyr = utc_keep(:,0)

; ------------------------------------
; Write out to netCDF file...
;   Using code from: https://www.ncl.ucar.edu/Applications/method_1.shtml

print("Creating netCDF file")

system("/bin/rm -f " + fileName_out)   ; remove any pre-existing file

; setfileoption("nc","format","LargeFile")  ; set to allow large variables to be written out 

ncdf = addfile(fileName_out,"c")  ; open output netCDF file

;===================================================================
; create global attributes of the file (optional)
;===================================================================
fAtt               = True            ; assign file attributes
fAtt@title         = "NCL used to get UTC; 4 early morning hours saved out."
fAtt@source_file   =  fileName_in
fAtt@Conventions   = "None"
fAtt@creation_date = systemfunc ("date")
fileattdef( ncdf, fAtt )            ; copy file attributes

;===================================================================
; make time an UNLIMITED dimension; recommended  for most applications
;===================================================================
filedimdef(ncdf,"time",-1,True) 

;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================

PS_keep!0 = "time"
PS_keep!1 = "lat"
PS_keep!2 = "lon"

PS_keep&time = rawTime_keep
PS_keep&lat  = latSel
PS_keep&lon  = lonSel

PS_keep@long_name = "Sfc Pressure in UTC time."
PS_keep@units = PS@units
ncdf->PS = PS_keep

; -----------------------------
UTChr!0 = "time"
UTCday!0 = "time"
UTCmon!0 = "time"
UTCyr!0 = "time"

UTChr&time = rawTime_keep
UTCday&time = rawTime_keep
UTCmon&time = rawTime_keep
UTCyr&time  = rawTime_keep

UTChr@long_name = "Hour in UTC"
UTCday@long_name = "Day in UTC"
UTCmon@long_name = "Mon in UTC"
UTCyr@long_name = "Yr in UTC"

ncdf->UTC_hr = UTChr
ncdf->UTC_day = UTCday
ncdf->UTC_mon = UTCmon
ncdf->UTC_yr = UTCyr

print("File created: ")
print(fileName_out)

end



