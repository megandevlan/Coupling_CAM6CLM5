;--- Goal: Convert model output time to UTC, then save relevant morning times for US.
;    Date: 31 Aug 2020
;    Author: Meg Fowler

begin

; Set filenames
Q2mFile_in = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrQREFHT.nc"
T2mFile_in = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrTREFHT.nc"
lhFile_in  = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrLHFLX.nc"
shFile_in  = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrSHFLX.nc"
pbhFile_in = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrPBLH.nc"
psFile_in  = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrPS.nc"

fileName_out = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrVars-CONUS.nc"


; Read in files 
fIn_Q  = addfile(Q2mFile_in, "r")
fIn_T  = addfile(T2mFile_in, "r")
fIn_LH = addfile(lhFile_in, "r") 
fIn_SH = addfile(shFile_in, "r") 
fIn_PS = addfile(psFile_in, "r") 
fIn_PBH = addfile(pbhFile_in,"r") 

; Read in time
print("Reading in time and converting to UTC...\n")
time = fIn_Q->time
lat  = fIn_Q->lat
lon  = fIn_Q->lon

; Convert to UTC  time
utc_date = cd_calendar(time, 0)

; Limit lat and lon as well... 
iLat = ind(lat.ge.20. .and. lat.le.60.)
iLon = ind(lon.ge.220. .and. lon.le.300.)
latSel = lat(iLat)
lonSel = lon(iLon)

; Isolate only these regions in the variable too
print("Selecting CONUS for ... QREFHT")
QREFHT = fIn_Q->QREFHT
QREFHT_keep = QREFHT(:,iLat,iLon)

print("                    ... TREFHT")
TREFHT = fIn_T->TREFHT
TREFHT_keep = TREFHT(:,iLat,iLon)

print("                    ... LHFLX")
LHFLX = fIn_LH->LHFLX
LHFLX_keep = LHFLX(:,iLat,iLon)

print("                    ... SHFLX")
SHFLX = fIn_SH->SHFLX
SHFLX_keep = SHFLX(:,iLat,iLon)

print("                    ... PS")
PS = fIn_PS->PS
PS_keep = PS(:,iLat,iLon)

print("                    ... PBLH")
PBLH = fIn_PBH->PBLH
PBLH_keep = PBLH(:,iLat,iLon)


; Also isolate yr/mon/day/hr arrays
UTChr  = utc_date(:,3)
UTCday = utc_date(:,2)
UTCmon = utc_date(:,1)
UTCyr  = utc_date(:,0)

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
fAtt@title         = "NCL used to get UTC; CONUS gridcells isolated and saved."
fAtt@source_file   = "f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.*"
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

QREFHT_keep!0 = "time"
QREFHT_keep!1 = "lat"
QREFHT_keep!2 = "lon"

QREFHT_keep&time = time
QREFHT_keep&lat  = latSel
QREFHT_keep&lon  = lonSel

QREFHT_keep@long_name = "Reference height humidity  in UTC time."
QREFHT_keep@units = QREFHT@units
ncdf->QREFHT = QREFHT_keep

; ---- 

TREFHT_keep!0 = "time"
TREFHT_keep!1 = "lat"
TREFHT_keep!2 = "lon"

TREFHT_keep&time = time
TREFHT_keep&lat  = latSel
TREFHT_keep&lon  = lonSel

TREFHT_keep@long_name = "Reference height temperature in UTC time."
TREFHT_keep@units = TREFHT@units
ncdf->TREFHT = TREFHT_keep

; ---- 

LHFLX_keep!0 = "time"
LHFLX_keep!1 = "lat"
LHFLX_keep!2 = "lon"

LHFLX_keep&time = time
LHFLX_keep&lat  = latSel
LHFLX_keep&lon  = lonSel

LHFLX_keep@long_name = "Latent heat flux in UTC time."
LHFLX_keep@units = LHFLX@units
ncdf->LHFLX = LHFLX_keep

; ---- 

SHFLX_keep!0 = "time"
SHFLX_keep!1 = "lat"
SHFLX_keep!2 = "lon"

SHFLX_keep&time = time
SHFLX_keep&lat  = latSel
SHFLX_keep&lon  = lonSel

SHFLX_keep@long_name = "Sensible heat flux in UTC time."
SHFLX_keep@units = SHFLX@units
ncdf->SHFLX = SHFLX_keep

; ---- 

PS_keep!0 = "time"
PS_keep!1 = "lat"
PS_keep!2 = "lon"

PS_keep&time = time
PS_keep&lat  = latSel
PS_keep&lon  = lonSel

PS_keep@long_name = "Surface presure in UTC time."
PS_keep@units = PS@units
ncdf->PS = PS_keep

; ---- 

PBLH_keep!0 = "time"
PBLH_keep!1 = "lat"
PBLH_keep!2 = "lon"

PBLH_keep&time = time
PBLH_keep&lat  = latSel
PBLH_keep&lon  = lonSel

PBLH_keep@long_name = "Planetary boundary layer height in UTC time."
PBLH_keep@units = PBLH@units
ncdf->PBLH = PBLH_keep

; -----------------------------
UTChr!0 = "time"
UTCday!0 = "time"
UTCmon!0 = "time"
UTCyr!0 = "time"

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

print("File created: ")
print(fileName_out)

end



