; --- Goal: Take hourly values of PRECT and get 3-hourly sums. 
;           Save those values out to a netCDF along with UTC time of first hour averaged, 
;           (e.g., a 00z average should be the average of 00z-03z). 
;     Author: Meg Fowler 
;     Date: 4 Sept 2020 

begin 

; Set filenames
fileName_in = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979_PRECT.nc"
fileName_out = "/glade/scratch/mdfowler/processedData/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979_PRECT-3hrly_Try2.nc"

; Read in file with data and time 
fIn = addfile(fileName_in, "r") 

; Read in time and PRECT
print("Reading in time and converting to UTC...") 
time = fIn->time
lat  = fIn->lat
lon  = fIn->lon
PRECT = fIn->PRECT 

; Convert to UTC time 
utc_date = cd_calendar(time, 0) 

; Now get 3-hourly sums as in example here: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_sum_n.shtml
nhr = 3

dimp = dimsizes(PRECT)
ntim = dimp(0)
nlat = dimp(1)
mlon = dimp(2)

nhrdim = ntim/nhr 

ptot    = new ( (/nhrdim,nlat,mlon/), typeof(PRECT), getFillValue(PRECT) ) 
UTChr   = new ( (/nhrdim/), typeof(PRECT), getFillValue(PRECT) ) 
UTCday  = new ( (/nhrdim/), typeof(PRECT), getFillValue(PRECT) )
UTCmon  = new ( (/nhrdim/), typeof(PRECT), getFillValue(PRECT) )
UTCyr  = new ( (/nhrdim/), typeof(PRECT), getFillValue(PRECT) )
rawTime_keep = new( (/nhrdim/), typeof(time), getFillValue(time) )

ntStrt = 0 
ntLast = nhr-1

do nt=0,ntim-1,nhr
     ptot(nt/nhr,:,:) = dim_sum_n( PRECT(ntStrt:ntLast,:,:) , 0)
     UTChr(nt/nhr)    = utc_date(ntStrt,3)
     UTCday(nt/nhr)    = utc_date(ntStrt,2)
     UTCmon(nt/nhr)    = utc_date(ntStrt,1)
     UTCyr(nt/nhr)     = utc_date(ntStrt,0)

     rawTime_keep(nt/nhr) = time(ntStrt)

     ntStrt = ntStrt + nhr
     ntLast = ntLast + nhr
end do

; ------- Write out to netCDF file ------ 
;   Using code from: https://www.ncl.ucar.edu/Applications/method_1.shtml

print("Creating netCDF file")

system("/bin/rm -f " + fileName_out)   ; remove any pre-existing file

; setfileoption("nc","format","LargeFile")  ; set to allow large variables to be written out 

ncdf = addfile(fileName_out,"c")  ; open output netCDF file

;===================================================================
; create global attributes of the file (optional)
;===================================================================
fAtt               = True            ; assign file attributes
fAtt@title         = "NCL used to get UTC and compute 3-hr sum of PRECT."
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

copy_VarMeta(PRECT(::nhr,:,:), ptot)          ; meta data;  ::nhr makes time assignment
ptot@long_name = "3-hr accumulated Precip"

ncdf->P_3hr = ptot


; -----------------------------

; -----------------------------
UTChr!0 = "time"
UTCday!0 = "time"
UTCmon!0 = "time"
UTCyr!0 = "time"

UTChr&time  = rawTime_keep
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


;copy_VarMeta(time(::nhr), UTChr)
;copy_VarMeta(time(::nhr), UTCday)
;copy_VarMeta(time(::nhr), UTCmon)

;ncdf->UTC_hr  = UTChr
;ncdf->UTC_day = UTCday
;ncdf->UTC_mon = UTCmon

print("File created: ")
print(fileName_out)

end


