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

filePS = "/Users/meganfowler/gp_fuse/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrPS-UTCsel.nc"
fileHybrid = "/Users/meganfowler/gp_fuse/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h0.1980-01.nc"
fileOut = "/Users/meganfowler/Documents/NCAR/Analysis/Coupling_initial/data/hrSim_CONUS/f.e21.FHIST_BGC.f09_f09_mg17.hourlyOutput.001.cam.h1.1979-1981_hrP-levels-UTCsel.nc"

; Open file with surface pressure
fPS = addfile(filePS,"r")

; Open a file with hybrid information
fHyb = addfile(fileHybrid,"r")

; Read a few variables
hyam = fHyb->hyam
hybm = fHyb->hybm
ps = fPS->PS
p0 = fHyb->P0
time = fPS->time
lat = fPS->lat
lon = fPS->lon

; ------ Compute pressure levels --------

; Compute pressure levels
pm = pres_hybrid_ccm(ps,p0,hyam,hybm)

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

print("File created!")

end
