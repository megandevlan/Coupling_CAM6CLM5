; --- Goal: Convert specific to relative humidity for monthly data using NCL's function. 
;     Date: 14 Sept 2020
;     Author: Meg D. Fowler
;
; -------------------------------------------------------------------------------------
begin

; Set filename
fileIn = "/glade/work/mdfowler/data/CESM2_cam5/f.e20.FHIST.f09_f09.cesm2_1_cam5.001.cam.h0.2000-2005_refHTs.nc"
fileName_out = "/glade/work/mdfowler/data/CESM2_cam5/f.e20.FHIST.f09_f09.cesm2_1_cam5.001.cam.h0.2000-2005_refHTs_RHREFHT.nc"

; Read in file 
fIn = addfile(fileIn,"r")

; Read in dimensions
time = fIn->time
lat  = fIn->lat
lon  = fIn->lon

; Read in variables 
print("Read in data...")
QREFHT = fIn->QREFHT
TREFHT = fIn->TREFHT
PS     = fIn->PS 

; Compute relative humidity using NCL function 
print("Computing RH...")
RH = relhum(TREFHT, QREFHT, PS) 

; ------------------------------------
; Write out to netCDF file...
;   Using code from: https://www.ncl.ucar.edu/Applications/method_1.shtml

print("Creating netCDF file...")

system("/bin/rm -f " + fileName_out)   ; remove any pre-existing file

; setfileoption("nc","format","LargeFile")  ; set to allow large variables to be written out 

ncdf = addfile(fileName_out,"c")  ; open output netCDF file

;===================================================================
; create global attributes of the file (optional)
;===================================================================
fAtt               = True            ; assign file attributes
fAtt@title         = "NCL used to compute RHREFHT from QREFHT, TREFHT, and PS."
fAtt@source_file   = fileIn
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
RH!0 = "time"
RH!1 = "lat"
RH!2 = "lon" 

RH&time = time
RH&lat  = lat
RH&lon  = lon
RH@long_name = "Reference height relative humidity" 
RH@units = "%"
ncdf->RHREFHT = RH 

; Add the other variables I want to keep in the file still 
ncdf->TREFHT = TREFHT
ncdf->PS = PS
ncdf->QREFHT = QREFHT

print("File created: ")
print(fileName_out)

end
