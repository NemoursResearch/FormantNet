#----------------------------------------------------------------------------------
# zcep -- Compute Bark Cepstral coefficients for a windowed region of speech
#
# Synopsis:  zcoefs <- zcep(sr, data, nc)
#            sr   - sample rate of the speech data needed for calculation of Bark
#                   filter bands.
#            data - a vector of windowed waveform samples such as returned by
#                   getFrame()
#            nc   - Number of Bark Cepstral coefficients to return
#
# Description:
#  The zcep function for R calls the libcdsp zcep funtion from the standard SRL
#  libcdsp via an interface/wrapper function called r_zcep that translates the
#  the passed args to the correct format.
#
#  When called, zcep first checks to see that the necessary r_zcep function is
#  loaded. If not, the shared object lib is loaded. The lib name/path is system-
#  dependent and right now, we check for either Linux or macOS (Darwin) to load
#  the correct lib. If run on another OS (e.g. Windows), the function will print
#  an error message and return NA.
#
#  If successful, the function will return a vector of cepstral coefficients.
#
#  HTB -- 3/24/21
#---------------------------------------------------------------------------------
#

zcep <- function(sr, data, nc) {
  #
  # Make sure the r_zcep function is accessible and if not, load libcdsp
  #
  if (!is.loaded("r_zcep")) {
    if (Sys.info()['sysname'] == "Linux") {
      dyn.load("/usr/local/srl/lib/libcdsp.so")
    } else if (Sys.info()['sysname'] == "Darwin") {
      dyn.load("/usr/local/srl/lib/libcdsp.dylib")
    } else {
      message("zcep: libcdsp not loaded!")
      return(NA)
    }
  }

  totenergy = as.double(0.0)
  rtn = .C("r_zcep", sr_p=as.double(sr),
               npts_p=as.integer(length(data)),
               data=as.single(data),
               ncoef_p=as.integer(nc),
               coefs=as.single(rep(0,nc)),
               totenergy)
  return(as.double(rtn$coefs))
}