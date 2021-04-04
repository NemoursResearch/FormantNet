#-------------------------------------------------------------------
# iniPars() -- Returns default parameter values for generating
#              model spectra from our standard 30-parameter model.
#
# Notes: This function declares separate VT and SRC parameter
#        vectors and assigns names to each element, but then
#        concatenates the two vectors into a single output.
#
# 1/10/20 -- htb
#-------------------------------------------------------------------
iniPars <- function() {
  vtPars <- c(500,80,60,
               1500,100,60,
               2500,120,60,
               3500,240,60,
               4500,480,60,
               5500,500,60,
               6500,500,60,
               7500,500,60)
  names(vtPars) <- c("f1f","f1b","f1a",
                      "f2f","f2b","f2a",
                      "f3f","f3b","f3a",
                      "f4f","f4b","f4a",
                      "f5f","f5b","f5a",
                      "f6f","f6b","f6a",
                      "f7f","f7b","f7a",
                      "f8f","f8b","f8a")
  srcPars <- c(.6,.667,.0002,24,0,0)
  names(srcPars) <- c("oq","tprat","ta","av","af","ah")
  
  pars <- c(srcPars,vtPars)
  pars
}
#-------------------------------------------------------------------
# lbPars() -- experimental vector with elements corresponding to the
#             pars vector that lists lower bounds on the allowed
#             value of each parameter. These need to be considered
#             much more carefully.
#-------------------------------------------------------------------
lbPars <- function() {
  vtPars <- c(150,20,0,
              750,20,0,
              1750,30,0,
              2750,30,0,
              3750,40,0,
              4750,40,0,
              5750,50,0,
              6750,50,0)
  names(vtPars) <- c("f1f","f1b","f1a",
                     "f2f","f2b","f2a",
                     "f3f","f3b","f3a",
                     "f4f","f4b","f4a",
                     "f5f","f5b","f5a",
                     "f6f","f6b","f6a",
                     "f7f","f7b","f7a",
                     "f8f","f8b","f8a")
  srcPars <- c(.1,.6,0.0,0,0,0)
  names(srcPars) <- c("oq","tprat","ta","av","af","ah")
  
  pars <- c(srcPars,vtPars)
  pars
}
#-------------------------------------------------------------------
# ubPars() -- experimental vector with elements corresponding to the
#             pars vector that lists upper bounds on the allowed
#             value of each parameter. These need to be considered
#             much more carefully.
#-------------------------------------------------------------------
ubPars <- function() {
  vtPars <- c(1000,1000,66,
              3000,1000,66,
              4000,1000,66,
              5000,1000,66,
              6000,1000,66,
              7000,1000,66,
              8000,1000,66,
              10000,1000,66)
  names(vtPars) <- c("f1f","f1b","f1a",
                     "f2f","f2b","f2a",
                     "f3f","f3b","f3a",
                     "f4f","f4b","f4a",
                     "f5f","f5b","f5a",
                     "f6f","f6b","f6a",
                     "f7f","f7b","f7a",
                     "f8f","f8b","f8a")
  srcPars <- c(.95,.95,.0009,68,68,68)
  names(srcPars) <- c("oq","tprat","ta","av","af","ah")

  pars <- c(srcPars,vtPars)
  pars
}

#-------------------------------------------------------------------
# estSpect -- Function to generate a spectrum from parameters for
#             use in fitting parameters.
# Args:
#  pars  -- Vector of parameters for both source function and VT
#           response function. Each element of the vector needs to
#           be named and is selected by name. See iniPars() for the
#           default values and names of each parameter.
#  np    -- Number of points in the output Log Magnitude spectrum.
#           The actual output will have np+1 values (DC is included)
#  fs    -- Sampling frequency in Hz.
#  vcd   -- The voicing status for the output to be generated. This
#           only effects source spectrum calculation.
#  t0    -- The epoch duration (i.e. 1/f0) in seconds. This only
#           directly effects source spectrum, however, "np" should
#           be chosen to cover at least fs*t0*2 samples of data
#           and must be the same for both voiced and voiceless
#           frames.
#  mode  -- One of "vt", "src", or "comb" to determine if the output
#           spectrum is based on only vocal tract, only source, or
#           both combined. This makes it possible to separately
#           display, or fit, the two contributions independetly.
#-------------------------------------------------------------------
estSpect <- function(pars, np=512, fs=16000, vcd=1, t0=0.01,
                     mode="vt") {
  
  if (!(mode %in% c("src","vt","comb")))
      stop("estSpect: error - mode must be one of 'vt', 'src', or 'comb'")
      
  if (mode %in% c("src","comb")) {
    src <- srcfn(pars, vcd, t0, fs)
    np2 <- 2 * np
    if (length(src) < (np2))
      src <- c(src, rep(0,np2-length(src)))
    else if (length(src > (np2)))  # Hope we don't need this!
      src <- src[1:np2]
    srcsp <- 20 * log10(1/np * abs(fft(src)[1:(np+1)]) + 1)
  } else {
    srcsp <- rep(0, np+1)
  }
  if (mode %in% c("vt","comb")) {
    vt <- vtfn(pars, np, fs)
    vtsp <- 20 * log10(vt)
  } else {
    vtsp <- rep(0, np+1)
  }
  return(srcsp+vtsp)
}

#
# Returns a vector of residual errors for each
# point in the target spectrum
#
errSpect <- function(pars, targ, np=512, fs=16000, vcd=1, t0=0.01,
                     mode="vt") {
  est <- estSpect(pars, np, fs, vcd, t0, mode)
  return(targ - est)
}

#
# Returns a Scaler sum-squared error value to
# express the distance from predicted to target.
#
errSpectS <- function(pars, targ, np=512, fs=16000, vcd=1, t0, mode) {
  est <- estSpect(pars, np, fs, vcd, t0, mode)
  return(sum((targ - est)^2))
}