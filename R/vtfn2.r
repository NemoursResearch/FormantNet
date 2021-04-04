#-------------------------------------------------------------------
# vtfn -- Generate spectrum of vocal tract response given formant
#         frequency, amplitude, and bandwidth.
#
# Args:
#  pars -- A vector of named parameters. The first parameter is the
#          number of poles and zeros described in subsequent triplets
#          (frequency, bandwidth, amplitude) of parameters. Poles are
#          distinguished from zeros by having positive instead of 
#          negative frequency values.
#
#  np   -- The number of points of spectrum values to cover the
#          Nyquist range from 0 - fs/2. This is deceptive because,
#          to be consistent with FFT concentions, there will really
#          be an np+1 point spectrum returned with the first value
#          corresponding to DC and the last to the Nyquist frequency.
#
#  fs   -- Sampling frequency in Hz.
#
# Notes: This is the "parallel formant" version with each formant
#        curve calculated independently and added to the others.
#        Each formant is also multiplied by an amplitude factor
#        that is passed as a quantity in dB via the pars vector.
#        Additionally, this function called the underlying formant
#        function with scaled=TRUE so that every formant response
#        curve is generated with a theoretical peak amplitude of 1.0.
#
# 1/10/20 -- htb
#-------------------------------------------------------------------

vtfn2 <- function(pars, np=512, fs=16000) {
  maxf <- fs/2
  nres <- pars[1]
  s <- rep(0,np+1)
  z <- rep(1,np+1)
  for (j in seq(2,nres*3,3)) {
    af <- 10^(pars[j+2]/20)
    if (pars[j] > 0) {
      s <- s + af * formant(pars[j],pars[j+1],np,maxf)
      #cat(paste0('pole: af=',af,'; f=',pars[j],'; b=',pars[j+1],'\n'))
    } else {
      z <- z * af / formant(pars[j],pars[j+1],np,maxf)
      #cat(paste0('zero: af=',af,'; f=',pars[j],'; b=',pars[j+1],'\n'))
    }
  }
  s*z
}