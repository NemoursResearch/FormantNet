#----------------------------------------------------------------------------
# frmAmpCorr -- Formant Amplitude Correction
#
# The amplitude of a resonator is proportional to both its frequency and
# bandwidth. This function is designed to calculate a scaling factor that
# will compensate both frequency and bandwidth factors to result in unit
# amplitude at the resonator center frequency (i.e. amplitude = 1.0 at
# the resonator characteristic frequency).
#
# The amplitude at the resonant frequency is inversely proportional to the
# bandwidth plus a constant offset that is a nonlinear function of the
# bandwidth. The latter function is approximated here via a 4th order
# polynomial.
#
# See also: function formant() that calculates the discrete spectrum of the
# resonator impulse response given the resonator frequency and bandwidth as
# well as the bandwidth of the spectrum and the number of discrete spectral
# bins. The formant function now also includes a scale argument that
# defaults to FALSE for compatibility with previous versions of the function.
# If scale=TRUE, the amplitude spectrum is scaled to have unit gain at the
# resonator frequency.
#
# 12/23/19 -- htb
#---------------------------------------------------------------------------
#
frmAmpCorr <- function(freq, band) {
  coe <- c(-1.683882e-03,4.234928e-04,-1.231765e-07,2.421068e-11,-2.066097e-15)
  m <- 1.0 / band
  b <- coe[1] + coe[2]*band + coe[3]*band^2 + coe[4]*band^3 + coe[5]*band^4
#  cat(paste0("m = ",m,"; b = ",b,"\n"))
  amp <- m*freq + b
  return(amp)
}
