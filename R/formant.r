#--------------------------------------------------------------------------
# formant -- Formant response spectrum
#
# Calculate a single-formant impulse response function given the formant 
# frequency (f), bandwidth (b), the number of points in the spectrum (np),
# and the maximum frequency.
#
# 12/23/19 -- htb
# Added scale option set to default to FALSE. If true, call frmAmpCorr()
# to obtain a scaling factor based on the frequency and bandwidth that
# can be applied to force unity gain for the resonator at the resonant
# frequency.
#

formant <- function(f, b, np, maxf, scale=FALSE) {
  if (length(f) != length(b) && length(b) > 1)
    return(print("formant: number of bandwidths != number of formants"))

  s <- matrix(seq(0, maxf, maxf/np),nrow=length(f),
              ncol=np+1, byrow=TRUE)
  if (scale)
    scale = 1/frmAmpCorr(f, b)
  else
    scale = 1.0

  bsq <- .25 * b^2
  anum <- (f^2 + bsq)
  return(scale * anum / sqrt(((s-f)^2 + bsq) * ((s+f)^2 + bsq)))
}

formant1d <- function(f, b, np, maxf) {
#
# Calculate a single-formant response function given the
# formant frequency (f), bandwidth (b), the number of points
# in the spectrum (np), and the maximum frequency.
#
  s <- seq(0, maxf, maxf/np)
  bsq <- .25 * b^2
  fsq <- f^2
  fdif <- s - f
  fsum <- s + f
  a <- fdif^2 + bsq
  b <- fsum^2 + bsq
  step1 <- 2.0 * f / sqrt(a*b)
  step2 <- (fsq + bsq) * (a*fsum - b*fdif) / (a*b)^1.5
  return (step1 - step2)
}

band1d <- function(f, b, np, maxf) {
#
# Calculate a single-formant response function given the
# formant frequency (f), bandwidth (b), the number of points
# in the spectrum (np), and the maximum frequency.
#
  s <- seq(0, maxf, maxf/np)
  bsq <- .25 * b^2
  bo2 <- 0.5 * b
  fsq <- f^2
  fdif <- s - f
  fsum <- s + f
  a <- fdif^2 + bsq
  b <- fsum^2 + bsq
  step1 <- bo2 / sqrt(a*b)
  step2 <- (fsq + bsq) * bo2 * (a+b) / (2.0*(a*b)^1.5)
  return (step1 - step2)
}

hpc1 <- function(nf, f, f1) {
#
# This is the higher pole correction originally given by
# Fant (1960) Acoustic Theory of Speech Production
#
  s <- pi^2/8
  for (j in c(1:nf)) {
    s <- s - 1.0/(j*2 - 1)^2
  }
  fr <- f/f1
  return(exp(fr^2 * s))
}

hpc2 <- function(nf, f, f1) {
#
# This is the second order higher pole correction given by
# Gold & Rabiner (1968) Analysis of Digital and Analog Formant
#   Synthesizers. IEEE Transactions on Audio and Electroacoustics,
#   16, 81-94.
#
  s <- pi^2/8
  t <- pi^4/96
  for (j in c(1:nf)) {
    s <- s - 1.0/(j*2 - 1)^2
    t <- t - 1.0/(j*2 - 1)^4
  }
  fr <- f/f1
  return(exp(fr^2 * s + 0.5 * fr^4 * t))
}

hpc3 <- function(nf, f, f1) {
#
# This is the second order higher pole correction given by
# Gold & Rabiner (1968) Analysis of Digital and Analog Formant
#   Synthesizers. IEEE Transactions on Audio and Electroacoustics,
#   16, 81-94.
#
  x <-  c(0.23370055, 0.01467803,0.122589439,0.002332353,0.0825894390,
          0.0007323526,0.0621812758,0.0003158595,0.0498355967,0.0001634437,
          4.157113e-02,9.514233e-05,3.565397e-02,6.012955e-05,3.120953e-02,
          4.037646e-05)
  dim(x) = c(2,8)
  x = t(x)
  s = x[nf,1]
  t = x[nf,2]
  fr <- f/f1
  return(exp(fr^2 * s + 0.5 * fr^4 * t))
}

frmnt <- function(f, ff, fb)
{
  if (ff < 0.0) {
    ff = abs(ff)
    pole = F
  } else {
    pole = T
  }
  bsq <- 0.25 * fb^2
  w <- ff^2 + bsq
  if (pole)
    return (w / sqrt(((f - ff)^2 + bsq) * ((f + ff)^2 + bsq)))
  else
    return (sqrt(((f - ff)^2 + bsq) * ((f + ff)^2 + bsq)) / w)
}
   
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
# 12/23/19 -- htb original code
#  coe <- c(-1.683882e-03,4.234928e-04,-1.231765e-07,2.421068e-11,-2.066097e-15)
#  m <- 1.0 / band
#  b <- coe[1] + coe[2]*band + coe[3]*band^2 + coe[4]*band^3 + coe[5]*band^4
#  cat(paste0("m = ",m,"; b = ",b,"\n"))
#  amp <- m*freq + b
#  return(amp)
#
# 5/5/20  -- htb
#  After looking at the actual formula in formant(), I realized that I was
#  swatting flies with a cannon and replaced the 4th order polynomial fit
#  from the original code, with a simple closed-form exact calculation.
#  YEESH!!
#
#---------------------------------------------------------------------------
#
frmAmpCorr <- function(f, b) {
  bsq <- .25 * b^2
  anum <- (f^2 + bsq)
  return(anum / sqrt(bsq * ((f+f)^2 + bsq)))
}
