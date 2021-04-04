#-------------------------------------------------------------------
# vtfn -- Generate spectrum of vocal tract response given formant
#         frequency, amplitude, and bandwidth.
#
# Args:
#  pars -- A vector of named parameters for 8 formants. Each
#          parameter has a name such as f[1-8][fba] to specify
#          formant number (1-8) and parameter (f(requency), 
#          b(andwidth), or a(mplitude)). All frequencies and 
#          bandwidths specified in Hertz, and amplitudes in dB.
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

vtfn <- function(pars, np=512, fs=16000) {
  maxf <- fs/2
  a1s <- 10^(pars['f1a']/20)
  a2s <- 10^(pars['f2a']/20)
  a3s <- 10^(pars['f3a']/20)
  a4s <- 10^(pars['f4a']/20)
  a5s <- 10^(pars['f5a']/20)
  a6s <- 10^(pars['f6a']/20)
  a7s <- 10^(pars['f7a']/20)
  a8s <- 10^(pars['f8a']/20)

  f1 <- formant(pars['f1f'], pars['f1b'], np, maxf, scale=F)
  f2 <- formant(pars['f2f'], pars['f2b'], np, maxf, scale=F)
  f3 <- formant(pars['f3f'], pars['f3b'], np, maxf, scale=F)
  f4 <- formant(pars['f4f'], pars['f4b'], np, maxf, scale=F)
  f5 <- formant(pars['f5f'], pars['f5b'], np, maxf, scale=F)
  f6 <- formant(pars['f6f'], pars['f6b'], np, maxf, scale=F)
  f7 <- formant(pars['f7f'], pars['f7b'], np, maxf, scale=F)
  f8 <- formant(pars['f8f'], pars['f8b'], np, maxf, scale=F)
  
  s <- a1s*f1 + a2s*f2 + a3s*f3 + a4s*f4 + a5s*f5 +
    a6s*f6 + a7s*f7 + a8s*f8
  s
}