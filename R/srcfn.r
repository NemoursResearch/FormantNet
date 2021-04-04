#-------------------------------------------------------------------
# srcfn -- Wrapper function to call lfm and/or generate random noise
#          to create a single epoch of pitch data for analysis or
#          synthesis.
#
# Args:
#  pars -- A vector of parameters that will be referenced by name
#          to select specific source-related parameters. Typically
#          this is the same vector that is also passed to vtfn to
#          generate the spectral response of the vocal tract. The
#          parameters used by srcfn are:
#          oq - open quotient - LTM Te = oq * T0
#          tprat - The ratio of tp/te clamped to the range:
#                  .5 <= tprat <= .95
#          ta - the duration of interval Ta in the LF model
#          av - Amplitude of voicing
#          ah - Amplitude of hiss applied only during the open phase
#               of voiced epochs.
#          af - Amplitude of frication applied only for voiceless
#               epochs.
#  vcd  -- Voicing status 1=voiced; 0=voiceless
#  t0   -- epoch duration in seconds.
#  fs   -- Sampling frequency
#
# 1/10/20 -- htb
#-------------------------------------------------------------------
srcfn <- function(pars, vcd=1, t0=0.01, fs=16000) {
  if (vcd) {
#    if (pars['tprat'] < 0.5)
#      pars['tprat'] = 0.5
#    else if (pars['tprat'] > 0.95)
#      pars['tprat'] = 0.95
#    if (pars['ta'] <.0001)
#      pars['ta'] <- .0001
#    if (pars['oq'] < .1)
#      pars['oq'] <- 0.1

    ug <- lfm(T0=t0,
              TE=pars['oq']*t0,
              TP=pars['oq']*t0*pars['tprat'],
              TA=pars['ta'],
              AV=pars['av'],
              Fs=fs)

    if (pars['ah'] > 0) {
#      n <- pars['oq'] * t0 * fs
      af <- 10^(pars['ah']/20)
#      ug[1:n] <- ug[1:n] + af * rnorm(n, 0, 1) * hanwin(n)
      TE = t0 * pars['oq']
      TP = TE * pars['tprat']
      na <- round(fs * (t0 - TE), 0)
      nb <- round(fs * TP, 0)
      nt0 <- round(fs *t0)
      nte <- round(fs * TE)
      n <- na + nb
      print(sprintf("TE=%f; TP=%f; na=%d; nb=%d; nt0=%d; nte=%d", TE, TP, na, nb, nt0, nte))
      tug <- af * rnorm(n, 0, 1) * hanwin(n)
      ug[1:nb] = ug[1:nb] + tug[(na+1):n]
      ug[(nte+1):nt0] = ug[(nte+1):nt0] + tug[1:na]
    }
  } else {
    n <- t0 * fs
    af <- 10^(pars['af']/20)
    ug <- af * rnorm(n, 0, 1)
  }

  ug
}