#------------------------------------------------------------
# wav2r -- Waveform to Autocorrelation
#
# Calculates m (default = 18) autocorrelation coefficients from an
# input signal.
#------------------------------------------------------------
wav2r <- function(s, m=18) {
  ns = length(s)
  r = array()
  r[1] = sum(s^2)
  for (j in 2:m) {
    r[j] = sum(s[1:(ns-j+1)] * s[j:ns]) 
  }
  return(r)
}
