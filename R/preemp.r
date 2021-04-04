#---------------------------------------------------------------------
# Calculates the first-differenced waveform for preemphasis.
#---------------------------------------------------------------------
preemp <- function(s, coef=1.0) {
  n <- length(s)
  return(c(0,s[2:n]-coef*s[1:(n-1)]))
}
