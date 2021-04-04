#---------------------------------------------------------------------
# cep  --  Cepstrally Smoothed Spectrum
#
# Calculates a cepstrally smoothed spectrum for the input signal 's'
# with at most 'b' real cepstrum coefficients retained and all others
# set to zero.
#
# This function assumes that the input is even length and returns
# N/2 + 1 points.
#--------------------------------------------------------------------
#
cep <- function(s, b=32)
{
  t <- fft(10*log10(abs(fft(s)) + 1 ))
  t[b:(length(s)-b)] = 0
  return(abs(fft(t,inverse=T))[1:(length(s)/2)]/length(s))
}
