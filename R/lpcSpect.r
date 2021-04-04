lpcSpect <- function(samps, np=256, m=18) {
  np2 = np + np
  lpc <- levnsn(wav2r(samps,m), m)
  h <- c(lpc$a, rep(0, np2 - m))
  af <- 1/np
  sp <- 20*log10(as.vector(af*sqrt(lpc$v))/abs(fft(h))[1:(np+1)])
  sp
}