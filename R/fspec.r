# fspec -- formant based frequency spectrum
#
fspec <- function(f, n=128, nyquist=8000) {
  sp = formant(f[1], 100, n, nyquist)
  for (j in 2:length(f)) {
    sp = sp + formant(f[j], 100, n, nyquist) / (f[j]/500)
  }
#  sp = sp * hpc2(length(f), seq(0, nyquist, nyquist/n), f[1])
  return(sp)
}
