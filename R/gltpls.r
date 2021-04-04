gltpls <- function(oph=.3, cph=.2, t0=0.01, fs=16000) {
  npc <- round(t0 * fs, 0)
  itp <- round(oph * npc, 0)
  itn <- round(cph * npc, 0)
  itz <- npc - (itp + itn)
  rato <- seq(1, itp) / itp
  ratc <- seq(1, itn) / itn
#  gx <- c(3*rato^2 - 2*rato^3, 1 - ratc^2, rep(0, itz))
  gx <- c(.5 - .5*cos(pi*rato), .5 + .5*cos(pi*ratc), rep(0, itz))
  gx
}

gltplsd <- function(oph=.3, cph=.2, t0=0.01, fs=16000) {
  gx <- gltpls(oph, cph, t0, fs)
  n <- length(gx)
  gx <- c(gx[2:n] - gx[1:(n-1)], 0)
  gx
}