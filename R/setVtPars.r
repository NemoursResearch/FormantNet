setVtPars <- function(par, spect, fsc) {
  pks <- peaks(spect)
  pks$x <- fsc * pks$x
  fidx <- c(7,10,13,16,19,22,25,28)
  aidx <- fidx + 2
  for (i in 1:length(pks$x)) {
    par[fidx[i]] <- pks$x[i]
    par[aidx[i]] <- pks$y[i]
  }
  par
}
