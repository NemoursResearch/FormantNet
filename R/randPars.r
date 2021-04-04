randPars <- function() {
  pars <- iniPars()
  ubp <- ubPars()
  lbp <- lbPars()
  for (j in 1:length(pars)) {
    m <- pars[j]
    s <- (ubp[j] - lbp[j]) / 6
    x <- rnorm(1, mean=m, sd=s)
    while ( !(x >= lbp[j] && x <= ubp[j])) {
      x <- rnorm(1, mean=m, sd=s)
    }
    pars[j] <- x
  }
  pars
}
