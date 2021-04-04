ola <- function(w, nrep) {
  l <- length(w)
  lo2 <- l/2
  tot <- l + nrep * lo2
  out <- rep(0, tot)
  offset <- lo2
  out[1:l] <- w
  for (j in 1:nrep) {
    out[offset:(offset+l-1)] <- out[offset:(offset+l-1)] + w
    offset <- offset + lo2
  }
  out
}