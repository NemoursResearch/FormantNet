
blkwin <- function(np) {
  rad <- seq(0,2*pi,2*pi/(np-1))
  w <- (.42 - .5*cos(rad) + .08*cos(2*rad))
  return(w)
}

hamwin <- function(np) {
  w <- .54 - .46 * cos(seq(0,2*pi,2*pi/(np-1)))
  return(w)
}

hanwin <- function(np) {
  w <- .5 - .5 * cos(seq(0,2*pi,2*pi/(np-1)))
  return(w)
}
