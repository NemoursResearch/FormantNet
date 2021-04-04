mstft <- function(x, asc=NULL) {
  n <- length(x)
  no2 <- n / 2
  no2p1 <- no2 + 1
  if (is.null(asc)) {
     af = 2.0 / n
  } else {
     af = 2.0 / asc
  }
  return(af * abs(fft(x))[1:no2p1])
}

mstftdb <- function(x, asc=NULL) {
  20 * log10(mstft(x, asc) + 1)
}