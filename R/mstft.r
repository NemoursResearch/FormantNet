mstft <- function(x) {
  n <- length(x)
  s <- attr(x, "win:sum")
  if (is.null(s))
    s <- n
  s <- 2.0 / s
  no2 <- n / 2
  no2p1 <- no2 + 1
  s * abs(fft(x))[1:no2p1]
}

mstftdb <- function(x) {
  20 * log10(mstft(x) + 1)
}