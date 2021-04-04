boxm <- function(n) {
  t <- runif(n+1)
  u1 <- t[1]
  r <- array()
  for (j in 1:n) {
    u2 <- t[j+1]
    r[j] <- sqrt(-2*log(u1)) * cos(2*pi*u2)
    u1 <- u2
  }
  r
}
