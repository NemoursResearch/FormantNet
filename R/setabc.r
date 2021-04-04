setabc <- function(f, fb, sr) {
  pit = pi/sr
  r = exp(-pit*fb)
  coefs = c(0,0,0)
  coefs[3] = -r * r
  coefs[2] = 2*r*cos(2*pit*f)
  coefs[1] = 1 - coefs[2] - coefs[3]
  if (f < 0.0) {
    coefs[1] = 1 / coefs[1]
    coefs[2] = -coefs[1] * coefs[2]
    coefs[3] = -coefs[1] * coefs[3]
  }
  return(coefs)
}

res <- function(x,f,fb,sr) {
  cf = setabc(f,fb,sr)
  m1 = 0
  m2 = 0
  y = x
  if (f >= 0) {
    for (j in 1:length(x)) {
      y[j] = x[j]*cf[1] + m1*cf[2] + m2*cf[3]
      m2 = m1
      m1 = y[j]
    }
  } else {
    for ( j in 1:length(x)) {
      y[j] = x[j]*cf[1] + m1*cf[2] + m2*cf[3]
      m2 = m1
      m1 = x[j]
    }
  }
  return(y)
}
