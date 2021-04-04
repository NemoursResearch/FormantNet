
notchFilt <- function(inp, n)
{
  for (j in 1:length(inp)) {
    out = n$a[1]*inp[j] + n$a[2]*n$xm[1] + n$a[3]*n$xm[2] + n$b[1]*n$ym[1] + n$b[2]*n$ym[2]
    n$xm[2] = n$xm[1]
    n$xm[1] = inp[j]
    n$ym[2] = n$ym[1]
    n$ym[1] = out
    inp[j] = out
  }
  inp
}

getNotch <- function(f, b, sr=16000)
{
  n <- list(a=c(0,0,0),b=c(0,0),xm=c(0,0),ym=c(0,0))
  tpf = 2.0 * pi * f / sr
  r = 1.0 - 3.0 * b/sr
  k = (1.0 - 2.0*r*cos(tpf) + r*r) / (2.0 - 2.0*cos(tpf))
  n$a[1] = k
  n$a[3] = k
  n$a[2] = -2.0 * k * cos(tpf)
  n$b[1] = 2.0 * r * cos(tpf)
  n$b[2] = -(r*r)
  n
}

getNarrowBand <- function(f, b, sr=16000)
{
  n <- list(a=c(0,0,0),b=c(0,0),xm=c(0,0),ym=c(0,0))
  tpf = 2.0 * pi * f / sr
  r = 1.0 - 3.0 * b/sr
  k = (1.0 - 2.0*r*cos(tpf) + r*r) / (2.0 - 2.0*cos(tpf))
  n$a[1] = 1.0 - k
  n$a[2] = 2.0 * (k - r) * cos(tpf)
  n$a[3] = r*r - k
  n$b[1] = 2.0 * r * cos(tpf)
  n$b[2] = -(r*r)
  n
}
