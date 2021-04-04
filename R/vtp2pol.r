asymp <-function(v=0.0, min=-10, max=+10)
{
  r = max - min;
  return (min + r * (pi/2 + atan(v)) / pi)
}

vtp2pol <- function(scale=0.0, resonance=0.0, shift=0.0, fbfact=0.0, rpfact=0.0)
{
  f = exp(scale/50) * 1000.0 * c(1,3,5,7,9,11,13,15) / 2
  b = exp(resonance/50) * 50.0 + 20.0 * 1:8

  for (j in 1:8) {
	if (j > 1)
	  minv = 50.0 + (f[j] + f[j-1]) / 2.0
	else
	  minv = 50.0
	if (j < 8)
	  maxv = (f[j+1] + f[j]) / 2.0 - 50.0
	else
	  maxv = f[j] + 450.0 * exp(scale/50)
	if (j%%2) {  # if j is odd positive values decrease freq
	  v = asymp(fbfact*.0325, maxv, minv)
	} else {  # if j is even positive values increase freq
	  v = asymp(fbfact*.0325, minv, maxv)
	}
	f[j] = v
  }

  if (rpfact < 0.0) {
	minv = f[1] + 50.0
	maxv = f[2] + (f[2] - minv)
	f[2] = asymp(rpfact*.0325, minv, maxv)
	minv = f[2] + 50.0
	maxv = f[3] + (f[3] - minv)
	f[3] = asymp(rpfact*.0325, minv, maxv)
  } else {
	maxv = f[4] - 50.0
	minv = f[3] - (maxv - f[3])
	f[3] = asymp(rpfact*.0325, minv, maxv)
	maxv = f[3] - 50.0
	minv = f[2] - (maxv - f[2])
	f[2] = asymp(rpfact*.0325, minv, maxv)
  }

  mid = exp(scale/50) * 500.0
  minv = 50.0 - f[1]
  maxv = -minv
  v = asymp(shift*0.325, minv, maxv)
  cat("minv=",minv," maxv=",maxv," v=",v,"\n")
  f = f + v

  rtn <- data.frame(f, b)
  return(rtn)
}

vtSpect <- function(vt)
{
    s=rep(1,101);
	for (i in 1:length(vt$f)) {
	  s = s * formant(vt$f[i], vt$b[i], 100, 8000)
    }
	s = s * hpc2(8, seq(0, 8000, 8000/100), 500)
    return(s)
}