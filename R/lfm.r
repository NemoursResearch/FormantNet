lfm <- function(T0=.01, TE=.006, TP=.004, TA=.0002, AV=1, Fs=16000)
{
  id = 3
  Ur = 0.0
  Xe = 0.0
  itot = round(T0 * Fs)
  dug = rep(0,itot)

#  avs = 10.0^((AV - 49.5)/20.0)
  avs = AV
  
  T = 1.0/Fs;
  TE = round(TE * Fs) * T

#
# These adjustments to Tp were not part of the Lin (1990)
# Fortran code, but have been added to silently force Tp
# to have reasonable values.
#
  if (TP >= .99*TE)
    TP = .99*TE
  if (TP / TE < .5)
     TP = .5 * TE
########################
  
  tb = T0 - TE
  if (TA >= tb) 
	  TA = 0.98 * tb
  if (TA == 0.0) {
	  id = 1
  }

  if (id == 3) {
  	Xe0 = 1.0 / TA
	  for (i in 0:199) {
	    xyz = exp(-tb * Xe0)
	    q = xyz - 1.0 + TA * Xe0;
	    dq = -tb * xyz + TA
	    xyz = q / dq
	    Xe = Xe0 - xyz
	    xyz = abs(-xyz / Xe0)
	    if (xyz <= 1.0e-4) 
		    break
	    Xe0 = Xe
	  }

	  Xe = Xe0
	  ex = 1.0 / (Xe * TA)
	  Ur = ex * (exp(-Xe * tb) * (1.0 / Xe + tb) - 1.0 / Xe)
  }

  og = pi / TP

  if (TP / TE > 0.5) { # Lin had Te/Tp
    og2 = og * og
    ogte = og * TE
    sd1 = sin(ogte)
    cd1 = cos(ogte)
    eesd = abs(1.0 / sd1)
    ogcd = og * cd1
  
    tw = (TE - TP) + TA
    al = 200.0
    if (tw/TP >= 0.85)
	    al = 40.0

    for (i in 0:199) {
	    ed1 = exp(al * TE)
      e0 = eesd / ed1
      gg = al * al + og2
      ue = e0 / gg * (ed1 * (al * sd1 - ogcd) + og)
      ut = ue + Ur
      utprime = -TE*ue + (-2.0*al*ue + e0*(ed1 * (TE * (al*sd1 - ogcd) + sd1))) / gg
      xyz = ut / utprime
      aa = al - xyz
      if (abs(xyz/al) <= 1.0e-2)
        break
      al = aa
    }

    if (avs < 0) {
	    e0 = abs(avs) * e0
    } else {
	    e0 = avs / og / (1.0 + exp(al / og*pi)) * gg
    }
  } else {
	  if (avs <= 0.0) {
	    cat("AV must be > 0.0 when TP/TE == 0.5\n")
	    return(dug)
	  }
	  id = 1
	  al = 0.0
	  e0 = avs * og / 2.0
  }

  ite = round(TE * Fs)

  afilter = exp(al / Fs)
  xinput = e0 * afilter * sin(og / Fs)
  bfilter = 2.0 * afilter * cos(og / Fs)
  afilter = afilter * afilter
  y1filter = 0.0
  yout = xinput

  for (i in 1:ite) {
	  dug[i] = yout
	  y2filter = y1filter
	  y1filter = yout
	  yout = bfilter * y1filter - afilter * y2filter
  }

  if (id == 3) {
	  edconst = exp(-tb*Xe)
	  gain = -dug[ite] / (1.0 - edconst)
	  edconst = gain * edconst
	  afilter = exp(-Xe/Fs)
	  y1filter = -gain
	  for (i in (ite+1):itot) {
	    yout = afilter * y1filter
	    y1filter = yout
	    dug[i] = (yout + edconst)
	  }
  } else {
	  for (i in (ite+1):itot)
	    dug[i] = 0.0
  }

  return(dug)
}
