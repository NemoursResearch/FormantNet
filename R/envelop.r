envelop <- function(x, np=3) {
  for (k in 1:np) {
#
# Fill minima
#
    dm1 = x[1]
    dz = x[2]
    for (j in 2:(length(x)-1)) {
      dp1 = x[j+1]
      if (dm1 > dz & dp1 > dz)
        x[j] <- 0.5 * (dm1 + dp1)
      dm1 = dz
      dz = dp1
    }
#
# Smooth non-maxima
#
    dm1 = x[1]
    dz = x[2]
    for (j in 2:(length(x)-1)) {
      dp1 = x[j+1]
      if (dm1 >= dz || dp1 >= dz)
        x[j] <- 0.25*dm1 + 0.5*dz + 0.25*dp1
      dm1 = dz
      dz = dp1
    }
  }
  x
}

polevne <- function(x, np=3) {
  for (k in 1:np) {
    #
    # Degrade maxima
    #
    dm1 = x[1]
    dz = x[2]
    for (j in 2:(length(x)-1)) {
      dp1 = x[j+1]
      if (dm1 < dz & dp1 < dz)
        x[j] <- 0.5 * (dm1 + dp1)
      dm1 = dz
      dz = dp1
    }
    #
    # Smooth non-maxima
    #
    dm1 = x[1]
    dz = x[2]
    for (j in 2:(length(x)-1)) {
      dp1 = x[j+1]
      if (dm1 >= dz || dp1 >= dz)
        x[j] <- 0.25*dm1 + 0.5*dz + 0.25*dp1
      dm1 = dz
      dz = dp1
    }
  }
  x
}
