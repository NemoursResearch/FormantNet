# Copy of the levinson() function from the signals package with
# a correction to remove deprecation warnings.
#
levnsn <- function (x, p = NULL) 
{
  fit <- function(acf, p) {
    ref <- numeric(p)
    g <- -acf[2]/acf[1]
    a <- g
    v <- Re((1 - g * Conj(g)) * acf[1])
    ref[1] <- g
    for (t in 2:p) {
      g <- -(acf[t + 1] + a %*% acf[seq(t, 2, by = -1)])/v
      a <- c((a + as.vector(g) * Conj(a[seq(t - 1, 1, -1)])), g)
      v <- v * (1 - Re(g * Conj(g)))
      ref[t] <- g
    }
    a <- c(1, a)
    return(list(a = a, v = v, ref = ref))
  }
  if ((!is.null(p) && (p != as.integer(p))) || (p < 2)) 
    stop("p must be integer >= 2.")
  if (is.numeric(x) && is.null(dim(x))) {
    lx <- length(x)
    if (is.null(p) || p >= lx) 
      p <- lx - 1
    r <- fit(x, p)
  }
  else {
    if (is.numeric(x) && !(is.null(dim(x)))) {
      lx <- dim(x)
      if (is.null(p) || p >= lx[1]) 
        p <- lx[1] - 1
      zr <- apply(x, 2, function(y) fit(y, p))
      zr <- matrix(unlist(zr), nrow = lx[2], byrow = TRUE)
      a <- zr[, 1:(p + 1), drop = FALSE]
      v <- zr[, p + 2]
      ref <- t(zr[, -(1:(p + 2)), drop = FALSE])
      r <- list(a = a, v = v, ref = ref)
    }
    else {
      stop("x must be a numeric vector or matrix.")
    }
  }
  return(r)
}
