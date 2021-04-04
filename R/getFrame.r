getFrame <- function(wave, time, absolute=F, winlen=25, winfn=blkwin) {
  srms <- wave$wav@samp.rate / 1000.0

  if (!absolute) {
    pidx <- which.min(abs(wave$pps$time-time))
    time <- wave$pps$time[pidx]
  } else {
    pidx = 0
  }
  if (is.numeric(winlen)) {
    winms <- winlen
  } else {
    winms = 0
  }
  if (pidx > 0 & winms == 0) {
    if (pidx < length(wave$pps$time) && pidx > 1) {
      winms <- 2 * min(wave$pps$time[pidx] - wave$pps$time[pidx-1],
                       wave$pps$time[pidx+1] - wave$pps$time[pidx])
    } else if (pidx > 1) {
      winms <- 2 * wave$pps$time[pidx+1] - wave$pps$time[pidx]
    } else {
      winms <- 2 * wave$pps$time[pidx] - wave$pps$time[pidx-1]
    }
  }

  halfwin <- round(srms * winms/2, 0)
  nsam <- length(wave$wav@left)
  ib <- round(srms * time, 0) - halfwin
  if (ib < 1) {
    lpad <- rep(0, abs(ib))
    ib <- 1
  } else {
    lpad <- NULL
  }
  ie <- ib + 2 * halfwin - 1
  if ( ie > nsam) {
    rpad <- rep(0, ie - nsam)
    ie <- nsam
  } else {
    rpad <- NULL
  }
  FUN <- match.fun(winfn)
  w <- c(lpad,wave$wav@left[ib:ie],rpad) * FUN(2*halfwin)
  w
}

