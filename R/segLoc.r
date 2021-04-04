segst <- function(wave, seg) {
  srms <- wave$wav@samp.rate / 1000.0
  i <- which(wave$seg$segnam == seg)
  if (length(i) == 1) {
    st = srms * wave$seg$beg[i] / 1000.0
  } else {
    st = 0
  }
  st
}

segnd <- function(wave, seg) {
  srms <- wave$wav@samp.rate / 1000.0
  i <- which(wave$seg$segnam == seg)
  if (length(i) == 1) {
    nd = srms * wave$seg$end[i] / 1000.0
  } else {
    nd = length(wave$wav@left)
  }
  nd
}

segLoc <- function(wave, seg1, seg2=NULL) {
  ib = segst(wave, seg1)
  ie = ifelse(is.null(seg2), segnd(wave, seg1), segnd(wave, seg2))
  return(c(ib,ie))
}
