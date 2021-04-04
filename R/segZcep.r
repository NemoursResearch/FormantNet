segZcep <- function(wav, seg, nc=12, step=10) {
  j <- which(wav$seg$segnam == seg)
  t1 <- wav$seg$beg[j] / 1000.0
  t2 <- wav$seg$end[j] / 1000.0
  srms <- wav$wav@samp.rate / 1000.0
  wl <- round(srms * 32)
  s <- rep(0, nc)
  sr <- as.double(wav$wav@samp.rate)
  nfr <- 0
  for (tx in seq(t1, t2, step)) {
    s <- s + zcep(sr, getFrame(wav, tx, absolute=F, winlen=32), nc)
    nfr <- nfr + 1
  }
  s <- s/nfr
  s
}