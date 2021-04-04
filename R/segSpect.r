segSpect <- function(wav, seg, step=10) {
  j <- which(wav$seg$segnam == seg)
  t1 <- wav$seg$beg[j] / 1000.0
  t2 <- wav$seg$end[j] / 1000.0
  srms <- wav$wav@samp.rate / 1000.0
  wl <- round(srms * 32)
  s <- rep(0, 1 + wl/2)
  nfr <- 0
  for (tx in seq(t1, t2, step)) {
    s <- s + mstft(getFrame(wav, tx, absolute=T, winlen=32))
    nfr <- nfr + 1
  }
  s <- s/nfr
  s
}