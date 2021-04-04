ascale <- function(freq, band) {
  rtn <- data.frame(amp=numeric(), freq=numeric(), band=numeric())
  n <- length(freq)
  for (i in 1:n) {
    a <- max(formant(freq[i], band[i], 8000, 8000))
    rtn <- rbind(rtn, data.frame(amp=a, freq=freq[i], band=band[i]))
  }
  return(rtn)
}

estAmp <- function() {
  rtn <- data.frame(band=numeric(), slope=numeric(), intercept=numeric(), row.names=NULL)
  for (b in seq(40,4000, 10)) {
    tmp <- data.frame(amp=numeric(), freq=numeric())
    for (f in seq(100,8000,10)) {
      a <- max(formant(f, b, 8000, 8000))
      tmp <- rbind(tmp, data.frame(amp=a, freq=f))
    }
    tmp.lm <- lm(amp ~ freq, data=tmp)
    rtn <- rbind(rtn, data.frame(band=b, intercept=tmp.lm$coefficients[1], 
                 slope=tmp.lm$coefficients[2]), row.names=NULL)
  }
  return(rtn)
}
