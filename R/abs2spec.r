abs2spec <- function(abs.df, frame, np=256) {
  nres <- as.numeric(abs.df[frame,13])
  vtp <- as.numeric(abs.df[frame,13:(13+(nres*3))])
  vts <- 20 * log10(0.001 + vtfn2(vtp, np=np))
  vts
}