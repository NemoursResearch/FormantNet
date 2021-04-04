#--------------------------------------------------------------
# getWaveData -- Load data for SRL .wav file with .pps & .seg
#               information (if available).
#
# Usage: waveData <- getWaveData(basename)
#
#  Args:
#   basename -- The basename (with path but sans extension) to 
#               the .wav file. This is also assumed to be the
#               basename of the .pps and .seg file in the same
#               location.
#
# Returns:
#   A list of three elements:
#     wav -- A class "Wave" object from package "tuneR".
#     pps -- A data.frame with columns time, f0, and vcd that
#            are the time in msec of the epoch marker, the f0
#            if a voiced epoch or 0 if unvoiced, and the voicing
#            status (1=voiced; 0=unvoiced) respectively.
#     seg -- A data.frame with columns segnam, beg, end listing
#            the segment name, beginning time, and ending time
#            of the segment in microseconds.
# Note: If .wav file is not found, the function returns NULL
#       rather than a list. If either or both of .pps or .seg
#       files is not found, that list element is set to NULL.
#
# 1/11/20 -- htb
#-------------------------------------------------------------
#
getWaveData <- function(basename) {
  require(tuneR)
  
  wname <- paste0(basename,".wav")
  pname <- paste0(basename,".pps")
  sname <- paste0(basename,".seg")
  
  if (!file.exists(wname))
    return(NULL)
  w <- readWave(wname)
  if(file.exists(pname)) {
    p <- read.table(pname)
    names(p) <- c("time","f0","vcd")
  } else {
    p <- NULL
  }
  if(file.exists(sname)) {
    s <- read.table(sname)
    names(s) <- c("segnam","beg","end")
  } else {
    s <- NULL
  }
  rtn <- list(wav=w, pps=p, seg=s)
  rtn
}
