#---------------------------------------------------------------------
# nn_norm -- Utility function for normalizing matrices for neural
#            network analyses.
#
# Usage:  normed_mat <- nn_norm(input, [type=c("mm","z")], [alt=normed_mat])
#
# Args:
#  input  -- An input vector or matrix to be normalized by columns.
#  type   -- One of "mm" for min-max normalization or "z" for z-scores
#  alt    -- An already normalized matrix with the same number of columns
#            as the input from which the scaling and offset are used.
#
# Description:
#  By default, nn_norm applies min-max scaling to the input matrix and
#  returns a normalized copy of the input. The offset and scaling factor
#  used for normalization are returned as attributes of the output 
#  matrix, specifically "nn_norm:offset" and "nn_norm:scale". These
#  attributes can then be used to restore the original matrix, or used
#  to apply equivalent normalization to another matrix when passed as
#  the alt argument.
#
#  See nn_unnorm for the inverse of this.
#
# Author: Bunnell -- 1/4/20
#----------------------------------------------------------------------
nn_norm <- function(inp, type="mm", alt=NULL) {
#
# Calculate or assign scale and offset
#
  if (is.null(alt)) {
    if (type == "mm") {
      #cat("scale MM\n")
      if (is.null(dim(inp))) {
        coff <- min(inp)
        cscl <- max(inp) - coff
        if (cscl == 0) {
          cscl = 1
        }
      } else {
        cmax <- apply(inp, 2, max)
        coff <- apply(inp, 2, min)
        cscl <- cmax - coff
        j <- which(cscl == 0 | is.nan(cscl) | is.na(cscl))
        cscl[j] <- 1
      }
    } else {
      #cat("scale Z\n")
      if (is.null(dim(inp))) {
        coff <- mean(inp)
        cscl <- sd(inp)
        if (cscl == 0 || is.na(cscl) || is.nan(cscl))
          cscl = 1
      } else {
        cscl <- apply(inp, 2, sd)
        coff <- apply(inp, 2, mean)
        j <- which(cscl == 0 | is.nan(cscl) | is.na(cscl))
        cscl[j] <- 1
      }
    }
  } else {
    #cat("alt scaling\n")
    coff <- attr(alt, "nn_norm:offset")
    cscl <- attr(alt, "nn_norm:scale")
  }
#
# Apply the scaling
#
  if (is.null(dim(inp))) {  # Must be a vector
    oup <- (inp - coff) / cscl
  } else {
    oup <- t((t(inp) - coff) / cscl)
  }
#
# Save scaling parameters as attributes
#
  attr(oup, 'nn_norm:offset') = coff
  attr(oup, 'nn_norm:scale') = cscl

  oup
}

#---------------------------------------------------------------------
# nn_unnorm -- Utility function to undo normalization that was applied
#              to a matrix or vector.
#
# Usage:  matrix <- nn_unnorm(input, [alt=normed_mat])
#
# Args:
#  input  -- An previously normalized input vector or matrix or similar
#            object that is presumed to be in the normalized space of
#            another object passed as the alt argument.
#  alt    -- An already normalized matrix with the same number of columns
#            as the input from which the scaling and offset are used.
#
# Description:
#  By default, nn_unnorm uses the attributes of the input matrix to
#  recover the offset and scaling parameters and applies the inverse
#  operation to restore the original unnormalized matrix.
#
#  If the alt argument is provided, nn_unnorm instead uses the
#  scaling attributes from the alt matrix to apply to the input matrix.
#
#  See nn_norm for the inverse of this.
#
# Author: Bunnell -- 1/4/20
#----------------------------------------------------------------------
#
nn_unnorm <- function(inp, alt=NULL) {
  if (is.null(alt)) {
    coff <- attr(inp, "nn_norm:offset")
    cscl <- attr(inp, "nn_norm:scale")
  } else {
    coff <- attr(alt, "nn_norm:offset")
    cscl <- attr(alt, "nn_norm:scale")
  }

  if (is.null(dim(inp))) {
    oup <- cscl * inp + coff
  } else {
    oup <- t((t(inp) * cscl) + coff)
  }

  oup
}