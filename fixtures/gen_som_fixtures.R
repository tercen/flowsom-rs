#!/usr/bin/env Rscript
# Reference fixtures for the Rust port of FlowSOM's SOM, dumped from the image Tercen's R
# CytoNorm operator runs (FlowSOM 1.22.0, R 4.0.4). Synthetic data only.
#
# Everything is written with %.17g. write.csv's default of 15 significant digits is not enough:
# the map's nearest-code test can be decided by the last bits, so a rounded fixture makes an
# exact port look wrong on a handful of nodes.
suppressMessages(library(FlowSOM))
out <- "/out/"
w17 <- function(m, file) {
  df <- as.data.frame(lapply(as.data.frame(m), function(c) sprintf("%.17g", c)))
  write.csv(df, paste0(out, file), row.names = FALSE, quote = FALSE)
}

set.seed(42)
n <- 3000; px <- 4
centres <- matrix(c(0,0,0,0,  5,5,0,0,  0,0,5,5,  5,0,5,0), nrow = 4, byrow = TRUE)
lab  <- sample(1:4, n, replace = TRUE)
data <- centres[lab, ] + matrix(rnorm(n * px, sd = 0.6), nrow = n)
colnames(data) <- paste0("m", seq_len(px))
w17(data, "som_input.csv")

xdim <- 5; ydim <- 5
set.seed(1)
init <- data[sample(1:nrow(data), xdim * ydim, replace = FALSE), , drop = FALSE]
w17(init, "som_init_codes.csv")

for (rlen in c(1, 2, 3, 10)) {
  set.seed(1)
  invisible(sample(1:nrow(data), xdim * ydim, replace = FALSE))
  som <- FlowSOM::SOM(data, xdim = xdim, ydim = ydim, rlen = rlen, codes = init, silent = TRUE)
  w17(som$codes, sprintf("som_codes_rlen%d.csv", rlen))
  if (rlen == 10) {
    w17(cbind(node = som$mapping[, 1], dist = som$mapping[, 2]), "som_mapping.csv")
    w17(som$codes, "som_codes.csv")
  }
  cat("rlen", rlen, "code[1,1]:", sprintf("%.17g", som$codes[1, 1]), "\n")
}

grid <- expand.grid(seq_len(xdim), seq_len(ydim))
w17(as.matrix(stats::dist(grid, method = "maximum")), "som_nhbrdist.csv")
