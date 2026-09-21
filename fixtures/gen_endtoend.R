# The whole chain the CytoNorm operator runs: FlowSOM() with a seed, then every cell's
# metacluster. Synthetic data, so it can be committed.
#
# The input dumped here is `fsom$data`, not the matrix handed to flowFrame(): flowCore stores
# expressions as 32-bit floats, so a flowFrame round trip moves every value by up to 2.4e-7.
# That is the R pipeline losing precision, and comparing against it needs the data FlowSOM
# actually saw.
suppressMessages(library(FlowSOM))
out <- "/out/"
w17 <- function(m, file) {
  df <- as.data.frame(lapply(as.data.frame(m), function(c)
    if (is.numeric(c)) sprintf("%.17g", c) else c))
  write.csv(df, paste0(out, file), row.names = FALSE, quote = FALSE)
}
data <- as.matrix(read.csv(paste0(out, "som_input.csv")))
ff <- flowCore::flowFrame(data)
fsom <- FlowSOM::FlowSOM(ff, colsToUse = colnames(data), xdim = 5, ydim = 5,
                         nClus = 5, scale = FALSE, seed = 1, silent = TRUE)
w17(fsom$FlowSOM$data, "e2e_input.csv")
w17(fsom$FlowSOM$map$codes, "e2e_codes.csv")
w17(matrix(as.integer(fsom$metaclustering), ncol = 1), "e2e_metaclustering.csv")
w17(matrix(as.integer(fsom$FlowSOM$map$mapping[, 1]), ncol = 1), "e2e_node.csv")
w17(matrix(as.integer(FlowSOM::GetMetaclusters(fsom)), ncol = 1), "e2e_cell_metacluster.csv")
cat("float32 shift:", max(abs(data - fsom$FlowSOM$data)), "\n")
cat("metaclusters:", paste(as.integer(fsom$metaclustering), collapse = " "), "\n")
