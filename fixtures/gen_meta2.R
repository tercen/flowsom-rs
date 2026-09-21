# SSE, findElbow, DetermineNumberOfClusters and scale(), for the maxMeta path.
suppressMessages({library(FlowSOM); library(ConsensusClusterPlus)})
out <- "/out/"
w17 <- function(m, file) {
  df <- as.data.frame(lapply(as.data.frame(m), function(c)
    if (is.numeric(c)) sprintf("%.17g", c) else c))
  write.csv(df, paste0(out, file), row.names = FALSE, quote = FALSE)
}
codes <- as.matrix(read.csv(paste0(out, "som_codes.csv")))

# SSE at a few clusterings
set.seed(1)
cl3 <- FlowSOM::metaClustering_consensus(codes, k = 3, seed = 1)
cl5 <- FlowSOM::metaClustering_consensus(codes, k = 5, seed = 1)
w17(matrix(c(FlowSOM:::SSE(codes, rep(1, nrow(codes))),
             FlowSOM:::SSE(codes, as.numeric(cl3)),
             FlowSOM:::SSE(codes, as.numeric(cl5))), ncol = 1), "sse.csv")

# findElbow on a couple of shapes
w17(matrix(c(FlowSOM:::findElbow(c(100, 40, 20, 15, 13, 12, 11.5, 11.2)),
             FlowSOM:::findElbow(c(50, 49, 48, 47, 20, 19, 18, 17)),
             FlowSOM:::findElbow(c(10, 9, 8, 7, 6, 5, 4, 3))), ncol = 1), "findelbow.csv")

# scale()
x <- as.matrix(read.csv(paste0(out, "hc_input.csv")))
w17(scale(x, center = TRUE, scale = TRUE), "scaled.csv")
cat("ok\n")
