# metaClustering_consensus on the codes the SOM fixture produced, at several k.
suppressMessages({library(FlowSOM); library(ConsensusClusterPlus)})
out <- "/out/"
w17 <- function(m, file) {
  df <- as.data.frame(lapply(as.data.frame(m), function(c)
    if (is.numeric(c)) sprintf("%.17g", c) else c))
  write.csv(df, paste0(out, file), row.names = FALSE, quote = FALSE)
}
codes <- as.matrix(read.csv(paste0(out, "som_codes.csv")))
cat("codes:", dim(codes), "\n")
for (k in c(3, 5, 8)) {
  ct <- FlowSOM::metaClustering_consensus(codes, k = k, seed = 1)
  w17(matrix(as.integer(ct), ncol = 1), sprintf("meta_k%d.csv", k))
  cat("k", k, ":", paste(as.integer(ct), collapse = " "), "\n")
}
# the consensus matrix itself at k = 5, so a mismatch can be localised
r <- suppressMessages(ConsensusClusterPlus::ConsensusClusterPlus(
  t(codes), maxK = 5, reps = 100, pItem = 0.9, pFeature = 1,
  title = tempdir(), plot = "pdf", verbose = FALSE,
  clusterAlg = "hc", distance = "euclidean", seed = 1))
w17(r[[5]]$consensusMatrix, "meta_consensus_k5.csv")
cat("ok\n")
