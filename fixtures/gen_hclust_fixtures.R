# hclust(method="average") and cutree, the two pieces ConsensusClusterPlus is built from.
out <- "/out/"
w17 <- function(m, file) {
  df <- as.data.frame(lapply(as.data.frame(m), function(c)
    if (is.numeric(c)) sprintf("%.17g", c) else c))
  write.csv(df, paste0(out, file), row.names = FALSE, quote = FALSE)
}
set.seed(11)
n <- 40; p <- 5
x <- matrix(rnorm(n * p), nrow = n) + rep(c(0, 4, 8, 12), each = 10)
w17(x, "hc_input.csv")
d <- dist(x, method = "euclidean")
w17(matrix(as.vector(d), ncol = 1), "hc_dist.csv")
hc <- hclust(d, method = "average")
w17(cbind(a = hc$merge[, 1], b = hc$merge[, 2], h = hc$height), "hc_merge.csv")
w17(matrix(hc$order, ncol = 1), "hc_order.csv")
ct <- sapply(2:8, function(k) cutree(hc, k))
colnames(ct) <- paste0("k", 2:8)
w17(ct, "hc_cutree.csv")

# a matrix with exact ties, which is what a consensus matrix is full of
set.seed(3)
y <- round(matrix(rnorm(20 * 3), nrow = 20) * 2) / 2
w17(y, "hc_ties_input.csv")
hy <- hclust(dist(y), method = "average")
w17(cbind(a = hy$merge[, 1], b = hy$merge[, 2], h = hy$height), "hc_ties_merge.csv")
w17(sapply(2:6, function(k) cutree(hy, k)), "hc_ties_cutree.csv")
cat("ok\n")
