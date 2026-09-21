suppressMessages(library(FlowSOM))
out <- "/out/"
w17 <- function(m, file) {
  df <- as.data.frame(lapply(as.data.frame(m), function(c) sprintf("%.17g", c)))
  write.csv(df, paste0(out, file), row.names = FALSE, quote = FALSE)
}
# Realistic within-cluster sum-of-squares curves: convex, decreasing, no exact ties.
cases <- list(
  c(100, 40, 20, 15, 13, 12, 11.5, 11.2),
  c(50, 49, 48, 47, 20, 19, 18, 17),
  c(420.98, 210.3, 160.44, 96.7, 47.99, 44.1, 41.8, 40.9, 40.5, 40.3),
  c(1000, 820, 700, 640, 610, 598, 592, 589, 587.5, 586.9, 586.5, 586.3)
)
w17(matrix(sapply(cases, FlowSOM:::findElbow), ncol = 1), "findelbow.csv")
for (i in seq_along(cases)) cat(i, ":", FlowSOM:::findElbow(cases[[i]]), "\n")
