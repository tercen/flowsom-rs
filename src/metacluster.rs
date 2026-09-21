//! `MetaClustering`: let FlowSOM choose the number of metaclusters, by the elbow of the
//! within-cluster sum of squares.
//!
//! One deliberate difference from R. `FlowSOM:::consensus` calls ConsensusClusterPlus **without
//! a seed**, and ConsensusClusterPlus then seeds itself from the clock, so R's `maxMeta` path
//! gives a different answer on every run. This one takes the seed, because an operator that
//! cannot be re-run to the same answer is a bug. Everything else — the smoothing, the sum of
//! squares, the elbow — is the original.
use crate::consensus::{self, Options};
use crate::hclust::Linkage;

/// `SSE(data, clustering)`: summed over clusters, `(n_j - 1)` times the sum of the per-column
/// variances of cluster `j`.
pub fn sse(data: &[f64], n: usize, p: usize, clustering: &[usize]) -> f64 {
    let max = clustering.iter().copied().max().unwrap_or(0);
    let mut total = 0.0;
    for j in 1..=max {
        let rows: Vec<usize> = (0..n).filter(|&i| clustering[i] == j).collect();
        if rows.len() <= 1 {
            continue;
        }
        let m = rows.len() as f64;
        let mut var_sum = 0.0;
        for col in 0..p {
            let mean = rows.iter().map(|&i| data[i + col * n]).sum::<f64>() / m;
            let v = rows
                .iter()
                .map(|&i| (data[i + col * n] - mean).powi(2))
                .sum::<f64>()
                / (m - 1.0);
            var_sum += v;
        }
        total += (m - 1.0) * var_sum;
    }
    total
}

/// `findElbow`: the split point that minimises the total absolute residual of two straight
/// lines fitted either side of it.
pub fn find_elbow(y: &[f64]) -> usize {
    let n = y.len();
    let mut min_r = f64::INFINITY;
    let mut optimal = 1usize;
    // R's loop is `for (i in 2:(n-1))`, and `data[1:(i-1), ]` is a single point when i = 2,
    // whose residual is zero.
    for i in 2..n {
        let r = abs_residuals(&y[..i - 1], 1.0) + abs_residuals(&y[i - 1..], i as f64);
        if r < min_r {
            min_r = r;
            optimal = i;
        }
    }
    optimal
}

/// Sum of `|residual|` from a least-squares line through `(x0 + k, y[k])`.
fn abs_residuals(y: &[f64], x0: f64) -> f64 {
    let n = y.len() as f64;
    if y.len() < 2 {
        return 0.0;
    }
    let xs: Vec<f64> = (0..y.len()).map(|k| x0 + k as f64).collect();
    let xbar = xs.iter().sum::<f64>() / n;
    let ybar = y.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|x| (x - xbar).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(y).map(|(x, v)| (x - xbar) * (v - ybar)).sum();
    let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
    let intercept = ybar - slope * xbar;
    xs.iter()
        .zip(y)
        .map(|(x, v)| (v - (intercept + slope * x)).abs())
        .sum()
}

/// `DetermineNumberOfClusters(data, max, "metaClustering_consensus")`.
///
/// Runs the consensus once up to `max_k` and scores every k from its own clustering, which is
/// what the original does — the hundred resamples are not repeated per k.
pub fn determine_number_of_clusters(
    data: &[f64],
    n: usize,
    p: usize,
    max_k: usize,
    seed: u32,
) -> usize {
    let opts = Options::default();
    let ml = consensus::consensus_matrices(data, n, p, max_k, seed, &opts);
    let mut res = vec![0.0; max_k + 1];
    // `res[1] <- SSE(data, rep(1, nrow(data)))` — everything in one cluster.
    res[1] = sse(data, n, p, &vec![1usize; n]);
    for k in 2..=max_k {
        let fm = &ml[k];
        let mut d = vec![0.0; n * (n - 1) / 2];
        for a in 0..n {
            for b in a + 1..n {
                d[b + a * n - (a + 1) * (a + 2) / 2] = 1.0 - fm[b + a * n];
            }
        }
        let hc = crate::hclust::hclust(d, n, Linkage::Average);
        res[k] = sse(data, n, p, &crate::hclust::cutree(&hc.merge, k));
    }
    // `smooth = 0.2`, applied in place, so later entries see already-smoothed neighbours.
    let smooth = 0.2;
    for i in 2..max_k {
        res[i] =
            (1.0 - smooth) * res[i] + (smooth / 2.0) * res[i - 1] + (smooth / 2.0) * res[i + 1];
    }
    find_elbow(&res[1..=max_k])
}

/// `MetaClustering(codes, "metaClustering_consensus", max)`: choose k, then cluster at it.
pub fn metaclustering(data: &[f64], n: usize, p: usize, max_k: usize, seed: u32) -> Vec<usize> {
    let k = determine_number_of_clusters(data, n, p, max_k, seed);
    consensus::metacluster_consensus(data, n, p, k, seed)
}

/// Neumaier's compensated sum: the correctly rounded total, whatever the order.
///
/// R accumulates `colMeans` and `sum` in **long double** — 64 bits of mantissa on x86 — and a
/// plain `f64` loop is off by a few ulp against it. That is not a rounding curiosity here: a few
/// ulp in the scaled data changes which node a cell is nearest, and SOM training is chaotic, so
/// a quarter of the cells end up on a different node (their metacluster, measured on the
/// reference data, does not move). Compensated summation is correctly rounded, which is what
/// R's extended precision is buying, so the two agree.
fn neumaier_sum(xs: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// `scale(x, center = TRUE, scale = TRUE)` per column — what `FlowSOM(scale = TRUE)` applies
/// before the map sees the data.
///
/// R's own steps, in R's own order: centre by `colMeans`, then divide by
/// `sqrt(sum(v^2) / (n - 1))` of the centred column. Both sums are compensated, for the reason
/// in [`neumaier_sum`].
pub fn scale_columns(data: &mut [f64], n: usize, p: usize) {
    for col in 0..p {
        let s = &mut data[col * n..(col + 1) * n];
        let mean = neumaier_sum(s.iter().copied()) / n as f64;
        for v in s.iter_mut() {
            *v -= mean;
        }
        let sd = (neumaier_sum(s.iter().map(|v| v * v)) / (n as f64 - 1.0)).sqrt();
        if sd > 0.0 {
            for v in s.iter_mut() {
                *v /= sd;
            }
        }
    }
}
