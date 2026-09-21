//! `ConsensusClusterPlus`, as far as FlowSOM's `metaClustering_consensus` uses it.
//!
//! The metaclusters a FlowSOM returns are not a clustering of the codes directly. They are a
//! clustering of a *consensus matrix*: resample 90% of the nodes a hundred times, cluster each
//! subsample with average-linkage hclust, count how often each pair lands together, and cluster
//! the resulting co-occurrence fractions. Everything about the answer therefore depends on R's
//! random stream and on hclust's tie-breaking, which is why both are ported exactly
//! ([`crate::rng`], [`crate::hclust`]).
//!
//! Only the path FlowSOM takes is here: `clusterAlg = "hc"`, `distance = "euclidean"`,
//! `reps = 100`, `pItem = 0.9`, `pFeature = 1`, average linkage inside and out.
//! `ml` is indexed by `k` throughout, with slots 0 and 1 unused, so that the code reads the way
//! `ml[[k]]` reads in the original.
#![allow(clippy::needless_range_loop)]
use crate::hclust::{self, Linkage};
use crate::rng::RRng;

/// FlowSOM's call: `reps = 100`, `pItem = 0.9`.
pub struct Options {
    pub reps: usize,
    pub p_item: f64,
    pub inner_linkage: Linkage,
    pub final_linkage: Linkage,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            reps: 100,
            p_item: 0.9,
            inner_linkage: Linkage::Average,
            final_linkage: Linkage::Average,
        }
    }
}

/// The consensus fractions at each `k` from 2 to `max_k`, each an `n × n` symmetric matrix in
/// column-major order. Index 0 and 1 are empty: `ml[[1]]` is unused in the original too.
pub fn consensus_matrices(
    items: &[f64],
    n: usize,
    p: usize,
    max_k: usize,
    seed: u32,
    opts: &Options,
) -> Vec<Vec<f64>> {
    assert!(max_k >= 2, "maxK must be at least 2");
    // `dist(t(d))` over the items — the SOM nodes — once, and every resample takes a submatrix
    // of it rather than recomputing. That is what the original does, and it matters: a
    // recomputed distance could differ in the last bits.
    let full = hclust::dist_euclidean(items, n, p);
    let at = |i: usize, j: usize| -> f64 {
        if i == j {
            0.0
        } else {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            full[b + a * n - (a + 1) * (a + 2) / 2]
        }
    };

    let mut rng = RRng::set_seed(seed);
    let sample_n = (n as f64 * opts.p_item).floor() as usize;
    let mut ml: Vec<Vec<f64>> = (0..=max_k).map(|_| vec![0.0; n * n]).collect();
    let mut m_count = vec![0.0; n * n];

    for _ in 0..opts.reps {
        // `sort(sample(space, sampleN, replace = FALSE))`: the subsample is used in ascending
        // order, so the hclust below sees the items in their original order.
        let mut cols = rng.sample_int(n, sample_n);
        cols.sort_unstable();

        let m = cols.len();
        let mut sub = vec![0.0; m * (m - 1) / 2];
        for a in 0..m {
            for b in a + 1..m {
                sub[b + a * m - (a + 1) * (a + 2) / 2] = at(cols[a], cols[b]);
            }
        }
        let hc = hclust::hclust(sub, m, opts.inner_linkage);

        // Every pair drawn together, counted once per rep, diagonal included.
        for &a in &cols {
            for &b in &cols {
                m_count[a + b * n] += 1.0;
            }
        }

        for k in 2..=max_k {
            let assignment = hclust::cutree(&hc.merge, k);
            for a in 0..m {
                for b in 0..m {
                    if assignment[a] == assignment[b] {
                        ml[k][cols[a] + cols[b] * n] += 1.0;
                    }
                }
            }
        }
    }

    // `triangle(ml, 3) / triangle(mCount, 3)`, which for the symmetric matrices these are is an
    // elementwise ratio, with zero wherever a pair was never drawn together.
    for k in 2..=max_k {
        for idx in 0..n * n {
            ml[k][idx] = if m_count[idx] == 0.0 {
                0.0
            } else {
                ml[k][idx] / m_count[idx]
            };
        }
    }
    ml
}

/// `metaClustering_consensus(codes, k, seed)`: the metacluster of every SOM node, 1-based.
///
/// `codes` is the map's codes, column-major, `ncodes × p`.
pub fn metacluster_consensus(
    codes: &[f64],
    ncodes: usize,
    p: usize,
    k: usize,
    seed: u32,
) -> Vec<usize> {
    let opts = Options::default();
    let ml = consensus_matrices(codes, ncodes, p, k, seed, &opts);
    let fm = &ml[k];
    // `hclust(as.dist(1 - fm), method = "average")` — the lower triangle of one minus the
    // consensus fractions.
    let mut d = vec![0.0; ncodes * (ncodes - 1) / 2];
    for a in 0..ncodes {
        for b in a + 1..ncodes {
            d[b + a * ncodes - (a + 1) * (a + 2) / 2] = 1.0 - fm[b + a * ncodes];
        }
    }
    let hc = hclust::hclust(d, ncodes, opts.final_linkage);
    hclust::cutree(&hc.merge, k)
}
