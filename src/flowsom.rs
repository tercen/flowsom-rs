//! The chain `FlowSOM()` runs, in one call: train a map, metacluster its nodes, and give every
//! cell the metacluster of the node it falls on.
//!
//! This is what CytoNorm asks FlowSOM for. `scale = FALSE`, because CytoNorm passes that and
//! scaling before a cofactor-transformed channel would be a second normalisation nobody asked
//! for.
use crate::consensus;
use crate::rng::RRng;
use crate::som::{self, Dist};

/// What `FlowSOM()` returns, as far as anything downstream uses it.
pub struct FlowSom {
    /// The map's codes, column-major, `xdim * ydim` rows.
    pub codes: Vec<f64>,
    pub xdim: usize,
    pub ydim: usize,
    /// One metacluster per node, 1-based.
    pub metaclustering: Vec<usize>,
    /// One node per training cell, 1-based.
    pub node: Vec<usize>,
}

impl FlowSom {
    pub fn ncodes(&self) -> usize {
        self.xdim * self.ydim
    }

    /// The metacluster of every row of `data` — `GetMetaclusters`, for cells the map has not
    /// seen before as well as for the training data.
    pub fn metacluster_of(&self, data: &[f64], n: usize, p: usize) -> Vec<usize> {
        som::map_data_to_codes(data, &self.codes, n, p, self.ncodes(), Dist::Euclidean)
            .iter()
            .map(|m| self.metaclustering[m.node - 1])
            .collect()
    }
}

/// `FlowSOM(input, xdim, ydim, nClus, scale = FALSE, seed)`.
///
/// `data` is column-major, `n × p`. The RNG is seeded once, as `FlowSOM()` does, and the
/// metaclustering reseeds from the same number, as `metaClustering_consensus(seed = seed)` does.
pub fn fit(
    data: &[f64],
    n: usize,
    p: usize,
    xdim: usize,
    ydim: usize,
    n_clus: usize,
    rlen: usize,
    seed: u32,
) -> FlowSom {
    let ncodes = xdim * ydim;
    let mut rng = RRng::set_seed(seed);
    // `codes <- data[sample(1:nrow(data), nCodes, replace = FALSE), ]`
    let drawn = rng.sample_int(n, ncodes);
    let mut codes = vec![0.0; ncodes * p];
    for (c, &row) in drawn.iter().enumerate() {
        for j in 0..p {
            codes[c + j * ncodes] = data[row + j * n];
        }
    }

    let nhbrdist = som::nhbrdist(xdim, ydim);
    let radii = som::default_radius(&nhbrdist);
    som::train(
        data,
        &mut codes,
        &nhbrdist,
        (0.05, 0.01),
        radii,
        n,
        p,
        ncodes,
        rlen,
        Dist::Euclidean,
        &mut rng,
    );

    let node = som::map_data_to_codes(data, &codes, n, p, ncodes, Dist::Euclidean)
        .iter()
        .map(|m| m.node)
        .collect();
    let metaclustering = consensus::metacluster_consensus(&codes, ncodes, p, n_clus, seed);

    FlowSom {
        codes,
        xdim,
        ydim,
        metaclustering,
        node,
    }
}
