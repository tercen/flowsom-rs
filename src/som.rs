//! `C_SOM` and `C_mapDataToCodes` from FlowSOM's `src/som.c`, and the R wrapper around them.
//!
//! Layout follows the C: `data` and `codes` are **column-major**, `data[i + j*n]` being
//! observation `i` of variable `j`. Keeping that layout is what makes the loops comparable with
//! the original line by line.
use crate::rng::RRng;

/// Distances the training loop can use. `Euclidean` is FlowSOM's default (`distf = 2`) and the
/// only one CytoNorm uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dist {
    Manhattan,
    Euclidean,
    Chebyshev,
    Cosine,
}

impl Dist {
    /// The C `distf`, including its two bugs: `manh` and `chebyshev` call C's integer `abs` on a
    /// double, so both truncate towards zero before taking the absolute value. Reproduced
    /// deliberately — the point of this crate is to give the same clusters as the R FlowSOM in
    /// use, not a better distance.
    fn of(
        self,
        data: &[f64],
        i: usize,
        n: usize,
        codes: &[f64],
        cd: usize,
        ncodes: usize,
        px: usize,
    ) -> f64 {
        match self {
            Dist::Euclidean => {
                let mut acc = 0.0;
                for j in 0..px {
                    let t = data[i + j * n] - codes[cd + j * ncodes];
                    acc += t * t;
                }
                acc.sqrt()
            }
            Dist::Manhattan => {
                let mut acc = 0.0;
                for j in 0..px {
                    let t = data[i + j * n] - codes[cd + j * ncodes];
                    acc += c_abs(t);
                }
                acc
            }
            Dist::Chebyshev => {
                let mut acc: f64 = 0.0;
                for j in 0..px {
                    let t = c_abs(data[i + j * n] - codes[cd + j * ncodes]);
                    if t > acc {
                        acc = t;
                    }
                }
                acc
            }
            Dist::Cosine => {
                let (mut nom, mut d1, mut d2) = (0.0, 0.0, 0.0);
                for j in 0..px {
                    let (a, b) = (data[i + j * n], codes[cd + j * ncodes]);
                    nom += a * b;
                    d1 += a * a;
                    d2 += b * b;
                }
                -nom / (d1.sqrt() * d2.sqrt()) + 1.0
            }
        }
    }
}

/// C's `abs(double)`: the argument is converted to `int` first. Out-of-range conversion is
/// undefined behaviour in C; saturating is the closest defined thing, and the values this sees
/// are cytometry channel differences, far inside the range.
fn c_abs(x: f64) -> f64 {
    (x as i32).unsigned_abs() as f64
}

/// The map's grid, and the Chebyshev distances between its nodes.
///
/// R builds it as `expand.grid(seq_len(xdim), seq_len(ydim))`, so node index varies fastest
/// along x, and `dist(grid, method = "maximum")` is Chebyshev.
pub fn nhbrdist(xdim: usize, ydim: usize) -> Vec<f64> {
    let ncodes = xdim * ydim;
    let coord = |k: usize| ((k % xdim) as f64, (k / xdim) as f64);
    let mut d = vec![0.0; ncodes * ncodes];
    for a in 0..ncodes {
        let (ax, ay) = coord(a);
        for b in 0..ncodes {
            let (bx, by) = coord(b);
            d[a + b * ncodes] = (ax - bx).abs().max((ay - by).abs());
        }
    }
    d
}

/// The radius FlowSOM defaults to: `quantile(nhbrdist, 0.67)` down to 0.
///
/// R's `quantile` type 7 on the whole distance matrix, diagonal included, which is what
/// `stats::quantile(nhbrdist, 0.67)` sees because `as.matrix(dist(...))` is square.
pub fn default_radius(nhbrdist: &[f64]) -> (f64, f64) {
    let mut v = nhbrdist.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let h = (n - 1) as f64 * 0.67;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    (v[lo] + (h - lo as f64) * (v[hi] - v[lo]), 0.0)
}

/// `C_SOM`: online training, `rlen` passes over the data in random order.
///
/// `codes` is updated in place. `rng` is the caller's, because in R the same stream drew the
/// initial codes a moment earlier and the sequence continues here.
#[allow(clippy::too_many_arguments)]
pub fn train(
    data: &[f64],
    codes: &mut [f64],
    nhbrdist: &[f64],
    alphas: (f64, f64),
    radii: (f64, f64),
    n: usize,
    px: usize,
    ncodes: usize,
    rlen: usize,
    dist: Dist,
    rng: &mut RRng,
) {
    let niter = rlen * n;
    let mut threshold = radii.0;
    let threshold_step = (radii.0 - radii.1) / niter as f64;
    let mut change = 1.0f64;
    let mut xdists = vec![0.0f64; ncodes];

    let mut k = 0usize;
    while k < niter {
        if k % n == 0 {
            // The original writes `k = niter` to break, then still runs the body once more
            // before the loop test. Reproduced: the epoch that decides to stop also trains.
            if change < 1.0 {
                k = niter;
            }
            change = 0.0;
        }

        let i = (n as f64 * rng.unif_rand()) as usize;

        let mut nearest = 0usize;
        for cd in 0..ncodes {
            xdists[cd] = dist.of(data, i, n, codes, cd, ncodes, px);
            if xdists[cd] < xdists[nearest] {
                nearest = cd;
            }
        }

        if threshold < 1.0 {
            threshold = 0.5;
        }
        let alpha = alphas.0 - (alphas.0 - alphas.1) * (k as f64) / (niter as f64);

        for cd in 0..ncodes {
            if nhbrdist[cd + ncodes * nearest] > threshold {
                continue;
            }
            for j in 0..px {
                let tmp = data[i + j * n] - codes[cd + j * ncodes];
                // `change += abs(tmp)` in C truncates to int first, so a step smaller than one
                // unit counts as nothing. That is what the early-stopping test measures.
                change += c_abs(tmp);
                codes[cd + j * ncodes] += tmp * alpha;
            }
        }

        threshold -= threshold_step;
        k += 1;
    }
}

/// One observation's nearest code.
pub struct Mapped {
    /// 1-based node index, as R returns it.
    pub node: usize,
    pub dist: f64,
}

/// `C_mapDataToCodes`: assign every observation to its nearest code.
pub fn map_data_to_codes(
    data: &[f64],
    codes: &[f64],
    nd: usize,
    px: usize,
    ncodes: usize,
    dist: Dist,
) -> Vec<Mapped> {
    (0..nd)
        .map(|i| {
            let mut minid = 0usize;
            let mut mindist = f64::MAX;
            for cd in 0..ncodes {
                let d = dist.of(data, i, nd, codes, cd, ncodes, px);
                if d < mindist {
                    mindist = d;
                    minid = cd;
                }
            }
            Mapped {
                node: minid + 1,
                dist: mindist,
            }
        })
        .collect()
}
