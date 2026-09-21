//! The map itself, against FlowSOM 1.22.0 running in the image Tercen's R CytoNorm operator
//! uses. `fixtures/gen_som_fixtures.R` produced these; the data is synthetic and seeded.
use flowsom::rng::RRng;
use flowsom::som::{self, Dist};

const XDIM: usize = 5;
const YDIM: usize = 5;
const RLEN: usize = 10;

/// A CSV of numbers, returned column-major — the layout the C works in.
fn read_matrix(path: &str) -> (Vec<f64>, usize, usize) {
    let s = std::fs::read_to_string(format!("{}/fixtures/{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let rows: Vec<Vec<f64>> = s
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|v| v.trim().trim_matches('"').parse().unwrap())
                .collect()
        })
        .collect();
    let (n, px) = (rows.len(), rows[0].len());
    let mut col = vec![0.0; n * px];
    for (i, r) in rows.iter().enumerate() {
        for (j, v) in r.iter().enumerate() {
            col[i + j * n] = *v;
        }
    }
    (col, n, px)
}

/// Bit equality, not a tolerance. The arithmetic is the same operations in the same order, so
/// anything else means a real difference — and the fixtures carry 17 significant digits so that
/// this is a fair test. At 15, `write.csv`'s default, the rounding alone moves a handful of
/// nodes: the nearest-code test can be decided by the last bits.
fn assert_identical(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}[{k}]: got {g:.17e}, R says {w:.17e}"
        );
    }
}

#[test]
fn the_grid_distances_match_r() {
    let (want, n, px) = read_matrix("som_nhbrdist.csv");
    assert_eq!((n, px), (XDIM * YDIM, XDIM * YDIM));
    let got = som::nhbrdist(XDIM, YDIM);
    assert_eq!(got, want, "dist(expand.grid(...), method = 'maximum')");
}

#[test]
fn training_reproduces_the_r_codes() {
    let (data, n, px) = read_matrix("som_input.csv");
    let (init, ncodes, px2) = read_matrix("som_init_codes.csv");
    assert_eq!((ncodes, px, px2), (25, 4, 4));

    // The fixture drew the initial codes from set.seed(1), then trained with the stream left
    // where that draw finished. Same here: draw and discard, then train.
    let mut rng = RRng::set_seed(1);
    let drawn = rng.sample_int(n, ncodes);
    for (c, &row) in drawn.iter().enumerate() {
        for j in 0..px {
            assert_eq!(
                init[c + j * ncodes],
                data[row + j * n],
                "initial code {c} should be data row {row}"
            );
        }
    }

    // Several lengths, because a single one cannot tell a correct loop from one that diverges
    // and happens to be close: the early-stopping test only bites after a few epochs.
    let nhbrdist = som::nhbrdist(XDIM, YDIM);
    let radii = som::default_radius(&nhbrdist);
    assert_eq!(radii, (3.0, 0.0), "quantile(nhbrdist, 0.67) * c(1, 0)");
    for rlen in [1usize, 2, 3, RLEN] {
        let (want, _, _) = read_matrix(&format!("som_codes_rlen{rlen}.csv"));
        let mut rng = RRng::set_seed(1);
        let _ = rng.sample_int(n, ncodes);
        let mut codes = init.clone();
        som::train(
            &data,
            &mut codes,
            &nhbrdist,
            (0.05, 0.01),
            radii,
            n,
            px,
            ncodes,
            rlen,
            Dist::Euclidean,
            &mut rng,
        );
        assert_identical(&codes, &want, &format!("codes at rlen {rlen}"));
    }
}

#[test]
fn the_mapping_matches_r() {
    let (data, n, px) = read_matrix("som_input.csv");
    let (codes, ncodes, _) = read_matrix("som_codes.csv");
    let (want, n2, two) = read_matrix("som_mapping.csv");
    assert_eq!((n2, two), (n, 2));

    let got = som::map_data_to_codes(&data, &codes, n, px, ncodes, Dist::Euclidean);
    let nodes: Vec<f64> = got.iter().map(|m| m.node as f64).collect();
    let dists: Vec<f64> = got.iter().map(|m| m.dist).collect();
    assert_eq!(
        nodes,
        want[0..n],
        "every cell should land on the node R gives it"
    );
    assert_identical(&dists, &want[n..2 * n], "mapping distances");
}
