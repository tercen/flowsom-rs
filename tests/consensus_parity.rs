//! `metaClustering_consensus`, against FlowSOM 1.22.0 and ConsensusClusterPlus in the reference
//! image. A hundred resamples, so this is also the test that says R's random stream is being
//! consumed in exactly the same order and amount.
use flowsom::consensus;

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

#[test]
fn the_consensus_matrix_matches_r() {
    let (codes, ncodes, p) = read_matrix("som_codes.csv");
    let (want, n, n2) = read_matrix("meta_consensus_k5.csv");
    assert_eq!((n, n2), (ncodes, ncodes));
    let ml = consensus::consensus_matrices(&codes, ncodes, p, 5, 1, &Default::default());
    for idx in 0..ncodes * ncodes {
        assert_eq!(
            ml[5][idx].to_bits(),
            want[idx].to_bits(),
            "consensus[{}, {}]: got {}, R says {}",
            idx % ncodes + 1,
            idx / ncodes + 1,
            ml[5][idx],
            want[idx]
        );
    }
}

#[test]
fn metaclusters_match_r() {
    let (codes, ncodes, p) = read_matrix("som_codes.csv");
    for k in [3usize, 5, 8] {
        let (want, n, one) = read_matrix(&format!("meta_k{k}.csv"));
        assert_eq!((n, one), (ncodes, 1));
        let want: Vec<usize> = want.iter().map(|v| *v as usize).collect();
        let got = consensus::metacluster_consensus(&codes, ncodes, p, k, 1);
        assert_eq!(
            got, want,
            "metaClustering_consensus(codes, k = {k}, seed = 1)"
        );
    }
}
