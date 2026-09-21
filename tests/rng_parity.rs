//! R's stream, value for value. Everything else in this crate is built on it, so if these fail
//! nothing above them means anything.
use flowsom::rng::RRng;

fn csv_col(path: &str) -> Vec<String> {
    let s = std::fs::read_to_string(format!("{}/fixtures/{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    s.lines()
        .skip(1)
        .map(|l| l.trim().trim_matches('"').to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

#[test]
fn unif_rand_matches_r() {
    for (seed, file) in [(1u32, "rng_runif_seed1.csv"), (7, "rng_runif_seed7.csv")] {
        let expect: Vec<f64> = csv_col(file).iter().map(|s| s.parse().unwrap()).collect();
        let mut rng = RRng::set_seed(seed);
        for (k, want) in expect.iter().enumerate() {
            let got = rng.unif_rand();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "set.seed({seed}); runif()[{}]: got {got:.17e}, R says {want:.17e}",
                k + 1
            );
        }
    }
}

#[test]
fn sample_without_replacement_matches_r() {
    // sample() is rejection-based since R 3.6 and consumes a variable number of uniforms, so
    // this checks the draw order, not just the set.
    for (seed, n, k, file) in [
        (1u32, 3000usize, 25usize, "rng_sample_3000_25_seed1.csv"),
        (42, 10, 10, "rng_sample_10_seed42.csv"),
        (7, 100, 30, "rng_sample_100_30_seed7.csv"),
    ] {
        let expect: Vec<usize> = csv_col(file).iter().map(|s| s.parse().unwrap()).collect();
        assert_eq!(expect.len(), k, "{file}");
        let got: Vec<usize> = RRng::set_seed(seed)
            .sample_int(n, k)
            .iter()
            .map(|i| i + 1)
            .collect();
        assert_eq!(got, expect, "set.seed({seed}); sample(1:{n}, {k})");
    }
}
