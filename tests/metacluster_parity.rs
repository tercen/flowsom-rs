//! `SSE`, `findElbow` and `scale()`, against R. These are the pieces `maxMeta` is built from.
use flowsom::{consensus, metacluster};

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
fn sse_matches_r() {
    let (codes, n, p) = read_matrix("som_codes.csv");
    let (want, three, _) = read_matrix("sse.csv");
    assert_eq!(three, 3);
    let one = vec![1usize; n];
    let cl3 = consensus::metacluster_consensus(&codes, n, p, 3, 1);
    let cl5 = consensus::metacluster_consensus(&codes, n, p, 5, 1);
    for (got, w) in [
        (metacluster::sse(&codes, n, p, &one), want[0]),
        (metacluster::sse(&codes, n, p, &cl3), want[1]),
        (metacluster::sse(&codes, n, p, &cl5), want[2]),
    ] {
        let rel = (got - w).abs() / w.abs();
        assert!(rel < 1e-14, "SSE: got {got}, R says {w} (relative {rel:e})");
    }
}

/// Realistic sum-of-squares curves: convex and decreasing. A perfectly straight series is not
/// tested, and should not be — every split then fits both halves exactly, so R's answer is
/// decided by whatever floating-point dust `lm`'s QR leaves behind, and it is not 2.
#[test]
fn find_elbow_matches_r() {
    let (want, four, _) = read_matrix("findelbow.csv");
    assert_eq!(four, 4);
    let cases: [&[f64]; 4] = [
        &[100.0, 40.0, 20.0, 15.0, 13.0, 12.0, 11.5, 11.2],
        &[50.0, 49.0, 48.0, 47.0, 20.0, 19.0, 18.0, 17.0],
        &[
            420.98, 210.3, 160.44, 96.7, 47.99, 44.1, 41.8, 40.9, 40.5, 40.3,
        ],
        &[
            1000.0, 820.0, 700.0, 640.0, 610.0, 598.0, 592.0, 589.0, 587.5, 586.9, 586.5, 586.3,
        ],
    ];
    for (k, case) in cases.iter().enumerate() {
        assert_eq!(
            metacluster::find_elbow(case) as f64,
            want[k],
            "findElbow case {k}"
        );
    }
}

#[test]
fn scale_matches_r() {
    let (mut x, n, p) = read_matrix("hc_input.csv");
    let (want, n2, p2) = read_matrix("scaled.csv");
    assert_eq!((n, p), (n2, p2));
    metacluster::scale_columns(&mut x, n, p);
    let worst = x
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-14,
        "worst absolute difference after scale(): {worst:e}"
    );
}
