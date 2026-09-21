//! `hclust(method = "average")` and `cutree`, against R 4.0.4 in the reference image.
//!
//! The second case is built from values rounded to halves, so the distance matrix is full of
//! exact ties — which is the shape a consensus matrix has, and the only way to check that ties
//! break the way the Fortran breaks them.
use flowsom::hclust::{self, Linkage};

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

fn check(input: &str, merge_file: &str, cutree_file: &str, ks: std::ops::RangeInclusive<usize>) {
    let (x, n, p) = read_matrix(input);
    let d = hclust::dist_euclidean(&x, n, p);
    let hc = hclust::hclust(d, n, Linkage::Average);

    let (want, rows, three) = read_matrix(merge_file);
    assert_eq!((rows, three), (n - 1, 3));
    for i in 0..n - 1 {
        assert_eq!(
            (hc.merge[i].0 as f64, hc.merge[i].1 as f64),
            (want[i], want[i + (n - 1)]),
            "merge row {}",
            i + 1
        );
        let h = want[i + 2 * (n - 1)];
        assert_eq!(hc.height[i].to_bits(), h.to_bits(), "height {}", i + 1);
    }

    let (ct, rows, cols) = read_matrix(cutree_file);
    assert_eq!(rows, n);
    for (c, k) in ks.enumerate() {
        assert!(c < cols);
        let got = hclust::cutree(&hc.merge, k);
        let want_k: Vec<usize> = (0..n).map(|i| ct[i + c * n] as usize).collect();
        assert_eq!(got, want_k, "cutree(k = {k}) on {input}");
    }
}

#[test]
fn average_linkage_matches_r() {
    check("hc_input.csv", "hc_merge.csv", "hc_cutree.csv", 2..=8);
}

#[test]
fn ties_break_the_way_the_fortran_breaks_them() {
    check(
        "hc_ties_input.csv",
        "hc_ties_merge.csv",
        "hc_ties_cutree.csv",
        2..=6,
    );
}

#[test]
fn the_dendrogram_order_matches_r() {
    let (x, n, p) = read_matrix("hc_input.csv");
    let hc = hclust::hclust(hclust::dist_euclidean(&x, n, p), n, Linkage::Average);
    let (want, rows, one) = read_matrix("hc_order.csv");
    assert_eq!((rows, one), (n, 1));
    let want: Vec<usize> = want.iter().map(|v| *v as usize).collect();
    assert_eq!(hc.order, want);
}
