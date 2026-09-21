//! `FlowSOM()` end to end, against the reference image: the codes, the node each cell lands on,
//! the metacluster of each node, and the metacluster of each cell.
use flowsom::flowsom;

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

fn ints(path: &str) -> Vec<usize> {
    let (v, _, _) = read_matrix(path);
    v.iter().map(|x| *x as usize).collect()
}

/// The input is `fsom$data`, the matrix FlowSOM actually trained on. It is not the matrix that
/// was handed to `flowFrame()`: flowCore stores expressions as 32-bit floats, so the round trip
/// through a flowFrame moves every value by up to 2.4e-7. Worth knowing before anyone expects
/// this crate and the R pipeline to agree on a cell's last bits — the loss is on the R side.
#[test]
fn the_whole_chain_matches_r() {
    let (data, n, p) = read_matrix("e2e_input.csv");
    let fsom = flowsom::fit(
        &data,
        n,
        p,
        &flowsom::Params {
            xdim: 5,
            ydim: 5,
            clusters: flowsom::Clusters::Fixed(5),
            rlen: 10,
            seed: 1,
        },
    );

    let (codes, ncodes, _) = read_matrix("e2e_codes.csv");
    assert_eq!(ncodes, 25);
    for (k, (g, w)) in fsom.codes.iter().zip(&codes).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "code entry {k}");
    }
    assert_eq!(fsom.node, ints("e2e_node.csv"), "node per cell");
    assert_eq!(
        fsom.metaclustering,
        ints("e2e_metaclustering.csv"),
        "metacluster per node"
    );
    assert_eq!(
        fsom.metacluster_of(&data, n, p),
        ints("e2e_cell_metacluster.csv"),
        "GetMetaclusters"
    );
}
