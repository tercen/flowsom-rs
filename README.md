# flowsom-rs

FlowSOM's self-organising map and its consensus metaclustering, in Rust, giving **the same
clusters** as the R implementation Tercen runs today — not similar ones.

Every number below is bit-identical to FlowSOM 1.22.0 on R 4.0.4, checked against fixtures
generated inside `lucas501/cytonorm_docker:1.1.9`, the image the R CytoNorm operator uses:

| | agreement with R |
|---|---|
| `unif_rand`, `sample()` | bit-identical, including the rejection sampler |
| SOM codes, at rlen 1, 2, 3 and 10 | bit-identical |
| `MapDataToCodes` nodes and distances | bit-identical |
| `hclust(method = "average")` merge, heights, order | bit-identical, ties included |
| `cutree` at k = 2..8 | identical |
| ConsensusClusterPlus's consensus matrix (100 reps) | bit-identical |
| `metaClustering_consensus` at k = 3, 5, 8 | identical |
| `FlowSOM()` end to end, and `GetMetaclusters` | identical |

```rust
use flowsom::flowsom;

// data column-major, n rows by p channels
let fsom = flowsom::fit(&data, n, p, 5, 5, /* nClus */ 5, /* rlen */ 10, /* seed */ 1);
let per_cell = fsom.metacluster_of(&data, n, p);
```

## Why it is a transliteration

A clustering is a discrete answer: a tie broken the other way is not a small error, it is a
different cluster for every cell in that node. So the parts that decide ties are copied rather
than rewritten —

- **R's random stream.** `set.seed` scrambles the seed through 50 rounds of an LCG before
  Mersenne-Twister sees it, and since R 3.6 `sample()` draws 16-bit blocks and rejects
  out-of-range ones, consuming a variable number of uniforms. The map draws its initial codes
  with `sample()` and then one observation per training step, and ConsensusClusterPlus resamples
  a hundred times, so nothing downstream survives a different stream.
- **`hclust`'s scan order.** R's Fortran takes the first minimum it meets, scanning ascending.
  The consensus matrix is full of exact ties, so this is not a corner case there, it is the
  normal case.
- **Two bugs in FlowSOM's C.** `manh` and `chebyshev` call C's integer `abs` on a double, and so
  does the accumulator that decides when training stops early. Reproducing them is the point.

## Two things to know before comparing against an R pipeline

- **Fixtures need 17 significant digits.** `write.csv` writes 15, and at 15 the nearest-code test
  flips on a couple of nodes — an exact port looks wrong. Every fixture here is `%.17g`.
- **flowCore stores expressions as 32-bit floats.** A `flowFrame` round trip moves every value by
  up to 2.4e-7, so the R pipeline is not working in double precision. When comparing, dump the
  matrix FlowSOM actually saw (`fsom$data`), not the one handed to `flowFrame()`. The precision
  is lost on the R side.

## Licence

**GPL-2.0-only**, and deliberately so: the metaclustering is a port of ConsensusClusterPlus,
which is "GPL version 2" without the "or later". FlowSOM itself is GPL (>= 2) and R's `hclust` is
GPL-2+, so the map alone could be GPL-3; the consensus module is what pins this.

That matters downstream: an AGPL-3 operator cannot link this crate at all — there is no
combination of GPL-2-only and AGPL-3 that may be distributed. `cytonorm_rust_operator` moved to
GPL-2-or-later for exactly this reason; its `LICENSING.md` has the chain.
