# flowsom-rs — status, 2026-09-21

Built to give CytoNorm its clustering. Published at `github.com/tercen/flowsom-rs`.

## Where it got to

The whole FlowSOM chain, bit-identical to FlowSOM 1.22.0 on R 4.0.4 — see `README.md` for the
table. That includes R's random stream, `hclust` with its tie-breaking, `cutree`,
ConsensusClusterPlus's hundred resamples, and `FlowSOM()` end to end.

`cargo test` is 11 tests, all parity against fixtures generated in `lucas501/cytonorm_docker:1.1.9`.
Everything committed is synthetic and seeded; no study data is here or will be.

## The licence, settled 2026-09-21

**GPL-2.0-only**, because ConsensusClusterPlus is "GPL version 2" with no "or later" and the
metaclustering here is a port of it. FlowSOM and R's `hclust` are GPL-2-*or-later*, so the map
alone could have been GPL-3; the consensus module is what pins the crate.

Faris chose to move `cytonorm_rust_operator` from AGPL-3 to **GPL-2-or-later** rather than split
this crate, so the operator keeps exact parity with the R pipeline and the distributed
combination is GPL-2-only — which is what the upstream asks for. The reasoning is in that
repository's `LICENSING.md`.

Note for later: `asinh_rust_operator` is still AGPL-3, so **it cannot link this crate** as it
stands. It has no reason to today.

## What is not here

- **`BuildMST`.** FlowSOM builds a minimum spanning tree for plotting and for `mst > 1` training.
  CytoNorm uses neither.
- **`AggregateFlowFrames`, `ReadInput`, scaling, `NewData`.** The operator's job, not the crate's.
- **Anything about speed.** The maps CytoNorm builds are 5x5 to 15x15 over a subsample, so the
  hundred-resample metaclustering is the expensive part and it takes 50 ms here.

## Next

1. Wire it into `cytonorm_rust_operator` so `cluster > 1` works: train on the reference batch's
   cells, then give every cell its metacluster, and fit a spline per (batch, metacluster,
   channel) — the code for which is already there and only ever sees one cluster today.
