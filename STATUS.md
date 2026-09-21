# flowsom-rs — status, 2026-09-21

Built to give CytoNorm its clustering. **Local git only: no remote, nothing published**, because
of the licence question below, which is not mine to decide.

## Where it got to

The whole FlowSOM chain, bit-identical to FlowSOM 1.22.0 on R 4.0.4 — see `README.md` for the
table. That includes R's random stream, `hclust` with its tie-breaking, `cutree`,
ConsensusClusterPlus's hundred resamples, and `FlowSOM()` end to end.

`cargo test` is 11 tests, all parity against fixtures generated in `lucas501/cytonorm_docker:1.1.9`.
Everything committed is synthetic and seeded; no study data is here or will be.

## The licence question, for Faris

| package | licence |
|---|---|
| FlowSOM | GPL (>= 2) |
| R's `hclust` | GPL-2+ |
| **ConsensusClusterPlus** | **GPL version 2**, no "or later" |
| `cytonorm_rust_operator` | AGPL-3.0 |
| `asinh_rust_operator` | AGPL-3.0 |

GPL-2-only and AGPL-3 are incompatible, so **the CytoNorm operator cannot link this crate while
the metaclustering is a port of ConsensusClusterPlus and the operator is AGPL-3**. Three ways
out, in the order I would consider them:

1. **Split the crate.** The map, the mapping, `hclust` and `cutree` are all GPL-2-*or-later*
   sources, so they can ship as GPL-3 and be linked from an AGPL-3 operator. Only the consensus
   module is pinned to GPL-2-only, and it would live in a second crate that the operator does not
   link. CytoNorm then needs a metaclustering that is not ConsensusClusterPlus — which changes
   the answer, so it would have to be stated plainly rather than presented as parity.
2. **Relicense the operator to GPL-2-or-later**, so it can link the whole crate. Cheapest
   technically; it is a product decision about what Tercen's operators ship under.
3. **Ask the ConsensusClusterPlus authors** for a GPL-3 or dual licence. Slow, and it may not
   come.

There is a fourth reading — that matching behaviour is not copying expression, and a
reimplementation of a published method (Monti et al. 2003) is not a derivative work. I have not
assumed it, because I wrote this by reading their source.

## What is not here

- **`BuildMST`.** FlowSOM builds a minimum spanning tree for plotting and for `mst > 1` training.
  CytoNorm uses neither.
- **`AggregateFlowFrames`, `ReadInput`, scaling, `NewData`.** The operator's job, not the crate's.
- **Anything about speed.** The maps CytoNorm builds are 5x5 to 15x15 over a subsample, so the
  hundred-resample metaclustering is the expensive part and it takes 50 ms here.

## Next

1. Faris decides the licence question.
2. Wire it into `cytonorm_rust_operator` so `cluster > 1` works: train on the reference batch's
   cells, then give every cell its metacluster, and fit a spline per (batch, metacluster,
   channel) — the code for which is already there and only ever sees one cluster today.
