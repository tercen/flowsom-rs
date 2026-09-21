//! FlowSOM's self-organising map, ported from the C and R that Tercen's CytoNorm operator runs
//! today (FlowSOM 1.22.0 on R 4.0.4, in `lucas501/cytonorm_docker:1.1.9`).
//!
//! The point of the port is that it gives the *same clusters*, not merely similar ones, so it
//! reproduces R's random stream ([`rng`]) and the training loop's arithmetic down to the
//! truncation quirks in the original C ([`som`]).
pub mod rng;
pub mod som;
