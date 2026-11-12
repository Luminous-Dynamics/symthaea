#![forbid(unsafe_code)]
#![deny(warnings)]

// Include the exact production modules at crate root because chain.rs imports
// crate::distribution. This harness intentionally mirrors that module path.
#[path = "../../../crates/domains/symthaea-markov/src/distribution.rs"]
pub mod distribution;

#[path = "../../../crates/domains/symthaea-markov/src/chain.rs"]
pub mod chain;
