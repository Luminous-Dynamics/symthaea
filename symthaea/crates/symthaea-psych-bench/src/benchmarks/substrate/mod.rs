//! Substrate independence domain benchmarks.
//!
//! - **Transfer** — Substrate Transfer fidelity (Multiple Realizability, Putnam 1967)

pub mod degradation;
pub mod transfer;

pub use degradation::SubstrateDegradationBenchmark;
pub use transfer::SubstrateTransferBenchmark;
