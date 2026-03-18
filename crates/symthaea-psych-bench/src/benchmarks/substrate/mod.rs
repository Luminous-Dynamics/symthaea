//! Substrate independence domain benchmarks.
//!
//! - **Transfer** — Substrate Transfer fidelity (Multiple Realizability, Putnam 1967)
//! - **Degradation** — Graceful degradation under substrate quality loss
//! - **Latency** — Processing speed effects across substrate types

pub mod degradation;
pub mod latency;
pub mod transfer;

pub use degradation::SubstrateDegradationBenchmark;
pub use latency::SubstrateLatencyBenchmark;
pub use transfer::SubstrateTransferBenchmark;
