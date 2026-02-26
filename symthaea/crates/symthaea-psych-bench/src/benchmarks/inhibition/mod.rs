//! Inhibition domain benchmarks.
//!
//! - **Go/No-Go** — Prepotent response inhibition (commission errors)
//! - **Stop Signal** — Inhibitory control latency (SSRT)

pub mod go_nogo;
pub mod stop_signal;

pub use go_nogo::GoNoGoBenchmark;
pub use stop_signal::StopSignalBenchmark;
