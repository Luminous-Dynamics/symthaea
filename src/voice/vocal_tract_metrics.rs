//! Vocal tract quality metrics — re-exported from `symthaea-vocal-tract` sub-crate.
//!
//! See `crates/symthaea-vocal-tract/src/metrics.rs` for the canonical implementation.

#[cfg(feature = "vocal-tract")]
pub use symthaea_vocal_tract::metrics::{load_wav, save_wav, VocalTractMetrics};
