//! Consciousness domain benchmarks.
//!
//! - **Blindsight** — Dissociation between forced-choice accuracy and subjective
//!   awareness report (Weiskrantz, 1986).

pub mod blindsight;
pub mod binocular_rivalry;

pub use blindsight::BlindSightBenchmark;
pub use binocular_rivalry::BinocularRivalryBenchmark;
