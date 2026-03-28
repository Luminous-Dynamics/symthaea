//! Sustained Attention domain benchmarks.
//!
//! - **SART** — Sustained Attention to Response Task (commission errors)
//! - **PVT** — Psychomotor Vigilance Task (fatigue-related RT slowing)
//! - **CPT** — Continuous Performance Task (2-back sequential matching)

pub mod cpt;
pub mod pvt;
pub mod sart;

pub use cpt::CptBenchmark;
pub use pvt::PvtBenchmark;
pub use sart::SartBenchmark;
