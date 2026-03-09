//! Motor learning domain benchmarks.
//!
//! - **SRTT** — Serial Reaction Time Task (implicit sequence learning)
//! - **FittsLaw** — Speed-accuracy tradeoff in motor targeting
//! - **Bimanual** — Dual-rhythm coordination interference
//! - **ProprioceptiveDrift** — Rubber Hand Illusion (multisensory body ownership)

pub mod bimanual;
pub mod fitts_law;
pub mod proprioceptive_drift;
pub mod srtt;

pub use bimanual::BimanualBenchmark;
pub use fitts_law::FittsLawBenchmark;
pub use proprioceptive_drift::ProprioceptiveDriftBenchmark;
pub use srtt::SrttBenchmark;
