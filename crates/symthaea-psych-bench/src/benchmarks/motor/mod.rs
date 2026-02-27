//! Motor learning domain benchmarks.
//!
//! - **SRTT** — Serial Reaction Time Task (implicit sequence learning)
//! - **FittsLaw** — Speed-accuracy tradeoff in motor targeting
//! - **Bimanual** — Dual-rhythm coordination interference

pub mod bimanual;
pub mod fitts_law;
pub mod srtt;

pub use bimanual::BimanualBenchmark;
pub use fitts_law::FittsLawBenchmark;
pub use srtt::SrttBenchmark;
