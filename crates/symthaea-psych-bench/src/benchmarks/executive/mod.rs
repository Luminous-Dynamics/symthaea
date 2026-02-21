//! Executive function benchmarks.
//!
//! - **WCST** — Wisconsin Card Sorting Test (rule learning, set-shifting, perseveration)
//! - **IGT** — Iowa Gambling Task (decision-making under ambiguity, loss aversion)
//! - **Raven's** — Progressive Matrices (fluid intelligence, pattern completion)
//! - **Stroop** — Color-Word Interference (inhibitory control, selective attention)
//! - **Flanker** — Eriksen Flanker Task (spatial attention filtering, response inhibition)

pub mod flanker;
pub mod iowa_gambling;
pub mod ravens;
pub mod stroop;
pub mod wisconsin;

pub use flanker::FlankerBenchmark;
pub use iowa_gambling::IowaGamblingBenchmark;
pub use ravens::RavensProgressiveMatricesBenchmark;
pub use stroop::StroopBenchmark;
pub use wisconsin::WisconsinCardSortingBenchmark;
