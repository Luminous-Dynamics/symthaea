//! Executive function benchmarks.
//!
//! - **WCST** — Wisconsin Card Sorting Test (rule learning, set-shifting, perseveration)
//! - **IGT** — Iowa Gambling Task (decision-making under ambiguity, loss aversion)
//! - **Raven's** — Progressive Matrices (fluid intelligence, pattern completion)

pub mod iowa_gambling;
pub mod ravens;
pub mod wisconsin;

pub use iowa_gambling::IowaGamblingBenchmark;
pub use ravens::RavensProgressiveMatricesBenchmark;
pub use wisconsin::WisconsinCardSortingBenchmark;
