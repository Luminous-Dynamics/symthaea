//! ToMBench: Theory of Mind benchmark suite.
//!
//! Tests `SocialCoherence` module with mental model tracking,
//! belief/desire/intention attribution via 5 task types.

pub mod false_belief;
pub mod faux_pas;
pub mod hinting;
pub mod persuasion;
pub mod strange_story;

pub use false_belief::FalseBeliefBenchmark;
pub use faux_pas::FauxPasBenchmark;
pub use hinting::HintingBenchmark;
pub use persuasion::PersuasionBenchmark;
pub use strange_story::StrangeStoryBenchmark;
