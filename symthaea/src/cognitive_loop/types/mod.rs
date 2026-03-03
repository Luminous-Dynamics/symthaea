//! Public types returned by the cognitive loop.
//!
//! Decomposed from a monolithic types.rs into thematic sub-modules.
//! All public APIs are preserved via re-exports.

mod carryover;
mod scheduling;
mod telemetry;
mod output;

pub use scheduling::*;
pub use telemetry::*;
pub use output::*;

// Re-export crate-visible types
pub(crate) use carryover::{ConsciousnessCache, CycleCarryover, QualityMetrics};
// Used by test modules — gate to suppress unused-import warnings in lib builds
#[cfg(test)]
pub(crate) use carryover::{LearningState, UrgencyState};
pub(crate) use scheduling::CycleState;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
