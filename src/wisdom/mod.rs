//! # Wisdom Module - Philosophical Grounding for Computational Cognition
//!
//! Re-exports from the `symthaea-wisdom` sub-crate.
//! See that crate for full documentation.

// Re-export all public items from the sub-crate
pub use symthaea_wisdom::*;

// Re-export submodules for path compatibility (crate::wisdom::autopoiesis::...)
pub use symthaea_wisdom::autopoiesis;
pub use symthaea_wisdom::harmonics;
pub use symthaea_wisdom::meta_cognition;
