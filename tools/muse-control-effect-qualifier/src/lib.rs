//! Strict isolated qualification target for the production MUSE control-effect evaluator.
//!
//! This imports the production source directly. It exists so targeted tests and
//! warnings-denied Clippy can run without inheriting unrelated legacy warnings
//! from the wider symthaea-muse crate.

#[path = "../../../crates/domains/symthaea-muse/src/control_effect.rs"]
pub mod control_effect;
