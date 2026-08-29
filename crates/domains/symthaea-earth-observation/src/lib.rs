//! Provider-neutral Earth-observation contracts for Symthaea Planetary Perception.
//!
//! The central rule is epistemic: what a sensor observed, what processing
//! derived, what a model inferred, and what later evidence verified are
//! different things. Provider-specific I/O belongs in bridge crates; this crate
//! owns Earth/sensor-physics semantics and deterministic feature math.

pub mod features;
pub mod sar;
mod model;

pub use model::*;
