//! Cognitive loop test suite, organized by focus area.
//!
//! - `core`: Basic service, flow state, emotion, curiosity, reflection, snapshot
//! - `cycle_properties`: Output bounds, NaN/Inf guards, monotonicity, determinism, clamping
//! - `subsystems`: Thalamic router, active inference, closed learning, episodic memory, goals, world model
//! - `integration`: Moral evaluation, FEP signals, stats, unified architecture, module enables
//! - `feedback`: Feedback loops, v0.6.3 modules, attestation, monitoring, synergy

#[allow(clippy::field_reassign_with_default)]
mod accessors_and_types;
mod core;
mod cycle_properties;
mod feedback;
mod helpers;
#[allow(clippy::field_reassign_with_default)]
mod integration;
#[allow(clippy::field_reassign_with_default)]
mod phase_coverage;
mod phase_results;
mod subsystems;
