//! Cognitive loop test suite, organized by focus area.
//!
//! - `core`: Basic service, flow state, emotion, curiosity, reflection, snapshot
//! - `subsystems`: Thalamic router, active inference, closed learning, episodic memory, goals, world model
//! - `integration`: Moral evaluation, FEP signals, stats, unified architecture, module enables
//! - `feedback`: Feedback loops, v0.6.3 modules, attestation, monitoring, synergy

mod core;
mod feedback;
mod helpers;
mod integration;
mod subsystems;
