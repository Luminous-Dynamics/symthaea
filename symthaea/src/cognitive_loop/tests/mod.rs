// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognitive loop test suite, organized by focus area.
//!
//! - `core`: Basic service, flow state, emotion, curiosity, reflection, snapshot
//! - `cycle_properties`: Output bounds, NaN/Inf guards, monotonicity, determinism, clamping
//! - `subsystems`: Thalamic router, active inference, closed learning, episodic memory, goals, world model
//! - `integration`: Moral evaluation, FEP signals, stats, unified architecture, module enables
//! - `feedback`: Feedback loops, v0.6.3 modules, attestation, monitoring, synergy

#[allow(clippy::field_reassign_with_default)]
mod accessors_and_types;
mod arousal_trap;
mod behavioral_modulation;
mod core;
mod crucible_integration;
mod cycle_properties;
mod dream_pipeline;
mod error_recovery;
mod feedback;
mod helpers;
#[allow(clippy::field_reassign_with_default)]
mod integration;
mod managers;
mod memory_pipeline;
mod moral_and_drives;
#[allow(clippy::field_reassign_with_default)]
mod phase_coverage;
mod phase_results;
mod proptest_cross_coupling;
mod proptest_feedback_consensus;
// proptest_feedback_loops: references MCE_NARRATIVE/MCE_SOCIAL/RESONANCE_FLOW constants
// that don't exist in thresholds.rs yet. Uncomment when those constants are added.
// mod proptest_feedback_loops;
mod proptest_nan_resilience;
mod proptest_substrate;
mod sensor_blending;
mod subsystem_smoke;
mod subsystems;
