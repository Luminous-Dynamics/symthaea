// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full consciousness pipeline wiring — same pattern as the manipulator demo.
//!
//! Feeds prediction error (from HDC cosine dissimilarity) and danger level
//! (from wind-gust magnitude) into the RoboticAgent, which runs:
//!
//! 1. FEP perceive (observation + precision)
//! 2. FEP select_action()
//! 3. MasterConsciousnessEquation → Φ
//! 4. SafetyTier from real Φ
//! 5. Motor gain ∈ [0,1]
//!
//! No shortcuts: Φ is not a clamped lerp of PE.

use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_robotics_bridge::agent::RoboticAgent;

/// Run one consciousness tick through the full pipeline.
///
/// `observation` packs: `[prediction_error, danger, altitude_norm, attitude_norm]`.
/// `danger_level` is passed separately to the agent's internal modulation.
///
/// Returns `(phi, safety_tier, motor_gain)`.
pub fn consciousness_tick(
    agent: &mut RoboticAgent,
    prediction_error: f32,
    danger_level: f64,
    altitude_norm: f64,
    attitude_norm: f64,
) -> (f64, SafetyTier, f64) {
    let observation = [
        prediction_error as f64,
        danger_level,
        altitude_norm,
        attitude_norm,
    ];

    let motor_gain = agent.tick(&observation, danger_level);

    (agent.phi(), agent.safety_tier, motor_gain)
}
