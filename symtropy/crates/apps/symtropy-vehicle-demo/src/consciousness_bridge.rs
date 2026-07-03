// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness pipeline wiring — same pattern as flight-demo + manipulator-demo.

use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_robotics_bridge::RoboticAgentTrait;
use symtropy_robotics_bridge::agent::RoboticAgent;

/// Run one consciousness tick.
///
/// `observation` packs: `[prediction_error, danger, speed_norm, slip_norm]`.
/// Returns `(phi, safety_tier, motor_gain)`.
pub fn consciousness_tick(
    agent: &mut RoboticAgent,
    prediction_error: f32,
    danger_level: f64,
    speed_norm: f64,
    slip_norm: f64,
) -> (f64, SafetyTier, f64) {
    let observation = [prediction_error as f64, danger_level, speed_norm, slip_norm];
    let motor_gain = agent.tick(&observation, danger_level);
    (agent.phi(), agent.safety_tier, motor_gain)
}
