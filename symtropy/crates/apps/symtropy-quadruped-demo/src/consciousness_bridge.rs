// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness tick wrapper.

use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_robotics_bridge::RoboticAgentTrait;
use symtropy_robotics_bridge::agent::RoboticAgent;

pub fn consciousness_tick(
    agent: &mut RoboticAgent,
    prediction_error: f32,
    danger_level: f64,
    height_norm: f64,
    terrain_roughness: f64,
) -> (f64, SafetyTier, f64) {
    let observation = [
        prediction_error as f64,
        danger_level,
        height_norm,
        terrain_roughness,
    ];
    let motor_gain = agent.tick(&observation, danger_level);
    (agent.phi(), agent.safety_tier, motor_gain)
}
