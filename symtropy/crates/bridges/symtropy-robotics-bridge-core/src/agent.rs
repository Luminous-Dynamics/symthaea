// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Core robotic agent traits and utilities.

use crate::motor::MotorPlanner;
use crate::platform::PlatformType;
use symtropy_physics::body::BodyHandle;

/// Trait for a robotic agent in the game world.
///
/// This trait defines the interface for an agent that can be ticked by the
/// game engine and produce motor commands. The core engine is agnostic to
/// the specific AI / consciousness model driving the agent.
pub trait RoboticAgent {
    /// Physics body handle.
    fn body(&self) -> BodyHandle;

    /// Platform type.
    fn platform(&self) -> PlatformType;

    /// Run one perception-action cycle.
    /// Returns the motor gain [0.0, 1.0] for this tick.
    fn tick(&mut self, observation: &[f64], danger_level: f64) -> f64;

    /// Current Φ level (if applicable, else 0.0).
    fn phi(&self) -> f64 {
        0.0
    }

    /// Current bottleneck name (for debug/UI).
    fn bottleneck(&self) -> &str {
        "unknown"
    }

    /// Whether this agent has enough consciousness/confidence for motor output.
    fn can_act(&self) -> bool {
        true
    }

    /// Plan per-joint motor commands with the given planner.
    fn tick_motor_commands_with<P: MotorPlanner>(
        &mut self,
        observation: &[f64],
        danger_level: f64,
        planner: &mut P,
    ) -> Vec<f64> {
        let gain = self.tick(observation, danger_level);
        planner.plan(observation, gain, self.platform().num_actuators())
    }

    /// Convenience: run one tick and plan per-joint motor commands using
    /// [`UniformGainPlanner`](crate::motor::UniformGainPlanner) (every actuator gets the same gain).
    fn tick_motor_commands(&mut self, observation: &[f64], danger_level: f64) -> Vec<f64> {
        let mut planner = crate::motor::UniformGainPlanner;
        self.tick_motor_commands_with(observation, danger_level, &mut planner)
    }
}

/// Spawn a robotic agent's body into the physics world.
pub fn spawn_robot_body<const D: usize>(
    physics: &mut symtropy_physics::PhysicsWorld<D>,
    platform: PlatformType,
    position: symtropy_math::Point<D>,
) -> BodyHandle {
    physics.add_sphere(position, platform.default_radius(), platform.default_mass())
}
