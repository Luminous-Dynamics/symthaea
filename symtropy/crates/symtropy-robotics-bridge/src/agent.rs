// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotic agent: FEP-driven entity with consciousness-gated motor authority.

use nalgebra::SVector;
use symthaea_consciousness_equation::{
    ConsciousnessInputs, ConsciousnessResult, MasterConsciousnessEquation,
};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;

use crate::platform::PlatformType;

/// A robotic agent in the game world.
///
/// Combines:
/// - An `ActiveInferenceAgent` (FEP) for perception-action decision making
/// - A `MasterConsciousnessEquation` for consciousness computation
/// - A `BodyHandle` linking to the physics world
/// - A `PlatformType` defining actuator count and physics properties
///
/// The agent's consciousness level gates its motor authority via `SafetyTier`.
pub struct RoboticAgent {
    /// Physics body handle.
    pub body: BodyHandle,
    /// Platform type.
    pub platform: PlatformType,
    /// FEP active inference agent.
    pub fep: ActiveInferenceAgent,
    /// Consciousness equation.
    pub consciousness: MasterConsciousnessEquation,
    /// Latest consciousness result.
    pub consciousness_result: Option<ConsciousnessResult>,
    /// Current safety tier (derived from Φ).
    pub safety_tier: SafetyTier,
    /// Name for display/dialogue.
    pub name: String,
    /// Caution level [0, 1] — modulated by FEP surprise.
    pub caution: f64,
    /// Whether the player is currently controlling this agent.
    pub player_controlled: bool,
}

impl RoboticAgent {
    /// Create a new robotic agent.
    pub fn new(
        body: BodyHandle,
        platform: PlatformType,
        name: impl Into<String>,
    ) -> Self {
        let config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: platform.num_actuators(),
            ..Default::default()
        };

        Self {
            body,
            platform,
            fep: ActiveInferenceAgent::new(config),
            consciousness: MasterConsciousnessEquation::default(),
            consciousness_result: None,
            safety_tier: SafetyTier::Green,
            name: name.into(),
            caution: 0.5,
            player_controlled: false,
        }
    }

    /// Run one perception-action cycle.
    ///
    /// 1. Build observation from environment state
    /// 2. FEP perceive + select action
    /// 3. Compute consciousness
    /// 4. Gate motor output by safety tier
    ///
    /// Returns the motor gain [0.0, 1.0] for this tick.
    pub fn tick(
        &mut self,
        observation: &[f64],
        danger_level: f64,
    ) -> f64 {
        // FEP perception-action cycle
        let obs = Observation::new(
            observation.to_vec(),
            0.8, // precision
            "game",
        );
        let _perception = self.fep.perceive(&obs);
        let _action = self.fep.select_action();

        // Update caution based on danger
        if danger_level > 0.5 {
            self.caution = (self.caution + 0.05).min(1.0);
        } else {
            self.caution = (self.caution - 0.02).max(0.0);
        }

        // Compute consciousness
        let inputs = ConsciousnessInputs {
            phi: (1.0 - self.caution) * 0.7 + 0.3,
            broadcast: (1.0 - danger_level * 0.4).max(0.2),
            working_memory: 0.7,
            attention: (danger_level * 0.3 + 0.5).min(1.0),
            recurrence: 0.6,
            embodiment: (1.0 - danger_level * 0.2).max(0.3),
            knowledge: 0.5,
            synchrony: 0.6,
        };
        let result = self.consciousness.compute(&inputs);
        let phi = result.consciousness_level;
        self.safety_tier = SafetyTier::from_phi(phi);
        self.consciousness_result = Some(result);

        self.safety_tier.motor_gain()
    }

    /// Current Φ level.
    pub fn phi(&self) -> f64 {
        self.consciousness_result
            .as_ref()
            .map(|r| r.consciousness_level)
            .unwrap_or(0.0)
    }

    /// Current bottleneck name.
    pub fn bottleneck(&self) -> &str {
        self.consciousness_result
            .as_ref()
            .map(|r| r.bottleneck_name.as_str())
            .unwrap_or("uncomputed")
    }

    /// Current free energy (surprise) from the FEP agent.
    pub fn free_energy(&self) -> f64 {
        self.fep.current_free_energy()
    }

    /// Whether this agent has enough consciousness for motor output.
    pub fn can_act(&self) -> bool {
        self.safety_tier.allows_output()
    }

    /// Transfer control to/from the player.
    pub fn set_player_controlled(&mut self, controlled: bool) {
        self.player_controlled = controlled;
    }
}

/// Spawn a robotic agent into the physics world and return the agent.
pub fn spawn_robot<const D: usize>(
    physics: &mut symtropy_physics::PhysicsWorld<D>,
    platform: PlatformType,
    position: symtropy_math::Point<D>,
    name: impl Into<String>,
) -> RoboticAgent {
    let handle = physics.add_sphere(
        position,
        platform.default_radius(),
        platform.default_mass(),
    );
    RoboticAgent::new(handle, platform, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_quadrotor_agent() {
        let agent = RoboticAgent::new(
            BodyHandle(0),
            PlatformType::Quadrotor,
            "Drone-1",
        );
        assert_eq!(agent.platform, PlatformType::Quadrotor);
        assert_eq!(agent.name, "Drone-1");
        assert_eq!(agent.safety_tier, SafetyTier::Green);
        assert!(!agent.player_controlled);
    }

    #[test]
    fn tick_produces_motor_gain() {
        let mut agent = RoboticAgent::new(
            BodyHandle(0),
            PlatformType::Vehicle,
            "Car-1",
        );
        let gain = agent.tick(&[0.5, 0.3, 0.1, 0.2], 0.0);
        // With low danger, should have reasonable consciousness → positive gain
        assert!(gain >= 0.0);
        assert!(gain <= 1.0);
    }

    #[test]
    fn high_danger_increases_caution() {
        let mut agent = RoboticAgent::new(
            BodyHandle(0),
            PlatformType::Humanoid,
            "Bot-1",
        );
        let initial_caution = agent.caution;
        agent.tick(&[0.5, 0.3, 0.9, 0.8], 0.9);
        assert!(agent.caution > initial_caution);
    }

    #[test]
    fn player_control_transfer() {
        let mut agent = RoboticAgent::new(
            BodyHandle(0),
            PlatformType::Quadrotor,
            "Drone",
        );
        assert!(!agent.player_controlled);
        agent.set_player_controlled(true);
        assert!(agent.player_controlled);
        agent.set_player_controlled(false);
        assert!(!agent.player_controlled);
    }

    #[test]
    fn spawn_robot_creates_body() {
        let mut physics = symtropy_physics::PhysicsWorld::<3>::new(
            SVector::from([0.0, -9.81, 0.0]),
        );
        let agent = spawn_robot(
            &mut physics,
            PlatformType::Quadrotor,
            symtropy_math::Point::new([0.0, 10.0, 0.0]),
            "Test-Drone",
        );
        assert_eq!(physics.body_count(), 1);
        assert_eq!(agent.platform, PlatformType::Quadrotor);
        assert_eq!(agent.name, "Test-Drone");

        // Body should exist in physics world
        let body = physics.body(agent.body).unwrap();
        assert!((body.position().coord(1) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn spawn_robot_4d() {
        let mut physics = symtropy_physics::PhysicsWorld::<4>::new(
            SVector::from([0.0, -9.81, 0.0, 0.0]),
        );
        let agent = spawn_robot(
            &mut physics,
            PlatformType::Auv,
            symtropy_math::Point::new([0.0, 0.0, -50.0, 0.0]),
            "Sub-1",
        );
        assert_eq!(physics.body_count(), 1);
        assert_eq!(agent.platform, PlatformType::Auv);
    }

    #[test]
    fn consciousness_computed_after_tick() {
        let mut agent = RoboticAgent::new(
            BodyHandle(0),
            PlatformType::Helicopter,
            "Heli-1",
        );
        assert!(agent.consciousness_result.is_none());

        agent.tick(&[0.5, 0.3, 0.1, 0.2], 0.2);

        assert!(agent.consciousness_result.is_some());
        assert!(agent.phi() >= 0.0);
    }

    #[test]
    fn all_platforms_spawn() {
        let platforms = [
            PlatformType::Quadrotor,
            PlatformType::Vehicle,
            PlatformType::Humanoid,
            PlatformType::Auv,
            PlatformType::Helicopter,
            PlatformType::Manipulator,
        ];

        let mut physics = symtropy_physics::PhysicsWorld::<3>::new(
            SVector::from([0.0, -9.81, 0.0]),
        );

        for (i, platform) in platforms.iter().enumerate() {
            let agent = spawn_robot(
                &mut physics,
                *platform,
                symtropy_math::Point::new([i as f64 * 10.0, 0.0, 0.0]),
                format!("{}-{}", platform.name(), i),
            );
            assert_eq!(agent.platform, *platform);
        }

        assert_eq!(physics.body_count(), 6);
    }
}
