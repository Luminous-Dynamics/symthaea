// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotic agent: FEP-driven entity with consciousness-gated motor authority.
//!
//! # Safety positioning (SOTIF / ISO 21448)
//!
//! The scalar `motor_gain` returned by [`RoboticAgent::tick`] is a
//! **Φ-derived triggering-condition monitor** in the SOTIF sense (ISO
//! 21448:2022 *Road vehicles — Safety of the intended functionality*).
//! It is an advisory, continuous, model-based *confidence* signal —
//! **not** a substitute for safety-rated hardware envelopes. Concretely:
//!
//! - SOTIF §5 defines triggering conditions as "specific conditions of
//!   a scenario that serve as an initiator for a subsequent system
//!   reaction contributing to either a hazardous behavior or an
//!   absence of prevention". The HDC prediction-error surge that
//!   drives Φ down is exactly such a triggering-condition signal.
//!
//! - A system that uses `tick()`'s `motor_gain` as its *only* safety
//!   layer will **not** satisfy ISO 13849-1 PLd Cat.3, ISO 26262
//!   ASIL-D, or IEC 80601-2-77 on its own. The recommended
//!   architecture is:
//!
//!     ```text
//!     [ rule-based hard envelope  ]  ← ISO 13849 / 26262 certified
//!     [   (speed limits, e-stop,  ]    single-failure-tolerant,
//!     [    PFL force caps, etc.)  ]    microsecond-deterministic
//!              ▲
//!              │ Φ *augments* but never overrides
//!     [ RoboticAgent::tick motor_gain  ] ← this crate
//!     [  (advisory / attenuating only) ]
//!     ```
//!
//! - For actions where a single-point failure is catastrophic (energy
//!   delivery, irreversible tissue contact, etc.), the Φ channel
//!   **must** be paired with an independent diverse-redundancy hard
//!   gate. `symtropy-surgical-demo`'s `hardware_cautery_gate()` is the
//!   canonical example: cautery fires iff Φ *and* distance-to-critical
//!   *and* tip-force all agree.
//!
//! Three legitimate roles for the Φ signal have been observed across
//! the C-series platform demos:
//!
//!   1. **Magnitude attenuation** — scale a commanded torque / thrust
//!      down when Φ drops. Appropriate when the underlying controller
//!      has no "safer action" carve-outs (flight, humanoid, quadrotor
//!      attitude).
//!   2. **Mode selection** — pick from a discrete set of operating
//!      modes whose factors + interlocks are pre-certified individually
//!      (exoskeleton `AssistanceMode`, surgical `SurgicalSafetyLevel`).
//!   3. **Mission-phase tracking** — gate authority on external-world
//!      temporal windows (orbital comm/solar).
//!
//! See `memory/symtropy_phase_c2_flight_demo.md` for the empirical
//! positioning against 2024-2026 industry practice (PX4 discrete
//! failsafes, Franka 3-zone, Waymo rule-based fallback, etc.).

use nalgebra::SVector;
use symthaea_consciousness_equation::{
    ConsciousnessInputs, ConsciousnessResult, MasterConsciousnessEquation,
};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;

use crate::motor::{MotorPlanner, UniformGainPlanner};
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
    pub fn new(body: BodyHandle, platform: PlatformType, name: impl Into<String>) -> Self {
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
    ///
    /// # Safety contract (SOTIF)
    ///
    /// The returned `motor_gain` is **advisory** — it is a Φ-derived
    /// scalar whose semantics are:
    ///
    /// > "Given the internal HDC/FEP prediction-error state at this
    /// > tick, the model's confidence in the intended behavior is `g`;
    /// > downstream actuation should be scaled by at most `g` of its
    /// > nominal authority."
    ///
    /// Critically, this method has **no visibility into hardware
    /// envelopes** (joint limits, PFL force caps, ISO/TS 15066
    /// protective distance, FDA-cleared interlocks). Callers that
    /// need safety-rated behavior **must** combine this gain with an
    /// independent hard-envelope check — see the surgical demo's
    /// `hardware_cautery_gate` for the canonical dual-channel pattern.
    ///
    /// In terms of the ISO 21448 (SOTIF) functional-insufficiency
    /// framework: `tick()` is a *triggering-condition monitor* — it
    /// detects scenarios where the model's own prediction is likely
    /// wrong. It is **not** a safety function in the ISO 26262 sense,
    /// and should never be the sole gate on a single-point-of-failure
    /// hazardous action.
    pub fn tick(&mut self, observation: &[f64], danger_level: f64) -> f64 {
        // FEP perception-action cycle
        let obs = Observation::new(
            observation.to_vec(),
            0.8, // precision
            "game",
        );
        let perception = self.fep.perceive(&obs);
        let action = self.fep.select_action();

        // Update caution based on danger
        if danger_level > 0.5 {
            self.caution = (self.caution + 0.05).min(1.0);
        } else {
            self.caution = (self.caution - 0.02).max(0.0);
        }

        // ── FEP-derived signals (platform-aware, observation-aware) ─────
        //
        // The observation vector goes through `fep.perceive()`, which
        // produces prediction error, free-energy components, precision,
        // and belief change. These reflect the model's live estimate of
        // "how well am I predicting this platform's sensor stream." We
        // thread them into `ConsciousnessInputs` so the consciousness
        // equation's output actually depends on what the robot is
        // seeing and how well its generative model fits, not just on
        // danger_level.
        //
        // Transform unbounded FEP magnitudes to [0, 1] with saturating
        // functions so a pathological spike can't push inputs outside
        // the consciousness equation's expected range.
        //
        // Each FEP-derived channel is blended with its prior hardcoded
        // default at a 35:65 ratio (FEP:hardcoded). This keeps the
        // output band close to the original [0.099, 0.145] under
        // representative inputs (no sudden regressions on the paper's
        // benchmarks) while letting platform-specific observation
        // streams introduce genuine variation. The blend weights can
        // be tuned once we see real per-platform traces.
        let pe_raw = perception.free_energy.prediction_error.abs();
        let pe_norm = 1.0 - (-pe_raw * 2.0).exp(); // 0 at pe=0, →1 as pe→∞
        let belief_change_norm = 1.0 - (-perception.belief_change.max(0.0) * 2.0).exp();
        let precision_norm = {
            // precision is in roughly [0, ∞); map to [0, 1] via a soft
            // logistic centered at 1.0 (our default `Observation`
            // precision).
            let p = perception.precision.max(0.0);
            p / (p + 1.0)
        };
        let fe_total_norm = {
            let fe = self.fep.current_free_energy().abs();
            1.0 / (1.0 + fe) // →1 when FE low (good fit), →0 when FE high
        };
        // Action-distribution concentration: uniform → broad (~1/num_actions per),
        // concentrated → one bin dominates. Use (max_prob - uniform) / (1 - uniform)
        // clipped to [0, 1] as a concentration score.
        let action_concentration = {
            let probs = &action.action_probabilities;
            if probs.is_empty() {
                0.0_f64
            } else {
                let uniform = 1.0 / probs.len() as f64;
                let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
                ((max_p - uniform) / (1.0 - uniform).max(1e-6)).clamp(0.0, 1.0)
            }
        };

        // 35% FEP-derived + 65% hardcoded-baseline blend.
        const FEP_WEIGHT: f64 = 0.35;
        const BASE_WEIGHT: f64 = 1.0 - FEP_WEIGHT;

        let inputs = ConsciousnessInputs {
            phi: (1.0 - self.caution) * 0.7 + 0.3,
            broadcast: ((1.0 - danger_level * 0.4).max(0.2) * BASE_WEIGHT
                + action_concentration * FEP_WEIGHT)
                .clamp(0.2, 1.0),
            working_memory: (0.7 * BASE_WEIGHT + precision_norm * FEP_WEIGHT).clamp(0.2, 0.95),
            attention: (((danger_level * 0.3 + 0.5).min(1.0)) * BASE_WEIGHT + pe_norm * FEP_WEIGHT)
                .clamp(0.2, 0.98),
            recurrence: (0.6 * BASE_WEIGHT + belief_change_norm * FEP_WEIGHT).clamp(0.2, 0.9),
            embodiment: (1.0 - danger_level * 0.2).max(0.3),
            knowledge: (0.5 * BASE_WEIGHT + fe_total_norm * FEP_WEIGHT).clamp(0.2, 0.9),
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

    /// Run one tick and plan per-joint motor commands with the given planner.
    ///
    /// Returns a vector of commands with length [`PlatformType::num_actuators`].
    /// The scalar NRC safety gain is fed into `planner.plan(observation, gain,
    /// num_actuators)`. Use [`UniformGainPlanner`] for a no-intelligence
    /// baseline, or provide your own planner to map observation + gain to
    /// real per-joint commands (e.g. wrap a Symthaea `EmbodimentBridge` step).
    pub fn tick_motor_commands_with<P: MotorPlanner>(
        &mut self,
        observation: &[f64],
        danger_level: f64,
        planner: &mut P,
    ) -> Vec<f64> {
        let gain = self.tick(observation, danger_level);
        planner.plan(observation, gain, self.platform.num_actuators())
    }

    /// Convenience: run one tick and plan per-joint motor commands using
    /// [`UniformGainPlanner`] (every actuator gets the same gain).
    ///
    /// Not a realistic controller — for real robotics, wire a platform-specific
    /// planner via [`tick_motor_commands_with`](Self::tick_motor_commands_with).
    pub fn tick_motor_commands(&mut self, observation: &[f64], danger_level: f64) -> Vec<f64> {
        let mut planner = UniformGainPlanner;
        self.tick_motor_commands_with(observation, danger_level, &mut planner)
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
    let handle = physics.add_sphere(position, platform.default_radius(), platform.default_mass());
    RoboticAgent::new(handle, platform, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_quadrotor_agent() {
        let agent = RoboticAgent::new(BodyHandle(0), PlatformType::Quadrotor, "Drone-1");
        assert_eq!(agent.platform, PlatformType::Quadrotor);
        assert_eq!(agent.name, "Drone-1");
        assert_eq!(agent.safety_tier, SafetyTier::Green);
        assert!(!agent.player_controlled);
    }

    #[test]
    fn tick_produces_motor_gain() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Vehicle, "Car-1");
        let gain = agent.tick(&[0.5, 0.3, 0.1, 0.2], 0.0);
        // With low danger, should have reasonable consciousness → positive gain
        assert!(gain >= 0.0);
        assert!(gain <= 1.0);
    }

    #[test]
    fn high_danger_increases_caution() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "Bot-1");
        let initial_caution = agent.caution;
        agent.tick(&[0.5, 0.3, 0.9, 0.8], 0.9);
        assert!(agent.caution > initial_caution);
    }

    #[test]
    fn player_control_transfer() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Quadrotor, "Drone");
        assert!(!agent.player_controlled);
        agent.set_player_controlled(true);
        assert!(agent.player_controlled);
        agent.set_player_controlled(false);
        assert!(!agent.player_controlled);
    }

    #[test]
    fn spawn_robot_creates_body() {
        let mut physics =
            symtropy_physics::PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));
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
        let mut physics =
            symtropy_physics::PhysicsWorld::<4>::new(SVector::from([0.0, -9.81, 0.0, 0.0]));
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
    fn tick_motor_commands_humanoid_produces_21_dof() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "Humanoid-1");
        let cmds = agent.tick_motor_commands(&[0.5, 0.3, 0.1, 0.2], 0.1);
        assert_eq!(cmds.len(), 21, "humanoid should emit 21 per-joint commands");
        // All joints should get the same scalar gain from the uniform planner.
        let g = cmds[0];
        assert!(g >= 0.0 && g <= 1.0);
        for c in &cmds {
            assert!((c - g).abs() < 1e-12);
        }
    }

    #[test]
    fn tick_motor_commands_scales_with_danger() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "Humanoid-1");
        // After many high-danger ticks, caution saturates → low phi → Red tier → 0 gain
        for _ in 0..100 {
            agent.tick_motor_commands(&[0.5, 0.3, 0.9, 0.8], 0.95);
        }
        let cmds = agent.tick_motor_commands(&[0.5, 0.3, 0.9, 0.8], 0.95);
        let max = cmds.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max <= 0.5,
            "sustained danger should suppress motor authority (got max={max})"
        );
    }

    #[test]
    fn tick_motor_commands_with_custom_planner() {
        use crate::motor::MotorPlanner;

        struct HalfPlanner;
        impl MotorPlanner for HalfPlanner {
            fn plan(&mut self, _: &[f64], gain: f64, n: usize) -> Vec<f64> {
                vec![gain * 0.5; n]
            }
        }

        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Manipulator, "Arm-1");
        let mut planner = HalfPlanner;
        let cmds = agent.tick_motor_commands_with(&[0.5, 0.3, 0.1, 0.2], 0.0, &mut planner);
        assert_eq!(cmds.len(), 8, "manipulator has 8 actuators");
        // Half-planner halves the gain uniformly.
        let expected = agent.safety_tier.motor_gain() * 0.5;
        for c in &cmds {
            assert!((c - expected).abs() < 1e-12, "c={c} expected={expected}");
        }
    }

    #[test]
    fn tick_motor_commands_every_platform() {
        // Coverage test for the full 10-platform enum — verifies every
        // PlatformType value flows through the FEP + consciousness +
        // UniformGainPlanner pipeline and emits the expected number of
        // per-actuator commands. Catches regressions when PlatformType is
        // extended but num_actuators() forgets a new variant.
        for p in PlatformType::ALL {
            let mut agent = RoboticAgent::new(BodyHandle(0), p, p.name());
            let cmds = agent.tick_motor_commands(&[0.5, 0.3, 0.1, 0.2], 0.1);
            assert_eq!(
                cmds.len(),
                p.num_actuators(),
                "{:?}: planner emitted {} commands, expected {}",
                p,
                cmds.len(),
                p.num_actuators(),
            );
            for c in &cmds {
                assert!(
                    (0.0..=1.0).contains(c),
                    "{:?}: motor gain {c} outside [0, 1]",
                    p,
                );
            }
            assert!(agent.consciousness_result.is_some(), "{:?}: no phi", p);
        }
    }

    #[test]
    fn consciousness_computed_after_tick() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Helicopter, "Heli-1");
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

        let mut physics =
            symtropy_physics::PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));

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
