// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge a Symthaea [`EmbodimentBridge`] into the Symtropy [`MotorPlanner`]
//! protocol.
//!
//! The Symtropy [`RoboticAgent`](symtropy_robotics_bridge::RoboticAgent)
//! computes a scalar NRC safety gain each tick and then hands that gain to a
//! [`MotorPlanner`] to produce per-actuator commands. Platform crates in
//! Symthaea (`symthaea-humanoid`, `symthaea-flight`, `symthaea-manipulator`,
//! etc.) implement [`EmbodimentBridge`], whose `step(thought_hv, dt, phi)`
//! internally translates thought to per-joint torques and advances the
//! platform's own physics.
//!
//! [`EmbodimentPlanner`] glues those two worlds together. It:
//! 1. Encodes the observation vector into a [`ContinuousHV`] "thought".
//! 2. Calls `bridge.step(&thought_hv, dt, gain as f64)`.
//! 3. Returns a per-actuator `Vec<f64>` sized to `num_actuators`, where each
//!    slot holds the bridge's commanded authority for that actuator — gated
//!    by the bridge's reported `MotorSafetyLevel` and capped by the incoming
//!    NRC gain.
//!
//! The commands are currently uniform across actuators, reflecting the
//! information available in [`EmbodimentResult`]. A future version can shift
//! to per-joint telemetry if [`EmbodimentBridge`] grows a
//! `last_motor_commands()` accessor.
//!
//! # Example
//!
//! ```ignore
//! use symtropy_robotics_bridge::{MotorPlanner, PlatformType, RoboticAgent};
//! use symtropy_symthaea_adapter::EmbodimentPlanner;
//! use symthaea_core::embodiment::EmbodimentBridge;
//! use symtropy_physics::body::BodyHandle;
//!
//! let bridge: Box<dyn EmbodimentBridge> = make_humanoid_bridge();
//! let mut planner = EmbodimentPlanner::new(bridge);
//!
//! let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "Bot");
//! let commands = agent.tick_motor_commands_with(&[0.5, 0.3, 0.1, 0.2], 0.1, &mut planner);
//! assert_eq!(commands.len(), 21);
//! ```

use symthaea_core::embodiment::EmbodimentBridge;
use symthaea_core::hdc::ContinuousHV;
use symtropy_robotics_bridge::MotorPlanner;

/// Default simulation timestep fed to `EmbodimentBridge::step` (60 Hz).
pub const DEFAULT_DT: f32 = 1.0 / 60.0;

/// Default hypervector dimension used for the encoded thought.
///
/// Matches Symthaea's 16,384D HDC convention. Platforms that expect a
/// different dimension will internally rebuild their encoders, but the
/// default is correct for every built-in platform.
pub const DEFAULT_HV_DIM: usize = 16_384;

/// Wraps a Symthaea [`EmbodimentBridge`] as a Symtropy [`MotorPlanner`].
pub struct EmbodimentPlanner {
    bridge: Box<dyn EmbodimentBridge>,
    dt: f32,
    hv_dim: usize,
}

impl EmbodimentPlanner {
    /// Wrap a bridge with default timestep (60 Hz) and 16,384D HVs.
    pub fn new(bridge: Box<dyn EmbodimentBridge>) -> Self {
        Self {
            bridge,
            dt: DEFAULT_DT,
            hv_dim: DEFAULT_HV_DIM,
        }
    }

    /// Override the simulation timestep fed to `bridge.step`.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Override the hypervector dimension used when encoding observations.
    pub fn with_hv_dim(mut self, hv_dim: usize) -> Self {
        self.hv_dim = hv_dim;
        self
    }

    /// Read-only access to the wrapped bridge.
    pub fn bridge(&self) -> &dyn EmbodimentBridge {
        self.bridge.as_ref()
    }

    /// Mutable access to the wrapped bridge (for reset, safety overrides, ...).
    pub fn bridge_mut(&mut self) -> &mut dyn EmbodimentBridge {
        self.bridge.as_mut()
    }

    /// Current simulation timestep.
    pub fn dt(&self) -> f32 {
        self.dt
    }
}

impl MotorPlanner for EmbodimentPlanner {
    fn plan(&mut self, observation: &[f64], gain: f64, num_actuators: usize) -> Vec<f64> {
        let thought_hv = encode_observation(observation, self.hv_dim);
        let result = self.bridge.step(&thought_hv, self.dt, gain);

        // Translate the bridge's scalar response into a per-actuator vector.
        //
        // The effective per-joint authority is the product of:
        //   - the bridge's post-step safety gain (its own NRC tier),
        //   - |control_effort| (how hard the bridge believes it is pushing),
        // then capped by `gain` (the caller's cap) so we never exceed the
        // Symtropy-side NRC ceiling. Uniform across actuators.
        let bridge_gain = result.safety_level.motor_gain() as f64;
        let effort = (result.control_effort as f64).abs();
        let per_joint = (bridge_gain * effort).min(gain).max(0.0);

        vec![per_joint; num_actuators]
    }
}

/// Encode a short observation vector into a dim-wide [`ContinuousHV`].
///
/// Empty observations produce a zero HV. Non-empty observations are tiled
/// into the HV (each slot `i` gets `obs[i % obs.len()]`), which is sufficient
/// for platforms whose encoders re-embed the thought via their own HDC
/// projection. Platforms that require a specific semantic encoding should
/// compose their own planner rather than relying on this default.
pub fn encode_observation(obs: &[f64], hv_dim: usize) -> ContinuousHV {
    if obs.is_empty() {
        return ContinuousHV::zero(hv_dim);
    }
    let mut values = Vec::with_capacity(hv_dim);
    for i in 0..hv_dim {
        values.push(obs[i % obs.len()] as f32);
    }
    ContinuousHV::from_vec(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment::{
        AgentIdentity, EmbodimentPlatform, EmbodimentResult, EmbodimentTelemetry,
        MotorSafetyLevel,
    };

    /// Minimal EmbodimentBridge for testing the adapter without pulling a
    /// real platform crate. Records every call to step() and returns a
    /// configurable EmbodimentResult.
    struct MockBridge {
        step_count: usize,
        last_thought_norm: f32,
        last_dt: f32,
        last_phi: f64,
        result_safety: MotorSafetyLevel,
        result_effort: f32,
        num_actuators: usize,
        safety_override: Option<MotorSafetyLevel>,
    }

    impl MockBridge {
        fn new() -> Self {
            Self {
                step_count: 0,
                last_thought_norm: 0.0,
                last_dt: 0.0,
                last_phi: 0.0,
                result_safety: MotorSafetyLevel::Green,
                result_effort: 1.0,
                num_actuators: 21,
                safety_override: None,
            }
        }
    }

    impl EmbodimentBridge for MockBridge {
        fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
            self.step_count += 1;
            self.last_thought_norm = thought_hv
                .values
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            self.last_dt = dt;
            self.last_phi = phi;
            EmbodimentResult {
                num_actuators: self.num_actuators,
                control_effort: self.result_effort,
                success: true,
                prediction_error: 0.1,
                safety_level: self.result_safety,
                epistemic_grounding: 0,
                observation_confidence: 0.9,
            }
        }

        fn encode_perception(&mut self) -> ContinuousHV {
            ContinuousHV::zero(DEFAULT_HV_DIM)
        }

        fn reset(&mut self) {
            self.step_count = 0;
        }

        fn safety_level(&self) -> MotorSafetyLevel {
            self.safety_override.unwrap_or(self.result_safety)
        }

        fn set_safety_override(&mut self, level: MotorSafetyLevel) {
            self.safety_override = Some(level);
        }

        fn clear_safety_override(&mut self) {
            self.safety_override = None;
        }

        fn platform(&self) -> EmbodimentPlatform {
            EmbodimentPlatform::None
        }

        fn num_actuators(&self) -> usize {
            self.num_actuators
        }

        fn total_steps(&self) -> usize {
            self.step_count
        }

        fn telemetry(&self) -> EmbodimentTelemetry {
            EmbodimentTelemetry {
                total_steps: self.step_count as u64,
                ..Default::default()
            }
        }

        fn agent_identity(&self) -> AgentIdentity {
            AgentIdentity::new("mock")
        }
    }

    fn make_planner() -> EmbodimentPlanner {
        EmbodimentPlanner::new(Box::new(MockBridge::new()))
    }

    #[test]
    fn default_dt_is_60hz() {
        let p = make_planner();
        assert!((p.dt() - 1.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn with_dt_overrides_timestep() {
        let p = EmbodimentPlanner::new(Box::new(MockBridge::new())).with_dt(0.01);
        assert!((p.dt() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn plan_returns_vec_of_requested_length() {
        let mut p = make_planner();
        let cmds = p.plan(&[0.5], 1.0, 21);
        assert_eq!(cmds.len(), 21);
    }

    #[test]
    fn plan_invokes_bridge_step_once() {
        let mut p = make_planner();
        let _ = p.plan(&[0.5, 0.3], 1.0, 21);
        assert_eq!(p.bridge().total_steps(), 1);
        let _ = p.plan(&[0.5, 0.3], 1.0, 21);
        assert_eq!(p.bridge().total_steps(), 2);
    }

    #[test]
    fn plan_forwards_gain_as_phi() {
        let mut p = make_planner();
        let _ = p.plan(&[0.5], 0.42, 21);
        // We can't see phi from the trait, but we can observe that a lower
        // gain cap is respected in the output (Green × 1.0 = 1.0, capped to
        // 0.42).
        let cmds = p.plan(&[0.5], 0.42, 4);
        for c in &cmds {
            assert!(
                *c <= 0.42 + 1e-12,
                "gain cap violated: {c} > 0.42"
            );
        }
    }

    #[test]
    fn red_safety_zeros_output() {
        let mut bridge = MockBridge::new();
        bridge.result_safety = MotorSafetyLevel::Red;
        let mut p = EmbodimentPlanner::new(Box::new(bridge));
        let cmds = p.plan(&[0.5], 1.0, 21);
        for c in &cmds {
            assert_eq!(*c, 0.0);
        }
    }

    #[test]
    fn yellow_safety_scales_output() {
        let mut bridge = MockBridge::new();
        bridge.result_safety = MotorSafetyLevel::Yellow;
        bridge.result_effort = 1.0;
        let mut p = EmbodimentPlanner::new(Box::new(bridge));
        let cmds = p.plan(&[0.5], 1.0, 4);
        // Yellow gain is 0.6, effort is 1.0, incoming cap is 1.0 → 0.6
        for c in &cmds {
            assert!((c - 0.6).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_actuators_yields_empty_vec() {
        let mut p = make_planner();
        let cmds = p.plan(&[0.5], 1.0, 0);
        assert!(cmds.is_empty());
    }

    #[test]
    fn empty_observation_produces_zero_hv_and_still_steps() {
        let mut p = make_planner();
        let _ = p.plan(&[], 1.0, 21);
        assert_eq!(p.bridge().total_steps(), 1);
    }

    #[test]
    fn reset_propagates_to_bridge() {
        let mut p = make_planner();
        let _ = p.plan(&[0.5], 1.0, 21);
        assert_eq!(p.bridge().total_steps(), 1);
        p.bridge_mut().reset();
        assert_eq!(p.bridge().total_steps(), 0);
    }

    #[test]
    fn safety_override_plumbs_through() {
        let mut p = make_planner();
        p.bridge_mut().set_safety_override(MotorSafetyLevel::Orange);
        assert_eq!(p.bridge().safety_level(), MotorSafetyLevel::Orange);
        p.bridge_mut().clear_safety_override();
        assert_eq!(p.bridge().safety_level(), MotorSafetyLevel::Green);
    }

    #[test]
    fn encode_observation_empty_is_zero() {
        let hv = encode_observation(&[], 64);
        assert_eq!(hv.values.len(), 64);
        for v in &hv.values {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn encode_observation_tiles_short_input() {
        let hv = encode_observation(&[0.25, -0.5], 8);
        assert_eq!(
            hv.values,
            vec![0.25, -0.5, 0.25, -0.5, 0.25, -0.5, 0.25, -0.5]
        );
    }

    #[test]
    fn encode_observation_respects_hv_dim() {
        let hv = encode_observation(&[1.0], 128);
        assert_eq!(hv.values.len(), 128);
    }

    #[test]
    fn control_effort_modulates_output() {
        let mut bridge = MockBridge::new();
        bridge.result_safety = MotorSafetyLevel::Green;
        bridge.result_effort = 0.5;
        let mut p = EmbodimentPlanner::new(Box::new(bridge));
        let cmds = p.plan(&[0.5], 1.0, 4);
        // Green gain = 1.0, effort = 0.5, cap = 1.0 → 0.5
        for c in &cmds {
            assert!((c - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn gain_cap_binds_even_when_bridge_authority_is_full() {
        let mut bridge = MockBridge::new();
        bridge.result_safety = MotorSafetyLevel::Green;
        bridge.result_effort = 1.0;
        let mut p = EmbodimentPlanner::new(Box::new(bridge));
        let cmds = p.plan(&[0.5], 0.3, 4);
        // Bridge authority = 1.0 × 1.0 = 1.0, but cap is 0.3
        for c in &cmds {
            assert!((c - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn negative_control_effort_is_abs_valued() {
        let mut bridge = MockBridge::new();
        bridge.result_safety = MotorSafetyLevel::Green;
        bridge.result_effort = -0.8;
        let mut p = EmbodimentPlanner::new(Box::new(bridge));
        let cmds = p.plan(&[0.5], 1.0, 4);
        // |-0.8| = 0.8
        for c in &cmds {
            assert!((c - 0.8).abs() < 1e-6);
        }
    }
}
