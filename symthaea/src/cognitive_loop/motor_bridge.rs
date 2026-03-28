// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Motor bridge: cognitive loop → embodied motor output.
//!
//! Defines the [`EmbodimentBridge`] trait for platform-agnostic motor control
//! and the [`MotorBridge`] (humanoid) concrete implementation.
//!
//! The bridge is bidirectional:
//! - **Motor output**: thought HV → controller → motor command → physics
//! - **Perception input**: body state → HDC encode → proprioceptive HV → next cycle
//!
//! ```text
//! CognitiveLoopService
//!   → current_thought (16384D ContinuousHV)
//!   → EmbodimentBridge::step()
//!   → Platform-specific controller → motor commands → physics sim
//!   → EmbodimentBridge::encode_perception()
//!   → proprioceptive HV → next cognitive cycle input
//! ```

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::humanoid::{
    HumanoidCommand, HumanoidConfig, HumanoidController, HumanoidHdcEncoder,
    HumanoidPhysicsSimulator, HumanoidState, SimpleHumanoidSimulator,
};

// ── Platform-agnostic embodiment trait ──────────────────────────────────────

/// NRC-inspired safety level for motor output gating.
///
/// Maps to consciousness level (Phi) thresholds:
/// - Green (Phi > 0.6): full authority
/// - Yellow (0.3–0.6): reduced speed/force
/// - Orange (0.1–0.3): retreat to safe pose
/// - Red (< 0.1): emergency stop, power cut
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MotorSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

impl MotorSafetyLevel {
    /// Determine safety level from current Phi value.
    pub fn from_phi(phi: f64) -> Self {
        if phi > 0.6 {
            Self::Green
        } else if phi > 0.3 {
            Self::Yellow
        } else if phi > 0.1 {
            Self::Orange
        } else {
            Self::Red
        }
    }

    /// Motor gain multiplier for this safety level.
    pub fn motor_gain(&self) -> f32 {
        match self {
            Self::Green => 1.0,
            Self::Yellow => 0.6,
            Self::Orange => 0.3,
            Self::Red => 0.0,
        }
    }
}

/// Result of a single embodiment step.
#[derive(Debug, Clone)]
pub struct EmbodimentResult {
    pub num_actuators: usize,
    pub control_effort: f32,
    pub success: bool,
    pub prediction_error: f32,
    pub safety_level: MotorSafetyLevel,
}

/// Telemetry from the embodiment subsystem, exposed in CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbodimentTelemetry {
    pub total_steps: u64,
    pub control_effort: f32,
    pub prediction_error: f32,
    pub safety_level: String,
    pub platform: String,
    pub num_actuators: usize,
}

/// Which embodiment platform to use.
///
/// Each variant (except `None` and `CareProvider`) maps to a robotics crate
/// behind a feature flag. `CareProvider` is virtual-only — it has no
/// physical embodiment and dispatches `None` in the constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbodimentPlatform {
    #[default]
    None,
    Humanoid,
    Quadrotor,
    Vehicle,
    Helicopter,
    Auv,
    Manipulator,
    /// Virtual caregiver agent — no physical embodiment.
    CareProvider,
}

/// Platform-agnostic embodiment bridge trait.
///
/// Any robotics platform that wants to close the proprioceptive loop
/// with the cognitive loop must implement this trait.
pub trait EmbodimentBridge: Send + Sync {
    /// Translate thought → motor commands, step physics, return result.
    /// `phi` is the current consciousness level for safety gating.
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult;

    /// Encode the current body state into a proprioceptive 16,384D HV.
    fn encode_perception(&mut self) -> ContinuousHV;

    /// Reset body to default state.
    fn reset(&mut self);

    /// Current safety level.
    fn safety_level(&self) -> MotorSafetyLevel;

    /// Override safety level from SafetyAgent. Merged with phi via max().
    fn set_safety_override(&mut self, level: MotorSafetyLevel);

    /// Clear external safety override.
    fn clear_safety_override(&mut self);

    /// Platform identifier.
    fn platform(&self) -> EmbodimentPlatform;

    /// Number of actuators.
    fn num_actuators(&self) -> usize;

    /// Total steps executed.
    fn total_steps(&self) -> usize;

    /// Get telemetry for CycleMetadata.
    fn telemetry(&self) -> EmbodimentTelemetry;
}

// ── Humanoid implementation ─────────────────────────────────────────────────

/// Bridge between cognitive loop and humanoid motor system.
///
/// Implements [`EmbodimentBridge`] for the 21-DOF bipedal humanoid.
pub struct MotorBridge {
    controller: HumanoidController,
    simulator: SimpleHumanoidSimulator,
    encoder: HumanoidHdcEncoder,
    config: HumanoidConfig,
    last_command: HumanoidCommand,
    last_perception: Option<ContinuousHV>,
    dt: f64,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl MotorBridge {
    /// Create a new motor bridge with default configuration.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = HumanoidConfig::default();
        let controller = HumanoidController::new(genesis, &config);
        let simulator = SimpleHumanoidSimulator::new();
        let encoder = HumanoidHdcEncoder::new(genesis, 32);
        let dt = config.physics_dt();

        Self {
            controller,
            simulator,
            encoder,
            config,
            last_command: HumanoidCommand::zero(),
            last_perception: None,
            dt,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
        }
    }

    /// Create with a pre-trained controller.
    pub fn from_controller(controller: HumanoidController, config: HumanoidConfig) -> Self {
        let simulator = SimpleHumanoidSimulator::new();
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let encoder = HumanoidHdcEncoder::new(&genesis, 32);
        let dt = config.physics_dt();

        Self {
            controller,
            simulator,
            encoder,
            config,
            last_command: HumanoidCommand::zero(),
            last_perception: None,
            dt,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
        }
    }

    /// Direct step (without consciousness gating). For standalone use.
    pub fn step_direct(&mut self, thought_hv: &ContinuousHV) -> (HumanoidCommand, ContinuousHV) {
        let command = self.controller.forward(thought_hv, self.dt as f32);
        self.simulator.step(&command, self.dt);
        let perception = self.encoder.encode(self.simulator.state());
        self.last_command = command.clone();
        self.last_perception = Some(perception.clone());
        self.total_steps += 1;
        (command, perception)
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let perception = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(perception.clone());
        perception
    }

    pub fn last_perception(&self) -> Option<&ContinuousHV> {
        self.last_perception.as_ref()
    }

    pub fn body_state(&self) -> &HumanoidState {
        self.simulator.state()
    }

    pub fn last_command(&self) -> &HumanoidCommand {
        &self.last_command
    }

    pub fn controller(&self) -> &HumanoidController {
        &self.controller
    }

    pub fn controller_mut(&mut self) -> &mut HumanoidController {
        &mut self.controller
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    #[cfg(feature = "hal")]
    pub fn write_to_hardware<I: symthaea_hal::I2cBus>(
        &self,
        safety: &mut symthaea_hal::SafetyInterlock,
        servo: &mut symthaea_hal::ServoOutput<I>,
    ) -> Result<(), symthaea_hal::HalError> {
        let safe_cmd = safety.filter_command(&self.last_command)?;
        servo.apply(&safe_cmd)
    }

    #[cfg(feature = "hal")]
    pub fn step_from_readings(
        &mut self,
        _readings: &[Option<Vec<f32>>],
        thought_hv: &ContinuousHV,
    ) -> HumanoidCommand {
        let (command, _) = self.step_direct(thought_hv);
        command
    }

    #[cfg(feature = "hal")]
    pub fn hal_callback<'a>(
        &'a mut self,
        thought_hv: &'a ContinuousHV,
    ) -> impl FnMut(&[Option<Vec<f32>>]) -> HumanoidCommand + 'a {
        move |readings| self.step_from_readings(readings, thought_hv)
    }
}

impl EmbodimentBridge for MotorBridge {
    fn step(&mut self, thought_hv: &ContinuousHV, _dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        let gain = self.current_safety.motor_gain();

        let mut command = self.controller.forward(thought_hv, self.dt as f32);
        if gain < 1.0 {
            for t in command.torques.iter_mut() {
                *t *= gain;
            }
        }

        let effort: f32 =
            command.torques.iter().map(|t| t.abs()).sum::<f32>() / command.torques.len() as f32;
        self.last_control_effort = effort;

        self.simulator.step(&command, self.dt);
        let perception = self.encoder.encode(self.simulator.state());

        let pred_error = if let Some(ref prev) = self.last_perception {
            let sim = perception.similarity(prev);
            let pe: f32 = (1.0 - sim.max(0.0)).min(1.0);
            pe
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;

        self.last_command = command;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        let success = self.simulator.state().head_height.is_finite();

        EmbodimentResult {
            num_actuators: 21,
            control_effort: effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
        }
    }

    fn encode_perception(&mut self) -> ContinuousHV {
        MotorBridge::encode_perception(self)
    }

    fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.encoder.reset();
        self.last_command = HumanoidCommand::zero();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
    }

    fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Humanoid
    }

    fn num_actuators(&self) -> usize {
        21
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "humanoid".to_string(),
            num_actuators: 21,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod embodiment_tests {
    use super::*;

    fn make_bridge() -> MotorBridge {
        let genesis = GenesisSeed::from_phrase("test-embodiment");
        MotorBridge::new(&genesis)
    }

    #[test]
    fn test_embodiment_bridge_step() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        let result = bridge.step(&hv, 0.025, 0.7);
        assert!(result.success);
        assert_eq!(result.num_actuators, 21);
        assert!(result.control_effort >= 0.0);
        assert_eq!(result.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_safety_gating_reduces_torque() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);

        let green = bridge.step(&hv, 0.025, 0.8);
        bridge.reset();
        let red = bridge.step(&hv, 0.025, 0.05);

        assert_eq!(red.safety_level, MotorSafetyLevel::Red);
        assert_eq!(red.control_effort, 0.0);
        assert!(green.control_effort >= red.control_effort);
    }

    #[test]
    fn test_safety_level_from_phi() {
        assert_eq!(MotorSafetyLevel::from_phi(0.8), MotorSafetyLevel::Green);
        assert_eq!(MotorSafetyLevel::from_phi(0.5), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.2), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.05), MotorSafetyLevel::Red);
    }

    #[test]
    fn test_telemetry_populated() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        let _ = bridge.step(&hv, 0.025, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "humanoid");
        assert_eq!(t.num_actuators, 21);
    }

    #[test]
    fn test_platform_enum() {
        let bridge = make_bridge();
        assert_eq!(bridge.platform(), EmbodimentPlatform::Humanoid);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        let _ = EmbodimentBridge::step(&mut bridge, &hv, 0.025, 0.7);
        assert_eq!(EmbodimentBridge::total_steps(&bridge), 1);
        EmbodimentBridge::reset(&mut bridge);
        assert_eq!(EmbodimentBridge::total_steps(&bridge), 0);
    }
}

#[cfg(test)]
#[cfg(feature = "hal")]
mod hal_tests {
    use super::*;

    fn make_bridge() -> MotorBridge {
        let genesis = GenesisSeed::from_phrase("test");
        MotorBridge::new(&genesis)
    }

    #[test]
    fn test_step_from_readings_basic() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::zero(16384);
        let cmd = bridge.step_from_readings(&[], &hv);
        for &t in &cmd.torques {
            assert!(t.is_finite());
        }
    }

    #[test]
    fn test_step_from_readings_increments_steps() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::zero(16384);
        assert_eq!(bridge.total_steps(), 0);
        let _ = bridge.step_from_readings(&[], &hv);
        assert_eq!(bridge.total_steps(), 1);
    }

    #[test]
    fn test_hal_callback_works() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::zero(16384);
        let mut cb = bridge.hal_callback(&hv);
        let cmd = cb(&[]);
        for &t in &cmd.torques {
            assert!(t.is_finite());
        }
    }
}
