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

#[cfg(feature = "humanoid")]
use symthaea_core::genesis::GenesisSeed;
#[cfg(feature = "humanoid")]
use symthaea_core::hdc::ContinuousHV;

#[cfg(feature = "humanoid")]
use crate::humanoid::{
    HumanoidCommand, HumanoidConfig, HumanoidController, HumanoidHdcEncoder,
    HumanoidPhysicsSimulator, HumanoidState, SimpleHumanoidSimulator,
};

// ── Re-export canonical embodiment types from symthaea-core ────────────────
// All platform crates also import from symthaea-core, so there is now a
// single Rust type across the entire workspace — no more E0308 mismatches.

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentBridge, EmbodimentPlatform,
    EmbodimentResult, EmbodimentTelemetry, MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};

// ── Humanoid implementation ─────────────────────────────────────────────────
// Everything below is the humanoid-specific MotorBridge. Other platforms
// implement EmbodimentBridge directly in their crate.

/// Bridge between cognitive loop and humanoid motor system.
///
/// Implements [`EmbodimentBridge`] for the 21-DOF bipedal humanoid.
///
/// # Deprecation notice (2026-04-19)
///
/// **Use [`symthaea_humanoid::embodiment::HumanoidEmbodiment`]
/// instead.** That type lives in the `symthaea-humanoid` sub-crate
/// alongside the other nine platforms' `*Embodiment` structs, matching
/// a uniform architectural pattern (commit `1a85fce8c8`). Production
/// construction paths in `constructor.rs` and
/// `accessors/system.rs` both switched to `HumanoidEmbodiment` in
/// commit `[THIS COMMIT]`.
///
/// `MotorBridge` is kept for backwards-compatibility — downstream
/// code that already references it will keep working — but it
/// receives no new features. All future humanoid-specific additions
/// (moral-gate paths, safe fallbacks, telemetry fields) should land
/// in `HumanoidEmbodiment` only.
///
/// This struct does not carry `#[deprecated]` because in-crate tests
/// and external downstream references still use it and the
/// deprecation warnings would be noisy without producing any signal
/// the next reader doesn't already have from this doc-comment.
#[cfg(feature = "humanoid")]
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
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

#[cfg(feature = "humanoid")]
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
            moral_safety: None,
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
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
        }
    }

    /// Apply moral gate from ethics engine.
    /// A humanoid can step on / collide with beings — ahimsa must force Red,
    /// consent violation forces Orange, caution forces Yellow.
    pub fn apply_moral_gate(&mut self, gate: symthaea_core::embodiment::MoralGateInput) {
        use symthaea_core::embodiment::MoralGateInput;
        self.moral_safety =
            if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED {
                Some(MotorSafetyLevel::Red)
            } else if gate.consent_violation {
                Some(MotorSafetyLevel::Orange)
            } else if gate.verdict == MoralGateInput::VERDICT_CAUTION {
                Some(MotorSafetyLevel::Yellow)
            } else {
                None
            };
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

#[cfg(feature = "humanoid")]
impl EmbodimentBridge for MotorBridge {
    fn step(&mut self, thought_hv: &ContinuousHV, _dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        // Ethics gate max-composes on top of phi + override.
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
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
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
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
        self.moral_safety = None;
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
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }

    fn apply_moral_gate(&mut self, gate: symthaea_core::embodiment::MoralGateInput) {
        MotorBridge::apply_moral_gate(self, gate)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "humanoid"))]
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

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        use symthaea_core::embodiment::MoralGateInput;
        let mut bridge = make_bridge();
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        // Phi 0.9 → Green normally; ahimsa forces Red.
        let r = bridge.step(&hv, 0.025, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red regardless of phi"
        );
        assert_eq!(r.control_effort, 0.0, "Red tier must zero torques");
    }

    #[test]
    fn test_moral_gate_consent_violation_forces_orange() {
        use symthaea_core::embodiment::MoralGateInput;
        let mut bridge = make_bridge();
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Orange,
            "consent violation → Orange"
        );
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
