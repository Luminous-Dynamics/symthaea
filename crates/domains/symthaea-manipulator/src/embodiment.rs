// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for 7-DOF industrial manipulator.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::ManipulatorController;
use crate::encoder::ManipulatorHdcEncoder;
use crate::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use crate::types::ManipulatorConfig;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    GroundingEstimator, MoralGateInput, MotorSafetyLevel, SafeFallback, GROUNDING_SENSORIMOTOR,
};

/// Emergency fallback posture for the manipulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManipulatorFallbackStage {
    /// Hold current pose via per-joint gravity-compensation torque; gripper frozen.
    GravityHold,
}

/// Manipulator embodiment bridge.
pub struct ManipulatorEmbodiment {
    controller: ManipulatorController,
    simulator: SimpleManipulatorSimulator,
    encoder: ManipulatorHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    moral_safety: Option<MotorSafetyLevel>,
    grounding: GroundingEstimator,
    last_grounding: u8,
    fallback_stage: ManipulatorFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl ManipulatorEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ManipulatorConfig::default();
        Self {
            controller: ManipulatorController::new(genesis, &config),
            simulator: SimpleManipulatorSimulator::new(),
            encoder: ManipulatorHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            moral_safety: None,
            grounding: GroundingEstimator::new(),
            last_grounding: GROUNDING_SENSORIMOTOR,
            fallback_stage: ManipulatorFallbackStage::GravityHold,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Current SafeFallback stage (always `GravityHold` for this platform).
    pub fn fallback_stage(&self) -> ManipulatorFallbackStage {
        self.fallback_stage
    }

    /// Override the commanded torques with a gravity-compensation hold and
    /// freeze the gripper at its current opening. Executes at full authority
    /// (not scaled by motor_gain) per the SafeFallback contract.
    fn apply_gravity_hold(&self, cmd: &mut crate::types::ManipulatorCommand) {
        cmd.joint_torques = self.simulator.gravity_compensation_torques();
        cmd.gripper = self.simulator.state().gripper_opening as f32;
    }

    /// Apply moral gate from the ethics engine.
    ///
    /// For the manipulator (force on humans/objects), ethics gating is critical:
    /// - Blocked → Red (power cut)
    /// - Caution → cap at Yellow (reduced force)
    /// - consent_violation → Orange (retreat to home)
    /// - ahimsa_violated → Red (emergency stop)
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED {
            self.moral_safety = Some(MotorSafetyLevel::Red);
        } else if gate.consent_violation {
            self.moral_safety = Some(MotorSafetyLevel::Orange);
        } else if gate.verdict == MoralGateInput::VERDICT_CAUTION {
            self.moral_safety = Some(MotorSafetyLevel::Yellow);
        } else {
            self.moral_safety = None;
        }
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        // Effective safety = max(phi_safety, safety_override, moral_safety)
        // Higher enum variant = stricter (Green < Yellow < Orange < Red)
        self.current_safety = phi_level;
        if let Some(override_level) = self.safety_override {
            self.current_safety = self.current_safety.max(override_level);
        }
        if let Some(moral_level) = self.moral_safety {
            self.current_safety = self.current_safety.max(moral_level);
        }
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── SafeFallback: GravityHold at Red ─────────────────────────
        // "No force" for a loaded arm means "drop whatever it's holding" —
        // gain=0.0 alone is NOT safe here. Hold the current pose against
        // gravity instead, at full torque authority (not gain-scaled).
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_gravity_hold(&mut cmd);
        } else {
            self.fallback_stage = ManipulatorFallbackStage::GravityHold;
            self.fallback_cycles_in_stage = 0;
            if gain < 1.0 {
                for t in &mut cmd.joint_torques {
                    *t *= gain;
                }
            }
        }

        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);

        let perception = self.encoder.encode(self.simulator.state());

        let pred_error = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        // Compute epistemic grounding from prediction error trend.
        // Manipulator has no swarm peers, so social grounding is not possible.
        self.last_grounding = self.grounding.estimate(pred_error, None);

        let success = self.simulator.state().is_finite();

        EmbodimentResult {
            num_actuators: 8, // 7 joints + 1 gripper
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: self.last_grounding,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(p.clone());
        p
    }

    pub fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.grounding.reset();
        self.last_grounding = GROUNDING_SENSORIMOTOR;
        self.fallback_stage = ManipulatorFallbackStage::GravityHold;
        self.fallback_cycles_in_stage = 0;
    }

    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "manipulator".to_string(),
            num_actuators: 8,
            epistemic_grounding: grounding_label(self.last_grounding).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: self.platform_telemetry_bytes(),
        }
    }

    /// Serialize joint angles and end-effector force as JSON bytes.
    pub fn platform_telemetry_bytes(&self) -> Vec<u8> {
        let state = self.simulator.state();
        serde_json::to_vec(&serde_json::json!({
            "joint_angles": state.joint_angles,
            "ee_force": state.end_effector_force,
            "gripper": state.gripper_opening,
        }))
        .unwrap_or_default()
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for ManipulatorEmbodiment {
    fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.step(hv, dt, phi)
    }
    fn encode_perception(&mut self) -> ContinuousHV {
        self.encode_perception()
    }
    fn reset(&mut self) {
        self.reset()
    }
    fn safety_level(&self) -> MotorSafetyLevel {
        self.safety_level()
    }
    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.set_safety_override(level)
    }
    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Manipulator
    }
    fn num_actuators(&self) -> usize {
        8
    }
    fn total_steps(&self) -> usize {
        self.total_steps()
    }
    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
    fn apply_moral_gate(&mut self, gate: symthaea_core::embodiment::MoralGateInput) {
        self.apply_moral_gate(gate)
    }
    fn platform_telemetry_bytes(&self) -> Vec<u8> {
        self.platform_telemetry_bytes()
    }
}

impl SafeFallback for ManipulatorEmbodiment {
    fn platform_name(&self) -> &'static str {
        "manipulator"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: grasping, in-contact (per SafeFallback trait's own scale)
    }
    fn safe_fallback_description(&self) -> &'static str {
        "GravityHold: per-joint gravity-compensation torque holds current pose; gripper frozen"
    }
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.7);
        assert!(r.success);
        assert_eq!(r.num_actuators, 8);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_perception() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "manipulator");
    }

    #[test]
    fn test_moral_gate_blocked_forces_red() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_BLOCKED,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9); // High phi, but blocked
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // NOTE: this used to assert control_effort == 0.0, but that was the
        // bug this fix closes: "no force" is not safe for a loaded arm — it
        // means "drop whatever it's holding". Red now commands GravityHold
        // (a deliberate, non-zero, purposeful hold torque) instead of a
        // passive zero. See test_red_triggers_gravity_hold below for the
        // behavior this actually asserts.
        assert!(r.control_effort.is_finite());
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_moral_gate_consent_forces_orange() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
    }

    #[test]
    fn test_moral_gate_caution_caps_yellow() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_CAUTION,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9); // Phi says Green, ethics caps at Yellow
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
    }

    #[test]
    fn test_moral_gate_safe_no_effect() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_reset() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_red_triggers_gravity_hold_stage() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.05);
        assert_eq!(
            bridge.fallback_stage(),
            ManipulatorFallbackStage::GravityHold
        );
    }

    #[test]
    fn test_red_gravity_hold_commands_nonzero_torque_not_passive_zero() {
        // The core safety bug: at Red, gain=0.0 alone means "drop the load".
        // GravityHold must command a real, non-zero holding torque derived
        // from the arm's own gravity model — not simply zero everything.
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert!(
            r.control_effort > 1e-4,
            "GravityHold must command non-zero holding torque, got {}",
            r.control_effort
        );
    }

    #[test]
    fn test_red_gravity_hold_matches_simulator_gravity_compensation() {
        // The commanded torque at Red must equal the simulator's own
        // gravity-compensation computation for the pre-step pose (full
        // authority, not scaled by motor_gain).
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let expected = bridge.simulator.gravity_compensation_torques();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.05);
        assert_eq!(bridge.last_control_effort, {
            expected.iter().map(|t| t.abs()).sum::<f32>() / expected.len() as f32
        });
    }

    #[test]
    fn test_red_gravity_hold_freezes_gripper() {
        // Gripper must stay at whatever opening it was, not snap to a
        // default — freezing state is part of "hold current pose".
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        // First step at Green to let the gripper move away from its default.
        bridge.step(&hv, 0.002, 0.9);
        let opening_before_red = bridge.simulator.state().gripper_opening;
        bridge.step(&hv, 0.002, 0.05); // now Red
        let opening_after_red = bridge.simulator.state().gripper_opening;
        assert!(
            (opening_after_red - opening_before_red).abs() < 1e-6,
            "gripper should freeze at Red, went from {} to {}",
            opening_before_red,
            opening_after_red
        );
    }

    #[test]
    fn test_safe_fallback_trait_impl() {
        let bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(bridge.platform_name(), "manipulator");
        assert_eq!(bridge.safe_fallback_priority(), 5);
        assert_eq!(bridge.safe_fallback_latency_cycles(), 1);
        assert!(bridge.safe_fallback_description().contains("GravityHold"));
    }

    #[test]
    fn test_safe_fallback_active_iff_red_or_orange() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.9); // Green
        assert!(!bridge.safe_fallback_active());
        bridge.step(&hv, 0.002, 0.05); // Red
        assert!(bridge.safe_fallback_active());
    }

    #[test]
    fn test_non_red_tiers_still_scale_by_gain_not_gravity_hold() {
        // Yellow/Orange must retain graduated authority (scaled command),
        // not jump straight to the Red fallback.
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.5); // Yellow, gain 0.6
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
        assert_eq!(bridge.fallback_cycles_in_stage, 0);
    }
}
