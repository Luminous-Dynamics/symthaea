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
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, GroundingEstimator,
    MoralGateInput, MotorSafetyLevel, grounding_from_prediction_error, grounding_label,
};

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
        }
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
        if gain < 1.0 {
            for t in &mut cmd.joint_torques {
                *t *= gain;
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
        assert_eq!(r.control_effort, 0.0, "Blocked should zero motor output");
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
}
