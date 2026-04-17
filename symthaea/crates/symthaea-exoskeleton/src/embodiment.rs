// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use crate::controller::ExoskeletonController;
use crate::encoder::ExoskeletonHdcEncoder;
use crate::simulator::{ExoskeletonPhysicsSimulator, SimpleExoskeletonSimulator};
use crate::types::{AssistanceMode, ExoskeletonConfig, NUM_ACTUATORS};
pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MoralGateInput, MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};

pub struct ExoskeletonEmbodiment {
    controller: ExoskeletonController, simulator: SimpleExoskeletonSimulator,
    encoder: ExoskeletonHdcEncoder, last_perception: Option<ContinuousHV>,
    total_steps: usize, current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32, last_prediction_error: f32,
    assistance_mode: AssistanceMode,
}

impl ExoskeletonEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ExoskeletonConfig::default();
        Self {
            controller: ExoskeletonController::new(genesis, &config),
            simulator: SimpleExoskeletonSimulator::new(),
            encoder: ExoskeletonHdcEncoder::new(genesis, 32),
            last_perception: None, total_steps: 0,
            current_safety: MotorSafetyLevel::Green, safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0, last_prediction_error: 0.0,
            assistance_mode: AssistanceMode::Responsive,
        }
    }
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) { self.safety_override = Some(level); }
    pub fn clear_safety_override(&mut self) { self.safety_override = None; }
    pub fn assistance_mode(&self) -> AssistanceMode { self.assistance_mode }

    /// Apply moral gate from ethics engine.
    /// Exoskeleton applies force directly to a human body — ahimsa forces backdrivable
    /// (Red gain=0.0 zeros assistive torques + stiffness), consent violation forces Orange retreat.
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.moral_safety = if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED { Some(MotorSafetyLevel::Red) }
            else if gate.consent_violation { Some(MotorSafetyLevel::Orange) }
            else if gate.verdict == MoralGateInput::VERDICT_CAUTION { Some(MotorSafetyLevel::Yellow) }
            else { None };
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.assistance_mode = AssistanceMode::from_phi(phi);
        let tf = self.assistance_mode.torque_factor();
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = phi_level;
        if let Some(ov) = self.safety_override { self.current_safety = self.current_safety.max(ov); }
        if let Some(m) = self.moral_safety { self.current_safety = self.current_safety.max(m); }
        let gain = self.current_safety.motor_gain();
        let mut cmd = self.controller.forward(thought_hv, dt);
        for t in &mut cmd.joint_torques { *t *= tf * gain; }
        cmd.stiffness_gain *= self.assistance_mode.stiffness_factor() as f32;
        if gain == 0.0 { cmd.stiffness_gain = 0.0; cmd.damping_gain = 0.0; }
        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);
        let perception = self.encoder.encode(self.simulator.state());
        let pred_error = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0) } else { 0.0f32 };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;
        EmbodimentResult {
            num_actuators: NUM_ACTUATORS, control_effort: self.last_control_effort,
            success: self.simulator.state().is_finite(), prediction_error: pred_error,
            safety_level: self.current_safety, epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.encoder.encode(self.simulator.state()); self.last_perception = Some(p.clone()); p
    }
    pub fn reset(&mut self) {
        self.simulator.reset(); self.controller.reset(); self.encoder.reset();
        self.last_perception = None; self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green; self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0; self.last_prediction_error = 0.0;
        self.assistance_mode = AssistanceMode::Responsive;
    }
    pub fn safety_level(&self) -> MotorSafetyLevel { self.current_safety }
    pub fn total_steps(&self) -> usize { self.total_steps }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64, control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "exoskeleton".to_string(), num_actuators: NUM_ACTUATORS,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for ExoskeletonEmbodiment {
    fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult { self.step(hv, dt, phi) }
    fn encode_perception(&mut self) -> ContinuousHV { self.encode_perception() }
    fn reset(&mut self) { self.reset() }
    fn safety_level(&self) -> MotorSafetyLevel { self.safety_level() }
    fn set_safety_override(&mut self, level: MotorSafetyLevel) { self.set_safety_override(level) }
    fn clear_safety_override(&mut self) { self.clear_safety_override() }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform { symthaea_core::embodiment::EmbodimentPlatform::Exoskeleton }
    fn num_actuators(&self) -> usize { NUM_ACTUATORS }
    fn total_steps(&self) -> usize { self.total_steps() }
    fn telemetry(&self) -> EmbodimentTelemetry { self.telemetry() }
    fn apply_moral_gate(&mut self, gate: MoralGateInput) { self.apply_moral_gate(gate) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_step() {
        let mut b = ExoskeletonEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.7);
        assert!(r.success); assert_eq!(r.num_actuators, 6);
    }
    #[test]
    fn test_red_backdrivable() {
        let mut b = ExoskeletonEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert!(r.control_effort < 0.01);
    }
    #[test]
    fn test_extended() {
        let mut b = ExoskeletonEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for i in 0..500 { let phi = 0.3 + 0.4 * ((i as f64 * 0.1).sin()); assert!(b.step(&hv, 0.005, phi).success); }
    }
    #[test]
    fn test_moral_gate_ahimsa_zeros_torques() {
        // Critical: exoskeleton strapped to human body. Ahimsa violation (e.g.,
        // commanded to dangerous posture) must make device backdrivable.
        let mut b = ExoskeletonEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput { verdict: MoralGateInput::VERDICT_SAFE, consent_violation: false, ahimsa_violated: true });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red, "ahimsa must force backdrivable Red regardless of phi");
        assert!(r.control_effort < 0.01, "Red must zero torques, got {}", r.control_effort);
    }
    #[test]
    fn test_moral_gate_consent_violation() {
        let mut b = ExoskeletonEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput { verdict: MoralGateInput::VERDICT_SAFE, consent_violation: true, ahimsa_violated: false });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange, "consent violation must force Orange retreat");
    }
}
