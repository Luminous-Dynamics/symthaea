// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_core::genesis::GenesisSeed; use symthaea_core::hdc::ContinuousHV;
use crate::controller::SurgicalController; use crate::encoder::SurgicalHdcEncoder;
use crate::simulator::{SimpleSurgicalSimulator, SurgicalPhysicsSimulator};
use crate::types::{SurgicalConfig, SurgicalSafetyLevel, NUM_ACTUATORS};
pub use symthaea_core::embodiment::{grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry, MotorSafetyLevel, GROUNDING_SENSORIMOTOR};

pub struct SurgicalEmbodiment { controller: SurgicalController, simulator: SimpleSurgicalSimulator, encoder: SurgicalHdcEncoder, last_perception: Option<ContinuousHV>, total_steps: usize, current_safety: MotorSafetyLevel, safety_override: Option<MotorSafetyLevel>, last_control_effort: f32, last_prediction_error: f32, surgical_safety: SurgicalSafetyLevel }

impl SurgicalEmbodiment {
    pub fn new(g: &GenesisSeed) -> Self { let c = SurgicalConfig::default(); Self { controller: SurgicalController::new(g, &c), simulator: SimpleSurgicalSimulator::new(), encoder: SurgicalHdcEncoder::new(g, 32), last_perception: None, total_steps: 0, current_safety: MotorSafetyLevel::Green, safety_override: None, last_control_effort: 0.0, last_prediction_error: 0.0, surgical_safety: SurgicalSafetyLevel::FullControl } }
    pub fn set_safety_override(&mut self, l: MotorSafetyLevel) { self.safety_override = Some(l); }
    pub fn clear_safety_override(&mut self) { self.safety_override = None; }
    pub fn surgical_safety(&self) -> SurgicalSafetyLevel { self.surgical_safety }
    pub fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.surgical_safety = SurgicalSafetyLevel::from_phi(phi); let tg = self.surgical_safety.torque_gain();
        let pl = MotorSafetyLevel::from_phi(phi); self.current_safety = match self.safety_override { Some(ov) => pl.max(ov), None => pl };
        let mut cmd = self.controller.forward(hv, dt);
        for t in &mut cmd.joint_torques { *t *= tg; }
        if !self.surgical_safety.cautery_allowed() { cmd.cautery = 0.0; }
        if self.surgical_safety == SurgicalSafetyLevel::Retract { cmd.joint_torques = [0.0,-0.3,0.0,0.0,0.0,0.0]; cmd.jaw = 0.0; cmd.cautery = 0.0; }
        self.last_control_effort = cmd.control_effort(); self.simulator.step(&cmd, dt as f64);
        let p = self.encoder.encode(self.simulator.state());
        let pe = if let Some(ref prev) = self.last_perception { (1.0-p.similarity(prev).max(0.0)).min(1.0) } else { 0.0f32 };
        self.last_prediction_error = pe; self.last_perception = Some(p); self.total_steps += 1;
        EmbodimentResult { num_actuators: NUM_ACTUATORS, control_effort: self.last_control_effort, success: self.simulator.state().is_finite(), prediction_error: pe, safety_level: self.current_safety, epistemic_grounding: GROUNDING_SENSORIMOTOR, observation_confidence: grounding_from_prediction_error(pe) }
    }
    pub fn encode_perception(&mut self) -> ContinuousHV { let p = self.encoder.encode(self.simulator.state()); self.last_perception = Some(p.clone()); p }
    pub fn reset(&mut self) { self.simulator.reset(); self.controller.reset(); self.encoder.reset(); self.last_perception = None; self.total_steps = 0; self.current_safety = MotorSafetyLevel::Green; self.safety_override = None; self.last_control_effort = 0.0; self.last_prediction_error = 0.0; self.surgical_safety = SurgicalSafetyLevel::FullControl; }
    pub fn safety_level(&self) -> MotorSafetyLevel { self.current_safety }
    pub fn total_steps(&self) -> usize { self.total_steps }
    pub fn telemetry(&self) -> EmbodimentTelemetry { EmbodimentTelemetry { total_steps: self.total_steps as u64, control_effort: self.last_control_effort, prediction_error: self.last_prediction_error, safety_level: format!("{:?}", self.current_safety), platform: "surgical".to_string(), num_actuators: NUM_ACTUATORS, epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(), observation_confidence: grounding_from_prediction_error(self.last_prediction_error) } }
}

impl symthaea_core::embodiment::EmbodimentBridge for SurgicalEmbodiment {
    fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult { self.step(hv, dt, phi) }
    fn encode_perception(&mut self) -> ContinuousHV { self.encode_perception() }
    fn reset(&mut self) { self.reset() }
    fn safety_level(&self) -> MotorSafetyLevel { self.safety_level() }
    fn set_safety_override(&mut self, l: MotorSafetyLevel) { self.set_safety_override(l) }
    fn clear_safety_override(&mut self) { self.clear_safety_override() }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform { symthaea_core::embodiment::EmbodimentPlatform::Surgical }
    fn num_actuators(&self) -> usize { NUM_ACTUATORS }
    fn total_steps(&self) -> usize { self.total_steps() }
    fn telemetry(&self) -> EmbodimentTelemetry { self.telemetry() }
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_step() { let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t")); assert!(b.step(&ContinuousHV::random(16384, 42), 0.001, 0.7).success); }
    #[test] fn test_retract() { let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t")); b.step(&ContinuousHV::random(16384, 42), 0.001, 0.05); assert_eq!(b.surgical_safety(), SurgicalSafetyLevel::Retract); }
    #[test] fn test_extended() { let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t")); let hv = ContinuousHV::random(16384, 42); for _ in 0..500 { assert!(b.step(&hv, 0.001, 0.7).success); } }
}
