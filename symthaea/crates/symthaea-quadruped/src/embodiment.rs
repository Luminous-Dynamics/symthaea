use symthaea_core::genesis::GenesisSeed; use symthaea_core::hdc::ContinuousHV;
use crate::controller::QuadrupedController; use crate::encoder::QuadrupedHdcEncoder;
use crate::simulator::{QuadrupedPhysicsSimulator, SimpleQuadrupedSimulator}; use crate::types::{GaitType, QuadrupedConfig, NUM_ACTUATORS};
pub use symthaea_core::embodiment::{grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry, MoralGateInput, MotorSafetyLevel, GROUNDING_SENSORIMOTOR};

pub struct QuadrupedEmbodiment { ctrl: QuadrupedController, sim: SimpleQuadrupedSimulator, enc: QuadrupedHdcEncoder, last_p: Option<ContinuousHV>, steps: usize, safety: MotorSafetyLevel, safety_ov: Option<MotorSafetyLevel>, moral_safety: Option<MotorSafetyLevel>, effort: f32, pe: f32, gait: GaitType }
impl QuadrupedEmbodiment {
    pub fn new(g: &GenesisSeed) -> Self { let c = QuadrupedConfig::default(); Self { ctrl: QuadrupedController::new(g, &c), sim: SimpleQuadrupedSimulator::new(), enc: QuadrupedHdcEncoder::new(g, 32), last_p: None, steps: 0, safety: MotorSafetyLevel::Green, safety_ov: None, moral_safety: None, effort: 0.0, pe: 0.0, gait: GaitType::Trot } }
    pub fn set_safety_override(&mut self, l: MotorSafetyLevel) { self.safety_ov = Some(l); }
    pub fn clear_safety_override(&mut self) { self.safety_ov = None; }
    /// Apply moral gate from ethics engine.
    /// A quadruped can step on beings — ahimsa must force Red, consent violation Orange.
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.moral_safety = if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED { Some(MotorSafetyLevel::Red) }
            else if gate.consent_violation { Some(MotorSafetyLevel::Orange) }
            else if gate.verdict == MoralGateInput::VERDICT_CAUTION { Some(MotorSafetyLevel::Yellow) }
            else { None };
    }
    pub fn gait(&self) -> GaitType { self.gait }
    pub fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.gait = GaitType::from_phi(phi); self.sim.set_gait(self.gait);
        let pl = MotorSafetyLevel::from_phi(phi);
        self.safety = pl;
        if let Some(ov) = self.safety_ov { self.safety = self.safety.max(ov); }
        if let Some(m) = self.moral_safety { self.safety = self.safety.max(m); }
        let gain = self.safety.motor_gain(); let mut cmd = self.ctrl.forward(hv, dt);
        if gain < 1.0 { for t in &mut cmd.joint_torques { *t *= gain; } }
        self.effort = cmd.control_effort(); self.sim.step(&cmd, dt as f64);
        let p = self.enc.encode(self.sim.state()); self.pe = if let Some(ref prev) = self.last_p { (1.0-p.similarity(prev).max(0.0)).min(1.0) } else { 0.0 };
        self.last_p = Some(p); self.steps += 1;
        EmbodimentResult { num_actuators: NUM_ACTUATORS, control_effort: self.effort, success: self.sim.state().is_finite(), prediction_error: self.pe, safety_level: self.safety, epistemic_grounding: GROUNDING_SENSORIMOTOR, observation_confidence: grounding_from_prediction_error(self.pe) }
    }
    pub fn encode_perception(&mut self) -> ContinuousHV { let p = self.enc.encode(self.sim.state()); self.last_p = Some(p.clone()); p }
    pub fn reset(&mut self) { self.sim.reset(); self.ctrl.reset(); self.enc.reset(); self.last_p = None; self.steps = 0; self.safety = MotorSafetyLevel::Green; self.safety_ov = None; self.moral_safety = None; self.effort = 0.0; self.pe = 0.0; self.gait = GaitType::Trot; }
    pub fn safety_level(&self) -> MotorSafetyLevel { self.safety }
    pub fn total_steps(&self) -> usize { self.steps }
    pub fn telemetry(&self) -> EmbodimentTelemetry { EmbodimentTelemetry { total_steps: self.steps as u64, control_effort: self.effort, prediction_error: self.pe, safety_level: format!("{:?}", self.safety), platform: "quadruped".to_string(), num_actuators: NUM_ACTUATORS, epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(), observation_confidence: grounding_from_prediction_error(self.pe), platform_specific: Vec::new() } }
}
impl symthaea_core::embodiment::EmbodimentBridge for QuadrupedEmbodiment {
    fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult { self.step(hv, dt, phi) }
    fn encode_perception(&mut self) -> ContinuousHV { self.encode_perception() } fn reset(&mut self) { self.reset() }
    fn safety_level(&self) -> MotorSafetyLevel { self.safety_level() } fn set_safety_override(&mut self, l: MotorSafetyLevel) { self.set_safety_override(l) } fn clear_safety_override(&mut self) { self.clear_safety_override() }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform { symthaea_core::embodiment::EmbodimentPlatform::Quadruped }
    fn num_actuators(&self) -> usize { NUM_ACTUATORS } fn total_steps(&self) -> usize { self.total_steps() } fn telemetry(&self) -> EmbodimentTelemetry { self.telemetry() }
    fn apply_moral_gate(&mut self, gate: MoralGateInput) { self.apply_moral_gate(gate) }
}
#[cfg(test)] mod tests { use super::*;
    #[test] fn test_step() { let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t")); assert!(b.step(&ContinuousHV::random(16384, 42), 0.005, 0.7).success); assert_eq!(b.gait(), GaitType::Trot); }
    #[test] fn test_gait_switch() { let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t")); b.step(&ContinuousHV::random(16384, 42), 0.005, 0.05); assert_eq!(b.gait(), GaitType::Collapse); }
    #[test] fn test_moral_gate_ahimsa_forces_red() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput { verdict: MoralGateInput::VERDICT_SAFE, consent_violation: false, ahimsa_violated: true });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red, "ahimsa must force Red even at high phi");
        assert!(r.control_effort < 0.01, "Red must zero torques, got {}", r.control_effort);
    }
    #[test] fn test_moral_gate_clears() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput { verdict: MoralGateInput::VERDICT_BLOCKED, consent_violation: false, ahimsa_violated: false });
        b.apply_moral_gate(MoralGateInput { verdict: MoralGateInput::VERDICT_SAFE, consent_violation: false, ahimsa_violated: false });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green, "clearing moral gate must return to phi-derived safety");
    }
}
