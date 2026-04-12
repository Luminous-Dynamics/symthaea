use crate::controller::OrbitalController;
use crate::encoder::OrbitalHdcEncoder;
use crate::simulator::{OrbitalPhysicsSimulator, SimpleOrbitalSimulator};
use crate::types::{OrbitalConfig, NUM_ACTUATORS};
pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub struct OrbitalEmbodiment {
    ctrl: OrbitalController,
    sim: SimpleOrbitalSimulator,
    enc: OrbitalHdcEncoder,
    last_p: Option<ContinuousHV>,
    steps: usize,
    safety: MotorSafetyLevel,
    safety_ov: Option<MotorSafetyLevel>,
    effort: f32,
    pe: f32,
}
impl OrbitalEmbodiment {
    pub fn new(g: &GenesisSeed) -> Self {
        let c = OrbitalConfig::default();
        Self {
            ctrl: OrbitalController::new(g, &c),
            sim: SimpleOrbitalSimulator::new(),
            enc: OrbitalHdcEncoder::new(g, 32),
            last_p: None,
            steps: 0,
            safety: MotorSafetyLevel::Green,
            safety_ov: None,
            effort: 0.0,
            pe: 0.0,
        }
    }
    pub fn set_safety_override(&mut self, l: MotorSafetyLevel) {
        self.safety_ov = Some(l);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_ov = None;
    }
    pub fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let pl = MotorSafetyLevel::from_phi(phi);
        self.safety = match self.safety_ov {
            Some(ov) => pl.max(ov),
            None => pl,
        };
        let gain = self.safety.motor_gain();
        let mut cmd = self.ctrl.forward(hv, dt);
        if gain <= 0.3 {
            cmd.joint_torques = [0.0; NUM_ACTUATORS];
        } else {
            for t in &mut cmd.joint_torques {
                *t *= gain;
            }
        }
        if self.sim.state().comm_window == 0.0 && gain > 0.3 {
            for t in &mut cmd.joint_torques {
                *t *= 0.5;
            }
        }
        self.effort = cmd.control_effort();
        self.sim.step(&cmd, dt as f64);
        let p = self.enc.encode(self.sim.state());
        self.pe = if let Some(ref prev) = self.last_p {
            (1.0 - p.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0
        };
        self.last_p = Some(p);
        self.steps += 1;
        EmbodimentResult {
            num_actuators: NUM_ACTUATORS,
            control_effort: self.effort,
            success: self.sim.state().is_finite(),
            prediction_error: self.pe,
            safety_level: self.safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(self.pe),
        }
    }
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.enc.encode(self.sim.state());
        self.last_p = Some(p.clone());
        p
    }
    pub fn reset(&mut self) {
        self.sim.reset();
        self.ctrl.reset();
        self.enc.reset();
        self.last_p = None;
        self.steps = 0;
        self.safety = MotorSafetyLevel::Green;
        self.safety_ov = None;
        self.effort = 0.0;
        self.pe = 0.0;
    }
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.safety
    }
    pub fn total_steps(&self) -> usize {
        self.steps
    }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.steps as u64,
            control_effort: self.effort,
            prediction_error: self.pe,
            safety_level: format!("{:?}", self.safety),
            platform: "orbital".to_string(),
            num_actuators: NUM_ACTUATORS,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.pe),
        }
    }
}
impl symthaea_core::embodiment::EmbodimentBridge for OrbitalEmbodiment {
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
    fn set_safety_override(&mut self, l: MotorSafetyLevel) {
        self.set_safety_override(l)
    }
    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Orbital
    }
    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }
    fn total_steps(&self) -> usize {
        self.total_steps()
    }
    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_step() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert!(b.step(&ContinuousHV::random(16384, 42), 0.01, 0.7).success);
    }
    #[test]
    fn test_orange_parks() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = b.step(&ContinuousHV::random(16384, 42), 0.01, 0.15);
        assert!(r.control_effort < 0.01);
    }
}
