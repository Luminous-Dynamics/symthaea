// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::SubterraneanController;
use crate::encoder::SubterraneanHdcEncoder;
use crate::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use crate::types::SubterraneanConfig;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};

pub struct SubterraneanEmbodiment {
    controller: SubterraneanController,
    simulator: SimpleSubterraneanSimulator,
    encoder: SubterraneanHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl SubterraneanEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = SubterraneanConfig::default();
        Self {
            controller: SubterraneanController::new(genesis, &config),
            simulator: SimpleSubterraneanSimulator::new(),
            encoder: SubterraneanHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
        }
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(o) => phi_level.max(o),
            None => phi_level,
        };
        let gain = self.current_safety.motor_gain();
        let mut cmd = self.controller.forward(thought_hv, dt);
        if gain < 1.0 {
            for t in &mut cmd.torques {
                *t *= gain;
            }
        }
        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);
        let perception = self.encoder.encode(self.simulator.state());
        let pe = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0
        };
        self.last_prediction_error = pe;
        self.last_perception = Some(perception);
        self.total_steps += 1;
        EmbodimentResult {
            num_actuators: 6,
            control_effort: self.last_control_effort,
            success: self.simulator.state().is_finite(),
            prediction_error: pe,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pe),
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
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
    }
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
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
            platform: "subterranean".to_string(),
            num_actuators: 6,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for SubterraneanEmbodiment {
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.step(thought_hv, dt, phi)
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
        symthaea_core::embodiment::EmbodimentPlatform::Subterranean
    }

    fn num_actuators(&self) -> usize {
        6
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
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }
}
