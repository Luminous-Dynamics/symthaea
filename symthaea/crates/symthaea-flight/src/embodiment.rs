// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for quadrotor flight platform.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use crate::types::FlightConfig;

pub use symthaea_hal::MotorSafetyLevel;

/// Quadrotor embodiment bridge.
pub struct FlightEmbodiment {
    controller: FlightController,
    simulator: SimplePhysicsSimulator,
    encoder: QuadrotorHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl FlightEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = FlightConfig::default();
        Self {
            controller: FlightController::new(genesis, &config),
            simulator: SimplePhysicsSimulator::new(),
            encoder: QuadrotorHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
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
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);
        if gain < 1.0 {
            // Scale moments; on Red, cut thrust to hover-only (autorotate equivalent)
            cmd.roll_moment *= gain;
            cmd.pitch_moment *= gain;
            cmd.yaw_moment *= gain;
            if gain == 0.0 {
                cmd.thrust = 0.0; // Motors off on Red
            } else {
                cmd.thrust *= gain;
            }
        }

        let effort = (cmd.thrust.abs()
            + cmd.roll_moment.abs()
            + cmd.pitch_moment.abs()
            + cmd.yaw_moment.abs())
            / 4.0;
        self.last_control_effort = effort;

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

        let success = self.simulator.state().position[2].is_finite();

        EmbodimentResult {
            num_actuators: 4,
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(p.clone());
        p
    }

    pub fn reset(&mut self) {
        self.simulator.reset(0.1);
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

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "quadrotor".to_string(),
            num_actuators: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbodimentResult {
    pub num_actuators: usize,
    pub control_effort: f32,
    pub success: bool,
    pub prediction_error: f32,
    pub safety_level: MotorSafetyLevel,
}

#[derive(Debug, Clone, Default)]
pub struct EmbodimentTelemetry {
    pub total_steps: u64,
    pub control_effort: f32,
    pub prediction_error: f32,
    pub safety_level: String,
    pub platform: String,
    pub num_actuators: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.7);
        assert!(r.success);
        assert_eq!(r.num_actuators, 4);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert_eq!(r.control_effort, 0.0);
    }

    #[test]
    fn test_perception() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "quadrotor");
    }

    #[test]
    fn test_reset() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }
}
