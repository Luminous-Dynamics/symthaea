// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for SAR helicopter.
//!
//! Enables the cognitive loop to fly a helicopter via the proprioceptive loop:
//! thought → motor commands → rotor dynamics + physics → proprioceptive HV.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};

use crate::controller::HelicopterController;
use crate::encoder::HelicopterHdcEncoder;
use crate::simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
use crate::types::HelicopterConfig;

/// Helicopter embodiment bridge.
///
/// Wraps controller + simulator + encoder into a single step function
/// that the cognitive loop can call each cycle.
pub struct HelicopterEmbodiment {
    controller: HelicopterController,
    simulator: SimpleHelicopterSimulator,
    encoder: HelicopterHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl HelicopterEmbodiment {
    /// Create from default config.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = HelicopterConfig::default();
        let controller = HelicopterController::new(genesis, &config);
        let simulator = SimpleHelicopterSimulator::new();
        let encoder = HelicopterHdcEncoder::new(genesis, 32);

        Self {
            controller,
            simulator,
            encoder,
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

    /// Step: thought → motor command (consciousness-gated) → physics → proprioceptive encoding.
    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);

        // Apply safety gain
        if gain < 1.0 {
            cmd.collective *= gain;
            cmd.cyclic_lon *= gain;
            cmd.cyclic_lat *= gain;
            cmd.pedal *= gain;
            cmd.thrust *= gain;
            cmd.tail_rotor *= gain;
        }

        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);

        let perception = self.encoder.encode(self.simulator.state());

        let pred_error = if let Some(ref prev) = self.last_perception {
            let sim = perception.similarity(prev);
            (1.0 - sim.max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        let success = self.simulator.state().is_finite();

        EmbodimentResult {
            num_actuators: 6,
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    /// Encode current body state as proprioceptive HV.
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let perception = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(perception.clone());
        perception
    }

    /// Reset to hover state.
    pub fn reset(&mut self) {
        self.simulator.reset(20.0);
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
    }

    pub fn safety_level(&self) -> MotorSafetyLevel { self.current_safety }
    pub fn total_steps(&self) -> usize { self.total_steps }
    pub fn last_perception(&self) -> Option<&ContinuousHV> { self.last_perception.as_ref() }

    /// Telemetry summary.
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "helicopter".to_string(),
            num_actuators: 6,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bridge() -> HelicopterEmbodiment {
        let genesis = GenesisSeed::from_phrase("test-heli-embodiment");
        HelicopterEmbodiment::new(&genesis)
    }

    #[test]
    fn test_step_produces_valid_result() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        let result = bridge.step(&hv, 1.0 / 300.0, 0.7);
        assert!(result.success);
        assert_eq!(result.num_actuators, 6);
        assert_eq!(result.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_proprioceptive_encoding() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 1.0 / 300.0, 0.7);
        let perception = bridge.encode_perception();
        assert_eq!(perception.dim(), 16384);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);

        // Red safety = zero motor gain
        let result = bridge.step(&hv, 1.0 / 300.0, 0.05);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        assert_eq!(result.control_effort, 0.0);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 1.0 / 300.0, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "helicopter");
        assert_eq!(t.num_actuators, 6);
    }

    #[test]
    fn test_reset() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 1.0 / 300.0, 0.7);
        assert_eq!(bridge.total_steps(), 1);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
        assert!(bridge.last_perception().is_none());
    }
}
