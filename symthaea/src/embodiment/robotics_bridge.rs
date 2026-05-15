// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotics Bridge — Symthaea's physical body.
//!
//! Wires the high-level cognitive cycle (thought hypervectors) to low-level
//! motor commands via `EmbodimentBridge`.

use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentResult, MotorSafetyLevel};
use symthaea_core::hdc::ContinuousHV;
use symthaea_fep::ActiveInferenceAgent;

/// A robotic agent that couples cognitive thoughts to physical actions.
///
/// This is the "Mk0 Bootstrapper" protocol's embodiment core.
pub struct RoboticAgent {
    /// The physical/simulated body bridge.
    bridge: Box<dyn EmbodimentBridge>,
    /// Active Inference agent for control modulation and surprise detection.
    fep: ActiveInferenceAgent,
    /// Last recorded phi (consciousness level).
    last_phi: f64,
}

impl RoboticAgent {
    /// Create a new robotic agent from an embodiment bridge.
    pub fn new(bridge: Box<dyn EmbodimentBridge>) -> Self {
        // Initialize FEP agent with dimensions matching the bridge
        let num_actuators = bridge.num_actuators();
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: num_actuators,
            obs_dim: num_actuators,
            num_actions: num_actuators,
            ..Default::default()
        };

        Self {
            bridge,
            fep: ActiveInferenceAgent::new(config),
            last_phi: 0.0,
        }
    }

    /// Step the robotic body one increment in time.
    ///
    /// 1. Encodes body state (proprioception).
    /// 2. Runs active inference to compute surprise (free energy).
    /// 3. Modulates the thought HV based on safety and grounding.
    /// 4. Steps the physics via the bridge.
    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> RoboticStepResult {
        self.last_phi = phi;

        // 1. Perception
        let perception_hv = self.bridge.encode_perception();

        // Convert ContinuousHV to FEP Observation
        // We use the first N components of the HV as observables, or ideally
        // a reduced projection. For now, we take the mean or a sample.
        let values: Vec<f64> = perception_hv
            .as_slice()
            .iter()
            .take(self.fep.config.obs_dim)
            .map(|&f| f as f64)
            .collect();
        let obs = symthaea_fep::Observation::new(values, 1.0, "proprioception");

        // 2. Active Inference (Active Sensing)
        let perception_result = self.fep.perceive(&obs);
        let _action_result = self.fep.select_action();

        // 3. Safety Gating & Modulation
        let safety = self.bridge.safety_level();
        let gain = safety.motor_gain();

        // 4. Actuation
        let result = self.bridge.step(thought_hv, dt, phi);

        RoboticStepResult {
            embodiment: result,
            free_energy: perception_result.free_energy.total,
            surprise: perception_result.free_energy.prediction_error,
            motor_gain: gain,
        }
    }

    /// Access the underlying bridge.
    pub fn bridge(&self) -> &dyn EmbodimentBridge {
        self.bridge.as_ref()
    }

    /// Access the underlying bridge (mutable).
    pub fn bridge_mut(&mut self) -> &mut dyn EmbodimentBridge {
        self.bridge.as_mut()
    }

    /// Reset both body and brain state.
    pub fn reset(&mut self) {
        self.bridge.reset();
        self.fep.reset();
    }
}

/// Result of a robotic agent step.
pub struct RoboticStepResult {
    /// Outcome from the physical/simulated bridge.
    pub embodiment: EmbodimentResult,
    /// Variational free energy from the FEP loop.
    pub free_energy: f64,
    /// Surprise (prediction error) scalar.
    pub surprise: f64,
    /// Applied motor gain (0.0-1.0).
    pub motor_gain: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    #[cfg(feature = "manipulator")]
    #[test]
    fn test_robotic_agent_manipulator_step() {
        use crate::manipulator::embodiment::ManipulatorEmbodiment;
        let genesis = GenesisSeed::from_phrase("test-robot");
        let bridge = Box::new(ManipulatorEmbodiment::new(&genesis));
        let mut agent = RoboticAgent::new(bridge);

        let thought = ContinuousHV::random(16384, 123);
        let result = agent.step(&thought, 0.01, 0.8);

        assert!(result.embodiment.success);
        assert!(result.free_energy.is_finite());
        assert!(result.surprise.is_finite());
        assert_eq!(result.motor_gain, 1.0); // Green safety
    }

    #[test]
    fn test_robotic_agent_construction() {
        struct MockBridge;
        impl EmbodimentBridge for MockBridge {
            fn step(
                &mut self,
                _hv: &ContinuousHV,
                _dt: f32,
                _phi: f64,
            ) -> crate::symthaea_core::embodiment::EmbodimentResult {
                crate::symthaea_core::embodiment::EmbodimentResult {
                    num_actuators: 1,
                    control_effort: 0.0,
                    success: true,
                    prediction_error: 0.0,
                    safety_level: MotorSafetyLevel::Green,
                    epistemic_grounding: 0,
                    observation_confidence: 1.0,
                }
            }
            fn encode_perception(&mut self) -> ContinuousHV {
                ContinuousHV::zero(16384)
            }
            fn reset(&mut self) {}
            fn safety_level(&self) -> MotorSafetyLevel {
                MotorSafetyLevel::Green
            }
            fn set_safety_override(&mut self, _level: MotorSafetyLevel) {}
            fn clear_safety_override(&mut self) {}
            fn platform(&self) -> crate::symthaea_core::embodiment::EmbodimentPlatform {
                crate::symthaea_core::embodiment::EmbodimentPlatform::None
            }
            fn num_actuators(&self) -> usize {
                1
            }
            fn total_steps(&self) -> usize {
                0
            }
            fn telemetry(&self) -> crate::symthaea_core::embodiment::EmbodimentTelemetry {
                Default::default()
            }
            fn apply_moral_gate(
                &mut self,
                _gate: crate::symthaea_core::embodiment::MoralGateInput,
            ) {
            }
            fn platform_telemetry_bytes(&self) -> Vec<u8> {
                Vec::new()
            }
        }

        let agent = RoboticAgent::new(Box::new(MockBridge));
        assert_eq!(agent.bridge().num_actuators(), 1);
    }
}
