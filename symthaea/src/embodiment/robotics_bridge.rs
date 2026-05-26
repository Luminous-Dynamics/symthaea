// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotics Bridge — Symthaea's physical body.
//!
//! Wires the high-level cognitive cycle (thought hypervectors) to low-level
//! motor commands via `EmbodimentBridge` with closed-loop Active Inference.

use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentResult, MotorSafetyLevel};
use symthaea_core::hdc::ContinuousHV;
use symthaea_fep::ActiveInferenceAgent;

/// A robotic agent that couples cognitive thoughts to physical actions.
pub struct RoboticAgent {
    bridge: Box<dyn EmbodimentBridge>,
    fep: ActiveInferenceAgent,
    last_phi: f64,
}

impl RoboticAgent {
    pub fn new(bridge: Box<dyn EmbodimentBridge>) -> Self {
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

    /// Step the robotic body one increment in time with true Active Inference modulation.
    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> RoboticStepResult {
        self.last_phi = phi;

        // 1. Perception (Proprioception Input)
        let perception_hv = self.bridge.encode_perception();
        let values: Vec<f64> = perception_hv
            .as_slice()
            .iter()
            .take(self.fep.config.obs_dim)
            .map(|&f| f as f64)
            .collect();
        let obs = symthaea_fep::Observation::new(values, 1.0, "proprioception");

        // 2. Active Inference Sensory Update & Action Selection
        let perception_result = self.fep.perceive(&obs);
        let action_result = self.fep.select_action();

        // 3. Precision-Weighted Cognitive Modulation
        let surprise = perception_result.free_energy.prediction_error;

        let modulated_hv = if surprise > 0.5 {
            let mut modified = thought_hv.clone();
            let slice = modified.values.as_mut_slice();

            // Map the discrete action selection index to apply a deterministic torque bias
            let bias = (action_result.action as f32 * 0.05 * surprise as f32).clamp(-1.0, 1.0);
            for (i, val) in slice.iter_mut().enumerate() {
                if i % 7 == action_result.action % 7 {
                    *val += bias;
                }
            }
            modified
        } else {
            thought_hv.clone()
        };

        // 4. Safety Gating
        let safety = self.bridge.safety_level();
        let gain = safety.motor_gain();

        // 5. Actuation
        let result = self.bridge.step(&modulated_hv, dt, phi);

        RoboticStepResult {
            embodiment: result,
            free_energy: perception_result.free_energy.total,
            surprise,
            motor_gain: gain,
        }
    }

    pub fn bridge(&self) -> &dyn EmbodimentBridge {
        self.bridge.as_ref()
    }

    pub fn bridge_mut(&mut self) -> &mut dyn EmbodimentBridge {
        self.bridge.as_mut()
    }

    pub fn reset(&mut self) {
        self.bridge.reset();
        self.fep.reset();
    }
}

pub struct RoboticStepResult {
    pub embodiment: EmbodimentResult,
    pub free_energy: f64,
    pub surprise: f64,
    pub motor_gain: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn test_robotic_agent_construction() {
        struct MockBridge;
        impl EmbodimentBridge for MockBridge {
            fn step(&mut self, _hv: &ContinuousHV, _dt: f32, _phi: f64) -> EmbodimentResult {
                EmbodimentResult {
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
