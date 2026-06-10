// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for the gravcraft platform.
//!
//! Unlike all other platforms that apply forces, this applies metric
//! perturbations to spacetime. Consciousness (Phi) directly gates
//! how much spacetime warping is allowed.

use crate::controller::MetricController;
use crate::encoder::GravcraftHdcEncoder;
use crate::simulator::MetricPerturbationSimulator;
use crate::types::{GravcraftConfig, MetricCommand};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MotorSafetyLevel,
    grounding_from_prediction_error, grounding_label,
};

/// Gravcraft embodiment bridge.
///
/// Safety gating unique to this platform:
/// - Green (Phi > 0.6): full metric authority, all 3 amplifiers
/// - Yellow (0.3-0.6): single amplifier only, reduced amplitude
/// - Orange (0.1-0.3): flat metric (zero perturbation), drift only
/// - Red (< 0.1): emergency metric neutralization
pub struct GravcraftEmbodiment {
    controller: MetricController,
    simulator: MetricPerturbationSimulator,
    encoder: GravcraftHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl GravcraftEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = GravcraftConfig::default();
        Self {
            controller: MetricController::new(genesis, &config),
            simulator: MetricPerturbationSimulator::new(config),
            encoder: GravcraftHdcEncoder::new(genesis),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
        }
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.current_safety = MotorSafetyLevel::from_phi(phi);

        // Get raw command from controller (Phi already gates amplitude internally)
        let mut cmd = self.controller.forward(thought_hv, phi);

        // Apply safety-level restrictions
        match self.current_safety {
            MotorSafetyLevel::Green => {
                // Full authority — all 3 amplifiers active
            }
            MotorSafetyLevel::Yellow => {
                // Single amplifier only, reduced amplitude
                cmd.amplifiers[1] = (0.0, 0.0, 0.0, 0.0);
                cmd.amplifiers[2] = (0.0, 0.0, 0.0, 0.0);
                let (amp, f, a, e) = cmd.amplifiers[0];
                cmd.amplifiers[0] = (amp * 0.5, f, a, e);
            }
            MotorSafetyLevel::Orange => {
                // Flat metric — zero perturbation, drift only
                cmd = MetricCommand::default();
            }
            MotorSafetyLevel::Red => {
                // Emergency metric neutralization
                cmd = MetricCommand::default();
            }
        }

        // Compute control effort (sum of absolute amplitudes)
        let effort: f64 = cmd.amplifiers.iter().map(|(a, _, _, _)| a.abs()).sum();
        self.last_control_effort = effort as f32;

        // Step the simulator
        self.simulator.step(&cmd, dt as f64);

        // Encode perception
        let perception = self.encoder.encode(self.simulator.state());

        let pred_error = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        let success = self
            .simulator
            .state()
            .position
            .iter()
            .all(|p| p.is_finite());

        EmbodimentResult {
            num_actuators: 12, // 3 amplifiers × 4 controls
            control_effort: effort as f32,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        self.encoder.encode(self.simulator.state())
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "gravcraft".to_string(),
            num_actuators: 12,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

    #[test]
    fn test_embodiment_step_valid() {
        let genesis = GenesisSeed::from_phrase("gravcraft embodiment");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xCAFE);

        let result = emb.step(&thought, 0.01, 0.7);
        assert!(result.success);
        assert_eq!(result.num_actuators, 12);
        assert!(result.control_effort.is_finite());
    }

    #[test]
    fn test_phi_zero_no_metric_authority() {
        let genesis = GenesisSeed::from_phrase("safety test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xBEEF);

        let result = emb.step(&thought, 0.01, 0.0);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        // With Red safety, there should be no control effort
        assert!(
            result.control_effort < 1e-6,
            "Red safety should zero control effort: {}",
            result.control_effort
        );
    }

    #[test]
    fn test_safety_level_transitions() {
        let genesis = GenesisSeed::from_phrase("transition test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xFACE);

        // Green
        let r = emb.step(&thought, 0.01, 0.8);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);

        // Yellow
        let r = emb.step(&thought, 0.01, 0.4);
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);

        // Orange
        let r = emb.step(&thought, 0.01, 0.2);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);

        // Red
        let r = emb.step(&thought, 0.01, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_encode_perception_returns_valid() {
        let genesis = GenesisSeed::from_phrase("perception test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let hv = emb.encode_perception();
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }
}
