// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for the bipedal humanoid platform.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::HumanoidController;
use crate::encoder::HumanoidHdcEncoder;
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::types::{HumanoidCommand, HumanoidConfig};

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback posture for bipedal humanoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidFallbackStage {
    /// Hold current pose with gravity-compensation torque baseline.
    StandingLock,
}

/// Bipedal humanoid embodiment bridge.
pub struct HumanoidEmbodiment {
    controller: HumanoidController,
    simulator: SimpleHumanoidSimulator,
    encoder: HumanoidHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: HumanoidFallbackStage,
    fallback_cycles_in_stage: u32,
    num_actuators: usize,
}

impl HumanoidEmbodiment {
    /// Gravity-compensation torque baseline applied to hip pitch joints during StandingLock.
    const GRAVITY_COMP_BASELINE: f32 = 0.05;

    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = HumanoidConfig::default();
        let simulator = SimpleHumanoidSimulator::new();
        let num_actuators = simulator.state().joint_angles.len();
        Self {
            controller: HumanoidController::new(genesis, &config),
            simulator,
            encoder: HumanoidHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: HumanoidFallbackStage::StandingLock,
            fallback_cycles_in_stage: 0,
            num_actuators,
        }
    }

    /// Apply moral gate from ethics engine.
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.moral_safety =
            if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED {
                Some(MotorSafetyLevel::Red)
            } else if gate.consent_violation {
                Some(MotorSafetyLevel::Orange)
            } else if gate.verdict == MoralGateInput::VERDICT_CAUTION {
                Some(MotorSafetyLevel::Yellow)
            } else {
                None
            };
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    pub fn fallback_stage(&self) -> HumanoidFallbackStage {
        self.fallback_stage
    }

    fn apply_standing_lock(&self, cmd: &mut HumanoidCommand) {
        // Zero all torques as the default safe state
        for t in cmd.torques.iter_mut() {
            *t = 0.0;
        }
        // Dynamically resolve hip pitch joints from the morphology layout map
        let names = crate::morphology::HumanoidMorphology::Dmc21.joint_names();
        for (idx, name) in names.iter().enumerate() {
            if name.contains("hip_y") && idx < cmd.torques.len() {
                cmd.torques[idx] = Self::GRAVITY_COMP_BASELINE;
            }
        }
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── StandingLock fallback for Red tier ──────────────────────
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_standing_lock(&mut cmd);
        } else {
            self.fallback_stage = HumanoidFallbackStage::StandingLock;
            self.fallback_cycles_in_stage = 0;
            if gain < 1.0 {
                for t in cmd.torques.iter_mut() {
                    *t *= gain;
                }
            }
        }

        self.last_control_effort = cmd.control_effort();

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

        let success = self.simulator.state().root_height.is_finite()
            && self.simulator.state().root_height > 0.2;

        EmbodimentResult {
            num_actuators: self.num_actuators,
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
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
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.fallback_stage = HumanoidFallbackStage::StandingLock;
        self.fallback_cycles_in_stage = 0;
    }

    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Access the underlying simulator (for testing and telemetry).
    pub fn simulator(&self) -> &SimpleHumanoidSimulator {
        &self.simulator
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "humanoid".to_string(),
            num_actuators: self.num_actuators,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for HumanoidEmbodiment {
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
    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.set_safety_override(level)
    }
    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Humanoid
    }
    fn num_actuators(&self) -> usize {
        self.num_actuators
    }
    fn total_steps(&self) -> usize {
        self.total_steps()
    }
    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.apply_moral_gate(gate)
    }
}

impl SafeFallback for HumanoidEmbodiment {
    fn platform_name(&self) -> &'static str {
        "humanoid"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        9
    }
    fn safe_fallback_description(&self) -> &'static str {
        "StandingLock: zero torque + gravity-comp hip pitch baseline"
    }
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.7);
        assert_eq!(r.num_actuators, 21);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_green_allows_full_authority() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_red_triggers_standing_lock() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.025, 0.05);
        assert_eq!(bridge.fallback_stage(), HumanoidFallbackStage::StandingLock);
    }

    #[test]
    fn test_standing_lock_applies_hip_gravity_comp() {
        // Regression: joint_names() used to return generic "j_N" names, so
        // the contains("hip_y") lookup never matched and the advertised
        // gravity-comp baseline was silently absent (pure zero torque).
        let bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = HumanoidCommand {
            torques: vec![1.0; 21],
        };
        bridge.apply_standing_lock(&mut cmd);
        let names = crate::morphology::HumanoidMorphology::Dmc21.joint_names();
        let hip_indices: Vec<usize> = names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.contains("hip_y"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            hip_indices.len(),
            2,
            "expected right_hip_y + left_hip_y in joint names, got {names:?}"
        );
        for (i, t) in cmd.torques.iter().enumerate() {
            if hip_indices.contains(&i) {
                assert!(
                    *t > 0.0,
                    "hip pitch joint {i} must carry the gravity-comp baseline"
                );
            } else {
                assert_eq!(*t, 0.0, "non-hip joint {i} must be zeroed at Red");
            }
        }
    }

    #[test]
    fn test_perception() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.025, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.025, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "humanoid");
        assert_eq!(t.num_actuators, 21);
    }

    #[test]
    fn test_reset() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.025, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_moral_gate_consent_violation_forces_orange() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
    }

    #[test]
    fn test_moral_gate_caution_forces_yellow() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_CAUTION,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.025, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
    }

    #[test]
    fn test_platform_identity() {
        use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform};
        let bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(bridge.platform(), EmbodimentPlatform::Humanoid);
    }
}
