// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for the bipedal humanoid platform.
//!
//! The public bridge deliberately does not apply learned motor output directly.
//! Goal-directed commands cross the shared [`HumanoidExecutionPipeline`], which
//! composes deterministic whole-body control with final command projection.
//! The state used for control is fused through the humanoid estimator, and its
//! explicit bounded uncertainty restricts epistemic goal authority.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::HumanoidController;
use crate::encoder::HumanoidHdcEncoder;
use crate::execution::{HumanoidAuthorityEnvelope, HumanoidExecutionPipeline};
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::state_estimation::{
    FusedHumanoidStateEstimator, ProprioceptiveMeasurement, StateEstimatorConfig,
};
use crate::state_uncertainty::HumanoidStateUncertaintyEnvelope;
use crate::types::{
    ActuationMode, HumanoidCommand, HumanoidConfig, HumanoidPdGains, HumanoidState, HumanoidTask,
    pd_standing_baseline,
};

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
    pipeline: HumanoidExecutionPipeline,
    pd_gains: HumanoidPdGains,
    simulator: SimpleHumanoidSimulator,
    encoder: HumanoidHdcEncoder,
    state_estimator: FusedHumanoidStateEstimator,
    state_estimator_config: StateEstimatorConfig,
    observation_sequence: u64,
    last_state_uncertainty: Option<HumanoidStateUncertaintyEnvelope>,
    last_epistemic_authority: f32,
    rejected_state_updates: u64,
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
    /// The generic bridge has no training curriculum, so it requests the
    /// hierarchy's minimum retained deterministic baseline rather than full PD
    /// curriculum authority.
    const EMBODIMENT_BASELINE_WEIGHT: f32 = 0.0;

    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = HumanoidConfig::default();
        let morphology = config.morphology;
        let simulator = SimpleHumanoidSimulator::new();
        let num_actuators = simulator.state().joint_angles.len();
        let state_estimator_config = StateEstimatorConfig::default();
        let mut state_estimator = FusedHumanoidStateEstimator::with_config(
            morphology,
            state_estimator_config.clone(),
        );
        state_estimator
            .reset(simulator.state())
            .expect("validated simulator state must initialize humanoid estimator");
        Self {
            controller: HumanoidController::new(genesis, &config),
            pipeline: HumanoidExecutionPipeline::new(morphology),
            pd_gains: HumanoidPdGains::for_morphology(morphology),
            simulator,
            encoder: HumanoidHdcEncoder::new(genesis, 32),
            state_estimator,
            state_estimator_config,
            observation_sequence: 0,
            last_state_uncertainty: None,
            last_epistemic_authority: 0.0,
            rejected_state_updates: 0,
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

    pub fn last_state_uncertainty(&self) -> Option<HumanoidStateUncertaintyEnvelope> {
        self.last_state_uncertainty
    }

    pub fn last_epistemic_authority(&self) -> f32 {
        self.last_epistemic_authority
    }

    pub fn rejected_state_updates(&self) -> u64 {
        self.rejected_state_updates
    }

    fn apply_standing_lock(&self, cmd: &mut HumanoidCommand) {
        // Zero all torques as the default safe state.
        for t in cmd.torques.iter_mut() {
            *t = 0.0;
        }
        // Resolve hip pitch joints from the active morphology rather than the
        // legacy DMC21 constants.
        let names = self.pipeline.morphology().joint_names();
        for (idx, name) in names.iter().enumerate() {
            if name.contains("hip_y") && idx < cmd.torques.len() {
                cmd.torques[idx] = Self::GRAVITY_COMP_BASELINE;
            }
        }
    }

    /// Fuse the current backend observation and derive the restrictive epistemic
    /// authority supported by this exact estimator update. Rejected evidence
    /// retains the previous state estimate but revokes goal authority for the tick.
    fn estimated_control_state(&mut self) -> (HumanoidState, f32) {
        let raw_state = self.simulator.state().clone();
        let contact_frame = self.simulator.contact_frame();
        self.observation_sequence = self.observation_sequence.saturating_add(1);
        let mut measurement = ProprioceptiveMeasurement::from_simulator(
            self.pipeline.morphology(),
            self.observation_sequence,
            raw_state,
            contact_frame,
        );
        measurement.received_at_s = self.simulator.state().timestamp;

        match self.state_estimator.update(&measurement) {
            Ok((estimate, report)) => {
                let estimate = estimate.clone();
                let uncertainty = HumanoidStateUncertaintyEnvelope::from_estimator_report(
                    report,
                    &self.state_estimator_config,
                );
                let authority = uncertainty.epistemic_authority();
                self.last_state_uncertainty = Some(uncertainty);
                self.last_epistemic_authority = authority;
                (estimate, authority)
            }
            Err(_) => {
                self.rejected_state_updates = self.rejected_state_updates.saturating_add(1);
                self.last_state_uncertainty = None;
                self.last_epistemic_authority = 0.0;
                (self.state_estimator.estimate().clone(), 0.0)
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

        let (state, epistemic_authority) = self.estimated_control_state();
        let goal_authority_revoked = epistemic_authority <= 0.0;
        let cmd = if matches!(self.current_safety, MotorSafetyLevel::Red) || goal_authority_revoked {
            // Red or rejected state evidence revokes goal-directed authority.
            // StandingLock retains independent minimum-safe authority and still
            // crosses final projection.
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            let mut fallback = HumanoidCommand::zero_for(self.num_actuators);
            self.apply_standing_lock(&mut fallback);
            self.pipeline
                .authorize_fallback(
                    &fallback,
                    &state,
                    ActuationMode::NormalizedTorque,
                    dt as f64,
                )
                .command
        } else {
            self.fallback_stage = HumanoidFallbackStage::StandingLock;
            self.fallback_cycles_in_stage = 0;

            let learned_residual = self.controller.forward(thought_hv, dt);
            let baseline = pd_standing_baseline(&state, &self.pd_gains);
            // This in-process simulator explicitly admits operator,
            // qualification, and physical-health authority. Epistemic authority
            // comes from the live estimator evidence; cognition may only reduce it.
            let authority = HumanoidAuthorityEnvelope {
                operator: 1.0,
                qualification: 1.0,
                physical: 1.0,
                epistemic: epistemic_authority,
                cognitive: self.current_safety.motor_gain(),
            };
            self.pipeline
                .authorize_with_authority(
                    HumanoidTask::Stand,
                    &state,
                    &baseline,
                    &learned_residual,
                    Self::EMBODIMENT_BASELINE_WEIGHT,
                    0.0,
                    authority,
                    ActuationMode::NormalizedTorque,
                    dt as f64,
                )
                .command
        };

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
        let observation_confidence =
            grounding_from_prediction_error(pred_error).min(self.last_epistemic_authority);

        EmbodimentResult {
            num_actuators: self.num_actuators,
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence,
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
        self.pipeline.reset();
        self.encoder.reset();
        let reset_state = self.simulator.state().clone();
        self.state_estimator
            .reset(&reset_state)
            .expect("validated reset state must reinitialize humanoid estimator");
        self.observation_sequence = 0;
        self.last_state_uncertainty = None;
        self.last_epistemic_authority = 0.0;
        self.rejected_state_updates = 0;
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
            safety_level: self.current_safety,
            platform: "humanoid".to_string(),
            num_actuators: self.num_actuators,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error)
                .min(self.last_epistemic_authority),
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
    fn test_state_evidence_restricts_runtime_authority() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.025, 0.9);
        let uncertainty = bridge
            .last_state_uncertainty()
            .expect("accepted simulator observation must produce uncertainty evidence");
        assert!(uncertainty.accepted);
        assert!(bridge.last_epistemic_authority().is_finite());
        assert!((0.0..=1.0).contains(&bridge.last_epistemic_authority()));
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
        let names = bridge.pipeline.morphology().joint_names();
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
        assert!((0.0..=1.0).contains(&t.observation_confidence));
    }

    #[test]
    fn test_reset() {
        let mut bridge = HumanoidEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.025, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
        assert!(bridge.last_state_uncertainty().is_none());
        assert_eq!(bridge.last_epistemic_authority(), 0.0);
        assert_eq!(bridge.rejected_state_updates(), 0);
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
