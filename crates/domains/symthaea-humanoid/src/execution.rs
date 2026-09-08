// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authoritative humanoid execution composition root.
//!
//! Learned or cognitive policies may propose motion, but this module owns the
//! transition from proposal to actuator-eligible command. Every proposal is
//! routed through the deterministic hierarchical controller and the final
//! morphology-aware safety projector before it may be applied to a simulator or
//! passed to a hardware backend.

use crate::controller::HumanoidController;
use crate::hierarchical::{HierarchicalControlReport, HierarchicalHumanoidController};
use crate::morphology::HumanoidMorphology;
use crate::safety::{HumanoidSafetyProjector, SafetyProjectionReport};
use crate::types::{ActuationMode, HumanoidCommand, HumanoidConfig, HumanoidState, HumanoidTask};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Evidence emitted for every authoritative humanoid command synthesis.
#[derive(Debug, Clone)]
pub struct HumanoidExecutionReport {
    pub hierarchy: HierarchicalControlReport,
    pub safety: SafetyProjectionReport,
}

/// Result of one authoritative command synthesis.
#[derive(Debug, Clone)]
pub struct HumanoidExecutionResult {
    /// The command that is eligible for backend actuation.
    pub command: HumanoidCommand,
    /// Deterministic control and final safety evidence for this command.
    pub report: HumanoidExecutionReport,
}

/// Single composition root for learned policy, deterministic whole-body control,
/// and final command projection.
///
/// Backends remain responsible for their own independent hardware interlocks
/// (watchdog, e-stop, over-current, calibration, bus faults, and servo limits).
pub struct HumanoidExecutionPipeline {
    morphology: HumanoidMorphology,
    learned_policy: HumanoidController,
    hierarchy: HierarchicalHumanoidController,
    safety: HumanoidSafetyProjector,
    baseline_weight: f32,
}

impl HumanoidExecutionPipeline {
    pub fn new(genesis: &GenesisSeed, config: &HumanoidConfig) -> Self {
        let morphology = config.morphology;
        Self {
            morphology,
            learned_policy: HumanoidController::new(genesis, config),
            hierarchy: HierarchicalHumanoidController::new(morphology),
            safety: HumanoidSafetyProjector::new(morphology),
            baseline_weight: 1.0,
        }
    }

    pub fn morphology(&self) -> HumanoidMorphology {
        self.morphology
    }

    /// Set the deterministic baseline curriculum weight admitted by the
    /// hierarchy. The hierarchy itself retains its configured minimum baseline.
    pub fn set_baseline_weight(&mut self, weight: f32) {
        self.baseline_weight = weight.clamp(0.0, 1.0);
    }

    /// Produce an actuator-eligible command from a cognitive/learned proposal.
    ///
    /// `authority_scale` is restrictive only: callers may reduce the learned
    /// residual authority, but cannot bypass the deterministic hierarchy or
    /// final safety projector.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        &mut self,
        thought_hv: &ContinuousHV,
        task: HumanoidTask,
        state: &HumanoidState,
        dt: f64,
        free_energy: f64,
        authority_scale: f32,
        actuation_mode: ActuationMode,
    ) -> HumanoidExecutionResult {
        let n = self.morphology.num_actuators();
        let baseline = HumanoidCommand::zero_for(n);
        let mut learned_residual = self.learned_policy.forward(thought_hv, dt as f32);
        let authority_scale = if authority_scale.is_finite() {
            authority_scale.clamp(0.0, 1.0)
        } else {
            0.0
        };
        for torque in &mut learned_residual.torques {
            *torque *= authority_scale;
        }

        let (candidate, hierarchy) = self.hierarchy.synthesize(
            task,
            state,
            &baseline,
            &learned_residual,
            self.baseline_weight,
            free_energy,
        );
        let projected = self
            .safety
            .project(&candidate, state, actuation_mode, dt);

        HumanoidExecutionResult {
            command: projected.command,
            report: HumanoidExecutionReport {
                hierarchy,
                safety: projected.report,
            },
        }
    }

    pub fn reset(&mut self) {
        self.learned_policy.reset();
        self.safety.reset();
        self.baseline_weight = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_authority_fails_restrictive() {
        let config = HumanoidConfig::default();
        let mut pipeline = HumanoidExecutionPipeline::new(&GenesisSeed::from_phrase("test"), &config);
        let state = HumanoidState::standing_for(config.morphology);
        let hv = ContinuousHV::random(16384, 42);
        let result = pipeline.synthesize(
            &hv,
            HumanoidTask::Stand,
            &state,
            0.025,
            0.0,
            f32::NAN,
            ActuationMode::NormalizedTorque,
        );
        assert!(result.command.torques.iter().all(|value| value.is_finite()));
        assert!(result.command.torques.iter().all(|value| value.abs() <= 1.0));
    }

    #[test]
    fn every_command_receives_safety_evidence() {
        let config = HumanoidConfig::default();
        let mut pipeline = HumanoidExecutionPipeline::new(&GenesisSeed::from_phrase("test"), &config);
        let state = HumanoidState::standing_for(config.morphology);
        let hv = ContinuousHV::random(16384, 7);
        let result = pipeline.synthesize(
            &hv,
            HumanoidTask::Stand,
            &state,
            0.025,
            0.0,
            1.0,
            ActuationMode::NormalizedTorque,
        );
        assert_eq!(result.command.num_actuators(), config.morphology.num_actuators());
        assert!(!result.report.safety.rejected);
    }
}
