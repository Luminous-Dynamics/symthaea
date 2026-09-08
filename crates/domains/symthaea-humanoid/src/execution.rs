// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authoritative humanoid execution boundary.
//!
//! Cognitive and learned systems may propose motion, but they do not own the
//! transition to actuator authority. This module composes the deterministic
//! hierarchical controller with the final morphology-aware safety projector.
//!
//! Backend-specific watchdog, e-stop, over-current, calibration, bus-fault, and
//! servo enforcement remains an additional independent responsibility of the
//! hardware abstraction layer.

use crate::hierarchical::{HierarchicalControlReport, HierarchicalHumanoidController};
use crate::morphology::HumanoidMorphology;
use crate::safety::{HumanoidSafetyProjector, SafetyProjectionReport};
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState, HumanoidTask};

/// Evidence emitted for every command that crosses the humanoid authority boundary.
#[derive(Debug, Clone)]
pub struct HumanoidExecutionReport {
    pub hierarchy: HierarchicalControlReport,
    pub safety: SafetyProjectionReport,
    /// Restrictive authority multiplier applied after deterministic synthesis.
    /// This can reduce requested motion but cannot bypass hierarchy or safety.
    pub authority_scale: f32,
}

/// Result of one authoritative command synthesis.
#[derive(Debug, Clone)]
pub struct HumanoidExecutionResult {
    /// Command eligible to continue to a simulator or independent hardware interlock.
    pub command: HumanoidCommand,
    /// Control and safety evidence associated with the command.
    pub report: HumanoidExecutionReport,
}

/// Single authority boundary shared by cognitive, training, simulation, HIL,
/// and future physical-runtime callers.
pub struct HumanoidExecutionPipeline {
    morphology: HumanoidMorphology,
    hierarchy: HierarchicalHumanoidController,
    safety: HumanoidSafetyProjector,
}

impl HumanoidExecutionPipeline {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self {
            morphology,
            hierarchy: HierarchicalHumanoidController::new(morphology),
            safety: HumanoidSafetyProjector::new(morphology),
        }
    }

    pub fn morphology(&self) -> HumanoidMorphology {
        self.morphology
    }

    /// Admit a baseline + learned residual through deterministic whole-body
    /// synthesis and final command projection.
    ///
    /// `authority_scale` is deliberately restrictive-only. Non-finite values
    /// fail closed to zero requested motion. A caller can reduce the synthesized
    /// command, but it cannot use this parameter to exceed the controller's
    /// normalized authority or bypass final safety projection.
    #[allow(clippy::too_many_arguments)]
    pub fn authorize(
        &mut self,
        task: HumanoidTask,
        state: &HumanoidState,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
        authority_scale: f32,
        actuation_mode: ActuationMode,
        dt: f64,
    ) -> HumanoidExecutionResult {
        let (mut candidate, hierarchy) = self.hierarchy.synthesize(
            task,
            state,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
        );

        let authority_scale = if authority_scale.is_finite() {
            authority_scale.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if authority_scale < 1.0 {
            for torque in &mut candidate.torques {
                *torque *= authority_scale;
            }
        }

        let projected = self
            .safety
            .project(&candidate, state, actuation_mode, dt);

        HumanoidExecutionResult {
            command: projected.command,
            report: HumanoidExecutionReport {
                hierarchy,
                safety: projected.report,
                authority_scale,
            },
        }
    }

    /// Route a platform minimum-safe fallback through the same final physical
    /// command projector. Safe fallback intentionally bypasses goal-directed
    /// hierarchy: it is the independent behavior used when goal authority has
    /// already been revoked.
    pub fn authorize_fallback(
        &mut self,
        fallback: &HumanoidCommand,
        state: &HumanoidState,
        actuation_mode: ActuationMode,
        dt: f64,
    ) -> crate::safety::ProjectedCommand {
        self.safety.project(fallback, state, actuation_mode, dt)
    }

    pub fn reset(&mut self) {
        self.safety.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_pair(morphology: HumanoidMorphology) -> (HumanoidCommand, HumanoidCommand) {
        let n = morphology.num_actuators();
        (HumanoidCommand::zero_for(n), HumanoidCommand::zero_for(n))
    }

    #[test]
    fn non_finite_authority_fails_restrictive() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let (baseline, residual) = zero_pair(morphology);
        let result = pipeline.authorize(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
            f32::NAN,
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert_eq!(result.report.authority_scale, 0.0);
        assert!(result.command.torques.iter().all(|value| value.is_finite()));
        assert!(result.command.torques.iter().all(|value| value.abs() <= 1.0));
    }

    #[test]
    fn every_goal_directed_command_receives_safety_evidence() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let (baseline, residual) = zero_pair(morphology);
        let result = pipeline.authorize(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
            1.0,
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert_eq!(result.command.num_actuators(), morphology.num_actuators());
        assert!(!result.report.safety.rejected);
    }

    #[test]
    fn fallback_still_crosses_final_projector() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let mut fallback = HumanoidCommand::zero_for(morphology.num_actuators());
        fallback.torques[0] = 4.0;
        let result = pipeline.authorize_fallback(
            &fallback,
            &state,
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert!(result.report.magnitude_clips > 0 || result.report.slew_clips > 0);
        assert!(result.command.torques[0].abs() <= 1.0);
    }
}
