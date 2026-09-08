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

/// Independent sources that may restrict goal-directed motor authority.
///
/// Every source is interpreted in `[0, 1]` and composition is most-restrictive
/// wins. Non-finite values fail closed to zero. No single source can grant more
/// authority than another source has admitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumanoidAuthorityEnvelope {
    /// Human/operator or authenticated task authority.
    pub operator: f32,
    /// Maximum authority supported by current qualification evidence.
    pub qualification: f32,
    /// Authority admitted by current body/hardware health before final projection.
    pub physical: f32,
    /// Authority admitted by perception/state-estimation confidence.
    pub epistemic: f32,
    /// Cognitive subsystem restriction (for example degraded cognition/Φ).
    pub cognitive: f32,
}

impl HumanoidAuthorityEnvelope {
    /// Explicit fully-admitted envelope. Callers should use this only when every
    /// source has independently been established; `Default` is fail-closed.
    pub const fn fully_admitted() -> Self {
        Self {
            operator: 1.0,
            qualification: 1.0,
            physical: 1.0,
            epistemic: 1.0,
            cognitive: 1.0,
        }
    }

    /// Legacy scalar compatibility: a single scalar restricts every source.
    pub const fn from_scalar(scale: f32) -> Self {
        Self {
            operator: scale,
            qualification: scale,
            physical: scale,
            epistemic: scale,
            cognitive: scale,
        }
    }

    /// Most-restrictive effective authority. Invalid inputs fail closed.
    pub fn effective_scale(self) -> f32 {
        [
            self.operator,
            self.qualification,
            self.physical,
            self.epistemic,
            self.cognitive,
        ]
        .into_iter()
        .map(restrictive_unit_interval)
        .fold(1.0, f32::min)
    }
}

impl Default for HumanoidAuthorityEnvelope {
    fn default() -> Self {
        Self {
            operator: 0.0,
            qualification: 0.0,
            physical: 0.0,
            epistemic: 0.0,
            cognitive: 0.0,
        }
    }
}

fn restrictive_unit_interval(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Evidence emitted for every command that crosses the humanoid authority boundary.
#[derive(Debug, Clone)]
pub struct HumanoidExecutionReport {
    pub hierarchy: HierarchicalControlReport,
    pub safety: SafetyProjectionReport,
    /// Per-source authority inputs used for this command.
    pub authority: HumanoidAuthorityEnvelope,
    /// Most-restrictive authority multiplier applied after deterministic synthesis.
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

    /// Compatibility entry point for callers that still provide one scalar.
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
        self.authorize_with_authority(
            task,
            state,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
            HumanoidAuthorityEnvelope::from_scalar(authority_scale),
            actuation_mode,
            dt,
        )
    }

    /// Admit a baseline + learned residual through deterministic whole-body
    /// synthesis and final command projection using independent authority sources.
    #[allow(clippy::too_many_arguments)]
    pub fn authorize_with_authority(
        &mut self,
        task: HumanoidTask,
        state: &HumanoidState,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
        authority: HumanoidAuthorityEnvelope,
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

        let authority_scale = authority.effective_scale();
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
                authority,
                authority_scale,
            },
        }
    }

    /// Route a platform minimum-safe fallback through the same final physical
    /// command projector. Safe fallback intentionally bypasses goal-directed
    /// hierarchy: it is the independent behavior used when goal authority has
    /// already been revoked.
    ///
    /// Entering fallback revokes the previous goal-directed command history
    /// before projection. This prevents a slew limiter from preserving stale
    /// unsafe authority while the minimum-safe behavior takes over.
    pub fn authorize_fallback(
        &mut self,
        fallback: &HumanoidCommand,
        state: &HumanoidState,
        actuation_mode: ActuationMode,
        dt: f64,
    ) -> crate::safety::ProjectedCommand {
        self.safety.reset();
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
    fn default_authority_is_fail_closed() {
        assert_eq!(HumanoidAuthorityEnvelope::default().effective_scale(), 0.0);
    }

    #[test]
    fn most_restrictive_source_wins() {
        let authority = HumanoidAuthorityEnvelope {
            operator: 1.0,
            qualification: 0.8,
            physical: 0.6,
            epistemic: 0.4,
            cognitive: 0.9,
        };
        assert!((authority.effective_scale() - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn non_finite_authority_source_fails_restrictive() {
        let authority = HumanoidAuthorityEnvelope {
            cognitive: f32::NAN,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        };
        assert_eq!(authority.effective_scale(), 0.0);
    }

    #[test]
    fn non_finite_legacy_authority_fails_restrictive() {
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
        let result = pipeline.authorize_with_authority(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
            HumanoidAuthorityEnvelope::fully_admitted(),
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert_eq!(result.command.num_actuators(), morphology.num_actuators());
        assert!(!result.report.safety.rejected);
        assert_eq!(result.report.authority_scale, 1.0);
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
