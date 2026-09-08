// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authoritative humanoid execution boundary.
//!
//! Cognitive and learned systems may propose motion, but they do not own the
//! transition to actuator authority. This module composes deterministic
//! hierarchical control, explicitly bounded pre-actuation modifiers, typed
//! restrictive goal authority, and the final morphology-aware safety projector.
//!
//! Protective fall/recovery behavior is deliberately independent of goal
//! authority: revoking a task must not revoke the torque needed to brace or
//! recover. Protective output still crosses final physical safety projection.
//!
//! Backend-specific watchdog, e-stop, over-current, calibration, bus-fault, and
//! servo enforcement remains an additional independent responsibility of the
//! hardware abstraction layer.

use crate::contact::ContactFrame;
use crate::dynamics::RigidBodyDynamicsProvider;
use crate::floating_base::FloatingBaseDynamicsProvider;
use crate::full_dynamics::FullRigidBodyDynamicsProvider;
use crate::hierarchical::{HierarchicalControlReport, HierarchicalHumanoidController};
use crate::morphology::HumanoidMorphology;
use crate::safety::{HumanoidSafetyProjector, SafetyProjectionReport};
use crate::terrain::TerrainProbe;
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

/// Failure while modifying a prepared, still-non-authoritative command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedCommandError {
    ActuatorCountMismatch,
    NonFiniteRetainedPolicy,
    NonFiniteProtectiveCommand,
    ExplorationNoiseCountMismatch,
    NonFiniteExplorationNoise,
}

impl std::fmt::Display for PreparedCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ActuatorCountMismatch => "protective command actuator count mismatch",
            Self::NonFiniteRetainedPolicy => "protective retained-policy fraction is non-finite",
            Self::NonFiniteProtectiveCommand => "protective command contains a non-finite value",
            Self::ExplorationNoiseCountMismatch => "exploration noise actuator count mismatch",
            Self::NonFiniteExplorationNoise => "exploration noise contains a non-finite value",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PreparedCommandError {}

/// A synthesized command that has **not** crossed the authority/safety boundary.
///
/// The underlying goal-directed command is deliberately private. Exploration
/// modifies only that goal-directed component. Protective fall/recovery output
/// is retained separately so task/cognitive/epistemic revocation cannot suppress
/// the minimum force needed for deterministic protective behavior.
#[derive(Debug, Clone)]
pub struct HumanoidPreparedCommand {
    goal_command: HumanoidCommand,
    hierarchy: HierarchicalControlReport,
    protective_override: Option<(HumanoidCommand, f32)>,
    exploration_applied: bool,
}

impl HumanoidPreparedCommand {
    /// Evidence from deterministic hierarchical synthesis is inspectable before
    /// finalization, but the raw motor command is not.
    pub fn hierarchy_report(&self) -> &HierarchicalControlReport {
        &self.hierarchy
    }

    pub fn goal_control_effort(&self) -> f32 {
        self.goal_command.control_effort()
    }

    /// Register a deterministic protective behavior to be blended only after
    /// goal authority has been applied. Invalid inputs are rejected without
    /// mutating the prepared command.
    pub fn apply_protective_override(
        &mut self,
        protective: &HumanoidCommand,
        retained_policy: f32,
    ) -> Result<(), PreparedCommandError> {
        if protective.num_actuators() != self.goal_command.num_actuators() {
            return Err(PreparedCommandError::ActuatorCountMismatch);
        }
        if !retained_policy.is_finite() {
            return Err(PreparedCommandError::NonFiniteRetainedPolicy);
        }
        if protective.torques.iter().any(|value| !value.is_finite()) {
            return Err(PreparedCommandError::NonFiniteProtectiveCommand);
        }
        self.protective_override = Some((protective.clone(), retained_policy.clamp(0.0, 1.0)));
        Ok(())
    }

    /// Apply exploration only to the goal-directed component while it is still
    /// non-authoritative. Exact actuator cardinality is required so noise cannot
    /// truncate a frame.
    pub fn apply_exploration_noise(
        &mut self,
        noise: &[f32],
    ) -> Result<(), PreparedCommandError> {
        if noise.len() != self.goal_command.num_actuators() {
            return Err(PreparedCommandError::ExplorationNoiseCountMismatch);
        }
        if noise.iter().any(|value| !value.is_finite()) {
            return Err(PreparedCommandError::NonFiniteExplorationNoise);
        }
        for (value, noise) in self.goal_command.torques.iter_mut().zip(noise.iter()) {
            *value = (*value + *noise).clamp(-1.0, 1.0);
        }
        self.exploration_applied = true;
        Ok(())
    }
}

/// Evidence emitted for every command that crosses the humanoid authority boundary.
#[derive(Debug, Clone)]
pub struct HumanoidExecutionReport {
    pub hierarchy: HierarchicalControlReport,
    pub safety: SafetyProjectionReport,
    /// Per-source goal authority inputs used for this command.
    pub authority: HumanoidAuthorityEnvelope,
    /// Most-restrictive goal-authority multiplier.
    pub authority_scale: f32,
    /// Whether independent protective behavior was blended after goal authority.
    pub protective_override_applied: bool,
    /// Whether exploration modified the goal-directed component before authority.
    pub exploration_applied: bool,
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

    /// Deterministic synthesis on the default estimated-contact/flat-terrain path.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        task: HumanoidTask,
        state: &HumanoidState,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
    ) -> HumanoidPreparedCommand {
        let (goal_command, hierarchy) = self.hierarchy.synthesize(
            task,
            state,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
        );
        HumanoidPreparedCommand {
            goal_command,
            hierarchy,
            protective_override: None,
            exploration_applied: false,
        }
    }

    /// Deterministic synthesis using the embodiment's real contact, terrain, and
    /// dynamics providers. This is the path training/HIL/hardware should use when
    /// richer evidence is available.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_with_environment<T>(
        &self,
        task: HumanoidTask,
        state: &HumanoidState,
        contacts: &ContactFrame,
        environment: &T,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
    ) -> HumanoidPreparedCommand
    where
        T: TerrainProbe
            + RigidBodyDynamicsProvider
            + FullRigidBodyDynamicsProvider
            + FloatingBaseDynamicsProvider
            + ?Sized,
    {
        let (goal_command, hierarchy) = self.hierarchy.synthesize_with_environment(
            task,
            state,
            contacts,
            environment,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
        );
        HumanoidPreparedCommand {
            goal_command,
            hierarchy,
            protective_override: None,
            exploration_applied: false,
        }
    }

    /// Consume a prepared command and cross typed goal authority + final physical
    /// projection. Protective output is blended after goal authority so loss of
    /// task/cognitive/epistemic authority cannot disable deterministic bracing.
    pub fn finalize_prepared(
        &mut self,
        mut prepared: HumanoidPreparedCommand,
        state: &HumanoidState,
        authority: HumanoidAuthorityEnvelope,
        actuation_mode: ActuationMode,
        dt: f64,
    ) -> HumanoidExecutionResult {
        let authority_scale = authority.effective_scale();
        if authority_scale < 1.0 {
            for torque in &mut prepared.goal_command.torques {
                *torque *= authority_scale;
            }
        }

        let protective_override_applied = prepared.protective_override.is_some();
        if let Some((protective, retained_policy)) = prepared.protective_override.take() {
            for (goal, protective) in prepared
                .goal_command
                .torques
                .iter_mut()
                .zip(protective.torques.iter())
            {
                *goal = (retained_policy * *goal + *protective).clamp(-1.0, 1.0);
            }
        }

        let projected = self
            .safety
            .project(&prepared.goal_command, state, actuation_mode, dt);

        HumanoidExecutionResult {
            command: projected.command,
            report: HumanoidExecutionReport {
                hierarchy: prepared.hierarchy,
                safety: projected.report,
                authority,
                authority_scale,
                protective_override_applied,
                exploration_applied: prepared.exploration_applied,
            },
        }
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

    /// Convenience path for ordinary callers with no pre-finalization modifiers.
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
        let prepared = self.prepare(
            task,
            state,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
        );
        self.finalize_prepared(prepared, state, authority, actuation_mode, dt)
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
    fn prepared_command_requires_exact_exploration_cardinality() {
        let morphology = HumanoidMorphology::Dmc21;
        let pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let (baseline, residual) = zero_pair(morphology);
        let mut prepared = pipeline.prepare(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
        );
        let err = prepared.apply_exploration_noise(&[0.0; 2]).unwrap_err();
        assert_eq!(err, PreparedCommandError::ExplorationNoiseCountMismatch);
    }

    #[test]
    fn protective_behavior_survives_goal_authority_revocation() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let (baseline, residual) = zero_pair(morphology);
        let mut prepared = pipeline.prepare(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
        );
        let mut protective = HumanoidCommand::zero_for(morphology.num_actuators());
        protective.torques[0] = 0.1;
        prepared.apply_protective_override(&protective, 0.0).unwrap();
        let result = pipeline.finalize_prepared(
            prepared,
            &state,
            HumanoidAuthorityEnvelope::default(),
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert_eq!(result.report.authority_scale, 0.0);
        assert!(result.report.protective_override_applied);
        assert!(result.command.torques[0] > 0.0);
    }

    #[test]
    fn protective_and_exploration_modifiers_are_evidenced() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let (baseline, residual) = zero_pair(morphology);
        let mut prepared = pipeline.prepare(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
        );
        let protective = HumanoidCommand::zero_for(morphology.num_actuators());
        prepared.apply_protective_override(&protective, 0.35).unwrap();
        prepared
            .apply_exploration_noise(&vec![0.0; morphology.num_actuators()])
            .unwrap();
        let result = pipeline.finalize_prepared(
            prepared,
            &state,
            HumanoidAuthorityEnvelope::fully_admitted(),
            ActuationMode::NormalizedTorque,
            0.025,
        );
        assert!(result.report.protective_override_applied);
        assert!(result.report.exploration_applied);
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
        assert!(!result.report.protective_override_applied);
        assert!(!result.report.exploration_applied);
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
