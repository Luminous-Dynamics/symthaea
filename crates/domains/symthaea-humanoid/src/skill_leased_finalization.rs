// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Keep a guarded skill epoch borrowed through humanoid command finalization.
//!
//! This is the first local bridge from the one-epoch semantic permit lineage to
//! the exact pipeline-finalized command lineage. The result retains the borrowed
//! skill lease, so safe Rust cannot advance the guarded skill owner while the
//! finalized productive command is still being carried toward a later physical
//! admission boundary.
//!
//! Composite and intentional human-contact skills are deliberately not collapsed
//! into one DMC benchmark task. They require dedicated coordinators/controllers
//! before they can enter this path.

use std::num::NonZeroU64;

use crate::execution::{HumanoidAuthorityEnvelope, HumanoidExecutionPipeline};
use crate::finalized_command_commitment::FinalizedCommandCommitmentError;
use crate::pipeline_finalized_command::{
    PipelineFinalizedHumanoidCommand, finalize_and_commit_humanoid_command,
};
use crate::skill_execution_lease::HumanoidSkillExecutionLease;
use crate::skill_runtime::{
    HumanoidSkillContract, HumanoidSkillIntent, HumanoidSkillRequirementRole,
};
use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask};

/// Process-local productive command whose semantic validation epoch remains
/// borrowed from the guarded skill owner.
///
/// Intentionally not `Clone`, `Copy`, `Serialize`, or `Deserialize`.
#[derive(Debug)]
pub struct SkillLeasedPipelineFinalizedHumanoidCommand<'a> {
    lease: HumanoidSkillExecutionLease<'a>,
    finalized: PipelineFinalizedHumanoidCommand,
}

impl<'a> SkillLeasedPipelineFinalizedHumanoidCommand<'a> {
    pub const fn validation_epoch(&self) -> NonZeroU64 {
        self.lease.validation_epoch()
    }

    pub fn contract(&self) -> &'a HumanoidSkillContract {
        self.lease.contract()
    }

    pub fn finalized(&self) -> &PipelineFinalizedHumanoidCommand {
        &self.finalized
    }

    /// Consume the semantic wrapper while preserving the lease for a later
    /// physical-admission owner that wants to keep the guarded epoch borrowed.
    pub fn into_parts(
        self,
    ) -> (
        HumanoidSkillExecutionLease<'a>,
        PipelineFinalizedHumanoidCommand,
    ) {
        (self.lease, self.finalized)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillLeasedFinalizationError {
    PipelineMorphologyMismatch,
    RequirementCountMismatch,
    RequirementRoleMismatch,
    RequirementTaskMismatch,
    CompositeSkillRequiresCoordinator,
    HumanInteractionRequiresSpecializedController,
    FinalizedCommand(FinalizedCommandCommitmentErrorKind),
}

/// Stable, copyable classification of command-commitment failures. The detailed
/// source is intentionally not promoted into semantic authority state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalizedCommandCommitmentErrorKind {
    InvalidCommand,
    LengthOverflow,
}

impl From<&FinalizedCommandCommitmentError> for FinalizedCommandCommitmentErrorKind {
    fn from(value: &FinalizedCommandCommitmentError) -> Self {
        match value {
            FinalizedCommandCommitmentError::InvalidCommand(_) => Self::InvalidCommand,
            FinalizedCommandCommitmentError::LengthOverflow => Self::LengthOverflow,
        }
    }
}

impl std::fmt::Display for SkillLeasedFinalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::PipelineMorphologyMismatch => {
                "guarded skill morphology differs from execution pipeline morphology"
            }
            Self::RequirementCountMismatch => {
                "atomic productive skill does not contain exactly one requirement"
            }
            Self::RequirementRoleMismatch => {
                "atomic productive skill requirement has the wrong semantic role"
            }
            Self::RequirementTaskMismatch => {
                "atomic productive skill requirement has the wrong benchmark task"
            }
            Self::CompositeSkillRequiresCoordinator => {
                "composite skill requires a dedicated multi-controller coordinator"
            }
            Self::HumanInteractionRequiresSpecializedController => {
                "intentional human interaction requires a specialized contact controller"
            }
            Self::FinalizedCommand(_) => "pipeline-finalized command commitment failed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SkillLeasedFinalizationError {}

/// Synthesize and finalize one atomic productive skill while retaining the
/// guarded semantic epoch borrow in the returned value.
///
/// This remains below full physical admission: the supplied authority envelope
/// is still the compatibility restriction type and must eventually be replaced
/// by a verifier-owned join before HAL/ROS dispatch.
#[allow(clippy::too_many_arguments)]
pub fn finalize_atomic_skill_under_lease<'a>(
    lease: HumanoidSkillExecutionLease<'a>,
    pipeline: &mut HumanoidExecutionPipeline,
    state: &HumanoidState,
    baseline: &HumanoidCommand,
    learned_residual: &HumanoidCommand,
    baseline_weight: f32,
    free_energy: f64,
    authority: HumanoidAuthorityEnvelope,
    dt: f64,
) -> Result<SkillLeasedPipelineFinalizedHumanoidCommand<'a>, SkillLeasedFinalizationError> {
    let contract = lease.contract();
    if pipeline.morphology() != contract.morphology {
        return Err(SkillLeasedFinalizationError::PipelineMorphologyMismatch);
    }
    let task = atomic_productive_task(contract)?;
    let prepared = pipeline.prepare(
        task,
        state,
        baseline,
        learned_residual,
        baseline_weight,
        free_energy,
    );
    let finalized = finalize_and_commit_humanoid_command(
        pipeline,
        prepared,
        state,
        authority,
        contract.actuation_mode,
        dt,
    )
    .map_err(|source| {
        SkillLeasedFinalizationError::FinalizedCommand(
            FinalizedCommandCommitmentErrorKind::from(&source),
        )
    })?;

    Ok(SkillLeasedPipelineFinalizedHumanoidCommand { lease, finalized })
}

fn atomic_productive_task(
    contract: &HumanoidSkillContract,
) -> Result<HumanoidTask, SkillLeasedFinalizationError> {
    let (task, role) = match contract.intent {
        HumanoidSkillIntent::Stand => (HumanoidTask::Stand, HumanoidSkillRequirementRole::Posture),
        HumanoidSkillIntent::Locomote { mode, .. } => {
            (mode.task(), HumanoidSkillRequirementRole::Locomotion)
        }
        HumanoidSkillIntent::Reach { .. } => {
            (HumanoidTask::Reach, HumanoidSkillRequirementRole::Manipulation)
        }
        HumanoidSkillIntent::Grasp { .. } => {
            (HumanoidTask::Grasp, HumanoidSkillRequirementRole::Manipulation)
        }
        HumanoidSkillIntent::Carry { .. } => {
            return Err(SkillLeasedFinalizationError::CompositeSkillRequiresCoordinator);
        }
        HumanoidSkillIntent::AssistHuman { .. } => {
            return Err(
                SkillLeasedFinalizationError::HumanInteractionRequiresSpecializedController,
            );
        }
    };

    if contract.requirements.len() != 1 {
        return Err(SkillLeasedFinalizationError::RequirementCountMismatch);
    }
    let requirement = &contract.requirements[0];
    if requirement.role != role {
        return Err(SkillLeasedFinalizationError::RequirementRoleMismatch);
    }
    if requirement.task != task {
        return Err(SkillLeasedFinalizationError::RequirementTaskMismatch);
    }
    Ok(task)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_request::HumanoidCapabilityRequest;
    use crate::morphology::HumanoidMorphology;
    use crate::skill_runtime::{
        HumanoidLocomotionMode, HumanoidSkillInvariant, HumanoidSkillPrecondition,
        HumanoidSkillRecoveryPolicy, HumanoidSkillRequirement,
    };
    use crate::types::ActuationMode;

    fn contract(
        intent: HumanoidSkillIntent,
        role: HumanoidSkillRequirementRole,
        task: HumanoidTask,
    ) -> HumanoidSkillContract {
        HumanoidSkillContract {
            intent,
            morphology: HumanoidMorphology::Dmc21,
            actuation_mode: ActuationMode::NormalizedTorque,
            backend_profile_id: "leased-finalization-test-v1".into(),
            requirements: vec![HumanoidSkillRequirement {
                role,
                task,
                request: HumanoidCapabilityRequest::stationary(17),
            }],
            preconditions: vec![HumanoidSkillPrecondition::GoalExecutionAuthority],
            invariants: vec![HumanoidSkillInvariant::CapabilityEnvelopeRespected],
            recovery: HumanoidSkillRecoveryPolicy::Replan,
        }
    }

    #[test]
    fn atomic_stand_maps_to_exact_benchmark_task() {
        let contract = contract(
            HumanoidSkillIntent::Stand,
            HumanoidSkillRequirementRole::Posture,
            HumanoidTask::Stand,
        );
        assert_eq!(atomic_productive_task(&contract), Ok(HumanoidTask::Stand));
    }

    #[test]
    fn requirement_task_mismatch_fails_closed() {
        let contract = contract(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.2,
                turn_rate_rad_s: 0.0,
            },
            HumanoidSkillRequirementRole::Locomotion,
            HumanoidTask::Run,
        );
        assert_eq!(
            atomic_productive_task(&contract),
            Err(SkillLeasedFinalizationError::RequirementTaskMismatch)
        );
    }

    #[test]
    fn carry_is_not_silently_decomposed_into_one_task() {
        let contract = contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.2,
                turn_rate_rad_s: 0.0,
                manipulation_speed_mps: 0.1,
                retention_force_n: 5.0,
                resulting_total_payload_kg: 1.0,
            },
            HumanoidSkillRequirementRole::Locomotion,
            HumanoidTask::Walk,
        );
        assert_eq!(
            atomic_productive_task(&contract),
            Err(SkillLeasedFinalizationError::CompositeSkillRequiresCoordinator)
        );
    }

    #[test]
    fn human_contact_is_not_silently_collapsed_into_reach() {
        let contract = contract(
            HumanoidSkillIntent::AssistHuman {
                end_effector_speed_mps: 0.1,
                human_contact_force_n: 5.0,
            },
            HumanoidSkillRequirementRole::HumanInteraction,
            HumanoidTask::Reach,
        );
        assert_eq!(
            atomic_productive_task(&contract),
            Err(SkillLeasedFinalizationError::HumanInteractionRequiresSpecializedController)
        );
    }
}
