// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical validation for caller-transportable humanoid skill contracts.
//!
//! `HumanoidSkillContract` is intentionally serializable for planning, audit, and
//! transport. That means possession of a structurally plausible value cannot prove
//! that the authoritative skill compiler actually produced it. In particular, a
//! caller must not be able to remove one requirement from a composite `Carry`
//! contract and then ask downstream authority code to bless the weakened object.
//!
//! This module makes the existing compiler the canonical owner of contract shape:
//! it reconstructs the exact qualification subjects implied by the contract's
//! morphology / task / actuation / backend identity, recompiles the intent, and
//! requires semantic equality with the supplied contract.

use crate::qualification::HumanoidQualificationSubject;
use crate::skill_runtime::{
    HumanoidLocomotionMode, HumanoidSkillCompileError, HumanoidSkillContract,
    HumanoidSkillIntent, HumanoidSkillQualificationSet, compile_humanoid_skill_contract,
};
use crate::types::HumanoidTask;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalHumanoidSkillContractError {
    Compile(HumanoidSkillCompileError),
    NonCanonicalContract,
}

impl std::fmt::Display for CanonicalHumanoidSkillContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(error) => write!(f, "canonical skill recompilation failed: {error:?}"),
            Self::NonCanonicalContract => {
                f.write_str("supplied humanoid skill contract differs from canonical compiler output")
            }
        }
    }
}

impl std::error::Error for CanonicalHumanoidSkillContractError {}

impl From<HumanoidSkillCompileError> for CanonicalHumanoidSkillContractError {
    fn from(value: HumanoidSkillCompileError) -> Self {
        Self::Compile(value)
    }
}

/// Re-run the authoritative compiler from the minimum subject identities implied
/// by a transported contract and require the exact canonical result.
///
/// This is deliberately reject-only. It never repairs, fills, clamps, reorders,
/// or otherwise normalizes a malformed contract into something executable.
pub fn validate_canonical_humanoid_skill_contract(
    contract: &HumanoidSkillContract,
) -> Result<(), CanonicalHumanoidSkillContractError> {
    let qualifications = HumanoidSkillQualificationSet::new(
        required_tasks(contract.intent)
            .into_iter()
            .map(|task| {
                HumanoidQualificationSubject::new(
                    contract.morphology,
                    task,
                    contract.actuation_mode,
                    contract.backend_profile_id.clone(),
                )
            })
            .collect(),
    );

    let canonical = compile_humanoid_skill_contract(contract.intent, &qualifications)?;
    if &canonical != contract {
        return Err(CanonicalHumanoidSkillContractError::NonCanonicalContract);
    }
    Ok(())
}

/// Exact benchmark/control subjects required to canonically compile one semantic
/// skill. Composite requirements stay closed as one set; callers cannot select a
/// subset and still pass canonical validation for the original intent.
fn required_tasks(intent: HumanoidSkillIntent) -> Vec<HumanoidTask> {
    match intent {
        HumanoidSkillIntent::Stand => vec![HumanoidTask::Stand],
        HumanoidSkillIntent::Locomote { mode, .. } => vec![mode.task()],
        HumanoidSkillIntent::Reach { .. } => vec![HumanoidTask::Reach],
        HumanoidSkillIntent::Grasp { .. } => vec![HumanoidTask::Grasp],
        HumanoidSkillIntent::Carry { mode, .. } => {
            vec![HumanoidTask::Grasp, locomotion_task(mode)]
        }
        HumanoidSkillIntent::AssistHuman { .. } => vec![HumanoidTask::Reach],
    }
}

const fn locomotion_task(mode: HumanoidLocomotionMode) -> HumanoidTask {
    mode.task()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::skill_runtime::{
        HumanoidSkillPrecondition, HumanoidSkillRequirementRole,
        compile_humanoid_skill_contract,
    };
    use crate::types::ActuationMode;

    fn qualifications(tasks: &[HumanoidTask]) -> HumanoidSkillQualificationSet {
        HumanoidSkillQualificationSet::new(
            tasks
                .iter()
                .copied()
                .map(|task| {
                    HumanoidQualificationSubject::new(
                        HumanoidMorphology::Dexterous53,
                        task,
                        ActuationMode::NormalizedTorque,
                        "canonical-skill-test-v1",
                    )
                })
                .collect(),
        )
    }

    fn carry() -> HumanoidSkillContract {
        compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.2,
                manipulation_speed_mps: 0.1,
                retention_force_n: 20.0,
                resulting_total_payload_kg: 4.0,
            },
            &qualifications(&[HumanoidTask::Grasp, HumanoidTask::Walk]),
        )
        .unwrap()
    }

    #[test]
    fn compiler_output_is_canonical() {
        validate_canonical_humanoid_skill_contract(&carry()).unwrap();
    }

    #[test]
    fn carry_cannot_drop_locomotion_requirement() {
        let mut contract = carry();
        contract
            .requirements
            .retain(|requirement| requirement.role != HumanoidSkillRequirementRole::Locomotion);
        assert_eq!(
            validate_canonical_humanoid_skill_contract(&contract),
            Err(CanonicalHumanoidSkillContractError::NonCanonicalContract)
        );
    }

    #[test]
    fn carry_cannot_change_one_compiled_demand() {
        let mut contract = carry();
        contract.requirements[0].request.object_contact_force_n += 1.0;
        assert_eq!(
            validate_canonical_humanoid_skill_contract(&contract),
            Err(CanonicalHumanoidSkillContractError::NonCanonicalContract)
        );
    }

    #[test]
    fn assist_human_cannot_drop_contact_consent_precondition() {
        let mut contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::AssistHuman {
                end_effector_speed_mps: 0.1,
                human_contact_force_n: 5.0,
            },
            &qualifications(&[HumanoidTask::Reach]),
        )
        .unwrap();
        contract
            .preconditions
            .retain(|precondition| *precondition != HumanoidSkillPrecondition::HumanContactConsent);
        assert_eq!(
            validate_canonical_humanoid_skill_contract(&contract),
            Err(CanonicalHumanoidSkillContractError::NonCanonicalContract)
        );
    }

    #[test]
    fn caller_cannot_change_recovery_semantics() {
        let mut contract = carry();
        contract.recovery = crate::skill_runtime::HumanoidSkillRecoveryPolicy::Replan;
        assert_eq!(
            validate_canonical_humanoid_skill_contract(&contract),
            Err(CanonicalHumanoidSkillContractError::NonCanonicalContract)
        );
    }
}
