// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed mission-authority requirements derived from humanoid skill contracts.
//!
//! This module deliberately does **not** implement authorization. It freezes the
//! humanoid-side translation from one exact closed semantic skill contract into
//! the task/resource/audience/operation bindings that a future adapter to the
//! generic `symthaea-authority` kernel must verify.
//!
//! In particular:
//!
//! ```text
//! HumanoidMissionAuthorityRequirementProfileV1 != CapabilityGrant
//! HumanoidMissionAuthorityRequirementProfileV1 != VerifiedMissionAuthority
//! mission world snapshot binding != current operating-environment evidence
//! ```
//!
//! A positive production witness must later be minted from verifier-owned
//! generic authority state, not from deserializing this ordinary data object.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::skill_runtime::{
    HumanoidLocomotionMode, HumanoidSkillContract, HumanoidSkillIntent, HumanoidSkillInvariant,
    HumanoidSkillPrecondition, HumanoidSkillRecoveryPolicy, HumanoidSkillRequirement,
    HumanoidSkillRequirementRole,
};
use crate::types::HumanoidTask;

pub const HUMANOID_MISSION_AUTHORITY_PROFILE_SCHEMA_VERSION: u16 = 1;
const PROFILE_DOMAIN: &[u8] = b"symthaea-humanoid-mission-authority-profile-v1";
const CONTRACT_DOMAIN: &[u8] = b"symthaea-humanoid-skill-contract-v1";

/// Stable humanoid operation names intended for a future adapter into the
/// generic `symthaea_authority::Operation` namespace.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum HumanoidMissionOperationV1 {
    PostureStand,
    LocomotionWalk,
    LocomotionRun,
    ManipulationReach,
    ManipulationGrasp,
    HumanInteractionAssist,
}

impl HumanoidMissionOperationV1 {
    pub const fn as_operation_name(self) -> &'static str {
        match self {
            Self::PostureStand => "robotics.humanoid.posture.stand",
            Self::LocomotionWalk => "robotics.humanoid.locomotion.walk",
            Self::LocomotionRun => "robotics.humanoid.locomotion.run",
            Self::ManipulationReach => "robotics.humanoid.manipulation.reach",
            Self::ManipulationGrasp => "robotics.humanoid.manipulation.grasp",
            Self::HumanInteractionAssist => "robotics.humanoid.human_interaction.assist",
        }
    }

    const fn stable_tag(self) -> u8 {
        match self {
            Self::PostureStand => 1,
            Self::LocomotionWalk => 2,
            Self::LocomotionRun => 3,
            Self::ManipulationReach => 4,
            Self::ManipulationGrasp => 5,
            Self::HumanInteractionAssist => 6,
        }
    }
}

/// Ordinary, serializable requirements for one exact humanoid mission exercise.
///
/// The profile carries no positive authority. A later production adapter must
/// derive the profile again from the exact live skill contract, verify a generic
/// grant and its proof-owned current authority state, and only then mint an
/// opaque process-local mission witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidMissionAuthorityRequirementProfileV1 {
    schema_version: u16,
    grant_subject_principal: String,
    runtime_audience_principal: String,
    mission_task_id: String,
    robot_resource: String,
    skill_contract_digest: [u8; 32],
    required_operations: Vec<HumanoidMissionOperationV1>,
    required_plan_digest: Option<[u8; 32]>,
    mission_world_snapshot_digest: Option<[u8; 32]>,
    requires_bounded_expiry: bool,
    profile_digest: [u8; 32],
}

impl HumanoidMissionAuthorityRequirementProfileV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn from_contract(
        contract: &HumanoidSkillContract,
        grant_subject_principal: impl Into<String>,
        runtime_audience_principal: impl Into<String>,
        mission_task_id: impl Into<String>,
        robot_resource: impl Into<String>,
        required_plan_digest: Option<[u8; 32]>,
        mission_world_snapshot_digest: Option<[u8; 32]>,
    ) -> Result<Self, HumanoidMissionAuthorityProfileError> {
        validate_contract_shape(contract)?;

        let grant_subject_principal = grant_subject_principal.into();
        let runtime_audience_principal = runtime_audience_principal.into();
        let mission_task_id = mission_task_id.into();
        let robot_resource = robot_resource.into();
        for value in [
            grant_subject_principal.as_str(),
            runtime_audience_principal.as_str(),
            mission_task_id.as_str(),
            robot_resource.as_str(),
        ] {
            if !valid_identifier(value) {
                return Err(HumanoidMissionAuthorityProfileError::InvalidIdentifier);
            }
        }
        if required_plan_digest.is_some_and(is_zero_digest)
            || mission_world_snapshot_digest.is_some_and(is_zero_digest)
        {
            return Err(HumanoidMissionAuthorityProfileError::ZeroOptionalDigest);
        }

        let required_operations = required_operations(contract)?;
        let skill_contract_digest = humanoid_skill_contract_digest_v1(contract)?;
        let mut profile = Self {
            schema_version: HUMANOID_MISSION_AUTHORITY_PROFILE_SCHEMA_VERSION,
            grant_subject_principal,
            runtime_audience_principal,
            mission_task_id,
            robot_resource,
            skill_contract_digest,
            required_operations,
            required_plan_digest,
            mission_world_snapshot_digest,
            requires_bounded_expiry: true,
            profile_digest: [0; 32],
        };
        profile.profile_digest = profile.compute_digest();
        profile.validate()?;
        Ok(profile)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn grant_subject_principal(&self) -> &str {
        &self.grant_subject_principal
    }

    pub fn runtime_audience_principal(&self) -> &str {
        &self.runtime_audience_principal
    }

    pub fn mission_task_id(&self) -> &str {
        &self.mission_task_id
    }

    pub fn robot_resource(&self) -> &str {
        &self.robot_resource
    }

    pub const fn skill_contract_digest(&self) -> [u8; 32] {
        self.skill_contract_digest
    }

    pub fn required_operations(&self) -> &[HumanoidMissionOperationV1] {
        &self.required_operations
    }

    pub const fn required_plan_digest(&self) -> Option<[u8; 32]> {
        self.required_plan_digest
    }

    /// This binds mission issuance to a declared world snapshot only. It does
    /// not prove the current operating environment still matches at execution.
    pub const fn mission_world_snapshot_digest(&self) -> Option<[u8; 32]> {
        self.mission_world_snapshot_digest
    }

    pub const fn requires_bounded_expiry(&self) -> bool {
        self.requires_bounded_expiry
    }

    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    /// Revalidate ordinary persisted profile data. Passing this check proves
    /// structural/content consistency only; it does not create authority.
    pub fn validate(&self) -> Result<(), HumanoidMissionAuthorityProfileError> {
        if self.schema_version != HUMANOID_MISSION_AUTHORITY_PROFILE_SCHEMA_VERSION {
            return Err(HumanoidMissionAuthorityProfileError::UnsupportedSchema);
        }
        for value in [
            self.grant_subject_principal.as_str(),
            self.runtime_audience_principal.as_str(),
            self.mission_task_id.as_str(),
            self.robot_resource.as_str(),
        ] {
            if !valid_identifier(value) {
                return Err(HumanoidMissionAuthorityProfileError::InvalidIdentifier);
            }
        }
        if is_zero_digest(self.skill_contract_digest) {
            return Err(HumanoidMissionAuthorityProfileError::InvalidContractDigest);
        }
        if self.required_plan_digest.is_some_and(is_zero_digest)
            || self.mission_world_snapshot_digest.is_some_and(is_zero_digest)
        {
            return Err(HumanoidMissionAuthorityProfileError::ZeroOptionalDigest);
        }
        if !self.requires_bounded_expiry {
            return Err(HumanoidMissionAuthorityProfileError::UnboundedExpiry);
        }
        if self.required_operations.is_empty()
            || self
                .required_operations
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(HumanoidMissionAuthorityProfileError::NonCanonicalOperations);
        }
        if self.compute_digest() != self.profile_digest {
            return Err(HumanoidMissionAuthorityProfileError::ProfileDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut transcript = Transcript::new(PROFILE_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.string(&self.grant_subject_principal);
        transcript.string(&self.runtime_audience_principal);
        transcript.string(&self.mission_task_id);
        transcript.string(&self.robot_resource);
        transcript.digest(self.skill_contract_digest);
        transcript.u32(self.required_operations.len() as u32);
        for operation in &self.required_operations {
            transcript.byte(operation.stable_tag());
            transcript.string(operation.as_operation_name());
        }
        transcript.optional_digest(self.required_plan_digest);
        transcript.optional_digest(self.mission_world_snapshot_digest);
        transcript.byte(u8::from(self.requires_bounded_expiry));
        *transcript.finish().as_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidMissionAuthorityProfileError {
    UnsupportedSchema,
    InvalidIdentifier,
    InvalidSkillContract,
    UnsupportedRequirement,
    ContractSerialization,
    InvalidContractDigest,
    ZeroOptionalDigest,
    NonCanonicalOperations,
    UnboundedExpiry,
    ProfileDigestMismatch,
}

/// Domain-separated implementation-lineage identity for the exact closed skill
/// contract. V1 deliberately commits the exact `serde_json` bytes produced by
/// this Rust schema. It is not claimed as a universal cross-language encoding.
pub fn humanoid_skill_contract_digest_v1(
    contract: &HumanoidSkillContract,
) -> Result<[u8; 32], HumanoidMissionAuthorityProfileError> {
    validate_contract_shape(contract)?;
    let encoded = serde_json::to_vec(contract)
        .map_err(|_| HumanoidMissionAuthorityProfileError::ContractSerialization)?;
    let mut transcript = Transcript::new(CONTRACT_DOMAIN);
    transcript.u32(encoded.len() as u32);
    transcript.bytes(&encoded);
    Ok(*transcript.finish().as_bytes())
}

fn required_operations(
    contract: &HumanoidSkillContract,
) -> Result<Vec<HumanoidMissionOperationV1>, HumanoidMissionAuthorityProfileError> {
    let mut operations = BTreeSet::new();
    for requirement in &contract.requirements {
        operations.insert(operation_for_requirement(requirement)?);
    }
    if operations.is_empty() {
        return Err(HumanoidMissionAuthorityProfileError::InvalidSkillContract);
    }
    Ok(operations.into_iter().collect())
}

fn operation_for_requirement(
    requirement: &HumanoidSkillRequirement,
) -> Result<HumanoidMissionOperationV1, HumanoidMissionAuthorityProfileError> {
    match (requirement.role, requirement.task) {
        (HumanoidSkillRequirementRole::Posture, HumanoidTask::Stand) => {
            Ok(HumanoidMissionOperationV1::PostureStand)
        }
        (HumanoidSkillRequirementRole::Locomotion, HumanoidTask::Walk) => {
            Ok(HumanoidMissionOperationV1::LocomotionWalk)
        }
        (HumanoidSkillRequirementRole::Locomotion, HumanoidTask::Run) => {
            Ok(HumanoidMissionOperationV1::LocomotionRun)
        }
        (HumanoidSkillRequirementRole::Manipulation, HumanoidTask::Reach) => {
            Ok(HumanoidMissionOperationV1::ManipulationReach)
        }
        (HumanoidSkillRequirementRole::Manipulation, HumanoidTask::Grasp) => {
            Ok(HumanoidMissionOperationV1::ManipulationGrasp)
        }
        (HumanoidSkillRequirementRole::HumanInteraction, HumanoidTask::Reach) => {
            Ok(HumanoidMissionOperationV1::HumanInteractionAssist)
        }
        _ => Err(HumanoidMissionAuthorityProfileError::UnsupportedRequirement),
    }
}

fn validate_contract_shape(
    contract: &HumanoidSkillContract,
) -> Result<(), HumanoidMissionAuthorityProfileError> {
    if !contract.intent.validate()
        || !valid_identifier(&contract.backend_profile_id)
        || contract.requirements.is_empty()
        || !contract
            .requirements
            .iter()
            .all(|requirement| requirement.request.validate())
        || !contract
            .preconditions
            .contains(&HumanoidSkillPrecondition::GoalExecutionAuthority)
        || !contract
            .invariants
            .contains(&HumanoidSkillInvariant::CapabilityEnvelopeRespected)
        || !contract
            .invariants
            .contains(&HumanoidSkillInvariant::QualificationSubjectStable)
        || !contract
            .invariants
            .contains(&HumanoidSkillInvariant::ProtectiveBehaviorMayPreemptGoal)
    {
        return Err(HumanoidMissionAuthorityProfileError::InvalidSkillContract);
    }

    let shape_ok = match contract.intent {
        HumanoidSkillIntent::Stand => {
            contract.requirements.len() == 1
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::Posture,
                    HumanoidTask::Stand,
                )
                && contract.recovery == HumanoidSkillRecoveryPolicy::Replan
        }
        HumanoidSkillIntent::Locomote { mode, .. } => {
            contract.requirements.len() == 1
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::Locomotion,
                    mode.task(),
                )
                && contract.recovery == HumanoidSkillRecoveryPolicy::Replan
        }
        HumanoidSkillIntent::Reach { .. } => {
            contract.requirements.len() == 1
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::Manipulation,
                    HumanoidTask::Reach,
                )
                && contract.recovery == HumanoidSkillRecoveryPolicy::Replan
        }
        HumanoidSkillIntent::Grasp {
            resulting_total_payload_kg,
            ..
        } => {
            contract.requirements.len() == 1
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::Manipulation,
                    HumanoidTask::Grasp,
                )
                && contract.recovery == HumanoidSkillRecoveryPolicy::SecureLoadThenReplan
                && (resulting_total_payload_kg == 0.0
                    || (contract
                        .preconditions
                        .contains(&HumanoidSkillPrecondition::LoadRetentionEvidence)
                        && contract
                            .invariants
                            .contains(&HumanoidSkillInvariant::LoadRetentionMaintained)))
        }
        HumanoidSkillIntent::Carry { mode, .. } => {
            contract.requirements.len() == 2
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::Manipulation,
                    HumanoidTask::Grasp,
                )
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::Locomotion,
                    mode.task(),
                )
                && contract
                    .preconditions
                    .contains(&HumanoidSkillPrecondition::LoadRetentionEvidence)
                && contract
                    .invariants
                    .contains(&HumanoidSkillInvariant::LoadRetentionMaintained)
                && contract.recovery == HumanoidSkillRecoveryPolicy::SecureLoadThenReplan
        }
        HumanoidSkillIntent::AssistHuman { .. } => {
            contract.requirements.len() == 1
                && has_exact_requirement(
                    contract,
                    HumanoidSkillRequirementRole::HumanInteraction,
                    HumanoidTask::Reach,
                )
                && contract
                    .preconditions
                    .contains(&HumanoidSkillPrecondition::HumanProximityEvidence)
                && contract
                    .preconditions
                    .contains(&HumanoidSkillPrecondition::HumanContactConsent)
                && contract
                    .invariants
                    .contains(&HumanoidSkillInvariant::HumanContactConsentMaintained)
                && contract.recovery == HumanoidSkillRecoveryPolicy::WithdrawThenReplan
        }
    };

    if !shape_ok {
        return Err(HumanoidMissionAuthorityProfileError::InvalidSkillContract);
    }

    // Every requirement pair must have an explicit v1 operation mapping. This
    // intentionally makes future skill/role variants fail closed until reviewed.
    for requirement in &contract.requirements {
        operation_for_requirement(requirement)?;
    }
    Ok(())
}

fn has_exact_requirement(
    contract: &HumanoidSkillContract,
    role: HumanoidSkillRequirementRole,
    task: HumanoidTask,
) -> bool {
    contract
        .requirements
        .iter()
        .filter(|requirement| requirement.role == role && requirement.task == task)
        .count()
        == 1
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn is_zero_digest(digest: [u8; 32]) -> bool {
    digest == [0; 32]
}

struct Transcript(blake3::Hasher);

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u32).to_be_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    fn byte(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.0.update(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.update(&value.to_be_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.0.update(value);
    }

    fn string(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.bytes(value.as_bytes());
    }

    fn digest(&mut self, value: [u8; 32]) {
        self.bytes(&value);
    }

    fn optional_digest(&mut self, value: Option<[u8; 32]>) {
        match value {
            Some(value) => {
                self.byte(1);
                self.digest(value);
            }
            None => self.byte(0),
        }
    }

    fn finish(self) -> blake3::Hash {
        self.0.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_runtime::{
        HumanoidSkillQualificationSet, compile_humanoid_skill_contract,
    };
    use crate::types::ActuationMode;

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "mission-profile-test-backend-v1",
        )
    }

    fn qualifications(tasks: &[HumanoidTask]) -> HumanoidSkillQualificationSet {
        HumanoidSkillQualificationSet::new(tasks.iter().copied().map(subject).collect())
    }

    fn profile(contract: &HumanoidSkillContract) -> HumanoidMissionAuthorityRequirementProfileV1 {
        HumanoidMissionAuthorityRequirementProfileV1::from_contract(
            contract,
            "principal:robot-17",
            "runtime:robot-17/humanoid",
            "mission:warehouse-381",
            "robot:17",
            Some([7; 32]),
            Some([9; 32]),
        )
        .unwrap()
    }

    #[test]
    fn carry_preserves_closed_manipulation_and_locomotion_requirements() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.2,
                manipulation_speed_mps: 0.1,
                retention_force_n: 30.0,
                resulting_total_payload_kg: 4.0,
            },
            &qualifications(&[HumanoidTask::Grasp, HumanoidTask::Walk]),
        )
        .unwrap();
        let profile = profile(&contract);
        assert_eq!(profile.required_operations().len(), 2);
        assert!(profile
            .required_operations()
            .contains(&HumanoidMissionOperationV1::ManipulationGrasp));
        assert!(profile
            .required_operations()
            .contains(&HumanoidMissionOperationV1::LocomotionWalk));
        assert!(profile.requires_bounded_expiry());
        profile.validate().unwrap();
    }

    #[test]
    fn skill_demand_change_changes_contract_and_profile_identity() {
        let qualifications = qualifications(&[HumanoidTask::Walk]);
        let first = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.1,
            },
            &qualifications,
        )
        .unwrap();
        let second = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.1,
            },
            &qualifications,
        )
        .unwrap();
        let first_profile = profile(&first);
        let second_profile = profile(&second);
        assert_ne!(
            first_profile.skill_contract_digest(),
            second_profile.skill_contract_digest()
        );
        assert_ne!(first_profile.profile_digest(), second_profile.profile_digest());
    }

    #[test]
    fn mission_binding_changes_profile_without_changing_skill_contract() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Stand,
            &qualifications(&[HumanoidTask::Stand]),
        )
        .unwrap();
        let first = profile(&contract);
        let second = HumanoidMissionAuthorityRequirementProfileV1::from_contract(
            &contract,
            "principal:robot-17",
            "runtime:robot-17/humanoid",
            "mission:warehouse-382",
            "robot:17",
            Some([7; 32]),
            Some([9; 32]),
        )
        .unwrap();
        assert_eq!(first.skill_contract_digest(), second.skill_contract_digest());
        assert_ne!(first.profile_digest(), second.profile_digest());
    }

    #[test]
    fn assist_human_is_not_laundered_into_generic_reach_authority() {
        let reach = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Reach {
                end_effector_speed_mps: 0.1,
            },
            &qualifications(&[HumanoidTask::Reach]),
        )
        .unwrap();
        let assist = compile_humanoid_skill_contract(
            HumanoidSkillIntent::AssistHuman {
                end_effector_speed_mps: 0.1,
                human_contact_force_n: 5.0,
            },
            &qualifications(&[HumanoidTask::Reach]),
        )
        .unwrap();
        assert_eq!(
            profile(&reach).required_operations(),
            &[HumanoidMissionOperationV1::ManipulationReach]
        );
        assert_eq!(
            profile(&assist).required_operations(),
            &[HumanoidMissionOperationV1::HumanInteractionAssist]
        );
    }

    #[test]
    fn malformed_raw_contract_cannot_be_profiled_as_compiled_skill() {
        let mut contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Stand,
            &qualifications(&[HumanoidTask::Stand]),
        )
        .unwrap();
        contract
            .preconditions
            .retain(|value| *value != HumanoidSkillPrecondition::GoalExecutionAuthority);
        assert_eq!(
            HumanoidMissionAuthorityRequirementProfileV1::from_contract(
                &contract,
                "principal:robot-17",
                "runtime:robot-17/humanoid",
                "mission:warehouse-381",
                "robot:17",
                None,
                None,
            )
            .unwrap_err(),
            HumanoidMissionAuthorityProfileError::InvalidSkillContract
        );
    }

    #[test]
    fn serialized_profile_remains_data_and_revalidates_its_content_identity() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Stand,
            &qualifications(&[HumanoidTask::Stand]),
        )
        .unwrap();
        let profile = profile(&contract);
        let encoded = serde_json::to_vec(&profile).unwrap();
        let decoded: HumanoidMissionAuthorityRequirementProfileV1 =
            serde_json::from_slice(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.profile_digest(), profile.profile_digest());
    }
}
