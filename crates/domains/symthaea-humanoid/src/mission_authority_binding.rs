// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact institutional mission binding for canonical humanoid skills.
//!
//! This module is deliberately an adapter over the generic Symthaea authority
//! lineage. It does not define a second robot-specific grant format and it does
//! not dispatch motors. Its job is narrower: prove that an already Xenia-verified
//! capability is scoped to one exact canonical humanoid skill, robot resource,
//! mission task, issuer/subject, executor workload, and current bounded lifetime.
//!
//! A successful [`VerifiedHumanoidMissionBinding`] is still not a one-use
//! execution reservation. Negative-authority facts and use/risk accounting must
//! be consumed transactionally by the Action Runtime before consequential
//! dispatch. Keeping that distinction explicit prevents a scope proof from
//! silently becoming physical authority.

use symthaea_authority::{
    CAPABILITY_GRANT_SCHEMA_VERSION, CapabilityGrant, Digest32, Operation, PrincipalId,
    ResourceRef, TaskId,
};
use symthaea_authority_time::VerifiedAuthorityTime;
use symthaea_xenia_authority::VerifiedXeniaCapability;

use crate::canonical_skill_contract::{
    CanonicalHumanoidSkillContractError, validate_canonical_humanoid_skill_contract,
};
use crate::skill_runtime::{HumanoidLocomotionMode, HumanoidSkillContract, HumanoidSkillIntent};

const HUMANOID_SKILL_CONTRACT_DIGEST_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.skill-contract.v1\0";

/// Policy/profile inputs supplied by the trusted humanoid integration boundary.
/// These are expectations, not authority by themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidMissionAuthorityProfileV1 {
    pub expected_issuer: PrincipalId,
    pub expected_subject: PrincipalId,
    pub expected_runtime_audience: PrincipalId,
    pub mission_task: TaskId,
    pub robot_resource: ResourceRef,
    pub expected_world_digest: Option<Digest32>,
    pub expected_workload_digest: Digest32,
    pub expected_xenia_session_id: [u8; 16],
    /// V1 defaults to direct execution: a robot-bound mission grant must not
    /// retain authority to create further delegation edges.
    pub require_non_delegable: bool,
}

impl HumanoidMissionAuthorityProfileV1 {
    pub fn validate(&self) -> bool {
        !self.expected_issuer.0.trim().is_empty()
            && !self.expected_subject.0.trim().is_empty()
            && !self.expected_runtime_audience.0.trim().is_empty()
            && !self.mission_task.0.trim().is_empty()
            && !self.robot_resource.0.trim().is_empty()
            && self.expected_workload_digest.0 != [0; 32]
            && self.expected_xenia_session_id != [0; 16]
    }
}

/// Process-local proof that one exact verified Xenia capability is bound to one
/// exact canonical humanoid mission scope.
///
/// Intentionally not Clone/Copy/Serde. It owns the affine Xenia proof and exact
/// grant so callers cannot swap authority state after binding.
#[derive(Debug)]
pub struct VerifiedHumanoidMissionBinding {
    xenia_capability: VerifiedXeniaCapability,
    grant: CapabilityGrant,
    contract_digest: Digest32,
    operation: Operation,
    valid_until_unix_s: u64,
}

impl VerifiedHumanoidMissionBinding {
    pub fn grant(&self) -> &CapabilityGrant {
        &self.grant
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.grant.digest()
    }

    pub fn contract_digest(&self) -> Digest32 {
        self.contract_digest
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn valid_until_unix_s(&self) -> u64 {
        self.valid_until_unix_s
    }

    pub fn authority_state_sequence(&self) -> u64 {
        self.xenia_capability.authority_state_sequence()
    }

    pub fn authority_state_digest(&self) -> Digest32 {
        self.xenia_capability.authority_state_digest()
    }

    /// Recheck the proof-owned authority-state freshness and bounded lifetime
    /// under a new trusted-time fact for the same exact grant.
    ///
    /// This still does not reserve a use or evaluate Action Runtime accounting.
    pub fn revalidate_current(
        &self,
        authority_time: &VerifiedAuthorityTime,
    ) -> Result<(), HumanoidMissionBindingError> {
        authority_time
            .require_subject(self.grant.digest().0)
            .map_err(|_| HumanoidMissionBindingError::AuthorityTimeInvalid)?;
        self.xenia_capability
            .authority_state()
            .ensure_fresh(&self.grant, authority_time)
            .map_err(|_| HumanoidMissionBindingError::AuthorityStateInvalid)?;
        let now = authority_time
            .conservative_now_unix_s()
            .map_err(|_| HumanoidMissionBindingError::AuthorityTimeInvalid)?;
        if now > self.valid_until_unix_s {
            return Err(HumanoidMissionBindingError::BindingExpired);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidMissionBindingError {
    InvalidProfile,
    NonCanonicalContract,
    ContractEncoding,
    UnsupportedGrantSchema,
    GrantDigestMismatch,
    IssuerMismatch,
    SubjectMismatch,
    AudienceMismatch,
    TaskMismatch,
    ResourceScopeMismatch,
    OperationScopeMismatch,
    PlanDigestMismatch,
    WorldDigestMismatch,
    UnboundedGrantExpiry,
    ZeroUseBudget,
    DelegationStillPermitted,
    WorkloadMismatch,
    XeniaSessionMismatch,
    AuthorityTimeInvalid,
    AuthorityStateInvalid,
    BindingExpired,
}

impl std::fmt::Display for HumanoidMissionBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidProfile => "humanoid mission authority profile is invalid",
            Self::NonCanonicalContract => "humanoid skill contract is not canonical compiler output",
            Self::ContractEncoding => "canonical humanoid skill contract could not be encoded",
            Self::UnsupportedGrantSchema => "capability grant schema is unsupported",
            Self::GrantDigestMismatch => "verified Xenia capability does not bind this exact grant",
            Self::IssuerMismatch => "capability issuer differs from the mission profile",
            Self::SubjectMismatch => "capability subject differs from the mission profile",
            Self::AudienceMismatch => "capability audience differs from the humanoid runtime",
            Self::TaskMismatch => "capability task differs from the exact mission task",
            Self::ResourceScopeMismatch => "capability does not contain exactly the robot resource",
            Self::OperationScopeMismatch => "capability operation is not exactly the whole humanoid skill",
            Self::PlanDigestMismatch => "capability plan digest does not bind the exact skill contract",
            Self::WorldDigestMismatch => "capability world digest differs from the expected mission world",
            Self::UnboundedGrantExpiry => "physical mission capability must have a bounded expiry",
            Self::ZeroUseBudget => "physical mission capability has no usable execution budget",
            Self::DelegationStillPermitted => "direct humanoid mission capability retains delegation authority",
            Self::WorkloadMismatch => "verified Xenia capability belongs to a different executor workload",
            Self::XeniaSessionMismatch => "verified Xenia capability belongs to a different Xenia session",
            Self::AuthorityTimeInvalid => "trusted authority-time fact is invalid or stale",
            Self::AuthorityStateInvalid => "verified authority-state fact is invalid or stale",
            Self::BindingExpired => "humanoid mission binding has expired",
        })
    }
}

impl std::error::Error for HumanoidMissionBindingError {}

/// Domain-separated commitment to the complete canonical semantic skill
/// contract. The authority grant uses this as its exact `plan_digest` in V1.
pub fn humanoid_skill_contract_digest_v1(
    contract: &HumanoidSkillContract,
) -> Result<Digest32, HumanoidMissionBindingError> {
    validate_canonical_humanoid_skill_contract(contract).map_err(map_canonical_error)?;
    let encoded = serde_json::to_vec(contract)
        .map_err(|_| HumanoidMissionBindingError::ContractEncoding)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(HUMANOID_SKILL_CONTRACT_DIGEST_DOMAIN_V1);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(Digest32(*hasher.finalize().as_bytes()))
}

/// One closed semantic operation per whole skill. Composite `Carry` is not
/// represented as independent Walk + Grasp operations, preventing authority
/// decomposition from the composite mission binding.
pub fn humanoid_skill_operation_v1(intent: HumanoidSkillIntent) -> Operation {
    let name = match intent {
        HumanoidSkillIntent::Stand => "robotics.humanoid.skill.stand.v1",
        HumanoidSkillIntent::Locomote {
            mode: HumanoidLocomotionMode::Walk,
            ..
        } => "robotics.humanoid.skill.walk.v1",
        HumanoidSkillIntent::Locomote {
            mode: HumanoidLocomotionMode::Run,
            ..
        } => "robotics.humanoid.skill.run.v1",
        HumanoidSkillIntent::Reach { .. } => "robotics.humanoid.skill.reach.v1",
        HumanoidSkillIntent::Grasp { .. } => "robotics.humanoid.skill.grasp.v1",
        HumanoidSkillIntent::Carry { .. } => "robotics.humanoid.skill.carry.v1",
        HumanoidSkillIntent::AssistHuman { .. } => "robotics.humanoid.skill.assist-human.v1",
    };
    Operation(name.to_string())
}

/// Validate grant scope without treating the result as live execution authority.
/// This pure helper is useful for tests and policy tooling; the positive runtime
/// transition is [`bind_verified_humanoid_mission`].
pub fn validate_humanoid_mission_scope_v1(
    grant: &CapabilityGrant,
    contract: &HumanoidSkillContract,
    profile: &HumanoidMissionAuthorityProfileV1,
) -> Result<Digest32, HumanoidMissionBindingError> {
    if !profile.validate() {
        return Err(HumanoidMissionBindingError::InvalidProfile);
    }
    if grant.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION {
        return Err(HumanoidMissionBindingError::UnsupportedGrantSchema);
    }
    let contract_digest = humanoid_skill_contract_digest_v1(contract)?;
    if grant.issuer != profile.expected_issuer {
        return Err(HumanoidMissionBindingError::IssuerMismatch);
    }
    if grant.subject != profile.expected_subject {
        return Err(HumanoidMissionBindingError::SubjectMismatch);
    }
    if grant.audience.as_ref() != Some(&profile.expected_runtime_audience) {
        return Err(HumanoidMissionBindingError::AudienceMismatch);
    }
    if grant.task.as_ref() != Some(&profile.mission_task) {
        return Err(HumanoidMissionBindingError::TaskMismatch);
    }
    if grant.resources.len() != 1 || !grant.resources.contains(&profile.robot_resource) {
        return Err(HumanoidMissionBindingError::ResourceScopeMismatch);
    }
    let operation = humanoid_skill_operation_v1(contract.intent);
    if grant.operations.len() != 1 || !grant.operations.contains(&operation) {
        return Err(HumanoidMissionBindingError::OperationScopeMismatch);
    }
    if grant.plan_digest != Some(contract_digest) {
        return Err(HumanoidMissionBindingError::PlanDigestMismatch);
    }
    if grant.world_digest != profile.expected_world_digest {
        return Err(HumanoidMissionBindingError::WorldDigestMismatch);
    }
    if grant.expires_at_unix_s.is_none() || grant.expires_at_unix_s == Some(0) {
        return Err(HumanoidMissionBindingError::UnboundedGrantExpiry);
    }
    if grant.max_uses == 0 {
        return Err(HumanoidMissionBindingError::ZeroUseBudget);
    }
    if profile.require_non_delegable && grant.delegation_depth_remaining != 0 {
        return Err(HumanoidMissionBindingError::DelegationStillPermitted);
    }
    Ok(contract_digest)
}

/// Bind one affine Xenia verification result to one exact canonical humanoid
/// mission scope under fresh trusted time.
///
/// The resulting value is a prerequisite for later execution admission, not an
/// execution permit. Action Runtime reservation + negative-fact evaluation still
/// must occur before any physical dispatch.
pub fn bind_verified_humanoid_mission_v1(
    xenia_capability: VerifiedXeniaCapability,
    grant: CapabilityGrant,
    contract: &HumanoidSkillContract,
    profile: &HumanoidMissionAuthorityProfileV1,
    authority_time: &VerifiedAuthorityTime,
) -> Result<VerifiedHumanoidMissionBinding, HumanoidMissionBindingError> {
    let contract_digest = validate_humanoid_mission_scope_v1(&grant, contract, profile)?;
    if xenia_capability.grant_digest() != grant.digest() {
        return Err(HumanoidMissionBindingError::GrantDigestMismatch);
    }
    if xenia_capability.workload_digest() != profile.expected_workload_digest {
        return Err(HumanoidMissionBindingError::WorkloadMismatch);
    }
    if xenia_capability.session_id() != profile.expected_xenia_session_id {
        return Err(HumanoidMissionBindingError::XeniaSessionMismatch);
    }
    authority_time
        .require_subject(grant.digest().0)
        .map_err(|_| HumanoidMissionBindingError::AuthorityTimeInvalid)?;
    xenia_capability
        .authority_state()
        .ensure_fresh(&grant, authority_time)
        .map_err(|_| HumanoidMissionBindingError::AuthorityStateInvalid)?;
    let now = authority_time
        .conservative_now_unix_s()
        .map_err(|_| HumanoidMissionBindingError::AuthorityTimeInvalid)?;
    let grant_expiry = grant
        .expires_at_unix_s
        .ok_or(HumanoidMissionBindingError::UnboundedGrantExpiry)?;
    let valid_until_unix_s = grant_expiry.min(xenia_capability.expires_at_unix_s());
    if now > valid_until_unix_s {
        return Err(HumanoidMissionBindingError::BindingExpired);
    }

    Ok(VerifiedHumanoidMissionBinding {
        operation: humanoid_skill_operation_v1(contract.intent),
        xenia_capability,
        grant,
        contract_digest,
        valid_until_unix_s,
    })
}

fn map_canonical_error(_: CanonicalHumanoidSkillContractError) -> HumanoidMissionBindingError {
    HumanoidMissionBindingError::NonCanonicalContract
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_runtime::{
        HumanoidSkillQualificationSet, compile_humanoid_skill_contract,
    };
    use crate::types::{ActuationMode, HumanoidTask};
    use symthaea_authority::{AuthorityEpoch, RiskBudget};

    fn carry_contract() -> HumanoidSkillContract {
        let qualifications = HumanoidSkillQualificationSet::new(vec![
            HumanoidQualificationSubject::new(
                HumanoidMorphology::Dexterous53,
                HumanoidTask::Grasp,
                ActuationMode::NormalizedTorque,
                "mission-binding-test-v1",
            ),
            HumanoidQualificationSubject::new(
                HumanoidMorphology::Dexterous53,
                HumanoidTask::Walk,
                ActuationMode::NormalizedTorque,
                "mission-binding-test-v1",
            ),
        ]);
        compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.1,
                manipulation_speed_mps: 0.1,
                retention_force_n: 15.0,
                resulting_total_payload_kg: 3.0,
            },
            &qualifications,
        )
        .unwrap()
    }

    fn profile() -> HumanoidMissionAuthorityProfileV1 {
        HumanoidMissionAuthorityProfileV1 {
            expected_issuer: PrincipalId("institution:warehouse-1".into()),
            expected_subject: PrincipalId("agent:mission-executor-17".into()),
            expected_runtime_audience: PrincipalId("robot:17/runtime".into()),
            mission_task: TaskId("mission:pallet-381".into()),
            robot_resource: ResourceRef("robot:17".into()),
            expected_world_digest: Some(Digest32([7; 32])),
            expected_workload_digest: Digest32([8; 32]),
            expected_xenia_session_id: [9; 16],
            require_non_delegable: true,
        }
    }

    fn grant(contract: &HumanoidSkillContract) -> CapabilityGrant {
        let profile = profile();
        let mut grant = CapabilityGrant::new(
            "grant:robot-17:pallet-381",
            profile.expected_issuer.clone(),
            profile.expected_subject.clone(),
            AuthorityEpoch(12),
        );
        grant.audience = Some(profile.expected_runtime_audience.clone());
        grant.task = Some(profile.mission_task.clone());
        grant.resources.insert(profile.robot_resource.clone());
        grant.operations.insert(humanoid_skill_operation_v1(contract.intent));
        grant.plan_digest = Some(humanoid_skill_contract_digest_v1(contract).unwrap());
        grant.world_digest = profile.expected_world_digest;
        grant.expires_at_unix_s = Some(2_000_000_000);
        grant.max_uses = 1;
        grant.delegation_depth_remaining = 0;
        grant.risk_budget = RiskBudget {
            mutation_units: 1,
            irreversible_units: 1,
            external_disclosure_bytes: 0,
            monetary_microunits: 0,
        };
        grant
    }

    #[test]
    fn exact_canonical_carry_scope_is_accepted() {
        let contract = carry_contract();
        assert_eq!(
            validate_humanoid_mission_scope_v1(&grant(&contract), &contract, &profile()),
            Ok(humanoid_skill_contract_digest_v1(&contract).unwrap())
        );
    }

    #[test]
    fn carry_grant_cannot_be_decomposed_into_walk_authority() {
        let contract = carry_contract();
        let mut grant = grant(&contract);
        grant.operations.clear();
        grant.operations.insert(Operation("robotics.humanoid.skill.walk.v1".into()));
        assert_eq!(
            validate_humanoid_mission_scope_v1(&grant, &contract, &profile()),
            Err(HumanoidMissionBindingError::OperationScopeMismatch)
        );
    }

    #[test]
    fn changed_contract_is_not_covered_by_old_plan_digest() {
        let contract = carry_contract();
        let grant = grant(&contract);
        let mut changed = contract.clone();
        changed.intent = HumanoidSkillIntent::Carry {
            mode: HumanoidLocomotionMode::Walk,
            horizontal_speed_mps: 0.5,
            turn_rate_rad_s: 0.1,
            manipulation_speed_mps: 0.1,
            retention_force_n: 15.0,
            resulting_total_payload_kg: 3.0,
        };
        assert_eq!(
            validate_humanoid_mission_scope_v1(&grant, &changed, &profile()),
            Err(HumanoidMissionBindingError::NonCanonicalContract)
        );
    }

    #[test]
    fn malformed_composite_contract_cannot_receive_mission_binding() {
        let mut contract = carry_contract();
        contract.requirements.pop();
        let mut grant = grant(&carry_contract());
        grant.plan_digest = Some(Digest32([3; 32]));
        assert_eq!(
            validate_humanoid_mission_scope_v1(&grant, &contract, &profile()),
            Err(HumanoidMissionBindingError::NonCanonicalContract)
        );
    }

    #[test]
    fn physical_mission_scope_requires_bounded_expiry() {
        let contract = carry_contract();
        let mut grant = grant(&contract);
        grant.expires_at_unix_s = None;
        assert_eq!(
            validate_humanoid_mission_scope_v1(&grant, &contract, &profile()),
            Err(HumanoidMissionBindingError::UnboundedGrantExpiry)
        );
    }

    #[test]
    fn direct_robot_profile_rejects_remaining_delegation_depth() {
        let contract = carry_contract();
        let mut grant = grant(&contract);
        grant.delegation_depth_remaining = 1;
        assert_eq!(
            validate_humanoid_mission_scope_v1(&grant, &contract, &profile()),
            Err(HumanoidMissionBindingError::DelegationStillPermitted)
        );
    }
}
