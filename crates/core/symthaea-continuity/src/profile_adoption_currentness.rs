// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only currentness revalidation for persisted verifier-profile adoptions.
//!
//! A persisted lineage head is intentionally too small to reconstruct runtime
//! authority: it does not retain the adoption validity interval, evidence-class
//! ceiling, or exact scope. This module therefore defines a non-authorizing,
//! replayable registry record that retains the exact candidate transition,
//! predecessor transition, root-provisioning snapshot, and local authority-grant
//! snapshot required to re-run policy at use time.
//!
//! Core theorem:
//!
//! `persisted write != current policy != cryptographic authority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::ValidatedContinuityContractV1;
use crate::profile_adoption::{
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionTransitionV1,
};
use crate::profile_adoption_admission::{
    VerifierProfileAdoptionAdmissionError, VerifierProfileAdoptionAdmissionPolicyV1,
    VerifierProfileAdoptionHeadV1,
};
use crate::profile_adoption_authority::{
    AuthorityGrantedVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantError,
    VerifierAdoptionAuthorityGrantV1, bind_policy_checked_adoption_to_authority_grant,
};
use crate::profile_adoption_commit::VerifierAdoptionAuthorityRootSnapshotV1;
use crate::verifier::VerifierProfileV1;
use crate::witness::EvidenceClass;

const RECORD_DOMAIN: &[u8] = b"symthaea.continuity.verifier-adoption.registry-record.v1\0";
pub const VERIFIER_PROFILE_ADOPTION_REGISTRY_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-profile-adoption-registry-record-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionRegistryRecordIdV1([u8; 32]);
impl VerifierProfileAdoptionRegistryRecordIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierAdoptionAuthorityRootRecordV1 {
    authority_subject: String,
    root_id: String,
    root_digest: [u8; 32],
    provisioning_epoch: u64,
}
impl VerifierAdoptionAuthorityRootRecordV1 {
    fn from_snapshot(snapshot: &VerifierAdoptionAuthorityRootSnapshotV1) -> Self {
        Self {
            authority_subject: snapshot.authority_subject().to_owned(),
            root_id: snapshot.root_id().to_owned(),
            root_digest: snapshot.root_digest(),
            provisioning_epoch: snapshot.epoch(),
        }
    }
    fn matches(&self, snapshot: &VerifierAdoptionAuthorityRootSnapshotV1) -> bool {
        self.authority_subject == snapshot.authority_subject()
            && self.root_id == snapshot.root_id()
            && self.root_digest == snapshot.root_digest()
            && self.provisioning_epoch == snapshot.epoch()
    }
    pub fn authority_subject(&self) -> &str { &self.authority_subject }
    pub fn root_id(&self) -> &str { &self.root_id }
    pub fn root_digest(&self) -> [u8; 32] { self.root_digest }
    pub fn provisioning_epoch(&self) -> u64 { self.provisioning_epoch }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierAdoptionAuthorityGrantRecordV1 {
    grant_id_text: String,
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    verifier_role_id: String,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: VerifierAdoptionScopeV1,
}
impl VerifierAdoptionAuthorityGrantRecordV1 {
    fn from_grant(grant: &VerifierAdoptionAuthorityGrantV1) -> Self {
        Self {
            grant_id_text: grant.grant_id_text().to_owned(),
            authority_subject: grant.authority_subject().to_owned(),
            authority_root_id: grant.authority_root_id().to_owned(),
            authority_root_digest: grant.authority_root_digest(),
            verifier_role_id: grant.verifier_role_id().to_owned(),
            grant_epoch: grant.grant_epoch(),
            maximum_evidence_class: grant.maximum_evidence_class(),
            allowed_scope: grant.allowed_scope().clone(),
        }
    }
    fn matches(&self, grant: &VerifierAdoptionAuthorityGrantV1) -> bool {
        self.grant_id_text == grant.grant_id_text()
            && self.authority_subject == grant.authority_subject()
            && self.authority_root_id == grant.authority_root_id()
            && self.authority_root_digest == grant.authority_root_digest()
            && self.verifier_role_id == grant.verifier_role_id()
            && self.grant_epoch == grant.grant_epoch()
            && self.maximum_evidence_class == grant.maximum_evidence_class()
            && self.allowed_scope == *grant.allowed_scope()
    }
    pub fn grant_id_text(&self) -> &str { &self.grant_id_text }
    pub fn authority_subject(&self) -> &str { &self.authority_subject }
    pub fn authority_root_id(&self) -> &str { &self.authority_root_id }
    pub fn authority_root_digest(&self) -> [u8; 32] { self.authority_root_digest }
    pub fn verifier_role_id(&self) -> &str { &self.verifier_role_id }
    pub fn grant_epoch(&self) -> u64 { self.grant_epoch }
    pub fn maximum_evidence_class(&self) -> EvidenceClass { self.maximum_evidence_class }
    pub fn allowed_scope(&self) -> &VerifierAdoptionScopeV1 { &self.allowed_scope }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionRegistryRecordV1 {
    schema_version: String,
    transition: VerifierProfileAdoptionTransitionV1,
    predecessor_transition: Option<VerifierProfileAdoptionTransitionV1>,
    authority_root: VerifierAdoptionAuthorityRootRecordV1,
    authority_grant: VerifierAdoptionAuthorityGrantRecordV1,
    record_id: VerifierProfileAdoptionRegistryRecordIdV1,
}
impl VerifierProfileAdoptionRegistryRecordV1 {
    pub fn new_uncommitted_projection(
        granted: &AuthorityGrantedVerifierProfileAdoptionV1,
        authority_root: &VerifierAdoptionAuthorityRootSnapshotV1,
        predecessor_transition: Option<&VerifierProfileAdoptionTransitionV1>,
    ) -> Result<Self, VerifierProfileAdoptionCurrentnessError> {
        let checked = granted.checked();
        let transition = checked.transition();
        transition.validate()?;
        let subject = transition.subject();
        if subject.authority_subject() != authority_root.authority_subject()
            || subject.authority_root_id() != authority_root.root_id()
            || subject.authority_root_digest() != authority_root.root_digest()
        {
            return Err(VerifierProfileAdoptionCurrentnessError::RootProjectionMismatch);
        }
        let predecessor_transition = match checked.expected_predecessor_head() {
            VerifierProfileAdoptionHeadV1::Uninitialized => {
                if predecessor_transition.is_some()
                    || transition.generation() != 1
                    || transition.predecessor() != VerifierProfileAdoptionPredecessorV1::Bootstrap
                {
                    return Err(VerifierProfileAdoptionCurrentnessError::UnexpectedPredecessorRecord);
                }
                None
            }
            expected @ VerifierProfileAdoptionHeadV1::Current(_) => {
                let predecessor = predecessor_transition
                    .ok_or(VerifierProfileAdoptionCurrentnessError::MissingPredecessorRecord)?;
                predecessor.validate()?;
                let derived = VerifierProfileAdoptionHeadV1::from_transition(predecessor)?;
                if &derived != expected {
                    return Err(VerifierProfileAdoptionCurrentnessError::PredecessorRecordMismatch);
                }
                Some(predecessor.clone())
            }
        };
        let authority_root_record = VerifierAdoptionAuthorityRootRecordV1::from_snapshot(authority_root);
        let authority_grant_record = VerifierAdoptionAuthorityGrantRecordV1::from_grant(granted.authority_grant());
        let record_id = VerifierProfileAdoptionRegistryRecordIdV1(hash_record(
            transition, predecessor_transition.as_ref(), &authority_root_record, &authority_grant_record,
        )?);
        Ok(Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_REGISTRY_RECORD_SCHEMA_V1.to_owned(),
            transition: transition.clone(),
            predecessor_transition,
            authority_root: authority_root_record,
            authority_grant: authority_grant_record,
            record_id,
        })
    }

    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionCurrentnessError> {
        if self.schema_version != VERIFIER_PROFILE_ADOPTION_REGISTRY_RECORD_SCHEMA_V1 {
            return Err(VerifierProfileAdoptionCurrentnessError::UnsupportedRecordSchema(self.schema_version.clone()));
        }
        self.transition.validate()?;
        validate_record_text("authority_subject", &self.authority_root.authority_subject)?;
        validate_record_text("root_id", &self.authority_root.root_id)?;
        if self.authority_root.root_digest == [0; 32] { return Err(VerifierProfileAdoptionCurrentnessError::ZeroAuthorityRootDigest); }
        if self.authority_root.provisioning_epoch == 0 { return Err(VerifierProfileAdoptionCurrentnessError::ZeroAuthorityRootEpoch); }
        validate_record_text("grant_id", &self.authority_grant.grant_id_text)?;
        validate_record_text("grant_authority_subject", &self.authority_grant.authority_subject)?;
        validate_record_text("grant_authority_root_id", &self.authority_grant.authority_root_id)?;
        validate_record_text("grant_verifier_role_id", &self.authority_grant.verifier_role_id)?;
        if self.authority_grant.authority_root_digest == [0; 32] { return Err(VerifierProfileAdoptionCurrentnessError::ZeroGrantAuthorityRootDigest); }
        if self.authority_grant.grant_epoch == 0 { return Err(VerifierProfileAdoptionCurrentnessError::ZeroGrantEpoch); }
        let subject = self.transition.subject();
        if subject.authority_subject() != self.authority_root.authority_subject
            || subject.authority_root_id() != self.authority_root.root_id
            || subject.authority_root_digest() != self.authority_root.root_digest
        { return Err(VerifierProfileAdoptionCurrentnessError::RootProjectionMismatch); }
        if subject.authority_subject() != self.authority_grant.authority_subject
            || subject.authority_root_id() != self.authority_grant.authority_root_id
            || subject.authority_root_digest() != self.authority_grant.authority_root_digest
            || subject.verifier_role_id() != self.authority_grant.verifier_role_id
        { return Err(VerifierProfileAdoptionCurrentnessError::GrantProjectionMismatch); }
        match (&self.predecessor_transition, self.transition.predecessor()) {
            (None, VerifierProfileAdoptionPredecessorV1::Bootstrap) if self.transition.generation() == 1 => {}
            (Some(predecessor), VerifierProfileAdoptionPredecessorV1::Previous(expected_digest)) => {
                predecessor.validate()?;
                if predecessor.transition_digest()? != expected_digest { return Err(VerifierProfileAdoptionCurrentnessError::PredecessorRecordMismatch); }
                let expected_generation = predecessor.generation().checked_add(1).ok_or(VerifierProfileAdoptionCurrentnessError::GenerationExhausted)?;
                if self.transition.generation() != expected_generation { return Err(VerifierProfileAdoptionCurrentnessError::PredecessorRecordMismatch); }
                if predecessor.subject().authority_subject() != subject.authority_subject()
                    || predecessor.subject().authority_root_id() != subject.authority_root_id()
                    || predecessor.subject().authority_root_digest() != subject.authority_root_digest()
                    || predecessor.subject().verifier_role_id() != subject.verifier_role_id()
                { return Err(VerifierProfileAdoptionCurrentnessError::PredecessorRecordMismatch); }
            }
            _ => return Err(VerifierProfileAdoptionCurrentnessError::PredecessorRecordMismatch),
        }
        let expected = VerifierProfileAdoptionRegistryRecordIdV1(hash_record(
            &self.transition, self.predecessor_transition.as_ref(), &self.authority_root, &self.authority_grant,
        )?);
        if expected != self.record_id { return Err(VerifierProfileAdoptionCurrentnessError::RecordIdentityMismatch); }
        Ok(())
    }
    pub fn id(&self) -> VerifierProfileAdoptionRegistryRecordIdV1 { self.record_id }
    pub fn transition(&self) -> &VerifierProfileAdoptionTransitionV1 { &self.transition }
    pub fn predecessor_transition(&self) -> Option<&VerifierProfileAdoptionTransitionV1> { self.predecessor_transition.as_ref() }
    pub fn authority_root(&self) -> &VerifierAdoptionAuthorityRootRecordV1 { &self.authority_root }
    pub fn authority_grant(&self) -> &VerifierAdoptionAuthorityGrantRecordV1 { &self.authority_grant }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCurrentVerifierProfileAdoptionV1 {
    record_id: VerifierProfileAdoptionRegistryRecordIdV1,
    granted: AuthorityGrantedVerifierProfileAdoptionV1,
    current_root: VerifierAdoptionAuthorityRootSnapshotV1,
    current_head: VerifierProfileAdoptionHeadV1,
}
impl PolicyCurrentVerifierProfileAdoptionV1 {
    pub fn record_id(&self) -> VerifierProfileAdoptionRegistryRecordIdV1 { self.record_id }
    pub fn profile(&self) -> &VerifierProfileV1 { self.granted.checked().profile() }
    pub fn transition(&self) -> &VerifierProfileAdoptionTransitionV1 { self.granted.checked().transition() }
    pub fn authority_grant(&self) -> &VerifierAdoptionAuthorityGrantV1 { self.granted.authority_grant() }
    pub fn current_root(&self) -> &VerifierAdoptionAuthorityRootSnapshotV1 { &self.current_root }
    pub fn current_head(&self) -> &VerifierProfileAdoptionHeadV1 { &self.current_head }
}

#[allow(clippy::too_many_arguments)]
pub fn check_persisted_adoption_currentness(
    record: &VerifierProfileAdoptionRegistryRecordV1,
    now_unix_ms: u64,
    current_root: &VerifierAdoptionAuthorityRootSnapshotV1,
    current_grant: &VerifierAdoptionAuthorityGrantV1,
    current_head: &VerifierProfileAdoptionHeadV1,
    current_profile: &VerifierProfileV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<PolicyCurrentVerifierProfileAdoptionV1, VerifierProfileAdoptionCurrentnessError> {
    record.validate()?;
    if !record.authority_root.matches(current_root) { return Err(VerifierProfileAdoptionCurrentnessError::AuthorityRootNoLongerCurrent); }
    if !record.authority_grant.matches(current_grant) { return Err(VerifierProfileAdoptionCurrentnessError::AuthorityGrantNoLongerCurrent); }
    let candidate_head = VerifierProfileAdoptionHeadV1::from_transition(&record.transition)?;
    if &candidate_head != current_head { return Err(VerifierProfileAdoptionCurrentnessError::AdoptionNoLongerCurrentHead); }
    let expected_predecessor_head = match &record.predecessor_transition {
        None => VerifierProfileAdoptionHeadV1::Uninitialized,
        Some(predecessor) => VerifierProfileAdoptionHeadV1::from_transition(predecessor)?,
    };
    let subject = record.transition.subject();
    let admission_policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
        subject.authority_subject(), subject.authority_root_id(), subject.authority_root_digest(),
        subject.verifier_role_id(), expected_predecessor_head,
    )?;
    let checked = admission_policy.check(now_unix_ms, &record.transition, current_profile, scope_contract)?;
    let granted = bind_policy_checked_adoption_to_authority_grant(checked, current_grant, scope_contract)?;
    Ok(PolicyCurrentVerifierProfileAdoptionV1 {
        record_id: record.record_id, granted, current_root: current_root.clone(), current_head: current_head.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionCurrentnessError {
    #[error(transparent)] Adoption(#[from] VerifierProfileAdoptionError),
    #[error(transparent)] Admission(#[from] VerifierProfileAdoptionAdmissionError),
    #[error(transparent)] AuthorityGrant(#[from] VerifierAdoptionAuthorityGrantError),
    #[error("unsupported verifier-adoption registry record schema {0}")] UnsupportedRecordSchema(String),
    #[error("{field} must not be blank")] BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")] TextTooLong { field: &'static str },
    #[error("{field} contains control characters")] ControlCharacters { field: &'static str },
    #[error("registry record authority root digest must be non-zero")] ZeroAuthorityRootDigest,
    #[error("registry record authority root provisioning epoch must be non-zero")] ZeroAuthorityRootEpoch,
    #[error("registry record grant authority root digest must be non-zero")] ZeroGrantAuthorityRootDigest,
    #[error("registry record grant epoch must be non-zero")] ZeroGrantEpoch,
    #[error("registry record root snapshot does not match the adoption subject")] RootProjectionMismatch,
    #[error("registry record grant snapshot does not match the adoption subject")] GrantProjectionMismatch,
    #[error("bootstrap record unexpectedly carries a predecessor or successor record omitted it")] UnexpectedPredecessorRecord,
    #[error("successor registry record requires the exact predecessor transition")] MissingPredecessorRecord,
    #[error("registry record predecessor transition does not match exact lineage")] PredecessorRecordMismatch,
    #[error("verifier-adoption generation space exhausted while validating record")] GenerationExhausted,
    #[error("stored verifier-adoption registry record identity is not canonical")] RecordIdentityMismatch,
    #[error("adoption-authority root provisioning state no longer matches persisted record")] AuthorityRootNoLongerCurrent,
    #[error("local verifier-adoption authority grant no longer matches persisted record")] AuthorityGrantNoLongerCurrent,
    #[error("persisted adoption is no longer the exact current lineage head")] AdoptionNoLongerCurrentHead,
}

fn hash_record(
    transition: &VerifierProfileAdoptionTransitionV1,
    predecessor: Option<&VerifierProfileAdoptionTransitionV1>,
    root: &VerifierAdoptionAuthorityRootRecordV1,
    grant: &VerifierAdoptionAuthorityGrantRecordV1,
) -> Result<[u8; 32], VerifierProfileAdoptionCurrentnessError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECORD_DOMAIN);
    hash_text(&mut hasher, VERIFIER_PROFILE_ADOPTION_REGISTRY_RECORD_SCHEMA_V1);
    hasher.update(transition.transition_digest()?.as_bytes());
    match predecessor {
        None => { hasher.update(&[0]); }
        Some(predecessor) => { hasher.update(&[1]); hasher.update(predecessor.transition_digest()?.as_bytes()); }
    }
    hash_text(&mut hasher, &root.authority_subject);
    hash_text(&mut hasher, &root.root_id);
    hasher.update(&root.root_digest);
    hasher.update(&root.provisioning_epoch.to_le_bytes());
    hash_text(&mut hasher, &grant.grant_id_text);
    hash_text(&mut hasher, &grant.authority_subject);
    hash_text(&mut hasher, &grant.authority_root_id);
    hasher.update(&grant.authority_root_digest);
    hash_text(&mut hasher, &grant.verifier_role_id);
    hasher.update(&grant.grant_epoch.to_le_bytes());
    hasher.update(&[evidence_class_tag(grant.maximum_evidence_class)]);
    hash_scope(&mut hasher, &grant.allowed_scope);
    Ok(*hasher.finalize().as_bytes())
}
fn hash_scope(hasher: &mut blake3::Hasher, scope: &VerifierAdoptionScopeV1) {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => { hasher.update(&[0]); }
        VerifierAdoptionScopeV1::Contract { contract_id } => { hasher.update(&[1]); hasher.update(contract_id.as_bytes()); }
        VerifierAdoptionScopeV1::Requirements { contract_id, requirement_ids } => {
            hasher.update(&[2]); hasher.update(contract_id.as_bytes());
            hasher.update(&(requirement_ids.len() as u64).to_le_bytes());
            for requirement_id in requirement_ids { hasher.update(requirement_id.as_bytes()); }
        }
    }
}
fn evidence_class_tag(class: EvidenceClass) -> u8 {
    match class {
        EvidenceClass::Declared => 1, EvidenceClass::Observed => 2, EvidenceClass::StaticAnalysis => 3,
        EvidenceClass::Simulated => 4, EvidenceClass::DifferentiallyVerified => 5,
        EvidenceClass::HardwareVerified => 6, EvidenceClass::IndependentlyReplicated => 7,
    }
}
fn hash_text(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes()); hasher.update(value.as_bytes());
}
fn validate_record_text(field: &'static str, value: &str) -> Result<(), VerifierProfileAdoptionCurrentnessError> {
    let trimmed = value.trim();
    if trimmed.is_empty() { return Err(VerifierProfileAdoptionCurrentnessError::BlankText { field }); }
    if trimmed.len() > 1024 { return Err(VerifierProfileAdoptionCurrentnessError::TextTooLong { field }); }
    if trimmed.chars().any(char::is_control) { return Err(VerifierProfileAdoptionCurrentnessError::ControlCharacters { field }); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate, RequirementCriticality};
    use crate::observation::{DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage, ObservationEnvelopeV1};
    use crate::profile_adoption::VerifierProfileAdoptionSubjectV1;

    fn profile(root: u8, epoch: u64) -> VerifierProfileV1 {
        VerifierProfileV1::new("hardware-verifier-v1", [root; 32], epoch, EvidenceClass::HardwareVerified).unwrap()
    }
    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new("machine-1", "workflow.dependency", "fixture", "1", 1_700_000_000_000,
            ObservationCoverage::Complete, EvidenceBasis::Tested, [1; 32], vec![]).unwrap();
        let dependency = DependencyClaimV1::new("role:research", "requires", "capability:cuda", DependencyBasis::Observed,
            vec![observation.id()], vec![]).unwrap();
        let requirement = ContinuityRequirementV1::new(dependency.id(), "cuda-workflow", RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario { scenario_id: "cuda-fixture-v1".into() }, ApprovalBasis::ExplicitPolicy, [2; 32]).unwrap();
        ContinuityContractV1::new("research-fleet", [3; 32], vec![requirement]).unwrap().validate().unwrap()
    }
    fn root_snapshot(epoch: u64) -> VerifierAdoptionAuthorityRootSnapshotV1 {
        VerifierAdoptionAuthorityRootSnapshotV1::new("organization:test", "adoption-root-1", [0x55; 32], epoch).unwrap()
    }
    fn grant(epoch: u64, max: EvidenceClass, scope: VerifierAdoptionScopeV1) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new("grant-1", "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", epoch, max, scope).unwrap()
    }
    fn subject(profile: &VerifierProfileV1, generation: u64, scope: VerifierAdoptionScopeV1) -> VerifierProfileAdoptionSubjectV1 {
        VerifierProfileAdoptionSubjectV1::new(format!("adopt-{generation}"), "organization:test", "adoption-root-1", [0x55; 32],
            profile, generation, 1_000, 2_000, EvidenceClass::DifferentiallyVerified, scope).unwrap()
    }
    fn granted_bootstrap(profile: &VerifierProfileV1, grant: &VerifierAdoptionAuthorityGrantV1,
        contract: &ValidatedContinuityContractV1) -> AuthorityGrantedVerifierProfileAdoptionV1 {
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject(profile, 1,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() })).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new("organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", VerifierProfileAdoptionHeadV1::Uninitialized).unwrap();
        let checked = policy.check(1_500, &transition, profile, Some(contract)).unwrap();
        bind_policy_checked_adoption_to_authority_grant(checked, grant, Some(contract)).unwrap()
    }

    #[test]
    fn exact_persisted_policy_can_be_revalidated_current_without_claiming_authorization() {
        let contract = contract(); let profile = profile(9, 7);
        let grant = grant(3, EvidenceClass::DifferentiallyVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        let granted = granted_bootstrap(&profile, &grant, &contract); let root = root_snapshot(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted, &root, None).unwrap();
        let current_head = VerifierProfileAdoptionHeadV1::from_transition(granted.checked().transition()).unwrap();
        let current = check_persisted_adoption_currentness(&record, 1_500, &root, &grant, &current_head, &profile, Some(&contract)).unwrap();
        assert_eq!(current.record_id(), record.id()); assert_eq!(current.profile().id(), profile.id());
    }

    #[test]
    fn root_reprovisioning_invalidates_runtime_currentness() {
        let contract = contract(); let profile = profile(9, 7);
        let grant = grant(3, EvidenceClass::DifferentiallyVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        let granted = granted_bootstrap(&profile, &grant, &contract); let root = root_snapshot(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted, &root, None).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(granted.checked().transition()).unwrap();
        assert_eq!(check_persisted_adoption_currentness(&record, 1_500, &root_snapshot(5), &grant, &head, &profile, Some(&contract)).unwrap_err(),
            VerifierProfileAdoptionCurrentnessError::AuthorityRootNoLongerCurrent);
    }

    #[test]
    fn grant_narrowing_invalidates_runtime_currentness() {
        let contract = contract(); let profile = profile(9, 7);
        let original = grant(3, EvidenceClass::DifferentiallyVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        let granted = granted_bootstrap(&profile, &original, &contract); let root = root_snapshot(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted, &root, None).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(granted.checked().transition()).unwrap();
        let narrowed = grant(4, EvidenceClass::Simulated, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        assert_eq!(check_persisted_adoption_currentness(&record, 1_500, &root, &narrowed, &head, &profile, Some(&contract)).unwrap_err(),
            VerifierProfileAdoptionCurrentnessError::AuthorityGrantNoLongerCurrent);
    }

    #[test]
    fn superseded_adoption_head_invalidates_runtime_currentness() {
        let contract = contract(); let profile_a = profile(9, 7);
        let grant = grant(3, EvidenceClass::DifferentiallyVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        let first = granted_bootstrap(&profile_a, &grant, &contract); let root = root_snapshot(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&first, &root, None).unwrap();
        let first_transition = first.checked().transition().clone(); let profile_b = profile(10, 8);
        let successor = VerifierProfileAdoptionTransitionV1::successor(subject(&profile_b, 2,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() }), &first_transition).unwrap();
        let current_head = VerifierProfileAdoptionHeadV1::from_transition(&successor).unwrap();
        assert_eq!(check_persisted_adoption_currentness(&record, 1_500, &root, &grant, &current_head, &profile_a, Some(&contract)).unwrap_err(),
            VerifierProfileAdoptionCurrentnessError::AdoptionNoLongerCurrentHead);
    }

    #[test]
    fn expiry_invalidates_runtime_currentness() {
        let contract = contract(); let profile = profile(9, 7);
        let grant = grant(3, EvidenceClass::DifferentiallyVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        let granted = granted_bootstrap(&profile, &grant, &contract); let root = root_snapshot(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted, &root, None).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(granted.checked().transition()).unwrap();
        assert!(matches!(check_persisted_adoption_currentness(&record, 2_000, &root, &grant, &head, &profile, Some(&contract)),
            Err(VerifierProfileAdoptionCurrentnessError::Admission(VerifierProfileAdoptionAdmissionError::Expired { .. }))));
    }

    #[test]
    fn successor_record_requires_exact_predecessor_transition() {
        let contract = contract(); let profile_a = profile(9, 7);
        let grant = grant(3, EvidenceClass::DifferentiallyVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        let first = granted_bootstrap(&profile_a, &grant, &contract); let first_transition = first.checked().transition().clone();
        let first_head = VerifierProfileAdoptionHeadV1::from_transition(&first_transition).unwrap(); let profile_b = profile(10, 8);
        let successor = VerifierProfileAdoptionTransitionV1::successor(subject(&profile_b, 2,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() }), &first_transition).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new("organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", first_head).unwrap();
        let checked = policy.check(1_500, &successor, &profile_b, Some(&contract)).unwrap();
        let granted_successor = bind_policy_checked_adoption_to_authority_grant(checked, &grant, Some(&contract)).unwrap();
        let root = root_snapshot(4);
        assert_eq!(VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted_successor, &root, None).unwrap_err(),
            VerifierProfileAdoptionCurrentnessError::MissingPredecessorRecord);
        VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted_successor, &root, Some(&first_transition)).unwrap();
    }
}
