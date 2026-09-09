// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifier-owned continuity evidence admission.
//!
//! `RawEvidence != ProfilePolicyCheckedEvidence != AuthorityCheckedEvidence != AuthenticatedEvidence != QualifiedWitness`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{
    ContinuityContractId, ContinuityRequirementId, ValidatedContinuityContractV1,
};
use crate::profile_adoption::{VerifierAdoptionScopeV1, VerifierProfileAdoptionTransitionDigest};
use crate::profile_adoption_root::VerifierProfileAdoptionAuthorityRootSnapshotId;
use crate::witness::{
    EvidenceClass, ObligationDispositionV1, TargetRealizationId, VerificationPolicyV1,
};

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.verifier-profile.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.verification-claim.v1\0";
const AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-verification-evidence.authority-bound.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileId([u8; 32]);
impl VerifierProfileId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerificationEvidenceClaimId([u8; 32]);
impl VerificationEvidenceClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedVerificationEvidenceId([u8; 32]);
impl AuthenticatedVerificationEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Provisioned verifier identity and the strongest evidence class it may issue.
///
/// This is capability/configuration, not organizational authority. A profile may
/// support `HardwareVerified` while an adopted authority grant restricts it to a
/// weaker class or narrower scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileV1 {
    profile_name: String,
    root_digest: [u8; 32],
    root_epoch: u64,
    evidence_class: EvidenceClass,
    profile_id: VerifierProfileId,
}

impl VerifierProfileV1 {
    pub fn new(
        profile_name: impl Into<String>,
        root_digest: [u8; 32],
        root_epoch: u64,
        evidence_class: EvidenceClass,
    ) -> Result<Self, VerificationAdmissionError> {
        let profile_name = checked_text("profile_name", profile_name.into())?;
        if root_digest == [0; 32] { return Err(VerificationAdmissionError::ZeroVerifierRootDigest); }
        if root_epoch == 0 { return Err(VerificationAdmissionError::ZeroVerifierRootEpoch); }
        let profile_id = VerifierProfileId(hash_profile(&profile_name, root_digest, root_epoch, evidence_class));
        Ok(Self { profile_name, root_digest, root_epoch, evidence_class, profile_id })
    }
    pub fn id(&self) -> VerifierProfileId { self.profile_id }
    pub fn profile_name(&self) -> &str { &self.profile_name }
    pub fn root_digest(&self) -> [u8; 32] { self.root_digest }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
    pub fn evidence_class(&self) -> EvidenceClass { self.evidence_class }
    pub fn validate(&self) -> Result<(), VerificationAdmissionError> {
        checked_text("profile_name", self.profile_name.clone())?;
        if self.root_digest == [0; 32] { return Err(VerificationAdmissionError::ZeroVerifierRootDigest); }
        if self.root_epoch == 0 { return Err(VerificationAdmissionError::ZeroVerifierRootEpoch); }
        let expected = VerifierProfileId(hash_profile(
            &self.profile_name, self.root_digest, self.root_epoch, self.evidence_class,
        ));
        if expected != self.profile_id { return Err(VerificationAdmissionError::VerifierProfileIdentityMismatch); }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcomeV1 {
    Satisfied,
    Failed,
    Inconclusive,
    InfrastructureFailure,
    NotExecuted,
}
impl VerificationOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Satisfied => 1,
            Self::Failed => 2,
            Self::Inconclusive => 3,
            Self::InfrastructureFailure => 4,
            Self::NotExecuted => 5,
        }
    }
}

/// Transportable verification claim. It carries no evidence-strength assertion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationEvidenceClaimV1 {
    contract_id: ContinuityContractId,
    target_realization_id: TargetRealizationId,
    requirement_id: ContinuityRequirementId,
    verifier_profile_id: VerifierProfileId,
    transaction_challenge: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: VerificationOutcomeV1,
    raw_evidence_digest: [u8; 32],
    claim_id: VerificationEvidenceClaimId,
}

impl VerificationEvidenceClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract_id: ContinuityContractId,
        target_realization_id: TargetRealizationId,
        requirement_id: ContinuityRequirementId,
        verifier_profile_id: VerifierProfileId,
        transaction_challenge: [u8; 32],
        observed_at_unix_ms: u64,
        outcome: VerificationOutcomeV1,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, VerificationAdmissionError> {
        if transaction_challenge == [0; 32] { return Err(VerificationAdmissionError::ZeroTransactionChallenge); }
        if observed_at_unix_ms == 0 { return Err(VerificationAdmissionError::ZeroObservationTime); }
        if raw_evidence_digest == [0; 32] { return Err(VerificationAdmissionError::ZeroRawEvidenceDigest); }
        let claim_id = VerificationEvidenceClaimId(hash_claim(
            contract_id, target_realization_id, requirement_id, verifier_profile_id,
            transaction_challenge, observed_at_unix_ms, outcome, raw_evidence_digest,
        ));
        Ok(Self {
            contract_id, target_realization_id, requirement_id, verifier_profile_id,
            transaction_challenge, observed_at_unix_ms, outcome, raw_evidence_digest, claim_id,
        })
    }
    pub fn id(&self) -> VerificationEvidenceClaimId { self.claim_id }
    pub fn contract_id(&self) -> ContinuityContractId { self.contract_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn requirement_id(&self) -> ContinuityRequirementId { self.requirement_id }
    pub fn verifier_profile_id(&self) -> VerifierProfileId { self.verifier_profile_id }
    pub fn transaction_challenge(&self) -> [u8; 32] { self.transaction_challenge }
    pub fn observed_at_unix_ms(&self) -> u64 { self.observed_at_unix_ms }
    pub fn outcome(&self) -> VerificationOutcomeV1 { self.outcome }
    pub fn raw_evidence_digest(&self) -> [u8; 32] { self.raw_evidence_digest }
    pub fn validate(&self) -> Result<(), VerificationAdmissionError> {
        if self.transaction_challenge == [0; 32] { return Err(VerificationAdmissionError::ZeroTransactionChallenge); }
        if self.observed_at_unix_ms == 0 { return Err(VerificationAdmissionError::ZeroObservationTime); }
        if self.raw_evidence_digest == [0; 32] { return Err(VerificationAdmissionError::ZeroRawEvidenceDigest); }
        let expected = VerificationEvidenceClaimId(hash_claim(
            self.contract_id, self.target_realization_id, self.requirement_id,
            self.verifier_profile_id, self.transaction_challenge, self.observed_at_unix_ms,
            self.outcome, self.raw_evidence_digest,
        ));
        if expected != self.claim_id { return Err(VerificationAdmissionError::VerificationClaimIdentityMismatch); }
        Ok(())
    }
}

/// Exact profile/policy/context admission. This is deliberately weaker than
/// organizational authority admission and proves no signature.
#[derive(Debug, Clone)]
pub(crate) struct ProfilePolicyCheckedVerificationEvidenceV1 {
    claim: VerificationEvidenceClaimV1,
    profile: VerifierProfileV1,
}

pub(crate) fn policy_check_verification_evidence(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    policy: &VerificationPolicyV1,
    profile: &VerifierProfileV1,
    expected_challenge: [u8; 32],
    claim: VerificationEvidenceClaimV1,
) -> Result<ProfilePolicyCheckedVerificationEvidenceV1, VerificationAdmissionError> {
    profile.validate()?;
    claim.validate()?;
    if expected_challenge == [0; 32] { return Err(VerificationAdmissionError::ZeroTransactionChallenge); }
    if policy.verifier_profile_id() != profile.profile_name() {
        return Err(VerificationAdmissionError::VerifierProfilePolicyMismatch);
    }
    if claim.verifier_profile_id() != profile.id() {
        return Err(VerificationAdmissionError::VerifierProfileClaimMismatch);
    }
    if claim.contract_id() != contract.id() { return Err(VerificationAdmissionError::ContractMismatch); }
    if claim.target_realization_id() != target { return Err(VerificationAdmissionError::TargetMismatch); }
    if claim.transaction_challenge() != expected_challenge { return Err(VerificationAdmissionError::ChallengeMismatch); }
    if !contract.requirements().iter().any(|r| r.id() == claim.requirement_id()) {
        return Err(VerificationAdmissionError::UnknownRequirement);
    }
    if !policy.entries().iter().any(|e| e.requirement_id() == claim.requirement_id()) {
        return Err(VerificationAdmissionError::RequirementOutsideVerificationPolicy);
    }
    Ok(ProfilePolicyCheckedVerificationEvidenceV1 { claim, profile: profile.clone() })
}

/// Profile-policy evidence after a separate current verifier-authority theorem has
/// restricted it to the actually granted evidence class and adoption lineage.
///
/// There is intentionally no production constructor yet. A future
/// `AuthorizedVerifierProfile` path must be the only production source and must
/// prove adoption scope/currentness before creating this value.
#[derive(Debug, Clone)]
pub(crate) struct AuthorityCheckedVerificationEvidenceV1 {
    profile_checked: ProfilePolicyCheckedVerificationEvidenceV1,
    effective_evidence_class: EvidenceClass,
    adoption_transition_digest: VerifierProfileAdoptionTransitionDigest,
    authority_root_snapshot_id: VerifierProfileAdoptionAuthorityRootSnapshotId,
}

impl AuthorityCheckedVerificationEvidenceV1 {
    pub(crate) fn effective_evidence_class(&self) -> EvidenceClass {
        self.effective_evidence_class
    }

    pub(crate) fn adoption_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.adoption_transition_digest
    }

    pub(crate) fn authority_root_snapshot_id(&self) -> VerifierProfileAdoptionAuthorityRootSnapshotId {
        self.authority_root_snapshot_id
    }

    /// Test-only model of the future `AuthorizedVerifierProfile` restriction step.
    ///
    /// All authority facts are derived from one exact adoption transition and one
    /// exact local root snapshot rather than passed as independently selectable
    /// ceiling/scope/identity fields.
    #[cfg(test)]
    pub(crate) fn authorize_for_test(
        profile_checked: ProfilePolicyCheckedVerificationEvidenceV1,
        adoption: &crate::profile_adoption::VerifierProfileAdoptionTransitionV1,
        authority_root: &crate::profile_adoption_root::VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) -> Result<Self, VerificationAdmissionError> {
        let subject = adoption.subject();
        if subject.verifier_profile_id() != profile_checked.profile.id()
            || subject.verifier_role_id() != profile_checked.profile.profile_name()
        {
            return Err(VerificationAdmissionError::AuthorityVerifierProfileMismatch);
        }
        if subject.authority_subject() != authority_root.authority_subject()
            || subject.authority_root_id() != authority_root.authority_root_id()
            || subject.authority_root_digest() != authority_root.authority_root_digest()
        {
            return Err(VerificationAdmissionError::AuthorityRootSnapshotMismatch);
        }
        if subject.evidence_class_ceiling() > profile_checked.profile.evidence_class() {
            return Err(VerificationAdmissionError::AuthorityEvidenceClassExceedsProfile {
                profile: profile_checked.profile.evidence_class(),
                authorized: subject.evidence_class_ceiling(),
            });
        }
        if !scope_allows_claim(subject.scope(), &profile_checked.claim) {
            return Err(VerificationAdmissionError::RequirementOutsideAdoptionScope);
        }
        let observed_at = profile_checked.claim.observed_at_unix_ms();
        if observed_at < subject.valid_from_unix_ms() || observed_at >= subject.valid_until_unix_ms() {
            return Err(VerificationAdmissionError::EvidenceObservationOutsideAdoptionValidity {
                observed_at_unix_ms: observed_at,
                valid_from_unix_ms: subject.valid_from_unix_ms(),
                valid_until_unix_ms: subject.valid_until_unix_ms(),
            });
        }
        let adoption_transition_digest = adoption
            .transition_digest()
            .expect("test authority transition must remain canonical");
        Ok(Self {
            profile_checked,
            effective_evidence_class: subject.evidence_class_ceiling(),
            adoption_transition_digest,
            authority_root_snapshot_id: authority_root.id(),
        })
    }
}

/// Authenticated evidence can only contain authority-checked evidence.
///
/// A future crypto/Xenia adapter must authenticate the exact claim under the
/// authorized verifier root, but it must not be able to skip the organizational
/// adoption-authority layer and authenticate a bare profile-policy result directly.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedVerificationEvidenceV1 {
    checked: AuthorityCheckedVerificationEvidenceV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedVerificationEvidenceId,
}

impl AuthenticatedVerificationEvidenceV1 {
    pub(crate) fn contract_id(&self) -> ContinuityContractId {
        self.checked.profile_checked.claim.contract_id()
    }
    pub(crate) fn target_realization_id(&self) -> TargetRealizationId {
        self.checked.profile_checked.claim.target_realization_id()
    }
    pub(crate) fn requirement_id(&self) -> ContinuityRequirementId {
        self.checked.profile_checked.claim.requirement_id()
    }
    pub(crate) fn profile_id(&self) -> VerifierProfileId {
        self.checked.profile_checked.profile.id()
    }
    pub(crate) fn root_epoch(&self) -> u64 {
        self.checked.profile_checked.profile.root_epoch()
    }
    pub(crate) fn observed_at_unix_ms(&self) -> u64 {
        self.checked.profile_checked.claim.observed_at_unix_ms()
    }
    pub(crate) fn effective_evidence_class(&self) -> EvidenceClass {
        self.checked.effective_evidence_class()
    }
    pub(crate) fn adoption_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.checked.adoption_transition_digest()
    }
    pub(crate) fn authority_root_snapshot_id(&self) -> VerifierProfileAdoptionAuthorityRootSnapshotId {
        self.checked.authority_root_snapshot_id()
    }
    pub(crate) fn id(&self) -> AuthenticatedVerificationEvidenceId { self.evidence_id }
    pub(crate) fn disposition(&self) -> ObligationDispositionV1 {
        let evidence_digest = *self.evidence_id.as_bytes();
        match self.checked.profile_checked.claim.outcome() {
            VerificationOutcomeV1::Satisfied => ObligationDispositionV1::Satisfied {
                evidence_digest,
                evidence_class: self.checked.effective_evidence_class(),
            },
            VerificationOutcomeV1::Failed => ObligationDispositionV1::Failed { evidence_digest },
            VerificationOutcomeV1::Inconclusive => ObligationDispositionV1::Inconclusive { evidence_digest },
            VerificationOutcomeV1::InfrastructureFailure => ObligationDispositionV1::InfrastructureFailure { evidence_digest },
            VerificationOutcomeV1::NotExecuted => ObligationDispositionV1::NotExecuted { evidence_digest },
        }
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: AuthorityCheckedVerificationEvidenceV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, VerificationAdmissionError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(VerificationAdmissionError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedVerificationEvidenceId(hash_authenticated(
            checked.profile_checked.claim.id(),
            checked.profile_checked.profile.id(),
            checked.effective_evidence_class,
            checked.adoption_transition_digest,
            checked.authority_root_snapshot_id,
            authentication_evidence_digest,
        ));
        Ok(Self { checked, authentication_evidence_digest, evidence_id })
    }
}

fn scope_allows_claim(scope: &VerifierAdoptionScopeV1, claim: &VerificationEvidenceClaimV1) -> bool {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => true,
        VerifierAdoptionScopeV1::Contract { contract_id } => *contract_id == claim.contract_id(),
        VerifierAdoptionScopeV1::Requirements {
            contract_id,
            requirement_ids,
        } => {
            *contract_id == claim.contract_id()
                && requirement_ids
                    .iter()
                    .any(|requirement_id| *requirement_id == claim.requirement_id())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerificationAdmissionError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("verifier root digest must be non-zero")]
    ZeroVerifierRootDigest,
    #[error("verifier root epoch must be non-zero")]
    ZeroVerifierRootEpoch,
    #[error("stored verifier profile identity does not match canonical fields")]
    VerifierProfileIdentityMismatch,
    #[error("transaction challenge must be non-zero")]
    ZeroTransactionChallenge,
    #[error("observation time must be non-zero")]
    ZeroObservationTime,
    #[error("raw verification evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("stored verification claim identity does not match canonical fields")]
    VerificationClaimIdentityMismatch,
    #[error("verification policy names a different verifier profile")]
    VerifierProfilePolicyMismatch,
    #[error("raw claim binds a different verifier profile snapshot")]
    VerifierProfileClaimMismatch,
    #[error("raw claim binds a different continuity contract")]
    ContractMismatch,
    #[error("raw claim binds a different target realization")]
    TargetMismatch,
    #[error("raw claim challenge does not match this verification transaction")]
    ChallengeMismatch,
    #[error("raw claim references a requirement outside the validated contract")]
    UnknownRequirement,
    #[error("raw claim references a requirement outside the verification policy")]
    RequirementOutsideVerificationPolicy,
    #[error("adoption authority binds a different exact verifier profile")]
    AuthorityVerifierProfileMismatch,
    #[error("adoption authority root snapshot does not match the adoption transition")]
    AuthorityRootSnapshotMismatch,
    #[error("authorized evidence class {authorized:?} exceeds verifier profile maximum {profile:?}")]
    AuthorityEvidenceClassExceedsProfile {
        profile: EvidenceClass,
        authorized: EvidenceClass,
    },
    #[error("verification claim requirement is outside the verifier adoption authority scope")]
    RequirementOutsideAdoptionScope,
    #[error("verification evidence observation {observed_at_unix_ms} is outside verifier adoption validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    EvidenceObservationOutsideAdoptionValidity {
        observed_at_unix_ms: u64,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    #[error("authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
}

fn checked_text(field: &'static str, value: String) -> Result<String, VerificationAdmissionError> {
    let trimmed = value.trim();
    if trimmed.is_empty() { return Err(VerificationAdmissionError::BlankText { field }); }
    if trimmed.len() > 1024 { return Err(VerificationAdmissionError::TextTooLong { field }); }
    if trimmed.chars().any(char::is_control) { return Err(VerificationAdmissionError::ControlCharacters { field }); }
    Ok(trimmed.to_string())
}

fn hash_profile(name: &str, root: [u8; 32], epoch: u64, class: EvidenceClass) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, name);
    bytes.extend_from_slice(&root);
    bytes.extend_from_slice(&epoch.to_le_bytes());
    bytes.push(evidence_class_tag(class));
    domain_hash(PROFILE_DOMAIN, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    contract_id: ContinuityContractId,
    target_id: TargetRealizationId,
    requirement_id: ContinuityRequirementId,
    profile_id: VerifierProfileId,
    challenge: [u8; 32],
    observed_at: u64,
    outcome: VerificationOutcomeV1,
    raw_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(contract_id.as_bytes());
    bytes.extend_from_slice(target_id.as_bytes());
    bytes.extend_from_slice(requirement_id.as_bytes());
    bytes.extend_from_slice(profile_id.as_bytes());
    bytes.extend_from_slice(&challenge);
    bytes.extend_from_slice(&observed_at.to_le_bytes());
    bytes.push(outcome.tag());
    bytes.extend_from_slice(&raw_digest);
    domain_hash(CLAIM_DOMAIN, &bytes)
}

fn hash_authenticated(
    claim_id: VerificationEvidenceClaimId,
    profile_id: VerifierProfileId,
    effective_evidence_class: EvidenceClass,
    adoption_transition_digest: VerifierProfileAdoptionTransitionDigest,
    authority_root_snapshot_id: VerifierProfileAdoptionAuthorityRootSnapshotId,
    auth_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(claim_id.as_bytes());
    bytes.extend_from_slice(profile_id.as_bytes());
    bytes.push(evidence_class_tag(effective_evidence_class));
    bytes.extend_from_slice(adoption_transition_digest.as_bytes());
    bytes.extend_from_slice(authority_root_snapshot_id.as_bytes());
    bytes.extend_from_slice(&auth_digest);
    domain_hash(AUTH_DOMAIN, &bytes)
}

fn evidence_class_tag(class: EvidenceClass) -> u8 {
    match class {
        EvidenceClass::Declared => 1,
        EvidenceClass::Observed => 2,
        EvidenceClass::StaticAnalysis => 3,
        EvidenceClass::Simulated => 4,
        EvidenceClass::DifferentiallyVerified => 5,
        EvidenceClass::HardwareVerified => 6,
        EvidenceClass::IndependentlyReplicated => 7,
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}
fn put_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
        RequirementCriticality,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };
    use crate::profile_adoption::{
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_root::VerifierProfileAdoptionAuthorityRootSnapshotV1;
    use crate::witness::VerificationPolicyEntryV1;

    const AUTH_VALID_FROM: u64 = 1_600_000_000_000;
    const AUTH_VALID_UNTIL: u64 = 1_800_000_000_000;

    fn contract_with_seed(seed: u8) -> ValidatedContinuityContractV1 {
        let obs = ObservationEnvelopeV1::new(
            format!("machine-{seed}"), "workflow.dependency", "fixture", "1",
            1_700_000_000_000 + seed as u64, ObservationCoverage::Complete,
            EvidenceBasis::Tested, [seed; 32], vec![],
        ).unwrap();
        let dep = DependencyClaimV1::new(
            "role:research", "requires", format!("capability-{seed}"), DependencyBasis::Observed,
            vec![obs.id()], vec![],
        ).unwrap();
        let req = ContinuityRequirementV1::new(
            dep.id(), format!("cuda-workflow-{seed}"), RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario { scenario_id: format!("cuda-fixture-{seed}") },
            ApprovalBasis::ExplicitPolicy, [seed.wrapping_add(20); 32],
        ).unwrap();
        ContinuityContractV1::new(
            format!("research-fleet-{seed}"),
            [seed.wrapping_add(40); 32],
            vec![req],
        ).unwrap().validate().unwrap()
    }
    fn contract() -> ValidatedContinuityContractV1 { contract_with_seed(1) }
    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new("hardware-verifier-v1", [9; 32], 7, EvidenceClass::HardwareVerified).unwrap()
    }
    fn policy(contract: &ValidatedContinuityContractV1) -> VerificationPolicyV1 {
        VerificationPolicyV1::new(
            "hardware-verifier-v1",
            vec![VerificationPolicyEntryV1::new(contract.requirements()[0].id(), EvidenceClass::Simulated)],
        ).unwrap()
    }
    fn claim_at(
        contract: &ValidatedContinuityContractV1,
        target: TargetRealizationId,
        profile: &VerifierProfileV1,
        challenge: [u8; 32],
        observed_at_unix_ms: u64,
    ) -> VerificationEvidenceClaimV1 {
        VerificationEvidenceClaimV1::new(
            contract.id(), target, contract.requirements()[0].id(), profile.id(), challenge,
            observed_at_unix_ms, VerificationOutcomeV1::Satisfied, [8; 32],
        ).unwrap()
    }
    fn claim(contract: &ValidatedContinuityContractV1, target: TargetRealizationId, profile: &VerifierProfileV1, challenge: [u8; 32]) -> VerificationEvidenceClaimV1 {
        claim_at(contract, target, profile, challenge, 1_700_000_000_111)
    }
    fn profile_checked_from_claim(
        contract: &ValidatedContinuityContractV1,
        target: TargetRealizationId,
        profile: &VerifierProfileV1,
        challenge: [u8; 32],
        claim: VerificationEvidenceClaimV1,
    ) -> ProfilePolicyCheckedVerificationEvidenceV1 {
        policy_check_verification_evidence(
            contract,
            target,
            &policy(contract),
            profile,
            challenge,
            claim,
        ).unwrap()
    }
    fn profile_checked(
        contract: &ValidatedContinuityContractV1,
        target: TargetRealizationId,
        profile: &VerifierProfileV1,
        challenge: [u8; 32],
    ) -> ProfilePolicyCheckedVerificationEvidenceV1 {
        profile_checked_from_claim(
            contract,
            target,
            profile,
            challenge,
            claim(contract, target, profile, challenge),
        )
    }
    fn authority_context(
        profile: &VerifierProfileV1,
        adoption_id: &str,
        scope: VerifierAdoptionScopeV1,
        ceiling: EvidenceClass,
        root_epoch: u64,
    ) -> (
        VerifierProfileAdoptionTransitionV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) {
        let subject = VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            profile,
            1,
            AUTH_VALID_FROM,
            AUTH_VALID_UNTIL,
            ceiling,
            scope,
        ).unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let root = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            root_epoch,
        ).unwrap();
        (transition, root)
    }

    #[test]
    fn raw_claim_has_no_caller_selectable_evidence_class() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        assert_eq!(claim(&contract, target, &profile, [5; 32]).verifier_profile_id(), profile.id());
    }

    #[test]
    fn cross_target_replay_fails_closed() {
        let contract = contract();
        let a = TargetRealizationId::from_digest([4; 32]).unwrap();
        let b = TargetRealizationId::from_digest([6; 32]).unwrap();
        let profile = profile();
        let result = policy_check_verification_evidence(&contract, b, &policy(&contract), &profile, [5; 32], claim(&contract, a, &profile, [5; 32]));
        assert_eq!(result.unwrap_err(), VerificationAdmissionError::TargetMismatch);
    }

    #[test]
    fn cross_transaction_replay_fails_closed() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let result = policy_check_verification_evidence(&contract, target, &policy(&contract), &profile, [6; 32], claim(&contract, target, &profile, [5; 32]));
        assert_eq!(result.unwrap_err(), VerificationAdmissionError::ChallengeMismatch);
    }

    #[test]
    fn root_epoch_changes_profile_identity() {
        let a = profile();
        let b = VerifierProfileV1::new("hardware-verifier-v1", [9; 32], 8, EvidenceClass::HardwareVerified).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn adopted_ceiling_not_profile_max_drives_authenticated_disposition() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let checked = profile_checked(&contract, target, &profile, [5; 32]);
        let (adoption, root) = authority_context(
            &profile,
            "adopt-restricted",
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::DifferentiallyVerified,
            9,
        );
        let authority_checked = AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
            checked,
            &adoption,
            &root,
        ).unwrap();
        let authenticated = AuthenticatedVerificationEvidenceV1::authenticate_for_test(
            authority_checked,
            [0x77; 32],
        ).unwrap();

        assert_eq!(authenticated.effective_evidence_class(), EvidenceClass::DifferentiallyVerified);
        assert!(matches!(
            authenticated.disposition(),
            ObligationDispositionV1::Satisfied {
                evidence_class: EvidenceClass::DifferentiallyVerified,
                ..
            }
        ));
    }

    #[test]
    fn adoption_profile_must_match_profile_policy_result() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile_a = profile();
        let profile_b = VerifierProfileV1::new(
            "hardware-verifier-v1",
            [0x0a; 32],
            8,
            EvidenceClass::HardwareVerified,
        ).unwrap();
        let checked = profile_checked(&contract, target, &profile_a, [5; 32]);
        let (adoption, root) = authority_context(
            &profile_b,
            "adopt-other-profile",
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::HardwareVerified,
            9,
        );
        assert_eq!(
            AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
                checked,
                &adoption,
                &root,
            ).unwrap_err(),
            VerificationAdmissionError::AuthorityVerifierProfileMismatch
        );
    }

    #[test]
    fn adoption_scope_is_a_second_gate_after_profile_policy() {
        let contract = contract();
        let foreign = contract_with_seed(2);
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let checked = profile_checked(&contract, target, &profile, [5; 32]);
        let (adoption, root) = authority_context(
            &profile,
            "adopt-foreign-scope",
            VerifierAdoptionScopeV1::Contract { contract_id: foreign.id() },
            EvidenceClass::HardwareVerified,
            9,
        );
        assert_eq!(
            AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
                checked,
                &adoption,
                &root,
            ).unwrap_err(),
            VerificationAdmissionError::RequirementOutsideAdoptionScope
        );
    }

    #[test]
    fn evidence_observation_must_fall_inside_adoption_validity() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let early = claim_at(
            &contract,
            target,
            &profile,
            [5; 32],
            AUTH_VALID_FROM - 1,
        );
        let checked = profile_checked_from_claim(&contract, target, &profile, [5; 32], early);
        let (adoption, root) = authority_context(
            &profile,
            "adopt-time-bound",
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::HardwareVerified,
            9,
        );
        assert_eq!(
            AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
                checked,
                &adoption,
                &root,
            ).unwrap_err(),
            VerificationAdmissionError::EvidenceObservationOutsideAdoptionValidity {
                observed_at_unix_ms: AUTH_VALID_FROM - 1,
                valid_from_unix_ms: AUTH_VALID_FROM,
                valid_until_unix_ms: AUTH_VALID_UNTIL,
            }
        );
    }

    #[test]
    fn authenticated_identity_commits_to_authority_grant_context() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let checked = profile_checked(&contract, target, &profile, [5; 32]);
        let (adoption_a, root_a) = authority_context(
            &profile,
            "adopt-a",
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::DifferentiallyVerified,
            9,
        );
        let (adoption_b, root_b) = authority_context(
            &profile,
            "adopt-b",
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::DifferentiallyVerified,
            10,
        );
        let a = AuthenticatedVerificationEvidenceV1::authenticate_for_test(
            AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
                checked.clone(), &adoption_a, &root_a,
            ).unwrap(),
            [0x77; 32],
        ).unwrap();
        let b = AuthenticatedVerificationEvidenceV1::authenticate_for_test(
            AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
                checked, &adoption_b, &root_b,
            ).unwrap(),
            [0x77; 32],
        ).unwrap();

        assert_ne!(a.id(), b.id());
        assert_ne!(a.adoption_transition_digest(), b.adoption_transition_digest());
        assert_ne!(a.authority_root_snapshot_id(), b.authority_root_snapshot_id());
    }
}
