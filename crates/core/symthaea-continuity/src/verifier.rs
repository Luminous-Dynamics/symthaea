// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifier-owned continuity evidence admission.
//!
//! Core invariant:
//!
//! `RawEvidence != PolicyCheckedEvidence != AuthenticatedEvidence != QualifiedWitness`.
//!
//! This module deliberately stops before production authentication. Raw evidence
//! and verifier profiles are public data, while the authenticated wrapper has no
//! production constructor yet. A future Xenia/crypto adapter must authenticate the
//! exact policy-checked claim before witness composition can consume it.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{
    ContinuityContractId, ContinuityRequirementId, ValidatedContinuityContractV1,
};
use crate::witness::{
    EvidenceClass, ObligationDispositionV1, TargetRealizationId, VerificationPolicyV1,
};

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.verifier-profile.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.verification-claim.v1\0";
const AUTHENTICATED_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-verification-evidence.v1\0";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct VerifierProfileId([u8; 32]);

impl VerifierProfileId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct VerificationEvidenceClaimId([u8; 32]);

impl VerificationEvidenceClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct AuthenticatedVerificationEvidenceId([u8; 32]);

impl AuthenticatedVerificationEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Locally provisioned verifier profile.
///
/// `evidence_class` is a property of the trusted verifier profile, not a field
/// supplied by each raw evidence claim. This prevents transport data from simply
/// declaring itself `HardwareVerified` or `IndependentlyReplicated`.
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
        if root_digest == [0; 32] {
            return Err(VerificationAdmissionError::ZeroVerifierRootDigest);
        }
        if root_epoch == 0 {
            return Err(VerificationAdmissionError::ZeroVerifierRootEpoch);
        }
        let profile_id = VerifierProfileId(hash_profile(
            &profile_name,
            root_digest,
            root_epoch,
            evidence_class,
        ));
        Ok(Self {
            profile_name,
            root_digest,
            root_epoch,
            evidence_class,
            profile_id,
        })
    }

    pub fn id(&self) -> VerifierProfileId {
        self.profile_id
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn root_digest(&self) -> [u8; 32] {
        self.root_digest
    }

    pub fn root_epoch(&self) -> u64 {
        self.root_epoch
    }

    pub fn evidence_class(&self) -> EvidenceClass {
        self.evidence_class
    }

    pub fn validate(&self) -> Result<(), VerificationAdmissionError> {
        checked_text("profile_name", self.profile_name.clone())?;
        if self.root_digest == [0; 32] {
            return Err(VerificationAdmissionError::ZeroVerifierRootDigest);
        }
        if self.root_epoch == 0 {
            return Err(VerificationAdmissionError::ZeroVerifierRootEpoch);
        }
        let expected = VerifierProfileId(hash_profile(
            &self.profile_name,
            self.root_digest,
            self.root_epoch,
            self.evidence_class,
        ));
        if expected != self.profile_id {
            return Err(VerificationAdmissionError::VerifierProfileIdentityMismatch);
        }
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

/// Transportable raw verifier claim.
///
/// It intentionally contains no caller-selectable evidence class. The class is
/// supplied by the locally provisioned verifier profile after exact policy/target
/// binding and eventual cryptographic authentication.
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
        if transaction_challenge == [0; 32] {
            return Err(VerificationAdmissionError::ZeroTransactionChallenge);
        }
        if observed_at_unix_ms == 0 {
            return Err(VerificationAdmissionError::ZeroObservationTime);
        }
        if raw_evidence_digest == [0; 32] {
            return Err(VerificationAdmissionError::ZeroRawEvidenceDigest);
        }
        let claim_id = VerificationEvidenceClaimId(hash_claim(
            contract_id,
            target_realization_id,
            requirement_id,
            verifier_profile_id,
            transaction_challenge,
            observed_at_unix_ms,
            outcome,
            raw_evidence_digest,
        ));
        Ok(Self {
            contract_id,
            target_realization_id,
            requirement_id,
            verifier_profile_id,
            transaction_challenge,
            observed_at_unix_ms,
            outcome,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn id(&self) -> VerificationEvidenceClaimId {
        self.claim_id
    }

    pub fn contract_id(&self) -> ContinuityContractId {
        self.contract_id
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.target_realization_id
    }

    pub fn requirement_id(&self) -> ContinuityRequirementId {
        self.requirement_id
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    pub fn transaction_challenge(&self) -> [u8; 32] {
        self.transaction_challenge
    }

    pub fn observed_at_unix_ms(&self) -> u64 {
        self.observed_at_unix_ms
    }

    pub fn outcome(&self) -> VerificationOutcomeV1 {
        self.outcome
    }

    pub fn raw_evidence_digest(&self) -> [u8; 32] {
        self.raw_evidence_digest
    }

    pub fn validate(&self) -> Result<(), VerificationAdmissionError> {
        if self.transaction_challenge == [0; 32] {
            return Err(VerificationAdmissionError::ZeroTransactionChallenge);
        }
        if self.observed_at_unix_ms == 0 {
            return Err(VerificationAdmissionError::ZeroObservationTime);
        }
        if self.raw_evidence_digest == [0; 32] {
            return Err(VerificationAdmissionError::ZeroRawEvidenceDigest);
        }
        let expected = VerificationEvidenceClaimId(hash_claim(
            self.contract_id,
            self.target_realization_id,
            self.requirement_id,
            self.verifier_profile_id,
            self.transaction_challenge,
            self.observed_at_unix_ms,
            self.outcome,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(VerificationAdmissionError::VerificationClaimIdentityMismatch);
        }
        Ok(())
    }
}

/// Structurally and contextually checked, but still unauthenticated, evidence.
#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedVerificationEvidenceV1 {
    claim: VerificationEvidenceClaimV1,
    profile: VerifierProfileV1,
}

/// Bind raw evidence to one exact contract, target, verifier profile, policy and
/// transaction challenge. This proves no signature.
pub(crate) fn policy_check_verification_evidence(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    policy: &VerificationPolicyV1,
    profile: &VerifierProfileV1,
    expected_challenge: [u8; 32],
    claim: VerificationEvidenceClaimV1,
) -> Result<PolicyCheckedVerificationEvidenceV1, VerificationAdmissionError> {
    profile.validate()?;
    claim.validate()?;
    if expected_challenge == [0; 32] {
        return Err(VerificationAdmissionError::ZeroTransactionChallenge);
    }
    if policy.verifier_profile_id() != profile.profile_name() {
        return Err(VerificationAdmissionError::VerifierProfilePolicyMismatch);
    }
    if claim.verifier_profile_id() != profile.id() {
        return Err(VerificationAdmissionError::VerifierProfileClaimMismatch);
    }
    if claim.contract_id() != contract.id() {
        return Err(VerificationAdmissionError::ContractMismatch);
    }
    if claim.target_realization_id() != target {
        return Err(VerificationAdmissionError::TargetMismatch);
    }
    if claim.transaction_challenge() != expected_challenge {
        return Err(VerificationAdmissionError::ChallengeMismatch);
    }
    if !contract
        .requirements()
        .iter()
        .any(|requirement| requirement.id() == claim.requirement_id())
    {
        return Err(VerificationAdmissionError::UnknownRequirement);
    }
    if !policy
        .entries()
        .iter()
        .any(|entry| entry.requirement_id() == claim.requirement_id())
    {
        return Err(VerificationAdmissionError::RequirementOutsideVerificationPolicy);
    }
    Ok(PolicyCheckedVerificationEvidenceV1 { claim, profile: profile.clone() })
}

/// Authenticated verifier-owned evidence. V1 intentionally has no production
/// constructor. A future crypto/Xenia adapter must authenticate the exact claim
/// under the captured verifier root before constructing this value.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedVerificationEvidenceV1 {
    checked: PolicyCheckedVerificationEvidenceV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedVerificationEvidenceId,
}

impl AuthenticatedVerificationEvidenceV1 {
    pub(crate) fn requirement_id(&self) -> ContinuityRequirementId {
        self.checked.claim.requirement_id()
    }

    pub(crate) fn profile_id(&self) -> VerifierProfileId {
        self.checked.profile.id()
    }

    pub(crate) fn root_epoch(&self) -> u64 {
        self.checked.profile.root_epoch()
    }

    pub(crate) fn observed_at_unix_ms(&self) -> u64 {
        self.checked.claim.observed_at_unix_ms()
    }

    pub(crate) fn id(&self) -> AuthenticatedVerificationEvidenceId {
        self.evidence_id
    }

    pub(crate) fn disposition(&self) -> ObligationDispositionV1 {
        let evidence_digest = *self.evidence_id.as_bytes();
        match self.checked.claim.outcome() {
            VerificationOutcomeV1::Satisfied => ObligationDispositionV1::Satisfied {
                evidence_digest,
                evidence_class: self.checked.profile.evidence_class(),
            },
            VerificationOutcomeV1::Failed => ObligationDispositionV1::Failed { evidence_digest },
            VerificationOutcomeV1::Inconclusive => {
                ObligationDispositionV1::Inconclusive { evidence_digest }
            }
            VerificationOutcomeV1::InfrastructureFailure => {
                ObligationDispositionV1::InfrastructureFailure { evidence_digest }
            }
            VerificationOutcomeV1::NotExecuted => {
                ObligationDispositionV1::NotExecuted { evidence_digest }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedVerificationEvidenceV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, VerificationAdmissionError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(VerificationAdmissionError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedVerificationEvidenceId(hash_authenticated(
            checked.claim.id(),
            checked.profile.id(),
            authentication_evidence_digest,
        ));
        Ok(Self {
            checked,
            authentication_evidence_digest,
            evidence_id,
        })
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
    #[error("authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
}

fn checked_text(field: &'static str, value: String) -> Result<String, VerificationAdmissionError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerificationAdmissionError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerificationAdmissionError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerificationAdmissionError::ControlCharacters { field });
    }
    Ok(trimmed.to_string())
}

fn hash_profile(
    profile_name: &str,
    root_digest: [u8; 32],
    root_epoch: u64,
    evidence_class: EvidenceClass,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, profile_name);
    bytes.extend_from_slice(&root_digest);
    bytes.extend_from_slice(&root_epoch.to_le_bytes());
    bytes.push(evidence_class_tag(evidence_class));
    domain_hash(PROFILE_DOMAIN, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    contract_id: ContinuityContractId,
    target_id: TargetRealizationId,
    requirement_id: ContinuityRequirementId,
    verifier_profile_id: VerifierProfileId,
    challenge: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: VerificationOutcomeV1,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(contract_id.as_bytes());
    bytes.extend_from_slice(target_id.as_bytes());
    bytes.extend_from_slice(requirement_id.as_bytes());
    bytes.extend_from_slice(verifier_profile_id.as_bytes());
    bytes.extend_from_slice(&challenge);
    bytes.extend_from_slice(&observed_at_unix_ms.to_le_bytes());
    bytes.push(outcome.tag());
    bytes.extend_from_slice(&raw_evidence_digest);
    domain_hash(CLAIM_DOMAIN, &bytes)
}

fn hash_authenticated(
    claim_id: VerificationEvidenceClaimId,
    verifier_profile_id: VerifierProfileId,
    authentication_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(claim_id.as_bytes());
    bytes.extend_from_slice(verifier_profile_id.as_bytes());
    bytes.extend_from_slice(&authentication_evidence_digest);
    domain_hash(AUTHENTICATED_DOMAIN, &bytes)
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
    use crate::witness::VerificationPolicyEntryV1;

    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            "machine-1",
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000,
            ObservationCoverage::Complete,
            EvidenceBasis::Tested,
            [1; 32],
            vec![],
        )
        .unwrap();
        let dependency = DependencyClaimV1::new(
            "role:research",
            "requires",
            "capability:cuda",
            DependencyBasis::Observed,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(),
            "cuda-workflow",
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: "cuda-fixture-v1".into(),
            },
            ApprovalBasis::ExplicitPolicy,
            [2; 32],
        )
        .unwrap();
        ContinuityContractV1::new("research-fleet", [3; 32], vec![requirement])
            .unwrap()
            .validate()
            .unwrap()
    }

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn policy(contract: &ValidatedContinuityContractV1) -> VerificationPolicyV1 {
        VerificationPolicyV1::new(
            "hardware-verifier-v1",
            vec![VerificationPolicyEntryV1::new(
                contract.requirements()[0].id(),
                EvidenceClass::Simulated,
            )],
        )
        .unwrap()
    }

    fn claim(
        contract: &ValidatedContinuityContractV1,
        target: TargetRealizationId,
        profile: &VerifierProfileV1,
        challenge: [u8; 32],
    ) -> VerificationEvidenceClaimV1 {
        VerificationEvidenceClaimV1::new(
            contract.id(),
            target,
            contract.requirements()[0].id(),
            profile.id(),
            challenge,
            1_700_000_000_111,
            VerificationOutcomeV1::Satisfied,
            [8; 32],
        )
        .unwrap()
    }

    #[test]
    fn raw_claim_has_no_caller_selectable_evidence_class() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let claim = claim(&contract, target, &profile, [5; 32]);
        assert_eq!(claim.verifier_profile_id(), profile.id());
        assert_eq!(profile.evidence_class(), EvidenceClass::HardwareVerified);
    }

    #[test]
    fn cross_target_replay_fails_closed() {
        let contract = contract();
        let target_a = TargetRealizationId::from_digest([4; 32]).unwrap();
        let target_b = TargetRealizationId::from_digest([6; 32]).unwrap();
        let profile = profile();
        let result = policy_check_verification_evidence(
            &contract,
            target_b,
            &policy(&contract),
            &profile,
            [5; 32],
            claim(&contract, target_a, &profile, [5; 32]),
        );
        assert_eq!(result.unwrap_err(), VerificationAdmissionError::TargetMismatch);
    }

    #[test]
    fn cross_transaction_challenge_replay_fails_closed() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let result = policy_check_verification_evidence(
            &contract,
            target,
            &policy(&contract),
            &profile,
            [6; 32],
            claim(&contract, target, &profile, [5; 32]),
        );
        assert_eq!(result.unwrap_err(), VerificationAdmissionError::ChallengeMismatch);
    }

    #[test]
    fn verifier_profile_root_epoch_participates_in_identity() {
        let left = profile();
        let right = VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            8,
            EvidenceClass::HardwareVerified,
        )
        .unwrap();
        assert_ne!(left.id(), right.id());
    }

    #[test]
    fn authenticated_identity_binds_authentication_evidence() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let checked = policy_check_verification_evidence(
            &contract,
            target,
            &policy(&contract),
            &profile,
            [5; 32],
            claim(&contract, target, &profile, [5; 32]),
        )
        .unwrap();
        let left = AuthenticatedVerificationEvidenceV1::authenticate_for_test(
            checked.clone(),
            [10; 32],
        )
        .unwrap();
        let right = AuthenticatedVerificationEvidenceV1::authenticate_for_test(
            checked,
            [11; 32],
        )
        .unwrap();
        assert_ne!(left.id(), right.id());
        assert!(matches!(
            left.disposition(),
            ObligationDispositionV1::Satisfied {
                evidence_class: EvidenceClass::HardwareVerified,
                ..
            }
        ));
    }
}
