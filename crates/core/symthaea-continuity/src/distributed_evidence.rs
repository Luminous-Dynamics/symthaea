// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Failure-domain and recovery-path currentness claims for distributed continuity.
//!
//! These transport claims are deliberately untrusted. They carry no evidence class
//! and cannot directly qualify a transition. Evidence strength is owned by the
//! provisioned verifier profile, and production authentication remains a separate
//! Xenia/crypto boundary.
//!
//! Core theorem:
//!
//! `RawCurrentnessClaim != PolicyCheckedEvidence != AuthenticatedEvidence != DistributedTransitionWitness`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::RecoveryPathClassV1;
use crate::distributed_state::{DistributedStateContextId, ValidatedDistributedStateContextV1};
use crate::failure_domain::{FailureDomainPolicyId, ValidatedFailureDomainPolicyV1};
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};

pub const FAILURE_DOMAIN_STATE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-failure-domain-state-claim-v1";
pub const RECOVERY_PATH_STATE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-recovery-path-state-claim-v1";

const FAILURE_DOMAIN_CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.failure-domain-state-claim.v1\0";
const FAILURE_DOMAIN_AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-failure-domain-state.v1\0";
const RECOVERY_CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.recovery-path-state-claim.v1\0";
const RECOVERY_AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-recovery-path-state.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FailureDomainStateClaimId([u8; 32]);

impl FailureDomainStateClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedFailureDomainStateEvidenceId([u8; 32]);

impl AuthenticatedFailureDomainStateEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RecoveryPathStateClaimId([u8; 32]);

impl RecoveryPathStateClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedRecoveryPathStateEvidenceId([u8; 32]);

impl AuthenticatedRecoveryPathStateEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Coarse verifier assertion about whether the currently observed placement matches
/// one exact failure-domain policy partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureDomainObservationOutcomeV1 {
    MatchesPolicy,
    Mismatch,
    Unknown,
}

impl FailureDomainObservationOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::MatchesPolicy => 1,
            Self::Mismatch => 2,
            Self::Unknown => 3,
        }
    }
}

/// Transportable, untrusted observation claim for one exact failure-domain policy.
///
/// `MatchesPolicy` means the verifier adapter claims that current observed placement
/// matches the complete partition committed by `failure_domain_policy_id`. The raw
/// evidence digest must carry the underlying adapter-specific observation record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureDomainStateClaimV1 {
    schema_version: String,
    context_id: DistributedStateContextId,
    failure_domain_policy_id: FailureDomainPolicyId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    outcome: FailureDomainObservationOutcomeV1,
    raw_evidence_digest: [u8; 32],
    claim_id: FailureDomainStateClaimId,
}

impl FailureDomainStateClaimV1 {
    pub fn new(
        context: &ValidatedDistributedStateContextV1,
        policy: &ValidatedFailureDomainPolicyV1,
        verifier_profile_id: VerifierProfileId,
        observed_at_unix_ms: u64,
        outcome: FailureDomainObservationOutcomeV1,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, DistributedEvidenceError> {
        validate_policy_context(context, policy)?;
        validate_observation_material(observed_at_unix_ms, raw_evidence_digest)?;

        let context_id = context.id();
        let failure_domain_policy_id = policy.id();
        let claim_id = FailureDomainStateClaimId(hash_failure_domain_claim(
            context_id,
            failure_domain_policy_id,
            verifier_profile_id,
            observed_at_unix_ms,
            outcome,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: FAILURE_DOMAIN_STATE_CLAIM_SCHEMA_V1.to_owned(),
            context_id,
            failure_domain_policy_id,
            verifier_profile_id,
            observed_at_unix_ms,
            outcome,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), DistributedEvidenceError> {
        if self.schema_version != FAILURE_DOMAIN_STATE_CLAIM_SCHEMA_V1 {
            return Err(DistributedEvidenceError::UnsupportedFailureDomainClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_observation_material(self.observed_at_unix_ms, self.raw_evidence_digest)?;
        let expected = FailureDomainStateClaimId(hash_failure_domain_claim(
            self.context_id,
            self.failure_domain_policy_id,
            self.verifier_profile_id,
            self.observed_at_unix_ms,
            self.outcome,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(DistributedEvidenceError::FailureDomainClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> FailureDomainStateClaimId {
        self.claim_id
    }

    pub fn context_id(&self) -> DistributedStateContextId {
        self.context_id
    }

    pub fn failure_domain_policy_id(&self) -> FailureDomainPolicyId {
        self.failure_domain_policy_id
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    pub fn observed_at_unix_ms(&self) -> u64 {
        self.observed_at_unix_ms
    }

    pub fn outcome(&self) -> FailureDomainObservationOutcomeV1 {
        self.outcome
    }

    pub fn raw_evidence_digest(&self) -> [u8; 32] {
        self.raw_evidence_digest
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedFailureDomainStateEvidenceV1 {
    claim: FailureDomainStateClaimV1,
    profile: VerifierProfileV1,
}

pub(crate) fn policy_check_failure_domain_state_claim(
    context: &ValidatedDistributedStateContextV1,
    policy: &ValidatedFailureDomainPolicyV1,
    profile: &VerifierProfileV1,
    claim: FailureDomainStateClaimV1,
) -> Result<PolicyCheckedFailureDomainStateEvidenceV1, DistributedEvidenceError> {
    profile.validate()?;
    context.as_raw().validate()?;
    policy.as_raw().validate()?;
    claim.validate()?;
    validate_policy_context(context, policy)?;
    if claim.context_id() != context.id() {
        return Err(DistributedEvidenceError::ClaimContextMismatch);
    }
    if claim.failure_domain_policy_id() != policy.id() {
        return Err(DistributedEvidenceError::FailureDomainPolicyMismatch);
    }
    if claim.verifier_profile_id() != profile.id() {
        return Err(DistributedEvidenceError::VerifierProfileClaimMismatch);
    }
    Ok(PolicyCheckedFailureDomainStateEvidenceV1 {
        claim,
        profile: profile.clone(),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedFailureDomainStateEvidenceV1 {
    checked: PolicyCheckedFailureDomainStateEvidenceV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedFailureDomainStateEvidenceId,
}

impl AuthenticatedFailureDomainStateEvidenceV1 {
    pub(crate) fn id(&self) -> AuthenticatedFailureDomainStateEvidenceId {
        self.evidence_id
    }

    pub(crate) fn context_id(&self) -> DistributedStateContextId {
        self.checked.claim.context_id()
    }

    pub(crate) fn policy_id(&self) -> FailureDomainPolicyId {
        self.checked.claim.failure_domain_policy_id()
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

    pub(crate) fn outcome(&self) -> FailureDomainObservationOutcomeV1 {
        self.checked.claim.outcome()
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedFailureDomainStateEvidenceV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, DistributedEvidenceError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(DistributedEvidenceError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedFailureDomainStateEvidenceId(hash_authenticated(
            FAILURE_DOMAIN_AUTH_DOMAIN,
            checked.claim.id().as_bytes(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryPathObservationOutcomeV1 {
    Available,
    Unavailable,
    Unknown,
}

impl RecoveryPathObservationOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Available => 1,
            Self::Unavailable => 2,
            Self::Unknown => 3,
        }
    }
}

/// Transportable, untrusted claim about one policy-acceptable recovery-path class.
///
/// `Available` requires both a stable opaque path identity and separate evidence
/// that the path is independent from the transition world. This prevents merely
/// naming the same in-band path as "recovery" from satisfying policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryPathStateClaimV1 {
    schema_version: String,
    context_id: DistributedStateContextId,
    recovery_path_class: RecoveryPathClassV1,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    outcome: RecoveryPathObservationOutcomeV1,
    recovery_path_identity_digest: Option<[u8; 32]>,
    independence_evidence_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: RecoveryPathStateClaimId,
}

impl RecoveryPathStateClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &ValidatedDistributedStateContextV1,
        recovery_path_class: RecoveryPathClassV1,
        verifier_profile_id: VerifierProfileId,
        observed_at_unix_ms: u64,
        outcome: RecoveryPathObservationOutcomeV1,
        recovery_path_identity_digest: Option<[u8; 32]>,
        independence_evidence_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, DistributedEvidenceError> {
        if context
            .budget()
            .recovery_path_any_of()
            .binary_search(&recovery_path_class)
            .is_err()
        {
            return Err(DistributedEvidenceError::RecoveryClassOutsideBudget);
        }
        validate_recovery_material(
            observed_at_unix_ms,
            outcome,
            recovery_path_identity_digest,
            independence_evidence_digest,
            raw_evidence_digest,
        )?;
        let context_id = context.id();
        let claim_id = RecoveryPathStateClaimId(hash_recovery_claim(
            context_id,
            &recovery_path_class,
            verifier_profile_id,
            observed_at_unix_ms,
            outcome,
            recovery_path_identity_digest,
            independence_evidence_digest,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: RECOVERY_PATH_STATE_CLAIM_SCHEMA_V1.to_owned(),
            context_id,
            recovery_path_class,
            verifier_profile_id,
            observed_at_unix_ms,
            outcome,
            recovery_path_identity_digest,
            independence_evidence_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), DistributedEvidenceError> {
        if self.schema_version != RECOVERY_PATH_STATE_CLAIM_SCHEMA_V1 {
            return Err(DistributedEvidenceError::UnsupportedRecoveryClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_recovery_material(
            self.observed_at_unix_ms,
            self.outcome,
            self.recovery_path_identity_digest,
            self.independence_evidence_digest,
            self.raw_evidence_digest,
        )?;
        let expected = RecoveryPathStateClaimId(hash_recovery_claim(
            self.context_id,
            &self.recovery_path_class,
            self.verifier_profile_id,
            self.observed_at_unix_ms,
            self.outcome,
            self.recovery_path_identity_digest,
            self.independence_evidence_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(DistributedEvidenceError::RecoveryClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> RecoveryPathStateClaimId {
        self.claim_id
    }

    pub fn context_id(&self) -> DistributedStateContextId {
        self.context_id
    }

    pub fn recovery_path_class(&self) -> &RecoveryPathClassV1 {
        &self.recovery_path_class
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    pub fn observed_at_unix_ms(&self) -> u64 {
        self.observed_at_unix_ms
    }

    pub fn outcome(&self) -> RecoveryPathObservationOutcomeV1 {
        self.outcome
    }

    pub fn recovery_path_identity_digest(&self) -> Option<[u8; 32]> {
        self.recovery_path_identity_digest
    }

    pub fn independence_evidence_digest(&self) -> Option<[u8; 32]> {
        self.independence_evidence_digest
    }

    pub fn raw_evidence_digest(&self) -> [u8; 32] {
        self.raw_evidence_digest
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedRecoveryPathStateEvidenceV1 {
    claim: RecoveryPathStateClaimV1,
    profile: VerifierProfileV1,
}

pub(crate) fn policy_check_recovery_path_state_claim(
    context: &ValidatedDistributedStateContextV1,
    profile: &VerifierProfileV1,
    claim: RecoveryPathStateClaimV1,
) -> Result<PolicyCheckedRecoveryPathStateEvidenceV1, DistributedEvidenceError> {
    profile.validate()?;
    context.as_raw().validate()?;
    claim.validate()?;
    if claim.context_id() != context.id() {
        return Err(DistributedEvidenceError::ClaimContextMismatch);
    }
    if claim.verifier_profile_id() != profile.id() {
        return Err(DistributedEvidenceError::VerifierProfileClaimMismatch);
    }
    if context
        .budget()
        .recovery_path_any_of()
        .binary_search(claim.recovery_path_class())
        .is_err()
    {
        return Err(DistributedEvidenceError::RecoveryClassOutsideBudget);
    }
    Ok(PolicyCheckedRecoveryPathStateEvidenceV1 {
        claim,
        profile: profile.clone(),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedRecoveryPathStateEvidenceV1 {
    checked: PolicyCheckedRecoveryPathStateEvidenceV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedRecoveryPathStateEvidenceId,
}

impl AuthenticatedRecoveryPathStateEvidenceV1 {
    pub(crate) fn id(&self) -> AuthenticatedRecoveryPathStateEvidenceId {
        self.evidence_id
    }

    pub(crate) fn context_id(&self) -> DistributedStateContextId {
        self.checked.claim.context_id()
    }

    pub(crate) fn recovery_path_class(&self) -> &RecoveryPathClassV1 {
        self.checked.claim.recovery_path_class()
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

    pub(crate) fn outcome(&self) -> RecoveryPathObservationOutcomeV1 {
        self.checked.claim.outcome()
    }

    pub(crate) fn recovery_path_identity_digest(&self) -> Option<[u8; 32]> {
        self.checked.claim.recovery_path_identity_digest()
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedRecoveryPathStateEvidenceV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, DistributedEvidenceError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(DistributedEvidenceError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedRecoveryPathStateEvidenceId(hash_authenticated(
            RECOVERY_AUTH_DOMAIN,
            checked.claim.id().as_bytes(),
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
pub enum DistributedEvidenceError {
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error("unsupported failure-domain state claim schema: {0}")]
    UnsupportedFailureDomainClaimSchema(String),
    #[error("unsupported recovery-path state claim schema: {0}")]
    UnsupportedRecoveryClaimSchema(String),
    #[error("distributed currentness observation time must be non-zero")]
    ZeroObservationTime,
    #[error("distributed currentness raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("recovery-path identity digest must be non-zero when present")]
    ZeroRecoveryPathIdentityDigest,
    #[error("recovery independence evidence digest must be non-zero when present")]
    ZeroIndependenceEvidenceDigest,
    #[error("available recovery path requires exact path identity")]
    MissingRecoveryPathIdentity,
    #[error("available recovery path requires independence evidence")]
    MissingIndependenceEvidence,
    #[error("unavailable recovery path may not claim independence evidence")]
    IndependenceEvidenceOnUnavailablePath,
    #[error("unknown recovery-path state may not claim path identity or independence evidence")]
    MaterialOnUnknownRecoveryPath,
    #[error("failure-domain policy is bound to a different distributed transaction world")]
    FailureDomainPolicyContextMismatch,
    #[error("failure-domain state claim belongs to a different transaction context")]
    ClaimContextMismatch,
    #[error("failure-domain state claim references a different failure-domain policy")]
    FailureDomainPolicyMismatch,
    #[error("distributed currentness claim binds a different verifier profile")]
    VerifierProfileClaimMismatch,
    #[error("recovery-path class is not permitted by the distributed change budget")]
    RecoveryClassOutsideBudget,
    #[error("stored failure-domain state claim identity does not match canonical fields")]
    FailureDomainClaimIdentityMismatch,
    #[error("stored recovery-path state claim identity does not match canonical fields")]
    RecoveryClaimIdentityMismatch,
    #[error("authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
}

fn validate_policy_context(
    context: &ValidatedDistributedStateContextV1,
    policy: &ValidatedFailureDomainPolicyV1,
) -> Result<(), DistributedEvidenceError> {
    if policy.aggregate_subject_id() != context.aggregate_subject_id()
        || policy.budget_id() != context.budget_id()
        || policy.budget_generation() != context.budget_generation()
    {
        return Err(DistributedEvidenceError::FailureDomainPolicyContextMismatch);
    }
    Ok(())
}

fn validate_observation_material(
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
) -> Result<(), DistributedEvidenceError> {
    if observed_at_unix_ms == 0 {
        return Err(DistributedEvidenceError::ZeroObservationTime);
    }
    if raw_evidence_digest == [0; 32] {
        return Err(DistributedEvidenceError::ZeroRawEvidenceDigest);
    }
    Ok(())
}

fn validate_recovery_material(
    observed_at_unix_ms: u64,
    outcome: RecoveryPathObservationOutcomeV1,
    recovery_path_identity_digest: Option<[u8; 32]>,
    independence_evidence_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), DistributedEvidenceError> {
    validate_observation_material(observed_at_unix_ms, raw_evidence_digest)?;
    if matches!(recovery_path_identity_digest, Some(digest) if digest == [0; 32]) {
        return Err(DistributedEvidenceError::ZeroRecoveryPathIdentityDigest);
    }
    if matches!(independence_evidence_digest, Some(digest) if digest == [0; 32]) {
        return Err(DistributedEvidenceError::ZeroIndependenceEvidenceDigest);
    }
    match outcome {
        RecoveryPathObservationOutcomeV1::Available => {
            if recovery_path_identity_digest.is_none() {
                return Err(DistributedEvidenceError::MissingRecoveryPathIdentity);
            }
            if independence_evidence_digest.is_none() {
                return Err(DistributedEvidenceError::MissingIndependenceEvidence);
            }
        }
        RecoveryPathObservationOutcomeV1::Unavailable => {
            if independence_evidence_digest.is_some() {
                return Err(DistributedEvidenceError::IndependenceEvidenceOnUnavailablePath);
            }
        }
        RecoveryPathObservationOutcomeV1::Unknown => {
            if recovery_path_identity_digest.is_some() || independence_evidence_digest.is_some() {
                return Err(DistributedEvidenceError::MaterialOnUnknownRecoveryPath);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_failure_domain_claim(
    context_id: DistributedStateContextId,
    policy_id: FailureDomainPolicyId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    outcome: FailureDomainObservationOutcomeV1,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(144);
    bytes.extend_from_slice(context_id.as_bytes());
    bytes.extend_from_slice(policy_id.as_bytes());
    bytes.extend_from_slice(verifier_profile_id.as_bytes());
    bytes.extend_from_slice(&observed_at_unix_ms.to_le_bytes());
    bytes.push(outcome.tag());
    bytes.extend_from_slice(&raw_evidence_digest);
    domain_hash(FAILURE_DOMAIN_CLAIM_DOMAIN, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn hash_recovery_claim(
    context_id: DistributedStateContextId,
    recovery_path_class: &RecoveryPathClassV1,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    outcome: RecoveryPathObservationOutcomeV1,
    recovery_path_identity_digest: Option<[u8; 32]>,
    independence_evidence_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(192);
    bytes.extend_from_slice(context_id.as_bytes());
    encode_recovery_class(&mut bytes, recovery_path_class);
    bytes.extend_from_slice(verifier_profile_id.as_bytes());
    bytes.extend_from_slice(&observed_at_unix_ms.to_le_bytes());
    bytes.push(outcome.tag());
    put_optional_digest(&mut bytes, recovery_path_identity_digest);
    put_optional_digest(&mut bytes, independence_evidence_digest);
    bytes.extend_from_slice(&raw_evidence_digest);
    domain_hash(RECOVERY_CLAIM_DOMAIN, &bytes)
}

fn hash_authenticated(
    domain: &[u8],
    claim_id: &[u8; 32],
    profile_id: VerifierProfileId,
    authentication_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(96);
    bytes.extend_from_slice(claim_id);
    bytes.extend_from_slice(profile_id.as_bytes());
    bytes.extend_from_slice(&authentication_evidence_digest);
    domain_hash(domain, &bytes)
}

fn encode_recovery_class(out: &mut Vec<u8>, class: &RecoveryPathClassV1) {
    match class {
        RecoveryPathClassV1::DeviceLocalAutomaticRollback => out.push(1),
        RecoveryPathClassV1::OutOfBandManagement => out.push(2),
        RecoveryPathClassV1::IndependentNetworkPath => out.push(3),
        RecoveryPathClassV1::LocalPhysicalIntervention => out.push(4),
        RecoveryPathClassV1::Custom { kind_id } => {
            out.push(255);
            put_len(out, kind_id.len());
            out.extend_from_slice(kind_id.as_bytes());
        }
    }
}

fn put_optional_digest(out: &mut Vec<u8>, digest: Option<[u8; 32]>) {
    match digest {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value);
        }
        None => out.push(0),
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{DistributedChangeBudgetV1, RecoveryPathClassV1};
    use crate::distributed_state::DistributedStateContextV1;
    use crate::failure_domain::{
        FailureDomainGroupV1, FailureDomainKindV1, FailureDomainPolicyV1,
    };
    use crate::scope::{ContinuityScopeV1, ContinuitySubjectV1};
    use crate::witness::EvidenceClass;

    fn subject(logical_id: &str, scope: ContinuityScopeV1) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new("org.example", logical_id, scope, None).unwrap()
    }

    fn fixture() -> (
        Vec<ContinuitySubjectV1>,
        ValidatedDistributedStateContextV1,
        ValidatedFailureDomainPolicyV1,
    ) {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let participants = vec![
            subject("node-a", ContinuityScopeV1::Machine),
            subject("node-b", ContinuityScopeV1::Machine),
            subject("node-c", ContinuityScopeV1::Machine),
        ];
        let raw_budget = DistributedChangeBudgetV1::new(
            &aggregate,
            3,
            participants.iter().map(ContinuitySubjectV1::id).collect(),
            1,
            2,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let budget = raw_budget.validate_against_subject(&aggregate).unwrap();
        let context = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let raw_policy = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            vec![
                FailureDomainGroupV1::new(
                    "rack-a",
                    vec![participants[0].id(), participants[1].id()],
                )
                .unwrap(),
                FailureDomainGroupV1::new("rack-b", vec![participants[2].id()]).unwrap(),
            ],
            1,
        )
        .unwrap();
        let policy = raw_policy.validate_against_budget(&budget).unwrap();
        (participants, context, policy)
    }

    fn profile(seed: u8) -> VerifierProfileV1 {
        VerifierProfileV1::new(
            format!("distributed-evidence-{seed}"),
            [seed; 32],
            4,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    #[test]
    fn failure_domain_match_requires_exact_context_policy_and_profile() {
        let (_, context, policy) = fixture();
        let verifier = profile(7);
        let claim = FailureDomainStateClaimV1::new(
            &context,
            &policy,
            verifier.id(),
            1_700_000_000_000,
            FailureDomainObservationOutcomeV1::MatchesPolicy,
            [4; 32],
        )
        .unwrap();
        let checked =
            policy_check_failure_domain_state_claim(&context, &policy, &verifier, claim).unwrap();
        let authenticated =
            AuthenticatedFailureDomainStateEvidenceV1::authenticate_for_test(checked, [5; 32])
                .unwrap();
        assert_eq!(authenticated.context_id(), context.id());
        assert_eq!(authenticated.policy_id(), policy.id());
        assert_eq!(
            authenticated.outcome(),
            FailureDomainObservationOutcomeV1::MatchesPolicy
        );
    }

    #[test]
    fn available_recovery_requires_identity_and_independence() {
        let (_, context, _) = fixture();
        let verifier = profile(7);
        assert_eq!(
            RecoveryPathStateClaimV1::new(
                &context,
                RecoveryPathClassV1::OutOfBandManagement,
                verifier.id(),
                1_700_000_000_000,
                RecoveryPathObservationOutcomeV1::Available,
                Some([6; 32]),
                None,
                [4; 32],
            )
            .unwrap_err(),
            DistributedEvidenceError::MissingIndependenceEvidence
        );
    }

    #[test]
    fn recovery_class_outside_budget_fails_closed() {
        let (_, context, _) = fixture();
        let verifier = profile(7);
        assert_eq!(
            RecoveryPathStateClaimV1::new(
                &context,
                RecoveryPathClassV1::IndependentNetworkPath,
                verifier.id(),
                1_700_000_000_000,
                RecoveryPathObservationOutcomeV1::Unknown,
                None,
                None,
                [4; 32],
            )
            .unwrap_err(),
            DistributedEvidenceError::RecoveryClassOutsideBudget
        );
    }

    #[test]
    fn available_recovery_can_be_policy_checked_and_authenticated() {
        let (_, context, _) = fixture();
        let verifier = profile(7);
        let claim = RecoveryPathStateClaimV1::new(
            &context,
            RecoveryPathClassV1::OutOfBandManagement,
            verifier.id(),
            1_700_000_000_000,
            RecoveryPathObservationOutcomeV1::Available,
            Some([6; 32]),
            Some([7; 32]),
            [4; 32],
        )
        .unwrap();
        let checked = policy_check_recovery_path_state_claim(&context, &verifier, claim).unwrap();
        let authenticated =
            AuthenticatedRecoveryPathStateEvidenceV1::authenticate_for_test(checked, [5; 32])
                .unwrap();
        assert_eq!(authenticated.context_id(), context.id());
        assert_eq!(
            authenticated.outcome(),
            RecoveryPathObservationOutcomeV1::Available
        );
        assert_eq!(authenticated.recovery_path_identity_digest(), Some([6; 32]));
    }

    #[test]
    fn unknown_recovery_preserves_absence_of_material() {
        let (_, context, _) = fixture();
        let verifier = profile(7);
        let claim = RecoveryPathStateClaimV1::new(
            &context,
            RecoveryPathClassV1::OutOfBandManagement,
            verifier.id(),
            1_700_000_000_000,
            RecoveryPathObservationOutcomeV1::Unknown,
            None,
            None,
            [4; 32],
        )
        .unwrap();
        claim.validate().unwrap();
    }

    #[test]
    fn unavailable_recovery_cannot_claim_independence() {
        let (_, context, _) = fixture();
        let verifier = profile(7);
        assert_eq!(
            RecoveryPathStateClaimV1::new(
                &context,
                RecoveryPathClassV1::OutOfBandManagement,
                verifier.id(),
                1_700_000_000_000,
                RecoveryPathObservationOutcomeV1::Unavailable,
                Some([6; 32]),
                Some([7; 32]),
                [4; 32],
            )
            .unwrap_err(),
            DistributedEvidenceError::IndependenceEvidenceOnUnavailablePath
        );
    }

    #[test]
    fn failure_domain_claim_from_other_context_is_rejected() {
        let (participants, context, policy) = fixture();
        let verifier = profile(7);
        let other_context = DistributedStateContextV1::new(
            context.budget(),
            vec![participants[1].id()],
            [8; 32],
        )
        .unwrap()
        .validate_against_budget(context.budget())
        .unwrap();
        let claim = FailureDomainStateClaimV1::new(
            &context,
            &policy,
            verifier.id(),
            1_700_000_000_000,
            FailureDomainObservationOutcomeV1::MatchesPolicy,
            [4; 32],
        )
        .unwrap();
        assert_eq!(
            policy_check_failure_domain_state_claim(
                &other_context,
                &policy,
                &verifier,
                claim,
            )
            .unwrap_err(),
            DistributedEvidenceError::ClaimContextMismatch
        );
    }
}
