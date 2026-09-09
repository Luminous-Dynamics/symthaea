// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independently authenticated authority for one exact continuity transition intent.
//!
//! Authority is deliberately separate from safety evidence. An operator may authorize
//! one exact subject/target/distributed transaction context, while participant health,
//! failure-domain state, recovery state, and local currentness are refreshed later.
//!
//! Core theorem:
//!
//! `TransitionIntent != AuthenticatedAuthority != CommitEligibility != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_qualification::QualifiedDistributedTransitionWitnessV1;
use crate::distributed_state::DistributedStateContextId;
use crate::scope::ContinuitySubjectId;
use crate::subject_witness::{
    SubjectBoundQualifiedContinuityWitnessId, SubjectBoundQualifiedContinuityWitnessV1,
    SubjectWitnessBindingError,
};
use crate::witness::TargetRealizationId;

pub const TRANSITION_AUTHORITY_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-transition-authority-profile-v1";
pub const TRANSITION_AUTHORITY_POLICY_SCHEMA_V1: &str =
    "symthaea-continuity-transition-authority-policy-v1";
pub const TRANSITION_AUTHORITY_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-transition-authority-claim-v1";
pub const TRANSITION_AUTHORITY_XENIA_PURPOSE: &str =
    "symthaea.continuity.transition-authority.v1";

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.transition-authority-profile.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.transition-authority-policy.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.transition-authority-claim.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-transition-authority.v1\0";
const CLAIM_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.transition-authority-wire.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_PROFILES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransitionAuthorityProfileId([u8; 32]);

impl TransitionAuthorityProfileId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransitionAuthorityPolicyId([u8; 32]);

impl TransitionAuthorityPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransitionAuthorityClaimId([u8; 32]);

impl TransitionAuthorityClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedTransitionAuthorityId([u8; 32]);

impl AuthenticatedTransitionAuthorityId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Provisioned authority root. This is intentionally distinct from a verifier
/// profile: permission to transition and evidence that a transition is safe are
/// different trust domains even when an organization happens to operate both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionAuthorityProfileV1 {
    schema_version: String,
    profile_name: String,
    root_digest: [u8; 32],
    root_epoch: u64,
    profile_id: TransitionAuthorityProfileId,
}

impl TransitionAuthorityProfileV1 {
    pub fn new(
        profile_name: impl Into<String>,
        root_digest: [u8; 32],
        root_epoch: u64,
    ) -> Result<Self, TransitionAuthorityError> {
        let profile_name = checked_text("authority profile name", profile_name.into())?;
        if root_digest == [0; 32] {
            return Err(TransitionAuthorityError::ZeroAuthorityRootDigest);
        }
        if root_epoch == 0 {
            return Err(TransitionAuthorityError::ZeroAuthorityRootEpoch);
        }
        let profile_id = TransitionAuthorityProfileId(hash_profile(
            &profile_name,
            root_digest,
            root_epoch,
        ));
        Ok(Self {
            schema_version: TRANSITION_AUTHORITY_PROFILE_SCHEMA_V1.to_owned(),
            profile_name,
            root_digest,
            root_epoch,
            profile_id,
        })
    }

    pub fn validate(&self) -> Result<(), TransitionAuthorityError> {
        if self.schema_version != TRANSITION_AUTHORITY_PROFILE_SCHEMA_V1 {
            return Err(TransitionAuthorityError::UnsupportedProfileSchema(
                self.schema_version.clone(),
            ));
        }
        let canonical = checked_text("authority profile name", self.profile_name.clone())?;
        if canonical != self.profile_name {
            return Err(TransitionAuthorityError::NonCanonicalProfileName);
        }
        if self.root_digest == [0; 32] {
            return Err(TransitionAuthorityError::ZeroAuthorityRootDigest);
        }
        if self.root_epoch == 0 {
            return Err(TransitionAuthorityError::ZeroAuthorityRootEpoch);
        }
        let expected = TransitionAuthorityProfileId(hash_profile(
            &self.profile_name,
            self.root_digest,
            self.root_epoch,
        ));
        if expected != self.profile_id {
            return Err(TransitionAuthorityError::ProfileIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> TransitionAuthorityProfileId {
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
}

/// Policy declaring which exact authority roots may approve a continuity transition
/// and the maximum lifetime of one approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionAuthorityPolicyV1 {
    schema_version: String,
    policy_generation: u64,
    allowed_profile_ids: Vec<TransitionAuthorityProfileId>,
    max_grant_lifetime_ms: u64,
    policy_id: TransitionAuthorityPolicyId,
}

impl TransitionAuthorityPolicyV1 {
    pub fn new(
        policy_generation: u64,
        mut allowed_profile_ids: Vec<TransitionAuthorityProfileId>,
        max_grant_lifetime_ms: u64,
    ) -> Result<Self, TransitionAuthorityError> {
        if policy_generation == 0 {
            return Err(TransitionAuthorityError::ZeroPolicyGeneration);
        }
        if max_grant_lifetime_ms == 0 {
            return Err(TransitionAuthorityError::ZeroGrantLifetime);
        }
        if allowed_profile_ids.is_empty() {
            return Err(TransitionAuthorityError::NoAllowedAuthorityProfiles);
        }
        if allowed_profile_ids.len() > MAX_PROFILES {
            return Err(TransitionAuthorityError::TooManyAuthorityProfiles);
        }
        allowed_profile_ids.sort();
        allowed_profile_ids.dedup();
        let policy_id = TransitionAuthorityPolicyId(hash_policy(
            policy_generation,
            &allowed_profile_ids,
            max_grant_lifetime_ms,
        ));
        Ok(Self {
            schema_version: TRANSITION_AUTHORITY_POLICY_SCHEMA_V1.to_owned(),
            policy_generation,
            allowed_profile_ids,
            max_grant_lifetime_ms,
            policy_id,
        })
    }

    pub fn validate(&self) -> Result<(), TransitionAuthorityError> {
        if self.schema_version != TRANSITION_AUTHORITY_POLICY_SCHEMA_V1 {
            return Err(TransitionAuthorityError::UnsupportedPolicySchema(
                self.schema_version.clone(),
            ));
        }
        if self.policy_generation == 0 {
            return Err(TransitionAuthorityError::ZeroPolicyGeneration);
        }
        if self.max_grant_lifetime_ms == 0 {
            return Err(TransitionAuthorityError::ZeroGrantLifetime);
        }
        if self.allowed_profile_ids.is_empty() {
            return Err(TransitionAuthorityError::NoAllowedAuthorityProfiles);
        }
        if self.allowed_profile_ids.len() > MAX_PROFILES {
            return Err(TransitionAuthorityError::TooManyAuthorityProfiles);
        }
        if self
            .allowed_profile_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(TransitionAuthorityError::NonCanonicalAuthorityProfiles);
        }
        let expected = TransitionAuthorityPolicyId(hash_policy(
            self.policy_generation,
            &self.allowed_profile_ids,
            self.max_grant_lifetime_ms,
        ));
        if expected != self.policy_id {
            return Err(TransitionAuthorityError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn validate_profile(
        &self,
        profile: &TransitionAuthorityProfileV1,
    ) -> Result<ValidatedTransitionAuthorityPolicyV1, TransitionAuthorityError> {
        self.validate()?;
        profile.validate()?;
        if self.allowed_profile_ids.binary_search(&profile.id()).is_err() {
            return Err(TransitionAuthorityError::AuthorityProfileNotAllowed);
        }
        Ok(ValidatedTransitionAuthorityPolicyV1 {
            inner: self.clone(),
            profile: profile.clone(),
        })
    }

    pub fn id(&self) -> TransitionAuthorityPolicyId {
        self.policy_id
    }

    pub fn policy_generation(&self) -> u64 {
        self.policy_generation
    }

    pub fn max_grant_lifetime_ms(&self) -> u64 {
        self.max_grant_lifetime_ms
    }

    pub fn allowed_profile_ids(&self) -> &[TransitionAuthorityProfileId] {
        &self.allowed_profile_ids
    }
}

/// Non-Serde authority policy rebound to one exact allowed authority root.
#[derive(Debug, Clone)]
pub struct ValidatedTransitionAuthorityPolicyV1 {
    inner: TransitionAuthorityPolicyV1,
    profile: TransitionAuthorityProfileV1,
}

impl ValidatedTransitionAuthorityPolicyV1 {
    pub fn id(&self) -> TransitionAuthorityPolicyId {
        self.inner.id()
    }

    pub fn policy_generation(&self) -> u64 {
        self.inner.policy_generation()
    }

    pub fn max_grant_lifetime_ms(&self) -> u64 {
        self.inner.max_grant_lifetime_ms()
    }

    pub fn profile_id(&self) -> TransitionAuthorityProfileId {
        self.profile.id()
    }

    pub fn profile_root_epoch(&self) -> u64 {
        self.profile.root_epoch()
    }

    pub fn as_raw(&self) -> &TransitionAuthorityPolicyV1 {
        &self.inner
    }
}

/// An authority grant over stable transition intent. It deliberately binds the
/// distributed *context* rather than a specific current-state witness so health can
/// be re-evaluated under the same exact authorized transaction without re-signing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionAuthorityClaimV1 {
    schema_version: String,
    subject_witness_id: SubjectBoundQualifiedContinuityWitnessId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    authority_profile_id: TransitionAuthorityProfileId,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    authority_basis_digest: [u8; 32],
    claim_id: TransitionAuthorityClaimId,
}

impl TransitionAuthorityClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
        distributed_witness: &QualifiedDistributedTransitionWitnessV1,
        authority_profile_id: TransitionAuthorityProfileId,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
        authority_basis_digest: [u8; 32],
    ) -> Result<Self, TransitionAuthorityError> {
        local_witness.validate()?;
        require_local_candidate(local_witness, distributed_witness)?;
        validate_grant_material(
            valid_from_unix_ms,
            valid_until_unix_ms,
            authority_basis_digest,
        )?;
        let subject_witness_id = local_witness.id();
        let subject_id = local_witness.subject_id();
        let target_realization_id = local_witness.target_realization_id();
        let distributed_context_id = distributed_witness.context_id();
        let claim_id = TransitionAuthorityClaimId(hash_claim(
            subject_witness_id,
            subject_id,
            target_realization_id,
            distributed_context_id,
            authority_profile_id,
            valid_from_unix_ms,
            valid_until_unix_ms,
            authority_basis_digest,
        ));
        Ok(Self {
            schema_version: TRANSITION_AUTHORITY_CLAIM_SCHEMA_V1.to_owned(),
            subject_witness_id,
            subject_id,
            target_realization_id,
            distributed_context_id,
            authority_profile_id,
            valid_from_unix_ms,
            valid_until_unix_ms,
            authority_basis_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), TransitionAuthorityError> {
        if self.schema_version != TRANSITION_AUTHORITY_CLAIM_SCHEMA_V1 {
            return Err(TransitionAuthorityError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_grant_material(
            self.valid_from_unix_ms,
            self.valid_until_unix_ms,
            self.authority_basis_digest,
        )?;
        let expected = TransitionAuthorityClaimId(hash_claim(
            self.subject_witness_id,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.authority_profile_id,
            self.valid_from_unix_ms,
            self.valid_until_unix_ms,
            self.authority_basis_digest,
        ));
        if expected != self.claim_id {
            return Err(TransitionAuthorityError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> TransitionAuthorityClaimId {
        self.claim_id
    }

    pub fn valid_from_unix_ms(&self) -> u64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }
}

/// Canonical transition-authority bytes for a future Xenia/signature adapter.
/// No serde representation is trusted as the signing payload.
pub fn canonical_transition_authority_claim_bytes(
    claim: &TransitionAuthorityClaimV1,
) -> Result<Vec<u8>, TransitionAuthorityError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(320);
    out.extend_from_slice(CLAIM_WIRE_DOMAIN);
    out.extend_from_slice(claim.subject_witness_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.target_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(claim.authority_profile_id.as_bytes());
    out.extend_from_slice(&claim.valid_from_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.valid_until_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.authority_basis_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_transition_authority_claim_digest(
    claim: &TransitionAuthorityClaimV1,
) -> Result<[u8; 32], TransitionAuthorityError> {
    let bytes = canonical_transition_authority_claim_bytes(claim)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedTransitionAuthorityV1 {
    claim: TransitionAuthorityClaimV1,
    policy_id: TransitionAuthorityPolicyId,
    policy_generation: u64,
    profile: TransitionAuthorityProfileV1,
}

pub(crate) fn policy_check_transition_authority_claim(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
    policy: &ValidatedTransitionAuthorityPolicyV1,
    claim: TransitionAuthorityClaimV1,
) -> Result<PolicyCheckedTransitionAuthorityV1, TransitionAuthorityError> {
    local_witness.validate()?;
    claim.validate()?;
    require_local_candidate(local_witness, distributed_witness)?;
    if claim.authority_profile_id != policy.profile_id() {
        return Err(TransitionAuthorityError::AuthorityProfileMismatch);
    }
    if claim.subject_witness_id != local_witness.id()
        || claim.subject_id != local_witness.subject_id()
        || claim.target_realization_id != local_witness.target_realization_id()
    {
        return Err(TransitionAuthorityError::LocalWitnessMismatch);
    }
    if claim.distributed_context_id != distributed_witness.context_id() {
        return Err(TransitionAuthorityError::DistributedContextMismatch);
    }
    let lifetime = claim.valid_until_unix_ms - claim.valid_from_unix_ms;
    if lifetime > policy.max_grant_lifetime_ms() {
        return Err(TransitionAuthorityError::GrantLifetimeExceedsPolicy {
            observed_ms: lifetime,
            allowed_ms: policy.max_grant_lifetime_ms(),
        });
    }
    Ok(PolicyCheckedTransitionAuthorityV1 {
        claim,
        policy_id: policy.id(),
        policy_generation: policy.policy_generation(),
        profile: policy.profile.clone(),
    })
}

/// Authenticated authority is constructor-owned. Production construction is reserved
/// for the Xenia/crypto adapter that verifies the canonical claim bytes against the
/// exact pinned authority root.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedTransitionAuthorityV1 {
    checked: PolicyCheckedTransitionAuthorityV1,
    authentication_evidence_digest: [u8; 32],
    authority_id: AuthenticatedTransitionAuthorityId,
}

impl AuthenticatedTransitionAuthorityV1 {
    pub(crate) fn id(&self) -> AuthenticatedTransitionAuthorityId {
        self.authority_id
    }

    pub(crate) fn claim_id(&self) -> TransitionAuthorityClaimId {
        self.checked.claim.id()
    }

    pub(crate) fn policy_id(&self) -> TransitionAuthorityPolicyId {
        self.checked.policy_id
    }

    pub(crate) fn policy_generation(&self) -> u64 {
        self.checked.policy_generation
    }

    pub(crate) fn profile_id(&self) -> TransitionAuthorityProfileId {
        self.checked.profile.id()
    }

    pub(crate) fn root_epoch(&self) -> u64 {
        self.checked.profile.root_epoch()
    }

    pub(crate) fn subject_witness_id(&self) -> SubjectBoundQualifiedContinuityWitnessId {
        self.checked.claim.subject_witness_id
    }

    pub(crate) fn subject_id(&self) -> ContinuitySubjectId {
        self.checked.claim.subject_id
    }

    pub(crate) fn target_realization_id(&self) -> TargetRealizationId {
        self.checked.claim.target_realization_id
    }

    pub(crate) fn distributed_context_id(&self) -> DistributedStateContextId {
        self.checked.claim.distributed_context_id
    }

    pub(crate) fn valid_from_unix_ms(&self) -> u64 {
        self.checked.claim.valid_from_unix_ms
    }

    pub(crate) fn valid_until_unix_ms(&self) -> u64 {
        self.checked.claim.valid_until_unix_ms
    }

    pub(crate) fn authority_basis_digest(&self) -> [u8; 32] {
        self.checked.claim.authority_basis_digest
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedTransitionAuthorityV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, TransitionAuthorityError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(TransitionAuthorityError::ZeroAuthenticationEvidenceDigest);
        }
        let authority_id = AuthenticatedTransitionAuthorityId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                checked.claim.id().as_bytes(),
                checked.policy_id.as_bytes(),
                checked.profile.id().as_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self {
            checked,
            authentication_evidence_digest,
            authority_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransitionAuthorityError {
    #[error(transparent)]
    SubjectWitness(#[from] SubjectWitnessBindingError),
    #[error("unsupported transition authority profile schema: {0}")]
    UnsupportedProfileSchema(String),
    #[error("unsupported transition authority policy schema: {0}")]
    UnsupportedPolicySchema(String),
    #[error("unsupported transition authority claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds the text bound")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("authority profile name is not canonical")]
    NonCanonicalProfileName,
    #[error("transition authority root digest must be non-zero")]
    ZeroAuthorityRootDigest,
    #[error("transition authority root epoch must be non-zero")]
    ZeroAuthorityRootEpoch,
    #[error("transition authority profile identity mismatch")]
    ProfileIdentityMismatch,
    #[error("transition authority policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("transition authority grant lifetime must be non-zero")]
    ZeroGrantLifetime,
    #[error("transition authority policy requires at least one allowed profile")]
    NoAllowedAuthorityProfiles,
    #[error("transition authority policy exceeds the allowed profile bound")]
    TooManyAuthorityProfiles,
    #[error("transition authority policy profile list must be canonical sorted unique")]
    NonCanonicalAuthorityProfiles,
    #[error("transition authority policy identity mismatch")]
    PolicyIdentityMismatch,
    #[error("authority profile is not allowed by policy")]
    AuthorityProfileNotAllowed,
    #[error("local continuity subject is not in the exact distributed transition candidate set")]
    LocalSubjectNotCandidate,
    #[error("transition authority validity times must be non-zero and strictly increasing")]
    InvalidGrantWindow,
    #[error("transition authority basis digest must be non-zero")]
    ZeroAuthorityBasisDigest,
    #[error("transition authority claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("transition authority claim profile does not match the selected allowed profile")]
    AuthorityProfileMismatch,
    #[error("transition authority claim does not bind the exact local witness/subject/target")]
    LocalWitnessMismatch,
    #[error("transition authority claim does not bind the exact distributed transaction context")]
    DistributedContextMismatch,
    #[error("transition authority grant lifetime {observed_ms} ms exceeds policy maximum {allowed_ms} ms")]
    GrantLifetimeExceedsPolicy { observed_ms: u64, allowed_ms: u64 },
    #[error("transition authority authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
}

fn require_local_candidate(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
) -> Result<(), TransitionAuthorityError> {
    if distributed_witness
        .candidate_subject_ids()
        .binary_search(&local_witness.subject_id())
        .is_err()
    {
        return Err(TransitionAuthorityError::LocalSubjectNotCandidate);
    }
    Ok(())
}

fn validate_grant_material(
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    authority_basis_digest: [u8; 32],
) -> Result<(), TransitionAuthorityError> {
    if valid_from_unix_ms == 0
        || valid_until_unix_ms == 0
        || valid_until_unix_ms <= valid_from_unix_ms
    {
        return Err(TransitionAuthorityError::InvalidGrantWindow);
    }
    if authority_basis_digest == [0; 32] {
        return Err(TransitionAuthorityError::ZeroAuthorityBasisDigest);
    }
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, TransitionAuthorityError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(TransitionAuthorityError::BlankText { field });
    }
    if trimmed.len() > MAX_TEXT_BYTES {
        return Err(TransitionAuthorityError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(TransitionAuthorityError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn hash_profile(name: &str, root_digest: [u8; 32], root_epoch: u64) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, name);
    bytes.extend_from_slice(&root_digest);
    bytes.extend_from_slice(&root_epoch.to_le_bytes());
    domain_hash(PROFILE_DOMAIN, &bytes)
}

fn hash_policy(
    policy_generation: u64,
    allowed_profile_ids: &[TransitionAuthorityProfileId],
    max_grant_lifetime_ms: u64,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&policy_generation.to_le_bytes());
    put_len(&mut bytes, allowed_profile_ids.len());
    for id in allowed_profile_ids {
        bytes.extend_from_slice(id.as_bytes());
    }
    bytes.extend_from_slice(&max_grant_lifetime_ms.to_le_bytes());
    domain_hash(POLICY_DOMAIN, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    subject_witness_id: SubjectBoundQualifiedContinuityWitnessId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    authority_profile_id: TransitionAuthorityProfileId,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    authority_basis_digest: [u8; 32],
) -> [u8; 32] {
    domain_hash_parts(
        CLAIM_DOMAIN,
        &[
            subject_witness_id.as_bytes(),
            subject_id.as_bytes(),
            target_realization_id.as_bytes(),
            distributed_context_id.as_bytes(),
            authority_profile_id.as_bytes(),
            &valid_from_unix_ms.to_le_bytes(),
            &valid_until_unix_ms.to_le_bytes(),
            &authority_basis_digest,
        ],
    )
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(name: &str, seed: u8, epoch: u64) -> TransitionAuthorityProfileV1 {
        TransitionAuthorityProfileV1::new(name, [seed; 32], epoch).unwrap()
    }

    #[test]
    fn authority_profile_identity_binds_root_and_epoch() {
        let a = profile("owner-root", 1, 1);
        let b = profile("owner-root", 2, 1);
        let c = profile("owner-root", 1, 2);
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
    }

    #[test]
    fn authority_policy_canonicalizes_allowed_roots() {
        let a = profile("a", 1, 1);
        let b = profile("b", 2, 1);
        let policy = TransitionAuthorityPolicyV1::new(1, vec![b.id(), a.id(), a.id()], 60_000)
            .unwrap();
        assert_eq!(policy.allowed_profile_ids().len(), 2);
        policy.validate_profile(&a).unwrap();
        policy.validate_profile(&b).unwrap();
    }

    #[test]
    fn same_name_substituted_authority_root_is_not_allowed() {
        let trusted = profile("owner", 1, 1);
        let substituted = profile("owner", 2, 1);
        let policy = TransitionAuthorityPolicyV1::new(1, vec![trusted.id()], 60_000).unwrap();
        assert_eq!(
            policy.validate_profile(&substituted).unwrap_err(),
            TransitionAuthorityError::AuthorityProfileNotAllowed
        );
    }

    #[test]
    fn invalid_or_overlong_grant_window_fails_closed() {
        assert_eq!(
            validate_grant_material(10, 10, [1; 32]).unwrap_err(),
            TransitionAuthorityError::InvalidGrantWindow
        );
        let trusted = profile("owner", 1, 1);
        let policy = TransitionAuthorityPolicyV1::new(1, vec![trusted.id()], 1_000).unwrap();
        let bound = policy.validate_profile(&trusted).unwrap();
        assert_eq!(bound.max_grant_lifetime_ms(), 1_000);
    }
}
