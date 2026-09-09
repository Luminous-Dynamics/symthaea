// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated clock/monotonic epoch binding for destructive continuity commits.
//!
//! Wall-clock values are useful diagnostics but must not become authority merely
//! because a caller supplied a `u64`. This layer requires an independently rooted
//! clock/epoch attestation for the exact commit-eligible transition.
//!
//! The design is generalized from Symthaea's existing trusted identity/time work:
//! exact hardware/clock root, boot counter, monotonic counter, multiple source
//! failure domains, bounded uncertainty, and explicit rollback detection.
//!
//! Core theorem:
//!
//! `CommitEligibleTransition != AuthenticatedClockClaim != TrustedCommitEligibility != ExecutionCapability`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::commit_eligibility::{CommitEligibleTransitionId, CommitEligibleTransitionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const TRUSTED_COMMIT_CLOCK_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-trusted-commit-clock-profile-v1";
pub const TRUSTED_COMMIT_EPOCH_POLICY_SCHEMA_V1: &str =
    "symthaea-continuity-trusted-commit-epoch-policy-v1";
pub const TRUSTED_COMMIT_EPOCH_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-trusted-commit-epoch-claim-v1";
pub const TRUSTED_COMMIT_EPOCH_XENIA_PURPOSE: &str =
    "symthaea.continuity.trusted-commit-epoch.v1";

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.trusted-commit-clock-profile.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.trusted-commit-epoch-policy.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.trusted-commit-epoch-claim.v1\0";
const CLAIM_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.trusted-commit-epoch-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-trusted-commit-epoch.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-trusted-commit-epoch.v1\0";
const ELIGIBILITY_DOMAIN: &[u8] = b"symthaea.continuity.trusted-commit-eligibility.v1\0";
const MAX_TEXT_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrustedCommitClockProfileId([u8; 32]);
impl TrustedCommitClockProfileId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrustedCommitEpochPolicyId([u8; 32]);
impl TrustedCommitEpochPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrustedCommitEpochClaimId([u8; 32]);
impl TrustedCommitEpochClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedTrustedCommitEpochId([u8; 32]);
impl AuthenticatedTrustedCommitEpochId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedTrustedCommitEpochId([u8; 32]);
impl QualifiedTrustedCommitEpochId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrustedCommitEligibilityId([u8; 32]);
impl TrustedCommitEligibilityId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Provisioned trust root for a platform/clock adapter.
///
/// This is distinct from both verifier roots and transition-authority roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedCommitClockProfileV1 {
    schema_version: String,
    profile_name: String,
    root_digest: [u8; 32],
    root_epoch: u64,
    profile_id: TrustedCommitClockProfileId,
}

impl TrustedCommitClockProfileV1 {
    pub fn new(
        profile_name: impl Into<String>,
        root_digest: [u8; 32],
        root_epoch: u64,
    ) -> Result<Self, TrustedCommitEpochError> {
        let profile_name = checked_text("clock profile name", profile_name.into())?;
        if root_digest == [0; 32] { return Err(TrustedCommitEpochError::ZeroClockRootDigest); }
        if root_epoch == 0 { return Err(TrustedCommitEpochError::ZeroClockRootEpoch); }
        let profile_id = TrustedCommitClockProfileId(hash_profile(&profile_name, root_digest, root_epoch));
        Ok(Self {
            schema_version: TRUSTED_COMMIT_CLOCK_PROFILE_SCHEMA_V1.to_owned(),
            profile_name,
            root_digest,
            root_epoch,
            profile_id,
        })
    }

    pub fn validate(&self) -> Result<(), TrustedCommitEpochError> {
        if self.schema_version != TRUSTED_COMMIT_CLOCK_PROFILE_SCHEMA_V1 {
            return Err(TrustedCommitEpochError::UnsupportedClockProfileSchema(self.schema_version.clone()));
        }
        let canonical = checked_text("clock profile name", self.profile_name.clone())?;
        if canonical != self.profile_name { return Err(TrustedCommitEpochError::NonCanonicalClockProfileName); }
        if self.root_digest == [0; 32] { return Err(TrustedCommitEpochError::ZeroClockRootDigest); }
        if self.root_epoch == 0 { return Err(TrustedCommitEpochError::ZeroClockRootEpoch); }
        let expected = TrustedCommitClockProfileId(hash_profile(&self.profile_name, self.root_digest, self.root_epoch));
        if expected != self.profile_id { return Err(TrustedCommitEpochError::ClockProfileIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> TrustedCommitClockProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
    pub fn profile_name(&self) -> &str { &self.profile_name }
}

/// Exact quality floor for commit-time epoch evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedCommitEpochPolicyV1 {
    schema_version: String,
    policy_generation: u64,
    clock_profile_id: TrustedCommitClockProfileId,
    minimum_time_sources: u16,
    minimum_failure_domains: u16,
    maximum_uncertainty_ms: u64,
    policy_id: TrustedCommitEpochPolicyId,
}

impl TrustedCommitEpochPolicyV1 {
    pub fn new(
        policy_generation: u64,
        profile: &TrustedCommitClockProfileV1,
        minimum_time_sources: u16,
        minimum_failure_domains: u16,
        maximum_uncertainty_ms: u64,
    ) -> Result<Self, TrustedCommitEpochError> {
        profile.validate()?;
        validate_policy_material(
            policy_generation,
            minimum_time_sources,
            minimum_failure_domains,
            maximum_uncertainty_ms,
        )?;
        if minimum_failure_domains > minimum_time_sources {
            return Err(TrustedCommitEpochError::FailureDomainsExceedSources);
        }
        let clock_profile_id = profile.id();
        let policy_id = TrustedCommitEpochPolicyId(hash_policy(
            policy_generation,
            clock_profile_id,
            minimum_time_sources,
            minimum_failure_domains,
            maximum_uncertainty_ms,
        ));
        Ok(Self {
            schema_version: TRUSTED_COMMIT_EPOCH_POLICY_SCHEMA_V1.to_owned(),
            policy_generation,
            clock_profile_id,
            minimum_time_sources,
            minimum_failure_domains,
            maximum_uncertainty_ms,
            policy_id,
        })
    }

    pub fn validate(&self) -> Result<(), TrustedCommitEpochError> {
        if self.schema_version != TRUSTED_COMMIT_EPOCH_POLICY_SCHEMA_V1 {
            return Err(TrustedCommitEpochError::UnsupportedEpochPolicySchema(self.schema_version.clone()));
        }
        validate_policy_material(
            self.policy_generation,
            self.minimum_time_sources,
            self.minimum_failure_domains,
            self.maximum_uncertainty_ms,
        )?;
        if self.minimum_failure_domains > self.minimum_time_sources {
            return Err(TrustedCommitEpochError::FailureDomainsExceedSources);
        }
        let expected = TrustedCommitEpochPolicyId(hash_policy(
            self.policy_generation,
            self.clock_profile_id,
            self.minimum_time_sources,
            self.minimum_failure_domains,
            self.maximum_uncertainty_ms,
        ));
        if expected != self.policy_id { return Err(TrustedCommitEpochError::EpochPolicyIdentityMismatch); }
        Ok(())
    }

    pub fn validate_against_profile(
        &self,
        profile: &TrustedCommitClockProfileV1,
    ) -> Result<ValidatedTrustedCommitEpochPolicyV1, TrustedCommitEpochError> {
        self.validate()?;
        profile.validate()?;
        if profile.id() != self.clock_profile_id { return Err(TrustedCommitEpochError::ClockProfileMismatch); }
        Ok(ValidatedTrustedCommitEpochPolicyV1 { inner: self.clone(), profile: profile.clone() })
    }

    pub fn id(&self) -> TrustedCommitEpochPolicyId { self.policy_id }
    pub fn policy_generation(&self) -> u64 { self.policy_generation }
    pub fn clock_profile_id(&self) -> TrustedCommitClockProfileId { self.clock_profile_id }
    pub fn minimum_time_sources(&self) -> u16 { self.minimum_time_sources }
    pub fn minimum_failure_domains(&self) -> u16 { self.minimum_failure_domains }
    pub fn maximum_uncertainty_ms(&self) -> u64 { self.maximum_uncertainty_ms }
}

#[derive(Debug, Clone)]
pub struct ValidatedTrustedCommitEpochPolicyV1 {
    inner: TrustedCommitEpochPolicyV1,
    profile: TrustedCommitClockProfileV1,
}

impl ValidatedTrustedCommitEpochPolicyV1 {
    pub fn id(&self) -> TrustedCommitEpochPolicyId { self.inner.id() }
    pub fn policy_generation(&self) -> u64 { self.inner.policy_generation() }
    pub fn clock_profile_id(&self) -> TrustedCommitClockProfileId { self.inner.clock_profile_id() }
    pub fn clock_root_epoch(&self) -> u64 { self.profile.root_epoch() }
    pub fn minimum_time_sources(&self) -> u16 { self.inner.minimum_time_sources() }
    pub fn minimum_failure_domains(&self) -> u16 { self.inner.minimum_failure_domains() }
    pub fn maximum_uncertainty_ms(&self) -> u64 { self.inner.maximum_uncertainty_ms() }
}

/// Untrusted transport claim emitted by a clock/platform adapter.
///
/// The source/failure-domain counts are claims by the authenticated adapter and are
/// backed by `source_set_digest` + `raw_evidence_digest`; callers cannot turn this
/// value into trusted eligibility without exact profile authentication and policy checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedCommitEpochClaimV1 {
    schema_version: String,
    commit_eligibility_id: CommitEligibleTransitionId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    clock_profile_id: TrustedCommitClockProfileId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    accepted_unix_ms: u64,
    uncertainty_ms: u64,
    source_count: u16,
    failure_domain_count: u16,
    source_set_digest: [u8; 32],
    raw_evidence_digest: [u8; 32],
    claim_id: TrustedCommitEpochClaimId,
}

impl TrustedCommitEpochClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eligibility: &CommitEligibleTransitionV1,
        clock_profile_id: TrustedCommitClockProfileId,
        boot_instance_digest: [u8; 32],
        boot_counter: u64,
        monotonic_counter: u64,
        accepted_unix_ms: u64,
        uncertainty_ms: u64,
        source_count: u16,
        failure_domain_count: u16,
        source_set_digest: [u8; 32],
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, TrustedCommitEpochError> {
        validate_claim_material(
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            accepted_unix_ms,
            source_count,
            failure_domain_count,
            source_set_digest,
            raw_evidence_digest,
        )?;
        let claim_id = TrustedCommitEpochClaimId(hash_claim(
            eligibility.id(),
            eligibility.subject_id(),
            eligibility.target_realization_id(),
            eligibility.distributed_context_id(),
            clock_profile_id,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            accepted_unix_ms,
            uncertainty_ms,
            source_count,
            failure_domain_count,
            source_set_digest,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: TRUSTED_COMMIT_EPOCH_CLAIM_SCHEMA_V1.to_owned(),
            commit_eligibility_id: eligibility.id(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            clock_profile_id,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            accepted_unix_ms,
            uncertainty_ms,
            source_count,
            failure_domain_count,
            source_set_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), TrustedCommitEpochError> {
        if self.schema_version != TRUSTED_COMMIT_EPOCH_CLAIM_SCHEMA_V1 {
            return Err(TrustedCommitEpochError::UnsupportedEpochClaimSchema(self.schema_version.clone()));
        }
        validate_claim_material(
            self.boot_instance_digest,
            self.boot_counter,
            self.monotonic_counter,
            self.accepted_unix_ms,
            self.source_count,
            self.failure_domain_count,
            self.source_set_digest,
            self.raw_evidence_digest,
        )?;
        let expected = TrustedCommitEpochClaimId(hash_claim(
            self.commit_eligibility_id,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.clock_profile_id,
            self.boot_instance_digest,
            self.boot_counter,
            self.monotonic_counter,
            self.accepted_unix_ms,
            self.uncertainty_ms,
            self.source_count,
            self.failure_domain_count,
            self.source_set_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id { return Err(TrustedCommitEpochError::EpochClaimIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> TrustedCommitEpochClaimId { self.claim_id }
}

pub fn canonical_trusted_commit_epoch_claim_bytes(
    claim: &TrustedCommitEpochClaimV1,
) -> Result<Vec<u8>, TrustedCommitEpochError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(CLAIM_WIRE_DOMAIN);
    out.extend_from_slice(claim.commit_eligibility_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.target_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(claim.clock_profile_id.as_bytes());
    out.extend_from_slice(&claim.boot_instance_digest);
    out.extend_from_slice(&claim.boot_counter.to_le_bytes());
    out.extend_from_slice(&claim.monotonic_counter.to_le_bytes());
    out.extend_from_slice(&claim.accepted_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.uncertainty_ms.to_le_bytes());
    out.extend_from_slice(&claim.source_count.to_le_bytes());
    out.extend_from_slice(&claim.failure_domain_count.to_le_bytes());
    out.extend_from_slice(&claim.source_set_digest);
    out.extend_from_slice(&claim.raw_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_trusted_commit_epoch_claim_digest(
    claim: &TrustedCommitEpochClaimV1,
) -> Result<[u8; 32], TrustedCommitEpochError> {
    Ok(*blake3::hash(&canonical_trusted_commit_epoch_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedTrustedCommitEpochV1 {
    claim: TrustedCommitEpochClaimV1,
    policy: ValidatedTrustedCommitEpochPolicyV1,
}

pub(crate) fn policy_check_trusted_commit_epoch_claim(
    eligibility: &CommitEligibleTransitionV1,
    policy: &ValidatedTrustedCommitEpochPolicyV1,
    claim: TrustedCommitEpochClaimV1,
) -> Result<PolicyCheckedTrustedCommitEpochV1, TrustedCommitEpochError> {
    claim.validate()?;
    if claim.clock_profile_id != policy.clock_profile_id() {
        return Err(TrustedCommitEpochError::ClockProfileMismatch);
    }
    if claim.commit_eligibility_id != eligibility.id()
        || claim.subject_id != eligibility.subject_id()
        || claim.target_realization_id != eligibility.target_realization_id()
        || claim.distributed_context_id != eligibility.distributed_context_id()
    {
        return Err(TrustedCommitEpochError::EligibilityContextMismatch);
    }
    if claim.accepted_unix_ms != eligibility.commit_time_unix_ms() {
        return Err(TrustedCommitEpochError::AcceptedTimeMismatch {
            accepted_unix_ms: claim.accepted_unix_ms,
            commit_time_unix_ms: eligibility.commit_time_unix_ms(),
        });
    }
    if claim.uncertainty_ms > policy.maximum_uncertainty_ms() {
        return Err(TrustedCommitEpochError::ExcessiveUncertainty {
            observed_ms: claim.uncertainty_ms,
            allowed_ms: policy.maximum_uncertainty_ms(),
        });
    }
    if claim.source_count < policy.minimum_time_sources() {
        return Err(TrustedCommitEpochError::InsufficientTimeSources {
            observed: claim.source_count,
            required: policy.minimum_time_sources(),
        });
    }
    if claim.failure_domain_count < policy.minimum_failure_domains() {
        return Err(TrustedCommitEpochError::InsufficientFailureDomains {
            observed: claim.failure_domain_count,
            required: policy.minimum_failure_domains(),
        });
    }
    Ok(PolicyCheckedTrustedCommitEpochV1 { claim, policy: policy.clone() })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedTrustedCommitEpochV1 {
    checked: PolicyCheckedTrustedCommitEpochV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedTrustedCommitEpochId,
}

impl AuthenticatedTrustedCommitEpochV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedTrustedCommitEpochV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, TrustedCommitEpochError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(TrustedCommitEpochError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedTrustedCommitEpochId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                checked.claim.id().as_bytes(),
                checked.policy.id().as_bytes(),
                checked.policy.clock_profile_id().as_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self { checked, authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde exact epoch accepted under one exact clock-root policy.
#[derive(Debug, Clone)]
pub struct QualifiedTrustedCommitEpochV1 {
    epoch_id: QualifiedTrustedCommitEpochId,
    commit_eligibility_id: CommitEligibleTransitionId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    policy_id: TrustedCommitEpochPolicyId,
    policy_generation: u64,
    clock_profile_id: TrustedCommitClockProfileId,
    clock_root_epoch: u64,
    authenticated_evidence_id: AuthenticatedTrustedCommitEpochId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    accepted_unix_ms: u64,
    uncertainty_ms: u64,
    source_count: u16,
    failure_domain_count: u16,
    source_set_digest: [u8; 32],
}

impl QualifiedTrustedCommitEpochV1 {
    pub fn id(&self) -> QualifiedTrustedCommitEpochId { self.epoch_id }
    pub fn commit_eligibility_id(&self) -> CommitEligibleTransitionId { self.commit_eligibility_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn clock_profile_id(&self) -> TrustedCommitClockProfileId { self.clock_profile_id }
    pub fn clock_root_epoch(&self) -> u64 { self.clock_root_epoch }
    pub fn boot_instance_digest(&self) -> [u8; 32] { self.boot_instance_digest }
    pub fn boot_counter(&self) -> u64 { self.boot_counter }
    pub fn monotonic_counter(&self) -> u64 { self.monotonic_counter }
    pub fn accepted_unix_ms(&self) -> u64 { self.accepted_unix_ms }
    pub fn uncertainty_ms(&self) -> u64 { self.uncertainty_ms }
    pub fn source_count(&self) -> u16 { self.source_count }
    pub fn failure_domain_count(&self) -> u16 { self.failure_domain_count }
    pub fn source_set_digest(&self) -> [u8; 32] { self.source_set_digest }
}

pub(crate) fn qualify_trusted_commit_epoch(
    eligibility: &CommitEligibleTransitionV1,
    evidence: &AuthenticatedTrustedCommitEpochV1,
) -> Result<QualifiedTrustedCommitEpochV1, TrustedCommitEpochError> {
    let claim = &evidence.checked.claim;
    let policy = &evidence.checked.policy;
    if claim.commit_eligibility_id != eligibility.id()
        || claim.subject_id != eligibility.subject_id()
        || claim.target_realization_id != eligibility.target_realization_id()
        || claim.distributed_context_id != eligibility.distributed_context_id()
        || claim.accepted_unix_ms != eligibility.commit_time_unix_ms()
    {
        return Err(TrustedCommitEpochError::EligibilityContextMismatch);
    }
    let epoch_id = QualifiedTrustedCommitEpochId(domain_hash_parts(
        QUALIFIED_DOMAIN,
        &[
            eligibility.id().as_bytes(),
            claim.id().as_bytes(),
            policy.id().as_bytes(),
            policy.clock_profile_id().as_bytes(),
            &policy.clock_root_epoch().to_le_bytes(),
            evidence.evidence_id.as_bytes(),
            &claim.boot_instance_digest,
            &claim.boot_counter.to_le_bytes(),
            &claim.monotonic_counter.to_le_bytes(),
            &claim.accepted_unix_ms.to_le_bytes(),
            &claim.uncertainty_ms.to_le_bytes(),
            &claim.source_set_digest,
        ],
    ));
    Ok(QualifiedTrustedCommitEpochV1 {
        epoch_id,
        commit_eligibility_id: eligibility.id(),
        subject_id: eligibility.subject_id(),
        target_realization_id: eligibility.target_realization_id(),
        distributed_context_id: eligibility.distributed_context_id(),
        policy_id: policy.id(),
        policy_generation: policy.policy_generation(),
        clock_profile_id: policy.clock_profile_id(),
        clock_root_epoch: policy.clock_root_epoch(),
        authenticated_evidence_id: evidence.evidence_id,
        boot_instance_digest: claim.boot_instance_digest,
        boot_counter: claim.boot_counter,
        monotonic_counter: claim.monotonic_counter,
        accepted_unix_ms: claim.accepted_unix_ms,
        uncertainty_ms: claim.uncertainty_ms,
        source_count: claim.source_count,
        failure_domain_count: claim.failure_domain_count,
        source_set_digest: claim.source_set_digest,
    })
}

/// Enforce rollback-resistant progression when an executor has a previous qualified
/// epoch anchor available. Persisting that anchor in rollback-resistant storage is
/// an adapter responsibility; the pure continuity kernel does not pretend ordinary
/// serialized storage is tamper-proof.
pub fn validate_trusted_commit_epoch_progression(
    previous: &QualifiedTrustedCommitEpochV1,
    next: &QualifiedTrustedCommitEpochV1,
) -> Result<(), TrustedCommitEpochError> {
    if previous.subject_id != next.subject_id || previous.clock_profile_id != next.clock_profile_id {
        return Err(TrustedCommitEpochError::EpochLineageMismatch);
    }
    if next.boot_counter < previous.boot_counter {
        return Err(TrustedCommitEpochError::BootCounterRollback {
            previous: previous.boot_counter,
            observed: next.boot_counter,
        });
    }
    if next.boot_counter == previous.boot_counter {
        if next.boot_instance_digest != previous.boot_instance_digest {
            return Err(TrustedCommitEpochError::BootInstanceDriftWithoutCounterAdvance);
        }
        if next.monotonic_counter <= previous.monotonic_counter {
            return Err(TrustedCommitEpochError::MonotonicCounterRollback {
                previous: previous.monotonic_counter,
                observed: next.monotonic_counter,
            });
        }
    }
    Ok(())
}

/// Non-Serde commit eligibility strengthened by an authenticated trusted epoch.
#[derive(Debug, Clone)]
pub struct TrustedCommitEligibilityV1 {
    trusted_eligibility_id: TrustedCommitEligibilityId,
    eligibility_id: CommitEligibleTransitionId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    commit_time_unix_ms: u64,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
}

impl TrustedCommitEligibilityV1 {
    pub fn id(&self) -> TrustedCommitEligibilityId { self.trusted_eligibility_id }
    pub fn eligibility_id(&self) -> CommitEligibleTransitionId { self.eligibility_id }
    pub fn trusted_epoch_id(&self) -> QualifiedTrustedCommitEpochId { self.trusted_epoch_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn commit_time_unix_ms(&self) -> u64 { self.commit_time_unix_ms }
    pub fn boot_instance_digest(&self) -> [u8; 32] { self.boot_instance_digest }
    pub fn boot_counter(&self) -> u64 { self.boot_counter }
    pub fn monotonic_counter(&self) -> u64 { self.monotonic_counter }
}

pub(crate) fn bind_trusted_commit_eligibility(
    eligibility: &CommitEligibleTransitionV1,
    epoch: &QualifiedTrustedCommitEpochV1,
) -> Result<TrustedCommitEligibilityV1, TrustedCommitEpochError> {
    if epoch.commit_eligibility_id != eligibility.id()
        || epoch.subject_id != eligibility.subject_id()
        || epoch.target_realization_id != eligibility.target_realization_id()
        || epoch.distributed_context_id != eligibility.distributed_context_id()
        || epoch.accepted_unix_ms != eligibility.commit_time_unix_ms()
    {
        return Err(TrustedCommitEpochError::EligibilityContextMismatch);
    }
    let trusted_eligibility_id = TrustedCommitEligibilityId(domain_hash_parts(
        ELIGIBILITY_DOMAIN,
        &[
            eligibility.id().as_bytes(),
            epoch.id().as_bytes(),
            eligibility.subject_id().as_bytes(),
            eligibility.target_realization_id().as_bytes(),
            eligibility.distributed_context_id().as_bytes(),
            &eligibility.commit_time_unix_ms().to_le_bytes(),
            &epoch.boot_instance_digest,
            &epoch.boot_counter.to_le_bytes(),
            &epoch.monotonic_counter.to_le_bytes(),
        ],
    ));
    Ok(TrustedCommitEligibilityV1 {
        trusted_eligibility_id,
        eligibility_id: eligibility.id(),
        trusted_epoch_id: epoch.id(),
        subject_id: eligibility.subject_id(),
        target_realization_id: eligibility.target_realization_id(),
        distributed_context_id: eligibility.distributed_context_id(),
        commit_time_unix_ms: eligibility.commit_time_unix_ms(),
        boot_instance_digest: epoch.boot_instance_digest,
        boot_counter: epoch.boot_counter,
        monotonic_counter: epoch.monotonic_counter,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TrustedCommitEpochError {
    #[error("unsupported trusted commit clock profile schema: {0}")]
    UnsupportedClockProfileSchema(String),
    #[error("unsupported trusted commit epoch policy schema: {0}")]
    UnsupportedEpochPolicySchema(String),
    #[error("unsupported trusted commit epoch claim schema: {0}")]
    UnsupportedEpochClaimSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds the text bound")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("clock profile name is not canonical")]
    NonCanonicalClockProfileName,
    #[error("trusted commit clock root digest must be non-zero")]
    ZeroClockRootDigest,
    #[error("trusted commit clock root epoch must be non-zero")]
    ZeroClockRootEpoch,
    #[error("trusted commit clock profile identity mismatch")]
    ClockProfileIdentityMismatch,
    #[error("trusted commit epoch policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("trusted commit epoch requires at least one time source")]
    NoTimeSources,
    #[error("trusted commit epoch requires at least one time-source failure domain")]
    NoFailureDomains,
    #[error("trusted commit epoch maximum uncertainty must be non-zero")]
    ZeroMaximumUncertainty,
    #[error("minimum failure domains cannot exceed minimum time sources")]
    FailureDomainsExceedSources,
    #[error("trusted commit epoch policy identity mismatch")]
    EpochPolicyIdentityMismatch,
    #[error("trusted commit clock profile does not match the pinned policy")]
    ClockProfileMismatch,
    #[error("trusted commit epoch boot-instance digest must be non-zero")]
    ZeroBootInstanceDigest,
    #[error("trusted commit epoch boot counter must be non-zero")]
    ZeroBootCounter,
    #[error("trusted commit epoch monotonic counter must be non-zero")]
    ZeroMonotonicCounter,
    #[error("trusted commit epoch accepted time must be non-zero")]
    ZeroAcceptedTime,
    #[error("trusted commit epoch source count must be non-zero")]
    ZeroSourceCount,
    #[error("trusted commit epoch failure-domain count must be non-zero")]
    ZeroFailureDomainCount,
    #[error("trusted commit epoch source-set digest must be non-zero")]
    ZeroSourceSetDigest,
    #[error("trusted commit epoch raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("trusted commit epoch failure-domain count exceeds source count")]
    ObservedFailureDomainsExceedSources,
    #[error("trusted commit epoch claim identity mismatch")]
    EpochClaimIdentityMismatch,
    #[error("trusted commit epoch does not bind the exact commit eligibility/subject/target/context")]
    EligibilityContextMismatch,
    #[error("authenticated accepted time {accepted_unix_ms} does not equal exact commit time {commit_time_unix_ms}")]
    AcceptedTimeMismatch { accepted_unix_ms: u64, commit_time_unix_ms: u64 },
    #[error("trusted time uncertainty {observed_ms} ms exceeds policy maximum {allowed_ms} ms")]
    ExcessiveUncertainty { observed_ms: u64, allowed_ms: u64 },
    #[error("trusted time source count {observed} below policy minimum {required}")]
    InsufficientTimeSources { observed: u16, required: u16 },
    #[error("trusted time failure-domain count {observed} below policy minimum {required}")]
    InsufficientFailureDomains { observed: u16, required: u16 },
    #[error("trusted commit epoch authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("trusted commit epoch progression belongs to a different subject/clock lineage")]
    EpochLineageMismatch,
    #[error("trusted commit boot counter rolled back from {previous} to {observed}")]
    BootCounterRollback { previous: u64, observed: u64 },
    #[error("boot instance changed without advancing the boot counter")]
    BootInstanceDriftWithoutCounterAdvance,
    #[error("trusted commit monotonic counter did not advance: previous {previous}, observed {observed}")]
    MonotonicCounterRollback { previous: u64, observed: u64 },
}

fn validate_policy_material(
    generation: u64,
    sources: u16,
    domains: u16,
    max_uncertainty_ms: u64,
) -> Result<(), TrustedCommitEpochError> {
    if generation == 0 { return Err(TrustedCommitEpochError::ZeroPolicyGeneration); }
    if sources == 0 { return Err(TrustedCommitEpochError::NoTimeSources); }
    if domains == 0 { return Err(TrustedCommitEpochError::NoFailureDomains); }
    if max_uncertainty_ms == 0 { return Err(TrustedCommitEpochError::ZeroMaximumUncertainty); }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_claim_material(
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    accepted_unix_ms: u64,
    source_count: u16,
    failure_domain_count: u16,
    source_set_digest: [u8; 32],
    raw_evidence_digest: [u8; 32],
) -> Result<(), TrustedCommitEpochError> {
    if boot_instance_digest == [0; 32] { return Err(TrustedCommitEpochError::ZeroBootInstanceDigest); }
    if boot_counter == 0 { return Err(TrustedCommitEpochError::ZeroBootCounter); }
    if monotonic_counter == 0 { return Err(TrustedCommitEpochError::ZeroMonotonicCounter); }
    if accepted_unix_ms == 0 { return Err(TrustedCommitEpochError::ZeroAcceptedTime); }
    if source_count == 0 { return Err(TrustedCommitEpochError::ZeroSourceCount); }
    if failure_domain_count == 0 { return Err(TrustedCommitEpochError::ZeroFailureDomainCount); }
    if failure_domain_count > source_count { return Err(TrustedCommitEpochError::ObservedFailureDomainsExceedSources); }
    if source_set_digest == [0; 32] { return Err(TrustedCommitEpochError::ZeroSourceSetDigest); }
    if raw_evidence_digest == [0; 32] { return Err(TrustedCommitEpochError::ZeroRawEvidenceDigest); }
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, TrustedCommitEpochError> {
    let trimmed = value.trim();
    if trimmed.is_empty() { return Err(TrustedCommitEpochError::BlankText { field }); }
    if trimmed.len() > MAX_TEXT_BYTES { return Err(TrustedCommitEpochError::TextTooLong { field }); }
    if trimmed.chars().any(char::is_control) { return Err(TrustedCommitEpochError::ControlCharacters { field }); }
    Ok(trimmed.to_owned())
}

fn hash_profile(name: &str, root: [u8; 32], epoch: u64) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, name);
    bytes.extend_from_slice(&root);
    bytes.extend_from_slice(&epoch.to_le_bytes());
    domain_hash(PROFILE_DOMAIN, &bytes)
}

fn hash_policy(
    generation: u64,
    profile_id: TrustedCommitClockProfileId,
    sources: u16,
    domains: u16,
    max_uncertainty_ms: u64,
) -> [u8; 32] {
    domain_hash_parts(POLICY_DOMAIN, &[
        &generation.to_le_bytes(),
        profile_id.as_bytes(),
        &sources.to_le_bytes(),
        &domains.to_le_bytes(),
        &max_uncertainty_ms.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    eligibility_id: CommitEligibleTransitionId,
    subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId,
    context_id: DistributedStateContextId,
    profile_id: TrustedCommitClockProfileId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    accepted_unix_ms: u64,
    uncertainty_ms: u64,
    source_count: u16,
    failure_domain_count: u16,
    source_set_digest: [u8; 32],
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    domain_hash_parts(CLAIM_DOMAIN, &[
        eligibility_id.as_bytes(),
        subject_id.as_bytes(),
        target_id.as_bytes(),
        context_id.as_bytes(),
        profile_id.as_bytes(),
        &boot_instance_digest,
        &boot_counter.to_le_bytes(),
        &monotonic_counter.to_le_bytes(),
        &accepted_unix_ms.to_le_bytes(),
        &uncertainty_ms.to_le_bytes(),
        &source_count.to_le_bytes(),
        &failure_domain_count.to_le_bytes(),
        &source_set_digest,
        &raw_evidence_digest,
    ])
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
    for part in parts { hasher.update(part); }
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) { out.extend_from_slice(&(len as u64).to_le_bytes()); }
fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(seed: u8) -> TrustedCommitClockProfileV1 {
        TrustedCommitClockProfileV1::new(format!("clock-{seed}"), [seed; 32], seed as u64).unwrap()
    }

    #[test]
    fn clock_profile_root_or_epoch_changes_identity() {
        let a = TrustedCommitClockProfileV1::new("clock", [1; 32], 1).unwrap();
        let b = TrustedCommitClockProfileV1::new("clock", [2; 32], 1).unwrap();
        let c = TrustedCommitClockProfileV1::new("clock", [1; 32], 2).unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
    }

    #[test]
    fn epoch_policy_requires_independent_source_floor() {
        let p = profile(1);
        assert_eq!(
            TrustedCommitEpochPolicyV1::new(1, &p, 1, 2, 50).unwrap_err(),
            TrustedCommitEpochError::FailureDomainsExceedSources
        );
        TrustedCommitEpochPolicyV1::new(1, &p, 2, 2, 50)
            .unwrap()
            .validate_against_profile(&p)
            .unwrap();
    }

    #[test]
    fn claim_material_rejects_counter_or_source_ambiguity() {
        assert_eq!(
            validate_claim_material([1; 32], 1, 0, 1, 2, 2, [2; 32], [3; 32]).unwrap_err(),
            TrustedCommitEpochError::ZeroMonotonicCounter
        );
        assert_eq!(
            validate_claim_material([1; 32], 1, 1, 1, 1, 2, [2; 32], [3; 32]).unwrap_err(),
            TrustedCommitEpochError::ObservedFailureDomainsExceedSources
        );
    }
}
