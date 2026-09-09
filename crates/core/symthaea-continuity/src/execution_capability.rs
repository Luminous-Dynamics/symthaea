// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executor-owned session and one-use physical transition capability.
//!
//! This module is intentionally generic: it does not know Nix, NETCONF, Redfish,
//! gNOI, databases, hypervisors, or storage APIs. Backend adapters consume the
//! capability by value and return a separate evidence receipt.
//!
//! Core theorem:
//!
//! `TrustedCommitEligibility != ExecutionSession != OneUseExecutionCapability != ExecutionReceipt`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_state::DistributedStateContextId;
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::{
    QualifiedTrustedCommitEpochV1, TrustedCommitEligibilityId, TrustedCommitEligibilityV1,
    TrustedCommitEpochError, validate_trusted_commit_epoch_progression,
};
use crate::witness::TargetRealizationId;

pub const EXECUTION_BACKEND_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-execution-backend-profile-v1";

const BACKEND_DOMAIN: &[u8] = b"symthaea.continuity.execution-backend.v1\0";
const SESSION_DOMAIN: &[u8] = b"symthaea.continuity.execution-session.v1\0";
const CAPABILITY_DOMAIN: &[u8] = b"symthaea.continuity.one-use-execution-capability.v1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.continuity.execution-attempt-receipt.v1\0";
const MAX_TEXT_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionBackendId([u8; 32]);
impl ExecutionBackendId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionSessionId([u8; 32]);
impl ExecutionSessionId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OneUseExecutionCapabilityId([u8; 32]);
impl OneUseExecutionCapabilityId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionAttemptReceiptId([u8; 32]);
impl ExecutionAttemptReceiptId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Exact executor implementation identity. This identifies the adapter/backend; it
/// does not grant authority and is safe to serialize as descriptive configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionBackendProfileV1 {
    schema_version: String,
    backend_name: String,
    implementation_digest: [u8; 32],
    backend_generation: u64,
    backend_id: ExecutionBackendId,
}

impl ExecutionBackendProfileV1 {
    pub fn new(
        backend_name: impl Into<String>,
        implementation_digest: [u8; 32],
        backend_generation: u64,
    ) -> Result<Self, ExecutionCapabilityError> {
        let backend_name = checked_text("execution backend name", backend_name.into())?;
        if implementation_digest == [0; 32] {
            return Err(ExecutionCapabilityError::ZeroBackendImplementationDigest);
        }
        if backend_generation == 0 {
            return Err(ExecutionCapabilityError::ZeroBackendGeneration);
        }
        let backend_id = ExecutionBackendId(hash_backend(
            &backend_name,
            implementation_digest,
            backend_generation,
        ));
        Ok(Self {
            schema_version: EXECUTION_BACKEND_PROFILE_SCHEMA_V1.to_owned(),
            backend_name,
            implementation_digest,
            backend_generation,
            backend_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExecutionCapabilityError> {
        if self.schema_version != EXECUTION_BACKEND_PROFILE_SCHEMA_V1 {
            return Err(ExecutionCapabilityError::UnsupportedBackendSchema(self.schema_version.clone()));
        }
        let canonical = checked_text("execution backend name", self.backend_name.clone())?;
        if canonical != self.backend_name { return Err(ExecutionCapabilityError::NonCanonicalBackendName); }
        if self.implementation_digest == [0; 32] {
            return Err(ExecutionCapabilityError::ZeroBackendImplementationDigest);
        }
        if self.backend_generation == 0 {
            return Err(ExecutionCapabilityError::ZeroBackendGeneration);
        }
        let expected = ExecutionBackendId(hash_backend(
            &self.backend_name,
            self.implementation_digest,
            self.backend_generation,
        ));
        if expected != self.backend_id { return Err(ExecutionCapabilityError::BackendIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_name(&self) -> &str { &self.backend_name }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
}

/// Whether opening an executor session may bootstrap without a previous
/// rollback-resistant epoch anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionEpochAnchorModeV1 {
    RequirePrevious,
    AllowBootstrap,
}

/// Executor-owned, non-Serde, non-Clone transaction session.
///
/// The internal minted set prevents the same trusted eligibility from producing
/// multiple capabilities within one exact executor session.
#[derive(Debug)]
pub struct ExecutionSessionV1 {
    session_id: ExecutionSessionId,
    backend: ExecutionBackendProfileV1,
    session_generation: u64,
    session_nonce: [u8; 32],
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    trusted_epoch_id: crate::trusted_commit_epoch::QualifiedTrustedCommitEpochId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_from_previous: bool,
    minted_eligibilities: BTreeSet<TrustedCommitEligibilityId>,
}

impl ExecutionSessionV1 {
    /// Crate-owned opening boundary for a future platform/Spore coordinator.
    pub(crate) fn open(
        eligibility: &TrustedCommitEligibilityV1,
        current_epoch: &QualifiedTrustedCommitEpochV1,
        previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
        anchor_mode: ExecutionEpochAnchorModeV1,
        backend: &ExecutionBackendProfileV1,
        session_generation: u64,
        session_nonce: [u8; 32],
    ) -> Result<Self, ExecutionCapabilityError> {
        backend.validate()?;
        if session_generation == 0 { return Err(ExecutionCapabilityError::ZeroSessionGeneration); }
        if session_nonce == [0; 32] { return Err(ExecutionCapabilityError::ZeroSessionNonce); }
        require_epoch_matches_eligibility(eligibility, current_epoch)?;

        let anchored_from_previous = match previous_epoch {
            Some(previous) => {
                validate_trusted_commit_epoch_progression(previous, current_epoch)?;
                true
            }
            None => {
                if anchor_mode == ExecutionEpochAnchorModeV1::RequirePrevious {
                    return Err(ExecutionCapabilityError::MissingPreviousEpochAnchor);
                }
                false
            }
        };

        let session_id = ExecutionSessionId(hash_session(
            backend.id(),
            session_generation,
            session_nonce,
            eligibility,
            current_epoch,
            anchored_from_previous,
        ));
        Ok(Self {
            session_id,
            backend: backend.clone(),
            session_generation,
            session_nonce,
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            trusted_epoch_id: current_epoch.id(),
            boot_instance_digest: current_epoch.boot_instance_digest(),
            boot_counter: current_epoch.boot_counter(),
            monotonic_counter: current_epoch.monotonic_counter(),
            anchored_from_previous,
            minted_eligibilities: BTreeSet::new(),
        })
    }

    pub fn id(&self) -> ExecutionSessionId { self.session_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend.id() }
    pub fn session_generation(&self) -> u64 { self.session_generation }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn anchored_from_previous(&self) -> bool { self.anchored_from_previous }
}

/// Opaque, non-Serde, non-Clone physical mutation capability.
///
/// Backend adapters should accept this value **by value**. There is no public
/// constructor and no method that recreates it from its serializable ID.
#[derive(Debug)]
pub struct OneUseExecutionCapabilityV1 {
    capability_id: OneUseExecutionCapabilityId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
    backend_id: ExecutionBackendId,
    session_id: ExecutionSessionId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
}

impl OneUseExecutionCapabilityV1 {
    pub fn id(&self) -> OneUseExecutionCapabilityId { self.capability_id }
    pub fn trusted_eligibility_id(&self) -> TrustedCommitEligibilityId { self.trusted_eligibility_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn session_id(&self) -> ExecutionSessionId { self.session_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }

    /// Check that a backend is still operating inside the exact session for which
    /// this capability was minted. The eventual adapter still consumes `self` by value.
    pub fn validate_for_session(&self, session: &ExecutionSessionV1) -> Result<(), ExecutionCapabilityError> {
        if self.session_id != session.id()
            || self.backend_id != session.backend_id()
            || self.subject_id != session.subject_id()
            || self.target_realization_id != session.target_realization_id()
            || self.distributed_context_id != session.distributed_context_id()
            || self.boot_instance_digest != session.boot_instance_digest
            || self.boot_counter != session.boot_counter
            || self.monotonic_counter != session.monotonic_counter
        {
            return Err(ExecutionCapabilityError::SessionMismatch);
        }
        Ok(())
    }

    /// Consume the capability and create a non-authoritative attempt receipt.
    ///
    /// The receipt records what the backend reports; it is not independently
    /// authenticated proof of physical success. A later adapter may authenticate
    /// the backend evidence separately.
    pub fn consume_for_session(
        self,
        session: &ExecutionSessionV1,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, ExecutionCapabilityError> {
        self.validate_for_session(session)?;
        if backend_evidence_digest == [0; 32] { return Err(ExecutionCapabilityError::ZeroBackendEvidenceDigest); }
        if result_digest == [0; 32] { return Err(ExecutionCapabilityError::ZeroResultDigest); }
        let receipt_id = ExecutionAttemptReceiptId(hash_receipt(
            self.capability_id,
            self.trusted_eligibility_id,
            self.backend_id,
            self.session_id,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            outcome,
            backend_evidence_digest,
            result_digest,
        ));
        Ok(ExecutionAttemptReceiptV1 {
            receipt_id,
            capability_id: self.capability_id,
            trusted_eligibility_id: self.trusted_eligibility_id,
            backend_id: self.backend_id,
            session_id: self.session_id,
            subject_id: self.subject_id,
            target_realization_id: self.target_realization_id,
            distributed_context_id: self.distributed_context_id,
            outcome,
            backend_evidence_digest,
            result_digest,
        })
    }
}

/// Mint exactly once per trusted eligibility within an executor-owned session.
pub(crate) fn mint_one_use_execution_capability(
    session: &mut ExecutionSessionV1,
    eligibility: &TrustedCommitEligibilityV1,
) -> Result<OneUseExecutionCapabilityV1, ExecutionCapabilityError> {
    if eligibility.subject_id() != session.subject_id
        || eligibility.target_realization_id() != session.target_realization_id
        || eligibility.distributed_context_id() != session.distributed_context_id
        || eligibility.trusted_epoch_id != session.trusted_epoch_id
        || eligibility.boot_instance_digest() != session.boot_instance_digest
        || eligibility.boot_counter() != session.boot_counter
        || eligibility.monotonic_counter() != session.monotonic_counter
    {
        return Err(ExecutionCapabilityError::EligibilitySessionMismatch);
    }
    if !session.minted_eligibilities.insert(eligibility.id()) {
        return Err(ExecutionCapabilityError::EligibilityAlreadyMintedInSession);
    }
    let capability_id = OneUseExecutionCapabilityId(hash_capability(
        eligibility,
        session,
    ));
    Ok(OneUseExecutionCapabilityV1 {
        capability_id,
        trusted_eligibility_id: eligibility.id(),
        backend_id: session.backend_id(),
        session_id: session.id(),
        subject_id: eligibility.subject_id(),
        target_realization_id: eligibility.target_realization_id(),
        distributed_context_id: eligibility.distributed_context_id(),
        boot_instance_digest: eligibility.boot_instance_digest(),
        boot_counter: eligibility.boot_counter(),
        monotonic_counter: eligibility.monotonic_counter(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionAttemptOutcomeV1 {
    Succeeded,
    Failed,
    RolledBack,
    Indeterminate,
}
impl ExecutionAttemptOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Succeeded => 1,
            Self::Failed => 2,
            Self::RolledBack => 3,
            Self::Indeterminate => 4,
        }
    }
}

/// Serializable audit material only. It never reconstructs a capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionAttemptReceiptV1 {
    receipt_id: ExecutionAttemptReceiptId,
    capability_id: OneUseExecutionCapabilityId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
    backend_id: ExecutionBackendId,
    session_id: ExecutionSessionId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    outcome: ExecutionAttemptOutcomeV1,
    backend_evidence_digest: [u8; 32],
    result_digest: [u8; 32],
}

impl ExecutionAttemptReceiptV1 {
    pub fn id(&self) -> ExecutionAttemptReceiptId { self.receipt_id }
    pub fn capability_id(&self) -> OneUseExecutionCapabilityId { self.capability_id }
    pub fn outcome(&self) -> ExecutionAttemptOutcomeV1 { self.outcome }
    pub fn backend_evidence_digest(&self) -> [u8; 32] { self.backend_evidence_digest }
    pub fn result_digest(&self) -> [u8; 32] { self.result_digest }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionCapabilityError {
    #[error(transparent)]
    TrustedEpoch(#[from] TrustedCommitEpochError),
    #[error("unsupported execution backend schema: {0}")]
    UnsupportedBackendSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds the text bound")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("execution backend name is not canonical")]
    NonCanonicalBackendName,
    #[error("execution backend implementation digest must be non-zero")]
    ZeroBackendImplementationDigest,
    #[error("execution backend generation must be non-zero")]
    ZeroBackendGeneration,
    #[error("execution backend identity mismatch")]
    BackendIdentityMismatch,
    #[error("execution session generation must be non-zero")]
    ZeroSessionGeneration,
    #[error("execution session nonce must be non-zero")]
    ZeroSessionNonce,
    #[error("execution session requires a previous trusted epoch anchor by policy")]
    MissingPreviousEpochAnchor,
    #[error("trusted epoch does not bind the exact trusted commit eligibility")]
    TrustedEpochEligibilityMismatch,
    #[error("trusted eligibility does not bind the exact executor session")]
    EligibilitySessionMismatch,
    #[error("trusted eligibility already minted an execution capability in this session")]
    EligibilityAlreadyMintedInSession,
    #[error("execution capability does not belong to the current executor session/backend")]
    SessionMismatch,
    #[error("execution backend evidence digest must be non-zero")]
    ZeroBackendEvidenceDigest,
    #[error("execution result digest must be non-zero")]
    ZeroResultDigest,
}

fn require_epoch_matches_eligibility(
    eligibility: &TrustedCommitEligibilityV1,
    epoch: &QualifiedTrustedCommitEpochV1,
) -> Result<(), ExecutionCapabilityError> {
    if eligibility.trusted_epoch_id() != epoch.id()
        || eligibility.subject_id() != epoch.subject_id()
        || eligibility.target_realization_id() != epoch.target_realization_id()
        || eligibility.distributed_context_id() != epoch.distributed_context_id()
        || eligibility.commit_time_unix_ms() != epoch.accepted_unix_ms()
        || eligibility.boot_instance_digest() != epoch.boot_instance_digest()
        || eligibility.boot_counter() != epoch.boot_counter()
        || eligibility.monotonic_counter() != epoch.monotonic_counter()
    {
        return Err(ExecutionCapabilityError::TrustedEpochEligibilityMismatch);
    }
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, ExecutionCapabilityError> {
    let trimmed = value.trim();
    if trimmed.is_empty() { return Err(ExecutionCapabilityError::BlankText { field }); }
    if trimmed.len() > MAX_TEXT_BYTES { return Err(ExecutionCapabilityError::TextTooLong { field }); }
    if trimmed.chars().any(char::is_control) { return Err(ExecutionCapabilityError::ControlCharacters { field }); }
    Ok(trimmed.to_owned())
}

fn hash_backend(name: &str, implementation_digest: [u8; 32], generation: u64) -> [u8; 32] {
    domain_hash_parts(BACKEND_DOMAIN, &[
        name.as_bytes(),
        &implementation_digest,
        &generation.to_le_bytes(),
    ])
}

fn hash_session(
    backend_id: ExecutionBackendId,
    generation: u64,
    nonce: [u8; 32],
    eligibility: &TrustedCommitEligibilityV1,
    epoch: &QualifiedTrustedCommitEpochV1,
    anchored: bool,
) -> [u8; 32] {
    let anchored_byte = [u8::from(anchored)];
    domain_hash_parts(SESSION_DOMAIN, &[
        backend_id.as_bytes(),
        &generation.to_le_bytes(),
        &nonce,
        eligibility.id().as_bytes(),
        eligibility.subject_id().as_bytes(),
        eligibility.target_realization_id().as_bytes(),
        eligibility.distributed_context_id().as_bytes(),
        epoch.id().as_bytes(),
        &epoch.boot_instance_digest(),
        &epoch.boot_counter().to_le_bytes(),
        &epoch.monotonic_counter().to_le_bytes(),
        &anchored_byte,
    ])
}

fn hash_capability(
    eligibility: &TrustedCommitEligibilityV1,
    session: &ExecutionSessionV1,
) -> [u8; 32] {
    domain_hash_parts(CAPABILITY_DOMAIN, &[
        eligibility.id().as_bytes(),
        session.id().as_bytes(),
        session.backend_id().as_bytes(),
        eligibility.subject_id().as_bytes(),
        eligibility.target_realization_id().as_bytes(),
        eligibility.distributed_context_id().as_bytes(),
        &session.session_nonce,
        &session.session_generation.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_receipt(
    capability_id: OneUseExecutionCapabilityId,
    eligibility_id: TrustedCommitEligibilityId,
    backend_id: ExecutionBackendId,
    session_id: ExecutionSessionId,
    subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId,
    context_id: DistributedStateContextId,
    outcome: ExecutionAttemptOutcomeV1,
    backend_evidence_digest: [u8; 32],
    result_digest: [u8; 32],
) -> [u8; 32] {
    let outcome_tag = [outcome.tag()];
    domain_hash_parts(RECEIPT_DOMAIN, &[
        capability_id.as_bytes(),
        eligibility_id.as_bytes(),
        backend_id.as_bytes(),
        session_id.as_bytes(),
        subject_id.as_bytes(),
        target_id.as_bytes(),
        context_id.as_bytes(),
        &outcome_tag,
        &backend_evidence_digest,
        &result_digest,
    ])
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts { hasher.update(part); }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_identity_changes_with_implementation_or_generation() {
        let a = ExecutionBackendProfileV1::new("spore", [1; 32], 1).unwrap();
        let b = ExecutionBackendProfileV1::new("spore", [2; 32], 1).unwrap();
        let c = ExecutionBackendProfileV1::new("spore", [1; 32], 2).unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
    }

    #[test]
    fn bootstrap_mode_is_explicit() {
        assert_ne!(ExecutionEpochAnchorModeV1::RequirePrevious, ExecutionEpochAnchorModeV1::AllowBootstrap);
    }
}
