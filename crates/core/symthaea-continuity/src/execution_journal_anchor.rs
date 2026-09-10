// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rollback-resistant anchoring contract for the reconstructed execution journal.
//!
//! The pure continuity kernel cannot make disk state rollback resistant. Instead it
//! defines the exact state a TPM-NV/BMC/platform adapter must authenticate and the
//! monotonic progression rules required before an anchored journal may be trusted.
//!
//! `ReconstructedJournal != AuthenticatedAnchor != RollbackResistanceByItself`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution_journal::{ExecutionJournalDigest, ReconstructedExecutionJournalV1};
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::{QualifiedTrustedCommitEpochId, QualifiedTrustedCommitEpochV1};

pub const EXECUTION_JOURNAL_ANCHOR_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-execution-journal-anchor-profile-v1";
pub const EXECUTION_JOURNAL_ANCHOR_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-execution-journal-anchor-claim-v1";
pub const EXECUTION_JOURNAL_ANCHOR_AUTH_PURPOSE: &str =
    "symthaea.continuity.execution-journal-anchor.v1";

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.execution-journal-anchor-profile.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.execution-journal-anchor-claim.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-execution-journal-anchor.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-execution-journal-anchor.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.execution-journal-anchor-wire.v1\0";
const MAX_TEXT_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionJournalAnchorProfileId([u8; 32]);
impl ExecutionJournalAnchorProfileId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionJournalAnchorClaimId([u8; 32]);
impl ExecutionJournalAnchorClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedExecutionJournalAnchorId([u8; 32]);
impl AuthenticatedExecutionJournalAnchorId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedExecutionJournalAnchorId([u8; 32]);
impl QualifiedExecutionJournalAnchorId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Exact platform/root identity allowed to authenticate rollback-resistant anchors.
/// This is descriptive trust configuration, not an anchor or capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionJournalAnchorProfileV1 {
    schema_version: String,
    profile_name: String,
    anchor_root_digest: [u8; 32],
    root_epoch: u64,
    implementation_digest: [u8; 32],
    profile_id: ExecutionJournalAnchorProfileId,
}

impl ExecutionJournalAnchorProfileV1 {
    pub fn new(
        profile_name: impl Into<String>,
        anchor_root_digest: [u8; 32],
        root_epoch: u64,
        implementation_digest: [u8; 32],
    ) -> Result<Self, ExecutionJournalAnchorError> {
        let profile_name = checked_text("journal anchor profile name", profile_name.into())?;
        if anchor_root_digest == [0; 32] {
            return Err(ExecutionJournalAnchorError::ZeroAnchorRootDigest);
        }
        if root_epoch == 0 {
            return Err(ExecutionJournalAnchorError::ZeroRootEpoch);
        }
        if implementation_digest == [0; 32] {
            return Err(ExecutionJournalAnchorError::ZeroImplementationDigest);
        }
        let profile_id = ExecutionJournalAnchorProfileId(hash_profile(
            &profile_name,
            anchor_root_digest,
            root_epoch,
            implementation_digest,
        ));
        Ok(Self {
            schema_version: EXECUTION_JOURNAL_ANCHOR_PROFILE_SCHEMA_V1.to_owned(),
            profile_name,
            anchor_root_digest,
            root_epoch,
            implementation_digest,
            profile_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExecutionJournalAnchorError> {
        if self.schema_version != EXECUTION_JOURNAL_ANCHOR_PROFILE_SCHEMA_V1 {
            return Err(ExecutionJournalAnchorError::UnsupportedProfileSchema(
                self.schema_version.clone(),
            ));
        }
        let canonical = checked_text("journal anchor profile name", self.profile_name.clone())?;
        if canonical != self.profile_name {
            return Err(ExecutionJournalAnchorError::NonCanonicalProfileName);
        }
        if self.anchor_root_digest == [0; 32] {
            return Err(ExecutionJournalAnchorError::ZeroAnchorRootDigest);
        }
        if self.root_epoch == 0 {
            return Err(ExecutionJournalAnchorError::ZeroRootEpoch);
        }
        if self.implementation_digest == [0; 32] {
            return Err(ExecutionJournalAnchorError::ZeroImplementationDigest);
        }
        let expected = ExecutionJournalAnchorProfileId(hash_profile(
            &self.profile_name,
            self.anchor_root_digest,
            self.root_epoch,
            self.implementation_digest,
        ));
        if expected != self.profile_id {
            return Err(ExecutionJournalAnchorError::ProfileIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ExecutionJournalAnchorProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
}

/// Serializable claim that one rollback-resistant adapter committed one exact
/// reconstructed journal state under one exact trusted commit epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionJournalAnchorClaimV1 {
    schema_version: String,
    profile_id: ExecutionJournalAnchorProfileId,
    anchor_sequence: u64,
    predecessor_anchor_id: Option<QualifiedExecutionJournalAnchorId>,
    subject_id: ContinuitySubjectId,
    journal_digest: ExecutionJournalDigest,
    journal_entry_count: u64,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
    claim_id: ExecutionJournalAnchorClaimId,
}

impl ExecutionJournalAnchorClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: &ExecutionJournalAnchorProfileV1,
        journal: &ReconstructedExecutionJournalV1,
        epoch: &QualifiedTrustedCommitEpochV1,
        anchor_sequence: u64,
        predecessor_anchor_id: Option<QualifiedExecutionJournalAnchorId>,
        anchored_at_unix_ms: u64,
        raw_anchor_evidence_digest: [u8; 32],
    ) -> Result<Self, ExecutionJournalAnchorError> {
        profile.validate()?;
        validate_claim_material(
            anchor_sequence,
            journal.len() as u64,
            epoch.boot_instance_digest(),
            epoch.boot_counter(),
            epoch.monotonic_counter(),
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
        )?;
        let profile_id = profile.id();
        let subject_id = epoch.subject_id();
        let journal_digest = journal.digest();
        let journal_entry_count = journal.len() as u64;
        let trusted_epoch_id = epoch.id();
        let boot_instance_digest = epoch.boot_instance_digest();
        let boot_counter = epoch.boot_counter();
        let monotonic_counter = epoch.monotonic_counter();
        let claim_id = ExecutionJournalAnchorClaimId(hash_claim(
            profile_id,
            anchor_sequence,
            predecessor_anchor_id,
            subject_id,
            journal_digest,
            journal_entry_count,
            trusted_epoch_id,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
        ));
        Ok(Self {
            schema_version: EXECUTION_JOURNAL_ANCHOR_CLAIM_SCHEMA_V1.to_owned(),
            profile_id,
            anchor_sequence,
            predecessor_anchor_id,
            subject_id,
            journal_digest,
            journal_entry_count,
            trusted_epoch_id,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExecutionJournalAnchorError> {
        if self.schema_version != EXECUTION_JOURNAL_ANCHOR_CLAIM_SCHEMA_V1 {
            return Err(ExecutionJournalAnchorError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_claim_material(
            self.anchor_sequence,
            self.journal_entry_count,
            self.boot_instance_digest,
            self.boot_counter,
            self.monotonic_counter,
            self.anchored_at_unix_ms,
            self.raw_anchor_evidence_digest,
        )?;
        let expected = ExecutionJournalAnchorClaimId(hash_claim(
            self.profile_id,
            self.anchor_sequence,
            self.predecessor_anchor_id,
            self.subject_id,
            self.journal_digest,
            self.journal_entry_count,
            self.trusted_epoch_id,
            self.boot_instance_digest,
            self.boot_counter,
            self.monotonic_counter,
            self.anchored_at_unix_ms,
            self.raw_anchor_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(ExecutionJournalAnchorError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ExecutionJournalAnchorClaimId { self.claim_id }
}

/// Stable authentication payload. Serde bytes are not the signing/attestation contract.
pub fn canonical_execution_journal_anchor_claim_bytes(
    claim: &ExecutionJournalAnchorClaimV1,
) -> Result<Vec<u8>, ExecutionJournalAnchorError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.profile_id.as_bytes());
    out.extend_from_slice(&claim.anchor_sequence.to_le_bytes());
    match claim.predecessor_anchor_id {
        Some(id) => {
            out.push(1);
            out.extend_from_slice(id.as_bytes());
        }
        None => out.push(0),
    }
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.journal_digest.as_bytes());
    out.extend_from_slice(&claim.journal_entry_count.to_le_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    out.extend_from_slice(&claim.boot_instance_digest);
    out.extend_from_slice(&claim.boot_counter.to_le_bytes());
    out.extend_from_slice(&claim.monotonic_counter.to_le_bytes());
    out.extend_from_slice(&claim.anchored_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_anchor_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_execution_journal_anchor_claim_digest(
    claim: &ExecutionJournalAnchorClaimV1,
) -> Result<[u8; 32], ExecutionJournalAnchorError> {
    Ok(*blake3::hash(&canonical_execution_journal_anchor_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedExecutionJournalAnchorV1 {
    claim: ExecutionJournalAnchorClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedExecutionJournalAnchorId,
}

impl AuthenticatedExecutionJournalAnchorV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: ExecutionJournalAnchorClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, ExecutionJournalAnchorError> {
        claim.validate()?;
        profile.validate()?;
        if claim.profile_id != profile.id() {
            return Err(ExecutionJournalAnchorError::ProfileMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(ExecutionJournalAnchorError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedExecutionJournalAnchorId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                claim.id().as_bytes(),
                profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self {
            claim,
            profile,
            authentication_evidence_digest,
            evidence_id,
        })
    }
}

/// Non-Serde exact journal anchor admitted against the actual reconstructed journal
/// and exact trusted epoch. Persistence/anti-rollback strength still comes from the
/// authenticated adapter/root, not from this Rust value by itself.
#[derive(Debug, Clone)]
pub struct QualifiedExecutionJournalAnchorV1 {
    qualified_id: QualifiedExecutionJournalAnchorId,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    anchor_sequence: u64,
    predecessor_anchor_id: Option<QualifiedExecutionJournalAnchorId>,
    subject_id: ContinuitySubjectId,
    journal_digest: ExecutionJournalDigest,
    journal_entry_count: u64,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    authentication_evidence_id: AuthenticatedExecutionJournalAnchorId,
}

impl QualifiedExecutionJournalAnchorV1 {
    pub(crate) fn qualify(
        journal: &ReconstructedExecutionJournalV1,
        epoch: &QualifiedTrustedCommitEpochV1,
        authenticated: &AuthenticatedExecutionJournalAnchorV1,
        previous: Option<&QualifiedExecutionJournalAnchorV1>,
    ) -> Result<Self, ExecutionJournalAnchorError> {
        let claim = &authenticated.claim;
        claim.validate()?;
        authenticated.profile.validate()?;
        if claim.profile_id != authenticated.profile.id() {
            return Err(ExecutionJournalAnchorError::ProfileMismatch);
        }
        if claim.subject_id != epoch.subject_id()
            || claim.journal_digest != journal.digest()
            || claim.journal_entry_count != journal.len() as u64
            || claim.trusted_epoch_id != epoch.id()
            || claim.boot_instance_digest != epoch.boot_instance_digest()
            || claim.boot_counter != epoch.boot_counter()
            || claim.monotonic_counter != epoch.monotonic_counter()
            || claim.anchored_at_unix_ms != epoch.accepted_unix_ms()
        {
            return Err(ExecutionJournalAnchorError::AnchorContextMismatch);
        }
        validate_progression(previous, claim, authenticated.profile.id(), authenticated.profile.root_epoch())?;

        let qualified_id = QualifiedExecutionJournalAnchorId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[
                claim.id().as_bytes(),
                authenticated.profile.id().as_bytes(),
                &authenticated.profile.root_epoch().to_le_bytes(),
                authenticated.evidence_id.as_bytes(),
                claim.journal_digest.as_bytes(),
                claim.trusted_epoch_id.as_bytes(),
            ],
        ));
        Ok(Self {
            qualified_id,
            profile_id: authenticated.profile.id(),
            root_epoch: authenticated.profile.root_epoch(),
            anchor_sequence: claim.anchor_sequence,
            predecessor_anchor_id: claim.predecessor_anchor_id,
            subject_id: claim.subject_id,
            journal_digest: claim.journal_digest,
            journal_entry_count: claim.journal_entry_count,
            trusted_epoch_id: claim.trusted_epoch_id,
            boot_instance_digest: claim.boot_instance_digest,
            boot_counter: claim.boot_counter,
            monotonic_counter: claim.monotonic_counter,
            anchored_at_unix_ms: claim.anchored_at_unix_ms,
            authentication_evidence_id: authenticated.evidence_id,
        })
    }

    pub fn id(&self) -> QualifiedExecutionJournalAnchorId { self.qualified_id }
    pub fn profile_id(&self) -> ExecutionJournalAnchorProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
    pub fn anchor_sequence(&self) -> u64 { self.anchor_sequence }
    pub fn predecessor_anchor_id(&self) -> Option<QualifiedExecutionJournalAnchorId> {
        self.predecessor_anchor_id
    }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn journal_digest(&self) -> ExecutionJournalDigest { self.journal_digest }
    pub fn journal_entry_count(&self) -> u64 { self.journal_entry_count }
    pub fn trusted_epoch_id(&self) -> QualifiedTrustedCommitEpochId { self.trusted_epoch_id }
    pub fn boot_counter(&self) -> u64 { self.boot_counter }
    pub fn monotonic_counter(&self) -> u64 { self.monotonic_counter }
    pub fn anchored_at_unix_ms(&self) -> u64 { self.anchored_at_unix_ms }
}

fn validate_progression(
    previous: Option<&QualifiedExecutionJournalAnchorV1>,
    next: &ExecutionJournalAnchorClaimV1,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
) -> Result<(), ExecutionJournalAnchorError> {
    match previous {
        None => {
            if next.anchor_sequence != 1 || next.predecessor_anchor_id.is_some() {
                return Err(ExecutionJournalAnchorError::InvalidInitialAnchor);
            }
        }
        Some(previous) => {
            if previous.profile_id != profile_id || previous.root_epoch != root_epoch {
                return Err(ExecutionJournalAnchorError::AnchorProfileLineageMismatch);
            }
            if previous.subject_id != next.subject_id {
                return Err(ExecutionJournalAnchorError::SubjectLineageMismatch);
            }
            let expected_sequence = previous
                .anchor_sequence
                .checked_add(1)
                .ok_or(ExecutionJournalAnchorError::AnchorSequenceOverflow)?;
            if next.anchor_sequence != expected_sequence
                || next.predecessor_anchor_id != Some(previous.id())
            {
                return Err(ExecutionJournalAnchorError::AnchorSequenceMismatch);
            }
            if next.journal_entry_count < previous.journal_entry_count {
                return Err(ExecutionJournalAnchorError::JournalEntryCountRollback {
                    previous: previous.journal_entry_count,
                    observed: next.journal_entry_count,
                });
            }
            if next.boot_counter < previous.boot_counter {
                return Err(ExecutionJournalAnchorError::BootCounterRollback {
                    previous: previous.boot_counter,
                    observed: next.boot_counter,
                });
            }
            if next.boot_counter == previous.boot_counter {
                if next.boot_instance_digest != previous.boot_instance_digest {
                    return Err(ExecutionJournalAnchorError::BootInstanceDriftWithoutCounterAdvance);
                }
                if next.monotonic_counter <= previous.monotonic_counter {
                    return Err(ExecutionJournalAnchorError::MonotonicCounterRollback {
                        previous: previous.monotonic_counter,
                        observed: next.monotonic_counter,
                    });
                }
            }
            if next.anchored_at_unix_ms < previous.anchored_at_unix_ms {
                return Err(ExecutionJournalAnchorError::AnchorTimeRollback);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionJournalAnchorError {
    #[error("unsupported execution journal anchor profile schema: {0}")]
    UnsupportedProfileSchema(String),
    #[error("unsupported execution journal anchor claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds the text bound")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("journal anchor profile name is not canonical")]
    NonCanonicalProfileName,
    #[error("journal anchor root digest must be non-zero")]
    ZeroAnchorRootDigest,
    #[error("journal anchor root epoch must be non-zero")]
    ZeroRootEpoch,
    #[error("journal anchor implementation digest must be non-zero")]
    ZeroImplementationDigest,
    #[error("journal anchor profile identity mismatch")]
    ProfileIdentityMismatch,
    #[error("journal anchor sequence must be non-zero")]
    ZeroAnchorSequence,
    #[error("journal anchor boot-instance digest must be non-zero")]
    ZeroBootInstanceDigest,
    #[error("journal anchor boot counter must be non-zero")]
    ZeroBootCounter,
    #[error("journal anchor monotonic counter must be non-zero")]
    ZeroMonotonicCounter,
    #[error("journal anchor time must be non-zero")]
    ZeroAnchorTime,
    #[error("journal anchor raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("journal anchor claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("journal anchor profile does not match claim")]
    ProfileMismatch,
    #[error("journal anchor authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("authenticated anchor does not match exact journal/epoch context")]
    AnchorContextMismatch,
    #[error("first journal anchor must be sequence 1 with no predecessor")]
    InvalidInitialAnchor,
    #[error("journal anchor profile/root lineage changed without a new trust transition")]
    AnchorProfileLineageMismatch,
    #[error("journal anchor subject lineage changed")]
    SubjectLineageMismatch,
    #[error("journal anchor sequence/predecessor is not the exact next anchor")]
    AnchorSequenceMismatch,
    #[error("journal anchor sequence overflow")]
    AnchorSequenceOverflow,
    #[error("journal entry count rolled back from {previous} to {observed}")]
    JournalEntryCountRollback { previous: u64, observed: u64 },
    #[error("journal anchor boot counter rolled back from {previous} to {observed}")]
    BootCounterRollback { previous: u64, observed: u64 },
    #[error("journal anchor boot identity changed without boot-counter advance")]
    BootInstanceDriftWithoutCounterAdvance,
    #[error("journal anchor monotonic counter did not strictly advance: previous {previous}, observed {observed}")]
    MonotonicCounterRollback { previous: u64, observed: u64 },
    #[error("journal anchor wall-clock time moved backwards")]
    AnchorTimeRollback,
}

fn validate_claim_material(
    anchor_sequence: u64,
    _journal_entry_count: u64,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
) -> Result<(), ExecutionJournalAnchorError> {
    if anchor_sequence == 0 { return Err(ExecutionJournalAnchorError::ZeroAnchorSequence); }
    if boot_instance_digest == [0; 32] { return Err(ExecutionJournalAnchorError::ZeroBootInstanceDigest); }
    if boot_counter == 0 { return Err(ExecutionJournalAnchorError::ZeroBootCounter); }
    if monotonic_counter == 0 { return Err(ExecutionJournalAnchorError::ZeroMonotonicCounter); }
    if anchored_at_unix_ms == 0 { return Err(ExecutionJournalAnchorError::ZeroAnchorTime); }
    if raw_anchor_evidence_digest == [0; 32] { return Err(ExecutionJournalAnchorError::ZeroRawEvidenceDigest); }
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, ExecutionJournalAnchorError> {
    let trimmed = value.trim();
    if trimmed.is_empty() { return Err(ExecutionJournalAnchorError::BlankText { field }); }
    if trimmed.len() > MAX_TEXT_BYTES { return Err(ExecutionJournalAnchorError::TextTooLong { field }); }
    if trimmed.chars().any(char::is_control) { return Err(ExecutionJournalAnchorError::ControlCharacters { field }); }
    Ok(trimmed.to_owned())
}

fn hash_profile(
    name: &str,
    root: [u8; 32],
    root_epoch: u64,
    implementation: [u8; 32],
) -> [u8; 32] {
    domain_hash_parts(PROFILE_DOMAIN, &[
        name.as_bytes(),
        &root,
        &root_epoch.to_le_bytes(),
        &implementation,
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    profile_id: ExecutionJournalAnchorProfileId,
    anchor_sequence: u64,
    predecessor: Option<QualifiedExecutionJournalAnchorId>,
    subject_id: ContinuitySubjectId,
    journal_digest: ExecutionJournalDigest,
    journal_entry_count: u64,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(profile_id.as_bytes());
    hasher.update(&anchor_sequence.to_le_bytes());
    match predecessor {
        Some(id) => {
            hasher.update(&[1]);
            hasher.update(id.as_bytes());
        }
        None => { hasher.update(&[0]); }
    }
    hasher.update(subject_id.as_bytes());
    hasher.update(journal_digest.as_bytes());
    hasher.update(&journal_entry_count.to_le_bytes());
    hasher.update(trusted_epoch_id.as_bytes());
    hasher.update(&boot_instance_digest);
    hasher.update(&boot_counter.to_le_bytes());
    hasher.update(&monotonic_counter.to_le_bytes());
    hasher.update(&anchored_at_unix_ms.to_le_bytes());
    hasher.update(&raw_anchor_evidence_digest);
    *hasher.finalize().as_bytes()
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
    fn anchor_profile_identity_changes_with_root_epoch() {
        let a = ExecutionJournalAnchorProfileV1::new("tpm-nv", [1; 32], 1, [2; 32]).unwrap();
        let b = ExecutionJournalAnchorProfileV1::new("tpm-nv", [1; 32], 2, [2; 32]).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn anchor_purpose_is_domain_specific() {
        assert_eq!(EXECUTION_JOURNAL_ANCHOR_AUTH_PURPOSE, "symthaea.continuity.execution-journal-anchor.v1");
    }
}