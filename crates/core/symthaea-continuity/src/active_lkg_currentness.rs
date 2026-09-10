// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh rollback-resistant currentness for the active Last Known Good selection.
//!
//! An `ActiveKnownGoodSelectionV1` proves that one selection event was valid. It does
//! not, by itself, prove that no newer selection has superseded it. This module binds
//! the exact selection head to a fresh challenge and a monotonic platform-root anchor.
//!
//! `WasActive != FreshlyAttestedCurrentActive != RecoveryAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::active_lkg::{
    ActiveKnownGoodSelectionError, ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionV1,
};
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1,
};
use crate::known_good::KnownGoodCheckpointId;
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const ACTIVE_LKG_CURRENTNESS_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-active-lkg-currentness-claim-v1";
pub const ACTIVE_LKG_CURRENTNESS_AUTH_PURPOSE: &str =
    "symthaea.continuity.active-lkg-currentness.v1";

const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.active-lkg-currentness-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.active-lkg-currentness-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-active-lkg-currentness.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-active-lkg-currentness.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActiveLkgCurrentnessClaimId([u8; 32]);
impl ActiveLkgCurrentnessClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedActiveLkgCurrentnessId([u8; 32]);
impl AuthenticatedActiveLkgCurrentnessId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedActiveLkgCurrentnessId([u8; 32]);
impl QualifiedActiveLkgCurrentnessId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Serializable claim that one rollback-resistant platform/root freshly attested the
/// exact active-LKG head. The claim is evidence only until authenticated and qualified
/// against the exact live selection and expected fresh challenge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveLkgCurrentnessClaimV1 {
    schema_version: String,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    anchor_sequence: u64,
    predecessor_currentness_id: Option<QualifiedActiveLkgCurrentnessId>,
    subject_id: ContinuitySubjectId,
    active_selection_id: ActiveKnownGoodSelectionId,
    selection_generation: u64,
    predecessor_selection_id: Option<ActiveKnownGoodSelectionId>,
    checkpoint_id: KnownGoodCheckpointId,
    checkpoint_generation: u64,
    realization_id: TargetRealizationId,
    selected_at_unix_ms: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
    claim_id: ActiveLkgCurrentnessClaimId,
}

impl ActiveLkgCurrentnessClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: &ExecutionJournalAnchorProfileV1,
        active: &ActiveKnownGoodSelectionV1,
        anchor_sequence: u64,
        predecessor_currentness_id: Option<QualifiedActiveLkgCurrentnessId>,
        freshness_challenge_digest: [u8; 32],
        boot_instance_digest: [u8; 32],
        boot_counter: u64,
        monotonic_counter: u64,
        anchored_at_unix_ms: u64,
        raw_anchor_evidence_digest: [u8; 32],
    ) -> Result<Self, ActiveLkgCurrentnessError> {
        profile.validate()?;
        active.record().validate()?;
        validate_anchor_material(
            anchor_sequence,
            freshness_challenge_digest,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
        )?;
        if anchored_at_unix_ms < active.selected_at_unix_ms() {
            return Err(ActiveLkgCurrentnessError::AnchorPredatesSelection);
        }

        let profile_id = profile.id();
        let root_epoch = profile.root_epoch();
        let subject_id = active.subject_id();
        let active_selection_id = active.id();
        let selection_generation = active.generation();
        let predecessor_selection_id = active.predecessor_selection_id();
        let checkpoint_id = active.checkpoint_id();
        let checkpoint_generation = active.checkpoint_generation();
        let realization_id = active.realization_id();
        let selected_at_unix_ms = active.selected_at_unix_ms();
        let claim_id = ActiveLkgCurrentnessClaimId(hash_claim(
            profile_id,
            root_epoch,
            anchor_sequence,
            predecessor_currentness_id,
            subject_id,
            active_selection_id,
            selection_generation,
            predecessor_selection_id,
            checkpoint_id,
            checkpoint_generation,
            realization_id,
            selected_at_unix_ms,
            freshness_challenge_digest,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
        ));
        Ok(Self {
            schema_version: ACTIVE_LKG_CURRENTNESS_CLAIM_SCHEMA_V1.to_owned(),
            profile_id,
            root_epoch,
            anchor_sequence,
            predecessor_currentness_id,
            subject_id,
            active_selection_id,
            selection_generation,
            predecessor_selection_id,
            checkpoint_id,
            checkpoint_generation,
            realization_id,
            selected_at_unix_ms,
            freshness_challenge_digest,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), ActiveLkgCurrentnessError> {
        if self.schema_version != ACTIVE_LKG_CURRENTNESS_CLAIM_SCHEMA_V1 {
            return Err(ActiveLkgCurrentnessError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.root_epoch == 0 {
            return Err(ActiveLkgCurrentnessError::ZeroRootEpoch);
        }
        if self.selection_generation == 0 {
            return Err(ActiveLkgCurrentnessError::ZeroSelectionGeneration);
        }
        if self.checkpoint_generation == 0 {
            return Err(ActiveLkgCurrentnessError::ZeroCheckpointGeneration);
        }
        if self.selected_at_unix_ms == 0 {
            return Err(ActiveLkgCurrentnessError::ZeroSelectionTime);
        }
        validate_anchor_material(
            self.anchor_sequence,
            self.freshness_challenge_digest,
            self.boot_instance_digest,
            self.boot_counter,
            self.monotonic_counter,
            self.anchored_at_unix_ms,
            self.raw_anchor_evidence_digest,
        )?;
        if self.anchored_at_unix_ms < self.selected_at_unix_ms {
            return Err(ActiveLkgCurrentnessError::AnchorPredatesSelection);
        }
        let expected = ActiveLkgCurrentnessClaimId(hash_claim(
            self.profile_id,
            self.root_epoch,
            self.anchor_sequence,
            self.predecessor_currentness_id,
            self.subject_id,
            self.active_selection_id,
            self.selection_generation,
            self.predecessor_selection_id,
            self.checkpoint_id,
            self.checkpoint_generation,
            self.realization_id,
            self.selected_at_unix_ms,
            self.freshness_challenge_digest,
            self.boot_instance_digest,
            self.boot_counter,
            self.monotonic_counter,
            self.anchored_at_unix_ms,
            self.raw_anchor_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(ActiveLkgCurrentnessError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActiveLkgCurrentnessClaimId { self.claim_id }
}

/// Stable authentication bytes for a fresh platform attestation. Serde bytes are not
/// the signing/attestation contract.
pub fn canonical_active_lkg_currentness_claim_bytes(
    claim: &ActiveLkgCurrentnessClaimV1,
) -> Result<Vec<u8>, ActiveLkgCurrentnessError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(640);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.profile_id.as_bytes());
    out.extend_from_slice(&claim.root_epoch.to_le_bytes());
    out.extend_from_slice(&claim.anchor_sequence.to_le_bytes());
    encode_optional_id(&mut out, claim.predecessor_currentness_id.map(|id| *id.as_bytes()));
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.active_selection_id.as_bytes());
    out.extend_from_slice(&claim.selection_generation.to_le_bytes());
    encode_optional_id(&mut out, claim.predecessor_selection_id.map(|id| *id.as_bytes()));
    out.extend_from_slice(claim.checkpoint_id.as_bytes());
    out.extend_from_slice(&claim.checkpoint_generation.to_le_bytes());
    out.extend_from_slice(claim.realization_id.as_bytes());
    out.extend_from_slice(&claim.selected_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.freshness_challenge_digest);
    out.extend_from_slice(&claim.boot_instance_digest);
    out.extend_from_slice(&claim.boot_counter.to_le_bytes());
    out.extend_from_slice(&claim.monotonic_counter.to_le_bytes());
    out.extend_from_slice(&claim.anchored_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_anchor_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_active_lkg_currentness_claim_digest(
    claim: &ActiveLkgCurrentnessClaimV1,
) -> Result<[u8; 32], ActiveLkgCurrentnessError> {
    Ok(*blake3::hash(&canonical_active_lkg_currentness_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedActiveLkgCurrentnessV1 {
    claim: ActiveLkgCurrentnessClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedActiveLkgCurrentnessId,
}

impl AuthenticatedActiveLkgCurrentnessV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: ActiveLkgCurrentnessClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, ActiveLkgCurrentnessError> {
        claim.validate()?;
        profile.validate()?;
        if claim.profile_id != profile.id() || claim.root_epoch != profile.root_epoch() {
            return Err(ActiveLkgCurrentnessError::ProfileMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(ActiveLkgCurrentnessError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedActiveLkgCurrentnessId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                claim.id().as_bytes(),
                profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde proof that a fresh rollback-resistant attestation names this exact
/// selection as the current active-LKG head under an unbroken monotonic sequence.
#[derive(Debug, Clone)]
pub struct QualifiedActiveLkgCurrentnessV1 {
    qualified_id: QualifiedActiveLkgCurrentnessId,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    anchor_sequence: u64,
    predecessor_currentness_id: Option<QualifiedActiveLkgCurrentnessId>,
    subject_id: ContinuitySubjectId,
    active_selection_id: ActiveKnownGoodSelectionId,
    selection_generation: u64,
    predecessor_selection_id: Option<ActiveKnownGoodSelectionId>,
    checkpoint_id: KnownGoodCheckpointId,
    checkpoint_generation: u64,
    realization_id: TargetRealizationId,
    selected_at_unix_ms: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    authentication_evidence_id: AuthenticatedActiveLkgCurrentnessId,
}

impl QualifiedActiveLkgCurrentnessV1 {
    pub(crate) fn qualify(
        active: &ActiveKnownGoodSelectionV1,
        authenticated: &AuthenticatedActiveLkgCurrentnessV1,
        previous: Option<&QualifiedActiveLkgCurrentnessV1>,
        expected_freshness_challenge_digest: [u8; 32],
    ) -> Result<Self, ActiveLkgCurrentnessError> {
        active.record().validate()?;
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        if expected_freshness_challenge_digest == [0; 32] {
            return Err(ActiveLkgCurrentnessError::ZeroFreshnessChallengeDigest);
        }
        let claim = &authenticated.claim;
        if claim.profile_id != authenticated.profile.id()
            || claim.root_epoch != authenticated.profile.root_epoch()
        {
            return Err(ActiveLkgCurrentnessError::ProfileMismatch);
        }
        if claim.freshness_challenge_digest != expected_freshness_challenge_digest {
            return Err(ActiveLkgCurrentnessError::FreshnessChallengeMismatch);
        }
        if claim.active_selection_id != active.id()
            || claim.selection_generation != active.generation()
            || claim.predecessor_selection_id != active.predecessor_selection_id()
            || claim.checkpoint_id != active.checkpoint_id()
            || claim.checkpoint_generation != active.checkpoint_generation()
            || claim.subject_id != active.subject_id()
            || claim.realization_id != active.realization_id()
            || claim.selected_at_unix_ms != active.selected_at_unix_ms()
        {
            return Err(ActiveLkgCurrentnessError::SelectionContextMismatch);
        }
        validate_progression(previous, claim)?;

        let qualified_id = QualifiedActiveLkgCurrentnessId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[
                claim.id().as_bytes(),
                authenticated.profile.id().as_bytes(),
                &authenticated.profile.root_epoch().to_le_bytes(),
                authenticated.evidence_id.as_bytes(),
                &expected_freshness_challenge_digest,
            ],
        ));
        Ok(Self {
            qualified_id,
            profile_id: claim.profile_id,
            root_epoch: claim.root_epoch,
            anchor_sequence: claim.anchor_sequence,
            predecessor_currentness_id: claim.predecessor_currentness_id,
            subject_id: claim.subject_id,
            active_selection_id: claim.active_selection_id,
            selection_generation: claim.selection_generation,
            predecessor_selection_id: claim.predecessor_selection_id,
            checkpoint_id: claim.checkpoint_id,
            checkpoint_generation: claim.checkpoint_generation,
            realization_id: claim.realization_id,
            selected_at_unix_ms: claim.selected_at_unix_ms,
            freshness_challenge_digest: claim.freshness_challenge_digest,
            boot_instance_digest: claim.boot_instance_digest,
            boot_counter: claim.boot_counter,
            monotonic_counter: claim.monotonic_counter,
            anchored_at_unix_ms: claim.anchored_at_unix_ms,
            authentication_evidence_id: authenticated.evidence_id,
        })
    }

    pub fn id(&self) -> QualifiedActiveLkgCurrentnessId { self.qualified_id }
    pub fn profile_id(&self) -> ExecutionJournalAnchorProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
    pub fn anchor_sequence(&self) -> u64 { self.anchor_sequence }
    pub fn predecessor_currentness_id(&self) -> Option<QualifiedActiveLkgCurrentnessId> {
        self.predecessor_currentness_id
    }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn active_selection_id(&self) -> ActiveKnownGoodSelectionId { self.active_selection_id }
    pub fn selection_generation(&self) -> u64 { self.selection_generation }
    pub fn checkpoint_id(&self) -> KnownGoodCheckpointId { self.checkpoint_id }
    pub fn checkpoint_generation(&self) -> u64 { self.checkpoint_generation }
    pub fn realization_id(&self) -> TargetRealizationId { self.realization_id }
    pub fn freshness_challenge_digest(&self) -> [u8; 32] { self.freshness_challenge_digest }
    pub fn boot_counter(&self) -> u64 { self.boot_counter }
    pub fn monotonic_counter(&self) -> u64 { self.monotonic_counter }
    pub fn anchored_at_unix_ms(&self) -> u64 { self.anchored_at_unix_ms }
}

fn validate_progression(
    previous: Option<&QualifiedActiveLkgCurrentnessV1>,
    next: &ActiveLkgCurrentnessClaimV1,
) -> Result<(), ActiveLkgCurrentnessError> {
    match previous {
        None => {
            if next.anchor_sequence != 1 || next.predecessor_currentness_id.is_some() {
                return Err(ActiveLkgCurrentnessError::InvalidInitialAnchor);
            }
            if next.selection_generation != 1 || next.predecessor_selection_id.is_some() {
                return Err(ActiveLkgCurrentnessError::InitialAnchorRequiresGenerationOneSelection);
            }
        }
        Some(previous) => {
            if previous.profile_id != next.profile_id || previous.root_epoch != next.root_epoch {
                return Err(ActiveLkgCurrentnessError::ProfileLineageMismatch);
            }
            if previous.subject_id != next.subject_id {
                return Err(ActiveLkgCurrentnessError::SubjectLineageMismatch);
            }
            let expected_sequence = previous
                .anchor_sequence
                .checked_add(1)
                .ok_or(ActiveLkgCurrentnessError::AnchorSequenceOverflow)?;
            if next.anchor_sequence != expected_sequence
                || next.predecessor_currentness_id != Some(previous.id())
            {
                return Err(ActiveLkgCurrentnessError::AnchorSequenceMismatch);
            }

            match next.selection_generation.cmp(&previous.selection_generation) {
                std::cmp::Ordering::Less => {
                    return Err(ActiveLkgCurrentnessError::SelectionGenerationRollback {
                        previous: previous.selection_generation,
                        observed: next.selection_generation,
                    });
                }
                std::cmp::Ordering::Equal => {
                    if next.active_selection_id != previous.active_selection_id
                        || next.predecessor_selection_id != previous.predecessor_selection_id
                        || next.checkpoint_id != previous.checkpoint_id
                        || next.checkpoint_generation != previous.checkpoint_generation
                        || next.realization_id != previous.realization_id
                        || next.selected_at_unix_ms != previous.selected_at_unix_ms
                    {
                        return Err(ActiveLkgCurrentnessError::SameGenerationSelectionDrift);
                    }
                }
                std::cmp::Ordering::Greater => {
                    let expected_generation = previous
                        .selection_generation
                        .checked_add(1)
                        .ok_or(ActiveLkgCurrentnessError::SelectionGenerationOverflow)?;
                    if next.selection_generation != expected_generation {
                        return Err(ActiveLkgCurrentnessError::SelectionGenerationSkip {
                            previous: previous.selection_generation,
                            observed: next.selection_generation,
                        });
                    }
                    if next.predecessor_selection_id != Some(previous.active_selection_id) {
                        return Err(ActiveLkgCurrentnessError::SelectionPredecessorMismatch);
                    }
                    if next.checkpoint_generation <= previous.checkpoint_generation {
                        return Err(ActiveLkgCurrentnessError::CheckpointGenerationDidNotAdvance {
                            previous: previous.checkpoint_generation,
                            observed: next.checkpoint_generation,
                        });
                    }
                    if next.selected_at_unix_ms <= previous.selected_at_unix_ms {
                        return Err(ActiveLkgCurrentnessError::SelectionTimeDidNotAdvance);
                    }
                }
            }

            if next.boot_counter < previous.boot_counter {
                return Err(ActiveLkgCurrentnessError::BootCounterRollback {
                    previous: previous.boot_counter,
                    observed: next.boot_counter,
                });
            }
            if next.boot_counter == previous.boot_counter {
                if next.boot_instance_digest != previous.boot_instance_digest {
                    return Err(ActiveLkgCurrentnessError::BootInstanceDriftWithoutCounterAdvance);
                }
                if next.monotonic_counter <= previous.monotonic_counter {
                    return Err(ActiveLkgCurrentnessError::MonotonicCounterDidNotAdvance {
                        previous: previous.monotonic_counter,
                        observed: next.monotonic_counter,
                    });
                }
            }
            if next.anchored_at_unix_ms < previous.anchored_at_unix_ms {
                return Err(ActiveLkgCurrentnessError::AnchorTimeRollback);
            }
            if next.freshness_challenge_digest == previous.freshness_challenge_digest {
                return Err(ActiveLkgCurrentnessError::FreshnessChallengeReplay);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActiveLkgCurrentnessError {
    #[error(transparent)]
    ActiveSelection(#[from] ActiveKnownGoodSelectionError),
    #[error(transparent)]
    PlatformProfile(#[from] ExecutionJournalAnchorError),
    #[error("unsupported active-LKG currentness claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("active-LKG currentness root epoch must be non-zero")]
    ZeroRootEpoch,
    #[error("active-LKG currentness selection generation must be non-zero")]
    ZeroSelectionGeneration,
    #[error("active-LKG currentness checkpoint generation must be non-zero")]
    ZeroCheckpointGeneration,
    #[error("active-LKG currentness selection time must be non-zero")]
    ZeroSelectionTime,
    #[error("active-LKG currentness anchor sequence must be non-zero")]
    ZeroAnchorSequence,
    #[error("active-LKG currentness freshness challenge digest must be non-zero")]
    ZeroFreshnessChallengeDigest,
    #[error("active-LKG currentness boot-instance digest must be non-zero")]
    ZeroBootInstanceDigest,
    #[error("active-LKG currentness boot counter must be non-zero")]
    ZeroBootCounter,
    #[error("active-LKG currentness monotonic counter must be non-zero")]
    ZeroMonotonicCounter,
    #[error("active-LKG currentness anchor time must be non-zero")]
    ZeroAnchorTime,
    #[error("active-LKG currentness raw anchor evidence digest must be non-zero")]
    ZeroRawAnchorEvidenceDigest,
    #[error("active-LKG currentness anchor predates selected LKG")]
    AnchorPredatesSelection,
    #[error("active-LKG currentness claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("active-LKG currentness platform profile/root does not match claim")]
    ProfileMismatch,
    #[error("active-LKG currentness authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("active-LKG currentness claim does not bind the expected fresh challenge")]
    FreshnessChallengeMismatch,
    #[error("active-LKG currentness claim does not bind the exact live selection")]
    SelectionContextMismatch,
    #[error("first active-LKG currentness anchor must be sequence 1 with no predecessor")]
    InvalidInitialAnchor,
    #[error("first active-LKG currentness anchor requires exact generation-one selection")]
    InitialAnchorRequiresGenerationOneSelection,
    #[error("active-LKG currentness platform profile/root lineage changed")]
    ProfileLineageMismatch,
    #[error("active-LKG currentness subject lineage changed")]
    SubjectLineageMismatch,
    #[error("active-LKG currentness sequence/predecessor is not the exact next anchor")]
    AnchorSequenceMismatch,
    #[error("active-LKG currentness anchor sequence overflow")]
    AnchorSequenceOverflow,
    #[error("active-LKG selection generation rolled back from {previous} to {observed}")]
    SelectionGenerationRollback { previous: u64, observed: u64 },
    #[error("same active-LKG selection generation changed identity or checkpoint fields")]
    SameGenerationSelectionDrift,
    #[error("active-LKG selection generation skipped from {previous} to {observed}")]
    SelectionGenerationSkip { previous: u64, observed: u64 },
    #[error("active-LKG selection generation overflow")]
    SelectionGenerationOverflow,
    #[error("new active-LKG selection does not name the exact previous selection")]
    SelectionPredecessorMismatch,
    #[error("active-LKG checkpoint generation did not advance: previous {previous}, observed {observed}")]
    CheckpointGenerationDidNotAdvance { previous: u64, observed: u64 },
    #[error("active-LKG selection time did not strictly advance")]
    SelectionTimeDidNotAdvance,
    #[error("active-LKG currentness boot counter rolled back from {previous} to {observed}")]
    BootCounterRollback { previous: u64, observed: u64 },
    #[error("active-LKG currentness boot identity changed without boot-counter advance")]
    BootInstanceDriftWithoutCounterAdvance,
    #[error("active-LKG currentness monotonic counter did not strictly advance: previous {previous}, observed {observed}")]
    MonotonicCounterDidNotAdvance { previous: u64, observed: u64 },
    #[error("active-LKG currentness anchor wall-clock time moved backwards")]
    AnchorTimeRollback,
    #[error("active-LKG currentness freshness challenge was replayed")]
    FreshnessChallengeReplay,
}

fn validate_anchor_material(
    anchor_sequence: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
) -> Result<(), ActiveLkgCurrentnessError> {
    if anchor_sequence == 0 { return Err(ActiveLkgCurrentnessError::ZeroAnchorSequence); }
    if freshness_challenge_digest == [0; 32] {
        return Err(ActiveLkgCurrentnessError::ZeroFreshnessChallengeDigest);
    }
    if boot_instance_digest == [0; 32] {
        return Err(ActiveLkgCurrentnessError::ZeroBootInstanceDigest);
    }
    if boot_counter == 0 { return Err(ActiveLkgCurrentnessError::ZeroBootCounter); }
    if monotonic_counter == 0 { return Err(ActiveLkgCurrentnessError::ZeroMonotonicCounter); }
    if anchored_at_unix_ms == 0 { return Err(ActiveLkgCurrentnessError::ZeroAnchorTime); }
    if raw_anchor_evidence_digest == [0; 32] {
        return Err(ActiveLkgCurrentnessError::ZeroRawAnchorEvidenceDigest);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    anchor_sequence: u64,
    predecessor_currentness_id: Option<QualifiedActiveLkgCurrentnessId>,
    subject_id: ContinuitySubjectId,
    active_selection_id: ActiveKnownGoodSelectionId,
    selection_generation: u64,
    predecessor_selection_id: Option<ActiveKnownGoodSelectionId>,
    checkpoint_id: KnownGoodCheckpointId,
    checkpoint_generation: u64,
    realization_id: TargetRealizationId,
    selected_at_unix_ms: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(profile_id.as_bytes());
    hasher.update(&root_epoch.to_le_bytes());
    hasher.update(&anchor_sequence.to_le_bytes());
    hash_optional_id(&mut hasher, predecessor_currentness_id.map(|id| *id.as_bytes()));
    hasher.update(subject_id.as_bytes());
    hasher.update(active_selection_id.as_bytes());
    hasher.update(&selection_generation.to_le_bytes());
    hash_optional_id(&mut hasher, predecessor_selection_id.map(|id| *id.as_bytes()));
    hasher.update(checkpoint_id.as_bytes());
    hasher.update(&checkpoint_generation.to_le_bytes());
    hasher.update(realization_id.as_bytes());
    hasher.update(&selected_at_unix_ms.to_le_bytes());
    hasher.update(&freshness_challenge_digest);
    hasher.update(&boot_instance_digest);
    hasher.update(&boot_counter.to_le_bytes());
    hasher.update(&monotonic_counter.to_le_bytes());
    hasher.update(&anchored_at_unix_ms.to_le_bytes());
    hasher.update(&raw_anchor_evidence_digest);
    *hasher.finalize().as_bytes()
}

fn encode_optional_id(out: &mut Vec<u8>, id: Option<[u8; 32]>) {
    match id {
        Some(id) => {
            out.push(1);
            out.extend_from_slice(&id);
        }
        None => out.push(0),
    }
}

fn hash_optional_id(hasher: &mut blake3::Hasher, id: Option<[u8; 32]>) {
    match id {
        Some(id) => {
            hasher.update(&[1]);
            hasher.update(&id);
        }
        None => { hasher.update(&[0]); }
    }
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
    fn currentness_domains_are_distinct() {
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }

    #[test]
    fn currentness_has_explicit_authentication_purpose() {
        assert_eq!(
            ACTIVE_LKG_CURRENTNESS_AUTH_PURPOSE,
            "symthaea.continuity.active-lkg-currentness.v1"
        );
    }

    #[test]
    fn zero_challenge_is_rejected() {
        assert_eq!(
            validate_anchor_material(1, [0; 32], [1; 32], 1, 1, 1, [2; 32]).unwrap_err(),
            ActiveLkgCurrentnessError::ZeroFreshnessChallengeDigest
        );
    }
}
