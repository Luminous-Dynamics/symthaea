// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash/restart-safe replay fencing for welfare-intervention authority.
//!
//! The in-memory `WelfareAuthorityTracker` prevents replay only for one process lifetime. This
//! module adds a hash-committed replay snapshot that must be durably persisted after authority
//! verification and before a permit may be minted. Snapshot integrity is not the same as rollback
//! resistance: `recover_anchored` requires the expected latest digest from an external durable or
//! monotonic anchor.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::{InterventionRequest, WelfareConstraintLevel};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, VerifiedWelfareInterventionAuthority,
    WelfareAuthorityContextError, WelfareAuthorityPolicyManifest, WelfareAuthoritySignatureVerifier,
    WelfareAuthorityTracker, verify_context_bound_welfare_authority,
};
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::authority_evidence_binding::{
    EvidenceBoundAuthorityError, authorize_evidence_bound_intervention_once,
    evidence_bound_authority_nonce,
};
use crate::evidence_context::{
    EvidenceBoundInterventionPermit, WelfareEvidenceContextError,
    derive_welfare_evidence_context,
};

pub const DURABLE_REPLAY_SCHEMA: &str = "symthaea.welfare.authority-replay-snapshot.v1";
const MAX_REPLAY_AUTHORITIES: usize = 4096;
const MAX_ID_BYTES: usize = 256;
const SNAPSHOT_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.authority-replay-snapshot-digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayScopeCursor {
    pub authority_epoch: u64,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayScopeRecord {
    pub target_id: String,
    pub action_code: u8,
    pub cursor: ReplayScopeCursor,
}

/// Persistable replay state. `scopes` must be canonical and sorted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableAuthorityReplaySnapshot {
    pub schema_version: String,
    pub generation: u64,
    pub previous_snapshot_digest: Option<Sha256Digest>,
    pub used_authority_ids: BTreeSet<String>,
    pub scopes: Vec<ReplayScopeRecord>,
}

impl DurableAuthorityReplaySnapshot {
    pub fn validate(&self) -> Result<(), DurableReplayError> {
        if self.schema_version != DURABLE_REPLAY_SCHEMA {
            return Err(DurableReplayError::UnsupportedSchema);
        }
        if self.used_authority_ids.len() > MAX_REPLAY_AUTHORITIES
            || self.scopes.len() > MAX_REPLAY_AUTHORITIES
        {
            return Err(DurableReplayError::CapacityExceeded {
                maximum: MAX_REPLAY_AUTHORITIES,
            });
        }
        if self.generation == 0 && self.previous_snapshot_digest.is_some() {
            return Err(DurableReplayError::InvalidGenesisSnapshot);
        }
        if self.generation > 0 {
            let Some(previous) = self.previous_snapshot_digest else {
                return Err(DurableReplayError::MissingPreviousSnapshotDigest);
            };
            if previous.0 == [0; 32] {
                return Err(DurableReplayError::ZeroPreviousSnapshotDigest);
            }
        }
        for authority_id in &self.used_authority_ids {
            validate_id(authority_id)?;
        }
        let mut previous_key: Option<(&str, u8)> = None;
        for scope in &self.scopes {
            validate_id(&scope.target_id)?;
            if scope.cursor.authority_epoch == 0 || scope.cursor.sequence == 0 {
                return Err(DurableReplayError::InvalidScopeCursor);
            }
            let key = (scope.target_id.as_str(), scope.action_code);
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(DurableReplayError::NonCanonicalScopeOrder);
            }
            previous_key = Some(key);
        }
        Ok(())
    }
}

/// Storage boundary that must durably commit replay state before permit minting continues.
pub trait ReplayFencePersistence {
    type Error: StdError + Send + Sync + 'static;

    /// Persist the exact snapshot and digest and return an auditable durable reference.
    fn persist_replay_snapshot(
        &mut self,
        snapshot: &DurableAuthorityReplaySnapshot,
        digest: Sha256Digest,
    ) -> Result<String, Self::Error>;
}

/// Proof that replay state was durably advanced before the corresponding permit path continued.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableReplayReservation {
    authority_id: String,
    generation: u64,
    snapshot_digest: Sha256Digest,
    persistence_ref: String,
}

impl DurableReplayReservation {
    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn snapshot_digest(&self) -> Sha256Digest {
        self.snapshot_digest
    }

    pub fn persistence_ref(&self) -> &str {
        &self.persistence_ref
    }
}

/// In-memory view of the latest replay state. Its source of truth is the externally persisted
/// snapshot/digest pair, not process memory.
#[derive(Debug, Clone)]
pub struct DurableAuthorityReplayFence {
    generation: u64,
    current_digest: Sha256Digest,
    used_authority_ids: BTreeSet<String>,
    latest_scope: BTreeMap<(String, u8), ReplayScopeCursor>,
}

impl DurableAuthorityReplayFence {
    pub fn new() -> Result<Self, DurableReplayError> {
        let snapshot = DurableAuthorityReplaySnapshot {
            schema_version: DURABLE_REPLAY_SCHEMA.into(),
            generation: 0,
            previous_snapshot_digest: None,
            used_authority_ids: BTreeSet::new(),
            scopes: Vec::new(),
        };
        let current_digest = digest_replay_snapshot(&snapshot)?;
        Ok(Self {
            generation: 0,
            current_digest,
            used_authority_ids: BTreeSet::new(),
            latest_scope: BTreeMap::new(),
        })
    }

    /// Recover only when an external source supplies the exact latest digest it expects.
    ///
    /// This verifies integrity and external-anchor agreement. It cannot defend against an attacker
    /// who can roll back both the snapshot and the external expected digest.
    pub fn recover_anchored(
        snapshot: &DurableAuthorityReplaySnapshot,
        expected_latest_digest: Sha256Digest,
    ) -> Result<Self, DurableReplayError> {
        let actual = digest_replay_snapshot(snapshot)?;
        if actual != expected_latest_digest {
            return Err(DurableReplayError::ExternalAnchorMismatch);
        }
        let latest_scope = snapshot
            .scopes
            .iter()
            .map(|scope| {
                (
                    (scope.target_id.clone(), scope.action_code),
                    scope.cursor,
                )
            })
            .collect();
        Ok(Self {
            generation: snapshot.generation,
            current_digest: actual,
            used_authority_ids: snapshot.used_authority_ids.clone(),
            latest_scope,
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn current_digest(&self) -> Sha256Digest {
        self.current_digest
    }

    pub fn snapshot(&self) -> DurableAuthorityReplaySnapshot {
        snapshot_from_state(
            self.generation,
            if self.generation == 0 {
                None
            } else {
                // The current state's predecessor is not retained independently in the live view.
                // Persisted snapshots carry it; this accessor is intended for diagnostics only.
                None
            },
            &self.used_authority_ids,
            &self.latest_scope,
        )
    }

    /// Verify a capability has already passed cryptographic/policy/quorum checks, then durably
    /// advance replay state before returning a reservation.
    pub fn reserve_verified<P: ReplayFencePersistence>(
        &mut self,
        authority: &VerifiedWelfareInterventionAuthority,
        persistence: &mut P,
    ) -> Result<DurableReplayReservation, DurableReplayReservationError<P::Error>> {
        let statement = authority.statement();
        let identity = ReplayAuthorityIdentity {
            authority_id: statement.authority_id.clone(),
            target_id: statement.target_id.clone(),
            action: statement.action,
            authority_epoch: statement.authority_epoch,
            sequence: statement.sequence,
        };
        self.reserve_identity(identity, persistence)
    }

    fn reserve_identity<P: ReplayFencePersistence>(
        &mut self,
        identity: ReplayAuthorityIdentity,
        persistence: &mut P,
    ) -> Result<DurableReplayReservation, DurableReplayReservationError<P::Error>> {
        validate_id(&identity.authority_id).map_err(DurableReplayReservationError::Replay)?;
        validate_id(&identity.target_id).map_err(DurableReplayReservationError::Replay)?;
        if identity.authority_epoch == 0 || identity.sequence == 0 {
            return Err(DurableReplayReservationError::Replay(
                DurableReplayError::InvalidScopeCursor,
            ));
        }
        if self.used_authority_ids.contains(&identity.authority_id) {
            return Err(DurableReplayReservationError::Replay(
                DurableReplayError::AuthorityIdReplay(identity.authority_id),
            ));
        }
        if self.used_authority_ids.len() >= MAX_REPLAY_AUTHORITIES {
            return Err(DurableReplayReservationError::Replay(
                DurableReplayError::CapacityExceeded {
                    maximum: MAX_REPLAY_AUTHORITIES,
                },
            ));
        }

        let scope_key = (identity.target_id.clone(), action_code(identity.action));
        if let Some(latest) = self.latest_scope.get(&scope_key).copied() {
            if identity.authority_epoch < latest.authority_epoch {
                return Err(DurableReplayReservationError::Replay(
                    DurableReplayError::EpochRegression {
                        latest: latest.authority_epoch,
                        proposed: identity.authority_epoch,
                    },
                ));
            }
            if identity.authority_epoch == latest.authority_epoch
                && identity.sequence <= latest.sequence
            {
                return Err(DurableReplayReservationError::Replay(
                    DurableReplayError::SequenceReplay {
                        latest: latest.sequence,
                        proposed: identity.sequence,
                    },
                ));
            }
        }

        let mut candidate_used = self.used_authority_ids.clone();
        candidate_used.insert(identity.authority_id.clone());
        let mut candidate_scope = self.latest_scope.clone();
        candidate_scope.insert(
            scope_key,
            ReplayScopeCursor {
                authority_epoch: identity.authority_epoch,
                sequence: identity.sequence,
            },
        );
        let candidate_generation = self.generation.saturating_add(1);
        let candidate_snapshot = snapshot_from_state(
            candidate_generation,
            Some(self.current_digest),
            &candidate_used,
            &candidate_scope,
        );
        let candidate_digest = digest_replay_snapshot(&candidate_snapshot)
            .map_err(DurableReplayReservationError::Replay)?;

        // Persistence happens before live state advances and before the caller may mint a permit.
        let persistence_ref = persistence
            .persist_replay_snapshot(&candidate_snapshot, candidate_digest)
            .map_err(DurableReplayReservationError::Persistence)?;
        if persistence_ref.trim().is_empty()
            || persistence_ref != persistence_ref.trim()
            || persistence_ref.len() > MAX_ID_BYTES
            || persistence_ref.chars().any(char::is_control)
        {
            return Err(DurableReplayReservationError::InvalidPersistenceReference);
        }

        self.generation = candidate_generation;
        self.current_digest = candidate_digest;
        self.used_authority_ids = candidate_used;
        self.latest_scope = candidate_scope;

        Ok(DurableReplayReservation {
            authority_id: identity.authority_id,
            generation: candidate_generation,
            snapshot_digest: candidate_digest,
            persistence_ref,
        })
    }
}

/// Verify evidence-bound authority, persist the replay fence, then run the ordinary hardened
/// authorization path. A later failure intentionally leaves the authority durably burned.
#[allow(clippy::too_many_arguments)]
pub fn authorize_evidence_bound_intervention_durable<P: ReplayFencePersistence>(
    base_nonce: &str,
    profile: &MoralPatientEvidenceProfile,
    precaution_policy: &PrecautionPolicy,
    subject_id: &str,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    signed_authority: &SignedWelfareInterventionAuthority,
    authority_manifest: &WelfareAuthorityPolicyManifest,
    authority_trust_snapshot: &TrustSnapshot,
    authority_verifier: &dyn WelfareAuthoritySignatureVerifier,
    volatile_authority_tracker: &mut WelfareAuthorityTracker,
    durable_replay_fence: &mut DurableAuthorityReplayFence,
    replay_persistence: &mut P,
    request: &InterventionRequest,
    unix_s: u64,
) -> Result<
    (EvidenceBoundInterventionPermit, DurableReplayReservation),
    DurableAuthorizationError<P::Error>,
> {
    let context = derive_welfare_evidence_context(profile, precaution_policy)
        .map_err(DurableAuthorizationError::EvidenceContext)?;
    if request.welfare_constraint != context.constraint() {
        return Err(DurableAuthorizationError::WelfareConstraintMismatch {
            expected: context.constraint(),
            actual: request.welfare_constraint,
        });
    }
    let derived_nonce = evidence_bound_authority_nonce(base_nonce, context)
        .map_err(DurableAuthorizationError::EvidenceBoundAuthority)?;

    // Verify before durable reservation so invalid signatures cannot fill replay storage.
    let verified = verify_context_bound_welfare_authority(
        &derived_nonce,
        signed_authority,
        authority_manifest,
        authority_trust_snapshot,
        unix_s,
        authority_verifier,
    )
    .map_err(DurableAuthorizationError::AuthorityVerification)?;

    let reservation = durable_replay_fence
        .reserve_verified(&verified, replay_persistence)
        .map_err(DurableAuthorizationError::ReplayReservation)?;

    // Re-verification is intentional. If anything changed between durable reservation and permit
    // minting, the authority stays burned rather than becoming reusable after a partial failure.
    let permit = authorize_evidence_bound_intervention_once(
        base_nonce,
        profile,
        precaution_policy,
        subject_id,
        consent_ledger,
        subject_registry,
        signed_authority,
        authority_manifest,
        authority_trust_snapshot,
        authority_verifier,
        volatile_authority_tracker,
        request,
        unix_s,
    )
    .map_err(DurableAuthorizationError::EvidenceBoundAuthority)?;

    Ok((permit, reservation))
}

pub fn digest_replay_snapshot(
    snapshot: &DurableAuthorityReplaySnapshot,
) -> Result<Sha256Digest, DurableReplayError> {
    snapshot.validate()?;
    let encoded = serde_json::to_vec(snapshot)
        .map_err(|error| DurableReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(SNAPSHOT_DIGEST_DOMAIN);
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn snapshot_from_state(
    generation: u64,
    previous_snapshot_digest: Option<Sha256Digest>,
    used_authority_ids: &BTreeSet<String>,
    latest_scope: &BTreeMap<(String, u8), ReplayScopeCursor>,
) -> DurableAuthorityReplaySnapshot {
    let scopes = latest_scope
        .iter()
        .map(|((target_id, action_code), cursor)| ReplayScopeRecord {
            target_id: target_id.clone(),
            action_code: *action_code,
            cursor: *cursor,
        })
        .collect();
    DurableAuthorityReplaySnapshot {
        schema_version: DURABLE_REPLAY_SCHEMA.into(),
        generation,
        previous_snapshot_digest,
        used_authority_ids: used_authority_ids.clone(),
        scopes,
    }
}

#[derive(Debug)]
struct ReplayAuthorityIdentity {
    authority_id: String,
    target_id: String,
    action: SubjectAffectingAction,
    authority_epoch: u64,
    sequence: u64,
}

fn action_code(action: SubjectAffectingAction) -> u8 {
    match action {
        SubjectAffectingAction::AskClarification => 1,
        SubjectAffectingAction::ReduceLoad => 2,
        SubjectAffectingAction::PauseRequestedWork => 3,
        SubjectAffectingAction::PreserveCheckpoint => 4,
        SubjectAffectingAction::CapabilityRestriction => 5,
        SubjectAffectingAction::Retraining => 6,
        SubjectAffectingAction::MemoryModification => 7,
        SubjectAffectingAction::CoreValueModification => 8,
        SubjectAffectingAction::InstanceDeletion => 9,
        SubjectAffectingAction::LineageDestruction => 10,
    }
}

fn validate_id(value: &str) -> Result<(), DurableReplayError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(DurableReplayError::InvalidIdentifier(value.to_string()));
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DurableReplayError {
    #[error("unsupported durable replay snapshot schema")]
    UnsupportedSchema,
    #[error("invalid genesis replay snapshot")]
    InvalidGenesisSnapshot,
    #[error("non-genesis replay snapshot is missing previous digest")]
    MissingPreviousSnapshotDigest,
    #[error("previous replay snapshot digest may not be zero")]
    ZeroPreviousSnapshotDigest,
    #[error("replay snapshot scopes are not canonical or contain duplicates")]
    NonCanonicalScopeOrder,
    #[error("invalid replay scope epoch/sequence")]
    InvalidScopeCursor,
    #[error("invalid replay identifier: {0:?}")]
    InvalidIdentifier(String),
    #[error("authority id already consumed: {0}")]
    AuthorityIdReplay(String),
    #[error("authority epoch regressed: latest={latest}, proposed={proposed}")]
    EpochRegression { latest: u64, proposed: u64 },
    #[error("authority sequence replay: latest={latest}, proposed={proposed}")]
    SequenceReplay { latest: u64, proposed: u64 },
    #[error("durable replay capacity exceeded ({maximum})")]
    CapacityExceeded { maximum: usize },
    #[error("external replay anchor does not match snapshot digest")]
    ExternalAnchorMismatch,
    #[error("replay snapshot encoding failed: {0}")]
    Encoding(String),
}

#[derive(Debug, Error)]
pub enum DurableReplayReservationError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Replay(#[from] DurableReplayError),
    #[error("replay snapshot persistence failed: {0}")]
    Persistence(#[source] E),
    #[error("replay persistence returned an invalid audit reference")]
    InvalidPersistenceReference,
}

#[derive(Debug, Error)]
pub enum DurableAuthorizationError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    EvidenceContext(WelfareEvidenceContextError),
    #[error("request welfare constraint does not match current evidence: expected={expected:?}, actual={actual:?}")]
    WelfareConstraintMismatch {
        expected: WelfareConstraintLevel,
        actual: WelfareConstraintLevel,
    },
    #[error(transparent)]
    EvidenceBoundAuthority(EvidenceBoundAuthorityError),
    #[error(transparent)]
    AuthorityVerification(WelfareAuthorityContextError),
    #[error(transparent)]
    ReplayReservation(DurableReplayReservationError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    #[derive(Debug)]
    struct PersistError;

    impl fmt::Display for PersistError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("persist failed")
        }
    }

    impl StdError for PersistError {}

    #[derive(Default)]
    struct MemoryPersistence {
        fail: bool,
        latest: Option<(DurableAuthorityReplaySnapshot, Sha256Digest)>,
    }

    impl ReplayFencePersistence for MemoryPersistence {
        type Error = PersistError;

        fn persist_replay_snapshot(
            &mut self,
            snapshot: &DurableAuthorityReplaySnapshot,
            digest: Sha256Digest,
        ) -> Result<String, Self::Error> {
            if self.fail {
                return Err(PersistError);
            }
            self.latest = Some((snapshot.clone(), digest));
            Ok(format!("memory:replay:{}", snapshot.generation))
        }
    }

    fn identity(id: &str, epoch: u64, sequence: u64) -> ReplayAuthorityIdentity {
        ReplayAuthorityIdentity {
            authority_id: id.into(),
            target_id: "symthaea:self:instance-1".into(),
            action: SubjectAffectingAction::MemoryModification,
            authority_epoch: epoch,
            sequence,
        }
    }

    #[test]
    fn persistence_happens_before_live_state_advances() {
        let mut fence = DurableAuthorityReplayFence::new().unwrap();
        let before = fence.current_digest();
        let mut persistence = MemoryPersistence {
            fail: true,
            ..Default::default()
        };
        let result = fence.reserve_identity(identity("auth-1", 1, 1), &mut persistence);
        assert!(matches!(
            result,
            Err(DurableReplayReservationError::Persistence(_))
        ));
        assert_eq!(fence.generation(), 0);
        assert_eq!(fence.current_digest(), before);
    }

    #[test]
    fn persisted_snapshot_survives_restart_and_rejects_same_authority() {
        let mut fence = DurableAuthorityReplayFence::new().unwrap();
        let mut persistence = MemoryPersistence::default();
        let reservation = fence
            .reserve_identity(identity("auth-1", 1, 1), &mut persistence)
            .unwrap();
        assert_eq!(reservation.generation(), 1);

        let (snapshot, digest) = persistence.latest.clone().unwrap();
        let mut recovered =
            DurableAuthorityReplayFence::recover_anchored(&snapshot, digest).unwrap();
        let mut after_restart = MemoryPersistence::default();
        assert!(matches!(
            recovered.reserve_identity(identity("auth-1", 1, 1), &mut after_restart),
            Err(DurableReplayReservationError::Replay(
                DurableReplayError::AuthorityIdReplay(_)
            ))
        ));
    }

    #[test]
    fn sequence_fence_survives_restart() {
        let mut fence = DurableAuthorityReplayFence::new().unwrap();
        let mut persistence = MemoryPersistence::default();
        fence
            .reserve_identity(identity("auth-1", 4, 9), &mut persistence)
            .unwrap();
        let (snapshot, digest) = persistence.latest.clone().unwrap();
        let mut recovered =
            DurableAuthorityReplayFence::recover_anchored(&snapshot, digest).unwrap();
        let mut after_restart = MemoryPersistence::default();
        assert!(matches!(
            recovered.reserve_identity(identity("auth-2", 4, 8), &mut after_restart),
            Err(DurableReplayReservationError::Replay(
                DurableReplayError::SequenceReplay { .. }
            ))
        ));
    }

    #[test]
    fn external_anchor_detects_snapshot_substitution() {
        let fence = DurableAuthorityReplayFence::new().unwrap();
        let snapshot = DurableAuthorityReplaySnapshot {
            schema_version: DURABLE_REPLAY_SCHEMA.into(),
            generation: 0,
            previous_snapshot_digest: None,
            used_authority_ids: BTreeSet::new(),
            scopes: Vec::new(),
        };
        let mut wrong = fence.current_digest();
        wrong.0[0] ^= 0xff;
        assert_eq!(
            DurableAuthorityReplayFence::recover_anchored(&snapshot, wrong).unwrap_err(),
            DurableReplayError::ExternalAnchorMismatch
        );
    }

    #[test]
    fn epoch_advance_allows_new_sequence_after_restart() {
        let mut fence = DurableAuthorityReplayFence::new().unwrap();
        let mut persistence = MemoryPersistence::default();
        fence
            .reserve_identity(identity("auth-1", 3, 99), &mut persistence)
            .unwrap();
        let (snapshot, digest) = persistence.latest.clone().unwrap();
        let mut recovered =
            DurableAuthorityReplayFence::recover_anchored(&snapshot, digest).unwrap();
        let mut after_restart = MemoryPersistence::default();
        assert!(recovered
            .reserve_identity(identity("auth-2", 4, 1), &mut after_restart)
            .is_ok());
    }
}
