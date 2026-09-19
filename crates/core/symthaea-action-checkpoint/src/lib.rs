// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical grant-bound checkpoints for Symthaea action authority v0.2.
//!
//! This crate binds ordinary [`GrantAccountSnapshotV2`] persistence data to an
//! exact authority-v0.2 grant and an explicit predecessor checkpoint identity.
//! It deliberately does not provide persistence, currentness, a trusted store,
//! CAS/consensus, dispatch authority, or effect execution.
//!
//! Core separation:
//!
//! ```text
//! checkpoint bytes / JSON / serde representation
//!     != canonical checkpoint identity
//!     != trusted current head
//!     != durable/linearizable persistence
//!     != dispatch authority
//! ```
//!
//! A supplied checkpoint chain may be internally self-consistent while still
//! being stale. Rollback resistance exists only relative to an independently
//! trusted expected head supplied by a stronger layer.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use symthaea_action_runtime::{
    GrantAccountSnapshotV2, GrantAccountV2, ReservationState, RuntimeV2Error,
};
use symthaea_authority::{CapabilityGrant, Digest32, RiskBudget};
use thiserror::Error;

/// Schema version for the canonical checkpoint transcript.
pub const ACTION_CHECKPOINT_SCHEMA_VERSION: u16 = 2;
/// Domain separator for language-neutral checkpoint identity.
pub const ACTION_CHECKPOINT_DOMAIN: &[u8] = b"symthaea.action-checkpoint.v2\0";

/// Externally retainable checkpoint head identity.
///
/// This value is ordinary data. Its presence does not prove that the head is
/// current or that the storage provider is trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHeadV2 {
    pub sequence: u64,
    pub digest: Digest32,
}

/// Persistable checkpoint generation.
///
/// `Serialize` / `Deserialize` are storage conveniences only. The authority
/// identity is produced by [`GrantAccountCheckpointV2::digest`] from the frozen
/// canonical transcript below, not from any serializer output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccountCheckpointV2 {
    pub schema_version: u16,
    pub sequence: u64,
    pub previous_checkpoint_digest: Option<Digest32>,
    pub grant_digest: Digest32,
    pub snapshot: GrantAccountSnapshotV2,
}

impl GrantAccountCheckpointV2 {
    /// Construct generation zero from a currently valid deterministic account.
    ///
    /// Construction rebinds the snapshot to the externally supplied exact grant;
    /// the account cannot self-describe a broader ceiling.
    pub fn first(
        grant: &CapabilityGrant,
        account: &GrantAccountV2,
    ) -> Result<Self, CheckpointV2Error> {
        let snapshot = account.snapshot();
        GrantAccountV2::restore(grant, snapshot.clone())?;
        Ok(Self {
            schema_version: ACTION_CHECKPOINT_SCHEMA_VERSION,
            sequence: 0,
            previous_checkpoint_digest: None,
            grant_digest: grant.digest(),
            snapshot,
        })
    }

    /// Construct the exact successor of the supplied checkpoint.
    ///
    /// This establishes a hash-chain edge only. It does not prove that
    /// `previous` is the externally trusted current head.
    pub fn successor(
        previous: &Self,
        grant: &CapabilityGrant,
        account: &GrantAccountV2,
    ) -> Result<Self, CheckpointV2Error> {
        previous.verify_payload(grant)?;
        let snapshot = account.snapshot();
        GrantAccountV2::restore(grant, snapshot.clone())?;
        let sequence = previous
            .sequence
            .checked_add(1)
            .ok_or(CheckpointV2Error::SequenceOverflow)?;
        Ok(Self {
            schema_version: ACTION_CHECKPOINT_SCHEMA_VERSION,
            sequence,
            previous_checkpoint_digest: Some(previous.digest()?),
            grant_digest: grant.digest(),
            snapshot,
        })
    }

    /// Language-neutral SHA-256 identity over every checkpoint-semantic field.
    ///
    /// Canonical transcript, in order:
    ///
    /// ```text
    /// domain bytes
    /// checkpoint schema: u16 big-endian
    /// sequence:          u64 big-endian
    /// predecessor:       u8 tag (0/1), then 32 bytes when present
    /// grant digest:      32 bytes
    /// snapshot schema:   u16 big-endian
    /// snapshot grant:    32 bytes
    /// max uses:          u32 big-endian
    /// account risk:      4 * u64 big-endian
    /// reservation count: u32 big-endian
    /// reservations in BTreeMap key order:
    ///   map key                 32 bytes
    ///   embedded reservation id 32 bytes
    ///   effect intent id         32 bytes
    ///   attempt id               32 bytes
    ///   effect binding digest    32 bytes
    ///   risk charge              4 * u64 big-endian
    ///   state code               u8
    /// ```
    ///
    /// Reservation state codes are frozen as:
    /// `Reserved=0`, `OutcomeUnknown=1`, `Committed=2`, `Released=3`.
    pub fn digest(&self) -> Result<Digest32, CheckpointV2Error> {
        let reservation_count = u32::try_from(self.snapshot.reservations.len())
            .map_err(|_| CheckpointV2Error::CanonicalLengthOverflow)?;

        let mut transcript = CanonicalTranscript::new();
        transcript.u16(self.schema_version);
        transcript.u64(self.sequence);
        transcript.optional_digest(self.previous_checkpoint_digest);
        transcript.digest(self.grant_digest);
        transcript.u16(self.snapshot.schema_version);
        transcript.digest(self.snapshot.grant_digest);
        transcript.u32(self.snapshot.max_uses);
        transcript.risk(self.snapshot.risk_budget);
        transcript.u32(reservation_count);

        for (map_key, reservation) in &self.snapshot.reservations {
            transcript.digest(map_key.0);
            transcript.digest(reservation.reservation_id.0);
            transcript.digest(reservation.effect_intent_id.0);
            transcript.digest(reservation.attempt_id.0);
            transcript.digest(reservation.effect_binding_digest.0);
            transcript.risk(reservation.risk_charge);
            transcript.byte(reservation_state_code(reservation.state));
        }

        Ok(transcript.finish())
    }

    pub fn head(&self) -> Result<CheckpointHeadV2, CheckpointV2Error> {
        Ok(CheckpointHeadV2 {
            sequence: self.sequence,
            digest: self.digest()?,
        })
    }

    /// Rebind this checkpoint's payload to the externally supplied exact grant.
    ///
    /// Success means structurally valid deterministic accounting for that grant.
    /// It does not mean this checkpoint is current.
    pub fn verify_payload(
        &self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        if self.schema_version != ACTION_CHECKPOINT_SCHEMA_VERSION {
            return Err(CheckpointV2Error::UnsupportedSchema);
        }
        let exact_grant_digest = grant.digest();
        if self.grant_digest != exact_grant_digest
            || self.snapshot.grant_digest != exact_grant_digest
        {
            return Err(CheckpointV2Error::GrantDigestMismatch);
        }
        GrantAccountV2::restore(grant, self.snapshot.clone()).map_err(CheckpointV2Error::Runtime)
    }

    /// Verify the checkpoint relative to one externally supplied expected prior
    /// head.
    ///
    /// `None` means the caller expects generation zero. `Some(head)` means this
    /// checkpoint must be the exact sequence+digest successor of that head.
    ///
    /// The function does not authenticate `expected_previous`; a stronger store
    /// or anti-rollback layer must establish why that head is trusted/current.
    pub fn verify_against_expected_head(
        &self,
        grant: &CapabilityGrant,
        expected_previous: Option<CheckpointHeadV2>,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        let account = self.verify_payload(grant)?;
        match expected_previous {
            None => {
                if self.sequence != 0 || self.previous_checkpoint_digest.is_some() {
                    return Err(CheckpointV2Error::UnexpectedGenesisShape);
                }
            }
            Some(previous) => {
                let expected_sequence = previous
                    .sequence
                    .checked_add(1)
                    .ok_or(CheckpointV2Error::SequenceOverflow)?;
                if self.sequence != expected_sequence {
                    return Err(CheckpointV2Error::SequenceMismatch);
                }
                if self.previous_checkpoint_digest != Some(previous.digest) {
                    return Err(CheckpointV2Error::PreviousDigestMismatch);
                }
            }
        }
        Ok(account)
    }
}

/// Verify only the internal consistency of a caller-supplied chain beginning at
/// generation zero.
///
/// This is deliberately named `verify_supplied_chain`: success does not prove
/// currentness or rollback resistance because a stale historical chain can also
/// be internally valid. Callers needing currentness must compare against an
/// independently trusted current head.
pub fn verify_supplied_chain(
    grant: &CapabilityGrant,
    checkpoints: &[GrantAccountCheckpointV2],
) -> Result<(GrantAccountV2, CheckpointHeadV2), CheckpointV2Error> {
    let first = checkpoints.first().ok_or(CheckpointV2Error::EmptyChain)?;
    let mut account = first.verify_against_expected_head(grant, None)?;
    let mut head = first.head()?;

    for checkpoint in &checkpoints[1..] {
        account = checkpoint.verify_against_expected_head(grant, Some(head))?;
        head = checkpoint.head()?;
    }

    Ok((account, head))
}

fn reservation_state_code(state: ReservationState) -> u8 {
    match state {
        ReservationState::Reserved => 0,
        ReservationState::OutcomeUnknown => 1,
        ReservationState::Committed => 2,
        ReservationState::Released => 3,
    }
}

struct CanonicalTranscript(Sha256);

impl CanonicalTranscript {
    fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(ACTION_CHECKPOINT_DOMAIN);
        Self(hasher)
    }

    fn byte(&mut self, value: u8) {
        self.0.update([value]);
    }

    fn u16(&mut self, value: u16) {
        self.0.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.update(value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_be_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.0.update(value.0);
    }

    fn optional_digest(&mut self, value: Option<Digest32>) {
        match value {
            Some(value) => {
                self.byte(1);
                self.digest(value);
            }
            None => self.byte(0),
        }
    }

    fn risk(&mut self, value: RiskBudget) {
        self.u64(value.mutation_units);
        self.u64(value.irreversible_units);
        self.u64(value.external_disclosure_bytes);
        self.u64(value.monetary_microunits);
    }

    fn finish(self) -> Digest32 {
        let output = self.0.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&output);
        Digest32(bytes)
    }
}

#[derive(Debug, Error)]
pub enum CheckpointV2Error {
    #[error("unsupported action checkpoint schema")]
    UnsupportedSchema,
    #[error("checkpoint or snapshot does not bind the supplied exact grant")]
    GrantDigestMismatch,
    #[error("checkpoint sequence overflow")]
    SequenceOverflow,
    #[error("reservation collection exceeds canonical u32 length")]
    CanonicalLengthOverflow,
    #[error("expected generation zero but checkpoint declares a predecessor or nonzero sequence")]
    UnexpectedGenesisShape,
    #[error("checkpoint sequence does not follow the expected prior head")]
    SequenceMismatch,
    #[error("checkpoint predecessor digest does not match the expected prior head")]
    PreviousDigestMismatch,
    #[error("supplied checkpoint chain is empty")]
    EmptyChain,
    #[error("runtime snapshot validation failed: {0}")]
    Runtime(#[from] RuntimeV2Error),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use symthaea_action_runtime::{
        AttemptId, EffectBindingDigest, EffectIntentId, ExecutionReservationV2, ReservationId,
        ACTION_RUNTIME_SCHEMA_VERSION,
    };
    use symthaea_authority::{
        AuthorityContextRef, AuthorityEpoch, Operation, PrincipalId, PurposeId, ResourceRef,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(mutation: u64, irreversible: u64, disclosure: u64, monetary: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: mutation,
            irreversible_units: irreversible,
            external_disclosure_bytes: disclosure,
            monetary_microunits: monetary,
        }
    }

    fn grant(id: &str) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            id,
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            PurposeId("test-effect".into()),
            AuthorityEpoch(7),
            AuthorityContextRef::new("symthaea.test.context.v1", digest(1)),
        );
        grant.resources.insert(ResourceRef("resource-1".into()));
        grant.operations.insert(Operation("operate".into()));
        grant.max_uses = 3;
        grant.risk_budget = risk(10, 2, 300, 4_000);
        grant
    }

    fn golden_fixture() -> GrantAccountCheckpointV2 {
        let first = ExecutionReservationV2 {
            reservation_id: ReservationId(digest(0x33)),
            effect_intent_id: EffectIntentId(digest(0x44)),
            attempt_id: AttemptId(digest(0x55)),
            effect_binding_digest: EffectBindingDigest(digest(0x66)),
            risk_charge: risk(1, 0, 2, 3),
            state: ReservationState::Reserved,
        };
        let second = ExecutionReservationV2 {
            reservation_id: ReservationId(digest(0x77)),
            effect_intent_id: EffectIntentId(digest(0x88)),
            attempt_id: AttemptId(digest(0x99)),
            effect_binding_digest: EffectBindingDigest(digest(0xaa)),
            risk_charge: risk(4, 1, 5, 6),
            state: ReservationState::OutcomeUnknown,
        };
        let mut reservations = BTreeMap::new();
        reservations.insert(first.reservation_id, first);
        reservations.insert(second.reservation_id, second);

        GrantAccountCheckpointV2 {
            schema_version: ACTION_CHECKPOINT_SCHEMA_VERSION,
            sequence: 7,
            previous_checkpoint_digest: Some(digest(0x11)),
            grant_digest: digest(0x22),
            snapshot: GrantAccountSnapshotV2 {
                schema_version: ACTION_RUNTIME_SCHEMA_VERSION,
                grant_digest: digest(0x22),
                max_uses: 5,
                risk_budget: risk(10, 2, 300, 4_000),
                reservations,
            },
        }
    }

    fn hex_digest(value: Digest32) -> String {
        value.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn canonical_checkpoint_vector_matches_independent_oracle() {
        assert_eq!(
            hex_digest(golden_fixture().digest().unwrap()),
            "3a756277007027ff06ccaf77c521220db8dd4ce4898c1e8922aa0000fc979efc"
        );
    }

    #[test]
    fn checkpoint_identity_is_independent_of_map_insertion_order() {
        let canonical = golden_fixture();
        let mut reversed = canonical.clone();
        let entries: Vec<_> = reversed
            .snapshot
            .reservations
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect();
        reversed.snapshot.reservations.clear();
        for (key, value) in entries.into_iter().rev() {
            reversed.snapshot.reservations.insert(key, value);
        }
        assert_eq!(canonical.digest().unwrap(), reversed.digest().unwrap());
    }

    #[test]
    fn semantic_mutations_change_checkpoint_identity() {
        let original = golden_fixture();

        let mut changed_sequence = original.clone();
        changed_sequence.sequence += 1;
        assert_ne!(original.digest().unwrap(), changed_sequence.digest().unwrap());

        let mut changed_state = original.clone();
        changed_state
            .snapshot
            .reservations
            .get_mut(&ReservationId(digest(0x33)))
            .unwrap()
            .state = ReservationState::Released;
        assert_ne!(original.digest().unwrap(), changed_state.digest().unwrap());
    }

    #[test]
    fn first_checkpoint_rebinds_snapshot_to_external_grant() {
        let grant = grant("grant-a");
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let checkpoint = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        assert_eq!(checkpoint.sequence, 0);
        assert!(checkpoint.previous_checkpoint_digest.is_none());
        assert!(checkpoint
            .verify_against_expected_head(&grant, None)
            .is_ok());
    }

    #[test]
    fn foreign_grant_cannot_restore_checkpoint() {
        let grant_a = grant("grant-a");
        let grant_b = grant("grant-b");
        let account = GrantAccountV2::new_root(&grant_a).unwrap();
        let checkpoint = GrantAccountCheckpointV2::first(&grant_a, &account).unwrap();
        assert!(matches!(
            checkpoint.verify_payload(&grant_b),
            Err(CheckpointV2Error::GrantDigestMismatch)
        ));
    }

    #[test]
    fn snapshot_cannot_rewrite_external_grant_ceiling() {
        let grant = grant("grant-a");
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let mut checkpoint = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        checkpoint.snapshot.max_uses += 1;
        assert!(matches!(
            checkpoint.verify_payload(&grant),
            Err(CheckpointV2Error::Runtime(RuntimeV2Error::GrantCeilingMismatch))
        ));
    }

    #[test]
    fn successor_binds_exact_expected_prior_head() {
        let grant = grant("grant-a");
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let first = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let first_head = first.head().unwrap();

        account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1, 0, 0, 0),
            )
            .unwrap();
        let second = GrantAccountCheckpointV2::successor(&first, &grant, &account).unwrap();
        assert!(second
            .verify_against_expected_head(&grant, Some(first_head))
            .is_ok());

        let wrong_head = CheckpointHeadV2 {
            sequence: first_head.sequence,
            digest: digest(0xee),
        };
        assert!(matches!(
            second.verify_against_expected_head(&grant, Some(wrong_head)),
            Err(CheckpointV2Error::PreviousDigestMismatch)
        ));
    }

    #[test]
    fn old_checkpoint_cannot_pose_as_successor_of_newer_head() {
        let grant = grant("grant-a");
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let first = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let first_head = first.head().unwrap();
        assert!(matches!(
            first.verify_against_expected_head(&grant, Some(first_head)),
            Err(CheckpointV2Error::SequenceMismatch)
        ));
    }

    #[test]
    fn supplied_chain_verification_is_structural_not_currentness() {
        let grant = grant("grant-a");
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let first = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1, 0, 0, 0),
            )
            .unwrap();
        let second = GrantAccountCheckpointV2::successor(&first, &grant, &account).unwrap();
        account.mark_outcome_unknown(reservation).unwrap();
        let third = GrantAccountCheckpointV2::successor(&second, &grant, &account).unwrap();

        let (_, final_head) = verify_supplied_chain(&grant, &[first, second, third.clone()]).unwrap();
        assert_eq!(final_head, third.head().unwrap());
    }
}
