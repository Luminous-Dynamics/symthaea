// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict public facade for action-runtime v0.2 checkpoints.
//!
//! The internal `identity` module owns the frozen canonical transcript and
//! independent golden vector. This facade adds the missing state-transition
//! theorem so a hash-linked sequence of individually valid snapshots cannot
//! silently delete, rewrite, or move historical reservations backward.

#![deny(unsafe_code)]

#[path = "lib.rs"]
mod identity;

use serde::{Deserialize, Serialize};
use symthaea_action_runtime::{
    ExecutionReservationV2, GrantAccountSnapshotV2, GrantAccountV2, ReservationState,
    RuntimeV2Error,
};
use symthaea_authority::{CapabilityGrant, Digest32};
use thiserror::Error;

pub use identity::{ACTION_CHECKPOINT_DOMAIN, ACTION_CHECKPOINT_SCHEMA_VERSION, CheckpointHeadV2};

/// Public checkpoint type with strict generation/transition semantics.
///
/// Serialization remains storage-only. The inner identity object computes the
/// language-neutral checkpoint digest; this wrapper prevents weaker successor
/// construction/verification APIs from becoming part of the public surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GrantAccountCheckpointV2 {
    inner: identity::GrantAccountCheckpointV2,
}

impl GrantAccountCheckpointV2 {
    /// Establish exact generation zero.
    ///
    /// Generation zero must contain no historical reservations. This freezes a
    /// pre-use frontier so later reservation authority cannot be backdated into
    /// the genesis checkpoint.
    pub fn first(
        grant: &CapabilityGrant,
        account: &GrantAccountV2,
    ) -> Result<Self, CheckpointV2Error> {
        let snapshot = account.snapshot();
        if !snapshot.reservations.is_empty() {
            return Err(CheckpointV2Error::NonEmptyGenesis);
        }
        let inner = identity::GrantAccountCheckpointV2::first(grant, account)
            .map_err(identity_error)?;
        Ok(Self { inner })
    }

    /// Construct the exact semantic successor of `previous`.
    ///
    /// The previous and next snapshots must both be valid under the exact grant,
    /// and the transition must preserve every historical reservation while
    /// allowing only the runtime state machine's forward edges.
    pub fn successor(
        previous: &Self,
        grant: &CapabilityGrant,
        account: &GrantAccountV2,
    ) -> Result<Self, CheckpointV2Error> {
        previous.verify_payload(grant)?;
        let next_snapshot = account.snapshot();
        GrantAccountV2::restore(grant, next_snapshot.clone())?;
        validate_snapshot_transition(&previous.inner.snapshot, &next_snapshot)?;
        let inner = identity::GrantAccountCheckpointV2::successor(
            &previous.inner,
            grant,
            account,
        )
        .map_err(identity_error)?;
        Ok(Self { inner })
    }

    pub fn digest(&self) -> Result<Digest32, CheckpointV2Error> {
        self.inner.digest().map_err(identity_error)
    }

    pub fn head(&self) -> Result<CheckpointHeadV2, CheckpointV2Error> {
        self.inner.head().map_err(identity_error)
    }

    pub fn sequence(&self) -> u64 {
        self.inner.sequence
    }

    pub fn previous_checkpoint_digest(&self) -> Option<Digest32> {
        self.inner.previous_checkpoint_digest
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.inner.grant_digest
    }

    pub fn snapshot(&self) -> &GrantAccountSnapshotV2 {
        &self.inner.snapshot
    }

    /// Rebind the checkpoint payload to the exact externally supplied grant.
    ///
    /// Success is pointwise structural validity only, not currentness.
    pub fn verify_payload(
        &self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        self.inner.verify_payload(grant).map_err(identity_error)
    }

    /// Verify exact generation-zero shape and payload.
    pub fn verify_genesis(
        &self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        if !self.inner.snapshot.reservations.is_empty() {
            return Err(CheckpointV2Error::NonEmptyGenesis);
        }
        self.inner
            .verify_against_expected_head(grant, None)
            .map_err(identity_error)
    }

    /// Verify sequence/digest shape relative to an externally supplied expected
    /// prior head.
    ///
    /// A head alone cannot prove a legal semantic state transition because it
    /// does not contain the prior snapshot. Use [`Self::verify_successor_of`]
    /// whenever the full previous checkpoint is available.
    pub fn verify_shape_against_expected_head(
        &self,
        grant: &CapabilityGrant,
        expected_previous: Option<CheckpointHeadV2>,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        if expected_previous.is_none() {
            return self.verify_genesis(grant);
        }
        self.inner
            .verify_against_expected_head(grant, expected_previous)
            .map_err(identity_error)
    }

    /// Verify that this checkpoint is the exact legal semantic successor of the
    /// supplied full previous checkpoint.
    pub fn verify_successor_of(
        &self,
        previous: &Self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        previous.verify_payload(grant)?;
        let previous_head = previous.head()?;
        let account = self
            .inner
            .verify_against_expected_head(grant, Some(previous_head))
            .map_err(identity_error)?;
        validate_snapshot_transition(&previous.inner.snapshot, &self.inner.snapshot)?;
        Ok(account)
    }
}

/// Verify a caller-supplied chain from an empty generation-zero checkpoint.
///
/// Every edge is checked for exact predecessor identity and legal account-state
/// transition. This still does not prove that the supplied final head is the
/// externally current head; V2C must provide that stronger store/CAS theorem.
pub fn verify_supplied_chain(
    grant: &CapabilityGrant,
    checkpoints: &[GrantAccountCheckpointV2],
) -> Result<(GrantAccountV2, CheckpointHeadV2), CheckpointV2Error> {
    let first = checkpoints.first().ok_or(CheckpointV2Error::EmptyChain)?;
    let mut account = first.verify_genesis(grant)?;
    let mut head = first.head()?;

    for pair in checkpoints.windows(2) {
        account = pair[1].verify_successor_of(&pair[0], grant)?;
        head = pair[1].head()?;
    }

    Ok((account, head))
}

fn validate_snapshot_transition(
    previous: &GrantAccountSnapshotV2,
    next: &GrantAccountSnapshotV2,
) -> Result<(), CheckpointV2Error> {
    for (reservation_id, old) in &previous.reservations {
        let Some(new) = next.reservations.get(reservation_id) else {
            return Err(CheckpointV2Error::ReservationRemoved);
        };
        if !same_reservation_identity(old, new) {
            return Err(CheckpointV2Error::ReservationIdentityChanged);
        }
        if !allowed_state_transition(old.state, new.state) {
            return Err(CheckpointV2Error::InvalidReservationStateTransition {
                from: old.state,
                to: new.state,
            });
        }
    }

    for (reservation_id, new) in &next.reservations {
        if !previous.reservations.contains_key(reservation_id)
            && new.state != ReservationState::Reserved
        {
            return Err(CheckpointV2Error::NewReservationNotReserved);
        }
    }

    Ok(())
}

fn same_reservation_identity(
    old: &ExecutionReservationV2,
    new: &ExecutionReservationV2,
) -> bool {
    old.reservation_id == new.reservation_id
        && old.effect_intent_id == new.effect_intent_id
        && old.attempt_id == new.attempt_id
        && old.effect_binding_digest == new.effect_binding_digest
        && old.risk_charge == new.risk_charge
}

fn allowed_state_transition(from: ReservationState, to: ReservationState) -> bool {
    match from {
        ReservationState::Reserved => matches!(
            to,
            ReservationState::Reserved
                | ReservationState::OutcomeUnknown
                | ReservationState::Released
        ),
        ReservationState::OutcomeUnknown => matches!(
            to,
            ReservationState::OutcomeUnknown
                | ReservationState::Committed
                | ReservationState::Released
        ),
        ReservationState::Committed => to == ReservationState::Committed,
        ReservationState::Released => to == ReservationState::Released,
    }
}

fn identity_error(error: identity::CheckpointV2Error) -> CheckpointV2Error {
    CheckpointV2Error::Identity(error.to_string())
}

#[derive(Debug, Error)]
pub enum CheckpointV2Error {
    #[error("canonical checkpoint identity/payload verification failed: {0}")]
    Identity(String),
    #[error("runtime snapshot validation failed: {0}")]
    Runtime(#[from] RuntimeV2Error),
    #[error("generation-zero checkpoint must contain no reservations")]
    NonEmptyGenesis,
    #[error("historical reservation disappeared from successor snapshot")]
    ReservationRemoved,
    #[error("historical reservation identity/risk binding changed in successor snapshot")]
    ReservationIdentityChanged,
    #[error("new reservation must first appear in Reserved state")]
    NewReservationNotReserved,
    #[error("invalid reservation state transition {from:?} -> {to:?}")]
    InvalidReservationStateTransition {
        from: ReservationState,
        to: ReservationState,
    },
    #[error("supplied checkpoint chain is empty")]
    EmptyChain,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_action_runtime::{AttemptId, EffectBindingDigest, EffectIntentId};
    use symthaea_authority::{
        AuthorityContextRef, AuthorityEpoch, Operation, PrincipalId, PurposeId, ResourceRef,
        RiskBudget,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-transition-v2",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            PurposeId("test-effect".into()),
            AuthorityEpoch(9),
            AuthorityContextRef::new("symthaea.test.context.v1", digest(1)),
        );
        grant.resources.insert(ResourceRef("resource-1".into()));
        grant.operations.insert(Operation("operate".into()));
        grant.max_uses = 3;
        grant.risk_budget = risk(3);
        grant
    }

    #[test]
    fn generation_zero_must_precede_all_reservations() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        assert!(matches!(
            GrantAccountCheckpointV2::first(&grant, &account),
            Err(CheckpointV2Error::NonEmptyGenesis)
        ));
    }

    #[test]
    fn new_reservation_must_be_persisted_reserved_before_outcome_unknown() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let genesis = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        account.mark_outcome_unknown(reservation).unwrap();
        assert!(matches!(
            GrantAccountCheckpointV2::successor(&genesis, &grant, &account),
            Err(CheckpointV2Error::NewReservationNotReserved)
        ));
    }

    #[test]
    fn historical_reservation_cannot_disappear() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let genesis = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let reserved = GrantAccountCheckpointV2::successor(&genesis, &grant, &account).unwrap();

        let mut tampered = reserved.clone();
        tampered.inner.sequence += 1;
        tampered.inner.previous_checkpoint_digest = Some(reserved.digest().unwrap());
        tampered.inner.snapshot.reservations.clear();

        assert!(matches!(
            tampered.verify_successor_of(&reserved, &grant),
            Err(CheckpointV2Error::ReservationRemoved)
        ));
    }

    #[test]
    fn committed_reservation_cannot_move_backward() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let genesis = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let reserved = GrantAccountCheckpointV2::successor(&genesis, &grant, &account).unwrap();
        account.mark_outcome_unknown(reservation).unwrap();
        let unknown = GrantAccountCheckpointV2::successor(&reserved, &grant, &account).unwrap();
        account.reconcile_applied(reservation).unwrap();
        let committed = GrantAccountCheckpointV2::successor(&unknown, &grant, &account).unwrap();

        let mut backward = committed.clone();
        backward.inner.sequence += 1;
        backward.inner.previous_checkpoint_digest = Some(committed.digest().unwrap());
        backward
            .inner
            .snapshot
            .reservations
            .get_mut(&reservation)
            .unwrap()
            .state = ReservationState::OutcomeUnknown;

        assert!(matches!(
            backward.verify_successor_of(&committed, &grant),
            Err(CheckpointV2Error::InvalidReservationStateTransition {
                from: ReservationState::Committed,
                to: ReservationState::OutcomeUnknown,
            })
        ));
    }

    #[test]
    fn strict_supplied_chain_preserves_runtime_state_machine() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let genesis = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let reserved = GrantAccountCheckpointV2::successor(&genesis, &grant, &account).unwrap();
        account.mark_outcome_unknown(reservation).unwrap();
        let unknown = GrantAccountCheckpointV2::successor(&reserved, &grant, &account).unwrap();

        let (_, head) = verify_supplied_chain(&grant, &[genesis, reserved, unknown.clone()]).unwrap();
        assert_eq!(head, unknown.head().unwrap());
    }
}
