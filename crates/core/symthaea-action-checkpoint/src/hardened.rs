// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardened public checkpoint facade for action authority v0.2.
//!
//! The R1 predecessor `strict.rs` protects canonical identity, genesis,
//! reservation preservation, forward state motion, and forbids ordinary
//! creation of `Released`. This facade preserves V2B's more specific theorem:
//!
//! ```text
//! OutcomeUnknown -> Released
//!     requires a separately verified reconciliation proof
//! ```
//!
//! Returning capacity is authority-increasing. Ordinary runtime/checkpoint APIs
//! do not perform that transition. A future verifier-owned reconciliation path
//! must prove both exact-attempt non-application and that no live/delayed
//! dispatch capability can still cross the effect boundary.

#![deny(unsafe_code)]

#[path = "strict.rs"]
mod predecessor;

use serde::{Deserialize, Serialize};
use symthaea_action_runtime::{GrantAccountSnapshotV2, GrantAccountV2, ReservationState};
use symthaea_authority::{CapabilityGrant, Digest32};
use thiserror::Error;

pub use predecessor::{ACTION_CHECKPOINT_DOMAIN, ACTION_CHECKPOINT_SCHEMA_VERSION, CheckpointHeadV2};

/// Public checkpoint type with non-refundable uncertain-state semantics.
///
/// Serialization is storage-only. The canonical digest remains the exact frozen
/// predecessor transcript and independent oracle; this wrapper preserves the
/// explicit `OutcomeUnknownReleaseRequiresProof` claim while the predecessor
/// forbids every ordinary transition that newly creates `Released`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GrantAccountCheckpointV2 {
    inner: predecessor::GrantAccountCheckpointV2,
}

impl GrantAccountCheckpointV2 {
    pub fn first(
        grant: &CapabilityGrant,
        account: &GrantAccountV2,
    ) -> Result<Self, CheckpointV2Error> {
        predecessor::GrantAccountCheckpointV2::first(grant, account)
            .map(|inner| Self { inner })
            .map_err(predecessor_error)
    }

    pub fn successor(
        previous: &Self,
        grant: &CapabilityGrant,
        account: &GrantAccountV2,
    ) -> Result<Self, CheckpointV2Error> {
        let next = account.snapshot();
        forbid_unproven_uncertain_release(previous.snapshot(), &next)?;
        predecessor::GrantAccountCheckpointV2::successor(&previous.inner, grant, account)
            .map(|inner| Self { inner })
            .map_err(predecessor_error)
    }

    pub fn digest(&self) -> Result<Digest32, CheckpointV2Error> {
        self.inner.digest().map_err(predecessor_error)
    }

    pub fn head(&self) -> Result<CheckpointHeadV2, CheckpointV2Error> {
        self.inner.head().map_err(predecessor_error)
    }

    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }

    pub fn previous_checkpoint_digest(&self) -> Option<Digest32> {
        self.inner.previous_checkpoint_digest()
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.inner.grant_digest()
    }

    pub fn snapshot(&self) -> &GrantAccountSnapshotV2 {
        self.inner.snapshot()
    }

    pub fn verify_payload(
        &self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        self.inner.verify_payload(grant).map_err(predecessor_error)
    }

    pub fn verify_genesis(
        &self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        self.inner.verify_genesis(grant).map_err(predecessor_error)
    }

    /// Head-shape validation only. An expected head is ordinary data unless a
    /// stronger layer authenticates/currentizes it.
    pub fn verify_shape_against_expected_head(
        &self,
        grant: &CapabilityGrant,
        expected_previous: Option<CheckpointHeadV2>,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        self.inner
            .verify_shape_against_expected_head(grant, expected_previous)
            .map_err(predecessor_error)
    }

    /// Verify exact predecessor identity, the R1 no-refund state machine, and
    /// the explicit V2B `OutcomeUnknown` claim ceiling.
    pub fn verify_successor_of(
        &self,
        previous: &Self,
        grant: &CapabilityGrant,
    ) -> Result<GrantAccountV2, CheckpointV2Error> {
        forbid_unproven_uncertain_release(previous.snapshot(), self.snapshot())?;
        self.inner
            .verify_successor_of(&previous.inner, grant)
            .map_err(predecessor_error)
    }
}

/// Verify a supplied history without treating its final head as externally current.
/// Every edge preserves R1's no-ordinary-release theorem and V2B's explicit
/// uncertain-release claim ceiling.
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

fn forbid_unproven_uncertain_release(
    previous: &GrantAccountSnapshotV2,
    next: &GrantAccountSnapshotV2,
) -> Result<(), CheckpointV2Error> {
    for (reservation_id, old) in &previous.reservations {
        if old.state == ReservationState::OutcomeUnknown
            && next
                .reservations
                .get(reservation_id)
                .is_some_and(|new| new.state == ReservationState::Released)
        {
            return Err(CheckpointV2Error::OutcomeUnknownReleaseRequiresProof);
        }
    }
    Ok(())
}

fn predecessor_error(error: predecessor::CheckpointV2Error) -> CheckpointV2Error {
    CheckpointV2Error::Predecessor(error.to_string())
}

#[derive(Debug, Error)]
pub enum CheckpointV2Error {
    #[error("strict predecessor checkpoint verification failed: {0}")]
    Predecessor(String),
    #[error(
        "OutcomeUnknown authority cannot be released without a separately verified no-live-dispatch/non-application proof"
    )]
    OutcomeUnknownReleaseRequiresProof,
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

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-no-refund-v2",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            PurposeId("effect".into()),
            AuthorityEpoch(4),
            AuthorityContextRef::new("symthaea.test.context.v1", digest(1)),
        );
        grant.resources.insert(ResourceRef("resource".into()));
        grant.operations.insert(Operation("operate".into()));
        grant.max_uses = 1;
        grant.risk_budget = RiskBudget {
            mutation_units: 1,
            ..RiskBudget::default()
        };
        grant
    }

    fn unknown_account(
        grant: &CapabilityGrant,
    ) -> (
        GrantAccountCheckpointV2,
        GrantAccountV2,
        symthaea_action_runtime::ReservationId,
    ) {
        let mut account = GrantAccountV2::new_root(grant).unwrap();
        let genesis = GrantAccountCheckpointV2::first(grant, &account).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                RiskBudget {
                    mutation_units: 1,
                    ..RiskBudget::default()
                },
            )
            .unwrap();
        let reserved = GrantAccountCheckpointV2::successor(&genesis, grant, &account).unwrap();
        account.mark_outcome_unknown(reservation).unwrap();
        let unknown = GrantAccountCheckpointV2::successor(&reserved, grant, &account).unwrap();
        (unknown, account, reservation)
    }

    #[test]
    fn ordinary_checkpoint_path_cannot_refund_outcome_unknown() {
        let grant = grant();
        let (unknown, account, reservation) = unknown_account(&grant);

        // Persisted snapshots are ordinary data. Construct a malicious/future-
        // shaped snapshot that claims Released and prove the public ordinary
        // checkpoint path still cannot publish that authority increase.
        let mut forged: GrantAccountSnapshotV2 = account.snapshot();
        forged.reservations.get_mut(&reservation).unwrap().state = ReservationState::Released;
        let forged_account = GrantAccountV2::restore(&grant, forged).unwrap();

        assert!(matches!(
            GrantAccountCheckpointV2::successor(&unknown, &grant, &forged_account),
            Err(CheckpointV2Error::OutcomeUnknownReleaseRequiresProof)
        ));
    }

    #[test]
    fn strict_predecessor_blocks_reserved_to_released_too() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let genesis = GrantAccountCheckpointV2::first(&grant, &account).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                RiskBudget {
                    mutation_units: 1,
                    ..RiskBudget::default()
                },
            )
            .unwrap();
        let reserved = GrantAccountCheckpointV2::successor(&genesis, &grant, &account).unwrap();
        let mut forged = account.snapshot();
        forged.reservations.get_mut(&reservation).unwrap().state = ReservationState::Released;
        let forged_account = GrantAccountV2::restore(&grant, forged).unwrap();
        assert!(matches!(
            GrantAccountCheckpointV2::successor(&reserved, &grant, &forged_account),
            Err(CheckpointV2Error::Predecessor(_))
        ));
    }

    #[test]
    fn committed_forward_transition_remains_available() {
        let grant = grant();
        let (unknown, mut account, reservation) = unknown_account(&grant);
        account.reconcile_applied(reservation).unwrap();
        assert!(GrantAccountCheckpointV2::successor(&unknown, &grant, &account).is_ok());
    }
}
