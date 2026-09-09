// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grant-bound anti-rollback checkpoints for `symthaea-action-runtime`.
//!
//! Runtime accounting is useful only if persisted state cannot rewrite its own
//! authority ceiling or resurrect an older pre-consumption view. This crate
//! binds every persisted [`GrantAccountSnapshot`] to the exact
//! [`CapabilityGrant`], verifies structural accounting consistency, and chains
//! successive checkpoints by monotonic sequence and digest.
//!
//! The caller must retain/authenticate the latest trusted [`CheckpointHead`]
//! outside the checkpoint being validated (for example via Xenia, TPM-backed
//! state, an append-only log, or another trusted supervisor). The hash chain
//! detects rollback only relative to such an external trusted head.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_action_runtime::{
    DelegationEscrow, EscrowState, ExecutionReservation, GrantAccount, GrantAccountSnapshot,
    ReservationState, RuntimeAccountingError,
};
use symthaea_authority::{CapabilityGrant, Digest32, RiskBudget};
use thiserror::Error;

pub const ACTION_CHECKPOINT_SCHEMA_VERSION: u16 = 1;
const ACTION_CHECKPOINT_DOMAIN: &[u8] = b"symthaea.action-checkpoint.v1\0";

/// Small externally retainable anti-rollback anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHead {
    pub sequence: u64,
    pub digest: Digest32,
}

/// One persisted runtime-accounting generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccountCheckpoint {
    pub schema_version: u16,
    pub sequence: u64,
    pub previous_checkpoint_digest: Option<Digest32>,
    pub grant_digest: Digest32,
    pub snapshot: GrantAccountSnapshot,
}

impl GrantAccountCheckpoint {
    /// Construct generation zero after validating the snapshot against the
    /// externally supplied capability grant.
    pub fn first(
        grant: &CapabilityGrant,
        snapshot: GrantAccountSnapshot,
    ) -> Result<Self, CheckpointError> {
        validate_snapshot_against_grant(grant, &snapshot)?;
        Ok(Self {
            schema_version: ACTION_CHECKPOINT_SCHEMA_VERSION,
            sequence: 0,
            previous_checkpoint_digest: None,
            grant_digest: grant.digest(),
            snapshot,
        })
    }

    /// Construct the next generation, cryptographically linking the exact
    /// predecessor checkpoint.
    pub fn successor(
        previous: &GrantAccountCheckpoint,
        grant: &CapabilityGrant,
        snapshot: GrantAccountSnapshot,
    ) -> Result<Self, CheckpointError> {
        previous.verify_payload(grant)?;
        validate_snapshot_against_grant(grant, &snapshot)?;
        let sequence = previous
            .sequence
            .checked_add(1)
            .ok_or(CheckpointError::SequenceOverflow)?;
        Ok(Self {
            schema_version: ACTION_CHECKPOINT_SCHEMA_VERSION,
            sequence,
            previous_checkpoint_digest: Some(previous.digest()?),
            grant_digest: grant.digest(),
            snapshot,
        })
    }

    /// Domain-separated commitment to the full checkpoint.
    ///
    /// Bincode v1 is intentionally part of this schema generation. A codec
    /// migration must increment `ACTION_CHECKPOINT_SCHEMA_VERSION` rather than
    /// silently reinterpret old commitments.
    pub fn digest(&self) -> Result<Digest32, CheckpointError> {
        let encoded = bincode::serialize(self).map_err(|_| CheckpointError::EncodingFailed)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(ACTION_CHECKPOINT_DOMAIN);
        hasher.update(&encoded);
        Ok(Digest32(*hasher.finalize().as_bytes()))
    }

    pub fn head(&self) -> Result<CheckpointHead, CheckpointError> {
        Ok(CheckpointHead {
            sequence: self.sequence,
            digest: self.digest()?,
        })
    }

    /// Verify payload integrity against the external grant, without asserting a
    /// chain predecessor. Useful before constructing a successor.
    pub fn verify_payload(&self, grant: &CapabilityGrant) -> Result<GrantAccount, CheckpointError> {
        if self.schema_version != ACTION_CHECKPOINT_SCHEMA_VERSION {
            return Err(CheckpointError::UnsupportedSchema);
        }
        if self.grant_digest != grant.digest() {
            return Err(CheckpointError::GrantDigestMismatch);
        }
        validate_snapshot_against_grant(grant, &self.snapshot)?;
        GrantAccount::from_snapshot(self.snapshot.clone()).map_err(CheckpointError::RuntimeState)
    }

    /// Verify this checkpoint as either the first generation or the exact
    /// successor of an externally trusted head.
    pub fn verify_against_head(
        &self,
        grant: &CapabilityGrant,
        expected_previous: Option<CheckpointHead>,
    ) -> Result<GrantAccount, CheckpointError> {
        let account = self.verify_payload(grant)?;
        match expected_previous {
            None => {
                if self.sequence != 0 || self.previous_checkpoint_digest.is_some() {
                    return Err(CheckpointError::UnexpectedPredecessor);
                }
            }
            Some(previous) => {
                let expected_sequence = previous
                    .sequence
                    .checked_add(1)
                    .ok_or(CheckpointError::SequenceOverflow)?;
                if self.sequence != expected_sequence {
                    return Err(CheckpointError::SequenceMismatch);
                }
                if self.previous_checkpoint_digest != Some(previous.digest) {
                    return Err(CheckpointError::PreviousDigestMismatch);
                }
            }
        }
        Ok(account)
    }
}

/// Verify an ordered checkpoint chain from generation zero and return the final
/// reconstructed account plus its trusted head.
pub fn verify_chain(
    grant: &CapabilityGrant,
    checkpoints: &[GrantAccountCheckpoint],
) -> Result<(GrantAccount, CheckpointHead), CheckpointError> {
    let first = checkpoints.first().ok_or(CheckpointError::EmptyChain)?;
    let mut account = first.verify_against_head(grant, None)?;
    let mut head = first.head()?;

    for checkpoint in &checkpoints[1..] {
        account = checkpoint.verify_against_head(grant, Some(head))?;
        head = checkpoint.head()?;
    }

    Ok((account, head))
}

fn validate_snapshot_against_grant(
    grant: &CapabilityGrant,
    snapshot: &GrantAccountSnapshot,
) -> Result<(), CheckpointError> {
    if snapshot.grant_digest != grant.digest()
        || snapshot.max_uses != grant.max_uses
        || snapshot.risk_budget != grant.risk_budget
    {
        return Err(CheckpointError::GrantSnapshotMismatch);
    }

    let (derived_committed_uses, derived_committed_risk) =
        derive_committed_accounting(snapshot)?;
    if snapshot.committed_uses != derived_committed_uses
        || snapshot.committed_risk != derived_committed_risk
    {
        return Err(CheckpointError::CommittedAccountingMismatch);
    }

    GrantAccount::from_snapshot(snapshot.clone()).map_err(CheckpointError::RuntimeState)?;
    Ok(())
}

fn derive_committed_accounting(
    snapshot: &GrantAccountSnapshot,
) -> Result<(u32, RiskBudget), CheckpointError> {
    let mut uses = 0u32;
    let mut risk = RiskBudget::default();

    for (key, reservation) in &snapshot.reservations {
        validate_reservation_key(key, reservation)?;
        if reservation.state == ReservationState::Committed {
            uses = uses.checked_add(1).ok_or(CheckpointError::ArithmeticOverflow)?;
            risk = risk_checked_add(risk, reservation.risk_charge)
                .ok_or(CheckpointError::ArithmeticOverflow)?;
        }
    }

    for (key, escrow) in &snapshot.escrows {
        validate_escrow_key(key, escrow)?;
        match escrow.state {
            EscrowState::Closed => {
                if escrow.committed_uses > escrow.allocated_uses
                    || !escrow.committed_risk.attenuates(escrow.allocated_risk)
                {
                    return Err(CheckpointError::InvalidClosedEscrow);
                }
                uses = uses
                    .checked_add(escrow.committed_uses)
                    .ok_or(CheckpointError::ArithmeticOverflow)?;
                risk = risk_checked_add(risk, escrow.committed_risk)
                    .ok_or(CheckpointError::ArithmeticOverflow)?;
            }
            EscrowState::Open | EscrowState::OutcomeUnknown => {
                if escrow.committed_uses != 0 || escrow.committed_risk != RiskBudget::default() {
                    return Err(CheckpointError::PrematureEscrowCommitment);
                }
            }
        }
    }

    Ok((uses, risk))
}

fn validate_reservation_key(
    key: &symthaea_action_runtime::ReservationId,
    reservation: &ExecutionReservation,
) -> Result<(), CheckpointError> {
    if key != &reservation.reservation_id {
        Err(CheckpointError::ReservationKeyMismatch)
    } else {
        Ok(())
    }
}

fn validate_escrow_key(
    key: &symthaea_action_runtime::EscrowId,
    escrow: &DelegationEscrow,
) -> Result<(), CheckpointError> {
    if key != &escrow.escrow_id {
        Err(CheckpointError::EscrowKeyMismatch)
    } else {
        Ok(())
    }
}

fn risk_checked_add(left: RiskBudget, right: RiskBudget) -> Option<RiskBudget> {
    Some(RiskBudget {
        mutation_units: left.mutation_units.checked_add(right.mutation_units)?,
        irreversible_units: left.irreversible_units.checked_add(right.irreversible_units)?,
        external_disclosure_bytes: left
            .external_disclosure_bytes
            .checked_add(right.external_disclosure_bytes)?,
        monetary_microunits: left
            .monetary_microunits
            .checked_add(right.monetary_microunits)?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CheckpointError {
    #[error("unsupported action checkpoint schema")]
    UnsupportedSchema,
    #[error("checkpoint does not bind the supplied capability grant")]
    GrantDigestMismatch,
    #[error("runtime snapshot ceiling does not match the supplied capability grant")]
    GrantSnapshotMismatch,
    #[error("runtime snapshot committed counters do not match its durable records")]
    CommittedAccountingMismatch,
    #[error("reservation map key does not match embedded reservation id")]
    ReservationKeyMismatch,
    #[error("escrow map key does not match embedded escrow id")]
    EscrowKeyMismatch,
    #[error("closed escrow exceeds its original allocation")]
    InvalidClosedEscrow,
    #[error("open or uncertain escrow contains prematurely committed accounting")]
    PrematureEscrowCommitment,
    #[error("checkpoint encoding failed")]
    EncodingFailed,
    #[error("checkpoint sequence overflow")]
    SequenceOverflow,
    #[error("first checkpoint unexpectedly declares a predecessor")]
    UnexpectedPredecessor,
    #[error("checkpoint sequence does not follow the trusted prior head")]
    SequenceMismatch,
    #[error("checkpoint predecessor digest does not match the trusted prior head")]
    PreviousDigestMismatch,
    #[error("checkpoint chain is empty")]
    EmptyChain,
    #[error("checkpoint accounting arithmetic overflow")]
    ArithmeticOverflow,
    #[error("runtime accounting state invalid: {0}")]
    RuntimeState(RuntimeAccountingError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_action_runtime::{EscrowId, ExecutionId, ReservationId};
    use symthaea_authority::{AuthorityEpoch, PrincipalId};

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "g1",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            AuthorityEpoch(5),
        );
        grant.max_uses = 4;
        grant.risk_budget = risk(4);
        grant
    }

    #[test]
    fn snapshot_cannot_rewrite_its_own_grant_ceiling() {
        let grant = grant();
        let account = GrantAccount::new(&grant);
        let mut snapshot = account.snapshot();
        snapshot.max_uses = 40;
        assert_eq!(
            GrantAccountCheckpoint::first(&grant, snapshot).unwrap_err(),
            CheckpointError::GrantSnapshotMismatch
        );
    }

    #[test]
    fn duplicated_committed_counter_cannot_hide_missing_records() {
        let grant = grant();
        let account = GrantAccount::new(&grant);
        let mut snapshot = account.snapshot();
        snapshot.committed_uses = 1;
        assert_eq!(
            GrantAccountCheckpoint::first(&grant, snapshot).unwrap_err(),
            CheckpointError::CommittedAccountingMismatch
        );
    }

    #[test]
    fn sequence_and_digest_bind_successor_to_exact_prior_head() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let first = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        let first_head = first.head().unwrap();

        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), risk(1))
            .unwrap();
        account.commit_observed(&id).unwrap();
        let second = GrantAccountCheckpoint::successor(&first, &grant, account.snapshot()).unwrap();
        assert!(second.verify_against_head(&grant, Some(first_head)).is_ok());

        let wrong_head = CheckpointHead {
            sequence: first_head.sequence,
            digest: Digest32([9; 32]),
        };
        assert_eq!(
            second.verify_against_head(&grant, Some(wrong_head)).unwrap_err(),
            CheckpointError::PreviousDigestMismatch
        );
    }

    #[test]
    fn old_checkpoint_cannot_validate_as_successor_of_newer_head() {
        let grant = grant();
        let first_account = GrantAccount::new(&grant);
        let first = GrantAccountCheckpoint::first(&grant, first_account.snapshot()).unwrap();
        let first_head = first.head().unwrap();
        assert_eq!(
            first.verify_against_head(&grant, Some(first_head)).unwrap_err(),
            CheckpointError::SequenceMismatch
        );
    }

    #[test]
    fn closed_escrow_accounting_is_reconstructed_from_records() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let escrow = EscrowId("child".into());
        account
            .reserve_delegation_escrow(escrow.clone(), 3, risk(3))
            .unwrap();
        account.close_escrow(&escrow, 1, risk(1)).unwrap();
        let checkpoint = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        assert!(checkpoint.verify_against_head(&grant, None).is_ok());
    }

    #[test]
    fn ordered_chain_verifies_and_returns_latest_head() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let first = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();

        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), risk(1))
            .unwrap();
        let second = GrantAccountCheckpoint::successor(&first, &grant, account.snapshot()).unwrap();

        account.mark_outcome_unknown(&id).unwrap();
        let third = GrantAccountCheckpoint::successor(&second, &grant, account.snapshot()).unwrap();

        let (_, head) = verify_chain(&grant, &[first, second, third.clone()]).unwrap();
        assert_eq!(head, third.head().unwrap());
    }
}
