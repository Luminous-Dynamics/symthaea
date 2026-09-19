// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-grant anti-rollback checkpoints for action accounting.
//!
//! Historical checkpointing bound a runtime snapshot to a grant and predecessor
//! chain. v0.2 additionally exposes an opaque `VerifiedGrantAccounting` only when
//! the checkpoint exactly matches an externally retained trusted head. Chain
//! reconstruction alone is not currentness.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_action_runtime::{GrantAccount, GrantAccountSnapshot, RuntimeAccountingError};
use symthaea_authority::{CapabilityGrant, Digest32, GrantUseState, RiskBudget};
use thiserror::Error;

pub const ACTION_CHECKPOINT_SCHEMA_VERSION: u16 = 2;
const ACTION_CHECKPOINT_DOMAIN: &[u8] = b"symthaea.action-checkpoint.v2\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHead {
    pub grant_digest: Digest32,
    pub sequence: u64,
    pub digest: Digest32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccountCheckpoint {
    pub schema_version: u16,
    pub sequence: u64,
    pub previous_checkpoint_digest: Option<Digest32>,
    pub grant_digest: Digest32,
    pub snapshot: GrantAccountSnapshot,
}

impl GrantAccountCheckpoint {
    pub fn first(
        grant: &CapabilityGrant,
        snapshot: GrantAccountSnapshot,
    ) -> Result<Self, CheckpointError> {
        GrantAccount::from_snapshot(grant, snapshot.clone())?;
        Ok(Self {
            schema_version: ACTION_CHECKPOINT_SCHEMA_VERSION,
            sequence: 0,
            previous_checkpoint_digest: None,
            grant_digest: grant.digest(),
            snapshot,
        })
    }

    pub fn successor(
        previous: &GrantAccountCheckpoint,
        grant: &CapabilityGrant,
        snapshot: GrantAccountSnapshot,
    ) -> Result<Self, CheckpointError> {
        previous.verify_payload(grant)?;
        GrantAccount::from_snapshot(grant, snapshot.clone())?;
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

    pub fn digest(&self) -> Result<Digest32, CheckpointError> {
        if self.schema_version != ACTION_CHECKPOINT_SCHEMA_VERSION {
            return Err(CheckpointError::UnsupportedSchema);
        }
        let encoded = bincode::serialize(self).map_err(|_| CheckpointError::EncodingFailed)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(ACTION_CHECKPOINT_DOMAIN);
        hasher.update(&encoded);
        Ok(Digest32(*hasher.finalize().as_bytes()))
    }

    pub fn head(&self) -> Result<CheckpointHead, CheckpointError> {
        Ok(CheckpointHead {
            grant_digest: self.grant_digest,
            sequence: self.sequence,
            digest: self.digest()?,
        })
    }

    pub fn verify_payload(&self, grant: &CapabilityGrant) -> Result<GrantAccount, CheckpointError> {
        if self.schema_version != ACTION_CHECKPOINT_SCHEMA_VERSION {
            return Err(CheckpointError::UnsupportedSchema);
        }
        if self.grant_digest != grant.digest() || self.snapshot.grant_digest != grant.digest() {
            return Err(CheckpointError::GrantDigestMismatch);
        }
        GrantAccount::from_snapshot(grant, self.snapshot.clone()).map_err(CheckpointError::Runtime)
    }

    pub fn verify_against_predecessor(
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
                if previous.grant_digest != self.grant_digest {
                    return Err(CheckpointError::PredecessorGrantMismatch);
                }
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

/// Opaque grant-bound accounting proof relative to one externally supplied
/// trusted checkpoint head. Not Clone and not Serde.
#[derive(Debug)]
pub struct VerifiedGrantAccounting {
    grant_digest: Digest32,
    checkpoint_head: CheckpointHead,
    use_state: GrantUseState,
    charged_risk: RiskBudget,
}

impl VerifiedGrantAccounting {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn checkpoint_head(&self) -> CheckpointHead {
        self.checkpoint_head
    }

    pub fn use_state(&self) -> &GrantUseState {
        &self.use_state
    }

    pub fn charged_risk(&self) -> RiskBudget {
        self.charged_risk
    }

    pub fn ensure_grant(&self, grant: &CapabilityGrant) -> Result<(), CheckpointError> {
        if grant.digest() != self.grant_digest {
            Err(CheckpointError::GrantDigestMismatch)
        } else {
            Ok(())
        }
    }
}

/// Verify the exact checkpoint selected by an externally authenticated/retained
/// head. Supplying a head is an explicit trust boundary; this crate does not
/// authenticate Xenia/TPM/log custody of that head.
pub fn verify_current_checkpoint(
    grant: &CapabilityGrant,
    checkpoint: &GrantAccountCheckpoint,
    trusted_head: CheckpointHead,
) -> Result<VerifiedGrantAccounting, CheckpointError> {
    if trusted_head.grant_digest != grant.digest() {
        return Err(CheckpointError::TrustedHeadGrantMismatch);
    }
    let account = checkpoint.verify_payload(grant)?;
    let actual_head = checkpoint.head()?;
    if actual_head != trusted_head {
        return Err(CheckpointError::TrustedHeadMismatch);
    }
    Ok(VerifiedGrantAccounting {
        grant_digest: grant.digest(),
        checkpoint_head: actual_head,
        use_state: account.unverified_use_state()?,
        charged_risk: account.unverified_charged_risk()?,
    })
}

/// Reconstruct a complete ordered chain for audit/recovery. This does not by
/// itself establish that the returned head is the externally current head.
pub fn verify_chain(
    grant: &CapabilityGrant,
    checkpoints: &[GrantAccountCheckpoint],
) -> Result<(GrantAccount, CheckpointHead), CheckpointError> {
    let first = checkpoints.first().ok_or(CheckpointError::EmptyChain)?;
    let mut account = first.verify_against_predecessor(grant, None)?;
    let mut head = first.head()?;
    for checkpoint in &checkpoints[1..] {
        account = checkpoint.verify_against_predecessor(grant, Some(head))?;
        head = checkpoint.head()?;
    }
    Ok((account, head))
}

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("unsupported action-checkpoint schema")]
    UnsupportedSchema,
    #[error("checkpoint does not bind the supplied capability grant")]
    GrantDigestMismatch,
    #[error("checkpoint encoding failed")]
    EncodingFailed,
    #[error("checkpoint sequence overflow")]
    SequenceOverflow,
    #[error("first checkpoint unexpectedly declares a predecessor")]
    UnexpectedPredecessor,
    #[error("predecessor head belongs to another grant")]
    PredecessorGrantMismatch,
    #[error("checkpoint sequence does not follow the expected predecessor")]
    SequenceMismatch,
    #[error("checkpoint predecessor digest does not match the expected head")]
    PreviousDigestMismatch,
    #[error("checkpoint chain is empty")]
    EmptyChain,
    #[error("trusted current head belongs to another grant")]
    TrustedHeadGrantMismatch,
    #[error("checkpoint does not exactly match the externally trusted current head")]
    TrustedHeadMismatch,
    #[error("runtime accounting failed: {0}")]
    Runtime(#[from] RuntimeAccountingError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_action_runtime::{ExecutionId, GrantAccount, ReservationId};
    use symthaea_authority::{
        AuthorityContextRef, AuthorityEpoch, Operation, PrincipalId, PurposeId, ResourceRef,
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

    fn grant(id: &str) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            id,
            PrincipalId("issuer".into()),
            PrincipalId("robot".into()),
            PurposeId("goal-directed-actuation".into()),
            AuthorityEpoch(7),
            AuthorityContextRef::new("swarm-controller", digest(9)),
        );
        grant.resources.insert(ResourceRef("robot".into()));
        grant.operations.insert(Operation("move".into()));
        grant.max_uses = 4;
        grant.risk_budget = risk(4);
        grant
    }

    #[test]
    fn exact_trusted_head_yields_opaque_grant_bound_accounting() {
        let grant = grant("g1");
        let mut account = GrantAccount::new(&grant).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(1),
                risk(1),
            )
            .unwrap();
        let checkpoint = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        let verified = verify_current_checkpoint(&grant, &checkpoint, checkpoint.head().unwrap())
            .unwrap();
        assert_eq!(verified.grant_digest(), grant.digest());
        assert_eq!(verified.use_state().committed, 0);
        assert_eq!(verified.use_state().reserved, 1);
        assert_eq!(verified.charged_risk(), risk(1));
    }

    #[test]
    fn stale_checkpoint_cannot_validate_against_newer_trusted_head() {
        let grant = grant("g1");
        let mut account = GrantAccount::new(&grant).unwrap();
        let first = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(1),
                risk(1),
            )
            .unwrap();
        let second = GrantAccountCheckpoint::successor(&first, &grant, account.snapshot()).unwrap();
        assert!(matches!(
            verify_current_checkpoint(&grant, &first, second.head().unwrap()),
            Err(CheckpointError::TrustedHeadMismatch)
        ));
    }

    #[test]
    fn trusted_head_is_grant_bound() {
        let grant_a = grant("a");
        let grant_b = grant("b");
        let account = GrantAccount::new(&grant_a).unwrap();
        let checkpoint = GrantAccountCheckpoint::first(&grant_a, account.snapshot()).unwrap();
        assert!(matches!(
            verify_current_checkpoint(&grant_b, &checkpoint, checkpoint.head().unwrap()),
            Err(CheckpointError::TrustedHeadGrantMismatch)
        ));
    }

    #[test]
    fn effect_identity_is_checkpoint_committed() {
        let grant = grant("g1");
        let mut account = GrantAccount::new(&grant).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(1),
                risk(1),
            )
            .unwrap();
        let first = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        let mut changed = account.snapshot();
        changed
            .reservations
            .get_mut(&ReservationId("r1".into()))
            .unwrap()
            .effect_digest = digest(2);
        let altered = GrantAccountCheckpoint::first(&grant, changed).unwrap();
        assert_ne!(first.digest().unwrap(), altered.digest().unwrap());
        assert!(matches!(
            verify_current_checkpoint(&grant, &altered, first.head().unwrap()),
            Err(CheckpointError::TrustedHeadMismatch)
        ));
    }

    #[test]
    fn chain_reconstruction_is_not_currentness_but_preserves_order() {
        let grant = grant("g1");
        let mut account = GrantAccount::new(&grant).unwrap();
        let first = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(1),
                risk(1),
            )
            .unwrap();
        let second = GrantAccountCheckpoint::successor(&first, &grant, account.snapshot()).unwrap();
        let (_, head) = verify_chain(&grant, &[first, second.clone()]).unwrap();
        assert_eq!(head, second.head().unwrap());
    }
}
