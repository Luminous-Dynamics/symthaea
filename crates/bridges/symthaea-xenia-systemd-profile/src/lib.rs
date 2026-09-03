// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composed Xenia-authorized systemd recovery profile.
//!
//! This crate adds no new system effect. It composes:
//!
//! - #305 typed `service.restart` broker semantics;
//! - CAS-backed checkpoint frontiers;
//! - independently verified Xenia capability/workload evidence;
//! - challenge-bound multi-authority time evidence;
//! - threshold-authenticated current authority epoch + negative facts.
//!
//! A verified Xenia proof is consumed by value immediately before the broker
//! enters its reservation/checkpoint/effect state machine. Cross-process replay
//! is constrained by the CAS frontier rather than by relying on Rust affinity
//! alone. Caller-provided wall-clock time, authority epoch, and revocation lists
//! are not part of the production effect-entry API.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, GrantUseState, ReservationId};
use symthaea_authority::{AuthorityContext, CapabilityGrant, Digest32};
use symthaea_authority_frontier::{
    CasCheckpointStoreAdapter, CheckpointCasStore, EstablishedGrantFrontier, FrontierError,
    establish_grant_frontier,
};
use symthaea_authority_state::{AuthorityStateError, VerifiedAuthorityState};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use symthaea_system_broker::{
    BrokerError, RecoveryReceipt, RestartPlan, ServiceBackend, SystemdRecoveryBroker,
};
use symthaea_xenia_authority::VerifiedXeniaCapability;
use thiserror::Error;

/// Xenia evidence attached to one successful typed systemd recovery receipt.
#[derive(Debug)]
pub struct XeniaSystemdRecoveryReceipt {
    pub xenia_authorization_id: [u8; 16],
    pub xenia_session_id: [u8; 16],
    pub xenia_ledger_entry_count: u64,
    pub xenia_ledger_head_hash: [u8; 32],
    pub workload_digest: Digest32,
    pub authority_state_digest: Digest32,
    pub authority_state_sequence: u64,
    pub recovery: RecoveryReceipt,
}

/// Ready-to-authorize systemd recovery session.
///
/// The profile owns one exact capability grant and a broker restored at a
/// trusted CAS frontier. Xenia should authorize the exact head returned by
/// [`Self::authorization_checkpoint_head`].
pub struct XeniaSystemdRecoveryProfile<B, S>
where
    B: ServiceBackend,
    S: CheckpointCasStore,
{
    grant: CapabilityGrant,
    grant_digest: Digest32,
    broker: SystemdRecoveryBroker<B, CasCheckpointStoreAdapter<S>>,
}

impl<B, S> XeniaSystemdRecoveryProfile<B, S>
where
    B: ServiceBackend,
    S: CheckpointCasStore,
{
    /// Establish generation zero before requesting Xenia authority.
    pub fn bootstrap(
        grant: CapabilityGrant,
        backend: B,
        store: S,
    ) -> Result<(Self, EstablishedGrantFrontier), ProfileBootstrapError<S::Error>> {
        let grant_digest = grant.digest();
        let grant_for_profile = grant.clone();
        let (frontier, adapter) = establish_grant_frontier(&grant, store)
            .map_err(ProfileBootstrapError::Frontier)?;
        let broker = SystemdRecoveryBroker::from_checkpoint(
            grant,
            frontier.checkpoint.clone(),
            frontier.head,
            backend,
            adapter,
        )
        .map_err(ProfileBootstrapError::Broker)?;
        Ok((
            Self {
                grant: grant_for_profile,
                grant_digest,
                broker,
            },
            frontier,
        ))
    }

    /// Restore an already-established profile from one externally trusted head.
    pub fn restore(
        grant: CapabilityGrant,
        checkpoint: GrantAccountCheckpoint,
        trusted_head: CheckpointHead,
        backend: B,
        store: S,
    ) -> Result<Self, BrokerError> {
        let grant_digest = grant.digest();
        let grant_for_profile = grant.clone();
        let adapter = CasCheckpointStoreAdapter::from_trusted_head(store, trusted_head);
        let broker = SystemdRecoveryBroker::from_checkpoint(
            grant,
            checkpoint,
            trusted_head,
            backend,
            adapter,
        )?;
        Ok(Self {
            grant: grant_for_profile,
            grant_digest,
            broker,
        })
    }

    /// Exact Agency Kernel frontier Xenia must bind before this profile mutates.
    pub fn authorization_checkpoint_head(&self) -> Result<CheckpointHead, ProfileRecoveryError> {
        self.broker
            .current_checkpoint_head()?
            .ok_or(ProfileRecoveryError::MissingCheckpoint)
    }

    pub fn is_contained(&self) -> bool {
        self.broker.is_contained()
    }

    /// Consume one independently verified Xenia authority proof and attempt the
    /// exact typed recovery.
    ///
    /// The proof must still bind this profile's exact current checkpoint and
    /// the exact fresh authority-state snapshot at the instant immediately
    /// before the broker enters effect admission. Time is re-derived from the
    /// short-lived challenged fact at this same boundary.
    ///
    /// No caller-selected current epoch or negative-fact list is accepted.
    pub fn recover_verified_once(
        &mut self,
        verified: VerifiedXeniaCapability,
        authority_time: &VerifiedAuthorityTime,
        authority_state: VerifiedAuthorityState,
        plan: &RestartPlan,
        execution_id: ExecutionId,
        reservation_id: ReservationId,
    ) -> Result<XeniaSystemdRecoveryReceipt, ProfileRecoveryError> {
        if verified.grant_digest() != self.grant_digest {
            return Err(ProfileRecoveryError::VerifiedGrantMismatch);
        }
        authority_time.require_subject(self.grant_digest.0)?;
        authority_state.ensure_fresh(&self.grant, authority_time)?;
        if verified.authority_state_digest() != authority_state.snapshot_digest()
            || verified.authority_state_sequence() != authority_state.state_sequence()
        {
            return Err(ProfileRecoveryError::AuthorityStateChangedAfterXeniaVerification);
        }

        let now_unix_s = authority_time.conservative_now_unix_s()?;
        if now_unix_s > verified.expires_at_unix_s() {
            return Err(ProfileRecoveryError::XeniaProofExpiredAtEffectEntry);
        }
        let current_head = self.authorization_checkpoint_head()?;
        if verified.prior_checkpoint() != current_head {
            return Err(ProfileRecoveryError::AuthorityFrontierAdvanced);
        }

        let authorization_id = verified.authorization_id();
        let session_id = verified.session_id();
        let workload_digest = verified.workload_digest();
        let authority_state_digest = verified.authority_state_digest();
        let authority_state_sequence = verified.authority_state_sequence();
        let (xenia_ledger_entry_count, xenia_ledger_head_hash) = verified.xenia_frontier();

        // `verified` is intentionally consumed here. Durable/cross-process
        // single-use semantics come from the broker's subsequent CAS checkpoint
        // transition rather than from the affine Rust value alone.
        drop(verified);

        let recovery = self.broker.recover_once(
            plan,
            execution_id,
            reservation_id,
            AuthorityContext {
                now_unix_s,
                current_epoch: authority_state.authority_epoch(),
                // The broker explicitly ignores caller use accounting and
                // substitutes its own durable GrantAccount state.
                use_state: GrantUseState::default(),
            },
            authority_state.negative_facts(),
        )?;

        Ok(XeniaSystemdRecoveryReceipt {
            xenia_authorization_id: authorization_id,
            xenia_session_id: session_id,
            xenia_ledger_entry_count,
            xenia_ledger_head_hash,
            workload_digest,
            authority_state_digest,
            authority_state_sequence,
            recovery,
        })
    }
}

#[derive(Debug, Error)]
pub enum ProfileBootstrapError<E>
where
    E: StdError + 'static,
{
    #[error("failed to establish CAS authority frontier: {0}")]
    Frontier(#[source] FrontierError<E>),
    #[error("failed to restore typed systemd broker at generation zero: {0}")]
    Broker(#[source] BrokerError),
}

#[derive(Debug, Error)]
pub enum ProfileRecoveryError {
    #[error("verified authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("verified authority state failed: {0}")]
    AuthorityState(#[from] AuthorityStateError),
    #[error("typed systemd broker failed: {0}")]
    Broker(#[from] BrokerError),
    #[error("profile has no established checkpoint")]
    MissingCheckpoint,
    #[error("verified Xenia proof belongs to a different capability grant")]
    VerifiedGrantMismatch,
    #[error("fresh authority state differs from the state bound into the verified Xenia proof")]
    AuthorityStateChangedAfterXeniaVerification,
    #[error("Xenia proof expired before effect entry")]
    XeniaProofExpiredAtEffectEntry,
    #[error("Agency Kernel frontier advanced after Xenia proof verification")]
    AuthorityFrontierAdvanced,
}
