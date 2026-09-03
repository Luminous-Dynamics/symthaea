// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composed Xenia-authorized systemd recovery profile.
//!
//! This crate adds no new system effect. It composes typed broker semantics,
//! CAS-backed authority accounting, challenged time, authenticated current
//! authority state, and a fresh witnessed process-bound executor workload.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, GrantUseState, ReservationId};
use symthaea_authority::{AuthorityContext, CapabilityGrant, Digest32};
use symthaea_authority_frontier::{
    CasCheckpointStoreAdapter, CheckpointCasStore, EstablishedGrantFrontier, FrontierError,
    establish_grant_frontier,
};
use symthaea_authority_state::AuthorityStateError;
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use symthaea_system_broker::{
    BrokerError, RecoveryReceipt, RestartPlan, ServiceBackend, SystemdRecoveryBroker,
};
use symthaea_xenia_authority::{VerifiedXeniaCapability, WorkloadIdentityError};
use thiserror::Error;

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

    pub fn authorization_checkpoint_head(&self) -> Result<CheckpointHead, ProfileRecoveryError> {
        self.broker
            .current_checkpoint_head()?
            .ok_or(ProfileRecoveryError::MissingCheckpoint)
    }

    pub fn is_contained(&self) -> bool {
        self.broker.is_contained()
    }

    /// Consume one independently verified Xenia proof and attempt exactly one
    /// typed recovery.
    ///
    /// Effect entry accepts no caller-selected wall clock, epoch, revocation
    /// list, authority-state object, workload identity, or use counter. The
    /// Xenia proof owns both authenticated authority state and witnessed
    /// executor identity. Immediately before broker admission we re-check that
    /// the workload proof is fresh and still describes this exact Linux process
    /// instance/artifact.
    pub fn recover_verified_once(
        &mut self,
        verified: VerifiedXeniaCapability,
        authority_time: &VerifiedAuthorityTime,
        plan: &RestartPlan,
        execution_id: ExecutionId,
        reservation_id: ReservationId,
    ) -> Result<XeniaSystemdRecoveryReceipt, ProfileRecoveryError> {
        if verified.grant_digest() != self.grant_digest {
            return Err(ProfileRecoveryError::VerifiedGrantMismatch);
        }
        authority_time.require_subject(self.grant_digest.0)?;
        verified
            .authority_state()
            .ensure_fresh(&self.grant, authority_time)?;
        verified
            .executor_workload()
            .ensure_fresh(&self.grant, authority_time)?;
        verified.executor_workload().require_current_process()?;

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
        let current_epoch = verified.authority_state().authority_epoch();

        let recovery = self.broker.recover_once(
            plan,
            execution_id,
            reservation_id,
            AuthorityContext {
                now_unix_s,
                current_epoch,
                use_state: GrantUseState::default(),
            },
            verified.authority_state().negative_facts(),
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
    #[error("verified executor workload failed at effect entry: {0}")]
    Workload(#[from] WorkloadIdentityError),
    #[error("typed systemd broker failed: {0}")]
    Broker(#[from] BrokerError),
    #[error("profile has no established checkpoint")]
    MissingCheckpoint,
    #[error("verified Xenia proof belongs to a different capability grant")]
    VerifiedGrantMismatch,
    #[error("Xenia proof expired before effect entry")]
    XeniaProofExpiredAtEffectEntry,
    #[error("Agency Kernel frontier advanced after Xenia proof verification")]
    AuthorityFrontierAdvanced,
}
