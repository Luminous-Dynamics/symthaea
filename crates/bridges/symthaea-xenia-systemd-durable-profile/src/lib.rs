// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable one-use Xenia-authorized systemd recovery profile.
//!
//! This profile composes Xenia delegation, challenged time, authenticated
//! authority state, a fresh witnessed process-bound executor, SQLite CAS
//! accounting, the typed systemd broker, and durable write-ahead attempt
//! evidence.

#![deny(unsafe_code)]

use std::path::{Path, PathBuf};

use serde::Serialize;
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_action_runtime::{ExecutionId, GrantUseState, ReservationId};
use symthaea_authority::{AuthorityContext, CapabilityGrant, Digest32};
use symthaea_authority_frontier::{
    CasCheckpointStoreAdapter, EstablishedGrantFrontier, FrontierError, establish_grant_frontier,
};
use symthaea_authority_frontier_sqlite::{SqliteCheckpointCasStore, SqliteFrontierError};
use symthaea_authority_state::AuthorityStateError;
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use symthaea_system_attempt_evidence::{
    AttemptEvidenceContext, AttemptEvidenceHead, SqliteAttemptEvidenceError,
    SqliteAttemptEvidenceJournal, instrument_attempt,
};
use symthaea_system_broker::{
    BrokerError, RecoveryReceipt, RestartPlan, ServiceBackend, SystemdRecoveryBroker,
};
use symthaea_xenia_authority::{
    VerifiedXeniaCapability, WorkloadIdentityError,
};
use thiserror::Error;

const XENIA_AUTHORITY_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.xenia-systemd.authority-evidence.v0.3\0";

pub struct DurableXeniaSystemdBootstrap<B>
where
    B: ServiceBackend,
{
    grant: CapabilityGrant,
    grant_digest: Digest32,
    backend: B,
    frontier: EstablishedGrantFrontier,
    store: SqliteCheckpointCasStore,
    state_path: PathBuf,
}

impl<B> DurableXeniaSystemdBootstrap<B>
where
    B: ServiceBackend,
{
    pub fn bootstrap(
        grant: CapabilityGrant,
        backend: B,
        state_path: impl AsRef<Path>,
    ) -> Result<Self, DurableBootstrapError> {
        let state_path = state_path.as_ref().to_path_buf();
        let store = SqliteCheckpointCasStore::open(&state_path)
            .map_err(DurableBootstrapError::OpenCheckpointStore)?;
        let grant_digest = grant.digest();
        let (frontier, adapter) = establish_grant_frontier(&grant, store)
            .map_err(DurableBootstrapError::EstablishFrontier)?;
        let store = adapter.into_inner();
        Ok(Self {
            grant,
            grant_digest,
            backend,
            frontier,
            store,
            state_path,
        })
    }

    pub fn authorization_checkpoint_head(&self) -> CheckpointHead {
        self.frontier.head
    }

    pub fn authorization_checkpoint(&self) -> &symthaea_action_checkpoint::GrantAccountCheckpoint {
        &self.frontier.checkpoint
    }

    /// Consume one verified Xenia capability and run the existing typed broker.
    ///
    /// Before any attempt journal is opened, the proof-owned workload is checked
    /// for freshness and re-measured against this exact Linux process instance.
    /// A process/artifact mismatch therefore cannot create `DispatchArmed`
    /// evidence, reserve a use, or call the backend.
    pub fn recover_verified_once(
        self,
        verified: VerifiedXeniaCapability,
        authority_time: &VerifiedAuthorityTime,
        plan: &RestartPlan,
        execution_id: ExecutionId,
        reservation_id: ReservationId,
    ) -> Result<DurableXeniaSystemdReceipt, DurableRecoveryError> {
        if verified.grant_digest() != self.grant_digest {
            return Err(DurableRecoveryError::VerifiedGrantMismatch);
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
            return Err(DurableRecoveryError::XeniaProofExpiredAtEffectEntry);
        }
        if verified.prior_checkpoint() != self.frontier.head {
            return Err(DurableRecoveryError::AuthorityFrontierAdvanced);
        }

        let xenia_authorization_id = verified.authorization_id();
        let xenia_session_id = verified.session_id();
        let workload_digest = verified.workload_digest();
        let authority_state_digest = verified.authority_state_digest();
        let authority_state_sequence = verified.authority_state_sequence();
        let current_epoch = verified.authority_state().authority_epoch();
        let (xenia_ledger_entry_count, xenia_ledger_head_hash) = verified.xenia_frontier();
        let authority_evidence_digest =
            xenia_authority_evidence_digest(self.grant_digest, &verified);

        let attempt_context = AttemptEvidenceContext::new(
            &execution_id,
            &reservation_id,
            self.grant_digest,
            plan.digest(),
            plan.world_digest,
            Some(authority_evidence_digest),
        );
        let attempt_key = attempt_context.attempt_key();
        let journal = SqliteAttemptEvidenceJournal::open(&self.state_path)
            .map_err(DurableRecoveryError::OpenAttemptJournal)?;

        let cas_adapter =
            CasCheckpointStoreAdapter::from_trusted_head(self.store, self.frontier.head);
        let (backend, checkpoint_store, evidence_handle) =
            instrument_attempt(self.backend, cas_adapter, journal, attempt_context);

        let mut broker = SystemdRecoveryBroker::from_checkpoint(
            self.grant,
            self.frontier.checkpoint,
            self.frontier.head,
            backend,
            checkpoint_store,
        )
        .map_err(DurableRecoveryError::BrokerRestore)?;

        let recovery = match broker.recover_once(
            plan,
            execution_id,
            reservation_id,
            AuthorityContext {
                now_unix_s,
                current_epoch,
                use_state: GrantUseState::default(),
            },
            verified.authority_state().negative_facts(),
        ) {
            Ok(receipt) => receipt,
            Err(source) => {
                let attempt_evidence = match evidence_handle.latest_head() {
                    Ok(head) => DurableEvidenceLocator::Known(head),
                    Err(error) => DurableEvidenceLocator::Unavailable {
                        diagnostic_digest: diagnostic(&error),
                    },
                };
                return Err(DurableRecoveryError::BrokerAttempt {
                    source,
                    attempt_key,
                    attempt_evidence,
                });
            }
        };

        let attempt_evidence = match evidence_handle.append_recovery_receipt(&recovery) {
            Ok(head) => DurableAttemptEvidenceStatus::RecoveryCompleted(head),
            Err(error) => {
                let last_durable_evidence = match evidence_handle.latest_head() {
                    Ok(head) => DurableEvidenceLocator::Known(head),
                    Err(locator_error) => DurableEvidenceLocator::Unavailable {
                        diagnostic_digest: diagnostic(&locator_error),
                    },
                };
                DurableAttemptEvidenceStatus::FinalizationIncomplete {
                    last_durable_evidence,
                    diagnostic_digest: diagnostic(&error),
                }
            }
        };

        Ok(DurableXeniaSystemdReceipt {
            xenia_authorization_id,
            xenia_session_id,
            xenia_ledger_entry_count,
            xenia_ledger_head_hash,
            workload_digest,
            authority_state_digest,
            authority_state_sequence,
            authority_evidence_digest,
            attempt_key,
            recovery,
            attempt_evidence,
        })
    }
}

#[derive(Debug)]
pub struct DurableXeniaSystemdReceipt {
    pub xenia_authorization_id: [u8; 16],
    pub xenia_session_id: [u8; 16],
    pub xenia_ledger_entry_count: u64,
    pub xenia_ledger_head_hash: [u8; 32],
    pub workload_digest: Digest32,
    pub authority_state_digest: Digest32,
    pub authority_state_sequence: u64,
    pub authority_evidence_digest: Digest32,
    pub attempt_key: Digest32,
    pub recovery: RecoveryReceipt,
    pub attempt_evidence: DurableAttemptEvidenceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurableEvidenceLocator {
    Known(Option<AttemptEvidenceHead>),
    Unavailable { diagnostic_digest: Digest32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurableAttemptEvidenceStatus {
    RecoveryCompleted(AttemptEvidenceHead),
    FinalizationIncomplete {
        last_durable_evidence: DurableEvidenceLocator,
        diagnostic_digest: Digest32,
    },
}

#[derive(Debug, Error)]
pub enum DurableBootstrapError {
    #[error("failed to open SQLite checkpoint store: {0}")]
    OpenCheckpointStore(#[source] SqliteFrontierError),
    #[error("failed to establish generation-zero authority frontier: {0}")]
    EstablishFrontier(#[source] FrontierError<SqliteFrontierError>),
}

#[derive(Debug, Error)]
pub enum DurableRecoveryError {
    #[error("verified authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("verified authority state failed: {0}")]
    AuthorityState(#[from] AuthorityStateError),
    #[error("verified executor workload failed at effect entry: {0}")]
    Workload(#[from] WorkloadIdentityError),
    #[error("verified Xenia proof belongs to a different capability grant")]
    VerifiedGrantMismatch,
    #[error("Xenia proof expired before effect entry")]
    XeniaProofExpiredAtEffectEntry,
    #[error("Agency Kernel frontier advanced after Xenia proof verification")]
    AuthorityFrontierAdvanced,
    #[error("failed to open durable attempt journal: {0}")]
    OpenAttemptJournal(#[source] SqliteAttemptEvidenceError),
    #[error("failed to restore typed broker at the authorized frontier: {0}")]
    BrokerRestore(#[source] BrokerError),
    #[error(
        "typed broker failed after admission; attempt {attempt_key:?}; durable attempt evidence locator {attempt_evidence:?}: {source}"
    )]
    BrokerAttempt {
        #[source]
        source: BrokerError,
        attempt_key: Digest32,
        attempt_evidence: DurableEvidenceLocator,
    },
}

#[derive(Debug, Serialize)]
struct XeniaAuthorityEvidenceCommitmentV1 {
    schema_version: u16,
    authorization_id: [u8; 16],
    session_id: [u8; 16],
    grant_digest: Digest32,
    workload_digest: Digest32,
    workload_process_digest: Digest32,
    workload_policy_digest: [u8; 32],
    workload_time_policy_digest: [u8; 32],
    ledger_entry_count: u64,
    ledger_head_hash: [u8; 32],
    prior_checkpoint: CheckpointHead,
    authority_state_digest: Digest32,
    authority_state_sequence: u64,
    authority_state_policy_digest: [u8; 32],
    authority_state_time_policy_digest: [u8; 32],
    expires_at_unix_s: u64,
}

fn xenia_authority_evidence_digest(
    grant_digest: Digest32,
    verified: &VerifiedXeniaCapability,
) -> Digest32 {
    let (ledger_entry_count, ledger_head_hash) = verified.xenia_frontier();
    let workload_process_digest = verified
        .executor_workload()
        .process()
        .digest()
        .expect("verified executor process commitment must remain valid");
    let commitment = XeniaAuthorityEvidenceCommitmentV1 {
        schema_version: 1,
        authorization_id: verified.authorization_id(),
        session_id: verified.session_id(),
        grant_digest,
        workload_digest: verified.workload_digest(),
        workload_process_digest,
        workload_policy_digest: verified.executor_workload().workload_policy_digest(),
        workload_time_policy_digest: verified.executor_workload().time_policy_digest(),
        ledger_entry_count,
        ledger_head_hash,
        prior_checkpoint: verified.prior_checkpoint(),
        authority_state_digest: verified.authority_state_digest(),
        authority_state_sequence: verified.authority_state_sequence(),
        authority_state_policy_digest: verified.authority_state().state_policy_digest(),
        authority_state_time_policy_digest: verified.authority_state().time_policy_digest(),
        expires_at_unix_s: verified.expires_at_unix_s(),
    };
    let encoded = bincode::serialize(&commitment)
        .expect("fixed Xenia authority evidence commitment must serialize");
    let mut hasher = blake3::Hasher::new();
    hasher.update(XENIA_AUTHORITY_EVIDENCE_DOMAIN);
    hasher.update(&encoded);
    Digest32(*hasher.finalize().as_bytes())
}

fn diagnostic(error: &impl std::fmt::Display) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.xenia-systemd.durable-profile.diagnostic.v0.3\0");
    hasher.update(error.to_string().as_bytes());
    Digest32(*hasher.finalize().as_bytes())
}
