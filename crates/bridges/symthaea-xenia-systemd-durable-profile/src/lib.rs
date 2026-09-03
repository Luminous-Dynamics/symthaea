// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable one-use Xenia-authorized systemd recovery profile.
//!
//! This profile composes the existing semantic/security layers rather than
//! replacing them:
//! - Xenia proof verification happens upstream in `symthaea-xenia-authority`;
//! - #305 remains the only systemd execution/accounting state machine;
//! - #316 supplies checkpoint CAS semantics;
//! - #320 supplies the concrete SQLite CAS backend;
//! - #326 supplies durable write-ahead attempt evidence.

#![deny(unsafe_code)]

use std::path::{Path, PathBuf};

use serde::Serialize;
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_action_runtime::{ExecutionId, ReservationId};
use symthaea_authority::{AuthorityContext, CapabilityGrant, Digest32, NegativeAuthorityFact};
use symthaea_authority_frontier::{
    CasCheckpointStoreAdapter, EstablishedGrantFrontier, FrontierError, establish_grant_frontier,
};
use symthaea_authority_frontier_sqlite::{SqliteCheckpointCasStore, SqliteFrontierError};
use symthaea_system_attempt_evidence::{
    AttemptEvidenceContext, AttemptEvidenceHead, SqliteAttemptEvidenceError,
    SqliteAttemptEvidenceJournal, instrument_attempt,
};
use symthaea_system_broker::{
    BrokerError, RecoveryReceipt, RestartPlan, ServiceBackend, SystemdRecoveryBroker,
};
use symthaea_xenia_authority::VerifiedXeniaCapability;
use thiserror::Error;

const XENIA_AUTHORITY_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.xenia-systemd.authority-evidence.v0.2\0";

/// Bootstrap object that exists before Xenia issues authority.
///
/// Generation zero is already durable when this object is returned. Callers
/// should obtain Xenia authorization bound to [`Self::authorization_checkpoint_head`]
/// and then consume this object through [`Self::recover_verified_once`].
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
    /// Create a fresh generation-zero Agency Kernel frontier in the SQLite
    /// state database. The database must not already contain a frontier for a
    /// previous session; restore/recovery is intentionally a separate future API.
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

    /// Exact generation-zero checkpoint Xenia must bind.
    pub fn authorization_checkpoint_head(&self) -> CheckpointHead {
        self.frontier.head
    }

    /// Exact generation-zero checkpoint payload corresponding to the head.
    pub fn authorization_checkpoint(&self) -> &symthaea_action_checkpoint::GrantAccountCheckpoint {
        &self.frontier.checkpoint
    }

    /// Consume the one-use bootstrap plus one verified Xenia capability and run
    /// the existing typed broker with SQLite CAS + durable attempt evidence.
    ///
    /// Consuming `self` is deliberate: a V0.2 object represents one exact
    /// single-use grant lineage. Durable replay prevention still comes from CAS,
    /// not from Rust move semantics alone.
    #[allow(clippy::too_many_arguments)]
    pub fn recover_verified_once(
        self,
        verified: VerifiedXeniaCapability,
        plan: &RestartPlan,
        execution_id: ExecutionId,
        reservation_id: ReservationId,
        authority_context: AuthorityContext,
        negative_facts: &[NegativeAuthorityFact],
    ) -> Result<DurableXeniaSystemdReceipt, DurableRecoveryError> {
        if verified.grant_digest() != self.grant_digest {
            return Err(DurableRecoveryError::VerifiedGrantMismatch);
        }
        if authority_context.now_unix_s > verified.expires_at_unix_s() {
            return Err(DurableRecoveryError::XeniaProofExpiredAtEffectEntry);
        }
        if verified.prior_checkpoint() != self.frontier.head {
            return Err(DurableRecoveryError::AuthorityFrontierAdvanced);
        }

        let xenia_authorization_id = verified.authorization_id();
        let xenia_session_id = verified.session_id();
        let workload_digest = verified.workload_digest();
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

        // The affine verifier proof has now been reduced to fixed evidence
        // commitments. Durable one-use semantics come from the next CAS-backed
        // reservation checkpoint, not from retaining the Rust proof value.
        drop(verified);

        let recovery = match broker.recover_once(
            plan,
            execution_id,
            reservation_id,
            authority_context,
            negative_facts,
        ) {
            Ok(receipt) => receipt,
            Err(source) => {
                let attempt_evidence_head = evidence_handle.latest_head().unwrap_or(None);
                return Err(DurableRecoveryError::BrokerAttempt {
                    source,
                    attempt_key,
                    attempt_evidence_head,
                });
            }
        };

        let attempt_evidence = match evidence_handle.append_recovery_receipt(&recovery) {
            Ok(head) => DurableAttemptEvidenceStatus::RecoveryCompleted(head),
            Err(error) => DurableAttemptEvidenceStatus::FinalizationIncomplete {
                last_durable_head: evidence_handle.latest_head().unwrap_or(None),
                diagnostic_digest: diagnostic(&error),
            },
        };

        Ok(DurableXeniaSystemdReceipt {
            xenia_authorization_id,
            xenia_session_id,
            xenia_ledger_entry_count,
            xenia_ledger_head_hash,
            workload_digest,
            authority_evidence_digest,
            attempt_key,
            recovery,
            attempt_evidence,
        })
    }
}

/// Success receipt joining Xenia provenance, #305 accounting, and attempt evidence.
#[derive(Debug)]
pub struct DurableXeniaSystemdReceipt {
    pub xenia_authorization_id: [u8; 16],
    pub xenia_session_id: [u8; 16],
    pub xenia_ledger_entry_count: u64,
    pub xenia_ledger_head_hash: [u8; 32],
    pub workload_digest: Digest32,
    pub authority_evidence_digest: Digest32,
    pub attempt_key: Digest32,
    pub recovery: RecoveryReceipt,
    pub attempt_evidence: DurableAttemptEvidenceStatus,
}

/// Whether broker-level success could also be appended as the final attempt
/// evidence record. A finalization failure does not erase the earlier durable
/// `DispatchArmed`/dispatch classification records or the successful #305
/// accounting checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurableAttemptEvidenceStatus {
    RecoveryCompleted(AttemptEvidenceHead),
    FinalizationIncomplete {
        last_durable_head: Option<AttemptEvidenceHead>,
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
        "typed broker failed after admission; attempt {attempt_key:?}; latest durable attempt evidence: {attempt_evidence_head:?}: {source}"
    )]
    BrokerAttempt {
        #[source]
        source: BrokerError,
        attempt_key: Digest32,
        attempt_evidence_head: Option<AttemptEvidenceHead>,
    },
}

#[derive(Debug, Serialize)]
struct XeniaAuthorityEvidenceCommitmentV1 {
    schema_version: u16,
    authorization_id: [u8; 16],
    session_id: [u8; 16],
    grant_digest: Digest32,
    workload_digest: Digest32,
    ledger_entry_count: u64,
    ledger_head_hash: [u8; 32],
    prior_checkpoint: CheckpointHead,
    expires_at_unix_s: u64,
}

fn xenia_authority_evidence_digest(
    grant_digest: Digest32,
    verified: &VerifiedXeniaCapability,
) -> Digest32 {
    let (ledger_entry_count, ledger_head_hash) = verified.xenia_frontier();
    let commitment = XeniaAuthorityEvidenceCommitmentV1 {
        schema_version: 1,
        authorization_id: verified.authorization_id(),
        session_id: verified.session_id(),
        grant_digest,
        workload_digest: verified.workload_digest(),
        ledger_entry_count,
        ledger_head_hash,
        prior_checkpoint: verified.prior_checkpoint(),
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
    hasher.update(b"symthaea.xenia-systemd.durable-profile.diagnostic.v0.2\0");
    hasher.update(error.to_string().as_bytes());
    Digest32(*hasher.finalize().as_bytes())
}
