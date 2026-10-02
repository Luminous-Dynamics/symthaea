// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Durable shared authorization-consumption domain.
//!
//! SQLite is the authoritative reservation/consumption state machine. The
//! external effect remains a separate boundary: an uncertain effect becomes
//! Indeterminate and requires explicit reconciliation.

use std::path::{Path, PathBuf};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

use super::{
    ActionAuthorizationWitness, AuthorizationConsumptionError, AuthorizationLease,
    AuthorizationLeaseState, EpistemicAction, ExecutionOutcome, ExecutionReceipt,
};

#[derive(Debug)]
pub enum AuthorizationStoreError {
    Sqlite(rusqlite::Error),
    Consumption(AuthorizationConsumptionError),
    NotFound(String),
    InvalidState(String),
}
impl std::fmt::Display for AuthorizationStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "SQLite error: {e}"),
            Self::Consumption(e) => write!(f, "authorization consumption error: {e:?}"),
            Self::NotFound(id) => write!(f, "authorization lease not found: {id}"),
            Self::InvalidState(s) => write!(f, "invalid persisted authorization state: {s}"),
        }
    }
}
impl std::error::Error for AuthorizationStoreError {}
impl From<rusqlite::Error> for AuthorizationStoreError {
    fn from(e: rusqlite::Error) -> Self { Self::Sqlite(e) }
}
impl From<AuthorizationConsumptionError> for AuthorizationStoreError {
    fn from(e: AuthorizationConsumptionError) -> Self { Self::Consumption(e) }
}

/// Relying-party authorization for recovering one exact attempt.
///
/// The authority/authentication layer is responsible for validating the issuer
/// and policy. The durable store enforces only the non-negotiable structural
/// binding: one authorization instance, one attempt, one boundary, one action
/// digest, and one authority epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryAuthorizationWitness {
    pub authorization_instance: String,
    pub attempt_id: String,
    pub boundary_id: String,
    pub action_digest: String,
    pub policy: String,
    pub authority_epoch: u64,
    pub issued_at: String,
}

impl RecoveryAuthorizationWitness {
    pub fn is_bound_to(&self, lease: &AuthorizationLease) -> bool {
        !self.authorization_instance.is_empty()
            && !self.attempt_id.is_empty()
            && !self.boundary_id.is_empty()
            && !self.policy.is_empty()
            && !self.issued_at.is_empty()
            && self.authorization_instance == lease.authorization_instance
            && self.action_digest == lease.action_digest
            && self.authority_epoch == lease.authority_epoch
    }
}

/// A durable shared consumption domain. Each operation uses a fresh connection,
/// allowing independent processes to contend on the same SQLite state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
/// The purpose of provider evidence presented to the verifier. A pre-entry lookup
/// is intentionally not interchangeable with terminal outcome evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderEvidenceKind {
    TerminalOutcome,
    PreEntryLookup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderTerminalEvidence {
    pub kind: ProviderEvidenceKind,
    pub outcome: ExecutionOutcome,
    /// Stable provider-side evidence identifier, if the provider exposes one.
    pub evidence_id: String,
    /// Digest of the authenticated provider evidence payload.
    pub evidence_digest: String,
    pub action_digest: String,
    pub attempt_id: String,
    pub provider_idempotency_key: String,
    pub target_identity: String,
    pub audience: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedProviderOutcome {
    pub evidence: ProviderTerminalEvidence,
    /// Relying-party configured verifier identity/revision.
    pub verifier_id: String,
    /// Digest of the verifier's authenticated verification statement.
    pub verification_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderVerificationError {
    InvalidEvidenceKind,
    InvalidBinding,
    IndeterminateNotTerminal,
    VerificationFailed,
}

/// The verifier owns provider authentication and semantic authority. The durable
/// store deliberately does not pretend that an adapter's local enum is provider
/// truth; it only accepts a verifier result that explicitly affirms terminal
/// evidence for the exact frozen dispatch record.
pub trait ProviderEvidenceVerifier {
    fn verify_terminal_outcome(
        &self,
        record: &DurableDispatchRecord,
        evidence: &ProviderTerminalEvidence,
    ) -> Result<VerifiedProviderOutcome, ProviderVerificationError>;
}

pub struct DurableDispatchRecord {
    pub authorization_instance: String,
    pub attempt_id: String,
    pub action_id: String,
    pub action_digest: String,
    pub provider_idempotency_key: String,
    pub target_identity: String,
    pub audience: String,
    pub adapter: String,
    /// Stable boundary identity. It scopes attempt ownership without entering
    /// the shared action key, so separate boundary instances cannot claim one
    /// another's dispatch records.
    pub boundary_id: String,
}

impl DurableDispatchRecord {
    pub fn new(
        authorization_instance: impl Into<String>,
        attempt_id: impl Into<String>,
        action_id: impl Into<String>,
        action_digest: impl Into<String>,
        provider_idempotency_key: impl Into<String>,
        effect: &super::ActionEffectBinding,
        boundary_id: impl Into<String>,
    ) -> Self {
        Self {
            authorization_instance: authorization_instance.into(),
            attempt_id: attempt_id.into(),
            action_id: action_id.into(),
            action_digest: action_digest.into(),
            provider_idempotency_key: provider_idempotency_key.into(),
            target_identity: effect.target_identity.clone(),
            audience: effect.audience.clone(),
            adapter: effect.adapter.clone(),
            boundary_id: boundary_id.into(),
        }
    }
}

pub struct SqliteAuthorizationStore { path: PathBuf }

impl SqliteAuthorizationStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AuthorizationStoreError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AuthorizationStoreError::InvalidState(e.to_string()))?;
        }
        let store = Self { path };
        let mut connection = store.connection()?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;",
        )?;
        migrate_legacy_schema(&mut connection)?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS authorization_leases (
               authorization_instance TEXT PRIMARY KEY,
               action_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               support_digest TEXT NOT NULL,
               policy TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               remaining_executions INTEGER NOT NULL,
               state TEXT NOT NULL,
               attempt_id TEXT,
               boundary_id TEXT
             );
             CREATE TABLE IF NOT EXISTS authorization_receipts (
               authorization_instance TEXT NOT NULL,
               action_id TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               phase TEXT NOT NULL,
               outcome TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               boundary_id TEXT,
               PRIMARY KEY(authorization_instance, attempt_id, phase)
             );
             CREATE TABLE IF NOT EXISTS authorization_recovery_markers (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               boundary_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               marker TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id, marker)
             );
             CREATE TABLE IF NOT EXISTS authorization_terminal_evidence (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               boundary_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               provider_idempotency_key TEXT NOT NULL,
               target_identity TEXT NOT NULL,
               audience TEXT NOT NULL,
               outcome TEXT NOT NULL,
               evidence_id TEXT NOT NULL,
               evidence_digest TEXT NOT NULL,
               verifier_id TEXT NOT NULL,
               verification_digest TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id)
             );
             CREATE TABLE IF NOT EXISTS authorization_dispatches (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               action_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               provider_idempotency_key TEXT NOT NULL,
               target_identity TEXT NOT NULL,
               audience TEXT NOT NULL,
               adapter TEXT NOT NULL,
               boundary_id TEXT NOT NULL,
               state TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id)
             );",
        )?;
        ensure_column(&mut connection, "authorization_leases", "boundary_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_receipts", "boundary_id", "TEXT")?;
        connection.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS authorization_lease_attempt_id_uq
               ON authorization_leases(attempt_id)
               WHERE boundary_id IS NOT NULL AND attempt_id IS NOT NULL;
             CREATE UNIQUE INDEX IF NOT EXISTS authorization_dispatch_attempt_id_uq
               ON authorization_dispatches(attempt_id)
               WHERE boundary_id IS NOT NULL AND attempt_id IS NOT NULL;",
        )?;
        Ok(store)
    }

    fn connection(&self) -> Result<Connection, AuthorizationStoreError> {
        Ok(Connection::open(&self.path)?)
    }

    pub fn register_lease(&self, lease: &AuthorizationLease) -> Result<(), AuthorizationStoreError> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT INTO authorization_leases
             (authorization_instance, action_id, action_digest, support_digest, policy, authority_epoch,
              remaining_executions, state, attempt_id, boundary_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL)",
            params![
                lease.authorization_instance, lease.action_id, lease.action_digest,
                lease.support_digest, lease.policy, lease.authority_epoch as i64,
                lease.remaining_executions as i64, encode_state(&lease.state),
                state_attempt(&lease.state)
            ],
        )?;
        Ok(())
    }

    pub fn prepare_for_execution(
        &self, witness: &ActionAuthorizationWitness, action: &EpistemicAction,
        current_frame: &str, attempt_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let bound: Option<String> = tx
            .query_row(
                "SELECT boundary_id FROM authorization_leases
                 WHERE authorization_instance=?1",
                params![witness.authorization_instance.as_str()],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        if bound.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;
        lease.prepare_for_execution(witness, action, current_frame, attempt_id)?;
        tx.execute(
            "UPDATE authorization_leases SET state='prepared', attempt_id=?2
             WHERE authorization_instance=?1 AND state='ready' AND remaining_executions>0
               AND boundary_id IS NULL",
            params![witness.authorization_instance.as_str(), attempt_id],
        )?;
        if tx.changes() != 1 {
            return Err(AuthorizationConsumptionError::NotReady.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Prepare an effectful attempt with durable boundary ownership.
    ///
    /// Persisting the boundary while the lease is still Prepared means a crash
    /// before DispatchPending cannot leave the reservation ownerless.
    /// Bound attempt IDs are unique within this durable state domain.
    pub fn prepare_for_execution_bound(
        &self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        current_frame: &str,
        attempt_id: &str,
        boundary_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if boundary_id.is_empty() || attempt_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let prior_owner: Option<(String, String)> = tx
            .query_row(
                "SELECT authorization_instance,boundary_id
                 FROM authorization_leases
                 WHERE attempt_id=?1 AND boundary_id IS NOT NULL
                   AND authorization_instance<>?2",
                params![attempt_id, witness.authorization_instance.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if prior_owner.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;
        lease.prepare_for_execution(witness, action, current_frame, attempt_id)?;

        let changed = tx.execute(
            "UPDATE authorization_leases
             SET state='prepared', attempt_id=?2, boundary_id=?3
             WHERE authorization_instance=?1 AND state='ready'
               AND remaining_executions>0 AND boundary_id IS NULL",
            params![witness.authorization_instance.as_str(), attempt_id, boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::NotReady.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Durably cross the pre-dispatch fence. This must commit before the
    /// executor enters the external effect sink.
    pub fn mark_dispatch_pending(
        &self, authorization_instance: &str, attempt_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if load_lease_boundary(&tx, authorization_instance)?.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        lease.mark_dispatch_pending(attempt_id)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='dispatch_pending', attempt_id=?2
             WHERE authorization_instance=?1 AND state='prepared' AND attempt_id=?2",
            params![authorization_instance, attempt_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Create the immutable provider-entry record and cross the pre-dispatch
    /// fence in one durable transaction. Effectful callers should use this path
    /// rather than the legacy attempt-only transition because it freezes every
    /// material field that the later provider boundary must consume.
    pub fn mark_dispatch_pending_bound(
        &self,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::ActionEffectBinding,
        boundary_id: &str,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        if action.effect_binding.as_ref() != Some(expected_effect) || boundary_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let prior_owner: Option<(String, String)> = tx
            .query_row(
                "SELECT authorization_instance,boundary_id
                 FROM authorization_leases
                 WHERE attempt_id=?1 AND boundary_id IS NOT NULL
                   AND authorization_instance<>?2",
                params![attempt_id, authorization_instance],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if prior_owner.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let current_boundary = load_lease_boundary(&tx, authorization_instance)?;
        if current_boundary.as_deref().is_some_and(|id| id != boundary_id) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let expected_digest = action.canonical_action_digest();
        if lease.action_id != action.id || lease.action_digest != expected_digest {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        lease.mark_dispatch_pending(attempt_id)?;
        let record = DurableDispatchRecord::new(
            authorization_instance,
            attempt_id,
            &action.id,
            &expected_digest,
            &lease.provider_idempotency_key(),
            expected_effect,
            boundary_id,
        );
        if current_boundary.is_none() {
            tx.execute(
                "UPDATE authorization_leases SET boundary_id=?2
                 WHERE authorization_instance=?1 AND state='prepared'
                   AND attempt_id=?3 AND boundary_id IS NULL",
                params![authorization_instance, boundary_id, attempt_id],
            )?;
        }
        tx.execute(
            "INSERT INTO authorization_dispatches
             (authorization_instance,attempt_id,action_id,action_digest,provider_idempotency_key,
              target_identity,audience,adapter,boundary_id,state)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,'dispatch_pending')",
            params![record.authorization_instance, record.attempt_id, record.action_id,
                record.action_digest, record.provider_idempotency_key, record.target_identity,
                record.audience, record.adapter, record.boundary_id],
        )?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='dispatch_pending', attempt_id=?2
             WHERE authorization_instance=?1 AND state='prepared' AND attempt_id=?2",
            params![authorization_instance, attempt_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        tx.commit()?;
        Ok(record)
    }

    /// Durably record provider entry after DispatchPending has committed.
    /// Cross the provider-entry boundary only for the exact immutable record
    /// created before dispatch. All identity/effect fields are checked again,
    /// preventing a stale executor from substituting a different sink or key.
    pub fn mark_invoked_bound(
        &self, record: &DurableDispatchRecord,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx.query_row(
            "SELECT action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,boundary_id,state
             FROM authorization_dispatches WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((
                r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?,
                r.get::<_,String>(3)?, r.get::<_,String>(4)?, r.get::<_,String>(5)?,
                r.get::<_,String>(6)?, r.get::<_,String>(7)?,
            )),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;
        if row.0 != record.action_id || row.1 != record.action_digest
            || row.2 != record.provider_idempotency_key || row.3 != record.target_identity
            || row.4 != record.audience || row.5 != record.adapter
            || row.6 != record.boundary_id || row.7 != "dispatch_pending"
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        lease.mark_invoked(&record.attempt_id)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='invoked', attempt_id=?2
             WHERE authorization_instance=?1 AND state='dispatch_pending' AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
        )?;
        if changed != 1 { return Err(AuthorizationConsumptionError::AttemptMismatch.into()); }
        let changed = tx.execute(
            "UPDATE authorization_dispatches SET state='invoked'
             WHERE authorization_instance=?1 AND attempt_id=?2 AND state='dispatch_pending'",
            params![record.authorization_instance, record.attempt_id],
        )?;
        if changed != 1 { return Err(AuthorizationConsumptionError::AttemptMismatch.into()); }
        tx.commit()?;
        Ok(())
    }

    pub fn mark_invoked(
        &self, authorization_instance: &str, attempt_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if load_lease_boundary(&tx, authorization_instance)?.is_some()
            || dispatch_boundary(&tx, authorization_instance, attempt_id)?.is_some()
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        lease.mark_invoked(attempt_id)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='invoked', attempt_id=?2
             WHERE authorization_instance=?1 AND state='dispatch_pending' AND attempt_id=?2",
            params![authorization_instance, attempt_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Commit an outcome only through the frozen dispatch record. The record
    /// is revalidated immediately before the lease transition so the provider
    /// outcome cannot be attached to a different sink contract.
    fn validate_verified_terminal_outcome(
        record: &DurableDispatchRecord,
        verified: &VerifiedProviderOutcome,
    ) -> Result<(), AuthorizationStoreError> {
        let evidence = &verified.evidence;
        if !matches!(evidence.kind, ProviderEvidenceKind::TerminalOutcome)
            || matches!(evidence.outcome, ExecutionOutcome::Indeterminate)
        {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        if verified.verifier_id.is_empty() || verified.verification_digest.is_empty()
            || evidence.evidence_id.is_empty() || evidence.evidence_digest.is_empty()
            || evidence.attempt_id != record.attempt_id
            || evidence.action_digest != record.action_digest
            || evidence.provider_idempotency_key != record.provider_idempotency_key
            || evidence.target_identity != record.target_identity
            || evidence.audience != record.audience
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(())
    }

    /// Commit a terminal provider outcome only after a relying-party configured
    /// verifier has authenticated and semantically bound the provider evidence
    /// to this exact frozen dispatch record.
    pub fn commit_bound_verified<V: ProviderEvidenceVerifier>(
        &self,
        record: &DurableDispatchRecord,
        evidence: &ProviderTerminalEvidence,
        verifier: &V,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let verified = verifier
            .verify_terminal_outcome(record, evidence)
            .map_err(|_| AuthorizationConsumptionError::ProviderEvidenceVerificationRequired)?;
        Self::validate_verified_terminal_outcome(record, &verified)?;

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx.query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![record.authorization_instance, record.attempt_id, record.boundary_id],
            |r| r.get::<_, String>(0),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;
        if !matches!(row.as_str(), "dispatch_pending" | "invoked" | "indeterminate") {
            if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
                return Ok(receipt);
            }
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }

        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.action_id != record.action_id
            || lease.action_digest != record.action_digest
            || lease.provider_idempotency_key() != record.provider_idempotency_key
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
            return Ok(receipt);
        }
        if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "indeterminate")? {
            return Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into());
        }

        let receipt = lease.commit(&record.attempt_id, evidence.outcome)?;
        update_lease_with_boundary(&tx, &lease, Some(&record.boundary_id))?;
        tx.execute(
            "INSERT OR REPLACE INTO authorization_terminal_evidence
             (authorization_instance,attempt_id,boundary_id,action_digest,provider_idempotency_key,
              target_identity,audience,outcome,evidence_id,evidence_digest,verifier_id,verification_digest)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            params![
                record.authorization_instance, record.attempt_id, record.boundary_id,
                record.action_digest, record.provider_idempotency_key, record.target_identity,
                record.audience,
                if matches!(evidence.outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                evidence.evidence_id, evidence.evidence_digest, verified.verifier_id,
                verified.verification_digest,
            ],
        )?;
        insert_receipt_with_boundary(&tx, &receipt, "final", Some(&record.boundary_id))?;
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?4
               AND state IN ('dispatch_pending','invoked','indeterminate')",
            params![record.authorization_instance, record.attempt_id,
                if matches!(evidence.outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                record.boundary_id],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    pub fn commit_bound(
        &self, record: &DurableDispatchRecord, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx.query_row(
            "SELECT action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,boundary_id,state
             FROM authorization_dispatches WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,
                  r.get::<_,String>(3)?,r.get::<_,String>(4)?,r.get::<_,String>(5)?,
                  r.get::<_,String>(6)?,r.get::<_,String>(7)?)),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;
        if row.0 != record.action_id || row.1 != record.action_digest
            || row.2 != record.provider_idempotency_key || row.3 != record.target_identity
            || row.4 != record.audience || row.5 != record.adapter || row.6 != record.boundary_id
            || !matches!(row.7.as_str(), "dispatch_pending" | "invoked" | "indeterminate" | "succeeded" | "failed")
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if !matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
            return Ok(r);
        }
        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "indeterminate")? {
            return if matches!(outcome, ExecutionOutcome::Indeterminate) { Ok(r) }
            else { Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into()) };
        }
        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.action_id != record.action_id || lease.action_digest != record.action_digest
            || lease.provider_idempotency_key() != record.provider_idempotency_key
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let receipt = lease.commit(&record.attempt_id, outcome)?;
        update_lease_with_boundary(&tx, &lease, Some(&record.boundary_id))?;
        let phase = if matches!(outcome, ExecutionOutcome::Indeterminate) { "indeterminate" } else { "final" };
        insert_receipt_with_boundary(&tx, &receipt, phase, Some(&record.boundary_id))?;
        let dispatch_state = match outcome {
            ExecutionOutcome::Succeeded => "succeeded",
            ExecutionOutcome::Failed => "failed",
            ExecutionOutcome::Indeterminate => "indeterminate",
        };
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id, dispatch_state],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    pub fn commit(
        &self, authorization_instance: &str, attempt_id: &str, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        if dispatch_boundary(&tx, authorization_instance, attempt_id)?.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some(r) = load_receipt(&tx, authorization_instance, attempt_id, "final")? { return Ok(r); }
        if let Some(r) = load_receipt(&tx, authorization_instance, attempt_id, "indeterminate")? {
            return if matches!(outcome, ExecutionOutcome::Indeterminate) {
                Ok(r)
            } else {
                Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into())
            };
        }

        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let receipt = lease.commit(attempt_id, outcome)?;
        update_lease(&tx, &lease)?;
        let phase = if matches!(outcome, ExecutionOutcome::Indeterminate) { "indeterminate" } else { "final" };
        insert_receipt(&tx, &receipt, phase)?;
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![authorization_instance, attempt_id, if matches!(outcome, ExecutionOutcome::Indeterminate) { "indeterminate" } else if matches!(outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" }],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    pub fn reconcile_indeterminate(
        &self, authorization_instance: &str, attempt_id: &str, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        if matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if dispatch_boundary(&tx, authorization_instance, attempt_id)?.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if let Some(r) = load_receipt(&tx, authorization_instance, attempt_id, "reconciled")? { return Ok(r); }

        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let receipt = lease.reconcile_indeterminate(attempt_id, outcome)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state=?2, attempt_id=?3, remaining_executions=?4
             WHERE authorization_instance=?1 AND state='indeterminate' AND attempt_id=?5",
            params![
                authorization_instance, encode_state(&lease.state), state_attempt(&lease.state),
                lease.remaining_executions as i64, attempt_id
            ],
        )?;
        if changed != 1 { return Err(AuthorizationConsumptionError::AttemptMismatch.into()); }
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND state='indeterminate'",
            params![authorization_instance, attempt_id,
                if matches!(outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" }],
        )?;
        insert_receipt(&tx, &receipt, "reconciled")?;
        tx.commit()?;
        Ok(receipt)
    }

    /// Recover one exact Prepared attempt that is proven to have stopped
    /// before DispatchPending.
    ///
    /// This is not outcome reconciliation. It atomically marks the attempt as
    /// not entered, releases its reservation back to Ready, and records the
    /// explicit not-entered marker. A later dispatch using the old attempt ID
    /// therefore fails rather than continuing after recovery.
    pub fn recover_pre_dispatch_attempt(
        &self,
        witness: &RecoveryAuthorizationWitness,
    ) -> Result<bool, AuthorizationStoreError> {
        if witness.boundary_id.is_empty() || witness.attempt_id.is_empty() {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;

        if !witness.is_bound_to(&lease) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some((boundary_id, action_digest)) = tx
            .query_row(
                "SELECT boundary_id,action_digest
                 FROM authorization_recovery_markers
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND marker='not_entered'",
                params![witness.authorization_instance.as_str(), witness.attempt_id.as_str()],
                |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?)),
            )
            .optional()?
        {
            if boundary_id == witness.boundary_id && action_digest == witness.action_digest {
                return Ok(false);
            }
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if !matches!(
            &lease.state,
            AuthorizationLeaseState::Prepared { attempt_id } if attempt_id == &witness.attempt_id
        ) {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        if dispatch_boundary(&tx, &witness.authorization_instance, &witness.attempt_id)?.is_some() {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        let changed = tx.execute(
            "UPDATE authorization_leases
             SET state='ready', attempt_id=NULL, boundary_id=NULL
             WHERE authorization_instance=?1 AND state='prepared'
               AND attempt_id=?2 AND boundary_id=?3",
            params![
                witness.authorization_instance.as_str(),
                witness.attempt_id.as_str(),
                witness.boundary_id.as_str()
            ],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        tx.execute(
            "INSERT INTO authorization_recovery_markers
             (authorization_instance,attempt_id,boundary_id,action_digest,authority_epoch,marker)
             VALUES (?1,?2,?3,?4,?5,'not_entered')",
            params![
                witness.authorization_instance.as_str(),
                witness.attempt_id.as_str(),
                witness.boundary_id.as_str(),
                witness.action_digest.as_str(),
                witness.authority_epoch as i64
            ],
        )?;

        tx.commit()?;
        Ok(true)
    }

    /// Reconcile an indeterminate effect using its exact frozen dispatch record.
    ///
    /// The durable dispatch record and current lease must name the same boundary
    /// before this operation can consume the authorization budget.
    pub fn reconcile_indeterminate_bound(
        &self,
        record: &DurableDispatchRecord,
        outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        if matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let row = tx.query_row(
            "SELECT action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,boundary_id,state
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance.as_str(), record.attempt_id.as_str()],
            |r| Ok((
                r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?,
                r.get::<_,String>(3)?, r.get::<_,String>(4)?, r.get::<_,String>(5)?,
                r.get::<_,String>(6)?, r.get::<_,String>(7)?,
            )),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;

        if row.0 != record.action_id || row.1 != record.action_digest
            || row.2 != record.provider_idempotency_key || row.3 != record.target_identity
            || row.4 != record.audience || row.5 != record.adapter
            || row.6 != record.boundary_id || row.7 != "indeterminate"
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "reconciled")? {
            return Ok(r);
        }

        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.action_id != record.action_id || lease.action_digest != record.action_digest
            || lease.provider_idempotency_key() != record.provider_idempotency_key
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let receipt = lease.reconcile_indeterminate(&record.attempt_id, outcome)?;
        update_lease_with_boundary(&tx, &lease, Some(&record.boundary_id))?;

        let dispatch_state = if matches!(outcome, ExecutionOutcome::Succeeded) {
            "succeeded"
        } else {
            "failed"
        };
        let changed = tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2
               AND boundary_id=?4 AND state='indeterminate'",
            params![record.authorization_instance, record.attempt_id, dispatch_state, record.boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        insert_receipt_with_boundary(&tx, &receipt, "reconciled", Some(&record.boundary_id))?;
        tx.commit()?;
        Ok(receipt)
    }

    /// Crash recovery is deliberately conservative: a process may have reached
    /// the external sink after its last durable local write. Any non-terminal
    /// prepared/dispatch-pending reservation therefore becomes Indeterminate
    /// before another execution can be admitted.
    /// Recover one exact attempt owned by one execution boundary.
    ///
    /// This is the preferred recovery primitive for an external effect: the
    /// boundary and attempt are both supplied explicitly, so recovery cannot
    /// claim a sibling attempt merely because it shares the same boundary.
    pub fn recover_incomplete_attempt_for_boundary(
        &self,
        boundary_id: &str,
        attempt_id: &str,
    ) -> Result<usize, AuthorizationStoreError> {
        if boundary_id.is_empty() || attempt_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        self.recover_incomplete_attempts_scoped(Some(boundary_id), Some(attempt_id))
    }

    /// Recover all incomplete attempts owned by one boundary.
    ///
    /// This batch form is intended for executor startup maintenance. Any
    /// per-attempt recovery authorization should use the exact-attempt API.
    pub fn recover_incomplete_attempts_for_boundary(
        &self,
        boundary_id: &str,
    ) -> Result<usize, AuthorizationStoreError> {
        if boundary_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        self.recover_incomplete_attempts_scoped(Some(boundary_id), None)
    }

    /// Recover only legacy/unscoped attempts. Bound effectful attempts should use
    /// one of the boundary-scoped recovery paths.
    pub fn recover_incomplete_attempts(&self) -> Result<usize, AuthorizationStoreError> {
        self.recover_incomplete_attempts_scoped(None, None)
    }

    fn recover_incomplete_attempts_scoped(
        &self,
        boundary_filter: Option<&str>,
        attempt_filter: Option<&str>,
    ) -> Result<usize, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut recovered = Vec::new();
        {
            let (mut stmt, query_params) = match (boundary_filter, attempt_filter) {
                (Some(_), Some(_)) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id=?1 AND attempt_id=?2",
                    )?,
                    vec![
                        boundary_filter.unwrap().to_owned(),
                        attempt_filter.unwrap().to_owned(),
                    ],
                ),
                (Some(_), None) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('prepared','dispatch_pending','invoked')
                           AND boundary_id=?1",
                    )?,
                    vec![boundary_filter.unwrap().to_owned()],
                ),
                (None, Some(_)) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id IS NULL AND attempt_id=?1",
                    )?,
                    vec![attempt_filter.unwrap().to_owned()],
                ),
                (None, None) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id IS NULL",
                    )?,
                    Vec::new(),
                ),
            };
            let rows = stmt.query_map(rusqlite::params_from_iter(query_params.iter()), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)? as u64,
                    row.get::<_, Option<String>>(5)?,
                ))
            })?;
            for row in rows {
                recovered.push(row?);
            }
        }

        for (instance, action_id, attempt_id, action_digest, authority_epoch, lease_boundary) in &recovered {
            if let Some(boundary_id) = lease_boundary.as_deref() {
                match dispatch_boundary(&tx, instance, attempt_id)? {
                    Some(dispatch_boundary_id) if dispatch_boundary_id == boundary_id => {}
                    Some(_) => return Err(AuthorizationConsumptionError::InvalidBinding.into()),
                    None => return Err(AuthorizationConsumptionError::InvalidBinding.into()),
                }
            }
            tx.execute(
                "UPDATE authorization_leases
                 SET state='indeterminate', attempt_id=?2
                 WHERE authorization_instance=?1
                   AND state IN ('prepared','dispatch_pending','invoked')
                   AND attempt_id=?2",
                params![instance, attempt_id],
            )?;
            let receipt = ExecutionReceipt {
                action_id: action_id.clone(),
                authorization_instance: instance.clone(),
                action_digest: action_digest.clone(),
                provider_idempotency_key: {
                    let lease = AuthorizationLease::new_with_instance(
                        instance.clone(),
                        action_id.clone(),
                        action_digest.clone(),
                        String::new(),
                        String::new(),
                        *authority_epoch,
                        1,
                    );
                    lease.provider_idempotency_key()
                },
                attempt_id: attempt_id.clone(),
                authority_epoch: *authority_epoch,
                outcome: ExecutionOutcome::Indeterminate,
            };
            tx.execute(
                "UPDATE authorization_dispatches SET state='indeterminate'
                 WHERE authorization_instance=?1 AND attempt_id=?2
                   AND state IN ('dispatch_pending','invoked')
                   AND (?3 IS NULL OR boundary_id=?3)",
                params![instance, attempt_id, lease_boundary],
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO authorization_receipts
                 (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,boundary_id)
                 VALUES (?1,?2,?3,'indeterminate','indeterminate',?4,?5,?6)",
                params![
                    receipt.authorization_instance,
                    receipt.action_id,
                    receipt.attempt_id,
                    receipt.action_digest,
                    receipt.authority_epoch as i64,
                    lease_boundary
                ],
            )?;
        }
        tx.commit()?;
        Ok(recovered.len())
    }
}

fn migrate_legacy_schema(connection: &mut Connection) -> Result<(), AuthorizationStoreError> {
    let has_lease_table: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='authorization_leases'
            )",
            [],
            |row| row.get(0),
        )?;

    if !has_lease_table {
        return Ok(());
    }

    let has_instance: bool = connection
        .prepare("PRAGMA table_info(authorization_leases)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|name| name == "authorization_instance");

    if has_instance {
        return Ok(());
    }

    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch(
        "ALTER TABLE authorization_leases RENAME TO authorization_leases_legacy;
         ALTER TABLE authorization_receipts RENAME TO authorization_receipts_legacy;

         CREATE TABLE authorization_leases (
           authorization_instance TEXT PRIMARY KEY,
           action_id TEXT NOT NULL,
           action_digest TEXT NOT NULL,
           support_digest TEXT NOT NULL,
           policy TEXT NOT NULL,
           authority_epoch INTEGER NOT NULL,
           remaining_executions INTEGER NOT NULL,
           state TEXT NOT NULL,
           attempt_id TEXT,
           boundary_id TEXT
         );

         CREATE TABLE authorization_receipts (
           authorization_instance TEXT NOT NULL,
           action_id TEXT NOT NULL,
           attempt_id TEXT NOT NULL,
           phase TEXT NOT NULL,
           outcome TEXT NOT NULL,
           action_digest TEXT NOT NULL,
           authority_epoch INTEGER NOT NULL,
           boundary_id TEXT,
           PRIMARY KEY(authorization_instance, attempt_id, phase)
         );",
    )?;
    tx.execute(
        "INSERT INTO authorization_leases
         (authorization_instance,action_id,action_digest,support_digest,policy,
          authority_epoch,remaining_executions,state,attempt_id,boundary_id)
         SELECT action_id,action_id,action_digest,support_digest,policy,
                authority_epoch,remaining_executions,state,attempt_id,NULL
         FROM authorization_leases_legacy",
        [],
    )?;
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,boundary_id)
         SELECT action_id,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,NULL
         FROM authorization_receipts_legacy",
        [],
    )?;
    tx.execute_batch(
        "DROP TABLE authorization_leases_legacy;
         DROP TABLE authorization_receipts_legacy;",
    )?;
    tx.commit()?;
    Ok(())
}

fn encode_state(s: &AuthorizationLeaseState) -> &'static str {
    match s {
        AuthorizationLeaseState::Ready => "ready",
        AuthorizationLeaseState::Prepared { .. } => "prepared",
        AuthorizationLeaseState::DispatchPending { .. } => "dispatch_pending",
        AuthorizationLeaseState::Invoked { .. } => "invoked",
        AuthorizationLeaseState::Indeterminate { .. } => "indeterminate",
        AuthorizationLeaseState::Exhausted => "exhausted",
        AuthorizationLeaseState::Revoked => "revoked",
        AuthorizationLeaseState::Expired => "expired",
    }
}
fn state_attempt(s: &AuthorizationLeaseState) -> Option<&str> {
    match s {
        AuthorizationLeaseState::Prepared { attempt_id }
        | AuthorizationLeaseState::DispatchPending { attempt_id }
        | AuthorizationLeaseState::Invoked { attempt_id }
        | AuthorizationLeaseState::Indeterminate { attempt_id } => Some(attempt_id),
        _ => None,
    }
}
fn load_lease(tx: &Transaction<'_>, id: &str) -> Result<Option<AuthorizationLease>, rusqlite::Error> {
    tx.query_row(
        "SELECT authorization_instance,action_id,action_digest,support_digest,policy,authority_epoch,
                remaining_executions,state,attempt_id FROM authorization_leases
         WHERE authorization_instance=?1",
        params![id],
        |r| {
            let state: String = r.get(7)?;
            let attempt: Option<String> = r.get(8)?;
            let decoded = match state.as_str() {
                "ready" => AuthorizationLeaseState::Ready,
                "prepared" => AuthorizationLeaseState::Prepared {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "dispatch_pending" => AuthorizationLeaseState::DispatchPending {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "invoked" => AuthorizationLeaseState::Invoked {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "indeterminate" => AuthorizationLeaseState::Indeterminate {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "exhausted" => AuthorizationLeaseState::Exhausted,
                "revoked" => AuthorizationLeaseState::Revoked,
                "expired" => AuthorizationLeaseState::Expired,
                _ => return Err(rusqlite::Error::InvalidQuery),
            };
            Ok(AuthorizationLease {
                authorization_instance: r.get(0)?,
                action_id: r.get(1)?,
                action_digest: r.get(2)?,
                support_digest: r.get(3)?,
                policy: r.get(4)?,
                authority_epoch: r.get::<_,i64>(5)? as u64,
                remaining_executions: r.get::<_,i64>(6)? as u32,
                state: decoded,
            })
        },
    ).optional()
}
fn ensure_column(
    connection: &mut Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<(), AuthorizationStoreError> {
    let names = connection
        .prepare(&format!("PRAGMA table_info({table})"))?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if names.iter().any(|name| name == column) {
        return Ok(());
    }
    connection.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {declaration}"),
        [],
    )?;
    Ok(())
}

fn dispatch_boundary(
    tx: &Transaction<'_>,
    authorization_instance: &str,
    attempt_id: &str,
) -> Result<Option<String>, AuthorizationStoreError> {
    tx.query_row(
        "SELECT boundary_id FROM authorization_dispatches
         WHERE authorization_instance=?1 AND attempt_id=?2",
        params![authorization_instance, attempt_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

fn load_lease_boundary(
    tx: &Transaction<'_>,
    authorization_instance: &str,
) -> Result<Option<String>, AuthorizationStoreError> {
    tx.query_row(
        "SELECT boundary_id FROM authorization_leases
         WHERE authorization_instance=?1",
        params![authorization_instance],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

fn update_lease(tx: &Transaction<'_>, lease: &AuthorizationLease) -> Result<(), AuthorizationStoreError> {
    update_lease_with_boundary(tx, lease, None)
}

fn update_lease_with_boundary(
    tx: &Transaction<'_>,
    lease: &AuthorizationLease,
    boundary_id: Option<&str>,
) -> Result<(), AuthorizationStoreError> {
    let current_boundary = match lease.state {
        AuthorizationLeaseState::Ready
        | AuthorizationLeaseState::Exhausted
        | AuthorizationLeaseState::Revoked
        | AuthorizationLeaseState::Expired => None,
        _ => boundary_id,
    };
    let changed = tx.execute(
        "UPDATE authorization_leases
         SET state=?2,attempt_id=?3,remaining_executions=?4,boundary_id=?5
         WHERE authorization_instance=?1",
        params![
            lease.authorization_instance,
            encode_state(&lease.state),
            state_attempt(&lease.state),
            lease.remaining_executions as i64,
            current_boundary
        ],
    )?;
    if changed != 1 {
        return Err(AuthorizationStoreError::NotFound(lease.authorization_instance.clone()));
    }
    Ok(())
}

fn insert_receipt(tx: &Transaction<'_>, r: &ExecutionReceipt, phase: &str) -> Result<(), AuthorizationStoreError> {
    insert_receipt_with_boundary(tx, r, phase, None)
}

fn insert_receipt_with_boundary(
    tx: &Transaction<'_>,
    r: &ExecutionReceipt,
    phase: &str,
    boundary_id: Option<&str>,
) -> Result<(), AuthorizationStoreError> {
    let outcome = match r.outcome {
        ExecutionOutcome::Succeeded => "succeeded",
        ExecutionOutcome::Failed => "failed",
        ExecutionOutcome::Indeterminate => "indeterminate",
    };
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,boundary_id)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        params![
            r.authorization_instance, r.action_id, r.attempt_id, phase, outcome,
            r.action_digest, r.authority_epoch as i64, boundary_id
        ],
    )?;
    Ok(())
}
fn load_receipt(
    tx: &Transaction<'_>, authorization_instance: &str, attempt_id: &str, phase: &str,
) -> Result<Option<ExecutionReceipt>, AuthorizationStoreError> {
    tx.query_row(
        "SELECT authorization_instance,action_id,attempt_id,outcome,action_digest,authority_epoch
         FROM authorization_receipts
         WHERE authorization_instance=?1 AND attempt_id=?2 AND phase=?3",
        params![authorization_instance,attempt_id,phase],
        |r| {
            let outcome = match r.get::<_,String>(3)?.as_str() {
                "succeeded" => ExecutionOutcome::Succeeded,
                "failed" => ExecutionOutcome::Failed,
                "indeterminate" => ExecutionOutcome::Indeterminate,
                _ => return Err(rusqlite::Error::InvalidQuery),
            };
            let authorization_instance: String = r.get(0)?;
            let action_id: String = r.get(1)?;
            let attempt_id: String = r.get(2)?;
            let action_digest: String = r.get(4)?;
            let authority_epoch = r.get::<_,i64>(5)? as u64;
            let lease = AuthorizationLease::new_with_instance(
                authorization_instance.clone(),
                action_id.clone(),
                action_digest.clone(),
                String::new(),
                String::new(),
                authority_epoch,
                1,
            );
            Ok(ExecutionReceipt {
                authorization_instance,
                action_id,
                attempt_id,
                outcome,
                action_digest,
                provider_idempotency_key: lease.provider_idempotency_key(),
                authority_epoch,
            })
        },
    ).optional().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    fn fixture(path: &Path) -> (SqliteAuthorizationStore, EpistemicAction, ActionAuthorizationWitness) {
        let store = SqliteAuthorizationStore::open(path).unwrap();
        let action = EpistemicAction::new("durable-action","intervention",super::super::ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:00:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new(action.id.clone(),digest,"sha256:support","policy-v1",1,1)).unwrap();
        (store,action,witness)
    }

    #[test]
    fn durable_state_survives_reopen_and_blocks_replay() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-1").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-1").unwrap();
        let first=store.commit(&action.id,"attempt-1",ExecutionOutcome::Succeeded).unwrap();
        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        reopened.recover_incomplete_attempts().unwrap();
        assert!(matches!(
            reopened.prepare_for_execution(&witness,&action,"frame@1","attempt-2"),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::BudgetExhausted))
        ));
        assert_eq!(reopened.commit(&action.id,"attempt-1",ExecutionOutcome::Succeeded).unwrap(),first);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn explicit_authorization_instances_allow_fresh_issuance_without_replay() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-instance-{}.db",std::process::id()));
        let (store,action,old_witness)=fixture(&path);
        store.prepare_for_execution(&old_witness,&action,"frame@1","old-attempt").unwrap();
        store.mark_dispatch_pending(&old_witness.authorization_instance,"old-attempt").unwrap();
        store.commit(&action.id,"old-attempt",ExecutionOutcome::Succeeded).unwrap();

        let digest=action.canonical_action_digest();
        let fresh_witness=ActionAuthorizationWitness {
            authorization_instance:"approval-new".into(),
            issued_at:"2026-10-02T20:02:00Z".into(),
            ..old_witness
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-new", action.id.clone(), digest, "sha256:support", "policy-v1", 1, 1,
        )).unwrap();

        store.prepare_for_execution(&fresh_witness,&action,"frame@1","new-attempt").unwrap();
        store.mark_dispatch_pending("approval-new","new-attempt").unwrap();
        let receipt=store.commit("approval-new","new-attempt",ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.authorization_instance,"approval-new");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reopen_converts_nonterminal_dispatch_to_indeterminate() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-recovery-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-crash").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-crash").unwrap();
        drop(store);

        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        assert!(matches!(
            reopened.prepare_for_execution(&witness,&action,"frame@1","attempt-retry"),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::IndeterminateRequiresReconciliation
            ))
        ));
        assert_eq!(
            reopened.commit(&witness.authorization_instance,"attempt-crash",ExecutionOutcome::Succeeded)
                .unwrap_err().to_string(),
            "authorization consumption error: IndeterminateRequiresReconciliation"
        );
        let reconciled=reopened.reconcile_indeterminate(
            &witness.authorization_instance,"attempt-crash",ExecutionOutcome::Succeeded
        ).unwrap();
        assert_eq!(reconciled.outcome,ExecutionOutcome::Succeeded);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn invoked_state_is_durable_and_committable() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-invoked-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-invoked").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-invoked").unwrap();
        store.mark_invoked(&witness.authorization_instance,"attempt-invoked").unwrap();
        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        let receipt=reopened.commit(
            &witness.authorization_instance,"attempt-invoked",ExecutionOutcome::Succeeded
        ).unwrap();
        assert_eq!(receipt.outcome,ExecutionOutcome::Succeeded);
        assert_eq!(receipt.provider_idempotency_key, AuthorizationLease::new(
            action.id.clone(), action.canonical_action_digest(), "sha256:support", "policy-v1", 1, 1
        ).provider_idempotency_key());
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn bound_dispatch_record_freezes_effect_and_boundary_identity() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-bound-dispatch-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("bound-dispatch-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:00:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new(action.id.clone(),digest,"sha256:support","policy-v1",1,1)).unwrap();
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-bound").unwrap();
        let record=store.mark_dispatch_pending_bound(&witness.authorization_instance,"attempt-bound",&action,&effect,"boundary-A").unwrap();
        assert_eq!(record.target_identity,"target-A");
        let mut tampered=record.clone(); tampered.adapter="adapter-B".into();
        assert!(matches!(store.mark_invoked_bound(&tampered),Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))));
        store.mark_invoked_bound(&record).unwrap();
        assert!(store.mark_invoked_bound(&record).is_err());
        let mut wrong_key=record.clone();
        wrong_key.provider_idempotency_key.push_str("-tampered");
        assert!(matches!(
            store.commit_bound(&wrong_key,ExecutionOutcome::Succeeded),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        let receipt=store.commit_bound(&record,ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.provider_idempotency_key,record.provider_idempotency_key);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn legacy_transitions_cannot_bypass_boundary_owned_attempts() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-legacy-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("boundary-legacy-fence","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-legacy-fence".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:09:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-legacy-fence",action.id.clone(),digest,"sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-legacy-fence","boundary-A"
        ).unwrap();
        assert!(matches!(
            store.mark_dispatch_pending("approval-legacy-fence","attempt-legacy-fence"),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let record=store.mark_dispatch_pending_bound(
            "approval-legacy-fence","attempt-legacy-fence",&action,&effect,"boundary-A"
        ).unwrap();
        assert!(matches!(
            store.mark_invoked("approval-legacy-fence","attempt-legacy-fence"),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        store.mark_invoked_bound(&record).unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_releases_prepared_attempt_and_records_marker() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("pre-recovery","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-pre-recovery".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:13:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-pre-recovery",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pre","boundary-A"
        ).unwrap();

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:"attempt-pre".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:14:00Z".into(),
        };
        assert!(store.recover_pre_dispatch_attempt(&recovery).unwrap());
        assert!(!store.recover_pre_dispatch_attempt(&recovery).unwrap());
        assert_eq!(store.recover_incomplete_attempts().unwrap(),0);
        assert_eq!(store.recover_incomplete_attempts_for_boundary("boundary-A").unwrap(),0);

        let old_attempt=store.mark_dispatch_pending_bound(
            &witness.authorization_instance,"attempt-pre",&action,
            &super::super::ActionEffectBinding::new("target-A","prod","adapter-A"),
            "boundary-A"
        );
        assert!(old_attempt.is_err());

        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pre-retry","boundary-A"
        ).unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_rejects_tampered_scope_and_digest() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-fence-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("pre-recovery-fence","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-pre-fence".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:15:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-pre-fence",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pre-fence","boundary-A"
        ).unwrap();

        let wrong_boundary=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-B".into(),
            action_digest:digest.clone(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:00Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&wrong_boundary),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let wrong_digest=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-A".into(),
            action_digest:"sha256:forged".into(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:01Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&wrong_digest),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let correct=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:02Z".into(),
        };
        assert!(store.recover_pre_dispatch_attempt(&correct).unwrap());
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_cannot_release_after_dispatch_pending() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-after-dispatch-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("pre-recovery-after-dispatch","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-after-dispatch".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:17:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-after-dispatch",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-after-dispatch","boundary-A"
        ).unwrap();
        let record=store.mark_dispatch_pending_bound(
            "approval-after-dispatch","attempt-after-dispatch",&action,&effect,"boundary-A"
        ).unwrap();

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:"approval-after-dispatch".into(),
            attempt_id:"attempt-after-dispatch".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:18:00Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&recovery),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed
            ))
        ));
        assert_eq!(record.boundary_id,"boundary-A");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn boundary_scoped_recovery_cannot_claim_another_boundary() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-recovery-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("boundary-recovery-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-boundary".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:10:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-boundary","boundary-A"
        ).unwrap();
        let record=store.mark_dispatch_pending_bound(
            &witness.authorization_instance,"attempt-boundary",&action,&effect,"boundary-A"
        ).unwrap();
        drop(store);

        let boundary_b=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(boundary_b.recover_incomplete_attempts().unwrap(),0);
        assert_eq!(
            boundary_b
                .recover_incomplete_attempt_for_boundary("boundary-B","attempt-boundary")
                .unwrap(),
            0
        );

        let mut wrong=record.clone();
        wrong.boundary_id="boundary-B".into();
        assert!(matches!(
            boundary_b.reconcile_indeterminate_bound(&wrong,ExecutionOutcome::Succeeded),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let boundary_a=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(
            boundary_a
                .recover_incomplete_attempt_for_boundary("boundary-A","attempt-boundary")
                .unwrap(),
            1
        );
        let receipt=boundary_a.reconcile_indeterminate_bound(&record,ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.outcome,ExecutionOutcome::Succeeded);
        assert_eq!(receipt.authorization_instance,"approval-boundary");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn boundary_ownership_survives_crash_before_dispatch_record() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-prepared-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("boundary-prepared","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-prepared".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:12:00Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-prepared",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-prepared","boundary-A"
        ).unwrap();
        drop(store);

        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(reopened.recover_incomplete_attempts().unwrap(),0);
        assert_eq!(
            reopened
                .recover_incomplete_attempt_for_boundary("boundary-A","attempt-prepared")
                .unwrap(),
            0
        );

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:"attempt-prepared".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:12:30Z".into(),
        };
        assert!(reopened.recover_pre_dispatch_attempt(&recovery).unwrap());
        assert!(matches!(
            reopened.mark_dispatch_pending_bound(
                &witness.authorization_instance,"attempt-prepared",&action,
                &super::super::ActionEffectBinding::new("target-A","prod","adapter-A"),
                "boundary-A"
            ),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::AttemptMismatch))
        ));
        let _=std::fs::remove_file(path);
    }


    #[test]
    fn bound_attempt_id_cannot_be_reused_across_boundaries() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-attempt-scope-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action_a=EpistemicAction::new("scope-a","a",super::super::ActionRisk::Critical);
        let action_b=EpistemicAction::new("scope-b","b",super::super::ActionRisk::Critical);
        let digest_a=action_a.canonical_action_digest();
        let digest_b=action_b.canonical_action_digest();
        let witness_a=ActionAuthorizationWitness {
            action_id:action_a.id.clone(), authorization_instance:"scope-approval-a".into(),
            action_digest:digest_a.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:11:00Z".into(), expires_at:None, authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            action_id:action_b.id.clone(), authorization_instance:"scope-approval-b".into(),
            action_digest:digest_b.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:11:01Z".into(), expires_at:None, authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "scope-approval-a",action_a.id.clone(),digest_a,"sha256:support","policy-v1",1,1
        )).unwrap();
        store.register_lease(&AuthorizationLease::new_with_instance(
            "scope-approval-b",action_b.id.clone(),digest_b,"sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness_a,&action_a,"frame@1","attempt-reused","boundary-A"
        ).unwrap();
        assert!(matches!(
            store.prepare_for_execution_bound(
                &witness_b,&action_b,"frame@1","attempt-reused","boundary-B"
            ),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn recovery_treats_invoked_as_indeterminate() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-invoked-recovery-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-crash").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-crash").unwrap();
        store.mark_invoked(&witness.authorization_instance,"attempt-crash").unwrap();
        drop(store);

        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(reopened.recover_incomplete_attempts().unwrap(),1);
        assert!(matches!(
            reopened.prepare_for_execution(&witness,&action,"frame@1","attempt-retry"),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::IndeterminateRequiresReconciliation
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn dispatch_pending_wrong_attempt_is_fenced() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-dispatch-fence-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-1").unwrap();
        assert_eq!(
            store.mark_dispatch_pending(&witness.authorization_instance,"attempt-2").unwrap_err().to_string(),
            "authorization consumption error: AttemptMismatch"
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn current_instance_store_is_upgraded_with_boundary_columns_and_indexes() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-migrate-{}.db",std::process::id()));
        {
            let connection=Connection::open(&path).unwrap();
            connection.execute_batch(
                "CREATE TABLE authorization_leases (
                   authorization_instance TEXT PRIMARY KEY, action_id TEXT NOT NULL,
                   action_digest TEXT NOT NULL, support_digest TEXT NOT NULL,
                   policy TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
                   remaining_executions INTEGER NOT NULL, state TEXT NOT NULL, attempt_id TEXT
                 );
                 CREATE TABLE authorization_receipts (
                   authorization_instance TEXT NOT NULL, action_id TEXT NOT NULL,
                   attempt_id TEXT NOT NULL, phase TEXT NOT NULL, outcome TEXT NOT NULL,
                   action_digest TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
                   PRIMARY KEY(authorization_instance,attempt_id,phase)
                 );
                 CREATE TABLE authorization_dispatches (
                   authorization_instance TEXT NOT NULL, attempt_id TEXT NOT NULL,
                   action_id TEXT NOT NULL, action_digest TEXT NOT NULL,
                   provider_idempotency_key TEXT NOT NULL, target_identity TEXT NOT NULL,
                   audience TEXT NOT NULL, adapter TEXT NOT NULL, boundary_id TEXT NOT NULL,
                   state TEXT NOT NULL, PRIMARY KEY(authorization_instance,attempt_id)
                 );",
            ).unwrap();
        }
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let lease=store.connection().unwrap();

        let lease_boundary:String=lease.query_row(
            "SELECT name FROM pragma_table_info('authorization_leases')
             WHERE name='boundary_id'",
            [], |row| row.get(0)
        ).unwrap();
        let receipt_boundary:String=lease.query_row(
            "SELECT name FROM pragma_table_info('authorization_receipts')
             WHERE name='boundary_id'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(lease_boundary,"boundary_id");
        assert_eq!(receipt_boundary,"boundary_id");

        let index_count:i64=lease.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='index' AND name IN
             ('authorization_lease_attempt_id_uq','authorization_dispatch_attempt_id_uq')",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(index_count,2);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn legacy_action_keyed_store_is_migrated_to_explicit_instances() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-migrate-{}.db",std::process::id()));
        {
            let connection=Connection::open(&path).unwrap();
            connection.execute_batch(
                "CREATE TABLE authorization_leases (
                   action_id TEXT PRIMARY KEY, action_digest TEXT NOT NULL,
                   support_digest TEXT NOT NULL, policy TEXT NOT NULL,
                   authority_epoch INTEGER NOT NULL, remaining_executions INTEGER NOT NULL,
                   state TEXT NOT NULL, attempt_id TEXT
                 );
                 CREATE TABLE authorization_receipts (
                   action_id TEXT NOT NULL, attempt_id TEXT NOT NULL, phase TEXT NOT NULL,
                   outcome TEXT NOT NULL, action_digest TEXT NOT NULL,
                   authority_epoch INTEGER NOT NULL,
                   PRIMARY KEY(action_id,attempt_id,phase)
                 );
                 INSERT INTO authorization_leases VALUES
                   ('legacy-action','sha256:action','sha256:support','policy-v1',1,1,'ready',NULL);",
            ).unwrap();
        }
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let lease=store.connection().unwrap();
        let instance:String=lease.query_row(
            "SELECT authorization_instance FROM authorization_leases WHERE action_id='legacy-action'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(instance,"legacy-action");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn concurrent_prepare_has_one_winner() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-concurrent-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let second=SqliteAuthorizationStore::open(&path).unwrap();
        let barrier=Arc::new(std::sync::Barrier::new(2));
        let a2=action.clone(); let w2=witness.clone();
        let b1=barrier.clone(); let b2=barrier.clone();
        let t1=thread::spawn(move||{b1.wait();store.prepare_for_execution(&witness,&action,"frame@1","a")});
        let t2=thread::spawn(move||{b2.wait();second.prepare_for_execution(&w2,&a2,"frame@1","b")});
        let r1=t1.join().unwrap(); let r2=t2.join().unwrap();
        assert_ne!(r1.is_ok(),r2.is_ok());
        let _=std::fs::remove_file(path);
    }
}
