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

/// A durable shared consumption domain. Each operation uses a fresh connection,
/// allowing independent processes to contend on the same SQLite state machine.
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
               attempt_id TEXT
             );
             CREATE TABLE IF NOT EXISTS authorization_receipts (
               authorization_instance TEXT NOT NULL,
               action_id TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               phase TEXT NOT NULL,
               outcome TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id, phase)
             );",
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
              remaining_executions, state, attempt_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
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
        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;
        lease.prepare_for_execution(witness, action, current_frame, attempt_id)?;
        tx.execute(
            "UPDATE authorization_leases SET state='prepared', attempt_id=?2
             WHERE authorization_instance=?1 AND state='ready' AND remaining_executions>0",
            params![witness.authorization_instance.as_str(), attempt_id],
        )?;
        if tx.changes() != 1 {
            return Err(AuthorizationConsumptionError::NotReady.into());
        }
        tx.commit()?;
        Ok(())
    }

    pub fn commit(
        &self, authorization_instance: &str, attempt_id: &str, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

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
        insert_receipt(&tx, &receipt, "reconciled")?;
        tx.commit()?;
        Ok(receipt)
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
           attempt_id TEXT
         );

         CREATE TABLE authorization_receipts (
           authorization_instance TEXT NOT NULL,
           action_id TEXT NOT NULL,
           attempt_id TEXT NOT NULL,
           phase TEXT NOT NULL,
           outcome TEXT NOT NULL,
           action_digest TEXT NOT NULL,
           authority_epoch INTEGER NOT NULL,
           PRIMARY KEY(authorization_instance, attempt_id, phase)
         );",
    )?;
    tx.execute(
        "INSERT INTO authorization_leases
         (authorization_instance,action_id,action_digest,support_digest,policy,
          authority_epoch,remaining_executions,state,attempt_id)
         SELECT action_id,action_id,action_digest,support_digest,policy,
                authority_epoch,remaining_executions,state,attempt_id
         FROM authorization_leases_legacy",
        [],
    )?;
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch)
         SELECT action_id,action_id,attempt_id,phase,outcome,action_digest,authority_epoch
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
        AuthorizationLeaseState::Indeterminate { .. } => "indeterminate",
        AuthorizationLeaseState::Exhausted => "exhausted",
        AuthorizationLeaseState::Revoked => "revoked",
        AuthorizationLeaseState::Expired => "expired",
    }
}
fn state_attempt(s: &AuthorizationLeaseState) -> Option<&str> {
    match s {
        AuthorizationLeaseState::Prepared { attempt_id } |
        AuthorizationLeaseState::Indeterminate { attempt_id } => Some(attempt_id),
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
fn update_lease(tx: &Transaction<'_>, lease: &AuthorizationLease) -> Result<(), AuthorizationStoreError> {
    let changed = tx.execute(
        "UPDATE authorization_leases
         SET state=?2,attempt_id=?3,remaining_executions=?4
         WHERE authorization_instance=?1",
        params![
            lease.authorization_instance,
            encode_state(&lease.state),
            state_attempt(&lease.state),
            lease.remaining_executions as i64,
        ],
    )?;
    if changed != 1 {
        return Err(AuthorizationStoreError::NotFound(lease.authorization_instance.clone()));
    }
    Ok(())
}
fn insert_receipt(tx: &Transaction<'_>, r: &ExecutionReceipt, phase: &str) -> Result<(), AuthorizationStoreError> {
    let outcome = match r.outcome {
        ExecutionOutcome::Succeeded => "succeeded",
        ExecutionOutcome::Failed => "failed",
        ExecutionOutcome::Indeterminate => "indeterminate",
    };
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch)
         VALUES (?1,?2,?3,?4,?5,?6,?7)",
        params![
            r.authorization_instance, r.action_id, r.attempt_id, phase, outcome,
            r.action_digest, r.authority_epoch as i64
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
        params![action_id,attempt_id,phase],
        |r| {
            let outcome = match r.get::<_,String>(3)?.as_str() {
                "succeeded" => ExecutionOutcome::Succeeded,
                "failed" => ExecutionOutcome::Failed,
                "indeterminate" => ExecutionOutcome::Indeterminate,
                _ => return Err(rusqlite::Error::InvalidQuery),
            };
            Ok(ExecutionReceipt {
                authorization_instance:r.get(0)?, action_id:r.get(1)?, attempt_id:r.get(2)?, outcome,
                action_digest:r.get(4)?, authority_epoch:r.get::<_,i64>(5)? as u64,
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
        let first=store.commit(&action.id,"attempt-1",ExecutionOutcome::Succeeded).unwrap();
        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
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
        let receipt=store.commit("approval-new","new-attempt",ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.authorization_instance,"approval-new");
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
