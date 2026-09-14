// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure retained-lock reconciliation for EUREKA-002 V2 qualifier checkpoints.
//!
//! A retained #3148 writer lock is evidence that checkpoint persistence ended
//! ambiguously. This module observes the exact self-bound lock and exact durable
//! target twice, rejects a moving observation surface, and classifies only what
//! the bytes establish. It deliberately contains no lock deletion, stale-age
//! policy, writer-liveness inference, governance admission, or execution
//! authority.

#![allow(dead_code)]

use super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackCheckpoint;
use super::v2_qualifier_checkpoint_store::{
    V2CheckpointStoreError, V2CheckpointWriteLockRecord, V2QualifierCheckpointStore,
};

pub(super) const V2_CHECKPOINT_RECONCILIATION_PLAN_REVISION: &str =
    "EUREKA.002.V2.CHECKPOINT_RECONCILIATION_PLAN.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2CheckpointReconciliationDisposition {
    NotCommitted,
    Committed,
    EscalationRequired(V2CheckpointReconciliationEscalation),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2CheckpointReconciliationEscalation {
    InvalidLockShape,
    MissingExpectedHighWater,
    UnexpectedTarget,
    DivergentTarget,
    InvalidIntendedCheckpointShape,
}

#[derive(Debug)]
pub(super) struct V2CheckpointReconciliationPlan {
    lock: V2CheckpointWriteLockRecord,
    observed_target: Option<V2QualifierAntiRollbackCheckpoint>,
    disposition: V2CheckpointReconciliationDisposition,
    commitment: [u8; 32],
}

impl V2CheckpointReconciliationPlan {
    pub(super) fn lock(&self) -> &V2CheckpointWriteLockRecord {
        &self.lock
    }

    pub(super) fn observed_target(&self) -> Option<&V2QualifierAntiRollbackCheckpoint> {
        self.observed_target.as_ref()
    }

    pub(super) const fn disposition(&self) -> V2CheckpointReconciliationDisposition {
        self.disposition
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2CheckpointReconciliationError {
    Store(V2CheckpointStoreError),
    NoWriterLock,
    LockChangedDuringInspection,
    TargetChangedDuringInspection,
}

/// Observe a retained lock and target without mutating either one.
///
/// The double-read is intentional: a plan is not emitted if the lock or target
/// changes while reconciliation is observing them. The resulting disposition
/// says only what exact bytes were stable across this inspection; it does not
/// establish that the previous writer is dead and does not authorize lock
/// removal.
pub(super) fn reconcile_checkpoint_writer_lock(
    store: &V2QualifierCheckpointStore,
) -> Result<V2CheckpointReconciliationPlan, V2CheckpointReconciliationError> {
    let first_lock = store
        .inspect_writer_lock()
        .map_err(V2CheckpointReconciliationError::Store)?
        .ok_or(V2CheckpointReconciliationError::NoWriterLock)?;
    let first_target = store
        .load()
        .map_err(V2CheckpointReconciliationError::Store)?;

    let second_lock = store
        .inspect_writer_lock()
        .map_err(V2CheckpointReconciliationError::Store)?
        .ok_or(V2CheckpointReconciliationError::LockChangedDuringInspection)?;
    if second_lock != first_lock {
        return Err(V2CheckpointReconciliationError::LockChangedDuringInspection);
    }

    let second_target = store
        .load()
        .map_err(V2CheckpointReconciliationError::Store)?;
    if second_target != first_target {
        return Err(V2CheckpointReconciliationError::TargetChangedDuringInspection);
    }

    let disposition = classify_stable_state(&first_lock, first_target.as_ref());
    let mut plan = V2CheckpointReconciliationPlan {
        lock: first_lock,
        observed_target: first_target,
        disposition,
        commitment: [0_u8; 32],
    };
    plan.commitment = reconciliation_plan_commitment(&plan);
    Ok(plan)
}

fn classify_stable_state(
    lock: &V2CheckpointWriteLockRecord,
    target: Option<&V2QualifierAntiRollbackCheckpoint>,
) -> V2CheckpointReconciliationDisposition {
    let expected = lock.expected_checkpoint_commitment();
    let intended = lock.intended_checkpoint_commitment();

    if expected == Some(intended) {
        return V2CheckpointReconciliationDisposition::EscalationRequired(
            V2CheckpointReconciliationEscalation::InvalidLockShape,
        );
    }

    match (expected, target) {
        (None, None) => V2CheckpointReconciliationDisposition::NotCommitted,
        (Some(_), None) => V2CheckpointReconciliationDisposition::EscalationRequired(
            V2CheckpointReconciliationEscalation::MissingExpectedHighWater,
        ),
        (None, Some(observed)) if observed.commitment() == intended => {
            if observed.authority_sequence() == 1
                && observed.signer_policy_sequence() == 1
                && observed.predecessor_checkpoint_commitment().is_none()
            {
                V2CheckpointReconciliationDisposition::Committed
            } else {
                V2CheckpointReconciliationDisposition::EscalationRequired(
                    V2CheckpointReconciliationEscalation::InvalidIntendedCheckpointShape,
                )
            }
        }
        (None, Some(_)) => V2CheckpointReconciliationDisposition::EscalationRequired(
            V2CheckpointReconciliationEscalation::UnexpectedTarget,
        ),
        (Some(expected_commitment), Some(observed))
            if observed.commitment() == expected_commitment =>
        {
            V2CheckpointReconciliationDisposition::NotCommitted
        }
        (Some(expected_commitment), Some(observed)) if observed.commitment() == intended => {
            if observed.predecessor_checkpoint_commitment() == Some(expected_commitment) {
                V2CheckpointReconciliationDisposition::Committed
            } else {
                V2CheckpointReconciliationDisposition::EscalationRequired(
                    V2CheckpointReconciliationEscalation::InvalidIntendedCheckpointShape,
                )
            }
        }
        (Some(_), Some(_)) => V2CheckpointReconciliationDisposition::EscalationRequired(
            V2CheckpointReconciliationEscalation::DivergentTarget,
        ),
    }
}

fn reconciliation_plan_commitment(plan: &V2CheckpointReconciliationPlan) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_CHECKPOINT_RECONCILIATION_PLAN_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&plan.lock.commitment());
    match plan.lock.expected_checkpoint_commitment() {
        Some(expected) => {
            bytes.push(1);
            bytes.extend_from_slice(&expected);
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&plan.lock.intended_checkpoint_commitment());
    match plan.observed_target.as_ref() {
        Some(target) => {
            bytes.push(1);
            bytes.extend_from_slice(&target.commitment());
            encode_bytes(&mut bytes, &target.persisted_bytes());
        }
        None => bytes.push(0),
    }
    encode_disposition(&mut bytes, plan.disposition);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_disposition(bytes: &mut Vec<u8>, disposition: V2CheckpointReconciliationDisposition) {
    match disposition {
        V2CheckpointReconciliationDisposition::NotCommitted => bytes.extend_from_slice(&[0, 0]),
        V2CheckpointReconciliationDisposition::Committed => bytes.extend_from_slice(&[1, 0]),
        V2CheckpointReconciliationDisposition::EscalationRequired(reason) => {
            bytes.push(2);
            bytes.push(match reason {
                V2CheckpointReconciliationEscalation::InvalidLockShape => 1,
                V2CheckpointReconciliationEscalation::MissingExpectedHighWater => 2,
                V2CheckpointReconciliationEscalation::UnexpectedTarget => 3,
                V2CheckpointReconciliationEscalation::DivergentTarget => 4,
                V2CheckpointReconciliationEscalation::InvalidIntendedCheckpointShape => 5,
            });
        }
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackGuard;
    use super::super::v2_qualifier_authority::{
        V2QualifierAuthorityProfile, V2QualifierAuthorityRecord,
    };
    use super::super::v2_qualifier_currentness::V2QualifierAuthorityLineage;
    use super::super::v2_qualifier_signer_policy::{
        V2GovernanceSigner, V2QualifierSignerPolicy,
    };
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn policy() -> V2QualifierSignerPolicy {
        let signer = V2GovernanceSigner::from_hex(
            "recovery-fixture@example.invalid",
            "ssh-ed25519",
            &"a".repeat(64),
        )
        .unwrap();
        V2QualifierSignerPolicy::genesis(1, 1, vec![signer]).unwrap()
    }

    fn checkpoint_pair(
        workflow: char,
        successor_workflow: char,
    ) -> (
        V2QualifierAntiRollbackCheckpoint,
        V2QualifierAntiRollbackCheckpoint,
    ) {
        let authority = V2QualifierAuthorityRecord::genesis(1, profile(workflow, 'c')).unwrap();
        let successor = V2QualifierAuthorityRecord::rotate(
            &authority,
            authority.commitment(),
            2,
            profile(successor_workflow, 'c'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis = guard.checkpoint().clone();
        let (guard, _) = guard.advance_authority(lineage, successor, &policy).unwrap();
        (genesis, guard.checkpoint().clone())
    }

    fn test_store() -> (PathBuf, V2QualifierCheckpointStore) {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "symthaea-eureka-v2-reconcile-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let store = V2QualifierCheckpointStore::new(dir.join("high-water.env")).unwrap();
        (dir, store)
    }

    #[cfg(unix)]
    fn retained_transition_lock(
        store: &V2QualifierCheckpointStore,
        expected: &V2QualifierAntiRollbackCheckpoint,
        intended: &V2QualifierAntiRollbackCheckpoint,
        durable_other: &V2QualifierAntiRollbackCheckpoint,
    ) {
        store.persist_genesis(durable_other).unwrap();
        assert_eq!(
            store.persist_successor(expected, intended).unwrap_err(),
            V2CheckpointStoreError::HighWaterMismatch
        );
        let lock = store.inspect_writer_lock().unwrap().unwrap();
        assert_eq!(lock.expected_checkpoint_commitment(), Some(expected.commitment()));
        assert_eq!(lock.intended_checkpoint_commitment(), intended.commitment());
    }

    #[cfg(unix)]
    #[test]
    fn exact_expected_target_classifies_not_committed_without_clearing() {
        let (expected, intended) = checkpoint_pair('b', 'd');
        let (other, _) = checkpoint_pair('e', 'f');
        let (dir, store) = test_store();
        retained_transition_lock(&store, &expected, &intended, &other);
        fs::write(store.path(), expected.persisted_bytes()).unwrap();

        let plan = reconcile_checkpoint_writer_lock(&store).unwrap();
        assert_eq!(plan.disposition(), V2CheckpointReconciliationDisposition::NotCommitted);
        assert_eq!(plan.observed_target(), Some(&expected));
        assert!(store.inspect_writer_lock().unwrap().is_some());
        assert_ne!(plan.commitment(), [0_u8; 32]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn exact_intended_target_classifies_committed_without_clearing() {
        let (expected, intended) = checkpoint_pair('b', 'd');
        let (other, _) = checkpoint_pair('e', 'f');
        let (dir, store) = test_store();
        retained_transition_lock(&store, &expected, &intended, &other);
        fs::write(store.path(), intended.persisted_bytes()).unwrap();

        let plan = reconcile_checkpoint_writer_lock(&store).unwrap();
        assert_eq!(plan.disposition(), V2CheckpointReconciliationDisposition::Committed);
        assert_eq!(plan.observed_target(), Some(&intended));
        assert!(store.inspect_writer_lock().unwrap().is_some());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn divergent_target_requires_escalation_instead_of_preference() {
        let (expected, intended) = checkpoint_pair('b', 'd');
        let (other, _) = checkpoint_pair('e', 'f');
        let (dir, store) = test_store();
        retained_transition_lock(&store, &expected, &intended, &other);

        let plan = reconcile_checkpoint_writer_lock(&store).unwrap();
        assert_eq!(
            plan.disposition(),
            V2CheckpointReconciliationDisposition::EscalationRequired(
                V2CheckpointReconciliationEscalation::DivergentTarget
            )
        );
        assert_eq!(plan.observed_target(), Some(&other));
        assert!(store.inspect_writer_lock().unwrap().is_some());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn missing_expected_high_water_requires_escalation() {
        let (expected, intended) = checkpoint_pair('b', 'd');
        let (dir, store) = test_store();
        assert_eq!(
            store.persist_successor(&expected, &intended).unwrap_err(),
            V2CheckpointStoreError::MissingCheckpoint
        );

        let plan = reconcile_checkpoint_writer_lock(&store).unwrap();
        assert_eq!(
            plan.disposition(),
            V2CheckpointReconciliationDisposition::EscalationRequired(
                V2CheckpointReconciliationEscalation::MissingExpectedHighWater
            )
        );
        assert!(plan.observed_target().is_none());
        assert!(store.inspect_writer_lock().unwrap().is_some());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_lock_has_no_reconciliation_plan() {
        let (dir, store) = test_store();
        assert_eq!(
            reconcile_checkpoint_writer_lock(&store).unwrap_err(),
            V2CheckpointReconciliationError::NoWriterLock
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reconciliation_source_cannot_clear_lock_or_grant_authority() {
        let production = include_str!("v2_checkpoint_reconciliation.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "remove_file",
            "stale_after",
            "SystemTime",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden reconciliation surface: {forbidden}"
            );
        }
        assert!(production.contains("inspect_writer_lock"));
        assert!(production.contains("TargetChangedDuringInspection"));
        assert!(production.contains("EscalationRequired"));
    }
}
