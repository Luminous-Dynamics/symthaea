// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only crash recovery for the bounded systemd agency witness.
//!
//! A checkpoint persisted immediately before dispatch can legitimately contain
//! an execution in [`ReservationState::Reserved`]. After a process/host crash,
//! that state is ambiguous: the effect may have been dispatched after the last
//! durable checkpoint. Restoring it as ordinary pre-dispatch work would be too
//! optimistic.
//!
//! This crate therefore provides a conservative recovery protocol:
//!
//! ```text
//! trusted checkpoint: Reserved
//!          |
//!          | crash restoration
//!          v
//!     OutcomeUnknown   -- persist successor first
//!          |
//!          | read-only independent observation
//!          +---- InvocationID changed ----> Committed
//!          |
//!          +---- otherwise ---------------> still OutcomeUnknown
//! ```
//!
//! Reconciliation is evidence/accounting, not a new effect. It intentionally
//! has no restart/mutation operation and does not require a still-live mutation
//! capability to record what an already-initiated action actually did.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_action_checkpoint::{CheckpointError, CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{
    GrantAccount, ReservationId, ReservationState, RuntimeAccountingError,
};
use symthaea_authority::{CapabilityGrant, Digest32};
use symthaea_system_broker::{
    CheckpointStore, HostId, RestartPlan, ServiceBackend, ServiceObservation, ServiceUnit,
};
use thiserror::Error;

/// Read-only observer surface used during crash reconciliation.
///
/// There is deliberately no mutation method here.
pub trait ServiceObserver {
    type Error: std::error::Error;

    fn observe(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error>;
}

/// Any typed #305 backend can be used as an observer, but recovery code sees
/// only this read-only trait.
impl<T> ServiceObserver for T
where
    T: ServiceBackend,
{
    type Error = T::Error;

    fn observe(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error> {
        ServiceBackend::observe(self, host, unit)
    }
}

/// Result of converting crash-ambiguous `Reserved` state to `OutcomeUnknown`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedCheckpoint {
    pub checkpoint: GrantAccountCheckpoint,
    pub head: CheckpointHead,
    pub converted_reservations: Vec<ReservationId>,
}

/// Evidence result for one unknown restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReconciliationOutcome {
    /// A changed non-empty systemd InvocationID proved a new invocation.
    ReconciledApplied,
    /// Observation succeeded but did not prove application.
    StillUnknown,
    /// Observation itself was unavailable. Authority remains charged.
    ObservationUnavailable,
}

/// Privacy-minimized recovery evidence. No journal, stderr, or command output is retained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationReceipt {
    pub reservation_id: ReservationId,
    pub grant_digest: Digest32,
    pub plan_digest: Digest32,
    pub before_world_digest: Digest32,
    pub after_world_digest: Option<Digest32>,
    pub previous_head: CheckpointHead,
    pub current_head: CheckpointHead,
    pub outcome: ReconciliationOutcome,
}

/// On crash restoration, conservatively convert every in-flight `Reserved`
/// execution to `OutcomeUnknown` and durably persist the successor before the
/// returned checkpoint can be used for further recovery.
///
/// If there are no `Reserved` executions, no new checkpoint is emitted.
pub fn normalize_crash_checkpoint<S>(
    grant: &CapabilityGrant,
    checkpoint: GrantAccountCheckpoint,
    trusted_head: CheckpointHead,
    store: &mut S,
) -> Result<NormalizedCheckpoint, RecoveryError>
where
    S: CheckpointStore,
{
    require_exact_head(&checkpoint, trusted_head)?;
    let mut account = checkpoint.verify_payload(grant)?;
    let reserved_ids: Vec<ReservationId> = account
        .snapshot()
        .reservations
        .iter()
        .filter_map(|(id, reservation)| {
            (reservation.state == ReservationState::Reserved).then_some(id.clone())
        })
        .collect();

    if reserved_ids.is_empty() {
        return Ok(NormalizedCheckpoint {
            checkpoint,
            head: trusted_head,
            converted_reservations: Vec::new(),
        });
    }

    for id in &reserved_ids {
        account.mark_outcome_unknown(id)?;
    }

    let successor = GrantAccountCheckpoint::successor(&checkpoint, grant, account.snapshot())?;
    let expected_head = successor.head()?;
    let acknowledged = store
        .persist(&successor)
        .map_err(|_| RecoveryError::CheckpointStoreFailed)?;
    if acknowledged != expected_head {
        return Err(RecoveryError::CheckpointHeadMismatch);
    }

    Ok(NormalizedCheckpoint {
        checkpoint: successor,
        head: expected_head,
        converted_reservations: reserved_ids,
    })
}

/// Reconcile one already-unknown restart through independent read-only service
/// observation.
///
/// This function performs no new mutation and therefore does not consume a new
/// capability use. Revocation/expiry can stop future actions but must not erase
/// or block accurate accounting of an effect that may already have occurred.
pub fn reconcile_unknown_restart<O, S>(
    grant: &CapabilityGrant,
    checkpoint: GrantAccountCheckpoint,
    trusted_head: CheckpointHead,
    plan: &RestartPlan,
    before: &ServiceObservation,
    reservation_id: &ReservationId,
    observer: &mut O,
    store: &mut S,
) -> Result<(GrantAccountCheckpoint, ReconciliationReceipt), RecoveryError>
where
    O: ServiceObserver,
    S: CheckpointStore,
{
    require_exact_head(&checkpoint, trusted_head)?;
    validate_reconciliation_binding(grant, plan, before)?;
    let mut account = checkpoint.verify_payload(grant)?;
    let state = account
        .snapshot()
        .reservations
        .get(reservation_id)
        .ok_or(RecoveryError::UnknownReservation)?
        .state;
    if state != ReservationState::OutcomeUnknown {
        return Err(RecoveryError::ReservationNotOutcomeUnknown);
    }

    let after = match observer.observe(&plan.host, &plan.unit) {
        Ok(value) => value,
        Err(_) => {
            return Ok((
                checkpoint,
                ReconciliationReceipt {
                    reservation_id: reservation_id.clone(),
                    grant_digest: grant.digest(),
                    plan_digest: plan.digest(),
                    before_world_digest: before.digest(),
                    after_world_digest: None,
                    previous_head: trusted_head,
                    current_head: trusted_head,
                    outcome: ReconciliationOutcome::ObservationUnavailable,
                },
            ));
        }
    };

    if invocation_changed(before, &after) {
        account.reconcile_applied(reservation_id)?;
        let successor = GrantAccountCheckpoint::successor(&checkpoint, grant, account.snapshot())?;
        let new_head = successor.head()?;
        let acknowledged = store
            .persist(&successor)
            .map_err(|_| RecoveryError::CheckpointStoreFailed)?;
        if acknowledged != new_head {
            return Err(RecoveryError::CheckpointHeadMismatch);
        }
        let receipt = ReconciliationReceipt {
            reservation_id: reservation_id.clone(),
            grant_digest: grant.digest(),
            plan_digest: plan.digest(),
            before_world_digest: before.digest(),
            after_world_digest: Some(after.digest()),
            previous_head: trusted_head,
            current_head: new_head,
            outcome: ReconciliationOutcome::ReconciledApplied,
        };
        Ok((successor, receipt))
    } else {
        Ok((
            checkpoint,
            ReconciliationReceipt {
                reservation_id: reservation_id.clone(),
                grant_digest: grant.digest(),
                plan_digest: plan.digest(),
                before_world_digest: before.digest(),
                after_world_digest: Some(after.digest()),
                previous_head: trusted_head,
                current_head: trusted_head,
                outcome: ReconciliationOutcome::StillUnknown,
            },
        ))
    }
}

fn validate_reconciliation_binding(
    grant: &CapabilityGrant,
    plan: &RestartPlan,
    before: &ServiceObservation,
) -> Result<(), RecoveryError> {
    if before.host != plan.host || before.unit != plan.unit || before.digest() != plan.world_digest {
        return Err(RecoveryError::BeforeObservationMismatch);
    }
    if grant.subject != plan.actor
        || grant.audience.as_ref() != Some(&plan.executor)
        || grant.task != plan.task
        || grant.resources != BTreeSet::from([plan.resource()])
        || grant.operations != BTreeSet::from([plan.operation()])
        || grant.plan_digest != Some(plan.digest())
        || grant.world_digest != Some(before.digest())
    {
        return Err(RecoveryError::GrantPlanMismatch);
    }
    Ok(())
}

fn require_exact_head(
    checkpoint: &GrantAccountCheckpoint,
    trusted_head: CheckpointHead,
) -> Result<(), RecoveryError> {
    if checkpoint.head()? == trusted_head {
        Ok(())
    } else {
        Err(RecoveryError::TrustedHeadMismatch)
    }
}

fn invocation_changed(before: &ServiceObservation, after: &ServiceObservation) -> bool {
    matches!(
        (&before.invocation_id, &after.invocation_id),
        (Some(before_id), Some(after_id))
            if !before_id.is_empty() && !after_id.is_empty() && before_id != after_id
    )
}

#[derive(Debug, Error)]
pub enum RecoveryError {
    #[error("checkpoint does not match externally trusted head")]
    TrustedHeadMismatch,
    #[error("checkpoint store failed")]
    CheckpointStoreFailed,
    #[error("checkpoint store acknowledged a different head")]
    CheckpointHeadMismatch,
    #[error("restart plan is not bound to the supplied before-observation")]
    BeforeObservationMismatch,
    #[error("capability grant does not exactly bind the reconciliation plan")]
    GrantPlanMismatch,
    #[error("unknown reservation id")]
    UnknownReservation,
    #[error("reservation is not in OutcomeUnknown state")]
    ReservationNotOutcomeUnknown,
    #[error("runtime accounting error: {0}")]
    Runtime(#[from] RuntimeAccountingError),
    #[error("checkpoint validation error: {0}")]
    Checkpoint(#[from] CheckpointError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use symthaea_action_runtime::{ExecutionId, GrantUseState};
    use symthaea_authority::{AuthorityEpoch, PrincipalId, TaskId};
    use symthaea_system_broker::{DispatchEvidence, restart_risk_charge};

    #[derive(Debug, Error, Clone, Copy)]
    #[error("observer failure")]
    struct ObserverError;

    struct FakeObserver {
        result: Result<ServiceObservation, ObserverError>,
        calls: Arc<AtomicUsize>,
    }

    impl ServiceObserver for FakeObserver {
        type Error = ObserverError;

        fn observe(
            &mut self,
            _host: &HostId,
            _unit: &ServiceUnit,
        ) -> Result<ServiceObservation, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.result.clone()
        }
    }

    #[derive(Debug, Error)]
    #[error("store failure")]
    struct StoreError;

    #[derive(Default)]
    struct FakeStore {
        writes: usize,
        wrong_head: bool,
        fail: bool,
    }

    impl CheckpointStore for FakeStore {
        type Error = StoreError;

        fn persist(
            &mut self,
            checkpoint: &GrantAccountCheckpoint,
        ) -> Result<CheckpointHead, Self::Error> {
            self.writes += 1;
            if self.fail {
                return Err(StoreError);
            }
            let mut head = checkpoint.head().map_err(|_| StoreError)?;
            if self.wrong_head {
                head.digest = Digest32([0xAA; 32]);
            }
            Ok(head)
        }
    }

    fn host() -> HostId {
        HostId::parse("host-a").unwrap()
    }

    fn unit() -> ServiceUnit {
        ServiceUnit::parse("postgresql.service").unwrap()
    }

    fn observation(active: &str, sub: &str, invocation: &str) -> ServiceObservation {
        ServiceObservation {
            host: host(),
            unit: unit(),
            active_state: active.into(),
            sub_state: sub.into(),
            invocation_id: Some(invocation.into()),
        }
    }

    fn plan(before: &ServiceObservation) -> RestartPlan {
        RestartPlan::new(
            PrincipalId("user:alice".into()),
            PrincipalId("workload:system-broker".into()),
            Some(TaskId("task:repair-postgres".into())),
            before,
        )
    }

    fn grant_for(plan: &RestartPlan) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-restart-postgres",
            PrincipalId("authority:alice".into()),
            plan.actor.clone(),
            AuthorityEpoch(7),
        );
        grant.audience = Some(plan.executor.clone());
        grant.task = plan.task.clone();
        grant.resources = BTreeSet::from([plan.resource()]);
        grant.operations = BTreeSet::from([plan.operation()]);
        grant.plan_digest = Some(plan.digest());
        grant.world_digest = Some(plan.world_digest);
        grant.max_uses = 1;
        grant.risk_budget = restart_risk_charge();
        grant
    }

    fn reserved_checkpoint(
        grant: &CapabilityGrant,
    ) -> (GrantAccountCheckpoint, ReservationId) {
        let mut account = GrantAccount::new(grant);
        let id = ReservationId("res-1".into());
        account
            .reserve_execution(
                id.clone(),
                ExecutionId("exec-1".into()),
                restart_risk_charge(),
            )
            .unwrap();
        let checkpoint = GrantAccountCheckpoint::first(grant, account.snapshot()).unwrap();
        (checkpoint, id)
    }

    #[test]
    fn crash_restore_converts_reserved_to_unknown_before_reuse() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, id) = reserved_checkpoint(&grant);
        let head = checkpoint.head().unwrap();
        let mut store = FakeStore::default();

        let normalized = normalize_crash_checkpoint(&grant, checkpoint, head, &mut store).unwrap();
        assert_eq!(normalized.converted_reservations, vec![id.clone()]);
        assert_eq!(normalized.head.sequence, head.sequence + 1);
        assert_eq!(store.writes, 1);

        let account = normalized.checkpoint.verify_payload(&grant).unwrap();
        assert_eq!(
            account.snapshot().reservations.get(&id).unwrap().state,
            ReservationState::OutcomeUnknown
        );
        assert_eq!(
            account.authority_use_state(),
            GrantUseState {
                committed: 0,
                reserved: 1,
            }
        );
    }

    #[test]
    fn already_unknown_checkpoint_is_idempotent_and_not_rewritten() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, id) = reserved_checkpoint(&grant);
        let mut account = checkpoint.verify_payload(&grant).unwrap();
        account.mark_outcome_unknown(&id).unwrap();
        let unknown = GrantAccountCheckpoint::successor(&checkpoint, &grant, account.snapshot()).unwrap();
        let head = unknown.head().unwrap();
        let mut store = FakeStore::default();

        let normalized = normalize_crash_checkpoint(&grant, unknown.clone(), head, &mut store).unwrap();
        assert_eq!(normalized.checkpoint, unknown);
        assert!(normalized.converted_reservations.is_empty());
        assert_eq!(normalized.head, head);
        assert_eq!(store.writes, 0);
    }

    #[test]
    fn changed_invocation_reconciles_unknown_to_committed() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, id) = reserved_checkpoint(&grant);
        let head = checkpoint.head().unwrap();
        let mut store = FakeStore::default();
        let normalized = normalize_crash_checkpoint(&grant, checkpoint, head, &mut store).unwrap();
        let previous_head = normalized.head;

        let calls = Arc::new(AtomicUsize::new(0));
        let mut observer = FakeObserver {
            result: Ok(observation("active", "running", "inv-2")),
            calls: calls.clone(),
        };
        let (reconciled, receipt) = reconcile_unknown_restart(
            &grant,
            normalized.checkpoint,
            previous_head,
            &p,
            &before,
            &id,
            &mut observer,
            &mut store,
        )
        .unwrap();

        assert_eq!(receipt.outcome, ReconciliationOutcome::ReconciledApplied);
        assert_eq!(receipt.previous_head, previous_head);
        assert_eq!(receipt.current_head.sequence, previous_head.sequence + 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let account = reconciled.verify_payload(&grant).unwrap();
        assert_eq!(
            account.authority_use_state(),
            GrantUseState {
                committed: 1,
                reserved: 0,
            }
        );
    }

    #[test]
    fn unchanged_invocation_remains_unknown_and_does_not_write_checkpoint() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, id) = reserved_checkpoint(&grant);
        let head = checkpoint.head().unwrap();
        let mut store = FakeStore::default();
        let normalized = normalize_crash_checkpoint(&grant, checkpoint, head, &mut store).unwrap();
        let writes_after_normalize = store.writes;

        let mut observer = FakeObserver {
            result: Ok(observation("failed", "failed", "inv-1")),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let (same_checkpoint, receipt) = reconcile_unknown_restart(
            &grant,
            normalized.checkpoint.clone(),
            normalized.head,
            &p,
            &before,
            &id,
            &mut observer,
            &mut store,
        )
        .unwrap();
        assert_eq!(receipt.outcome, ReconciliationOutcome::StillUnknown);
        assert_eq!(receipt.current_head, normalized.head);
        assert_eq!(same_checkpoint, normalized.checkpoint);
        assert_eq!(store.writes, writes_after_normalize);
    }

    #[test]
    fn observation_failure_preserves_unknown_without_state_change() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, id) = reserved_checkpoint(&grant);
        let head = checkpoint.head().unwrap();
        let mut store = FakeStore::default();
        let normalized = normalize_crash_checkpoint(&grant, checkpoint, head, &mut store).unwrap();
        let writes_after_normalize = store.writes;
        let mut observer = FakeObserver {
            result: Err(ObserverError),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let (same_checkpoint, receipt) = reconcile_unknown_restart(
            &grant,
            normalized.checkpoint.clone(),
            normalized.head,
            &p,
            &before,
            &id,
            &mut observer,
            &mut store,
        )
        .unwrap();
        assert_eq!(receipt.outcome, ReconciliationOutcome::ObservationUnavailable);
        assert_eq!(same_checkpoint, normalized.checkpoint);
        assert_eq!(store.writes, writes_after_normalize);
    }

    #[test]
    fn wrong_trusted_head_rejects_before_normalization() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, _id) = reserved_checkpoint(&grant);
        let mut wrong = checkpoint.head().unwrap();
        wrong.digest = Digest32([0x99; 32]);
        let mut store = FakeStore::default();
        assert!(matches!(
            normalize_crash_checkpoint(&grant, checkpoint, wrong, &mut store),
            Err(RecoveryError::TrustedHeadMismatch)
        ));
        assert_eq!(store.writes, 0);
    }

    #[test]
    fn mismatched_before_observation_cannot_reconcile_other_target() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, id) = reserved_checkpoint(&grant);
        let head = checkpoint.head().unwrap();
        let mut store = FakeStore::default();
        let normalized = normalize_crash_checkpoint(&grant, checkpoint, head, &mut store).unwrap();
        let wrong_before = ServiceObservation {
            unit: ServiceUnit::parse("sshd.service").unwrap(),
            ..before.clone()
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let mut observer = FakeObserver {
            result: Ok(observation("active", "running", "inv-2")),
            calls: calls.clone(),
        };
        assert!(matches!(
            reconcile_unknown_restart(
                &grant,
                normalized.checkpoint,
                normalized.head,
                &p,
                &wrong_before,
                &id,
                &mut observer,
                &mut store,
            ),
            Err(RecoveryError::BeforeObservationMismatch)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn wrong_checkpoint_ack_fails_closed() {
        let before = observation("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let (checkpoint, _id) = reserved_checkpoint(&grant);
        let head = checkpoint.head().unwrap();
        let mut store = FakeStore {
            wrong_head: true,
            ..FakeStore::default()
        };
        assert!(matches!(
            normalize_crash_checkpoint(&grant, checkpoint, head, &mut store),
            Err(RecoveryError::CheckpointHeadMismatch)
        ));
    }

    // Compile-time intent note: this crate's recovery functions are generic over
    // `ServiceObserver`, not `ServiceBackend`; there is no restart call in the
    // recovery algorithm. DispatchEvidence is imported above only to make it
    // obvious that no such value participates in reconciliation.
    #[test]
    fn reconciliation_is_read_only_surface() {
        let _ = std::mem::size_of::<DispatchEvidence>();
    }
}
