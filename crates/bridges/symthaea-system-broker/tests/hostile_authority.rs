// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-API hostile tests for the bounded systemd recovery witness.

use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, GrantAccount, GrantUseState, ReservationId};
use symthaea_authority::{
    AuthorityContext, AuthorityEpoch, CapabilityGrant, Digest32, Operation, PrincipalId, ResourceRef,
    TaskId,
};
use symthaea_system_broker::{
    BrokerError, CheckpointStore, DispatchEvidence, HostId, RestartPlan, ServiceBackend,
    ServiceObservation, ServiceUnit, SystemdRecoveryBroker, restart_risk_charge,
};
use thiserror::Error;

#[derive(Debug, Error, Clone, Copy)]
#[error("hostile test backend failure")]
struct BackendError;

#[derive(Clone)]
struct CallCounters {
    observe: Arc<AtomicUsize>,
    restart: Arc<AtomicUsize>,
}

impl Default for CallCounters {
    fn default() -> Self {
        Self {
            observe: Arc::new(AtomicUsize::new(0)),
            restart: Arc::new(AtomicUsize::new(0)),
        }
    }
}

struct FakeBackend {
    observations: VecDeque<ServiceObservation>,
    dispatch: Result<DispatchEvidence, BackendError>,
    calls: CallCounters,
}

impl FakeBackend {
    fn new(
        observations: Vec<ServiceObservation>,
        dispatch: Result<DispatchEvidence, BackendError>,
    ) -> (Self, CallCounters) {
        let calls = CallCounters::default();
        (
            Self {
                observations: observations.into(),
                dispatch,
                calls: calls.clone(),
            },
            calls,
        )
    }
}

impl ServiceBackend for FakeBackend {
    type Error = BackendError;

    fn observe(
        &mut self,
        _host: &HostId,
        _unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error> {
        self.calls.observe.fetch_add(1, Ordering::SeqCst);
        self.observations.pop_front().ok_or(BackendError)
    }

    fn restart(
        &mut self,
        _host: &HostId,
        _unit: &ServiceUnit,
    ) -> Result<DispatchEvidence, Self::Error> {
        self.calls.restart.fetch_add(1, Ordering::SeqCst);
        self.dispatch
    }
}

#[derive(Debug, Error)]
#[error("hostile test checkpoint failure")]
struct StoreError;

#[derive(Default)]
struct FakeStore {
    wrong_head: bool,
}

impl CheckpointStore for FakeStore {
    type Error = StoreError;

    fn persist(
        &mut self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error> {
        let mut head = checkpoint.head().map_err(|_| StoreError)?;
        if self.wrong_head {
            head.digest = Digest32([0xA5; 32]);
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

fn context(epoch: u64) -> AuthorityContext {
    AuthorityContext {
        now_unix_s: 1_800_000_000,
        current_epoch: AuthorityEpoch(epoch),
        // Deliberately bogus. The broker must use its own GrantAccount state.
        use_state: GrantUseState {
            committed: 999,
            reserved: 999,
        },
    }
}

fn assert_no_backend_calls(calls: &CallCounters) {
    assert_eq!(calls.observe.load(Ordering::SeqCst), 0);
    assert_eq!(calls.restart.load(Ordering::SeqCst), 0);
}

#[test]
fn wrong_subject_denied_before_backend() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let mut grant = grant_for(&p);
    grant.subject = PrincipalId("agent:other".into());
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-subject".into()),
            ReservationId("res-subject".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::ActorMismatch)
    ));
    assert_no_backend_calls(&calls);
}

#[test]
fn wrong_executor_denied_before_backend() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let mut grant = grant_for(&p);
    grant.audience = Some(PrincipalId("workload:other-broker".into()));
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-audience".into()),
            ReservationId("res-audience".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::ExecutorMismatch)
    ));
    assert_no_backend_calls(&calls);
}

#[test]
fn wrong_task_denied_before_backend() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let mut grant = grant_for(&p);
    grant.task = Some(TaskId("task:repair-sshd".into()));
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-task".into()),
            ReservationId("res-task".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::TaskMismatch)
    ));
    assert_no_backend_calls(&calls);
}

#[test]
fn second_operation_is_rejected_as_broad_authority() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let mut grant = grant_for(&p);
    grant.operations.insert(Operation("service.stop".into()));
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-ops".into()),
            ReservationId("res-ops".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::OperationScopeNotExact)
    ));
    assert_no_backend_calls(&calls);
}

#[test]
fn world_binding_substitution_is_rejected_before_backend() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let mut grant = grant_for(&p);
    grant.world_digest = Some(Digest32([0x33; 32]));
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-world".into()),
            ReservationId("res-world".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::WorldBindingMismatch)
    ));
    assert_no_backend_calls(&calls);
}

#[test]
fn stale_authority_epoch_denied_before_backend() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let grant = grant_for(&p);
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-epoch".into()),
            ReservationId("res-epoch".into()),
            context(8),
            &[],
        ),
        Err(BrokerError::AuthorityDenied(
            symthaea_authority::DenyReason::EpochStale
        ))
    ));
    assert_no_backend_calls(&calls);
}

#[test]
fn caller_cannot_fake_fresh_use_state_after_commit() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let grant = grant_for(&p);
    let (backend, calls) = FakeBackend::new(
        vec![
            before.clone(),
            before.clone(),
            observation("active", "running", "inv-2"),
        ],
        Ok(DispatchEvidence::Applied),
    );
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    broker
        .recover_once(
            &p,
            ExecutionId("exec-first".into()),
            ReservationId("res-first".into()),
            context(7),
            &[],
        )
        .unwrap();
    let observes_after_first = calls.observe.load(Ordering::SeqCst);
    let restarts_after_first = calls.restart.load(Ordering::SeqCst);

    // context() still claims absurd caller-controlled counters; the broker uses
    // the real committed account and denies before another backend call.
    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-second".into()),
            ReservationId("res-second".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::AuthorityDenied(
            symthaea_authority::DenyReason::UseBudgetExhausted
        ))
    ));
    assert_eq!(calls.observe.load(Ordering::SeqCst), observes_after_first);
    assert_eq!(calls.restart.load(Ordering::SeqCst), restarts_after_first);
}

#[test]
fn second_attempt_while_outcome_unknown_is_denied_before_backend() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let grant = grant_for(&p);
    let (backend, calls) = FakeBackend::new(
        vec![before.clone(), before.clone(), before],
        Ok(DispatchEvidence::OutcomeUnknown {
            diagnostic_digest: Digest32([0x44; 32]),
        }),
    );
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    let receipt = broker
        .recover_once(
            &p,
            ExecutionId("exec-unknown-1".into()),
            ReservationId("res-unknown-1".into()),
            context(7),
            &[],
        )
        .unwrap();
    assert_eq!(receipt.use_state.committed, 0);
    assert_eq!(receipt.use_state.reserved, 1);
    let observes_after_first = calls.observe.load(Ordering::SeqCst);
    let restarts_after_first = calls.restart.load(Ordering::SeqCst);

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-unknown-2".into()),
            ReservationId("res-unknown-2".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::AuthorityDenied(
            symthaea_authority::DenyReason::UseBudgetExhausted
        ))
    ));
    assert_eq!(calls.observe.load(Ordering::SeqCst), observes_after_first);
    assert_eq!(calls.restart.load(Ordering::SeqCst), restarts_after_first);
}

#[test]
fn wrong_checkpoint_ack_latches_containment_before_dispatch() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let grant = grant_for(&p);
    let (backend, calls) = FakeBackend::new(vec![before], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(
        grant,
        backend,
        FakeStore { wrong_head: true },
    );

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-wrong-head".into()),
            ReservationId("res-wrong-head".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::CheckpointHeadMismatch)
    ));
    assert!(broker.is_contained());
    assert_eq!(calls.restart.load(Ordering::SeqCst), 0);

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-contained".into()),
            ReservationId("res-contained".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::ContainmentRequired)
    ));
}

#[test]
fn restore_requires_exact_externally_trusted_checkpoint_head() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let grant = grant_for(&p);
    let account = GrantAccount::new(&grant);
    let checkpoint = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
    let (backend, _calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let wrong_head = CheckpointHead {
        sequence: checkpoint.head().unwrap().sequence,
        digest: Digest32([0x77; 32]),
    };

    assert!(matches!(
        SystemdRecoveryBroker::from_checkpoint(
            grant,
            checkpoint,
            wrong_head,
            backend,
            FakeStore::default(),
        ),
        Err(BrokerError::CheckpointHeadMismatch)
    ));
}

#[test]
fn unrelated_resource_name_cannot_be_smuggled_into_exact_grant() {
    let before = observation("failed", "failed", "inv-1");
    let p = plan(&before);
    let mut grant = grant_for(&p);
    grant.resources.insert(ResourceRef(
        "host://host-a/systemd/unit/sshd.service".into(),
    ));
    let (backend, calls) = FakeBackend::new(vec![], Ok(DispatchEvidence::Applied));
    let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());

    assert!(matches!(
        broker.recover_once(
            &p,
            ExecutionId("exec-resource".into()),
            ReservationId("res-resource".into()),
            context(7),
            &[],
        ),
        Err(BrokerError::ResourceScopeNotExact)
    ));
    assert_no_backend_calls(&calls);
}
