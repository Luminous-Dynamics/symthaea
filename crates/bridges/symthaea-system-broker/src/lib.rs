// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, capability-bound systemd service recovery for bounded Symthaea agency.
//!
//! This crate is intentionally narrow. v0.1 exposes only two semantic effects:
//!
//! - observe one exact `.service` unit;
//! - restart one exact `.service` unit.
//!
//! It deliberately provides no `exec(String)`, shell, arbitrary systemctl
//! subcommand, unit glob, or generic command escape hatch.
//!
//! The mutation path composes three lower layers:
//!
//! 1. `symthaea-authority` proves the exact one-use grant is currently valid;
//! 2. `symthaea-action-runtime` reserves the use/risk budget before dispatch;
//! 3. `symthaea-action-checkpoint` requires that reservation state to be
//!    durably acknowledged before the broker invokes the backend.
//!
//! The broker then re-observes immediately before dispatch to close the
//! authorization/actuation TOCTOU window, and independently observes again
//! after dispatch to verify service health. Unknown external outcomes remain
//! charged until invocation identity proves that a restart took effect.

#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::process::Command;

use serde::{Deserialize, Serialize};
use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{
    ExecutionId, GrantAccount, GrantUseState, ReservationId, RuntimeAccountingError,
};
use symthaea_authority::{
    evaluate_authority, AuthorityContext, AuthorityDecision, CapabilityGrant, Digest32,
    NegativeAuthorityFact, Operation, PrincipalId, ResourceRef, RiskBudget, TaskId,
};
use thiserror::Error;

pub const SYSTEM_BROKER_SCHEMA_VERSION: u16 = 1;
const OBSERVATION_DOMAIN: &[u8] = b"symthaea.system-broker.observation.v1\0";
const RESTART_PLAN_DOMAIN: &[u8] = b"symthaea.system-broker.restart-plan.v1\0";
const DIAGNOSTIC_DOMAIN: &[u8] = b"symthaea.system-broker.diagnostic.v1\0";
const OP_RESTART: &str = "service.restart";

/// Stable host identity used in exact resource commitments.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HostId(String);

impl HostId {
    pub fn parse(value: impl Into<String>) -> Result<Self, BrokerError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 255
            || !value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b':'))
        {
            return Err(BrokerError::InvalidHostId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HostId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Exact systemd service unit identity.
///
/// v0.1 deliberately accepts only `.service` units and a conservative ASCII
/// subset. Other unit kinds should receive separately reviewed typed effects.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServiceUnit(String);

impl ServiceUnit {
    pub fn parse(value: impl Into<String>) -> Result<Self, BrokerError> {
        let value = value.into();
        let stem = value
            .strip_suffix(".service")
            .ok_or(BrokerError::InvalidServiceUnit)?;
        if stem.is_empty()
            || value.len() > 255
            || value.contains("..")
            || !value.bytes().all(|b| {
                b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b':' | b'@')
            })
        {
            return Err(BrokerError::InvalidServiceUnit);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ServiceUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Read-only service state used for stale-state and postcondition checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceObservation {
    pub host: HostId,
    pub unit: ServiceUnit,
    pub active_state: String,
    pub sub_state: String,
    /// systemd `InvocationID` when available. A changed non-empty ID is strong
    /// local evidence that a new invocation occurred.
    pub invocation_id: Option<String>,
}

impl ServiceObservation {
    pub fn is_healthy(&self) -> bool {
        self.active_state == "active"
            && !matches!(self.sub_state.as_str(), "failed" | "dead" | "stop-sigterm")
    }

    pub fn digest(&self) -> Digest32 {
        digest_serialized(OBSERVATION_DOMAIN, self)
    }
}

/// Exact semantic plan reviewed by authority before restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestartPlan {
    pub schema_version: u16,
    pub actor: PrincipalId,
    pub executor: PrincipalId,
    pub task: Option<TaskId>,
    pub host: HostId,
    pub unit: ServiceUnit,
    /// Exact state that was inspected when this plan was authorized.
    pub world_digest: Digest32,
}

impl RestartPlan {
    pub fn new(
        actor: PrincipalId,
        executor: PrincipalId,
        task: Option<TaskId>,
        observation: &ServiceObservation,
    ) -> Self {
        Self {
            schema_version: SYSTEM_BROKER_SCHEMA_VERSION,
            actor,
            executor,
            task,
            host: observation.host.clone(),
            unit: observation.unit.clone(),
            world_digest: observation.digest(),
        }
    }

    pub fn digest(&self) -> Digest32 {
        digest_serialized(RESTART_PLAN_DOMAIN, self)
    }

    pub fn resource(&self) -> ResourceRef {
        service_resource(&self.host, &self.unit)
    }

    pub fn operation(&self) -> Operation {
        Operation(OP_RESTART.to_string())
    }
}

/// Semantic result of a backend dispatch attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchEvidence {
    /// Backend proved the restart request was accepted/applied.
    Applied,
    /// A process was dispatched but the broker cannot prove whether/how much of
    /// the external effect occurred. The use remains charged.
    OutcomeUnknown { diagnostic_digest: Digest32 },
}

/// Post-dispatch independent verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationResult {
    Healthy,
    Unhealthy,
    Unavailable,
    /// Dispatch outcome remains unresolved and no independent observation proved
    /// that a new invocation occurred.
    OutcomeUnknown,
}

/// Stable effect accounting classification retained in the durable receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryOutcome {
    Applied,
    ReconciledApplied,
    OutcomeUnknown,
}

/// Privacy-minimized recovery evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryReceipt {
    pub schema_version: u16,
    pub execution_id: ExecutionId,
    pub reservation_id: ReservationId,
    pub grant_digest: Digest32,
    pub plan_digest: Digest32,
    pub before_world_digest: Digest32,
    pub after_world_digest: Option<Digest32>,
    pub outcome: RecoveryOutcome,
    pub verification: VerificationResult,
    pub checkpoint_head: CheckpointHead,
    pub use_state: GrantUseState,
}

/// Backend boundary. Implementations may use D-Bus, systemd APIs, or a tightly
/// scoped `systemctl` process, but callers cannot supply arbitrary commands.
pub trait ServiceBackend {
    type Error: std::error::Error;

    fn observe(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error>;

    fn restart(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<DispatchEvidence, Self::Error>;
}

/// External durability boundary. `persist` must not return success until the
/// caller's required durability policy is satisfied.
///
/// The broker verifies that the returned head is exactly the checkpoint it
/// asked to persist. This trait does not by itself prove fsync, TPM, Xenia, or
/// remote witness semantics; those are properties of the concrete store.
pub trait CheckpointStore {
    type Error: std::error::Error;

    fn persist(
        &mut self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error>;
}

/// Local `systemctl` backend with no shell interpretation.
///
/// It accepts one configured local host identity and only the typed unit passed
/// by the broker. A non-zero `systemctl restart` exit is conservatively treated
/// as `OutcomeUnknown` because partial stop/start effects may already have
/// occurred even when the command reports failure.
pub struct SystemctlBackend {
    local_host: HostId,
}

impl SystemctlBackend {
    pub fn new(local_host: HostId) -> Self {
        Self { local_host }
    }

    fn require_local(&self, host: &HostId) -> Result<(), SystemctlError> {
        if host == &self.local_host {
            Ok(())
        } else {
            Err(SystemctlError::WrongHost)
        }
    }
}

impl ServiceBackend for SystemctlBackend {
    type Error = SystemctlError;

    fn observe(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error> {
        self.require_local(host)?;
        let output = Command::new("systemctl")
            .args([
                "show",
                "--no-pager",
                "--property=ActiveState,SubState,InvocationID",
                "--",
                unit.as_str(),
            ])
            .output()
            .map_err(SystemctlError::Spawn)?;
        if !output.status.success() {
            return Err(SystemctlError::ObservationFailed(digest_diagnostic(
                &output.stderr,
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut active_state = None;
        let mut sub_state = None;
        let mut invocation_id = None;
        for line in stdout.lines() {
            if let Some((key, value)) = line.split_once('=') {
                match key {
                    "ActiveState" => active_state = Some(value.to_string()),
                    "SubState" => sub_state = Some(value.to_string()),
                    "InvocationID" if !value.is_empty() => invocation_id = Some(value.to_string()),
                    _ => {}
                }
            }
        }

        Ok(ServiceObservation {
            host: host.clone(),
            unit: unit.clone(),
            active_state: active_state.ok_or(SystemctlError::MissingProperty("ActiveState"))?,
            sub_state: sub_state.ok_or(SystemctlError::MissingProperty("SubState"))?,
            invocation_id,
        })
    }

    fn restart(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<DispatchEvidence, Self::Error> {
        self.require_local(host)?;
        let output = Command::new("systemctl")
            .args(["restart", "--", unit.as_str()])
            .output()
            .map_err(SystemctlError::Spawn)?;
        if output.status.success() {
            Ok(DispatchEvidence::Applied)
        } else {
            Ok(DispatchEvidence::OutcomeUnknown {
                diagnostic_digest: digest_diagnostic(&output.stderr),
            })
        }
    }
}

#[derive(Debug, Error)]
pub enum SystemctlError {
    #[error("systemctl backend is bound to a different host identity")]
    WrongHost,
    #[error("failed to spawn systemctl: {0}")]
    Spawn(std::io::Error),
    #[error("systemctl observation failed; diagnostic commitment {0:?}")]
    ObservationFailed(Digest32),
    #[error("systemctl output missing required property {0}")]
    MissingProperty(&'static str),
}

/// One narrowly-scoped recovery session for exactly one capability grant.
pub struct SystemdRecoveryBroker<B, S> {
    grant: CapabilityGrant,
    account: GrantAccount,
    checkpoint: Option<GrantAccountCheckpoint>,
    backend: B,
    store: S,
}

impl<B, S> SystemdRecoveryBroker<B, S>
where
    B: ServiceBackend,
    S: CheckpointStore,
{
    pub fn new(grant: CapabilityGrant, backend: B, store: S) -> Self {
        let account = GrantAccount::new(&grant);
        Self {
            grant,
            account,
            checkpoint: None,
            backend,
            store,
        }
    }

    /// Restore from an externally trusted exact checkpoint head.
    pub fn from_checkpoint(
        grant: CapabilityGrant,
        checkpoint: GrantAccountCheckpoint,
        trusted_head: CheckpointHead,
        backend: B,
        store: S,
    ) -> Result<Self, BrokerError> {
        let actual_head = checkpoint.head().map_err(BrokerError::Checkpoint)?;
        if actual_head != trusted_head {
            return Err(BrokerError::CheckpointHeadMismatch);
        }
        let account = checkpoint
            .verify_payload(&grant)
            .map_err(BrokerError::Checkpoint)?;
        Ok(Self {
            grant,
            account,
            checkpoint: Some(checkpoint),
            backend,
            store,
        })
    }

    pub fn account_use_state(&self) -> GrantUseState {
        self.account.authority_use_state()
    }

    pub fn current_checkpoint_head(&self) -> Result<Option<CheckpointHead>, BrokerError> {
        self.checkpoint
            .as_ref()
            .map(GrantAccountCheckpoint::head)
            .transpose()
            .map_err(BrokerError::Checkpoint)
    }

    /// Execute one exact failed-service recovery attempt.
    ///
    /// No Phi/confidence value exists in this API. Authority is evaluated from
    /// the grant, epoch/use state, negative facts, exact plan/world bindings,
    /// and exact actor/executor/task/resource/operation scope.
    pub fn recover_once(
        &mut self,
        plan: &RestartPlan,
        execution_id: ExecutionId,
        reservation_id: ReservationId,
        authority_context: AuthorityContext,
        negative_facts: &[NegativeAuthorityFact],
    ) -> Result<RecoveryReceipt, BrokerError> {
        self.validate_plan_and_authority(plan, authority_context, negative_facts)?;

        // Independent re-observation before reserving authority. The authorized
        // snapshot must still be the exact world we are about to mutate.
        let before = self.observe_backend(&plan.host, &plan.unit)?;
        if before.digest() != plan.world_digest {
            return Err(BrokerError::StaleWorld);
        }
        if before.is_healthy() {
            return Err(BrokerError::ServiceAlreadyHealthy);
        }

        let restart_risk = restart_risk_charge();
        self.account
            .reserve_execution(reservation_id.clone(), execution_id.clone(), restart_risk)
            .map_err(BrokerError::Runtime)?;

        // The reservation MUST be durably acknowledged before an effect is
        // dispatched. If persistence fails, release the never-dispatched use.
        if let Err(error) = self.persist_successor() {
            self.account
                .cancel_before_dispatch(&reservation_id)
                .map_err(BrokerError::Runtime)?;
            return Err(error);
        }

        // Close the time-of-check/time-of-use window after the potentially slow
        // durability step. Any material service-state change invalidates this
        // exact plan and releases the reservation before actuation.
        let predispatch = self.observe_backend(&plan.host, &plan.unit)?;
        if predispatch.digest() != plan.world_digest {
            self.account
                .cancel_before_dispatch(&reservation_id)
                .map_err(BrokerError::Runtime)?;
            self.persist_successor()?;
            return Err(BrokerError::StaleWorld);
        }

        let dispatch = match self.backend.restart(&plan.host, &plan.unit) {
            Ok(value) => value,
            Err(error) => {
                // Backend failure means the typed backend could not establish a
                // dispatch result. Conservatively move to unknown unless the
                // backend failed before spawning; the trait cannot prove that
                // distinction generically, so this branch remains charged.
                self.account
                    .mark_outcome_unknown(&reservation_id)
                    .map_err(BrokerError::Runtime)?;
                self.persist_successor()?;
                return Err(BrokerError::BackendOutcomeUnknown(digest_text(
                    &error.to_string(),
                )));
            }
        };

        let mut outcome = match dispatch {
            DispatchEvidence::Applied => {
                self.account
                    .commit_observed(&reservation_id)
                    .map_err(BrokerError::Runtime)?;
                RecoveryOutcome::Applied
            }
            DispatchEvidence::OutcomeUnknown { .. } => {
                self.account
                    .mark_outcome_unknown(&reservation_id)
                    .map_err(BrokerError::Runtime)?;
                RecoveryOutcome::OutcomeUnknown
            }
        };
        self.persist_successor()?;

        // Independent post-effect observation. A changed non-empty InvocationID
        // can reconcile an unknown dispatch as applied. Absence/equality never
        // proves non-application, so unknown remains charged.
        let after = match self.observe_backend(&plan.host, &plan.unit) {
            Ok(after) => Some(after),
            Err(_) => None,
        };

        if outcome == RecoveryOutcome::OutcomeUnknown {
            if let Some(after) = &after {
                if invocation_changed(&before, after) {
                    self.account
                        .reconcile_applied(&reservation_id)
                        .map_err(BrokerError::Runtime)?;
                    self.persist_successor()?;
                    outcome = RecoveryOutcome::ReconciledApplied;
                }
            }
        }

        let verification = match (&after, outcome) {
            (Some(after), _) if after.is_healthy() => VerificationResult::Healthy,
            (Some(_), RecoveryOutcome::OutcomeUnknown) => VerificationResult::OutcomeUnknown,
            (Some(_), _) => VerificationResult::Unhealthy,
            (None, RecoveryOutcome::OutcomeUnknown) => VerificationResult::OutcomeUnknown,
            (None, _) => VerificationResult::Unavailable,
        };
        let checkpoint_head = self
            .current_checkpoint_head()?
            .ok_or(BrokerError::MissingCheckpoint)?;

        Ok(RecoveryReceipt {
            schema_version: SYSTEM_BROKER_SCHEMA_VERSION,
            execution_id,
            reservation_id,
            grant_digest: self.grant.digest(),
            plan_digest: plan.digest(),
            before_world_digest: before.digest(),
            after_world_digest: after.as_ref().map(ServiceObservation::digest),
            outcome,
            verification,
            checkpoint_head,
            use_state: self.account.authority_use_state(),
        })
    }

    fn validate_plan_and_authority(
        &self,
        plan: &RestartPlan,
        context: AuthorityContext,
        negative_facts: &[NegativeAuthorityFact],
    ) -> Result<(), BrokerError> {
        if plan.schema_version != SYSTEM_BROKER_SCHEMA_VERSION {
            return Err(BrokerError::UnsupportedSchema);
        }
        if self.account.snapshot().grant_digest != self.grant.digest() {
            return Err(BrokerError::GrantAccountMismatch);
        }
        if self.grant.subject != plan.actor {
            return Err(BrokerError::ActorMismatch);
        }
        if self.grant.audience.as_ref() != Some(&plan.executor) {
            return Err(BrokerError::ExecutorMismatch);
        }
        if self.grant.task != plan.task {
            return Err(BrokerError::TaskMismatch);
        }

        let expected_resources = BTreeSet::from([plan.resource()]);
        if self.grant.resources != expected_resources {
            return Err(BrokerError::ResourceScopeNotExact);
        }
        let expected_operations = BTreeSet::from([plan.operation()]);
        if self.grant.operations != expected_operations {
            return Err(BrokerError::OperationScopeNotExact);
        }
        if self.grant.plan_digest != Some(plan.digest()) {
            return Err(BrokerError::PlanBindingMismatch);
        }
        if self.grant.world_digest != Some(plan.world_digest) {
            return Err(BrokerError::WorldBindingMismatch);
        }
        if self.grant.max_uses != 1 {
            return Err(BrokerError::GrantMustBeSingleUse);
        }
        if !restart_risk_charge().attenuates(self.grant.risk_budget) {
            return Err(BrokerError::RiskBudgetInsufficient);
        }

        let decision = evaluate_authority(
            &self.grant,
            AuthorityContext {
                now_unix_s: context.now_unix_s,
                current_epoch: context.current_epoch,
                // Never trust caller-supplied use accounting.
                use_state: self.account.authority_use_state(),
            },
            negative_facts,
        );
        match decision {
            AuthorityDecision::Allow => Ok(()),
            AuthorityDecision::Deny(reason) => Err(BrokerError::AuthorityDenied(reason)),
        }
    }

    fn observe_backend(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<ServiceObservation, BrokerError> {
        self.backend
            .observe(host, unit)
            .map_err(|error| BrokerError::BackendObservation(digest_text(&error.to_string())))
    }

    fn persist_successor(&mut self) -> Result<CheckpointHead, BrokerError> {
        let snapshot = self.account.snapshot();
        let checkpoint = match &self.checkpoint {
            Some(previous) => GrantAccountCheckpoint::successor(previous, &self.grant, snapshot),
            None => GrantAccountCheckpoint::first(&self.grant, snapshot),
        }
        .map_err(BrokerError::Checkpoint)?;
        let expected_head = checkpoint.head().map_err(BrokerError::Checkpoint)?;
        let acknowledged = self
            .store
            .persist(&checkpoint)
            .map_err(|error| BrokerError::CheckpointStore(digest_text(&error.to_string())))?;
        if acknowledged != expected_head {
            return Err(BrokerError::CheckpointHeadMismatch);
        }
        self.checkpoint = Some(checkpoint);
        Ok(expected_head)
    }
}

pub fn service_resource(host: &HostId, unit: &ServiceUnit) -> ResourceRef {
    ResourceRef(format!("host://{}/systemd/unit/{}", host.as_str(), unit.as_str()))
}

pub fn restart_operation() -> Operation {
    Operation(OP_RESTART.to_string())
}

pub fn restart_risk_charge() -> RiskBudget {
    RiskBudget {
        mutation_units: 1,
        irreversible_units: 0,
        external_disclosure_bytes: 0,
        monetary_microunits: 0,
    }
}

fn invocation_changed(before: &ServiceObservation, after: &ServiceObservation) -> bool {
    matches!(
        (&before.invocation_id, &after.invocation_id),
        (Some(before), Some(after)) if !before.is_empty() && !after.is_empty() && before != after
    )
}

fn digest_serialized<T: Serialize>(domain: &[u8], value: &T) -> Digest32 {
    let encoded = bincode::serialize(value).expect("security type serialization must be infallible");
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&encoded);
    Digest32(*hasher.finalize().as_bytes())
}

fn digest_diagnostic(bytes: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DIAGNOSTIC_DOMAIN);
    hasher.update(bytes);
    Digest32(*hasher.finalize().as_bytes())
}

fn digest_text(text: &str) -> Digest32 {
    digest_diagnostic(text.as_bytes())
}

#[derive(Debug, Error)]
pub enum BrokerError {
    #[error("invalid host identity")]
    InvalidHostId,
    #[error("invalid or unsupported systemd service unit")]
    InvalidServiceUnit,
    #[error("unsupported broker schema")]
    UnsupportedSchema,
    #[error("grant account does not match capability grant")]
    GrantAccountMismatch,
    #[error("plan actor does not match capability subject")]
    ActorMismatch,
    #[error("capability must be bound to the exact broker executor")]
    ExecutorMismatch,
    #[error("plan task does not exactly match capability task binding")]
    TaskMismatch,
    #[error("capability resource scope is not exactly one requested service")]
    ResourceScopeNotExact,
    #[error("capability operation scope is not exactly service.restart")]
    OperationScopeNotExact,
    #[error("capability is not bound to the exact restart plan")]
    PlanBindingMismatch,
    #[error("capability is not bound to the exact inspected world")]
    WorldBindingMismatch,
    #[error("systemd recovery witness requires a single-use capability")]
    GrantMustBeSingleUse,
    #[error("capability risk budget cannot cover one service restart")]
    RiskBudgetInsufficient,
    #[error("authority kernel denied the grant: {0:?}")]
    AuthorityDenied(symthaea_authority::DenyReason),
    #[error("authorized world snapshot is stale")]
    StaleWorld,
    #[error("service is already healthy; restart is not minimally necessary")]
    ServiceAlreadyHealthy,
    #[error("runtime accounting error: {0}")]
    Runtime(#[from] RuntimeAccountingError),
    #[error("checkpoint validation failed: {0}")]
    Checkpoint(symthaea_action_checkpoint::CheckpointError),
    #[error("checkpoint store failed; diagnostic commitment {0:?}")]
    CheckpointStore(Digest32),
    #[error("checkpoint store acknowledged the wrong chain head")]
    CheckpointHeadMismatch,
    #[error("no durable checkpoint exists for the completed attempt")]
    MissingCheckpoint,
    #[error("backend observation failed; diagnostic commitment {0:?}")]
    BackendObservation(Digest32),
    #[error("backend dispatch outcome is unknown; diagnostic commitment {0:?}")]
    BackendOutcomeUnknown(Digest32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Debug, Error)]
    #[error("fake backend failure")]
    struct FakeBackendError;

    struct FakeBackend {
        observations: VecDeque<ServiceObservation>,
        dispatch: Result<DispatchEvidence, FakeBackendError>,
        restart_calls: usize,
    }

    impl FakeBackend {
        fn new(
            observations: Vec<ServiceObservation>,
            dispatch: Result<DispatchEvidence, FakeBackendError>,
        ) -> Self {
            Self {
                observations: observations.into(),
                dispatch,
                restart_calls: 0,
            }
        }
    }

    impl ServiceBackend for FakeBackend {
        type Error = FakeBackendError;

        fn observe(
            &mut self,
            _host: &HostId,
            _unit: &ServiceUnit,
        ) -> Result<ServiceObservation, Self::Error> {
            self.observations.pop_front().ok_or(FakeBackendError)
        }

        fn restart(
            &mut self,
            _host: &HostId,
            _unit: &ServiceUnit,
        ) -> Result<DispatchEvidence, Self::Error> {
            self.restart_calls += 1;
            self.dispatch
        }
    }

    #[derive(Debug, Error)]
    #[error("fake checkpoint failure")]
    struct FakeStoreError;

    #[derive(Default)]
    struct FakeStore {
        checkpoints: Vec<GrantAccountCheckpoint>,
        fail: bool,
    }

    impl CheckpointStore for FakeStore {
        type Error = FakeStoreError;

        fn persist(
            &mut self,
            checkpoint: &GrantAccountCheckpoint,
        ) -> Result<CheckpointHead, Self::Error> {
            if self.fail {
                return Err(FakeStoreError);
            }
            self.checkpoints.push(checkpoint.clone());
            checkpoint.head().map_err(|_| FakeStoreError)
        }
    }

    fn host() -> HostId {
        HostId::parse("host-a").unwrap()
    }

    fn unit() -> ServiceUnit {
        ServiceUnit::parse("postgresql.service").unwrap()
    }

    fn obs(active: &str, sub: &str, invocation: &str) -> ServiceObservation {
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
            PrincipalId("user:alice".into()),
            plan.actor.clone(),
            symthaea_authority::AuthorityEpoch(7),
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

    fn ctx() -> AuthorityContext {
        AuthorityContext {
            now_unix_s: 1_800_000_000,
            current_epoch: symthaea_authority::AuthorityEpoch(7),
            // The broker deliberately ignores this caller value.
            use_state: GrantUseState {
                committed: 999,
                reserved: 999,
            },
        }
    }

    #[test]
    fn unit_parser_rejects_non_service_and_shell_like_names() {
        assert!(ServiceUnit::parse("nginx.service").is_ok());
        assert!(ServiceUnit::parse("nginx.socket").is_err());
        assert!(ServiceUnit::parse("nginx.service;rm").is_err());
        assert!(ServiceUnit::parse("../nginx.service").is_err());
        assert!(ServiceUnit::parse("nginx service.service").is_err());
    }

    #[test]
    fn successful_recovery_is_one_use_and_verified() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let backend = FakeBackend::new(
            vec![before.clone(), before.clone(), obs("active", "running", "inv-2")],
            Ok(DispatchEvidence::Applied),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        let receipt = broker
            .recover_once(
                &p,
                ExecutionId("exec-1".into()),
                ReservationId("res-1".into()),
                ctx(),
                &[],
            )
            .unwrap();
        assert_eq!(receipt.outcome, RecoveryOutcome::Applied);
        assert_eq!(receipt.verification, VerificationResult::Healthy);
        assert_eq!(receipt.use_state.committed, 1);
        assert_eq!(receipt.use_state.reserved, 0);
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-2".into()),
                ReservationId("res-2".into()),
                ctx(),
                &[],
            ),
            Err(BrokerError::AuthorityDenied(
                symthaea_authority::DenyReason::UseBudgetExhausted
            ))
        ));
    }

    #[test]
    fn stale_state_after_durable_reservation_never_dispatches() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let changed = obs("inactive", "dead", "inv-1");
        let backend = FakeBackend::new(
            vec![before, changed],
            Ok(DispatchEvidence::Applied),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-stale".into()),
                ReservationId("res-stale".into()),
                ctx(),
                &[],
            ),
            Err(BrokerError::StaleWorld)
        ));
        assert_eq!(broker.account_use_state(), GrantUseState::default());
        assert_eq!(broker.backend.restart_calls, 0);
    }

    #[test]
    fn checkpoint_failure_prevents_dispatch_and_releases_use() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let backend = FakeBackend::new(
            vec![before],
            Ok(DispatchEvidence::Applied),
        );
        let store = FakeStore {
            checkpoints: Vec::new(),
            fail: true,
        };
        let mut broker = SystemdRecoveryBroker::new(grant, backend, store);
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-store-fail".into()),
                ReservationId("res-store-fail".into()),
                ctx(),
                &[],
            ),
            Err(BrokerError::CheckpointStore(_))
        ));
        assert_eq!(broker.account_use_state(), GrantUseState::default());
        assert_eq!(broker.backend.restart_calls, 0);
    }

    #[test]
    fn unknown_dispatch_reconciles_only_when_invocation_changes() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let backend = FakeBackend::new(
            vec![before.clone(), before, obs("active", "running", "inv-2")],
            Ok(DispatchEvidence::OutcomeUnknown {
                diagnostic_digest: digest_text("lost response"),
            }),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        let receipt = broker
            .recover_once(
                &p,
                ExecutionId("exec-unknown".into()),
                ReservationId("res-unknown".into()),
                ctx(),
                &[],
            )
            .unwrap();
        assert_eq!(receipt.outcome, RecoveryOutcome::ReconciledApplied);
        assert_eq!(receipt.verification, VerificationResult::Healthy);
        assert_eq!(receipt.use_state.committed, 1);
        assert_eq!(receipt.use_state.reserved, 0);
    }

    #[test]
    fn unresolved_unknown_outcome_remains_reserved() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let backend = FakeBackend::new(
            vec![before.clone(), before.clone(), before],
            Ok(DispatchEvidence::OutcomeUnknown {
                diagnostic_digest: digest_text("timeout"),
            }),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        let receipt = broker
            .recover_once(
                &p,
                ExecutionId("exec-unknown".into()),
                ReservationId("res-unknown".into()),
                ctx(),
                &[],
            )
            .unwrap();
        assert_eq!(receipt.outcome, RecoveryOutcome::OutcomeUnknown);
        assert_eq!(receipt.verification, VerificationResult::OutcomeUnknown);
        assert_eq!(receipt.use_state.committed, 0);
        assert_eq!(receipt.use_state.reserved, 1);
    }

    #[test]
    fn negative_authority_dominates_before_any_backend_call() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let grant = grant_for(&p);
        let fact = NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        };
        let backend = FakeBackend::new(
            vec![before],
            Ok(DispatchEvidence::Applied),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-revoked".into()),
                ReservationId("res-revoked".into()),
                ctx(),
                &[fact],
            ),
            Err(BrokerError::AuthorityDenied(
                symthaea_authority::DenyReason::ExplicitlyRevoked
            ))
        ));
        assert_eq!(broker.backend.restart_calls, 0);
    }

    #[test]
    fn broad_multi_service_grant_is_rejected_by_witness() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let mut grant = grant_for(&p);
        grant.resources.insert(ResourceRef(
            "host://host-a/systemd/unit/sshd.service".into(),
        ));
        let backend = FakeBackend::new(
            vec![before],
            Ok(DispatchEvidence::Applied),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-broad".into()),
                ReservationId("res-broad".into()),
                ctx(),
                &[],
            ),
            Err(BrokerError::ResourceScopeNotExact)
        ));
        assert_eq!(broker.backend.restart_calls, 0);
    }

    #[test]
    fn already_healthy_service_is_not_restarted() {
        let healthy = obs("active", "running", "inv-1");
        let p = plan(&healthy);
        let grant = grant_for(&p);
        let backend = FakeBackend::new(
            vec![healthy],
            Ok(DispatchEvidence::Applied),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-healthy".into()),
                ReservationId("res-healthy".into()),
                ctx(),
                &[],
            ),
            Err(BrokerError::ServiceAlreadyHealthy)
        ));
        assert_eq!(broker.backend.restart_calls, 0);
    }

    #[test]
    fn plan_and_world_binding_are_exact() {
        let before = obs("failed", "failed", "inv-1");
        let p = plan(&before);
        let mut grant = grant_for(&p);
        grant.plan_digest = Some(Digest32([9; 32]));
        let backend = FakeBackend::new(
            vec![before],
            Ok(DispatchEvidence::Applied),
        );
        let mut broker = SystemdRecoveryBroker::new(grant, backend, FakeStore::default());
        assert!(matches!(
            broker.recover_once(
                &p,
                ExecutionId("exec-plan".into()),
                ReservationId("res-plan".into()),
                ctx(),
                &[],
            ),
            Err(BrokerError::PlanBindingMismatch)
        ));
        assert_eq!(broker.backend.restart_calls, 0);
    }
}
