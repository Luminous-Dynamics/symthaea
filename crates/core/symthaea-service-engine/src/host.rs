// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Daemon-facing host for the single-owner Symthaea service runtime.
//!
//! The host makes the responsiveness boundary explicit:
//! mutating operations enter the bounded owner mailbox, while status,
//! introspection, partnership, and semantic-event subscriptions are read from
//! independently owned runtime planes and never await mutable cognition.

use std::fmt;
use std::path::PathBuf;

use symthaea::Symthaea;
use symthaea::symthaea::SleepReport;
use symthaea_interface_events::{EventPlaneError, SemanticEventSubscriber, SubscribeFrom};
use symthaea_interface_runtime::StatePlaneError;
use symthaea_interface_types::RuntimeId;
use symthaea_runtime_owner::{OwnerCompletionError, OwnerSubmitError, RuntimeOwnerExit};
use symthaea_service_read_model::{
    CognitiveStatusRead, IntrospectionRead, PartnershipRead, ServiceReadModel,
    ServiceReadModelError,
};
use symthaea_service_runtime::observed::ObservationIssue;
use symthaea_service_runtime::{ProcessOrigin, ServiceMutationCommand, ServiceRuntimeSnapshot};
use tokio::task::JoinHandle;

use crate::semantic::SemanticQueryExecution;
use crate::service::{
    SymthaeaObservedServiceReply, SymthaeaServiceExecution, SymthaeaServiceHandle,
    SymthaeaServiceRuntime, SymthaeaServiceRuntimeSpawnError, spawn_service_runtime,
};
use crate::telemetry::BridgeTelemetrySnapshot;

/// Cloneable daemon capability surface.
///
/// Cloning this value clones only immutable runtime identity, bounded command
/// handles, and read-plane hubs; it never clones or exposes the mutable
/// `Symthaea` facade.
#[derive(Clone)]
pub struct ServiceRuntimeHost {
    runtime_id: RuntimeId,
    commands: SymthaeaServiceHandle,
    state: symthaea_service_runtime::state_plane::ServiceStatePlaneHub,
    events: symthaea_service_events::ServiceEventHub,
}

/// Runtime host plus the owner task lifecycle handle.
pub struct HostedServiceRuntime {
    pub host: ServiceRuntimeHost,
    pub task: JoinHandle<RuntimeOwnerExit>,
}

/// Query-specific owner reply. The transport never needs to inspect the generic
/// service mutation enum to find these fields.
#[derive(Debug)]
pub struct ServiceQueryReply {
    pub execution: SemanticQueryExecution,
    pub snapshot: ServiceRuntimeSnapshot,
    pub telemetry: Option<BridgeTelemetrySnapshot>,
    pub observation_issues: Vec<ObservationIssue<StatePlaneError>>,
}

#[derive(Debug)]
pub struct ServiceSleepReply {
    pub result: Result<SleepReport, anyhow::Error>,
    pub snapshot: ServiceRuntimeSnapshot,
    pub observation_issues: Vec<ObservationIssue<StatePlaneError>>,
}

#[derive(Debug)]
pub struct ServiceSaveReply {
    pub path: PathBuf,
    pub result: Result<(), anyhow::Error>,
    pub snapshot: ServiceRuntimeSnapshot,
    pub observation_issues: Vec<ObservationIssue<StatePlaneError>>,
}

#[derive(Debug)]
pub struct ServiceShutdownReply {
    pub path: Option<PathBuf>,
    pub result: Result<(), anyhow::Error>,
    pub snapshot: ServiceRuntimeSnapshot,
    pub observation_issues: Vec<ObservationIssue<StatePlaneError>>,
}

/// Admission/completion failure at the daemon edge.
///
/// The rejected command payload is intentionally not exposed through this API. A
/// wire caller can retry its original request; service code should branch on the
/// bounded-runtime condition rather than inspect cognition input through an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceHostCommandError {
    Busy,
    Closed,
    SequenceExhausted,
    OwnerStopped,
    UnexpectedReply,
}

impl fmt::Display for ServiceHostCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => write!(f, "Symthaea runtime command mailbox is full"),
            Self::Closed => write!(f, "Symthaea runtime command mailbox is closed"),
            Self::SequenceExhausted => write!(f, "Symthaea runtime command sequence exhausted"),
            Self::OwnerStopped => write!(f, "Symthaea runtime owner stopped before replying"),
            Self::UnexpectedReply => {
                write!(f, "Symthaea runtime returned an unexpected command reply")
            }
        }
    }
}

impl std::error::Error for ServiceHostCommandError {}

impl ServiceHostCommandError {
    fn from_submit(error: OwnerSubmitError<ServiceMutationCommand>) -> Self {
        match error {
            OwnerSubmitError::Full(_) => Self::Busy,
            OwnerSubmitError::Closed(_) => Self::Closed,
            OwnerSubmitError::SequenceExhausted(_) => Self::SequenceExhausted,
        }
    }

    fn from_completion(error: OwnerCompletionError) -> Self {
        match error {
            OwnerCompletionError::OwnerStopped => Self::OwnerStopped,
        }
    }
}

#[derive(Debug)]
pub enum ServiceHostReadError {
    Subscribe(StatePlaneError),
    Read(ServiceReadModelError),
}

impl fmt::Display for ServiceHostReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Subscribe(error) => write!(f, "failed to subscribe to service state: {error}"),
            Self::Read(error) => write!(f, "failed to read service state: {error}"),
        }
    }
}

impl std::error::Error for ServiceHostReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Subscribe(error) => Some(error),
            Self::Read(error) => Some(error),
        }
    }
}

fn resolve_query_reply(
    reply: SymthaeaObservedServiceReply,
) -> Result<ServiceQueryReply, ServiceHostCommandError> {
    let execution = reply.execution;
    let snapshot = reply.snapshot;
    let observation_issues = reply.observation_issues;
    match execution {
        SymthaeaServiceExecution::Query {
            execution,
            telemetry,
        } => Ok(ServiceQueryReply {
            execution,
            snapshot,
            telemetry,
            observation_issues,
        }),
        _ => Err(ServiceHostCommandError::UnexpectedReply),
    }
}

fn resolve_sleep_reply(
    reply: SymthaeaObservedServiceReply,
) -> Result<ServiceSleepReply, ServiceHostCommandError> {
    let execution = reply.execution;
    let snapshot = reply.snapshot;
    let observation_issues = reply.observation_issues;
    match execution {
        SymthaeaServiceExecution::Sleep(result) => Ok(ServiceSleepReply {
            result,
            snapshot,
            observation_issues,
        }),
        _ => Err(ServiceHostCommandError::UnexpectedReply),
    }
}

fn resolve_save_reply(
    reply: SymthaeaObservedServiceReply,
) -> Result<ServiceSaveReply, ServiceHostCommandError> {
    let execution = reply.execution;
    let snapshot = reply.snapshot;
    let observation_issues = reply.observation_issues;
    match execution {
        SymthaeaServiceExecution::Save { path, result } => Ok(ServiceSaveReply {
            path,
            result,
            snapshot,
            observation_issues,
        }),
        _ => Err(ServiceHostCommandError::UnexpectedReply),
    }
}

fn resolve_shutdown_reply(
    reply: SymthaeaObservedServiceReply,
) -> Result<ServiceShutdownReply, ServiceHostCommandError> {
    let execution = reply.execution;
    let snapshot = reply.snapshot;
    let observation_issues = reply.observation_issues;
    match execution {
        SymthaeaServiceExecution::ShutdownPersist { path, result } => Ok(ServiceShutdownReply {
            path,
            result,
            snapshot,
            observation_issues,
        }),
        _ => Err(ServiceHostCommandError::UnexpectedReply),
    }
}

impl ServiceRuntimeHost {
    /// Stable identity of the running semantic runtime represented by this host.
    pub fn runtime_id(&self) -> &RuntimeId {
        &self.runtime_id
    }

    /// Submit a text turn through the bounded sole-owner mailbox and wait only for
    /// that admitted command's completion ticket.
    pub async fn query(
        &self,
        content: impl Into<String>,
        origin: ProcessOrigin,
    ) -> Result<ServiceQueryReply, ServiceHostCommandError> {
        let ticket = self
            .commands
            .try_query(content, origin)
            .map_err(ServiceHostCommandError::from_submit)?;
        let reply = ticket
            .resolve()
            .await
            .map_err(ServiceHostCommandError::from_completion)?;
        resolve_query_reply(reply)
    }

    pub async fn sleep(&self) -> Result<ServiceSleepReply, ServiceHostCommandError> {
        let ticket = self
            .commands
            .try_sleep()
            .map_err(ServiceHostCommandError::from_submit)?;
        let reply = ticket
            .resolve()
            .await
            .map_err(ServiceHostCommandError::from_completion)?;
        resolve_sleep_reply(reply)
    }

    pub async fn save(
        &self,
        path: PathBuf,
    ) -> Result<ServiceSaveReply, ServiceHostCommandError> {
        let ticket = self
            .commands
            .try_save(path)
            .map_err(ServiceHostCommandError::from_submit)?;
        let reply = ticket
            .resolve()
            .await
            .map_err(ServiceHostCommandError::from_completion)?;
        resolve_save_reply(reply)
    }

    pub async fn shutdown_persist(
        &self,
        path: Option<PathBuf>,
    ) -> Result<ServiceShutdownReply, ServiceHostCommandError> {
        let ticket = self
            .commands
            .try_shutdown_persist(path)
            .map_err(ServiceHostCommandError::from_submit)?;
        let reply = ticket
            .resolve()
            .await
            .map_err(ServiceHostCommandError::from_completion)?;
        resolve_shutdown_reply(reply)
    }

    /// Read the latest runtime activity and last completed cognitive snapshot.
    ///
    /// This path is intentionally synchronous and never enters the mutation
    /// mailbox. A long-running `query()` therefore cannot make status wait for
    /// mutable cognition to become available.
    pub fn read_model(&self) -> Result<ServiceReadModel, ServiceHostReadError> {
        let subscription = self
            .state
            .subscribe()
            .map_err(ServiceHostReadError::Subscribe)?;
        ServiceReadModel::peek(&subscription).map_err(ServiceHostReadError::Read)
    }

    pub fn status(&self) -> Result<CognitiveStatusRead, ServiceHostReadError> {
        Ok(self.read_model()?.status())
    }

    pub fn introspection(&self) -> Result<IntrospectionRead, ServiceHostReadError> {
        Ok(self.read_model()?.introspection())
    }

    pub fn partnership(&self) -> Result<PartnershipRead, ServiceHostReadError> {
        Ok(self.read_model()?.partnership())
    }

    pub fn subscribe_events(
        &self,
        start: SubscribeFrom,
    ) -> Result<SemanticEventSubscriber, EventPlaneError> {
        self.events.subscribe(start)
    }

    pub fn mailbox_capacity(&self) -> usize {
        self.commands.mailbox_capacity()
    }

    pub fn remaining_capacity(&self) -> usize {
        self.commands.remaining_capacity()
    }
}

impl HostedServiceRuntime {
    fn new(runtime_id: RuntimeId, runtime: SymthaeaServiceRuntime) -> Self {
        let SymthaeaServiceRuntime {
            commands,
            state,
            events,
            task,
        } = runtime;
        Self {
            host: ServiceRuntimeHost {
                runtime_id,
                commands,
                state,
                events,
            },
            task,
        }
    }
}

/// Construct the daemon-facing host around one initialized concrete Symthaea
/// facade. Runtime identity is explicit, retained on the read/control surface, and
/// shared with the semantic event lineage.
pub fn spawn_service_runtime_host(
    symthaea: Symthaea,
    runtime_id: RuntimeId,
    mailbox_capacity: usize,
    event_retention_capacity: usize,
) -> Result<HostedServiceRuntime, SymthaeaServiceRuntimeSpawnError> {
    let host_runtime_id = runtime_id.clone();
    spawn_service_runtime(
        symthaea,
        runtime_id,
        mailbox_capacity,
        event_retention_capacity,
    )
    .map(|runtime| HostedServiceRuntime::new(host_runtime_id, runtime))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_service_runtime::observed::ObservedServiceReply;
    use symthaea_service_runtime::{CognitiveSummary, PartnershipSummary};

    fn initialized_snapshot() -> ServiceRuntimeSnapshot {
        ServiceRuntimeSnapshot::initialized(
            CognitiveSummary {
                consciousness_level: 0.2,
                self_loops: 0,
                graph_size: 0,
                complexity: 0.0,
                short_term_memories: 0,
                long_term_memories: 0,
            },
            PartnershipSummary {
                stage: "test".into(),
                trust: 0.0,
                vulnerability: 0.0,
                reciprocity: 0.0,
                phi_dyad: 0.0,
                interactions: 0,
                trajectory_points: 0,
            },
        )
    }

    #[test]
    fn bounded_submit_failures_map_to_stable_daemon_conditions() {
        assert_eq!(
            ServiceHostCommandError::from_submit(OwnerSubmitError::Full(
                ServiceMutationCommand::Sleep,
            )),
            ServiceHostCommandError::Busy
        );
        assert_eq!(
            ServiceHostCommandError::from_submit(OwnerSubmitError::Closed(
                ServiceMutationCommand::Sleep,
            )),
            ServiceHostCommandError::Closed
        );
        assert_eq!(
            ServiceHostCommandError::from_submit(OwnerSubmitError::SequenceExhausted(
                ServiceMutationCommand::Sleep,
            )),
            ServiceHostCommandError::SequenceExhausted
        );
        assert_eq!(
            ServiceHostCommandError::from_completion(OwnerCompletionError::OwnerStopped),
            ServiceHostCommandError::OwnerStopped
        );
    }

    #[test]
    fn operation_specific_reply_adapter_fails_closed_on_wrong_variant() {
        let reply = ObservedServiceReply {
            execution: SymthaeaServiceExecution::Sleep(Ok(SleepReport {
                scaled: 0,
                consolidated: 0,
                pruned: 0,
                patterns_extracted: 0,
            })),
            snapshot: initialized_snapshot(),
            observation_issues: Vec::new(),
        };

        assert!(matches!(
            resolve_query_reply(reply),
            Err(ServiceHostCommandError::UnexpectedReply)
        ));
    }
}
