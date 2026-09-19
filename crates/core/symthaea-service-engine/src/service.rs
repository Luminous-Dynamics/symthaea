// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full demonstrated mutation surface for moving the service daemon off
//! `Mutex<Symthaea>`.
//!
//! This adapter intentionally supports only mutations the current daemon actually
//! performs: query processing, sleep/consolidation, explicit save, and optional
//! shutdown persistence. `BackgroundCycle` remains unsupported because the daemon's
//! ordinary maintenance loop is currently read-only introspection.

use std::fmt;
use std::path::PathBuf;

use symthaea::Symthaea;
use symthaea::symthaea::SleepReport;
use symthaea_interface_events::EventPlaneError;
use symthaea_interface_runtime::StatePlaneError;
use symthaea_interface_types::RuntimeId;
use symthaea_runtime_owner::{
    HandlerFuture, OwnerCommandContext, OwnerSubmitError, RuntimeCommandTicket,
    RuntimeOwnerConfigError, RuntimeOwnerExit, RuntimeOwnerHandle, spawn_runtime_owner,
};
use symthaea_service_events::{ServiceEventHub, service_event_plane};
use symthaea_service_runtime::observed::{
    ObservedServiceHandler, ObservedServiceReply, ServiceMutationExecutor,
};
use symthaea_service_runtime::state_plane::{ServiceStatePlaneHub, service_state_planes};
use symthaea_service_runtime::{BackgroundCycleKind, ProcessOrigin, ServiceMutationCommand};
use tokio::task::JoinHandle;

use super::semantic::{SemanticQueryExecution, SemanticSymthaeaQueryExecutor};
use super::{SymthaeaSnapshotter, startup_snapshot};

/// Result of one owner-executed service mutation.
#[derive(Debug)]
pub enum SymthaeaServiceExecution {
    Query(SemanticQueryExecution),
    Sleep(Result<SleepReport, anyhow::Error>),
    Save {
        path: PathBuf,
        result: Result<(), anyhow::Error>,
    },
    ShutdownPersist {
        path: Option<PathBuf>,
        result: Result<(), anyhow::Error>,
    },
    UnsupportedBackgroundCycle(BackgroundCycleKind),
}

/// Concrete executor for the mutation surface demonstrated by the current daemon.
pub struct SymthaeaServiceExecutor {
    query: SemanticSymthaeaQueryExecutor,
}

impl SymthaeaServiceExecutor {
    pub fn new(query: SemanticSymthaeaQueryExecutor) -> Self {
        Self { query }
    }
}

impl ServiceMutationExecutor<Symthaea> for SymthaeaServiceExecutor {
    type Reply = SymthaeaServiceExecution;

    fn execute<'a>(
        &'a mut self,
        engine: &'a mut Symthaea,
        context: OwnerCommandContext,
        command: ServiceMutationCommand,
    ) -> HandlerFuture<'a, Self::Reply> {
        Box::pin(async move {
            match command {
                ServiceMutationCommand::ProcessText { content, origin } => {
                    let execution = self
                        .query
                        .execute(
                            engine,
                            context,
                            ServiceMutationCommand::ProcessText { content, origin },
                        )
                        .await;
                    SymthaeaServiceExecution::Query(execution)
                }
                ServiceMutationCommand::Sleep => {
                    SymthaeaServiceExecution::Sleep(engine.sleep().await)
                }
                ServiceMutationCommand::Save { path } => {
                    let result = engine.pause(&path.to_string_lossy());
                    SymthaeaServiceExecution::Save { path, result }
                }
                ServiceMutationCommand::ShutdownPersist { path } => {
                    let result = match path.as_ref() {
                        Some(path) => engine.pause(&path.to_string_lossy()),
                        None => Ok(()),
                    };
                    SymthaeaServiceExecution::ShutdownPersist { path, result }
                }
                ServiceMutationCommand::BackgroundCycle { kind } => {
                    SymthaeaServiceExecution::UnsupportedBackgroundCycle(kind)
                }
            }
        })
    }
}

pub type SymthaeaObservedServiceReply =
    ObservedServiceReply<SymthaeaServiceExecution, StatePlaneError>;

/// Narrow service mutation capability. There is still no raw mutable `Symthaea`
/// access and no public generic `submit(ServiceMutationCommand)` escape hatch.
#[derive(Clone)]
pub struct SymthaeaServiceHandle {
    inner: RuntimeOwnerHandle<ServiceMutationCommand, SymthaeaObservedServiceReply>,
}

impl SymthaeaServiceHandle {
    pub fn try_query(
        &self,
        content: impl Into<String>,
        origin: ProcessOrigin,
    ) -> Result<
        RuntimeCommandTicket<SymthaeaObservedServiceReply>,
        OwnerSubmitError<ServiceMutationCommand>,
    > {
        self.inner.try_submit(ServiceMutationCommand::ProcessText {
            content: content.into(),
            origin,
        })
    }

    pub fn try_sleep(
        &self,
    ) -> Result<
        RuntimeCommandTicket<SymthaeaObservedServiceReply>,
        OwnerSubmitError<ServiceMutationCommand>,
    > {
        self.inner.try_submit(ServiceMutationCommand::Sleep)
    }

    pub fn try_save(
        &self,
        path: PathBuf,
    ) -> Result<
        RuntimeCommandTicket<SymthaeaObservedServiceReply>,
        OwnerSubmitError<ServiceMutationCommand>,
    > {
        self.inner.try_submit(ServiceMutationCommand::Save { path })
    }

    pub fn try_shutdown_persist(
        &self,
        path: Option<PathBuf>,
    ) -> Result<
        RuntimeCommandTicket<SymthaeaObservedServiceReply>,
        OwnerSubmitError<ServiceMutationCommand>,
    > {
        self.inner
            .try_submit(ServiceMutationCommand::ShutdownPersist { path })
    }

    pub fn mailbox_capacity(&self) -> usize {
        self.inner.mailbox_capacity()
    }

    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity()
    }
}

/// Runtime capabilities needed to replace the daemon's service-wide cognitive
/// mutex while keeping read traffic outside the mutation mailbox.
pub struct SymthaeaServiceRuntime {
    pub commands: SymthaeaServiceHandle,
    pub state: ServiceStatePlaneHub,
    pub events: ServiceEventHub,
    pub task: JoinHandle<RuntimeOwnerExit>,
}

#[derive(Debug)]
pub enum SymthaeaServiceRuntimeSpawnError {
    StatePlane(StatePlaneError),
    EventPlane(EventPlaneError),
    Owner(RuntimeOwnerConfigError),
}

impl fmt::Display for SymthaeaServiceRuntimeSpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StatePlane(error) => write!(f, "failed to initialize service state plane: {error}"),
            Self::EventPlane(error) => write!(f, "failed to initialize semantic event plane: {error}"),
            Self::Owner(error) => write!(f, "failed to spawn Symthaea runtime owner: {error}"),
        }
    }
}

impl std::error::Error for SymthaeaServiceRuntimeSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StatePlane(error) => Some(error),
            Self::EventPlane(error) => Some(error),
            Self::Owner(error) => Some(error),
        }
    }
}

fn validate_mailbox_capacity(
    mailbox_capacity: usize,
) -> Result<(), SymthaeaServiceRuntimeSpawnError> {
    if mailbox_capacity == 0 {
        return Err(SymthaeaServiceRuntimeSpawnError::Owner(
            RuntimeOwnerConfigError::ZeroMailboxCapacity,
        ));
    }
    Ok(())
}

/// Create the complete demonstrated service mutation owner.
pub fn spawn_service_runtime(
    symthaea: Symthaea,
    runtime_id: RuntimeId,
    mailbox_capacity: usize,
    event_retention_capacity: usize,
) -> Result<SymthaeaServiceRuntime, SymthaeaServiceRuntimeSpawnError> {
    validate_mailbox_capacity(mailbox_capacity)?;

    let (event_emitter, events) = service_event_plane(runtime_id, event_retention_capacity)
        .map_err(SymthaeaServiceRuntimeSpawnError::EventPlane)?;

    let initial_snapshot = startup_snapshot(&symthaea);
    let (state_publisher, state) = service_state_planes(initial_snapshot)
        .map_err(SymthaeaServiceRuntimeSpawnError::StatePlane)?;

    let query = SemanticSymthaeaQueryExecutor::new(event_emitter);
    let executor = SymthaeaServiceExecutor::new(query);
    let handler = ObservedServiceHandler::new(executor, SymthaeaSnapshotter, state_publisher);
    let (inner, task) = spawn_runtime_owner(symthaea, handler, mailbox_capacity)
        .map_err(SymthaeaServiceRuntimeSpawnError::Owner)?;

    Ok(SymthaeaServiceRuntime {
        commands: SymthaeaServiceHandle { inner },
        state,
        events,
        task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_mailbox_capacity_fails_before_runtime_bootstrap() {
        assert!(matches!(
            validate_mailbox_capacity(0),
            Err(SymthaeaServiceRuntimeSpawnError::Owner(
                RuntimeOwnerConfigError::ZeroMailboxCapacity
            ))
        ));
        assert!(validate_mailbox_capacity(1).is_ok());
    }
}
