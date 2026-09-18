// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic-event-aware query runtime for the concrete `Symthaea` facade.
//!
//! The current facade returns a whole `ProcessResponse`, so this layer emits only
//! lifecycle facts it can prove: response start, then response finish on success or
//! a stable error event on failure. It does not fabricate incremental deltas.

use std::fmt;

use symthaea::Symthaea;
use symthaea_interface_events::EventPlaneError;
use symthaea_interface_runtime::StatePlaneError;
use symthaea_interface_types::{ErrorCode, RuntimeEventKind, RuntimeId, TurnId};
use symthaea_runtime_owner::{
    HandlerFuture, OwnerCommandContext, OwnerSubmitError, RuntimeCommandTicket,
    RuntimeOwnerConfigError, RuntimeOwnerExit, RuntimeOwnerHandle, spawn_runtime_owner,
};
use symthaea_service_events::{
    ServiceEventEmitter, ServiceEventError, ServiceEventHub, service_event_plane,
    turn_id_for_owner_context,
};
use symthaea_service_runtime::observed::{
    ObservedServiceHandler, ObservedServiceReply, ServiceMutationExecutor,
};
use symthaea_service_runtime::state_plane::{ServiceStatePlaneHub, service_state_planes};
use symthaea_service_runtime::{ProcessOrigin, ServiceMutationCommand};
use tokio::task::JoinHandle;

use super::{ProcessedQuery, SymthaeaQueryError, SymthaeaSnapshotter, startup_snapshot};

/// Stage at which semantic lifecycle publication degraded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticQueryEventStage {
    ResponseStarted,
    ResponseFinished,
    Error,
}

/// Semantic event failure kept separate from cognitive query execution.
#[derive(Debug)]
pub struct SemanticQueryEventIssue {
    pub stage: SemanticQueryEventStage,
    pub error: ServiceEventError,
}

/// Concrete query execution plus semantic-observation status.
#[derive(Debug)]
pub struct SemanticQueryExecution {
    pub turn_id: Option<TurnId>,
    pub execution: Result<ProcessedQuery, SymthaeaQueryError>,
    pub event_issues: Vec<SemanticQueryEventIssue>,
}

impl SemanticQueryExecution {
    pub fn semantic_observations_clean(&self) -> bool {
        self.event_issues.is_empty()
    }
}

/// Query executor that owns the sole semantic-event minting capability.
pub struct SemanticSymthaeaQueryExecutor {
    events: ServiceEventEmitter,
}

impl SemanticSymthaeaQueryExecutor {
    pub fn new(events: ServiceEventEmitter) -> Self {
        Self { events }
    }
}

impl ServiceMutationExecutor<Symthaea> for SemanticSymthaeaQueryExecutor {
    type Reply = SemanticQueryExecution;

    fn execute<'a>(
        &'a mut self,
        engine: &'a mut Symthaea,
        context: OwnerCommandContext,
        command: ServiceMutationCommand,
    ) -> HandlerFuture<'a, Self::Reply> {
        Box::pin(async move {
            let (content, origin) = match command {
                ServiceMutationCommand::ProcessText { content, origin } => (content, origin),
                other => {
                    return SemanticQueryExecution {
                        turn_id: None,
                        execution: Err(SymthaeaQueryError::UnsupportedCommand(other.kind())),
                        event_issues: Vec::new(),
                    };
                }
            };

            let mut event_issues = Vec::new();
            let turn_id = match turn_id_for_owner_context(context) {
                Ok(turn_id) => Some(turn_id),
                Err(error) => {
                    event_issues.push(SemanticQueryEventIssue {
                        stage: SemanticQueryEventStage::ResponseStarted,
                        error: ServiceEventError::Identity(error),
                    });
                    None
                }
            };

            let mut lifecycle_started = false;
            if let Some(turn_id) = turn_id.as_ref() {
                match self.events.emit(
                    None,
                    RuntimeEventKind::ResponseStarted {
                        turn_id: turn_id.clone(),
                    },
                ) {
                    Ok(_) => lifecycle_started = true,
                    Err(error) => event_issues.push(SemanticQueryEventIssue {
                        stage: SemanticQueryEventStage::ResponseStarted,
                        error,
                    }),
                }
            }

            let execution = engine
                .process(&content)
                .await
                .map(|response| ProcessedQuery { origin, response })
                .map_err(|source| SymthaeaQueryError::Process { origin, source });

            if lifecycle_started && execution.is_ok() {
                if let Some(turn_id) = turn_id.as_ref()
                    && let Err(error) = self.events.emit(
                        None,
                        RuntimeEventKind::ResponseFinished {
                            turn_id: turn_id.clone(),
                        },
                    )
                {
                    event_issues.push(SemanticQueryEventIssue {
                        stage: SemanticQueryEventStage::ResponseFinished,
                        error,
                    });
                }
            } else if lifecycle_started {
                match ErrorCode::new("query_process_failed") {
                    Ok(code) => {
                        if let Err(error) = self.events.emit(None, RuntimeEventKind::Error { code }) {
                            event_issues.push(SemanticQueryEventIssue {
                                stage: SemanticQueryEventStage::Error,
                                error,
                            });
                        }
                    }
                    Err(error) => event_issues.push(SemanticQueryEventIssue {
                        stage: SemanticQueryEventStage::Error,
                        error: ServiceEventError::Identity(error),
                    }),
                }
            }

            SemanticQueryExecution {
                turn_id,
                execution,
                event_issues,
            }
        })
    }
}

pub type SemanticObservedQueryReply =
    ObservedServiceReply<SemanticQueryExecution, StatePlaneError>;

/// Narrow query admission capability for the event-aware runtime.
#[derive(Clone)]
pub struct SemanticQueryHandle {
    inner: RuntimeOwnerHandle<ServiceMutationCommand, SemanticObservedQueryReply>,
}

impl SemanticQueryHandle {
    pub fn try_query(
        &self,
        content: impl Into<String>,
        origin: ProcessOrigin,
    ) -> Result<
        RuntimeCommandTicket<SemanticObservedQueryReply>,
        OwnerSubmitError<ServiceMutationCommand>,
    > {
        self.inner.try_submit(ServiceMutationCommand::ProcessText {
            content: content.into(),
            origin,
        })
    }

    pub fn mailbox_capacity(&self) -> usize {
        self.inner.mailbox_capacity()
    }

    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity()
    }
}

/// Concrete runtime exposing query admission plus read-only state and semantic
/// event capabilities.
pub struct SemanticQueryRuntime {
    pub queries: SemanticQueryHandle,
    pub state: ServiceStatePlaneHub,
    pub events: ServiceEventHub,
    pub task: JoinHandle<RuntimeOwnerExit>,
}

#[derive(Debug)]
pub enum SemanticQueryRuntimeSpawnError {
    StatePlane(StatePlaneError),
    EventPlane(EventPlaneError),
    Owner(RuntimeOwnerConfigError),
}

impl fmt::Display for SemanticQueryRuntimeSpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StatePlane(error) => write!(f, "failed to initialize service state plane: {error}"),
            Self::EventPlane(error) => write!(f, "failed to initialize semantic event plane: {error}"),
            Self::Owner(error) => write!(f, "failed to spawn Symthaea runtime owner: {error}"),
        }
    }
}

impl std::error::Error for SemanticQueryRuntimeSpawnError {
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
) -> Result<(), SemanticQueryRuntimeSpawnError> {
    if mailbox_capacity == 0 {
        return Err(SemanticQueryRuntimeSpawnError::Owner(
            RuntimeOwnerConfigError::ZeroMailboxCapacity,
        ));
    }
    Ok(())
}

/// Hand one initialized facade to the sole owner and bind its query lifecycle to
/// one explicit semantic runtime identity.
pub fn spawn_semantic_query_runtime(
    symthaea: Symthaea,
    runtime_id: RuntimeId,
    mailbox_capacity: usize,
    event_retention_capacity: usize,
) -> Result<SemanticQueryRuntime, SemanticQueryRuntimeSpawnError> {
    validate_mailbox_capacity(mailbox_capacity)?;

    // Validate semantic retention before publishing any startup cognitive state.
    let (event_emitter, events) = service_event_plane(runtime_id, event_retention_capacity)
        .map_err(SemanticQueryRuntimeSpawnError::EventPlane)?;

    let initial_snapshot = startup_snapshot(&symthaea);
    let (state_publisher, state) = service_state_planes(initial_snapshot)
        .map_err(SemanticQueryRuntimeSpawnError::StatePlane)?;

    let executor = SemanticSymthaeaQueryExecutor::new(event_emitter);
    let handler = ObservedServiceHandler::new(executor, SymthaeaSnapshotter, state_publisher);
    let (inner, task) = spawn_runtime_owner(symthaea, handler, mailbox_capacity)
        .map_err(SemanticQueryRuntimeSpawnError::Owner)?;

    Ok(SemanticQueryRuntime {
        queries: SemanticQueryHandle { inner },
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
            Err(SemanticQueryRuntimeSpawnError::Owner(
                RuntimeOwnerConfigError::ZeroMailboxCapacity
            ))
        ));
        assert!(validate_mailbox_capacity(1).is_ok());
    }

    #[test]
    fn query_failure_code_is_a_valid_stable_interface_identity() {
        let code = ErrorCode::new("query_process_failed").unwrap();
        assert_eq!(code.as_str(), "query_process_failed");
    }
}
