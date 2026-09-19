// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Concrete adapter from the `Symthaea` facade into the single-owner service runtime.
//!
//! This tranche is intentionally query-only. Callers receive a narrow query
//! capability rather than a raw `ServiceMutationCommand` handle, while the owner
//! still rejects unsupported mutation variants defensively.

pub mod host;
pub mod protocol;
pub mod protocol_error;
pub mod read_wire;
pub mod semantic;
pub mod service;
pub mod telemetry;
pub mod wire;

use std::fmt;

use symthaea::Symthaea;
use symthaea::symthaea::{IntrospectionResult, PartnershipState, ProcessResponse};
use symthaea_interface_runtime::StatePlaneError;
use symthaea_runtime_owner::{
    HandlerFuture, OwnerSubmitError, RuntimeCommandTicket, RuntimeOwnerConfigError,
    RuntimeOwnerExit, RuntimeOwnerHandle, spawn_runtime_owner,
};
use symthaea_service_runtime::observed::{
    ObservedServiceHandler, ObservedServiceReply, ServiceMutationExecutor, ServiceSnapshotter,
};
use symthaea_service_runtime::state_plane::{ServiceStatePlaneHub, service_state_planes};
use symthaea_service_runtime::{
    CognitiveSummary, PartnershipSummary, ProcessOrigin, ServiceCommandKind,
    ServiceMutationCommand, ServiceRuntimeSnapshot,
};
use tokio::task::JoinHandle;

/// Successful processing result with the ingress origin preserved for later
/// semantic-event correlation.
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub origin: ProcessOrigin,
    pub response: ProcessResponse,
}

/// Query execution failure. Unsupported command kinds are explicit rather than
/// silently routed around the owner boundary.
#[derive(Debug)]
pub enum SymthaeaQueryError {
    UnsupportedCommand(ServiceCommandKind),
    Process {
        origin: ProcessOrigin,
        source: anyhow::Error,
    },
}

impl fmt::Display for SymthaeaQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCommand(kind) => {
                write!(f, "query-only Symthaea runtime does not support {kind:?}")
            }
            Self::Process { origin, source } => {
                write!(f, "Symthaea process failed for {origin:?}: {source}")
            }
        }
    }
}

impl std::error::Error for SymthaeaQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Process { source, .. } => Some(source.as_ref()),
            Self::UnsupportedCommand(_) => None,
        }
    }
}

/// Concrete owner-side query executor for the production `Symthaea` facade.
#[derive(Debug, Default, Clone, Copy)]
pub struct SymthaeaQueryExecutor;

impl ServiceMutationExecutor<Symthaea> for SymthaeaQueryExecutor {
    type Reply = Result<ProcessedQuery, SymthaeaQueryError>;

    fn execute<'a>(
        &'a mut self,
        engine: &'a mut Symthaea,
        _context: symthaea_runtime_owner::OwnerCommandContext,
        command: ServiceMutationCommand,
    ) -> HandlerFuture<'a, Self::Reply> {
        Box::pin(async move {
            match command {
                ServiceMutationCommand::ProcessText { content, origin } => engine
                    .process(&content)
                    .await
                    .map(|response| ProcessedQuery { origin, response })
                    .map_err(|source| SymthaeaQueryError::Process { origin, source }),
                other => Err(SymthaeaQueryError::UnsupportedCommand(other.kind())),
            }
        })
    }
}

fn summaries_from_read_models(
    introspection: &IntrospectionResult,
    partnership: &PartnershipState,
) -> (CognitiveSummary, PartnershipSummary) {
    (
        CognitiveSummary {
            consciousness_level: introspection.consciousness_level,
            self_loops: introspection.self_loops,
            graph_size: introspection.graph_size,
            complexity: introspection.complexity,
            short_term_memories: introspection.memory_stats.short_term_count,
            long_term_memories: introspection.memory_stats.long_term_count,
        },
        PartnershipSummary {
            stage: format!("{:?}", partnership.stage),
            trust: partnership.trust,
            vulnerability: partnership.vulnerability,
            reciprocity: partnership.reciprocity,
            phi_dyad: partnership.phi_dyad,
            interactions: partnership.interactions,
            trajectory_points: partnership.trajectory_points,
        },
    )
}

fn current_summaries(engine: &Symthaea) -> (CognitiveSummary, PartnershipSummary) {
    let introspection = engine.introspect();
    let partnership = engine.partnership_state();
    summaries_from_read_models(&introspection, &partnership)
}

/// Capture the real initialized facade state before ownership handoff.
pub fn startup_snapshot(engine: &Symthaea) -> ServiceRuntimeSnapshot {
    let (cognition, partnership) = current_summaries(engine);
    ServiceRuntimeSnapshot::initialized(cognition, partnership)
}

/// Extract a completed post-command snapshot while still inside the owner task.
#[derive(Debug, Default, Clone, Copy)]
pub struct SymthaeaSnapshotter;

impl ServiceSnapshotter<Symthaea> for SymthaeaSnapshotter {
    fn snapshot(
        &mut self,
        engine: &Symthaea,
        context: symthaea_runtime_owner::OwnerCommandContext,
    ) -> ServiceRuntimeSnapshot {
        let (cognition, partnership) = current_summaries(engine);
        ServiceRuntimeSnapshot::after_command(context, cognition, partnership)
    }
}

pub type SymthaeaObservedQueryReply =
    ObservedServiceReply<Result<ProcessedQuery, SymthaeaQueryError>, StatePlaneError>;

/// Narrow query capability. It intentionally does not expose the underlying raw
/// `RuntimeOwnerHandle<ServiceMutationCommand, _>`.
#[derive(Clone)]
pub struct SymthaeaQueryHandle {
    inner: RuntimeOwnerHandle<ServiceMutationCommand, SymthaeaObservedQueryReply>,
}

impl SymthaeaQueryHandle {
    pub fn try_query(
        &self,
        content: impl Into<String>,
        origin: ProcessOrigin,
    ) -> Result<
        RuntimeCommandTicket<SymthaeaObservedQueryReply>,
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

/// Running query-only Symthaea owner plus read-only latest-wins state hub.
pub struct SymthaeaQueryRuntime {
    pub queries: SymthaeaQueryHandle,
    pub state: ServiceStatePlaneHub,
    pub task: JoinHandle<RuntimeOwnerExit>,
}

#[derive(Debug)]
pub enum SymthaeaQueryRuntimeSpawnError {
    StatePlane(StatePlaneError),
    Owner(RuntimeOwnerConfigError),
}

impl fmt::Display for SymthaeaQueryRuntimeSpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StatePlane(error) => write!(f, "failed to initialize service state plane: {error}"),
            Self::Owner(error) => write!(f, "failed to spawn Symthaea runtime owner: {error}"),
        }
    }
}

impl std::error::Error for SymthaeaQueryRuntimeSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StatePlane(error) => Some(error),
            Self::Owner(error) => Some(error),
        }
    }
}

fn validate_mailbox_capacity(
    mailbox_capacity: usize,
) -> Result<(), SymthaeaQueryRuntimeSpawnError> {
    if mailbox_capacity == 0 {
        return Err(SymthaeaQueryRuntimeSpawnError::Owner(
            RuntimeOwnerConfigError::ZeroMailboxCapacity,
        ));
    }
    Ok(())
}

/// Hand one initialized `Symthaea` facade to the sole mutable owner and expose only
/// query admission plus read-only state subscriptions.
pub fn spawn_query_runtime(
    symthaea: Symthaea,
    mailbox_capacity: usize,
) -> Result<SymthaeaQueryRuntime, SymthaeaQueryRuntimeSpawnError> {
    validate_mailbox_capacity(mailbox_capacity)?;

    let initial_snapshot = startup_snapshot(&symthaea);
    let (publisher, state) =
        service_state_planes(initial_snapshot).map_err(SymthaeaQueryRuntimeSpawnError::StatePlane)?;

    let handler = ObservedServiceHandler::new(SymthaeaQueryExecutor, SymthaeaSnapshotter, publisher);
    let (inner, task) = spawn_runtime_owner(symthaea, handler, mailbox_capacity)
        .map_err(SymthaeaQueryRuntimeSpawnError::Owner)?;

    Ok(SymthaeaQueryRuntime {
        queries: SymthaeaQueryHandle { inner },
        state,
        task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea::hdc::relational_consciousness::RelationshipStage;
    use symthaea::symthaea::MemoryStats;

    #[test]
    fn read_models_map_without_threshold_derived_labels() {
        let introspection = IntrospectionResult {
            consciousness_level: 0.73,
            self_loops: 4,
            graph_size: 12,
            complexity: 2.5,
            memory_stats: MemoryStats {
                short_term_count: 10,
                long_term_count: 7,
            },
        };
        let partnership = PartnershipState {
            stage: RelationshipStage::Contact,
            trust: 0.4,
            vulnerability: 0.2,
            reciprocity: 0.3,
            phi_dyad: 0.11,
            interactions: 8,
            trajectory_points: 5,
        };

        let (cognition, relational) = summaries_from_read_models(&introspection, &partnership);

        assert_eq!(cognition.consciousness_level, 0.73);
        assert_eq!(cognition.total_memories(), 17);
        assert_eq!(relational.stage, "Contact");
        assert_eq!(relational.interactions, 8);
        assert_eq!(relational.trajectory_points, 5);
    }

    #[test]
    fn zero_mailbox_capacity_fails_before_runtime_bootstrap() {
        assert!(matches!(
            validate_mailbox_capacity(0),
            Err(SymthaeaQueryRuntimeSpawnError::Owner(
                RuntimeOwnerConfigError::ZeroMailboxCapacity
            ))
        ));
        assert!(validate_mailbox_capacity(1).is_ok());
    }
}
