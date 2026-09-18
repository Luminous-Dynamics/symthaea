// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed contracts for migrating the service daemon onto the single-owner runtime.
//!
//! This crate separates mutating commands, fast runtime activity, and immutable
//! completed cognitive state. It intentionally does not implement the concrete
//! `Symthaea` handler or duplicate the daemon wire protocol.

use std::path::PathBuf;

use symthaea_runtime_owner::{OwnerCommandContext, OwnerCommandSeq};

/// Why text entered the cognitive processing path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessOrigin {
    ServiceQuery,
    VoiceTurn,
    VoiceTranscription,
}

/// Background work is still admitted through the same bounded owner mailbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundCycleKind {
    Periodic,
    PreSleep,
    Maintenance,
}

/// Stable command category suitable for responsive activity presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceCommandKind {
    ProcessText,
    Sleep,
    Save,
    ShutdownPersist,
    BackgroundCycle,
}

/// Operations that may mutate the cognitive facade and therefore must execute
/// through the sole runtime owner after startup handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceMutationCommand {
    ProcessText {
        content: String,
        origin: ProcessOrigin,
    },
    Sleep,
    Save {
        path: PathBuf,
    },
    ShutdownPersist {
        path: Option<PathBuf>,
    },
    BackgroundCycle {
        kind: BackgroundCycleKind,
    },
}

impl ServiceMutationCommand {
    pub fn kind(&self) -> ServiceCommandKind {
        match self {
            Self::ProcessText { .. } => ServiceCommandKind::ProcessText,
            Self::Sleep => ServiceCommandKind::Sleep,
            Self::Save { .. } => ServiceCommandKind::Save,
            Self::ShutdownPersist { .. } => ServiceCommandKind::ShutdownPersist,
            Self::BackgroundCycle { .. } => ServiceCommandKind::BackgroundCycle,
        }
    }

    pub fn process_origin(&self) -> Option<ProcessOrigin> {
        match self {
            Self::ProcessText { origin, .. } => Some(*origin),
            _ => None,
        }
    }
}

/// Privately constructible proof that the sole owner has begun one command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessingActivity {
    owner_command_seq: OwnerCommandSeq,
    kind: ServiceCommandKind,
}

impl ProcessingActivity {
    pub fn owner_command_seq(self) -> OwnerCommandSeq {
        self.owner_command_seq
    }

    pub fn kind(self) -> ServiceCommandKind {
        self.kind
    }
}

/// Privately constructible proof that shutdown persistence is executing inside the
/// owner rather than merely having been requested by a client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownActivity {
    owner_command_seq: OwnerCommandSeq,
}

impl ShutdownActivity {
    pub fn owner_command_seq(self) -> OwnerCommandSeq {
        self.owner_command_seq
    }
}

/// Fast control-plane activity. Activity is deliberately separate from cognitive
/// state: `Processing` does not imply any newer Phi/coherence/memory observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeActivity {
    #[default]
    Idle,
    Processing(ProcessingActivity),
    ShuttingDown(ShutdownActivity),
}

impl RuntimeActivity {
    /// Construct processing activity only from context delivered inside the owner.
    pub fn processing(context: OwnerCommandContext, kind: ServiceCommandKind) -> Self {
        Self::Processing(ProcessingActivity {
            owner_command_seq: context.sequence(),
            kind,
        })
    }

    /// Construct shutdown activity only from context delivered inside the owner.
    pub fn shutting_down(context: OwnerCommandContext) -> Self {
        Self::ShuttingDown(ShutdownActivity {
            owner_command_seq: context.sequence(),
        })
    }

    pub fn owner_command_seq(self) -> Option<OwnerCommandSeq> {
        match self {
            Self::Idle => None,
            Self::Processing(activity) => Some(activity.owner_command_seq()),
            Self::ShuttingDown(activity) => Some(activity.owner_command_seq()),
        }
    }

    pub fn is_busy(self) -> bool {
        !matches!(self, Self::Idle)
    }
}

/// Values copied directly from completed facade introspection.
///
/// Presentation-derived labels such as `is_conscious = consciousness_level > 0.5`
/// intentionally do not belong in canonical runtime state.
#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveSummary {
    pub consciousness_level: f32,
    pub self_loops: usize,
    pub graph_size: usize,
    pub complexity: f32,
    pub short_term_memories: usize,
    pub long_term_memories: usize,
}

impl CognitiveSummary {
    pub fn total_memories(&self) -> usize {
        self.short_term_memories
            .saturating_add(self.long_term_memories)
    }
}

/// Direct partnership/relational values copied from the facade.
#[derive(Debug, Clone, PartialEq)]
pub struct PartnershipSummary {
    pub stage: String,
    pub trust: f32,
    pub vulnerability: f32,
    pub reciprocity: f32,
    pub phi_dyad: f64,
    pub interactions: u64,
    pub trajectory_points: usize,
}

/// Why an immutable snapshot exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotOrigin {
    Initialized,
    AfterCommand(OwnerCommandSeq),
}

/// Immutable completed cognitive state for status/introspection/UI readers.
///
/// Fields are private so an `AfterCommand` correlation cannot be fabricated from a
/// caller-side ticket; the owner-side constructor requires `OwnerCommandContext`.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceRuntimeSnapshot {
    origin: SnapshotOrigin,
    cognition: CognitiveSummary,
    partnership: PartnershipSummary,
}

impl ServiceRuntimeSnapshot {
    pub fn initialized(cognition: CognitiveSummary, partnership: PartnershipSummary) -> Self {
        Self {
            origin: SnapshotOrigin::Initialized,
            cognition,
            partnership,
        }
    }

    pub fn after_command(
        context: OwnerCommandContext,
        cognition: CognitiveSummary,
        partnership: PartnershipSummary,
    ) -> Self {
        Self {
            origin: SnapshotOrigin::AfterCommand(context.sequence()),
            cognition,
            partnership,
        }
    }

    pub fn origin(&self) -> SnapshotOrigin {
        self.origin
    }

    pub fn cognition(&self) -> &CognitiveSummary {
        &self.cognition
    }

    pub fn partnership(&self) -> &PartnershipSummary {
        &self.partnership
    }

    pub fn memory_count(&self) -> usize {
        self.cognition.total_memories()
    }
}

/// Operational counters remain outside canonical cognitive state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServiceCounters {
    pub requests_processed: u64,
    pub sleep_cycles: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_runtime_owner::{HandlerFuture, RuntimeCommandHandler, spawn_runtime_owner};

    #[test]
    fn every_mutating_command_has_a_stable_activity_kind() {
        let commands = [
            ServiceMutationCommand::ProcessText {
                content: "hello".into(),
                origin: ProcessOrigin::ServiceQuery,
            },
            ServiceMutationCommand::Sleep,
            ServiceMutationCommand::Save {
                path: PathBuf::from("state.bin"),
            },
            ServiceMutationCommand::ShutdownPersist { path: None },
            ServiceMutationCommand::BackgroundCycle {
                kind: BackgroundCycleKind::Periodic,
            },
        ];

        assert_eq!(commands[0].kind(), ServiceCommandKind::ProcessText);
        assert_eq!(commands[1].kind(), ServiceCommandKind::Sleep);
        assert_eq!(commands[2].kind(), ServiceCommandKind::Save);
        assert_eq!(commands[3].kind(), ServiceCommandKind::ShutdownPersist);
        assert_eq!(commands[4].kind(), ServiceCommandKind::BackgroundCycle);
    }

    #[test]
    fn canonical_summary_keeps_daemon_counters_and_threshold_labels_out() {
        let summary = CognitiveSummary {
            consciousness_level: 0.73,
            self_loops: 4,
            graph_size: 12,
            complexity: 2.5,
            short_term_memories: 10,
            long_term_memories: 7,
        };
        assert_eq!(summary.total_memories(), 17);

        let counters = ServiceCounters {
            requests_processed: 99,
            sleep_cycles: 3,
        };
        assert_eq!(counters.requests_processed, 99);
        assert_eq!(counters.sleep_cycles, 3);
    }

    #[test]
    fn idle_is_the_only_default_activity() {
        assert_eq!(RuntimeActivity::default(), RuntimeActivity::Idle);
        assert!(!RuntimeActivity::Idle.is_busy());
        assert_eq!(RuntimeActivity::Idle.owner_command_seq(), None);
    }

    struct ContextBindingHandler;

    impl RuntimeCommandHandler<(), ServiceMutationCommand> for ContextBindingHandler {
        type Reply = (RuntimeActivity, ServiceRuntimeSnapshot);

        fn handle<'a>(
            &'a mut self,
            _engine: &'a mut (),
            context: OwnerCommandContext,
            command: ServiceMutationCommand,
        ) -> HandlerFuture<'a, Self::Reply> {
            Box::pin(async move {
                let activity = RuntimeActivity::processing(context, command.kind());
                let snapshot = ServiceRuntimeSnapshot::after_command(
                    context,
                    CognitiveSummary {
                        consciousness_level: 0.5,
                        self_loops: 1,
                        graph_size: 2,
                        complexity: 0.3,
                        short_term_memories: 3,
                        long_term_memories: 4,
                    },
                    PartnershipSummary {
                        stage: "forming".into(),
                        trust: 0.2,
                        vulnerability: 0.1,
                        reciprocity: 0.3,
                        phi_dyad: 0.0,
                        interactions: 1,
                        trajectory_points: 1,
                    },
                );
                (activity, snapshot)
            })
        }
    }

    #[tokio::test]
    async fn owner_context_binds_activity_and_snapshot_to_the_same_ticket() {
        let (handle, task) = spawn_runtime_owner((), ContextBindingHandler, 1).unwrap();
        let ticket = handle
            .try_submit(ServiceMutationCommand::ProcessText {
                content: "hello".into(),
                origin: ProcessOrigin::ServiceQuery,
            })
            .unwrap();
        let expected = ticket.sequence();
        let (activity, snapshot) = ticket.resolve().await.unwrap();

        assert_eq!(activity.owner_command_seq(), Some(expected));
        assert_eq!(snapshot.origin(), SnapshotOrigin::AfterCommand(expected));
        assert_eq!(snapshot.memory_count(), 7);

        drop(handle);
        assert_eq!(task.await.unwrap().commands_completed, 1);
    }
}
