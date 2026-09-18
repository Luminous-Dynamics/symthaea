// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed contracts for migrating the service daemon onto the single-owner runtime.
//!
//! These types deliberately separate three concerns that are currently mixed behind
//! `Mutex<Symthaea>` in the daemon:
//!
//! 1. commands that are allowed to mutate cognition;
//! 2. runtime activity used to keep interfaces responsive during long commands;
//! 3. immutable completed cognitive state used by status/introspection clients.
//!
//! This crate does not implement the concrete `Symthaea` command handler. It is a
//! dependency-light contract so the daemon can migrate in small tranches without
//! moving its wire `Request`/`Response` enums or duplicating cognitive ownership.

use std::path::PathBuf;

use symthaea_runtime_owner::OwnerCommandSeq;

/// Why text is entering the cognitive processing path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessOrigin {
    ServiceQuery,
    VoiceTurn,
    VoiceTranscription,
}

/// Background mutation source. A periodic tick is never permission to bypass the
/// bounded owner mailbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundCycleKind {
    Periodic,
    PreSleep,
    Maintenance,
}

/// Stable command category suitable for runtime-activity presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceCommandKind {
    ProcessText,
    Sleep,
    Save,
    ShutdownPersist,
    BackgroundCycle,
}

/// Mutating operations that must execute through the sole runtime owner after the
/// service hands the initialized `Symthaea` facade into that owner.
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

/// Fast control-plane activity. This is intentionally separate from cognitive
/// state: `Processing` means a command is in flight, not that any newer Phi,
/// coherence, consciousness, or memory observation exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeActivity {
    #[default]
    Idle,
    Processing {
        owner_command_seq: OwnerCommandSeq,
        kind: ServiceCommandKind,
    },
    ShuttingDown,
}

impl RuntimeActivity {
    pub fn processing(owner_command_seq: OwnerCommandSeq, kind: ServiceCommandKind) -> Self {
        Self::Processing {
            owner_command_seq,
            kind,
        }
    }

    pub fn is_busy(self) -> bool {
        !matches!(self, Self::Idle)
    }
}

/// Values obtained directly from the facade introspection result after a completed
/// mutation boundary. Derived presentation thresholds such as `is_conscious =
/// consciousness_level > 0.5` intentionally do not live here.
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

/// Direct partnership/relational values that existing query/UI paths may need.
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

/// Immutable cognitive state published after a completed owner command.
///
/// `captured_after` correlates this state with owner control flow only. It is not a
/// semantic runtime cursor and must never substitute for `RuntimeCursor/EventSeq`.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceRuntimeSnapshot {
    pub captured_after: Option<OwnerCommandSeq>,
    pub cognition: CognitiveSummary,
    pub partnership: PartnershipSummary,
}

impl ServiceRuntimeSnapshot {
    pub fn memory_count(&self) -> usize {
        self.cognition.total_memories()
    }
}

/// Service-local counters intentionally kept outside the canonical cognitive
/// snapshot. Uptime/request counts/sleep counts describe daemon operation, not the
/// mutable cognitive state itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServiceCounters {
    pub requests_processed: u64,
    pub sleep_cycles: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(value: u64) -> OwnerCommandSeq {
        // OwnerCommandSeq is produced by the owner at runtime. Tests construct it
        // through an accepted command in that crate rather than exposing a public
        // arbitrary constructor here; this helper is intentionally unused until
        // integration tests bind both crates.
        let _ = value;
        panic!("OwnerCommandSeq construction belongs to symthaea-runtime-owner")
    }

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
    fn voice_and_service_text_share_one_mutation_class_but_keep_origin() {
        let query = ServiceMutationCommand::ProcessText {
            content: "same path".into(),
            origin: ProcessOrigin::ServiceQuery,
        };
        let voice = ServiceMutationCommand::ProcessText {
            content: "same path".into(),
            origin: ProcessOrigin::VoiceTurn,
        };

        assert_eq!(query.kind(), ServiceCommandKind::ProcessText);
        assert_eq!(voice.kind(), ServiceCommandKind::ProcessText);
        assert_eq!(query.process_origin(), Some(ProcessOrigin::ServiceQuery));
        assert_eq!(voice.process_origin(), Some(ProcessOrigin::VoiceTurn));
    }

    #[test]
    fn canonical_cognitive_summary_does_not_need_daemon_counters_or_threshold_labels() {
        let summary = CognitiveSummary {
            consciousness_level: 0.73,
            self_loops: 4,
            graph_size: 12,
            complexity: 2.5,
            short_term_memories: 10,
            long_term_memories: 7,
        };

        assert_eq!(summary.total_memories(), 17);
    }

    #[test]
    fn daemon_counters_remain_separate_from_cognitive_state() {
        let counters = ServiceCounters {
            requests_processed: 99,
            sleep_cycles: 3,
        };
        assert_eq!(counters.requests_processed, 99);
        assert_eq!(counters.sleep_cycles, 3);
    }

    #[test]
    fn idle_is_the_default_activity() {
        assert_eq!(RuntimeActivity::default(), RuntimeActivity::Idle);
        assert!(!RuntimeActivity::Idle.is_busy());
        assert!(RuntimeActivity::ShuttingDown.is_busy());
    }

    // Keep the helper type-checked so a future public arbitrary constructor cannot
    // accidentally become part of this service contract without a deliberate edit.
    #[allow(dead_code)]
    fn owner_sequence_constructor_stays_private_to_owner_crate(value: u64) -> OwnerCommandSeq {
        seq(value)
    }
}
