// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-rich measurement-only read responses for responsive interfaces.
//!
//! Runtime activity and cognitive snapshot freshness are deliberately separate.
//! A client may therefore learn that Symthaea is processing command N while the
//! displayed cognition still truthfully belongs to the last completed command.

use serde::Serialize;
use symthaea_service_read_model::{ActivitySnapshotRelation, IntrospectionRead};
use symthaea_service_runtime::SnapshotOrigin;

/// Machine-readable provenance for one measurement-only read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReadProvenanceWire {
    pub runtime_activity: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_command_seq: Option<u64>,
    pub snapshot_origin: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_command_seq: Option<u64>,
    pub activity_revision: u64,
    pub snapshot_revision: u64,
}

impl ReadProvenanceWire {
    fn from_introspection(read: &IntrospectionRead) -> Self {
        let (runtime_activity, active_command_seq, snapshot_origin) = match read.relation {
            ActivitySnapshotRelation::Idle { snapshot_origin } => {
                ("idle", None, snapshot_origin)
            }
            ActivitySnapshotRelation::Processing {
                active_command,
                snapshot_origin,
            } => ("processing", Some(active_command.get()), snapshot_origin),
            ActivitySnapshotRelation::ShuttingDown {
                active_command,
                snapshot_origin,
            } => ("shutting_down", Some(active_command.get()), snapshot_origin),
        };

        let (snapshot_origin, snapshot_command_seq) = match snapshot_origin {
            SnapshotOrigin::Initialized => ("initialized", None),
            SnapshotOrigin::AfterCommand(sequence) => {
                ("after_command", Some(sequence.get()))
            }
        };

        Self {
            runtime_activity,
            active_command_seq,
            snapshot_origin,
            snapshot_command_seq,
            activity_revision: read.activity_revision.get(),
            snapshot_revision: read.snapshot_revision.get(),
        }
    }
}

/// Versioned introspection response containing only measured cognitive values plus
/// explicit runtime/snapshot provenance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MeasuredIntrospectionV2 {
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub consciousness_level: f32,
    pub self_loops: usize,
    pub graph_size: usize,
    pub complexity: f32,
    pub short_term_memories: usize,
    pub long_term_memories: usize,
    pub epistemic_status: &'static str,
    #[serde(flatten)]
    pub provenance: ReadProvenanceWire,
}

impl From<IntrospectionRead> for MeasuredIntrospectionV2 {
    fn from(read: IntrospectionRead) -> Self {
        let provenance = ReadProvenanceWire::from_introspection(&read);
        Self {
            response_type: "introspection_v2",
            consciousness_level: read.consciousness_level,
            self_loops: read.self_loops,
            graph_size: read.graph_size,
            complexity: read.complexity,
            short_term_memories: read.short_term_memories,
            long_term_memories: read.long_term_memories,
            epistemic_status: "measured_runtime_snapshot",
            provenance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interface_runtime::StateRevision;
    use symthaea_runtime_owner::{
        HandlerFuture, OwnerCommandContext, RuntimeCommandHandler, spawn_runtime_owner,
    };

    fn revision(value: u64) -> StateRevision {
        StateRevision::new(value).expect("test revision is non-zero")
    }

    struct Noop;

    impl RuntimeCommandHandler<(), ()> for Noop {
        type Reply = ();

        fn handle<'a>(
            &'a mut self,
            _engine: &'a mut (),
            _context: OwnerCommandContext,
            _command: (),
        ) -> HandlerFuture<'a, Self::Reply> {
            Box::pin(async {})
        }
    }

    #[test]
    fn initialized_idle_read_has_explicit_freshness_metadata() {
        let wire = MeasuredIntrospectionV2::from(IntrospectionRead {
            relation: ActivitySnapshotRelation::Idle {
                snapshot_origin: SnapshotOrigin::Initialized,
            },
            activity_revision: revision(1),
            snapshot_revision: revision(1),
            consciousness_level: 0.4,
            self_loops: 2,
            graph_size: 8,
            complexity: 1.2,
            short_term_memories: 3,
            long_term_memories: 5,
        });

        let json = serde_json::to_value(wire).unwrap();
        assert_eq!(json["type"], "introspection_v2");
        assert_eq!(json["runtime_activity"], "idle");
        assert_eq!(json["snapshot_origin"], "initialized");
        assert_eq!(json["activity_revision"], 1);
        assert_eq!(json["snapshot_revision"], 1);
        assert!(json.get("active_command_seq").is_none());
        assert!(json.get("snapshot_command_seq").is_none());
    }

    #[tokio::test]
    async fn processing_read_can_point_at_an_older_completed_snapshot() {
        let (owner, task) = spawn_runtime_owner((), Noop, 1).unwrap();
        let ticket = owner.try_submit(()).unwrap();
        let active = ticket.sequence();

        let wire = MeasuredIntrospectionV2::from(IntrospectionRead {
            relation: ActivitySnapshotRelation::Processing {
                active_command: active,
                snapshot_origin: SnapshotOrigin::Initialized,
            },
            activity_revision: revision(4),
            snapshot_revision: revision(2),
            consciousness_level: 0.7,
            self_loops: 4,
            graph_size: 11,
            complexity: 2.0,
            short_term_memories: 6,
            long_term_memories: 9,
        });

        let json = serde_json::to_value(wire).unwrap();
        assert_eq!(json["runtime_activity"], "processing");
        assert_eq!(json["active_command_seq"], active.get());
        assert_eq!(json["snapshot_origin"], "initialized");
        assert_eq!(json["activity_revision"], 4);
        assert_eq!(json["snapshot_revision"], 2);

        ticket.resolve().await.unwrap();
        drop(owner);
        assert_eq!(task.await.unwrap().commands_completed, 1);
    }
}
