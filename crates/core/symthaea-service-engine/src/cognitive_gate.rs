// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measurement-only cognition input for conservative shell/action policy.
//!
//! Cognitive telemetry may inhibit or request stronger review, but it does not
//! create command capability or authority. This adapter therefore exposes only a
//! measured consciousness value from an explicitly idle runtime. If cognition is
//! processing or shutting down, no stale value is returned for action gating.

use symthaea_interface_runtime::StateRevision;
use symthaea_service_read_model::{ActivitySnapshotRelation, IntrospectionRead};
use symthaea_service_runtime::SnapshotOrigin;

use crate::protocol_error::ServiceProtocolFailure;

/// Measured cognitive input that is safe to consult as an additional conservative
/// policy signal. Possessing this value grants no execution authority.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredCognitiveGate {
    pub consciousness_level: f32,
    pub snapshot_origin: SnapshotOrigin,
    pub activity_revision: StateRevision,
    pub snapshot_revision: StateRevision,
}

impl MeasuredCognitiveGate {
    /// Construct a gate observation only from an explicitly idle runtime.
    ///
    /// A processing/shutdown relation may carry a perfectly valid *older* cognitive
    /// snapshot for UI display, but that stale snapshot must not silently authorize
    /// an action gate merely because its previous value happened to be high.
    pub fn try_from_read(read: IntrospectionRead) -> Result<Self, ServiceProtocolFailure> {
        let snapshot_origin = match read.relation {
            ActivitySnapshotRelation::Idle { snapshot_origin } => snapshot_origin,
            ActivitySnapshotRelation::Processing { .. }
            | ActivitySnapshotRelation::ShuttingDown { .. } => {
                return Err(ServiceProtocolFailure::cognitive_gate_not_idle());
            }
        };

        Ok(Self {
            consciousness_level: read.consciousness_level,
            snapshot_origin,
            activity_revision: read.activity_revision,
            snapshot_revision: read.snapshot_revision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_runtime_owner::{
        HandlerFuture, OwnerCommandContext, RuntimeCommandHandler, spawn_runtime_owner,
    };

    fn revision(value: u64) -> StateRevision {
        StateRevision::new(value).expect("test revision is non-zero")
    }

    fn idle_read() -> IntrospectionRead {
        IntrospectionRead {
            relation: ActivitySnapshotRelation::Idle {
                snapshot_origin: SnapshotOrigin::Initialized,
            },
            activity_revision: revision(2),
            snapshot_revision: revision(2),
            consciousness_level: 0.72,
            self_loops: 2,
            graph_size: 8,
            complexity: 1.5,
            short_term_memories: 3,
            long_term_memories: 4,
        }
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
    fn idle_measurement_is_available_without_granting_authority() {
        let gate = MeasuredCognitiveGate::try_from_read(idle_read()).unwrap();
        assert_eq!(gate.consciousness_level, 0.72);
        assert_eq!(gate.snapshot_origin, SnapshotOrigin::Initialized);
        assert_eq!(gate.activity_revision.get(), 2);
        assert_eq!(gate.snapshot_revision.get(), 2);
    }

    #[tokio::test]
    async fn processing_relation_fails_closed_instead_of_using_stale_high_value() {
        let (owner, task) = spawn_runtime_owner((), Noop, 1).unwrap();
        let ticket = owner.try_submit(()).unwrap();
        let active = ticket.sequence();

        let stale_high_read = IntrospectionRead {
            relation: ActivitySnapshotRelation::Processing {
                active_command: active,
                snapshot_origin: SnapshotOrigin::Initialized,
            },
            activity_revision: revision(4),
            snapshot_revision: revision(2),
            consciousness_level: 0.99,
            self_loops: 5,
            graph_size: 12,
            complexity: 2.0,
            short_term_memories: 7,
            long_term_memories: 9,
        };

        let failure = MeasuredCognitiveGate::try_from_read(stale_high_read).unwrap_err();
        assert_eq!(failure.code, "cognitive_gate_not_idle");
        assert!(failure.retryable);

        ticket.resolve().await.unwrap();
        drop(owner);
        assert_eq!(task.await.unwrap().commands_completed, 1);
    }
}
