// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only service views composed from runtime activity plus the last completed
//! cognitive snapshot.
//!
//! The two planes are intentionally independent. A read may therefore report an
//! in-flight owner command while preserving an older completed cognitive snapshot.
//! This module keeps that relation explicit instead of manufacturing freshness,
//! consciousness thresholds, or awakening labels.

use std::fmt;

use symthaea_interface_runtime::{StatePlaneError, StateRevision};
use symthaea_runtime_owner::OwnerCommandSeq;
use symthaea_service_runtime::state_plane::ServiceStatePlaneSubscription;
use symthaea_service_runtime::{
    CognitiveSummary, PartnershipSummary, RuntimeActivity, SnapshotOrigin,
};

/// Descriptive relation between fast runtime activity and the last completed
/// cognitive snapshot. It does not assert semantic event continuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivitySnapshotRelation {
    Idle {
        snapshot_origin: SnapshotOrigin,
    },
    Processing {
        active_command: OwnerCommandSeq,
        snapshot_origin: SnapshotOrigin,
    },
    ShuttingDown {
        active_command: OwnerCommandSeq,
        snapshot_origin: SnapshotOrigin,
    },
}

/// One read-only projection of the two latest-wins service state planes.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceReadModel {
    pub activity: RuntimeActivity,
    pub activity_revision: StateRevision,
    pub snapshot_revision: StateRevision,
    pub snapshot_origin: SnapshotOrigin,
    pub cognition: CognitiveSummary,
    pub partnership: PartnershipSummary,
}

impl ServiceReadModel {
    /// Peek both latest-wins slots without advancing the subscription cursors.
    pub fn peek(
        subscription: &ServiceStatePlaneSubscription,
    ) -> Result<Self, ServiceReadModelError> {
        let activity = subscription
            .peek_activity()
            .map_err(ServiceReadModelError::StatePlane)?
            .ok_or(ServiceReadModelError::MissingActivity)?;
        let snapshot = subscription
            .peek_snapshot()
            .map_err(ServiceReadModelError::StatePlane)?
            .ok_or(ServiceReadModelError::MissingSnapshot)?;

        Ok(Self {
            activity: *activity.value,
            activity_revision: activity.revision,
            snapshot_revision: snapshot.revision,
            snapshot_origin: snapshot.value.origin(),
            cognition: snapshot.value.cognition().clone(),
            partnership: snapshot.value.partnership().clone(),
        })
    }

    pub fn relation(&self) -> ActivitySnapshotRelation {
        match self.activity {
            RuntimeActivity::Idle => ActivitySnapshotRelation::Idle {
                snapshot_origin: self.snapshot_origin,
            },
            RuntimeActivity::Processing(activity) => ActivitySnapshotRelation::Processing {
                active_command: activity.owner_command_seq(),
                snapshot_origin: self.snapshot_origin,
            },
            RuntimeActivity::ShuttingDown(activity) => ActivitySnapshotRelation::ShuttingDown {
                active_command: activity.owner_command_seq(),
                snapshot_origin: self.snapshot_origin,
            },
        }
    }

    pub fn status(&self) -> CognitiveStatusRead {
        CognitiveStatusRead {
            relation: self.relation(),
            activity_revision: self.activity_revision,
            snapshot_revision: self.snapshot_revision,
            consciousness_level: self.cognition.consciousness_level,
            memory_count: self.cognition.total_memories(),
        }
    }

    pub fn introspection(&self) -> IntrospectionRead {
        IntrospectionRead {
            relation: self.relation(),
            activity_revision: self.activity_revision,
            snapshot_revision: self.snapshot_revision,
            consciousness_level: self.cognition.consciousness_level,
            self_loops: self.cognition.self_loops,
            graph_size: self.cognition.graph_size,
            complexity: self.cognition.complexity,
            short_term_memories: self.cognition.short_term_memories,
            long_term_memories: self.cognition.long_term_memories,
        }
    }

    pub fn partnership(&self) -> PartnershipRead {
        PartnershipRead {
            relation: self.relation(),
            activity_revision: self.activity_revision,
            snapshot_revision: self.snapshot_revision,
            stage: self.partnership.stage.clone(),
            trust: self.partnership.trust,
            vulnerability: self.partnership.vulnerability,
            reciprocity: self.partnership.reciprocity,
            phi_dyad: self.partnership.phi_dyad,
            interactions: self.partnership.interactions,
            trajectory_points: self.partnership.trajectory_points,
        }
    }
}

/// Canonical cognitive portion of service status. Daemon-local uptime/request/sleep
/// counters remain separate operational data and can be composed at the wire edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CognitiveStatusRead {
    pub relation: ActivitySnapshotRelation,
    pub activity_revision: StateRevision,
    pub snapshot_revision: StateRevision,
    pub consciousness_level: f32,
    pub memory_count: usize,
}

/// Measured introspection fields only. Deliberately excludes threshold-derived
/// `is_conscious`, `meta_awareness`, `phenomenal_state`, and pseudo-Phi labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntrospectionRead {
    pub relation: ActivitySnapshotRelation,
    pub activity_revision: StateRevision,
    pub snapshot_revision: StateRevision,
    pub consciousness_level: f32,
    pub self_loops: usize,
    pub graph_size: usize,
    pub complexity: f32,
    pub short_term_memories: usize,
    pub long_term_memories: usize,
}

/// Direct relational fields copied from the last completed facade snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct PartnershipRead {
    pub relation: ActivitySnapshotRelation,
    pub activity_revision: StateRevision,
    pub snapshot_revision: StateRevision,
    pub stage: String,
    pub trust: f32,
    pub vulnerability: f32,
    pub reciprocity: f32,
    pub phi_dyad: f64,
    pub interactions: u64,
    pub trajectory_points: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceReadModelError {
    StatePlane(StatePlaneError),
    MissingActivity,
    MissingSnapshot,
}

impl fmt::Display for ServiceReadModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StatePlane(error) => write!(f, "service state-plane read failed: {error}"),
            Self::MissingActivity => write!(f, "service activity state is not initialized"),
            Self::MissingSnapshot => write!(f, "service cognitive snapshot is not initialized"),
        }
    }
}

impl std::error::Error for ServiceReadModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StatePlane(error) => Some(error),
            Self::MissingActivity | Self::MissingSnapshot => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_service_runtime::state_plane::service_state_planes;
    use symthaea_service_runtime::{CognitiveSummary, PartnershipSummary, ServiceRuntimeSnapshot};

    fn initialized_snapshot() -> ServiceRuntimeSnapshot {
        ServiceRuntimeSnapshot::initialized(
            CognitiveSummary {
                consciousness_level: 0.61,
                self_loops: 3,
                graph_size: 9,
                complexity: 1.7,
                short_term_memories: 4,
                long_term_memories: 6,
            },
            PartnershipSummary {
                stage: "Contact".into(),
                trust: 0.4,
                vulnerability: 0.2,
                reciprocity: 0.3,
                phi_dyad: 0.12,
                interactions: 7,
                trajectory_points: 5,
            },
        )
    }

    #[test]
    fn initialized_read_model_is_truthful_and_threshold_free() {
        let (_publisher, hub) = service_state_planes(initialized_snapshot()).unwrap();
        let subscription = hub.subscribe().unwrap();
        let read = ServiceReadModel::peek(&subscription).unwrap();

        assert_eq!(
            read.relation(),
            ActivitySnapshotRelation::Idle {
                snapshot_origin: SnapshotOrigin::Initialized,
            }
        );

        let status = read.status();
        assert_eq!(status.consciousness_level, 0.61);
        assert_eq!(status.memory_count, 10);

        let introspection = read.introspection();
        assert_eq!(introspection.self_loops, 3);
        assert_eq!(introspection.graph_size, 9);
        assert_eq!(introspection.short_term_memories, 4);
        assert_eq!(introspection.long_term_memories, 6);

        let partnership = read.partnership();
        assert_eq!(partnership.stage, "Contact");
        assert_eq!(partnership.interactions, 7);
    }
}
