// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Revision-pinned state coordinates layered over environment-qualified stable subjects.
//!
//! This module deliberately does **not** call a revision number a content snapshot:
//!
//! ```text
//! stable subject identity
//!     != revision-pinned state coordinate
//!     != content-committed snapshot identity
//! ```
//!
//! A revision coordinate detects ordinary state drift within one trusted graph lineage.
//! It does not cryptographically commit the graph/entity/relation contents. A future
//! content-committed snapshot must therefore be a separate type/theorem.

use crate::system_state::{EntityId, RelationId, SystemEntityV1, SystemRelationV1};
use crate::system_state_identity::{
    SystemEntityRefV1, SystemEnvironmentIdV1, SystemRelationRefV1, SystemStateGraphScopeV1,
    SystemStateIdentityErrorV1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Environment-qualified whole-graph revision coordinate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SystemGraphRevisionRefV1 {
    pub environment_id: SystemEnvironmentIdV1,
    pub graph_revision: u64,
}

/// Stable entity subject plus an entity-local revision coordinate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SystemEntityRevisionRefV1 {
    pub subject: SystemEntityRefV1,
    pub entity_revision: u64,
}

/// Stable relation subject plus a relation-local revision coordinate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SystemRelationRevisionRefV1 {
    pub subject: SystemRelationRefV1,
    pub relation_revision: u64,
}

/// Read-only helpers for creating and resolving revision-pinned state coordinates.
///
/// Resolution is against the current projection only. Historical state is not
/// reconstructed. A revision mismatch fails closed rather than silently following
/// the subject to newer state.
pub trait SystemStateRevisionScopeExtV1 {
    fn graph_revision_ref(
        &self,
    ) -> Result<SystemGraphRevisionRefV1, SystemStateRevisionErrorV1>;

    fn entity_revision_ref(
        &self,
        id: &EntityId,
    ) -> Result<SystemEntityRevisionRefV1, SystemStateRevisionErrorV1>;

    fn relation_revision_ref(
        &self,
        id: &RelationId,
    ) -> Result<SystemRelationRevisionRefV1, SystemStateRevisionErrorV1>;

    fn resolve_graph_revision_ref(
        &self,
        reference: &SystemGraphRevisionRefV1,
    ) -> Result<(), SystemStateRevisionErrorV1>;

    fn resolve_entity_revision_ref(
        &self,
        reference: &SystemEntityRevisionRefV1,
    ) -> Result<&SystemEntityV1, SystemStateRevisionErrorV1>;

    fn resolve_relation_revision_ref(
        &self,
        reference: &SystemRelationRevisionRefV1,
    ) -> Result<&SystemRelationV1, SystemStateRevisionErrorV1>;
}

impl SystemStateRevisionScopeExtV1 for SystemStateGraphScopeV1<'_> {
    fn graph_revision_ref(
        &self,
    ) -> Result<SystemGraphRevisionRefV1, SystemStateRevisionErrorV1> {
        let revision = self.graph_revision();
        if revision == u64::MAX {
            return Err(SystemStateRevisionErrorV1::GraphRevisionExhausted);
        }
        Ok(SystemGraphRevisionRefV1 {
            environment_id: self.environment_id().clone(),
            graph_revision: revision,
        })
    }

    fn entity_revision_ref(
        &self,
        id: &EntityId,
    ) -> Result<SystemEntityRevisionRefV1, SystemStateRevisionErrorV1> {
        let subject = self.entity_ref(id)?;
        let entity = self.resolve_entity_ref(&subject)?;
        if entity.revision == u64::MAX {
            return Err(SystemStateRevisionErrorV1::EntityRevisionExhausted {
                entity: subject,
            });
        }
        Ok(SystemEntityRevisionRefV1 {
            subject,
            entity_revision: entity.revision,
        })
    }

    fn relation_revision_ref(
        &self,
        id: &RelationId,
    ) -> Result<SystemRelationRevisionRefV1, SystemStateRevisionErrorV1> {
        let subject = self.relation_ref(id)?;
        let relation = self.resolve_relation_ref(&subject)?;
        if relation.revision == u64::MAX {
            return Err(SystemStateRevisionErrorV1::RelationRevisionExhausted {
                relation: subject,
            });
        }
        Ok(SystemRelationRevisionRefV1 {
            subject,
            relation_revision: relation.revision,
        })
    }

    fn resolve_graph_revision_ref(
        &self,
        reference: &SystemGraphRevisionRefV1,
    ) -> Result<(), SystemStateRevisionErrorV1> {
        if self.environment_id() != &reference.environment_id {
            return Err(SystemStateRevisionErrorV1::Identity(
                SystemStateIdentityErrorV1::EnvironmentMismatch {
                    expected: self.environment_id().clone(),
                    actual: reference.environment_id.clone(),
                },
            ));
        }
        if reference.graph_revision == u64::MAX || self.graph_revision() == u64::MAX {
            return Err(SystemStateRevisionErrorV1::GraphRevisionExhausted);
        }
        if self.graph_revision() != reference.graph_revision {
            return Err(SystemStateRevisionErrorV1::GraphRevisionMismatch {
                expected: reference.graph_revision,
                actual: self.graph_revision(),
            });
        }
        Ok(())
    }

    fn resolve_entity_revision_ref(
        &self,
        reference: &SystemEntityRevisionRefV1,
    ) -> Result<&SystemEntityV1, SystemStateRevisionErrorV1> {
        let entity = self.resolve_entity_ref(&reference.subject)?;
        if reference.entity_revision == u64::MAX || entity.revision == u64::MAX {
            return Err(SystemStateRevisionErrorV1::EntityRevisionExhausted {
                entity: reference.subject.clone(),
            });
        }
        if entity.revision != reference.entity_revision {
            return Err(SystemStateRevisionErrorV1::EntityRevisionMismatch {
                entity: reference.subject.clone(),
                expected: reference.entity_revision,
                actual: entity.revision,
            });
        }
        Ok(entity)
    }

    fn resolve_relation_revision_ref(
        &self,
        reference: &SystemRelationRevisionRefV1,
    ) -> Result<&SystemRelationV1, SystemStateRevisionErrorV1> {
        let relation = self.resolve_relation_ref(&reference.subject)?;
        if reference.relation_revision == u64::MAX || relation.revision == u64::MAX {
            return Err(SystemStateRevisionErrorV1::RelationRevisionExhausted {
                relation: reference.subject.clone(),
            });
        }
        if relation.revision != reference.relation_revision {
            return Err(SystemStateRevisionErrorV1::RelationRevisionMismatch {
                relation: reference.subject.clone(),
                expected: reference.relation_revision,
                actual: relation.revision,
            });
        }
        Ok(relation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemStateRevisionErrorV1 {
    Identity(SystemStateIdentityErrorV1),
    GraphRevisionExhausted,
    EntityRevisionExhausted {
        entity: SystemEntityRefV1,
    },
    RelationRevisionExhausted {
        relation: SystemRelationRefV1,
    },
    GraphRevisionMismatch {
        expected: u64,
        actual: u64,
    },
    EntityRevisionMismatch {
        entity: SystemEntityRefV1,
        expected: u64,
        actual: u64,
    },
    RelationRevisionMismatch {
        relation: SystemRelationRefV1,
        expected: u64,
        actual: u64,
    },
}

impl From<SystemStateIdentityErrorV1> for SystemStateRevisionErrorV1 {
    fn from(value: SystemStateIdentityErrorV1) -> Self {
        Self::Identity(value)
    }
}

impl fmt::Display for SystemStateRevisionErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(error) => write!(f, "{error}"),
            Self::GraphRevisionExhausted => write!(
                f,
                "graph revision is exhausted; revision equality can no longer prove unchanged state"
            ),
            Self::EntityRevisionExhausted { entity } => write!(
                f,
                "entity {:?} revision is exhausted; revision equality can no longer prove unchanged state",
                entity
            ),
            Self::RelationRevisionExhausted { relation } => write!(
                f,
                "relation {:?} revision is exhausted; revision equality can no longer prove unchanged state",
                relation
            ),
            Self::GraphRevisionMismatch { expected, actual } => write!(
                f,
                "graph revision mismatch: expected {expected}, got {actual}"
            ),
            Self::EntityRevisionMismatch {
                entity,
                expected,
                actual,
            } => write!(
                f,
                "entity {:?} revision mismatch: expected {expected}, got {actual}",
                entity
            ),
            Self::RelationRevisionMismatch {
                relation,
                expected,
                actual,
            } => write!(
                f,
                "relation {:?} revision mismatch: expected {expected}, got {actual}",
                relation
            ),
        }
    }
}

impl Error for SystemStateRevisionErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_state::{
        EntityKindV1, ObservationClockV1, ObservationId, ObservationProvenanceV1,
        ObservationSourceKindV1, RelationKindV1, StateValueV1, SystemObservationV1,
        SystemStateGraphV1,
    };
    use std::collections::BTreeMap;

    fn environment() -> SystemEnvironmentIdV1 {
        SystemEnvironmentIdV1::try_new("env-a").unwrap()
    }

    fn observation(id: &str, subject: &str, time: u64) -> SystemObservationV1 {
        SystemObservationV1 {
            id: ObservationId(id.into()),
            subject: EntityId(subject.into()),
            provenance: ObservationProvenanceV1 {
                source_id: format!("source:{subject}"),
                source_kind: ObservationSourceKindV1::SyntheticTest,
                collector: "state-revision-test".into(),
                collector_version: Some("1".into()),
                schema_version: None,
                artifact_digest: None,
            },
            clock: ObservationClockV1 {
                event_time_unix_ms: None,
                observed_at_unix_ms: time,
                ingested_at_unix_ms: Some(time),
                max_age_ms: Some(1_000),
                clock_uncertainty_ms: None,
            },
            confidence: 1.0,
            facts: BTreeMap::new(),
        }
    }

    fn populated_graph_with_status(status: &str) -> SystemStateGraphV1 {
        let mut graph = SystemStateGraphV1::new();
        let obs = observation("obs-1", "host-1", 1_000);
        let obs_id = obs.id.clone();
        graph.record_observation(obs).unwrap();

        let mut host_attributes = BTreeMap::new();
        host_attributes.insert("status".into(), StateValueV1::Text(status.into()));
        graph
            .upsert_entity(
                EntityId("host-1".into()),
                EntityKindV1::Host,
                host_attributes,
                &obs_id,
            )
            .unwrap();
        graph
            .upsert_entity(
                EntityId("dns-1".into()),
                EntityKindV1::Service,
                BTreeMap::new(),
                &obs_id,
            )
            .unwrap();
        graph
            .upsert_relation(
                RelationId("rel-1".into()),
                EntityId("host-1".into()),
                RelationKindV1::DependsOn,
                EntityId("dns-1".into()),
                BTreeMap::new(),
                &obs_id,
            )
            .unwrap();
        graph
    }

    fn populated_graph() -> SystemStateGraphV1 {
        populated_graph_with_status("healthy")
    }

    #[test]
    fn stable_subject_survives_unrelated_graph_drift_but_graph_revision_does_not() {
        let mut graph = populated_graph();
        let (stable, entity_revision, graph_revision) = {
            let scope = SystemStateGraphScopeV1::new(environment(), &graph);
            (
                scope.entity_ref(&EntityId("host-1".into())).unwrap(),
                scope
                    .entity_revision_ref(&EntityId("host-1".into()))
                    .unwrap(),
                scope.graph_revision_ref().unwrap(),
            )
        };

        graph
            .record_observation(observation("obs-2", "other-1", 1_100))
            .unwrap();
        let scope = SystemStateGraphScopeV1::new(environment(), &graph);

        assert!(scope.resolve_entity_ref(&stable).is_ok());
        assert!(scope.resolve_entity_revision_ref(&entity_revision).is_ok());
        assert!(matches!(
            scope.resolve_graph_revision_ref(&graph_revision),
            Err(SystemStateRevisionErrorV1::GraphRevisionMismatch { .. })
        ));
    }

    #[test]
    fn entity_revision_ref_fails_closed_after_subject_revision_changes() {
        let mut graph = populated_graph();
        let revision = {
            let scope = SystemStateGraphScopeV1::new(environment(), &graph);
            scope
                .entity_revision_ref(&EntityId("host-1".into()))
                .unwrap()
        };

        let obs = observation("obs-2", "host-1", 1_100);
        let obs_id = obs.id.clone();
        graph.record_observation(obs).unwrap();
        graph
            .upsert_entity(
                EntityId("host-1".into()),
                EntityKindV1::Host,
                BTreeMap::new(),
                &obs_id,
            )
            .unwrap();

        let scope = SystemStateGraphScopeV1::new(environment(), &graph);
        assert!(matches!(
            scope.resolve_entity_revision_ref(&revision),
            Err(SystemStateRevisionErrorV1::EntityRevisionMismatch { .. })
        ));
    }

    #[test]
    fn relation_revision_ref_fails_closed_after_relation_revision_changes() {
        let mut graph = populated_graph();
        let revision = {
            let scope = SystemStateGraphScopeV1::new(environment(), &graph);
            scope
                .relation_revision_ref(&RelationId("rel-1".into()))
                .unwrap()
        };

        let obs = observation("obs-2", "host-1", 1_100);
        let obs_id = obs.id.clone();
        graph.record_observation(obs).unwrap();
        graph
            .upsert_relation(
                RelationId("rel-1".into()),
                EntityId("host-1".into()),
                RelationKindV1::DependsOn,
                EntityId("dns-1".into()),
                BTreeMap::new(),
                &obs_id,
            )
            .unwrap();

        let scope = SystemStateGraphScopeV1::new(environment(), &graph);
        assert!(matches!(
            scope.resolve_relation_revision_ref(&revision),
            Err(SystemStateRevisionErrorV1::RelationRevisionMismatch { .. })
        ));
    }

    #[test]
    fn graph_revision_exhaustion_fails_closed() {
        let mut graph = populated_graph();
        graph.revision = u64::MAX;
        let scope = SystemStateGraphScopeV1::new(environment(), &graph);

        assert!(matches!(
            scope.graph_revision_ref(),
            Err(SystemStateRevisionErrorV1::GraphRevisionExhausted)
        ));
        assert!(matches!(
            scope.resolve_graph_revision_ref(&SystemGraphRevisionRefV1 {
                environment_id: environment(),
                graph_revision: u64::MAX,
            }),
            Err(SystemStateRevisionErrorV1::GraphRevisionExhausted)
        ));
    }

    #[test]
    fn revision_coordinate_does_not_claim_content_commitment() {
        let graph_a = populated_graph_with_status("healthy");
        let graph_b = populated_graph_with_status("compromised");
        assert_eq!(graph_a.revision, graph_b.revision);

        let scope_a = SystemStateGraphScopeV1::new(environment(), &graph_a);
        let scope_b = SystemStateGraphScopeV1::new(environment(), &graph_b);
        let revision = scope_a
            .entity_revision_ref(&EntityId("host-1".into()))
            .unwrap();

        let a = scope_a.resolve_entity_revision_ref(&revision).unwrap();
        let b = scope_b.resolve_entity_revision_ref(&revision).unwrap();
        assert_ne!(a.attributes, b.attributes);
    }
}
