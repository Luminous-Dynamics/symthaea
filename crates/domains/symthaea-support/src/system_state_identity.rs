// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Environment-qualified durable references for `SystemStateGraphV1`.
//!
//! `EntityId`, `RelationId`, and `ObservationId` are graph-local identities. This
//! module provides the cross-crate persistence boundary that prevents those local
//! identifiers from being rebound across independently modeled environments.
//!
//! Core theorem:
//!
//! ```text
//! parsed graph != validated durable graph scope
//! local graph ID + graph revision != durable global subject identity
//! map key != embedded record identity unless explicitly checked
//! ```
//!
//! The environment namespace is ordinary provenance/identity only. It grants no
//! authority and does not require a global registry.

use crate::system_state::{
    EntityId, ObservationId, RelationId, SystemEntityV1, SystemObservationV1, SystemRelationV1,
    SystemStateGraphV1,
};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use std::error::Error;
use std::fmt;

/// Stable namespace for one modeled environment.
///
/// Callers should preserve this identity across replay/cloning of the same modeled
/// environment and issue a distinct identity for a genuinely distinct environment,
/// even when the initial graph contents happen to be byte-identical.
///
/// Deserialization goes through the same validation path as ordinary construction;
/// parsing persisted data cannot bypass the non-empty/canonical identity contract.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SystemEnvironmentIdV1(String);

impl SystemEnvironmentIdV1 {
    pub fn try_new(value: impl Into<String>) -> Result<Self, SystemStateIdentityErrorV1> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(SystemStateIdentityErrorV1::EmptyEnvironmentId);
        }
        if value.trim() != value {
            return Err(SystemStateIdentityErrorV1::NonCanonicalEnvironmentId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SystemEnvironmentIdV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_new(value).map_err(D::Error::custom)
    }
}

/// Durable reference to one graph-local entity.
///
/// Fields are private so callers cannot mint an unchecked durable coordinate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SystemEntityRefV1 {
    environment_id: SystemEnvironmentIdV1,
    entity_id: EntityId,
}

impl SystemEntityRefV1 {
    pub fn environment_id(&self) -> &SystemEnvironmentIdV1 {
        &self.environment_id
    }

    pub fn entity_id(&self) -> &EntityId {
        &self.entity_id
    }
}

impl<'de> Deserialize<'de> for SystemEntityRefV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            environment_id: SystemEnvironmentIdV1,
            entity_id: EntityId,
        }

        let raw = Raw::deserialize(deserializer)?;
        validate_local_id(&raw.entity_id.0, "entity").map_err(D::Error::custom)?;
        Ok(Self {
            environment_id: raw.environment_id,
            entity_id: raw.entity_id,
        })
    }
}

/// Durable reference to one graph-local relation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SystemRelationRefV1 {
    environment_id: SystemEnvironmentIdV1,
    relation_id: RelationId,
}

impl SystemRelationRefV1 {
    pub fn environment_id(&self) -> &SystemEnvironmentIdV1 {
        &self.environment_id
    }

    pub fn relation_id(&self) -> &RelationId {
        &self.relation_id
    }
}

impl<'de> Deserialize<'de> for SystemRelationRefV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            environment_id: SystemEnvironmentIdV1,
            relation_id: RelationId,
        }

        let raw = Raw::deserialize(deserializer)?;
        validate_local_id(&raw.relation_id.0, "relation").map_err(D::Error::custom)?;
        Ok(Self {
            environment_id: raw.environment_id,
            relation_id: raw.relation_id,
        })
    }
}

/// Durable reference to one graph-local observation/evidence item.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SystemObservationRefV1 {
    environment_id: SystemEnvironmentIdV1,
    observation_id: ObservationId,
}

impl SystemObservationRefV1 {
    pub fn environment_id(&self) -> &SystemEnvironmentIdV1 {
        &self.environment_id
    }

    pub fn observation_id(&self) -> &ObservationId {
        &self.observation_id
    }
}

impl<'de> Deserialize<'de> for SystemObservationRefV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            environment_id: SystemEnvironmentIdV1,
            observation_id: ObservationId,
        }

        let raw = Raw::deserialize(deserializer)?;
        validate_local_id(&raw.observation_id.0, "observation").map_err(D::Error::custom)?;
        Ok(Self {
            environment_id: raw.environment_id,
            observation_id: raw.observation_id,
        })
    }
}

/// Read-only, validated environment binding for durable cross-crate references.
///
/// The underlying graph remains unchanged: local graph operations continue using
/// their existing IDs. Durable scoping is stricter than ordinary local use because
/// a deserialized graph must first pass both #1102's referential validation and
/// index-identity validation (`map key == embedded record ID`).
pub struct SystemStateGraphScopeV1<'a> {
    environment_id: SystemEnvironmentIdV1,
    graph: &'a SystemStateGraphV1,
}

impl<'a> SystemStateGraphScopeV1<'a> {
    /// Construct the durable scope only after validating the persisted graph.
    ///
    /// There is intentionally no public unchecked constructor.
    pub fn try_new(
        environment_id: SystemEnvironmentIdV1,
        graph: &'a SystemStateGraphV1,
    ) -> Result<Self, SystemStateIdentityErrorV1> {
        validate_graph_for_durable_scope(graph)?;
        Ok(Self {
            environment_id,
            graph,
        })
    }

    /// Test-only shorthand. Production builds expose only `try_new`.
    #[cfg(test)]
    pub(crate) fn new(environment_id: SystemEnvironmentIdV1, graph: &'a SystemStateGraphV1) -> Self {
        Self::try_new(environment_id, graph).expect("test graph must be durable-scope valid")
    }

    pub fn environment_id(&self) -> &SystemEnvironmentIdV1 {
        &self.environment_id
    }

    pub fn graph_revision(&self) -> u64 {
        self.graph.revision
    }

    pub fn entity_ref(
        &self,
        id: &EntityId,
    ) -> Result<SystemEntityRefV1, SystemStateIdentityErrorV1> {
        validate_local_id(&id.0, "entity")?;
        if self.graph.entity(id).is_none() {
            return Err(SystemStateIdentityErrorV1::UnknownEntity(id.clone()));
        }
        Ok(SystemEntityRefV1 {
            environment_id: self.environment_id.clone(),
            entity_id: id.clone(),
        })
    }

    pub fn relation_ref(
        &self,
        id: &RelationId,
    ) -> Result<SystemRelationRefV1, SystemStateIdentityErrorV1> {
        validate_local_id(&id.0, "relation")?;
        if self.graph.relation(id).is_none() {
            return Err(SystemStateIdentityErrorV1::UnknownRelation(id.clone()));
        }
        Ok(SystemRelationRefV1 {
            environment_id: self.environment_id.clone(),
            relation_id: id.clone(),
        })
    }

    pub fn observation_ref(
        &self,
        id: &ObservationId,
    ) -> Result<SystemObservationRefV1, SystemStateIdentityErrorV1> {
        validate_local_id(&id.0, "observation")?;
        if self.graph.observation(id).is_none() {
            return Err(SystemStateIdentityErrorV1::UnknownObservation(id.clone()));
        }
        Ok(SystemObservationRefV1 {
            environment_id: self.environment_id.clone(),
            observation_id: id.clone(),
        })
    }

    pub fn resolve_entity_ref(
        &self,
        reference: &SystemEntityRefV1,
    ) -> Result<&'a SystemEntityV1, SystemStateIdentityErrorV1> {
        self.require_environment(reference.environment_id())?;
        validate_local_id(&reference.entity_id().0, "entity")?;
        self.graph
            .entity(reference.entity_id())
            .ok_or_else(|| SystemStateIdentityErrorV1::UnknownEntity(reference.entity_id().clone()))
    }

    pub fn resolve_relation_ref(
        &self,
        reference: &SystemRelationRefV1,
    ) -> Result<&'a SystemRelationV1, SystemStateIdentityErrorV1> {
        self.require_environment(reference.environment_id())?;
        validate_local_id(&reference.relation_id().0, "relation")?;
        self.graph.relation(reference.relation_id()).ok_or_else(|| {
            SystemStateIdentityErrorV1::UnknownRelation(reference.relation_id().clone())
        })
    }

    pub fn resolve_observation_ref(
        &self,
        reference: &SystemObservationRefV1,
    ) -> Result<&'a SystemObservationV1, SystemStateIdentityErrorV1> {
        self.require_environment(reference.environment_id())?;
        validate_local_id(&reference.observation_id().0, "observation")?;
        self.graph
            .observation(reference.observation_id())
            .ok_or_else(|| {
                SystemStateIdentityErrorV1::UnknownObservation(reference.observation_id().clone())
            })
    }

    fn require_environment(
        &self,
        environment_id: &SystemEnvironmentIdV1,
    ) -> Result<(), SystemStateIdentityErrorV1> {
        if &self.environment_id == environment_id {
            Ok(())
        } else {
            Err(SystemStateIdentityErrorV1::EnvironmentMismatch {
                expected: self.environment_id.clone(),
                actual: environment_id.clone(),
            })
        }
    }
}

fn validate_graph_for_durable_scope(
    graph: &SystemStateGraphV1,
) -> Result<(), SystemStateIdentityErrorV1> {
    graph
        .validate()
        .map_err(|error| SystemStateIdentityErrorV1::InvalidGraph(error.to_string()))?;

    for entity in graph.entities() {
        validate_local_id(&entity.id.0, "entity")?;
        match graph.entity(&entity.id) {
            Some(indexed) if std::ptr::eq(indexed, entity) => {}
            _ => {
                return Err(SystemStateIdentityErrorV1::EntityIndexIdentityMismatch(
                    entity.id.clone(),
                ));
            }
        }
    }

    for relation in graph.relations() {
        validate_local_id(&relation.id.0, "relation")?;
        match graph.relation(&relation.id) {
            Some(indexed) if std::ptr::eq(indexed, relation) => {}
            _ => {
                return Err(SystemStateIdentityErrorV1::RelationIndexIdentityMismatch(
                    relation.id.clone(),
                ));
            }
        }
    }

    for observation in graph.observations() {
        validate_local_id(&observation.id.0, "observation")?;
        match graph.observation(&observation.id) {
            Some(indexed) if std::ptr::eq(indexed, observation) => {}
            _ => {
                return Err(SystemStateIdentityErrorV1::ObservationIndexIdentityMismatch(
                    observation.id.clone(),
                ));
            }
        }
    }

    Ok(())
}

fn validate_local_id(value: &str, kind: &'static str) -> Result<(), SystemStateIdentityErrorV1> {
    if value.trim().is_empty() {
        return Err(SystemStateIdentityErrorV1::EmptyLocalIdentifier(kind));
    }
    if value.trim() != value {
        return Err(SystemStateIdentityErrorV1::NonCanonicalLocalIdentifier(kind));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemStateIdentityErrorV1 {
    EmptyEnvironmentId,
    NonCanonicalEnvironmentId,
    EmptyLocalIdentifier(&'static str),
    NonCanonicalLocalIdentifier(&'static str),
    InvalidGraph(String),
    EntityIndexIdentityMismatch(EntityId),
    RelationIndexIdentityMismatch(RelationId),
    ObservationIndexIdentityMismatch(ObservationId),
    UnknownEntity(EntityId),
    UnknownRelation(RelationId),
    UnknownObservation(ObservationId),
    EnvironmentMismatch {
        expected: SystemEnvironmentIdV1,
        actual: SystemEnvironmentIdV1,
    },
}

impl fmt::Display for SystemStateIdentityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEnvironmentId => write!(f, "environment identifier must not be empty"),
            Self::NonCanonicalEnvironmentId => {
                write!(f, "environment identifier contains surrounding whitespace")
            }
            Self::EmptyLocalIdentifier(kind) => {
                write!(f, "{kind} identifier must not be empty at the durable boundary")
            }
            Self::NonCanonicalLocalIdentifier(kind) => write!(
                f,
                "{kind} identifier contains surrounding whitespace at the durable boundary"
            ),
            Self::InvalidGraph(message) => {
                write!(f, "system-state graph is not valid for durable scoping: {message}")
            }
            Self::EntityIndexIdentityMismatch(id) => write!(
                f,
                "entity record {:?} is not stored under its embedded entity identity",
                id
            ),
            Self::RelationIndexIdentityMismatch(id) => write!(
                f,
                "relation record {:?} is not stored under its embedded relation identity",
                id
            ),
            Self::ObservationIndexIdentityMismatch(id) => write!(
                f,
                "observation record {:?} is not stored under its embedded observation identity",
                id
            ),
            Self::UnknownEntity(id) => write!(f, "unknown entity {:?}", id),
            Self::UnknownRelation(id) => write!(f, "unknown relation {:?}", id),
            Self::UnknownObservation(id) => write!(f, "unknown observation {:?}", id),
            Self::EnvironmentMismatch { expected, actual } => write!(
                f,
                "environment mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
        }
    }
}

impl Error for SystemStateIdentityErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_state::{
        EntityKindV1, ObservationClockV1, ObservationProvenanceV1, ObservationSourceKindV1,
        RelationKindV1, SystemObservationV1,
    };
    use std::collections::BTreeMap;

    fn environment(id: &str) -> SystemEnvironmentIdV1 {
        SystemEnvironmentIdV1::try_new(id).unwrap()
    }

    fn observation(id: &str, subject: &str) -> SystemObservationV1 {
        SystemObservationV1 {
            id: ObservationId(id.into()),
            subject: EntityId(subject.into()),
            provenance: ObservationProvenanceV1 {
                source_id: format!("source:{subject}"),
                source_kind: ObservationSourceKindV1::SyntheticTest,
                collector: "state-identity-test".into(),
                collector_version: Some("1".into()),
                schema_version: None,
                artifact_digest: None,
            },
            clock: ObservationClockV1 {
                event_time_unix_ms: None,
                observed_at_unix_ms: 1_000,
                ingested_at_unix_ms: Some(1_000),
                max_age_ms: Some(1_000),
                clock_uncertainty_ms: None,
            },
            confidence: 1.0,
            facts: BTreeMap::new(),
        }
    }

    fn observation_only_graph() -> SystemStateGraphV1 {
        let mut graph = SystemStateGraphV1::new();
        graph
            .record_observation(observation("obs-1", "host-1"))
            .unwrap();
        graph
    }

    fn entity_only_graph() -> SystemStateGraphV1 {
        let mut graph = observation_only_graph();
        graph
            .upsert_entity(
                EntityId("host-1".into()),
                EntityKindV1::Host,
                BTreeMap::new(),
                &ObservationId("obs-1".into()),
            )
            .unwrap();
        graph
    }

    fn populated_graph() -> SystemStateGraphV1 {
        let mut graph = entity_only_graph();
        graph
            .upsert_entity(
                EntityId("dns-1".into()),
                EntityKindV1::Service,
                BTreeMap::new(),
                &ObservationId("obs-1".into()),
            )
            .unwrap();
        graph
            .upsert_relation(
                RelationId("rel-1".into()),
                EntityId("host-1".into()),
                RelationKindV1::DependsOn,
                EntityId("dns-1".into()),
                BTreeMap::new(),
                &ObservationId("obs-1".into()),
            )
            .unwrap();
        graph
    }

    fn rewrite_map_key(
        graph: &SystemStateGraphV1,
        field: &str,
        old_key: &str,
        new_key: &str,
    ) -> SystemStateGraphV1 {
        let mut value = serde_json::to_value(graph).unwrap();
        let map = value
            .get_mut(field)
            .and_then(serde_json::Value::as_object_mut)
            .unwrap();
        let record = map.remove(old_key).unwrap();
        map.insert(new_key.into(), record);
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn same_local_ids_in_distinct_environments_produce_distinct_durable_refs() {
        let graph_a = populated_graph();
        let graph_b = populated_graph();
        let scope_a = SystemStateGraphScopeV1::try_new(environment("env-a"), &graph_a).unwrap();
        let scope_b = SystemStateGraphScopeV1::try_new(environment("env-b"), &graph_b).unwrap();

        assert_ne!(
            scope_a.entity_ref(&EntityId("host-1".into())).unwrap(),
            scope_b.entity_ref(&EntityId("host-1".into())).unwrap()
        );
        assert_ne!(
            scope_a.observation_ref(&ObservationId("obs-1".into())).unwrap(),
            scope_b.observation_ref(&ObservationId("obs-1".into())).unwrap()
        );
        assert_ne!(
            scope_a.relation_ref(&RelationId("rel-1".into())).unwrap(),
            scope_b.relation_ref(&RelationId("rel-1".into())).unwrap()
        );
    }

    #[test]
    fn cross_environment_resolution_fails_closed() {
        let graph_a = populated_graph();
        let graph_b = populated_graph();
        let scope_a = SystemStateGraphScopeV1::try_new(environment("env-a"), &graph_a).unwrap();
        let scope_b = SystemStateGraphScopeV1::try_new(environment("env-b"), &graph_b).unwrap();
        let foreign = scope_b.entity_ref(&EntityId("host-1".into())).unwrap();

        assert!(matches!(
            scope_a.resolve_entity_ref(&foreign),
            Err(SystemStateIdentityErrorV1::EnvironmentMismatch { .. })
        ));
    }

    #[test]
    fn replay_of_same_environment_can_preserve_reference_identity_explicitly() {
        let graph = populated_graph();
        let replay = graph.clone();
        let env = environment("env-a");
        let original_scope = SystemStateGraphScopeV1::try_new(env.clone(), &graph).unwrap();
        let replay_scope = SystemStateGraphScopeV1::try_new(env, &replay).unwrap();

        assert_eq!(
            original_scope
                .entity_ref(&EntityId("host-1".into()))
                .unwrap(),
            replay_scope
                .entity_ref(&EntityId("host-1".into()))
                .unwrap()
        );
    }

    #[test]
    fn byte_identical_graphs_are_distinct_when_environment_identity_is_distinct() {
        let graph_a = populated_graph();
        let graph_b = graph_a.clone();
        let scope_a = SystemStateGraphScopeV1::try_new(environment("env-a"), &graph_a).unwrap();
        let scope_b = SystemStateGraphScopeV1::try_new(environment("env-b"), &graph_b).unwrap();

        assert_ne!(
            scope_a.entity_ref(&EntityId("host-1".into())).unwrap(),
            scope_b.entity_ref(&EntityId("host-1".into())).unwrap()
        );
    }

    #[test]
    fn empty_environment_identity_is_rejected() {
        assert_eq!(
            SystemEnvironmentIdV1::try_new("   "),
            Err(SystemStateIdentityErrorV1::EmptyEnvironmentId)
        );
    }

    #[test]
    fn noncanonical_environment_identity_is_rejected() {
        assert_eq!(
            SystemEnvironmentIdV1::try_new(" env-a "),
            Err(SystemStateIdentityErrorV1::NonCanonicalEnvironmentId)
        );
    }

    #[test]
    fn deserialization_cannot_bypass_environment_or_local_ref_validation() {
        for invalid in ["\"\"", "\"   \"", "\" env-a \""] {
            assert!(serde_json::from_str::<SystemEnvironmentIdV1>(invalid).is_err());
        }
        let parsed: SystemEnvironmentIdV1 = serde_json::from_str("\"env-a\"").unwrap();
        assert_eq!(parsed, environment("env-a"));

        assert!(serde_json::from_str::<SystemEntityRefV1>(
            r#"{"environment_id":"env-a","entity_id":" host-1 "}"#
        )
        .is_err());
    }

    #[test]
    fn durable_scope_rejects_entity_index_key_rebinding_missed_by_base_validation() {
        let malformed = rewrite_map_key(&entity_only_graph(), "entities", "host-1", "alias-host");
        malformed.validate().unwrap();

        assert!(matches!(
            SystemStateGraphScopeV1::try_new(environment("env-a"), &malformed),
            Err(SystemStateIdentityErrorV1::EntityIndexIdentityMismatch(_))
        ));
    }

    #[test]
    fn durable_scope_rejects_relation_index_key_rebinding_missed_by_base_validation() {
        let malformed = rewrite_map_key(&populated_graph(), "relations", "rel-1", "alias-rel");
        malformed.validate().unwrap();

        assert!(matches!(
            SystemStateGraphScopeV1::try_new(environment("env-a"), &malformed),
            Err(SystemStateIdentityErrorV1::RelationIndexIdentityMismatch(_))
        ));
    }

    #[test]
    fn durable_scope_rejects_observation_index_key_rebinding_missed_by_base_validation() {
        let malformed =
            rewrite_map_key(&observation_only_graph(), "observations", "obs-1", "alias-obs");
        malformed.validate().unwrap();

        assert!(matches!(
            SystemStateGraphScopeV1::try_new(environment("env-a"), &malformed),
            Err(SystemStateIdentityErrorV1::ObservationIndexIdentityMismatch(_))
        ));
    }
}
