// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound live system-state model for IT support reasoning.
//!
//! `SystemStateGraphV1` is intentionally distinct from long-lived semantic knowledge:
//! it represents what appears to be true about one concrete environment now, together
//! with the observations that justify those beliefs.
//!
//! Core non-equivalences:
//!
//! ```text
//! knowledge != current state
//! observation != truth
//! confidence != currentness
//! diagnosis != authority
//! proposed change != execution permission
//! ```
//!
//! V1 is deliberately framework-neutral and carries no execution capability.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// Stable identifier for an entity in one modeled environment.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

/// Stable identifier for a relation in one modeled environment.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RelationId(pub String);

/// Stable identifier for an observation/evidence item.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObservationId(pub String);

/// Broad entity vocabulary. Domain-specific adapters may use `Custom` without
/// weakening the identity of the entity itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKindV1 {
    Organization,
    Site,
    Network,
    Device,
    Host,
    VirtualMachine,
    Container,
    Process,
    Service,
    Storage,
    Identity,
    Policy,
    Certificate,
    Software,
    Configuration,
    TelemetrySource,
    Change,
    Incident,
    Custom(String),
}

/// Typed dependency/topology vocabulary for the live system model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationKindV1 {
    RunsOn,
    DependsOn,
    ConnectsTo,
    RoutesVia,
    ResolvesVia,
    AuthenticatesVia,
    AuthorizedBy,
    MountedFrom,
    ReplicatesTo,
    DeployedFrom,
    ConfiguredBy,
    ObservedBy,
    ChangedBy,
    MemberOf,
    ProtectedBy,
    Custom(String),
}

/// Framework-neutral scalar values for state attributes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StateValueV1 {
    Text(String),
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    TextList(Vec<String>),
}

/// Where an observation came from. Source kind is evidence metadata, not an
/// authority ranking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationSourceKindV1 {
    WindowsEventLog,
    Syslog,
    Metric,
    Trace,
    PacketCapture,
    Snmp,
    Gnmi,
    Netconf,
    Restconf,
    CloudApi,
    ConfigSnapshot,
    OperatorInput,
    SyntheticTest,
    Other(String),
}

/// Observation provenance needed to reconstruct how state was learned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationProvenanceV1 {
    /// Logical source identity, for example `host-27:eventlog:System`.
    pub source_id: String,
    pub source_kind: ObservationSourceKindV1,
    /// Collector/adapter implementation identity.
    pub collector: String,
    /// Optional collector/provider version.
    pub collector_version: Option<String>,
    /// Optional source schema/model version (e.g. OpenConfig model revision).
    pub schema_version: Option<String>,
    /// Optional immutable artifact/config/log digest when available.
    pub artifact_digest: Option<String>,
}

/// Times and uncertainty for one observation.
///
/// `event_time_unix_ms` may come from a remote clock and therefore is never
/// assumed to be strictly ordered with local collection/ingestion time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationClockV1 {
    /// Source-declared event time, if the source provides one.
    pub event_time_unix_ms: Option<u64>,
    /// Time the collector observed the state, on the collector's clock.
    pub observed_at_unix_ms: u64,
    /// Local ingestion time, if tracked separately.
    pub ingested_at_unix_ms: Option<u64>,
    /// Maximum age for which this observation may satisfy a currentness check.
    pub max_age_ms: Option<u64>,
    /// Declared clock uncertainty/error bound when known.
    pub clock_uncertainty_ms: Option<u64>,
}

/// Result of evaluating whether evidence is current enough for a consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentnessStatusV1 {
    Fresh,
    Stale,
    Indeterminate,
}

impl ObservationClockV1 {
    /// Evaluate freshness without pretending that missing validity semantics are
    /// equivalent to fresh evidence.
    pub fn currentness_at(&self, now_unix_ms: u64) -> CurrentnessStatusV1 {
        let Some(max_age_ms) = self.max_age_ms else {
            return CurrentnessStatusV1::Indeterminate;
        };

        let Some(age_ms) = now_unix_ms.checked_sub(self.observed_at_unix_ms) else {
            // Collector timestamp appears to be in the future relative to this clock.
            return CurrentnessStatusV1::Indeterminate;
        };

        if age_ms <= max_age_ms {
            CurrentnessStatusV1::Fresh
        } else {
            CurrentnessStatusV1::Stale
        }
    }
}

/// Immutable evidence record stored alongside graph state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemObservationV1 {
    pub id: ObservationId,
    /// Entity that the observation was primarily about. Relations can still cite
    /// the same evidence when the adapter learned a dependency/topology fact.
    pub subject: EntityId,
    pub provenance: ObservationProvenanceV1,
    pub clock: ObservationClockV1,
    /// Epistemic confidence in the normalized observation, independent of age.
    pub confidence: f32,
    /// Optional adapter-specific normalized facts retained for audit/replay.
    pub facts: BTreeMap<String, StateValueV1>,
}

impl SystemObservationV1 {
    pub fn validate(&self) -> Result<(), SystemStateGraphError> {
        if self.id.0.trim().is_empty() {
            return Err(SystemStateGraphError::EmptyIdentifier("observation"));
        }
        if self.subject.0.trim().is_empty() {
            return Err(SystemStateGraphError::EmptyIdentifier("subject"));
        }
        if self.provenance.source_id.trim().is_empty() {
            return Err(SystemStateGraphError::EmptyIdentifier("source_id"));
        }
        if self.provenance.collector.trim().is_empty() {
            return Err(SystemStateGraphError::EmptyIdentifier("collector"));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(SystemStateGraphError::InvalidConfidence(self.confidence));
        }
        Ok(())
    }
}

/// One entity's current normalized state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemEntityV1 {
    pub id: EntityId,
    pub kind: EntityKindV1,
    pub attributes: BTreeMap<String, StateValueV1>,
    /// Evidence items that contributed to the current projection.
    pub evidence: BTreeSet<ObservationId>,
    /// Monotonic graph-local entity revision.
    pub revision: u64,
}

/// One typed relation between current entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemRelationV1 {
    pub id: RelationId,
    pub from: EntityId,
    pub kind: RelationKindV1,
    pub to: EntityId,
    pub attributes: BTreeMap<String, StateValueV1>,
    pub evidence: BTreeSet<ObservationId>,
    /// Monotonic graph-local relation revision.
    pub revision: u64,
}

/// Framework-neutral live system-state graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemStateGraphV1 {
    /// Monotonic revision of the whole graph projection.
    pub revision: u64,
    entities: BTreeMap<EntityId, SystemEntityV1>,
    relations: BTreeMap<RelationId, SystemRelationV1>,
    observations: BTreeMap<ObservationId, SystemObservationV1>,
}

impl SystemStateGraphV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entity(&self, id: &EntityId) -> Option<&SystemEntityV1> {
        self.entities.get(id)
    }

    pub fn relation(&self, id: &RelationId) -> Option<&SystemRelationV1> {
        self.relations.get(id)
    }

    pub fn observation(&self, id: &ObservationId) -> Option<&SystemObservationV1> {
        self.observations.get(id)
    }

    pub fn entities(&self) -> impl Iterator<Item = &SystemEntityV1> {
        self.entities.values()
    }

    pub fn relations(&self) -> impl Iterator<Item = &SystemRelationV1> {
        self.relations.values()
    }

    pub fn observations(&self) -> impl Iterator<Item = &SystemObservationV1> {
        self.observations.values()
    }

    /// Record immutable evidence. Replaying byte-equivalent normalized evidence is
    /// idempotent; reusing the same observation identity for different evidence is
    /// rejected.
    pub fn record_observation(
        &mut self,
        observation: SystemObservationV1,
    ) -> Result<bool, SystemStateGraphError> {
        observation.validate()?;

        if let Some(existing) = self.observations.get(&observation.id) {
            if existing == &observation {
                return Ok(false);
            }
            return Err(SystemStateGraphError::ObservationIdentityConflict(
                observation.id,
            ));
        }

        self.observations.insert(observation.id.clone(), observation);
        self.bump_revision();
        Ok(true)
    }

    /// Project an entity update from already-recorded evidence.
    ///
    /// Attributes are merged. An entity identity cannot silently change kind.
    pub fn upsert_entity(
        &mut self,
        id: EntityId,
        kind: EntityKindV1,
        attributes: BTreeMap<String, StateValueV1>,
        evidence_id: &ObservationId,
    ) -> Result<u64, SystemStateGraphError> {
        self.require_evidence(evidence_id)?;
        validate_nonempty(&id.0, "entity")?;

        let entity_revision = match self.entities.get_mut(&id) {
            Some(existing) => {
                if existing.kind != kind {
                    return Err(SystemStateGraphError::EntityKindConflict(id));
                }
                existing.attributes.extend(attributes);
                existing.evidence.insert(evidence_id.clone());
                existing.revision = existing.revision.saturating_add(1);
                existing.revision
            }
            None => {
                let mut evidence = BTreeSet::new();
                evidence.insert(evidence_id.clone());
                self.entities.insert(
                    id.clone(),
                    SystemEntityV1 {
                        id,
                        kind,
                        attributes,
                        evidence,
                        revision: 1,
                    },
                );
                1
            }
        };

        self.bump_revision();
        Ok(entity_revision)
    }

    /// Project a relation update from already-recorded evidence.
    ///
    /// Both endpoints must already exist. A relation identity cannot silently be
    /// rebound to different endpoints or semantics.
    pub fn upsert_relation(
        &mut self,
        id: RelationId,
        from: EntityId,
        kind: RelationKindV1,
        to: EntityId,
        attributes: BTreeMap<String, StateValueV1>,
        evidence_id: &ObservationId,
    ) -> Result<u64, SystemStateGraphError> {
        self.require_evidence(evidence_id)?;
        validate_nonempty(&id.0, "relation")?;
        self.require_entity(&from)?;
        self.require_entity(&to)?;

        let relation_revision = match self.relations.get_mut(&id) {
            Some(existing) => {
                if existing.from != from || existing.to != to || existing.kind != kind {
                    return Err(SystemStateGraphError::RelationIdentityConflict(id));
                }
                existing.attributes.extend(attributes);
                existing.evidence.insert(evidence_id.clone());
                existing.revision = existing.revision.saturating_add(1);
                existing.revision
            }
            None => {
                let mut evidence = BTreeSet::new();
                evidence.insert(evidence_id.clone());
                self.relations.insert(
                    id.clone(),
                    SystemRelationV1 {
                        id,
                        from,
                        kind,
                        to,
                        attributes,
                        evidence,
                        revision: 1,
                    },
                );
                1
            }
        };

        self.bump_revision();
        Ok(relation_revision)
    }

    /// Return outgoing relations for dependency/topology traversal.
    pub fn outgoing_relations<'a>(
        &'a self,
        entity_id: &'a EntityId,
    ) -> impl Iterator<Item = &'a SystemRelationV1> + 'a {
        self.relations
            .values()
            .filter(move |relation| &relation.from == entity_id)
    }

    /// Return incoming relations for blast-radius/dependency traversal.
    pub fn incoming_relations<'a>(
        &'a self,
        entity_id: &'a EntityId,
    ) -> impl Iterator<Item = &'a SystemRelationV1> + 'a {
        self.relations
            .values()
            .filter(move |relation| &relation.to == entity_id)
    }

    /// Validate referential and evidence integrity of a deserialized graph.
    pub fn validate(&self) -> Result<(), SystemStateGraphError> {
        for observation in self.observations.values() {
            observation.validate()?;
        }

        for entity in self.entities.values() {
            validate_nonempty(&entity.id.0, "entity")?;
            for evidence_id in &entity.evidence {
                self.require_evidence(evidence_id)?;
            }
        }

        for relation in self.relations.values() {
            validate_nonempty(&relation.id.0, "relation")?;
            self.require_entity(&relation.from)?;
            self.require_entity(&relation.to)?;
            for evidence_id in &relation.evidence {
                self.require_evidence(evidence_id)?;
            }
        }

        Ok(())
    }

    fn require_evidence(&self, id: &ObservationId) -> Result<(), SystemStateGraphError> {
        if self.observations.contains_key(id) {
            Ok(())
        } else {
            Err(SystemStateGraphError::UnknownObservation(id.clone()))
        }
    }

    fn require_entity(&self, id: &EntityId) -> Result<(), SystemStateGraphError> {
        if self.entities.contains_key(id) {
            Ok(())
        } else {
            Err(SystemStateGraphError::UnknownEntity(id.clone()))
        }
    }

    fn bump_revision(&mut self) {
        self.revision = self.revision.saturating_add(1);
    }
}

fn validate_nonempty(value: &str, field: &'static str) -> Result<(), SystemStateGraphError> {
    if value.trim().is_empty() {
        Err(SystemStateGraphError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemStateGraphError {
    EmptyIdentifier(&'static str),
    InvalidConfidence(f32),
    ObservationIdentityConflict(ObservationId),
    UnknownObservation(ObservationId),
    UnknownEntity(EntityId),
    EntityKindConflict(EntityId),
    RelationIdentityConflict(RelationId),
}

impl fmt::Display for SystemStateGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(f, "{field} identifier must not be empty"),
            Self::InvalidConfidence(value) => {
                write!(f, "observation confidence must be finite and in [0, 1], got {value}")
            }
            Self::ObservationIdentityConflict(id) => {
                write!(f, "observation identity {:?} was reused for different evidence", id)
            }
            Self::UnknownObservation(id) => write!(f, "unknown observation {:?}", id),
            Self::UnknownEntity(id) => write!(f, "unknown entity {:?}", id),
            Self::EntityKindConflict(id) => {
                write!(f, "entity {:?} cannot silently change kind", id)
            }
            Self::RelationIdentityConflict(id) => {
                write!(f, "relation {:?} cannot silently change endpoints or kind", id)
            }
        }
    }
}

impl Error for SystemStateGraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(id: &str, subject: &str, observed_at: u64) -> SystemObservationV1 {
        SystemObservationV1 {
            id: ObservationId(id.to_string()),
            subject: EntityId(subject.to_string()),
            provenance: ObservationProvenanceV1 {
                source_id: format!("source:{subject}"),
                source_kind: ObservationSourceKindV1::SyntheticTest,
                collector: "system-state-test".to_string(),
                collector_version: Some("1".to_string()),
                schema_version: None,
                artifact_digest: None,
            },
            clock: ObservationClockV1 {
                event_time_unix_ms: None,
                observed_at_unix_ms: observed_at,
                ingested_at_unix_ms: Some(observed_at),
                max_age_ms: Some(1_000),
                clock_uncertainty_ms: None,
            },
            confidence: 0.9,
            facts: BTreeMap::new(),
        }
    }

    #[test]
    fn currentness_is_explicit_and_fail_indeterminate_when_undefined() {
        let clock = ObservationClockV1 {
            event_time_unix_ms: None,
            observed_at_unix_ms: 1_000,
            ingested_at_unix_ms: Some(1_010),
            max_age_ms: Some(500),
            clock_uncertainty_ms: Some(5),
        };

        assert_eq!(clock.currentness_at(1_200), CurrentnessStatusV1::Fresh);
        assert_eq!(clock.currentness_at(1_501), CurrentnessStatusV1::Stale);
        assert_eq!(clock.currentness_at(900), CurrentnessStatusV1::Indeterminate);

        let undefined = ObservationClockV1 {
            max_age_ms: None,
            ..clock
        };
        assert_eq!(
            undefined.currentness_at(1_200),
            CurrentnessStatusV1::Indeterminate
        );
    }

    #[test]
    fn observation_identity_is_idempotent_but_not_rebindable() {
        let mut graph = SystemStateGraphV1::new();
        let obs = observation("obs-1", "host-1", 1_000);

        assert_eq!(graph.record_observation(obs.clone()).unwrap(), true);
        assert_eq!(graph.record_observation(obs.clone()).unwrap(), false);

        let mut conflicting = obs;
        conflicting.confidence = 0.5;
        assert!(matches!(
            graph.record_observation(conflicting),
            Err(SystemStateGraphError::ObservationIdentityConflict(_))
        ));
    }

    #[test]
    fn graph_requires_evidence_and_existing_relation_endpoints() {
        let mut graph = SystemStateGraphV1::new();
        let obs = observation("obs-1", "host-1", 1_000);
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

        let result = graph.upsert_relation(
            RelationId("rel-1".into()),
            EntityId("host-1".into()),
            RelationKindV1::DependsOn,
            EntityId("dns-1".into()),
            BTreeMap::new(),
            &obs_id,
        );

        assert!(matches!(result, Err(SystemStateGraphError::UnknownEntity(_))));
    }

    #[test]
    fn entity_and_relation_updates_accumulate_evidence() {
        let mut graph = SystemStateGraphV1::new();
        let obs1 = observation("obs-1", "app-1", 1_000);
        let obs2 = observation("obs-2", "app-1", 1_100);
        let obs1_id = obs1.id.clone();
        let obs2_id = obs2.id.clone();
        graph.record_observation(obs1).unwrap();
        graph.record_observation(obs2).unwrap();

        let mut app_attrs = BTreeMap::new();
        app_attrs.insert("status".into(), StateValueV1::Text("healthy".into()));
        graph
            .upsert_entity(
                EntityId("app-1".into()),
                EntityKindV1::Service,
                app_attrs,
                &obs1_id,
            )
            .unwrap();
        graph
            .upsert_entity(
                EntityId("db-1".into()),
                EntityKindV1::Service,
                BTreeMap::new(),
                &obs1_id,
            )
            .unwrap();

        graph
            .upsert_relation(
                RelationId("app-db".into()),
                EntityId("app-1".into()),
                RelationKindV1::DependsOn,
                EntityId("db-1".into()),
                BTreeMap::new(),
                &obs1_id,
            )
            .unwrap();
        graph
            .upsert_relation(
                RelationId("app-db".into()),
                EntityId("app-1".into()),
                RelationKindV1::DependsOn,
                EntityId("db-1".into()),
                BTreeMap::new(),
                &obs2_id,
            )
            .unwrap();

        let relation = graph.relation(&RelationId("app-db".into())).unwrap();
        assert_eq!(relation.revision, 2);
        assert_eq!(relation.evidence.len(), 2);
        assert_eq!(graph.incoming_relations(&EntityId("db-1".into())).count(), 1);
        assert_eq!(graph.outgoing_relations(&EntityId("app-1".into())).count(), 1);
        graph.validate().unwrap();
    }

    #[test]
    fn entity_identity_cannot_silently_change_kind() {
        let mut graph = SystemStateGraphV1::new();
        let obs = observation("obs-1", "node-1", 1_000);
        let obs_id = obs.id.clone();
        graph.record_observation(obs).unwrap();
        graph
            .upsert_entity(
                EntityId("node-1".into()),
                EntityKindV1::Host,
                BTreeMap::new(),
                &obs_id,
            )
            .unwrap();

        assert!(matches!(
            graph.upsert_entity(
                EntityId("node-1".into()),
                EntityKindV1::Network,
                BTreeMap::new(),
                &obs_id,
            ),
            Err(SystemStateGraphError::EntityKindConflict(_))
        ));
    }

    #[test]
    fn invalid_confidence_is_rejected() {
        let mut graph = SystemStateGraphV1::new();
        let mut obs = observation("obs-1", "host-1", 1_000);
        obs.confidence = f32::NAN;

        assert!(matches!(
            graph.record_observation(obs),
            Err(SystemStateGraphError::InvalidConfidence(_))
        ));
    }
}
