// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Vendor-neutral entity and relationship discovery contracts.
//!
//! Topology edges keep their semantic relation separate from the epistemic
//! basis for that edge. A Kubernetes ownerReference, an observed flow, a
//! vendor causal claim, and a model hypothesis must never become equivalent
//! merely because they connect the same two entities.

use crate::observation::{EntityRef, ObservationId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum RelationKind {
    DependsOn,
    HostedOn,
    RoutesThrough,
    AuthenticatedBy,
    ConfiguredBy,
    OwnedBy,
    Serves,
    CommunicatesWith,
    ObservedBy,
    MemberOf,
    ConnectedTo,
    Other(String),
}

/// Epistemic basis for a topology edge.
///
/// This is deliberately independent from [`RelationKind`]. `DependsOn` may be
/// a declared structural dependency, a runtime observation, a third-party
/// causal claim, or only a model hypothesis; downstream reasoning must retain
/// that distinction.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum RelationBasis {
    Structural,
    Observational,
    CausalClaim,
    Hypothesis,
}

impl Default for RelationBasis {
    fn default() -> Self {
        Self::Structural
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRelation {
    pub from: EntityRef,
    pub to: EntityRef,
    pub kind: RelationKind,
    #[serde(default)]
    pub basis: RelationBasis,
    /// Confidence that this edge exists with this semantic meaning, [0, 1].
    pub confidence: f32,
    /// Source-native time for an observed/claimed relation, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at_unix_ms: Option<u64>,
    /// Runtime observations supporting this edge. Structural edges may have no
    /// observation IDs when their evidence is the discovered object itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_observation_ids: Vec<ObservationId>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

impl EntityRelation {
    pub fn validate(&self) -> Result<(), TopologyValidationError> {
        validate_entity(&self.from)?;
        validate_entity(&self.to)?;
        if self.from == self.to {
            return Err(TopologyValidationError::SelfRelation(
                self.from.canonical_key(),
            ));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(TopologyValidationError::ConfidenceOutOfRange(
                self.confidence,
            ));
        }

        let mut evidence = BTreeSet::new();
        for id in &self.evidence_observation_ids {
            if id.as_str().trim().is_empty() {
                return Err(TopologyValidationError::EmptyEvidenceObservationId);
            }
            if !evidence.insert(id.clone()) {
                return Err(TopologyValidationError::DuplicateEvidenceObservationId(
                    id.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredEntity {
    pub entity: EntityRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoverySnapshot {
    pub integration_id: String,
    pub discovered_at_unix_ms: u64,
    pub entities: Vec<DiscoveredEntity>,
    pub relations: Vec<EntityRelation>,
}

impl DiscoverySnapshot {
    pub fn validate(&self) -> Result<(), TopologyValidationError> {
        if self.integration_id.trim().is_empty() {
            return Err(TopologyValidationError::EmptyIntegrationId);
        }

        let mut entity_keys = BTreeSet::new();
        for entity in &self.entities {
            validate_entity(&entity.entity)?;
            let key = entity.entity.canonical_key();
            if !entity_keys.insert(key.clone()) {
                return Err(TopologyValidationError::DuplicateEntity(key));
            }
        }

        let mut relation_keys = BTreeSet::new();
        for relation in &self.relations {
            relation.validate()?;
            let from = relation.from.canonical_key();
            let to = relation.to.canonical_key();
            if !entity_keys.contains(&from) {
                return Err(TopologyValidationError::DanglingRelationEndpoint(from));
            }
            if !entity_keys.contains(&to) {
                return Err(TopologyValidationError::DanglingRelationEndpoint(to));
            }
            let key = (
                relation.from.clone(),
                relation.to.clone(),
                relation.kind.clone(),
                relation.basis,
            );
            if !relation_keys.insert(key) {
                return Err(TopologyValidationError::DuplicateRelation {
                    from,
                    to,
                    kind: relation.kind.clone(),
                    basis: relation.basis,
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TopologyValidationError {
    #[error("discovery snapshot integration id is empty")]
    EmptyIntegrationId,
    #[error("entity field `{0}` is empty")]
    EmptyEntityField(&'static str),
    #[error("duplicate discovered entity `{0}`")]
    DuplicateEntity(String),
    #[error("topology relation points to undiscovered entity `{0}`")]
    DanglingRelationEndpoint(String),
    #[error("topology relation cannot point an entity at itself: `{0}`")]
    SelfRelation(String),
    #[error("relation confidence must be finite and within [0,1], got {0}")]
    ConfidenceOutOfRange(f32),
    #[error("relation evidence observation id is empty")]
    EmptyEvidenceObservationId,
    #[error("duplicate relation evidence observation id `{0}`")]
    DuplicateEvidenceObservationId(ObservationId),
    #[error("duplicate topology relation {from} -> {to} ({kind:?}, {basis:?})")]
    DuplicateRelation {
        from: String,
        to: String,
        kind: RelationKind,
        basis: RelationBasis,
    },
}

fn validate_entity(entity: &EntityRef) -> Result<(), TopologyValidationError> {
    if entity.namespace.trim().is_empty() {
        return Err(TopologyValidationError::EmptyEntityField("namespace"));
    }
    if entity.kind.trim().is_empty() {
        return Err(TopologyValidationError::EmptyEntityField("kind"));
    }
    if entity.id.trim().is_empty() {
        return Err(TopologyValidationError::EmptyEntityField("id"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(id: &str) -> EntityRef {
        EntityRef::new("site:lab", "service", id)
    }

    fn discovered(id: &str) -> DiscoveredEntity {
        DiscoveredEntity {
            entity: entity(id),
            display_name: None,
            attributes: BTreeMap::new(),
        }
    }

    fn relation(from: &str, to: &str) -> EntityRelation {
        EntityRelation {
            from: entity(from),
            to: entity(to),
            kind: RelationKind::DependsOn,
            basis: RelationBasis::Structural,
            confidence: 1.0,
            observed_at_unix_ms: None,
            evidence_observation_ids: vec![],
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn discovery_snapshot_rejects_duplicate_entities() {
        let duplicate = discovered("api");
        let snapshot = DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 1,
            entities: vec![duplicate.clone(), duplicate],
            relations: vec![],
        };
        assert!(matches!(
            snapshot.validate(),
            Err(TopologyValidationError::DuplicateEntity(_))
        ));
    }

    #[test]
    fn relation_confidence_fails_closed() {
        let mut relation = relation("api", "db");
        relation.confidence = f32::NAN;
        assert!(matches!(
            relation.validate(),
            Err(TopologyValidationError::ConfidenceOutOfRange(_))
        ));
    }

    #[test]
    fn dangling_relation_endpoint_is_rejected() {
        let snapshot = DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 1,
            entities: vec![discovered("api")],
            relations: vec![relation("api", "db")],
        };
        assert!(matches!(
            snapshot.validate(),
            Err(TopologyValidationError::DanglingRelationEndpoint(_))
        ));
    }

    #[test]
    fn duplicate_relation_is_rejected_not_double_counted() {
        let edge = relation("api", "db");
        let snapshot = DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 1,
            entities: vec![discovered("api"), discovered("db")],
            relations: vec![edge.clone(), edge],
        };
        assert!(matches!(
            snapshot.validate(),
            Err(TopologyValidationError::DuplicateRelation { .. })
        ));
    }

    #[test]
    fn relation_basis_keeps_structural_and_hypothesis_edges_distinct() {
        let structural = relation("api", "db");
        let mut hypothesis = structural.clone();
        hypothesis.basis = RelationBasis::Hypothesis;
        let snapshot = DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 1,
            entities: vec![discovered("api"), discovered("db")],
            relations: vec![structural, hypothesis],
        };
        assert!(snapshot.validate().is_ok());
    }
}
