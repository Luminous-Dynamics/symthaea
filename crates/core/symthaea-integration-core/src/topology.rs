// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Vendor-neutral entity and relationship discovery contracts.

use crate::observation::EntityRef;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRelation {
    pub from: EntityRef,
    pub to: EntityRef,
    pub kind: RelationKind,
    /// Confidence that this edge exists with this semantic meaning, [0, 1].
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

impl EntityRelation {
    pub fn validate(&self) -> Result<(), TopologyValidationError> {
        validate_entity(&self.from)?;
        validate_entity(&self.to)?;
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(TopologyValidationError::ConfidenceOutOfRange(
                self.confidence,
            ));
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

        for relation in &self.relations {
            relation.validate()?;
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
    #[error("relation confidence must be finite and within [0,1], got {0}")]
    ConfidenceOutOfRange(f32),
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

    #[test]
    fn discovery_snapshot_rejects_duplicate_entities() {
        let duplicate = DiscoveredEntity {
            entity: entity("api"),
            display_name: None,
            attributes: BTreeMap::new(),
        };
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
        let relation = EntityRelation {
            from: entity("api"),
            to: entity("db"),
            kind: RelationKind::DependsOn,
            confidence: f32::NAN,
            attributes: BTreeMap::new(),
        };
        assert!(matches!(
            relation.validate(),
            Err(TopologyValidationError::ConfidenceOutOfRange(_))
        ));
    }
}
