// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resource/cardinality admission budgets for discovered topology.
//!
//! Discovery is read-only but can still become an amplification path: one
//! malformed inventory dump may contain enormous entity IDs, attributes, or
//! relation fan-out. These deterministic limits are applied before topology
//! reaches the world model.

use crate::{DiscoverySnapshot, RelationKind, TopologyValidationError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyLimits {
    pub max_entities: usize,
    pub max_relations: usize,
    pub max_entity_namespace_bytes: usize,
    pub max_entity_kind_bytes: usize,
    pub max_entity_id_bytes: usize,
    pub max_display_name_bytes: usize,
    pub max_attributes_per_entity: usize,
    pub max_attributes_per_relation: usize,
    pub max_attribute_key_bytes: usize,
    pub max_attribute_value_bytes: usize,
    pub max_other_relation_kind_bytes: usize,
    pub max_evidence_refs_per_relation: usize,
    /// Approximate cumulative UTF-8 string bytes admitted in one snapshot.
    pub max_total_string_bytes: usize,
}

impl Default for TopologyLimits {
    fn default() -> Self {
        Self {
            max_entities: 50_000,
            max_relations: 200_000,
            max_entity_namespace_bytes: 1_024,
            max_entity_kind_bytes: 128,
            max_entity_id_bytes: 4_096,
            max_display_name_bytes: 4_096,
            max_attributes_per_entity: 128,
            max_attributes_per_relation: 128,
            max_attribute_key_bytes: 256,
            max_attribute_value_bytes: 16_384,
            max_other_relation_kind_bytes: 512,
            max_evidence_refs_per_relation: 64,
            max_total_string_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TopologyBudgetError {
    #[error("topology snapshot is structurally invalid: {0}")]
    InvalidSnapshot(#[from] TopologyValidationError),
    #[error("topology contains {actual} entities; limit is {limit}")]
    TooManyEntities { actual: usize, limit: usize },
    #[error("topology contains {actual} relations; limit is {limit}")]
    TooManyRelations { actual: usize, limit: usize },
    #[error("entity {index} namespace is {actual} bytes; limit is {limit}")]
    EntityNamespaceTooLarge { index: usize, actual: usize, limit: usize },
    #[error("entity {index} kind is {actual} bytes; limit is {limit}")]
    EntityKindTooLarge { index: usize, actual: usize, limit: usize },
    #[error("entity {index} id is {actual} bytes; limit is {limit}")]
    EntityIdTooLarge { index: usize, actual: usize, limit: usize },
    #[error("entity {index} display name is {actual} bytes; limit is {limit}")]
    DisplayNameTooLarge { index: usize, actual: usize, limit: usize },
    #[error("entity {index} has {actual} attributes; limit is {limit}")]
    TooManyEntityAttributes { index: usize, actual: usize, limit: usize },
    #[error("relation {index} has {actual} attributes; limit is {limit}")]
    TooManyRelationAttributes { index: usize, actual: usize, limit: usize },
    #[error("topology attribute key is {actual} bytes; limit is {limit}")]
    AttributeKeyTooLarge { actual: usize, limit: usize },
    #[error("topology attribute value is {actual} bytes; limit is {limit}")]
    AttributeValueTooLarge { actual: usize, limit: usize },
    #[error("relation {index} custom kind is {actual} bytes; limit is {limit}")]
    RelationKindTooLarge { index: usize, actual: usize, limit: usize },
    #[error("relation {index} has {actual} evidence references; limit is {limit}")]
    TooManyEvidenceRefs { index: usize, actual: usize, limit: usize },
    #[error("topology string footprint exceeded {limit} bytes (observed at least {actual})")]
    TotalStringBytesExceeded { actual: usize, limit: usize },
}

impl DiscoverySnapshot {
    pub fn validate_with_limits(
        &self,
        limits: &TopologyLimits,
    ) -> Result<(), TopologyBudgetError> {
        self.validate()?;

        if self.entities.len() > limits.max_entities {
            return Err(TopologyBudgetError::TooManyEntities {
                actual: self.entities.len(),
                limit: limits.max_entities,
            });
        }
        if self.relations.len() > limits.max_relations {
            return Err(TopologyBudgetError::TooManyRelations {
                actual: self.relations.len(),
                limit: limits.max_relations,
            });
        }

        let mut total = self.integration_id.len();
        for (index, discovered) in self.entities.iter().enumerate() {
            check_len(
                discovered.entity.namespace.len(),
                limits.max_entity_namespace_bytes,
                |actual, limit| TopologyBudgetError::EntityNamespaceTooLarge {
                    index,
                    actual,
                    limit,
                },
            )?;
            check_len(
                discovered.entity.kind.len(),
                limits.max_entity_kind_bytes,
                |actual, limit| TopologyBudgetError::EntityKindTooLarge {
                    index,
                    actual,
                    limit,
                },
            )?;
            check_len(
                discovered.entity.id.len(),
                limits.max_entity_id_bytes,
                |actual, limit| TopologyBudgetError::EntityIdTooLarge {
                    index,
                    actual,
                    limit,
                },
            )?;
            total = total
                .saturating_add(discovered.entity.namespace.len())
                .saturating_add(discovered.entity.kind.len())
                .saturating_add(discovered.entity.id.len());

            if let Some(display_name) = &discovered.display_name {
                check_len(display_name.len(), limits.max_display_name_bytes, |actual, limit| {
                    TopologyBudgetError::DisplayNameTooLarge {
                        index,
                        actual,
                        limit,
                    }
                })?;
                total = total.saturating_add(display_name.len());
            }

            if discovered.attributes.len() > limits.max_attributes_per_entity {
                return Err(TopologyBudgetError::TooManyEntityAttributes {
                    index,
                    actual: discovered.attributes.len(),
                    limit: limits.max_attributes_per_entity,
                });
            }
            for (key, value) in &discovered.attributes {
                validate_attribute(key, value, limits)?;
                total = total.saturating_add(key.len()).saturating_add(value.len());
            }
            check_total(total, limits)?;
        }

        for (index, relation) in self.relations.iter().enumerate() {
            total = total
                .saturating_add(relation.from.namespace.len())
                .saturating_add(relation.from.kind.len())
                .saturating_add(relation.from.id.len())
                .saturating_add(relation.to.namespace.len())
                .saturating_add(relation.to.kind.len())
                .saturating_add(relation.to.id.len());

            if let RelationKind::Other(kind) = &relation.kind {
                check_len(kind.len(), limits.max_other_relation_kind_bytes, |actual, limit| {
                    TopologyBudgetError::RelationKindTooLarge {
                        index,
                        actual,
                        limit,
                    }
                })?;
                total = total.saturating_add(kind.len());
            }

            if relation.evidence_observation_ids.len() > limits.max_evidence_refs_per_relation {
                return Err(TopologyBudgetError::TooManyEvidenceRefs {
                    index,
                    actual: relation.evidence_observation_ids.len(),
                    limit: limits.max_evidence_refs_per_relation,
                });
            }
            for evidence in &relation.evidence_observation_ids {
                total = total.saturating_add(evidence.as_str().len());
            }

            if relation.attributes.len() > limits.max_attributes_per_relation {
                return Err(TopologyBudgetError::TooManyRelationAttributes {
                    index,
                    actual: relation.attributes.len(),
                    limit: limits.max_attributes_per_relation,
                });
            }
            for (key, value) in &relation.attributes {
                validate_attribute(key, value, limits)?;
                total = total.saturating_add(key.len()).saturating_add(value.len());
            }
            check_total(total, limits)?;
        }

        Ok(())
    }
}

fn validate_attribute(
    key: &str,
    value: &str,
    limits: &TopologyLimits,
) -> Result<(), TopologyBudgetError> {
    if key.len() > limits.max_attribute_key_bytes {
        return Err(TopologyBudgetError::AttributeKeyTooLarge {
            actual: key.len(),
            limit: limits.max_attribute_key_bytes,
        });
    }
    if value.len() > limits.max_attribute_value_bytes {
        return Err(TopologyBudgetError::AttributeValueTooLarge {
            actual: value.len(),
            limit: limits.max_attribute_value_bytes,
        });
    }
    Ok(())
}

fn check_len<F>(actual: usize, limit: usize, error: F) -> Result<(), TopologyBudgetError>
where
    F: FnOnce(usize, usize) -> TopologyBudgetError,
{
    if actual > limit {
        Err(error(actual, limit))
    } else {
        Ok(())
    }
}

fn check_total(total: usize, limits: &TopologyLimits) -> Result<(), TopologyBudgetError> {
    if total > limits.max_total_string_bytes {
        Err(TopologyBudgetError::TotalStringBytesExceeded {
            actual: total,
            limit: limits.max_total_string_bytes,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DiscoveredEntity, EntityRef, EntityRelation, RelationBasis, RelationKind};
    use std::collections::BTreeMap;

    fn snapshot() -> DiscoverySnapshot {
        let api = EntityRef::new("cluster:lab", "service", "api");
        let db = EntityRef::new("cluster:lab", "service", "db");
        DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 1,
            entities: vec![
                DiscoveredEntity {
                    entity: api.clone(),
                    display_name: Some("api".into()),
                    attributes: BTreeMap::new(),
                },
                DiscoveredEntity {
                    entity: db.clone(),
                    display_name: Some("db".into()),
                    attributes: BTreeMap::new(),
                },
            ],
            relations: vec![EntityRelation {
                from: api,
                to: db,
                kind: RelationKind::DependsOn,
                basis: RelationBasis::Structural,
                confidence: 1.0,
                observed_at_unix_ms: None,
                evidence_observation_ids: vec![],
                attributes: BTreeMap::new(),
            }],
        }
    }

    #[test]
    fn ordinary_topology_fits_default_budget() {
        assert!(snapshot()
            .validate_with_limits(&TopologyLimits::default())
            .is_ok());
    }

    #[test]
    fn entity_cardinality_fails_closed() {
        let limits = TopologyLimits {
            max_entities: 1,
            ..Default::default()
        };
        assert!(matches!(
            snapshot().validate_with_limits(&limits),
            Err(TopologyBudgetError::TooManyEntities { .. })
        ));
    }

    #[test]
    fn relation_evidence_cardinality_fails_closed() {
        let mut snapshot = snapshot();
        snapshot.relations[0].evidence_observation_ids = vec![
            crate::ObservationId::new("a"),
            crate::ObservationId::new("b"),
        ];
        let limits = TopologyLimits {
            max_evidence_refs_per_relation: 1,
            ..Default::default()
        };
        assert!(matches!(
            snapshot.validate_with_limits(&limits),
            Err(TopologyBudgetError::TooManyEvidenceRefs { .. })
        ));
    }
}
