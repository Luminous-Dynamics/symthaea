// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resource/cardinality admission budgets for integration output.
//!
//! Read-only does not mean harmless: an exporter can still exhaust memory or
//! destabilize a reasoning loop with pathological cardinality. These limits
//! bound the shape of an admitted observation batch before it reaches the world
//! model. They are deterministic schema/resource checks, not transport quotas.

use crate::{BatchValidationError, ObservationBatch, ObservationValue};

/// Conservative default limits for one integration batch.
///
/// Deployments may choose tighter limits. Raising them should be an explicit
/// operational decision rather than an adapter-controlled behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationLimits {
    pub max_batch_observations: usize,
    pub max_labels_per_observation: usize,
    pub max_label_key_bytes: usize,
    pub max_label_value_bytes: usize,
    pub max_signal_bytes: usize,
    pub max_text_value_bytes: usize,
    pub max_attributes_per_observation: usize,
    pub max_attribute_key_bytes: usize,
    pub max_attribute_value_bytes: usize,
    pub max_lineage_parents: usize,
    pub max_transforms: usize,
    /// Approximate cumulative UTF-8 string bytes admitted in one batch.
    pub max_total_string_bytes: usize,
}

impl Default for ObservationLimits {
    fn default() -> Self {
        Self {
            max_batch_observations: 50_000,
            max_labels_per_observation: 64,
            max_label_key_bytes: 256,
            max_label_value_bytes: 4_096,
            max_signal_bytes: 512,
            max_text_value_bytes: 65_536,
            max_attributes_per_observation: 128,
            max_attribute_key_bytes: 256,
            max_attribute_value_bytes: 16_384,
            max_lineage_parents: 64,
            max_transforms: 32,
            max_total_string_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObservationBudgetError {
    #[error("observation batch is structurally invalid: {0}")]
    InvalidBatch(#[from] BatchValidationError),
    #[error("batch contains {actual} observations; limit is {limit}")]
    TooManyObservations { actual: usize, limit: usize },
    #[error("observation {index} signal is {actual} bytes; limit is {limit}")]
    SignalTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} contains {actual} labels; limit is {limit}")]
    TooManyLabels {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} label key is {actual} bytes; limit is {limit}")]
    LabelKeyTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} label value is {actual} bytes; limit is {limit}")]
    LabelValueTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} text value is {actual} bytes; limit is {limit}")]
    TextValueTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} contains {actual} attributes; limit is {limit}")]
    TooManyAttributes {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} attribute key is {actual} bytes; limit is {limit}")]
    AttributeKeyTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} attribute value is {actual} bytes; limit is {limit}")]
    AttributeValueTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} has {actual} lineage parents; limit is {limit}")]
    TooManyLineageParents {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("observation {index} has {actual} transforms; limit is {limit}")]
    TooManyTransforms {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("batch string footprint exceeded {limit} bytes (observed at least {actual})")]
    TotalStringBytesExceeded { actual: usize, limit: usize },
}

impl ObservationBatch {
    /// Validate both structural correctness and explicit resource/cardinality
    /// limits before admitting this batch to a world model.
    pub fn validate_with_limits(
        &self,
        limits: &ObservationLimits,
    ) -> Result<(), ObservationBudgetError> {
        self.validate()?;

        if self.observations.len() > limits.max_batch_observations {
            return Err(ObservationBudgetError::TooManyObservations {
                actual: self.observations.len(),
                limit: limits.max_batch_observations,
            });
        }

        let mut total_string_bytes = self.integration_id.len();
        for (index, observation) in self.observations.iter().enumerate() {
            check_len(
                index,
                observation.signal.len(),
                limits.max_signal_bytes,
                |index, actual, limit| ObservationBudgetError::SignalTooLarge {
                    index,
                    actual,
                    limit,
                },
            )?;
            total_string_bytes = total_string_bytes.saturating_add(observation.signal.len());

            if observation.labels.len() > limits.max_labels_per_observation {
                return Err(ObservationBudgetError::TooManyLabels {
                    index,
                    actual: observation.labels.len(),
                    limit: limits.max_labels_per_observation,
                });
            }
            for (key, value) in &observation.labels {
                if key.len() > limits.max_label_key_bytes {
                    return Err(ObservationBudgetError::LabelKeyTooLarge {
                        index,
                        actual: key.len(),
                        limit: limits.max_label_key_bytes,
                    });
                }
                if value.len() > limits.max_label_value_bytes {
                    return Err(ObservationBudgetError::LabelValueTooLarge {
                        index,
                        actual: value.len(),
                        limit: limits.max_label_value_bytes,
                    });
                }
                total_string_bytes = total_string_bytes
                    .saturating_add(key.len())
                    .saturating_add(value.len());
            }

            if observation.lineage.parent_ids.len() > limits.max_lineage_parents {
                return Err(ObservationBudgetError::TooManyLineageParents {
                    index,
                    actual: observation.lineage.parent_ids.len(),
                    limit: limits.max_lineage_parents,
                });
            }
            if observation.lineage.transforms.len() > limits.max_transforms {
                return Err(ObservationBudgetError::TooManyTransforms {
                    index,
                    actual: observation.lineage.transforms.len(),
                    limit: limits.max_transforms,
                });
            }

            total_string_bytes = total_string_bytes
                .saturating_add(observation.observation_id.0.len())
                .saturating_add(observation.entity.namespace.len())
                .saturating_add(observation.entity.kind.len())
                .saturating_add(observation.entity.id.len())
                .saturating_add(observation.source.integration_id.len())
                .saturating_add(observation.source.measurement_method.len())
                .saturating_add(observation.lineage.lineage_id.len());

            if let Some(value) = &observation.source.collector_id {
                total_string_bytes = total_string_bytes.saturating_add(value.len());
            }
            if let Some(value) = &observation.source.upstream_origin {
                total_string_bytes = total_string_bytes.saturating_add(value.len());
            }
            if let Some(value) = &observation.source.tenant {
                total_string_bytes = total_string_bytes.saturating_add(value.len());
            }
            if let Some(value) = &observation.lineage.independence_group {
                total_string_bytes = total_string_bytes.saturating_add(value.len());
            }
            for parent in &observation.lineage.parent_ids {
                total_string_bytes = total_string_bytes.saturating_add(parent.0.len());
            }
            for transform in &observation.lineage.transforms {
                total_string_bytes = total_string_bytes.saturating_add(transform.name.len());
                if let Some(version) = &transform.version {
                    total_string_bytes = total_string_bytes.saturating_add(version.len());
                }
            }

            match &observation.value {
                ObservationValue::Text(value) => {
                    if value.len() > limits.max_text_value_bytes {
                        return Err(ObservationBudgetError::TextValueTooLarge {
                            index,
                            actual: value.len(),
                            limit: limits.max_text_value_bytes,
                        });
                    }
                    total_string_bytes = total_string_bytes.saturating_add(value.len());
                }
                ObservationValue::Attributes(attributes) => {
                    if attributes.len() > limits.max_attributes_per_observation {
                        return Err(ObservationBudgetError::TooManyAttributes {
                            index,
                            actual: attributes.len(),
                            limit: limits.max_attributes_per_observation,
                        });
                    }
                    for (key, value) in attributes {
                        if key.len() > limits.max_attribute_key_bytes {
                            return Err(ObservationBudgetError::AttributeKeyTooLarge {
                                index,
                                actual: key.len(),
                                limit: limits.max_attribute_key_bytes,
                            });
                        }
                        if value.len() > limits.max_attribute_value_bytes {
                            return Err(ObservationBudgetError::AttributeValueTooLarge {
                                index,
                                actual: value.len(),
                                limit: limits.max_attribute_value_bytes,
                            });
                        }
                        total_string_bytes = total_string_bytes
                            .saturating_add(key.len())
                            .saturating_add(value.len());
                    }
                }
                ObservationValue::Number { unit, .. } => {
                    if let Some(unit) = unit {
                        total_string_bytes = total_string_bytes.saturating_add(unit.len());
                    }
                }
                ObservationValue::Integer(_)
                | ObservationValue::Unsigned(_)
                | ObservationValue::Boolean(_) => {}
            }

            if total_string_bytes > limits.max_total_string_bytes {
                return Err(ObservationBudgetError::TotalStringBytesExceeded {
                    actual: total_string_bytes,
                    limit: limits.max_total_string_bytes,
                });
            }
        }

        Ok(())
    }
}

fn check_len<F>(
    index: usize,
    actual: usize,
    limit: usize,
    error: F,
) -> Result<(), ObservationBudgetError>
where
    F: FnOnce(usize, usize, usize) -> ObservationBudgetError,
{
    if actual > limit {
        Err(error(index, actual, limit))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EntityRef, ObservationEnvelope, ObservationId, ObservationKind, ObservationLineage,
        ObservationQuality, ObservationSource,
    };
    use std::collections::BTreeMap;

    fn observation() -> ObservationEnvelope {
        ObservationEnvelope::new(
            ObservationId::new("obs-1"),
            1_000,
            1_010,
            EntityRef::new("site:lab", "host", "node-1"),
            ObservationKind::Metric,
            "system.cpu.utilization",
            ObservationValue::Number {
                value: 0.5,
                unit: Some("1".into()),
            },
            ObservationSource {
                integration_id: "fixture".into(),
                collector_id: None,
                upstream_origin: None,
                measurement_method: "fixture".into(),
                tenant: None,
            },
            ObservationQuality::observed(1.0),
            ObservationLineage {
                lineage_id: "lineage-1".into(),
                parent_ids: vec![],
                independence_group: None,
                transforms: vec![],
            },
        )
    }

    fn batch(observations: Vec<ObservationEnvelope>) -> ObservationBatch {
        ObservationBatch {
            integration_id: "fixture".into(),
            collected_at_unix_ms: 1_010,
            observations,
        }
    }

    #[test]
    fn ordinary_batch_fits_default_budget() {
        assert!(batch(vec![observation()])
            .validate_with_limits(&ObservationLimits::default())
            .is_ok());
    }

    #[test]
    fn excessive_label_cardinality_fails_closed() {
        let mut observation = observation();
        observation.labels = (0..4)
            .map(|index| (format!("key-{index}"), "value".to_string()))
            .collect::<BTreeMap<_, _>>();
        let limits = ObservationLimits {
            max_labels_per_observation: 3,
            ..Default::default()
        };
        assert!(matches!(
            batch(vec![observation]).validate_with_limits(&limits),
            Err(ObservationBudgetError::TooManyLabels { .. })
        ));
    }

    #[test]
    fn oversized_text_payload_fails_closed() {
        let mut observation = observation();
        observation.value = ObservationValue::Text("0123456789".into());
        let limits = ObservationLimits {
            max_text_value_bytes: 8,
            ..Default::default()
        };
        assert!(matches!(
            batch(vec![observation]).validate_with_limits(&limits),
            Err(ObservationBudgetError::TextValueTooLarge { .. })
        ));
    }

    #[test]
    fn total_string_budget_catches_many_individually_small_values() {
        let mut observation = observation();
        observation.labels.insert("a".into(), "b".into());
        let limits = ObservationLimits {
            max_total_string_bytes: 8,
            ..Default::default()
        };
        assert!(matches!(
            batch(vec![observation]).validate_with_limits(&limits),
            Err(ObservationBudgetError::TotalStringBytesExceeded { .. })
        ));
    }
}
