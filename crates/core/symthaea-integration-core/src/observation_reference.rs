// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-qualified observation references for cross-integration evidence.
//!
//! [`ObservationId`] is intentionally source-local: collectors may preserve a
//! native event identifier such as `syslog-42`, while other bridges derive
//! deterministic local IDs. Cross-source reasoning must therefore bind that
//! local identifier to the source namespace instead of treating the raw string
//! as globally unique.

use crate::{ObservationEnvelope, ObservationId};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Collision-resistant logical reference to one source-local observation.
///
/// `upstream_origin` is deliberately not part of the identity namespace. It is
/// provenance about how the measurement was produced and may be discovered or
/// enriched later without changing which local observation is being referenced.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceQualifiedObservationRef {
    pub integration_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    pub observation_id: ObservationId,
}

impl SourceQualifiedObservationRef {
    pub fn new(integration_id: impl Into<String>, observation_id: ObservationId) -> Self {
        Self {
            integration_id: integration_id.into(),
            collector_id: None,
            tenant: None,
            observation_id,
        }
    }

    pub fn from_observation(observation: &ObservationEnvelope) -> Self {
        Self {
            integration_id: observation.source.integration_id.clone(),
            collector_id: observation.source.collector_id.clone(),
            tenant: observation.source.tenant.clone(),
            observation_id: observation.observation_id.clone(),
        }
    }

    pub fn matches_observation(&self, observation: &ObservationEnvelope) -> bool {
        self == &Self::from_observation(observation)
    }

    pub fn validate(&self) -> Result<(), SourceQualifiedObservationRefError> {
        require_non_empty("integration_id", &self.integration_id)?;
        require_non_empty("observation_id", self.observation_id.as_str())?;
        if self
            .collector_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(SourceQualifiedObservationRefError::EmptyField("collector_id"));
        }
        if self
            .tenant
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(SourceQualifiedObservationRefError::EmptyField("tenant"));
        }
        Ok(())
    }

    /// Collision-safe deterministic encoding suitable for indexes and evidence.
    pub fn canonical_key(&self) -> Result<String, SourceQualifiedObservationRefError> {
        self.validate()?;
        let mut key = String::from("observation-ref-v1");
        push_required_component(&mut key, &self.integration_id);
        push_optional_component(&mut key, self.collector_id.as_deref());
        push_optional_component(&mut key, self.tenant.as_deref());
        push_required_component(&mut key, self.observation_id.as_str());
        Ok(key)
    }
}

impl ObservationEnvelope {
    /// Source-qualified identity for cross-integration reasoning and evidence.
    pub fn source_qualified_ref(&self) -> SourceQualifiedObservationRef {
        SourceQualifiedObservationRef::from_observation(self)
    }
}

impl fmt::Display for SourceQualifiedObservationRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.canonical_key() {
            Ok(key) => f.write_str(&key),
            Err(_) => write!(
                f,
                "observation-ref-invalid:{}:{}",
                self.integration_id, self.observation_id
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SourceQualifiedObservationRefError {
    #[error("required field `{0}` is empty")]
    EmptyField(&'static str),
}

fn require_non_empty(
    field: &'static str,
    value: &str,
) -> Result<(), SourceQualifiedObservationRefError> {
    if value.trim().is_empty() {
        Err(SourceQualifiedObservationRefError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn push_required_component(output: &mut String, value: &str) {
    output.push('|');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
}

fn push_optional_component(output: &mut String, value: Option<&str>) {
    match value {
        None => output.push_str("|N"),
        Some(value) => {
            output.push_str("|S");
            output.push_str(&value.len().to_string());
            output.push(':');
            output.push_str(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EntityRef, ObservationKind, ObservationLineage, ObservationQuality, ObservationSource,
        ObservationValue,
    };

    fn observation(id: &str, integration: &str, collector: Option<&str>) -> ObservationEnvelope {
        ObservationEnvelope::new(
            ObservationId::new(id),
            100,
            100,
            EntityRef::new("site:lab", "host", "node-1"),
            ObservationKind::Metric,
            "system.cpu.utilization",
            ObservationValue::Unsigned(1),
            ObservationSource {
                integration_id: integration.into(),
                collector_id: collector.map(str::to_string),
                upstream_origin: None,
                measurement_method: "fixture".into(),
                tenant: Some("tenant-a".into()),
            },
            ObservationQuality::observed(1.0),
            ObservationLineage {
                lineage_id: "lineage".into(),
                parent_ids: vec![],
                independence_group: None,
                transforms: vec![],
            },
        )
    }

    #[test]
    fn equal_local_ids_from_different_integrations_are_distinct() {
        let left = observation("obs-1", "source-a", Some("collector"));
        let right = observation("obs-1", "source-b", Some("collector"));
        assert_ne!(left.source_qualified_ref(), right.source_qualified_ref());
    }

    #[test]
    fn collector_and_tenant_are_part_of_the_local_namespace() {
        let left = observation("obs-1", "source", Some("collector-a"));
        let right = observation("obs-1", "source", Some("collector-b"));
        assert_ne!(left.source_qualified_ref(), right.source_qualified_ref());
    }

    #[test]
    fn upstream_origin_does_not_change_observation_identity() {
        let mut left = observation("obs-1", "source", Some("collector"));
        let mut right = left.clone();
        left.source.upstream_origin = Some("origin-a".into());
        right.source.upstream_origin = Some("origin-b".into());
        assert_eq!(left.source_qualified_ref(), right.source_qualified_ref());
    }

    #[test]
    fn canonical_key_is_versioned_and_boundary_safe() {
        let reference = observation("obs|1", "source:a", Some("collector:b"))
            .source_qualified_ref();
        let key = reference.canonical_key().unwrap();
        assert!(key.starts_with("observation-ref-v1|"));
        assert!(key.contains("|8:source:a"));
    }
}
