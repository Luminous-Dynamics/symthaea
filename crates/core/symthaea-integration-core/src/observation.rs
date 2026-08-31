// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical runtime observation envelope for infrastructure integrations.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Current serialized observation schema version.
pub const OBSERVATION_SCHEMA_VERSION: u16 = 1;

/// Caller-supplied, stable identity for an observation.
///
/// The library deliberately does not generate IDs internally: collectors that
/// need replay/deduplication should derive deterministic IDs from their native
/// event identity or explicitly generate them at the boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObservationId(pub String);

impl ObservationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ObservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable reference to an entity in the observed world.
///
/// `namespace` scopes the identifier (for example `aws:123456789012:us-east-1`
/// or `site:dc1`), `kind` gives the semantic entity class, and `id` is the
/// source-stable identifier within that namespace.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityRef {
    pub namespace: String,
    pub kind: String,
    pub id: String,
}

impl EntityRef {
    pub fn new(
        namespace: impl Into<String>,
        kind: impl Into<String>,
        id: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            kind: kind.into(),
            id: id.into(),
        }
    }

    pub fn canonical_key(&self) -> String {
        format!("{}:{}:{}", self.namespace, self.kind, self.id)
    }
}

/// Broad semantic class of an observation. Source-specific detail belongs in
/// `signal` and labels, not in an ever-growing vendor-specific enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationKind {
    Metric,
    Log,
    Trace,
    Event,
    Configuration,
    Topology,
    Flow,
    Health,
    Security,
    Inventory,
    Other(String),
}

/// Portable value representation for v0.1 observations.
///
/// Deliberately avoids `serde_json::Value` so the core contract remains small,
/// deterministic, and usable by non-JSON transports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObservationValue {
    Number {
        value: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
    },
    Integer(i64),
    Unsigned(u64),
    Boolean(bool),
    Text(String),
    Attributes(BTreeMap<String, String>),
}

/// Epistemic/availability state of a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationState {
    /// A current measurement was successfully obtained.
    Observed,
    /// A measurement exists but is older than its accepted freshness window.
    Stale,
    /// The source cannot currently determine the state.
    Unknown,
    /// The target cannot be observed through this source/capability.
    Unobservable,
    /// Independent measurements disagree materially.
    Conflicting,
    /// Only a subset of the expected measurement was obtained.
    Partial,
}

/// Quality metadata kept separate from the measured value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationQuality {
    /// Confidence in this source/mapping for this observation, [0, 1].
    pub source_confidence: f32,
    /// Fraction of expected fields/samples represented, [0, 1].
    pub completeness: f32,
    pub state: ObservationState,
    /// Age relative to the collector's expected freshness model, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub staleness_ms: Option<u64>,
}

impl ObservationQuality {
    pub fn observed(source_confidence: f32) -> Self {
        Self {
            source_confidence,
            completeness: 1.0,
            state: ObservationState::Observed,
            staleness_ms: Some(0),
        }
    }
}

/// Describes where a measurement came from, independently of what it measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSource {
    /// Integration manifest id, e.g. `prometheus` or `aws-cloudwatch`.
    pub integration_id: String,
    /// Concrete collector/process identity when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collector_id: Option<String>,
    /// Upstream raw origin when multiple products re-export the same signal.
    /// Example: several vendors may ultimately derive CPU from `/proc/stat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_origin: Option<String>,
    /// Measurement technique/protocol, e.g. `snmp-poll`, `otlp-metric`, `ebpf`.
    pub measurement_method: String,
    /// Tenant/administrative boundary, if relevant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

/// One transformation applied between the raw source and this observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformStep {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Whether identical input is expected to produce identical output.
    pub deterministic: bool,
}

/// Provenance lineage for correlation-resistant evidence handling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationLineage {
    /// Stable lineage identifier assigned by the source/bridge.
    pub lineage_id: String,
    /// Parent observations when this value is derived/aggregated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_ids: Vec<ObservationId>,
    /// Measurements with the same non-empty group are explicitly declared to
    /// share a measurement lineage. Distinct non-empty groups are treated as
    /// declared independent; missing groups remain epistemically unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub independence_group: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformStep>,
}

/// Conservative relationship between two observations' measurement lineages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineageRelationship {
    SameObservation,
    SharedOrigin,
    DeclaredIndependent,
    Unknown,
}

/// Universal runtime observation passed from integrations into the world model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationEnvelope {
    pub schema_version: u16,
    pub observation_id: ObservationId,
    /// Source-native event/measurement time.
    pub observed_at_unix_ms: u64,
    /// Time the integration admitted the observation into Symthaea.
    pub ingested_at_unix_ms: u64,
    pub entity: EntityRef,
    pub kind: ObservationKind,
    /// Vendor-neutral semantic name when available, otherwise a stable
    /// integration-qualified signal name.
    pub signal: String,
    pub value: ObservationValue,
    pub source: ObservationSource,
    pub quality: ObservationQuality,
    pub lineage: ObservationLineage,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

impl ObservationEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        observation_id: ObservationId,
        observed_at_unix_ms: u64,
        ingested_at_unix_ms: u64,
        entity: EntityRef,
        kind: ObservationKind,
        signal: impl Into<String>,
        value: ObservationValue,
        source: ObservationSource,
        quality: ObservationQuality,
        lineage: ObservationLineage,
    ) -> Self {
        Self {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            observation_id,
            observed_at_unix_ms,
            ingested_at_unix_ms,
            entity,
            kind,
            signal: signal.into(),
            value,
            source,
            quality,
            lineage,
            labels: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.schema_version != OBSERVATION_SCHEMA_VERSION {
            return Err(ObservationValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        require_non_empty("observation_id", self.observation_id.as_str())?;
        require_non_empty("entity.namespace", &self.entity.namespace)?;
        require_non_empty("entity.kind", &self.entity.kind)?;
        require_non_empty("entity.id", &self.entity.id)?;
        require_non_empty("signal", &self.signal)?;
        require_non_empty("source.integration_id", &self.source.integration_id)?;
        require_non_empty("source.measurement_method", &self.source.measurement_method)?;
        require_non_empty("lineage.lineage_id", &self.lineage.lineage_id)?;

        validate_probability("quality.source_confidence", self.quality.source_confidence)?;
        validate_probability("quality.completeness", self.quality.completeness)?;

        if let ObservationValue::Number { value, .. } = &self.value {
            if !value.is_finite() {
                return Err(ObservationValidationError::NonFiniteNumber);
            }
        }

        let mut parents = BTreeSet::new();
        for parent in &self.lineage.parent_ids {
            require_non_empty("lineage.parent_id", parent.as_str())?;
            if parent == &self.observation_id {
                return Err(ObservationValidationError::SelfParent);
            }
            if !parents.insert(parent.clone()) {
                return Err(ObservationValidationError::DuplicateParent(parent.clone()));
            }
        }

        for transform in &self.lineage.transforms {
            require_non_empty("lineage.transform.name", &transform.name)?;
        }

        Ok(())
    }

    /// Assess whether two reports are plausibly independent. Missing lineage
    /// metadata never gets upgraded to independence.
    pub fn lineage_relationship(&self, other: &Self) -> LineageRelationship {
        if self.observation_id == other.observation_id {
            return LineageRelationship::SameObservation;
        }
        if self.lineage.lineage_id == other.lineage.lineage_id {
            return LineageRelationship::SharedOrigin;
        }
        if let (Some(a), Some(b)) = (
            self.source.upstream_origin.as_deref(),
            other.source.upstream_origin.as_deref(),
        ) {
            if a == b {
                return LineageRelationship::SharedOrigin;
            }
        }
        match (
            self.lineage.independence_group.as_deref(),
            other.lineage.independence_group.as_deref(),
        ) {
            (Some(a), Some(b)) if a == b => LineageRelationship::SharedOrigin,
            (Some(_), Some(_)) => LineageRelationship::DeclaredIndependent,
            _ => LineageRelationship::Unknown,
        }
    }

    pub fn age_ms_at(&self, now_unix_ms: u64) -> u64 {
        now_unix_ms.saturating_sub(self.observed_at_unix_ms)
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// One collector result. Batches retain the producing integration identity so
/// adapters cannot silently mix observations from unrelated manifests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationBatch {
    pub integration_id: String,
    pub collected_at_unix_ms: u64,
    pub observations: Vec<ObservationEnvelope>,
}

impl ObservationBatch {
    pub fn validate(&self) -> Result<(), BatchValidationError> {
        if self.integration_id.trim().is_empty() {
            return Err(BatchValidationError::EmptyIntegrationId);
        }

        let mut ids = BTreeSet::new();
        for (index, observation) in self.observations.iter().enumerate() {
            observation
                .validate()
                .map_err(|reason| BatchValidationError::InvalidObservation { index, reason })?;
            if observation.source.integration_id != self.integration_id {
                return Err(BatchValidationError::IntegrationMismatch {
                    index,
                    batch: self.integration_id.clone(),
                    observation: observation.source.integration_id.clone(),
                });
            }
            if !ids.insert(observation.observation_id.clone()) {
                return Err(BatchValidationError::DuplicateObservationId(
                    observation.observation_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ObservationValidationError {
    #[error("unsupported observation schema version {0}")]
    UnsupportedSchemaVersion(u16),
    #[error("required field `{0}` is empty")]
    EmptyField(&'static str),
    #[error("{field} must be finite and within [0,1], got {value}")]
    ProbabilityOutOfRange { field: &'static str, value: f32 },
    #[error("numeric observation value must be finite")]
    NonFiniteNumber,
    #[error("an observation cannot list itself as a lineage parent")]
    SelfParent,
    #[error("duplicate lineage parent {0}")]
    DuplicateParent(ObservationId),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BatchValidationError {
    #[error("batch integration id is empty")]
    EmptyIntegrationId,
    #[error("observation {index} is invalid: {reason}")]
    InvalidObservation {
        index: usize,
        reason: ObservationValidationError,
    },
    #[error(
        "observation {index} belongs to integration `{observation}`, batch declares `{batch}`"
    )]
    IntegrationMismatch {
        index: usize,
        batch: String,
        observation: String,
    },
    #[error("duplicate observation id {0} in batch")]
    DuplicateObservationId(ObservationId),
}

fn require_non_empty(
    field: &'static str,
    value: &str,
) -> Result<(), ObservationValidationError> {
    if value.trim().is_empty() {
        Err(ObservationValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_probability(
    field: &'static str,
    value: f32,
) -> Result<(), ObservationValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ObservationValidationError::ProbabilityOutOfRange { field, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str, integration: &str, group: Option<&str>) -> ObservationEnvelope {
        ObservationEnvelope::new(
            ObservationId::new(id),
            1_000,
            1_010,
            EntityRef::new("site:lab", "host", "node-1"),
            ObservationKind::Metric,
            "system.cpu.utilization",
            ObservationValue::Number {
                value: 0.75,
                unit: Some("1".into()),
            },
            ObservationSource {
                integration_id: integration.into(),
                collector_id: Some("collector-a".into()),
                upstream_origin: None,
                measurement_method: "test".into(),
                tenant: None,
            },
            ObservationQuality::observed(0.9),
            ObservationLineage {
                lineage_id: id.into(),
                parent_ids: vec![],
                independence_group: group.map(str::to_string),
                transforms: vec![],
            },
        )
    }

    #[test]
    fn valid_observation_passes() {
        assert!(sample("obs-1", "test", Some("sensor-a")).validate().is_ok());
    }

    #[test]
    fn non_finite_number_fails_closed() {
        let mut obs = sample("obs-1", "test", None);
        obs.value = ObservationValue::Number {
            value: f64::NAN,
            unit: None,
        };
        assert_eq!(obs.validate(), Err(ObservationValidationError::NonFiniteNumber));
    }

    #[test]
    fn missing_independence_metadata_is_unknown() {
        let a = sample("a", "one", None);
        let b = sample("b", "two", None);
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn distinct_declared_groups_are_independent() {
        let a = sample("a", "one", Some("physical-bmc"));
        let b = sample("b", "two", Some("kernel-procfs"));
        assert_eq!(
            a.lineage_relationship(&b),
            LineageRelationship::DeclaredIndependent
        );
    }

    #[test]
    fn duplicate_batch_ids_fail() {
        let obs = sample("same", "test", None);
        let batch = ObservationBatch {
            integration_id: "test".into(),
            collected_at_unix_ms: 2_000,
            observations: vec![obs.clone(), obs],
        };
        assert!(matches!(
            batch.validate(),
            Err(BatchValidationError::DuplicateObservationId(_))
        ));
    }

    #[test]
    fn serde_roundtrip_preserves_lineage() {
        let obs = sample("obs-1", "test", Some("sensor-a"));
        let json = serde_json::to_string(&obs).unwrap();
        let restored: ObservationEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, obs);
    }
}
