// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical runtime observation envelope for infrastructure integrations.

use crate::observation_reference::SourceQualifiedObservationRef;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Current serialized observation schema version.
pub const OBSERVATION_SCHEMA_VERSION: u16 = 1;

/// Caller-supplied, stable **source-local** identity for an observation.
///
/// The library deliberately does not generate IDs internally: collectors that
/// need replay/deduplication should derive deterministic IDs from their native
/// event identity or explicitly generate them at the boundary. Cross-source
/// reasoning must use [`SourceQualifiedObservationRef`] rather than assuming the
/// raw ID string is globally unique.
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

    /// Collision-safe deterministic encoding for hashes, indexes and evidence.
    ///
    /// Length prefixes make component boundaries unambiguous even when vendor-
    /// controlled namespace/kind/id strings themselves contain `:`, `|`, or
    /// other separators. The version prefix makes future encoding migrations
    /// explicit instead of silently changing persisted deterministic IDs.
    pub fn canonical_key(&self) -> String {
        let mut key = String::from("entity-v1");
        push_len_prefixed(&mut key, &self.namespace);
        push_len_prefixed(&mut key, &self.kind);
        push_len_prefixed(&mut key, &self.id);
        key
    }
}

fn push_len_prefixed(output: &mut String, value: &str) {
    output.push('|');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
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
    /// Stable source-local lineage identifier assigned by the source/bridge.
    /// It is meaningful only together with the source namespace: integration,
    /// collector, tenant, measurement method, and explicit upstream origin when
    /// present. Equal local strings outside that namespace do not prove shared
    /// lineage.
    pub lineage_id: String,
    /// Source-local parent observations when this value is derived/aggregated.
    /// The enclosing observation source supplies their namespace in v0.1.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_ids: Vec<ObservationId>,
    /// Measurements with the same non-empty group are conservatively declared
    /// to share a measurement lineage. Different group labels are descriptive
    /// only and do not prove independence without a separately qualified
    /// independence authority.
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
    /// Reserved for a future relationship established by an explicitly
    /// qualified independence authority. Adapter-local group labels never
    /// produce this relationship on their own.
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

        for (field, value) in [
            ("source.collector_id", self.source.collector_id.as_deref()),
            ("source.upstream_origin", self.source.upstream_origin.as_deref()),
            ("source.tenant", self.source.tenant.as_deref()),
            (
                "lineage.independence_group",
                self.lineage.independence_group.as_deref(),
            ),
        ] {
            if let Some(value) = value {
                require_non_empty(field, value)?;
            }
        }

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

        debug_assert!(self.source_qualified_ref().validate().is_ok());
        Ok(())
    }

    /// Assess whether two reports share known lineage. Positive independence is
    /// never inferred from different adapter-supplied group labels. This method
    /// also stays conservative for invalid/unadmitted envelopes: empty optional
    /// provenance strings never become shared-origin evidence.
    pub fn lineage_relationship(&self, other: &Self) -> LineageRelationship {
        let self_ref = self.source_qualified_ref();
        let other_ref = other.source_qualified_ref();
        if self_ref.validate().is_ok() && other_ref.validate().is_ok() && self_ref == other_ref {
            return LineageRelationship::SameObservation;
        }

        if !self.lineage.lineage_id.trim().is_empty()
            && !other.lineage.lineage_id.trim().is_empty()
            && !self.source.integration_id.trim().is_empty()
            && !other.source.integration_id.trim().is_empty()
            && !self.source.measurement_method.trim().is_empty()
            && !other.source.measurement_method.trim().is_empty()
            && self.lineage.lineage_id == other.lineage.lineage_id
            && self.source.integration_id == other.source.integration_id
            && self.source.collector_id == other.source.collector_id
            && self.source.tenant == other.source.tenant
            && self.source.measurement_method == other.source.measurement_method
            && self.source.upstream_origin == other.source.upstream_origin
        {
            return LineageRelationship::SharedOrigin;
        }

        if let (Some(a), Some(b)) = (
            self.source
                .upstream_origin
                .as_deref()
                .filter(|value| !value.trim().is_empty()),
            other
                .source
                .upstream_origin
                .as_deref()
                .filter(|value| !value.trim().is_empty()),
        ) {
            if a == b {
                return LineageRelationship::SharedOrigin;
            }
        }

        match (
            self.lineage
                .independence_group
                .as_deref()
                .filter(|value| !value.trim().is_empty()),
            other
                .lineage
                .independence_group
                .as_deref()
                .filter(|value| !value.trim().is_empty()),
        ) {
            (Some(a), Some(b)) if a == b => LineageRelationship::SharedOrigin,
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
    /// Local integration time at which this batch was completed/admitted. Source
    /// event clocks remain independent and are not ordered against this field.
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
            if observation.ingested_at_unix_ms > self.collected_at_unix_ms {
                return Err(BatchValidationError::FutureIngestionTimestamp {
                    index,
                    observation_id: observation.observation_id.clone(),
                    ingested_at_unix_ms: observation.ingested_at_unix_ms,
                    collected_at_unix_ms: self.collected_at_unix_ms,
                });
            }
            let reference = observation.source_qualified_ref();
            if !ids.insert(reference.clone()) {
                return Err(BatchValidationError::DuplicateObservationReference(reference));
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
    #[error(
        "observation {index} `{observation_id}` was ingested at {ingested_at_unix_ms} after batch collection {collected_at_unix_ms}"
    )]
    FutureIngestionTimestamp {
        index: usize,
        observation_id: ObservationId,
        ingested_at_unix_ms: u64,
        collected_at_unix_ms: u64,
    },
    #[error("duplicate source-qualified observation reference {0} in batch")]
    DuplicateObservationReference(SourceQualifiedObservationRef),
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
    fn entity_canonical_key_is_length_prefixed_and_versioned() {
        let entity = EntityRef::new("site:lab", "host", "node-1");
        assert_eq!(
            entity.canonical_key(),
            "entity-v1|8:site:lab|4:host|6:node-1"
        );
    }

    #[test]
    fn entity_canonical_key_cannot_collide_on_separator_placement() {
        let left = EntityRef::new("a", "b:c", "d");
        let right = EntityRef::new("a:b", "c", "d");
        assert_ne!(left, right);
        assert_ne!(left.canonical_key(), right.canonical_key());
    }

    #[test]
    fn valid_observation_passes() {
        let observation = sample("obs-1", "test", Some("sensor-a"));
        assert!(observation.validate().is_ok());
        assert!(observation.source_qualified_ref().validate().is_ok());
    }

    #[test]
    fn optional_provenance_fields_cannot_be_present_but_empty() {
        let mutations: [fn(&mut ObservationEnvelope); 4] = [
            |observation| observation.source.collector_id = Some("".into()),
            |observation| observation.source.upstream_origin = Some(" ".into()),
            |observation| observation.source.tenant = Some("".into()),
            |observation| observation.lineage.independence_group = Some("\t".into()),
        ];
        for mutate in mutations {
            let mut observation = sample("obs-1", "test", None);
            mutate(&mut observation);
            assert!(matches!(
                observation.validate(),
                Err(ObservationValidationError::EmptyField(_))
            ));
        }
    }

    #[test]
    fn invalid_empty_provenance_cannot_create_shared_origin() {
        let mut a = sample("a", "one", None);
        let mut b = sample("b", "two", None);
        a.source.upstream_origin = Some("".into());
        b.source.upstream_origin = Some("".into());
        a.lineage.independence_group = Some("".into());
        b.lineage.independence_group = Some("".into());
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn source_clock_ahead_of_ingestion_is_not_structurally_rejected() {
        let mut obs = sample("clock-skew", "test", None);
        obs.observed_at_unix_ms = 1_020;
        obs.ingested_at_unix_ms = 1_010;
        assert!(obs.validate().is_ok());

        let batch = ObservationBatch {
            integration_id: "test".into(),
            collected_at_unix_ms: 1_010,
            observations: vec![obs],
        };
        assert!(batch.validate().is_ok());
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
    fn equal_raw_observation_ids_across_integrations_are_not_same_observation() {
        let a = sample("same", "one", None);
        let b = sample("same", "two", None);
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn equal_raw_observation_ids_across_collectors_are_not_same_observation() {
        let a = sample("same", "one", None);
        let mut b = a.clone();
        b.source.collector_id = Some("collector-b".into());
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn equal_source_local_lineage_ids_across_integrations_do_not_collapse() {
        let mut a = sample("a", "one", None);
        let mut b = sample("b", "two", None);
        a.lineage.lineage_id = "local-lineage".into();
        b.lineage.lineage_id = "local-lineage".into();
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn equal_lineage_ids_with_same_source_context_share_origin() {
        let mut a = sample("a", "same", None);
        let mut b = sample("b", "same", None);
        a.lineage.lineage_id = "same-lineage".into();
        b.lineage.lineage_id = "same-lineage".into();
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::SharedOrigin);
    }

    #[test]
    fn equal_lineage_ids_with_different_measurement_methods_do_not_collapse() {
        let mut a = sample("a", "same", None);
        let mut b = sample("b", "same", None);
        a.lineage.lineage_id = "same-lineage".into();
        b.lineage.lineage_id = "same-lineage".into();
        b.source.measurement_method = "other-method".into();
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn equal_lineage_ids_with_different_upstream_origins_do_not_collapse() {
        let mut a = sample("a", "same", None);
        let mut b = sample("b", "same", None);
        a.lineage.lineage_id = "same-lineage".into();
        b.lineage.lineage_id = "same-lineage".into();
        a.source.upstream_origin = Some("upstream-a".into());
        b.source.upstream_origin = Some("upstream-b".into());
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn reused_log_like_id_with_distinct_origins_is_not_shared_lineage() {
        let mut a = sample("syslog-42", "symthaea-logparse", None);
        let mut b = a.clone();
        a.source.collector_id = None;
        b.source.collector_id = None;
        a.source.measurement_method = "syslog".into();
        b.source.measurement_method = "syslog".into();
        a.source.upstream_origin = Some("log-stream:host-a:sshd".into());
        b.source.upstream_origin = Some("log-stream:host-b:sshd".into());
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn equal_lineage_ids_across_tenants_do_not_collapse() {
        let mut a = sample("a", "same", None);
        let mut b = sample("b", "same", None);
        a.lineage.lineage_id = "same-lineage".into();
        b.lineage.lineage_id = "same-lineage".into();
        a.source.tenant = Some("tenant-a".into());
        b.source.tenant = Some("tenant-b".into());
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn distinct_self_declared_groups_remain_epistemically_unknown() {
        let a = sample("a", "one", Some("physical-bmc"));
        let b = sample("b", "two", Some("kernel-procfs"));
        assert_eq!(a.lineage_relationship(&b), LineageRelationship::Unknown);
    }

    #[test]
    fn same_declared_group_is_conservatively_shared_origin() {
        let a = sample("a", "one", Some("same-origin"));
        let b = sample("b", "two", Some("same-origin"));
        assert_eq!(
            a.lineage_relationship(&b),
            LineageRelationship::SharedOrigin
        );
    }

    #[test]
    fn duplicate_batch_references_fail() {
        let obs = sample("same", "test", None);
        let batch = ObservationBatch {
            integration_id: "test".into(),
            collected_at_unix_ms: 2_000,
            observations: vec![obs.clone(), obs],
        };
        assert!(matches!(
            batch.validate(),
            Err(BatchValidationError::DuplicateObservationReference(_))
        ));
    }

    #[test]
    fn equal_local_ids_from_distinct_collectors_are_allowed_in_one_batch() {
        let a = sample("same", "test", None);
        let mut b = a.clone();
        b.source.collector_id = Some("collector-b".into());
        b.lineage.lineage_id = "collector-b-lineage".into();
        let batch = ObservationBatch {
            integration_id: "test".into(),
            collected_at_unix_ms: 2_000,
            observations: vec![a, b],
        };
        assert!(batch.validate().is_ok());
    }

    #[test]
    fn batch_rejects_ingestion_after_its_collection_boundary() {
        let mut obs = sample("future-ingest", "test", None);
        obs.ingested_at_unix_ms = 2_001;
        let batch = ObservationBatch {
            integration_id: "test".into(),
            collected_at_unix_ms: 2_000,
            observations: vec![obs],
        };
        assert!(matches!(
            batch.validate(),
            Err(BatchValidationError::FutureIngestionTimestamp { .. })
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
