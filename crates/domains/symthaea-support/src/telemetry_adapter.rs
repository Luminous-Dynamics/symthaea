// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy-aware, provider-neutral external telemetry -> `SystemObservationV1` bridge.
//!
//! This layer normalizes metrics/traces/events from external collectors. It does
//! not infer canonical graph identity from telemetry resource attributes and it
//! does not treat telemetry payload as truth merely because a collector emitted it.

use crate::scrubber::PrivacyScrubber;
use crate::system_state::{
    EntityId, ObservationClockV1, ObservationId, ObservationProvenanceV1,
    ObservationSourceKindV1, StateValueV1, SystemObservationV1, SystemStateGraphError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelemetrySignalKindV1 {
    Metric,
    TraceSpan,
    Event,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelemetrySchemaStabilityV1 {
    Development,
    Alpha,
    Beta,
    ReleaseCandidate,
    Stable,
    Mixed,
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySchemaIdentityV1 {
    /// Schema/convention family such as `opentelemetry-semconv` or `openconfig`.
    pub family: String,
    pub version: Option<String>,
    pub stability: TelemetrySchemaStabilityV1,
}

impl TelemetrySchemaIdentityV1 {
    pub fn validate(&self) -> Result<(), TelemetryAdapterErrorV1> {
        require_nonempty(&self.family, "schema family")?;
        if self.version.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(TelemetryAdapterErrorV1::EmptyField("schema version"));
        }
        Ok(())
    }

    fn provenance_version(&self) -> String {
        match &self.version {
            Some(version) => format!("{}:{}:{:?}", self.family, version, self.stability),
            None => format!("{}:unversioned:{:?}", self.family, self.stability),
        }
    }
}

/// Provider-neutral normalized telemetry record before support privacy policy is applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalTelemetryRecordV1 {
    /// Upstream record/point/span identity. It is hashed into observation identity
    /// but is not retained as a graph fact by default.
    pub record_id: String,
    pub kind: TelemetrySignalKindV1,
    pub name: String,
    pub schema: TelemetrySchemaIdentityV1,
    pub event_time_unix_ms: Option<u64>,
    pub clock_uncertainty_ms: Option<u64>,
    /// Useful for spans or timed events.
    pub duration_ns: Option<u64>,
    /// Metric point/value where representable by the state scalar vocabulary.
    pub metric_value: Option<StateValueV1>,
    pub unit: Option<String>,
    pub status: Option<String>,
    #[serde(default)]
    pub resource_attributes: BTreeMap<String, StateValueV1>,
    #[serde(default)]
    pub attributes: BTreeMap<String, StateValueV1>,
}

impl ExternalTelemetryRecordV1 {
    pub fn validate(&self) -> Result<(), TelemetryAdapterErrorV1> {
        require_nonempty(&self.record_id, "record id")?;
        require_nonempty(&self.name, "signal name")?;
        self.schema.validate()?;
        if self.unit.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(TelemetryAdapterErrorV1::EmptyField("metric unit"));
        }
        if self.status.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(TelemetryAdapterErrorV1::EmptyField("status"));
        }
        if let Some(value) = &self.metric_value {
            validate_state_value(value, "metric value")?;
        }
        for (key, value) in self
            .resource_attributes
            .iter()
            .chain(self.attributes.iter())
        {
            require_nonempty(key, "telemetry attribute key")?;
            validate_state_value(value, "telemetry attribute value")?;
        }
        Ok(())
    }
}

/// Retention is deny-by-default for arbitrary attributes. Numeric metric values
/// are useful enough to retain by default; textual metric values are not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryRetentionPolicyV1 {
    #[serde(default)]
    pub resource_attribute_allowlist: BTreeSet<String>,
    #[serde(default)]
    pub signal_attribute_allowlist: BTreeSet<String>,
    pub retain_numeric_metric_value: bool,
    pub retain_text_metric_value: bool,
    pub retain_status: bool,
}

impl Default for TelemetryRetentionPolicyV1 {
    fn default() -> Self {
        Self {
            resource_attribute_allowlist: BTreeSet::new(),
            signal_attribute_allowlist: BTreeSet::new(),
            retain_numeric_metric_value: true,
            retain_text_metric_value: false,
            retain_status: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryObservationAdapterConfigV1 {
    pub source_id: String,
    pub collector: String,
    pub collector_version: Option<String>,
    pub artifact_digest: Option<String>,
    pub max_age_ms: Option<u64>,
    /// Confidence in normalization mechanics, not objective truth of telemetry.
    pub normalization_confidence: f32,
}

pub struct TelemetryObservationAdapterV1 {
    config: TelemetryObservationAdapterConfigV1,
    policy: TelemetryRetentionPolicyV1,
    scrubber: PrivacyScrubber,
}

impl TelemetryObservationAdapterV1 {
    pub fn new(
        config: TelemetryObservationAdapterConfigV1,
        policy: TelemetryRetentionPolicyV1,
    ) -> Result<Self, TelemetryAdapterErrorV1> {
        require_nonempty(&config.source_id, "source id")?;
        require_nonempty(&config.collector, "collector")?;
        if config
            .collector_version
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(TelemetryAdapterErrorV1::EmptyField("collector version"));
        }
        if config
            .artifact_digest
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(TelemetryAdapterErrorV1::EmptyField("artifact digest"));
        }
        if !config.normalization_confidence.is_finite()
            || !(0.0..=1.0).contains(&config.normalization_confidence)
        {
            return Err(TelemetryAdapterErrorV1::InvalidConfidence(
                config.normalization_confidence,
            ));
        }
        Ok(Self {
            config,
            policy,
            scrubber: PrivacyScrubber::new(),
        })
    }

    /// Normalize telemetry for an explicitly resolved canonical graph subject.
    pub fn normalize(
        &self,
        subject: EntityId,
        record: &ExternalTelemetryRecordV1,
        observed_at_unix_ms: u64,
        ingested_at_unix_ms: Option<u64>,
    ) -> Result<SystemObservationV1, TelemetryAdapterErrorV1> {
        if subject.0.trim().is_empty() {
            return Err(TelemetryAdapterErrorV1::EmptyField("subject"));
        }
        record.validate()?;

        let mut facts = BTreeMap::new();
        facts.insert(
            "telemetry.signal.kind".into(),
            StateValueV1::Text(signal_kind_name(record.kind).into()),
        );
        facts.insert(
            "telemetry.signal.name".into(),
            StateValueV1::Text(self.scrubber.scrub(&record.name).scrubbed_text),
        );
        facts.insert(
            "telemetry.schema.family".into(),
            StateValueV1::Text(record.schema.family.clone()),
        );
        facts.insert(
            "telemetry.schema.stability".into(),
            StateValueV1::Text(format!("{:?}", record.schema.stability).to_ascii_lowercase()),
        );
        if let Some(version) = &record.schema.version {
            facts.insert(
                "telemetry.schema.version".into(),
                StateValueV1::Text(version.clone()),
            );
        }
        if let Some(duration_ns) = record.duration_ns {
            facts.insert("telemetry.duration_ns".into(), StateValueV1::U64(duration_ns));
        }
        if let Some(unit) = &record.unit {
            facts.insert(
                "telemetry.metric.unit".into(),
                StateValueV1::Text(self.scrubber.scrub(unit).scrubbed_text),
            );
        }
        if self.policy.retain_status {
            if let Some(status) = &record.status {
                facts.insert(
                    "telemetry.status".into(),
                    StateValueV1::Text(self.scrubber.scrub(status).scrubbed_text),
                );
            }
        }
        if let Some(value) = &record.metric_value {
            if let Some(retained) = self.retain_metric_value(value) {
                facts.insert("telemetry.metric.value".into(), retained);
            }
        }

        self.retain_attributes(
            "telemetry.resource",
            &record.resource_attributes,
            &self.policy.resource_attribute_allowlist,
            &mut facts,
        );
        self.retain_attributes(
            "telemetry.attribute",
            &record.attributes,
            &self.policy.signal_attribute_allowlist,
            &mut facts,
        );

        let clock = ObservationClockV1 {
            event_time_unix_ms: record.event_time_unix_ms,
            observed_at_unix_ms,
            ingested_at_unix_ms,
            max_age_ms: self.config.max_age_ms,
            clock_uncertainty_ms: record.clock_uncertainty_ms,
        };
        let provenance = ObservationProvenanceV1 {
            source_id: self.config.source_id.clone(),
            source_kind: observation_source_kind(record.kind),
            collector: self.config.collector.clone(),
            collector_version: self.config.collector_version.clone(),
            schema_version: Some(record.schema.provenance_version()),
            artifact_digest: self.config.artifact_digest.clone(),
        };
        let id = ObservationId(self.observation_id(
            &subject,
            record,
            &clock,
            &provenance,
            &facts,
        )?);
        let observation = SystemObservationV1 {
            id,
            subject,
            provenance,
            clock,
            confidence: self.config.normalization_confidence,
            facts,
        };
        observation.validate()?;
        Ok(observation)
    }

    fn retain_metric_value(&self, value: &StateValueV1) -> Option<StateValueV1> {
        match value {
            StateValueV1::I64(_)
            | StateValueV1::U64(_)
            | StateValueV1::F64(_)
            | StateValueV1::Bool(_)
                if self.policy.retain_numeric_metric_value => Some(value.clone()),
            StateValueV1::Text(text) if self.policy.retain_text_metric_value => {
                Some(StateValueV1::Text(self.scrubber.scrub(text).scrubbed_text))
            }
            StateValueV1::TextList(values) if self.policy.retain_text_metric_value => {
                Some(StateValueV1::TextList(
                    values
                        .iter()
                        .map(|value| self.scrubber.scrub(value).scrubbed_text)
                        .collect(),
                ))
            }
            _ => None,
        }
    }

    fn retain_attributes(
        &self,
        prefix: &str,
        source: &BTreeMap<String, StateValueV1>,
        allowlist: &BTreeSet<String>,
        facts: &mut BTreeMap<String, StateValueV1>,
    ) {
        for key in allowlist {
            let Some(value) = source.get(key) else {
                continue;
            };
            facts.insert(
                format!("{prefix}.{}", sanitize_key(key)),
                self.scrub_state_value(value),
            );
        }
    }

    fn scrub_state_value(&self, value: &StateValueV1) -> StateValueV1 {
        match value {
            StateValueV1::Text(text) => {
                StateValueV1::Text(self.scrubber.scrub(text).scrubbed_text)
            }
            StateValueV1::TextList(values) => StateValueV1::TextList(
                values
                    .iter()
                    .map(|value| self.scrubber.scrub(value).scrubbed_text)
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    fn observation_id(
        &self,
        subject: &EntityId,
        record: &ExternalTelemetryRecordV1,
        clock: &ObservationClockV1,
        provenance: &ObservationProvenanceV1,
        facts: &BTreeMap<String, StateValueV1>,
    ) -> Result<String, TelemetryAdapterErrorV1> {
        let normalized = serde_json::to_vec(&(
            "symthaea-external-telemetry-observation-v1",
            subject,
            &record.record_id,
            record.kind,
            &record.schema,
            clock,
            provenance,
            facts,
        ))
        .map_err(|err| TelemetryAdapterErrorV1::Serialization(err.to_string()))?;
        Ok(format!(
            "telemetry:{}",
            blake3::hash(&normalized).to_hex()
        ))
    }
}

fn signal_kind_name(kind: TelemetrySignalKindV1) -> &'static str {
    match kind {
        TelemetrySignalKindV1::Metric => "metric",
        TelemetrySignalKindV1::TraceSpan => "trace-span",
        TelemetrySignalKindV1::Event => "event",
    }
}

fn observation_source_kind(kind: TelemetrySignalKindV1) -> ObservationSourceKindV1 {
    match kind {
        TelemetrySignalKindV1::Metric => ObservationSourceKindV1::Metric,
        TelemetrySignalKindV1::TraceSpan => ObservationSourceKindV1::Trace,
        TelemetrySignalKindV1::Event => {
            ObservationSourceKindV1::Other("external-telemetry-event".into())
        }
    }
}

fn sanitize_key(key: &str) -> String {
    let normalized: String = key
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if normalized.is_empty() {
        "unnamed".into()
    } else {
        normalized
    }
}

fn validate_state_value(
    value: &StateValueV1,
    field: &'static str,
) -> Result<(), TelemetryAdapterErrorV1> {
    if let StateValueV1::F64(number) = value {
        if !number.is_finite() {
            return Err(TelemetryAdapterErrorV1::NonFiniteValue {
                field,
                value: *number,
            });
        }
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), TelemetryAdapterErrorV1> {
    if value.trim().is_empty() {
        Err(TelemetryAdapterErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum TelemetryAdapterErrorV1 {
    EmptyField(&'static str),
    InvalidConfidence(f32),
    NonFiniteValue { field: &'static str, value: f64 },
    Serialization(String),
    State(SystemStateGraphError),
}

impl fmt::Display for TelemetryAdapterErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty telemetry adapter field {field}"),
            Self::InvalidConfidence(value) => write!(f, "invalid normalization confidence {value}"),
            Self::NonFiniteValue { field, value } => {
                write!(f, "non-finite telemetry {field}: {value}")
            }
            Self::Serialization(message) => {
                write!(f, "telemetry identity serialization failed: {message}")
            }
            Self::State(err) => write!(f, "invalid telemetry observation: {err}"),
        }
    }
}

impl Error for TelemetryAdapterErrorV1 {}

impl From<SystemStateGraphError> for TelemetryAdapterErrorV1 {
    fn from(value: SystemStateGraphError) -> Self {
        Self::State(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric_record() -> ExternalTelemetryRecordV1 {
        ExternalTelemetryRecordV1 {
            record_id: "point-1".into(),
            kind: TelemetrySignalKindV1::Metric,
            name: "system.memory.usage".into(),
            schema: TelemetrySchemaIdentityV1 {
                family: "opentelemetry-semconv".into(),
                version: Some("1.44.0".into()),
                stability: TelemetrySchemaStabilityV1::Mixed,
            },
            event_time_unix_ms: Some(1_700_000_000_000),
            clock_uncertainty_ms: Some(5),
            duration_ns: None,
            metric_value: Some(StateValueV1::U64(4096)),
            unit: Some("By".into()),
            status: None,
            resource_attributes: BTreeMap::from([
                ("host.name".into(), StateValueV1::Text("server-01".into())),
                (
                    "host.ip".into(),
                    StateValueV1::Text("192.168.1.44".into()),
                ),
            ]),
            attributes: BTreeMap::from([(
                "user.email".into(),
                StateValueV1::Text("admin@example.com".into()),
            )]),
        }
    }

    fn adapter(policy: TelemetryRetentionPolicyV1) -> TelemetryObservationAdapterV1 {
        TelemetryObservationAdapterV1::new(
            TelemetryObservationAdapterConfigV1 {
                source_id: "otel:collector-a".into(),
                collector: "symthaea-otel-adapter-v1".into(),
                collector_version: Some("1".into()),
                artifact_digest: None,
                max_age_ms: Some(30_000),
                normalization_confidence: 1.0,
            },
            policy,
        )
        .unwrap()
    }

    #[test]
    fn default_policy_keeps_numeric_metric_but_denies_arbitrary_attributes() {
        let obs = adapter(Default::default())
            .normalize(
                EntityId("host:canonical-01".into()),
                &metric_record(),
                1_700_000_000_100,
                None,
            )
            .unwrap();
        assert_eq!(
            obs.facts["telemetry.metric.value"],
            StateValueV1::U64(4096)
        );
        assert!(!obs
            .facts
            .keys()
            .any(|key| key.starts_with("telemetry.resource.")));
        assert!(!obs
            .facts
            .keys()
            .any(|key| key.starts_with("telemetry.attribute.")));
        assert_eq!(obs.subject, EntityId("host:canonical-01".into()));
    }

    #[test]
    fn allowlisted_sensitive_text_is_scrubbed_and_does_not_rebind_subject() {
        let mut policy = TelemetryRetentionPolicyV1::default();
        policy.resource_attribute_allowlist.insert("host.ip".into());
        policy.signal_attribute_allowlist.insert("user.email".into());
        let obs = adapter(policy)
            .normalize(
                EntityId("host:canonical-01".into()),
                &metric_record(),
                1_700_000_000_100,
                None,
            )
            .unwrap();
        let ip = match &obs.facts["telemetry.resource.host.ip"] {
            StateValueV1::Text(value) => value,
            _ => panic!("IP must be text"),
        };
        let email = match &obs.facts["telemetry.attribute.user.email"] {
            StateValueV1::Text(value) => value,
            _ => panic!("email must be text"),
        };
        assert!(!ip.contains("192.168.1.44"));
        assert!(!email.contains("admin@example.com"));
        assert_eq!(obs.subject, EntityId("host:canonical-01".into()));
    }

    #[test]
    fn exact_normalized_replay_is_idempotent_but_reobservation_is_distinct() {
        let adapter = adapter(Default::default());
        let record = metric_record();
        let first = adapter
            .normalize(
                EntityId("host:a".into()),
                &record,
                1_700_000_000_100,
                Some(1_700_000_000_120),
            )
            .unwrap();
        let replay = adapter
            .normalize(
                EntityId("host:a".into()),
                &record,
                1_700_000_000_100,
                Some(1_700_000_000_120),
            )
            .unwrap();
        let recollected = adapter
            .normalize(
                EntityId("host:a".into()),
                &record,
                1_700_000_000_200,
                Some(1_700_000_000_220),
            )
            .unwrap();
        assert_eq!(first.id, replay.id);
        assert_ne!(first.id, recollected.id);
    }

    #[test]
    fn schema_identity_and_stability_are_retained() {
        let obs = adapter(Default::default())
            .normalize(
                EntityId("host:a".into()),
                &metric_record(),
                1_700_000_000_100,
                None,
            )
            .unwrap();
        assert_eq!(
            obs.facts["telemetry.schema.version"],
            StateValueV1::Text("1.44.0".into())
        );
        assert_eq!(
            obs.facts["telemetry.schema.stability"],
            StateValueV1::Text("mixed".into())
        );
    }

    #[test]
    fn trace_maps_to_trace_provenance() {
        let mut record = metric_record();
        record.kind = TelemetrySignalKindV1::TraceSpan;
        record.record_id = "span-1".into();
        record.name = "GET /customers/192.168.1.44".into();
        record.metric_value = None;
        record.duration_ns = Some(2_000_000);
        let obs = adapter(Default::default())
            .normalize(EntityId("service:web".into()), &record, 100, None)
            .unwrap();
        assert_eq!(obs.provenance.source_kind, ObservationSourceKindV1::Trace);
        let name = match &obs.facts["telemetry.signal.name"] {
            StateValueV1::Text(value) => value,
            _ => panic!("name must be text"),
        };
        assert!(!name.contains("192.168.1.44"));
    }

    #[test]
    fn invalid_record_identity_is_rejected() {
        let mut record = metric_record();
        record.record_id.clear();
        assert!(adapter(Default::default())
            .normalize(EntityId("host:a".into()), &record, 100, None)
            .is_err());
    }

    #[test]
    fn non_finite_metric_value_is_rejected_before_identity_serialization() {
        let mut record = metric_record();
        record.metric_value = Some(StateValueV1::F64(f64::NAN));
        let err = adapter(Default::default())
            .normalize(EntityId("host:a".into()), &record, 100, None)
            .unwrap_err();
        assert!(matches!(
            err,
            TelemetryAdapterErrorV1::NonFiniteValue {
                field: "metric value",
                ..
            }
        ));
    }

    #[test]
    fn non_finite_attribute_is_rejected_even_when_retention_denies_it() {
        let mut record = metric_record();
        record
            .attributes
            .insert("bad.value".into(), StateValueV1::F64(f64::INFINITY));
        let err = adapter(Default::default())
            .normalize(EntityId("host:a".into()), &record, 100, None)
            .unwrap_err();
        assert!(matches!(
            err,
            TelemetryAdapterErrorV1::NonFiniteValue {
                field: "telemetry attribute value",
                ..
            }
        ));
    }

    #[test]
    fn empty_optional_provenance_metadata_is_rejected() {
        let err = TelemetryObservationAdapterV1::new(
            TelemetryObservationAdapterConfigV1 {
                source_id: "otel:collector-a".into(),
                collector: "symthaea-otel-adapter-v1".into(),
                collector_version: Some(" ".into()),
                artifact_digest: None,
                max_age_ms: Some(30_000),
                normalization_confidence: 1.0,
            },
            Default::default(),
        )
        .err()
        .expect("empty collector version must fail");
        assert!(matches!(
            err,
            TelemetryAdapterErrorV1::EmptyField("collector version")
        ));

        let err = TelemetryObservationAdapterV1::new(
            TelemetryObservationAdapterConfigV1 {
                source_id: "otel:collector-a".into(),
                collector: "symthaea-otel-adapter-v1".into(),
                collector_version: Some("1".into()),
                artifact_digest: Some(" ".into()),
                max_age_ms: Some(30_000),
                normalization_confidence: 1.0,
            },
            Default::default(),
        )
        .err()
        .expect("empty artifact digest must fail");
        assert!(matches!(
            err,
            TelemetryAdapterErrorV1::EmptyField("artifact digest")
        ));
    }
}
