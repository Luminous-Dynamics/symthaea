// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy-aware adapter from `symthaea-logparse::LogEvent` to evidence-bound
//! `SystemObservationV1` records.
//!
//! The adapter requires an explicit graph subject. It does not infer globally
//! stable host identity from an arbitrary hostname string, and by default it
//! retains only normalized metadata rather than raw message bodies/fields.

use crate::scrubber::PrivacyScrubber;
use crate::system_state::{
    EntityId, ObservationClockV1, ObservationId, ObservationProvenanceV1,
    ObservationSourceKindV1, StateValueV1, SystemObservationV1, SystemStateGraphError,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use symthaea_logparse::{LogEvent, Severity, Source};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogObservationPolicyV1 {
    /// Raw message text is omitted by default. If retained, it is privacy-scrubbed.
    pub retain_message: bool,
    /// Structured source fields are omitted by default. If retained, values are scrubbed.
    pub retain_fields: bool,
    /// Currentness validity horizon for the normalized observation.
    pub max_age_ms: Option<u64>,
    /// Adapter confidence in the syntactic normalization, not confidence that the
    /// source event's message is objectively true.
    pub normalization_confidence: f32,
}

impl Default for LogObservationPolicyV1 {
    fn default() -> Self {
        Self {
            retain_message: false,
            retain_fields: false,
            max_age_ms: None,
            normalization_confidence: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogObservationAdapterConfigV1 {
    /// Stable identity of the source stream, e.g. `host-27:eventlog:System`.
    pub source_id: String,
    /// Adapter/collector implementation identity.
    pub collector: String,
    pub collector_version: Option<String>,
    pub schema_version: Option<String>,
    /// Optional immutable source artifact digest for offline corpora.
    pub artifact_digest: Option<String>,
}

pub struct LogObservationAdapterV1 {
    config: LogObservationAdapterConfigV1,
    policy: LogObservationPolicyV1,
    scrubber: PrivacyScrubber,
}

impl LogObservationAdapterV1 {
    pub fn new(
        config: LogObservationAdapterConfigV1,
        policy: LogObservationPolicyV1,
    ) -> Result<Self, LogObservationAdapterError> {
        if config.source_id.trim().is_empty() {
            return Err(LogObservationAdapterError::EmptyConfig("source_id"));
        }
        if config.collector.trim().is_empty() {
            return Err(LogObservationAdapterError::EmptyConfig("collector"));
        }
        if !policy.normalization_confidence.is_finite()
            || !(0.0..=1.0).contains(&policy.normalization_confidence)
        {
            return Err(LogObservationAdapterError::InvalidConfidence(
                policy.normalization_confidence,
            ));
        }
        Ok(Self {
            config,
            policy,
            scrubber: PrivacyScrubber::new(),
        })
    }

    /// Normalize a parsed log event into immutable evidence for an explicitly
    /// resolved system subject.
    pub fn normalize(
        &self,
        subject: EntityId,
        event: &LogEvent,
        observed_at_unix_ms: u64,
        ingested_at_unix_ms: Option<u64>,
    ) -> Result<SystemObservationV1, LogObservationAdapterError> {
        if subject.0.trim().is_empty() {
            return Err(LogObservationAdapterError::EmptySubject);
        }

        let event_time = event.timestamp.timestamp_millis();
        let event_time_unix_ms = u64::try_from(event_time)
            .map_err(|_| LogObservationAdapterError::PreEpochTimestamp(event_time))?;

        let mut facts = BTreeMap::new();
        facts.insert(
            "log.severity".into(),
            StateValueV1::Text(severity_name(event.severity).into()),
        );
        facts.insert(
            "log.component".into(),
            StateValueV1::Text(event.component.clone()),
        );
        facts.insert(
            "log.provider".into(),
            StateValueV1::Text(event.provider.clone()),
        );
        facts.insert("log.event_id".into(), StateValueV1::U64(event.event_id as u64));

        if let Some(host) = event.host.as_deref() {
            // Source host is useful evidence but is not treated as the canonical
            // graph subject identity. Scrub before retention.
            facts.insert(
                "log.reported_host".into(),
                StateValueV1::Text(self.scrubber.scrub(host).scrubbed_text),
            );
        }

        if self.policy.retain_message && !event.message.is_empty() {
            facts.insert(
                "log.message".into(),
                StateValueV1::Text(self.scrubber.scrub(&event.message).scrubbed_text),
            );
        }

        if self.policy.retain_fields {
            for (key, value) in &event.fields {
                let sanitized_key = sanitize_fact_key(key);
                let scrubbed = self.scrubber.scrub(value).scrubbed_text;
                facts.insert(
                    format!("log.field.{sanitized_key}"),
                    StateValueV1::Text(scrubbed),
                );
            }
        }

        let observation = SystemObservationV1 {
            id: ObservationId(self.observation_id(&subject, event)),
            subject,
            provenance: ObservationProvenanceV1 {
                source_id: self.config.source_id.clone(),
                source_kind: source_kind(event.source),
                collector: self.config.collector.clone(),
                collector_version: self.config.collector_version.clone(),
                schema_version: self.config.schema_version.clone(),
                artifact_digest: self.config.artifact_digest.clone(),
            },
            clock: ObservationClockV1 {
                event_time_unix_ms: Some(event_time_unix_ms),
                observed_at_unix_ms,
                ingested_at_unix_ms,
                max_age_ms: self.policy.max_age_ms,
                clock_uncertainty_ms: None,
            },
            confidence: self.policy.normalization_confidence,
            facts,
        };
        observation.validate()?;
        Ok(observation)
    }

    fn observation_id(&self, subject: &EntityId, event: &LogEvent) -> String {
        let mut hasher = blake3::Hasher::new();
        hash_part(&mut hasher, b"symthaea-log-observation-v1");
        hash_part(&mut hasher, subject.0.as_bytes());
        hash_part(&mut hasher, self.config.source_id.as_bytes());
        hash_part(&mut hasher, source_tag(event.source).as_bytes());
        hash_part(&mut hasher, event.timestamp.to_rfc3339().as_bytes());
        hash_part(&mut hasher, event.provider.as_bytes());
        hash_part(&mut hasher, &event.event_id.to_le_bytes());
        hash_part(&mut hasher, event.component.as_bytes());
        hash_part(&mut hasher, event.message.as_bytes());
        if let Some(host) = &event.host {
            hash_part(&mut hasher, host.as_bytes());
        }
        for (key, value) in &event.fields {
            hash_part(&mut hasher, key.as_bytes());
            hash_part(&mut hasher, value.as_bytes());
        }
        format!("log:{}", hasher.finalize().to_hex())
    }
}

fn hash_part(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn sanitize_fact_key(key: &str) -> String {
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

fn source_kind(source: Source) -> ObservationSourceKindV1 {
    match source {
        Source::WindowsEvent => ObservationSourceKindV1::WindowsEventLog,
        Source::Syslog => ObservationSourceKindV1::Syslog,
        Source::Snmp => ObservationSourceKindV1::Snmp,
        Source::Other => ObservationSourceKindV1::Other("logparse-other".into()),
    }
}

fn source_tag(source: Source) -> &'static str {
    match source {
        Source::WindowsEvent => "windows-event",
        Source::Syslog => "syslog",
        Source::Snmp => "snmp",
        Source::Other => "other",
    }
}

fn severity_name(severity: Severity) -> &'static str {
    match severity {
        Severity::Debug => "debug",
        Severity::Info => "info",
        Severity::Notice => "notice",
        Severity::Warning => "warning",
        Severity::Error => "error",
        Severity::Critical => "critical",
        Severity::Alert => "alert",
        Severity::Emergency => "emergency",
    }
}

#[derive(Debug)]
pub enum LogObservationAdapterError {
    EmptyConfig(&'static str),
    EmptySubject,
    InvalidConfidence(f32),
    PreEpochTimestamp(i64),
    State(SystemStateGraphError),
}

impl fmt::Display for LogObservationAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyConfig(field) => write!(f, "empty log adapter config field {field}"),
            Self::EmptySubject => write!(f, "log observation subject is empty"),
            Self::InvalidConfidence(value) => write!(f, "invalid normalization confidence {value}"),
            Self::PreEpochTimestamp(value) => write!(f, "pre-epoch log timestamp {value}ms is unsupported"),
            Self::State(err) => write!(f, "invalid normalized observation: {err}"),
        }
    }
}

impl Error for LogObservationAdapterError {}

impl From<SystemStateGraphError> for LogObservationAdapterError {
    fn from(value: SystemStateGraphError) -> Self {
        Self::State(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use std::collections::BTreeMap;

    fn event() -> LogEvent {
        LogEvent {
            timestamp: Utc.timestamp_millis_opt(1_700_000_000_000).single().unwrap(),
            source: Source::Syslog,
            severity: Severity::Error,
            component: "sshd".into(),
            provider: "auth".into(),
            event_id: 42,
            message: "failed login from 192.168.1.44 token sk-abcdefghijklmnopqr".into(),
            fields: BTreeMap::from([
                ("remote_ip".into(), "192.168.1.44".into()),
                ("user name".into(), "admin@example.com".into()),
            ]),
            host: Some("server-01".into()),
            label: None,
        }
    }

    fn adapter(policy: LogObservationPolicyV1) -> LogObservationAdapterV1 {
        LogObservationAdapterV1::new(
            LogObservationAdapterConfigV1 {
                source_id: "server-01:syslog:auth".into(),
                collector: "symthaea-support-logparse-adapter-v1".into(),
                collector_version: Some("1".into()),
                schema_version: Some("logevent-v1".into()),
                artifact_digest: None,
            },
            policy,
        )
        .unwrap()
    }

    #[test]
    fn default_policy_does_not_retain_raw_message_or_fields() {
        let obs = adapter(Default::default())
            .normalize(EntityId("host:server-01".into()), &event(), 1_700_000_000_100, None)
            .unwrap();
        assert!(!obs.facts.contains_key("log.message"));
        assert!(!obs.facts.keys().any(|key| key.starts_with("log.field.")));
        assert_eq!(obs.facts["log.component"], StateValueV1::Text("sshd".into()));
    }

    #[test]
    fn retained_sensitive_text_is_scrubbed() {
        let policy = LogObservationPolicyV1 {
            retain_message: true,
            retain_fields: true,
            ..Default::default()
        };
        let obs = adapter(policy)
            .normalize(EntityId("host:server-01".into()), &event(), 1_700_000_000_100, None)
            .unwrap();
        let message = match &obs.facts["log.message"] {
            StateValueV1::Text(value) => value,
            _ => panic!("message must be text"),
        };
        assert!(!message.contains("192.168.1.44"));
        assert!(!message.contains("sk-abcdefghijklmnopqr"));
        assert!(message.contains("[REDACTED_IP_"));
        let email = match &obs.facts["log.field.user_name"] {
            StateValueV1::Text(value) => value,
            _ => panic!("field must be text"),
        };
        assert!(!email.contains("admin@example.com"));
    }

    #[test]
    fn observation_identity_is_content_deterministic() {
        let adapter = adapter(Default::default());
        let first = adapter
            .normalize(EntityId("host:server-01".into()), &event(), 1_700_000_000_100, None)
            .unwrap();
        let second = adapter
            .normalize(EntityId("host:server-01".into()), &event(), 1_700_000_000_999, None)
            .unwrap();
        assert_eq!(first.id, second.id);
        assert_ne!(first.clock.observed_at_unix_ms, second.clock.observed_at_unix_ms);
    }

    #[test]
    fn same_event_for_different_canonical_subject_has_different_identity() {
        let adapter = adapter(Default::default());
        let first = adapter
            .normalize(EntityId("host:a".into()), &event(), 1_700_000_000_100, None)
            .unwrap();
        let second = adapter
            .normalize(EntityId("host:b".into()), &event(), 1_700_000_000_100, None)
            .unwrap();
        assert_ne!(first.id, second.id);
    }
}
