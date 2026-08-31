//! Bridge normalized log events into the domain-neutral integration fabric.

use crate::{LogEvent, Severity, Source};
use std::collections::BTreeMap;
use symthaea_integration_core::{
    AccessMode, CapabilityClass, CapabilityDeclaration, EntityRef, IntegrationId,
    IntegrationManifest, MaturityLevel, ObservationEnvelope, ObservationId, ObservationKind,
    ObservationLineage, ObservationQuality, ObservationSource, ObservationState,
    ObservationValidationError, ObservationValue, RiskClass, TransformStep,
    INTEGRATION_MANIFEST_SCHEMA_VERSION,
};

pub const LOGPARSE_INTEGRATION_ID: &str = "symthaea-logparse";

/// Deployment-specific context that is not present in a normalized `LogEvent`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIntegrationContext {
    /// Entity namespace for hosts emitted by this collector, e.g. `site:dc1`.
    pub namespace: String,
    pub collector_id: Option<String>,
    pub tenant: Option<String>,
    /// Explicit independent measurement lineage. Missing stays epistemically unknown.
    pub independence_group: Option<String>,
    pub source_confidence: f32,
}

impl Default for LogIntegrationContext {
    fn default() -> Self {
        Self {
            namespace: "site:unknown".into(),
            collector_id: None,
            tenant: None,
            independence_group: None,
            source_confidence: 0.8,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LogIntegrationError {
    #[error("log event timestamp predates unix epoch: {0}")]
    NegativeTimestamp(i64),
    #[error("normalized observation failed validation: {0}")]
    InvalidObservation(#[from] ObservationValidationError),
}

/// Honest v0.1 manifest for the existing parser: fixture-qualified, read-only,
/// and restricted to normalized log observation.
pub fn integration_manifest() -> IntegrationManifest {
    IntegrationManifest {
        schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
        id: IntegrationId::new(LOGPARSE_INTEGRATION_ID),
        display_name: "Symthaea normalized log ingestion".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        provider: "Luminous Dynamics".into(),
        protocols: vec![
            "windows-evtx".into(),
            "syslog-rfc5424".into(),
            "syslog-rfc3164".into(),
        ],
        entity_kinds: vec!["host".into(), "service".into()],
        capabilities: vec![CapabilityDeclaration {
            name: "observe.log.normalized".into(),
            class: CapabilityClass::Observe,
            access: AccessMode::ReadOnly,
            risk: RiskClass::ReadOnly,
            reversible: false,
            default_enabled: true,
        }],
        credentials: vec![],
        maturity: MaturityLevel::E1FixtureParsing,
        default_read_only: true,
    }
}

/// Convert one already-normalized log event into the universal runtime
/// observation contract. The caller supplies the stable observation id so
/// replay/deduplication semantics remain under source control.
pub fn log_event_to_observation(
    event: &LogEvent,
    context: &LogIntegrationContext,
    observation_id: ObservationId,
    ingested_at_unix_ms: u64,
) -> Result<ObservationEnvelope, LogIntegrationError> {
    let observed_at = event.timestamp.timestamp_millis();
    let observed_at_unix_ms = u64::try_from(observed_at)
        .map_err(|_| LogIntegrationError::NegativeTimestamp(observed_at))?;

    let host = event.host.clone().unwrap_or_else(|| "unknown".into());
    let host_known = event.host.is_some();

    let mut payload = event.fields.clone();
    payload.insert("message".into(), event.message.clone());

    let source_name = source_name(event.source);
    let upstream_origin = event.host.as_ref().map(|host| {
        format!(
            "log-stream:{source_name}:{host}:{}",
            if event.provider.is_empty() {
                "unknown-provider"
            } else {
                event.provider.as_str()
            }
        )
    });

    let quality = ObservationQuality {
        source_confidence: context.source_confidence,
        completeness: if host_known { 1.0 } else { 0.8 },
        state: if host_known {
            ObservationState::Observed
        } else {
            ObservationState::Partial
        },
        staleness_ms: Some(ingested_at_unix_ms.saturating_sub(observed_at_unix_ms)),
    };

    let lineage_id = format!("logparse:{}", observation_id.as_str());
    let mut observation = ObservationEnvelope::new(
        observation_id,
        observed_at_unix_ms,
        ingested_at_unix_ms,
        EntityRef::new(&context.namespace, "host", host),
        ObservationKind::Log,
        "log.event",
        ObservationValue::Attributes(payload),
        ObservationSource {
            integration_id: LOGPARSE_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            upstream_origin,
            measurement_method: measurement_method(event.source).into(),
            tenant: context.tenant.clone(),
        },
        quality,
        ObservationLineage {
            lineage_id,
            parent_ids: vec![],
            independence_group: context.independence_group.clone(),
            transforms: vec![TransformStep {
                name: "symthaea-logparse.normalization".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
                deterministic: true,
            }],
        },
    );

    observation.labels = BTreeMap::from([
        ("log.source".into(), source_name.into()),
        ("log.severity".into(), severity_name(event.severity).into()),
        ("log.component".into(), event.component.clone()),
        ("log.provider".into(), event.provider.clone()),
        ("log.event_id".into(), event.event_id.to_string()),
    ]);

    observation.validate()?;
    Ok(observation)
}

fn source_name(source: Source) -> &'static str {
    match source {
        Source::WindowsEvent => "windows-event",
        Source::Syslog => "syslog",
        Source::Snmp => "snmp",
        Source::Other => "other",
    }
}

fn measurement_method(source: Source) -> &'static str {
    match source {
        Source::WindowsEvent => "windows-evtx",
        Source::Syslog => "syslog",
        Source::Snmp => "snmp-trap",
        Source::Other => "normalized-log",
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    fn event(host: Option<&str>) -> LogEvent {
        LogEvent {
            timestamp: "2026-08-31T00:00:00Z"
                .parse::<DateTime<Utc>>()
                .unwrap(),
            source: Source::Syslog,
            severity: Severity::Warning,
            component: "sshd".into(),
            provider: "auth".into(),
            event_id: 42,
            message: "authentication retry".into(),
            fields: BTreeMap::from([("pid".into(), "123".into())]),
            host: host.map(str::to_string),
            label: None,
        }
    }

    #[test]
    fn manifest_is_strictly_read_only() {
        assert!(integration_manifest().validate_read_only_profile().is_ok());
    }

    #[test]
    fn normalized_log_becomes_valid_observation() {
        let observation = log_event_to_observation(
            &event(Some("host-a")),
            &LogIntegrationContext {
                namespace: "site:lab".into(),
                independence_group: Some("host-kernel-log".into()),
                ..Default::default()
            },
            ObservationId::new("syslog-42"),
            1_777_593_601_000,
        )
        .unwrap();

        assert_eq!(observation.entity.id, "host-a");
        assert_eq!(observation.kind, ObservationKind::Log);
        assert_eq!(observation.signal, "log.event");
        assert!(observation.validate().is_ok());
    }

    #[test]
    fn missing_host_is_partial_not_falsely_complete() {
        let observation = log_event_to_observation(
            &event(None),
            &LogIntegrationContext::default(),
            ObservationId::new("unknown-host-log"),
            1_777_593_601_000,
        )
        .unwrap();
        assert_eq!(observation.quality.state, ObservationState::Partial);
        assert!(observation.quality.completeness < 1.0);
    }
}
