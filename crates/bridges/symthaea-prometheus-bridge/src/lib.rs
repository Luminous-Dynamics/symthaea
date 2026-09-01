// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only Prometheus text exposition bridge for Symthaea.
//!
//! v0.1 is intentionally an E1 fixture/replay adapter: callers supply an
//! already-retrieved Prometheus text scrape plus an explicit collection time.
//! Network transport, authentication, service discovery, and full OpenMetrics
//! semantics are deliberately deferred to separately qualified tranches.
//!
//! This bridge preserves the integration-fabric invariants:
//! - no actuation capability or mutation-capable credentials;
//! - deterministic observation identities for replay/deduplication;
//! - explicit source/upstream lineage rather than treating every scrape as an
//!   independent measurement;
//! - non-finite Prometheus values (`NaN`, `+Inf`, `-Inf`) remain visible as
//!   epistemically unknown observations rather than poisoning numeric state;
//! - Prometheus target identity remains local by default. Cross-protocol
//!   OpenTelemetry service identity is an explicit opt-in compatibility mode.

#![forbid(unsafe_code)]

mod identity_provider;

use blake3::Hasher;
use chrono::{TimeZone, Utc};
use prometheus_parse::{Sample, Scrape, Value};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_integration_core::{
    AccessMode, CapabilityClass, CapabilityDeclaration, EntityRef, ExternalIdentifier,
    IdentifierStability, IdentifierUniqueness, IdentityClaim, IdentityClaimSource,
    IdentityStrength, IntegrationError, IntegrationFuture, IntegrationId, IntegrationIdentity,
    IntegrationManifest, MaturityLevel, ObservationBatch, ObservationEnvelope, ObservationId,
    ObservationKind, ObservationLineage, ObservationQuality, ObservationRequest,
    ObservationSource, ObservationState, ObservationValue, Observer, RiskClass, TransformStep,
    INTEGRATION_MANIFEST_SCHEMA_VERSION,
};

pub const PROMETHEUS_INTEGRATION_ID: &str = "prometheus-text";
pub const PROMETHEUS_OBSERVE_CAPABILITY: &str = "observe.prometheus.metrics";
pub const PROMETHEUS_IDENTITY_CAPABILITY: &str = "discover.prometheus.identity";

/// How Prometheus `job`/`instance` labels may be interpreted for entity identity.
///
/// `NativeTarget` is deliberately the default because ordinary Prometheus
/// deployments do not guarantee OpenTelemetry service semantics. The
/// `OtelPrometheusCompatibility` mode is only correct when the scrape labels are
/// known to follow OpenTelemetry's Prometheus compatibility mapping:
/// `job = service.namespace/service.name` (or just service.name) and
/// `instance = service.instance.id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrometheusIdentityMapping {
    #[default]
    NativeTarget,
    OtelPrometheusCompatibility,
}

/// Boundary context supplied by the collector/replay harness.
///
/// Nothing here grants network access. `scrape_id` names one logical scrape
/// source and `default_target_id` is used only when the exposition omits the
/// conventional `instance` label.
#[derive(Debug, Clone, PartialEq)]
pub struct PrometheusFixtureContext {
    pub namespace: String,
    pub scrape_id: String,
    pub collector_id: Option<String>,
    pub default_target_id: String,
    pub tenant: Option<String>,
    pub upstream_origin: Option<String>,
    pub source_confidence: f32,
    pub identity_mapping: PrometheusIdentityMapping,
}

impl Default for PrometheusFixtureContext {
    fn default() -> Self {
        Self {
            namespace: "prometheus:fixture".into(),
            scrape_id: "fixture-scrape".into(),
            collector_id: None,
            default_target_id: "unknown-target".into(),
            tenant: None,
            upstream_origin: None,
            source_confidence: 0.9,
            identity_mapping: PrometheusIdentityMapping::NativeTarget,
        }
    }
}

/// E1 observer over one parsed Prometheus scrape.
#[derive(Debug, Clone)]
pub struct PrometheusTextObserver {
    manifest: IntegrationManifest,
    context: PrometheusFixtureContext,
    batch: ObservationBatch,
    identity_claims: Vec<IdentityClaim>,
}

impl PrometheusTextObserver {
    /// Parse a Prometheus text exposition body at an explicit collection time.
    ///
    /// The parser's sample time is set to `collected_at_unix_ms`, so samples
    /// without an explicit timestamp still acquire a deterministic observation
    /// time supplied by the ingestion boundary rather than wall-clock time.
    pub fn from_text(
        context: PrometheusFixtureContext,
        payload: &str,
        collected_at_unix_ms: u64,
    ) -> Result<Self, IntegrationError> {
        validate_context(&context)?;

        let collected_i64 = i64::try_from(collected_at_unix_ms).map_err(|_| {
            IntegrationError::InvalidRequest(format!(
                "collection time {collected_at_unix_ms} does not fit chrono timestamp range"
            ))
        })?;
        let sample_time = Utc
            .timestamp_millis_opt(collected_i64)
            .single()
            .ok_or_else(|| {
                IntegrationError::InvalidRequest(format!(
                    "invalid collection timestamp {collected_at_unix_ms}"
                ))
            })?;

        let lines = payload
            .lines()
            .map(|line| std::io::Result::Ok(line.to_owned()));
        let scrape = Scrape::parse_at(lines, sample_time)
            .map_err(|error| IntegrationError::Protocol(error.to_string()))?;

        let mut observations = Vec::new();
        let mut identity_claims: BTreeMap<String, IdentityClaim> = BTreeMap::new();
        for sample in &scrape.samples {
            let sample_observations = sample_to_observations(sample, &context, collected_at_unix_ms)?;
            let evidence_ids: Vec<ObservationId> = sample_observations
                .iter()
                .map(|observation| observation.observation_id.clone())
                .collect();
            let labels: BTreeMap<String, String> = sample
                .labels
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let entity = entity_for_labels(&context, &labels);
            let claim = identity_claim_for_target(
                &context,
                &labels,
                entity,
                collected_at_unix_ms,
                evidence_ids,
            )?;
            merge_identity_claim(&mut identity_claims, claim)?;
            observations.extend(sample_observations);
        }

        let batch = ObservationBatch {
            integration_id: PROMETHEUS_INTEGRATION_ID.into(),
            collected_at_unix_ms,
            observations,
        };
        batch
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;

        let identity_claims: Vec<IdentityClaim> = identity_claims.into_values().collect();
        for claim in &identity_claims {
            claim
                .validate()
                .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        }

        Ok(Self {
            manifest: integration_manifest(),
            context,
            batch,
            identity_claims,
        })
    }

    pub fn context(&self) -> &PrometheusFixtureContext {
        &self.context
    }

    pub fn batch(&self) -> &ObservationBatch {
        &self.batch
    }

    pub fn identity_claims(&self) -> &[IdentityClaim] {
        &self.identity_claims
    }

    /// Deterministic synchronous implementation used by tests and by the
    /// object-safe async `Observer` facade.
    pub fn observe_sync(
        &self,
        request: ObservationRequest,
    ) -> Result<ObservationBatch, IntegrationError> {
        request.validate()?;

        let observations = self
            .batch
            .observations
            .iter()
            .filter(|observation| {
                request.entities.is_empty()
                    || request.entities.iter().any(|entity| entity == &observation.entity)
            })
            .filter(|observation| {
                request.signals.is_empty()
                    || request
                        .signals
                        .iter()
                        .any(|signal| signal == &observation.signal)
            })
            .filter(|observation| {
                request
                    .since_unix_ms
                    .is_none_or(|since| observation.observed_at_unix_ms >= since)
            })
            .filter(|observation| {
                request
                    .until_unix_ms
                    .is_none_or(|until| observation.observed_at_unix_ms <= until)
            })
            .filter(|observation| {
                request
                    .filters
                    .iter()
                    .all(|(key, value)| observation.labels.get(key) == Some(value))
            })
            .cloned()
            .collect();

        let batch = ObservationBatch {
            integration_id: PROMETHEUS_INTEGRATION_ID.into(),
            collected_at_unix_ms: self.batch.collected_at_unix_ms,
            observations,
        };
        batch
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(batch)
    }
}

impl IntegrationIdentity for PrometheusTextObserver {
    fn manifest(&self) -> &IntegrationManifest {
        &self.manifest
    }
}

impl Observer for PrometheusTextObserver {
    fn observe<'a>(
        &'a self,
        request: ObservationRequest,
    ) -> IntegrationFuture<'a, Result<ObservationBatch, IntegrationError>> {
        Box::pin(async move { self.observe_sync(request) })
    }
}

pub fn integration_manifest() -> IntegrationManifest {
    IntegrationManifest {
        schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
        id: IntegrationId::new(PROMETHEUS_INTEGRATION_ID),
        display_name: "Prometheus text exposition".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        provider: "Prometheus ecosystem".into(),
        protocols: vec!["prometheus-text".into()],
        entity_kinds: vec!["prometheus_target".into()],
        capabilities: vec![
            CapabilityDeclaration {
                name: PROMETHEUS_OBSERVE_CAPABILITY.into(),
                class: CapabilityClass::Observe,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            },
            CapabilityDeclaration {
                name: PROMETHEUS_IDENTITY_CAPABILITY.into(),
                class: CapabilityClass::Discover,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            },
        ],
        credentials: vec![],
        maturity: MaturityLevel::E1FixtureParsing,
        default_read_only: true,
    }
}

fn validate_context(context: &PrometheusFixtureContext) -> Result<(), IntegrationError> {
    for (name, value) in [
        ("namespace", context.namespace.as_str()),
        ("scrape_id", context.scrape_id.as_str()),
        ("default_target_id", context.default_target_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(IntegrationError::InvalidRequest(format!(
                "Prometheus context `{name}` may not be empty"
            )));
        }
    }
    Ok(())
}

fn sample_to_observations(
    sample: &Sample,
    context: &PrometheusFixtureContext,
    ingested_at_unix_ms: u64,
) -> Result<Vec<ObservationEnvelope>, IntegrationError> {
    let observed_i64 = sample.timestamp.timestamp_millis();
    let observed_at_unix_ms = u64::try_from(observed_i64).map_err(|_| {
        IntegrationError::InvalidOutput(format!(
            "Prometheus sample `{}` predates the unix epoch: {observed_i64}",
            sample.metric
        ))
    })?;

    let labels: BTreeMap<String, String> = sample
        .labels
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let entity = entity_for_labels(context, &labels);
    let lineage_id = sample_lineage_id(context, sample, &labels, observed_at_unix_ms);

    match &sample.value {
        Value::Counter(value) => Ok(vec![scalar_observation(
            sample,
            context,
            &labels,
            entity,
            lineage_id,
            observed_at_unix_ms,
            ingested_at_unix_ms,
            sample.metric.clone(),
            "counter",
            *value,
            None,
        )]),
        Value::Gauge(value) => Ok(vec![scalar_observation(
            sample,
            context,
            &labels,
            entity,
            lineage_id,
            observed_at_unix_ms,
            ingested_at_unix_ms,
            sample.metric.clone(),
            "gauge",
            *value,
            None,
        )]),
        Value::Untyped(value) => Ok(vec![scalar_observation(
            sample,
            context,
            &labels,
            entity,
            lineage_id,
            observed_at_unix_ms,
            ingested_at_unix_ms,
            sample.metric.clone(),
            "untyped",
            *value,
            None,
        )]),
        Value::Histogram(buckets) => Ok(buckets
            .iter()
            .enumerate()
            .map(|(index, bucket)| {
                scalar_observation(
                    sample,
                    context,
                    &labels,
                    entity.clone(),
                    lineage_id.clone(),
                    observed_at_unix_ms,
                    ingested_at_unix_ms,
                    format!("{}.bucket", sample.metric),
                    "histogram_bucket",
                    bucket.count,
                    Some(("le", prometheus_number_label(bucket.less_than))),
                )
                .with_label("prometheus.component_index", index.to_string())
            })
            .collect()),
        Value::Summary(quantiles) => Ok(quantiles
            .iter()
            .enumerate()
            .map(|(index, quantile)| {
                scalar_observation(
                    sample,
                    context,
                    &labels,
                    entity.clone(),
                    lineage_id.clone(),
                    observed_at_unix_ms,
                    ingested_at_unix_ms,
                    format!("{}.quantile", sample.metric),
                    "summary_quantile",
                    quantile.count,
                    Some(("quantile", prometheus_number_label(quantile.quantile))),
                )
                .with_label("prometheus.component_index", index.to_string())
            })
            .collect()),
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_observation(
    sample: &Sample,
    context: &PrometheusFixtureContext,
    labels: &BTreeMap<String, String>,
    entity: EntityRef,
    lineage_id: String,
    observed_at_unix_ms: u64,
    ingested_at_unix_ms: u64,
    signal: String,
    sample_type: &'static str,
    value: f64,
    component_label: Option<(&'static str, String)>,
) -> ObservationEnvelope {
    let mut canonical_labels = labels.clone();
    canonical_labels.insert("prometheus.sample_type".into(), sample_type.into());
    canonical_labels.insert("prometheus.scrape_id".into(), context.scrape_id.clone());
    if let Some((key, value)) = component_label {
        canonical_labels.insert(key.into(), value);
    }

    let component = canonical_labels
        .get("le")
        .map(|value| format!("le={value}"))
        .or_else(|| {
            canonical_labels
                .get("quantile")
                .map(|value| format!("quantile={value}"))
        })
        .unwrap_or_else(|| sample_type.to_string());
    let observation_id = stable_observation_id(
        context,
        &sample.metric,
        &canonical_labels,
        observed_at_unix_ms,
        &component,
    );

    let (observation_value, state) = portable_value(value);
    let confidence = normalized_confidence(context.source_confidence);
    let upstream_origin = context.upstream_origin.clone().or_else(|| {
        labels
            .get("instance")
            .map(|instance| format!("prometheus-target:{instance}"))
    });

    ObservationEnvelope {
        schema_version: symthaea_integration_core::OBSERVATION_SCHEMA_VERSION,
        observation_id,
        observed_at_unix_ms,
        ingested_at_unix_ms,
        entity,
        kind: ObservationKind::Metric,
        signal,
        value: observation_value,
        source: ObservationSource {
            integration_id: PROMETHEUS_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            upstream_origin,
            measurement_method: "prometheus-text".into(),
            tenant: context.tenant.clone(),
        },
        quality: ObservationQuality {
            source_confidence: confidence,
            completeness: 1.0,
            state,
            staleness_ms: Some(ingested_at_unix_ms.saturating_sub(observed_at_unix_ms)),
        },
        lineage: ObservationLineage {
            lineage_id,
            parent_ids: vec![],
            // A scrape alone does not prove independence from another collector.
            independence_group: None,
            transforms: vec![TransformStep {
                name: "prometheus-text-normalization".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
                deterministic: true,
            }],
        },
        labels: canonical_labels,
    }
}

fn entity_for_labels(
    context: &PrometheusFixtureContext,
    labels: &BTreeMap<String, String>,
) -> EntityRef {
    let id = labels
        .get("instance")
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| context.default_target_id.clone());
    EntityRef::new(&context.namespace, "prometheus_target", id)
}

fn identity_claim_for_target(
    context: &PrometheusFixtureContext,
    labels: &BTreeMap<String, String>,
    subject: EntityRef,
    collected_at_unix_ms: u64,
    evidence_observation_ids: Vec<ObservationId>,
) -> Result<IdentityClaim, IntegrationError> {
    let (identifier, strength) = match context.identity_mapping {
        PrometheusIdentityMapping::NativeTarget => {
            native_target_identifier(context, labels, &subject)
        }
        PrometheusIdentityMapping::OtelPrometheusCompatibility => {
            otel_compatibility_identifier(labels)
                .map(|identifier| (identifier, IdentityStrength::Strong))
                .unwrap_or_else(|| native_target_identifier(context, labels, &subject))
        }
    };

    let identifier_key = identifier
        .canonical_key()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    let scope_material = format!(
        "{}|{}|{}|{}",
        PROMETHEUS_INTEGRATION_ID,
        context.scrape_id,
        subject.canonical_key(),
        identifier_key
    );
    let digest = blake3::hash(scope_material.as_bytes()).to_hex().to_string();
    Ok(IdentityClaim {
        claim_id: format!("prometheus-identity:{digest}"),
        subject,
        identifier,
        strength,
        source_confidence: normalized_confidence(context.source_confidence),
        source: IdentityClaimSource {
            integration_id: PROMETHEUS_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            tenant: context.tenant.clone(),
        },
        observed_at_unix_ms: collected_at_unix_ms,
        valid_from_unix_ms: None,
        valid_until_unix_ms: None,
        evidence_observation_ids,
    })
}

fn native_target_identifier(
    context: &PrometheusFixtureContext,
    labels: &BTreeMap<String, String>,
    subject: &EntityRef,
) -> (ExternalIdentifier, IdentityStrength) {
    let target = labels
        .get("instance")
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| subject.id.clone());
    let scope = format!(
        "prometheus:{}:{}|{}:{}",
        context.namespace.len(),
        context.namespace,
        context.scrape_id.len(),
        context.scrape_id
    );
    (
        ExternalIdentifier {
            scheme: "prometheus.target".into(),
            value: target,
            scope: Some(scope),
            uniqueness: IdentifierUniqueness::Scoped,
            stability: IdentifierStability::Session,
            case_sensitive: true,
        },
        IdentityStrength::Moderate,
    )
}

fn otel_compatibility_identifier(
    labels: &BTreeMap<String, String>,
) -> Option<ExternalIdentifier> {
    let job = labels
        .get("job")
        .map(String::as_str)
        .filter(|value| !value.trim().is_empty())?;
    let instance = labels
        .get("instance")
        .map(String::as_str)
        .filter(|value| !value.trim().is_empty())?;
    let (service_namespace, service_name) = invert_otel_prometheus_job(job)?;
    Some(ExternalIdentifier {
        scheme: "otel.service.instance.triplet".into(),
        value: length_prefixed_triplet(service_namespace, service_name, instance),
        scope: None,
        uniqueness: IdentifierUniqueness::Global,
        stability: IdentifierStability::Session,
        case_sensitive: true,
    })
}

/// Conservatively invert OpenTelemetry's Prometheus `job` mapping.
///
/// No slash means an empty service namespace. Exactly one slash maps to
/// namespace/name. More than one slash is not losslessly invertible without
/// additional metadata, so the caller must fall back to native target identity.
fn invert_otel_prometheus_job(job: &str) -> Option<(&str, &str)> {
    let slash_count = job.as_bytes().iter().filter(|byte| **byte == b'/').count();
    match slash_count {
        0 if !job.is_empty() => Some(("", job)),
        1 => {
            let (namespace, name) = job.split_once('/')?;
            if namespace.is_empty() || name.is_empty() {
                None
            } else {
                Some((namespace, name))
            }
        }
        _ => None,
    }
}

fn length_prefixed_triplet(first: &str, second: &str, third: &str) -> String {
    format!(
        "{}:{first}|{}:{second}|{}:{third}",
        first.len(),
        second.len(),
        third.len()
    )
}

fn merge_identity_claim(
    claims: &mut BTreeMap<String, IdentityClaim>,
    mut incoming: IdentityClaim,
) -> Result<(), IntegrationError> {
    let identifier_key = incoming
        .identifier
        .canonical_key()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    let key = format!("{}|{identifier_key}", incoming.subject.canonical_key());
    match claims.get_mut(&key) {
        None => {
            incoming.evidence_observation_ids.sort();
            incoming.evidence_observation_ids.dedup();
            claims.insert(key, incoming);
        }
        Some(existing) => {
            if existing.claim_id != incoming.claim_id
                || existing.subject != incoming.subject
                || existing.identifier != incoming.identifier
                || existing.strength != incoming.strength
                || existing.source != incoming.source
            {
                return Err(IntegrationError::InvalidOutput(
                    "Prometheus identity claim collision has incompatible metadata".into(),
                ));
            }
            existing.source_confidence = existing
                .source_confidence
                .min(incoming.source_confidence);
            existing.observed_at_unix_ms = existing
                .observed_at_unix_ms
                .min(incoming.observed_at_unix_ms);
            let mut evidence: BTreeSet<ObservationId> = existing
                .evidence_observation_ids
                .iter()
                .cloned()
                .collect();
            evidence.extend(incoming.evidence_observation_ids);
            existing.evidence_observation_ids = evidence.into_iter().collect();
        }
    }
    Ok(())
}

fn sample_lineage_id(
    context: &PrometheusFixtureContext,
    sample: &Sample,
    labels: &BTreeMap<String, String>,
    observed_at_unix_ms: u64,
) -> String {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-prometheus-lineage-v1\0");
    feed_string(&mut hasher, &context.scrape_id);
    feed_string(&mut hasher, &sample.metric);
    feed_labels(&mut hasher, labels);
    hasher.update(&observed_at_unix_ms.to_le_bytes());
    format!("prometheus-lineage:{}", hasher.finalize().to_hex())
}

fn stable_observation_id(
    context: &PrometheusFixtureContext,
    metric: &str,
    labels: &BTreeMap<String, String>,
    observed_at_unix_ms: u64,
    component: &str,
) -> ObservationId {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-prometheus-observation-v1\0");
    feed_string(&mut hasher, &context.scrape_id);
    feed_string(&mut hasher, metric);
    feed_labels(&mut hasher, labels);
    hasher.update(&observed_at_unix_ms.to_le_bytes());
    feed_string(&mut hasher, component);
    ObservationId::new(format!("prometheus:{}", hasher.finalize().to_hex()))
}

fn feed_string(hasher: &mut Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn feed_labels(hasher: &mut Hasher, labels: &BTreeMap<String, String>) {
    for (key, value) in labels {
        feed_string(hasher, key);
        feed_string(hasher, value);
    }
}

fn portable_value(value: f64) -> (ObservationValue, ObservationState) {
    if value.is_finite() {
        return (
            ObservationValue::Number { value, unit: None },
            ObservationState::Observed,
        );
    }

    let text = if value.is_nan() {
        "NaN"
    } else if value.is_sign_positive() {
        "+Inf"
    } else {
        "-Inf"
    };
    (ObservationValue::Text(text.into()), ObservationState::Unknown)
}

fn prometheus_number_label(value: f64) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value == f64::INFINITY {
        "+Inf".into()
    } else if value == f64::NEG_INFINITY {
        "-Inf".into()
    } else {
        value.to_string()
    }
}

fn normalized_confidence(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_integration_core::{ResolutionStatus, assess_entity_pair};

    fn fixture() -> PrometheusTextObserver {
        let payload = r#"
# HELP http_requests_total Total HTTP requests.
# TYPE http_requests_total counter
http_requests_total{instance="api-1:9100",job="api",code="200"} 1027
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes{instance="api-1:9100",job="api"} 12345
"#;
        PrometheusTextObserver::from_text(
            PrometheusFixtureContext {
                namespace: "site:lab".into(),
                scrape_id: "scrape-42".into(),
                collector_id: Some("prometheus-a".into()),
                default_target_id: "fallback".into(),
                tenant: Some("tenant-a".into()),
                upstream_origin: None,
                source_confidence: 0.95,
                identity_mapping: PrometheusIdentityMapping::NativeTarget,
            },
            payload,
            1_777_593_600_000,
        )
        .unwrap()
    }

    #[test]
    fn manifest_is_strictly_read_only_e1() {
        let manifest = integration_manifest();
        assert_eq!(manifest.maturity, MaturityLevel::E1FixtureParsing);
        assert!(manifest.validate_read_only_profile().is_ok());
        assert!(manifest.declares(PROMETHEUS_OBSERVE_CAPABILITY));
        assert!(manifest.declares(PROMETHEUS_IDENTITY_CAPABILITY));
    }

    #[test]
    fn text_scrape_becomes_valid_metric_observations() {
        let observer = fixture();
        assert_eq!(observer.batch().observations.len(), 2);
        assert!(observer.batch().validate().is_ok());
        assert!(observer
            .batch()
            .observations
            .iter()
            .all(|observation| observation.entity.id == "api-1:9100"));
    }

    #[test]
    fn native_target_claim_deduplicates_across_metrics_and_keeps_evidence() {
        let observer = fixture();
        assert_eq!(observer.identity_claims().len(), 1);
        let claim = &observer.identity_claims()[0];
        assert_eq!(claim.identifier.scheme, "prometheus.target");
        assert_eq!(claim.identifier.uniqueness, IdentifierUniqueness::Scoped);
        assert_eq!(claim.strength, IdentityStrength::Moderate);
        assert_eq!(claim.evidence_observation_ids.len(), 2);
    }

    #[test]
    fn otel_compatibility_mode_emits_the_same_service_triplet_scheme() {
        let observer = PrometheusTextObserver::from_text(
            PrometheusFixtureContext {
                identity_mapping: PrometheusIdentityMapping::OtelPrometheusCompatibility,
                ..PrometheusFixtureContext::default()
            },
            "# TYPE up gauge\nup{job=\"shop/api\",instance=\"api-17\"} 1\n",
            1_777_593_600_000,
        )
        .unwrap();
        let claim = &observer.identity_claims()[0];
        assert_eq!(claim.identifier.scheme, "otel.service.instance.triplet");
        assert_eq!(claim.identifier.value, "4:shop|3:api|6:api-17");
        assert_eq!(claim.identifier.uniqueness, IdentifierUniqueness::Global);
        assert_eq!(claim.strength, IdentityStrength::Strong);
    }

    #[test]
    fn malformed_compatibility_job_falls_back_to_native_identity() {
        let observer = PrometheusTextObserver::from_text(
            PrometheusFixtureContext {
                identity_mapping: PrometheusIdentityMapping::OtelPrometheusCompatibility,
                ..PrometheusFixtureContext::default()
            },
            "# TYPE up gauge\nup{job=\"too/many/slashes\",instance=\"api-17\"} 1\n",
            1_777_593_600_000,
        )
        .unwrap();
        assert_eq!(
            observer.identity_claims()[0].identifier.scheme,
            "prometheus.target"
        );
    }

    #[test]
    fn low_source_confidence_prevents_strong_cross_source_resolution() {
        let observer = PrometheusTextObserver::from_text(
            PrometheusFixtureContext {
                source_confidence: 0.5,
                identity_mapping: PrometheusIdentityMapping::OtelPrometheusCompatibility,
                ..PrometheusFixtureContext::default()
            },
            "# TYPE up gauge\nup{job=\"shop/api\",instance=\"api-17\"} 1\n",
            1_777_593_600_000,
        )
        .unwrap();
        let prometheus_claim = observer.identity_claims()[0].clone();
        let otlp_subject = EntityRef::new("otel", "service_instance", "otlp-api");
        let otlp_claim = IdentityClaim {
            claim_id: "otlp-claim".into(),
            subject: otlp_subject.clone(),
            identifier: ExternalIdentifier {
                scheme: "otel.service.instance.triplet".into(),
                value: "4:shop|3:api|6:api-17".into(),
                scope: None,
                uniqueness: IdentifierUniqueness::Global,
                stability: IdentifierStability::Session,
                case_sensitive: true,
            },
            strength: IdentityStrength::Strong,
            source_confidence: 1.0,
            source: IdentityClaimSource {
                integration_id: "otlp-metrics".into(),
                collector_id: None,
                tenant: None,
            },
            observed_at_unix_ms: 1_777_593_600_000,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        };
        let proposal = assess_entity_pair(
            &prometheus_claim.subject,
            &otlp_subject,
            &[prometheus_claim, otlp_claim],
            &[],
            1_777_593_600_000,
        )
        .unwrap();
        assert_eq!(proposal.status, ResolutionStatus::CandidateSame);
    }

    #[test]
    fn labels_and_signal_filters_are_applied() {
        let batch = fixture()
            .observe_sync(ObservationRequest {
                signals: vec!["http_requests_total".into()],
                filters: BTreeMap::from([("code".into(), "200".into())]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(batch.observations.len(), 1);
        assert_eq!(batch.observations[0].signal, "http_requests_total");
    }

    #[test]
    fn observation_ids_are_replay_deterministic() {
        let a = fixture();
        let b = fixture();
        let ids_a: Vec<_> = a
            .batch()
            .observations
            .iter()
            .map(|observation| observation.observation_id.clone())
            .collect();
        let ids_b: Vec<_> = b
            .batch()
            .observations
            .iter()
            .map(|observation| observation.observation_id.clone())
            .collect();
        assert_eq!(ids_a, ids_b);
        assert_eq!(a.identity_claims(), b.identity_claims());
    }

    #[test]
    fn context_confidence_is_bounded_before_validation() {
        let mut observer = fixture();
        observer.context.source_confidence = f32::NAN;
        // Existing parsed observations remain valid and immutable; a fresh
        // parse with NaN confidence normalizes confidence to zero.
        let reparsed = PrometheusTextObserver::from_text(
            PrometheusFixtureContext {
                source_confidence: f32::NAN,
                ..PrometheusFixtureContext::default()
            },
            "# TYPE up gauge\nup{instance=\"node-1\"} 1\n",
            1_777_593_600_000,
        )
        .unwrap();
        assert_eq!(
            reparsed.batch().observations[0].quality.source_confidence,
            0.0
        );
        assert_eq!(reparsed.identity_claims()[0].source_confidence, 0.0);
    }
}
