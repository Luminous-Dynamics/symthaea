// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only OTLP metrics replay bridge for Symthaea's integration fabric.
//!
//! v0.1 intentionally accepts already-decoded OTLP metric messages and does not
//! open a gRPC/HTTP receiver. This isolates semantic translation from transport,
//! authentication, rate limiting and live-listener hardening. The manifest only
//! claims scalar Gauge/Sum observation at E1 fixture/replay maturity.

#![forbid(unsafe_code)]

use opentelemetry_proto::tonic::{
    collector::metrics::v1::ExportMetricsServiceRequest,
    common::v1::{KeyValue, any_value},
    metrics::v1::{DataPointFlags, NumberDataPoint, metric, number_data_point},
};
use std::collections::BTreeMap;
use symthaea_integration_core::{
    AccessMode, CapabilityClass, CapabilityDeclaration, EntityRef, IntegrationError,
    IntegrationFuture, IntegrationId, IntegrationIdentity, IntegrationManifest, MaturityLevel,
    ObservationBatch, ObservationEnvelope, ObservationId, ObservationKind, ObservationLineage,
    ObservationQuality, ObservationRequest, ObservationSource, ObservationState, ObservationValue,
    Observer, RiskClass, TransformStep, INTEGRATION_MANIFEST_SCHEMA_VERSION,
};

pub const OTLP_METRICS_INTEGRATION_ID: &str = "otlp-metrics";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtlpMetricsContext {
    pub namespace: String,
    pub collector_id: Option<String>,
    pub tenant: Option<String>,
    /// Preserve the real upstream producer/collector identity when known. OTLP
    /// itself does not make an observation independent of another feed.
    pub upstream_origin: Option<String>,
    /// Only set when deployment evidence establishes a genuinely independent
    /// measurement lineage. None is deliberately the default.
    pub independence_group: Option<String>,
}

impl Default for OtlpMetricsContext {
    fn default() -> Self {
        Self {
            namespace: "otel".into(),
            collector_id: None,
            tenant: None,
            upstream_origin: None,
            independence_group: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OtlpMetricsObserver {
    manifest: IntegrationManifest,
    context: OtlpMetricsContext,
    request: ExportMetricsServiceRequest,
    ingested_at_unix_ms: u64,
}

impl OtlpMetricsObserver {
    pub fn new(
        context: OtlpMetricsContext,
        request: ExportMetricsServiceRequest,
        ingested_at_unix_ms: u64,
    ) -> Self {
        Self {
            manifest: integration_manifest(),
            context,
            request,
            ingested_at_unix_ms,
        }
    }

    pub fn translate(&self) -> Result<OtlpMetricsTranslation, IntegrationError> {
        translate_metrics_request(&self.request, &self.context, self.ingested_at_unix_ms)
    }

    pub fn observe_sync(
        &self,
        request: ObservationRequest,
    ) -> Result<ObservationBatch, IntegrationError> {
        request.validate()?;
        let mut batch = self.translate()?.batch;
        batch.observations.retain(|observation| {
            (request.entities.is_empty() || request.entities.contains(&observation.entity))
                && (request.signals.is_empty() || request.signals.contains(&observation.signal))
                && request
                    .since_unix_ms
                    .is_none_or(|since| observation.observed_at_unix_ms >= since)
                && request
                    .until_unix_ms
                    .is_none_or(|until| observation.observed_at_unix_ms <= until)
                && request
                    .filters
                    .iter()
                    .all(|(key, value)| observation.labels.get(key) == Some(value))
        });
        batch
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(batch)
    }
}

impl IntegrationIdentity for OtlpMetricsObserver {
    fn manifest(&self) -> &IntegrationManifest {
        &self.manifest
    }
}

impl Observer for OtlpMetricsObserver {
    fn observe<'a>(
        &'a self,
        request: ObservationRequest,
    ) -> IntegrationFuture<'a, Result<ObservationBatch, IntegrationError>> {
        Box::pin(async move { self.observe_sync(request) })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OtlpMetricsTranslation {
    pub batch: ObservationBatch,
    /// Histogram, exponential-histogram and summary data are intentionally not
    /// flattened into misleading scalar values in v0.1. They remain explicit
    /// skipped capability surface until the canonical envelope gains a lossless
    /// distribution representation.
    pub skipped_non_scalar_metrics: usize,
}

pub fn integration_manifest() -> IntegrationManifest {
    IntegrationManifest {
        schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
        id: IntegrationId::new(OTLP_METRICS_INTEGRATION_ID),
        display_name: "OpenTelemetry OTLP Metrics Replay".into(),
        version: "0.1.0".into(),
        provider: "OpenTelemetry / Symthaea".into(),
        protocols: vec!["OTLP/protobuf decoded-message replay".into()],
        entity_kinds: vec![
            "service_instance".into(),
            "k8s_pod".into(),
            "container".into(),
            "host".into(),
            "cloud_resource".into(),
            "service".into(),
            "otel_resource".into(),
        ],
        capabilities: vec![CapabilityDeclaration {
            name: "observe.metrics.scalar".into(),
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

pub fn translate_metrics_request(
    request: &ExportMetricsServiceRequest,
    context: &OtlpMetricsContext,
    ingested_at_unix_ms: u64,
) -> Result<OtlpMetricsTranslation, IntegrationError> {
    let mut observations = Vec::new();
    let mut skipped_non_scalar_metrics = 0usize;

    for (resource_index, resource_metrics) in request.resource_metrics.iter().enumerate() {
        let resource_attributes = primitive_attributes(
            resource_metrics
                .resource
                .as_ref()
                .map(|resource| resource.attributes.as_slice())
                .unwrap_or_default(),
            "resource",
        )?;
        let resource_dropped = resource_metrics
            .resource
            .as_ref()
            .map_or(0, |resource| resource.dropped_attributes_count as usize);
        let (entity, weak_identity) = resource_entity(
            &resource_attributes.values,
            context,
            resource_index,
        );

        for scope_metrics in &resource_metrics.scope_metrics {
            let scope_name = scope_metrics
                .scope
                .as_ref()
                .map(|scope| scope.name.as_str())
                .unwrap_or("");
            let scope_version = scope_metrics
                .scope
                .as_ref()
                .map(|scope| scope.version.as_str())
                .unwrap_or("");

            for metric in &scope_metrics.metrics {
                if metric.name.trim().is_empty() {
                    return Err(IntegrationError::InvalidOutput(
                        "OTLP metric name is empty".into(),
                    ));
                }

                match metric.data.as_ref() {
                    Some(metric::Data::Gauge(gauge)) => {
                        for point in &gauge.data_points {
                            observations.push(number_point_to_observation(
                                point,
                                metric.name.as_str(),
                                metric.description.as_str(),
                                metric.unit.as_str(),
                                "gauge",
                                &entity,
                                weak_identity,
                                &resource_attributes,
                                resource_dropped,
                                scope_name,
                                scope_version,
                                resource_metrics.schema_url.as_str(),
                                scope_metrics.schema_url.as_str(),
                                context,
                                ingested_at_unix_ms,
                            )?);
                        }
                    }
                    Some(metric::Data::Sum(sum)) => {
                        for point in &sum.data_points {
                            let mut observation = number_point_to_observation(
                                point,
                                metric.name.as_str(),
                                metric.description.as_str(),
                                metric.unit.as_str(),
                                "sum",
                                &entity,
                                weak_identity,
                                &resource_attributes,
                                resource_dropped,
                                scope_name,
                                scope_version,
                                resource_metrics.schema_url.as_str(),
                                scope_metrics.schema_url.as_str(),
                                context,
                                ingested_at_unix_ms,
                            )?;
                            observation.labels.insert(
                                "otel.sum.aggregation_temporality".into(),
                                sum.aggregation_temporality.to_string(),
                            );
                            observation
                                .labels
                                .insert("otel.sum.monotonic".into(), sum.is_monotonic.to_string());
                            observations.push(observation);
                        }
                    }
                    Some(
                        metric::Data::Histogram(_)
                        | metric::Data::ExponentialHistogram(_)
                        | metric::Data::Summary(_),
                    ) => skipped_non_scalar_metrics += 1,
                    None => {
                        return Err(IntegrationError::InvalidOutput(format!(
                            "OTLP metric `{}` has no data payload",
                            metric.name
                        )));
                    }
                }
            }
        }
    }

    let batch = ObservationBatch {
        integration_id: OTLP_METRICS_INTEGRATION_ID.into(),
        collected_at_unix_ms: ingested_at_unix_ms,
        observations,
    };
    batch
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;

    Ok(OtlpMetricsTranslation {
        batch,
        skipped_non_scalar_metrics,
    })
}

#[allow(clippy::too_many_arguments)]
fn number_point_to_observation(
    point: &NumberDataPoint,
    metric_name: &str,
    metric_description: &str,
    metric_unit: &str,
    metric_type: &str,
    entity: &EntityRef,
    weak_identity: bool,
    resource_attributes: &PrimitiveAttributes,
    resource_dropped: usize,
    scope_name: &str,
    scope_version: &str,
    resource_schema_url: &str,
    scope_schema_url: &str,
    context: &OtlpMetricsContext,
    ingested_at_unix_ms: u64,
) -> Result<ObservationEnvelope, IntegrationError> {
    if point.time_unix_nano == 0 {
        return Err(IntegrationError::InvalidOutput(format!(
            "OTLP scalar metric `{metric_name}` has required time_unix_nano=0"
        )));
    }

    let point_attributes = primitive_attributes(&point.attributes, "data point")?;
    let no_recorded_value = point.flags & DataPointFlags::NoRecordedValueMask as u32 != 0;

    let (value, mut quality) = if no_recorded_value {
        (
            ObservationValue::Text("no_recorded_value".into()),
            ObservationQuality {
                source_confidence: 1.0,
                completeness: 0.0,
                state: ObservationState::Unknown,
                staleness_ms: None,
            },
        )
    } else {
        match point.value {
            Some(number_data_point::Value::AsDouble(value)) if value.is_finite() => (
                ObservationValue::Number {
                    value,
                    unit: non_empty(metric_unit),
                },
                ObservationQuality::observed(1.0),
            ),
            Some(number_data_point::Value::AsDouble(value)) => (
                ObservationValue::Text(value.to_string()),
                ObservationQuality {
                    source_confidence: 1.0,
                    completeness: 0.0,
                    state: ObservationState::Unknown,
                    staleness_ms: None,
                },
            ),
            Some(number_data_point::Value::AsInt(value)) => (
                ObservationValue::Integer(value),
                ObservationQuality::observed(1.0),
            ),
            None => {
                return Err(IntegrationError::InvalidOutput(format!(
                    "OTLP scalar metric `{metric_name}` has no numeric value and no no-recorded-value flag"
                )));
            }
        }
    };

    let omitted_attributes = resource_attributes.omitted
        + point_attributes.omitted
        + resource_dropped;
    if quality.state == ObservationState::Observed && (weak_identity || omitted_attributes > 0) {
        quality.state = ObservationState::Partial;
        quality.completeness = if weak_identity { 0.85 } else { 0.95 };
        if omitted_attributes > 0 {
            quality.completeness = quality.completeness.min(0.9);
        }
    }

    let observed_at_unix_ms = point.time_unix_nano / 1_000_000;
    let mut labels = BTreeMap::new();
    for (key, value) in &resource_attributes.values {
        labels.insert(format!("resource.{key}"), value.clone());
    }
    for (key, value) in &point_attributes.values {
        labels.insert(key.clone(), value.clone());
    }
    labels.insert("otel.metric.type".into(), metric_type.into());
    labels.insert("otel.time_unix_nano".into(), point.time_unix_nano.to_string());
    if point.start_time_unix_nano != 0 {
        labels.insert(
            "otel.start_time_unix_nano".into(),
            point.start_time_unix_nano.to_string(),
        );
    }
    if !metric_description.is_empty() {
        labels.insert("otel.metric.description".into(), metric_description.into());
    }
    if !metric_unit.is_empty() {
        labels.insert("otel.metric.unit".into(), metric_unit.into());
    }
    if !scope_name.is_empty() {
        labels.insert("otel.scope.name".into(), scope_name.into());
    }
    if !scope_version.is_empty() {
        labels.insert("otel.scope.version".into(), scope_version.into());
    }
    if !resource_schema_url.is_empty() {
        labels.insert("otel.resource.schema_url".into(), resource_schema_url.into());
    }
    if !scope_schema_url.is_empty() {
        labels.insert("otel.scope.schema_url".into(), scope_schema_url.into());
    }
    if omitted_attributes > 0 {
        labels.insert(
            "otel.omitted_or_dropped_attributes".into(),
            omitted_attributes.to_string(),
        );
    }

    let lineage_material = canonical_measurement_material(
        entity,
        metric_name,
        metric_type,
        point,
        &point_attributes.values,
        scope_name,
    );
    let lineage_hash = blake3::hash(lineage_material.as_bytes()).to_hex().to_string();
    let value_material = format!("{lineage_material}|value={value:?}|flags={}", point.flags);
    let observation_hash = blake3::hash(value_material.as_bytes()).to_hex().to_string();

    let mut observation = ObservationEnvelope::new(
        ObservationId::new(format!("otlp:{observation_hash}")),
        observed_at_unix_ms,
        ingested_at_unix_ms,
        entity.clone(),
        ObservationKind::Metric,
        metric_name,
        value,
        ObservationSource {
            integration_id: OTLP_METRICS_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            upstream_origin: context.upstream_origin.clone(),
            measurement_method: "otlp-metrics-replay".into(),
            tenant: context.tenant.clone(),
        },
        quality,
        ObservationLineage {
            lineage_id: format!("otlp-lineage:{lineage_hash}"),
            parent_ids: vec![],
            independence_group: context.independence_group.clone(),
            transforms: vec![TransformStep {
                name: "otlp-scalar-metric-to-observation".into(),
                version: Some("0.1.0".into()),
                deterministic: true,
            }],
        },
    );
    observation.labels = labels;
    observation
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    Ok(observation)
}

fn canonical_measurement_material(
    entity: &EntityRef,
    metric_name: &str,
    metric_type: &str,
    point: &NumberDataPoint,
    point_attributes: &BTreeMap<String, String>,
    scope_name: &str,
) -> String {
    let mut material = format!(
        "{}|{metric_name}|{metric_type}|{}|{}|scope={scope_name}",
        entity.canonical_key(),
        point.start_time_unix_nano,
        point.time_unix_nano,
    );
    for (key, value) in point_attributes {
        material.push('|');
        material.push_str(key);
        material.push('=');
        material.push_str(value);
    }
    material
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrimitiveAttributes {
    values: BTreeMap<String, String>,
    omitted: usize,
}

fn primitive_attributes(
    attributes: &[KeyValue],
    context: &str,
) -> Result<PrimitiveAttributes, IntegrationError> {
    let mut values = BTreeMap::new();
    let mut omitted = 0usize;
    for attribute in attributes {
        if attribute.key.trim().is_empty() {
            return Err(IntegrationError::InvalidOutput(format!(
                "OTLP {context} contains an empty attribute key"
            )));
        }
        let Some(value) = attribute.value.as_ref().and_then(any_value_to_string) else {
            omitted += 1;
            continue;
        };
        if values.insert(attribute.key.clone(), value).is_some() {
            return Err(IntegrationError::InvalidOutput(format!(
                "OTLP {context} contains duplicate attribute `{}`",
                attribute.key
            )));
        }
    }
    Ok(PrimitiveAttributes { values, omitted })
}

fn any_value_to_string(value: &opentelemetry_proto::tonic::common::v1::AnyValue) -> Option<String> {
    match value.value.as_ref()? {
        any_value::Value::StringValue(value) => Some(value.clone()),
        any_value::Value::BoolValue(value) => Some(value.to_string()),
        any_value::Value::IntValue(value) => Some(value.to_string()),
        any_value::Value::DoubleValue(value) if value.is_finite() => Some(value.to_string()),
        any_value::Value::DoubleValue(_) => None,
        any_value::Value::ArrayValue(_)
        | any_value::Value::KvlistValue(_)
        | any_value::Value::BytesValue(_)
        | any_value::Value::StringValueStrindex(_) => None,
    }
}

fn resource_entity(
    attributes: &BTreeMap<String, String>,
    context: &OtlpMetricsContext,
    resource_index: usize,
) -> (EntityRef, bool) {
    const STRONG_KEYS: [(&str, &str); 5] = [
        ("service.instance.id", "service_instance"),
        ("k8s.pod.uid", "k8s_pod"),
        ("container.id", "container"),
        ("host.id", "host"),
        ("cloud.resource_id", "cloud_resource"),
    ];
    for (key, kind) in STRONG_KEYS {
        if let Some(id) = attributes.get(key).filter(|id| !id.trim().is_empty()) {
            return (EntityRef::new(&context.namespace, kind, id), false);
        }
    }
    if let Some(service_name) = attributes
        .get("service.name")
        .filter(|name| !name.trim().is_empty())
    {
        return (
            EntityRef::new(&context.namespace, "service", service_name),
            true,
        );
    }

    if attributes.is_empty() {
        return (
            EntityRef::new(
                &context.namespace,
                "otel_resource",
                format!("anonymous-resource-{resource_index}"),
            ),
            true,
        );
    }

    let mut canonical = String::new();
    for (key, value) in attributes {
        canonical.push_str(key);
        canonical.push('=');
        canonical.push_str(value);
        canonical.push('|');
    }
    let digest = blake3::hash(canonical.as_bytes()).to_hex().to_string();
    (
        EntityRef::new(
            &context.namespace,
            "otel_resource",
            format!("attrs-{digest}"),
        ),
        true,
    )
}

fn non_empty(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_proto::tonic::{
        common::v1::{AnyValue, InstrumentationScope},
        metrics::v1::{Gauge, Metric, ResourceMetrics, ScopeMetrics, Sum},
        resource::v1::Resource,
    };

    fn string_kv(key: &str, value: &str) -> KeyValue {
        KeyValue {
            key: key.into(),
            value: Some(AnyValue {
                value: Some(any_value::Value::StringValue(value.into())),
            }),
        }
    }

    fn point(value: number_data_point::Value) -> NumberDataPoint {
        NumberDataPoint {
            attributes: vec![string_kv("cpu", "0")],
            start_time_unix_nano: 1_000_000_000,
            time_unix_nano: 2_000_000_000,
            exemplars: vec![],
            flags: 0,
            value: Some(value),
        }
    }

    fn request(data: metric::Data) -> ExportMetricsServiceRequest {
        ExportMetricsServiceRequest {
            resource_metrics: vec![ResourceMetrics {
                resource: Some(Resource {
                    attributes: vec![
                        string_kv("service.name", "api"),
                        string_kv("service.instance.id", "api-17"),
                    ],
                    dropped_attributes_count: 0,
                    entity_refs: vec![],
                }),
                scope_metrics: vec![ScopeMetrics {
                    scope: Some(InstrumentationScope {
                        name: "fixture".into(),
                        version: "1.0".into(),
                        attributes: vec![],
                        dropped_attributes_count: 0,
                    }),
                    metrics: vec![Metric {
                        name: "system.cpu.utilization".into(),
                        description: "CPU utilization".into(),
                        unit: "1".into(),
                        metadata: vec![],
                        data: Some(data),
                    }],
                    schema_url: "https://opentelemetry.io/schemas/1.0.0".into(),
                }],
                schema_url: "https://opentelemetry.io/schemas/1.0.0".into(),
            }],
        }
    }

    #[test]
    fn manifest_is_strictly_read_only_and_e1() {
        let manifest = integration_manifest();
        assert_eq!(manifest.maturity, MaturityLevel::E1FixtureParsing);
        assert!(manifest.validate_read_only_profile().is_ok());
    }

    #[test]
    fn gauge_maps_service_instance_and_scalar_value() {
        let translated = translate_metrics_request(
            &request(metric::Data::Gauge(Gauge {
                data_points: vec![point(number_data_point::Value::AsDouble(0.75))],
            })),
            &OtlpMetricsContext::default(),
            2_100,
        )
        .unwrap();
        assert_eq!(translated.skipped_non_scalar_metrics, 0);
        assert_eq!(translated.batch.observations.len(), 1);
        let observation = &translated.batch.observations[0];
        assert_eq!(observation.entity.kind, "service_instance");
        assert_eq!(observation.entity.id, "api-17");
        assert_eq!(observation.signal, "system.cpu.utilization");
        assert!(matches!(
            observation.value,
            ObservationValue::Number { value, .. } if (value - 0.75).abs() < f64::EPSILON
        ));
        assert_eq!(observation.observed_at_unix_ms, 2_000);
    }

    #[test]
    fn sum_preserves_integer_and_sum_semantics() {
        let observer = OtlpMetricsObserver::new(
            OtlpMetricsContext::default(),
            request(metric::Data::Sum(Sum {
                data_points: vec![point(number_data_point::Value::AsInt(9))],
                aggregation_temporality: 2,
                is_monotonic: true,
            })),
            2_100,
        );
        let batch = observer.observe_sync(ObservationRequest::default()).unwrap();
        assert!(matches!(batch.observations[0].value, ObservationValue::Integer(9)));
        assert_eq!(
            batch.observations[0].labels.get("otel.sum.monotonic"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn no_recorded_value_is_explicit_unknown_not_fake_zero() {
        let mut missing = point(number_data_point::Value::AsInt(0));
        missing.flags = DataPointFlags::NoRecordedValueMask as u32;
        missing.value = None;
        let translated = translate_metrics_request(
            &request(metric::Data::Gauge(Gauge {
                data_points: vec![missing],
            })),
            &OtlpMetricsContext::default(),
            2_100,
        )
        .unwrap();
        let observation = &translated.batch.observations[0];
        assert_eq!(observation.quality.state, ObservationState::Unknown);
        assert!(matches!(observation.value, ObservationValue::Text(_)));
    }

    #[test]
    fn zero_required_timestamp_fails_closed() {
        let mut invalid = point(number_data_point::Value::AsInt(1));
        invalid.time_unix_nano = 0;
        let result = translate_metrics_request(
            &request(metric::Data::Gauge(Gauge {
                data_points: vec![invalid],
            })),
            &OtlpMetricsContext::default(),
            2_100,
        );
        assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
    }

    #[test]
    fn duplicate_point_attributes_fail_closed() {
        let mut duplicate = point(number_data_point::Value::AsInt(1));
        duplicate.attributes.push(string_kv("cpu", "1"));
        let result = translate_metrics_request(
            &request(metric::Data::Gauge(Gauge {
                data_points: vec![duplicate],
            })),
            &OtlpMetricsContext::default(),
            2_100,
        );
        assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
    }

    #[test]
    fn observer_filters_canonical_output() {
        let observer = OtlpMetricsObserver::new(
            OtlpMetricsContext::default(),
            request(metric::Data::Gauge(Gauge {
                data_points: vec![point(number_data_point::Value::AsDouble(0.75))],
            })),
            2_100,
        );
        let batch = observer
            .observe_sync(ObservationRequest {
                signals: vec!["system.cpu.utilization".into()],
                filters: BTreeMap::from([("cpu".into(), "0".into())]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(batch.observations.len(), 1);
    }
}
