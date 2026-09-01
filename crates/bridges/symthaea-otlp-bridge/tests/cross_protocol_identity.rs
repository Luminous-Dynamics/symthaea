// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-protocol proof that source-local Prometheus and OTLP entities can be
//! resolved without making their local `EntityRef`s identical.

use opentelemetry_proto::tonic::{
    collector::metrics::v1::ExportMetricsServiceRequest,
    common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value},
    metrics::v1::{Gauge, Metric, NumberDataPoint, ResourceMetrics, ScopeMetrics, metric, number_data_point},
    resource::v1::Resource,
};
use symthaea_integration_core::{ResolutionStatus, resolve_identity_claims};
use symthaea_otlp_bridge::{OtlpMetricsContext, translate_metrics_request};
use symthaea_prometheus_bridge::{
    PrometheusFixtureContext, PrometheusIdentityMapping, PrometheusTextObserver,
};

fn string_kv(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.into(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.into())),
        }),
    }
}

#[test]
fn otlp_and_prometheus_compatibility_resolve_one_service_instance() {
    let otlp_request = ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(Resource {
                attributes: vec![
                    string_kv("service.namespace", "shop"),
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
                    description: String::new(),
                    unit: "1".into(),
                    metadata: vec![],
                    data: Some(metric::Data::Gauge(Gauge {
                        data_points: vec![NumberDataPoint {
                            attributes: vec![],
                            start_time_unix_nano: 1_000_000_000,
                            time_unix_nano: 2_000_000_000,
                            exemplars: vec![],
                            flags: 0,
                            value: Some(number_data_point::Value::AsDouble(0.75)),
                        }],
                    })),
                }],
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    };
    let otlp = translate_metrics_request(
        &otlp_request,
        &OtlpMetricsContext::default(),
        2_100,
    )
    .unwrap();
    assert_eq!(otlp.identity_claims.len(), 1);

    let prometheus = PrometheusTextObserver::from_text(
        PrometheusFixtureContext {
            source_confidence: 0.95,
            identity_mapping: PrometheusIdentityMapping::OtelPrometheusCompatibility,
            ..PrometheusFixtureContext::default()
        },
        "# TYPE up gauge\nup{job=\"shop/api\",instance=\"api-17\"} 1\n",
        2_100,
    )
    .unwrap();
    assert_eq!(prometheus.identity_claims().len(), 1);

    let claims = otlp
        .identity_claims
        .iter()
        .chain(prometheus.identity_claims())
        .cloned()
        .collect::<Vec<_>>();
    let resolved = resolve_identity_claims(&claims, &[], 2_100).unwrap();

    assert_eq!(resolved.proposals.len(), 1);
    let proposal = &resolved.proposals[0];
    assert_eq!(proposal.status, ResolutionStatus::StrongCandidateSame);
    assert_ne!(proposal.left, proposal.right);
    assert_eq!(proposal.identifier_matches.len(), 1);
    assert!(proposal.identifier_matches[0]
        .canonical_identifier
        .starts_with("otel.service.instance.triplet|"));
}
