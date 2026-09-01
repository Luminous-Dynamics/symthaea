// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-protocol Kubernetes identity proofs.
//!
//! These tests deliberately run identity normalization as an explicit stage.
//! The resolver itself knows no adapter-specific aliases.

use opentelemetry_proto::tonic::{
    collector::metrics::v1::ExportMetricsServiceRequest,
    common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value},
    metrics::v1::{
        Gauge, Metric, NumberDataPoint, ResourceMetrics, ScopeMetrics, metric, number_data_point,
    },
    resource::v1::Resource,
};
use serde_json::json;
use std::sync::Arc;
use symthaea_integration_core::{
    IdentityRequest, IntegrationIdentity, IntegrationRegistry, ResolutionStatus,
    kubernetes_cluster_uid_from_topology, normalize_kubernetes_uid_snapshot,
    resolve_registry_identity_snapshots,
};
use symthaea_kubernetes_bridge::{KubernetesReplayContext, KubernetesReplayDiscoverer};
use symthaea_otlp_bridge::{OtlpMetricsContext, OtlpMetricsObserver};

const CLUSTER_UID: &str = "218fc5a9-a5f1-4b54-aa05-46717d0ab26d";
const OTHER_CLUSTER_UID: &str = "318fc5a9-a5f1-4b54-aa05-46717d0ab26d";
const POD_UID: &str = "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff";

fn string_kv(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.into(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.into())),
        }),
    }
}

fn otlp_request(cluster_uid: &str) -> ExportMetricsServiceRequest {
    ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(Resource {
                attributes: vec![
                    string_kv("k8s.cluster.uid", cluster_uid),
                    string_kv("k8s.pod.uid", POD_UID),
                    string_kv("k8s.pod.name", "api-17"),
                    string_kv("k8s.namespace.name", "shop"),
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
    }
}

fn kubernetes_replay() -> KubernetesReplayDiscoverer {
    KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext {
            cluster_id: "prod-eu-1".into(),
            ..Default::default()
        },
        &[
            json!({
                "apiVersion":"v1",
                "kind":"Namespace",
                "metadata":{"name":"kube-system","uid":CLUSTER_UID}
            }),
            json!({
                "apiVersion":"v1",
                "kind":"Pod",
                "metadata":{"name":"api-17","namespace":"shop","uid":POD_UID}
            }),
        ],
        2_100,
    )
    .unwrap()
}

#[test]
fn same_cluster_and_pod_uid_resolve_across_kubernetes_and_otlp() {
    let kubernetes = Arc::new(kubernetes_replay());
    let otlp = Arc::new(OtlpMetricsObserver::new(
        OtlpMetricsContext::default(),
        otlp_request(CLUSTER_UID),
        2_100,
    ));

    let mut registry = IntegrationRegistry::new();
    registry.register_discoverer(kubernetes.clone()).unwrap();
    registry
        .register_identity_provider(kubernetes.clone())
        .unwrap();
    registry.register_identity_provider(otlp.clone()).unwrap();

    registry
        .admit_discovery_snapshot(&kubernetes.manifest().id, kubernetes.topology())
        .unwrap();
    let cluster_uid = kubernetes_cluster_uid_from_topology(kubernetes.topology())
        .unwrap()
        .expect("kube-system namespace must provide a real cluster UID");
    assert_eq!(cluster_uid, CLUSTER_UID);

    let kubernetes_snapshot = normalize_kubernetes_uid_snapshot(
        &kubernetes
            .identity_snapshot_sync(IdentityRequest::default())
            .unwrap(),
        &cluster_uid,
    )
    .unwrap();
    let otlp_snapshot = normalize_kubernetes_uid_snapshot(
        &otlp
            .identity_snapshot_sync(IdentityRequest::default())
            .unwrap(),
        &cluster_uid,
    )
    .unwrap();

    let kubernetes_pod = kubernetes_snapshot
        .claims
        .iter()
        .find(|claim| claim.subject.kind == "k8s_pod")
        .unwrap();
    let otlp_pod = otlp_snapshot
        .claims
        .iter()
        .find(|claim| claim.subject.kind == "k8s_pod")
        .unwrap();
    assert_eq!(kubernetes_pod.identifier, otlp_pod.identifier);
    assert_ne!(kubernetes_pod.subject, otlp_pod.subject);

    let resolved = resolve_registry_identity_snapshots(
        &registry,
        &[kubernetes_snapshot, otlp_snapshot],
        2_100,
    )
    .unwrap();
    let proposal = resolved
        .proposals
        .iter()
        .find(|proposal| {
            proposal.left.kind == "k8s_pod" && proposal.right.kind == "k8s_pod"
        })
        .expect("shared scoped Pod UID should generate a resolution proposal");
    assert_eq!(proposal.status, ResolutionStatus::StrongCandidateSame);
    assert_eq!(proposal.identifier_matches.len(), 1);
}

#[test]
fn identical_pod_uid_in_different_cluster_scope_does_not_resolve() {
    let kubernetes = Arc::new(kubernetes_replay());
    let otlp = Arc::new(OtlpMetricsObserver::new(
        OtlpMetricsContext::default(),
        otlp_request(OTHER_CLUSTER_UID),
        2_100,
    ));

    let mut registry = IntegrationRegistry::new();
    registry
        .register_identity_provider(kubernetes.clone())
        .unwrap();
    registry.register_identity_provider(otlp.clone()).unwrap();

    let kubernetes_snapshot = normalize_kubernetes_uid_snapshot(
        &kubernetes
            .identity_snapshot_sync(IdentityRequest::default())
            .unwrap(),
        CLUSTER_UID,
    )
    .unwrap();
    let otlp_snapshot = normalize_kubernetes_uid_snapshot(
        &otlp
            .identity_snapshot_sync(IdentityRequest::default())
            .unwrap(),
        OTHER_CLUSTER_UID,
    )
    .unwrap();

    let resolved = resolve_registry_identity_snapshots(
        &registry,
        &[kubernetes_snapshot, otlp_snapshot],
        2_100,
    )
    .unwrap();
    assert!(
        resolved
            .proposals
            .iter()
            .all(|proposal| !(proposal.left.kind == "k8s_pod"
                && proposal.right.kind == "k8s_pod")),
        "the same Pod UID string under different real cluster UIDs must not correlate"
    );
}
