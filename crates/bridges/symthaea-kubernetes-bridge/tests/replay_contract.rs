// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use symthaea_integration_core::{
    DiscoveryRequest, IntegrationError, IntegrationId, IntegrationRegistry, TopologyLimits,
};
use symthaea_kubernetes_bridge::{
    KUBERNETES_INTEGRATION_ID, KubernetesReplayContext, KubernetesReplayDiscoverer,
};
use std::sync::Arc;

#[test]
fn kubernetes_list_documents_are_replayed_as_individual_objects() {
    let replay = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[json!({
            "apiVersion":"v1",
            "kind":"PodList",
            "items":[
                {"apiVersion":"v1","kind":"Pod","metadata":{"name":"a","namespace":"ns","uid":"a"}},
                {"apiVersion":"v1","kind":"Pod","metadata":{"name":"b","namespace":"ns","uid":"b"}}
            ]
        })],
        100,
    )
    .unwrap();

    assert_eq!(
        replay
            .topology()
            .entities
            .iter()
            .filter(|entity| entity.entity.kind == "k8s_pod")
            .count(),
        2
    );
}

#[test]
fn malformed_non_string_label_fails_closed() {
    let result = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[json!({
            "apiVersion":"v1","kind":"Pod",
            "metadata":{
                "name":"bad","namespace":"ns","uid":"bad",
                "labels":{"replicas":3}
            }
        })],
        100,
    );
    assert!(matches!(result, Err(IntegrationError::Protocol(_))));
}

#[test]
fn duplicate_kind_name_with_different_uid_is_rejected() {
    let result = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[
            json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{"name":"same","namespace":"ns","uid":"uid-a"}
            }),
            json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{"name":"same","namespace":"ns","uid":"uid-b"}
            }),
        ],
        100,
    );
    assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
}

#[test]
fn unknown_crd_kind_survives_as_generic_object_without_schema_guessing() {
    let replay = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[json!({
            "apiVersion":"example.io/v1","kind":"Widget",
            "metadata":{"name":"w1","namespace":"ns","uid":"widget-1"},
            "spec":{"opaque":{"future":"schema"}}
        })],
        100,
    )
    .unwrap();

    let widget = replay
        .topology()
        .entities
        .iter()
        .find(|entity| entity.attributes.get("k8s.kind") == Some(&"Widget".to_string()))
        .unwrap();
    assert_eq!(widget.entity.kind, "k8s_object");
    assert!(!widget.attributes.values().any(|value| value == "schema"));
}

#[test]
fn discovery_kind_filter_keeps_snapshot_structurally_closed() {
    let replay = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[
            json!({
                "apiVersion":"v1","kind":"Node",
                "metadata":{"name":"node","uid":"node-1"}
            }),
            json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{"name":"pod","namespace":"ns","uid":"pod-1"},
                "spec":{"nodeName":"node"}
            }),
        ],
        100,
    )
    .unwrap();

    let filtered = replay
        .discover_sync(DiscoveryRequest {
            entity_kinds: vec!["k8s_pod".into()],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(filtered.entities.len(), 1);
    assert!(filtered.relations.is_empty());
    assert!(filtered.validate().is_ok());
}

#[test]
fn registry_topology_budget_applies_to_real_kubernetes_adapter_output() {
    let replay = Arc::new(
        KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext::default(),
            &[json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{"name":"pod","namespace":"ns","uid":"pod-1"}
            })],
            100,
        )
        .unwrap(),
    );
    let mut registry = IntegrationRegistry::new();
    registry.register_discoverer(replay.clone()).unwrap();

    let limits = TopologyLimits {
        max_entities: 1,
        ..Default::default()
    };
    // Cluster + Pod + placeholder Namespace necessarily exceeds one entity.
    assert!(matches!(
        registry.admit_discovery_snapshot_with_limits(
            &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
            replay.topology(),
            &limits,
        ),
        Err(IntegrationError::InvalidOutput(_))
    ));
}
