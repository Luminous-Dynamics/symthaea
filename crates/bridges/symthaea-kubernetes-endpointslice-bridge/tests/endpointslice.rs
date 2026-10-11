// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::Arc;
use symthaea_integration_core::{IntegrationId, IntegrationRegistry, RelationKind};
use symthaea_kubernetes_bridge::{
    KUBERNETES_INTEGRATION_ID, KubernetesReplayContext, KubernetesReplayDiscoverer,
    integration_manifest,
};
use symthaea_kubernetes_endpointslice_bridge::augment_endpoint_slices;

fn base(at: u64) -> KubernetesReplayDiscoverer {
    KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext {
            cluster_id: "cluster-a".into(),
            source_confidence: 0.99,
            ..Default::default()
        },
        &[
            json!({"apiVersion":"v1","kind":"Namespace","metadata":{"name":"shop","uid":"ns-1"}}),
            json!({"apiVersion":"v1","kind":"Node","metadata":{"name":"node-a","uid":"node-1"}}),
            json!({"apiVersion":"v1","kind":"Pod","metadata":{"name":"api-pod","namespace":"shop","uid":"pod-1"}}),
            json!({"apiVersion":"v1","kind":"Service","metadata":{"name":"api","namespace":"shop","uid":"svc-1"}}),
        ],
        at,
    )
    .unwrap()
}

fn slice() -> Value {
    json!({
        "apiVersion":"discovery.k8s.io/v1",
        "kind":"EndpointSlice",
        "metadata":{
            "name":"api-abc","namespace":"shop","uid":"slice-1",
            "labels":{
                "kubernetes.io/service-name":"api",
                "endpointslice.kubernetes.io/managed-by":"endpointslice-controller.k8s.io"
            }
        },
        "addressType":"IPv4",
        "ports":[{"name":"http","protocol":"TCP","port":8080}],
        "endpoints":[{
            "addresses":["10.1.2.3"],
            "conditions":{"ready":true,"serving":true,"terminating":false},
            "nodeName":"node-a","zone":"zone-a",
            "targetRef":{
                "apiVersion":"v1","kind":"Pod","name":"api-pod",
                "namespace":"shop","uid":"pod-1"
            },
            "hints":{"forZones":[{"name":"zone-a"}]}
        }]
    })
}

fn role_count(snapshot: &symthaea_integration_core::DiscoverySnapshot, role: &str) -> usize {
    snapshot
        .entities
        .iter()
        .filter(|entity| {
            entity
                .attributes
                .get("symthaea.k8s.role")
                .map(String::as_str)
                == Some(role)
        })
        .count()
}

#[test]
fn adds_service_slice_endpoint_target_and_node_topology() {
    let base = base(100);
    let snapshot = augment_endpoint_slices(&base, &[slice()], 100).unwrap();
    snapshot.validate().unwrap();
    assert_eq!(role_count(&snapshot, "endpoint_slice"), 1);
    assert_eq!(role_count(&snapshot, "endpoint_membership"), 1);
    assert!(snapshot.relations.iter().any(|relation| {
        relation.kind == RelationKind::Other("EndpointSliceFor".into())
    }));
    assert!(snapshot.relations.iter().any(|relation| {
        relation.kind == RelationKind::Other("Targets".into()) && relation.to.kind == "k8s_pod"
    }));
    assert!(snapshot.relations.iter().any(|relation| {
        relation.kind == RelationKind::HostedOn && relation.to.kind == "k8s_node"
    }));
}

#[test]
fn nil_conditions_preserve_kubernetes_defaults_and_explicitness() {
    let base = base(100);
    let mut slice = slice();
    slice["endpoints"][0]["conditions"] = json!({});
    let snapshot = augment_endpoint_slices(&base, &[slice], 100).unwrap();
    let endpoint = snapshot
        .entities
        .iter()
        .find(|entity| {
            entity
                .attributes
                .get("symthaea.k8s.role")
                .map(String::as_str)
                == Some("endpoint_membership")
        })
        .unwrap();
    assert_eq!(
        endpoint.attributes.get("k8s.endpoint.ready").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        endpoint
            .attributes
            .get("k8s.endpoint.ready.explicit")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        endpoint
            .attributes
            .get("k8s.endpoint.serving")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        endpoint
            .attributes
            .get("k8s.endpoint.terminating")
            .map(String::as_str),
        Some("false")
    );
}

#[test]
fn underspecified_target_does_not_wildcard_match_other_namespace() {
    let base = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[
            json!({"apiVersion":"v1","kind":"Namespace","metadata":{"name":"shop","uid":"ns-1"}}),
            json!({"apiVersion":"v1","kind":"Namespace","metadata":{"name":"other","uid":"ns-2"}}),
            json!({"apiVersion":"v1","kind":"Pod","metadata":{"name":"api-pod","namespace":"other","uid":"other-pod"}}),
            json!({"apiVersion":"v1","kind":"Service","metadata":{"name":"api","namespace":"shop","uid":"svc-1"}}),
        ],
        100,
    )
    .unwrap();
    let mut slice = slice();
    slice["endpoints"][0]["targetRef"] = json!({"kind":"Pod","name":"api-pod"});
    let snapshot = augment_endpoint_slices(&base, &[slice], 100).unwrap();
    assert!(snapshot.entities.iter().any(|entity| {
        entity
            .attributes
            .get("symthaea.k8s.role")
            .map(String::as_str)
            == Some("target_reference")
    }));
    assert!(!snapshot.relations.iter().any(|relation| {
        relation.kind == RelationKind::Other("Targets".into()) && relation.to.id == "other-pod"
    }));
}

#[test]
fn uid_kind_conflict_fails_closed() {
    let base = base(100);
    let mut slice = slice();
    slice["endpoints"][0]["targetRef"]["kind"] = json!("Service");
    assert!(augment_endpoint_slices(&base, &[slice], 100).is_err());
}

#[test]
fn duplicate_addresses_wrong_family_and_duplicate_membership_fail_closed() {
    let base = base(100);
    let mut duplicate_address = slice();
    duplicate_address["endpoints"][0]["addresses"] = json!(["10.1.2.3", "10.1.2.3"]);
    assert!(augment_endpoint_slices(&base, &[duplicate_address], 100).is_err());

    let mut wrong_family = slice();
    wrong_family["endpoints"][0]["addresses"] = json!(["2001:db8::1"]);
    assert!(augment_endpoint_slices(&base, &[wrong_family], 100).is_err());

    let mut duplicate_endpoint = slice();
    let endpoint = duplicate_endpoint["endpoints"][0].clone();
    duplicate_endpoint["endpoints"] = json!([endpoint.clone(), endpoint]);
    assert!(augment_endpoint_slices(&base, &[duplicate_endpoint], 100).is_err());
}

#[test]
fn existing_generic_endpointslice_object_is_enriched_not_duplicated() {
    let slice = slice();
    let base = KubernetesReplayDiscoverer::from_objects(
        KubernetesReplayContext::default(),
        &[
            json!({"apiVersion":"v1","kind":"Namespace","metadata":{"name":"shop","uid":"ns-1"}}),
            slice.clone(),
        ],
        100,
    )
    .unwrap();
    let before = base
        .topology()
        .entities
        .iter()
        .filter(|entity| {
            entity.attributes.get("k8s.uid").map(String::as_str) == Some("slice-1")
        })
        .count();
    let snapshot = augment_endpoint_slices(&base, &[slice], 100).unwrap();
    let after = snapshot
        .entities
        .iter()
        .filter(|entity| {
            entity.attributes.get("k8s.uid").map(String::as_str) == Some("slice-1")
        })
        .count();
    assert_eq!(before, 1);
    assert_eq!(after, 1);
}

#[test]
fn emitted_entity_kinds_stay_inside_declared_kubernetes_manifest() {
    let base = base(100);
    let snapshot = augment_endpoint_slices(&base, &[slice()], 100).unwrap();
    let declared = integration_manifest()
        .entity_kinds
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert!(
        snapshot
            .entities
            .iter()
            .all(|entity| declared.contains(&entity.entity.kind))
    );
}

#[test]
fn api_max_port_and_address_shapes_fit_default_topology_admission() {
    let base = Arc::new(base(100));
    let mut slice = slice();
    slice["ports"] = Value::Array(
        (0..100)
            .map(|index| {
                json!({
                    "name":format!("p{index}"),
                    "protocol":"TCP",
                    "port":1000 + index
                })
            })
            .collect(),
    );
    slice["endpoints"][0]["addresses"] = Value::Array(
        (1..=100)
            .map(|index| Value::String(format!("10.1.2.{index}")))
            .collect(),
    );
    slice["endpoints"][0]["hints"] = json!({
        "forZones": (0..8).map(|i| json!({"name":format!("zone-{i}")})).collect::<Vec<_>>(),
        "forNodes": (0..8).map(|i| json!({"name":format!("node-{i}")})).collect::<Vec<_>>()
    });

    let snapshot = augment_endpoint_slices(base.as_ref(), &[slice], 100).unwrap();
    let mut registry = IntegrationRegistry::new();
    registry.register_discoverer(base).unwrap();
    registry
        .admit_discovery_snapshot(
            &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
            &snapshot,
        )
        .unwrap();
}

#[test]
fn same_capture_time_is_required_and_list_replay_does_not_claim_completeness() {
    let base = base(100);
    assert!(augment_endpoint_slices(&base, &[slice()], 101).is_err());

    let list = json!({
        "apiVersion":"discovery.k8s.io/v1",
        "kind":"EndpointSliceList",
        "items":[slice()]
    });
    let snapshot = augment_endpoint_slices(&base, &[list], 100).unwrap();
    assert_eq!(role_count(&snapshot, "endpoint_slice"), 1);
    // This crate exposes no Discoverer implementation, so it cannot independently
    // claim the core `discover.snapshot.complete` capability.
}

#[test]
fn fqdn_address_type_is_preserved_but_marked_semantically_undefined() {
    let base = base(100);
    let mut slice = slice();
    slice["addressType"] = json!("FQDN");
    slice["endpoints"][0]["addresses"] = json!(["example.internal"]);
    let snapshot = augment_endpoint_slices(&base, &[slice], 100).unwrap();
    let slice = snapshot
        .entities
        .iter()
        .find(|entity| {
            entity
                .attributes
                .get("symthaea.k8s.role")
                .map(String::as_str)
                == Some("endpoint_slice")
        })
        .unwrap();
    assert_eq!(
        slice
            .attributes
            .get("k8s.endpointslice.address_semantics")
            .map(String::as_str),
        Some("deprecated_undefined")
    );
}
