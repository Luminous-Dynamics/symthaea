// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use symthaea_integration_core::{
    StateAssessmentStatus, StateComparisonPolicy, assess_state_dimension,
};
use symthaea_kubernetes_bridge::KubernetesReplayContext;
use symthaea_kubernetes_endpointslice_bridge::augment_endpoint_slices;
use symthaea_kubernetes_state_bridge::KubernetesStateReplay;
use symthaea_kubernetes_workload_state_bridge::augment_workload_state;

#[test]
fn replicas_in_sync_does_not_imply_endpoint_readiness() {
    let at = 100;
    let objects = vec![
        json!({
            "apiVersion":"v1","kind":"Namespace",
            "metadata":{"name":"shop","uid":"ns-1"}
        }),
        json!({
            "apiVersion":"apps/v1","kind":"Deployment",
            "metadata":{
                "name":"api","namespace":"shop","uid":"dep-1","generation":3
            },
            "spec":{"replicas":3},
            "status":{
                "replicas":3,"readyReplicas":2,"availableReplicas":2,
                "updatedReplicas":3,"observedGeneration":3,
                "conditions":[{
                    "type":"Available","status":"False",
                    "reason":"MinimumReplicasUnavailable"
                }]
            }
        }),
        json!({
            "apiVersion":"v1","kind":"Pod",
            "metadata":{"name":"api-0","namespace":"shop","uid":"pod-0","labels":{"app":"api"}}
        }),
        json!({
            "apiVersion":"v1","kind":"Pod",
            "metadata":{"name":"api-1","namespace":"shop","uid":"pod-1","labels":{"app":"api"}}
        }),
        json!({
            "apiVersion":"v1","kind":"Pod",
            "metadata":{"name":"api-2","namespace":"shop","uid":"pod-2","labels":{"app":"api"}}
        }),
        json!({
            "apiVersion":"v1","kind":"Service",
            "metadata":{"name":"api","namespace":"shop","uid":"svc-1"},
            "spec":{"selector":{"app":"api"}}
        }),
    ];
    let context = KubernetesReplayContext {
        cluster_id: "cluster-a".into(),
        source_confidence: 0.99,
        ..Default::default()
    };
    let base = KubernetesStateReplay::from_objects(context, &objects, at).unwrap();
    let state = augment_workload_state(&base, &objects, at).unwrap();

    let deployment = base
        .topology()
        .topology()
        .entities
        .iter()
        .find(|entity| {
            entity.attributes.get("k8s.kind").map(String::as_str) == Some("Deployment")
                && entity.attributes.get("k8s.name").map(String::as_str) == Some("api")
        })
        .unwrap()
        .entity
        .clone();
    let replicas = assess_state_dimension(
        &state.assertions,
        &deployment,
        "workload.replicas",
        at,
        StateComparisonPolicy::Exact,
    )
    .unwrap();
    assert_eq!(replicas.status, StateAssessmentStatus::InSync);

    let slice = json!({
        "apiVersion":"discovery.k8s.io/v1","kind":"EndpointSlice",
        "metadata":{
            "name":"api-v4","namespace":"shop","uid":"slice-1",
            "labels":{"kubernetes.io/service-name":"api"}
        },
        "addressType":"IPv4",
        "ports":[{"name":"http","protocol":"TCP","port":8080}],
        "endpoints":[
            {
                "addresses":["10.0.0.10"],
                "conditions":{"ready":true,"serving":true,"terminating":false},
                "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-0","uid":"pod-0"}
            },
            {
                "addresses":["10.0.0.11"],
                "conditions":{"ready":true,"serving":true,"terminating":false},
                "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-1","uid":"pod-1"}
            },
            {
                "addresses":["10.0.0.12"],
                "conditions":{"ready":false,"serving":true,"terminating":false},
                "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-2","uid":"pod-2"}
            }
        ]
    });
    let topology = augment_endpoint_slices(base.topology(), &[slice], at).unwrap();
    let endpoints = topology
        .entities
        .iter()
        .filter(|entity| {
            entity
                .attributes
                .get("symthaea.k8s.role")
                .map(String::as_str)
                == Some("endpoint_membership")
        })
        .collect::<Vec<_>>();
    assert_eq!(endpoints.len(), 3);
    assert_eq!(
        endpoints
            .iter()
            .filter(|entity| {
                entity.attributes.get("k8s.endpoint.ready").map(String::as_str) == Some("true")
            })
            .count(),
        2
    );
    assert_eq!(
        endpoints
            .iter()
            .filter(|entity| {
                entity.attributes.get("k8s.endpoint.serving").map(String::as_str) == Some("true")
            })
            .count(),
        3
    );

    // These facts intentionally coexist: the controller has observed all three
    // requested replicas, but endpoint readiness exposes only two traffic-ready
    // backends. No layer is allowed to synthesize one fact from the other.
    assert_eq!(replicas.status, StateAssessmentStatus::InSync);
}
