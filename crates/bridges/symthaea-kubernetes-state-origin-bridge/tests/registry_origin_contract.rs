// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use std::sync::Arc;
use symthaea_integration_core::{
    DESIRED_STATE_ORIGIN_ATTRIBUTE, IntegrationId, IntegrationRegistry, StateRole,
};
use symthaea_kubernetes_bridge::{KUBERNETES_INTEGRATION_ID, KubernetesReplayContext};
use symthaea_kubernetes_state_bridge::KubernetesStateReplay;

fn replay() -> KubernetesStateReplay {
    KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[json!({
            "apiVersion":"apps/v1",
            "kind":"Deployment",
            "metadata":{
                "name":"api",
                "namespace":"shop",
                "uid":"dep-1",
                "generation":2
            },
            "spec":{"replicas":2},
            "status":{"replicas":2,"observedGeneration":2}
        })],
        100,
    )
    .unwrap()
}

#[test]
fn registry_rejects_desired_origin_smuggled_onto_observed_state() {
    let replay = replay();
    let mut snapshot = replay.snapshot().clone();
    snapshot
        .assertions
        .iter_mut()
        .find(|assertion| assertion.role == StateRole::Observed)
        .unwrap()
        .attributes
        .insert(DESIRED_STATE_ORIGIN_ATTRIBUTE.into(), "declared".into());

    let mut registry = IntegrationRegistry::new();
    registry
        .register_discoverer(Arc::new(replay.topology().clone()))
        .unwrap();

    assert!(registry
        .admit_state_snapshot(
            &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
            &snapshot,
        )
        .is_err());
}

#[test]
fn registry_rejects_unknown_desired_origin_string() {
    let replay = replay();
    let mut snapshot = replay.snapshot().clone();
    snapshot
        .assertions
        .iter_mut()
        .find(|assertion| assertion.role == StateRole::Desired)
        .unwrap()
        .attributes
        .insert(
            DESIRED_STATE_ORIGIN_ATTRIBUTE.into(),
            "maybe-from-some-controller".into(),
        );

    let mut registry = IntegrationRegistry::new();
    registry
        .register_discoverer(Arc::new(replay.topology().clone()))
        .unwrap();

    assert!(registry
        .admit_state_snapshot(
            &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
            &snapshot,
        )
        .is_err());
}
