// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use std::sync::Arc;
use symthaea_integration_core::{
    IntegrationId, IntegrationRegistry, StateAssessmentStatus, StateComparisonPolicy, StateHistory,
    StateLimits, TemporalStatePolicy, TemporalStateStatus, assess_state_dimension,
    assess_state_dimension_temporally, assess_state_dimension_with_history,
};
use symthaea_kubernetes_bridge::{KUBERNETES_INTEGRATION_ID, KubernetesReplayContext};
use symthaea_kubernetes_state_bridge::KubernetesStateReplay;

fn deployment() -> serde_json::Value {
    json!({
        "apiVersion":"apps/v1",
        "kind":"Deployment",
        "metadata":{
            "name":"api",
            "namespace":"shop",
            "uid":"deployment-uid",
            "generation":9,
            "resourceVersion":"42"
        },
        "spec":{"replicas":5},
        "status":{
            "replicas":3,
            "readyReplicas":2,
            "observedGeneration":8
        }
    })
}

fn registered_registry(replay: &KubernetesStateReplay) -> IntegrationRegistry {
    let mut registry = IntegrationRegistry::new();
    registry
        .register_discoverer(Arc::new(replay.topology().clone()))
        .unwrap();
    registry
}

#[test]
fn registered_kubernetes_source_admits_state_before_drift_assessment() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();

    let registry = registered_registry(&replay);
    let id = IntegrationId::new(KUBERNETES_INTEGRATION_ID);
    registry.admit_state_snapshot(&id, replay.snapshot()).unwrap();

    let entity = replay.snapshot().assertions[0].subject.clone();
    let assessment = assess_state_dimension(
        &replay.snapshot().assertions,
        &entity,
        "workload.replicas",
        100,
        StateComparisonPolicy::Exact,
    )
    .unwrap();
    assert_eq!(assessment.status, StateAssessmentStatus::Drift);
}

#[test]
fn kubernetes_replay_does_not_invent_rollout_age() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let entity = replay.snapshot().assertions[0].subject.clone();
    let assessment = assess_state_dimension_temporally(
        &replay.snapshot().assertions,
        &entity,
        "workload.replicas",
        100,
        TemporalStatePolicy {
            comparison: StateComparisonPolicy::Exact,
            max_desired_age_ms: Some(1_000),
            max_observed_age_ms: Some(1_000),
            convergence_window_ms: 60_000,
        },
    )
    .unwrap();
    assert_eq!(assessment.instantaneous.status, StateAssessmentStatus::Drift);
    assert_eq!(assessment.status, TemporalStateStatus::DriftAgeUnknown);
    assert_eq!(assessment.drift_age_ms, None);
}

#[test]
fn replay_history_supports_persistence_without_claiming_proof() {
    let first = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let latest = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        300,
    )
    .unwrap();
    let entity = latest.snapshot().assertions[0].subject.clone();
    let history = StateHistory {
        integration_id: KUBERNETES_INTEGRATION_ID.into(),
        snapshots: vec![first.snapshot().clone(), latest.snapshot().clone()],
    };

    let registry = registered_registry(&latest);
    let id = IntegrationId::new(KUBERNETES_INTEGRATION_ID);
    registry.admit_state_history(&id, &history).unwrap();

    let assessment = assess_state_dimension_with_history(
        &history,
        &entity,
        "workload.replicas",
        300,
        TemporalStatePolicy {
            comparison: StateComparisonPolicy::Exact,
            max_desired_age_ms: Some(1_000),
            max_observed_age_ms: Some(1_000),
            convergence_window_ms: 150,
        },
    )
    .unwrap();

    assert_eq!(assessment.current.status, TemporalStateStatus::DriftAgeUnknown);
    assert_eq!(
        assessment.continuously_observed_desired_age_lower_bound_ms,
        Some(200)
    );
    assert_eq!(
        assessment.continuously_observed_drift_age_lower_bound_ms,
        Some(200)
    );
    assert_eq!(
        assessment.drift_continuity.as_ref().unwrap().consecutive_snapshots,
        2
    );
    assert!(assessment.sampled_persistence_supported);
    assert!(!assessment.persistent_drift_proven);
}

#[test]
fn future_dated_assertion_is_rejected_at_snapshot_admission() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let registry = registered_registry(&replay);
    let id = IntegrationId::new(KUBERNETES_INTEGRATION_ID);
    let mut snapshot = replay.snapshot().clone();
    snapshot.assertions[0].observed_at_unix_ms = 101;

    assert!(registry.admit_state_snapshot(&id, &snapshot).is_err());
}

#[test]
fn future_dated_assertion_is_rejected_inside_history_admission() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let registry = registered_registry(&replay);
    let id = IntegrationId::new(KUBERNETES_INTEGRATION_ID);
    let mut snapshot = replay.snapshot().clone();
    snapshot.assertions[0].observed_at_unix_ms = 101;
    let history = StateHistory {
        integration_id: KUBERNETES_INTEGRATION_ID.into(),
        snapshots: vec![snapshot],
    };

    assert!(registry.admit_state_history(&id, &history).is_err());
}

#[test]
fn unregistered_state_source_is_rejected() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let registry = IntegrationRegistry::new();
    assert!(
        registry
            .admit_state_snapshot(
                &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
                replay.snapshot(),
            )
            .is_err()
    );
}

#[test]
fn unregistered_state_history_is_rejected() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let history = StateHistory {
        integration_id: KUBERNETES_INTEGRATION_ID.into(),
        snapshots: vec![replay.snapshot().clone()],
    };
    let registry = IntegrationRegistry::new();
    assert!(
        registry
            .admit_state_history(&IntegrationId::new(KUBERNETES_INTEGRATION_ID), &history)
            .is_err()
    );
}

#[test]
fn central_state_budget_rejects_oversized_snapshot() {
    let replay = KubernetesStateReplay::from_objects(
        KubernetesReplayContext::default(),
        &[deployment()],
        100,
    )
    .unwrap();
    let registry = registered_registry(&replay);
    let limits = StateLimits {
        max_assertions: 1,
        ..Default::default()
    };
    assert!(
        registry
            .admit_state_snapshot_with_limits(
                &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
                replay.snapshot(),
                &limits,
            )
            .is_err()
    );
}
