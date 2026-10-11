// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative Kubernetes desired-state provenance normalization.
//!
//! This is a pure semantic augmenter over an existing
//! `kubernetes-object-replay` `StateSnapshot`; it is not another source and it
//! has no live Kubernetes authority. Only origins that can be proven from the
//! Kubernetes field semantics are upgraded automatically. In particular,
//! `spec.*` does **not** imply `Declared`: a server-returned object may contain
//! API-defaulted values, so replay alone cannot distinguish authored intent from
//! materialized defaults.

#![forbid(unsafe_code)]

use blake3::Hasher;
use symthaea_integration_core::{
    DESIRED_STATE_ORIGIN_ATTRIBUTE, DesiredStateOrigin, IntegrationError, StateAssertion,
    StateRole, StateSnapshot, validate_state_snapshot_origins,
};
use symthaea_kubernetes_bridge::KUBERNETES_INTEGRATION_ID;

pub const ORIGIN_NORMALIZATION_PARENT_ATTRIBUTE: &str =
    "symthaea.desired_origin.parent_assertion_id";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KubernetesStateOriginReport {
    pub total_assertions: usize,
    pub desired_assertions: usize,
    pub observed_assertions: usize,
    pub explicit_origins: usize,
    pub normalized_system_derived: usize,
    pub normalized_controller_derived: usize,
    /// Desired assertions whose source does not prove a stronger origin.
    pub unresolved_desired: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KubernetesStateOriginNormalization {
    pub snapshot: StateSnapshot,
    pub report: KubernetesStateOriginReport,
}

/// Upgrade only desired-state origins that are provable from Kubernetes API
/// semantics, preserving all other desired assertions as explicit epistemic
/// unknowns (`Unspecified`).
///
/// Transformations derive a new assertion ID and retain the parent assertion ID
/// so origin enrichment cannot silently change semantics under the same ID.
pub fn normalize_kubernetes_state_origins(
    input: &StateSnapshot,
) -> Result<KubernetesStateOriginNormalization, IntegrationError> {
    if input.integration_id != KUBERNETES_INTEGRATION_ID {
        return Err(IntegrationError::InvalidRequest(format!(
            "Kubernetes state-origin normalization requires `{KUBERNETES_INTEGRATION_ID}` state, got `{}`",
            input.integration_id
        )));
    }
    input
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    validate_state_snapshot_origins(input).map_err(|error| {
        IntegrationError::InvalidOutput(format!(
            "Kubernetes state origin input violates typed origin contract: {error}"
        ))
    })?;

    let mut snapshot = input.clone();
    let mut report = KubernetesStateOriginReport {
        total_assertions: snapshot.assertions.len(),
        desired_assertions: 0,
        observed_assertions: 0,
        explicit_origins: 0,
        normalized_system_derived: 0,
        normalized_controller_derived: 0,
        unresolved_desired: 0,
    };

    for assertion in &mut snapshot.assertions {
        match assertion.role {
            StateRole::Observed => {
                report.observed_assertions += 1;
                // Input validation already rejects desired-origin metadata here.
            }
            StateRole::Desired => {
                report.desired_assertions += 1;
                let evidence = assertion
                    .desired_state_origin()
                    .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?
                    .expect("desired assertions always return origin evidence");
                let expected = provable_origin(assertion);

                if evidence.explicit && evidence.origin != DesiredStateOrigin::Unspecified {
                    if let Some(expected) = expected {
                        if evidence.origin != expected {
                            return Err(IntegrationError::InvalidOutput(format!(
                                "Kubernetes desired assertion `{}` claims origin `{}` but field `{}` proves `{}`",
                                assertion.assertion_id,
                                evidence.origin.as_str(),
                                assertion
                                    .attributes
                                    .get("k8s.field")
                                    .map(String::as_str)
                                    .unwrap_or("<missing>"),
                                expected.as_str()
                            )));
                        }
                    }
                    report.explicit_origins += 1;
                    continue;
                }

                match expected {
                    Some(origin) => {
                        transform_origin(assertion, origin)?;
                        match origin {
                            DesiredStateOrigin::SystemDerived => {
                                report.normalized_system_derived += 1;
                            }
                            DesiredStateOrigin::ControllerDerived => {
                                report.normalized_controller_derived += 1;
                            }
                            _ => unreachable!("provable Kubernetes mapping returned unexpected origin"),
                        }
                    }
                    None => {
                        // `spec.*` is intentionally left unresolved: an API
                        // response may contain a server-defaulted value.
                        report.unresolved_desired += 1;
                    }
                }
            }
        }
    }

    snapshot
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    validate_state_snapshot_origins(&snapshot).map_err(|error| {
        IntegrationError::InvalidOutput(format!(
            "Kubernetes state origin output violates typed origin contract: {error}"
        ))
    })?;

    Ok(KubernetesStateOriginNormalization { snapshot, report })
}

fn provable_origin(assertion: &StateAssertion) -> Option<DesiredStateOrigin> {
    match assertion.attributes.get("k8s.field").map(String::as_str) {
        // Kubernetes metadata.generation is maintained by the API machinery as
        // the desired-state generation marker. It is not user-authored value.
        Some("metadata.generation") => Some(DesiredStateOrigin::SystemDerived),
        // DaemonSet computes this target from node eligibility.
        Some("status.desiredNumberScheduled") => Some(DesiredStateOrigin::ControllerDerived),
        // Presence beneath spec is not enough to distinguish authored intent
        // from an API-defaulted value in a returned object.
        _ => None,
    }
}

fn transform_origin(
    assertion: &mut StateAssertion,
    origin: DesiredStateOrigin,
) -> Result<(), IntegrationError> {
    let parent_assertion_id = assertion.assertion_id.clone();
    assertion
        .set_desired_state_origin(origin)
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    assertion.attributes.insert(
        ORIGIN_NORMALIZATION_PARENT_ATTRIBUTE.into(),
        parent_assertion_id.clone(),
    );
    assertion.assertion_id = normalized_assertion_id(&parent_assertion_id, origin);
    Ok(())
}

fn normalized_assertion_id(parent_assertion_id: &str, origin: DesiredStateOrigin) -> String {
    let mut hasher = Hasher::new();
    feed(&mut hasher, "symthaea-state-origin-normalization-v1");
    feed(&mut hasher, parent_assertion_id);
    feed(&mut hasher, origin.as_str());
    format!("state-origin:{}", hasher.finalize().to_hex())
}

fn feed(hasher: &mut Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use symthaea_integration_core::{
        IntegrationId, IntegrationRegistry, StateAssessmentStatus, StateComparisonPolicy,
        assess_state_dimension,
    };
    use symthaea_kubernetes_bridge::KubernetesReplayContext;
    use symthaea_kubernetes_state_bridge::KubernetesStateReplay;
    use symthaea_kubernetes_workload_state_bridge::augment_workload_state;

    fn deployment() -> serde_json::Value {
        json!({
            "apiVersion":"apps/v1",
            "kind":"Deployment",
            "metadata":{
                "name":"api",
                "namespace":"shop",
                "uid":"dep-1",
                "generation":7,
                "resourceVersion":"42"
            },
            "spec":{"replicas":3},
            "status":{"replicas":3,"observedGeneration":7}
        })
    }

    #[test]
    fn spec_replicas_remains_unspecified_but_generation_is_system_derived() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment()],
            100,
        )
        .unwrap();
        let normalized = normalize_kubernetes_state_origins(replay.snapshot()).unwrap();

        let replicas = normalized
            .snapshot
            .assertions
            .iter()
            .find(|assertion| {
                assertion.role == StateRole::Desired
                    && assertion.dimension == "workload.replicas"
            })
            .unwrap();
        let replica_origin = replicas.desired_state_origin().unwrap().unwrap();
        assert_eq!(replica_origin.origin, DesiredStateOrigin::Unspecified);
        assert!(!replica_origin.explicit);

        let generation = normalized
            .snapshot
            .assertions
            .iter()
            .find(|assertion| {
                assertion.role == StateRole::Desired
                    && assertion.dimension == "controller.generation"
            })
            .unwrap();
        let generation_origin = generation.desired_state_origin().unwrap().unwrap();
        assert_eq!(generation_origin.origin, DesiredStateOrigin::SystemDerived);
        assert!(generation_origin.explicit);
        assert!(generation
            .attributes
            .contains_key(ORIGIN_NORMALIZATION_PARENT_ATTRIBUTE));

        assert_eq!(normalized.report.unresolved_desired, 1);
        assert_eq!(normalized.report.normalized_system_derived, 1);
    }

    #[test]
    fn daemonset_controller_target_is_typed_controller_derived() {
        let object = json!({
            "apiVersion":"apps/v1",
            "kind":"DaemonSet",
            "metadata":{
                "name":"agent",
                "namespace":"ops",
                "uid":"ds-1",
                "generation":4
            },
            "status":{
                "desiredNumberScheduled":5,
                "currentNumberScheduled":4,
                "observedGeneration":4
            }
        });
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[object.clone()],
            100,
        )
        .unwrap();
        let augmented = augment_workload_state(&replay, &[object], 100).unwrap();
        let normalized = normalize_kubernetes_state_origins(&augmented).unwrap();

        let desired = normalized
            .snapshot
            .assertions
            .iter()
            .find(|assertion| {
                assertion.role == StateRole::Desired
                    && assertion.dimension == "daemonset.scheduled_nodes"
            })
            .unwrap();
        let origin = desired.desired_state_origin().unwrap().unwrap();
        assert_eq!(origin.origin, DesiredStateOrigin::ControllerDerived);
        assert!(origin.explicit);

        let assessment = assess_state_dimension(
            &normalized.snapshot.assertions,
            &desired.subject,
            "daemonset.scheduled_nodes",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::Drift);
    }

    #[test]
    fn conflicting_explicit_origin_fails_closed() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment()],
            100,
        )
        .unwrap();
        let mut snapshot = replay.snapshot().clone();
        let generation = snapshot
            .assertions
            .iter_mut()
            .find(|assertion| {
                assertion.role == StateRole::Desired
                    && assertion.dimension == "controller.generation"
            })
            .unwrap();
        generation
            .set_desired_state_origin(DesiredStateOrigin::Declared)
            .unwrap();

        assert!(normalize_kubernetes_state_origins(&snapshot).is_err());
    }

    #[test]
    fn normalization_is_idempotent() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment()],
            100,
        )
        .unwrap();
        let once = normalize_kubernetes_state_origins(replay.snapshot()).unwrap();
        let twice = normalize_kubernetes_state_origins(&once.snapshot).unwrap();
        assert_eq!(once.snapshot, twice.snapshot);
    }

    #[test]
    fn normalized_snapshot_passes_registered_state_admission() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment()],
            100,
        )
        .unwrap();
        let normalized = normalize_kubernetes_state_origins(replay.snapshot()).unwrap();

        let mut registry = IntegrationRegistry::new();
        registry
            .register_discoverer(Arc::new(replay.topology().clone()))
            .unwrap();
        registry
            .admit_state_snapshot(
                &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
                &normalized.snapshot,
            )
            .unwrap();
    }

    #[test]
    fn observed_assertions_cannot_smuggle_desired_origin() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment()],
            100,
        )
        .unwrap();
        let mut snapshot = replay.snapshot().clone();
        let observed = snapshot
            .assertions
            .iter_mut()
            .find(|assertion| assertion.role == StateRole::Observed)
            .unwrap();
        observed.attributes.insert(
            DESIRED_STATE_ORIGIN_ATTRIBUTE.into(),
            "declared".into(),
        );
        assert!(normalize_kubernetes_state_origins(&snapshot).is_err());
    }
}
