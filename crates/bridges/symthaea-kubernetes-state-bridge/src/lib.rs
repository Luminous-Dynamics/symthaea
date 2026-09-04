// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! E1 Kubernetes desired/observed state replay.
//!
//! This bridge intentionally reuses `symthaea-kubernetes-bridge` for source-local
//! entity identity/topology and adds only state extraction. It opens no API
//! connection and performs no mutation. Absence of a field remains absence of
//! evidence; Kubernetes defaults are not guessed from incomplete fixture JSON.

#![forbid(unsafe_code)]

use blake3::Hasher;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use symthaea_integration_core::{
    EntityRef, IntegrationError, StateAssertion, StateAssertionSource, StateRole, StateSnapshot,
    StateValue,
};
use symthaea_kubernetes_bridge::{
    KUBERNETES_INTEGRATION_ID, KubernetesReplayContext, KubernetesReplayDiscoverer,
};

/// State evidence is another read-only output of the same Kubernetes replay
/// source, not a second logical integration identity.
pub const KUBERNETES_STATE_INTEGRATION_ID: &str = KUBERNETES_INTEGRATION_ID;

#[derive(Debug, Clone)]
pub struct KubernetesStateReplay {
    topology: KubernetesReplayDiscoverer,
    snapshot: StateSnapshot,
}

impl KubernetesStateReplay {
    pub fn from_objects(
        context: KubernetesReplayContext,
        documents: &[Value],
        collected_at_unix_ms: u64,
    ) -> Result<Self, IntegrationError> {
        let topology = KubernetesReplayDiscoverer::from_objects(
            context.clone(),
            documents,
            collected_at_unix_ms,
        )?;
        let objects = expand_documents(documents)?;
        let mut assertions = Vec::new();

        for (index, value) in objects.into_iter().enumerate() {
            let object = value.as_object().ok_or_else(|| {
                IntegrationError::Protocol(format!(
                    "Kubernetes state document {index} is not a JSON object"
                ))
            })?;
            let kind = required_string(object.get("kind"), index, "kind")?;
            if kind != "Deployment" {
                continue;
            }
            let api_version = required_string(object.get("apiVersion"), index, "apiVersion")?;
            let metadata = object
                .get("metadata")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    IntegrationError::Protocol(format!(
                        "Kubernetes state document {index} has no object `metadata`"
                    ))
                })?;
            let name = required_string(metadata.get("name"), index, "metadata.name")?;
            let namespace = optional_string(metadata.get("namespace"));
            let entity = find_topology_entity(&topology, &kind, namespace.as_deref(), &name)?;
            let resource_version = optional_string(metadata.get("resourceVersion"));

            if let Some(value) = optional_u64(
                object
                    .get("spec")
                    .and_then(Value::as_object)
                    .and_then(|spec| spec.get("replicas")),
                index,
                "spec.replicas",
            )? {
                assertions.push(assertion(
                    &context,
                    &entity,
                    "workload.replicas",
                    StateRole::Desired,
                    StateValue::Unsigned(value),
                    "spec.replicas",
                    &api_version,
                    &kind,
                    &name,
                    namespace.as_deref(),
                    resource_version.as_deref(),
                    collected_at_unix_ms,
                ));
            }

            let status = object.get("status").and_then(Value::as_object);
            push_observed_u64(
                &mut assertions,
                &context,
                &entity,
                status,
                "replicas",
                "workload.replicas",
                "status.replicas",
                &api_version,
                &kind,
                &name,
                namespace.as_deref(),
                resource_version.as_deref(),
                index,
                collected_at_unix_ms,
            )?;
            push_observed_u64(
                &mut assertions,
                &context,
                &entity,
                status,
                "readyReplicas",
                "workload.ready_replicas",
                "status.readyReplicas",
                &api_version,
                &kind,
                &name,
                namespace.as_deref(),
                resource_version.as_deref(),
                index,
                collected_at_unix_ms,
            )?;
            push_observed_u64(
                &mut assertions,
                &context,
                &entity,
                status,
                "availableReplicas",
                "workload.available_replicas",
                "status.availableReplicas",
                &api_version,
                &kind,
                &name,
                namespace.as_deref(),
                resource_version.as_deref(),
                index,
                collected_at_unix_ms,
            )?;
            push_observed_u64(
                &mut assertions,
                &context,
                &entity,
                status,
                "updatedReplicas",
                "workload.updated_replicas",
                "status.updatedReplicas",
                &api_version,
                &kind,
                &name,
                namespace.as_deref(),
                resource_version.as_deref(),
                index,
                collected_at_unix_ms,
            )?;
            push_observed_u64(
                &mut assertions,
                &context,
                &entity,
                status,
                "unavailableReplicas",
                "workload.unavailable_replicas",
                "status.unavailableReplicas",
                &api_version,
                &kind,
                &name,
                namespace.as_deref(),
                resource_version.as_deref(),
                index,
                collected_at_unix_ms,
            )?;

            if let Some(generation) = optional_u64(
                metadata.get("generation"),
                index,
                "metadata.generation",
            )? {
                assertions.push(assertion(
                    &context,
                    &entity,
                    "controller.generation",
                    StateRole::Desired,
                    StateValue::Unsigned(generation),
                    "metadata.generation",
                    &api_version,
                    &kind,
                    &name,
                    namespace.as_deref(),
                    resource_version.as_deref(),
                    collected_at_unix_ms,
                ));
            }
            if let Some(observed_generation) = optional_u64(
                status.and_then(|status| status.get("observedGeneration")),
                index,
                "status.observedGeneration",
            )? {
                assertions.push(assertion(
                    &context,
                    &entity,
                    "controller.generation",
                    StateRole::Observed,
                    StateValue::Unsigned(observed_generation),
                    "status.observedGeneration",
                    &api_version,
                    &kind,
                    &name,
                    namespace.as_deref(),
                    resource_version.as_deref(),
                    collected_at_unix_ms,
                ));
            }
        }

        let snapshot = StateSnapshot {
            integration_id: KUBERNETES_STATE_INTEGRATION_ID.into(),
            collected_at_unix_ms,
            assertions,
        };
        snapshot
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(Self { topology, snapshot })
    }

    pub fn topology(&self) -> &KubernetesReplayDiscoverer {
        &self.topology
    }

    pub fn snapshot(&self) -> &StateSnapshot {
        &self.snapshot
    }
}

#[allow(clippy::too_many_arguments)]
fn push_observed_u64(
    assertions: &mut Vec<StateAssertion>,
    context: &KubernetesReplayContext,
    entity: &EntityRef,
    status: Option<&Map<String, Value>>,
    key: &str,
    dimension: &str,
    field: &str,
    api_version: &str,
    kind: &str,
    name: &str,
    namespace: Option<&str>,
    resource_version: Option<&str>,
    index: usize,
    collected_at_unix_ms: u64,
) -> Result<(), IntegrationError> {
    if let Some(value) = optional_u64(status.and_then(|status| status.get(key)), index, field)? {
        assertions.push(assertion(
            context,
            entity,
            dimension,
            StateRole::Observed,
            StateValue::Unsigned(value),
            field,
            api_version,
            kind,
            name,
            namespace,
            resource_version,
            collected_at_unix_ms,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assertion(
    context: &KubernetesReplayContext,
    entity: &EntityRef,
    dimension: &str,
    role: StateRole,
    value: StateValue,
    field: &str,
    api_version: &str,
    kind: &str,
    name: &str,
    namespace: Option<&str>,
    resource_version: Option<&str>,
    collected_at_unix_ms: u64,
) -> StateAssertion {
    let mut attributes = BTreeMap::from([
        ("k8s.api_version".into(), api_version.into()),
        ("k8s.kind".into(), kind.into()),
        ("k8s.name".into(), name.into()),
        ("k8s.field".into(), field.into()),
    ]);
    if let Some(namespace) = namespace {
        attributes.insert("k8s.namespace".into(), namespace.into());
    }
    if let Some(resource_version) = resource_version {
        attributes.insert("k8s.resource_version".into(), resource_version.into());
    }

    let assertion_id = assertion_id(entity, dimension, role, &value, field, collected_at_unix_ms);
    StateAssertion {
        assertion_id,
        subject: entity.clone(),
        dimension: dimension.into(),
        role,
        value,
        source_confidence: context.source_confidence,
        source: StateAssertionSource {
            integration_id: KUBERNETES_STATE_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            tenant: context.tenant.clone(),
        },
        observed_at_unix_ms: collected_at_unix_ms,
        valid_from_unix_ms: None,
        valid_until_unix_ms: None,
        evidence_observation_ids: vec![],
        attributes,
    }
}

fn assertion_id(
    entity: &EntityRef,
    dimension: &str,
    role: StateRole,
    value: &StateValue,
    field: &str,
    collected_at_unix_ms: u64,
) -> String {
    let mut hasher = Hasher::new();
    feed(&mut hasher, "symthaea-kubernetes-state-v1");
    feed(&mut hasher, &entity.canonical_key());
    feed(&mut hasher, dimension);
    feed(
        &mut hasher,
        match role {
            StateRole::Desired => "desired",
            StateRole::Observed => "observed",
        },
    );
    feed(&mut hasher, &canonical_value(value));
    feed(&mut hasher, field);
    feed(&mut hasher, &collected_at_unix_ms.to_string());
    format!("k8s-state:{}", hasher.finalize().to_hex())
}

fn canonical_value(value: &StateValue) -> String {
    match value {
        StateValue::Number(value) => format!("number:{:016x}", value.to_bits()),
        StateValue::Integer(value) => format!("integer:{value}"),
        StateValue::Unsigned(value) => format!("unsigned:{value}"),
        StateValue::Boolean(value) => format!("boolean:{value}"),
        StateValue::Text(value) => format!("text:{}:{value}", value.len()),
    }
}

fn feed(hasher: &mut Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn find_topology_entity(
    topology: &KubernetesReplayDiscoverer,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
) -> Result<EntityRef, IntegrationError> {
    let matches = topology
        .topology()
        .entities
        .iter()
        .filter(|entity| entity.attributes.get("k8s.kind").map(String::as_str) == Some(kind))
        .filter(|entity| entity.attributes.get("k8s.name").map(String::as_str) == Some(name))
        .filter(|entity| {
            entity.attributes.get("k8s.namespace").map(String::as_str) == namespace
        })
        .map(|entity| entity.entity.clone())
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [entity] => Ok(entity.clone()),
        [] => Err(IntegrationError::InvalidOutput(format!(
            "Kubernetes state object `{kind}/{name}` has no matching topology entity"
        ))),
        _ => Err(IntegrationError::InvalidOutput(format!(
            "Kubernetes state object `{kind}/{name}` maps to multiple topology entities"
        ))),
    }
}

fn optional_u64(
    value: Option<&Value>,
    index: usize,
    field: &str,
) -> Result<Option<u64>, IntegrationError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value.as_u64().map(Some).ok_or_else(|| {
        IntegrationError::Protocol(format!(
            "Kubernetes state document {index} `{field}` must be a non-negative integer"
        ))
    })
}

fn required_string(
    value: Option<&Value>,
    index: usize,
    field: &str,
) -> Result<String, IntegrationError> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes state document {index} requires non-empty `{field}`"
            ))
        })
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn expand_documents<'a>(documents: &'a [Value]) -> Result<Vec<&'a Value>, IntegrationError> {
    let mut queue = documents.iter().collect::<Vec<_>>();
    let mut output = Vec::new();
    while let Some(value) = queue.pop() {
        let kind = value.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind.ends_with("List") {
            let items = value
                .get("items")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    IntegrationError::Protocol(format!(
                        "Kubernetes state `{kind}` document has no array `items`"
                    ))
                })?;
            queue.extend(items.iter());
        } else {
            output.push(value);
        }
    }
    output.reverse();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use symthaea_integration_core::{
        StateAssessmentStatus, StateComparisonPolicy, assess_state_dimension,
    };

    fn deployment(
        spec: u64,
        observed: Option<u64>,
        generation: u64,
        observed_generation: u64,
    ) -> Value {
        let mut status = serde_json::Map::new();
        if let Some(observed) = observed {
            status.insert("replicas".into(), json!(observed));
        }
        status.insert("readyReplicas".into(), json!(observed.unwrap_or(0)));
        status.insert("observedGeneration".into(), json!(observed_generation));
        json!({
            "apiVersion":"apps/v1",
            "kind":"Deployment",
            "metadata":{
                "name":"api",
                "namespace":"shop",
                "uid":"deployment-uid",
                "generation":generation,
                "resourceVersion":"42"
            },
            "spec":{"replicas":spec},
            "status":status
        })
    }

    #[test]
    fn deployment_replicas_become_real_drift_evidence() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment(5, Some(3), 9, 8)],
            100,
        )
        .unwrap();
        let entity = replay
            .topology()
            .topology()
            .entities
            .iter()
            .find(|entity| entity.entity.kind == "k8s_deployment")
            .unwrap()
            .entity
            .clone();
        let assessment = assess_state_dimension(
            &replay.snapshot().assertions,
            &entity,
            "workload.replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::Drift);
        assert_eq!(assessment.desired_value, Some(StateValue::Unsigned(5)));
        assert_eq!(assessment.observed_value, Some(StateValue::Unsigned(3)));
    }

    #[test]
    fn generation_lag_is_separate_from_replica_drift() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment(3, Some(3), 9, 8)],
            100,
        )
        .unwrap();
        let entity = replay.snapshot().assertions[0].subject.clone();
        let replicas = assess_state_dimension(
            &replay.snapshot().assertions,
            &entity,
            "workload.replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        let generation = assess_state_dimension(
            &replay.snapshot().assertions,
            &entity,
            "controller.generation",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(replicas.status, StateAssessmentStatus::InSync);
        assert_eq!(generation.status, StateAssessmentStatus::Drift);
    }

    #[test]
    fn absent_status_is_missing_evidence_not_zero() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment(5, None, 9, 9)],
            100,
        )
        .unwrap();
        let entity = replay.snapshot().assertions[0].subject.clone();
        let assessment = assess_state_dimension(
            &replay.snapshot().assertions,
            &entity,
            "workload.replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::MissingObserved);
    }

    #[test]
    fn ready_replicas_remain_observed_only_without_policy_guessing() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment(5, Some(3), 9, 9)],
            100,
        )
        .unwrap();
        let entity = replay.snapshot().assertions[0].subject.clone();
        let assessment = assess_state_dimension(
            &replay.snapshot().assertions,
            &entity,
            "workload.ready_replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::MissingDesired);
    }

    #[test]
    fn state_entity_is_exact_topology_entity_not_parallel_identity() {
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &[deployment(3, Some(3), 1, 1)],
            100,
        )
        .unwrap();
        let topology_entity = replay
            .topology()
            .topology()
            .entities
            .iter()
            .find(|entity| entity.entity.kind == "k8s_deployment")
            .unwrap();
        assert!(
            replay
                .snapshot()
                .assertions
                .iter()
                .all(|assertion| assertion.subject == topology_entity.entity)
        );
        assert_eq!(replay.snapshot().integration_id, KUBERNETES_INTEGRATION_ID);
    }

    #[test]
    fn malformed_negative_replica_count_fails_closed() {
        let value = json!({
            "apiVersion":"apps/v1",
            "kind":"Deployment",
            "metadata":{"name":"api","namespace":"shop","uid":"deployment-uid"},
            "spec":{"replicas":-1}
        });
        assert!(matches!(
            KubernetesStateReplay::from_objects(
                KubernetesReplayContext::default(),
                &[value],
                100,
            ),
            Err(IntegrationError::Protocol(_))
        ));
    }
}
