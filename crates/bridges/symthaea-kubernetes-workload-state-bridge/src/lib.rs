// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Controller-specific Kubernetes workload-state augmentation for E1 replay.
//!
//! This crate is deliberately not a second Kubernetes state source. It consumes
//! an existing `KubernetesStateReplay`, reuses its source-local entities and
//! integration identity, and adds state assertions whose semantics are specific
//! to ReplicaSet, StatefulSet, DaemonSet, and workload Conditions.

#![forbid(unsafe_code)]

use blake3::Hasher;
use chrono::DateTime;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_integration_core::{
    EntityRef, IntegrationError, StateAssertion, StateAssertionSource, StateRole, StateSnapshot,
    StateValue,
};
use symthaea_kubernetes_bridge::{KUBERNETES_INTEGRATION_ID, KubernetesReplayContext};
use symthaea_kubernetes_state_bridge::KubernetesStateReplay;

const MAX_CONDITION_TYPE_BYTES: usize = 256;
const MAX_CONDITION_MESSAGE_INLINE_BYTES: usize = 2 * 1024;

/// Add controller-specific workload and Condition assertions to an existing
/// Kubernetes E1 state replay.
///
/// The input documents must represent the same capture instant as `base`.
/// Combining separately timed captures would manufacture simultaneity.
pub fn augment_workload_state(
    base: &KubernetesStateReplay,
    documents: &[Value],
    collected_at_unix_ms: u64,
) -> Result<StateSnapshot, IntegrationError> {
    if base.snapshot().collected_at_unix_ms != collected_at_unix_ms {
        return Err(IntegrationError::InvalidRequest(format!(
            "Kubernetes workload-state augmentation requires same-capture input: base={}, augmentation={collected_at_unix_ms}",
            base.snapshot().collected_at_unix_ms
        )));
    }

    let context = base.topology().context().clone();
    let objects = expand_documents(documents)?;
    let mut assertions = base.snapshot().assertions.clone();

    for (index, value) in objects.into_iter().enumerate() {
        let object = value.as_object().ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes workload-state document {index} is not a JSON object"
            ))
        })?;
        let kind = required_string(object.get("kind"), index, "kind")?;
        if !matches!(
            kind.as_str(),
            "Deployment" | "ReplicaSet" | "StatefulSet" | "DaemonSet"
        ) {
            continue;
        }

        let info = object_info(base, object, &kind, index)?;
        let spec = object.get("spec").and_then(Value::as_object);
        let status = object.get("status").and_then(Value::as_object);

        match kind.as_str() {
            // Deployment count/generation semantics already live in the base
            // Kubernetes state replay. This augmenter adds Conditions only.
            "Deployment" => {}
            "ReplicaSet" => {
                push_u64_field(
                    &mut assertions,
                    &context,
                    &info,
                    spec.and_then(|spec| spec.get("replicas")),
                    index,
                    "spec.replicas",
                    "workload.replicas",
                    StateRole::Desired,
                    BTreeMap::new(),
                    collected_at_unix_ms,
                )?;
                push_observed_status_fields(
                    &mut assertions,
                    &context,
                    &info,
                    status,
                    index,
                    collected_at_unix_ms,
                    &[
                        ("replicas", "workload.replicas"),
                        ("readyReplicas", "workload.ready_replicas"),
                        ("availableReplicas", "workload.available_replicas"),
                        ("fullyLabeledReplicas", "workload.fully_labeled_replicas"),
                        ("terminatingReplicas", "workload.terminating_replicas"),
                    ],
                )?;
                push_generation_pair(
                    &mut assertions,
                    &context,
                    &info,
                    object,
                    status,
                    index,
                    collected_at_unix_ms,
                )?;
            }
            "StatefulSet" => {
                push_u64_field(
                    &mut assertions,
                    &context,
                    &info,
                    spec.and_then(|spec| spec.get("replicas")),
                    index,
                    "spec.replicas",
                    "workload.replicas",
                    StateRole::Desired,
                    BTreeMap::new(),
                    collected_at_unix_ms,
                )?;
                push_observed_status_fields(
                    &mut assertions,
                    &context,
                    &info,
                    status,
                    index,
                    collected_at_unix_ms,
                    &[
                        ("replicas", "workload.replicas"),
                        ("readyReplicas", "workload.ready_replicas"),
                        ("availableReplicas", "workload.available_replicas"),
                        ("currentReplicas", "workload.current_revision_replicas"),
                        ("updatedReplicas", "workload.updated_replicas"),
                    ],
                )?;
                push_observed_text_field(
                    &mut assertions,
                    &context,
                    &info,
                    status.and_then(|status| status.get("currentRevision")),
                    "status.currentRevision",
                    "workload.current_revision",
                    collected_at_unix_ms,
                )?;
                push_observed_text_field(
                    &mut assertions,
                    &context,
                    &info,
                    status.and_then(|status| status.get("updateRevision")),
                    "status.updateRevision",
                    "workload.update_revision",
                    collected_at_unix_ms,
                )?;
                push_generation_pair(
                    &mut assertions,
                    &context,
                    &info,
                    object,
                    status,
                    index,
                    collected_at_unix_ms,
                )?;
            }
            "DaemonSet" => {
                // DaemonSet has no user-authored replica count. Kubernetes
                // computes desiredNumberScheduled from node eligibility. Keep
                // that value on the Desired side but explicitly mark its origin
                // as controller-derived rather than configuration intent.
                push_u64_field(
                    &mut assertions,
                    &context,
                    &info,
                    status.and_then(|status| status.get("desiredNumberScheduled")),
                    index,
                    "status.desiredNumberScheduled",
                    "daemonset.scheduled_nodes",
                    StateRole::Desired,
                    BTreeMap::from([(
                        "symthaea.desired_origin".into(),
                        "controller_derived".into(),
                    )]),
                    collected_at_unix_ms,
                )?;
                push_u64_field(
                    &mut assertions,
                    &context,
                    &info,
                    status.and_then(|status| status.get("currentNumberScheduled")),
                    index,
                    "status.currentNumberScheduled",
                    "daemonset.scheduled_nodes",
                    StateRole::Observed,
                    BTreeMap::new(),
                    collected_at_unix_ms,
                )?;
                push_observed_status_fields(
                    &mut assertions,
                    &context,
                    &info,
                    status,
                    index,
                    collected_at_unix_ms,
                    &[
                        ("numberReady", "daemonset.ready_nodes"),
                        ("numberAvailable", "daemonset.available_nodes"),
                        ("numberUnavailable", "daemonset.unavailable_nodes"),
                        ("numberMisscheduled", "daemonset.misscheduled_nodes"),
                        ("updatedNumberScheduled", "daemonset.updated_nodes"),
                        ("collisionCount", "controller.collision_count"),
                    ],
                )?;
                push_generation_pair(
                    &mut assertions,
                    &context,
                    &info,
                    object,
                    status,
                    index,
                    collected_at_unix_ms,
                )?;
            }
            _ => unreachable!(),
        }

        push_conditions(
            &mut assertions,
            &context,
            &info,
            status,
            index,
            collected_at_unix_ms,
        )?;
    }

    let snapshot = StateSnapshot {
        integration_id: KUBERNETES_INTEGRATION_ID.into(),
        collected_at_unix_ms,
        assertions,
    };
    snapshot
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    Ok(snapshot)
}

#[derive(Debug, Clone)]
struct ObjectInfo {
    entity: EntityRef,
    api_version: String,
    kind: String,
    name: String,
    namespace: Option<String>,
    resource_version: Option<String>,
}

fn object_info(
    base: &KubernetesStateReplay,
    object: &Map<String, Value>,
    kind: &str,
    index: usize,
) -> Result<ObjectInfo, IntegrationError> {
    let api_version = required_string(object.get("apiVersion"), index, "apiVersion")?;
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes workload-state document {index} has no object `metadata`"
            ))
        })?;
    let name = required_string(metadata.get("name"), index, "metadata.name")?;
    let namespace = optional_string(metadata.get("namespace"));
    let resource_version = optional_string(metadata.get("resourceVersion"));
    let entity = find_topology_entity(base, kind, namespace.as_deref(), &name)?;
    Ok(ObjectInfo {
        entity,
        api_version,
        kind: kind.into(),
        name,
        namespace,
        resource_version,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_generation_pair(
    assertions: &mut Vec<StateAssertion>,
    context: &KubernetesReplayContext,
    info: &ObjectInfo,
    object: &Map<String, Value>,
    status: Option<&Map<String, Value>>,
    index: usize,
    collected_at_unix_ms: u64,
) -> Result<(), IntegrationError> {
    push_u64_field(
        assertions,
        context,
        info,
        object
            .get("metadata")
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get("generation")),
        index,
        "metadata.generation",
        "controller.generation",
        StateRole::Desired,
        BTreeMap::new(),
        collected_at_unix_ms,
    )?;
    push_u64_field(
        assertions,
        context,
        info,
        status.and_then(|status| status.get("observedGeneration")),
        index,
        "status.observedGeneration",
        "controller.generation",
        StateRole::Observed,
        BTreeMap::new(),
        collected_at_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_observed_status_fields(
    assertions: &mut Vec<StateAssertion>,
    context: &KubernetesReplayContext,
    info: &ObjectInfo,
    status: Option<&Map<String, Value>>,
    index: usize,
    collected_at_unix_ms: u64,
    fields: &[(&str, &str)],
) -> Result<(), IntegrationError> {
    for (field, dimension) in fields {
        push_u64_field(
            assertions,
            context,
            info,
            status.and_then(|status| status.get(*field)),
            index,
            &format!("status.{field}"),
            dimension,
            StateRole::Observed,
            BTreeMap::new(),
            collected_at_unix_ms,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_u64_field(
    assertions: &mut Vec<StateAssertion>,
    context: &KubernetesReplayContext,
    info: &ObjectInfo,
    raw: Option<&Value>,
    index: usize,
    field: &str,
    dimension: &str,
    role: StateRole,
    extra_attributes: BTreeMap<String, String>,
    collected_at_unix_ms: u64,
) -> Result<(), IntegrationError> {
    let Some(value) = optional_u64(raw, index, field)? else {
        return Ok(());
    };
    assertions.push(assertion(
        context,
        info,
        dimension,
        role,
        StateValue::Unsigned(value),
        field,
        None,
        extra_attributes,
        collected_at_unix_ms,
    ));
    Ok(())
}

fn push_observed_text_field(
    assertions: &mut Vec<StateAssertion>,
    context: &KubernetesReplayContext,
    info: &ObjectInfo,
    raw: Option<&Value>,
    field: &str,
    dimension: &str,
    collected_at_unix_ms: u64,
) -> Result<(), IntegrationError> {
    let Some(value) = optional_string(raw) else {
        return Ok(());
    };
    assertions.push(assertion(
        context,
        info,
        dimension,
        StateRole::Observed,
        StateValue::Text(value),
        field,
        None,
        BTreeMap::new(),
        collected_at_unix_ms,
    ));
    Ok(())
}

fn push_conditions(
    assertions: &mut Vec<StateAssertion>,
    context: &KubernetesReplayContext,
    info: &ObjectInfo,
    status: Option<&Map<String, Value>>,
    index: usize,
    collected_at_unix_ms: u64,
) -> Result<(), IntegrationError> {
    let Some(raw_conditions) = status.and_then(|status| status.get("conditions")) else {
        return Ok(());
    };
    if raw_conditions.is_null() {
        return Ok(());
    }
    let conditions = raw_conditions.as_array().ok_or_else(|| {
        IntegrationError::Protocol(format!(
            "Kubernetes workload-state document {index} `status.conditions` is not an array"
        ))
    })?;
    let mut condition_types = BTreeSet::new();

    for (condition_index, raw_condition) in conditions.iter().enumerate() {
        let condition = raw_condition.as_object().ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes workload-state document {index} condition {condition_index} is not an object"
            ))
        })?;
        let condition_type = required_string(
            condition.get("type"),
            index,
            "status.conditions.type",
        )?;
        if condition_type.len() > MAX_CONDITION_TYPE_BYTES {
            return Err(IntegrationError::Protocol(format!(
                "Kubernetes condition type is {} bytes; limit is {MAX_CONDITION_TYPE_BYTES}",
                condition_type.len()
            )));
        }
        if !condition_types.insert(condition_type.clone()) {
            return Err(IntegrationError::Protocol(format!(
                "Kubernetes `{}` object `{}` contains duplicate condition type `{condition_type}`",
                info.kind, info.name
            )));
        }
        let condition_status = required_string(
            condition.get("status"),
            index,
            "status.conditions.status",
        )?;
        if !matches!(condition_status.as_str(), "True" | "False" | "Unknown") {
            return Err(IntegrationError::Protocol(format!(
                "Kubernetes condition `{condition_type}` has invalid status `{condition_status}`; expected True, False, or Unknown"
            )));
        }

        let last_transition = optional_string(condition.get("lastTransitionTime"));
        let valid_from_unix_ms = last_transition
            .as_deref()
            .map(|value| parse_rfc3339_unix_ms(value, index, "status.conditions.lastTransitionTime"))
            .transpose()?;

        let mut attributes = BTreeMap::from([(
            "k8s.condition.type".into(),
            condition_type.clone(),
        )]);
        if let Some(reason) = optional_string(condition.get("reason")) {
            attributes.insert("k8s.condition.reason".into(), reason);
        }
        if let Some(value) = last_transition {
            attributes.insert("k8s.condition.last_transition_time".into(), value);
        }
        if let Some(value) = optional_string(condition.get("lastUpdateTime")) {
            // DeploymentCondition defines this field; other workload condition
            // types may simply omit it.
            parse_rfc3339_unix_ms(&value, index, "status.conditions.lastUpdateTime")?;
            attributes.insert("k8s.condition.last_update_time".into(), value);
        }
        if let Some(message) = optional_string(condition.get("message")) {
            let mut hasher = Hasher::new();
            hasher.update(message.as_bytes());
            attributes.insert(
                "k8s.condition.message_blake3".into(),
                hasher.finalize().to_hex().to_string(),
            );
            attributes.insert(
                "k8s.condition.message_bytes".into(),
                message.len().to_string(),
            );
            if message.len() <= MAX_CONDITION_MESSAGE_INLINE_BYTES {
                attributes.insert("k8s.condition.message".into(), message);
            } else {
                attributes.insert("k8s.condition.message_omitted".into(), "true".into());
            }
        }

        assertions.push(assertion(
            context,
            info,
            &format!("k8s.condition.{condition_type}.status"),
            StateRole::Observed,
            StateValue::Text(condition_status),
            "status.conditions",
            valid_from_unix_ms,
            attributes,
            collected_at_unix_ms,
        ));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assertion(
    context: &KubernetesReplayContext,
    info: &ObjectInfo,
    dimension: &str,
    role: StateRole,
    value: StateValue,
    field: &str,
    valid_from_unix_ms: Option<u64>,
    extra_attributes: BTreeMap<String, String>,
    collected_at_unix_ms: u64,
) -> StateAssertion {
    let mut attributes = BTreeMap::from([
        ("k8s.api_version".into(), info.api_version.clone()),
        ("k8s.kind".into(), info.kind.clone()),
        ("k8s.name".into(), info.name.clone()),
        ("k8s.field".into(), field.into()),
    ]);
    if let Some(namespace) = &info.namespace {
        attributes.insert("k8s.namespace".into(), namespace.clone());
    }
    if let Some(resource_version) = &info.resource_version {
        attributes.insert("k8s.resource_version".into(), resource_version.clone());
    }
    attributes.extend(extra_attributes);

    let assertion_id = assertion_id(
        &info.entity,
        dimension,
        role,
        &value,
        field,
        collected_at_unix_ms,
    );
    StateAssertion {
        assertion_id,
        subject: info.entity.clone(),
        dimension: dimension.into(),
        role,
        value,
        source_confidence: context.source_confidence,
        source: StateAssertionSource {
            integration_id: KUBERNETES_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            tenant: context.tenant.clone(),
        },
        observed_at_unix_ms: collected_at_unix_ms,
        valid_from_unix_ms,
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
    feed(&mut hasher, "symthaea-kubernetes-workload-state-v1");
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
    format!("k8s-workload-state:{}", hasher.finalize().to_hex())
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
    base: &KubernetesStateReplay,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
) -> Result<EntityRef, IntegrationError> {
    let matches = base
        .topology()
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
            "Kubernetes workload-state object `{kind}/{name}` has no matching topology entity"
        ))),
        _ => Err(IntegrationError::InvalidOutput(format!(
            "Kubernetes workload-state object `{kind}/{name}` maps to multiple topology entities"
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
            "Kubernetes workload-state document {index} `{field}` must be a non-negative integer"
        ))
    })
}

fn parse_rfc3339_unix_ms(
    value: &str,
    index: usize,
    field: &str,
) -> Result<u64, IntegrationError> {
    let timestamp = DateTime::parse_from_rfc3339(value).map_err(|error| {
        IntegrationError::Protocol(format!(
            "Kubernetes workload-state document {index} `{field}` is not RFC3339: {error}"
        ))
    })?;
    u64::try_from(timestamp.timestamp_millis()).map_err(|_| {
        IntegrationError::Protocol(format!(
            "Kubernetes workload-state document {index} `{field}` predates Unix epoch"
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
                "Kubernetes workload-state document {index} requires non-empty `{field}`"
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
                        "Kubernetes workload-state `{kind}` document has no array `items`"
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
    use std::sync::Arc;
    use symthaea_integration_core::{
        IntegrationId, IntegrationRegistry, StateAssessmentStatus, StateComparisonPolicy,
        assess_state_dimension,
    };

    fn base(objects: &[Value], at: u64) -> KubernetesStateReplay {
        KubernetesStateReplay::from_objects(
            KubernetesReplayContext {
                cluster_id: "cluster-a".into(),
                source_confidence: 0.99,
                ..Default::default()
            },
            objects,
            at,
        )
        .unwrap()
    }

    #[test]
    fn replicaset_replica_gap_is_real_drift_and_generation_is_separate() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"ReplicaSet",
            "metadata":{"name":"api-rs","namespace":"shop","uid":"rs-1","generation":7},
            "spec":{"replicas":4},
            "status":{"replicas":3,"readyReplicas":2,"availableReplicas":2,"observedGeneration":6}
        })];
        let base = base(&objects, 100);
        let snapshot = augment_workload_state(&base, &objects, 100).unwrap();
        let entity = find_topology_entity(&base, "ReplicaSet", Some("shop"), "api-rs").unwrap();
        let replicas = assess_state_dimension(
            &snapshot.assertions,
            &entity,
            "workload.replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(replicas.status, StateAssessmentStatus::Drift);
        let generation = assess_state_dimension(
            &snapshot.assertions,
            &entity,
            "controller.generation",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(generation.status, StateAssessmentStatus::Drift);
    }

    #[test]
    fn statefulset_revisions_remain_distinct_observed_dimensions() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"StatefulSet",
            "metadata":{"name":"db","namespace":"shop","uid":"sts-1","generation":4},
            "spec":{"replicas":3},
            "status":{
                "replicas":3,"readyReplicas":2,"currentReplicas":2,"updatedReplicas":1,
                "currentRevision":"db-a","updateRevision":"db-b","observedGeneration":4
            }
        })];
        let base = base(&objects, 100);
        let snapshot = augment_workload_state(&base, &objects, 100).unwrap();
        assert!(snapshot.assertions.iter().any(|assertion| {
            assertion.dimension == "workload.current_revision"
                && assertion.value == StateValue::Text("db-a".into())
                && assertion.role == StateRole::Observed
        }));
        assert!(snapshot.assertions.iter().any(|assertion| {
            assertion.dimension == "workload.update_revision"
                && assertion.value == StateValue::Text("db-b".into())
                && assertion.role == StateRole::Observed
        }));
        assert!(!snapshot.assertions.iter().any(|assertion| {
            assertion.dimension == "workload.ready_replicas"
                && assertion.role == StateRole::Desired
        }));
    }

    #[test]
    fn daemonset_controller_derived_target_can_be_compared_without_becoming_user_intent() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"DaemonSet",
            "metadata":{"name":"agent","namespace":"ops","uid":"ds-1","generation":3},
            "status":{
                "desiredNumberScheduled":5,"currentNumberScheduled":4,"numberReady":3,
                "numberMisscheduled":1,"updatedNumberScheduled":4,"observedGeneration":3
            }
        })];
        let base = base(&objects, 100);
        let snapshot = augment_workload_state(&base, &objects, 100).unwrap();
        let entity = find_topology_entity(&base, "DaemonSet", Some("ops"), "agent").unwrap();
        let assessment = assess_state_dimension(
            &snapshot.assertions,
            &entity,
            "daemonset.scheduled_nodes",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::Drift);
        let desired = snapshot.assertions.iter().find(|assertion| {
            assertion.subject == entity
                && assertion.dimension == "daemonset.scheduled_nodes"
                && assertion.role == StateRole::Desired
        }).unwrap();
        assert_eq!(
            desired.attributes.get("symthaea.desired_origin").map(String::as_str),
            Some("controller_derived")
        );
    }

    #[test]
    fn workload_condition_is_tri_state_observed_evidence_with_transition_time() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"Deployment",
            "metadata":{"name":"api","namespace":"shop","uid":"dep-1","generation":2},
            "spec":{"replicas":2},
            "status":{
                "replicas":2,"observedGeneration":2,
                "conditions":[{
                    "type":"Available","status":"False","reason":"MinimumReplicasUnavailable",
                    "message":"Deployment does not have minimum availability.",
                    "lastTransitionTime":"2026-09-01T18:00:00Z",
                    "lastUpdateTime":"2026-09-01T18:01:00Z"
                }]
            }
        })];
        let base = base(&objects, 100);
        let snapshot = augment_workload_state(&base, &objects, 100).unwrap();
        let condition = snapshot.assertions.iter().find(|assertion| {
            assertion.dimension == "k8s.condition.Available.status"
        }).unwrap();
        assert_eq!(condition.role, StateRole::Observed);
        assert_eq!(condition.value, StateValue::Text("False".into()));
        assert!(condition.valid_from_unix_ms.is_some());
        assert_eq!(
            condition.attributes.get("k8s.condition.reason").map(String::as_str),
            Some("MinimumReplicasUnavailable")
        );
    }

    #[test]
    fn duplicate_condition_type_is_rejected() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"ReplicaSet",
            "metadata":{"name":"rs","namespace":"shop","uid":"rs-1"},
            "status":{"conditions":[
                {"type":"ReplicaFailure","status":"True"},
                {"type":"ReplicaFailure","status":"False"}
            ]}
        })];
        let base = base(&objects, 100);
        assert!(augment_workload_state(&base, &objects, 100).is_err());
    }

    #[test]
    fn invalid_condition_status_is_rejected() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"StatefulSet",
            "metadata":{"name":"db","namespace":"shop","uid":"sts-1"},
            "status":{"conditions":[{"type":"Ready","status":"Maybe"}]}
        })];
        let base = base(&objects, 100);
        assert!(augment_workload_state(&base, &objects, 100).is_err());
    }

    #[test]
    fn large_condition_message_is_digest_only() {
        let message = "x".repeat(MAX_CONDITION_MESSAGE_INLINE_BYTES + 1);
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"DaemonSet",
            "metadata":{"name":"agent","namespace":"ops","uid":"ds-1"},
            "status":{
                "desiredNumberScheduled":1,"currentNumberScheduled":1,
                "conditions":[{"type":"Ready","status":"True","message":message}]
            }
        })];
        let base = base(&objects, 100);
        let snapshot = augment_workload_state(&base, &objects, 100).unwrap();
        let condition = snapshot.assertions.iter().find(|assertion| {
            assertion.dimension == "k8s.condition.Ready.status"
        }).unwrap();
        assert!(!condition.attributes.contains_key("k8s.condition.message"));
        assert_eq!(
            condition.attributes.get("k8s.condition.message_omitted").map(String::as_str),
            Some("true")
        );
        assert!(condition.attributes.contains_key("k8s.condition.message_blake3"));
    }

    #[test]
    fn list_replay_and_registry_admission_use_the_existing_kubernetes_source() {
        let item = json!({
            "apiVersion":"apps/v1","kind":"ReplicaSet",
            "metadata":{"name":"rs","namespace":"shop","uid":"rs-1"},
            "spec":{"replicas":2},"status":{"replicas":2}
        });
        let list = json!({"apiVersion":"apps/v1","kind":"ReplicaSetList","items":[item]});
        let base = base(&[list.clone()], 100);
        let snapshot = augment_workload_state(&base, &[list], 100).unwrap();

        let mut registry = IntegrationRegistry::new();
        registry
            .register_discoverer(Arc::new(base.topology().clone()))
            .unwrap();
        registry
            .admit_state_snapshot(&IntegrationId::new(KUBERNETES_INTEGRATION_ID), &snapshot)
            .unwrap();
    }

    #[test]
    fn capture_time_mismatch_is_rejected() {
        let objects = vec![json!({
            "apiVersion":"apps/v1","kind":"ReplicaSet",
            "metadata":{"name":"rs","namespace":"shop","uid":"rs-1"}
        })];
        let base = base(&objects, 100);
        assert!(augment_workload_state(&base, &objects, 101).is_err());
    }
}
