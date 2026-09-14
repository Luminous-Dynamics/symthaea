// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage-aware Kubernetes service-capacity evidence.
//!
//! This module deliberately does **not** emit a synthetic health score. It
//! combines already-admitted workload state with already-admitted EndpointSlice
//! topology and reports independently observed facts. Because E1 Kubernetes
//! replay is non-exhaustive, endpoint counts are lower bounds over the observed
//! corpus unless a later qualified completeness contract proves otherwise.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use symthaea_integration_core::{
    DesiredStateOrigin, DiscoverySnapshot, EntityRef, IntegrationError, IntegrationId,
    IntegrationRegistry, RelationBasis, RelationKind, StateAssessment, StateAssertion,
    StateComparisonPolicy, StateRole, StateSnapshot, assess_state_dimension,
    validate_state_snapshot_origins,
};
use symthaea_kubernetes_bridge::KUBERNETES_INTEGRATION_ID;

const ENDPOINT_MEMBERSHIP_ROLE: &str = "endpoint_membership";
const MAX_OWNERSHIP_DEPTH: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceCapacityRequest {
    pub service: EntityRef,
    pub workload: EntityRef,
    /// Usually `workload.replicas`; callers may select another explicitly
    /// modeled desired/observed dimension such as `daemonset.scheduled_nodes`.
    pub replica_dimension: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointCoverage {
    /// E1 replay cannot prove that every EndpointSlice for the Service was
    /// captured. All endpoint counts in this assessment are therefore lower
    /// bounds over the observed corpus, not exhaustive service capacity.
    ObservedSubsetLowerBound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesiredOriginObservation {
    pub assertion_id: String,
    pub origin: DesiredStateOrigin,
    pub explicit: bool,
}

/// Consensus over effective boolean condition values for unique resolved Pod
/// targets. Duplicate memberships for the same Pod do not increase these
/// counts. If different memberships disagree about the same Pod, that Pod is
/// counted once under `conflicting_targets` rather than forced to true/false.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UniquePodConditionConsensus {
    pub true_targets: usize,
    pub false_targets: usize,
    pub conflicting_targets: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceCapacityAssessment {
    pub service: EntityRef,
    pub workload: EntityRef,
    pub replica_dimension: String,
    pub replica_state: StateAssessment,
    pub desired_origins: Vec<DesiredOriginObservation>,
    pub endpoint_coverage: EndpointCoverage,
    /// EndpointSlice membership-level observations. A Pod repeated across slices
    /// contributes multiple memberships here because the duplicate references
    /// are themselves topology evidence.
    pub endpoint_memberships_observed: usize,
    pub endpoint_ready_observed: usize,
    pub endpoint_ready_explicit: usize,
    pub endpoint_serving_observed: usize,
    pub endpoint_serving_explicit: usize,
    pub endpoint_terminating_observed: usize,
    pub endpoint_terminating_explicit: usize,
    pub targetless_endpoint_memberships: usize,
    pub unresolved_target_references: usize,
    pub non_pod_target_references: usize,
    /// Unique resolved Pod targets observed across all EndpointSlice memberships.
    /// A Pod repeated during slice churn/rebalancing is counted once here.
    pub pod_targets_observed: usize,
    /// Effective ready/serving/terminating evidence deduplicated by Pod target.
    /// These are separate from the membership-level counters above.
    pub unique_pod_ready: UniquePodConditionConsensus,
    pub unique_pod_serving: UniquePodConditionConsensus,
    pub unique_pod_terminating: UniquePodConditionConsensus,
    /// Unique observed Pod targets with a structural ownership path to the
    /// requested workload.
    pub workload_owned_pod_targets_confirmed: usize,
    /// Unique Pod targets for which the partial topology cannot prove an
    /// ownership path to the requested workload. This is deliberately not
    /// labeled "foreign".
    pub workload_ownership_unresolved: usize,
}

/// Apply the existing registry admission boundaries to both topology and state
/// before producing a service-capacity assessment.
pub fn assess_registry_service_capacity(
    registry: &IntegrationRegistry,
    topology: &DiscoverySnapshot,
    state: &StateSnapshot,
    request: &ServiceCapacityRequest,
) -> Result<ServiceCapacityAssessment, IntegrationError> {
    let id = IntegrationId::new(KUBERNETES_INTEGRATION_ID);
    registry.admit_discovery_snapshot(&id, topology)?;
    registry.admit_state_snapshot(&id, state)?;
    assess_service_capacity(topology, state, request)
}

/// Assess one Service/workload pair without making an overall health judgment.
///
/// Both inputs must represent the same Kubernetes replay capture instant. This
/// function preserves non-completeness: observed endpoint counts are lower
/// bounds and absence of an endpoint or ownership edge is never negative proof.
pub fn assess_service_capacity(
    topology: &DiscoverySnapshot,
    state: &StateSnapshot,
    request: &ServiceCapacityRequest,
) -> Result<ServiceCapacityAssessment, IntegrationError> {
    validate_request(request)?;
    if topology.integration_id != KUBERNETES_INTEGRATION_ID
        || state.integration_id != KUBERNETES_INTEGRATION_ID
    {
        return Err(IntegrationError::InvalidRequest(format!(
            "service-capacity assessment requires `{KUBERNETES_INTEGRATION_ID}` topology/state"
        )));
    }
    if topology.discovered_at_unix_ms != state.collected_at_unix_ms {
        return Err(IntegrationError::InvalidRequest(format!(
            "service-capacity assessment requires same-capture evidence: topology={}, state={}",
            topology.discovered_at_unix_ms, state.collected_at_unix_ms
        )));
    }
    topology
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    state
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    validate_state_snapshot_origins(state)
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    require_entity(topology, &request.service, "service")?;
    require_entity(topology, &request.workload, "workload")?;

    let at = state.collected_at_unix_ms;
    let replica_state = assess_state_dimension(
        &state.assertions,
        &request.workload,
        &request.replica_dimension,
        at,
        StateComparisonPolicy::Exact,
    )
    .map_err(|error| {
        IntegrationError::InvalidOutput(format!("replica-state assessment failed: {error}"))
    })?;
    let desired_origins = collect_desired_origins(
        &state.assertions,
        &request.workload,
        &request.replica_dimension,
        at,
    )?;

    let entity_map = topology
        .entities
        .iter()
        .map(|entity| (entity.entity.canonical_key(), entity))
        .collect::<BTreeMap<_, _>>();

    let memberships = topology
        .relations
        .iter()
        .filter(|relation| {
            relation.from == request.service
                && relation.kind == RelationKind::Serves
                && relation.basis == RelationBasis::Structural
        })
        .filter_map(|relation| {
            entity_map
                .get(&relation.to.canonical_key())
                .filter(|entity| {
                    entity
                        .attributes
                        .get("symthaea.k8s.role")
                        .map(String::as_str)
                        == Some(ENDPOINT_MEMBERSHIP_ROLE)
                })
                .map(|_| relation.to.clone())
        })
        .collect::<BTreeSet<_>>();

    let mut assessment = ServiceCapacityAssessment {
        service: request.service.clone(),
        workload: request.workload.clone(),
        replica_dimension: request.replica_dimension.clone(),
        replica_state,
        desired_origins,
        endpoint_coverage: EndpointCoverage::ObservedSubsetLowerBound,
        endpoint_memberships_observed: memberships.len(),
        endpoint_ready_observed: 0,
        endpoint_ready_explicit: 0,
        endpoint_serving_observed: 0,
        endpoint_serving_explicit: 0,
        endpoint_terminating_observed: 0,
        endpoint_terminating_explicit: 0,
        targetless_endpoint_memberships: 0,
        unresolved_target_references: 0,
        non_pod_target_references: 0,
        pod_targets_observed: 0,
        unique_pod_ready: UniquePodConditionConsensus::default(),
        unique_pod_serving: UniquePodConditionConsensus::default(),
        unique_pod_terminating: UniquePodConditionConsensus::default(),
        workload_owned_pod_targets_confirmed: 0,
        workload_ownership_unresolved: 0,
    };
    let mut pod_targets = BTreeSet::new();
    let mut pod_condition_evidence = BTreeMap::<EntityRef, PodConditionEvidence>::new();
    let mut workload_owned_pod_targets = BTreeSet::new();
    let mut workload_ownership_unresolved = BTreeSet::new();

    for membership in memberships {
        let entity = entity_map
            .get(&membership.canonical_key())
            .ok_or_else(|| {
                IntegrationError::InvalidOutput(format!(
                    "EndpointSlice membership `{}` is missing from topology entities",
                    membership.canonical_key()
                ))
            })?;
        let conditions = accumulate_condition_counts(&mut assessment, &entity.attributes)?;

        let targets = topology
            .relations
            .iter()
            .filter(|relation| {
                relation.from == membership
                    && relation.basis == RelationBasis::Structural
                    && relation.kind == RelationKind::Other("Targets".into())
            })
            .map(|relation| relation.to.clone())
            .collect::<Vec<_>>();
        match targets.as_slice() {
            [] => {
                assessment.targetless_endpoint_memberships += 1;
            }
            [target] => {
                let target_entity = entity_map
                    .get(&target.canonical_key())
                    .ok_or_else(|| {
                        IntegrationError::InvalidOutput(format!(
                            "EndpointSlice target `{}` is missing from topology entities",
                            target.canonical_key()
                        ))
                    })?;
                if target_entity
                    .attributes
                    .get("k8s.reference")
                    .map(String::as_str)
                    == Some("true")
                {
                    assessment.unresolved_target_references += 1;
                    continue;
                }
                let is_pod = target.kind == "k8s_pod"
                    || target_entity
                        .attributes
                        .get("k8s.kind")
                        .map(String::as_str)
                        == Some("Pod");
                if !is_pod {
                    assessment.non_pod_target_references += 1;
                    continue;
                }
                pod_targets.insert(target.clone());
                pod_condition_evidence
                    .entry(target.clone())
                    .or_default()
                    .observe(conditions);
                if ownership_reaches(topology, target, &request.workload) {
                    workload_owned_pod_targets.insert(target.clone());
                    workload_ownership_unresolved.remove(target);
                } else if !workload_owned_pod_targets.contains(target) {
                    workload_ownership_unresolved.insert(target.clone());
                }
            }
            _ => {
                return Err(IntegrationError::InvalidOutput(format!(
                    "EndpointSlice membership `{}` has multiple structural Targets relations",
                    membership.canonical_key()
                )));
            }
        }
    }

    assessment.pod_targets_observed = pod_targets.len();
    assessment.unique_pod_ready = summarize_unique_pod_condition(
        pod_condition_evidence.values().map(|evidence| &evidence.ready),
    );
    assessment.unique_pod_serving = summarize_unique_pod_condition(
        pod_condition_evidence
            .values()
            .map(|evidence| &evidence.serving),
    );
    assessment.unique_pod_terminating = summarize_unique_pod_condition(
        pod_condition_evidence
            .values()
            .map(|evidence| &evidence.terminating),
    );
    assessment.workload_owned_pod_targets_confirmed = workload_owned_pod_targets.len();
    assessment.workload_ownership_unresolved = workload_ownership_unresolved.len();
    Ok(assessment)
}

fn validate_request(request: &ServiceCapacityRequest) -> Result<(), IntegrationError> {
    if request.replica_dimension.trim().is_empty() {
        return Err(IntegrationError::InvalidRequest(
            "service-capacity replica_dimension is empty".into(),
        ));
    }
    for (name, entity) in [
        ("service", &request.service),
        ("workload", &request.workload),
    ] {
        if entity.namespace.trim().is_empty()
            || entity.kind.trim().is_empty()
            || entity.id.trim().is_empty()
        {
            return Err(IntegrationError::InvalidRequest(format!(
                "service-capacity {name} EntityRef contains an empty field"
            )));
        }
    }
    Ok(())
}

fn require_entity(
    topology: &DiscoverySnapshot,
    entity: &EntityRef,
    role: &str,
) -> Result<(), IntegrationError> {
    if topology
        .entities
        .iter()
        .any(|candidate| candidate.entity == *entity)
    {
        Ok(())
    } else {
        Err(IntegrationError::InvalidRequest(format!(
            "service-capacity {role} `{}` is not present in topology",
            entity.canonical_key()
        )))
    }
}

fn collect_desired_origins(
    assertions: &[StateAssertion],
    subject: &EntityRef,
    dimension: &str,
    at_unix_ms: u64,
) -> Result<Vec<DesiredOriginObservation>, IntegrationError> {
    let mut origins = assertions
        .iter()
        .filter(|assertion| {
            assertion.role == StateRole::Desired
                && assertion.subject == *subject
                && assertion.dimension == dimension
                && assertion.is_active_at(at_unix_ms)
        })
        .map(|assertion| {
            let evidence = assertion
                .desired_state_origin()
                .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?
                .expect("desired assertion has desired origin evidence");
            Ok(DesiredOriginObservation {
                assertion_id: assertion.assertion_id.clone(),
                origin: evidence.origin,
                explicit: evidence.explicit,
            })
        })
        .collect::<Result<Vec<_>, IntegrationError>>()?;
    origins.sort_by(|left, right| {
        (left.origin, left.explicit, left.assertion_id.as_str()).cmp(&(
            right.origin,
            right.explicit,
            right.assertion_id.as_str(),
        ))
    });
    Ok(origins)
}

#[derive(Debug, Clone, Copy)]
struct EndpointConditionValues {
    ready: bool,
    serving: bool,
    terminating: bool,
}

#[derive(Debug, Default)]
struct PodConditionEvidence {
    ready: BTreeSet<bool>,
    serving: BTreeSet<bool>,
    terminating: BTreeSet<bool>,
}

impl PodConditionEvidence {
    fn observe(&mut self, values: EndpointConditionValues) {
        self.ready.insert(values.ready);
        self.serving.insert(values.serving);
        self.terminating.insert(values.terminating);
    }
}

fn accumulate_condition_counts(
    assessment: &mut ServiceCapacityAssessment,
    attributes: &BTreeMap<String, String>,
) -> Result<EndpointConditionValues, IntegrationError> {
    let ready = bool_attribute(attributes, "k8s.endpoint.ready")?;
    let ready_explicit = bool_attribute(attributes, "k8s.endpoint.ready.explicit")?;
    let serving = bool_attribute(attributes, "k8s.endpoint.serving")?;
    let serving_explicit = bool_attribute(attributes, "k8s.endpoint.serving.explicit")?;
    let terminating = bool_attribute(attributes, "k8s.endpoint.terminating")?;
    let terminating_explicit = bool_attribute(attributes, "k8s.endpoint.terminating.explicit")?;

    assessment.endpoint_ready_observed += usize::from(ready);
    assessment.endpoint_ready_explicit += usize::from(ready_explicit);
    assessment.endpoint_serving_observed += usize::from(serving);
    assessment.endpoint_serving_explicit += usize::from(serving_explicit);
    assessment.endpoint_terminating_observed += usize::from(terminating);
    assessment.endpoint_terminating_explicit += usize::from(terminating_explicit);
    Ok(EndpointConditionValues {
        ready,
        serving,
        terminating,
    })
}

fn summarize_unique_pod_condition<'a>(
    evidence: impl Iterator<Item = &'a BTreeSet<bool>>,
) -> UniquePodConditionConsensus {
    let mut summary = UniquePodConditionConsensus::default();
    for values in evidence {
        match (values.contains(&true), values.contains(&false)) {
            (true, true) => summary.conflicting_targets += 1,
            (true, false) => summary.true_targets += 1,
            (false, true) => summary.false_targets += 1,
            (false, false) => {}
        }
    }
    summary
}

fn bool_attribute(
    attributes: &BTreeMap<String, String>,
    key: &str,
) -> Result<bool, IntegrationError> {
    match attributes.get(key).map(String::as_str) {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(value) => Err(IntegrationError::InvalidOutput(format!(
            "EndpointSlice membership attribute `{key}` has non-boolean value `{value}`"
        ))),
        None => Err(IntegrationError::InvalidOutput(format!(
            "EndpointSlice membership is missing required attribute `{key}`"
        ))),
    }
}

fn ownership_reaches(
    topology: &DiscoverySnapshot,
    start: &EntityRef,
    target: &EntityRef,
) -> bool {
    if start == target {
        return true;
    }
    let mut queue = VecDeque::from([(start.clone(), 0usize)]);
    let mut visited = BTreeSet::new();
    visited.insert(start.clone());

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= MAX_OWNERSHIP_DEPTH {
            continue;
        }
        for relation in topology.relations.iter().filter(|relation| {
            relation.from == current
                && relation.kind == RelationKind::OwnedBy
                && relation.basis == RelationBasis::Structural
        }) {
            if relation.to == *target {
                return true;
            }
            if visited.insert(relation.to.clone()) {
                queue.push_back((relation.to.clone(), depth + 1));
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use symthaea_integration_core::{StateAssessmentStatus, StateValue};
    use symthaea_kubernetes_bridge::KubernetesReplayContext;
    use symthaea_kubernetes_endpointslice_bridge::augment_endpoint_slices;
    use symthaea_kubernetes_state_bridge::KubernetesStateReplay;
    use symthaea_kubernetes_state_origin_bridge::normalize_kubernetes_state_origins;

    fn documents() -> Vec<serde_json::Value> {
        vec![
            json!({
                "apiVersion":"v1","kind":"Namespace",
                "metadata":{"name":"shop","uid":"ns-1"}
            }),
            json!({
                "apiVersion":"apps/v1","kind":"Deployment",
                "metadata":{"name":"api","namespace":"shop","uid":"dep-1","generation":2},
                "spec":{"replicas":3},
                "status":{"replicas":3,"observedGeneration":2}
            }),
            json!({
                "apiVersion":"apps/v1","kind":"ReplicaSet",
                "metadata":{
                    "name":"api-rs","namespace":"shop","uid":"rs-1",
                    "ownerReferences":[{
                        "apiVersion":"apps/v1","kind":"Deployment","name":"api","uid":"dep-1"
                    }]
                }
            }),
            json!({
                "apiVersion":"v1","kind":"Service",
                "metadata":{"name":"api","namespace":"shop","uid":"svc-1"},
                "spec":{"selector":{"app":"api"}}
            }),
            pod("api-1", "pod-1"),
            pod("api-2", "pod-2"),
            pod("api-3", "pod-3"),
        ]
    }

    fn pod(name: &str, uid: &str) -> serde_json::Value {
        json!({
            "apiVersion":"v1","kind":"Pod",
            "metadata":{
                "name":name,"namespace":"shop","uid":uid,"labels":{"app":"api"},
                "ownerReferences":[{
                    "apiVersion":"apps/v1","kind":"ReplicaSet","name":"api-rs","uid":"rs-1"
                }]
            }
        })
    }

    fn endpoints() -> serde_json::Value {
        json!({
            "apiVersion":"discovery.k8s.io/v1",
            "kind":"EndpointSlice",
            "metadata":{
                "name":"api-abc","namespace":"shop","uid":"slice-1",
                "labels":{"kubernetes.io/service-name":"api"}
            },
            "addressType":"IPv4",
            "endpoints":[
                {
                    "addresses":["10.0.0.1"],
                    "conditions":{"ready":true,"serving":true,"terminating":false},
                    "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-1","uid":"pod-1"}
                },
                {
                    "addresses":["10.0.0.2"],
                    "conditions":{"ready":true,"serving":true,"terminating":false},
                    "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-2","uid":"pod-2"}
                },
                {
                    "addresses":["10.0.0.3"],
                    "conditions":{"ready":false,"serving":true,"terminating":true},
                    "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-3","uid":"pod-3"}
                }
            ]
        })
    }

    fn duplicate_api_1_slice(ready: bool) -> serde_json::Value {
        json!({
            "apiVersion":"discovery.k8s.io/v1",
            "kind":"EndpointSlice",
            "metadata":{
                "name":"api-rebalanced","namespace":"shop","uid":"slice-2",
                "labels":{"kubernetes.io/service-name":"api"}
            },
            "addressType":"IPv4",
            "endpoints":[{
                "addresses":["10.0.1.1"],
                "conditions":{"ready":ready,"serving":true,"terminating":false},
                "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"api-1","uid":"pod-1"}
            }]
        })
    }

    fn find_entity(topology: &DiscoverySnapshot, kind: &str, name: &str) -> EntityRef {
        topology
            .entities
            .iter()
            .find(|entity| {
                entity.attributes.get("k8s.kind").map(String::as_str) == Some(kind)
                    && entity.attributes.get("k8s.name").map(String::as_str) == Some(name)
            })
            .unwrap()
            .entity
            .clone()
    }

    fn request(topology: &DiscoverySnapshot) -> ServiceCapacityRequest {
        ServiceCapacityRequest {
            service: find_entity(topology, "Service", "api"),
            workload: find_entity(topology, "Deployment", "api"),
            replica_dimension: "workload.replicas".into(),
        }
    }

    #[test]
    fn replica_convergence_and_endpoint_readiness_remain_independent() {
        let docs = documents();
        let context = KubernetesReplayContext::default();
        let replay = KubernetesStateReplay::from_objects(context, &docs, 100).unwrap();
        let normalized = normalize_kubernetes_state_origins(replay.snapshot()).unwrap();
        let topology = augment_endpoint_slices(replay.topology(), &[endpoints()], 100).unwrap();
        let request = request(&topology);

        let mut registry = IntegrationRegistry::new();
        registry
            .register_discoverer(Arc::new(replay.topology().clone()))
            .unwrap();
        let assessment = assess_registry_service_capacity(
            &registry,
            &topology,
            &normalized.snapshot,
            &request,
        )
        .unwrap();

        assert_eq!(assessment.replica_state.status, StateAssessmentStatus::InSync);
        assert_eq!(
            assessment.replica_state.desired_value,
            Some(StateValue::Unsigned(3))
        );
        assert_eq!(
            assessment.replica_state.observed_value,
            Some(StateValue::Unsigned(3))
        );
        assert_eq!(
            assessment.endpoint_coverage,
            EndpointCoverage::ObservedSubsetLowerBound
        );
        assert_eq!(assessment.endpoint_memberships_observed, 3);
        assert_eq!(assessment.endpoint_ready_observed, 2);
        assert_eq!(assessment.endpoint_serving_observed, 3);
        assert_eq!(assessment.endpoint_terminating_observed, 1);
        assert_eq!(assessment.pod_targets_observed, 3);
        assert_eq!(assessment.unique_pod_ready.true_targets, 2);
        assert_eq!(assessment.unique_pod_ready.false_targets, 1);
        assert_eq!(assessment.unique_pod_ready.conflicting_targets, 0);
        assert_eq!(assessment.unique_pod_serving.true_targets, 3);
        assert_eq!(assessment.unique_pod_serving.false_targets, 0);
        assert_eq!(assessment.unique_pod_serving.conflicting_targets, 0);
        assert_eq!(assessment.unique_pod_terminating.true_targets, 1);
        assert_eq!(assessment.unique_pod_terminating.false_targets, 2);
        assert_eq!(assessment.unique_pod_terminating.conflicting_targets, 0);
        assert_eq!(assessment.workload_owned_pod_targets_confirmed, 3);
        assert_eq!(assessment.workload_ownership_unresolved, 0);
        assert_eq!(assessment.desired_origins.len(), 1);
        assert_eq!(
            assessment.desired_origins[0].origin,
            DesiredStateOrigin::Unspecified
        );
        assert!(!assessment.desired_origins[0].explicit);
    }

    #[test]
    fn repeated_pod_across_slices_does_not_inflate_unique_capacity() {
        let docs = documents();
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &docs,
            100,
        )
        .unwrap();
        let topology = augment_endpoint_slices(
            replay.topology(),
            &[endpoints(), duplicate_api_1_slice(true)],
            100,
        )
        .unwrap();
        let assessment =
            assess_service_capacity(&topology, replay.snapshot(), &request(&topology)).unwrap();

        assert_eq!(assessment.endpoint_memberships_observed, 4);
        assert_eq!(assessment.endpoint_ready_observed, 3);
        assert_eq!(assessment.pod_targets_observed, 3);
        assert_eq!(assessment.unique_pod_ready.true_targets, 2);
        assert_eq!(assessment.unique_pod_ready.false_targets, 1);
        assert_eq!(assessment.unique_pod_ready.conflicting_targets, 0);
        assert_eq!(assessment.workload_owned_pod_targets_confirmed, 3);
        assert_eq!(assessment.workload_ownership_unresolved, 0);
    }

    #[test]
    fn conflicting_slice_conditions_surface_per_unique_pod() {
        let docs = documents();
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &docs,
            100,
        )
        .unwrap();
        let topology = augment_endpoint_slices(
            replay.topology(),
            &[endpoints(), duplicate_api_1_slice(false)],
            100,
        )
        .unwrap();
        let assessment =
            assess_service_capacity(&topology, replay.snapshot(), &request(&topology)).unwrap();

        assert_eq!(assessment.endpoint_memberships_observed, 4);
        assert_eq!(assessment.pod_targets_observed, 3);
        assert_eq!(assessment.unique_pod_ready.true_targets, 1);
        assert_eq!(assessment.unique_pod_ready.false_targets, 1);
        assert_eq!(assessment.unique_pod_ready.conflicting_targets, 1);
    }

    #[test]
    fn mismatched_capture_times_are_rejected() {
        let docs = documents();
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &docs,
            100,
        )
        .unwrap();
        let mut state = replay.snapshot().clone();
        state.collected_at_unix_ms = 101;
        let topology = augment_endpoint_slices(replay.topology(), &[endpoints()], 100).unwrap();
        assert!(assess_service_capacity(&topology, &state, &request(&topology)).is_err());
    }

    #[test]
    fn missing_target_reference_is_counted_as_unknown_not_foreign() {
        let docs = documents();
        let replay = KubernetesStateReplay::from_objects(
            KubernetesReplayContext::default(),
            &docs,
            100,
        )
        .unwrap();
        let slice = json!({
            "apiVersion":"discovery.k8s.io/v1",
            "kind":"EndpointSlice",
            "metadata":{
                "name":"api-missing","namespace":"shop","uid":"slice-missing",
                "labels":{"kubernetes.io/service-name":"api"}
            },
            "addressType":"IPv4",
            "endpoints":[{
                "addresses":["10.0.0.9"],
                "targetRef":{"apiVersion":"v1","kind":"Pod","namespace":"shop","name":"missing","uid":"missing-pod"}
            }]
        });
        let topology = augment_endpoint_slices(replay.topology(), &[slice], 100).unwrap();
        let assessment =
            assess_service_capacity(&topology, replay.snapshot(), &request(&topology)).unwrap();
        assert_eq!(assessment.unresolved_target_references, 1);
        assert_eq!(assessment.workload_ownership_unresolved, 0);
    }
}
