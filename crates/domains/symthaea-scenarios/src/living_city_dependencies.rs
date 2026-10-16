//! Deterministic propagation of declared infrastructure dependency failures.
//!
//! This is a topology/logic screening layer. It does not calculate electrical
//! load flow, hydraulic pressure, thermal dynamics, capacity, failure rates, or
//! restoration times. Domain solvers and evidence-backed hazard profiles must
//! identify initiating failures; this module exposes the consequences of the
//! declared dependency graph so cross-sector cascades are not silently omitted.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceDomain {
    Electricity,
    Water,
    Wastewater,
    Cooling,
    Communications,
    Healthcare,
    EmergencyResponse,
    Mobility,
    Stormwater,
    Food,
    Housing,
    DigitalControl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceNode {
    pub id: String,
    pub domain: ServiceDomain,
}

/// Required dependencies propagate failure; degraded dependencies mark the
/// dependent service degraded but do not automatically declare it unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyMode {
    Required,
    Degraded,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceDependency {
    /// Service that depends on the prerequisite.
    pub dependent_id: String,
    /// Service whose failure affects the dependent.
    pub prerequisite_id: String,
    pub mode: DependencyMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Operational,
    Degraded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceOutcome {
    pub id: String,
    pub domain: ServiceDomain,
    pub status: ServiceStatus,
    pub initially_failed: bool,
    /// Required prerequisites failed in the final fixed-point result.
    pub failed_prerequisites: Vec<String>,
    /// Optional/degraded prerequisites failed in the final result.
    pub degraded_prerequisites: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyAssessment {
    /// Sorted by node ID for deterministic reports.
    pub outcomes: Vec<ServiceOutcome>,
    /// Number of rounds that discovered one or more additional failed services.
    pub propagation_rounds: usize,
}

impl DependencyAssessment {
    pub fn failed_ids(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.status == ServiceStatus::Failed)
            .map(|outcome| outcome.id.as_str())
            .collect()
    }

    pub fn degraded_ids(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.status == ServiceStatus::Degraded)
            .map(|outcome| outcome.id.as_str())
            .collect()
    }

    pub fn outcome(&self, id: &str) -> Option<&ServiceOutcome> {
        self.outcomes.iter().find(|outcome| outcome.id == id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyGraphError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for DependencyGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid infrastructure dependency graph {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for DependencyGraphError {}

fn invalid(field: &'static str, reason: &'static str) -> DependencyGraphError {
    DependencyGraphError { field, reason }
}

fn validate_graph<'a>(
    nodes: &'a [ServiceNode],
    dependencies: &[ServiceDependency],
) -> Result<BTreeMap<&'a str, ServiceDomain>, DependencyGraphError> {
    if nodes.is_empty() {
        return Err(invalid("nodes", "at least one service node is required"));
    }

    let mut index = BTreeMap::new();
    for node in nodes {
        if node.id.trim().is_empty() {
            return Err(invalid("nodes.id", "service IDs must be non-empty"));
        }
        if index.insert(node.id.as_str(), node.domain).is_some() {
            return Err(invalid("nodes.id", "service IDs must be unique"));
        }
    }

    let mut unique_edges = BTreeSet::new();
    for dependency in dependencies {
        if dependency.dependent_id.trim().is_empty()
            || dependency.prerequisite_id.trim().is_empty()
        {
            return Err(invalid("dependencies", "dependency endpoint IDs must be non-empty"));
        }
        if dependency.dependent_id == dependency.prerequisite_id {
            return Err(invalid("dependencies", "self-dependencies are invalid"));
        }
        if !index.contains_key(dependency.dependent_id.as_str())
            || !index.contains_key(dependency.prerequisite_id.as_str())
        {
            return Err(invalid(
                "dependencies",
                "every dependency endpoint must reference a declared service",
            ));
        }
        if !unique_edges.insert(dependency) {
            return Err(invalid("dependencies", "duplicate dependency edges are invalid"));
        }
    }
    Ok(index)
}

/// Propagate a set of initiating failures through the declared graph.
///
/// The algorithm is monotonic: a service can move from operational to failed,
/// but never be restored during this analysis. Restoration requires a separate
/// time-dependent model. Cycles are accepted and converge because the failed
/// set can only grow and is bounded by the finite node set.
pub fn assess_dependency_failures(
    nodes: &[ServiceNode],
    dependencies: &[ServiceDependency],
    initiating_failures: &[String],
) -> Result<DependencyAssessment, DependencyGraphError> {
    let node_index = validate_graph(nodes, dependencies)?;
    let mut initial: BTreeSet<String> = BTreeSet::new();
    for id in initiating_failures {
        if !node_index.contains_key(id.as_str()) {
            return Err(invalid(
                "initiating_failures",
                "every initiating failure must reference a declared service",
            ));
        }
        if !initial.insert(id.as_str().to_owned()) {
            return Err(invalid(
                "initiating_failures",
                "initiating failure IDs must be unique",
            ));
        }
    }

    let mut failed: BTreeSet<String> = initial.clone();
    let mut sorted_edges = dependencies.to_vec();
    sorted_edges.sort();

    let mut propagation_rounds = 0usize;
    loop {
        let mut newly_failed = Vec::new();
        for dependency in &sorted_edges {
            if dependency.mode == DependencyMode::Required
                && failed.contains(dependency.prerequisite_id.as_str())
                && !failed.contains(dependency.dependent_id.as_str())
            {
                newly_failed.push(dependency.dependent_id.clone());
            }
        }

        if newly_failed.is_empty() {
            break;
        }
        let prior_count = failed.len();
        failed.extend(newly_failed);
        if failed.len() == prior_count {
            break;
        }
        propagation_rounds += 1;
    }

    let mut outcomes = Vec::with_capacity(nodes.len());
    for (id, domain) in &node_index {
        let initially_failed = initial.contains(*id);
        let is_failed = failed.contains(*id);
        let mut failed_prerequisites = BTreeSet::new();
        let mut degraded_prerequisites = BTreeSet::new();

        for dependency in &sorted_edges {
            if dependency.dependent_id.as_str() != *id
                || !failed.contains(dependency.prerequisite_id.as_str())
            {
                continue;
            }
            match dependency.mode {
                DependencyMode::Required => {
                    failed_prerequisites.insert(dependency.prerequisite_id.clone());
                }
                DependencyMode::Degraded => {
                    degraded_prerequisites.insert(dependency.prerequisite_id.clone());
                }
            }
        }

        let status = if is_failed {
            ServiceStatus::Failed
        } else if !degraded_prerequisites.is_empty() {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Operational
        };

        outcomes.push(ServiceOutcome {
            id: (*id).to_owned(),
            domain: *domain,
            status,
            initially_failed,
            failed_prerequisites: failed_prerequisites.into_iter().collect(),
            degraded_prerequisites: degraded_prerequisites.into_iter().collect(),
        });
    }

    Ok(DependencyAssessment {
        outcomes,
        propagation_rounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, domain: ServiceDomain) -> ServiceNode {
        ServiceNode { id: id.into(), domain }
    }

    fn edge(dependent: &str, prerequisite: &str, mode: DependencyMode) -> ServiceDependency {
        ServiceDependency {
            dependent_id: dependent.into(),
            prerequisite_id: prerequisite.into(),
            mode,
        }
    }

    #[test]
    fn grid_outage_propagates_to_water_cooling_comms_dispatch_and_hospital() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("water-pump", ServiceDomain::Water),
            node("cooling-center", ServiceDomain::Cooling),
            node("telecom-tower", ServiceDomain::Communications),
            node("emergency-dispatch", ServiceDomain::EmergencyResponse),
            node("hospital", ServiceDomain::Healthcare),
        ];
        let dependencies = vec![
            edge("water-pump", "grid", DependencyMode::Required),
            edge("cooling-center", "grid", DependencyMode::Required),
            edge("telecom-tower", "grid", DependencyMode::Required),
            edge("emergency-dispatch", "telecom-tower", DependencyMode::Required),
            edge("hospital", "water-pump", DependencyMode::Required),
            edge("hospital", "emergency-dispatch", DependencyMode::Required),
        ];

        let result = assess_dependency_failures(&nodes, &dependencies, &["grid".into()]).unwrap();
        assert_eq!(
            result.failed_ids(),
            vec![
                "cooling-center",
                "emergency-dispatch",
                "grid",
                "hospital",
                "telecom-tower",
                "water-pump",
            ]
        );
        let hospital = result.outcome("hospital").unwrap();
        assert_eq!(hospital.status, ServiceStatus::Failed);
        assert_eq!(
            hospital.failed_prerequisites,
            vec!["emergency-dispatch".to_owned(), "water-pump".to_owned()]
        );
        assert!(result.propagation_rounds >= 2);
    }

    #[test]
    fn degraded_dependency_does_not_turn_service_into_failure() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("public-wifi", ServiceDomain::Communications),
        ];
        let dependencies = vec![edge("public-wifi", "grid", DependencyMode::Degraded)];
        let result = assess_dependency_failures(&nodes, &dependencies, &["grid".into()]).unwrap();
        assert_eq!(result.failed_ids(), vec!["grid"]);
        assert_eq!(result.degraded_ids(), vec!["public-wifi"]);
        assert_eq!(
            result.outcome("public-wifi").unwrap().degraded_prerequisites,
            vec!["grid".to_owned()]
        );
    }

    #[test]
    fn cycles_terminate_and_failure_propagates_monotonically() {
        let nodes = vec![
            node("a", ServiceDomain::DigitalControl),
            node("b", ServiceDomain::DigitalControl),
        ];
        let dependencies = vec![
            edge("a", "b", DependencyMode::Required),
            edge("b", "a", DependencyMode::Required),
        ];
        let result = assess_dependency_failures(&nodes, &dependencies, &["a".into()]).unwrap();
        assert_eq!(result.failed_ids(), vec!["a", "b"]);
        assert_eq!(result.propagation_rounds, 1);
    }

    #[test]
    fn no_initiating_failures_preserves_declared_services() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("water", ServiceDomain::Water),
        ];
        let dependencies = vec![edge("water", "grid", DependencyMode::Required)];
        let result = assess_dependency_failures(&nodes, &dependencies, &[]).unwrap();
        assert!(result.failed_ids().is_empty());
        assert!(result.degraded_ids().is_empty());
        assert_eq!(result.propagation_rounds, 0);
    }

    #[test]
    fn input_order_does_not_change_dependency_assessment() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("water-pump", ServiceDomain::Water),
            node("cooling-center", ServiceDomain::Cooling),
            node("telecom-tower", ServiceDomain::Communications),
            node("emergency-dispatch", ServiceDomain::EmergencyResponse),
        ];
        let dependencies = vec![
            edge("water-pump", "grid", DependencyMode::Required),
            edge("cooling-center", "grid", DependencyMode::Required),
            edge("telecom-tower", "grid", DependencyMode::Required),
            edge("emergency-dispatch", "telecom-tower", DependencyMode::Required),
            edge("cooling-center", "telecom-tower", DependencyMode::Degraded),
        ];
        let expected =
            assess_dependency_failures(&nodes, &dependencies, &["grid".into()]).unwrap();

        let mut reversed_nodes = nodes.clone();
        reversed_nodes.reverse();
        let mut reversed_dependencies = dependencies.clone();
        reversed_dependencies.reverse();
        let actual = assess_dependency_failures(
            &reversed_nodes,
            &reversed_dependencies,
            &["grid".into()],
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn adding_failures_never_restores_a_service_in_a_static_scenario() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("water-pump", ServiceDomain::Water),
            node("hospital", ServiceDomain::Healthcare),
            node("public-wifi", ServiceDomain::Communications),
        ];
        let dependencies = vec![
            edge("water-pump", "grid", DependencyMode::Required),
            edge("hospital", "water-pump", DependencyMode::Required),
            edge("public-wifi", "grid", DependencyMode::Degraded),
        ];
        let one_failure =
            assess_dependency_failures(&nodes, &dependencies, &["grid".into()]).unwrap();
        let two_failures = assess_dependency_failures(
            &nodes,
            &dependencies,
            &["grid".into(), "public-wifi".into()],
        )
        .unwrap();

        for id in one_failure.failed_ids() {
            assert!(
                two_failures.failed_ids().contains(&id),
                "service {id} was restored after adding another initiating failure"
            );
        }
    }

    #[test]
    fn unknown_endpoint_duplicate_edges_and_unknown_initiators_fail_closed() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("water", ServiceDomain::Water),
        ];
        assert_eq!(
            assess_dependency_failures(
                &nodes,
                &[edge("water", "unknown", DependencyMode::Required)],
                &["grid".into()],
            )
            .unwrap_err()
            .field,
            "dependencies"
        );

        let duplicate = vec![
            edge("water", "grid", DependencyMode::Required),
            edge("water", "grid", DependencyMode::Required),
        ];
        assert_eq!(
            assess_dependency_failures(&nodes, &duplicate, &[]).unwrap_err().field,
            "dependencies"
        );
        assert_eq!(
            assess_dependency_failures(&nodes, &[], &["missing".into()])
                .unwrap_err()
                .field,
            "initiating_failures"
        );
    }

    #[test]
    fn duplicate_nodes_self_edges_and_duplicate_initiators_fail_closed() {
        let nodes = vec![
            node("grid", ServiceDomain::Electricity),
            node("grid", ServiceDomain::Water),
        ];
        assert_eq!(
            assess_dependency_failures(&nodes, &[], &[]).unwrap_err().field,
            "nodes.id"
        );

        let nodes = vec![node("grid", ServiceDomain::Electricity)];
        assert_eq!(
            assess_dependency_failures(
                &nodes,
                &[edge("grid", "grid", DependencyMode::Required)],
                &[],
            )
            .unwrap_err()
            .field,
            "dependencies"
        );
        assert_eq!(
            assess_dependency_failures(&nodes, &[], &["grid".into(), "grid".into()])
                .unwrap_err()
                .field,
            "initiating_failures"
        );
    }
}
