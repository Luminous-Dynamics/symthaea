// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Essential-service resilience and recovery-objective assessment.
//!
//! Services are evaluated as a dependency graph with bounded outage and
//! recovery objectives. A declared fallback only counts when it is available,
//! independent enough, and proven by evidence.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ServiceCriticality {
    Optional,
    Required,
    Essential,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResilientService {
    pub service_id: String,
    pub criticality: ServiceCriticality,
    pub dependency_ids: Vec<String>,
    pub failure_domain: String,
    pub fallback_service_id: Option<String>,
    pub maximum_outage_ms: u64,
    pub recovery_objective_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceAvailability {
    Available,
    Degraded,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceObservation {
    pub service_id: String,
    pub availability: ServiceAvailability,
    pub outage_started_at_ms: Option<u64>,
    pub recovered_at_ms: Option<u64>,
    pub fallback_active: bool,
    pub evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceResiliencePolicy {
    pub required_service_ids: BTreeSet<String>,
    pub require_independent_fallback_for: BTreeSet<ServiceCriticality>,
    pub fail_on_dependency_cycle: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceResilienceStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceResilienceIssue {
    EmptyIdentity,
    DuplicateService(String),
    DuplicateObservation(String),
    MissingRequiredService(String),
    MissingObservation(String),
    UnknownObservation(String),
    MissingDependency {
        service_id: String,
        dependency_id: String,
    },
    DependencyCycle(Vec<String>),
    UnknownFallback {
        service_id: String,
        fallback_id: String,
    },
    SharedFallbackFailureDomain {
        service_id: String,
        fallback_id: String,
    },
    EssentialServiceUnavailable(String),
    OutageExceeded {
        service_id: String,
        observed_ms: u64,
        maximum_ms: u64,
    },
    RecoveryObjectiveMissed {
        service_id: String,
        observed_ms: u64,
        maximum_ms: u64,
    },
    DependencyUnavailable {
        service_id: String,
        dependency_id: String,
    },
    FallbackDeclaredButInactive(String),
    MissingEvidence(String),
    UnknownAvailability(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceResilienceReport {
    pub status: ServiceResilienceStatus,
    pub assessed_services: usize,
    pub available_services: usize,
    pub degraded_services: usize,
    pub unavailable_services: usize,
    pub issues: Vec<ServiceResilienceIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceResilienceError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct ServiceResilienceAssessor {
    policy: ServiceResiliencePolicy,
}

impl ServiceResilienceAssessor {
    pub fn new(policy: ServiceResiliencePolicy) -> Result<Self, ServiceResilienceError> {
        if policy.required_service_ids.is_empty()
            || policy
                .required_service_ids
                .iter()
                .any(|id| id.trim().is_empty())
        {
            return Err(ServiceResilienceError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        services: &[ResilientService],
        observations: &[ServiceObservation],
        now_ms: u64,
    ) -> ServiceResilienceReport {
        let mut issues = Vec::new();
        let mut by_id = BTreeMap::<&str, &ResilientService>::new();
        for service in services {
            if service.service_id.trim().is_empty()
                || service.failure_domain.trim().is_empty()
                || service.maximum_outage_ms == 0
                || service.recovery_objective_ms == 0
            {
                issues.push(ServiceResilienceIssue::EmptyIdentity);
            }
            if by_id.insert(service.service_id.as_str(), service).is_some() {
                issues.push(ServiceResilienceIssue::DuplicateService(
                    service.service_id.clone(),
                ));
            }
        }
        for required in &self.policy.required_service_ids {
            if !by_id.contains_key(required.as_str()) {
                issues.push(ServiceResilienceIssue::MissingRequiredService(
                    required.clone(),
                ));
            }
        }

        for service in services {
            for dependency in &service.dependency_ids {
                if !by_id.contains_key(dependency.as_str()) {
                    issues.push(ServiceResilienceIssue::MissingDependency {
                        service_id: service.service_id.clone(),
                        dependency_id: dependency.clone(),
                    });
                }
            }
            if let Some(fallback_id) = &service.fallback_service_id {
                match by_id.get(fallback_id.as_str()) {
                    None => issues.push(ServiceResilienceIssue::UnknownFallback {
                        service_id: service.service_id.clone(),
                        fallback_id: fallback_id.clone(),
                    }),
                    Some(fallback)
                        if self
                            .policy
                            .require_independent_fallback_for
                            .contains(&service.criticality)
                            && fallback.failure_domain == service.failure_domain =>
                    {
                        issues.push(ServiceResilienceIssue::SharedFallbackFailureDomain {
                            service_id: service.service_id.clone(),
                            fallback_id: fallback_id.clone(),
                        });
                    }
                    Some(_) => {}
                }
            }
        }

        if self.policy.fail_on_dependency_cycle {
            for cycle in dependency_cycles(services, &by_id) {
                issues.push(ServiceResilienceIssue::DependencyCycle(cycle));
            }
        }

        let mut observation_by_id = BTreeMap::<&str, &ServiceObservation>::new();
        for observation in observations {
            if !by_id.contains_key(observation.service_id.as_str()) {
                issues.push(ServiceResilienceIssue::UnknownObservation(
                    observation.service_id.clone(),
                ));
            }
            if observation_by_id
                .insert(observation.service_id.as_str(), observation)
                .is_some()
            {
                issues.push(ServiceResilienceIssue::DuplicateObservation(
                    observation.service_id.clone(),
                ));
            }
        }

        let mut available = 0usize;
        let mut degraded = 0usize;
        let mut unavailable = 0usize;
        for service in services {
            let Some(observation) = observation_by_id.get(service.service_id.as_str()) else {
                issues.push(ServiceResilienceIssue::MissingObservation(
                    service.service_id.clone(),
                ));
                continue;
            };
            if observation
                .evidence_id
                .as_ref()
                .is_none_or(|id| id.trim().is_empty())
            {
                issues.push(ServiceResilienceIssue::MissingEvidence(
                    service.service_id.clone(),
                ));
            }
            match observation.availability {
                ServiceAvailability::Available => available += 1,
                ServiceAvailability::Degraded => degraded += 1,
                ServiceAvailability::Unavailable => unavailable += 1,
                ServiceAvailability::Unknown => {
                    issues.push(ServiceResilienceIssue::UnknownAvailability(
                        service.service_id.clone(),
                    ));
                    continue;
                }
            }

            for dependency in &service.dependency_ids {
                if observation_by_id.get(dependency.as_str()).is_some_and(
                    |dependency_observation| {
                        matches!(
                            dependency_observation.availability,
                            ServiceAvailability::Unavailable | ServiceAvailability::Unknown
                        )
                    },
                ) && !observation.fallback_active
                {
                    issues.push(ServiceResilienceIssue::DependencyUnavailable {
                        service_id: service.service_id.clone(),
                        dependency_id: dependency.clone(),
                    });
                }
            }

            if observation.availability == ServiceAvailability::Unavailable {
                let outage = observation.outage_started_at_ms.map(|start| {
                    observation
                        .recovered_at_ms
                        .unwrap_or(now_ms)
                        .saturating_sub(start)
                });
                if let Some(outage_ms) = outage {
                    if outage_ms > service.maximum_outage_ms {
                        issues.push(ServiceResilienceIssue::OutageExceeded {
                            service_id: service.service_id.clone(),
                            observed_ms: outage_ms,
                            maximum_ms: service.maximum_outage_ms,
                        });
                    }
                    if observation.recovered_at_ms.is_some()
                        && outage_ms > service.recovery_objective_ms
                    {
                        issues.push(ServiceResilienceIssue::RecoveryObjectiveMissed {
                            service_id: service.service_id.clone(),
                            observed_ms: outage_ms,
                            maximum_ms: service.recovery_objective_ms,
                        });
                    }
                }
                if service.criticality == ServiceCriticality::Essential
                    && !(service.fallback_service_id.is_some() && observation.fallback_active)
                {
                    issues.push(ServiceResilienceIssue::EssentialServiceUnavailable(
                        service.service_id.clone(),
                    ));
                }
                if service.fallback_service_id.is_some() && !observation.fallback_active {
                    issues.push(ServiceResilienceIssue::FallbackDeclaredButInactive(
                        service.service_id.clone(),
                    ));
                }
            }
        }

        let status = if issues.iter().any(is_failure) {
            ServiceResilienceStatus::Fail
        } else if issues.is_empty() {
            ServiceResilienceStatus::Pass
        } else {
            ServiceResilienceStatus::Incomplete
        };
        ServiceResilienceReport {
            status,
            assessed_services: services.len(),
            available_services: available,
            degraded_services: degraded,
            unavailable_services: unavailable,
            issues,
        }
    }
}

fn dependency_cycles<'a>(
    services: &'a [ResilientService],
    by_id: &BTreeMap<&'a str, &'a ResilientService>,
) -> Vec<Vec<String>> {
    let mut cycles = Vec::new();
    for service in services {
        let mut queue = VecDeque::from([(
            service.service_id.as_str(),
            vec![service.service_id.clone()],
        )]);
        while let Some((current, path)) = queue.pop_front() {
            let Some(node) = by_id.get(current) else {
                continue;
            };
            for dependency in &node.dependency_ids {
                if dependency == &service.service_id {
                    let mut cycle = path.clone();
                    cycle.push(dependency.clone());
                    cycles.push(cycle);
                    queue.clear();
                    break;
                }
                if !path.contains(dependency) && path.len() <= services.len() {
                    let mut next_path = path.clone();
                    next_path.push(dependency.clone());
                    queue.push_back((dependency.as_str(), next_path));
                }
            }
        }
    }
    cycles.sort();
    cycles.dedup();
    cycles
}

fn is_failure(issue: &ServiceResilienceIssue) -> bool {
    matches!(
        issue,
        ServiceResilienceIssue::DependencyCycle(_)
            | ServiceResilienceIssue::SharedFallbackFailureDomain { .. }
            | ServiceResilienceIssue::EssentialServiceUnavailable(_)
            | ServiceResilienceIssue::OutageExceeded { .. }
            | ServiceResilienceIssue::RecoveryObjectiveMissed { .. }
            | ServiceResilienceIssue::DependencyUnavailable { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service(id: &str, criticality: ServiceCriticality, domain: &str) -> ResilientService {
        ResilientService {
            service_id: id.into(),
            criticality,
            dependency_ids: Vec::new(),
            failure_domain: domain.into(),
            fallback_service_id: None,
            maximum_outage_ms: 500,
            recovery_objective_ms: 300,
        }
    }

    fn policy() -> ServiceResiliencePolicy {
        ServiceResiliencePolicy {
            required_service_ids: BTreeSet::from(["flight-control".into()]),
            require_independent_fallback_for: BTreeSet::from([ServiceCriticality::Essential]),
            fail_on_dependency_cycle: true,
        }
    }

    #[test]
    fn independent_fallback_preserves_essential_service() {
        let mut primary = service("flight-control", ServiceCriticality::Essential, "compute-a");
        primary.fallback_service_id = Some("fallback".into());
        let fallback = service("fallback", ServiceCriticality::Essential, "compute-b");
        let report = ServiceResilienceAssessor::new(policy()).unwrap().assess(
            &[primary, fallback],
            &[
                ServiceObservation {
                    service_id: "flight-control".into(),
                    availability: ServiceAvailability::Unavailable,
                    outage_started_at_ms: Some(1_000),
                    recovered_at_ms: None,
                    fallback_active: true,
                    evidence_id: Some("evt-1".into()),
                },
                ServiceObservation {
                    service_id: "fallback".into(),
                    availability: ServiceAvailability::Available,
                    outage_started_at_ms: None,
                    recovered_at_ms: None,
                    fallback_active: false,
                    evidence_id: Some("evt-2".into()),
                },
            ],
            1_200,
        );
        assert_eq!(report.status, ServiceResilienceStatus::Pass);
    }

    #[test]
    fn shared_domain_fallback_fails() {
        let mut primary = service("flight-control", ServiceCriticality::Essential, "compute-a");
        primary.fallback_service_id = Some("fallback".into());
        let fallback = service("fallback", ServiceCriticality::Essential, "compute-a");
        let report = ServiceResilienceAssessor::new(policy()).unwrap().assess(
            &[primary, fallback],
            &[
                ServiceObservation {
                    service_id: "flight-control".into(),
                    availability: ServiceAvailability::Unavailable,
                    outage_started_at_ms: Some(1_000),
                    recovered_at_ms: None,
                    fallback_active: true,
                    evidence_id: Some("evt-1".into()),
                },
                ServiceObservation {
                    service_id: "fallback".into(),
                    availability: ServiceAvailability::Available,
                    outage_started_at_ms: None,
                    recovered_at_ms: None,
                    fallback_active: false,
                    evidence_id: Some("evt-2".into()),
                },
            ],
            1_100,
        );
        assert_eq!(report.status, ServiceResilienceStatus::Fail);
    }
}
