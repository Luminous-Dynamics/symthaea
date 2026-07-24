// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent implementation verification.
//!
//! A second executable is not automatically independent. This module records
//! implementation domains, requires complete result matrices, and refuses to
//! treat correlated implementations or missing evidence as verification.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VerificationCriticality {
    Development,
    Mission,
    SafetyCritical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationImplementation {
    pub implementation_id: String,
    pub organization_id: String,
    pub team_id: String,
    pub language: String,
    pub compiler_family: String,
    pub algorithm_family: String,
    pub source_digest: String,
    pub dependency_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationVector {
    pub vector_id: String,
    pub input_digest: String,
    pub criticality: VerificationCriticality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationResult {
    pub implementation_id: String,
    pub vector_id: String,
    pub output_digest: String,
    pub metrics: BTreeMap<String, f64>,
    pub evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndependentVerificationPolicy {
    pub minimum_implementations: usize,
    pub minimum_organizations: usize,
    pub require_distinct_source: bool,
    pub require_distinct_dependency_graph: bool,
    pub require_distinct_algorithm_for: BTreeSet<VerificationCriticality>,
    pub exact_output_for: BTreeSet<VerificationCriticality>,
    pub maximum_metric_delta: BTreeMap<String, f64>,
}

impl Default for IndependentVerificationPolicy {
    fn default() -> Self {
        Self {
            minimum_implementations: 2,
            minimum_organizations: 2,
            require_distinct_source: true,
            require_distinct_dependency_graph: true,
            require_distinct_algorithm_for: BTreeSet::from([
                VerificationCriticality::SafetyCritical,
            ]),
            exact_output_for: BTreeSet::from([
                VerificationCriticality::Mission,
                VerificationCriticality::SafetyCritical,
            ]),
            maximum_metric_delta: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndependentVerificationStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndependentVerificationIssue {
    EmptyIdentity,
    DuplicateImplementation(String),
    DuplicateVector(String),
    InsufficientImplementations {
        required: usize,
        observed: usize,
    },
    InsufficientOrganizations {
        required: usize,
        observed: usize,
    },
    SharedSourceDigest {
        left: String,
        right: String,
    },
    SharedDependencyDigest {
        left: String,
        right: String,
    },
    SharedAlgorithmFamily {
        vector_id: String,
        family: String,
    },
    MissingResult {
        implementation_id: String,
        vector_id: String,
    },
    DuplicateResult {
        implementation_id: String,
        vector_id: String,
    },
    UnknownImplementation(String),
    UnknownVector(String),
    MissingEvidence {
        implementation_id: String,
        vector_id: String,
    },
    InvalidMetric {
        implementation_id: String,
        vector_id: String,
        metric: String,
    },
    OutputDisagreement {
        vector_id: String,
    },
    MetricDisagreement {
        vector_id: String,
        metric: String,
        observed: f64,
        maximum: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndependentVerificationReport {
    pub status: IndependentVerificationStatus,
    pub implementations: usize,
    pub vectors: usize,
    pub complete_cells: usize,
    pub issues: Vec<IndependentVerificationIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependentVerificationError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct IndependentVerificationGate {
    policy: IndependentVerificationPolicy,
}

impl IndependentVerificationGate {
    pub fn new(
        policy: IndependentVerificationPolicy,
    ) -> Result<Self, IndependentVerificationError> {
        if policy.minimum_implementations < 2
            || policy.minimum_organizations == 0
            || policy.minimum_organizations > policy.minimum_implementations
            || policy
                .maximum_metric_delta
                .values()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(IndependentVerificationError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        implementations: &[VerificationImplementation],
        vectors: &[VerificationVector],
        results: &[VerificationResult],
    ) -> IndependentVerificationReport {
        let mut issues = Vec::new();
        let mut impls = BTreeMap::new();
        let mut vector_map = BTreeMap::new();

        for implementation in implementations {
            if fields_empty([
                &implementation.implementation_id,
                &implementation.organization_id,
                &implementation.team_id,
                &implementation.language,
                &implementation.compiler_family,
                &implementation.algorithm_family,
                &implementation.source_digest,
                &implementation.dependency_digest,
            ]) {
                issues.push(IndependentVerificationIssue::EmptyIdentity);
            }
            if impls
                .insert(implementation.implementation_id.as_str(), implementation)
                .is_some()
            {
                issues.push(IndependentVerificationIssue::DuplicateImplementation(
                    implementation.implementation_id.clone(),
                ));
            }
        }
        for vector in vectors {
            if fields_empty([&vector.vector_id, &vector.input_digest]) {
                issues.push(IndependentVerificationIssue::EmptyIdentity);
            }
            if vector_map
                .insert(vector.vector_id.as_str(), vector)
                .is_some()
            {
                issues.push(IndependentVerificationIssue::DuplicateVector(
                    vector.vector_id.clone(),
                ));
            }
        }

        if impls.len() < self.policy.minimum_implementations {
            issues.push(IndependentVerificationIssue::InsufficientImplementations {
                required: self.policy.minimum_implementations,
                observed: impls.len(),
            });
        }
        let organizations = implementations
            .iter()
            .map(|i| i.organization_id.as_str())
            .collect::<BTreeSet<_>>();
        if organizations.len() < self.policy.minimum_organizations {
            issues.push(IndependentVerificationIssue::InsufficientOrganizations {
                required: self.policy.minimum_organizations,
                observed: organizations.len(),
            });
        }

        for (index, left) in implementations.iter().enumerate() {
            for right in implementations.iter().skip(index + 1) {
                if self.policy.require_distinct_source && left.source_digest == right.source_digest
                {
                    issues.push(IndependentVerificationIssue::SharedSourceDigest {
                        left: left.implementation_id.clone(),
                        right: right.implementation_id.clone(),
                    });
                }
                if self.policy.require_distinct_dependency_graph
                    && left.dependency_digest == right.dependency_digest
                {
                    issues.push(IndependentVerificationIssue::SharedDependencyDigest {
                        left: left.implementation_id.clone(),
                        right: right.implementation_id.clone(),
                    });
                }
            }
        }

        let mut cells = BTreeMap::<(&str, &str), Vec<&VerificationResult>>::new();
        for result in results {
            if !impls.contains_key(result.implementation_id.as_str()) {
                issues.push(IndependentVerificationIssue::UnknownImplementation(
                    result.implementation_id.clone(),
                ));
            }
            if !vector_map.contains_key(result.vector_id.as_str()) {
                issues.push(IndependentVerificationIssue::UnknownVector(
                    result.vector_id.clone(),
                ));
            }
            if result.evidence_id.trim().is_empty() || result.output_digest.trim().is_empty() {
                issues.push(IndependentVerificationIssue::MissingEvidence {
                    implementation_id: result.implementation_id.clone(),
                    vector_id: result.vector_id.clone(),
                });
            }
            for (metric, value) in &result.metrics {
                if metric.trim().is_empty() || !value.is_finite() {
                    issues.push(IndependentVerificationIssue::InvalidMetric {
                        implementation_id: result.implementation_id.clone(),
                        vector_id: result.vector_id.clone(),
                        metric: metric.clone(),
                    });
                }
            }
            cells
                .entry((result.implementation_id.as_str(), result.vector_id.as_str()))
                .or_default()
                .push(result);
        }

        let mut complete_cells = 0usize;
        for implementation in implementations {
            for vector in vectors {
                let key = (
                    implementation.implementation_id.as_str(),
                    vector.vector_id.as_str(),
                );
                match cells.get(&key) {
                    None => issues.push(IndependentVerificationIssue::MissingResult {
                        implementation_id: implementation.implementation_id.clone(),
                        vector_id: vector.vector_id.clone(),
                    }),
                    Some(entries) if entries.len() > 1 => {
                        issues.push(IndependentVerificationIssue::DuplicateResult {
                            implementation_id: implementation.implementation_id.clone(),
                            vector_id: vector.vector_id.clone(),
                        });
                    }
                    Some(_) => complete_cells += 1,
                }
            }
        }

        for vector in vectors {
            let matching = results
                .iter()
                .filter(|result| result.vector_id == vector.vector_id)
                .collect::<Vec<_>>();
            if self
                .policy
                .require_distinct_algorithm_for
                .contains(&vector.criticality)
            {
                let families = matching
                    .iter()
                    .filter_map(|result| impls.get(result.implementation_id.as_str()))
                    .map(|implementation| implementation.algorithm_family.as_str())
                    .collect::<BTreeSet<_>>();
                if matching.len() >= 2 && families.len() < 2 {
                    issues.push(IndependentVerificationIssue::SharedAlgorithmFamily {
                        vector_id: vector.vector_id.clone(),
                        family: families.iter().next().copied().unwrap_or("").to_string(),
                    });
                }
            }
            if self.policy.exact_output_for.contains(&vector.criticality) {
                let outputs = matching
                    .iter()
                    .map(|result| result.output_digest.as_str())
                    .collect::<BTreeSet<_>>();
                if outputs.len() > 1 {
                    issues.push(IndependentVerificationIssue::OutputDisagreement {
                        vector_id: vector.vector_id.clone(),
                    });
                }
            }
            for (metric, maximum) in &self.policy.maximum_metric_delta {
                let values = matching
                    .iter()
                    .filter_map(|result| result.metrics.get(metric).copied())
                    .filter(|value| value.is_finite())
                    .collect::<Vec<_>>();
                if values.len() >= 2 {
                    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let maximum_observed = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let delta = maximum_observed - minimum;
                    if delta > *maximum {
                        issues.push(IndependentVerificationIssue::MetricDisagreement {
                            vector_id: vector.vector_id.clone(),
                            metric: metric.clone(),
                            observed: delta,
                            maximum: *maximum,
                        });
                    }
                }
            }
        }

        let status = if issues.iter().any(is_failure) {
            IndependentVerificationStatus::Fail
        } else if issues.is_empty() {
            IndependentVerificationStatus::Pass
        } else {
            IndependentVerificationStatus::Incomplete
        };
        IndependentVerificationReport {
            status,
            implementations: implementations.len(),
            vectors: vectors.len(),
            complete_cells,
            issues,
        }
    }
}

fn fields_empty<'a>(fields: impl IntoIterator<Item = &'a String>) -> bool {
    fields.into_iter().any(|field| field.trim().is_empty())
}

fn is_failure(issue: &IndependentVerificationIssue) -> bool {
    matches!(
        issue,
        IndependentVerificationIssue::SharedSourceDigest { .. }
            | IndependentVerificationIssue::SharedDependencyDigest { .. }
            | IndependentVerificationIssue::SharedAlgorithmFamily { .. }
            | IndependentVerificationIssue::OutputDisagreement { .. }
            | IndependentVerificationIssue::MetricDisagreement { .. }
            | IndependentVerificationIssue::InvalidMetric { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn implementation(id: &str, org: &str, algorithm: &str) -> VerificationImplementation {
        VerificationImplementation {
            implementation_id: id.into(),
            organization_id: org.into(),
            team_id: format!("team-{id}"),
            language: if id == "primary" { "Rust" } else { "C++" }.into(),
            compiler_family: if id == "primary" { "LLVM" } else { "GCC" }.into(),
            algorithm_family: algorithm.into(),
            source_digest: format!("sha256:source-{id}"),
            dependency_digest: format!("sha256:deps-{id}"),
        }
    }

    fn vector() -> VerificationVector {
        VerificationVector {
            vector_id: "hover-trim".into(),
            input_digest: "sha256:input".into(),
            criticality: VerificationCriticality::SafetyCritical,
        }
    }

    fn result(id: &str, digest: &str, metric: f64) -> VerificationResult {
        VerificationResult {
            implementation_id: id.into(),
            vector_id: "hover-trim".into(),
            output_digest: digest.into(),
            metrics: BTreeMap::from([("residual".into(), metric)]),
            evidence_id: format!("evidence-{id}"),
        }
    }

    #[test]
    fn independent_matching_implementations_pass() {
        let mut policy = IndependentVerificationPolicy::default();
        policy.maximum_metric_delta.insert("residual".into(), 0.01);
        let gate = IndependentVerificationGate::new(policy).unwrap();
        let report = gate.assess(
            &[
                implementation("primary", "luminous", "momentum-theory"),
                implementation("oracle", "independent-lab", "blade-element"),
            ],
            &[vector()],
            &[
                result("primary", "sha256:out", 0.10),
                result("oracle", "sha256:out", 0.105),
            ],
        );
        assert_eq!(report.status, IndependentVerificationStatus::Pass);
        assert_eq!(report.complete_cells, 2);
    }

    #[test]
    fn correlated_or_disagreeing_implementations_fail() {
        let gate =
            IndependentVerificationGate::new(IndependentVerificationPolicy::default()).unwrap();
        let mut oracle = implementation("oracle", "luminous", "momentum-theory");
        oracle.source_digest = "sha256:source-primary".into();
        let report = gate.assess(
            &[
                implementation("primary", "luminous", "momentum-theory"),
                oracle,
            ],
            &[vector()],
            &[
                result("primary", "sha256:a", 0.1),
                result("oracle", "sha256:b", 0.1),
            ],
        );
        assert_eq!(report.status, IndependentVerificationStatus::Fail);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            IndependentVerificationIssue::OutputDisagreement { .. }
        )));
    }

    #[test]
    fn missing_matrix_cell_is_incomplete() {
        let gate =
            IndependentVerificationGate::new(IndependentVerificationPolicy::default()).unwrap();
        let report = gate.assess(
            &[
                implementation("primary", "luminous", "momentum-theory"),
                implementation("oracle", "independent-lab", "blade-element"),
            ],
            &[vector()],
            &[result("primary", "sha256:out", 0.1)],
        );
        assert_eq!(report.status, IndependentVerificationStatus::Incomplete);
    }
}
