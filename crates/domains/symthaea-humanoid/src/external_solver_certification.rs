// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification evidence for external sparse-QP deployments.
//!
//! Transport success alone is not solver qualification. A production backend
//! must cover distinct matrix structures, report bounded residuals, reuse warm
//! state when expected, and produce no malformed or rejected responses.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSparseQpQualificationCase {
    pub case_id: String,
    pub backend_id: String,
    pub structure_fingerprint: u64,
    pub state_fingerprint: u64,
    pub response_admitted: bool,
    pub converged: bool,
    pub warm_start_requested: bool,
    pub warm_start_used: bool,
    pub symbolic_pattern_reused: bool,
    pub maximum_equality_residual: f64,
    pub maximum_bound_violation: f64,
    pub elapsed_micros: u64,
}

impl ExternalSparseQpQualificationCase {
    pub fn validate(&self) -> bool {
        !self.case_id.trim().is_empty()
            && !self.backend_id.trim().is_empty()
            && self.structure_fingerprint != 0
            && self.state_fingerprint != 0
            && self.maximum_equality_residual.is_finite()
            && self.maximum_equality_residual >= 0.0
            && self.maximum_bound_violation.is_finite()
            && self.maximum_bound_violation >= 0.0
            && self.elapsed_micros > 0
            && (!self.warm_start_used || self.warm_start_requested)
            && (!self.symbolic_pattern_reused || self.warm_start_requested)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExternalSparseQpQualificationCriteria {
    pub minimum_cases: usize,
    pub minimum_distinct_states: usize,
    pub minimum_distinct_structures: usize,
    pub minimum_warm_started_cases: usize,
    pub minimum_symbolic_reuse_cases: usize,
    pub maximum_equality_residual: f64,
    pub maximum_bound_violation: f64,
    pub maximum_elapsed_micros: u64,
}

impl Default for ExternalSparseQpQualificationCriteria {
    fn default() -> Self {
        Self {
            minimum_cases: 256,
            minimum_distinct_states: 256,
            minimum_distinct_structures: 3,
            minimum_warm_started_cases: 128,
            minimum_symbolic_reuse_cases: 128,
            maximum_equality_residual: 1.0e-6,
            maximum_bound_violation: 1.0e-6,
            maximum_elapsed_micros: 2_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSparseQpQualificationCertificate {
    pub backend_id: Option<String>,
    pub valid_cases: usize,
    pub admitted_cases: usize,
    pub converged_cases: usize,
    pub distinct_states: usize,
    pub distinct_structures: usize,
    pub warm_started_cases: usize,
    pub symbolic_reuse_cases: usize,
    pub maximum_observed_equality_residual: f64,
    pub maximum_observed_bound_violation: f64,
    pub maximum_observed_elapsed_micros: u64,
    pub passed: bool,
    pub failures: Vec<String>,
}

pub fn certify_external_sparse_qp_backend(
    cases: &[ExternalSparseQpQualificationCase],
    criteria: ExternalSparseQpQualificationCriteria,
) -> ExternalSparseQpQualificationCertificate {
    let valid = cases
        .iter()
        .filter(|case| case.validate())
        .collect::<Vec<_>>();
    let backend_ids = valid
        .iter()
        .map(|case| case.backend_id.as_str())
        .collect::<BTreeSet<_>>();
    let backend_id = (backend_ids.len() == 1)
        .then(|| backend_ids.first().map(|value| (*value).to_string()))
        .flatten();
    let admitted_cases = valid.iter().filter(|case| case.response_admitted).count();
    let converged_cases = valid.iter().filter(|case| case.converged).count();
    let distinct_states = valid
        .iter()
        .map(|case| case.state_fingerprint)
        .collect::<BTreeSet<_>>()
        .len();
    let distinct_structures = valid
        .iter()
        .map(|case| case.structure_fingerprint)
        .collect::<BTreeSet<_>>()
        .len();
    let warm_started_cases = valid.iter().filter(|case| case.warm_start_used).count();
    let symbolic_reuse_cases = valid
        .iter()
        .filter(|case| case.symbolic_pattern_reused)
        .count();
    let maximum_observed_equality_residual = valid
        .iter()
        .map(|case| case.maximum_equality_residual)
        .fold(0.0, f64::max);
    let maximum_observed_bound_violation = valid
        .iter()
        .map(|case| case.maximum_bound_violation)
        .fold(0.0, f64::max);
    let maximum_observed_elapsed_micros = valid
        .iter()
        .map(|case| case.elapsed_micros)
        .max()
        .unwrap_or(0);

    let mut failures = Vec::new();
    if criteria.minimum_cases == 0
        || criteria.minimum_distinct_states == 0
        || criteria.minimum_distinct_structures == 0
        || !criteria.maximum_equality_residual.is_finite()
        || criteria.maximum_equality_residual < 0.0
        || !criteria.maximum_bound_violation.is_finite()
        || criteria.maximum_bound_violation < 0.0
        || criteria.maximum_elapsed_micros == 0
    {
        failures.push("external sparse-QP criteria are invalid".to_string());
    }
    if backend_id.is_none() {
        failures.push("qualification cases do not identify exactly one backend".to_string());
    }
    if valid.len() < criteria.minimum_cases {
        failures.push(format!(
            "only {} valid cases were supplied; {} required",
            valid.len(),
            criteria.minimum_cases
        ));
    }
    if admitted_cases != valid.len() || converged_cases != valid.len() {
        failures.push(format!(
            "{} cases were rejected or failed convergence",
            valid
                .len()
                .saturating_sub(admitted_cases.min(converged_cases))
        ));
    }
    if distinct_states < criteria.minimum_distinct_states {
        failures.push(format!(
            "only {distinct_states} distinct states were covered; {} required",
            criteria.minimum_distinct_states
        ));
    }
    if distinct_structures < criteria.minimum_distinct_structures {
        failures.push(format!(
            "only {distinct_structures} QP structures were covered; {} required",
            criteria.minimum_distinct_structures
        ));
    }
    if warm_started_cases < criteria.minimum_warm_started_cases {
        failures.push(format!(
            "only {warm_started_cases} warm-started cases were admitted; {} required",
            criteria.minimum_warm_started_cases
        ));
    }
    if symbolic_reuse_cases < criteria.minimum_symbolic_reuse_cases {
        failures.push(format!(
            "only {symbolic_reuse_cases} symbolic-reuse cases were admitted; {} required",
            criteria.minimum_symbolic_reuse_cases
        ));
    }
    if maximum_observed_equality_residual > criteria.maximum_equality_residual {
        failures.push(format!(
            "maximum equality residual {:.6e} exceeds {:.6e}",
            maximum_observed_equality_residual, criteria.maximum_equality_residual
        ));
    }
    if maximum_observed_bound_violation > criteria.maximum_bound_violation {
        failures.push(format!(
            "maximum bound violation {:.6e} exceeds {:.6e}",
            maximum_observed_bound_violation, criteria.maximum_bound_violation
        ));
    }
    if maximum_observed_elapsed_micros > criteria.maximum_elapsed_micros {
        failures.push(format!(
            "maximum elapsed time {maximum_observed_elapsed_micros}us exceeds {}us",
            criteria.maximum_elapsed_micros
        ));
    }

    ExternalSparseQpQualificationCertificate {
        backend_id,
        valid_cases: valid.len(),
        admitted_cases,
        converged_cases,
        distinct_states,
        distinct_structures,
        warm_started_cases,
        symbolic_reuse_cases,
        maximum_observed_equality_residual,
        maximum_observed_bound_violation,
        maximum_observed_elapsed_micros,
        passed: failures.is_empty(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_backend_identity_fails() {
        let cases = ["one", "two"]
            .into_iter()
            .enumerate()
            .map(|(index, backend)| ExternalSparseQpQualificationCase {
                case_id: format!("case-{index}"),
                backend_id: backend.to_string(),
                structure_fingerprint: index as u64 + 1,
                state_fingerprint: index as u64 + 10,
                response_admitted: true,
                converged: true,
                warm_start_requested: true,
                warm_start_used: true,
                symbolic_pattern_reused: true,
                maximum_equality_residual: 0.0,
                maximum_bound_violation: 0.0,
                elapsed_micros: 1,
            })
            .collect::<Vec<_>>();
        let certificate = certify_external_sparse_qp_backend(
            &cases,
            ExternalSparseQpQualificationCriteria {
                minimum_cases: 2,
                minimum_distinct_states: 2,
                minimum_distinct_structures: 2,
                minimum_warm_started_cases: 2,
                minimum_symbolic_reuse_cases: 2,
                maximum_equality_residual: 1.0e-6,
                maximum_bound_violation: 1.0e-6,
                maximum_elapsed_micros: 10,
            },
        );
        assert!(!certificate.passed);
    }
}
