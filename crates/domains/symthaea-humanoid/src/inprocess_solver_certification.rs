// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification evidence for persistent in-process sparse-QP engines.
//!
//! A backend must demonstrate repeated-state warm starts, multiple sparsity
//! structures, bounded iterations, residuals, and wall time before a release
//! policy can name it as a production control solver.

use std::collections::BTreeSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::csc_qp::CscEqualityQuadraticProgram;
use crate::equality_qp::DenseEqualityQuadraticProgram;
use crate::sparse_qp_backend::SparseQpBackend;

#[derive(Debug, Clone)]
pub struct InProcessSparseQpQualificationProblem {
    pub case_id: String,
    pub state_fingerprint: u64,
    pub problem: DenseEqualityQuadraticProgram,
    pub warm_start: Option<Vec<f64>>,
}

impl InProcessSparseQpQualificationProblem {
    pub fn validate(&self) -> bool {
        !self.case_id.trim().is_empty()
            && self.state_fingerprint != 0
            && self.problem.validate()
            && self
                .warm_start
                .as_ref()
                .map(|values| {
                    values.len() == self.problem.diagonal_hessian.len()
                        && values.iter().all(|value| value.is_finite())
                })
                .unwrap_or(true)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InProcessSparseQpQualificationCase {
    pub case_id: String,
    pub backend_id: String,
    pub state_fingerprint: u64,
    pub structure_fingerprint: u64,
    pub admitted: bool,
    pub converged: bool,
    pub warm_start_requested: bool,
    pub warm_start_used: bool,
    pub symbolic_pattern_reused: bool,
    pub iterations: usize,
    pub maximum_equality_residual: f64,
    pub maximum_bound_violation: f64,
    pub elapsed_micros: u64,
}

impl InProcessSparseQpQualificationCase {
    pub fn validate(&self) -> bool {
        !self.case_id.trim().is_empty()
            && !self.backend_id.trim().is_empty()
            && self.state_fingerprint != 0
            && self.structure_fingerprint != 0
            && self.iterations > 0
            && self.maximum_equality_residual.is_finite()
            && self.maximum_equality_residual >= 0.0
            && self.maximum_bound_violation.is_finite()
            && self.maximum_bound_violation >= 0.0
            && self.elapsed_micros > 0
            && (!self.warm_start_used || self.warm_start_requested || self.symbolic_pattern_reused)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InProcessSparseQpQualificationCriteria {
    pub minimum_cases: usize,
    pub minimum_distinct_states: usize,
    pub minimum_distinct_structures: usize,
    pub minimum_warm_started_cases: usize,
    pub minimum_symbolic_reuse_cases: usize,
    pub maximum_iterations: usize,
    pub maximum_equality_residual: f64,
    pub maximum_bound_violation: f64,
    pub maximum_elapsed_micros: u64,
}

impl Default for InProcessSparseQpQualificationCriteria {
    fn default() -> Self {
        Self {
            minimum_cases: 512,
            minimum_distinct_states: 256,
            minimum_distinct_structures: 3,
            minimum_warm_started_cases: 256,
            minimum_symbolic_reuse_cases: 256,
            maximum_iterations: 128,
            maximum_equality_residual: 1.0e-6,
            maximum_bound_violation: 1.0e-6,
            maximum_elapsed_micros: 1_000,
        }
    }
}

impl InProcessSparseQpQualificationCriteria {
    pub fn validate(&self) -> bool {
        self.minimum_cases > 0
            && self.minimum_distinct_states > 0
            && self.minimum_distinct_structures > 0
            && self.maximum_iterations > 0
            && self.maximum_equality_residual.is_finite()
            && self.maximum_equality_residual >= 0.0
            && self.maximum_bound_violation.is_finite()
            && self.maximum_bound_violation >= 0.0
            && self.maximum_elapsed_micros > 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InProcessSparseQpQualificationCertificate {
    pub schema_version: u32,
    pub backend_id: Option<String>,
    pub total_cases: usize,
    pub valid_cases: usize,
    pub admitted_cases: usize,
    pub converged_cases: usize,
    pub distinct_states: usize,
    pub distinct_structures: usize,
    pub warm_started_cases: usize,
    pub symbolic_reuse_cases: usize,
    pub maximum_observed_iterations: usize,
    pub maximum_observed_equality_residual: f64,
    pub maximum_observed_bound_violation: f64,
    pub maximum_observed_elapsed_micros: u64,
    pub passed: bool,
    pub failures: Vec<String>,
    pub cases: Vec<InProcessSparseQpQualificationCase>,
}

pub fn qualify_inprocess_sparse_qp_backend(
    backend: &dyn SparseQpBackend,
    problems: &[InProcessSparseQpQualificationProblem],
    criteria: InProcessSparseQpQualificationCriteria,
) -> InProcessSparseQpQualificationCertificate {
    let mut cases = Vec::new();
    for problem in problems.iter().filter(|problem| problem.validate()) {
        let Some(sparse) = CscEqualityQuadraticProgram::from_dense(&problem.problem, 1.0e-14)
        else {
            continue;
        };
        let start = Instant::now();
        let solution = backend.solve(
            &problem.problem,
            sparse.structure_fingerprint,
            problem.warm_start.as_deref(),
        );
        let elapsed_micros = start.elapsed().as_micros().max(1) as u64;
        let case = match solution {
            Some(solution) => InProcessSparseQpQualificationCase {
                case_id: problem.case_id.clone(),
                backend_id: solution.backend_id,
                state_fingerprint: problem.state_fingerprint,
                structure_fingerprint: sparse.structure_fingerprint,
                admitted: true,
                converged: solution.solution.converged,
                warm_start_requested: problem.warm_start.is_some(),
                warm_start_used: solution.solution.warm_start_used,
                symbolic_pattern_reused: solution.symbolic_pattern_reused,
                iterations: solution.solution.active_set_iterations,
                maximum_equality_residual: solution.solution.maximum_equality_residual,
                maximum_bound_violation: solution.solution.maximum_bound_violation,
                elapsed_micros,
            },
            None => InProcessSparseQpQualificationCase {
                case_id: problem.case_id.clone(),
                backend_id: backend.backend_id().to_string(),
                state_fingerprint: problem.state_fingerprint,
                structure_fingerprint: sparse.structure_fingerprint,
                admitted: false,
                converged: false,
                warm_start_requested: problem.warm_start.is_some(),
                warm_start_used: false,
                symbolic_pattern_reused: false,
                iterations: 1,
                maximum_equality_residual: f64::MAX,
                maximum_bound_violation: f64::MAX,
                elapsed_micros,
            },
        };
        cases.push(case);
    }
    certify_inprocess_sparse_qp_cases(problems.len(), cases, criteria)
}

pub fn certify_inprocess_sparse_qp_cases(
    total_cases: usize,
    cases: Vec<InProcessSparseQpQualificationCase>,
    criteria: InProcessSparseQpQualificationCriteria,
) -> InProcessSparseQpQualificationCertificate {
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
    let admitted_cases = valid.iter().filter(|case| case.admitted).count();
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
    let maximum_observed_iterations = valid.iter().map(|case| case.iterations).max().unwrap_or(0);
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
    if !criteria.validate() {
        failures.push("in-process sparse-QP qualification criteria are invalid".to_string());
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
            "only {distinct_structures} sparse structures were covered; {} required",
            criteria.minimum_distinct_structures
        ));
    }
    if warm_started_cases < criteria.minimum_warm_started_cases {
        failures.push(format!(
            "only {warm_started_cases} warm starts were admitted; {} required",
            criteria.minimum_warm_started_cases
        ));
    }
    if symbolic_reuse_cases < criteria.minimum_symbolic_reuse_cases {
        failures.push(format!(
            "only {symbolic_reuse_cases} symbolic reuses were admitted; {} required",
            criteria.minimum_symbolic_reuse_cases
        ));
    }
    if maximum_observed_iterations > criteria.maximum_iterations {
        failures.push(format!(
            "maximum solver iterations {maximum_observed_iterations} exceed {}",
            criteria.maximum_iterations
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

    InProcessSparseQpQualificationCertificate {
        schema_version: 1,
        backend_id,
        total_cases,
        valid_cases: valid.len(),
        admitted_cases,
        converged_cases,
        distinct_states,
        distinct_structures,
        warm_started_cases,
        symbolic_reuse_cases,
        maximum_observed_iterations,
        maximum_observed_equality_residual,
        maximum_observed_bound_violation,
        maximum_observed_elapsed_micros,
        passed: failures.is_empty(),
        failures,
        cases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inprocess_sparse_qp::{InProcessSparseQpBackend, ReferenceProjectedAdmmEngine};

    fn problem(case: usize, warm: bool) -> InProcessSparseQpQualificationProblem {
        InProcessSparseQpQualificationProblem {
            case_id: format!("case-{case}"),
            state_fingerprint: case as u64 + 1,
            problem: DenseEqualityQuadraticProgram {
                diagonal_hessian: vec![1.0, 1.0],
                linear_term: vec![0.0, 0.0],
                lower_bounds: vec![-1.0, -1.0],
                upper_bounds: vec![1.0, 1.0],
                equality_matrix: vec![vec![1.0, 1.0]],
                equality_target: vec![0.0],
            },
            warm_start: warm.then(|| vec![0.0, 0.0]),
        }
    }

    #[test]
    fn small_reference_qualification_can_pass_explicit_criteria() {
        let backend = InProcessSparseQpBackend::new(ReferenceProjectedAdmmEngine::default())
            .unwrap()
            .with_validation_tolerance(1.0e-5);
        let cases = vec![problem(0, false), problem(1, true)];
        let certificate = qualify_inprocess_sparse_qp_backend(
            &backend,
            &cases,
            InProcessSparseQpQualificationCriteria {
                minimum_cases: 2,
                minimum_distinct_states: 2,
                minimum_distinct_structures: 1,
                minimum_warm_started_cases: 1,
                minimum_symbolic_reuse_cases: 1,
                maximum_iterations: 256,
                maximum_equality_residual: 1.0e-5,
                maximum_bound_violation: 1.0e-5,
                maximum_elapsed_micros: u64::MAX,
            },
        );
        assert!(certificate.passed, "{:?}", certificate.failures);
    }
}

/// Production-oriented criteria for the feature-gated OSQP backend. These
/// thresholds remain explicit evidence policy, not claims about a particular
/// machine until a qualification certificate records actual measurements.
pub fn osqp_production_qualification_criteria() -> InProcessSparseQpQualificationCriteria {
    InProcessSparseQpQualificationCriteria {
        minimum_cases: 2_048,
        minimum_distinct_states: 1_024,
        minimum_distinct_structures: 4,
        minimum_warm_started_cases: 1_024,
        minimum_symbolic_reuse_cases: 1_024,
        maximum_iterations: 4_000,
        maximum_equality_residual: 1.0e-6,
        maximum_bound_violation: 1.0e-7,
        maximum_elapsed_micros: 1_500,
    }
}

pub fn certificate_identifies_backend(
    certificate: &InProcessSparseQpQualificationCertificate,
    expected_backend_id: &str,
) -> bool {
    certificate.passed
        && !expected_backend_id.trim().is_empty()
        && certificate.backend_id.as_deref() == Some(expected_backend_id)
        && certificate.valid_cases == certificate.admitted_cases
        && certificate.admitted_cases == certificate.converged_cases
}
