// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification summaries for floating-base QP execution.

use serde::{Deserialize, Serialize};

use crate::floating_base_inverse_dynamics::FloatingBaseInverseDynamicsReport;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingBaseQpCase {
    pub case_id: String,
    pub state_fingerprint: u64,
    pub report: FloatingBaseInverseDynamicsReport,
}

impl FloatingBaseQpCase {
    pub fn validate(&self) -> bool {
        !self.case_id.trim().is_empty()
            && self.state_fingerprint != 0
            && self.report.solver_derived_model
            && self
                .report
                .solver_backend_id
                .as_ref()
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FloatingBaseQpCriteria {
    pub minimum_cases: usize,
    pub minimum_warm_started_cases: usize,
    pub maximum_fallback_fraction: f64,
    pub maximum_deadline_miss_fraction: f64,
    pub maximum_dynamics_residual: f64,
}

impl Default for FloatingBaseQpCriteria {
    fn default() -> Self {
        Self {
            minimum_cases: 100,
            minimum_warm_started_cases: 50,
            maximum_fallback_fraction: 0.01,
            maximum_deadline_miss_fraction: 0.0,
            maximum_dynamics_residual: 1.0e-4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingBaseQpCertificate {
    pub valid_cases: usize,
    pub converged_cases: usize,
    pub warm_started_cases: usize,
    pub symbolic_reuse_cases: usize,
    pub fallback_cases: usize,
    pub deadline_miss_cases: usize,
    pub distinct_state_fingerprints: usize,
    pub maximum_observed_dynamics_residual: f64,
    pub backend_ids: Vec<String>,
    pub passed: bool,
    pub failures: Vec<String>,
}

pub fn certify_floating_base_qp(
    cases: &[FloatingBaseQpCase],
    criteria: FloatingBaseQpCriteria,
) -> FloatingBaseQpCertificate {
    let valid = cases
        .iter()
        .filter(|case| case.validate())
        .collect::<Vec<_>>();
    let converged_cases = valid.iter().filter(|case| case.report.converged).count();
    let warm_started_cases = valid
        .iter()
        .filter(|case| case.report.warm_start_used)
        .count();
    let symbolic_reuse_cases = valid
        .iter()
        .filter(|case| case.report.symbolic_pattern_reused)
        .count();
    let fallback_cases = valid
        .iter()
        .filter(|case| case.report.used_fallback)
        .count();
    let deadline_miss_cases = valid
        .iter()
        .filter(|case| case.report.budget.deadline_missed)
        .count();
    let distinct_state_fingerprints = valid
        .iter()
        .map(|case| case.state_fingerprint)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let maximum_observed_dynamics_residual = valid
        .iter()
        .map(|case| case.report.maximum_dynamics_residual)
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    let backend_ids = valid
        .iter()
        .filter_map(|case| case.report.solver_backend_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut failures = Vec::new();
    if criteria.minimum_cases == 0
        || !criteria.maximum_fallback_fraction.is_finite()
        || !criteria.maximum_deadline_miss_fraction.is_finite()
        || !criteria.maximum_dynamics_residual.is_finite()
        || !(0.0..=1.0).contains(&criteria.maximum_fallback_fraction)
        || !(0.0..=1.0).contains(&criteria.maximum_deadline_miss_fraction)
        || criteria.maximum_dynamics_residual < 0.0
    {
        failures.push("floating-base QP criteria are invalid".to_string());
    }
    if valid.len() < criteria.minimum_cases {
        failures.push(format!(
            "only {} valid QP cases were supplied; {} required",
            valid.len(),
            criteria.minimum_cases
        ));
    }
    if warm_started_cases < criteria.minimum_warm_started_cases {
        failures.push(format!(
            "only {warm_started_cases} warm-started cases were observed; {} required",
            criteria.minimum_warm_started_cases
        ));
    }
    let denominator = valid.len().max(1) as f64;
    let fallback_fraction = fallback_cases as f64 / denominator;
    if fallback_fraction > criteria.maximum_fallback_fraction {
        failures.push(format!(
            "fallback fraction {fallback_fraction:.6} exceeds {:.6}",
            criteria.maximum_fallback_fraction
        ));
    }
    let deadline_fraction = deadline_miss_cases as f64 / denominator;
    if deadline_fraction > criteria.maximum_deadline_miss_fraction {
        failures.push(format!(
            "deadline-miss fraction {deadline_fraction:.6} exceeds {:.6}",
            criteria.maximum_deadline_miss_fraction
        ));
    }
    if maximum_observed_dynamics_residual > criteria.maximum_dynamics_residual {
        failures.push(format!(
            "maximum dynamics residual {:.6e} exceeds {:.6e}",
            maximum_observed_dynamics_residual, criteria.maximum_dynamics_residual
        ));
    }
    if distinct_state_fingerprints != valid.len() {
        failures.push("QP certification contains duplicate state fingerprints".to_string());
    }

    FloatingBaseQpCertificate {
        valid_cases: valid.len(),
        converged_cases,
        warm_started_cases,
        symbolic_reuse_cases,
        fallback_cases,
        deadline_miss_cases,
        distinct_state_fingerprints,
        maximum_observed_dynamics_residual,
        backend_ids,
        passed: failures.is_empty(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_budget::SolverBudgetEvidence;

    fn report() -> FloatingBaseInverseDynamicsReport {
        FloatingBaseInverseDynamicsReport {
            solver_derived_model: true,
            converged: true,
            used_fallback: false,
            active_contacts: 2,
            active_set_iterations: 3,
            warm_start_used: true,
            warm_start_active_bounds: 2,
            qp_structure_fingerprint: 7,
            solver_backend_id: Some("test-backend".to_string()),
            symbolic_pattern_reused: true,
            maximum_dynamics_residual: 1.0e-6,
            maximum_contact_acceleration_residual: 1.0e-6,
            maximum_friction_utilization: 0.5,
            objective: 1.0,
            budget: SolverBudgetEvidence {
                admitted: true,
                elapsed_micros: 100,
                deadline_missed: false,
                estimated_operations: 1000,
                variables: 10,
                constraints: 5,
            },
            model_id: Some("test-model".to_string()),
        }
    }

    #[test]
    fn well_formed_small_suite_passes_custom_criteria() {
        let cases = (1..=2)
            .map(|id| FloatingBaseQpCase {
                case_id: format!("case-{id}"),
                state_fingerprint: id,
                report: report(),
            })
            .collect::<Vec<_>>();
        let certificate = certify_floating_base_qp(
            &cases,
            FloatingBaseQpCriteria {
                minimum_cases: 2,
                minimum_warm_started_cases: 2,
                maximum_fallback_fraction: 0.0,
                maximum_deadline_miss_fraction: 0.0,
                maximum_dynamics_residual: 1.0e-4,
            },
        );
        assert!(certificate.passed);
    }
}
