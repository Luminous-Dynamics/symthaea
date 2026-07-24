// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerical oracle comparisons for floating-base dynamics.
//!
//! A solver-derived snapshot is only useful when it agrees with an independent
//! implementation or a recorded simulator oracle.  This module compares every
//! load-bearing component with explicit absolute and relative tolerances and
//! refuses to pass models with mismatched morphology, coordinate ordering, or
//! contact-site identity.

use serde::{Deserialize, Serialize};

use crate::floating_base::FloatingBaseDynamicsSnapshot;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DynamicsOracleTolerances {
    pub absolute_mass_matrix: f64,
    pub relative_mass_matrix: f64,
    pub absolute_bias_force: f64,
    pub relative_bias_force: f64,
    pub absolute_centroidal_matrix: f64,
    pub relative_centroidal_matrix: f64,
    pub absolute_contact_jacobian: f64,
    pub relative_contact_jacobian: f64,
    pub absolute_torque_limit_nm: f64,
    pub relative_torque_limit: f64,
}

impl Default for DynamicsOracleTolerances {
    fn default() -> Self {
        Self {
            absolute_mass_matrix: 1.0e-8,
            relative_mass_matrix: 2.0e-5,
            absolute_bias_force: 1.0e-7,
            relative_bias_force: 5.0e-5,
            absolute_centroidal_matrix: 1.0e-8,
            relative_centroidal_matrix: 5.0e-5,
            absolute_contact_jacobian: 1.0e-8,
            relative_contact_jacobian: 5.0e-5,
            absolute_torque_limit_nm: 1.0e-8,
            relative_torque_limit: 1.0e-6,
        }
    }
}

impl DynamicsOracleTolerances {
    pub fn validate(&self) -> bool {
        [
            self.absolute_mass_matrix,
            self.relative_mass_matrix,
            self.absolute_bias_force,
            self.relative_bias_force,
            self.absolute_centroidal_matrix,
            self.relative_centroidal_matrix,
            self.absolute_contact_jacobian,
            self.relative_contact_jacobian,
            self.absolute_torque_limit_nm,
            self.relative_torque_limit,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsOracleReport {
    pub candidate_model_id: String,
    pub oracle_model_id: String,
    pub structure_matches: bool,
    pub maximum_mass_matrix_absolute_error: f64,
    pub maximum_mass_matrix_relative_error: f64,
    pub maximum_bias_force_absolute_error: f64,
    pub maximum_bias_force_relative_error: f64,
    pub maximum_centroidal_absolute_error: f64,
    pub maximum_centroidal_relative_error: f64,
    pub maximum_contact_jacobian_absolute_error: f64,
    pub maximum_contact_jacobian_relative_error: f64,
    pub maximum_torque_limit_absolute_error_nm: f64,
    pub maximum_torque_limit_relative_error: f64,
    pub compared_contact_sites: usize,
    pub passed: bool,
    pub failures: Vec<String>,
}

impl DynamicsOracleReport {
    pub fn validate(&self) -> bool {
        !self.candidate_model_id.trim().is_empty()
            && !self.oracle_model_id.trim().is_empty()
            && self.compared_contact_sites > 0
            && [
                self.maximum_mass_matrix_absolute_error,
                self.maximum_mass_matrix_relative_error,
                self.maximum_bias_force_absolute_error,
                self.maximum_bias_force_relative_error,
                self.maximum_centroidal_absolute_error,
                self.maximum_centroidal_relative_error,
                self.maximum_contact_jacobian_absolute_error,
                self.maximum_contact_jacobian_relative_error,
                self.maximum_torque_limit_absolute_error_nm,
                self.maximum_torque_limit_relative_error,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && self.passed == (self.structure_matches && self.failures.is_empty())
    }
}

pub fn compare_floating_base_dynamics(
    candidate: &FloatingBaseDynamicsSnapshot,
    oracle: &FloatingBaseDynamicsSnapshot,
    tolerance: DynamicsOracleTolerances,
) -> DynamicsOracleReport {
    let mut failures = Vec::new();
    if !tolerance.validate() {
        failures.push("oracle tolerances are invalid".to_string());
    }
    if !candidate.validate() {
        failures.push("candidate snapshot failed validation".to_string());
    }
    if !oracle.validate() {
        failures.push("oracle snapshot failed validation".to_string());
    }

    let structure_matches = candidate.morphology == oracle.morphology
        && candidate.generalized_velocity_count == oracle.generalized_velocity_count
        && candidate.actuator_velocity_indices == oracle.actuator_velocity_indices
        && candidate.contacts.len() == oracle.contacts.len()
        && candidate
            .contacts
            .iter()
            .zip(oracle.contacts.iter())
            .all(|(left, right)| left.site_id == right.site_id);
    if !structure_matches {
        failures.push(
            "morphology, generalized coordinates, or contact-site ordering differs".to_string(),
        );
    }

    let mass = error_summary(&candidate.mass_matrix, &oracle.mass_matrix);
    let bias = error_summary(&candidate.bias_force, &oracle.bias_force);
    let centroidal = error_summary(
        &candidate.centroidal_momentum_matrix,
        &oracle.centroidal_momentum_matrix,
    );
    let torque = error_summary(&candidate.torque_limits_nm, &oracle.torque_limits_nm);

    let mut contact_absolute = 0.0f64;
    let mut contact_relative = 0.0f64;
    let mut compared_contact_sites = 0usize;
    if structure_matches {
        for (candidate_contact, oracle_contact) in
            candidate.contacts.iter().zip(oracle.contacts.iter())
        {
            compared_contact_sites += 1;
            for row in 0..6 {
                let error = error_summary(&candidate_contact.rows[row], &oracle_contact.rows[row]);
                contact_absolute = contact_absolute.max(error.absolute);
                contact_relative = contact_relative.max(error.relative);
            }
        }
    }

    check_limit(
        "mass matrix",
        mass,
        tolerance.absolute_mass_matrix,
        tolerance.relative_mass_matrix,
        &mut failures,
    );
    check_limit(
        "bias force",
        bias,
        tolerance.absolute_bias_force,
        tolerance.relative_bias_force,
        &mut failures,
    );
    check_limit(
        "centroidal matrix",
        centroidal,
        tolerance.absolute_centroidal_matrix,
        tolerance.relative_centroidal_matrix,
        &mut failures,
    );
    check_limit(
        "contact Jacobian",
        ErrorSummary {
            absolute: contact_absolute,
            relative: contact_relative,
        },
        tolerance.absolute_contact_jacobian,
        tolerance.relative_contact_jacobian,
        &mut failures,
    );
    check_limit(
        "torque limit",
        torque,
        tolerance.absolute_torque_limit_nm,
        tolerance.relative_torque_limit,
        &mut failures,
    );

    let passed = structure_matches && failures.is_empty();
    DynamicsOracleReport {
        candidate_model_id: candidate.model_id.clone(),
        oracle_model_id: oracle.model_id.clone(),
        structure_matches,
        maximum_mass_matrix_absolute_error: mass.absolute,
        maximum_mass_matrix_relative_error: mass.relative,
        maximum_bias_force_absolute_error: bias.absolute,
        maximum_bias_force_relative_error: bias.relative,
        maximum_centroidal_absolute_error: centroidal.absolute,
        maximum_centroidal_relative_error: centroidal.relative,
        maximum_contact_jacobian_absolute_error: contact_absolute,
        maximum_contact_jacobian_relative_error: contact_relative,
        maximum_torque_limit_absolute_error_nm: torque.absolute,
        maximum_torque_limit_relative_error: torque.relative,
        compared_contact_sites,
        passed,
        failures,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsOracleCase {
    pub case_id: String,
    pub sampled_state_fingerprint: u64,
    pub report: DynamicsOracleReport,
}

impl DynamicsOracleCase {
    pub fn validate(&self) -> bool {
        !self.case_id.trim().is_empty()
            && self.sampled_state_fingerprint != 0
            && self.report.validate()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DynamicsOracleCertificationCriteria {
    pub minimum_cases: usize,
    pub minimum_distinct_state_fingerprints: usize,
    pub require_all_cases_pass: bool,
}

impl Default for DynamicsOracleCertificationCriteria {
    fn default() -> Self {
        Self {
            minimum_cases: 16,
            minimum_distinct_state_fingerprints: 16,
            require_all_cases_pass: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsOracleCertificate {
    pub total_cases: usize,
    pub valid_cases: usize,
    pub passing_cases: usize,
    pub distinct_state_fingerprints: usize,
    pub passed: bool,
    pub failures: Vec<String>,
}

pub fn certify_dynamics_oracle_cases(
    cases: &[DynamicsOracleCase],
    criteria: DynamicsOracleCertificationCriteria,
) -> DynamicsOracleCertificate {
    let valid: Vec<_> = cases.iter().filter(|case| case.validate()).collect();
    let passing_cases = valid.iter().filter(|case| case.report.passed).count();
    let distinct_state_fingerprints = valid
        .iter()
        .map(|case| case.sampled_state_fingerprint)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let mut failures = Vec::new();
    if criteria.minimum_cases == 0 || criteria.minimum_distinct_state_fingerprints == 0 {
        failures.push("certification criteria must require non-zero coverage".to_string());
    }
    if valid.len() < criteria.minimum_cases {
        failures.push(format!(
            "only {} valid cases were supplied; {} required",
            valid.len(),
            criteria.minimum_cases
        ));
    }
    if distinct_state_fingerprints < criteria.minimum_distinct_state_fingerprints {
        failures.push(format!(
            "only {distinct_state_fingerprints} distinct states were covered; {} required",
            criteria.minimum_distinct_state_fingerprints
        ));
    }
    if criteria.require_all_cases_pass && passing_cases != valid.len() {
        failures.push(format!(
            "{} of {} valid oracle cases failed",
            valid.len().saturating_sub(passing_cases),
            valid.len()
        ));
    }
    DynamicsOracleCertificate {
        total_cases: cases.len(),
        valid_cases: valid.len(),
        passing_cases,
        distinct_state_fingerprints,
        passed: failures.is_empty(),
        failures,
    }
}

#[derive(Debug, Clone, Copy)]
struct ErrorSummary {
    absolute: f64,
    relative: f64,
}

fn error_summary(candidate: &[f64], oracle: &[f64]) -> ErrorSummary {
    if candidate.len() != oracle.len() || candidate.is_empty() {
        return ErrorSummary {
            absolute: f64::INFINITY,
            relative: f64::INFINITY,
        };
    }
    candidate.iter().zip(oracle.iter()).fold(
        ErrorSummary {
            absolute: 0.0,
            relative: 0.0,
        },
        |mut summary, (candidate, oracle)| {
            let absolute = (candidate - oracle).abs();
            let scale = candidate.abs().max(oracle.abs()).max(1.0e-12);
            summary.absolute = summary.absolute.max(absolute);
            summary.relative = summary.relative.max(absolute / scale);
            summary
        },
    )
}

fn check_limit(
    name: &str,
    error: ErrorSummary,
    absolute_limit: f64,
    relative_limit: f64,
    failures: &mut Vec<String>,
) {
    if !error.absolute.is_finite()
        || !error.relative.is_finite()
        || (error.absolute > absolute_limit && error.relative > relative_limit)
    {
        failures.push(format!(
            "{name} error exceeds tolerance: abs={:.6e}, rel={:.6e}",
            error.absolute, error.relative
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::floating_base::{FLOATING_BASE_DOF, FloatingBaseDynamicsSnapshot};
    use crate::full_dynamics::{DynamicsProvenance, SpatialContactJacobian};
    use crate::morphology::HumanoidMorphology;

    fn snapshot(model_id: &str) -> FloatingBaseDynamicsSnapshot {
        let morphology = HumanoidMorphology::Dmc21;
        let nv = FLOATING_BASE_DOF + morphology.num_actuators();
        let mut mass_matrix = vec![0.0; nv * nv];
        for index in 0..nv {
            mass_matrix[index * nv + index] = 1.0 + index as f64 * 0.1;
        }
        FloatingBaseDynamicsSnapshot {
            morphology,
            sampled_at_s: 1.0,
            total_mass_kg: 70.0,
            gravity_world_mps2: [0.0, 0.0, -9.81],
            generalized_velocity_count: nv,
            mass_matrix,
            bias_force: vec![0.25; nv],
            actuator_velocity_indices: (FLOATING_BASE_DOF..nv).collect(),
            torque_limits_nm: vec![100.0; morphology.num_actuators()],
            centroidal_momentum_matrix: vec![0.5; 6 * nv],
            contacts: vec![SpatialContactJacobian {
                site_id: "r_foot_site".to_string(),
                rows: std::array::from_fn(|row| vec![row as f64 * 0.01; nv]),
                confidence: 1.0,
            }],
            provenance: DynamicsProvenance::mujoco_solver_with_morphology_limits(),
            model_id: model_id.to_string(),
        }
    }

    #[test]
    fn identical_snapshot_passes() {
        let candidate = snapshot("candidate");
        let oracle = snapshot("oracle");
        let report = compare_floating_base_dynamics(
            &candidate,
            &oracle,
            DynamicsOracleTolerances::default(),
        );
        assert!(report.passed);
        assert!(report.validate());
    }

    #[test]
    fn material_mass_error_fails() {
        let mut candidate = snapshot("candidate");
        let oracle = snapshot("oracle");
        candidate.mass_matrix[0] *= 1.1;
        let report = compare_floating_base_dynamics(
            &candidate,
            &oracle,
            DynamicsOracleTolerances::default(),
        );
        assert!(!report.passed);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("mass matrix"))
        );
    }

    #[test]
    fn certification_rejects_duplicate_state_coverage() {
        let candidate = snapshot("candidate");
        let oracle = snapshot("oracle");
        let report = compare_floating_base_dynamics(
            &candidate,
            &oracle,
            DynamicsOracleTolerances::default(),
        );
        let cases = vec![
            DynamicsOracleCase {
                case_id: "a".to_string(),
                sampled_state_fingerprint: 7,
                report: report.clone(),
            },
            DynamicsOracleCase {
                case_id: "b".to_string(),
                sampled_state_fingerprint: 7,
                report,
            },
        ];
        let certificate = certify_dynamics_oracle_cases(
            &cases,
            DynamicsOracleCertificationCriteria {
                minimum_cases: 2,
                minimum_distinct_state_fingerprints: 2,
                require_all_cases_pass: true,
            },
        );
        assert!(!certificate.passed);
    }
}
