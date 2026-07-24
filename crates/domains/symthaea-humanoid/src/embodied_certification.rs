// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned control-substrate certification evidence.
//!
//! This is deterministic simulation/bench evidence, not a physical safety
//! certification. It is designed to make fallback, deadline, uncertainty, and
//! dynamics-fidelity dependence visible in release artifacts.

use serde::{Deserialize, Serialize};

use crate::full_dynamics::DynamicsFidelity;
use crate::hierarchical::HierarchicalControlReport;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedControlCase {
    pub scenario_id: String,
    pub dynamics_fidelity: DynamicsFidelity,
    pub active_contacts: usize,
    pub terrain_height_std_m: f64,
    pub terrain_evidence_age_s: f64,
    pub report: HierarchicalControlReport,
    pub fell: bool,
    pub recovered: bool,
}

impl EmbodiedControlCase {
    pub fn validate(&self) -> bool {
        !self.scenario_id.trim().is_empty()
            && self.terrain_height_std_m.is_finite()
            && self.terrain_height_std_m >= 0.0
            && self.terrain_evidence_age_s.is_finite()
            && self.terrain_evidence_age_s >= 0.0
            && self.active_contacts <= 8
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmbodiedCertificationCriteria {
    pub maximum_fallback_rate: f64,
    pub maximum_budget_miss_rate: f64,
    pub maximum_fall_rate: f64,
    pub minimum_recovery_rate: f64,
    pub maximum_terrain_height_std_m: f64,
    pub maximum_terrain_evidence_age_s: f64,
    pub require_solver_derived_case: bool,
    pub require_floating_base_case: bool,
    pub require_upper_body_contact_case: bool,
}

impl Default for EmbodiedCertificationCriteria {
    fn default() -> Self {
        Self {
            maximum_fallback_rate: 0.05,
            maximum_budget_miss_rate: 0.0,
            maximum_fall_rate: 0.05,
            minimum_recovery_rate: 0.95,
            maximum_terrain_height_std_m: 0.08,
            maximum_terrain_evidence_age_s: 0.20,
            require_solver_derived_case: true,
            require_floating_base_case: true,
            require_upper_body_contact_case: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedControlCertificate {
    pub schema_version: u32,
    pub scenario_fingerprint: u64,
    pub total_cases: usize,
    pub fallback_rate: f64,
    pub budget_miss_rate: f64,
    pub fall_rate: f64,
    pub recovery_rate: f64,
    pub maximum_terrain_height_std_m: f64,
    pub maximum_terrain_evidence_age_s: f64,
    pub solver_derived_cases: usize,
    pub floating_base_cases: usize,
    pub upper_body_contact_cases: usize,
    pub accepted: bool,
    pub failures: Vec<String>,
}

pub fn certify_embodied_control(
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
) -> EmbodiedControlCertificate {
    let valid = cases
        .iter()
        .filter(|case| case.validate())
        .collect::<Vec<_>>();
    let total = valid.len();
    let denominator = total.max(1) as f64;
    let fallback_rate = valid
        .iter()
        .filter(|case| {
            case.report.contact_dynamics_fallback
                || case.report.inverse_dynamics_fallback
                || case.report.floating_base_dynamics_fallback
        })
        .count() as f64
        / denominator;
    let budget_miss_rate = valid
        .iter()
        .filter(|case| {
            case.report.contact_solver_budget_missed
                || case.report.floating_base_solver_budget_missed
        })
        .count() as f64
        / denominator;
    let fall_rate = valid.iter().filter(|case| case.fell).count() as f64 / denominator;
    let fallen = valid.iter().filter(|case| case.fell).count();
    let recovery_rate = if fallen == 0 {
        1.0
    } else {
        valid
            .iter()
            .filter(|case| case.fell && case.recovered)
            .count() as f64
            / fallen as f64
    };
    let maximum_terrain_height_std_m = valid
        .iter()
        .map(|case| case.terrain_height_std_m)
        .fold(0.0, f64::max);
    let maximum_terrain_evidence_age_s = valid
        .iter()
        .map(|case| case.terrain_evidence_age_s)
        .fold(0.0, f64::max);
    let solver_derived_cases = valid
        .iter()
        .filter(|case| case.dynamics_fidelity == DynamicsFidelity::SolverDerived)
        .count();
    let floating_base_cases = valid
        .iter()
        .filter(|case| {
            case.dynamics_fidelity == DynamicsFidelity::SolverDerived
                && case.report.floating_base_model_available
                && case.report.floating_base_dynamics_converged
        })
        .count();
    let upper_body_contact_cases = valid.iter().filter(|case| case.active_contacts > 2).count();
    let mut failures = Vec::new();
    if total != cases.len() || total == 0 {
        failures.push("invalid or empty scenario corpus".to_string());
    }
    if fallback_rate > criteria.maximum_fallback_rate {
        failures.push(format!("fallback rate {fallback_rate:.4} exceeds limit"));
    }
    if budget_miss_rate > criteria.maximum_budget_miss_rate {
        failures.push(format!(
            "budget miss rate {budget_miss_rate:.4} exceeds limit"
        ));
    }
    if fall_rate > criteria.maximum_fall_rate {
        failures.push(format!("fall rate {fall_rate:.4} exceeds limit"));
    }
    if recovery_rate < criteria.minimum_recovery_rate {
        failures.push(format!("recovery rate {recovery_rate:.4} below minimum"));
    }
    if maximum_terrain_height_std_m > criteria.maximum_terrain_height_std_m {
        failures.push("terrain height uncertainty exceeds limit".to_string());
    }
    if maximum_terrain_evidence_age_s > criteria.maximum_terrain_evidence_age_s {
        failures.push("terrain evidence age exceeds limit".to_string());
    }
    if criteria.require_solver_derived_case && solver_derived_cases == 0 {
        failures.push("no solver-derived dynamics case present".to_string());
    }
    if criteria.require_floating_base_case && floating_base_cases == 0 {
        failures.push("no converged floating-base dynamics case present".to_string());
    }
    if criteria.require_upper_body_contact_case && upper_body_contact_cases == 0 {
        failures.push("no upper-body multi-contact case present".to_string());
    }
    EmbodiedControlCertificate {
        schema_version: 2,
        scenario_fingerprint: fingerprint_cases(&valid),
        total_cases: total,
        fallback_rate,
        budget_miss_rate,
        fall_rate,
        recovery_rate,
        maximum_terrain_height_std_m,
        maximum_terrain_evidence_age_s,
        solver_derived_cases,
        floating_base_cases,
        upper_body_contact_cases,
        accepted: failures.is_empty(),
        failures,
    }
}

fn fingerprint_cases(cases: &[&EmbodiedControlCase]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for case in cases {
        for byte in case.scenario_id.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= case.active_contacts as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        hash ^= case.fell as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        hash ^= case.recovered as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HierarchicalHumanoidController;
    use crate::morphology::HumanoidMorphology;
    use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask};

    #[test]
    fn incomplete_corpus_fails_closed() {
        let certificate = certify_embodied_control(&[], EmbodiedCertificationCriteria::default());
        assert!(!certificate.accepted);
    }

    #[test]
    fn reduced_only_corpus_cannot_claim_solver_derived_coverage() {
        let state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        let zero = HumanoidCommand::zero();
        let (_, report) = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21)
            .synthesize(HumanoidTask::Stand, &state, &zero, &zero, 1.0, 0.0);
        let case = EmbodiedControlCase {
            scenario_id: "reduced-only".to_string(),
            dynamics_fidelity: DynamicsFidelity::ReducedOrder,
            active_contacts: 2,
            terrain_height_std_m: 0.0,
            terrain_evidence_age_s: 0.0,
            report,
            fell: false,
            recovered: false,
        };
        let certificate =
            certify_embodied_control(&[case], EmbodiedCertificationCriteria::default());
        assert!(!certificate.accepted);
        assert!(
            certificate
                .failures
                .iter()
                .any(|failure| failure.contains("solver-derived"))
        );
    }
}
