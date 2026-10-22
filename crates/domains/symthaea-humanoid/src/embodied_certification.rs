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

/// One predeclared scenario execution, including its exact random seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodiedScenarioSpec {
    pub scenario_id: String,
    pub random_seed: u64,
}

/// Frozen, exact scenario inventory for one candidate/model build.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbodiedScenarioManifest {
    pub schema_version: u32,
    pub manifest_id: String,
    pub candidate_build_id: String,
    pub model_artifact_sha256: String,
    pub controller_artifact_sha256: String,
    pub simulator_id: String,
    pub simulator_version: String,
    pub test_harness_sha256: String,
    pub scenario_config_sha256: String,
    #[serde(default)]
    pub criteria: EmbodiedCertificationCriteria,
    /// Every (scenario ID, seed) pair must appear exactly once in the evaluated corpus.
    pub scenarios: Vec<EmbodiedScenarioSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedControlCase {
    pub scenario_id: String,
    /// The exact seed used to generate this trial, included in evidence identity.
    pub random_seed: u64,
    /// True when this case is an intentionally designed recovery challenge.
    pub recovery_challenge: bool,
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
            && (!self.recovered || self.recovery_challenge)
            && report_numeric_fields_are_finite(&self.report)
    }
}

fn report_numeric_fields_are_finite(report: &HierarchicalControlReport) -> bool {
    let scalar_values = [
        report.residual_authority as f64,
        report.balance_effort as f64,
        report.baseline_weight as f64,
        report.capture_margin_m,
        report.recovery_effort as f64,
        report.terrain_confidence,
        report.terrain_clearance_m,
        report.whole_body_joint_utilization,
        report.whole_body_objective_residual,
        report.inverse_dynamics_max_violation,
        report.terrain_mpc_cost,
        report.contact_dynamics_residual_nm,
        report.contact_acceleration_residual,
        report.contact_friction_utilization,
        report.terrain_max_height_std_m,
        report.terrain_max_evidence_age_s,
        report.contact_trust as f64,
        report.centroidal_authority,
        report.centroidal_correction_norm,
        report.angular_momentum_norm,
        report.linear_momentum_norm,
        report.floating_base_dynamics_residual,
    ];
    let nonnegative_values = [
        report.residual_authority as f64,
        report.balance_effort as f64,
        report.recovery_effort as f64,
        report.terrain_clearance_m,
        report.whole_body_joint_utilization,
        report.whole_body_objective_residual,
        report.inverse_dynamics_max_violation,
        report.contact_dynamics_residual_nm,
        report.contact_acceleration_residual,
        report.contact_friction_utilization,
        report.terrain_max_height_std_m,
        report.terrain_max_evidence_age_s,
        report.centroidal_authority,
        report.centroidal_correction_norm,
        report.angular_momentum_norm,
        report.linear_momentum_norm,
        report.floating_base_dynamics_residual,
    ];
    let bounded_values = [
        report.residual_authority as f64,
        report.baseline_weight as f64,
        report.terrain_confidence,
        report.contact_trust as f64,
    ];
    scalar_values.iter().all(|value| value.is_finite())
        && nonnegative_values.iter().all(|value| *value >= 0.0)
        && bounded_values
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
        && report.terrain_max_height_std_m >= 0.0
        && report.terrain_max_evidence_age_s >= 0.0
        && report.planned_step.as_ref().map_or(true, |step| {
            step.target_world_m.iter().all(|value| value.is_finite())
                && step
                    .predicted_capture_point_world_m
                    .iter()
                    .all(|value| value.is_finite())
                && [
                    step.duration_s,
                    step.clearance_m,
                    step.reach_utilization,
                    step.confidence,
                ]
                .iter()
                .all(|value| value.is_finite())
                && step.duration_s > 0.0
                && step.clearance_m >= 0.0
                && step.reach_utilization >= 0.0
                && (0.0..=1.0).contains(&step.confidence)
        })
        && report.planned_swing.as_ref().map_or(true, |swing| {
            swing.start_world_m.iter().all(|value| value.is_finite())
                && swing.apex_world_m.iter().all(|value| value.is_finite())
                && swing.target_world_m.iter().all(|value| value.is_finite())
                && swing.target_normal_world.iter().all(|value| value.is_finite())
                && [
                    swing.duration_s,
                    swing.touchdown_velocity_mps,
                    swing.maximum_obstacle_height_m,
                    swing.clearance_m,
                    swing.terrain_confidence,
                ]
                .iter()
                .all(|value| value.is_finite())
                && swing.duration_s > 0.0
                && swing.maximum_obstacle_height_m >= 0.0
                && swing.clearance_m >= 0.0
                && (0.0..=1.0).contains(&swing.terrain_confidence)
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedCertificationCriteria {
    pub maximum_fallback_rate: f64,
    pub maximum_budget_miss_rate: f64,
    pub maximum_fall_rate: f64,
    pub minimum_recovery_rate: f64,
    pub maximum_terrain_height_std_m: f64,
    pub maximum_terrain_evidence_age_s: f64,
    pub maximum_inverse_dynamics_violation: f64,
    pub maximum_contact_dynamics_residual_nm: f64,
    pub maximum_contact_acceleration_residual: f64,
    pub maximum_contact_friction_utilization: f64,
    pub maximum_floating_base_dynamics_residual: f64,
    pub maximum_joint_utilization: f64,
    /// Minimum total cases required for a release-level statistical claim.
    pub minimum_case_count: usize,
    /// Minimum explicit recovery challenges used to measure the recovery rate.
    pub minimum_recovery_trials: usize,
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
            maximum_inverse_dynamics_violation: 1.0e-7,
            maximum_contact_dynamics_residual_nm: 1.0e-4,
            maximum_contact_acceleration_residual: 2.0e-5,
            maximum_contact_friction_utilization: 1.0 + 1.0e-6,
            maximum_floating_base_dynamics_residual: 1.0e-4,
            maximum_joint_utilization: 1.0 + 1.0e-5,
            minimum_case_count: 59,
            minimum_recovery_trials: 59,
            require_solver_derived_case: true,
            require_floating_base_case: true,
            require_upper_body_contact_case: true,
        }
    }
}

impl EmbodiedCertificationCriteria {
    /// Reject non-finite or out-of-domain limits that could make comparisons
    /// vacuously pass (for example, `NaN` thresholds).
    pub fn is_valid(self) -> bool {
        // A caller may select stricter bounds, but cannot weaken this
        // qualification contract by supplying oversized thresholds or disabling
        // mandatory solver/dynamics coverage.
        [
            self.maximum_fallback_rate,
            self.maximum_budget_miss_rate,
            self.maximum_fall_rate,
            self.minimum_recovery_rate,
        ]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && self.maximum_fallback_rate <= 0.05
            && self.maximum_budget_miss_rate == 0.0
            && self.maximum_fall_rate <= 0.05
            && self.minimum_recovery_rate >= 0.95
            && self.maximum_terrain_height_std_m.is_finite()
            && (0.0..=0.08).contains(&self.maximum_terrain_height_std_m)
            && self.maximum_terrain_evidence_age_s.is_finite()
            && (0.0..=0.20).contains(&self.maximum_terrain_evidence_age_s)
            && self.maximum_inverse_dynamics_violation.is_finite()
            && (0.0..=1.0e-7).contains(&self.maximum_inverse_dynamics_violation)
            && self.maximum_contact_dynamics_residual_nm.is_finite()
            && (0.0..=1.0e-4).contains(&self.maximum_contact_dynamics_residual_nm)
            && self.maximum_contact_acceleration_residual.is_finite()
            && (0.0..=2.0e-5).contains(&self.maximum_contact_acceleration_residual)
            && self.maximum_contact_friction_utilization.is_finite()
            && (0.0..=1.0 + 1.0e-6).contains(&self.maximum_contact_friction_utilization)
            && self.maximum_floating_base_dynamics_residual.is_finite()
            && (0.0..=1.0e-4).contains(&self.maximum_floating_base_dynamics_residual)
            && self.maximum_joint_utilization.is_finite()
            && (0.0..=1.0 + 1.0e-5).contains(&self.maximum_joint_utilization)
            && self.minimum_case_count >= 59
            && self.minimum_recovery_trials >= 59
            && self.require_solver_derived_case
            && self.require_floating_base_case
            && self.require_upper_body_contact_case
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedControlCertificate {
    pub schema_version: u32,
    pub scenario_fingerprint: u64,
    /// Exact threshold policy used to decide acceptance.
    #[serde(default)]
    pub criteria: EmbodiedCertificationCriteria,
    /// Immutable candidate/model identity inherited from the frozen manifest.
    #[serde(default)]
    pub candidate_build_id: String,
    #[serde(default)]
    pub model_artifact_sha256: String,
    #[serde(default)]
    pub controller_artifact_sha256: String,
    #[serde(default)]
    pub simulator_id: String,
    #[serde(default)]
    pub simulator_version: String,
    #[serde(default)]
    pub test_harness_sha256: String,
    #[serde(default)]
    pub scenario_config_sha256: String,
    #[serde(default)]
    pub scenario_manifest_id: String,
    /// BLAKE3 digest of the frozen manifest and its exact case-ID inventory.
    #[serde(default)]
    pub scenario_manifest_digest: String,
    /// True only when every input case ID exactly matches the manifest inventory.
    #[serde(default)]
    pub manifest_verified: bool,
    /// BLAKE3 digest of the complete canonical serialized valid-case corpus.
    /// Empty only when serialization fails; that condition always rejects the certificate.
    #[serde(default)]
    pub corpus_digest: String,
    pub total_cases: usize,
    pub fallback_rate: f64,
    pub budget_miss_rate: f64,
    pub fall_rate: f64,
    pub recovery_rate: f64,
    /// One-sided 95% Wilson upper bound on the observed fall probability.
    pub fall_rate_upper_95: f64,
    /// One-sided 95% Wilson lower bound on recovery success probability.
    pub recovery_rate_lower_95: f64,
    pub fall_count: usize,
    pub recovery_trials: usize,
    pub recovery_successes: usize,
    pub maximum_terrain_height_std_m: f64,
    pub maximum_terrain_evidence_age_s: f64,
    pub maximum_inverse_dynamics_violation: f64,
    pub maximum_contact_dynamics_residual_nm: f64,
    pub maximum_contact_acceleration_residual: f64,
    pub maximum_contact_friction_utilization: f64,
    pub maximum_floating_base_dynamics_residual: f64,
    pub maximum_joint_utilization: f64,
    pub whole_body_infeasible_cases: usize,
    pub solver_derived_cases: usize,
    pub floating_base_cases: usize,
    pub upper_body_contact_cases: usize,
    pub accepted: bool,
    pub failures: Vec<String>,
}

/// Legacy entry point. It deliberately fails closed because a bare case slice
/// cannot prove that the submitted corpus matches a frozen expected inventory.
pub fn certify_embodied_control(
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
) -> EmbodiedControlCertificate {
    certify_embodied_control_internal(cases, criteria, None)
}

/// Certify cases only when their exact IDs match a frozen, candidate-bound manifest.
pub fn certify_embodied_control_against_manifest(
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
    manifest: &EmbodiedScenarioManifest,
) -> EmbodiedControlCertificate {
    certify_embodied_control_internal(cases, criteria, Some(manifest))
}

fn certify_embodied_control_internal(
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
    manifest: Option<&EmbodiedScenarioManifest>,
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
    let fall_count = valid.iter().filter(|case| case.fell).count();
    let fall_rate = fall_count as f64 / denominator;
    let recovery_trials = valid.iter().filter(|case| case.recovery_challenge).count();
    let recovery_successes = valid
        .iter()
        .filter(|case| case.recovery_challenge && case.recovered)
        .count();
    let recovery_rate = if recovery_trials == 0 {
        0.0
    } else {
        recovery_successes as f64 / recovery_trials as f64
    };
    let fall_rate_upper_95 = wilson_upper_95(fall_count, total);
    let recovery_rate_lower_95 = wilson_lower_95(recovery_successes, recovery_trials);
    let maximum_terrain_height_std_m = valid
        .iter()
        .map(|case| case.terrain_height_std_m.max(case.report.terrain_max_height_std_m))
        .fold(0.0, f64::max);
    let maximum_terrain_evidence_age_s = valid
        .iter()
        .map(|case| case.terrain_evidence_age_s.max(case.report.terrain_max_evidence_age_s))
        .fold(0.0, f64::max);
    let maximum_inverse_dynamics_violation = valid
        .iter()
        .map(|case| case.report.inverse_dynamics_max_violation)
        .fold(0.0, f64::max);
    let maximum_contact_dynamics_residual_nm = valid
        .iter()
        .map(|case| case.report.contact_dynamics_residual_nm)
        .fold(0.0, f64::max);
    let maximum_contact_acceleration_residual = valid
        .iter()
        .map(|case| case.report.contact_acceleration_residual)
        .fold(0.0, f64::max);
    let maximum_contact_friction_utilization = valid
        .iter()
        .map(|case| case.report.contact_friction_utilization)
        .fold(0.0, f64::max);
    let maximum_floating_base_dynamics_residual = valid
        .iter()
        .filter(|case| case.report.floating_base_model_available)
        .map(|case| case.report.floating_base_dynamics_residual)
        .fold(0.0, f64::max);
    let maximum_joint_utilization = valid
        .iter()
        .map(|case| case.report.whole_body_joint_utilization)
        .fold(0.0, f64::max);
    let whole_body_infeasible_cases = valid
        .iter()
        .filter(|case| !case.report.whole_body_feasible)
        .count();
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
    let corpus_digest = canonical_corpus_digest(&valid);
    let mut failures = Vec::new();
    if !criteria.is_valid() {
        failures.push("qualification criteria contain non-finite or out-of-domain thresholds".to_string());
    }
    if total != cases.len() || total == 0 {
        failures.push("invalid or empty scenario corpus".to_string());
    }
    if total < criteria.minimum_case_count {
        failures.push(format!(
            "scenario corpus has {total} cases; minimum is {}",
            criteria.minimum_case_count
        ));
    }
    if recovery_trials < criteria.minimum_recovery_trials {
        failures.push(format!(
            "recovery campaign has {recovery_trials} trials; minimum is {}",
            criteria.minimum_recovery_trials
        ));
    }
    if corpus_digest.is_none() {
        failures.push("complete evidence corpus could not be canonically serialized".to_string());
    }
    let manifest_validation = match manifest {
        Some(manifest) => validate_scenario_manifest(manifest, cases, criteria),
        None => Err("missing frozen scenario manifest".to_string()),
    };
    if let Err(reason) = &manifest_validation {
        failures.push(format!("scenario manifest rejected: {reason}"));
    }
    let manifest_verified =
        manifest_validation.is_ok() && total == cases.len() && total > 0;
    let scenario_manifest_digest = manifest_validation.unwrap_or_default();
    let (
        scenario_manifest_id,
        candidate_build_id,
        model_artifact_sha256,
        controller_artifact_sha256,
        simulator_id,
        simulator_version,
        test_harness_sha256,
        scenario_config_sha256,
    ) = match manifest {
        Some(manifest) => (
            manifest.manifest_id.clone(),
            manifest.candidate_build_id.clone(),
            manifest.model_artifact_sha256.clone(),
            manifest.controller_artifact_sha256.clone(),
            manifest.simulator_id.clone(),
            manifest.simulator_version.clone(),
            manifest.test_harness_sha256.clone(),
            manifest.scenario_config_sha256.clone(),
        ),
        None => (
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
    };
    if fallback_rate > criteria.maximum_fallback_rate {
        failures.push(format!("fallback rate {fallback_rate:.4} exceeds limit"));
    }
    if budget_miss_rate > criteria.maximum_budget_miss_rate {
        failures.push(format!(
            "budget miss rate {budget_miss_rate:.4} exceeds limit"
        ));
    }
    if fall_rate_upper_95 > criteria.maximum_fall_rate {
        failures.push(format!(
            "one-sided 95% fall-rate upper bound {fall_rate_upper_95:.4} exceeds limit"
        ));
    }
    if recovery_rate_lower_95 < criteria.minimum_recovery_rate {
        failures.push(format!(
            "one-sided 95% recovery-rate lower bound {recovery_rate_lower_95:.4} below minimum"
        ));
    }
    if maximum_terrain_height_std_m > criteria.maximum_terrain_height_std_m {
        failures.push("terrain height uncertainty exceeds limit".to_string());
    }
    if maximum_terrain_evidence_age_s > criteria.maximum_terrain_evidence_age_s {
        failures.push("terrain evidence age exceeds limit".to_string());
    }
    if maximum_inverse_dynamics_violation > criteria.maximum_inverse_dynamics_violation {
        failures.push("inverse-dynamics constraint violation exceeds limit".to_string());
    }
    if maximum_contact_dynamics_residual_nm > criteria.maximum_contact_dynamics_residual_nm {
        failures.push("contact-dynamics residual exceeds limit".to_string());
    }
    if maximum_contact_acceleration_residual > criteria.maximum_contact_acceleration_residual {
        failures.push("contact-acceleration residual exceeds limit".to_string());
    }
    if maximum_contact_friction_utilization > criteria.maximum_contact_friction_utilization {
        failures.push("contact friction-cone utilization exceeds limit".to_string());
    }
    if maximum_floating_base_dynamics_residual > criteria.maximum_floating_base_dynamics_residual {
        failures.push("floating-base dynamics residual exceeds limit".to_string());
    }
    if maximum_joint_utilization > criteria.maximum_joint_utilization {
        failures.push("joint utilization exceeds admissible bounds".to_string());
    }
    if whole_body_infeasible_cases > 0 {
        failures.push("one or more whole-body allocations were infeasible".to_string());
    }
    if valid.iter().any(|case| {
        case.report.planned_step.as_ref().is_some_and(|step| !step.feasible)
            || case.report.planned_swing.as_ref().is_some_and(|swing| !swing.feasible)
    }) {
        failures.push("one or more emitted recovery plans were infeasible".to_string());
    }
    if valid.iter().any(|case| {
        // Convergence and fallback are mutually exclusive states. When the
        // floating-base model is unavailable, it must not claim convergence.
        (case.report.contact_dynamics_converged == case.report.contact_dynamics_fallback)
            || (case.report.floating_base_model_available
                && (case.report.floating_base_dynamics_converged
                    == case.report.floating_base_dynamics_fallback))
            || (!case.report.floating_base_model_available
                && case.report.floating_base_dynamics_converged)
    }) {
        failures.push("dynamics solver state is inconsistent with fallback evidence".to_string());
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
        schema_version: 5,
        scenario_fingerprint: fingerprint_cases(&valid),
        criteria,
        candidate_build_id,
        model_artifact_sha256,
        controller_artifact_sha256,
        simulator_id,
        simulator_version,
        test_harness_sha256,
        scenario_config_sha256,
        scenario_manifest_id,
        scenario_manifest_digest,
        manifest_verified,
        corpus_digest: corpus_digest.unwrap_or_default(),
        total_cases: total,
        fallback_rate,
        budget_miss_rate,
        fall_rate,
        recovery_rate,
        fall_rate_upper_95,
        recovery_rate_lower_95,
        fall_count,
        recovery_trials,
        recovery_successes,
        maximum_terrain_height_std_m,
        maximum_terrain_evidence_age_s,
        maximum_inverse_dynamics_violation,
        maximum_contact_dynamics_residual_nm,
        maximum_contact_acceleration_residual,
        maximum_contact_friction_utilization,
        maximum_floating_base_dynamics_residual,
        maximum_joint_utilization,
        whole_body_infeasible_cases,
        solver_derived_cases,
        floating_base_cases,
        upper_body_contact_cases,
        accepted: failures.is_empty(),
        failures,
    }
}

/// Compute the stable BLAKE3 identity of a frozen manifest before a scenario
/// campaign runs. Store this digest in a protected build/qualification record;
/// the release verifier receives that trusted value separately from evidence JSON.
pub fn compute_scenario_manifest_digest(
    manifest: &EmbodiedScenarioManifest,
) -> std::result::Result<String, String> {
    if manifest.schema_version != 3 {
        return Err("unsupported manifest schema".to_string());
    }
    if manifest.manifest_id.trim().is_empty() {
        return Err("manifest identity is empty".to_string());
    }
    if manifest.candidate_build_id.trim().is_empty() {
        return Err("candidate build identity is empty".to_string());
    }
    if !is_lower_hex_sha256(&manifest.model_artifact_sha256)
        || !is_lower_hex_sha256(&manifest.controller_artifact_sha256)
        || !is_lower_hex_sha256(&manifest.test_harness_sha256)
        || !is_lower_hex_sha256(&manifest.scenario_config_sha256)
    {
        return Err(
            "model, controller, harness, and scenario config identities must be canonical SHA-256 hex"
                .to_string(),
        );
    }
    if manifest.simulator_id.trim().is_empty() || manifest.simulator_version.trim().is_empty() {
        return Err("simulator identity and version are required".to_string());
    }
    if !manifest.criteria.is_valid() {
        return Err("manifest qualification criteria are invalid".to_string());
    }
    if manifest.scenarios.len() < manifest.criteria.minimum_case_count {
        return Err("scenario inventory is below the minimum qualification case count".to_string());
    }
    if manifest.scenarios.iter().any(|scenario| scenario.scenario_id.trim().is_empty()) {
        return Err("expected scenario inventory contains a blank ID".to_string());
    }

    let mut expected = manifest.scenarios.clone();
    expected.sort_unstable_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then_with(|| a.random_seed.cmp(&b.random_seed))
    });
    if expected
        .windows(2)
        .any(|pair| pair[0].scenario_id == pair[1].scenario_id)
    {
        return Err("expected scenario inventory contains duplicate IDs".to_string());
    }
    let unique_seeds = expected
        .iter()
        .map(|scenario| scenario.random_seed)
        .collect::<std::collections::BTreeSet<_>>();
    if unique_seeds.len() != expected.len() {
        return Err("expected scenario inventory reuses random seeds".to_string());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-embodied-scenario-manifest-v3\0");
    hash_manifest_field(&mut hasher, manifest.manifest_id.as_bytes());
    hash_manifest_field(&mut hasher, manifest.candidate_build_id.as_bytes());
    hash_manifest_field(&mut hasher, manifest.model_artifact_sha256.as_bytes());
    hash_manifest_field(&mut hasher, manifest.controller_artifact_sha256.as_bytes());
    hash_manifest_field(&mut hasher, manifest.simulator_id.as_bytes());
    hash_manifest_field(&mut hasher, manifest.simulator_version.as_bytes());
    hash_manifest_field(&mut hasher, manifest.test_harness_sha256.as_bytes());
    hash_manifest_field(&mut hasher, manifest.scenario_config_sha256.as_bytes());
    let criteria_bytes = serde_json::to_vec(&manifest.criteria)
        .map_err(|_| "qualification criteria cannot be serialized".to_string())?;
    hash_manifest_field(&mut hasher, &criteria_bytes);
    hasher.update(&(expected.len() as u64).to_le_bytes());
    for scenario in expected {
        hash_manifest_field(&mut hasher, scenario.scenario_id.as_bytes());
        hasher.update(&scenario.random_seed.to_le_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_scenario_manifest(
    manifest: &EmbodiedScenarioManifest,
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
) -> std::result::Result<String, String> {
    if manifest.criteria != criteria {
        return Err("manifest qualification criteria do not match the enforced policy".to_string());
    }
    let digest = compute_scenario_manifest_digest(manifest)?;
    let mut expected = manifest.scenarios.clone();
    expected.sort_unstable_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then_with(|| a.random_seed.cmp(&b.random_seed))
    });
    let mut actual = cases
        .iter()
        .map(|case| EmbodiedScenarioSpec {
            scenario_id: case.scenario_id.clone(),
            random_seed: case.random_seed,
        })
        .collect::<Vec<_>>();
    actual.sort_unstable_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then_with(|| a.random_seed.cmp(&b.random_seed))
    });
    if actual != expected {
        return Err("scenario IDs or seeds do not exactly match the frozen inventory".to_string());
    }
    let unique_seeds = actual
        .iter()
        .map(|scenario| scenario.random_seed)
        .collect::<std::collections::BTreeSet<_>>();
    if unique_seeds.len() != actual.len() {
        return Err("evaluated scenario corpus reuses random seeds".to_string());
    }
    Ok(digest)
}

fn hash_manifest_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Canonicalize the complete case records independently of input ordering.
///
/// Each serialized record includes the controller report, dynamics fidelity,
/// terrain evidence, contact data, and observed outcome fields. Sorting the
/// byte encodings makes the corpus digest order-independent while retaining
/// duplicate records. The digest identifies evidence bytes; it does not prove
/// that the requested/frozen scenario manifest is complete or that evidence is true.
fn canonical_corpus_digest(cases: &[&EmbodiedControlCase]) -> Option<String> {
    let mut encoded = cases
        .iter()
        .map(|case| serde_json::to_vec(*case))
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?;
    encoded.sort_unstable();

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-embodied-control-corpus-v1\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    for record in encoded {
        hasher.update(&(record.len() as u64).to_le_bytes());
        hasher.update(&record);
    }
    Some(hasher.finalize().to_hex().to_string())
}

fn fingerprint_cases(cases: &[&EmbodiedControlCase]) -> u64 {
    let mut ordered = cases.to_vec();
    ordered.sort_unstable_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for case in ordered {
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

    fn sample_case(scenario_id: &str, terrain_evidence_age_s: f64) -> EmbodiedControlCase {
        let state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        let zero = HumanoidCommand::zero();
        let (_, report) = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21)
            .synthesize(HumanoidTask::Stand, &state, &zero, &zero, 1.0, 0.0);
        let random_seed = scenario_id
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
            });
        EmbodiedControlCase {
            scenario_id: scenario_id.to_string(),
            random_seed,
            recovery_challenge: true,
            dynamics_fidelity: DynamicsFidelity::ReducedOrder,
            active_contacts: 2,
            terrain_height_std_m: 0.01,
            terrain_evidence_age_s,
            report,
            fell: false,
            recovered: true,
        }
    }

    fn sample_corpus() -> Vec<EmbodiedControlCase> {
        (0..59)
            .map(|index| sample_case(&format!("case-{index:03}"), 0.02))
            .collect()
    }

    #[test]
    fn non_finite_controller_evidence_is_rejected() {
        let mut case = sample_case("non-finite-report", 0.02);
        case.report.contact_dynamics_residual_nm = f64::NAN;
        assert!(!case.validate());

        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("invalid or empty scenario corpus")
        }));
    }

    fn manifest_for(cases: &[EmbodiedControlCase]) -> EmbodiedScenarioManifest {
        EmbodiedScenarioManifest {
            schema_version: 3,
            manifest_id: "frozen-corpus-1".to_string(),
            candidate_build_id: "candidate-build-1".to_string(),
            model_artifact_sha256: "a".repeat(64),
            controller_artifact_sha256: "b".repeat(64),
            simulator_id: "mujoco".to_string(),
            simulator_version: "3.3.0".to_string(),
            test_harness_sha256: "c".repeat(64),
            scenario_config_sha256: "d".repeat(64),
            criteria: EmbodiedCertificationCriteria::default(),
            scenarios: cases
                .iter()
                .map(|case| EmbodiedScenarioSpec {
                    scenario_id: case.scenario_id.clone(),
                    random_seed: case.random_seed,
                })
                .collect(),
        }
    }

    #[test]
    fn negative_magnitudes_and_inconsistent_recovery_labels_are_rejected() {
        let mut negative = sample_case("negative-residual", 0.01);
        negative.report.inverse_dynamics_max_violation = -1.0e-3;
        assert!(!negative.validate());

        let mut contradictory = sample_case("recovered-without-fall", 0.01);
        contradictory.recovered = true;
        contradictory.recovery_challenge = false;
        contradictory.fell = false;
        assert!(!contradictory.validate());
    }

    #[test]
    fn body_report_uncertainty_cannot_be_hidden_by_case_summary() {
        let mut case = sample_case("terrain-disagreement", 0.01);
        case.report.terrain_max_height_std_m = 0.09;
        case.report.terrain_max_evidence_age_s = 0.01;
        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("terrain height uncertainty exceeds limit")
        }));
    }

    #[test]
    fn infeasible_whole_body_report_fails_qualification() {
        let mut case = sample_case("infeasible-body", 0.01);
        case.report.whole_body_feasible = false;
        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert_eq!(certificate.whole_body_infeasible_cases, 1);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("whole-body allocations were infeasible")
        }));
    }

    #[test]
    fn finite_but_excessive_solver_violation_fails_qualification() {
        let mut case = sample_case("qp-violation", 0.01);
        case.report.inverse_dynamics_max_violation = 1.0e-4;
        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("inverse-dynamics constraint violation exceeds limit")
        }));
    }

    #[test]
    fn wilson_bounds_fail_closed_on_empty_samples_and_bound_zero_failures() {
        assert_eq!(wilson_upper_95(0, 0), 1.0);
        assert_eq!(wilson_lower_95(0, 0), 0.0);
        assert!(wilson_upper_95(0, 59) < 0.05);
        assert!(wilson_lower_95(59, 59) >= 0.95);
        assert!(wilson_lower_95(58, 59) < 0.95);
    }

    #[test]
    fn contradictory_contact_solver_convergence_and_fallback_fails() {
        let mut case = sample_case("contact-both-true", 0.02);
        case.report.contact_dynamics_converged = true;
        case.report.contact_dynamics_fallback = true;
        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("inconsistent with fallback evidence")
        }));
    }

    #[test]
    fn missing_contact_solver_outcome_fails() {
        let mut case = sample_case("contact-neither", 0.02);
        case.report.contact_dynamics_converged = false;
        case.report.contact_dynamics_fallback = false;
        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("inconsistent with fallback evidence")
        }));
    }

    #[test]
    fn certificate_rejects_a_recovery_rate_without_enough_successes() {
        let mut cases = sample_corpus();
        cases[0].recovered = false;
        let manifest = manifest_for(&cases);
        let certificate = certify_embodied_control_against_manifest(
            &cases,
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert_eq!(certificate.recovery_trials, 59);
        assert_eq!(certificate.recovery_successes, 58);
        assert!(certificate.recovery_rate_lower_95 < 0.95);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("recovery-rate lower bound")
        }));
    }

    #[test]
    fn certificate_uses_fall_confidence_bound_not_point_rate() {
        let mut cases = sample_corpus();
        cases[0].fell = true;
        let manifest = manifest_for(&cases);
        let certificate = certify_embodied_control_against_manifest(
            &cases,
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(certificate.fall_rate < 0.05);
        assert!(certificate.fall_rate_upper_95 > 0.05);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("fall-rate upper bound")
        }));
    }

    #[test]
    fn frozen_manifest_rejects_seed_drift_and_reused_seeds() {
        let cases = sample_corpus();
        let manifest = manifest_for(&cases);
        let mut drifted = manifest.clone();
        drifted.scenarios[0].random_seed = drifted.scenarios[0].random_seed.wrapping_add(1);
        let certificate = certify_embodied_control_against_manifest(
            &cases,
            EmbodiedCertificationCriteria::default(),
            &drifted,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("scenario IDs or seeds do not exactly match")
        }));

        let mut duplicated = manifest;
        duplicated.scenarios[1].random_seed = duplicated.scenarios[0].random_seed;
        assert!(compute_scenario_manifest_digest(&duplicated).is_err());
    }

    #[test]
    fn criteria_cannot_weaken_solver_or_safety_contracts() {
        let mut criteria = EmbodiedCertificationCriteria::default();
        criteria.maximum_fall_rate = 0.50;
        assert!(!criteria.is_valid());

        let mut criteria = EmbodiedCertificationCriteria::default();
        criteria.maximum_contact_friction_utilization = 1.5;
        assert!(!criteria.is_valid());

        let mut criteria = EmbodiedCertificationCriteria::default();
        criteria.require_floating_base_case = false;
        assert!(!criteria.is_valid());
    }

    #[test]
    fn non_finite_qualification_thresholds_fail_closed() {
        let cases = vec![sample_case("case-a", 0.02)];
        let manifest = manifest_for(&cases);
        let mut criteria = EmbodiedCertificationCriteria::default();
        criteria.maximum_fall_rate = f64::NAN;
        let certificate =
            certify_embodied_control_against_manifest(&cases, criteria, &manifest);
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("non-finite or out-of-domain thresholds")
        }));
    }

    #[test]
    fn manifest_digest_binds_artifact_and_qualification_policy_identity() {
        let cases = sample_corpus();
        let manifest = manifest_for(&cases);
        let digest = compute_scenario_manifest_digest(&manifest).unwrap();

        let mut changed = manifest.clone();
        changed.controller_artifact_sha256 = "e".repeat(64);
        assert_ne!(compute_scenario_manifest_digest(&changed).unwrap(), digest);

        let mut changed_policy = manifest.clone();
        changed_policy.criteria.maximum_fall_rate = 0.04;
        assert_ne!(
            compute_scenario_manifest_digest(&changed_policy).unwrap(),
            digest
        );
        let certificate = certify_embodied_control_against_manifest(
            &cases,
            EmbodiedCertificationCriteria::default(),
            &changed_policy,
        );
        assert!(!certificate.accepted);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("do not match the enforced policy")
        }));
    }

    #[test]
    fn missing_manifest_fails_closed_for_nonempty_corpus() {
        let cases = vec![sample_case("case-a", 0.02)];
        let certificate = certify_embodied_control(
            &cases,
            EmbodiedCertificationCriteria::default(),
        );
        assert!(!certificate.accepted);
        assert!(!certificate.manifest_verified);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("missing frozen scenario manifest")
        }));
    }

    #[test]
    fn frozen_manifest_rejects_truncated_corpus() {
        let cases = sample_corpus();
        let manifest = manifest_for(&cases);
        let certificate = certify_embodied_control_against_manifest(
            &cases[..58],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(!certificate.manifest_verified);
        assert!(certificate.failures.iter().any(|failure| {
            failure.contains("do not exactly match")
        }));
    }

    #[test]
    fn exact_frozen_manifest_is_bound_to_candidate_and_model_identity() {
        let cases = sample_corpus();
        let manifest = manifest_for(&cases);
        let certificate = certify_embodied_control_against_manifest(
            &cases,
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(certificate.manifest_verified);
        assert_eq!(certificate.candidate_build_id, manifest.candidate_build_id);
        assert_eq!(certificate.model_artifact_sha256, manifest.model_artifact_sha256);
        assert_eq!(certificate.scenario_manifest_digest.len(), 64);
        assert_eq!(
            compute_scenario_manifest_digest(&manifest).unwrap(),
            certificate.scenario_manifest_digest,
        );

        let reversed = cases.iter().cloned().rev().collect::<Vec<_>>();
        let recomputed = certify_embodied_control_against_manifest(
            &reversed,
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert_eq!(
            serde_json::to_vec(&certificate).unwrap(),
            serde_json::to_vec(&recomputed).unwrap(),
            "full certificate identity must not depend on case order",
        );
    }

    #[test]
    fn corpus_digest_is_order_independent_and_binds_complete_records() {
        let cases = vec![sample_case("case-a", 0.02), sample_case("case-b", 0.03)];
        let forward = cases.iter().collect::<Vec<_>>();
        let mut reverse = cases.iter().collect::<Vec<_>>();
        reverse.reverse();
        assert_eq!(
            canonical_corpus_digest(&forward),
            canonical_corpus_digest(&reverse)
        );

        let mut changed = cases.clone();
        changed[0].terrain_evidence_age_s = 0.04;
        let changed_refs = changed.iter().collect::<Vec<_>>();
        assert_ne!(
            canonical_corpus_digest(&forward),
            canonical_corpus_digest(&changed_refs)
        );
    }

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
            random_seed: 0x52_45_44_55_43_45_44,
            recovery_challenge: true,
            dynamics_fidelity: DynamicsFidelity::ReducedOrder,
            active_contacts: 2,
            terrain_height_std_m: 0.0,
            terrain_evidence_age_s: 0.0,
            report,
            fell: false,
            recovered: false,
        };
        let manifest = manifest_for(std::slice::from_ref(&case));
        let certificate = certify_embodied_control_against_manifest(
            &[case],
            EmbodiedCertificationCriteria::default(),
            &manifest,
        );
        assert!(!certificate.accepted);
        assert!(
            certificate
                .failures
                .iter()
                .any(|failure| failure.contains("solver-derived"))
        );
    }
}
