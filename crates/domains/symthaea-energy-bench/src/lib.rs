// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent semantic checks for versioned energy-conversion benchmark records.
//!
//! This crate checks cross-field semantics JSON Schema cannot express easily:
//! time coverage, interval ordering/continuity, energy conservation, provenance
//! metadata shape, cached aggregate metrics, and synthetic-data decision guards.
//! It does not claim model validity or propagate measurement uncertainty.

use chrono::DateTime;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const EXPECTED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkRun {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub not_for_decision: bool,
    pub mission: Mission,
    pub boundary: Boundary,
    pub candidate: Candidate,
    pub solver: Solver,
    pub intervals: Vec<EnergyInterval>,
    pub evaluation_policy: EvaluationPolicy,
    pub uncertainty: Uncertainty,
    pub data_provenance: Vec<ProvenanceArtifact>,
    pub computed_metrics: Option<CachedMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mission {
    pub mission_id: String,
    pub duty_cycle_id: String,
    pub expected_duration_s: f64,
    pub ambient_temperature_k: Option<f64>,
    pub ambient_pressure_pa: Option<f64>,
    pub required_phases: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Boundary {
    pub boundary_id: String,
    pub auxiliary_treatment: String,
    pub storage_treatment: String,
    pub upstream_scope: String,
    pub custom_scope_definition: Option<String>,
    pub energy_unit: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidate {
    pub candidate_id: String,
    pub configuration_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Solver {
    pub method: String,
    pub version: String,
    pub code_revision: String,
    pub configuration_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnergyInterval {
    pub index: usize,
    pub phase: String,
    pub start_s: f64,
    pub end_s: f64,
    pub energy_j: EnergyFlows,
    pub standard_uncertainty_j: EnergyUncertainty,
    pub source_artifact_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnergyFlows {
    pub external_input: f64,
    pub net_useful_output: f64,
    pub auxiliary: f64,
    pub rejected_or_loss: f64,
    pub stored_energy_change: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnergyUncertainty {
    pub external_input: f64,
    pub net_useful_output: f64,
    pub auxiliary: f64,
    pub rejected_or_loss: f64,
    pub stored_energy_change: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationPolicy {
    pub policy_id: String,
    pub registered_before_search: bool,
    pub preregistered_at: String,
    pub relative_balance_tolerance: f64,
    pub time_continuity_tolerance_s: f64,
    pub minimum_superiority_margin: f64,
    pub baseline_id: String,
    pub selection_rule: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Uncertainty {
    pub quantity_type: String,
    pub coverage_factor: f64,
    pub correlation_model: String,
    pub covariance_matrix_artifact_id: Option<String>,
    pub monte_carlo_artifact_id: Option<String>,
    pub method: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProvenanceArtifact {
    pub artifact_id: String,
    pub kind: String,
    pub sha256: Option<String>,
    pub retrieved_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CachedMetrics {
    pub total_external_input_j: Option<f64>,
    pub total_net_useful_output_j: Option<f64>,
    pub net_stored_energy_change_j: Option<f64>,
    pub net_cycle_efficiency: Option<f64>,
    /// This field is deliberately ignored: a single run cannot establish comparison superiority.
    pub candidate_beats_baseline: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedTotals {
    pub total_external_input_j: f64,
    pub total_net_useful_output_j: f64,
    pub total_auxiliary_j: f64,
    pub total_rejected_or_loss_j: f64,
    pub net_stored_energy_change_j: f64,
    pub total_energy_throughput_j: f64,
    pub aggregate_residual_j: f64,
    pub maximum_interval_relative_balance_error: f64,
    pub net_cycle_efficiency: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub issues: Vec<String>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            return f.write_str("benchmark validation failed");
        }
        write!(f, "benchmark validation failed: {}", self.issues.join("; "))
    }
}

impl std::error::Error for ValidationError {}

/// Parse JSON and run independent semantic checks. Unknown extra fields are left
/// to the published JSON Schema; this function validates physical/accounting semantics.
pub fn validate_json(bytes: &[u8]) -> Result<VerifiedTotals, ValidationError> {
    let run: BenchmarkRun = serde_json::from_slice(bytes).map_err(|error| ValidationError {
        issues: vec![format!("invalid benchmark JSON or required record shape: {error}")],
    })?;
    validate_run(&run)
}

/// Validate mission coverage, provenance metadata, per-interval balances, and aggregates.
pub fn validate_run(run: &BenchmarkRun) -> Result<VerifiedTotals, ValidationError> {
    let mut issues = Vec::new();
    let policy = &run.evaluation_policy;
    let time_tolerance = policy.time_continuity_tolerance_s;
    let balance_tolerance = policy.relative_balance_tolerance;

    if run.schema_version != EXPECTED_SCHEMA_VERSION {
        issues.push(format!("unsupported schema_version {}; expected {EXPECTED_SCHEMA_VERSION}", run.schema_version));
    }
    if [
        run.benchmark_id.as_str(),
        run.mission.mission_id.as_str(),
        run.mission.duty_cycle_id.as_str(),
        run.boundary.boundary_id.as_str(),
        run.candidate.candidate_id.as_str(),
        run.solver.method.as_str(),
        run.solver.version.as_str(),
    ].iter().any(|value| value.trim().is_empty()) {
        issues.push("benchmark, mission, boundary, candidate, and solver identifiers must be non-empty".into());
    }
    if run.evidence_class == "synthetic_fixture" && !run.not_for_decision {
        issues.push("synthetic fixtures must set not_for_decision=true".into());
    }
    if !matches!(run.evidence_class.as_str(), "synthetic_fixture" | "published_data" | "measured" | "simulated" | "mixed") {
        issues.push(format!("unsupported evidence_class {:?}", run.evidence_class));
    }
    if run.boundary.energy_unit != "J" {
        issues.push("benchmark v1 energy_unit must be J".into());
    }
    if run.boundary.auxiliary_treatment != "separate_nonuseful_energy" {
        issues.push("auxiliary energy must be declared separately from useful output".into());
    }
    if run.boundary.storage_treatment != "signed_change_in_stored_energy" {
        issues.push("storage must use signed change in stored energy".into());
    }
    if run.boundary.upstream_scope == "custom" && run.boundary.custom_scope_definition.as_deref().unwrap_or("").trim().is_empty() {
        issues.push("custom upstream scope requires custom_scope_definition".into());
    }
    if !valid_git_revision(&run.solver.code_revision) {
        issues.push("solver.code_revision must be a 40-character hexadecimal Git revision".into());
    }
    for (label, value) in [
        ("candidate.configuration_sha256", run.candidate.configuration_sha256.as_deref()),
        ("solver.configuration_sha256", run.solver.configuration_sha256.as_deref()),
    ] {
        if value.is_some_and(|value| !valid_sha256(value)) {
            issues.push(format!("{label} must be a 64-character lowercase hexadecimal SHA-256"));
        }
    }

    if !policy.registered_before_search {
        issues.push("evaluation policy was not registered before search".into());
    }
    if DateTime::parse_from_rfc3339(&policy.preregistered_at).is_err() {
        issues.push("evaluation_policy.preregistered_at must be RFC 3339".into());
    }
    if !balance_tolerance.is_finite() || !(0.0..1.0).contains(&balance_tolerance) {
        issues.push("relative_balance_tolerance must be finite and in [0, 1)".into());
    }
    if !time_tolerance.is_finite() || time_tolerance < 0.0 {
        issues.push("time_continuity_tolerance_s must be finite and non-negative".into());
    }
    if !policy.minimum_superiority_margin.is_finite() || !(0.0..=1.0).contains(&policy.minimum_superiority_margin) {
        issues.push("minimum_superiority_margin must be finite and in [0, 1]".into());
    }
    if policy.policy_id.trim().is_empty() || policy.baseline_id.trim().is_empty() || policy.selection_rule.trim().is_empty() {
        issues.push("policy ID, baseline ID, and selection rule must be non-empty".into());
    }
    if !run.mission.expected_duration_s.is_finite() || run.mission.expected_duration_s <= 0.0 {
        issues.push("mission expected duration must be finite and positive".into());
    }
    for (name, value) in [
        ("ambient_temperature_k", run.mission.ambient_temperature_k),
        ("ambient_pressure_pa", run.mission.ambient_pressure_pa),
    ] {
        if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            issues.push(format!("mission {name} must be finite and positive when supplied"));
        }
    }
    if !run.uncertainty.coverage_factor.is_finite() || run.uncertainty.coverage_factor <= 0.0 {
        issues.push("uncertainty coverage_factor must be finite and positive".into());
    }
    if run.uncertainty.quantity_type != "combined_standard_uncertainty" {
        issues.push("unsupported uncertainty quantity_type".into());
    }
    if run.uncertainty.method.trim().is_empty() {
        issues.push("uncertainty method must not be empty".into());
    }
    match run.uncertainty.correlation_model.as_str() {
        "covariance_matrix_artifact" if run.uncertainty.covariance_matrix_artifact_id.as_deref().unwrap_or("").is_empty() => {
            issues.push("covariance-matrix correlation model requires an artifact ID".into());
        }
        "monte_carlo_propagation" if run.uncertainty.monte_carlo_artifact_id.as_deref().unwrap_or("").is_empty() => {
            issues.push("Monte Carlo correlation model requires an artifact ID".into());
        }
        "independent" | "covariance_matrix_artifact" | "monte_carlo_propagation" | "unknown" => {}
        other => issues.push(format!("unsupported uncertainty correlation model {other:?}")),
    }

    if run.intervals.is_empty() {
        issues.push("duty cycle must contain at least one interval".into());
        return Err(ValidationError { issues });
    }
    if run.data_provenance.is_empty() {
        issues.push("at least one provenance artifact is required".into());
    }
    let mut artifact_ids = BTreeSet::new();
    for artifact in &run.data_provenance {
        if artifact.artifact_id.trim().is_empty() || !artifact_ids.insert(artifact.artifact_id.as_str()) {
            issues.push(format!("provenance artifact ID is empty or duplicated: {:?}", artifact.artifact_id));
        }
        if artifact.kind != "synthetic_fixture" && !artifact.sha256.as_deref().is_some_and(valid_sha256) {
            issues.push(format!("non-synthetic artifact {:?} requires a valid SHA-256", artifact.artifact_id));
        }
        if artifact.sha256.as_deref().is_some_and(|hash| !valid_sha256(hash)) {
            issues.push(format!("artifact {:?} has malformed SHA-256", artifact.artifact_id));
        }
        if artifact.retrieved_at.as_deref().is_some_and(|value| DateTime::parse_from_rfc3339(value).is_err()) {
            issues.push(format!("artifact {:?} retrieved_at must be RFC 3339", artifact.artifact_id));
        }
    }

    let required_phases: BTreeSet<&str> = run.mission.required_phases.iter().map(String::as_str).collect();
    let mut observed_phases = BTreeSet::new();
    let mut total_input = 0.0;
    let mut total_useful = 0.0;
    let mut total_auxiliary = 0.0;
    let mut total_loss = 0.0;
    let mut total_storage_change = 0.0;
    let mut total_throughput = 0.0;
    let mut max_interval_error: f64 = 0.0;
    let mut previous_end: Option<f64> = None;

    for (position, interval) in run.intervals.iter().enumerate() {
        if interval.index != position {
            issues.push(format!("interval index {} must equal zero-based position {position}", interval.index));
        }
        if interval.phase.trim().is_empty() {
            issues.push(format!("interval {} has an empty phase", interval.index));
        }
        observed_phases.insert(interval.phase.as_str());
        if !interval.start_s.is_finite() || !interval.end_s.is_finite() {
            issues.push(format!("interval {} timestamps must be finite", interval.index));
        } else {
            if interval.end_s <= interval.start_s {
                issues.push(format!("interval {} must have positive duration", interval.index));
            }
            if let Some(end) = previous_end {
                if (interval.start_s - end).abs() > time_tolerance {
                    issues.push(format!("interval {} is discontinuous: starts at {}, previous interval ends at {end}", interval.index, interval.start_s));
                }
            } else if interval.start_s.abs() > time_tolerance {
                issues.push(format!("first interval must start at mission time zero, got {}", interval.start_s));
            }
            previous_end = Some(interval.end_s);
        }

        let flows = &interval.energy_j;
        for (name, value) in [
            ("external_input", flows.external_input),
            ("net_useful_output", flows.net_useful_output),
            ("auxiliary", flows.auxiliary),
            ("rejected_or_loss", flows.rejected_or_loss),
        ] {
            if !value.is_finite() || value < 0.0 {
                issues.push(format!("interval {} {name} must be finite and non-negative", interval.index));
            }
        }
        if !flows.stored_energy_change.is_finite() {
            issues.push(format!("interval {} stored_energy_change must be finite", interval.index));
        }
        for (name, value) in [
            ("external_input", interval.standard_uncertainty_j.external_input),
            ("net_useful_output", interval.standard_uncertainty_j.net_useful_output),
            ("auxiliary", interval.standard_uncertainty_j.auxiliary),
            ("rejected_or_loss", interval.standard_uncertainty_j.rejected_or_loss),
            ("stored_energy_change", interval.standard_uncertainty_j.stored_energy_change),
        ] {
            if !value.is_finite() || value < 0.0 {
                issues.push(format!("interval {} standard uncertainty {name} must be finite and non-negative", interval.index));
            }
        }
        if let Some(ids) = &interval.source_artifact_ids {
            for id in ids {
                if !artifact_ids.contains(id.as_str()) {
                    issues.push(format!("interval {} references unknown source artifact {id:?}", interval.index));
                }
            }
        }

        let throughput = flows.external_input + (-flows.stored_energy_change).max(0.0);
        let residual = flows.external_input
            - flows.net_useful_output
            - flows.auxiliary
            - flows.rejected_or_loss
            - flows.stored_energy_change;
        if !throughput.is_finite() || !residual.is_finite() {
            issues.push(format!("interval {} energy arithmetic overflowed", interval.index));
        } else if throughput <= 0.0 {
            issues.push(format!("interval {} has no positive energy throughput", interval.index));
        } else {
            let relative_error = residual.abs() / throughput;
            if !relative_error.is_finite() {
                issues.push(format!("interval {} relative energy error is non-finite", interval.index));
            } else {
                max_interval_error = max_interval_error.max(relative_error);
                if relative_error > balance_tolerance {
                    issues.push(format!("interval {} first-law error {:.6e} exceeds preregistered tolerance {:.6e}", interval.index, relative_error, balance_tolerance));
                }
            }
        }

        total_input += flows.external_input;
        total_useful += flows.net_useful_output;
        total_auxiliary += flows.auxiliary;
        total_loss += flows.rejected_or_loss;
        total_storage_change += flows.stored_energy_change;
        total_throughput += throughput;
        if [total_input, total_useful, total_auxiliary, total_loss, total_storage_change, total_throughput]
            .iter().any(|value| !value.is_finite()) {
            issues.push("aggregate energy sum overflowed".into());
            break;
        }
    }

    for phase in &required_phases {
        if !observed_phases.contains(phase) {
            issues.push(format!("required mission phase {phase:?} is missing"));
        }
    }
    if let Some(last) = run.intervals.last() {
        if last.end_s.is_finite()
            && run.mission.expected_duration_s.is_finite()
            && (last.end_s - run.mission.expected_duration_s).abs() > time_tolerance {
            issues.push(format!("last interval ends at {}, mission duration is {}", last.end_s, run.mission.expected_duration_s));
        }
    }

    let aggregate_residual = total_input - total_useful - total_auxiliary - total_loss - total_storage_change;
    if !aggregate_residual.is_finite() || !total_throughput.is_finite() || total_throughput <= 0.0 {
        issues.push("aggregate energy balance is non-finite or has no throughput".into());
    } else {
        let aggregate_relative_error = aggregate_residual.abs() / total_throughput;
        if !aggregate_relative_error.is_finite() || aggregate_relative_error > balance_tolerance {
            issues.push(format!("aggregate first-law error {:.6e} exceeds preregistered tolerance {:.6e}", aggregate_relative_error, balance_tolerance));
        }
    }

    let storage_is_balanced = total_storage_change.is_finite()
        && total_throughput.is_finite()
        && total_storage_change.abs() <= balance_tolerance * total_throughput;
    let efficiency = if storage_is_balanced && total_storage_change >= 0.0 && total_input > 0.0 {
        Some(total_useful / total_input)
    } else {
        None
    };
    if efficiency.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        issues.push("charge-sustaining net cycle efficiency is outside [0, 1]".into());
    }

    if let Some(metrics) = &run.computed_metrics {
        let metric_tolerance = balance_tolerance.max(1e-12);
        for (name, cached, recomputed) in [
            ("total_external_input_j", metrics.total_external_input_j, total_input),
            ("total_net_useful_output_j", metrics.total_net_useful_output_j, total_useful),
            ("net_stored_energy_change_j", metrics.net_stored_energy_change_j, total_storage_change),
        ] {
            if let Some(cached) = cached {
                if !cached.is_finite() || !approximately_equal(cached, recomputed, metric_tolerance) {
                    issues.push(format!("cached metric {name} does not match recomputed total"));
                }
            }
        }
        match (metrics.net_cycle_efficiency, efficiency) {
            (Some(cached), Some(recomputed)) if approximately_equal(cached, recomputed, metric_tolerance) => {}
            (Some(_), _) => issues.push("cached net_cycle_efficiency is present but cannot be independently confirmed".into()),
            (None, Some(_)) => issues.push("cached net_cycle_efficiency is missing despite charge-sustaining totals".into()),
            (None, None) => {}
        }
        // candidate_beats_baseline is deliberately ignored: comparison requires
        // two independently validated manifests under the same frozen policy.
    }

    if issues.is_empty() {
        Ok(VerifiedTotals {
            total_external_input_j: total_input,
            total_net_useful_output_j: total_useful,
            total_auxiliary_j: total_auxiliary,
            total_rejected_or_loss_j: total_loss,
            net_stored_energy_change_j: total_storage_change,
            total_energy_throughput_j: total_throughput,
            aggregate_residual_j: aggregate_residual,
            maximum_interval_relative_balance_error: max_interval_error,
            net_cycle_efficiency: efficiency,
        })
    } else {
        Err(ValidationError { issues })
    }
}

/// Verify a SHA-256 digest against exact artifact bytes. URI resolution and access
/// permissions remain the caller's responsibility.
pub fn verify_sha256(expected_lowercase_hex: &str, bytes: &[u8]) -> bool {
    if !valid_sha256(expected_lowercase_hex) {
        return false;
    }
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>() == expected_lowercase_hex
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_git_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) || (b'A'..=b'F').contains(&byte))
}

fn approximately_equal(left: f64, right: f64, tolerance: f64) -> bool {
    left.is_finite() && right.is_finite()
        && (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../../../docs/research/examples/energy-conversion-benchmark.synthetic.json");

    #[test]
    fn synthetic_fixture_closes_each_interval_and_aggregate() {
        let report = validate_json(FIXTURE.as_bytes()).expect("fixture should validate");
        assert_eq!(report.total_external_input_j, 112.0);
        assert!((report.total_net_useful_output_j - 35.7).abs() < 1e-12);
        assert!((report.total_auxiliary_j - 6.1).abs() < 1e-12);
        assert!((report.total_rejected_or_loss_j - 70.2).abs() < 1e-12);
        assert_eq!(report.net_stored_energy_change_j, 0.0);
        assert!(report.aggregate_residual_j.abs() < 1e-12);
        assert_eq!(report.net_cycle_efficiency, Some(35.7 / 112.0));
    }

    #[test]
    fn timeline_gaps_are_rejected() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.intervals[1].start_s += 1.0;
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("discontinuous")));
    }

    #[test]
    fn missing_required_shutdown_phase_is_rejected() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.intervals.pop();
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("required mission phase \"shutdown\" is missing")));
    }

    #[test]
    fn synthetic_record_cannot_be_marked_decision_eligible() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.not_for_decision = false;
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("synthetic fixtures must set not_for_decision=true")));
    }

    #[test]
    fn cached_metrics_are_recomputed_not_trusted() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.computed_metrics.as_mut().unwrap().total_external_input_j = Some(113.0);
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("cached metric total_external_input_j")));
    }

    #[test]
    fn exact_bytes_are_checked_against_sha256() {
        const ABC_SHA256: &str =
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify_sha256(ABC_SHA256, b"abc"));
        assert!(!verify_sha256(ABC_SHA256, b"abd"));
        assert!(!verify_sha256("not-a-digest", b"abc"));
    }

    #[test]
    fn discharge_does_not_produce_naive_cycle_efficiency() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.intervals[0].energy_j.stored_energy_change = 0.0;
        run.intervals[0].energy_j.rejected_or_loss = 8.5;
        run.computed_metrics.as_mut().unwrap().net_stored_energy_change_j = Some(-2.0);
        run.computed_metrics.as_mut().unwrap().net_cycle_efficiency = None;
        let report = validate_run(&run).unwrap();
        assert_eq!(report.net_stored_energy_change_j, -2.0);
        assert_eq!(report.net_cycle_efficiency, None);
    }
}
