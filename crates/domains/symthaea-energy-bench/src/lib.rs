// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent semantic checks for versioned energy-conversion benchmark records.
//!
//! This crate checks cross-field semantics JSON Schema cannot express easily:
//! time coverage, interval ordering/continuity, energy conservation, provenance
//! metadata shape, cached aggregate metrics, and synthetic-data decision guards.
//! Candidate comparison uses a narrow first-order uncertainty approximation only
//! under declared independence and subtracts a registered worst-case differential-
//! bias bound; it is not a general covariance/Monte Carlo measurement model.

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
    pub description: String,
    pub platform: String,
    pub duty_cycle_id: String,
    pub profile_source_uri: Option<String>,
    pub expected_duration_s: f64,
    pub ambient_temperature_k: Option<f64>,
    pub ambient_pressure_pa: Option<f64>,
    pub required_phases: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Boundary {
    pub boundary_id: String,
    pub input_definition: String,
    pub useful_output_definition: String,
    pub auxiliary_treatment: String,
    pub storage_treatment: String,
    pub upstream_scope: String,
    pub custom_scope_definition: Option<String>,
    pub energy_unit: String,
    pub power_unit: Option<String>,
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
    pub model_fidelity: String,
    pub random_seed: Option<i64>,
    #[serde(default)]
    pub calibration_dataset_ids: Vec<String>,
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
    pub search_started_at: String,
    pub relative_balance_tolerance: f64,
    pub time_continuity_tolerance_s: f64,
    pub minimum_superiority_margin: f64,
    /// Registered worst-case absolute bound on unmodeled differential bias
    /// between candidate and baseline efficiency, expressed as a fraction.
    /// This is a conservative bound, not a standard uncertainty.
    pub maximum_unmodeled_differential_bias_efficiency: f64,
    /// Evidence/reasoning used to justify the registered bound. This text is
    /// an audit aid, not independent authentication of the evidence.
    pub differential_bias_bound_basis: String,
    /// Content-addressed reference to the calibration record in data_provenance.
    /// The caller must separately verify the referenced artifact's exact bytes.
    pub differential_bias_calibration_artifact_id: String,
    pub differential_bias_calibration_artifact_sha256: String,
    pub baseline_id: String,
    pub selection_rule: String,
    pub comparison_metric_id: String,
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
pub enum ComparisonDecision {
    /// The declared expanded-uncertainty lower bound exceeds the preregistered margin.
    LowerBoundClearsRegisteredMargin,
    /// The estimated advantage did not clear the preregistered margin.
    LowerBoundDoesNotClearMargin,
    /// Evidence or uncertainty assumptions are insufficient for a decision.
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EfficiencyComparison {
    pub decision: ComparisonDecision,
    pub evidence_class: String,
    pub candidate_efficiency: Option<f64>,
    pub baseline_efficiency: Option<f64>,
    pub candidate_standard_uncertainty: Option<f64>,
    pub baseline_standard_uncertainty: Option<f64>,
    pub estimated_efficiency_difference: Option<f64>,
    pub expanded_difference_uncertainty: Option<f64>,
    pub conservative_lower_bound: Option<f64>,
    /// Absolute efficiency fraction (e.g. 0.02 means two percentage points).
    pub required_superiority_margin: f64,
    /// Registered worst-case bound subtracted directly from the efficiency
    /// difference after expanded measurement uncertainty.
    pub maximum_unmodeled_differential_bias_efficiency: f64,
    pub coverage_factor: f64,
    pub reason: Option<String>,
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
        issues.push(format!(
            "unsupported schema_version {}; expected {EXPECTED_SCHEMA_VERSION}",
            run.schema_version
        ));
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
        issues.push(
            "benchmark, mission, boundary, candidate, and solver identifiers must be non-empty"
                .into(),
        );
    }
    if run.evidence_class == "synthetic_fixture" && !run.not_for_decision {
        issues.push("synthetic fixtures must set not_for_decision=true".into());
    }
    if !matches!(
        run.evidence_class.as_str(),
        "synthetic_fixture" | "published_data" | "measured" | "simulated" | "mixed"
    ) {
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
    if run.boundary.upstream_scope == "custom"
        && run
            .boundary
            .custom_scope_definition
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
    {
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
    let preregistered_at = DateTime::parse_from_rfc3339(&policy.preregistered_at);
    let search_started_at = DateTime::parse_from_rfc3339(&policy.search_started_at);
    if preregistered_at.is_err() {
        issues.push("evaluation_policy.preregistered_at must be RFC 3339".into());
    }
    if search_started_at.is_err() {
        issues.push("evaluation_policy.search_started_at must be RFC 3339".into());
    }
    if let (Ok(policy_time), Ok(search_time)) = (preregistered_at, search_started_at) {
        if policy_time >= search_time {
            issues.push("evaluation policy must be timestamped before search_started_at".into());
        }
    }
    if !balance_tolerance.is_finite() || !(0.0..1.0).contains(&balance_tolerance) {
        issues.push("relative_balance_tolerance must be finite and in [0, 1)".into());
    }
    if !time_tolerance.is_finite() || time_tolerance < 0.0 {
        issues.push("time_continuity_tolerance_s must be finite and non-negative".into());
    }
    if !policy.minimum_superiority_margin.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_superiority_margin)
    {
        issues.push("minimum_superiority_margin must be finite and in [0, 1]".into());
    }
    if !policy.maximum_unmodeled_differential_bias_efficiency.is_finite()
        || !(0.0..=1.0).contains(&policy.maximum_unmodeled_differential_bias_efficiency)
    {
        issues.push(
            "maximum_unmodeled_differential_bias_efficiency must be finite and in [0, 1]"
                .into(),
        );
    }
    if policy.policy_id.trim().is_empty()
        || policy.baseline_id.trim().is_empty()
        || policy.selection_rule.trim().is_empty()
        || policy.differential_bias_bound_basis.trim().is_empty()
        || policy.differential_bias_calibration_artifact_id.trim().is_empty()
    {
        issues.push(
            "policy ID, baseline ID, selection rule, differential-bias-bound basis, and calibration artifact ID must be non-empty"
                .into(),
        );
    }
    if policy.comparison_metric_id != "net_cycle_efficiency" {
        issues.push(
            "benchmark contract v1 only supports comparison_metric_id=net_cycle_efficiency".into(),
        );
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
        "covariance_matrix_artifact"
            if run
                .uncertainty
                .covariance_matrix_artifact_id
                .as_deref()
                .unwrap_or("")
                .is_empty() =>
        {
            issues.push("covariance-matrix correlation model requires an artifact ID".into());
        }
        "monte_carlo_propagation"
            if run
                .uncertainty
                .monte_carlo_artifact_id
                .as_deref()
                .unwrap_or("")
                .is_empty() =>
        {
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
    if !valid_sha256(&policy.differential_bias_calibration_artifact_sha256) {
        issues.push(
            "evaluation_policy.differential_bias_calibration_artifact_sha256 must be a 64-character lowercase hexadecimal SHA-256".into(),
        );
    }
    let mut artifact_ids = BTreeSet::new();
    for artifact in &run.data_provenance {
        if artifact.artifact_id.trim().is_empty()
            || !artifact_ids.insert(artifact.artifact_id.as_str())
        {
            issues.push(format!(
                "provenance artifact ID is empty or duplicated: {:?}",
                artifact.artifact_id
            ));
        }
        if artifact.kind != "synthetic_fixture"
            && !artifact.sha256.as_deref().is_some_and(valid_sha256)
        {
            issues.push(format!(
                "non-synthetic artifact {:?} requires a valid SHA-256",
                artifact.artifact_id
            ));
        }
        if artifact.sha256.as_deref().is_some_and(|hash| !valid_sha256(hash)) {
            issues.push(format!("artifact {:?} has malformed SHA-256", artifact.artifact_id));
        }
        if artifact
            .retrieved_at
            .as_deref()
            .is_some_and(|value| DateTime::parse_from_rfc3339(value).is_err())
        {
            issues.push(format!(
                "artifact {:?} retrieved_at must be RFC 3339",
                artifact.artifact_id
            ));
        }
    }

    match run.data_provenance.iter().find(|artifact| {
        artifact.artifact_id == policy.differential_bias_calibration_artifact_id
    }) {
        Some(artifact)
            if artifact.sha256.as_deref()
                == Some(policy.differential_bias_calibration_artifact_sha256.as_str()) => {}
        Some(_) => issues.push(
            "evaluation policy calibration artifact SHA-256 does not match its provenance entry".into(),
        ),
        None => issues.push(
            "evaluation policy calibration artifact ID is not present in data_provenance".into(),
        ),
    }

    let required_phases: BTreeSet<&str> = run
        .mission
        .required_phases
        .iter()
        .map(String::as_str)
        .collect();
    let mut observed_phases = BTreeSet::new();
    let mut total_input = 0.0;
    let mut total_useful = 0.0;
    let mut total_auxiliary = 0.0;
    let mut total_loss = 0.0;
    let mut total_storage_change = 0.0;
    let mut max_interval_error: f64 = 0.0;
    let mut previous_end: Option<f64> = None;

    for (position, interval) in run.intervals.iter().enumerate() {
        if interval.index != position {
            issues.push(format!(
                "interval index {} must equal zero-based position {position}",
                interval.index
            ));
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
                    issues.push(format!(
                        "interval {} is discontinuous: starts at {}, previous interval ends at {end}",
                        interval.index, interval.start_s
                    ));
                }
            } else if interval.start_s.abs() > time_tolerance {
                issues.push(format!(
                    "first interval must start at mission time zero, got {}",
                    interval.start_s
                ));
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
                issues.push(format!(
                    "interval {} {name} must be finite and non-negative",
                    interval.index
                ));
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
                issues.push(format!(
                    "interval {} standard uncertainty {name} must be finite and non-negative",
                    interval.index
                ));
            }
        }
        if let Some(ids) = &interval.source_artifact_ids {
            for id in ids {
                if !artifact_ids.contains(id.as_str()) {
                    issues.push(format!(
                        "interval {} references unknown source artifact {id:?}",
                        interval.index
                    ));
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
                issues.push(format!(
                    "interval {} relative energy error is non-finite",
                    interval.index
                ));
            } else {
                max_interval_error = max_interval_error.max(relative_error);
                if relative_error > balance_tolerance {
                    issues.push(format!(
                        "interval {} first-law error {:.6e} exceeds preregistered tolerance {:.6e}",
                        interval.index, relative_error, balance_tolerance
                    ));
                }
            }
        }

        total_input += flows.external_input;
        total_useful += flows.net_useful_output;
        total_auxiliary += flows.auxiliary;
        total_loss += flows.rejected_or_loss;
        total_storage_change += flows.stored_energy_change;
        if [total_input, total_useful, total_auxiliary, total_loss, total_storage_change]
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
            issues.push(format!(
                "last interval ends at {}, mission duration is {}",
                last.end_s, run.mission.expected_duration_s
            ));
        }
    }

    // Match the crate's cycle-level convention: storage charged and then
    // discharged within the same cycle is not counted twice as external input.
    let total_throughput = total_input + (-total_storage_change).max(0.0);
    let aggregate_residual =
        total_input - total_useful - total_auxiliary - total_loss - total_storage_change;
    if !aggregate_residual.is_finite() || !total_throughput.is_finite() || total_throughput <= 0.0 {
        issues.push("aggregate energy balance is non-finite or has no throughput".into());
    } else {
        let aggregate_relative_error = aggregate_residual.abs() / total_throughput;
        if !aggregate_relative_error.is_finite() || aggregate_relative_error > balance_tolerance {
            issues.push(format!(
                "aggregate first-law error {:.6e} exceeds preregistered tolerance {:.6e}",
                aggregate_relative_error, balance_tolerance
            ));
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
            (
                "net_stored_energy_change_j",
                metrics.net_stored_energy_change_j,
                total_storage_change,
            ),
        ] {
            if let Some(cached) = cached {
                if !cached.is_finite()
                    || !approximately_equal(cached, recomputed, metric_tolerance)
                {
                    issues.push(format!("cached metric {name} does not match recomputed total"));
                }
            }
        }
        match (metrics.net_cycle_efficiency, efficiency) {
            (Some(cached), Some(recomputed))
                if approximately_equal(cached, recomputed, metric_tolerance) => {}
            (Some(_), _) => issues.push(
                "cached net_cycle_efficiency is present but cannot be independently confirmed"
                    .into(),
            ),
            (None, Some(_)) => issues.push(
                "cached net_cycle_efficiency is missing despite charge-sustaining totals".into(),
            ),
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

/// Compare a candidate against its registered baseline using first-order
/// uncertainty propagation under an explicit *independent-input* assumption.
///
/// The comparison is intentionally conservative: it returns Indeterminate when
/// either record is a synthetic/not-for-decision fixture, cycle efficiency is
/// unavailable, or the manifest does not declare independent uncertainty inputs.
/// That declaration assumes the energy-channel and interval standard uncertainties
/// are mutually independent within each run and between candidate and baseline.
/// If common sensors, calibration, ambient conditions, or model parameters induce
/// correlation, provide a covariance/Monte Carlo model in a future implementation;
/// do not use this independent-input result.
///
/// This function compares two already-integrated cycle records; it is not a
/// physical validation and does not establish real-world superiority.
pub fn compare_candidates(
    candidate: &BenchmarkRun,
    baseline: &BenchmarkRun,
) -> Result<EfficiencyComparison, ValidationError> {
    let candidate_totals = validate_run(candidate)?;
    let baseline_totals = validate_run(baseline)?;
    let mut issues = Vec::new();

    if candidate.candidate.candidate_id == baseline.candidate.candidate_id {
        issues.push("candidate and baseline must have different candidate IDs".into());
    }
    if candidate.benchmark_id != baseline.benchmark_id {
        issues.push("candidate and baseline benchmark_id differ".into());
    }
    if candidate.mission.mission_id != baseline.mission.mission_id
        || candidate.mission.description != baseline.mission.description
        || candidate.mission.platform != baseline.mission.platform
        || candidate.mission.duty_cycle_id != baseline.mission.duty_cycle_id
        || candidate.mission.profile_source_uri != baseline.mission.profile_source_uri
        || candidate.mission.required_phases != baseline.mission.required_phases
        || !time_equal_within_tolerance(
            candidate.mission.expected_duration_s,
            baseline.mission.expected_duration_s,
            candidate.evaluation_policy.time_continuity_tolerance_s,
        )
        || candidate.mission.ambient_temperature_k != baseline.mission.ambient_temperature_k
        || candidate.mission.ambient_pressure_pa != baseline.mission.ambient_pressure_pa
    {
        issues.push("candidate and baseline missions or operating conditions differ".into());
    }
    if candidate.intervals.len() != baseline.intervals.len()
        || candidate.intervals.iter().zip(&baseline.intervals).any(|(left, right)| {
            left.index != right.index
                || left.phase != right.phase
                || !time_equal_within_tolerance(
                    left.start_s,
                    right.start_s,
                    candidate.evaluation_policy.time_continuity_tolerance_s,
                )
                || !time_equal_within_tolerance(
                    left.end_s,
                    right.end_s,
                    candidate.evaluation_policy.time_continuity_tolerance_s,
                )
        })
    {
        issues.push("candidate and baseline interval schedules differ".into());
    }
    if candidate.solver.method != baseline.solver.method
        || candidate.solver.version != baseline.solver.version
        || candidate.solver.code_revision != baseline.solver.code_revision
        || candidate.solver.model_fidelity != baseline.solver.model_fidelity
        || candidate.solver.random_seed != baseline.solver.random_seed
        || candidate.solver.calibration_dataset_ids != baseline.solver.calibration_dataset_ids
    {
        issues.push(
            "candidate and baseline solver method/version/revision/fidelity or calibration data differ"
                .into(),
        );
    }

    let cb = &candidate.boundary;
    let bb = &baseline.boundary;
    if cb.boundary_id != bb.boundary_id
        || cb.input_definition != bb.input_definition
        || cb.useful_output_definition != bb.useful_output_definition
        || cb.auxiliary_treatment != bb.auxiliary_treatment
        || cb.storage_treatment != bb.storage_treatment
        || cb.upstream_scope != bb.upstream_scope
        || cb.custom_scope_definition != bb.custom_scope_definition
        || cb.energy_unit != bb.energy_unit
        || cb.power_unit != bb.power_unit
    {
        issues.push("candidate and baseline system boundaries differ".into());
    }

    let cp = &candidate.evaluation_policy;
    let bp = &baseline.evaluation_policy;
    if cp.policy_id != bp.policy_id
        || cp.preregistered_at != bp.preregistered_at
        || cp.search_started_at != bp.search_started_at
        || cp.relative_balance_tolerance != bp.relative_balance_tolerance
        || cp.time_continuity_tolerance_s != bp.time_continuity_tolerance_s
        || cp.minimum_superiority_margin != bp.minimum_superiority_margin
        || cp.maximum_unmodeled_differential_bias_efficiency
            != bp.maximum_unmodeled_differential_bias_efficiency
        || cp.differential_bias_bound_basis != bp.differential_bias_bound_basis
        || cp.differential_bias_calibration_artifact_id
            != bp.differential_bias_calibration_artifact_id
        || cp.differential_bias_calibration_artifact_sha256
            != bp.differential_bias_calibration_artifact_sha256
        || cp.selection_rule != bp.selection_rule
        || cp.baseline_id != bp.baseline_id
        || cp.comparison_metric_id != bp.comparison_metric_id
    {
        issues.push("candidate and baseline evaluation policies differ".into());
    }
    if cp.baseline_id != baseline.candidate.candidate_id {
        issues.push(
            "candidate policy baseline_id does not identify the supplied baseline candidate".into(),
        );
    }
    if candidate.uncertainty.coverage_factor != baseline.uncertainty.coverage_factor {
        issues.push("candidate and baseline uncertainty coverage factors differ".into());
    }
    if candidate.uncertainty.quantity_type != baseline.uncertainty.quantity_type {
        issues.push("candidate and baseline uncertainty quantity types differ".into());
    }

    if !issues.is_empty() {
        return Err(ValidationError { issues });
    }

    let coverage_factor = candidate.uncertainty.coverage_factor;
    let required_margin = cp.minimum_superiority_margin;
    let mut report = EfficiencyComparison {
        decision: ComparisonDecision::Indeterminate,
        evidence_class: candidate.evidence_class.clone(),
        candidate_efficiency: candidate_totals.net_cycle_efficiency,
        baseline_efficiency: baseline_totals.net_cycle_efficiency,
        candidate_standard_uncertainty: None,
        baseline_standard_uncertainty: None,
        estimated_efficiency_difference: None,
        expanded_difference_uncertainty: None,
        conservative_lower_bound: None,
        required_superiority_margin: required_margin,
        maximum_unmodeled_differential_bias_efficiency: cp
            .maximum_unmodeled_differential_bias_efficiency,
        coverage_factor,
        reason: None,
    };

    if candidate.not_for_decision
        || baseline.not_for_decision
        || candidate.evidence_class == "synthetic_fixture"
        || baseline.evidence_class == "synthetic_fixture"
    {
        report.reason = Some(
            "synthetic or not-for-decision records cannot support a comparison decision".into(),
        );
        return Ok(report);
    }
    if candidate.evidence_class != baseline.evidence_class {
        return Err(ValidationError {
            issues: vec!["candidate and baseline evidence classes differ".into()],
        });
    }
    if candidate.data_provenance.iter().any(|a| a.kind == "synthetic_fixture")
        || baseline.data_provenance.iter().any(|a| a.kind == "synthetic_fixture")
    {
        report.reason = Some("synthetic provenance cannot support a comparison decision".into());
        return Ok(report);
    }
    if candidate.uncertainty.correlation_model != "independent"
        || baseline.uncertainty.correlation_model != "independent"
    {
        report.reason = Some(
            "uncertainty inputs are correlated or unknown; independent approximation is invalid"
                .into()
        );
        return Ok(report);
    }
    let (Some(candidate_efficiency), Some(baseline_efficiency)) = (
        candidate_totals.net_cycle_efficiency,
        baseline_totals.net_cycle_efficiency,
    ) else {
        report.reason = Some(
            "one or both duty cycles are not charge-sustaining; cycle efficiency is unavailable"
                .into(),
        );
        return Ok(report);
    };

    let candidate_u = efficiency_standard_uncertainty(
        candidate,
        candidate_totals.total_external_input_j,
        candidate_totals.total_net_useful_output_j,
    )?;
    let baseline_u = efficiency_standard_uncertainty(
        baseline,
        baseline_totals.total_external_input_j,
        baseline_totals.total_net_useful_output_j,
    )?;
    let expanded_delta_u = coverage_factor * candidate_u.hypot(baseline_u);
    let difference = candidate_efficiency - baseline_efficiency;
    // Treat unmodeled differential bias as a worst-case absolute bound, not
    // as a random standard uncertainty. Subtract it linearly and conservatively.
    let lower_bound = difference
        - expanded_delta_u
        - cp.maximum_unmodeled_differential_bias_efficiency;
    if !expanded_delta_u.is_finite() || !difference.is_finite() || !lower_bound.is_finite() {
        return Err(ValidationError {
            issues: vec!["efficiency comparison uncertainty arithmetic overflowed".into()],
        });
    }

    report.candidate_standard_uncertainty = Some(candidate_u);
    report.baseline_standard_uncertainty = Some(baseline_u);
    report.estimated_efficiency_difference = Some(difference);
    report.expanded_difference_uncertainty = Some(expanded_delta_u);
    report.conservative_lower_bound = Some(lower_bound);
    report.decision = if lower_bound > required_margin {
        ComparisonDecision::LowerBoundClearsRegisteredMargin
    } else {
        ComparisonDecision::LowerBoundDoesNotClearMargin
    };
    report.reason = Some(
        "numeric result only; preregistration and physical validity remain unverified"
            .into()
    );
    Ok(report)
}

fn time_equal_within_tolerance(left: f64, right: f64, tolerance: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= tolerance
}

fn efficiency_standard_uncertainty(
    run: &BenchmarkRun,
    total_input_j: f64,
    total_useful_output_j: f64,
) -> Result<f64, ValidationError> {
    if !total_input_j.is_finite() || total_input_j <= 0.0 || !total_useful_output_j.is_finite() {
        return Err(ValidationError {
            issues: vec![
                "efficiency uncertainty requires positive input and finite output".into(),
            ],
        });
    }

    // RSS is valid here only under the explicit independence declaration.
    let mut input_uncertainty_j: f64 = 0.0;
    let mut output_uncertainty_j: f64 = 0.0;
    for interval in &run.intervals {
        input_uncertainty_j =
            input_uncertainty_j.hypot(interval.standard_uncertainty_j.external_input);
        output_uncertainty_j =
            output_uncertainty_j.hypot(interval.standard_uncertainty_j.net_useful_output);
    }
    let efficiency = total_useful_output_j / total_input_j;
    let input_term = output_uncertainty_j / total_input_j;
    // Algebraically equivalent to Q_out * u(Q_in) / Q_in^2, but avoids
    // squaring a large finite Q_in and overflowing the denominator.
    let output_term = efficiency * (input_uncertainty_j / total_input_j);
    let uncertainty = input_term.hypot(output_term);
    if !uncertainty.is_finite() {
        return Err(ValidationError {
            issues: vec!["efficiency standard uncertainty overflowed".into()],
        });
    }
    Ok(uncertainty)
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
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_git_revision(value: &str) -> bool {
    value.len() == 40
        && value.bytes().all(|byte| {
            byte.is_ascii_digit()
                || (b'a'..=b'f').contains(&byte)
                || (b'A'..=b'F').contains(&byte)
        })
}

fn approximately_equal(left: f64, right: f64, tolerance: f64) -> bool {
    left.is_finite() && right.is_finite()
        && (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!(
            "../../../../docs/research/examples/energy-conversion-benchmark.synthetic.json"
        );

    #[test]
    fn synthetic_fixture_closes_each_interval_and_aggregate() {
        let report = validate_json(FIXTURE.as_bytes()).expect("fixture should validate");
        assert_eq!(report.total_external_input_j, 112.0);
        assert!((report.total_net_useful_output_j - 35.7).abs() < 1e-12);
        assert!((report.total_auxiliary_j - 6.1).abs() < 1e-12);
        assert!((report.total_rejected_or_loss_j - 70.2).abs() < 1e-12);
        assert_eq!(report.net_stored_energy_change_j, 0.0);
        assert_eq!(report.total_energy_throughput_j, 112.0);
        assert!(report.aggregate_residual_j.abs() < 1e-12);
        assert_eq!(report.net_cycle_efficiency, Some(35.7 / 112.0));
    }

    #[test]
    fn non_closing_interval_is_rejected_even_if_totals_are_cached() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.intervals[1].energy_j.rejected_or_loss -= 1.0;
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("first-law error")));
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
        assert!(
            error
                .issues
                .iter()
                .any(|issue| issue.contains("required mission phase \"shutdown\" is missing"))
        );
    }


    #[test]
    fn policy_timestamp_must_precede_search_start() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.evaluation_policy.preregistered_at = "2026-10-15T02:00:00Z".into();
        run.evaluation_policy.search_started_at = "2026-10-15T01:00:00Z".into();
        let error = validate_run(&run).unwrap_err();
        assert!(
            error
                .issues
                .iter()
                .any(|issue| issue.contains("timestamped before search_started_at"))
        );
    }

    #[test]
    fn synthetic_record_cannot_be_marked_decision_eligible() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.not_for_decision = false;
        let error = validate_run(&run).unwrap_err();
        assert!(
            error
                .issues
                .iter()
                .any(|issue| issue.contains("synthetic fixtures must set not_for_decision=true"))
        );
    }

    #[test]
    fn cached_metrics_are_recomputed_not_trusted() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.computed_metrics.as_mut().unwrap().total_external_input_j = Some(113.0);
        let error = validate_run(&run).unwrap_err();
        assert!(
            error
                .issues
                .iter()
                .any(|issue| issue.contains("cached metric total_external_input_j"))
        );
    }

    #[test]
    fn missing_bias_calibration_artifact_reference_is_rejected() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.evaluation_policy.differential_bias_calibration_artifact_id =
            "missing-calibration-artifact".into();
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| {
            issue.contains("calibration artifact ID is not present in data_provenance")
        }));
    }

    #[test]
    fn bias_calibration_hash_must_match_its_provenance_entry() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.evaluation_policy.differential_bias_calibration_artifact_sha256 = "0".repeat(64);
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| {
            issue.contains("calibration artifact SHA-256 does not match its provenance entry")
        }));
    }

    #[test]
    fn candidate_and_baseline_calibration_references_must_match() {
        let (mut candidate, baseline) = comparison_runs();
        let alternate_id = "alternate-calibration-reference";
        let alternate_sha = "d".repeat(64);
        candidate.evaluation_policy.differential_bias_calibration_artifact_id =
            alternate_id.into();
        candidate.evaluation_policy.differential_bias_calibration_artifact_sha256 =
            alternate_sha.clone();
        candidate.data_provenance.push(ProvenanceArtifact {
            artifact_id: alternate_id.into(),
            kind: "synthetic_fixture".into(),
            sha256: Some(alternate_sha),
            retrieved_at: None,
        });

        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(error
            .issues
            .iter()
            .any(|issue| issue.contains("candidate and baseline evaluation policies differ")));
    }

    fn comparison_runs() -> (BenchmarkRun, BenchmarkRun) {
        let mut baseline: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        baseline.evidence_class = "simulated".into();
        baseline.not_for_decision = false;
        baseline.data_provenance[0].kind = "simulation_output".into();
        baseline.data_provenance[0].sha256 = Some("b".repeat(64));
        baseline.uncertainty.correlation_model = "independent".into();
        baseline.uncertainty.method = "test-only independent primitive uncertainties".into();
        baseline.candidate.candidate_id = "registered-baseline".into();
        baseline.evaluation_policy.baseline_id = baseline.candidate.candidate_id.clone();
        baseline.evaluation_policy.comparison_metric_id = "net_cycle_efficiency".into();
        baseline.evaluation_policy.minimum_superiority_margin = 0.02;
        baseline.evaluation_policy.maximum_unmodeled_differential_bias_efficiency = 0.0;
        baseline.evaluation_policy.differential_bias_bound_basis =
            "Test-only synthetic fixture; zero bias bound is not a physical assertion.".into();

        let mut candidate = baseline.clone();
        candidate.candidate.candidate_id = "candidate-better".into();
        candidate.data_provenance[0].sha256 = Some("a".repeat(64));
        candidate.intervals[1].energy_j.net_useful_output = 40.0;
        candidate.intervals[1].energy_j.rejected_or_loss = 54.0;
        let metrics = candidate.computed_metrics.as_mut().unwrap();
        metrics.total_net_useful_output_j = Some(40.7);
        metrics.net_cycle_efficiency = Some(40.7 / 112.0);
        (candidate, baseline)
    }

    #[test]
    fn comparison_only_supports_improvement_when_lower_bound_clears_margin() {
        let (candidate, baseline) = comparison_runs();
        let result = compare_candidates(&candidate, &baseline).unwrap();
        assert_eq!(result.decision, ComparisonDecision::LowerBoundClearsRegisteredMargin);
        assert!(result.estimated_efficiency_difference.unwrap() > 0.02);
        assert!(result.conservative_lower_bound.unwrap() > result.required_superiority_margin);
        assert!(result.expanded_difference_uncertainty.unwrap() > 0.0);
    }

    #[test]
    fn registered_differential_bias_bound_reduces_the_superiority_lower_bound() {
        let (candidate, mut baseline) = comparison_runs();
        let unadjusted = compare_candidates(&candidate, &baseline).unwrap();
        assert_eq!(
            unadjusted.decision,
            ComparisonDecision::LowerBoundClearsRegisteredMargin
        );

        baseline
            .evaluation_policy
            .maximum_unmodeled_differential_bias_efficiency = 0.025;
        baseline.evaluation_policy.differential_bias_bound_basis =
            "Synthetic unit test: known worst-case candidate-baseline bias bound.".into();
        let mut adjusted_candidate = candidate;
        adjusted_candidate
            .evaluation_policy
            .maximum_unmodeled_differential_bias_efficiency = 0.025;
        adjusted_candidate.evaluation_policy.differential_bias_bound_basis =
            baseline.evaluation_policy.differential_bias_bound_basis.clone();

        let adjusted = compare_candidates(&adjusted_candidate, &baseline).unwrap();
        assert!(
            adjusted.conservative_lower_bound.unwrap()
                < unadjusted.conservative_lower_bound.unwrap()
        );
        assert!(
            (unadjusted.conservative_lower_bound.unwrap()
                - adjusted.conservative_lower_bound.unwrap()
                - 0.025)
                .abs()
                < 1e-12
        );
        assert_eq!(
            adjusted.decision,
            ComparisonDecision::LowerBoundDoesNotClearMargin
        );
    }

    #[test]
    fn invalid_differential_bias_bound_is_rejected() {
        let (mut candidate, _) = comparison_runs();
        candidate
            .evaluation_policy
            .maximum_unmodeled_differential_bias_efficiency = 1.01;
        let error = validate_run(&candidate).unwrap_err();
        assert!(
            error.issues.iter().any(|issue| issue.contains(
                "maximum_unmodeled_differential_bias_efficiency must be finite and in [0, 1]"
            ))
        );
    }

    #[test]
    fn missing_bias_bound_basis_is_rejected() {
        let (mut candidate, _) = comparison_runs();
        candidate.evaluation_policy.differential_bias_bound_basis.clear();
        let error = validate_run(&candidate).unwrap_err();
        assert!(
            error.issues.iter().any(|issue| issue.contains("differential-bias-bound basis"))
        );
    }

    #[test]
    fn candidate_and_baseline_bias_bounds_must_match() {
        let (candidate, mut baseline) = comparison_runs();
        baseline
            .evaluation_policy
            .maximum_unmodeled_differential_bias_efficiency = 0.01;
        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("evaluation policies differ")));
    }

    #[test]
    fn candidate_and_baseline_bias_bound_bases_must_match() {
        let (candidate, mut baseline) = comparison_runs();
        baseline
            .evaluation_policy
            .differential_bias_bound_basis = "different registered basis".into();
        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("evaluation policies differ")));
    }

    #[test]
    fn large_uncertainty_does_not_turn_point_estimate_into_a_win() {
        let (mut candidate, baseline) = comparison_runs();
        candidate.intervals[1].standard_uncertainty_j.external_input = 25.0;
        candidate.intervals[1].standard_uncertainty_j.net_useful_output = 20.0;
        let result = compare_candidates(&candidate, &baseline).unwrap();
        assert_eq!(result.decision, ComparisonDecision::LowerBoundDoesNotClearMargin);
        assert!(result.conservative_lower_bound.unwrap() <= result.required_superiority_margin);
    }

    #[test]
    fn v1_rejects_an_unimplemented_comparison_metric() {
        let mut run: BenchmarkRun = serde_json::from_str(FIXTURE).unwrap();
        run.evaluation_policy.comparison_metric_id = "coefficient_of_performance".into();
        let error = validate_run(&run).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("only supports comparison_metric_id")));
    }

    #[test]
    fn v1_does_not_silently_compare_heat_pump_cop_as_cycle_efficiency() {
        let (mut candidate, baseline) = comparison_runs();
        candidate.evaluation_policy.comparison_metric_id =
            "coefficient_of_performance".into();
        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(
            error.issues.iter().any(|issue| issue.contains(
                "only supports comparison_metric_id=net_cycle_efficiency"
            ))
        );
    }

    #[test]
    fn candidate_and_baseline_solver_provenance_must_match() {
        let (candidate, mut baseline) = comparison_runs();
        baseline.solver.model_fidelity = "high_fidelity_simulation".into();
        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("solver method/version/revision/fidelity")));
    }

    #[test]
    fn unknown_or_correlated_inputs_are_indeterminate() {
        let (candidate, mut baseline) = comparison_runs();
        baseline.uncertainty.correlation_model = "unknown".into();
        let result = compare_candidates(&candidate, &baseline).unwrap();
        assert_eq!(result.decision, ComparisonDecision::Indeterminate);
        assert!(result.reason.unwrap().contains("independent standard uncertainties"));
    }

    #[test]
    fn candidates_with_different_interval_schedules_cannot_be_compared() {
        let (mut candidate, baseline) = comparison_runs();
        candidate.intervals[1].end_s = 3500.0;
        candidate.intervals[2].start_s = 3500.0;
        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("interval schedules differ")));
    }

    #[test]
    fn candidates_with_different_energy_boundaries_cannot_be_compared() {
        let (candidate, mut baseline) = comparison_runs();
        baseline.boundary.useful_output_definition =
            "shaft work, not electricity at system terminals".into();
        let error = compare_candidates(&candidate, &baseline).unwrap_err();
        assert!(error.issues.iter().any(|issue| issue.contains("system boundaries differ")));
    }

    #[test]
    fn synthetic_fixture_cannot_support_a_comparison_decision() {
        let (candidate, baseline) = comparison_runs();
        let result = compare_candidates(&candidate, &baseline).unwrap();
        assert_ne!(result.decision, ComparisonDecision::Indeterminate);
        let mut synthetic = baseline.clone();
        synthetic.evidence_class = "synthetic_fixture".into();
        synthetic.not_for_decision = true;
        synthetic.data_provenance[0].kind = "synthetic_fixture".into();
        synthetic.data_provenance[0].sha256 = None;
        let result = compare_candidates(&candidate, &synthetic).unwrap();
        assert_eq!(result.decision, ComparisonDecision::Indeterminate);
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


/// Deterministic Phase A experiment ranking; not an active-learning claim.
pub mod experiment_selection;

/// Oracle-backed known-answer benchmarks for experiment-selection strategies.
pub mod experiment_benchmark;

/// Synthetic oracle for false-superiority and false-negative candidate decisions.
pub mod candidate_decision_benchmark;
