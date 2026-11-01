//! Calibration-bound energy uncertainty and conservative pairwise comparison.
//!
//! Type A repeatability is estimated from independent inference-segment energy replicates.
//! Type B relative standard uncertainty is supplied by a pinned calibration/metrology budget.
//! A manufacturer accuracy bound is not silently converted into a standard uncertainty.
//! This module does not establish that a calibration certificate is authentic; its digest and
//! trust policy must be verified by the acquisition/provenance layer.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ppk2_trace_audit::{
    compute_ppk2_trace_audit_receipt_sha256, validate_ppk2_trace_audit_receipt,
    Ppk2TraceAuditReceipt, Ppk2SegmentMetrics,
};
use crate::multi_resource_budget::{
    remaining_resource_headroom, ResourceBudget, ResourceCostVector, ResourceHeadroom,
    ResourceQuantity, ResourceRegistry,
};

const MANIFEST_SCHEMA: &str = "symthaea.nas.metrology-uncertainty-manifest";
const RECEIPT_SCHEMA: &str = "symthaea.nas.energy-uncertainty-receipt";
const VERSION: &str = "0.1.0";
const MANIFEST_DOMAIN: &[u8] = b"symthaea.nas.metrology-uncertainty-manifest.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.energy-uncertainty-receipt.v0.1";
const REPLICATION_CONFIG_DOMAIN: &[u8] = b"symthaea.nas.replication-planning-config.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetrologyUncertaintyError {
    pub field: String,
    pub message: String,
}

impl MetrologyUncertaintyError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for MetrologyUncertaintyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for MetrologyUncertaintyError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetrologyUncertaintyManifest {
    pub schema: String,
    pub version: String,
    pub instrument_model: String,
    pub instrument_serial: String,
    pub current_range_id: String,
    pub calibration_certificate_sha256: String,
    /// Relative standard uncertainty as a fraction, not a percentage or manufacturer accuracy limit.
    pub type_b_relative_standard_uncertainty: f64,
    /// Explicitly describes how the Type B value was obtained.
    pub type_b_basis: String,
    /// RSS combination is permitted only when the protocol supports uncorrelated Type A and Type B terms.
    pub type_a_type_b_uncorrelated: bool,
    /// Coverage factor chosen from the declared uncertainty methodology.
    pub coverage_factor: f64,
    pub replicate_protocol_sha256: String,
    /// Must be true before the usual sample standard error is used as Type A uncertainty.
    pub replicates_independent: bool,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyUncertaintyReceipt {
    pub schema: String,
    pub version: String,
    pub trace_receipt_sha256: String,
    pub metrology_manifest_sha256: String,
    pub replicate_count: usize,
    pub mean_energy_uj: f64,
    pub sample_standard_deviation_uj: f64,
    pub type_a_standard_uncertainty_uj: f64,
    pub type_b_standard_uncertainty_uj: f64,
    pub combined_standard_uncertainty_uj: f64,
    pub coverage_factor: f64,
    pub expanded_uncertainty_uj: f64,
    pub lower_bound_uj: f64,
    pub upper_bound_uj: f64,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyComparisonDecision {
    CandidateStrictlyLower,
    CandidateStrictlyHigher,
    IndeterminateIntervalsOverlap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyComparison {
    pub candidate_receipt_sha256: String,
    pub reference_receipt_sha256: String,
    pub decision: EnergyComparisonDecision,
    pub candidate_lower_uj: f64,
    pub candidate_upper_uj: f64,
    pub reference_lower_uj: f64,
    pub reference_upper_uj: f64,
    pub difference_candidate_minus_reference_uj: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationPlanKind {
    CandidateOnly,
    ReferenceOnly,
    PairedCandidateAndReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationPlanningStatus {
    AlreadyResolved,
    MetrologyLimited,
    AdditionalReplicatesRequired,
    NoPlanWithinBudgetOrHorizon,
}

/// Frozen resource-vector inputs to replicate planning. No scalar exchange rate is used:
/// each plan is checked component-wise against the registered remaining headroom.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicationPlanningConfig {
    pub resource_registry: ResourceRegistry,
    pub resource_budget: ResourceBudget,
    /// Cost of one additional candidate-side replicate, including explicit zeros if applicable.
    pub candidate_replicate_cost: ResourceCostVector,
    /// Cost of one additional reference-side replicate, including explicit zeros if applicable.
    pub reference_replicate_cost: ResourceCostVector,
    pub maximum_additional_replicates_per_side: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdditionalReplicationPlan {
    pub kind: ReplicationPlanKind,
    pub additional_candidate_replicates: usize,
    pub additional_reference_replicates: usize,
    pub resource_cost: ResourceCostVector,
    pub projected_candidate_expanded_uncertainty_uj: f64,
    pub projected_reference_expanded_uncertainty_uj: f64,
    /// Positive only when the projected candidate/reference expanded intervals would be disjoint.
    pub projected_interval_gap_uj: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyReplicationPlanningResult {
    pub status: ReplicationPlanningStatus,
    pub current_decision: EnergyComparisonDecision,
    pub mean_energy_gap_uj: f64,
    pub irreducible_type_b_interval_floor_uj: f64,
    pub planning_config_sha256: String,
    pub resource_registry_sha256: String,
    pub resource_headroom: Vec<ResourceHeadroom>,
    /// Non-dominated plans in benefit (projected interval separation) × resource-vector space.
    pub plan_frontier: Vec<AdditionalReplicationPlan>,
}

fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub fn compute_metrology_manifest_sha256(
    manifest: &MetrologyUncertaintyManifest,
) -> Result<String, MetrologyUncertaintyError> {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        MetrologyUncertaintyError::new("manifest", format!("cannot serialize: {error}"))
    })?;
    Ok(digest(MANIFEST_DOMAIN, &bytes))
}

pub fn validate_metrology_manifest(
    manifest: &MetrologyUncertaintyManifest,
) -> Result<(), MetrologyUncertaintyError> {
    if manifest.schema != MANIFEST_SCHEMA || manifest.version != VERSION {
        return Err(MetrologyUncertaintyError::new("schema_version", "unsupported schema/version"));
    }
    for (field, value) in [
        ("instrument_model", manifest.instrument_model.as_str()),
        ("instrument_serial", manifest.instrument_serial.as_str()),
        ("current_range_id", manifest.current_range_id.as_str()),
        ("type_b_basis", manifest.type_b_basis.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(MetrologyUncertaintyError::new(field, "must be non-empty"));
        }
    }
    for (field, value) in [
        ("calibration_certificate_sha256", manifest.calibration_certificate_sha256.as_str()),
        ("replicate_protocol_sha256", manifest.replicate_protocol_sha256.as_str()),
        ("manifest_sha256", manifest.manifest_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(MetrologyUncertaintyError::new(field, "must be lowercase SHA-256 hex"));
        }
    }
    if !manifest.type_b_relative_standard_uncertainty.is_finite()
        || !(0.0..1.0).contains(&manifest.type_b_relative_standard_uncertainty)
    {
        return Err(MetrologyUncertaintyError::new(
            "type_b_relative_standard_uncertainty",
            "must be finite and in [0, 1)",
        ));
    }
    if !manifest.coverage_factor.is_finite() || manifest.coverage_factor <= 0.0 {
        return Err(MetrologyUncertaintyError::new(
            "coverage_factor",
            "must be finite and positive",
        ));
    }
    if !manifest.replicates_independent {
        return Err(MetrologyUncertaintyError::new(
            "replicates_independent",
            "Type A standard error is not issued unless the preregistered replicate design supports independence",
        ));
    }
    if !manifest.type_a_type_b_uncorrelated {
        return Err(MetrologyUncertaintyError::new(
            "type_a_type_b_uncorrelated",
            "RSS combination requires a justified uncorrelated Type A/Type B model; covariance is not implemented in v0.1",
        ));
    }
    if compute_metrology_manifest_sha256(manifest)? != manifest.manifest_sha256 {
        return Err(MetrologyUncertaintyError::new(
            "manifest_sha256",
            "does not match the canonical metrology manifest",
        ));
    }
    Ok(())
}

fn validate_trace_segment(segment: &Ppk2SegmentMetrics) -> Result<(), MetrologyUncertaintyError> {
    if segment.segment_id.trim().is_empty() {
        return Err(MetrologyUncertaintyError::new("trace.segment_id", "must be non-empty"));
    }
    for (field, value) in [
        ("energy_uj", segment.energy_uj),
        ("duration_ms", segment.duration_ms),
        ("time_weighted_mean_current_ua", segment.time_weighted_mean_current_ua),
        ("mean_power_uw", segment.mean_power_uw),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(MetrologyUncertaintyError::new(
                format!("trace.{}.{}", segment.segment_id, field),
                "must be finite and positive",
            ));
        }
    }
    Ok(())
}

fn receipt_sha256(
    receipt: &EnergyUncertaintyReceipt,
) -> Result<String, MetrologyUncertaintyError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        MetrologyUncertaintyError::new("receipt", format!("cannot serialize: {error}"))
    })?;
    Ok(digest(RECEIPT_DOMAIN, &bytes))
}

/// Compute Type A repeatability and Type B calibration uncertainty from the segment energies
/// in an already-created PPK2 trace receipt.
pub fn evaluate_ppk2_energy_uncertainty(
    trace: &Ppk2TraceAuditReceipt,
    manifest: &MetrologyUncertaintyManifest,
) -> Result<EnergyUncertaintyReceipt, MetrologyUncertaintyError> {
    validate_metrology_manifest(manifest)?;
    validate_ppk2_trace_audit_receipt(trace)
        .map_err(|error| MetrologyUncertaintyError::new("trace_receipt", error.to_string()))?;
    if !is_sha256(&trace.receipt_sha256) || !is_sha256(&trace.sample_csv_sha256) {
        return Err(MetrologyUncertaintyError::new(
            "trace_receipt_sha256",
            "trace receipt must contain valid source and receipt digests",
        ));
    }
    if trace.segments.len() < 2 {
        return Err(MetrologyUncertaintyError::new(
            "trace.segments",
            "at least two independent inference segments are required to estimate Type A repeatability",
        ));
    }
    if trace.segments.len() != trace.segment_count {
        return Err(MetrologyUncertaintyError::new(
            "trace.segment_count",
            "segment count does not match the segment inventory",
        ));
    }
    for segment in &trace.segments {
        validate_trace_segment(segment)?;
    }

    // Welford's stable one-pass sample mean and variance over inference-level energies.
    let mut count = 0usize;
    let mut mean = 0.0_f64;
    let mut sum_squared_deviation = 0.0_f64;
    for segment in &trace.segments {
        count += 1;
        let delta = segment.energy_uj - mean;
        mean += delta / count as f64;
        let delta_after = segment.energy_uj - mean;
        sum_squared_deviation += delta * delta_after;
    }
    let sample_variance = sum_squared_deviation / (count - 1) as f64;
    let sample_standard_deviation = sample_variance.sqrt();
    let type_a = sample_standard_deviation / (count as f64).sqrt();
    let type_b = mean.abs() * manifest.type_b_relative_standard_uncertainty;
    let combined = type_a.hypot(type_b);
    let expanded = manifest.coverage_factor * combined;

    if !mean.is_finite()
        || !sample_standard_deviation.is_finite()
        || !type_a.is_finite()
        || !type_b.is_finite()
        || !combined.is_finite()
        || !expanded.is_finite()
    {
        return Err(MetrologyUncertaintyError::new("uncertainty", "derived uncertainty is non-finite"));
    }

    let mut receipt = EnergyUncertaintyReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        trace_receipt_sha256: trace.receipt_sha256.clone(),
        metrology_manifest_sha256: manifest.manifest_sha256.clone(),
        replicate_count: count,
        mean_energy_uj: mean,
        sample_standard_deviation_uj: sample_standard_deviation,
        type_a_standard_uncertainty_uj: type_a,
        type_b_standard_uncertainty_uj: type_b,
        combined_standard_uncertainty_uj: combined,
        coverage_factor: manifest.coverage_factor,
        expanded_uncertainty_uj: expanded,
        lower_bound_uj: mean - expanded,
        upper_bound_uj: mean + expanded,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_sha256(&receipt)?;
    Ok(receipt)
}

pub fn validate_energy_uncertainty_receipt(
    receipt: &EnergyUncertaintyReceipt,
) -> Result<(), MetrologyUncertaintyError> {
    if receipt.schema != RECEIPT_SCHEMA || receipt.version != VERSION {
        return Err(MetrologyUncertaintyError::new("schema_version", "unsupported receipt schema/version"));
    }
    for (field, value) in [
        ("trace_receipt_sha256", receipt.trace_receipt_sha256.as_str()),
        ("metrology_manifest_sha256", receipt.metrology_manifest_sha256.as_str()),
        ("receipt_sha256", receipt.receipt_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(MetrologyUncertaintyError::new(field, "must be lowercase SHA-256 hex"));
        }
    }
    for (field, value) in [
        ("mean_energy_uj", receipt.mean_energy_uj),
        ("sample_standard_deviation_uj", receipt.sample_standard_deviation_uj),
        ("type_a_standard_uncertainty_uj", receipt.type_a_standard_uncertainty_uj),
        ("type_b_standard_uncertainty_uj", receipt.type_b_standard_uncertainty_uj),
        ("combined_standard_uncertainty_uj", receipt.combined_standard_uncertainty_uj),
        ("coverage_factor", receipt.coverage_factor),
        ("expanded_uncertainty_uj", receipt.expanded_uncertainty_uj),
        ("lower_bound_uj", receipt.lower_bound_uj),
        ("upper_bound_uj", receipt.upper_bound_uj),
    ] {
        if !value.is_finite() {
            return Err(MetrologyUncertaintyError::new(field, "must be finite"));
        }
    }
    if receipt.replicate_count < 2
        || receipt.mean_energy_uj <= 0.0
        || receipt.sample_standard_deviation_uj < 0.0
        || receipt.type_a_standard_uncertainty_uj < 0.0
        || receipt.type_b_standard_uncertainty_uj < 0.0
        || receipt.coverage_factor <= 0.0
    {
        return Err(MetrologyUncertaintyError::new("receipt", "replicate count and uncertainty values are outside valid bounds"));
    }
    let expected_type_a = receipt.sample_standard_deviation_uj / (receipt.replicate_count as f64).sqrt();
    let expected_combined = receipt.type_a_standard_uncertainty_uj.hypot(receipt.type_b_standard_uncertainty_uj);
    let expected_expanded = receipt.coverage_factor * expected_combined;
    let tolerance = 1e-10 * receipt.mean_energy_uj.abs().max(1.0);
    if (receipt.type_a_standard_uncertainty_uj - expected_type_a).abs() > tolerance
        || (receipt.combined_standard_uncertainty_uj - expected_combined).abs() > tolerance
        || (receipt.expanded_uncertainty_uj - expected_expanded).abs() > tolerance
        || (receipt.lower_bound_uj - (receipt.mean_energy_uj - expected_expanded)).abs() > tolerance
        || (receipt.upper_bound_uj - (receipt.mean_energy_uj + expected_expanded)).abs() > tolerance
    {
        return Err(MetrologyUncertaintyError::new("receipt.derived_fields", "uncertainty fields are inconsistent"));
    }
    if receipt_sha256(receipt)? != receipt.receipt_sha256 {
        return Err(MetrologyUncertaintyError::new("receipt_sha256", "does not match canonical receipt"));
    }
    Ok(())
}

const MAX_REPLICATION_SEARCH: usize = 1_000_000;

fn projected_expanded_uncertainty(
    receipt: &EnergyUncertaintyReceipt,
    total_replicates: usize,
) -> f64 {
    let type_a = receipt.sample_standard_deviation_uj / (total_replicates as f64).sqrt();
    receipt.coverage_factor
        * type_a.hypot(receipt.type_b_standard_uncertainty_uj)
}

fn validate_replicate_cost_template(
    registry: &ResourceRegistry,
    vector: &ResourceCostVector,
    field: &str,
) -> Result<(), MetrologyUncertaintyError> {
    if vector.action_id.trim().is_empty() {
        return Err(MetrologyUncertaintyError::new(
            format!("{field}.action_id"),
            "must be non-empty",
        ));
    }
    if vector.resource_registry_sha256 != registry.registry_sha256 {
        return Err(MetrologyUncertaintyError::new(
            format!("{field}.resource_registry_sha256"),
            "cost template is bound to a different resource registry",
        ));
    }
    if vector.resources.len() != registry.resources.len() {
        return Err(MetrologyUncertaintyError::new(
            format!("{field}.resources"),
            "cost template must explicitly contain every registered resource",
        ));
    }
    for (index, (quantity, definition)) in vector.resources.iter().zip(&registry.resources).enumerate() {
        if quantity.resource_id != definition.resource_id
            || quantity.unit != definition.unit
            || quantity.cost_model_sha256 != definition.cost_model_sha256
        {
            return Err(MetrologyUncertaintyError::new(
                format!("{field}.resources[{index}]"),
                "resource identity, unit or cost model does not match the frozen registry",
            ));
        }
    }
    if !vector.resources.iter().any(|quantity| quantity.units > 0) {
        return Err(MetrologyUncertaintyError::new(
            format!("{field}.resources"),
            "replicate cost must consume at least one declared resource",
        ));
    }
    Ok(())
}

fn build_replication_cost_vector(
    registry: &ResourceRegistry,
    candidate_template: &ResourceCostVector,
    reference_template: &ResourceCostVector,
    candidate_replicates: usize,
    reference_replicates: usize,
    kind: ReplicationPlanKind,
) -> Result<ResourceCostVector, MetrologyUncertaintyError> {
    let candidate_count = u64::try_from(candidate_replicates).map_err(|_| {
        MetrologyUncertaintyError::new("candidate_replicates", "does not fit in u64")
    })?;
    let reference_count = u64::try_from(reference_replicates).map_err(|_| {
        MetrologyUncertaintyError::new("reference_replicates", "does not fit in u64")
    })?;
    let action_id = match kind {
        ReplicationPlanKind::CandidateOnly => "replicate_candidate_only",
        ReplicationPlanKind::ReferenceOnly => "replicate_reference_only",
        ReplicationPlanKind::PairedCandidateAndReference => "replicate_paired_candidate_reference",
    };

    let mut resources = Vec::with_capacity(registry.resources.len());
    for ((definition, candidate), reference) in registry.resources.iter()
        .zip(&candidate_template.resources)
        .zip(&reference_template.resources)
    {
        let candidate_units = candidate.units.checked_mul(candidate_count).ok_or_else(|| {
            MetrologyUncertaintyError::new("resource_cost", "candidate resource multiplication overflow")
        })?;
        let reference_units = reference.units.checked_mul(reference_count).ok_or_else(|| {
            MetrologyUncertaintyError::new("resource_cost", "reference resource multiplication overflow")
        })?;
        let units = candidate_units.checked_add(reference_units).ok_or_else(|| {
            MetrologyUncertaintyError::new("resource_cost", "combined resource addition overflow")
        })?;
        resources.push(ResourceQuantity {
            resource_id: definition.resource_id.clone(),
            unit: definition.unit.clone(),
            units,
            cost_model_sha256: definition.cost_model_sha256.clone(),
        });
    }

    if !resources.iter().any(|quantity| quantity.units > 0) {
        return Err(MetrologyUncertaintyError::new(
            "resource_cost",
            "replication plan must consume at least one resource",
        ));
    }

    Ok(ResourceCostVector {
        action_id: action_id.to_owned(),
        resource_registry_sha256: registry.registry_sha256.clone(),
        resources,
    })
}

fn resource_cost_fits(
    cost: &ResourceCostVector,
    headroom: &[ResourceHeadroom],
) -> bool {
    cost.resources.len() == headroom.len()
        && cost.resources.iter().zip(headroom).all(|(quantity, available)| {
            quantity.resource_id == available.resource_id
                && quantity.unit == available.unit
                && quantity.units <= available.available_units
        })
}

fn replication_plan_dominates(
    left: &AdditionalReplicationPlan,
    right: &AdditionalReplicationPlan,
) -> bool {
    if left.resource_cost.resources.len() != right.resource_cost.resources.len() {
        return false;
    }
    let all_resources_no_worse = left.resource_cost.resources.iter()
        .zip(&right.resource_cost.resources)
        .all(|(l, r)| l.units <= r.units);
    let benefit_no_worse = left.projected_interval_gap_uj >= right.projected_interval_gap_uj;
    let strictly_better = left.resource_cost.resources.iter()
        .zip(&right.resource_cost.resources)
        .any(|(l, r)| l.units < r.units)
        || left.projected_interval_gap_uj > right.projected_interval_gap_uj;
    all_resources_no_worse && benefit_no_worse && strictly_better
}

fn replication_plan_frontier(
    plans: Vec<AdditionalReplicationPlan>,
) -> Vec<AdditionalReplicationPlan> {
    let mut frontier = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        let dominated = plans.iter().enumerate().any(|(other_index, other)| {
            other_index != index && replication_plan_dominates(other, plan)
        });
        if !dominated {
            frontier.push(plan.clone());
        }
    }
    frontier.sort_by(|left, right| {
        left.kind.cmp(&right.kind)
            .then_with(|| left.additional_candidate_replicates.cmp(&right.additional_candidate_replicates))
            .then_with(|| left.additional_reference_replicates.cmp(&right.additional_reference_replicates))
    });
    frontier
}

/// Plan additional replicates under the existing typed vector-resource budget.
///
/// Type A is projected as s/sqrt(n); Type B is held fixed. Candidate-only, reference-only,
/// and paired strategies are evaluated independently. The return value is a non-dominated
/// frontier in projected interval separation versus the complete resource-cost vector; no
/// scalar exchange rate is invented between instrument time, target occupancy, money, energy,
/// storage or any other registered resource.
pub fn plan_additional_energy_replicates(
    candidate: &EnergyUncertaintyReceipt,
    reference: &EnergyUncertaintyReceipt,
    config: &ReplicationPlanningConfig,
) -> Result<EnergyReplicationPlanningResult, MetrologyUncertaintyError> {
    validate_energy_uncertainty_receipt(candidate)
        .map_err(|error| MetrologyUncertaintyError::new("candidate", error.to_string()))?;
    validate_energy_uncertainty_receipt(reference)
        .map_err(|error| MetrologyUncertaintyError::new("reference", error.to_string()))?;

    if candidate.receipt_sha256 == reference.receipt_sha256 {
        return Err(MetrologyUncertaintyError::new(
            "comparison_scope",
            "candidate and reference must not reuse the exact same uncertainty receipt",
        ));
    }
    if config.maximum_additional_replicates_per_side > MAX_REPLICATION_SEARCH {
        return Err(MetrologyUncertaintyError::new(
            "maximum_additional_replicates_per_side",
            format!("must not exceed {MAX_REPLICATION_SEARCH}"),
        ));
    }
    validate_replicate_cost_template(
        &config.resource_registry,
        &config.candidate_replicate_cost,
        "candidate_replicate_cost",
    )?;
    validate_replicate_cost_template(
        &config.resource_registry,
        &config.reference_replicate_cost,
        "reference_replicate_cost",
    )?;
    let headroom = remaining_resource_headroom(&config.resource_registry, &config.resource_budget)
        .map_err(|error| MetrologyUncertaintyError::new("resource_budget", error.to_string()))?;

    let config_bytes = serde_json::to_vec(config).map_err(|error| {
        MetrologyUncertaintyError::new("planning_config", format!("cannot serialize: {error}"))
    })?;
    let planning_config_sha256 = digest(REPLICATION_CONFIG_DOMAIN, &config_bytes);

    let current = compare_energy_uncertainty(candidate, reference)?;
    let gap = (candidate.mean_energy_uj - reference.mean_energy_uj).abs();
    let type_b_floor =
        candidate.coverage_factor * candidate.type_b_standard_uncertainty_uj
            + reference.coverage_factor * reference.type_b_standard_uncertainty_uj;

    let result = |status, plan_frontier| EnergyReplicationPlanningResult {
        status,
        current_decision: current.decision,
        mean_energy_gap_uj: gap,
        irreducible_type_b_interval_floor_uj: type_b_floor,
        planning_config_sha256: planning_config_sha256.clone(),
        resource_registry_sha256: config.resource_registry.registry_sha256.clone(),
        resource_headroom: headroom.clone(),
        plan_frontier,
    };

    if current.decision != EnergyComparisonDecision::IndeterminateIntervalsOverlap {
        return Ok(result(ReplicationPlanningStatus::AlreadyResolved, Vec::new()));
    }
    if gap <= type_b_floor {
        return Ok(result(ReplicationPlanningStatus::MetrologyLimited, Vec::new()));
    }

    let mut plans = Vec::<AdditionalReplicationPlan>::new();
    let limit = config.maximum_additional_replicates_per_side;

    // Candidate-only repetitions.
    for additional in 1..=limit {
        let total = candidate.replicate_count.checked_add(additional).ok_or_else(|| {
            MetrologyUncertaintyError::new("candidate.replicate_count", "replicate count overflow")
        })?;
        let candidate_u = projected_expanded_uncertainty(candidate, total);
        let reference_u = reference.expanded_uncertainty_uj;
        if gap > candidate_u + reference_u {
            let cost = build_replication_cost_vector(
                &config.resource_registry,
                &config.candidate_replicate_cost,
                &config.reference_replicate_cost,
                additional,
                0,
                ReplicationPlanKind::CandidateOnly,
            )?;
            if resource_cost_fits(&cost, &headroom) {
                plans.push(AdditionalReplicationPlan {
                    kind: ReplicationPlanKind::CandidateOnly,
                    additional_candidate_replicates: additional,
                    additional_reference_replicates: 0,
                    resource_cost: cost,
                    projected_candidate_expanded_uncertainty_uj: candidate_u,
                    projected_reference_expanded_uncertainty_uj: reference_u,
                    projected_interval_gap_uj: gap - candidate_u - reference_u,
                });
            }
            break;
        }
    }

    // Reference-only repetitions.
    for additional in 1..=limit {
        let total = reference.replicate_count.checked_add(additional).ok_or_else(|| {
            MetrologyUncertaintyError::new("reference.replicate_count", "replicate count overflow")
        })?;
        let candidate_u = candidate.expanded_uncertainty_uj;
        let reference_u = projected_expanded_uncertainty(reference, total);
        if gap > candidate_u + reference_u {
            let cost = build_replication_cost_vector(
                &config.resource_registry,
                &config.candidate_replicate_cost,
                &config.reference_replicate_cost,
                0,
                additional,
                ReplicationPlanKind::ReferenceOnly,
            )?;
            if resource_cost_fits(&cost, &headroom) {
                plans.push(AdditionalReplicationPlan {
                    kind: ReplicationPlanKind::ReferenceOnly,
                    additional_candidate_replicates: 0,
                    additional_reference_replicates: additional,
                    resource_cost: cost,
                    projected_candidate_expanded_uncertainty_uj: candidate_u,
                    projected_reference_expanded_uncertainty_uj: reference_u,
                    projected_interval_gap_uj: gap - candidate_u - reference_u,
                });
            }
            break;
        }
    }

    // Paired blocked repetitions.
    for additional in 1..=limit {
        let candidate_total = candidate.replicate_count.checked_add(additional).ok_or_else(|| {
            MetrologyUncertaintyError::new("candidate.replicate_count", "replicate count overflow")
        })?;
        let reference_total = reference.replicate_count.checked_add(additional).ok_or_else(|| {
            MetrologyUncertaintyError::new("reference.replicate_count", "replicate count overflow")
        })?;
        let candidate_u = projected_expanded_uncertainty(candidate, candidate_total);
        let reference_u = projected_expanded_uncertainty(reference, reference_total);
        if gap > candidate_u + reference_u {
            let cost = build_replication_cost_vector(
                &config.resource_registry,
                &config.candidate_replicate_cost,
                &config.reference_replicate_cost,
                additional,
                additional,
                ReplicationPlanKind::PairedCandidateAndReference,
            )?;
            if resource_cost_fits(&cost, &headroom) {
                plans.push(AdditionalReplicationPlan {
                    kind: ReplicationPlanKind::PairedCandidateAndReference,
                    additional_candidate_replicates: additional,
                    additional_reference_replicates: additional,
                    resource_cost: cost,
                    projected_candidate_expanded_uncertainty_uj: candidate_u,
                    projected_reference_expanded_uncertainty_uj: reference_u,
                    projected_interval_gap_uj: gap - candidate_u - reference_u,
                });
            }
            break;
        }
    }

    let frontier = replication_plan_frontier(plans);
    if frontier.is_empty() {
        Ok(result(ReplicationPlanningStatus::NoPlanWithinBudgetOrHorizon, frontier))
    } else {
        Ok(result(ReplicationPlanningStatus::AdditionalReplicatesRequired, frontier))
    }
}

/// Only claim strict superiority when the expanded intervals do not overlap.
/// Shared systematic errors are not assumed to cancel; paired-covariance modeling is a
/// separate future extension and must be justified by the acquisition design.
pub fn compare_energy_uncertainty(
    candidate: &EnergyUncertaintyReceipt,
    reference: &EnergyUncertaintyReceipt,
) -> Result<EnergyComparison, MetrologyUncertaintyError> {
    validate_energy_uncertainty_receipt(candidate)
        .map_err(|error| MetrologyUncertaintyError::new("candidate", error.to_string()))?;
    validate_energy_uncertainty_receipt(reference)
        .map_err(|error| MetrologyUncertaintyError::new("reference", error.to_string()))?;
    let decision = if candidate.upper_bound_uj < reference.lower_bound_uj {
        EnergyComparisonDecision::CandidateStrictlyLower
    } else if candidate.lower_bound_uj > reference.upper_bound_uj {
        EnergyComparisonDecision::CandidateStrictlyHigher
    } else {
        EnergyComparisonDecision::IndeterminateIntervalsOverlap
    };
    Ok(EnergyComparison {
        candidate_receipt_sha256: candidate.receipt_sha256.clone(),
        reference_receipt_sha256: reference.receipt_sha256.clone(),
        decision,
        candidate_lower_uj: candidate.lower_bound_uj,
        candidate_upper_uj: candidate.upper_bound_uj,
        reference_lower_uj: reference.lower_bound_uj,
        reference_upper_uj: reference.upper_bound_uj,
        difference_candidate_minus_reference_uj: candidate.mean_energy_uj - reference.mean_energy_uj,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_resource_budget::{
        compute_resource_registry_sha256, ResourceBudgetLimit, ResourceClass,
        ResourceDefinition,
    };
    use crate::ppk2_trace_audit::{Ppk2SegmentMetrics, Ppk2TraceAuditReceipt};

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn manifest() -> MetrologyUncertaintyManifest {
        let mut value = MetrologyUncertaintyManifest {
            schema: MANIFEST_SCHEMA.into(),
            version: VERSION.into(),
            instrument_model: "Nordic PPK2".into(),
            instrument_serial: "ppk2-fixture-01".into(),
            current_range_id: "R3".into(),
            calibration_certificate_sha256: sha('a'),
            type_b_relative_standard_uncertainty: 0.02,
            type_b_basis: "fixture calibration certificate; relative standard uncertainty".into(),
            type_a_type_b_uncorrelated: true,
            coverage_factor: 2.0,
            replicate_protocol_sha256: sha('b'),
            replicates_independent: true,
            manifest_sha256: String::new(),
        };
        value.manifest_sha256 = compute_metrology_manifest_sha256(&value).unwrap();
        value
    }

    fn trace(energies: &[f64]) -> Ppk2TraceAuditReceipt {
        let segments: Vec<Ppk2SegmentMetrics> = energies.iter().enumerate().map(|(i, energy)| {
            let mean_power_uw = energy * 1_000.0;
            let mean_current_ua = mean_power_uw / 3.3;
            Ppk2SegmentMetrics {
                segment_id: i.to_string(),
                sample_count: 100,
                first_time_ms: 0.0,
                last_time_ms: 1.0,
                duration_ms: 1.0,
                minimum_current_ua: 0.0,
                maximum_current_ua: mean_current_ua * 2.0,
                time_weighted_mean_current_ua: mean_current_ua,
                mean_power_uw,
                energy_uj: *energy,
                median_interval_ms: 0.01,
                maximum_interval_ms: 0.01,
                median_sample_rate_hz: 100_000.0,
                maximum_gap_ratio: 1.0,
            }
        }).collect();
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let mean_power = mean_energy * 1_000.0;
        let mut receipt = Ppk2TraceAuditReceipt {
            schema: "symthaea.nas.ppk2-trace-audit".into(),
            version: "0.1.0".into(),
            sample_csv_sha256: sha('c'),
            sample_csv_bytes: 1000,
            voltage_v: 3.3,
            expected_sample_rate_hz: 100_000.0,
            sample_rate_tolerance_fraction: 0.2,
            maximum_gap_ratio: 2.5,
            segment_count: segments.len(),
            total_sample_count: energies.len() * 100,
            mean_segment_current_ua: mean_power / 3.3,
            mean_segment_power_uw: mean_power,
            mean_segment_energy_uj: mean_energy,
            minimum_segment_energy_uj: energies.iter().copied().fold(f64::INFINITY, f64::min),
            maximum_segment_energy_uj: energies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            mean_segment_duration_ms: 1.0,
            median_interval_ms: 0.01,
            median_sample_rate_hz: 100_000.0,
            maximum_observed_gap_ratio: 1.0,
            segments,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = compute_ppk2_trace_audit_receipt_sha256(&receipt).unwrap();
        receipt
    }

    #[test]
    fn combines_type_a_and_calibration_type_b_uncertainty() {
        let t = trace(&[10.0, 11.0, 9.0, 10.0]);
        let r = evaluate_ppk2_energy_uncertainty(&t, &manifest()).unwrap();
        assert_eq!(r.replicate_count, 4);
        assert!((r.mean_energy_uj - 10.0).abs() < 1e-12);
        assert!((r.type_a_standard_uncertainty_uj - (2.0_f64 / 3.0).sqrt() / 2.0).abs() < 1e-12);
        assert!((r.type_b_standard_uncertainty_uj - 0.2).abs() < 1e-12);
        assert!(r.expanded_uncertainty_uj > r.type_a_standard_uncertainty_uj);
    }

    #[test]
    fn rejects_single_replicate_and_non_independent_design() {
        assert!(evaluate_ppk2_energy_uncertainty(&trace(&[10.0]), &manifest()).is_err());
        let mut m = manifest();
        m.replicates_independent = false;
        m.manifest_sha256 = compute_metrology_manifest_sha256(&m).unwrap();
        assert!(evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 11.0]), &m).is_err());
    }

    #[test]
    fn rejects_tampered_trace_receipt_before_uncertainty_analysis() {
        let mut t = trace(&[10.0, 11.0, 9.0]);
        t.mean_segment_energy_uj = 0.0;
        assert!(evaluate_ppk2_energy_uncertainty(&t, &manifest()).is_err());
    }

    #[test]
    fn rejects_unjustified_type_a_type_b_rss_combination() {
        let mut m = manifest();
        m.type_a_type_b_uncorrelated = false;
        m.manifest_sha256 = compute_metrology_manifest_sha256(&m).unwrap();
        assert!(validate_metrology_manifest(&m).is_err());
    }

    #[test]
    fn rejects_invalid_calibration_uncertainty_and_digest() {
        let mut m = manifest();
        m.type_b_relative_standard_uncertainty = 1.5;
        m.manifest_sha256 = compute_metrology_manifest_sha256(&m).unwrap();
        assert!(validate_metrology_manifest(&m).is_err());

        let mut m = manifest();
        m.calibration_certificate_sha256 = "not-a-digest".into();
        m.manifest_sha256 = compute_metrology_manifest_sha256(&m).unwrap();
        assert!(validate_metrology_manifest(&m).is_err());
    }

    #[test]
    fn conservative_comparison_refuses_overlapping_intervals() {
        let t_a = trace(&[10.0, 10.1, 9.9, 10.0]);
        let t_b = trace(&[9.9, 10.0, 9.8, 9.9]);
        let a = evaluate_ppk2_energy_uncertainty(&t_a, &manifest()).unwrap();
        let b = evaluate_ppk2_energy_uncertainty(&t_b, &manifest()).unwrap();
        let result = compare_energy_uncertainty(&b, &a).unwrap();
        assert_eq!(result.decision, EnergyComparisonDecision::IndeterminateIntervalsOverlap);
    }

    #[test]
    fn rejects_tampered_uncertainty_receipt_before_comparison() {
        let a = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 10.0, 10.0]), &manifest()).unwrap();
        let mut forged = evaluate_ppk2_energy_uncertainty(&trace(&[5.0, 5.0, 5.0]), &manifest()).unwrap();
        forged.lower_bound_uj = 0.0;
        assert!(validate_energy_uncertainty_receipt(&forged).is_err());
        assert!(compare_energy_uncertainty(&forged, &a).is_err());
    }

    #[test]
    fn uncertainty_receipts_roundtrip_and_reject_unknown_fields() {
        let receipt = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 11.0, 9.0]), &manifest()).unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: EnergyUncertaintyReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, receipt);
        let with_unknown = json.trim_end_matches('}').to_owned() + ",\"unexpected\":true}";
        assert!(serde_json::from_str::<EnergyUncertaintyReceipt>(&with_unknown).is_err());
    }

    #[test]
    fn rejects_rehashed_receipt_with_forged_type_a_term() {
        let mut receipt = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 11.0, 9.0]), &manifest()).unwrap();
        receipt.type_a_standard_uncertainty_uj = 0.0;
        receipt.receipt_sha256 = receipt_sha256(&receipt).unwrap();
        assert!(validate_energy_uncertainty_receipt(&receipt).is_err());
    }

    fn replication_config(
        candidate_costs: [u64; 2],
        reference_costs: [u64; 2],
        per_resource_budget: u64,
        maximum_additional_replicates: usize,
    ) -> ReplicationPlanningConfig {
        let mut registry = ResourceRegistry {
            schema: "symthaea.nas.resource-registry".into(),
            version: "0.1.0".into(),
            resources: vec![
                ResourceDefinition {
                    resource_id: "instrument_time".into(),
                    unit: "seconds".into(),
                    class: ResourceClass::InstrumentTime,
                    cost_model_sha256: sha('e'),
                },
                ResourceDefinition {
                    resource_id: "target_device_time".into(),
                    unit: "seconds".into(),
                    class: ResourceClass::TargetDeviceTime,
                    cost_model_sha256: sha('f'),
                },
            ],
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = compute_resource_registry_sha256(&registry).unwrap();

        let budget = ResourceBudget {
            resource_registry_sha256: registry.registry_sha256.clone(),
            limits: vec![
                ResourceBudgetLimit {
                    resource_id: "instrument_time".into(),
                    unit: "seconds".into(),
                    total_units: per_resource_budget,
                    already_spent_units: 0,
                    protected_reserve_units: 0,
                },
                ResourceBudgetLimit {
                    resource_id: "target_device_time".into(),
                    unit: "seconds".into(),
                    total_units: per_resource_budget,
                    already_spent_units: 0,
                    protected_reserve_units: 0,
                },
            ],
        };

        let candidate_replicate_cost = ResourceCostVector {
            action_id: "candidate_replicate".into(),
            resource_registry_sha256: registry.registry_sha256.clone(),
            resources: vec![
                ResourceQuantity {
                    resource_id: "instrument_time".into(),
                    unit: "seconds".into(),
                    units: candidate_costs[0],
                    cost_model_sha256: sha('e'),
                },
                ResourceQuantity {
                    resource_id: "target_device_time".into(),
                    unit: "seconds".into(),
                    units: candidate_costs[1],
                    cost_model_sha256: sha('f'),
                },
            ],
        };
        let reference_replicate_cost = ResourceCostVector {
            action_id: "reference_replicate".into(),
            resource_registry_sha256: registry.registry_sha256.clone(),
            resources: vec![
                ResourceQuantity {
                    resource_id: "instrument_time".into(),
                    unit: "seconds".into(),
                    units: reference_costs[0],
                    cost_model_sha256: sha('e'),
                },
                ResourceQuantity {
                    resource_id: "target_device_time".into(),
                    unit: "seconds".into(),
                    units: reference_costs[1],
                    cost_model_sha256: sha('f'),
                },
            ],
        };

        ReplicationPlanningConfig {
            resource_registry: registry,
            resource_budget: budget,
            candidate_replicate_cost,
            reference_replicate_cost,
            maximum_additional_replicates_per_side: maximum_additional_replicates,
        }
    }

    #[test]
    fn replication_planner_finds_candidate_only_plan_with_vector_cost() {
        let candidate = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 12.0]), &manifest()).unwrap();
        let reference = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 10.0]), &manifest()).unwrap();
        let plan = plan_additional_energy_replicates(
            &candidate,
            &reference,
            &replication_config([1, 5], [5, 1], 100_000, 10_000),
        ).unwrap();
        assert_eq!(plan.status, ReplicationPlanningStatus::AdditionalReplicatesRequired);
        assert_eq!(plan.plan_frontier.len(), 1);
        assert_eq!(plan.plan_frontier[0].kind, ReplicationPlanKind::CandidateOnly);
        assert_eq!(plan.plan_frontier[0].resource_cost.resources[0].units,
            plan.plan_frontier[0].additional_candidate_replicates as u64);
    }

    #[test]
    fn replication_planner_returns_resource_vector_tradeoff_frontier() {
        let candidate = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 11.0]), &manifest()).unwrap();
        let reference = evaluate_ppk2_energy_uncertainty(&trace(&[8.0, 9.0]), &manifest()).unwrap();
        let plan = plan_additional_energy_replicates(
            &candidate,
            &reference,
            &replication_config([1, 5], [5, 1], 1_000, 10),
        ).unwrap();
        assert_eq!(plan.status, ReplicationPlanningStatus::AdditionalReplicatesRequired);
        let kinds: std::collections::HashSet<ReplicationPlanKind> =
            plan.plan_frontier.iter().map(|item| item.kind).collect();
        assert!(kinds.contains(&ReplicationPlanKind::CandidateOnly));
        assert!(kinds.contains(&ReplicationPlanKind::ReferenceOnly));
        assert!(kinds.contains(&ReplicationPlanKind::PairedCandidateAndReference));
        for plan in &plan.plan_frontier {
            assert!(plan.resource_cost.resources.iter().all(|item| item.units <= 1_000));
        }
    }

    #[test]
    fn replication_planner_stops_when_type_b_floor_exceeds_mean_gap() {
        let mut m = manifest();
        m.type_b_relative_standard_uncertainty = 0.10;
        m.manifest_sha256 = compute_metrology_manifest_sha256(&m).unwrap();
        let candidate = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 10.0, 10.0]), &m).unwrap();
        let reference = evaluate_ppk2_energy_uncertainty(&trace(&[9.0, 9.0, 9.0]), &m).unwrap();
        let plan = plan_additional_energy_replicates(
            &candidate,
            &reference,
            &replication_config([1, 1], [1, 1], 10_000, 1_000),
        ).unwrap();
        assert_eq!(plan.status, ReplicationPlanningStatus::MetrologyLimited);
        assert!(plan.plan_frontier.is_empty());
    }

    #[test]
    fn replication_planner_respects_each_vector_budget_component() {
        let candidate = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 12.0]), &manifest()).unwrap();
        let reference = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 10.0]), &manifest()).unwrap();
        let plan = plan_additional_energy_replicates(
            &candidate,
            &reference,
            &replication_config([100, 100], [100, 100], 1, 10),
        ).unwrap();
        assert_eq!(plan.status, ReplicationPlanningStatus::NoPlanWithinBudgetOrHorizon);
        assert!(plan.plan_frontier.is_empty());
    }

    #[test]
    fn reports_strict_win_only_when_intervals_are_disjoint() {
        let a = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 10.0, 10.0]), &manifest()).unwrap();
        let b = evaluate_ppk2_energy_uncertainty(&trace(&[5.0, 5.0, 5.0]), &manifest()).unwrap();
        let result = compare_energy_uncertainty(&b, &a).unwrap();
        assert_eq!(result.decision, EnergyComparisonDecision::CandidateStrictlyLower);
    }
}
