//! Uncertainty for repeated, matched, successful application-event energy.
//!
//! This is deliberately distinct from per-inference PPK2 uncertainty. It consumes a validated
//! phase-energy receipt, selects repeated *complete, accepted, deadline-met* events for one
//! exact input/artifact/device-session tuple, and computes the event-energy uncertainty budget.
//!
//! The worst-case clock-boundary energy term is not treated as random Type B uncertainty. It is
//! added linearly after the expanded Type A/Type B statistical interval.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::metrology_uncertainty::{
    validate_metrology_manifest, MetrologyUncertaintyManifest,
};
use crate::phase_energy_audit::{
    validate_application_phase_protocol, validate_phase_energy_audit_receipt,
    ApplicationPhaseProtocolManifest, PhaseEnergyAuditReceipt, PhaseEnergyEventStatus,
};

const VERSION: &str = "0.1.0";
const REPEAT_SCHEMA: &str = "symthaea.nas.application-event-repeatability-protocol";
const RECEIPT_SCHEMA: &str = "symthaea.nas.application-event-energy-uncertainty";
const COMPARISON_SCHEMA: &str = "symthaea.nas.application-event-energy-comparison";
const REPEAT_DOMAIN: &[u8] = b"symthaea.nas.application-event-repeatability-protocol.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.application-event-energy-uncertainty.v0.1";
const COMPARISON_DOMAIN: &[u8] = b"symthaea.nas.application-event-energy-comparison.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplicationEventUncertaintyError {
    pub field: String,
    pub message: String,
}
impl ApplicationEventUncertaintyError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl fmt::Display for ApplicationEventUncertaintyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for ApplicationEventUncertaintyError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationEventRepeatabilityProtocol {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub repeatability_id: String,
    pub target_id: String,
    pub board_revision: String,
    pub phase_protocol_sha256: String,
    /// Independent repeat-design hash, also required by the metrology manifest.
    pub replicate_design_sha256: String,
    pub candidate_artifact_sha256: String,
    pub device_session_sha256: String,
    pub matched_input_sha256: String,
    pub voltage_source_id: String,
    /// Digest of the calibrated voltage readback/evidence for this repeat session.
    pub voltage_readback_sha256: String,
    pub minimum_replicates: usize,
    pub protocol_sha256: String,
}

pub fn compute_application_event_repeatability_protocol_sha256(
    protocol: &ApplicationEventRepeatabilityProtocol,
) -> Result<String, ApplicationEventUncertaintyError> {
    let mut canonical = protocol.clone();
    canonical.protocol_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        ApplicationEventUncertaintyError::new("repeatability_protocol", error.to_string())
    })?;
    Ok(digest(REPEAT_DOMAIN, &bytes))
}

pub fn validate_application_event_repeatability_protocol(
    protocol: &ApplicationEventRepeatabilityProtocol,
) -> Result<(), ApplicationEventUncertaintyError> {
    if protocol.schema != REPEAT_SCHEMA || protocol.version != VERSION {
        return Err(ApplicationEventUncertaintyError::new("schema_version", "unsupported repeatability protocol"));
    }
    for (field, value) in [
        ("benchmark_id", protocol.benchmark_id.as_str()),
        ("repeatability_id", protocol.repeatability_id.as_str()),
        ("target_id", protocol.target_id.as_str()),
        ("board_revision", protocol.board_revision.as_str()),
        ("voltage_source_id", protocol.voltage_source_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(ApplicationEventUncertaintyError::new(field, "must be non-empty"));
        }
    }
    for (field, value) in [
        ("phase_protocol_sha256", protocol.phase_protocol_sha256.as_str()),
        ("replicate_design_sha256", protocol.replicate_design_sha256.as_str()),
        ("candidate_artifact_sha256", protocol.candidate_artifact_sha256.as_str()),
        ("device_session_sha256", protocol.device_session_sha256.as_str()),
        ("matched_input_sha256", protocol.matched_input_sha256.as_str()),
        ("voltage_readback_sha256", protocol.voltage_readback_sha256.as_str()),
        ("protocol_sha256", protocol.protocol_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(ApplicationEventUncertaintyError::new(field, "must be lowercase SHA-256 hex"));
        }
    }
    if protocol.minimum_replicates < 2 {
        return Err(ApplicationEventUncertaintyError::new("minimum_replicates", "must be at least two"));
    }
    if compute_application_event_repeatability_protocol_sha256(protocol)? != protocol.protocol_sha256 {
        return Err(ApplicationEventUncertaintyError::new("protocol_sha256", "does not match canonical repeatability protocol"));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationEventEnergyUncertaintyReceipt {
    pub schema: String,
    pub version: String,
    pub phase_energy_receipt_sha256: String,
    pub phase_protocol_sha256: String,
    pub repeatability_protocol_sha256: String,
    pub metrology_manifest_sha256: String,
    pub candidate_artifact_sha256: String,
    pub device_session_sha256: String,
    pub target_id: String,
    pub board_revision: String,
    pub instrument_serial: String,
    pub current_range_id: String,
    pub calibration_certificate_sha256: String,
    pub matched_input_sha256: String,
    pub voltage_source_id: String,
    pub voltage_readback_sha256: String,
    pub voltage_v: f64,
    pub matched_attempt_count: usize,
    pub successful_complete_replicate_count: usize,
    pub rejected_outcome_count: usize,
    pub rejected_deadline_count: usize,
    pub rejected_coverage_count: usize,
    pub rejected_missing_energy_count: usize,
    pub replicate_event_ids: Vec<String>,
    pub replicate_count: usize,
    pub mean_energy_uj: f64,
    pub sample_standard_deviation_uj: f64,
    pub type_a_standard_uncertainty_uj: f64,
    pub type_b_standard_uncertainty_uj: f64,
    pub combined_standard_uncertainty_uj: f64,
    pub coverage_factor: f64,
    pub expanded_statistical_uncertainty_uj: f64,
    /// Maximum observed deterministic clock-boundary energy bound across included repeats.
    pub maximum_boundary_alignment_bound_uj: f64,
    /// Expanded statistical uncertainty plus the worst-case boundary bound (linear addition).
    pub expanded_total_uncertainty_uj: f64,
    pub lower_bound_uj: f64,
    pub upper_bound_uj: f64,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicationEventEnergyDecision {
    CandidateStrictlyLower,
    CandidateStrictlyHigher,
    IndeterminateIntervalsOverlap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationEventEnergyComparison {
    pub schema: String,
    pub version: String,
    pub candidate_receipt_sha256: String,
    pub reference_receipt_sha256: String,
    pub decision: ApplicationEventEnergyDecision,
    pub candidate_lower_uj: f64,
    pub candidate_upper_uj: f64,
    pub reference_lower_uj: f64,
    pub reference_upper_uj: f64,
    pub candidate_minus_reference_mean_uj: f64,
    pub comparison_sha256: String,
}

fn compute_event_uncertainty_receipt_sha256(
    receipt: &ApplicationEventEnergyUncertaintyReceipt,
) -> Result<String, ApplicationEventUncertaintyError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        ApplicationEventUncertaintyError::new("receipt", error.to_string())
    })?;
    Ok(digest(RECEIPT_DOMAIN, &bytes))
}

/// Compute Type A, Type B and deterministic boundary terms from repeated successful events.
///
/// All events in the input phase receipt must identify the same candidate artifact and physical
/// device session. Only complete, protocol-accepted, deadline-met events for the exact matched
/// input are included in Type A. Other attempts are counted by exclusion reason, not silently
/// treated as successful repeats.
pub fn evaluate_application_event_energy_uncertainty(
    phase_receipt: &PhaseEnergyAuditReceipt,
    phase_protocol: &ApplicationPhaseProtocolManifest,
    repeatability_protocol: &ApplicationEventRepeatabilityProtocol,
    metrology_manifest: &MetrologyUncertaintyManifest,
) -> Result<ApplicationEventEnergyUncertaintyReceipt, ApplicationEventUncertaintyError> {
    validate_phase_energy_audit_receipt(phase_receipt)
        .map_err(|error| ApplicationEventUncertaintyError::new("phase_energy_receipt", error.to_string()))?;
    validate_application_phase_protocol(phase_protocol)
        .map_err(|error| ApplicationEventUncertaintyError::new("phase_protocol", error.to_string()))?;
    validate_application_event_repeatability_protocol(repeatability_protocol)?;
    validate_metrology_manifest(metrology_manifest)
        .map_err(|error| ApplicationEventUncertaintyError::new("metrology_manifest", error.to_string()))?;

    if phase_receipt.phase_protocol_sha256 != phase_protocol.protocol_sha256
        || repeatability_protocol.phase_protocol_sha256 != phase_protocol.protocol_sha256
        || repeatability_protocol.benchmark_id != phase_protocol.benchmark_id
    {
        return Err(ApplicationEventUncertaintyError::new(
            "phase_protocol_binding",
            "phase receipt, repeatability protocol and benchmark protocol do not match",
        ));
    }
    if metrology_manifest.replicate_protocol_sha256 != repeatability_protocol.replicate_design_sha256 {
        return Err(ApplicationEventUncertaintyError::new(
            "metrology_manifest.replicate_protocol_sha256",
            "metrology manifest is not bound to this exact repeat-design protocol",
        ));
    }
    if phase_receipt.events.is_empty() {
        return Err(ApplicationEventUncertaintyError::new("phase_receipt.events", "must not be empty"));
    }

    let mut matched_attempt_count = 0usize;
    let mut rejected_outcome_count = 0usize;
    let mut rejected_deadline_count = 0usize;
    let mut rejected_coverage_count = 0usize;
    let mut rejected_missing_energy_count = 0usize;
    let mut energies = Vec::<f64>::new();
    let mut boundary_bounds = Vec::<f64>::new();
    let mut event_ids = Vec::<String>::new();

    for event in &phase_receipt.events {
        if event.artifact_sha256 != repeatability_protocol.candidate_artifact_sha256
            || event.device_session_sha256 != repeatability_protocol.device_session_sha256
        {
            return Err(ApplicationEventUncertaintyError::new(
                format!("phase_receipt.events.{}.deployment_identity", event.event_id),
                "phase receipt mixes candidate artifact or physical device session identities",
            ));
        }
        if event.input_sha256 != repeatability_protocol.matched_input_sha256 {
            continue;
        }
        matched_attempt_count += 1;

        if !event.accepted_success_outcome {
            rejected_outcome_count += 1;
            continue;
        }
        if !event.deadline_met {
            rejected_deadline_count += 1;
            continue;
        }
        if event.status != PhaseEnergyEventStatus::CompletePhaseCoverage {
            rejected_coverage_count += 1;
            continue;
        }
        let energy = event.total_event_energy_uj.ok_or_else(|| {
            ApplicationEventUncertaintyError::new(
                format!("phase_receipt.events.{}.total_event_energy_uj", event.event_id),
                "complete successful event is missing total event energy",
            )
        })?;
        let boundary = event.total_event_boundary_alignment_uncertainty_uj.ok_or_else(|| {
            ApplicationEventUncertaintyError::new(
                format!("phase_receipt.events.{}.total_event_boundary_alignment_uncertainty_uj", event.event_id),
                "complete successful event is missing boundary alignment uncertainty",
            )
        })?;
        if !energy.is_finite() || energy <= 0.0 || !boundary.is_finite() || boundary < 0.0 {
            rejected_missing_energy_count += 1;
            continue;
        }
        energies.push(energy);
        boundary_bounds.push(boundary);
        event_ids.push(event.event_id.clone());
    }

    let replicate_count = energies.len();
    if replicate_count < repeatability_protocol.minimum_replicates
        || replicate_count < 2
    {
        return Err(ApplicationEventUncertaintyError::new(
            "replicate_count",
            format!(
                "only {replicate_count} complete accepted deadline-met repeats; protocol requires at least {}",
                repeatability_protocol.minimum_replicates
            ),
        ));
    }

    // Welford's numerically stable sample mean and variance.
    let mut count = 0usize;
    let mut mean = 0.0f64;
    let mut sum_squared_deviation = 0.0f64;
    for energy in &energies {
        count += 1;
        let delta = *energy - mean;
        mean += delta / count as f64;
        let delta_after = *energy - mean;
        sum_squared_deviation += delta * delta_after;
    }
    let sample_standard_deviation = (sum_squared_deviation / (count - 1) as f64).sqrt();
    let type_a = sample_standard_deviation / (count as f64).sqrt();
    let type_b = mean.abs() * metrology_manifest.type_b_relative_standard_uncertainty;
    let combined = type_a.hypot(type_b);
    let expanded_statistical = metrology_manifest.coverage_factor * combined;
    let maximum_boundary = boundary_bounds.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
    // Boundary alignment is a worst-case deterministic bound, so add linearly rather than
    // treating it as an independent Gaussian standard uncertainty.
    let expanded_total = expanded_statistical + maximum_boundary;

    if !mean.is_finite()
        || !sample_standard_deviation.is_finite()
        || !type_a.is_finite()
        || !type_b.is_finite()
        || !combined.is_finite()
        || !expanded_statistical.is_finite()
        || !maximum_boundary.is_finite()
        || !expanded_total.is_finite()
    {
        return Err(ApplicationEventUncertaintyError::new("uncertainty", "derived uncertainty is non-finite"));
    }

    event_ids.sort();
    let mut receipt = ApplicationEventEnergyUncertaintyReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        phase_energy_receipt_sha256: phase_receipt.receipt_sha256.clone(),
        phase_protocol_sha256: phase_protocol.protocol_sha256.clone(),
        repeatability_protocol_sha256: repeatability_protocol.protocol_sha256.clone(),
        metrology_manifest_sha256: metrology_manifest.manifest_sha256.clone(),
        candidate_artifact_sha256: repeatability_protocol.candidate_artifact_sha256.clone(),
        device_session_sha256: repeatability_protocol.device_session_sha256.clone(),
        target_id: repeatability_protocol.target_id.clone(),
        board_revision: repeatability_protocol.board_revision.clone(),
        instrument_serial: metrology_manifest.instrument_serial.clone(),
        current_range_id: metrology_manifest.current_range_id.clone(),
        calibration_certificate_sha256: metrology_manifest.calibration_certificate_sha256.clone(),
        matched_input_sha256: repeatability_protocol.matched_input_sha256.clone(),
        voltage_source_id: repeatability_protocol.voltage_source_id.clone(),
        voltage_readback_sha256: repeatability_protocol.voltage_readback_sha256.clone(),
        voltage_v: phase_receipt.voltage_v,
        matched_attempt_count,
        successful_complete_replicate_count: replicate_count,
        rejected_outcome_count,
        rejected_deadline_count,
        rejected_coverage_count,
        rejected_missing_energy_count,
        replicate_event_ids: event_ids,
        replicate_count,
        mean_energy_uj: mean,
        sample_standard_deviation_uj: sample_standard_deviation,
        type_a_standard_uncertainty_uj: type_a,
        type_b_standard_uncertainty_uj: type_b,
        combined_standard_uncertainty_uj: combined,
        coverage_factor: metrology_manifest.coverage_factor,
        expanded_statistical_uncertainty_uj: expanded_statistical,
        maximum_boundary_alignment_bound_uj: maximum_boundary,
        expanded_total_uncertainty_uj: expanded_total,
        lower_bound_uj: (mean - expanded_total).max(0.0),
        upper_bound_uj: mean + expanded_total,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_event_uncertainty_receipt_sha256(&receipt)?;
    Ok(receipt)
}

pub fn validate_application_event_energy_uncertainty_receipt(
    receipt: &ApplicationEventEnergyUncertaintyReceipt,
) -> Result<(), ApplicationEventUncertaintyError> {
    if receipt.schema != RECEIPT_SCHEMA || receipt.version != VERSION {
        return Err(ApplicationEventUncertaintyError::new("schema_version", "unsupported receipt schema/version"));
    }
    for (field, value) in [
        ("phase_energy_receipt_sha256", receipt.phase_energy_receipt_sha256.as_str()),
        ("phase_protocol_sha256", receipt.phase_protocol_sha256.as_str()),
        ("repeatability_protocol_sha256", receipt.repeatability_protocol_sha256.as_str()),
        ("metrology_manifest_sha256", receipt.metrology_manifest_sha256.as_str()),
        ("candidate_artifact_sha256", receipt.candidate_artifact_sha256.as_str()),
        ("device_session_sha256", receipt.device_session_sha256.as_str()),
        ("calibration_certificate_sha256", receipt.calibration_certificate_sha256.as_str()),
        ("matched_input_sha256", receipt.matched_input_sha256.as_str()),
        ("voltage_readback_sha256", receipt.voltage_readback_sha256.as_str()),
        ("receipt_sha256", receipt.receipt_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(ApplicationEventUncertaintyError::new(field, "must be lowercase SHA-256 hex"));
        }
    }
    if receipt.target_id.trim().is_empty()
        || receipt.board_revision.trim().is_empty()
        || receipt.instrument_serial.trim().is_empty()
        || receipt.current_range_id.trim().is_empty()
        || receipt.voltage_source_id.trim().is_empty()
        || receipt.replicate_count < 2
        || receipt.replicate_count != receipt.successful_complete_replicate_count
        || receipt.replicate_event_ids.len() != receipt.replicate_count
        || receipt.replicate_event_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ApplicationEventUncertaintyError::new("receipt.inventory", "invalid identity or replicate inventory"));
    }
    for (field, value) in [
        ("voltage_v", receipt.voltage_v),
        ("mean_energy_uj", receipt.mean_energy_uj),
        ("sample_standard_deviation_uj", receipt.sample_standard_deviation_uj),
        ("type_a_standard_uncertainty_uj", receipt.type_a_standard_uncertainty_uj),
        ("type_b_standard_uncertainty_uj", receipt.type_b_standard_uncertainty_uj),
        ("combined_standard_uncertainty_uj", receipt.combined_standard_uncertainty_uj),
        ("coverage_factor", receipt.coverage_factor),
        ("expanded_statistical_uncertainty_uj", receipt.expanded_statistical_uncertainty_uj),
        ("maximum_boundary_alignment_bound_uj", receipt.maximum_boundary_alignment_bound_uj),
        ("expanded_total_uncertainty_uj", receipt.expanded_total_uncertainty_uj),
        ("lower_bound_uj", receipt.lower_bound_uj),
        ("upper_bound_uj", receipt.upper_bound_uj),
    ] {
        if !value.is_finite() {
            return Err(ApplicationEventUncertaintyError::new(field, "must be finite"));
        }
    }
    if receipt.voltage_v <= 0.0
        || receipt.mean_energy_uj <= 0.0
        || receipt.sample_standard_deviation_uj < 0.0
        || receipt.type_a_standard_uncertainty_uj < 0.0
        || receipt.type_b_standard_uncertainty_uj < 0.0
        || receipt.coverage_factor <= 0.0
        || receipt.maximum_boundary_alignment_bound_uj < 0.0
        || receipt.matched_attempt_count != receipt.successful_complete_replicate_count
            .saturating_add(receipt.rejected_outcome_count)
            .saturating_add(receipt.rejected_deadline_count)
            .saturating_add(receipt.rejected_coverage_count)
            .saturating_add(receipt.rejected_missing_energy_count)
    {
        return Err(ApplicationEventUncertaintyError::new("receipt.values", "uncertainty values or counts are outside valid bounds"));
    }
    let expected_type_a = receipt.sample_standard_deviation_uj / (receipt.replicate_count as f64).sqrt();
    let expected_combined = receipt.type_a_standard_uncertainty_uj.hypot(receipt.type_b_standard_uncertainty_uj);
    let expected_statistical = receipt.coverage_factor * expected_combined;
    let expected_total = expected_statistical + receipt.maximum_boundary_alignment_bound_uj;
    let close = |left: f64, right: f64| {
        (left - right).abs() <= 1e-10 * left.abs().max(right.abs()).max(1.0)
    };
    if !close(receipt.type_a_standard_uncertainty_uj, expected_type_a)
        || !close(receipt.combined_standard_uncertainty_uj, expected_combined)
        || !close(receipt.expanded_statistical_uncertainty_uj, expected_statistical)
        || !close(receipt.expanded_total_uncertainty_uj, expected_total)
        || !close(receipt.lower_bound_uj, (receipt.mean_energy_uj - expected_total).max(0.0))
        || !close(receipt.upper_bound_uj, receipt.mean_energy_uj + expected_total)
    {
        return Err(ApplicationEventUncertaintyError::new("receipt.derived_fields", "uncertainty fields are inconsistent"));
    }
    if compute_event_uncertainty_receipt_sha256(receipt)? != receipt.receipt_sha256 {
        return Err(ApplicationEventUncertaintyError::new("receipt.receipt_sha256", "does not match canonical receipt"));
    }
    Ok(())
}

pub fn compare_application_event_energy(
    candidate: &ApplicationEventEnergyUncertaintyReceipt,
    reference: &ApplicationEventEnergyUncertaintyReceipt,
) -> Result<ApplicationEventEnergyComparison, ApplicationEventUncertaintyError> {
    validate_application_event_energy_uncertainty_receipt(candidate)
        .map_err(|error| ApplicationEventUncertaintyError::new("candidate", error.to_string()))?;
    validate_application_event_energy_uncertainty_receipt(reference)
        .map_err(|error| ApplicationEventUncertaintyError::new("reference", error.to_string()))?;

    // Match the workload, target and metrology conditions. The candidate artifacts and device
    // session receipts intentionally differ; their target/board and instrument calibration must
    // remain matched. Systematic calibration error is not assumed to cancel.
    for (field, left, right) in [
        ("phase_protocol_sha256", candidate.phase_protocol_sha256.as_str(), reference.phase_protocol_sha256.as_str()),
        ("matched_input_sha256", candidate.matched_input_sha256.as_str(), reference.matched_input_sha256.as_str()),
        ("target_id", candidate.target_id.as_str(), reference.target_id.as_str()),
        ("board_revision", candidate.board_revision.as_str(), reference.board_revision.as_str()),
        ("instrument_serial", candidate.instrument_serial.as_str(), reference.instrument_serial.as_str()),
        ("current_range_id", candidate.current_range_id.as_str(), reference.current_range_id.as_str()),
        ("calibration_certificate_sha256", candidate.calibration_certificate_sha256.as_str(), reference.calibration_certificate_sha256.as_str()),
        ("voltage_source_id", candidate.voltage_source_id.as_str(), reference.voltage_source_id.as_str()),
        ("voltage_readback_sha256", candidate.voltage_readback_sha256.as_str(), reference.voltage_readback_sha256.as_str()),
    ] {
        if left != right {
            return Err(ApplicationEventUncertaintyError::new(
                format!("comparison.{field}"),
                "candidate and reference must use matched workload, target, board, instrument, range and calibration",
            ));
        }
    }
    if (candidate.voltage_v - reference.voltage_v).abs() > 1e-9 {
        return Err(ApplicationEventUncertaintyError::new("comparison.voltage_v", "candidate and reference supply voltages differ"));
    }
    let decision = if candidate.upper_bound_uj < reference.lower_bound_uj {
        ApplicationEventEnergyDecision::CandidateStrictlyLower
    } else if candidate.lower_bound_uj > reference.upper_bound_uj {
        ApplicationEventEnergyDecision::CandidateStrictlyHigher
    } else {
        ApplicationEventEnergyDecision::IndeterminateIntervalsOverlap
    };

    let mut comparison = ApplicationEventEnergyComparison {
        schema: COMPARISON_SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        candidate_receipt_sha256: candidate.receipt_sha256.clone(),
        reference_receipt_sha256: reference.receipt_sha256.clone(),
        decision,
        candidate_lower_uj: candidate.lower_bound_uj,
        candidate_upper_uj: candidate.upper_bound_uj,
        reference_lower_uj: reference.lower_bound_uj,
        reference_upper_uj: reference.upper_bound_uj,
        candidate_minus_reference_mean_uj: candidate.mean_energy_uj - reference.mean_energy_uj,
        comparison_sha256: String::new(),
    };
    let bytes = serde_json::to_vec(&comparison).map_err(|error| {
        ApplicationEventUncertaintyError::new("comparison", error.to_string())
    })?;
    comparison.comparison_sha256 = digest(COMPARISON_DOMAIN, &bytes);
    Ok(comparison)
}

