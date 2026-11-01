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
    pub benchmark_id: String,
    pub repeatability_id: String,
    pub phase_protocol_sha256: String,
    pub replicate_design_sha256: String,
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
        benchmark_id: repeatability_protocol.benchmark_id.clone(),
        repeatability_id: repeatability_protocol.repeatability_id.clone(),
        phase_protocol_sha256: phase_protocol.protocol_sha256.clone(),
        replicate_design_sha256: repeatability_protocol.replicate_design_sha256.clone(),
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
        ("replicate_design_sha256", receipt.replicate_design_sha256.as_str()),
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
    if receipt.benchmark_id.trim().is_empty()
        || receipt.repeatability_id.trim().is_empty()
        || receipt.target_id.trim().is_empty()
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
        ("benchmark_id", candidate.benchmark_id.as_str(), reference.benchmark_id.as_str()),
        ("repeatability_id", candidate.repeatability_id.as_str(), reference.repeatability_id.as_str()),
        ("phase_protocol_sha256", candidate.phase_protocol_sha256.as_str(), reference.phase_protocol_sha256.as_str()),
        ("replicate_design_sha256", candidate.replicate_design_sha256.as_str(), reference.replicate_design_sha256.as_str()),
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase_energy_audit::{
        audit_application_phase_energy, compute_application_phase_protocol_sha256,
        compute_phase_clock_mapping_sha256, ApplicationEventOutcome, ApplicationPhaseDefinition,
        PhaseClockMappingManifest,
    };
    use crate::ppk2_trace_audit::Ppk2TraceAuditConfig;
    use crate::metrology_uncertainty::compute_metrology_manifest_sha256;
    use std::fmt::Write as _;

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn phase_protocol() -> ApplicationPhaseProtocolManifest {
        let mut protocol = ApplicationPhaseProtocolManifest {
            schema: "symthaea.nas.application-phase-protocol".into(),
            version: VERSION.into(),
            benchmark_id: "edge-event-energy-test".into(),
            protocol_id: "wake-infer-post-v1".into(),
            phases: vec![
                ApplicationPhaseDefinition { phase_id: "wake".into(), order: 0, required: true },
                ApplicationPhaseDefinition { phase_id: "inference".into(), order: 1, required: true },
                ApplicationPhaseDefinition { phase_id: "postprocess".into(), order: 2, required: true },
            ],
            accepted_success_outcomes: vec![ApplicationEventOutcome::Correct],
            outcome_evaluator_sha256: sha('e'),
            maximum_unattributed_duration_ms: 0.0,
            protocol_sha256: String::new(),
        };
        protocol.protocol_sha256 = compute_application_phase_protocol_sha256(&protocol).unwrap();
        protocol
    }

    fn clock_mapping() -> PhaseClockMappingManifest {
        let mut mapping = PhaseClockMappingManifest {
            schema: "symthaea.nas.phase-clock-mapping".into(),
            version: VERSION.into(),
            mapping_id: "test-clock-map".into(),
            marker_clock_domain_id: "logic-analyzer-clock".into(),
            marker_source_id: "gpio-sync-1".into(),
            ppk2_clock_domain_id: "ppk2-time-ms".into(),
            marker_origin_ns: 0,
            ppk2_origin_ms: 0.0,
            ppk2_ms_per_marker_ns: 1e-6,
            scale_tolerance_fraction: 0.001,
            maximum_alignment_error_ms: 0.0001,
            clock_calibration_evidence_sha256: sha('f'),
            mapping_sha256: String::new(),
        };
        mapping.mapping_sha256 = compute_phase_clock_mapping_sha256(&mapping).unwrap();
        mapping
    }

    fn phase_receipt(
        currents_ua: &[u64],
        artifact_tag: char,
        device_tag: char,
    ) -> PhaseEnergyAuditReceipt {
        let mut samples = String::from("segment_id,time_ms,current_uA\n");
        let mut markers = String::from(
            "event_id,segment_id,phase_id,edge,marker_timestamp_ns,sequence,clock_domain_id,marker_source_id\n"
        );
        let mut outcomes = String::from(
            "event_id,outcome,artifact_sha256,device_session_sha256,input_sha256,output_sha256,quality_metric_id,quality_value,deadline_met,retry_count\n"
        );

        for (index, current) in currents_ua.iter().enumerate() {
            let event_id = format!("event-{index}");
            for sample_index in 0..=4 {
                let time_ms = sample_index as f64 / 100.0;
                writeln!(&mut samples, "{event_id},{time_ms:.2},{current}").unwrap();
            }
            let marker_rows = [
                ("wake", "START", 0u64, 0u64),
                ("wake", "END", 10_000u64, 1u64),
                ("inference", "START", 10_000u64, 2u64),
                ("inference", "END", 30_000u64, 3u64),
                ("postprocess", "START", 30_000u64, 4u64),
                ("postprocess", "END", 40_000u64, 5u64),
            ];
            for (phase, edge, timestamp, sequence) in marker_rows {
                writeln!(
                    &mut markers,
                    "{event_id},{event_id},{phase},{edge},{timestamp},{sequence},logic-analyzer-clock,gpio-sync-1"
                ).unwrap();
            }
            writeln!(
                &mut outcomes,
                "{event_id},correct,{},{},{},{},accuracy,1.0,true,0",
                sha(artifact_tag),
                sha(device_tag),
                sha('c'),
                sha('d'),
            ).unwrap();
        }

        let config = Ppk2TraceAuditConfig {
            voltage_v: 3.3,
            expected_sample_rate_hz: 100_000.0,
            sample_rate_tolerance_fraction: 0.01,
            maximum_gap_ratio: 1.1,
        };
        audit_application_phase_energy(
            &samples,
            &markers,
            &outcomes,
            config,
            &clock_mapping(),
            &phase_protocol(),
        ).unwrap().receipt
    }

    fn repeat_protocol(
        phase_protocol_sha: &str,
        artifact_tag: char,
        device_tag: char,
    ) -> ApplicationEventRepeatabilityProtocol {
        let mut protocol = ApplicationEventRepeatabilityProtocol {
            schema: REPEAT_SCHEMA.into(),
            version: VERSION.into(),
            benchmark_id: "edge-event-energy-test".into(),
            repeatability_id: "matched-event-repeats-v1".into(),
            target_id: "nrf5340-fixture".into(),
            board_revision: "board-rev-a".into(),
            phase_protocol_sha256: phase_protocol_sha.into(),
            replicate_design_sha256: sha('a'),
            candidate_artifact_sha256: sha(artifact_tag),
            device_session_sha256: sha(device_tag),
            matched_input_sha256: sha('c'),
            voltage_source_id: "calibrated-vdd-source".into(),
            voltage_readback_sha256: sha('b'),
            minimum_replicates: 3,
            protocol_sha256: String::new(),
        };
        protocol.protocol_sha256 = compute_application_event_repeatability_protocol_sha256(&protocol).unwrap();
        protocol
    }

    fn metrology_manifest() -> MetrologyUncertaintyManifest {
        let mut manifest = MetrologyUncertaintyManifest {
            schema: "symthaea.nas.metrology-uncertainty-manifest".into(),
            version: VERSION.into(),
            instrument_model: "Nordic PPK2".into(),
            instrument_serial: "fixture-ppk2-01".into(),
            current_range_id: "range-r3".into(),
            calibration_certificate_sha256: sha('d'),
            type_b_relative_standard_uncertainty: 0.01,
            type_b_basis: "fixture calibration standard uncertainty budget".into(),
            type_a_type_b_uncorrelated: true,
            coverage_factor: 2.0,
            replicate_protocol_sha256: sha('a'),
            replicates_independent: true,
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = compute_metrology_manifest_sha256(&manifest).unwrap();
        manifest
    }

    fn evaluated(currents: &[u64], artifact: char, device: char) -> ApplicationEventEnergyUncertaintyReceipt {
        let phase = phase_receipt(currents, artifact, device);
        let phase_protocol = phase_protocol();
        let repeat = repeat_protocol(&phase_protocol.protocol_sha256, artifact, device);
        evaluate_application_event_energy_uncertainty(
            &phase,
            &phase_protocol,
            &repeat,
            &metrology_manifest(),
        ).unwrap()
    }

    #[test]
    fn only_matched_successful_complete_events_enter_type_a_repeats() {
        let receipt = evaluated(&[1_000, 1_100, 900], '8', '9');
        assert_eq!(receipt.matched_attempt_count, 3);
        assert_eq!(receipt.replicate_count, 3);
        assert_eq!(receipt.successful_complete_replicate_count, 3);
        assert!((receipt.mean_energy_uj - 0.132).abs() < 1e-9);
        assert!(receipt.type_a_standard_uncertainty_uj > 0.0);
        assert!(receipt.type_b_standard_uncertainty_uj > 0.0);
        assert!(receipt.maximum_boundary_alignment_bound_uj > 0.0);
        assert!(
            (receipt.expanded_total_uncertainty_uj
                - receipt.expanded_statistical_uncertainty_uj
                - receipt.maximum_boundary_alignment_bound_uj).abs() < 1e-12
        );
        validate_application_event_energy_uncertainty_receipt(&receipt).unwrap();
    }

    #[test]
    fn event_energy_comparison_requires_disjoint_uncertainty_intervals() {
        let candidate = evaluated(&[1_000, 1_100, 900], '8', '9');
        let reference = evaluated(&[2_000, 2_200, 1_800], '1', '2');
        let result = compare_application_event_energy(&candidate, &reference).unwrap();
        assert_eq!(result.decision, ApplicationEventEnergyDecision::CandidateStrictlyLower);
        assert!(is_sha256(&result.comparison_sha256));
    }

    #[test]
    fn close_event_energy_intervals_remain_indeterminate() {
        let candidate = evaluated(&[1_000, 1_001, 999], '8', '9');
        let reference = evaluated(&[990, 991, 989], '1', '2');
        let result = compare_application_event_energy(&candidate, &reference).unwrap();
        assert_eq!(result.decision, ApplicationEventEnergyDecision::IndeterminateIntervalsOverlap);
    }

    #[test]
    fn rejects_fewer_repeats_than_preregistered_minimum() {
        let phase = phase_receipt(&[1_000, 1_100], '8', '9');
        let protocol = phase_protocol();
        let repeat = repeat_protocol(&protocol.protocol_sha256, '8', '9');
        let result = evaluate_application_event_energy_uncertainty(
            &phase, &protocol, &repeat, &metrology_manifest(),
        );
        assert!(result.unwrap_err().to_string().contains("requires at least 3"));
    }

    #[test]
    fn rejects_artifact_or_device_session_mixing() {
        let phase = phase_receipt(&[1_000, 1_100, 900], '8', '9');
        let protocol = phase_protocol();
        let repeat = repeat_protocol(&protocol.protocol_sha256, '1', '9');
        let result = evaluate_application_event_energy_uncertainty(
            &phase, &protocol, &repeat, &metrology_manifest(),
        );
        assert!(result.unwrap_err().to_string().contains("deployment identity"));
    }

    #[test]
    fn rejects_rehashed_uncertainty_interval_tampering() {
        let mut receipt = evaluated(&[1_000, 1_100, 900], '8', '9');
        receipt.lower_bound_uj = 0.0;
        assert!(validate_application_event_energy_uncertainty_receipt(&receipt).is_err());
    }

    #[test]
    fn comparison_rejects_different_calibration_or_voltage_readback() {
        let candidate = evaluated(&[1_000, 1_100, 900], '8', '9');
        let mut reference = evaluated(&[2_000, 2_100, 1_900], '1', '2');
        reference.voltage_readback_sha256 = sha('7');
        assert!(compare_application_event_energy(&candidate, &reference).is_err());
    }
}
