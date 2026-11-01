//! Calibration-bound energy uncertainty and conservative pairwise comparison.
//!
//! Type A repeatability is estimated from independent inference-segment energy replicates.
//! Type B relative standard uncertainty is supplied by a pinned calibration/metrology budget.
//! A manufacturer accuracy bound is not silently converted into a standard uncertainty.
//! This module does not establish that a calibration certificate is authentic; its digest and
//! trust policy must be verified by the acquisition/provenance layer.

use std::error::Error;
use std::fmt;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::ppk2_trace_audit::{Ppk2TraceAuditReceipt, Ppk2SegmentMetrics};

const MANIFEST_SCHEMA: &str = "symthaea.nas.metrology-uncertainty-manifest";
const RECEIPT_SCHEMA: &str = "symthaea.nas.energy-uncertainty-receipt";
const VERSION: &str = "0.1.0";
const MANIFEST_DOMAIN: &[u8] = b"symthaea.nas.metrology-uncertainty-manifest.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.energy-uncertainty-receipt.v0.1";

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

#[derive(Debug, Clone, PartialEq, Serialize)]
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
    /// Coverage factor chosen from the declared uncertainty methodology.
    pub coverage_factor: f64,
    pub replicate_protocol_sha256: String,
    /// Must be true before the usual sample standard error is used as Type A uncertainty.
    pub replicates_independent: bool,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyComparisonDecision {
    CandidateStrictlyLower,
    CandidateStrictlyHigher,
    IndeterminateIntervalsOverlap,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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

/// Only claim strict superiority when the expanded intervals do not overlap.
/// Shared systematic errors are not assumed to cancel; paired-covariance modeling is a
/// separate future extension and must be justified by the acquisition design.
pub fn compare_energy_uncertainty(
    candidate: &EnergyUncertaintyReceipt,
    reference: &EnergyUncertaintyReceipt,
) -> Result<EnergyComparison, MetrologyUncertaintyError> {
    for (field, value) in [
        ("candidate.receipt_sha256", candidate.receipt_sha256.as_str()),
        ("reference.receipt_sha256", reference.receipt_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(MetrologyUncertaintyError::new(field, "must be lowercase SHA-256 hex"));
        }
    }
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
            Ppk2SegmentMetrics {
                segment_id: i.to_string(),
                sample_count: 100,
                first_time_ms: 0.0,
                last_time_ms: 1.0,
                duration_ms: 1.0,
                minimum_current_ua: 100.0,
                maximum_current_ua: 200.0,
                time_weighted_mean_current_ua: 150.0,
                mean_power_uw: 495.0,
                energy_uj: *energy,
                median_interval_ms: 0.01,
                maximum_interval_ms: 0.01,
                median_sample_rate_hz: 100_000.0,
                maximum_gap_ratio: 1.0,
            }
        }).collect();
        Ppk2TraceAuditReceipt {
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
            mean_segment_current_ua: 150.0,
            mean_segment_power_uw: 495.0,
            mean_segment_energy_uj: energies.iter().sum::<f64>() / energies.len() as f64,
            minimum_segment_energy_uj: energies.iter().copied().fold(f64::INFINITY, f64::min),
            maximum_segment_energy_uj: energies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            mean_segment_duration_ms: 1.0,
            median_interval_ms: 0.01,
            median_sample_rate_hz: 100_000.0,
            maximum_observed_gap_ratio: 1.0,
            segments,
            receipt_sha256: sha('d'),
        }
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
    fn reports_strict_win_only_when_intervals_are_disjoint() {
        let a = evaluate_ppk2_energy_uncertainty(&trace(&[10.0, 10.0, 10.0]), &manifest()).unwrap();
        let b = evaluate_ppk2_energy_uncertainty(&trace(&[5.0, 5.0, 5.0]), &manifest()).unwrap();
        let result = compare_energy_uncertainty(&b, &a).unwrap();
        assert_eq!(result.decision, EnergyComparisonDecision::CandidateStrictlyLower);
    }
}
