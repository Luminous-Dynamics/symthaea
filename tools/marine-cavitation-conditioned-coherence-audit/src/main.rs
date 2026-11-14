use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const PSD_REL_TOLERANCE: f64 = 1.0e-8;
const WINDOW_TIME_TOLERANCE_S: f64 = 1.0e-9;

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind { Synthetic, Measured }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus { PublicOrPermitted, Unknown, Restricted }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEvidence {
    dataset_id: String,
    citation: String,
    raw_data_sha256: String,
    rights_status: RightsStatus,
    independently_reviewed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    intermittency_report_case_id: String,
    intermittency_report_sha256: String,
    intermittency_disposition: String,
    vessel_id: String,
    propeller_revision: String,
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_binding_receipt_sha256: String,
    synchronization_receipt_sha256: String,
    shared_timebase_id: String,
    hydrophone_instrument_id: String,
    hydrophone_channel_id: String,
    hydrophone_calibration_certificate_sha256: String,
    vibration_instrument_id: String,
    vibration_channel_id: String,
    vibration_reference_location: String,
    vibration_reference_orientation: String,
    vibration_calibration_certificate_sha256: String,
    maximum_sync_alignment_uncertainty_s: f64,
    frequency_match_tolerance_fraction: f64,
    coherence_estimator_fingerprint_sha256: String,
    cross_spectrum_convention: String,
    minimum_windows_per_state_per_run: usize,
    minimum_independent_runs: usize,
    windows: Vec<JoinedWindow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinedWindow {
    run_id: String,
    window_id: String,
    start_time_s: f64,
    end_time_s: f64,
    acoustic_trace_sha256: String,
    cavitation_observation_sha256: String,
    cavitation_annotation_raw: Option<bool>,
    cavitation_state_for_conditioning: Option<bool>,
    acoustic_cavitation_alignment_offset_s: f64,
    acoustic_cavitation_alignment_expanded_uncertainty_s: f64,
    acoustic_cavitation_alignment_eligible: bool,
    timebase_receipt_sha256: String,
    acoustic_cavitation_sync_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    hydrophone_calibration_certificate_sha256: String,
    coherence: CoherenceWindow,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoherenceWindow {
    case_id: String,
    report_sha256: String,
    evidence_kind: EvidenceKind,
    disposition: String,
    source_dataset_id: String,
    source_data_sha256: String,
    vessel_id: String,
    propeller_revision: String,
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_binding_receipt_sha256: String,
    synchronization_receipt_sha256: String,
    shared_timebase_id: String,
    pressure_instrument_id: String,
    pressure_channel_id: String,
    pressure_calibration_certificate_sha256: String,
    pressure_raw_trace_sha256: String,
    vibration_instrument_id: String,
    vibration_channel_id: String,
    vibration_calibration_certificate_sha256: String,
    vibration_reference_location: String,
    vibration_reference_orientation: String,
    vibration_raw_trace_sha256: String,
    pressure_window_start_s: f64,
    pressure_window_end_s: f64,
    sample_rate_hz: f64,
    sample_count: u64,
    estimator_fingerprint_sha256: String,
    cross_spectrum_convention: String,
    bands: Vec<CoherenceBand>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoherenceBand {
    band_id: String,
    frequency_hz: f64,
    pressure_psd_pa2_per_hz: f64,
    coherence_magnitude_squared: f64,
    coherence_lower_bound: Option<f64>,
    coherence_upper_bound: Option<f64>,
    vibration_reference_coherent_pressure_psd_pa2_per_hz: f64,
    vibration_reference_coherent_pressure_psd_lower_bound: Option<f64>,
    vibration_reference_coherent_pressure_psd_upper_bound: Option<f64>,
    non_coherent_residual_pressure_psd_pa2_per_hz: f64,
    non_coherent_residual_pressure_psd_lower_bound: Option<f64>,
    non_coherent_residual_pressure_psd_upper_bound: Option<f64>,
    uncertainty_status: String,
    marginal_uncertainty_coverage_level: f64,
    phase_status: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_citation: String,
    source_data_sha256: String,
    intermittency_report_case_id: String,
    intermittency_report_sha256: String,
    evidence_kind: &'static str,
    vessel_id: String,
    propeller_revision: String,
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_binding_receipt_sha256: String,
    synchronization_receipt_sha256: String,
    hydrophone_instrument_id: String,
    vibration_instrument_id: String,
    vibration_reference_location: String,
    vibration_reference_orientation: String,
    disposition: &'static str,
    claim_ceiling: &'static str,
    run_reports: Vec<RunReport>,
    aggregate_band_reports: Vec<AggregateBandReport>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RunReport {
    run_id: String,
    total_windows: usize,
    cavitation_present_windows: usize,
    cavitation_absent_windows: usize,
    excluded_unknown_cavitation_state: usize,
    excluded_alignment_ineligible: usize,
    excluded_uncertainty_unresolved: usize,
    phase_unresolved_windows: usize,
    band_reports: Vec<RunBandReport>,
}

#[derive(Debug, Serialize, Clone)]
struct RunBandReport {
    band_id: String,
    frequency_hz: f64,
    cavitation_present: Option<CohortSummary>,
    cavitation_absent: Option<CohortSummary>,
    present_minus_absent_mean_coherence: Option<f64>,
    present_minus_absent_mean_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    present_minus_absent_mean_residual_pressure_psd_pa2_per_hz: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
struct CohortSummary {
    window_count: usize,
    mean_coherence: f64,
    median_coherence: f64,
    p05_coherence: f64,
    p95_coherence: f64,
    mean_coherence_marginal_lower_bound: Option<f64>,
    mean_coherence_marginal_upper_bound: Option<f64>,
    mean_pressure_psd_pa2_per_hz: f64,
    mean_coherent_pressure_psd_pa2_per_hz: f64,
    p05_coherent_pressure_psd_pa2_per_hz: f64,
    p95_coherent_pressure_psd_pa2_per_hz: f64,
    mean_coherent_pressure_psd_marginal_lower_bound: Option<f64>,
    mean_coherent_pressure_psd_marginal_upper_bound: Option<f64>,
    mean_residual_pressure_psd_pa2_per_hz: f64,
    p05_residual_pressure_psd_pa2_per_hz: f64,
    p95_residual_pressure_psd_pa2_per_hz: f64,
    mean_residual_pressure_psd_marginal_lower_bound: Option<f64>,
    mean_residual_pressure_psd_marginal_upper_bound: Option<f64>,
    marginal_uncertainty_coverage_level: f64,
    uncertainty_interpretation: &'static str,
}

#[derive(Debug, Serialize)]
struct AggregateBandReport {
    band_id: String,
    frequency_hz: f64,
    independent_runs_with_both_cohorts: usize,
    cavitation_present_equal_run_weighted_mean_coherence: Option<f64>,
    cavitation_absent_equal_run_weighted_mean_coherence: Option<f64>,
    present_minus_absent_equal_run_weighted_mean_coherence: Option<f64>,
    between_run_sd_of_coherence_delta: Option<f64>,
    cavitation_present_equal_run_weighted_mean_coherent_psd_pa2_per_hz: Option<f64>,
    cavitation_absent_equal_run_weighted_mean_coherent_psd_pa2_per_hz: Option<f64>,
    present_minus_absent_equal_run_weighted_mean_coherent_psd_pa2_per_hz: Option<f64>,
    cavitation_present_equal_run_weighted_mean_residual_psd_pa2_per_hz: Option<f64>,
    cavitation_absent_equal_run_weighted_mean_residual_psd_pa2_per_hz: Option<f64>,
    present_minus_absent_equal_run_weighted_mean_residual_psd_pa2_per_hz: Option<f64>,
    weighting: &'static str,
    interpretation: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    coherence: f64,
    coherence_lower: f64,
    coherence_upper: f64,
    pressure_psd: f64,
    coherent_psd: f64,
    coherent_lower: f64,
    coherent_upper: f64,
    residual_psd: f64,
    residual_lower: f64,
    residual_upper: f64,
    coverage: f64,
}

#[derive(Debug, Default)]
struct RunAccumulator {
    total_windows: usize,
    present_windows: usize,
    absent_windows: usize,
    unknown_state: usize,
    alignment_ineligible: usize,
    uncertainty_unresolved: usize,
    phase_unresolved: usize,
    present: BTreeMap<String, Vec<Sample>>,
    absent: BTreeMap<String, Vec<Sample>>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn close_time(a: f64, b: f64) -> bool {
    (a - b).abs() <= WINDOW_TIME_TOLERANCE_S
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let x = p * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo as f64))
}

fn mean_option(values: &[f64]) -> Option<f64> {
    if values.is_empty() { None } else { mean(values) }
}

fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values)?;
    let ss = values.iter().map(|value| (value - m).powi(2)).sum::<f64>();
    Some((ss / (values.len() - 1) as f64).sqrt())
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    for (label, value) in [
        ("case_id", case.case_id.as_str()),
        ("source dataset_id", case.source.dataset_id.as_str()),
        ("source citation", case.source.citation.as_str()),
        ("intermittency_report_case_id", case.intermittency_report_case_id.as_str()),
        ("vessel_id", case.vessel_id.as_str()),
        ("propeller_revision", case.propeller_revision.as_str()),
        ("operating_condition_id", case.operating_condition_id.as_str()),
        ("shared_timebase_id", case.shared_timebase_id.as_str()),
        ("hydrophone_instrument_id", case.hydrophone_instrument_id.as_str()),
        ("hydrophone_channel_id", case.hydrophone_channel_id.as_str()),
        ("vibration_instrument_id", case.vibration_instrument_id.as_str()),
        ("vibration_channel_id", case.vibration_channel_id.as_str()),
        ("vibration_reference_location", case.vibration_reference_location.as_str()),
        ("vibration_reference_orientation", case.vibration_reference_orientation.as_str()),
        ("cross_spectrum_convention", case.cross_spectrum_convention.as_str()),
    ] {
        if value.trim().is_empty() {
            errors.push(format!("{label} is required"));
        }
    }
    for (label, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("intermittency report", &case.intermittency_report_sha256),
        ("operating point", &case.operating_point_receipt_sha256),
        ("service fingerprint", &case.service_fingerprint_sha256),
        ("environment fingerprint", &case.environment_fingerprint_sha256),
        ("timebase binding", &case.timebase_binding_receipt_sha256),
        ("synchronization receipt", &case.synchronization_receipt_sha256),
        ("hydrophone calibration", &case.hydrophone_calibration_certificate_sha256),
        ("vibration calibration", &case.vibration_calibration_certificate_sha256),
        ("coherence estimator", &case.coherence_estimator_fingerprint_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label} digest must be 64 hexadecimal characters"));
        }
    }
    if !finite_positive(case.maximum_sync_alignment_uncertainty_s)
        || !case.frequency_match_tolerance_fraction.is_finite()
        || case.frequency_match_tolerance_fraction <= 0.0
        || case.frequency_match_tolerance_fraction >= 1.0
        || case.minimum_windows_per_state_per_run == 0
        || case.minimum_independent_runs < 2
    {
        errors.push("alignment/frequency tolerances and minimum independent window/run counts are invalid".to_string());
    }
    if case.windows.is_empty() {
        errors.push("at least one joined window is required".to_string());
    }

    let mut joined_ids = BTreeSet::new();
    let mut acoustic_digests = BTreeSet::new();
    let mut cavitation_digests = BTreeSet::new();
    let mut coherence_case_ids = BTreeSet::new();
    let mut coherence_report_digests = BTreeSet::new();

    for window in &case.windows {
        let identity = format!("{}::{}", window.run_id, window.window_id);
        if window.run_id.trim().is_empty()
            || window.window_id.trim().is_empty()
            || !joined_ids.insert(identity.clone())
        {
            errors.push(format!("empty or duplicate joined window identity {identity}"));
        }
        if !is_sha256(&window.acoustic_trace_sha256)
            || !is_sha256(&window.cavitation_observation_sha256)
            || !is_sha256(&window.timebase_receipt_sha256)
            || !is_sha256(&window.acoustic_cavitation_sync_receipt_sha256)
            || !is_sha256(&window.operating_point_receipt_sha256)
            || !is_sha256(&window.service_fingerprint_sha256)
            || !is_sha256(&window.environment_fingerprint_sha256)
            || !is_sha256(&window.hydrophone_calibration_certificate_sha256)
        {
            errors.push(format!("{identity}: malformed intermittency trace/receipt digest"));
        }
        if !acoustic_digests.insert(window.acoustic_trace_sha256.clone()) {
            errors.push(format!("{identity}: acoustic trace digest is reused"));
        }
        if !cavitation_digests.insert(window.cavitation_observation_sha256.clone()) {
            errors.push(format!("{identity}: cavitation observation digest is reused"));
        }
        if !window.start_time_s.is_finite()
            || !window.end_time_s.is_finite()
            || window.end_time_s <= window.start_time_s
            || !window.acoustic_cavitation_alignment_offset_s.is_finite()
            || !finite_nonnegative(window.acoustic_cavitation_alignment_expanded_uncertainty_s)
        {
            errors.push(format!("{identity}: invalid time interval or alignment values"));
        }
        let computed_alignment_eligible =
            window.acoustic_cavitation_alignment_offset_s.abs()
                + window.acoustic_cavitation_alignment_expanded_uncertainty_s
                <= case.maximum_sync_alignment_uncertainty_s;
        if window.acoustic_cavitation_alignment_eligible != computed_alignment_eligible {
            errors.push(format!("{identity}: alignment eligibility does not match offset plus expanded uncertainty"));
        }
        if !computed_alignment_eligible && window.cavitation_state_for_conditioning.is_some() {
            errors.push(format!("{identity}: unaligned cavitation state must not enter conditioned cohorts"));
        }
        if computed_alignment_eligible && window.cavitation_state_for_conditioning != window.cavitation_annotation_raw {
            errors.push(format!("{identity}: aligned conditioning state must preserve the raw annotation"));
        }

        let coh = &window.coherence;
        if coh.case_id.trim().is_empty() || !coherence_case_ids.insert(coh.case_id.clone()) {
            errors.push(format!("{identity}: empty or reused coherence case ID"));
        }
        if !is_sha256(&coh.report_sha256) || !coherence_report_digests.insert(coh.report_sha256.clone()) {
            errors.push(format!("{identity}: malformed or reused coherence report digest"));
        }
        if coh.vibration_raw_trace_sha256 == coh.pressure_raw_trace_sha256 {
            errors.push(format!("{identity}: pressure and vibration raw-trace digests must be distinct"));
        }
        for (label, digest) in [
            ("coherence source data", &coh.source_data_sha256),
            ("coherence operating point", &coh.operating_point_receipt_sha256),
            ("coherence service", &coh.service_fingerprint_sha256),
            ("coherence environment", &coh.environment_fingerprint_sha256),
            ("coherence timebase binding", &coh.timebase_binding_receipt_sha256),
            ("coherence synchronization", &coh.synchronization_receipt_sha256),
            ("pressure calibration", &coh.pressure_calibration_certificate_sha256),
            ("pressure raw trace", &coh.pressure_raw_trace_sha256),
            ("vibration calibration", &coh.vibration_calibration_certificate_sha256),
            ("vibration raw trace", &coh.vibration_raw_trace_sha256),
            ("coherence estimator", &coh.estimator_fingerprint_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("{identity}: {label} digest is malformed"));
            }
        }
        if coh.pressure_raw_trace_sha256 != window.acoustic_trace_sha256 {
            errors.push(format!("{identity}: coherence pressure raw-trace digest does not match intermittency acoustic trace"));
        }
        if !close_time(coh.pressure_window_start_s, window.start_time_s)
            || !close_time(coh.pressure_window_end_s, window.end_time_s)
        {
            errors.push(format!("{identity}: coherence pressure window does not match cavitation/acoustic window"));
        }
        if !finite_positive(coh.sample_rate_hz) || coh.sample_count < 2 {
            errors.push(format!("{identity}: coherence sample rate/count are invalid"));
        }
        if coh.bands.is_empty() {
            errors.push(format!("{identity}: coherence report contains no frequency bands"));
        }
        let mut band_ids = BTreeSet::new();
        for band in &coh.bands {
            if band.band_id.trim().is_empty() || !band_ids.insert(band.band_id.clone()) {
                errors.push(format!("{identity}: empty or duplicate coherence band ID"));
            }
            if !finite_positive(band.frequency_hz)
                || !finite_positive(band.pressure_psd_pa2_per_hz)
                || !finite_nonnegative(band.coherence_magnitude_squared)
                || band.coherence_magnitude_squared > 1.0 + 1.0e-9
                || !finite_nonnegative(band.vibration_reference_coherent_pressure_psd_pa2_per_hz)
                || !finite_nonnegative(band.non_coherent_residual_pressure_psd_pa2_per_hz)
                || !band.marginal_uncertainty_coverage_level.is_finite()
                || band.marginal_uncertainty_coverage_level <= 0.0
                || band.marginal_uncertainty_coverage_level >= 1.0
            {
                errors.push(format!("{identity} / {}: invalid coherence or spectral power values", band.band_id));
            }
            let recomposed = band.vibration_reference_coherent_pressure_psd_pa2_per_hz
                + band.non_coherent_residual_pressure_psd_pa2_per_hz;
            let allowed = PSD_REL_TOLERANCE * band.pressure_psd_pa2_per_hz.max(f64::MIN_POSITIVE) + 1.0e-24;
            if (recomposed - band.pressure_psd_pa2_per_hz).abs() > allowed {
                errors.push(format!("{identity} / {}: coherent plus residual PSD does not recover measured pressure PSD", band.band_id));
            }
            let intervals = [
                (band.coherence_lower_bound, band.coherence_upper_bound, band.coherence_magnitude_squared, "coherence"),
                (band.vibration_reference_coherent_pressure_psd_lower_bound, band.vibration_reference_coherent_pressure_psd_upper_bound, band.vibration_reference_coherent_pressure_psd_pa2_per_hz, "coherent PSD"),
                (band.non_coherent_residual_pressure_psd_lower_bound, band.non_coherent_residual_pressure_psd_upper_bound, band.non_coherent_residual_pressure_psd_pa2_per_hz, "residual PSD"),
            ];
            match band.uncertainty_status.as_str() {
                "marginal_interval_envelope_available" => {
                    for (lower, upper, value, label) in intervals {
                        match (lower, upper) {
                            (Some(lo), Some(hi))
                                if lo.is_finite()
                                    && hi.is_finite()
                                    && lo >= 0.0
                                    && lo <= value
                                    && value <= hi
                                    && (label != "coherence" || hi <= 1.0 + 1.0e-9) => {}
                            _ => errors.push(format!("{identity} / {}: {label} must have an ordered conservative marginal interval", band.band_id)),
                        }
                    }
                }
                "coherent_residual_interval_unresolved" => {}
                _ => errors.push(format!("{identity} / {}: unsupported uncertainty status", band.band_id)),
            }
            if band.phase_status != "phase_resolved_within_declared_limit" && band.phase_status != "phase_unresolved" {
                errors.push(format!("{identity} / {}: unsupported phase status", band.band_id));
            }
        }
    }
    errors
}

fn validate_cross_record_identity(case: &Case) -> (bool, Vec<String>) {
    let mut incomparable = false;
    let mut reasons = Vec::new();
    let reference_sample_rate = case.windows.first().map(|w| w.coherence.sample_rate_hz).unwrap_or_default();
    let reference_sample_count = case.windows.first().map(|w| w.coherence.sample_count).unwrap_or_default();
    let reference_band_coverage: BTreeMap<String, f64> = case.windows.first()
        .map(|window| window.coherence.bands.iter().map(|band| (band.band_id.clone(), band.marginal_uncertainty_coverage_level)).collect())
        .unwrap_or_default();
    let reference_band_ids: BTreeSet<String> = case.windows.first()
        .map(|window| window.coherence.bands.iter().map(|band| band.band_id.clone()).collect())
        .unwrap_or_default();
    let reference_frequencies: BTreeMap<String, f64> = case.windows.first()
        .map(|window| window.coherence.bands.iter().map(|band| (band.band_id.clone(), band.frequency_hz)).collect())
        .unwrap_or_default();

    for window in &case.windows {
        let coh = &window.coherence;
        let current_band_ids: BTreeSet<String> = coh.bands.iter().map(|band| band.band_id.clone()).collect();
        if current_band_ids != reference_band_ids {
            incomparable = true;
            reasons.push(format!("{}: coherence band ID set differs across windows", window.window_id));
        }
        if (coh.sample_rate_hz - reference_sample_rate).abs() > 1.0e-9 || coh.sample_count != reference_sample_count {
            incomparable = true;
            reasons.push(format!("{}: coherence sample rate/count differs across windows", window.window_id));
        }
        for band in &coh.bands {
            if reference_band_coverage.get(&band.band_id).is_some_and(|coverage| (coverage - band.marginal_uncertainty_coverage_level).abs() > 1.0e-9) {
                incomparable = true;
                reasons.push(format!("{}: marginal uncertainty coverage differs for band {}", window.window_id, band.band_id));
            }
            if let Some(reference_frequency) = reference_frequencies.get(&band.band_id) {
                let relative = (reference_frequency - band.frequency_hz).abs() / reference_frequency.max(1.0);
                if relative > case.frequency_match_tolerance_fraction {
                    incomparable = true;
                    reasons.push(format!("{}: frequency grid differs for band {}", window.window_id, band.band_id));
                }
            }
        }
        let checks = [
            (coh.source_dataset_id == case.source.dataset_id, "source dataset identity differs"),
            (coh.source_data_sha256 == case.source.raw_data_sha256, "source data digest differs"),
            (coh.vessel_id == case.vessel_id, "vessel identity differs"),
            (coh.propeller_revision == case.propeller_revision, "propeller revision differs"),
            (coh.operating_condition_id == case.operating_condition_id, "operating condition differs"),
            (coh.operating_point_receipt_sha256 == case.operating_point_receipt_sha256 && window.operating_point_receipt_sha256 == case.operating_point_receipt_sha256, "operating-point receipt differs"),
            (coh.service_fingerprint_sha256 == case.service_fingerprint_sha256 && window.service_fingerprint_sha256 == case.service_fingerprint_sha256, "service fingerprint differs"),
            (coh.environment_fingerprint_sha256 == case.environment_fingerprint_sha256 && window.environment_fingerprint_sha256 == case.environment_fingerprint_sha256, "environment fingerprint differs"),
            (coh.timebase_binding_receipt_sha256 == case.timebase_binding_receipt_sha256 && window.timebase_receipt_sha256 == case.timebase_binding_receipt_sha256, "shared timebase binding differs"),
            (coh.synchronization_receipt_sha256 == case.synchronization_receipt_sha256 && window.acoustic_cavitation_sync_receipt_sha256 == case.synchronization_receipt_sha256, "synchronization receipt differs"),
            (coh.shared_timebase_id == case.shared_timebase_id, "shared timebase identity differs"),
            (coh.pressure_instrument_id == case.hydrophone_instrument_id && coh.pressure_channel_id == case.hydrophone_channel_id && coh.pressure_calibration_certificate_sha256 == case.hydrophone_calibration_certificate_sha256 && window.hydrophone_instrument_id == case.hydrophone_instrument_id && window.hydrophone_calibration_certificate_sha256 == case.hydrophone_calibration_certificate_sha256, "hydrophone identity/calibration differs"),
            (coh.vibration_instrument_id == case.vibration_instrument_id && coh.vibration_channel_id == case.vibration_channel_id && coh.vibration_calibration_certificate_sha256 == case.vibration_calibration_certificate_sha256 && coh.vibration_reference_location == case.vibration_reference_location && coh.vibration_reference_orientation == case.vibration_reference_orientation, "vibration reference identity/calibration/location differs"),
            (coh.evidence_kind == case.evidence_kind, "coherence evidence kind differs"),
            (coh.estimator_fingerprint_sha256 == case.coherence_estimator_fingerprint_sha256, "coherence estimator configuration differs"),
            (coh.cross_spectrum_convention == case.cross_spectrum_convention, "cross-spectrum convention differs"),
        ];
        for (ok, reason) in checks {
            if !ok {
                incomparable = true;
                reasons.push(format!("{}: {reason}", window.window_id));
            }
        }
        let phase_unresolved_in_bands = coh.bands.iter().any(|band| band.phase_status == "phase_unresolved");
        let uncertainty_unresolved_in_bands = coh.bands.iter().any(|band| band.uncertainty_status == "coherent_residual_interval_unresolved");
        match coh.disposition.as_str() {
            "COHERENCE_ESTIMATE_ONLY" if !phase_unresolved_in_bands && !uncertainty_unresolved_in_bands => {}
            "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED" if phase_unresolved_in_bands && !uncertainty_unresolved_in_bands => {}
            "UNCERTAINTY_UNRESOLVED" if uncertainty_unresolved_in_bands => {
                reasons.push(format!("{}: coherence uncertainty is unresolved", window.window_id));
            }
            "COHERENCE_ESTIMATE_ONLY" | "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED" | "UNCERTAINTY_UNRESOLVED" => {
                reasons.push(format!("{}: coherence report disposition conflicts with per-band phase/uncertainty status", window.window_id));
            }
            "INVALID" => reasons.push(format!("{}: parent coherence report is invalid", window.window_id)),
            _ => reasons.push(format!("{}: parent coherence report is incomplete or unsupported", window.window_id)),
        }
    }
    (incomparable, reasons)
}

fn cohort_summary(samples: &[Sample]) -> Option<CohortSummary> {
    if samples.is_empty() {
        return None;
    }
    let mut coherence: Vec<f64> = samples.iter().map(|s| s.coherence).collect();
    coherence.sort_by(f64::total_cmp);
    let mut coherent: Vec<f64> = samples.iter().map(|s| s.coherent_psd).collect();
    coherent.sort_by(f64::total_cmp);
    let mut residual: Vec<f64> = samples.iter().map(|s| s.residual_psd).collect();
    residual.sort_by(f64::total_cmp);
    let mean_values = |f: fn(&Sample) -> f64| mean(&samples.iter().map(f).collect::<Vec<_>>());
    Some(CohortSummary {
        window_count: samples.len(),
        mean_coherence: mean_values(|s| s.coherence)?,
        median_coherence: quantile(&coherence, 0.5)?,
        p05_coherence: quantile(&coherence, 0.05)?,
        p95_coherence: quantile(&coherence, 0.95)?,
        mean_coherence_marginal_lower_bound: mean_values(|s| s.coherence_lower),
        mean_coherence_marginal_upper_bound: mean_values(|s| s.coherence_upper),
        mean_pressure_psd_pa2_per_hz: mean_values(|s| s.pressure_psd)?,
        mean_coherent_pressure_psd_pa2_per_hz: mean_values(|s| s.coherent_psd)?,
        p05_coherent_pressure_psd_pa2_per_hz: quantile(&coherent, 0.05)?,
        p95_coherent_pressure_psd_pa2_per_hz: quantile(&coherent, 0.95)?,
        mean_coherent_pressure_psd_marginal_lower_bound: mean_values(|s| s.coherent_lower),
        mean_coherent_pressure_psd_marginal_upper_bound: mean_values(|s| s.coherent_upper),
        mean_residual_pressure_psd_pa2_per_hz: mean_values(|s| s.residual_psd)?,
        p05_residual_pressure_psd_pa2_per_hz: quantile(&residual, 0.05)?,
        p95_residual_pressure_psd_pa2_per_hz: quantile(&residual, 0.95)?,
        mean_residual_pressure_psd_marginal_lower_bound: mean_values(|s| s.residual_lower),
        mean_residual_pressure_psd_marginal_upper_bound: mean_values(|s| s.residual_upper),
        marginal_uncertainty_coverage_level: samples.iter().map(|s| s.coverage).fold(f64::INFINITY, f64::min),
        uncertainty_interpretation: "mean_of_per_window_marginal_bounds_not_a_joint_confidence_interval",
    })
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let invalid_input = !validation_errors.is_empty();
    let (has_incomparable_identity, mut reasons) = validate_cross_record_identity(case);
    reasons.extend(validation_errors);

    let mut accumulators: BTreeMap<String, RunAccumulator> = BTreeMap::new();
    let mut band_grid: BTreeMap<String, f64> = BTreeMap::new();
    let mut phase_unresolved_total = 0usize;

    if !invalid_input {
        for window in &case.windows {
            let run = accumulators.entry(window.run_id.clone()).or_default();
            run.total_windows += 1;
            if !window.acoustic_cavitation_alignment_eligible {
                run.alignment_ineligible += 1;
                continue;
            }
            let Some(state) = window.cavitation_state_for_conditioning else {
                run.unknown_state += 1;
                continue;
            };
            if window.coherence.disposition == "UNCERTAINTY_UNRESOLVED" {
                run.uncertainty_unresolved += 1;
                continue;
            }
            if window.coherence.disposition == "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED" {
                run.phase_unresolved += 1;
                phase_unresolved_total += 1;
            }
            let cohort = if state { &mut run.present } else { &mut run.absent };
            if state { run.present_windows += 1; } else { run.absent_windows += 1; }
            for band in &window.coherence.bands {
                if let Some(existing) = band_grid.get(&band.band_id) {
                    let rel = (existing - band.frequency_hz).abs() / existing.max(1.0);
                    if rel > case.frequency_match_tolerance_fraction {
                        reasons.push(format!("{}: frequency for band {} differs across joined windows", window.window_id, band.band_id));
                    }
                } else {
                    band_grid.insert(band.band_id.clone(), band.frequency_hz);
                }
                cohort.entry(band.band_id.clone()).or_default().push(Sample {
                    coherence: band.coherence_magnitude_squared,
                    coherence_lower: band.coherence_lower_bound.unwrap_or(band.coherence_magnitude_squared),
                    coherence_upper: band.coherence_upper_bound.unwrap_or(band.coherence_magnitude_squared),
                    pressure_psd: band.pressure_psd_pa2_per_hz,
                    coherent_psd: band.vibration_reference_coherent_pressure_psd_pa2_per_hz,
                    coherent_lower: band.vibration_reference_coherent_pressure_psd_lower_bound.unwrap_or(band.vibration_reference_coherent_pressure_psd_pa2_per_hz),
                    coherent_upper: band.vibration_reference_coherent_pressure_psd_upper_bound.unwrap_or(band.vibration_reference_coherent_pressure_psd_pa2_per_hz),
                    residual_psd: band.non_coherent_residual_pressure_psd_pa2_per_hz,
                    residual_lower: band.non_coherent_residual_pressure_psd_lower_bound.unwrap_or(band.non_coherent_residual_pressure_psd_pa2_per_hz),
                    residual_upper: band.non_coherent_residual_pressure_psd_upper_bound.unwrap_or(band.non_coherent_residual_pressure_psd_pa2_per_hz),
                    coverage: band.marginal_uncertainty_coverage_level,
                });
            }
        }
    }

    let run_ids: Vec<String> = accumulators.keys().cloned().collect();
    let mut run_reports = Vec::new();
    let mut per_band_run_reports: BTreeMap<String, Vec<RunBandReport>> = BTreeMap::new();
    let mut runs_with_both_cohorts = 0usize;
    let mut insufficient_runs = false;

    for run_id in run_ids {
        let acc = accumulators.remove(&run_id).expect("run accumulator exists");
        let has_both = acc.present_windows >= case.minimum_windows_per_state_per_run
            && acc.absent_windows >= case.minimum_windows_per_state_per_run;
        if has_both { runs_with_both_cohorts += 1; } else { insufficient_runs = true; }
        let mut band_reports = Vec::new();
        for (band_id, frequency_hz) in &band_grid {
            let present = acc.present.get(band_id).and_then(|values| cohort_summary(values));
            let absent = acc.absent.get(band_id).and_then(|values| cohort_summary(values));
            let delta_coherence = match (&present, &absent) {
                (Some(p), Some(a)) => Some(p.mean_coherence - a.mean_coherence),
                _ => None,
            };
            let delta_coherent = match (&present, &absent) {
                (Some(p), Some(a)) => Some(p.mean_coherent_pressure_psd_pa2_per_hz - a.mean_coherent_pressure_psd_pa2_per_hz),
                _ => None,
            };
            let delta_residual = match (&present, &absent) {
                (Some(p), Some(a)) => Some(p.mean_residual_pressure_psd_pa2_per_hz - a.mean_residual_pressure_psd_pa2_per_hz),
                _ => None,
            };
            let report = RunBandReport {
                band_id: band_id.clone(),
                frequency_hz: *frequency_hz,
                cavitation_present: present,
                cavitation_absent: absent,
                present_minus_absent_mean_coherence: delta_coherence,
                present_minus_absent_mean_coherent_pressure_psd_pa2_per_hz: delta_coherent,
                present_minus_absent_mean_residual_pressure_psd_pa2_per_hz: delta_residual,
            };
            if has_both {
                per_band_run_reports.entry(band_id.clone()).or_default().push(report.clone());
            }
            band_reports.push(report);
        }
        run_reports.push(RunReport {
            run_id,
            total_windows: acc.total_windows,
            cavitation_present_windows: acc.present_windows,
            cavitation_absent_windows: acc.absent_windows,
            excluded_unknown_cavitation_state: acc.unknown_state,
            excluded_alignment_ineligible: acc.alignment_ineligible,
            excluded_uncertainty_unresolved: acc.uncertainty_unresolved,
            phase_unresolved_windows: acc.phase_unresolved,
            band_reports,
        });
    }

    let mut aggregate_band_reports = Vec::new();
    for (band_id, reports) in per_band_run_reports {
        let paired: Vec<&RunBandReport> = reports.iter()
            .filter(|r| r.present_minus_absent_mean_coherence.is_some())
            .collect();
        let p_coh: Vec<f64> = paired.iter().filter_map(|r| r.cavitation_present.as_ref().map(|x| x.mean_coherence)).collect();
        let a_coh: Vec<f64> = paired.iter().filter_map(|r| r.cavitation_absent.as_ref().map(|x| x.mean_coherence)).collect();
        let d_coh: Vec<f64> = paired.iter().filter_map(|r| r.present_minus_absent_mean_coherence).collect();
        let p_psd: Vec<f64> = paired.iter().filter_map(|r| r.cavitation_present.as_ref().map(|x| x.mean_coherent_pressure_psd_pa2_per_hz)).collect();
        let a_psd: Vec<f64> = paired.iter().filter_map(|r| r.cavitation_absent.as_ref().map(|x| x.mean_coherent_pressure_psd_pa2_per_hz)).collect();
        let d_psd: Vec<f64> = paired.iter().filter_map(|r| r.present_minus_absent_mean_coherent_pressure_psd_pa2_per_hz).collect();
        let p_res: Vec<f64> = paired.iter().filter_map(|r| r.cavitation_present.as_ref().map(|x| x.mean_residual_pressure_psd_pa2_per_hz)).collect();
        let a_res: Vec<f64> = paired.iter().filter_map(|r| r.cavitation_absent.as_ref().map(|x| x.mean_residual_pressure_psd_pa2_per_hz)).collect();
        let d_res: Vec<f64> = paired.iter().filter_map(|r| r.present_minus_absent_mean_residual_pressure_psd_pa2_per_hz).collect();
        let frequency_hz = band_grid.get(&band_id).copied().unwrap_or_default();
        aggregate_band_reports.push(AggregateBandReport {
            band_id,
            frequency_hz,
            independent_runs_with_both_cohorts: paired.len(),
            cavitation_present_equal_run_weighted_mean_coherence: mean_option(&p_coh),
            cavitation_absent_equal_run_weighted_mean_coherence: mean_option(&a_coh),
            present_minus_absent_equal_run_weighted_mean_coherence: mean_option(&d_coh),
            between_run_sd_of_coherence_delta: sample_sd(&d_coh),
            cavitation_present_equal_run_weighted_mean_coherent_psd_pa2_per_hz: mean_option(&p_psd),
            cavitation_absent_equal_run_weighted_mean_coherent_psd_pa2_per_hz: mean_option(&a_psd),
            present_minus_absent_equal_run_weighted_mean_coherent_psd_pa2_per_hz: mean_option(&d_psd),
            cavitation_present_equal_run_weighted_mean_residual_psd_pa2_per_hz: mean_option(&p_res),
            cavitation_absent_equal_run_weighted_mean_residual_psd_pa2_per_hz: mean_option(&a_res),
            present_minus_absent_equal_run_weighted_mean_residual_psd_pa2_per_hz: mean_option(&d_res),
            weighting: "equal_weight_per_independent_run_after_within_run_summary",
            interpretation: "descriptive_conditioned_association_not_source_attribution_or_causal_effect",
        });
    }

    let invalid_coherence_parent = reasons.iter().any(|reason| {
        reason.contains("parent coherence report is invalid")
            || reason.contains("disposition conflicts with per-band phase status")
    });
    let frequency_incomparable = reasons.iter().any(|reason| reason.contains("frequency for band"));
    let unresolved_parent = reasons.iter().any(|reason| {
        reason.contains("coherence uncertainty is unresolved")
            || reason.contains("parent coherence report is incomplete")
    });
    let evidence_incomplete = case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || case.intermittency_disposition != "DISTRIBUTIONAL_SUMMARY_ONLY"
        || insufficient_runs
        || runs_with_both_cohorts < case.minimum_independent_runs
        || run_reports.iter().any(|r| r.excluded_unknown_cavitation_state > 0 || r.excluded_alignment_ineligible > 0);
    let disposition = if invalid_input || invalid_coherence_parent {
        "INVALID"
    } else if has_incomparable_identity || frequency_incomparable {
        "INCOMPARABLE"
    } else if unresolved_parent || evidence_incomplete {
        "INCOMPLETE"
    } else {
        "CONDITIONED_SUMMARY_ONLY"
    };
    if invalid_input {
        reasons.push("one or more joined input records are malformed".to_string());
    }

    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot support a measured cavitation-conditioned coherence result".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source data rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source evidence has not been independently reviewed".to_string());
    }
    if phase_unresolved_total > 0 {
        reasons.push(format!("{phase_unresolved_total} windows permit magnitude coherence only; phase is unresolved"));
    }
    if runs_with_both_cohorts < case.minimum_independent_runs {
        reasons.push("insufficient independent runs contain both cavitation-present and cavitation-absent cohorts".to_string());
    }

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        source_dataset_id: case.source.dataset_id.clone(),
        source_citation: case.source.citation.clone(),
        source_data_sha256: case.source.raw_data_sha256.clone(),
        intermittency_report_case_id: case.intermittency_report_case_id.clone(),
        intermittency_report_sha256: case.intermittency_report_sha256.clone(),
        evidence_kind: match case.evidence_kind { EvidenceKind::Synthetic => "synthetic", EvidenceKind::Measured => "measured" },
        vessel_id: case.vessel_id.clone(),
        propeller_revision: case.propeller_revision.clone(),
        operating_condition_id: case.operating_condition_id.clone(),
        operating_point_receipt_sha256: case.operating_point_receipt_sha256.clone(),
        service_fingerprint_sha256: case.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: case.environment_fingerprint_sha256.clone(),
        timebase_binding_receipt_sha256: case.timebase_binding_receipt_sha256.clone(),
        synchronization_receipt_sha256: case.synchronization_receipt_sha256.clone(),
        hydrophone_instrument_id: case.hydrophone_instrument_id.clone(),
        vibration_instrument_id: case.vibration_instrument_id.clone(),
        vibration_reference_location: case.vibration_reference_location.clone(),
        vibration_reference_orientation: case.vibration_reference_orientation.clone(),
        disposition,
        claim_ceiling: "window_conditioned_descriptive_association_of_magnitude_coherence_and_pressure_power_with_declared_accelerometer_reference_no_unique_source_causal_or_ecological_claim",
        run_reports,
        aggregate_band_reports,
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-cavitation-conditioned-coherence-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-cavitation-conditioned-coherence-audit <case.json>");
        process::exit(64);
    }
    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            process::exit(64);
        }
    };
    let case: Case = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid case JSON: {error}");
            process::exit(64);
        }
    };
    let report = audit(&case);
    println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition {
        "CONDITIONED_SUMMARY_ONLY" => {}
        "INVALID" | "INCOMPARABLE" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: usize) -> String {
        format!("{seed:064x}")
    }

    fn band(coherence: f64, coherent: f64) -> CoherenceBand {
        let pressure = 1.0;
        CoherenceBand {
            band_id: "narrowband-100Hz".to_string(),
            frequency_hz: 100.0,
            pressure_psd_pa2_per_hz: pressure,
            coherence_magnitude_squared: coherence,
            coherence_lower_bound: Some((coherence - 0.05).max(0.0)),
            coherence_upper_bound: Some((coherence + 0.05).min(1.0)),
            vibration_reference_coherent_pressure_psd_pa2_per_hz: coherent,
            vibration_reference_coherent_pressure_psd_lower_bound: Some((coherent - 0.05).max(0.0)),
            vibration_reference_coherent_pressure_psd_upper_bound: Some((coherent + 0.05).min(1.0)),
            non_coherent_residual_pressure_psd_pa2_per_hz: pressure - coherent,
            non_coherent_residual_pressure_psd_lower_bound: Some((pressure - coherent - 0.05).max(0.0)),
            non_coherent_residual_pressure_psd_upper_bound: Some((pressure - coherent + 0.05).min(1.0)),
            uncertainty_status: "marginal_interval_envelope_available".to_string(),
            marginal_uncertainty_coverage_level: 0.95,
            phase_status: "phase_resolved_within_declared_limit".to_string(),
        }
    }

    fn window(run_index: usize, index: usize, present: bool, coherence: f64) -> JoinedWindow {
        let start = index as f64;
        let trace_seed = 10_000 + run_index * 100 + index;
        let coh = CoherenceWindow {
            case_id: format!("coherence-case-{run_index}-{index}"),
            report_sha256: digest(40_000 + run_index * 100 + index),
            evidence_kind: EvidenceKind::Synthetic,
            disposition: "COHERENCE_ESTIMATE_ONLY".to_string(),
            source_dataset_id: "synthetic-marine".to_string(),
            source_data_sha256: digest(1),
            vessel_id: "vessel-1".to_string(),
            propeller_revision: "prop-1".to_string(),
            operating_condition_id: "J0.8-sigma1".to_string(),
            operating_point_receipt_sha256: digest(2),
            service_fingerprint_sha256: digest(3),
            environment_fingerprint_sha256: digest(4),
            timebase_binding_receipt_sha256: digest(5),
            synchronization_receipt_sha256: digest(6),
            shared_timebase_id: "clock-1".to_string(),
            pressure_instrument_id: "hydrophone-1".to_string(),
            pressure_channel_id: "pressure-A".to_string(),
            pressure_calibration_certificate_sha256: digest(7),
            pressure_raw_trace_sha256: digest(trace_seed),
            vibration_instrument_id: "accelerometer-1".to_string(),
            vibration_channel_id: "accel-A".to_string(),
            vibration_calibration_certificate_sha256: digest(8),
            vibration_reference_location: "stern-bearing".to_string(),
            vibration_reference_orientation: "radial".to_string(),
            vibration_raw_trace_sha256: digest(50_000 + run_index * 100 + index),
            pressure_window_start_s: start,
            pressure_window_end_s: start + 1.0,
            sample_rate_hz: 48_000.0,
            sample_count: 48_000,
            estimator_fingerprint_sha256: digest(9),
            cross_spectrum_convention: "pressure_times_conjugate_acceleration".to_string(),
            bands: vec![band(coherence, coherence)],
        };
        JoinedWindow {
            run_id: format!("run-{run_index}"),
            window_id: format!("window-{run_index}-{index}"),
            start_time_s: start,
            end_time_s: start + 1.0,
            acoustic_trace_sha256: digest(trace_seed),
            cavitation_observation_sha256: digest(60_000 + run_index * 100 + index),
            cavitation_annotation_raw: Some(present),
            cavitation_state_for_conditioning: Some(present),
            acoustic_cavitation_alignment_offset_s: 0.0,
            acoustic_cavitation_alignment_expanded_uncertainty_s: 0.001,
            acoustic_cavitation_alignment_eligible: true,
            timebase_receipt_sha256: digest(5),
            acoustic_cavitation_sync_receipt_sha256: digest(6),
            operating_point_receipt_sha256: digest(2),
            service_fingerprint_sha256: digest(3),
            environment_fingerprint_sha256: digest(4),
            hydrophone_instrument_id: "hydrophone-1".to_string(),
            hydrophone_calibration_certificate_sha256: digest(7),
            coherence: coh,
        }
    }

    fn case() -> Case {
        let mut windows = Vec::new();
        for run in 0..2 {
            windows.push(window(run, 0, false, 0.2));
            windows.push(window(run, 1, true, 0.8));
            windows.push(window(run, 2, false, 0.3));
            windows.push(window(run, 3, true, 0.7));
        }
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-conditioned-coherence-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-marine".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest(1),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            intermittency_report_case_id: "intermittency-case-001".to_string(),
            intermittency_report_sha256: digest(11),
            intermittency_disposition: "INCOMPLETE".to_string(),
            vessel_id: "vessel-1".to_string(),
            propeller_revision: "prop-1".to_string(),
            operating_condition_id: "J0.8-sigma1".to_string(),
            operating_point_receipt_sha256: digest(2),
            service_fingerprint_sha256: digest(3),
            environment_fingerprint_sha256: digest(4),
            timebase_binding_receipt_sha256: digest(5),
            synchronization_receipt_sha256: digest(6),
            shared_timebase_id: "clock-1".to_string(),
            hydrophone_instrument_id: "hydrophone-1".to_string(),
            hydrophone_channel_id: "pressure-A".to_string(),
            hydrophone_calibration_certificate_sha256: digest(7),
            vibration_instrument_id: "accelerometer-1".to_string(),
            vibration_channel_id: "accel-A".to_string(),
            vibration_reference_location: "stern-bearing".to_string(),
            vibration_reference_orientation: "radial".to_string(),
            vibration_calibration_certificate_sha256: digest(8),
            maximum_sync_alignment_uncertainty_s: 0.01,
            frequency_match_tolerance_fraction: 0.001,
            coherence_estimator_fingerprint_sha256: digest(9),
            cross_spectrum_convention: "pressure_times_conjugate_acceleration".to_string(),
            minimum_windows_per_state_per_run: 2,
            minimum_independent_runs: 2,
            windows,
        }
    }

    fn measured(input: &mut Case) {
        input.evidence_kind = EvidenceKind::Measured;
        input.source.independently_reviewed = true;
        input.intermittency_disposition = "DISTRIBUTIONAL_SUMMARY_ONLY".to_string();
        for window in &mut input.windows {
            window.coherence.evidence_kind = EvidenceKind::Measured;
        }
    }

    #[test]
    fn committed_schema_and_fixture_match_runtime_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-cavitation-conditioned-coherence-v0.1.0.schema.json"
        )).expect("committed schema is valid JSON");
        assert_eq!(schema["$schema"], "https://json-schema.org/draft/2020-12/schema");
        for field in [
            "intermittency_report_sha256",
            "coherence_estimator_fingerprint_sha256",
            "cross_spectrum_convention",
            "minimum_independent_runs",
            "windows",
        ] {
            assert!(schema["required"].as_array().unwrap().iter().any(|value| value == field));
        }
        assert_eq!(
            schema["$defs"]["coherenceBand"]["properties"]["coherence_lower_bound"]["type"],
            serde_json::json!(["number", "null"])
        );

        let fixture: Case = serde_json::from_str(include_str!(
            "../../../docs/research/marine/fixtures/synthetic-marine-cavitation-conditioned-coherence-v0.1.0.json"
        )).expect("committed synthetic fixture matches runtime input contract");
        let report = audit(&fixture);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports.len(), 2);
        assert!((report.aggregate_band_reports[0].present_minus_absent_equal_run_weighted_mean_coherence.unwrap() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn source_dataset_citation_and_digest_survive_receipt() {
        let input = case();
        let report = audit(&input);
        assert_eq!(report.source_dataset_id, input.source.dataset_id);
        assert_eq!(report.source_citation, input.source.citation);
        assert_eq!(report.source_data_sha256, input.source.raw_data_sha256);
    }

    #[test]
    fn synthetic_fixture_never_qualifies_physical_conditioned_coherence() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports.len(), 2);
        assert_eq!(report.aggregate_band_reports[0].present_minus_absent_equal_run_weighted_mean_coherence, Some(0.5));
        assert!(report.claim_ceiling.contains("no_unique_source"));
    }

    #[test]
    fn measured_balanced_groups_produce_conditioned_summary_only() {
        let mut input = case();
        measured(&mut input);
        let report = audit(&input);
        assert_eq!(report.disposition, "CONDITIONED_SUMMARY_ONLY");
        assert_eq!(report.aggregate_band_reports[0].independent_runs_with_both_cohorts, 2);
    }

    #[test]
    fn unknown_cavitation_state_is_not_coerced_to_absent() {
        let mut input = case();
        measured(&mut input);
        input.windows[0].cavitation_annotation_raw = None;
        input.windows[0].cavitation_state_for_conditioning = None;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
        assert_eq!(audit(&input).run_reports[0].excluded_unknown_cavitation_state, 1);
    }

    #[test]
    fn unaligned_raw_annotation_is_not_conditioned() {
        let mut input = case();
        measured(&mut input);
        input.windows[0].acoustic_cavitation_alignment_offset_s = 0.02;
        input.windows[0].acoustic_cavitation_alignment_eligible = false;
        input.windows[0].cavitation_state_for_conditioning = None;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports[0].excluded_alignment_ineligible, 1);
    }

    #[test]
    fn exact_pressure_raw_trace_match_is_required() {
        let mut input = case();
        input.windows[0].coherence.pressure_raw_trace_sha256 = digest(999);
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn exact_window_start_and_end_are_required() {
        let mut input = case();
        input.windows[0].coherence.pressure_window_end_s += 0.1;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn timebase_identity_mismatch_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.shared_timebase_id = "clock-2".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn synchronization_receipt_mismatch_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.synchronization_receipt_sha256 = digest(77);
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn hydrophone_calibration_mismatch_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.pressure_calibration_certificate_sha256 = digest(77);
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn vibration_instrument_mismatch_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.vibration_instrument_id = "accelerometer-2".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn operating_condition_mismatch_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.operating_point_receipt_sha256 = digest(78);
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn frequency_grid_mismatch_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.bands[0].frequency_hz = 101.0;
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn coherence_uncertainty_unresolved_is_not_eligible() {
        let mut input = case();
        measured(&mut input);
        input.windows[0].coherence.disposition = "UNCERTAINTY_UNRESOLVED".to_string();
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports[0].excluded_uncertainty_unresolved, 1);
    }

    #[test]
    fn phase_unresolved_magnitude_coherence_remains_eligible() {
        let mut input = case();
        measured(&mut input);
        input.windows[0].coherence.disposition = "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED".to_string();
        input.windows[0].coherence.bands[0].phase_status = "phase_unresolved".to_string();
        let report = audit(&input);
        assert_eq!(report.disposition, "CONDITIONED_SUMMARY_ONLY");
        assert_eq!(report.run_reports[0].phase_unresolved_windows, 1);
    }

    #[test]
    fn one_run_many_windows_does_not_satisfy_independent_run_requirement() {
        let mut input = case();
        measured(&mut input);
        input.windows.retain(|window| window.run_id == "run-0");
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn duplicate_window_identity_is_invalid() {
        let mut input = case();
        input.windows[1].window_id = input.windows[0].window_id.clone();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn reused_coherence_report_digest_is_invalid() {
        let mut input = case();
        input.windows[1].coherence.report_sha256 = input.windows[0].coherence.report_sha256.clone();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn one_continuous_vibration_trace_may_back_multiple_windowed_reports() {
        let mut input = case();
        measured(&mut input);
        let shared_trace = input.windows[0].coherence.vibration_raw_trace_sha256.clone();
        for window in input.windows.iter_mut().skip(1) {
            window.coherence.vibration_raw_trace_sha256 = shared_trace.clone();
        }
        assert_eq!(audit(&input).disposition, "CONDITIONED_SUMMARY_ONLY");
    }

    #[test]
    fn uncertainty_unresolved_parent_window_is_incomplete_not_invalid() {
        let mut input = case();
        measured(&mut input);
        let band = &mut input.windows[0].coherence.bands[0];
        band.uncertainty_status = "coherent_residual_interval_unresolved".to_string();
        band.coherence_lower_bound = None;
        band.coherence_upper_bound = None;
        band.vibration_reference_coherent_pressure_psd_lower_bound = None;
        band.vibration_reference_coherent_pressure_psd_upper_bound = None;
        band.non_coherent_residual_pressure_psd_lower_bound = None;
        band.non_coherent_residual_pressure_psd_upper_bound = None;
        input.windows[0].coherence.disposition = "UNCERTAINTY_UNRESOLVED".to_string();
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports[0].excluded_uncertainty_unresolved, 1);
    }

    #[test]
    fn unresolved_band_bounds_with_resolved_parent_disposition_are_invalid() {
        let mut input = case();
        input.windows[0].coherence.bands[0].uncertainty_status = "coherent_residual_interval_unresolved".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn invalid_parent_coherence_report_remains_invalid_in_join() {
        let mut input = case();
        input.windows[0].coherence.disposition = "INVALID".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn resolved_parent_report_requires_all_marginal_bounds() {
        let mut input = case();
        input.windows[0].coherence.bands[0].coherence_lower_bound = None;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn uncertainty_unresolved_parent_report_must_have_matching_band_status() {
        let mut input = case();
        input.windows[0].coherence.disposition = "UNCERTAINTY_UNRESOLVED".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn coherent_plus_residual_pressure_psd_must_recompose_total() {
        let mut input = case();
        input.windows[0].coherence.bands[0].non_coherent_residual_pressure_psd_pa2_per_hz = 0.8;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn coherence_must_remain_between_zero_and_one() {
        let mut input = case();
        input.windows[0].coherence.bands[0].coherence_magnitude_squared = 1.1;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn equal_run_weighting_prevents_window_count_dominance() {
        let mut input = case();
        measured(&mut input);
        for index in 0..20 {
            input.windows.push(window(0, 100 + index, true, 1.0));
        }
        let report = audit(&input);
        let band = &report.aggregate_band_reports[0];
        assert_eq!(band.independent_runs_with_both_cohorts, 2);
        // Aggregate present mean is the mean of the two run means, not all windows pooled.
        assert!((band.cavitation_present_equal_run_weighted_mean_coherence.unwrap() - 0.8636363636363636).abs() < 1.0e-12);
    }

    #[test]
    fn under_sampled_run_remains_visible_but_is_excluded_from_campaign_aggregate() {
        let mut input = case();
        measured(&mut input);
        input.windows.retain(|window| !(window.run_id == "run-0" && window.window_id == "window-0-1"));
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports.len(), 2);
        assert_eq!(report.aggregate_band_reports[0].independent_runs_with_both_cohorts, 1);
        assert!((report.aggregate_band_reports[0].cavitation_present_equal_run_weighted_mean_coherence.unwrap() - 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn swapping_annotations_moves_membership_without_changing_per_window_metrics() {
        let mut input = case();
        measured(&mut input);
        let before = audit(&input);
        let original = input.windows[0].coherence.bands[0].coherence_magnitude_squared;
        input.windows[0].cavitation_annotation_raw = Some(true);
        input.windows[0].cavitation_state_for_conditioning = Some(true);
        let after = audit(&input);
        assert_eq!(input.windows[0].coherence.bands[0].coherence_magnitude_squared, original);
        assert_ne!(
            before.run_reports[0].band_reports[0].cavitation_absent.as_ref().unwrap().window_count,
            after.run_reports[0].band_reports[0].cavitation_absent.as_ref().unwrap().window_count
        );
    }

    #[test]
    fn unknown_coherence_uncertainty_status_is_not_accepted() {
        let mut input = case();
        input.windows[0].coherence.bands[0].uncertainty_status = "unknown".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn coherence_estimator_change_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.estimator_fingerprint_sha256 = digest(88);
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn cross_spectrum_convention_change_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.cross_spectrum_convention = "acceleration_times_conjugate_pressure".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn missing_band_id_in_one_window_is_incomparable() {
        let mut input = case();
        input.windows[0].coherence.bands[0].band_id = "different-band".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn all_parent_reports_must_reference_the_same_dataset() {
        let mut input = case();
        input.windows[0].coherence.source_dataset_id = "other-dataset".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }
}
