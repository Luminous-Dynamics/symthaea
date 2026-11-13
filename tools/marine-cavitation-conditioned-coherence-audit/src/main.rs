use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const FREQUENCY_TOLERANCE_HZ: f64 = 1.0e-9;
const WINDOW_TIME_TOLERANCE_S: f64 = 1.0e-9;
const PSD_PARTITION_RELATIVE_TOLERANCE: f64 = 1.0e-8;
const UNCERTAINTY_AVAILABLE: &str = "marginal_interval_envelope_available";
const COHERENCE_ESTIMATE_ONLY: &str = "COHERENCE_ESTIMATE_ONLY";
const MAGNITUDE_ONLY: &str = "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED";

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Synthetic,
    Measured,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus {
    PublicOrPermitted,
    Unknown,
    Restricted,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    minimum_windows_per_group_per_run: usize,
    minimum_runs_per_group: usize,
    intermittency_receipt_sha256: String,
    intermittency_report: IntermittencyReport,
    coherence_windows: Vec<CoherenceWindowInput>,
}

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
struct IntermittencyReport {
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: String,
    operating_condition_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    cavitation_instrument_id: String,
    cavitation_observation_method_id: String,
    cavitation_calibration_or_validation_sha256: String,
    maximum_sync_alignment_uncertainty_s: f64,
    disposition: String,
    run_reports: Vec<IntermittencyRun>,
}

#[derive(Debug, Deserialize)]
struct IntermittencyRun {
    run_id: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    acoustic_cavitation_sync_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    window_evidence: Vec<IntermittencyWindow>,
}

#[derive(Debug, Deserialize)]
struct IntermittencyWindow {
    window_id: String,
    acoustic_trace_sha256: String,
    cavitation_observation_sha256: String,
    cavitation_annotation_raw: Option<bool>,
    cavitation_state_for_conditioning: Option<bool>,
    start_time_s: f64,
    end_time_s: f64,
    acoustic_cavitation_alignment_offset_s: f64,
    acoustic_cavitation_alignment_expanded_uncertainty_s: f64,
    acoustic_cavitation_alignment_eligible: bool,
    disposition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoherenceWindowInput {
    run_id: String,
    window_id: String,
    coherence_report_sha256: String,
    coherence_report: CoherenceReport,
}

#[derive(Debug, Deserialize)]
struct CoherenceReport {
    case_id: String,
    evidence_kind: String,
    vessel_id: String,
    propeller_revision: String,
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_binding_receipt_sha256: String,
    synchronization_receipt_sha256: String,
    pressure_instrument_id: String,
    pressure_channel_id: String,
    vibration_instrument_id: String,
    vibration_channel_id: String,
    pressure_calibration_certificate_sha256: String,
    pressure_raw_trace_sha256: String,
    shared_timebase_id: String,
    sample_rate_hz: f64,
    sample_count: u64,
    pressure_window_start_s: f64,
    pressure_window_end_s: f64,
    disposition: String,
    band_reports: Vec<CoherenceBand>,
}

#[derive(Debug, Deserialize)]
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
    phase_status: String,
    uncertainty_status: String,
}

#[derive(Debug, Clone)]
struct JoinedWindow {
    run_id: String,
    window_id: String,
    start_time_s: f64,
    end_time_s: f64,
    cavitation_state: Option<bool>,
    coherence_report_sha256: String,
    coherence_case_id: String,
    phase_status: String,
    bands: Vec<CoherenceBand>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: &'static str,
    intermittency_case_id: String,
    intermittency_receipt_sha256: String,
    operating_condition_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    pressure_instrument_id: String,
    pressure_calibration_certificate_sha256: String,
    vibration_reference_instrument_ids: Vec<String>,
    vibration_reference_channel_ids: Vec<String>,
    shared_timebase_ids: Vec<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    run_reports: Vec<RunReport>,
    band_reports: Vec<BandAggregate>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RunReport {
    run_id: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    acoustic_cavitation_sync_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    total_joined_windows: usize,
    cavitation_present_windows: usize,
    cavitation_absent_windows: usize,
    cavitation_unknown_windows: usize,
    excluded_windows: usize,
    eligible_for_magnitude_coherence_windows: usize,
    phase_resolved_windows: usize,
    phase_unresolved_windows: usize,
    uncertainty_unresolved_windows: usize,
    run_band_summaries: Vec<RunBandSummary>,
    joined_window_receipts: Vec<JoinedWindowReceipt>,
}

#[derive(Debug, Serialize)]
struct JoinedWindowReceipt {
    window_id: String,
    start_time_s: f64,
    end_time_s: f64,
    coherence_case_id: String,
    coherence_report_sha256: String,
    acoustic_trace_sha256: String,
    cavitation_observation_sha256: String,
    cavitation_annotation_raw: Option<bool>,
    cavitation_state_for_conditioning: Option<bool>,
    alignment_eligible: bool,
    coherence_disposition: String,
    phase_status_by_band: Vec<BandPhaseStatus>,
}

#[derive(Debug, Serialize)]
struct BandPhaseStatus {
    band_id: String,
    phase_status: String,
    uncertainty_status: String,
}

#[derive(Debug, Serialize)]
struct GroupStats {
    window_count: usize,
    mean_coherence_magnitude_squared: Option<f64>,
    p05_coherence_magnitude_squared: Option<f64>,
    p95_coherence_magnitude_squared: Option<f64>,
    mean_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    p05_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    p95_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    mean_residual_pressure_psd_pa2_per_hz: Option<f64>,
    p05_residual_pressure_psd_pa2_per_hz: Option<f64>,
    p95_residual_pressure_psd_pa2_per_hz: Option<f64>,
    mean_coherent_pressure_psd_lower_bound_pa2_per_hz: Option<f64>,
    mean_coherent_pressure_psd_upper_bound_pa2_per_hz: Option<f64>,
    mean_residual_pressure_psd_lower_bound_pa2_per_hz: Option<f64>,
    mean_residual_pressure_psd_upper_bound_pa2_per_hz: Option<f64>,
    uncertainty_interpretation: &'static str,
}

#[derive(Debug, Serialize)]
struct RunBandSummary {
    band_id: String,
    frequency_hz: f64,
    cavitation_present: GroupStats,
    cavitation_absent: GroupStats,
    present_minus_absent_mean_coherence: Option<f64>,
    present_minus_absent_mean_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    present_minus_absent_mean_residual_pressure_psd_pa2_per_hz: Option<f64>,
    interpretation: &'static str,
}

#[derive(Debug, Serialize)]
struct BandAggregate {
    band_id: String,
    frequency_hz: f64,
    independent_runs_with_present_group: usize,
    independent_runs_with_absent_group: usize,
    equal_run_weighted_present_mean_coherence: Option<f64>,
    equal_run_weighted_absent_mean_coherence: Option<f64>,
    equal_run_weighted_present_minus_absent_coherence: Option<f64>,
    equal_run_weighted_present_mean_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    equal_run_weighted_absent_mean_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    equal_run_weighted_present_minus_absent_coherent_pressure_psd_pa2_per_hz: Option<f64>,
    equal_run_weighted_present_mean_residual_pressure_psd_pa2_per_hz: Option<f64>,
    equal_run_weighted_absent_mean_residual_pressure_psd_pa2_per_hz: Option<f64>,
    equal_run_weighted_present_minus_absent_residual_pressure_psd_pa2_per_hz: Option<f64>,
    interpretation: &'static str,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * fraction)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn validate_coherence_band(band: &CoherenceBand, report_case: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if band.band_id.trim().is_empty()
        || !finite_positive(band.frequency_hz)
        || !finite_positive(band.pressure_psd_pa2_per_hz)
        || !band.coherence_magnitude_squared.is_finite()
        || !(0.0..=1.0).contains(&band.coherence_magnitude_squared)
        || !finite_nonnegative(band.vibration_reference_coherent_pressure_psd_pa2_per_hz)
        || !finite_nonnegative(band.non_coherent_residual_pressure_psd_pa2_per_hz)
    {
        errors.push(format!("coherence report {report_case} has invalid band identity, frequency, coherence or PSD"));
    }
    for (name, bound) in [
        ("coherence lower", band.coherence_lower_bound),
        ("coherence upper", band.coherence_upper_bound),
        ("coherent PSD lower", band.vibration_reference_coherent_pressure_psd_lower_bound),
        ("coherent PSD upper", band.vibration_reference_coherent_pressure_psd_upper_bound),
        ("residual PSD lower", band.non_coherent_residual_pressure_psd_lower_bound),
        ("residual PSD upper", band.non_coherent_residual_pressure_psd_upper_bound),
    ] {
        if bound.is_some_and(|value| !finite_nonnegative(value)) {
            errors.push(format!("coherence report {report_case} has invalid {name} bound"));
        }
    }
    if band.coherence_lower_bound.is_some_and(|x| x > band.coherence_magnitude_squared)
        || band.coherence_upper_bound.is_some_and(|x| x < band.coherence_magnitude_squared)
        || band.coherence_lower_bound.is_some_and(|x| x > 1.0)
        || band.coherence_upper_bound.is_some_and(|x| x > 1.0)
        || band.vibration_reference_coherent_pressure_psd_lower_bound
            .is_some_and(|x| x > band.vibration_reference_coherent_pressure_psd_pa2_per_hz)
        || band.vibration_reference_coherent_pressure_psd_upper_bound
            .is_some_and(|x| x < band.vibration_reference_coherent_pressure_psd_pa2_per_hz)
        || band.non_coherent_residual_pressure_psd_lower_bound
            .is_some_and(|x| x > band.non_coherent_residual_pressure_psd_pa2_per_hz)
        || band.non_coherent_residual_pressure_psd_upper_bound
            .is_some_and(|x| x < band.non_coherent_residual_pressure_psd_pa2_per_hz)
    {
        errors.push(format!("coherence report {report_case} has uncertainty bounds that do not enclose the point estimate"));
    }
    let partition_sum = band.vibration_reference_coherent_pressure_psd_pa2_per_hz
        + band.non_coherent_residual_pressure_psd_pa2_per_hz;
    let partition_relative_error = (partition_sum - band.pressure_psd_pa2_per_hz).abs()
        / band.pressure_psd_pa2_per_hz.max(f64::MIN_POSITIVE);
    if partition_relative_error > PSD_PARTITION_RELATIVE_TOLERANCE {
        errors.push(format!("coherence report {report_case} coherent plus residual PSD does not conserve the pressure PSD"));
    }
    errors
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    if case.case_id.trim().is_empty()
        || case.source.dataset_id.trim().is_empty()
        || case.source.citation.trim().is_empty()
    {
        errors.push("case and source dataset identities/citation are required".to_string());
    }
    for (name, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("intermittency receipt", &case.intermittency_receipt_sha256),
        ("intermittency report raw data", &case.intermittency_report.source_data_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{name} digest must be SHA-256 hex"));
        }
    }
    if case.minimum_windows_per_group_per_run == 0 || case.minimum_runs_per_group < 2 {
        errors.push("minimum windows per group must be positive and minimum independent runs per group must be at least 2".to_string());
    }
    if case.source.raw_data_sha256 != case.intermittency_report.source_data_sha256
        || case.source.dataset_id != case.intermittency_report.source_dataset_id
    {
        errors.push("outer source identity does not match the embedded intermittency report source identity".to_string());
    }
    if !finite_positive(case.intermittency_report.maximum_sync_alignment_uncertainty_s) {
        errors.push("intermittency report has an invalid maximum sync alignment bound".to_string());
    }
    if case.intermittency_report.run_reports.is_empty() {
        errors.push("intermittency report has no run reports".to_string());
    }

    let mut run_ids = BTreeSet::new();
    let mut window_keys = BTreeSet::new();
    let mut expected_windows = BTreeMap::new();
    for run in &case.intermittency_report.run_reports {
        if run.run_id.trim().is_empty() || !run_ids.insert(run.run_id.clone()) {
            errors.push(format!("empty or duplicate intermittency run_id {}", run.run_id));
        }
        for (name, digest) in [
            ("run timebase receipt", &run.timebase_receipt_sha256),
            ("run operating-point receipt", &run.operating_point_receipt_sha256),
            ("run video/acoustic synchronization receipt", &run.acoustic_cavitation_sync_receipt_sha256),
            ("run service fingerprint", &run.service_fingerprint_sha256),
            ("run environment fingerprint", &run.environment_fingerprint_sha256),
            ("run hydrophone calibration", &run.calibration_certificate_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("run {} has malformed {name} digest", run.run_id));
            }
        }
        if run.service_fingerprint_sha256 != case.intermittency_report.service_fingerprint_sha256
            || run.environment_fingerprint_sha256 != case.intermittency_report.environment_fingerprint_sha256
            || run.hydrophone_instrument_id != case.intermittency_report.hydrophone_instrument_id
            || run.calibration_certificate_sha256 != case.intermittency_report.calibration_certificate_sha256
        {
            errors.push(format!("run {} service/environment/hydrophone calibration differs from frozen intermittency case", run.run_id));
        }
        for window in &run.window_evidence {
            let key = (run.run_id.clone(), window.window_id.clone());
            if window.window_id.trim().is_empty() || !window_keys.insert(key.clone()) {
                errors.push(format!("run {} has empty or duplicate window_id {}", run.run_id, window.window_id));
            }
            if !is_sha256(&window.acoustic_trace_sha256)
                || !is_sha256(&window.cavitation_observation_sha256)
                || !window.start_time_s.is_finite()
                || !window.end_time_s.is_finite()
                || window.end_time_s <= window.start_time_s
                || !window.acoustic_cavitation_alignment_offset_s.is_finite()
                || !finite_nonnegative(window.acoustic_cavitation_alignment_expanded_uncertainty_s)
            {
                errors.push(format!("run {} window {} has invalid identity, digest, timing or alignment metadata", run.run_id, window.window_id));
            }
            let alignment_bound = window.acoustic_cavitation_alignment_offset_s.abs()
                + window.acoustic_cavitation_alignment_expanded_uncertainty_s;
            let alignment_expected = alignment_bound
                <= case.intermittency_report.maximum_sync_alignment_uncertainty_s;
            if alignment_expected != window.acoustic_cavitation_alignment_eligible {
                errors.push(format!("run {} window {} alignment eligibility flag conflicts with its bound", run.run_id, window.window_id));
            }
            if window.acoustic_cavitation_alignment_eligible
                && window.cavitation_annotation_raw != window.cavitation_state_for_conditioning
            {
                errors.push(format!("run {} window {} conditioned state differs from raw annotation despite eligible alignment", run.run_id, window.window_id));
            }
            if !window.acoustic_cavitation_alignment_eligible && window.cavitation_state_for_conditioning.is_some() {
                errors.push(format!("run {} window {} has a conditioned state despite ineligible alignment", run.run_id, window.window_id));
            }
            expected_windows.insert(key, window);
        }
    }

    let mut coherence_case_ids = BTreeSet::new();
    let mut coherence_report_digests = BTreeSet::new();
    let mut shared_timebase_ids = BTreeSet::new();
    let mut vessel_ids = BTreeSet::new();
    let mut propeller_revisions = BTreeSet::new();
    let mut vibration_instrument_ids = BTreeSet::new();
    let mut vibration_channel_ids = BTreeSet::new();
    let mut expected_band_signature: Option<BTreeMap<String, f64>> = None;
    let mut vibration_instruments = BTreeSet::new();
    let mut vibration_channels = BTreeSet::new();
    let mut matched_window_keys = BTreeSet::new();
    for input in &case.coherence_windows {
        let key = (input.run_id.clone(), input.window_id.clone());
        if !coherence_case_ids.insert(input.coherence_report.case_id.clone())
            || !coherence_report_digests.insert(input.coherence_report_sha256.clone())
        {
            errors.push(format!("duplicate coherence case ID/report digest for {}/{}", input.run_id, input.window_id));
        }
        if !is_sha256(&input.coherence_report_sha256) {
            errors.push(format!("coherence report for {}/{} has malformed receipt digest", input.run_id, input.window_id));
        }
        if !matched_window_keys.insert(key.clone()) {
            errors.push(format!("duplicate coherence report for {}/{}", input.run_id, input.window_id));
        }
        let Some(window) = expected_windows.get(&key) else {
            errors.push(format!("coherence report {}/{} has no matching intermittency window receipt", input.run_id, input.window_id));
            continue;
        };
        let Some(run) = case.intermittency_report.run_reports.iter().find(|run| run.run_id == input.run_id) else {
            errors.push(format!("coherence report {}/{} refers to unknown run", input.run_id, input.window_id));
            continue;
        };
        let report = &input.coherence_report;
        if report.evidence_kind != "measured" {
            errors.push(format!("coherence report {} is not measured evidence", report.case_id));
        }
        if report.disposition == "UNCERTAINTY_UNRESOLVED" {
            errors.push(format!("INCOMPLETE: coherence report {} has unresolved coherent/residual uncertainty", report.case_id));
        } else if report.disposition != COHERENCE_ESTIMATE_ONLY && report.disposition != MAGNITUDE_ONLY {
            errors.push(format!("INCOMPLETE: coherence report {} disposition {} does not permit magnitude-coherence conditioning", report.case_id, report.disposition));
        }
        for (name, digest) in [
            ("coherence report operating-point receipt", &report.operating_point_receipt_sha256),
            ("coherence report service fingerprint", &report.service_fingerprint_sha256),
            ("coherence report environment fingerprint", &report.environment_fingerprint_sha256),
            ("coherence report timebase binding receipt", &report.timebase_binding_receipt_sha256),
            ("coherence report synchronization receipt", &report.synchronization_receipt_sha256),
            ("coherence report pressure calibration", &report.pressure_calibration_certificate_sha256),
            ("coherence report pressure raw trace", &report.pressure_raw_trace_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("coherence report {} has malformed {name}", report.case_id));
            }
        }
        if report.vessel_id.trim().is_empty()
            || report.propeller_revision.trim().is_empty()
            || report.pressure_channel_id.trim().is_empty()
            || report.vibration_instrument_id.trim().is_empty()
            || report.vibration_channel_id.trim().is_empty()
            || report.shared_timebase_id.trim().is_empty()
        {
            errors.push(format!("coherence report {} lacks vessel/propeller/channel/timebase identity", report.case_id));
        }
        if report.operating_condition_id != case.intermittency_report.operating_condition_id
            || report.operating_point_receipt_sha256 != run.operating_point_receipt_sha256
            || report.service_fingerprint_sha256 != run.service_fingerprint_sha256
            || report.environment_fingerprint_sha256 != run.environment_fingerprint_sha256
            || report.timebase_binding_receipt_sha256 != run.timebase_receipt_sha256
            || report.pressure_instrument_id != run.hydrophone_instrument_id
            || report.pressure_calibration_certificate_sha256 != run.calibration_certificate_sha256
        {
            errors.push(format!("INCOMPARABLE: coherence report {} differs in operating condition, service/environment, instrument calibration or shared timebase binding", report.case_id));
        }
        if report.pressure_raw_trace_sha256 != window.acoustic_trace_sha256
            || (report.pressure_window_start_s - window.start_time_s).abs() > WINDOW_TIME_TOLERANCE_S
            || (report.pressure_window_end_s - window.end_time_s).abs() > WINDOW_TIME_TOLERANCE_S
        {
            errors.push(format!("coherence report {} pressure raw-trace digest or exact window bounds do not match the intermittency receipt", report.case_id));
        }
        if !finite_positive(report.sample_rate_hz)
            || report.sample_count < 2
            || !report.pressure_window_start_s.is_finite()
            || !report.pressure_window_end_s.is_finite()
            || report.pressure_window_end_s <= report.pressure_window_start_s
        {
            errors.push(format!("coherence report {} has invalid sample window metadata", report.case_id));
        }
        shared_timebase_ids.insert(report.shared_timebase_id.clone());
        vessel_ids.insert(report.vessel_id.clone());
        propeller_revisions.insert(report.propeller_revision.clone());
        vibration_instrument_ids.insert(report.vibration_instrument_id.clone());
        vibration_channel_ids.insert(report.vibration_channel_id.clone());

        if report.band_reports.is_empty() {
            errors.push(format!("coherence report {} has no band reports", report.case_id));
        }
        let signature: BTreeMap<String, f64> = report.band_reports.iter()
            .map(|band| (band.band_id.clone(), band.frequency_hz))
            .collect();
        if signature.len() != report.band_reports.len() {
            errors.push(format!("coherence report {} repeats a band identity", report.case_id));
        }
        if let Some(expected) = &expected_band_signature {
            if expected.len() != signature.len()
                || expected.iter().any(|(id, frequency)| {
                    signature.get(id).is_none_or(|other| (other - frequency).abs() > FREQUENCY_TOLERANCE_HZ)
                })
            {
                errors.push(format!("INCOMPARABLE: coherence report {} uses a different frequency-bin identity/grid", report.case_id));
            }
        } else {
            expected_band_signature = Some(signature);
        }
        let mut band_ids = BTreeSet::new();
        for band in &report.band_reports {
            if !band_ids.insert(band.band_id.clone()) {
                errors.push(format!("coherence report {} repeats band ID {}", report.case_id, band.band_id));
            }
            errors.extend(validate_coherence_band(band, &report.case_id));
            if band.uncertainty_status != UNCERTAINTY_AVAILABLE {
                errors.push(format!("INCOMPLETE: coherence report {} band {} has unresolved coherent/residual uncertainty", report.case_id, band.band_id));
            }
            if band.coherence_lower_bound.is_none()
                || band.coherence_upper_bound.is_none()
                || band.vibration_reference_coherent_pressure_psd_lower_bound.is_none()
                || band.vibration_reference_coherent_pressure_psd_upper_bound.is_none()
                || band.non_coherent_residual_pressure_psd_lower_bound.is_none()
                || band.non_coherent_residual_pressure_psd_upper_bound.is_none()
            {
                errors.push(format!("INCOMPLETE: coherence report {} band {} lacks required per-window marginal uncertainty bounds", report.case_id, band.band_id));
            }
        }
    }
    if matched_window_keys != window_keys {
        errors.push("coherence window receipts do not form a one-to-one cover of all intermittency windows".to_string());
    }
    if shared_timebase_ids.len() > 1 {
        errors.push("coherence reports use different shared timebase IDs within one joined campaign".to_string());
    }
    if case.coherence_windows.is_empty() {
        errors.push("no per-window coherence reports were supplied".to_string());
    }
    if case.intermittency_report.evidence_kind != "measured"
        || case.intermittency_report.disposition != "DISTRIBUTIONAL_SUMMARY_ONLY"
    {
        errors.push("intermittency report is not a measured, distributionally eligible receipt".to_string());
    }
    errors
}

fn summarize_group(windows: &[&JoinedWindow], band_id: &str) -> GroupStats {
    let bands: Vec<&CoherenceBand> = windows.iter()
        .filter_map(|window| window.bands.iter().find(|band| band.band_id == band_id))
        .collect();
    let coherence: Vec<f64> = bands.iter().map(|band| band.coherence_magnitude_squared).collect();
    let coherent: Vec<f64> = bands.iter().map(|band| band.vibration_reference_coherent_pressure_psd_pa2_per_hz).collect();
    let residual: Vec<f64> = bands.iter().map(|band| band.non_coherent_residual_pressure_psd_pa2_per_hz).collect();
    let coherent_lower: Vec<f64> = bands.iter().filter_map(|band| band.vibration_reference_coherent_pressure_psd_lower_bound).collect();
    let coherent_upper: Vec<f64> = bands.iter().filter_map(|band| band.vibration_reference_coherent_pressure_psd_upper_bound).collect();
    let residual_lower: Vec<f64> = bands.iter().filter_map(|band| band.non_coherent_residual_pressure_psd_lower_bound).collect();
    let residual_upper: Vec<f64> = bands.iter().filter_map(|band| band.non_coherent_residual_pressure_psd_upper_bound).collect();

    GroupStats {
        window_count: bands.len(),
        mean_coherence_magnitude_squared: mean(&coherence),
        p05_coherence_magnitude_squared: quantile(&coherence, 0.05),
        p95_coherence_magnitude_squared: quantile(&coherence, 0.95),
        mean_coherent_pressure_psd_pa2_per_hz: mean(&coherent),
        p05_coherent_pressure_psd_pa2_per_hz: quantile(&coherent, 0.05),
        p95_coherent_pressure_psd_pa2_per_hz: quantile(&coherent, 0.95),
        mean_residual_pressure_psd_pa2_per_hz: mean(&residual),
        p05_residual_pressure_psd_pa2_per_hz: quantile(&residual, 0.05),
        p95_residual_pressure_psd_pa2_per_hz: quantile(&residual, 0.95),
        mean_coherent_pressure_psd_lower_bound_pa2_per_hz: mean(&coherent_lower),
        mean_coherent_pressure_psd_upper_bound_pa2_per_hz: mean(&coherent_upper),
        mean_residual_pressure_psd_lower_bound_pa2_per_hz: mean(&residual_lower),
        mean_residual_pressure_psd_upper_bound_pa2_per_hz: mean(&residual_upper),
        uncertainty_interpretation: "arithmetic_mean_of_per_window_marginal_bounds_no_joint_confidence_level_or_independence_assumption",
    }
}

fn summarize_run(run_id: &str, windows: &[JoinedWindow], band_ids: &[(String, f64)]) -> RunReport {
    let present: Vec<&JoinedWindow> = windows.iter().filter(|window| window.cavitation_state == Some(true)).collect();
    let absent: Vec<&JoinedWindow> = windows.iter().filter(|window| window.cavitation_state == Some(false)).collect();
    let unknown_count = windows.iter().filter(|window| window.cavitation_state.is_none()).count();
    let phase_resolved_count = windows.iter().filter(|window| window.phase_status == "phase_resolved_within_declared_limit").count();
    let phase_unresolved_count = windows.len().saturating_sub(phase_resolved_count);
    let run_band_summaries = band_ids.iter().map(|(band_id, frequency)| {
        let present_stats = summarize_group(&present, band_id);
        let absent_stats = summarize_group(&absent, band_id);
        RunBandSummary {
            band_id: band_id.clone(),
            frequency_hz: *frequency,
            present_minus_absent_mean_coherence: present_stats.mean_coherence_magnitude_squared
                .zip(absent_stats.mean_coherence_magnitude_squared).map(|(a, b)| a - b),
            present_minus_absent_mean_coherent_pressure_psd_pa2_per_hz: present_stats.mean_coherent_pressure_psd_pa2_per_hz
                .zip(absent_stats.mean_coherent_pressure_psd_pa2_per_hz).map(|(a, b)| a - b),
            present_minus_absent_mean_residual_pressure_psd_pa2_per_hz: present_stats.mean_residual_pressure_psd_pa2_per_hz
                .zip(absent_stats.mean_residual_pressure_psd_pa2_per_hz).map(|(a, b)| a - b),
            cavitation_present: present_stats,
            cavitation_absent: absent_stats,
            interpretation: "within_run_descriptive_difference_only_not_significance_or_causal_effect",
        }
    }).collect();

    RunReport {
        run_id: run_id.to_string(),
        timebase_receipt_sha256: run.timebase_receipt_sha256.clone(),
        operating_point_receipt_sha256: run.operating_point_receipt_sha256.clone(),
        acoustic_cavitation_sync_receipt_sha256: run.acoustic_cavitation_sync_receipt_sha256.clone(),
        service_fingerprint_sha256: run.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: run.environment_fingerprint_sha256.clone(),
        total_joined_windows: windows.len(),
        cavitation_present_windows: present.len(),
        cavitation_absent_windows: absent.len(),
        cavitation_unknown_windows: unknown_count,
        excluded_windows: unknown_count,
        eligible_for_magnitude_coherence_windows: windows.len(),
        phase_resolved_windows: phase_resolved_count,
        phase_unresolved_windows: phase_unresolved_count,
        uncertainty_unresolved_windows: 0,
        run_band_summaries,
        joined_window_receipts: windows.iter().map(|window| JoinedWindowReceipt {
            window_id: window.window_id.clone(),
            start_time_s: window.start_time_s,
            end_time_s: window.end_time_s,
            coherence_case_id: window.coherence_case_id.clone(),
            coherence_report_sha256: window.coherence_report_sha256.clone(),
            acoustic_trace_sha256: window.acoustic_trace_sha256.clone(),
            cavitation_observation_sha256: window.cavitation_observation_sha256.clone(),
            cavitation_annotation_raw: window.cavitation_annotation_raw,
            cavitation_state_for_conditioning: window.cavitation_state,
            alignment_eligible: window.alignment_eligible,
            coherence_disposition: window.coherence_disposition.clone(),
            phase_status_by_band: window.bands.iter().map(|band| BandPhaseStatus {
                band_id: band.band_id.clone(),
                phase_status: band.phase_status.clone(),
                uncertainty_status: band.uncertainty_status.clone(),
            }).collect(),
        }).collect(),
    }
}

fn aggregate_band(run_reports: &[RunReport], band_id: &str, frequency_hz: f64) -> BandAggregate {
    let present_coherence: Vec<f64> = run_reports.iter()
        .filter_map(|run| run.run_band_summaries.iter().find(|x| x.band_id == band_id))
        .filter_map(|x| x.cavitation_present.mean_coherence_magnitude_squared)
        .collect();
    let absent_coherence: Vec<f64> = run_reports.iter()
        .filter_map(|run| run.run_band_summaries.iter().find(|x| x.band_id == band_id))
        .filter_map(|x| x.cavitation_absent.mean_coherence_magnitude_squared)
        .collect();
    let present_coherent: Vec<f64> = run_reports.iter()
        .filter_map(|run| run.run_band_summaries.iter().find(|x| x.band_id == band_id))
        .filter_map(|x| x.cavitation_present.mean_coherent_pressure_psd_pa2_per_hz)
        .collect();
    let absent_coherent: Vec<f64> = run_reports.iter()
        .filter_map(|run| run.run_band_summaries.iter().find(|x| x.band_id == band_id))
        .filter_map(|x| x.cavitation_absent.mean_coherent_pressure_psd_pa2_per_hz)
        .collect();
    let present_residual: Vec<f64> = run_reports.iter()
        .filter_map(|run| run.run_band_summaries.iter().find(|x| x.band_id == band_id))
        .filter_map(|x| x.cavitation_present.mean_residual_pressure_psd_pa2_per_hz)
        .collect();
    let absent_residual: Vec<f64> = run_reports.iter()
        .filter_map(|run| run.run_band_summaries.iter().find(|x| x.band_id == band_id))
        .filter_map(|x| x.cavitation_absent.mean_residual_pressure_psd_pa2_per_hz)
        .collect();

    let present_c = mean(&present_coherence);
    let absent_c = mean(&absent_coherence);
    let present_coh = mean(&present_coherent);
    let absent_coh = mean(&absent_coherent);
    let present_res = mean(&present_residual);
    let absent_res = mean(&absent_residual);

    BandAggregate {
        band_id: band_id.to_string(),
        frequency_hz,
        independent_runs_with_present_group: present_coherence.len(),
        independent_runs_with_absent_group: absent_coherence.len(),
        equal_run_weighted_present_mean_coherence: present_c,
        equal_run_weighted_absent_mean_coherence: absent_c,
        equal_run_weighted_present_minus_absent_coherence: present_c.zip(absent_c).map(|(a,b)|a-b),
        equal_run_weighted_present_mean_coherent_pressure_psd_pa2_per_hz: present_coh,
        equal_run_weighted_absent_mean_coherent_pressure_psd_pa2_per_hz: absent_coh,
        equal_run_weighted_present_minus_absent_coherent_pressure_psd_pa2_per_hz: present_coh.zip(absent_coh).map(|(a,b)|a-b),
        equal_run_weighted_present_mean_residual_pressure_psd_pa2_per_hz: present_res,
        equal_run_weighted_absent_mean_residual_pressure_psd_pa2_per_hz: absent_res,
        equal_run_weighted_present_minus_absent_residual_pressure_psd_pa2_per_hz: present_res.zip(absent_res).map(|(a,b)|a-b),
        interpretation: "equal_weighted_descriptive_mean_of_independent_run_means_no_p_value_ci_or_causal_claim",
    }
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut run_reports = Vec::new();
    let mut band_ids: Vec<(String, f64)> = Vec::new();
    let mut joined_by_run: BTreeMap<String, Vec<JoinedWindow>> = BTreeMap::new();
    let mut vibration_instruments = BTreeSet::new();
    let mut vibration_channels = BTreeSet::new();
    let mut shared_timebase_ids = BTreeSet::new();

    if validation_errors.is_empty() {
        let windows_by_key: BTreeMap<(String, String), &IntermittencyWindow> = case.intermittency_report.run_reports.iter()
            .flat_map(|run| run.window_evidence.iter().map(move |window| ((run.run_id.clone(), window.window_id.clone()), window)))
            .collect();

        for input in &case.coherence_windows {
            let report = &input.coherence_report;
            vibration_instruments.insert(report.vibration_instrument_id.clone());
            vibration_channels.insert(report.vibration_channel_id.clone());
            shared_timebase_ids.insert(report.shared_timebase_id.clone());
            for band in &report.band_reports {
                if !band_ids.iter().any(|(id, _)| id == &band.band_id) {
                    band_ids.push((band.band_id.clone(), band.frequency_hz));
                }
            }
            let (window, _) = (&windows_by_key[&(input.run_id.clone(), input.window_id.clone())], &input.coherence_report);
            let joined = JoinedWindow {
                run_id: input.run_id.clone(),
                window_id: input.window_id.clone(),
                start_time_s: window.start_time_s,
                end_time_s: window.end_time_s,
                cavitation_state: window.cavitation_state_for_conditioning,
                coherence_report_sha256: input.coherence_report_sha256.clone(),
                coherence_case_id: report.case_id.clone(),
                phase_status: if report.band_reports.iter().all(|band| band.phase_status == "phase_resolved_within_declared_limit") {
                    "phase_resolved_within_declared_limit".to_string()
                } else {
                    "phase_unresolved".to_string()
                },
                bands: report.band_reports.clone(),
            };
            joined_by_run.entry(input.run_id.clone()).or_default().push(joined);
        }

        band_ids.sort_by(|a,b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        for run in &case.intermittency_report.run_reports {
            let mut windows = joined_by_run.remove(&run.run_id).unwrap_or_default();
            windows.sort_by(|a,b| a.window_id.cmp(&b.window_id));
            run_reports.push(summarize_run(&run.run_id, &windows, &band_ids));
        }
        run_reports.sort_by(|a,b| a.run_id.cmp(&b.run_id));
    }

    let band_reports: Vec<BandAggregate> = band_ids.iter()
        .map(|(band_id, frequency)| aggregate_band(&run_reports, band_id, *frequency))
        .collect();

    let present_runs = band_reports.iter().map(|band| band.independent_runs_with_present_group).min().unwrap_or(0);
    let absent_runs = band_reports.iter().map(|band| band.independent_runs_with_absent_group).min().unwrap_or(0);
    if present_runs < case.minimum_runs_per_group {
        reasons.push("INCOMPLETE: too few independent runs contain the required cavitation-present window cohort".to_string());
    }
    if absent_runs < case.minimum_runs_per_group {
        reasons.push("INCOMPLETE: too few independent runs contain the required cavitation-absent window cohort".to_string());
    }
    for run in &run_reports {
        for band in &run.run_band_summaries {
            if band.cavitation_present.window_count < case.minimum_windows_per_group_per_run {
                reasons.push(format!("INCOMPLETE: run {} band {} has too few cavitation-present windows", run.run_id, band.band_id));
            }
            if band.cavitation_absent.window_count < case.minimum_windows_per_group_per_run {
                reasons.push(format!("INCOMPLETE: run {} band {} has too few cavitation-absent windows", run.run_id, band.band_id));
            }
        }
    }
    if shared_timebase_ids.len() > 1
        || vessel_ids.len() > 1
        || propeller_revisions.len() > 1
        || vibration_instrument_ids.len() > 1
        || vibration_channel_ids.len() > 1
    {
        reasons.push("INCOMPARABLE: coherence reports do not share one vessel/propeller/vibration-reference/shared-clock stratum".to_string());
    }
    let alignment_unknown = run_reports.iter().any(|run| run.cavitation_unknown_windows > 0);
    if alignment_unknown {
        reasons.push("INCOMPLETE: unresolved or missing cavitation annotations remain unconditioned and prevent a complete two-cohort comparison".to_string());
    }

    let has_incomparable = reasons.iter().any(|reason| reason.starts_with("INCOMPARABLE:"));
    let has_incomplete_evidence = reasons.iter().any(|reason| reason.starts_with("INCOMPLETE:"));
    let disposition = if reasons.iter().any(|reason| !reason.starts_with("INCOMPARABLE:") && !reason.starts_with("INCOMPLETE:")) {
        "INVALID"
    } else if has_incomparable {
        "INCOMPARABLE"
    } else if has_incomplete_evidence
        || case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || reasons.iter().any(|reason| reason.contains("too few independent runs") || reason.contains("too few cavitation-") || reason.contains("unresolved or missing"))
    {
        "INCOMPLETE"
    } else {
        "CONDITIONED_SUMMARY_ONLY"
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        source_dataset_id: case.source.dataset_id.clone(),
        source_data_sha256: case.source.raw_data_sha256.clone(),
        evidence_kind: match case.evidence_kind {
            EvidenceKind::Synthetic => "synthetic",
            EvidenceKind::Measured => "measured",
        },
        intermittency_case_id: case.intermittency_report.case_id.clone(),
        intermittency_receipt_sha256: case.intermittency_receipt_sha256.clone(),
        operating_condition_id: case.intermittency_report.operating_condition_id.clone(),
        service_fingerprint_sha256: case.intermittency_report.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: case.intermittency_report.environment_fingerprint_sha256.clone(),
        pressure_instrument_id: case.intermittency_report.hydrophone_instrument_id.clone(),
        pressure_calibration_certificate_sha256: case.intermittency_report.calibration_certificate_sha256.clone(),
        vibration_reference_instrument_ids: vibration_instruments.into_iter().collect(),
        vibration_reference_channel_ids: vibration_channels.into_iter().collect(),
        shared_timebase_ids: shared_timebase_ids.into_iter().collect(),
        disposition,
        claim_ceiling: "condition_stratified_descriptive_association_of_magnitude_coherence_and_reference_coherent_residual_pressure_power_only_no_unique_source_causality_significance_noise_reduction_ecological_or_regulatory_qualification",
        run_reports,
        band_reports,
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

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn intermittency_window(run_index: usize, index: usize, state: Option<bool>) -> IntermittencyWindow {
        let seed = run_index * 100 + index + 1;
        IntermittencyWindow {
            window_id: format!("window-{index}"),
            acoustic_trace_sha256: format!("{seed:064x}"),
            cavitation_observation_sha256: format!("{:064x}", seed + 10_000),
            cavitation_annotation_raw: state,
            cavitation_state_for_conditioning: state,
            start_time_s: index as f64,
            end_time_s: index as f64 + 1.0,
            acoustic_cavitation_alignment_offset_s: 0.0,
            acoustic_cavitation_alignment_expanded_uncertainty_s: 0.001,
            acoustic_cavitation_alignment_eligible: true,
            disposition: "included".to_string(),
        }
    }

    fn coherence_band(index: usize, present: bool, band_id: &str, frequency: f64) -> CoherenceBand {
        let _ = index;
        let coherence = if present { 0.8 } else { 0.2 };
        let coherent = if present { 8.0 } else { 2.0 };
        let residual = if present { 2.0 } else { 8.0 };
        CoherenceBand {
            band_id: band_id.to_string(),
            frequency_hz: frequency,
            pressure_psd_pa2_per_hz: 10.0,
            coherence_magnitude_squared: coherence,
            coherence_lower_bound: Some(coherence - 0.05),
            coherence_upper_bound: Some(coherence + 0.05),
            vibration_reference_coherent_pressure_psd_pa2_per_hz: coherent,
            vibration_reference_coherent_pressure_psd_lower_bound: Some(coherent - 0.5),
            vibration_reference_coherent_pressure_psd_upper_bound: Some(coherent + 0.5),
            non_coherent_residual_pressure_psd_pa2_per_hz: residual,
            non_coherent_residual_pressure_psd_lower_bound: Some(residual - 0.5),
            non_coherent_residual_pressure_psd_upper_bound: Some(residual + 0.5),
            phase_status: "phase_resolved_within_declared_limit".to_string(),
            uncertainty_status: UNCERTAINTY_AVAILABLE.to_string(),
        }
    }

    fn coherence_report(
        window: &IntermittencyWindow,
        run: &IntermittencyRun,
        state: bool,
        index: usize,
        band_id: &str,
        frequency: f64,
    ) -> CoherenceReport {
        CoherenceReport {
            case_id: format!("coh-{}-{}", run.run_id, window.window_id),
            evidence_kind: "measured".to_string(),
            vessel_id: "vessel-1".to_string(),
            propeller_revision: "propeller-r3".to_string(),
            operating_condition_id: "condition-j08".to_string(),
            operating_point_receipt_sha256: run.operating_point_receipt_sha256.clone(),
            service_fingerprint_sha256: run.service_fingerprint_sha256.clone(),
            environment_fingerprint_sha256: run.environment_fingerprint_sha256.clone(),
            timebase_binding_receipt_sha256: run.timebase_receipt_sha256.clone(),
            synchronization_receipt_sha256: digest('c'),
            pressure_instrument_id: run.hydrophone_instrument_id.clone(),
            pressure_channel_id: "pressure-channel-1".to_string(),
            vibration_instrument_id: "accelerometer-1".to_string(),
            vibration_channel_id: "shaft-radial-x".to_string(),
            pressure_calibration_certificate_sha256: run.calibration_certificate_sha256.clone(),
            pressure_raw_trace_sha256: window.acoustic_trace_sha256.clone(),
            shared_timebase_id: "shared-clock-1".to_string(),
            sample_rate_hz: 20_000.0,
            sample_count: 20_000,
            pressure_window_start_s: window.start_time_s,
            pressure_window_end_s: window.end_time_s,
            disposition: COHERENCE_ESTIMATE_ONLY.to_string(),
            band_reports: vec![coherence_band(index, state, band_id, frequency)],
        }
    }

    fn case() -> Case {
        let mut run_reports = Vec::new();
        let mut coherence_windows = Vec::new();
        for run_index in 0..2 {
            let run_id = format!("run-{}", run_index + 1);
            let windows: Vec<_> = (0..4)
                .map(|index| intermittency_window(run_index, index, Some(index % 2 == 0)))
                .collect();
            let run = IntermittencyRun {
                run_id: run_id.clone(),
                timebase_receipt_sha256: digest('a'),
                operating_point_receipt_sha256: digest('b'),
                acoustic_cavitation_sync_receipt_sha256: digest('d'),
                service_fingerprint_sha256: digest('e'),
                environment_fingerprint_sha256: digest('f'),
                hydrophone_instrument_id: "hydrophone-H1".to_string(),
                calibration_certificate_sha256: digest('9'),
                window_evidence: windows.clone(),
            };
            for (index, window) in windows.iter().enumerate() {
                let state = window.cavitation_state_for_conditioning.unwrap();
                let report = coherence_report(window, &run, state, index, "band-100", 100.0);
                coherence_windows.push(CoherenceWindowInput {
                    run_id: run_id.clone(),
                    window_id: window.window_id.clone(),
                    coherence_report_sha256: format!("{:064x}", 50_000 + run_index * 100 + index),
                    coherence_report: report,
                });
            }
            run_reports.push(run);
        }
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-conditioned-coherence-1".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-conditioned-coherence".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('1'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            minimum_windows_per_group_per_run: 1,
            minimum_runs_per_group: 2,
            intermittency_receipt_sha256: digest('2'),
            intermittency_report: IntermittencyReport {
                case_id: "intermittency-case-1".to_string(),
                source_dataset_id: "synthetic-conditioned-coherence".to_string(),
                source_data_sha256: digest('1'),
                evidence_kind: "measured".to_string(),
                operating_condition_id: "condition-j08".to_string(),
                service_fingerprint_sha256: digest('e'),
                environment_fingerprint_sha256: digest('f'),
                hydrophone_instrument_id: "hydrophone-H1".to_string(),
                calibration_certificate_sha256: digest('9'),
                cavitation_instrument_id: "camera-1".to_string(),
                cavitation_observation_method_id: "phase-video-annotation".to_string(),
                cavitation_calibration_or_validation_sha256: digest('8'),
                maximum_sync_alignment_uncertainty_s: 0.01,
                disposition: "DISTRIBUTIONAL_SUMMARY_ONLY".to_string(),
                run_reports,
            },
            coherence_windows,
        }
    }

    #[test]
    fn synthetic_fixture_never_becomes_physical_conditioned_summary() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports.len(), 2);
        assert_eq!(report.band_reports.len(), 1);
        assert!(report.reasons.iter().any(|reason| reason.contains("synthetic")));
    }


    #[test]
    fn present_and_absent_cohorts_are_conditioned_separately() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        let report = audit(&input);
        assert_eq!(report.disposition, "CONDITIONED_SUMMARY_ONLY");
        assert_eq!(report.run_reports.len(), 2);
        for run in &report.run_reports {
            assert_eq!(run.cavitation_present_windows, 2);
            assert_eq!(run.cavitation_absent_windows, 2);
            assert_eq!(run.cavitation_unknown_windows, 0);
            let band = &run.run_band_summaries[0];
            assert!((band.cavitation_present.mean_coherence_magnitude_squared.unwrap() - 0.8).abs() < 1e-12);
            assert!((band.cavitation_absent.mean_coherence_magnitude_squared.unwrap() - 0.2).abs() < 1e-12);
            assert!((band.present_minus_absent_mean_coherence.unwrap() - 0.6).abs() < 1e-12);
            assert!((band.cavitation_present.mean_coherent_pressure_psd_pa2_per_hz.unwrap() - 8.0).abs() < 1e-12);
            assert!((band.cavitation_absent.mean_residual_pressure_psd_pa2_per_hz.unwrap() - 8.0).abs() < 1e-12);
        }
    }

    #[test]
    fn independent_run_means_are_equal_weighted_not_window_weighted() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        for item in &mut input.coherence_windows {
            if item.run_id == "run-2" {
                let state = input.intermittency_report.run_reports[1].window_evidence.iter()
                    .find(|window| window.window_id == item.window_id)
                    .and_then(|window| window.cavitation_state_for_conditioning);
                for band in &mut item.coherence_report.band_reports {
                    band.coherence_magnitude_squared = if state == Some(true) { 0.4 } else { 0.1 };
                }
            }
        }
        let report = audit(&input);
        assert_eq!(report.disposition, "CONDITIONED_SUMMARY_ONLY");
        let band = &report.band_reports[0];
        assert!((band.equal_run_weighted_present_mean_coherence.unwrap() - 0.6).abs() < 1e-12);
        assert!((band.equal_run_weighted_absent_mean_coherence.unwrap() - 0.15).abs() < 1e-12);
        assert!((band.equal_run_weighted_present_minus_absent_coherence.unwrap() - 0.45).abs() < 1e-12);
        assert_eq!(band.independent_runs_with_present_group, 2);
        assert_eq!(band.independent_runs_with_absent_group, 2);
    }

    #[test]
    fn unknown_cavitation_state_is_never_promoted_to_absent() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        let target = &mut input.intermittency_report.run_reports[0].window_evidence[0];
        target.cavitation_annotation_raw = None;
        target.cavitation_state_for_conditioning = None;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports[0].cavitation_unknown_windows, 1);
        assert_eq!(report.run_reports[0].cavitation_absent_windows, 1);
        assert_eq!(report.run_reports[0].cavitation_present_windows, 2);
    }

    #[test]
    fn misaligned_raw_positive_annotation_remains_unknown_for_cohorting() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        let window = &mut input.intermittency_report.run_reports[0].window_evidence[0];
        window.acoustic_cavitation_alignment_eligible = false;
        window.acoustic_cavitation_alignment_offset_s = 0.02;
        window.cavitation_annotation_raw = Some(true);
        window.cavitation_state_for_conditioning = None;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports[0].cavitation_unknown_windows, 1);
        let receipt = &report.run_reports[0].joined_window_receipts[0];
        assert_eq!(receipt.cavitation_annotation_raw, Some(true));
        assert_eq!(receipt.cavitation_state_for_conditioning, None);
        assert!(!receipt.alignment_eligible);
    }

    #[test]
    fn exact_pressure_trace_mismatch_is_invalid() {
        let mut input = case();
        input.coherence_windows[0].coherence_report.pressure_raw_trace_sha256 = digest('z');
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn exact_time_window_mismatch_is_invalid() {
        let mut input = case();
        input.coherence_windows[0].coherence_report.pressure_window_end_s += 0.01;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn service_environment_or_timebase_mismatch_is_incomparable() {
        let mut input = case();
        input.coherence_windows[0].coherence_report.service_fingerprint_sha256 = digest('x');
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
        let mut input = case();
        input.coherence_windows[0].coherence_report.timebase_binding_receipt_sha256 = digest('x');
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn phase_unresolved_reports_still_support_magnitude_coherence_summary() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        for window in &mut input.coherence_windows {
            window.coherence_report.disposition = MAGNITUDE_ONLY.to_string();
            for band in &mut window.coherence_report.band_reports {
                band.phase_status = "phase_unresolved".to_string();
            }
        }
        let report = audit(&input);
        assert_eq!(report.disposition, "CONDITIONED_SUMMARY_ONLY");
        assert!(report.run_reports.iter().all(|run| run.phase_unresolved_windows == 4));
    }

    #[test]
    fn uncertainty_unresolved_band_is_incomplete() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        input.coherence_windows[0].coherence_report.band_reports[0].uncertainty_status =
            "coherent_residual_interval_unresolved".to_string();
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn frequency_grid_mismatch_is_incomparable() {
        let mut input = case();
        input.coherence_windows[0].coherence_report.band_reports[0].frequency_hz = 101.0;
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn duplicate_coherence_report_digest_is_invalid() {
        let mut input = case();
        input.coherence_windows[1].coherence_report_sha256 =
            input.coherence_windows[0].coherence_report_sha256.clone();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn insufficient_independent_runs_is_incomplete() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        input.intermittency_report.run_reports.truncate(1);
        input.coherence_windows.retain(|window| window.run_id == "run-1");
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn missing_per_window_uncertainty_bounds_is_incomplete() {
        let mut input = case();
        input.coherence_windows[0].coherence_report.band_reports[0]
            .vibration_reference_coherent_pressure_psd_lower_bound = None;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn a_window_is_joined_exactly_once() {
        let mut input = case();
        input.coherence_windows.push(input.coherence_windows[0].clone());
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn input_window_permutation_preserves_run_and_band_summaries() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.source.rights_status = RightsStatus::PublicOrPermitted;
        let first = audit(&input);
        input.coherence_windows.reverse();
        input.intermittency_report.run_reports.reverse();
        for run in &mut input.intermittency_report.run_reports {
            run.window_evidence.reverse();
        }
        let second = audit(&input);
        assert_eq!(first.disposition, second.disposition);
        assert_eq!(first.run_reports.len(), second.run_reports.len());
        assert_eq!(first.band_reports[0].equal_run_weighted_present_mean_coherence,
            second.band_reports[0].equal_run_weighted_present_mean_coherence);
        assert_eq!(first.band_reports[0].equal_run_weighted_absent_mean_coherence,
            second.band_reports[0].equal_run_weighted_absent_mean_coherence);
    }

    #[test]
    fn coherence_one_does_not_promote_a_unique_source_claim() {
        let mut input = case();
        input.coherence_windows[0].coherence_report.band_reports[0].coherence_magnitude_squared = 1.0;
        let report = audit(&input);
        assert!(report.claim_ceiling.contains("no_unique_source"));
    }

    #[test]
    fn source_and_receipt_digests_are_preserved() {
        let report = audit(&case());
        assert_eq!(report.intermittency_receipt_sha256, digest('2'));
        assert_eq!(report.source_data_sha256, digest('1'));
        assert_eq!(report.run_reports[0].joined_window_receipts[0].coherence_report_sha256,
            format!("{:064x}", 50_000));
    }
}
