use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const SPL_QUANTITY: &str = "received_spl_db_re_1upa";
const WINDOW_DURATION_TOLERANCE_S: f64 = 1.0e-6;

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

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BackgroundMethod {
    BareHub,
    PressureSuppressed,
    Unknown,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum SusceptibilityMeasure {
    TotalGasContent,
    CavitationSusceptibilityAlpha,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
struct WaterCondition {
    measure: SusceptibilityMeasure,
    value: f64,
    unit: String,
    expanded_uncertainty: f64,
    equivalence_tolerance: f64,
    method_id: String,
    instrument_id: String,
    calibration_certificate_sha256: String,
    evidence_sha256: String,
    nuclei_distribution_evidence_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    operating_condition_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    cavitation_instrument_id: String,
    cavitation_observation_method_id: String,
    cavitation_calibration_or_validation_sha256: String,
    maximum_sync_alignment_uncertainty_s: f64,
    band: BandDefinition,
    window_duration_s: f64,
    minimum_usable_windows_per_run: usize,
    minimum_runs: usize,
    runs: Vec<Run>,
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
#[serde(deny_unknown_fields)]
struct BandDefinition {
    band_id: String,
    quantity_id: String,
    center_frequency_hz: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    run_id: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    acoustic_cavitation_sync_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    background_method: BackgroundMethod,
    background_trace_sha256: String,
    water_condition: WaterCondition,
    windows: Vec<Window>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Window {
    window_id: String,
    start_time_s: f64,
    end_time_s: f64,
    acoustic_cavitation_alignment_offset_s: f64,
    acoustic_cavitation_alignment_expanded_uncertainty_s: f64,
    acoustic_trace_sha256: String,
    cavitation_observation_sha256: String,
    cavitation_present: Option<bool>,
    total_spl_db: f64,
    background_spl_db: f64,
    total_expanded_uncertainty_db: f64,
    background_expanded_uncertainty_db: f64,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: &'static str,
    operating_condition_id: String,
    band_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    cavitation_instrument_id: String,
    cavitation_observation_method_id: String,
    cavitation_calibration_or_validation_sha256: String,
    maximum_sync_alignment_uncertainty_s: f64,
    quantity_id: &'static str,
    center_frequency_hz: f64,
    disposition: &'static str,
    claim_ceiling: &'static str,
    run_reports: Vec<RunReport>,
    nuclei_distribution_coverage_fraction: f64,
    run_to_run_energy_mean_sd_db: Option<f64>,
    run_to_run_energy_mean_range_db: Option<f64>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct WindowEvidenceRef {
    window_id: String,
    acoustic_trace_sha256: String,
    cavitation_observation_sha256: String,
    start_time_s: f64,
    end_time_s: f64,
    acoustic_cavitation_alignment_offset_s: f64,
    acoustic_cavitation_alignment_expanded_uncertainty_s: f64,
    acoustic_cavitation_alignment_eligible: bool,
    disposition: &'static str,
    exclusion_reason: Option<&'static str>,
    corrected_spl_db: Option<f64>,
    corrected_spl_expanded_lower_bound_db: Option<f64>,
    corrected_spl_expanded_upper_bound_db: Option<f64>,
    cavitation_component_upper_limit_db: Option<f64>,
}

#[derive(Debug, Serialize)]
struct RunReport {
    run_id: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    acoustic_cavitation_sync_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    background_method: &'static str,
    background_trace_sha256: String,
    water_condition: WaterCondition,
    window_evidence: Vec<WindowEvidenceRef>,
    total_windows: usize,
    usable_windows: usize,
    excluded_background_dominated: usize,
    excluded_exact_ten_db_boundary: usize,
    excluded_uncertainty_unresolved: usize,
    excluded_unknown_background_method: usize,
    excluded_alignment_unresolved: usize,
    cavitation_annotation_coverage: f64,
    cavitation_occupancy_fraction: Option<f64>,
    observed_cavitation_onsets: Option<usize>,
    observed_cavitation_onsets_per_hour: Option<f64>,
    left_censored_active_segments: Option<usize>,
    right_censored_active_segments: Option<usize>,
    acquisition_duration_s: f64,
    ittc_recommended_acquisition_duration_met: bool,
    energy_equivalent_mean_spl_db: Option<f64>,
    median_spl_db: Option<f64>,
    p05_spl_db: Option<f64>,
    p95_spl_db: Option<f64>,
    cavitation_present_energy_mean_spl_db: Option<f64>,
    cavitation_absent_energy_mean_spl_db: Option<f64>,
    cavitation_conditioned_delta_db: Option<f64>,
    usable_observation_duration_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackgroundFailure {
    BackgroundDominated,
    ExactTenDbBoundaryUnspecified,
}

#[derive(Debug, Clone, Copy)]
struct UsableWindow {
    start_time_s: f64,
    end_time_s: f64,
    corrected_spl_db: f64,
    cavitation_present: Option<bool>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn energy_mean_db(levels_db: &[f64]) -> Option<f64> {
    if levels_db.is_empty() || levels_db.iter().any(|level| !level.is_finite()) {
        return None;
    }
    let max_level = levels_db.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_relative_energy = levels_db
        .iter()
        .map(|level| 10.0_f64.powf((level - max_level) / 10.0))
        .sum::<f64>()
        / levels_db.len() as f64;
    let result = max_level + 10.0 * mean_relative_energy.log10();
    result.is_finite().then_some(result)
}

fn empirical_quantile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * fraction)
}

fn ittc_background_corrected_spl(
    total_db: f64,
    background_db: f64,
) -> Result<f64, BackgroundFailure> {
    if !total_db.is_finite() || !background_db.is_finite() || total_db <= background_db {
        return Err(BackgroundFailure::BackgroundDominated);
    }
    let margin_db = total_db - background_db;
    if margin_db < 3.0 {
        return Err(BackgroundFailure::BackgroundDominated);
    }
    if margin_db == 10.0 {
        return Err(BackgroundFailure::ExactTenDbBoundaryUnspecified);
    }
    if margin_db > 10.0 {
        return Ok(total_db);
    }
    let background_power_ratio = 10.0_f64.powf((background_db - total_db) / 10.0);
    let net_power_ratio = 1.0 - background_power_ratio;
    if !net_power_ratio.is_finite() || net_power_ratio <= 0.0 {
        return Err(BackgroundFailure::BackgroundDominated);
    }
    let corrected = total_db + 10.0 * net_power_ratio.log10();
    if corrected.is_finite() {
        Ok(corrected)
    } else {
        Err(BackgroundFailure::BackgroundDominated)
    }
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    if case.case_id.trim().is_empty()
        || case.source.dataset_id.trim().is_empty()
        || case.source.citation.trim().is_empty()
        || case.operating_condition_id.trim().is_empty()
        || case.band.band_id.trim().is_empty()
    {
        errors.push("case, dataset, operating-condition and band IDs/citation are required".to_string());
    }
    for (label, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("service fingerprint", &case.service_fingerprint_sha256),
        ("environment fingerprint", &case.environment_fingerprint_sha256),
        ("hydrophone calibration certificate", &case.calibration_certificate_sha256),
        ("cavitation calibration/validation record", &case.cavitation_calibration_or_validation_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label} digest must be 64 hexadecimal characters"));
        }
    }
    if case.band.quantity_id != SPL_QUANTITY {
        errors.push(format!("band quantity_id must be {SPL_QUANTITY}"));
    }
    if !finite_positive(case.band.center_frequency_hz)
        || !finite_positive(case.window_duration_s)
        || case.minimum_usable_windows_per_run == 0
        || case.minimum_runs == 0
    {
        errors.push("band frequency, window duration and minimum sample counts must be positive".to_string());
    }
    if case.hydrophone_instrument_id.trim().is_empty()
        || case.cavitation_instrument_id.trim().is_empty()
        || case.cavitation_observation_method_id.trim().is_empty()
    {
        errors.push("hydrophone/cavitation instrument and cavitation observation method IDs are required".to_string());
    }
    if !finite_positive(case.maximum_sync_alignment_uncertainty_s)
        || case.maximum_sync_alignment_uncertainty_s >= case.window_duration_s / 2.0
    {
        errors.push("maximum_sync_alignment_uncertainty_s must be positive and less than half the window duration".to_string());
    }
    if case.runs.is_empty() {
        errors.push("at least one run is required".to_string());
    }

    let mut run_ids = BTreeSet::new();
    for run in &case.runs {
        if run.run_id.trim().is_empty() || !run_ids.insert(run.run_id.clone()) {
            errors.push(format!("empty or duplicate run_id {}", run.run_id));
        }
        let water = &run.water_condition;
        if !water.value.is_finite()
            || water.value < 0.0
            || !finite_positive(water.expanded_uncertainty)
            || !finite_positive(water.equivalence_tolerance)
            || water.unit.trim().is_empty()
            || water.method_id.trim().is_empty()
            || water.instrument_id.trim().is_empty()
        {
            errors.push(format!("run {} has invalid water susceptibility value, unit, method, instrument or uncertainty", run.run_id));
        }
        for (label, digest) in [
            ("water susceptibility calibration certificate", &water.calibration_certificate_sha256),
            ("water susceptibility measurement evidence", &water.evidence_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("run {}: {label} digest must be 64 hexadecimal characters", run.run_id));
            }
        }
        if let Some(digest) = &water.nuclei_distribution_evidence_sha256
            && !is_sha256(digest)
        {
            errors.push(format!(
                "run {}: nuclei-distribution evidence digest must be 64 hexadecimal characters",
                run.run_id
            ));
        }
        for (label, digest) in [
            ("timebase receipt", &run.timebase_receipt_sha256),
            ("operating-point receipt", &run.operating_point_receipt_sha256),
            ("acoustic/cavitation synchronization receipt", &run.acoustic_cavitation_sync_receipt_sha256),
            ("run service fingerprint", &run.service_fingerprint_sha256),
            ("run environment fingerprint", &run.environment_fingerprint_sha256),
            ("run calibration certificate", &run.calibration_certificate_sha256),
            ("background trace", &run.background_trace_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("run {}: {label} digest must be 64 hexadecimal characters", run.run_id));
            }
        }
        if run.service_fingerprint_sha256 != case.service_fingerprint_sha256 {
            errors.push(format!("run {} service fingerprint differs from the frozen case", run.run_id));
        }
        if run.environment_fingerprint_sha256 != case.environment_fingerprint_sha256 {
            errors.push(format!("run {} environment fingerprint differs from the frozen case", run.run_id));
        }
        if run.hydrophone_instrument_id != case.hydrophone_instrument_id {
            errors.push(format!("run {} hydrophone instrument differs from the frozen case", run.run_id));
        }
        if run.calibration_certificate_sha256 != case.calibration_certificate_sha256 {
            errors.push(format!("run {} calibration certificate differs from the frozen case", run.run_id));
        }
        if run.windows.is_empty() {
            errors.push(format!("run {} has no windows", run.run_id));
        }
        let mut window_ids = BTreeSet::new();
        let mut acoustic_trace_digests = BTreeSet::new();
        let mut cavitation_observation_digests = BTreeSet::new();
        let mut previous_end: Option<f64> = None;
        for window in &run.windows {
            if window.window_id.trim().is_empty() || !window_ids.insert(window.window_id.clone()) {
                errors.push(format!("run {} has empty or duplicate window_id", run.run_id));
            }
            if !is_sha256(&window.acoustic_trace_sha256)
                || !is_sha256(&window.cavitation_observation_sha256)
            {
                errors.push(format!("run {} window {} has malformed acoustic/cavitation evidence digest", run.run_id, window.window_id));
            }
            if !acoustic_trace_digests.insert(window.acoustic_trace_sha256.clone()) {
                errors.push(format!("run {} reuses an acoustic trace digest across windows", run.run_id));
            }
            if !cavitation_observation_digests.insert(window.cavitation_observation_sha256.clone()) {
                errors.push(format!("run {} reuses a cavitation-observation digest across windows", run.run_id));
            }
            if !window.start_time_s.is_finite()
                || window.start_time_s < 0.0
                || !window.end_time_s.is_finite()
                || window.end_time_s <= window.start_time_s
                || ((window.end_time_s - window.start_time_s) - case.window_duration_s).abs()
                    > WINDOW_DURATION_TOLERANCE_S
                || !window.acoustic_cavitation_alignment_offset_s.is_finite()
                || !finite_positive(window.acoustic_cavitation_alignment_expanded_uncertainty_s)
                || !window.total_spl_db.is_finite()
                || !window.background_spl_db.is_finite()
                || !finite_positive(window.total_expanded_uncertainty_db)
                || !finite_positive(window.background_expanded_uncertainty_db)
            {
                errors.push(format!("run {} window {} has invalid times, duration, SPL or uncertainty", run.run_id, window.window_id));
            }
            if let Some(end) = previous_end
                && window.start_time_s < end
            {
                errors.push(format!("run {} windows overlap or are not chronological at {}", run.run_id, window.window_id));
            }
            previous_end = Some(window.end_time_s);
        }
    }
    errors
}

fn summarize_run(run: &Run, case: &Case) -> RunReport {
    let mut usable = Vec::new();
    let mut window_evidence = Vec::with_capacity(run.windows.len());
    let mut excluded_background_dominated = 0;
    let mut excluded_exact_ten_db_boundary = 0;
    let mut excluded_uncertainty_unresolved = 0;
    let mut excluded_unknown_background_method = 0;
    let mut excluded_alignment_unresolved = 0;

    for window in &run.windows {
        let alignment_eligible = window.acoustic_cavitation_alignment_offset_s.abs()
            + window.acoustic_cavitation_alignment_expanded_uncertainty_s
            <= case.maximum_sync_alignment_uncertainty_s;
        if !alignment_eligible {
            excluded_alignment_unresolved += 1;
        }
        let evidence_ref = |disposition, exclusion_reason, corrected_spl_db, lower_bound, upper_bound, upper_limit| WindowEvidenceRef {
            window_id: window.window_id.clone(),
            acoustic_trace_sha256: window.acoustic_trace_sha256.clone(),
            cavitation_observation_sha256: window.cavitation_observation_sha256.clone(),
            start_time_s: window.start_time_s,
            end_time_s: window.end_time_s,
            acoustic_cavitation_alignment_offset_s: window.acoustic_cavitation_alignment_offset_s,
            acoustic_cavitation_alignment_expanded_uncertainty_s: window.acoustic_cavitation_alignment_expanded_uncertainty_s,
            acoustic_cavitation_alignment_eligible: alignment_eligible,
            disposition,
            exclusion_reason,
            corrected_spl_db,
            corrected_spl_expanded_lower_bound_db: lower_bound,
            corrected_spl_expanded_upper_bound_db: upper_bound,
            cavitation_component_upper_limit_db: upper_limit,
        };

        if run.background_method == BackgroundMethod::Unknown {
            excluded_unknown_background_method += 1;
            window_evidence.push(evidence_ref(
                "excluded",
                Some("unknown_background_measurement_method"),
                None,
                None,
                None,
                None,
            ));
            continue;
        }

        match ittc_background_corrected_spl(window.total_spl_db, window.background_spl_db) {
            Err(BackgroundFailure::BackgroundDominated) => {
                excluded_background_dominated += 1;
                window_evidence.push(evidence_ref(
                    "excluded",
                    Some("background_margin_below_3_db_censored_upper_limit"),
                    None,
                    None,
                    None,
                    Some(window.total_spl_db + window.total_expanded_uncertainty_db),
                ));
            }
            Err(BackgroundFailure::ExactTenDbBoundaryUnspecified) => {
                excluded_exact_ten_db_boundary += 1;
                window_evidence.push(evidence_ref(
                    "excluded",
                    Some("exact_10_db_boundary_unspecified_by_ittc_rev03"),
                    None,
                    None,
                    None,
                    None,
                ));
            }
            Ok(level) => {
                let lower = ittc_background_corrected_spl(
                    window.total_spl_db - window.total_expanded_uncertainty_db,
                    window.background_spl_db + window.background_expanded_uncertainty_db,
                );
                let upper = ittc_background_corrected_spl(
                    window.total_spl_db + window.total_expanded_uncertainty_db,
                    window.background_spl_db - window.background_expanded_uncertainty_db,
                );
                let (Ok(lower_bound), Ok(upper_bound)) = (lower, upper) else {
                    excluded_uncertainty_unresolved += 1;
                    window_evidence.push(evidence_ref(
                        "excluded",
                        Some("expanded_uncertainty_crosses_background_eligibility_boundary"),
                        None,
                        None,
                        None,
                        None,
                    ));
                    continue;
                };
                if !level.is_finite() {
                    excluded_uncertainty_unresolved += 1;
                    window_evidence.push(evidence_ref(
                        "excluded",
                        Some("corrected_level_non_finite"),
                        None,
                        None,
                        None,
                        None,
                    ));
                    continue;
                }
                window_evidence.push(evidence_ref(
                    "included",
                    None,
                    Some(level),
                    Some(lower_bound),
                    Some(upper_bound),
                    None,
                ));
                usable.push(UsableWindow {
                    start_time_s: window.start_time_s,
                    end_time_s: window.end_time_s,
                    corrected_spl_db: level,
                    cavitation_present: if alignment_eligible {
                        window.cavitation_present
                    } else {
                        None
                    },
                });
            }
        }
    }

    let levels: Vec<f64> = usable.iter().map(|window| window.corrected_spl_db).collect();
    let mut sorted_levels = levels.clone();
    sorted_levels.sort_by(f64::total_cmp);
    let energy_mean = energy_mean_db(&levels);
    let median = empirical_quantile(&sorted_levels, 0.5);
    let p05 = empirical_quantile(&sorted_levels, 0.05);
    let p95 = empirical_quantile(&sorted_levels, 0.95);

    let usable_duration_s: f64 = usable
        .iter()
        .map(|window| window.end_time_s - window.start_time_s)
        .sum();
    let annotation_coverage = if run.windows.is_empty() {
        0.0
    } else {
        run.windows.iter().filter(|window| {
            window.acoustic_cavitation_alignment_offset_s.abs()
                + window.acoustic_cavitation_alignment_expanded_uncertainty_s
                <= case.maximum_sync_alignment_uncertainty_s
                && window.cavitation_present.is_some()
        }).count() as f64 / run.windows.len() as f64
    };
    let annotated_usable = usable
        .iter()
        .filter(|window| window.cavitation_present.is_some())
        .count();

    let (
        occupancy,
        onsets,
        onsets_per_hour,
        left_censored,
        right_censored,
        on_mean,
        off_mean,
        conditioned_delta,
    ) = if annotated_usable == usable.len() && !usable.is_empty() {
        let present: Vec<f64> = usable
            .iter()
            .filter(|window| window.cavitation_present == Some(true))
            .map(|window| window.corrected_spl_db)
            .collect();
        let absent: Vec<f64> = usable
            .iter()
            .filter(|window| window.cavitation_present == Some(false))
            .map(|window| window.corrected_spl_db)
            .collect();
        let occupancy = present.len() as f64 / usable.len() as f64;

        let mut observed_onsets = 0usize;
        let mut left_censored_segments = 0usize;
        for index in 0..usable.len() {
            if usable[index].cavitation_present != Some(true) {
                continue;
            }
            if index == 0 {
                // Acquisition starts while this active segment is already present.
                left_censored_segments += 1;
                continue;
            }
            let contiguous = (usable[index].start_time_s - usable[index - 1].end_time_s).abs()
                <= WINDOW_DURATION_TOLERANCE_S;
            if !contiguous {
                // A gap hides the transition into this active segment.
                left_censored_segments += 1;
            } else if usable[index - 1].cavitation_present == Some(false) {
                observed_onsets += 1;
            }
        }
        let right_censored_segments = if usable
            .last()
            .is_some_and(|window| window.cavitation_present == Some(true))
        {
            1
        } else {
            0
        };
        let onsets_per_hour = if usable_duration_s > 0.0 {
            Some(observed_onsets as f64 * 3600.0 / usable_duration_s)
        } else {
            None
        };
        let on_mean = energy_mean_db(&present);
        let off_mean = energy_mean_db(&absent);
        let delta = match (on_mean, off_mean) {
            (Some(on), Some(off)) => Some(on - off),
            _ => None,
        };
        (
            Some(occupancy),
            Some(observed_onsets),
            onsets_per_hour,
            Some(left_censored_segments),
            Some(right_censored_segments),
            on_mean,
            off_mean,
            delta,
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    // Use the sum of observed window durations, not wall-clock span:
    // gaps between windows are not acquired data and must not inflate the
    // ITTC-recommended acquisition-period diagnostic.
    let acquisition_duration_s: f64 = run
        .windows
        .iter()
        .map(|window| window.end_time_s - window.start_time_s)
        .sum();

    RunReport {
        run_id: run.run_id.clone(),
        timebase_receipt_sha256: run.timebase_receipt_sha256.clone(),
        service_fingerprint_sha256: run.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: run.environment_fingerprint_sha256.clone(),
        hydrophone_instrument_id: run.hydrophone_instrument_id.clone(),
        calibration_certificate_sha256: run.calibration_certificate_sha256.clone(),
        operating_point_receipt_sha256: run.operating_point_receipt_sha256.clone(),
        acoustic_cavitation_sync_receipt_sha256: run.acoustic_cavitation_sync_receipt_sha256.clone(),
        background_method: match run.background_method {
            BackgroundMethod::BareHub => "bare_hub",
            BackgroundMethod::PressureSuppressed => "pressure_suppressed",
            BackgroundMethod::Unknown => "unknown",
        },
        background_trace_sha256: run.background_trace_sha256.clone(),
        water_condition: run.water_condition.clone(),
        window_evidence,
        total_windows: run.windows.len(),
        usable_windows: usable.len(),
        excluded_background_dominated,
        excluded_exact_ten_db_boundary,
        excluded_uncertainty_unresolved,
        excluded_unknown_background_method,
        excluded_alignment_unresolved,
        cavitation_annotation_coverage: annotation_coverage,
        cavitation_occupancy_fraction: occupancy,
        observed_cavitation_onsets: onsets,
        observed_cavitation_onsets_per_hour: onsets_per_hour,
        left_censored_active_segments: left_censored,
        right_censored_active_segments: right_censored,
        acquisition_duration_s,
        ittc_recommended_acquisition_duration_met: acquisition_duration_s > 20.0,
        energy_equivalent_mean_spl_db: energy_mean,
        median_spl_db: median,
        p05_spl_db: p05,
        p95_spl_db: p95,
        cavitation_present_energy_mean_spl_db: on_mean,
        cavitation_absent_energy_mean_spl_db: off_mean,
        cavitation_conditioned_delta_db: conditioned_delta,
        usable_observation_duration_s: usable_duration_s,
    }
}

fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_squares = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    Some((sum_squares / (values.len() - 1) as f64).sqrt())
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let run_reports: Vec<RunReport> = if validation_errors.is_empty() {
        case.runs.iter().map(|run| summarize_run(run, case)).collect()
    } else {
        Vec::new()
    };

    let mut water_condition_incomparable = false;
    let mut water_condition_unresolved = false;
    let nuclei_distribution_coverage_fraction = if case.runs.is_empty() {
        0.0
    } else {
        case.runs.iter()
            .filter(|run| run.water_condition.nuclei_distribution_evidence_sha256.is_some())
            .count() as f64 / case.runs.len() as f64
    };

    if validation_errors.is_empty() {
        let reference_water = &case.runs[0].water_condition;
        if nuclei_distribution_coverage_fraction > 0.0
            && nuclei_distribution_coverage_fraction < 1.0
        {
            reasons.push("nuclei number/size distribution evidence is present for only part of the run cohort".to_string());
            water_condition_unresolved = true;
        } else if nuclei_distribution_coverage_fraction == 0.0 {
            reasons.push("recommended nuclei number/size distribution evidence was not supplied; water comparison relies on the declared gas-content or cavitation-susceptibility measure".to_string());
        }

        for run in case.runs.iter().skip(1) {
            let water = &run.water_condition;
            if water.measure != reference_water.measure
                || water.unit != reference_water.unit
                || water.method_id != reference_water.method_id
                || water.instrument_id != reference_water.instrument_id
                || water.calibration_certificate_sha256 != reference_water.calibration_certificate_sha256
                || (water.equivalence_tolerance - reference_water.equivalence_tolerance).abs() > 1.0e-12
            {
                reasons.push(format!("run {} water-condition measure, unit, method, instrument, calibration or equivalence tolerance differs from the comparison stratum", run.run_id));
                water_condition_incomparable = true;
                continue;
            }
            let delta = (water.value - reference_water.value).abs();
            let uncertainty = water.expanded_uncertainty + reference_water.expanded_uncertainty;
            if delta - uncertainty > reference_water.equivalence_tolerance {
                reasons.push(format!("run {} water susceptibility is demonstrably outside the predeclared equivalence tolerance", run.run_id));
                water_condition_incomparable = true;
            } else if delta + uncertainty > reference_water.equivalence_tolerance {
                reasons.push(format!("run {} water susceptibility uncertainty does not establish equivalence to the reference run", run.run_id));
                water_condition_unresolved = true;
            }
        }

        let common_method = case.runs.first().map(|run| &run.background_method);
        if case.runs.iter().any(|run| Some(&run.background_method) != common_method) {
            reasons.push("background measurement method differs across runs in one operating-condition stratum".to_string());
        }
        if case.runs.iter().any(|run| run.windows.iter().any(|window| window.cavitation_present.is_none())) {
            reasons.push("one or more windows lack independent cavitation observations; occupancy and conditioned summaries are incomplete".to_string());
        }
        if run_reports.iter().any(|report| report.excluded_alignment_unresolved > 0) {
            reasons.push("one or more cavitation annotations exceed the declared acoustic/video synchronization tolerance; event occupancy and conditioned comparisons are incomplete".to_string());
        }
        if run_reports.iter().any(|report| report.usable_windows < case.minimum_usable_windows_per_run) {
            reasons.push("one or more runs have fewer than the required usable windows".to_string());
        }
        if case.runs.len() < case.minimum_runs {
            reasons.push("fewer than the required independent runs are available".to_string());
        }
        if run_reports.iter().any(|report| !report.ittc_recommended_acquisition_duration_met) {
            reasons.push("one or more runs are shorter than the ITTC-recommended acquisition duration of greater than 20 seconds; this is advisory context, not a hard rejection".to_string());
        }
    }

    let run_means: Vec<f64> = run_reports.iter()
        .filter_map(|report| report.energy_equivalent_mean_spl_db)
        .collect();
    let run_to_run_sd = sample_sd(&run_means);
    let (range_lower, range_upper) = if run_means.is_empty() {
        (None, None)
    } else {
        let low = run_means.iter().copied().fold(f64::INFINITY, f64::min);
        let high = run_means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (Some(low), Some(high))
    };
    let run_range = match (range_lower, range_upper) {
        (Some(low), Some(high)) => Some(high - low),
        _ => None,
    };

    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot qualify physical cavitation intermittency".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source dataset reuse rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source evidence has not been independently reviewed".to_string());
    }
    if case.runs.iter().any(|run| run.background_method == BackgroundMethod::Unknown) {
        reasons.push("one or more runs have an unknown background measurement method".to_string());
    }

    let disposition = if !validation_errors.is_empty() {
        "INVALID"
    } else if water_condition_incomparable {
        "INCOMPARABLE_CONDITION"
    } else if water_condition_unresolved
        || run_reports.iter().any(|report| report.usable_windows < case.minimum_usable_windows_per_run)
        || case.runs.len() < case.minimum_runs
        || case.runs.iter().any(|run| run.windows.iter().any(|window| window.cavitation_present.is_none()))
        || run_reports.iter().any(|report| report.excluded_alignment_unresolved > 0)
        || case.runs.iter().any(|run| run.background_method == BackgroundMethod::Unknown)
        || reasons.iter().any(|reason| reason.contains("background measurement method differs"))
        || case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
    {
        "INCOMPLETE"
    } else {
        "DISTRIBUTIONAL_SUMMARY_ONLY"
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
        operating_condition_id: case.operating_condition_id.clone(),
        band_id: case.band.band_id.clone(),
        service_fingerprint_sha256: case.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: case.environment_fingerprint_sha256.clone(),
        hydrophone_instrument_id: case.hydrophone_instrument_id.clone(),
        calibration_certificate_sha256: case.calibration_certificate_sha256.clone(),
        cavitation_instrument_id: case.cavitation_instrument_id.clone(),
        cavitation_observation_method_id: case.cavitation_observation_method_id.clone(),
        cavitation_calibration_or_validation_sha256: case.cavitation_calibration_or_validation_sha256.clone(),
        maximum_sync_alignment_uncertainty_s: case.maximum_sync_alignment_uncertainty_s,
        quantity_id: SPL_QUANTITY,
        center_frequency_hz: case.band.center_frequency_hz,
        disposition,
        claim_ceiling: "distributional_summary_of_declared_model_scale_windows_only_no_full_scale_transfer_source_decomposition_causal_or_ecological_qualification",
        run_reports,
        nuclei_distribution_coverage_fraction,
        run_to_run_energy_mean_sd_db: run_to_run_sd,
        run_to_run_energy_mean_range_db: run_range,
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-cavitation-intermittency-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-cavitation-intermittency-audit <case.json>");
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
        "DISTRIBUTIONAL_SUMMARY_ONLY" => {}
        "INVALID" | "INCOMPARABLE_CONDITION" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(letter: char) -> String {
        std::iter::repeat_n(letter, 64).collect()
    }

    fn digest_seed(seed: usize) -> String {
        format!("{seed:064x}")
    }

    fn sample_window(index: usize, run_index: usize, level: f64, cavitation: bool) -> Window {
        let start = index as f64;
        let seed = run_index * 100 + index + 1;
        Window {
            window_id: format!("window-{index}"),
            start_time_s: start,
            end_time_s: start + 1.0,
            acoustic_cavitation_alignment_offset_s: 0.0,
            acoustic_cavitation_alignment_expanded_uncertainty_s: 0.001,
            acoustic_trace_sha256: digest_seed(seed),
            cavitation_observation_sha256: digest_seed(seed + 10_000),
            cavitation_present: Some(cavitation),
            total_spl_db: level,
            background_spl_db: 90.0,
            total_expanded_uncertainty_db: 0.2,
            background_expanded_uncertainty_db: 0.2,
        }
    }

    fn run(id: &str, run_index: usize, levels: &[f64], cavitation: &[bool]) -> Run {
        let windows = levels.iter().zip(cavitation)
            .enumerate()
            .map(|(index, (level, present))| sample_window(index, run_index, *level, *present))
            .collect();
        Run {
            run_id: id.to_string(),
            timebase_receipt_sha256: digest('e'),
            operating_point_receipt_sha256: digest(if run_index == 0 { 'a' } else { 'b' }),
            acoustic_cavitation_sync_receipt_sha256: digest(if run_index == 0 { 'c' } else { 'd' }),
            service_fingerprint_sha256: digest('f'),
            environment_fingerprint_sha256: digest('3'),
            hydrophone_instrument_id: "hydrophone-H1".to_string(),
            calibration_certificate_sha256: digest('4'),
            background_method: BackgroundMethod::BareHub,
            background_trace_sha256: digest(if run_index == 0 { '1' } else { '2' }),
            water_condition: WaterCondition {
                measure: SusceptibilityMeasure::CavitationSusceptibilityAlpha,
                value: 0.62 + run_index as f64 * 0.01,
                unit: "dimensionless".to_string(),
                expanded_uncertainty: 0.01,
                equivalence_tolerance: 0.05,
                method_id: "synthetic-alpha-method".to_string(),
                instrument_id: "synthetic-alpha-meter-1".to_string(),
                calibration_certificate_sha256: digest('5'),
                evidence_sha256: digest(if run_index == 0 { '6' } else { '7' }),
                nuclei_distribution_evidence_sha256: Some(digest(if run_index == 0 { '8' } else { '9' })),
            },
            windows,
        }
    }

    fn case() -> Case {
        let levels = [101.0, 111.0, 101.0, 111.0, 101.0, 111.0, 101.0, 111.0];
        let states = [false, true, false, true, false, true, false, true];
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-intermittency-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-intermittency".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('2'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            operating_condition_id: "J-0.8-sigma-nD-1".to_string(),
            service_fingerprint_sha256: digest('f'),
            environment_fingerprint_sha256: digest('3'),
            hydrophone_instrument_id: "hydrophone-H1".to_string(),
            calibration_certificate_sha256: digest('4'),
            cavitation_instrument_id: "high-speed-camera-C1".to_string(),
            cavitation_observation_method_id: "synchronized-blade-phase-annotation-v1".to_string(),
            cavitation_calibration_or_validation_sha256: digest('9'),
            maximum_sync_alignment_uncertainty_s: 0.01,
            band: BandDefinition {
                band_id: "one-third-octave-1000Hz".to_string(),
                quantity_id: SPL_QUANTITY.to_string(),
                center_frequency_hz: 1000.0,
            },
            window_duration_s: 1.0,
            minimum_usable_windows_per_run: 4,
            minimum_runs: 2,
            runs: vec![run("run-1", 0, &levels, &states), run("run-2", 1, &levels, &states)],
        }
    }

    #[test]
    fn stable_water_susceptibility_allows_distributional_comparison() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        assert_eq!(audit(&input).disposition, "DISTRIBUTIONAL_SUMMARY_ONLY");
    }

    #[test]
    fn water_susceptibility_drift_outside_tolerance_is_incomparable() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.runs[1].water_condition.value = 0.90;
        assert_eq!(audit(&input).disposition, "INCOMPARABLE_CONDITION");
    }

    #[test]
    fn water_susceptibility_uncertainty_crossing_tolerance_is_incomplete() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.runs[1].water_condition.value = 0.66;
        input.runs[1].water_condition.expanded_uncertainty = 0.04;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn water_susceptibility_method_or_calibration_change_is_incomparable() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.runs[1].water_condition.method_id = "different-method".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE_CONDITION");
    }

    #[test]
    fn malformed_nuclei_distribution_digest_is_invalid() {
        let mut input = case();
        input.runs[0].water_condition.nuclei_distribution_evidence_sha256 =
            Some("not-a-digest".to_string());
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn partial_nuclei_distribution_evidence_coverage_is_incomplete() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.runs[1].water_condition.nuclei_distribution_evidence_sha256 = None;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.nuclei_distribution_coverage_fraction, 0.5);
    }

    #[test]
    fn total_gas_content_is_an_alternative_water_condition_measure() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        for run in &mut input.runs {
            run.water_condition.measure = SusceptibilityMeasure::TotalGasContent;
            run.water_condition.value = 40.0;
            run.water_condition.unit = "percent_saturation".to_string();
        }
        assert_eq!(audit(&input).disposition, "DISTRIBUTIONAL_SUMMARY_ONLY");
    }

    #[test]
    fn no_nuclei_distribution_data_is_reported_as_missing_not_fabricated() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        for run in &mut input.runs {
            run.water_condition.nuclei_distribution_evidence_sha256 = None;
        }
        let report = audit(&input);
        assert_eq!(report.nuclei_distribution_coverage_fraction, 0.0);
        assert_eq!(report.disposition, "DISTRIBUTIONAL_SUMMARY_ONLY");
        assert!(report.reasons.iter().any(|reason| reason.contains("nuclei number/size distribution evidence was not supplied")));
    }

    #[test]
    fn run_report_preserves_water_condition_and_evidence_lineage() {
        let report = audit(&case());
        let water = &report.run_reports[0].water_condition;
        assert_eq!(water.measure, SusceptibilityMeasure::CavitationSusceptibilityAlpha);
        assert_eq!(water.method_id, "synthetic-alpha-method");
        assert_eq!(water.evidence_sha256, digest('6'));
        assert_eq!(report.nuclei_distribution_coverage_fraction, 1.0);
    }

    #[test]
    fn schema_and_fixture_require_water_condition_for_every_run() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-cavitation-intermittency-v0.1.0.schema.json"
        )).expect("schema JSON parses");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/fixtures/synthetic-marine-cavitation-intermittency-v0.1.0.json"
        )).expect("fixture JSON parses");
        assert!(schema["$defs"]["run"]["required"]
            .as_array().unwrap().iter().any(|value| value == "water_condition"));
        for required in [
            "measure", "value", "unit", "expanded_uncertainty",
            "equivalence_tolerance", "method_id", "instrument_id",
            "calibration_certificate_sha256", "evidence_sha256",
            "nuclei_distribution_evidence_sha256"
        ] {
            assert!(schema["$defs"]["waterCondition"]["required"]
                .as_array().unwrap().iter().any(|value| value == required));
        }
        for run in fixture["runs"].as_array().unwrap() {
            assert!(run.get("water_condition").is_some());
            assert_eq!(run["water_condition"]["evidence_sha256"].as_str().unwrap().len(), 64);
        }
    }

    #[test]
    fn energy_mean_is_not_arithmetic_mean_in_db() {
        let levels = [100.0, 110.0];
        let result = energy_mean_db(&levels).unwrap();
        assert!((result - 107.40362689494244).abs() < 1.0e-10);
        assert!((result - 105.0).abs() > 2.0);
    }

    #[test]
    fn background_dominated_windows_preserve_censored_upper_limits() {
        let mut input = case();
        input.runs[0].windows[0].total_spl_db = 92.9;
        let report = audit(&input);
        let run = &report.run_reports[0];
        assert_eq!(run.excluded_background_dominated, 1);
        assert_eq!(run.window_evidence[0].disposition, "excluded");
        let upper_limit = run.window_evidence[0].cavitation_component_upper_limit_db.unwrap();
        assert!((upper_limit - 93.1).abs() < 1.0e-12);
        assert_eq!(run.usable_windows, 7);
        assert_eq!(run.window_evidence[0].corrected_spl_db, None);
        assert_eq!(run.window_evidence[0].corrected_spl_expanded_lower_bound_db, None);
    }

    #[test]
    fn corrected_spl_expanded_bounds_are_preserved_per_window() {
        let report = audit(&case());
        let window = &report.run_reports[0].window_evidence[0];
        let central = window.corrected_spl_db.unwrap();
        let lower = window.corrected_spl_expanded_lower_bound_db.unwrap();
        let upper = window.corrected_spl_expanded_upper_bound_db.unwrap();
        assert!(lower <= central);
        assert!(central <= upper);
        assert_eq!(window.cavitation_component_upper_limit_db, None);
    }

    #[test]
    fn exact_ten_db_background_margin_is_unresolved() {
        assert!(matches!(
            ittc_background_corrected_spl(100.0, 90.0),
            Err(BackgroundFailure::ExactTenDbBoundaryUnspecified)
        ));
    }

    #[test]
    fn three_db_to_below_ten_uses_energy_subtraction() {
        let corrected = ittc_background_corrected_spl(99.9, 90.0).unwrap();
        assert!(corrected < 99.9);
        assert!(ittc_background_corrected_spl(93.0, 90.0).is_ok());
    }

    #[test]
    fn greater_than_ten_db_requires_no_adjustment() {
        assert_eq!(ittc_background_corrected_spl(100.1, 90.0), Ok(100.1));
    }

    #[test]
    fn below_three_db_is_not_usable_for_cavitation_summary() {
        assert!(matches!(
            ittc_background_corrected_spl(92.9, 90.0),
            Err(BackgroundFailure::BackgroundDominated)
        ));
    }

    #[test]
    fn synthetic_fixture_never_becomes_physical_qualification() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports.len(), 2);
        assert_eq!(report.run_reports[0].cavitation_occupancy_fraction, Some(0.5));
        assert!(report.reasons.iter().any(|reason| reason.contains("synthetic evidence")));
    }

    #[test]
    fn occupancy_and_onset_rate_are_reported() {
        let report = audit(&case());
        let run = &report.run_reports[0];
        assert_eq!(run.cavitation_occupancy_fraction, Some(0.5));
        assert_eq!(run.observed_cavitation_onsets, Some(4));
        assert!(run.observed_cavitation_onsets_per_hour.unwrap() > 0.0);
    }

    #[test]
    fn annotation_coverage_unknown_is_not_zero_occupancy() {
        let mut input = case();
        input.runs[0].windows[0].cavitation_present = None;
        let report = audit(&input);
        assert_eq!(report.run_reports[0].cavitation_occupancy_fraction, None);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn cavitation_conditioned_levels_are_separated() {
        let report = audit(&case());
        let run = &report.run_reports[0];
        assert!(run.cavitation_present_energy_mean_spl_db.unwrap()
            > run.cavitation_absent_energy_mean_spl_db.unwrap());
        assert!(run.cavitation_conditioned_delta_db.unwrap() > 0.0);
    }

    #[test]
    fn window_overlap_is_invalid() {
        let mut input = case();
        input.runs[0].windows[1].start_time_s = 0.5;
        input.runs[0].windows[1].end_time_s = 1.5;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn malformed_timebase_or_background_digest_is_invalid() {
        let mut input = case();
        input.runs[0].timebase_receipt_sha256 = "bad".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn uncertainty_crossing_background_threshold_excludes_window() {
        let mut input = case();
        input.runs[0].windows[0].total_spl_db = 93.1;
        input.runs[0].windows[0].background_spl_db = 90.0;
        input.runs[0].windows[0].total_expanded_uncertainty_db = 0.2;
        input.runs[0].windows[0].background_expanded_uncertainty_db = 0.2;
        let report = audit(&input);
        assert!(report.run_reports[0].excluded_uncertainty_unresolved > 0);
    }

    #[test]
    fn one_run_is_not_enough_for_run_to_run_dispersion() {
        let mut input = case();
        input.runs.truncate(1);
        let report = audit(&input);
        assert_eq!(report.run_to_run_energy_mean_sd_db, None);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn separate_runs_remain_separate_replicates() {
        let report = audit(&case());
        assert_eq!(report.run_reports.len(), 2);
        assert_eq!(report.run_reports[0].total_windows, 8);
        assert_eq!(report.run_reports[1].total_windows, 8);
    }

    #[test]
    fn per_run_sensor_environment_and_calibration_must_match_frozen_case() {
        let mut input = case();
        input.runs[1].environment_fingerprint_sha256 = digest('9');
        assert_eq!(audit(&input).disposition, "INVALID");
        let mut input = case();
        input.runs[0].hydrophone_instrument_id = "hydrophone-H2".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
        let mut input = case();
        input.runs[0].calibration_certificate_sha256 = digest('8');
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn cavitation_video_alignment_outside_tolerance_invalidates_event_annotation_only() {
        let mut input = case();
        input.runs[0].windows[0].acoustic_cavitation_alignment_offset_s = 0.01;
        input.runs[0].windows[0].acoustic_cavitation_alignment_expanded_uncertainty_s = 0.002;
        let report = audit(&input);
        let run = &report.run_reports[0];
        assert_eq!(run.excluded_alignment_unresolved, 1);
        assert_eq!(run.usable_windows, 8);
        assert_eq!(run.cavitation_annotation_coverage, 7.0 / 8.0);
        assert_eq!(run.cavitation_occupancy_fraction, None);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn alignment_uncertainty_at_declared_limit_is_eligible() {
        let mut input = case();
        input.runs[0].windows[0].acoustic_cavitation_alignment_offset_s = 0.005;
        input.runs[0].windows[0].acoustic_cavitation_alignment_expanded_uncertainty_s = 0.005;
        let report = audit(&input);
        assert_eq!(report.run_reports[0].excluded_alignment_unresolved, 0);
    }

    #[test]
    fn malformed_operating_point_or_sync_receipt_is_invalid() {
        let mut input = case();
        input.runs[0].operating_point_receipt_sha256 = "missing".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
        let mut input = case();
        input.runs[0].acoustic_cavitation_sync_receipt_sha256 = "missing".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn malformed_cavitation_camera_validation_digest_is_invalid() {
        let mut input = case();
        input.cavitation_calibration_or_validation_sha256 = "missing".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn event_rate_does_not_invent_onsets_across_gaps_or_at_run_start() {
        let mut input = case();
        // A run begins while cavitation is already present (left-censored),
        // then a gap occurs before another cavitation-present window.
        let windows = &mut input.runs[0].windows;
        windows[0].cavitation_present = Some(true);
        windows[1].cavitation_present = Some(false);
        for window in windows.iter_mut().skip(2) {
            window.start_time_s += 1.0;
            window.end_time_s += 1.0;
        }
        windows[2].cavitation_present = Some(true);
        let report = audit(&input);
        let run = &report.run_reports[0];
        assert_eq!(run.observed_cavitation_onsets, Some(2));
        assert_eq!(run.left_censored_active_segments, Some(2));
        assert_eq!(run.right_censored_active_segments, Some(1));
    }

    #[test]
    fn every_window_has_a_distinct_raw_trace_and_cavitation_receipt() {
        let report = audit(&case());
        let refs = &report.run_reports[0].window_evidence;
        let acoustic: BTreeSet<_> = refs.iter().map(|item| &item.acoustic_trace_sha256).collect();
        let cavitation: BTreeSet<_> = refs.iter().map(|item| &item.cavitation_observation_sha256).collect();
        assert_eq!(acoustic.len(), refs.len());
        assert_eq!(cavitation.len(), refs.len());
    }

    #[test]
    fn trailing_active_episode_is_right_censored() {
        let mut input = case();
        input.runs[0].windows[7].cavitation_present = Some(true);
        let report = audit(&input);
        assert_eq!(report.run_reports[0].right_censored_active_segments, Some(1));
    }

    #[test]
    fn background_exclusion_is_preserved_per_window() {
        let mut input = case();
        input.runs[0].windows[0].total_spl_db = 92.0;
        input.runs[0].windows[0].background_spl_db = 90.0;
        let report = audit(&input);
        let first = &report.run_reports[0].window_evidence[0];
        assert_eq!(first.disposition, "excluded");
        assert_eq!(
            first.exclusion_reason,
            Some("background_margin_below_3_db_censored_upper_limit")
        );
        assert!(first.cavitation_component_upper_limit_db.is_some());
        assert_eq!(first.corrected_spl_db, None);
    }

    #[test]
    fn unknown_background_method_excludes_every_window_from_spl_summary() {
        let mut input = case();
        input.runs[0].background_method = BackgroundMethod::Unknown;
        let report = audit(&input);
        assert_eq!(report.run_reports[0].usable_windows, 0);
        assert_eq!(report.run_reports[0].excluded_unknown_background_method, 8);
        assert!(report.run_reports[0].energy_equivalent_mean_spl_db.is_none());
    }

    #[test]
    fn short_acquisition_duration_is_explicitly_flagged() {
        let report = audit(&case());
        assert_eq!(report.run_reports[0].acquisition_duration_s, 8.0);
        assert!(!report.run_reports[0].ittc_recommended_acquisition_duration_met);
        assert!(report.reasons.iter().any(|reason| reason.contains("greater than 20 seconds")));
    }

    #[test]
    fn duplicate_window_trace_digests_are_invalid() {
        let mut input = case();
        input.runs[0].windows[1].acoustic_trace_sha256 =
            input.runs[0].windows[0].acoustic_trace_sha256.clone();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn empirical_quantile_uses_linear_interpolation() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(empirical_quantile(&values, 0.5), Some(2.5));
    }
}
