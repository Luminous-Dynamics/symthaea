use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const TWO_PI: f64 = std::f64::consts::TAU;

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
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    shaft_encoder_instrument_id: String,
    shaft_encoder_calibration_sha256: String,
    timebase_receipt_sha256: String,
    shaft_phase_alignment_receipt_sha256: String,
    shaft_phase_zero_reference: String,
    blade_count: u32,
    maximum_phase_uncertainty_rad: f64,
    maximum_shaft_rpm_timebase_consistency_fraction: f64,
    phase_bin_count: usize,
    minimum_samples_per_phase_bin: usize,
    minimum_phase_bin_coverage_fraction: f64,
    orders_to_measure: Vec<u32>,
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
struct Run {
    run_id: String,
    hydrophone_instrument_id: String,
    hydrophone_calibration_sha256: String,
    hydrophone_trace_sha256: String,
    run_timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    carrier_bands: Vec<CarrierBand>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CarrierBand {
    band_id: String,
    lower_frequency_hz: f64,
    upper_frequency_hz: f64,
    envelope_quantity_id: String,
    observations: Vec<PhaseObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhaseObservation {
    sample_id: String,
    timestamp_s: f64,
    revolution_index: u64,
    shaft_phase_unwrapped_rad: f64,
    shaft_phase_expanded_uncertainty_rad: f64,
    measured_shaft_rpm: f64,
    measured_shaft_rpm_expanded_uncertainty: f64,
    envelope_power: f64,
    cavitation_present: Option<bool>,
    acoustic_window_sha256: String,
    shaft_phase_evidence_sha256: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: &'static str,
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    shaft_encoder_instrument_id: String,
    shaft_encoder_calibration_sha256: String,
    timebase_receipt_sha256: String,
    shaft_phase_alignment_receipt_sha256: String,
    shaft_phase_zero_reference: String,
    blade_count: u32,
    phase_bin_count: usize,
    minimum_samples_per_phase_bin: usize,
    minimum_phase_bin_coverage_fraction: f64,
    orders_to_measure: Vec<u32>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    run_reports: Vec<RunReport>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RunReport {
    run_id: String,
    hydrophone_instrument_id: String,
    hydrophone_calibration_sha256: String,
    hydrophone_trace_sha256: String,
    run_timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    band_reports: Vec<BandReport>,
}

#[derive(Debug, Serialize)]
struct BandReport {
    band_id: String,
    lower_frequency_hz: f64,
    upper_frequency_hz: f64,
    envelope_quantity_id: &'static str,
    total_observation_count: usize,
    observed_revolution_count: usize,
    occupied_phase_bin_count: usize,
    phase_bin_coverage_fraction: f64,
    full_phase_circle_covered: bool,
    phase_alignment_eligible: bool,
    minimum_observations_per_occupied_phase_bin: usize,
    cavitation_annotation_coverage: f64,
    cavitation_window_presence_fraction: Option<f64>,
    equal_angle_mean_envelope_power: Option<f64>,
    modulation_orders: Vec<OrderModulation>,
    phase_bin_reports: Vec<PhaseBinReport>,
    disposition: &'static str,
}

#[derive(Debug, Serialize)]
struct OrderModulation {
    shaft_order: u32,
    blade_passing_order_multiple: Option<u32>,
    coefficient_real: Option<f64>,
    coefficient_imag: Option<f64>,
    modulation_amplitude_peak: Option<f64>,
    modulation_phase_rad: Option<f64>,
    modulation_index_fraction_of_mean: Option<f64>,
    phase_alignment_eligible: bool,
    disposition: &'static str,
}

#[derive(Debug, Serialize)]
struct PhaseBinReport {
    phase_bin_index: usize,
    phase_center_rad: f64,
    observation_count: usize,
    mean_envelope_power: Option<f64>,
    cavitation_presence_fraction: Option<f64>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
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
        || case.shaft_encoder_instrument_id.trim().is_empty()
        || case.shaft_phase_zero_reference.trim().is_empty()
    {
        errors.push("case/source/operating condition/encoder/shaft phase zero identity is required".to_string());
    }
    for (label, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("operating point", &case.operating_point_receipt_sha256),
        ("service fingerprint", &case.service_fingerprint_sha256),
        ("environment fingerprint", &case.environment_fingerprint_sha256),
        ("shaft encoder calibration", &case.shaft_encoder_calibration_sha256),
        ("timebase receipt", &case.timebase_receipt_sha256),
        ("shaft phase alignment receipt", &case.shaft_phase_alignment_receipt_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label} digest must be 64 hexadecimal characters"));
        }
    }
    if case.blade_count == 0
        || case.phase_bin_count < 4
        || case.minimum_samples_per_phase_bin == 0
        || !finite_positive(case.maximum_phase_uncertainty_rad)
        || !case.maximum_shaft_rpm_timebase_consistency_fraction.is_finite()
        || case.maximum_shaft_rpm_timebase_consistency_fraction <= 0.0
        || case.maximum_shaft_rpm_timebase_consistency_fraction >= 0.5
        || !case.minimum_phase_bin_coverage_fraction.is_finite()
        || case.minimum_phase_bin_coverage_fraction <= 0.0
        || case.minimum_phase_bin_coverage_fraction > 1.0
        || case.orders_to_measure.is_empty()
    {
        errors.push("blade count, phase bins, samples/bin, phase uncertainty, phase coverage and shaft orders must be positive/in range".to_string());
    }
    let mut orders = BTreeSet::new();
    for order in &case.orders_to_measure {
        if *order == 0 || *order as usize >= case.phase_bin_count / 2 || !orders.insert(*order) {
            errors.push("shaft orders must be unique, positive and below the phase-bin Nyquist order".to_string());
        }
    }
    if case.runs.is_empty() {
        errors.push("at least one run is required".to_string());
    }

    let mut run_ids = BTreeSet::new();
    for run in &case.runs {
        if run.run_id.trim().is_empty() || !run_ids.insert(run.run_id.clone()) {
            errors.push(format!("run {} has empty or duplicate run ID", run.run_id));
        }
        if run.hydrophone_instrument_id.trim().is_empty() {
            errors.push(format!("run {} hydrophone instrument ID is required", run.run_id));
        }
        for (label, digest) in [
            ("hydrophone calibration", &run.hydrophone_calibration_sha256),
            ("hydrophone raw trace", &run.hydrophone_trace_sha256),
            ("run timebase", &run.run_timebase_receipt_sha256),
            ("run operating point", &run.operating_point_receipt_sha256),
            ("run service", &run.service_fingerprint_sha256),
            ("run environment", &run.environment_fingerprint_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("run {} {label} digest is malformed", run.run_id));
            }
        }
        if run.run_timebase_receipt_sha256 != case.timebase_receipt_sha256
            || run.service_fingerprint_sha256 != case.service_fingerprint_sha256
            || run.environment_fingerprint_sha256 != case.environment_fingerprint_sha256
            || run.operating_point_receipt_sha256 != case.operating_point_receipt_sha256
        {
            errors.push(format!("run {} does not match the frozen timebase/service/environment/operating-point stratum", run.run_id));
        }
        if run.carrier_bands.is_empty() {
            errors.push(format!("run {} has no carrier bands", run.run_id));
        }
        let mut band_ids = BTreeSet::new();
        for band in &run.carrier_bands {
            if band.band_id.trim().is_empty() || !band_ids.insert(band.band_id.clone()) {
                errors.push(format!("run {} has empty or duplicate carrier-band ID", run.run_id));
            }
            if !finite_positive(band.lower_frequency_hz)
                || !finite_positive(band.upper_frequency_hz)
                || band.upper_frequency_hz <= band.lower_frequency_hz
                || band.envelope_quantity_id != "linear_envelope_power"
            {
                errors.push(format!("run {} band {} has invalid frequency bounds or envelope quantity", run.run_id, band.band_id));
            }
            if band.observations.is_empty() {
                errors.push(format!("run {} band {} has no phase observations", run.run_id, band.band_id));
            }
            let mut sample_ids = BTreeSet::new();
            let mut acoustic_digests = BTreeSet::new();
            let mut phase_digests = BTreeSet::new();
            let mut previous_phase: Option<f64> = None;
            let mut previous_timestamp: Option<f64> = None;
            let mut previous_rpm: Option<f64> = None;
            let mut previous_rpm_uncertainty: Option<f64> = None;
            for observation in &band.observations {
                if observation.sample_id.trim().is_empty() || !sample_ids.insert(observation.sample_id.clone()) {
                    errors.push(format!("run {} band {} has empty or duplicate sample ID", run.run_id, band.band_id));
                }
                if !is_sha256(&observation.acoustic_window_sha256)
                    || !is_sha256(&observation.shaft_phase_evidence_sha256)
                {
                    errors.push(format!("run {} band {} sample {} evidence digest malformed", run.run_id, band.band_id, observation.sample_id));
                }
                if !acoustic_digests.insert(observation.acoustic_window_sha256.clone())
                    || !phase_digests.insert(observation.shaft_phase_evidence_sha256.clone())
                {
                    errors.push(format!("run {} band {} reuses an acoustic-window or shaft-phase evidence digest", run.run_id, band.band_id));
                }
                if !observation.timestamp_s.is_finite()
                    || observation.timestamp_s < 0.0
                    || !observation.shaft_phase_unwrapped_rad.is_finite()
                    || observation.shaft_phase_unwrapped_rad < 0.0
                    || !finite_positive(observation.shaft_phase_expanded_uncertainty_rad)
                    || !finite_positive(observation.measured_shaft_rpm)
                    || !finite_positive(observation.measured_shaft_rpm_expanded_uncertainty)
                    || !finite_positive(observation.envelope_power)
                {
                    errors.push(format!("run {} band {} sample {} has invalid time/phase/uncertainty/power", run.run_id, band.band_id, observation.sample_id));
                }
                let expected_revolution = (observation.shaft_phase_unwrapped_rad / TWO_PI).floor() as u64;
                if expected_revolution != observation.revolution_index {
                    errors.push(format!("run {} band {} sample {} revolution_index disagrees with unwrapped phase", run.run_id, band.band_id, observation.sample_id));
                }
                if let Some(previous) = previous_phase
                    && observation.shaft_phase_unwrapped_rad <= previous
                {
                    errors.push(format!("run {} band {} shaft phase is reset, duplicate or non-monotonic", run.run_id, band.band_id));
                }
                if let Some(previous) = previous_timestamp
                    && observation.timestamp_s <= previous
                {
                    errors.push(format!("run {} band {} timestamps are not strictly increasing", run.run_id, band.band_id));
                }
                if let (Some(previous_phase_value), Some(previous_time_value), Some(previous_rpm_value), Some(previous_rpm_u)) =
                    (previous_phase, previous_timestamp, previous_rpm, previous_rpm_uncertainty)
                {
                    let elapsed_s = observation.timestamp_s - previous_time_value;
                    let phase_delta = observation.shaft_phase_unwrapped_rad - previous_phase_value;
                    if elapsed_s > 0.0 && phase_delta > 0.0 {
                        let derived_rpm = phase_delta / elapsed_s * 60.0 / TWO_PI;
                        let reference_rpm = (previous_rpm_value + observation.measured_shaft_rpm) / 2.0;
                        let uncertainty = (previous_rpm_u + observation.measured_shaft_rpm_expanded_uncertainty) / 2.0;
                        let tolerance = reference_rpm * case.maximum_shaft_rpm_timebase_consistency_fraction
                            + uncertainty;
                        if (derived_rpm - reference_rpm).abs() > tolerance {
                            errors.push(format!("run {} band {} sample {} phase/time slope disagrees with measured shaft RPM", run.run_id, band.band_id, observation.sample_id));
                        }
                    }
                }
                previous_phase = Some(observation.shaft_phase_unwrapped_rad);
                previous_timestamp = Some(observation.timestamp_s);
                previous_rpm = Some(observation.measured_shaft_rpm);
                previous_rpm_uncertainty = Some(observation.measured_shaft_rpm_expanded_uncertainty);
            }
        }
    }
    errors
}

fn audit_band(
    band: &CarrierBand,
    case: &Case,
) -> BandReport {
    let mut bin_sums = vec![0.0; case.phase_bin_count];
    let mut bin_counts = vec![0usize; case.phase_bin_count];
    let mut bin_cavitation_present = vec![0usize; case.phase_bin_count];
    let mut bin_cavitation_annotated = vec![0usize; case.phase_bin_count];

    for observation in &band.observations {
        let phase = observation.shaft_phase_unwrapped_rad.rem_euclid(TWO_PI);
        let index = ((phase / TWO_PI) * case.phase_bin_count as f64).floor() as usize;
        let index = index.min(case.phase_bin_count - 1);
        bin_sums[index] += observation.envelope_power;
        bin_counts[index] += 1;
        if let Some(present) = observation.cavitation_present {
            bin_cavitation_annotated[index] += 1;
            if present {
                bin_cavitation_present[index] += 1;
            }
        }
    }

    let occupied = bin_counts.iter().filter(|count| **count >= case.minimum_samples_per_phase_bin).count();
    let coverage = occupied as f64 / case.phase_bin_count as f64;
    let full_phase_circle_covered = occupied == case.phase_bin_count;
    let phase_alignment_eligible = band.observations.iter().all(|observation| {
        observation.shaft_phase_expanded_uncertainty_rad <= case.maximum_phase_uncertainty_rad
    });
    // The simple Fourier coefficient below assumes a uniformly sampled
    // complete phase circle. Missing bins are never interpolated.
    let sufficiently_covered = coverage >= case.minimum_phase_bin_coverage_fraction
        && full_phase_circle_covered
        && phase_alignment_eligible;
    let phase_bin_reports: Vec<PhaseBinReport> = (0..case.phase_bin_count)
        .map(|index| {
            let count = bin_counts[index];
            PhaseBinReport {
                phase_bin_index: index,
                phase_center_rad: (index as f64 + 0.5) * TWO_PI / case.phase_bin_count as f64,
                observation_count: count,
                mean_envelope_power: (count >= case.minimum_samples_per_phase_bin)
                    .then(|| bin_sums[index] / count as f64),
                cavitation_presence_fraction: (bin_cavitation_annotated[index] > 0)
                    .then(|| bin_cavitation_present[index] as f64 / bin_cavitation_annotated[index] as f64),
            }
        })
        .collect();

    let total_annotations: usize = bin_cavitation_annotated.iter().sum();
    let total_cavitation_present: usize = bin_cavitation_present.iter().sum();
    let annotation_coverage = total_annotations as f64 / band.observations.len() as f64;
    let cavitation_presence = (total_annotations == band.observations.len())
        .then(|| total_cavitation_present as f64 / total_annotations as f64);

    let complete_bins: Vec<(f64, f64)> = phase_bin_reports
        .iter()
        .filter_map(|report| report.mean_envelope_power.map(|mean| (report.phase_center_rad, mean)))
        .collect();
    let equal_angle_mean = if sufficiently_covered && !complete_bins.is_empty() {
        Some(complete_bins.iter().map(|(_, mean)| *mean).sum::<f64>() / complete_bins.len() as f64)
    } else {
        None
    };

    let mut modulation_orders = Vec::new();
    for order in &case.orders_to_measure {
        if !sufficiently_covered || complete_bins.is_empty() {
            modulation_orders.push(OrderModulation {
                shaft_order: *order,
                blade_passing_order_multiple: if *order % case.blade_count == 0 {
                    Some(*order / case.blade_count)
                } else {
                    None
                },
                coefficient_real: None,
                coefficient_imag: None,
                modulation_amplitude_peak: None,
                modulation_phase_rad: None,
                modulation_index_fraction_of_mean: None,
                phase_alignment_eligible,
                disposition: if !phase_alignment_eligible {
                    "INCOMPLETE_PHASE_ALIGNMENT"
                } else {
                    "INCOMPLETE_PHASE_BIN_COVERAGE"
                },
            });
            continue;
        }

        let n = complete_bins.len() as f64;
        let mut real = 0.0;
        let mut imag = 0.0;
        for (phase, power) in &complete_bins {
            let angle = *order as f64 * *phase;
            real += power * angle.cos();
            imag -= power * angle.sin();
        }
        // Peak sinusoidal modulation amplitude for a fully populated,
        // uniformly sampled phase circle.
        let coefficient_real = 2.0 * real / n;
        let coefficient_imag = 2.0 * imag / n;
        let amplitude = coefficient_real.hypot(coefficient_imag);
        let phase = coefficient_imag.atan2(coefficient_real);
        let index = equal_angle_mean.filter(|mean| *mean > 0.0).map(|mean| amplitude / mean);
        modulation_orders.push(OrderModulation {
            shaft_order: *order,
            blade_passing_order_multiple: if *order % case.blade_count == 0 {
                Some(*order / case.blade_count)
            } else {
                None
            },
            coefficient_real: Some(coefficient_real),
            coefficient_imag: Some(coefficient_imag),
            modulation_amplitude_peak: Some(amplitude),
            modulation_phase_rad: Some(phase),
            modulation_index_fraction_of_mean: index,
            phase_alignment_eligible,
            disposition: "ORDER_SYNCHRONOUS_MODULATION_COMPUTED",
        });
    }

    let disposition = if !phase_alignment_eligible {
        "INCOMPLETE_PHASE_ALIGNMENT"
    } else if !full_phase_circle_covered || coverage < case.minimum_phase_bin_coverage_fraction {
        "INCOMPLETE_PHASE_BIN_COVERAGE"
    } else if annotation_coverage < 1.0 {
        "MODULATION_ONLY_CAVITATION_ANNOTATION_INCOMPLETE"
    } else {
        "ORDER_SYNCHRONOUS_MODULATION_COMPUTED"
    };

    BandReport {
        band_id: band.band_id.clone(),
        lower_frequency_hz: band.lower_frequency_hz,
        upper_frequency_hz: band.upper_frequency_hz,
        envelope_quantity_id: "linear_envelope_power",
        total_observation_count: band.observations.len(),
        observed_revolution_count: band.observations.iter().map(|observation| observation.revolution_index).collect::<BTreeSet<_>>().len(),
        occupied_phase_bin_count: occupied,
        phase_bin_coverage_fraction: coverage,
        full_phase_circle_covered,
        phase_alignment_eligible,
        minimum_observations_per_occupied_phase_bin: bin_counts.iter().copied().filter(|count| *count >= case.minimum_samples_per_phase_bin).min().unwrap_or(0),
        cavitation_annotation_coverage: annotation_coverage,
        cavitation_window_presence_fraction: cavitation_presence,
        equal_angle_mean_envelope_power: equal_angle_mean,
        modulation_orders,
        phase_bin_reports,
        disposition,
    }
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut run_reports = Vec::new();
    if validation_errors.is_empty() {
        for run in &case.runs {
            let mut band_reports = Vec::new();
            for band in &run.carrier_bands {
                band_reports.push(audit_band(band, case));
            }
            run_reports.push(RunReport {
                run_id: run.run_id.clone(),
                hydrophone_instrument_id: run.hydrophone_instrument_id.clone(),
                hydrophone_calibration_sha256: run.hydrophone_calibration_sha256.clone(),
                hydrophone_trace_sha256: run.hydrophone_trace_sha256.clone(),
                run_timebase_receipt_sha256: run.run_timebase_receipt_sha256.clone(),
                operating_point_receipt_sha256: run.operating_point_receipt_sha256.clone(),
                service_fingerprint_sha256: run.service_fingerprint_sha256.clone(),
                environment_fingerprint_sha256: run.environment_fingerprint_sha256.clone(),
                band_reports,
            });
        }
    }
    if run_reports.iter().flat_map(|run| &run.band_reports)
        .any(|band| band.disposition == "INCOMPLETE_PHASE_BIN_COVERAGE")
    {
        reasons.push("one or more carrier-band runs do not provide a complete, sufficiently sampled phase circle".to_string());
    }
    if run_reports.iter().flat_map(|run| &run.band_reports)
        .any(|band| band.disposition == "INCOMPLETE_PHASE_ALIGNMENT")
    {
        reasons.push("one or more carrier-band runs exceed the predeclared phase alignment uncertainty".to_string());
    }
    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot qualify measured propeller modulation".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source data rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source metadata has not been independently reviewed".to_string());
    }
    let disposition = if !validation_errors.is_empty() {
        "INVALID"
    } else if case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || reasons.iter().any(|reason| reason.contains("phase circle"))
        || reasons.iter().any(|reason| reason.contains("phase alignment uncertainty"))
    {
        "INCOMPLETE"
    } else {
        "ORDER_SYNCHRONOUS_MODULATION_ONLY"
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
        operating_point_receipt_sha256: case.operating_point_receipt_sha256.clone(),
        service_fingerprint_sha256: case.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: case.environment_fingerprint_sha256.clone(),
        shaft_encoder_instrument_id: case.shaft_encoder_instrument_id.clone(),
        shaft_encoder_calibration_sha256: case.shaft_encoder_calibration_sha256.clone(),
        timebase_receipt_sha256: case.timebase_receipt_sha256.clone(),
        shaft_phase_alignment_receipt_sha256: case.shaft_phase_alignment_receipt_sha256.clone(),
        shaft_phase_zero_reference: case.shaft_phase_zero_reference.clone(),
        blade_count: case.blade_count,
        phase_bin_count: case.phase_bin_count,
        minimum_samples_per_phase_bin: case.minimum_samples_per_phase_bin,
        minimum_phase_bin_coverage_fraction: case.minimum_phase_bin_coverage_fraction,
        orders_to_measure: case.orders_to_measure.clone(),
        disposition,
        claim_ceiling: "shaft_phase_locked_linear_envelope_modulation_for_declared_carrier_band_only_no_unique_source_causality_full_scale_transfer_ecological_or_regulatory_qualification",
        run_reports,
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-rotation-synchronous-modulation-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-rotation-synchronous-modulation-audit <case.json>");
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
        "ORDER_SYNCHRONOUS_MODULATION_ONLY" => {}
        "INVALID" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    fn sample(
        index: usize,
        bins: usize,
        revolutions: usize,
        order: u32,
        mean: f64,
        amplitude: f64,
        phase_offset: f64,
    ) -> Vec<PhaseObservation> {
        let mut output = Vec::new();
        for revolution in 0..revolutions {
            for bin in 0..bins {
                let phase = revolution as f64 * TWO_PI
                    + (bin as f64 + 0.5) * TWO_PI / bins as f64;
                let power = mean + amplitude * (order as f64 * phase + phase_offset).cos();
                let item_index = revolution * bins + bin;
                output.push(PhaseObservation {
                    sample_id: format!("sample-{index}-{item_index}"),
                    timestamp_s: phase / (TWO_PI * 10.0),
                    revolution_index: revolution as u64,
                    shaft_phase_unwrapped_rad: phase,
                    shaft_phase_expanded_uncertainty_rad: 0.0001,
                    measured_shaft_rpm: 600.0,
                    measured_shaft_rpm_expanded_uncertainty: 0.1,
                    envelope_power: power,
                    cavitation_present: Some((phase.sin() + 1.0) > 1.0),
                    acoustic_window_sha256: format!("{:064x}", 10_000 + index * 1_000 + item_index),
                    shaft_phase_evidence_sha256: format!("{:064x}", 50_000 + index * 1_000 + item_index),
                });
            }
        }
        output
    }

    fn case() -> Case {
        let bins = 36;
        let runs = (0..2).map(|run_index| Run {
            run_id: format!("run-{run_index}"),
            hydrophone_instrument_id: "hydrophone-H1".to_string(),
            hydrophone_calibration_sha256: digest('a'),
            hydrophone_trace_sha256: digest(if run_index == 0 { 'b' } else { 'c' }),
            run_timebase_receipt_sha256: digest('d'),
            operating_point_receipt_sha256: digest('e'),
            service_fingerprint_sha256: digest('f'),
            environment_fingerprint_sha256: digest('1'),
            carrier_bands: vec![
                CarrierBand {
                    band_id: "2-5kHz".to_string(),
                    lower_frequency_hz: 2000.0,
                    upper_frequency_hz: 5000.0,
                    envelope_quantity_id: "linear_envelope_power".to_string(),
                    observations: sample(run_index, bins, 4, 5, 10.0, 2.0, 0.3),
                },
                CarrierBand {
                    band_id: "30-80kHz".to_string(),
                    lower_frequency_hz: 30000.0,
                    upper_frequency_hz: 80000.0,
                    envelope_quantity_id: "linear_envelope_power".to_string(),
                    observations: sample(run_index + 2, bins, 4, 7, 8.0, 0.0, 0.0),
                },
            ],
        }).collect();
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-rotation-synchronous-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-phase-locked-envelope".to_string(),
                citation: "Synthetic analytic known-answer fixture".to_string(),
                raw_data_sha256: digest('2'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            operating_condition_id: "J0.8-sigma-nD1".to_string(),
            operating_point_receipt_sha256: digest('e'),
            service_fingerprint_sha256: digest('f'),
            environment_fingerprint_sha256: digest('1'),
            shaft_encoder_instrument_id: "shaft-encoder-1".to_string(),
            shaft_encoder_calibration_sha256: digest('3'),
            timebase_receipt_sha256: digest('d'),
            shaft_phase_alignment_receipt_sha256: digest('4'),
            shaft_phase_zero_reference: "index_pulse_at_blade_1_reference".to_string(),
            blade_count: 5,
            maximum_phase_uncertainty_rad: 0.001,
            maximum_shaft_rpm_timebase_consistency_fraction: 0.01,
            phase_bin_count: bins,
            minimum_samples_per_phase_bin: 4,
            minimum_phase_bin_coverage_fraction: 1.0,
            orders_to_measure: vec![1, 5, 7, 10],
            runs,
        }
    }

    #[test]
    fn known_order_modulation_amplitude_and_phase_are_recovered() {
        let report = audit(&case());
        let band = &report.run_reports[0].band_reports[0];
        let order = band.modulation_orders.iter().find(|entry| entry.shaft_order == 5).unwrap();
        assert!((order.modulation_amplitude_peak.unwrap() - 2.0).abs() < 1.0e-9);
        assert!((order.modulation_phase_rad.unwrap() - 0.3).abs() < 1.0e-9);
        assert!((order.modulation_index_fraction_of_mean.unwrap() - 0.2).abs() < 1.0e-9);
    }

    #[test]
    fn constant_envelope_has_zero_order_modulation() {
        let report = audit(&case());
        let band = &report.run_reports[0].band_reports[1];
        for order in &band.modulation_orders {
            assert!(order.modulation_amplitude_peak.unwrap().abs() < 1.0e-9);
            assert!(order.modulation_index_fraction_of_mean.unwrap().abs() < 1.0e-9);
        }
    }

    #[test]
    fn order_shift_changes_reported_order_not_mean_power() {
        let input = case();
        let mut shifted = case();
        for run in &mut shifted.runs {
            run.carrier_bands[0].observations = sample(99, 36, 4, 7, 10.0, 2.0, 0.3);
        }
        let base = audit(&input);
        let next = audit(&shifted);
        let base_band = &base.run_reports[0].band_reports[0];
        let next_band = &next.run_reports[0].band_reports[0];
        assert!((base_band.equal_angle_mean_envelope_power.unwrap() - next_band.equal_angle_mean_envelope_power.unwrap()).abs() < 1.0e-9);
        assert!(base_band.modulation_orders.iter().find(|x| x.shaft_order == 5).unwrap().modulation_amplitude_peak.unwrap() > 1.0);
        assert!(next_band.modulation_orders.iter().find(|x| x.shaft_order == 7).unwrap().modulation_amplitude_peak.unwrap() > 1.0);
    }

    #[test]
    fn phase_offset_rotates_complex_coefficient_without_changing_magnitude() {
        let mut shifted = case();
        for run in &mut shifted.runs {
            run.carrier_bands[0].observations = sample(99, 36, 4, 5, 10.0, 2.0, 1.1);
        }
        let report = audit(&shifted);
        let order = report.run_reports[0].band_reports[0].modulation_orders.iter().find(|x| x.shaft_order == 5).unwrap();
        assert!((order.modulation_amplitude_peak.unwrap() - 2.0).abs() < 1.0e-9);
        assert!((order.modulation_phase_rad.unwrap() - 1.1).abs() < 1.0e-9);
    }

    #[test]
    fn phase_reset_and_reversed_samples_are_invalid() {
        let mut input = case();
        input.runs[0].carrier_bands[0].observations[1].shaft_phase_unwrapped_rad = 0.1;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn missing_phase_bin_makes_the_modulation_incomplete() {
        let mut input = case();
        input.runs[0].carrier_bands[0].observations.retain(|sample| {
            (sample.shaft_phase_unwrapped_rad.rem_euclid(TWO_PI) / TWO_PI * 36.0).floor() as usize != 10
        });
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("phase-bin coverage")));
    }

    #[test]
    fn inconsistent_phase_time_slope_is_invalid() {
        let mut input = case();
        input.runs[0].carrier_bands[0].observations[1].timestamp_s += 0.01;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn excessive_phase_uncertainty_is_incomplete_not_invalid() {
        let mut input = case();
        input.runs[0].carrier_bands[0].observations[0].shaft_phase_expanded_uncertainty_rad = 0.01;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.run_reports[0].band_reports[0].disposition, "INCOMPLETE_PHASE_ALIGNMENT");
        assert!(report.run_reports[0].band_reports[0].modulation_orders[0].modulation_amplitude_peak.is_none());
    }

    #[test]
    fn unannotated_cavitation_does_not_become_zero_occupancy() {
        let mut input = case();
        for sample in &mut input.runs[0].carrier_bands[0].observations {
            sample.cavitation_present = None;
        }
        let report = audit(&input);
        assert_eq!(report.run_reports[0].band_reports[0].cavitation_window_presence_fraction, None);
    }

    #[test]
    fn synthetic_result_is_never_physically_qualified() {
        assert_eq!(audit(&case()).disposition, "INCOMPLETE");
    }
}
