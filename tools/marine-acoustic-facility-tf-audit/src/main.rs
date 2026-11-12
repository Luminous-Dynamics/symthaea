use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind { Synthetic, Measured }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum RightsStatus { PublicOrPermitted, Unknown, Restricted }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum TvRBasis { IndependentlyMeasured, ManufacturerFreeField, Unknown }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum FrequencyQuantity { OneThirdOctaveBandSpl, PowerSpectralDensity }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum SignalType { BroadbandNoise, Tonal, Chirp, PseudoRandom }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    facility: Facility,
    projector: Projector,
    receiver: Receiver,
    repeatability_tolerance_db: f64,
    samples: Vec<Sample>,
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
struct Facility {
    facility_id: String,
    test_section_revision: String,
    geometry_sha256: String,
    is_closed_test_section: bool,
    operating_condition_fingerprint_sha256: String,
    water_condition_evidence_sha256: String,
    flow_state_id: String,
    free_surface_state_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Projector {
    instrument_id: String,
    projector_type: String,
    projector_revision: String,
    transmitter_chain_id: String,
    transmitter_chain_evidence_sha256: String,
    signal_type: SignalType,
    tvr_basis: TvRBasis,
    tvr_or_reference_source_level_evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receiver {
    hydrophone_instrument_id: String,
    hydrophone_calibration_certificate_sha256: String,
    acquisition_chain_id: String,
    acquisition_chain_evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Position {
    position_id: String,
    x_m: f64,
    y_m: f64,
    z_m: f64,
    expanded_position_uncertainty_m: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    sample_id: String,
    repeat_id: String,
    source_position: Position,
    receiver_position: Position,
    frequency_band_id: String,
    center_frequency_hz: f64,
    quantity: FrequencyQuantity,
    bandwidth_id: String,
    source_level_db: f64,
    source_level_expanded_uncertainty_db: f64,
    received_spl_db: f64,
    received_spl_expanded_uncertainty_db: f64,
    additional_source_basis_uncertainty_db: Option<f64>,
    source_trace_sha256: String,
    received_trace_sha256: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    facility_id: String,
    facility_revision: String,
    projector_id: String,
    receiver_id: String,
    tvr_basis: &'static str,
    disposition: &'static str,
    claim_ceiling: &'static str,
    position_band_results: Vec<PositionBandResult>,
    spatial_band_results: Vec<SpatialBandResult>,
    facility_wide_representative_factor_established: bool,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PositionBandResult {
    source_position_id: String,
    receiver_position_id: String,
    frequency_band_id: String,
    center_frequency_hz: f64,
    quantity: &'static str,
    bandwidth_id: String,
    repeat_count: usize,
    mean_transfer_function_db: Option<f64>,
    minimum_transfer_function_db: Option<f64>,
    maximum_transfer_function_db: Option<f64>,
    repeat_standard_deviation_db: Option<f64>,
    maximum_single_measurement_expanded_bound_db: Option<f64>,
    conservative_mean_expanded_bound_db: Option<f64>,
    repeatability_tolerance_db: f64,
    repeatability_disposition: &'static str,
    source_position: Position,
    receiver_position: Position,
    samples: Vec<SampleReceipt>,
}

#[derive(Debug, Serialize)]
struct SampleReceipt {
    sample_id: String,
    repeat_id: String,
    transfer_function_db: Option<f64>,
    expanded_bound_db: Option<f64>,
    disposition: &'static str,
    reason: Option<&'static str>,
    source_trace_sha256: String,
    received_trace_sha256: String,
}

#[derive(Debug, Serialize)]
struct SpatialBandResult {
    frequency_band_id: String,
    center_frequency_hz: f64,
    position_pair_count: usize,
    minimum_position_mean_tf_db: Option<f64>,
    maximum_position_mean_tf_db: Option<f64>,
    spatial_range_db: Option<f64>,
    conservative_spatial_range_lower_db: Option<f64>,
    conservative_spatial_range_upper_db: Option<f64>,
    spatial_consistency_tolerance_db: f64,
    spatial_consistency_disposition: &'static str,
    extrapolation_performed: bool,
}

#[derive(Debug)]
struct GroupKey {
    source_position_id: String,
    receiver_position_id: String,
    band_id: String,
}

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.source_position_id == other.source_position_id
            && self.receiver_position_id == other.receiver_position_id
            && self.band_id == other.band_id
    }
}
impl Eq for GroupKey {}
impl PartialOrd for GroupKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GroupKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.source_position_id, &self.receiver_position_id, &self.band_id)
            .cmp(&(&other.source_position_id, &other.receiver_position_id, &other.band_id))
    }
}

fn digest_ok(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn finite_positive(value: f64) -> bool { value.is_finite() && value > 0.0 }

fn tvr_name(value: TvRBasis) -> &'static str {
    match value {
        TvRBasis::IndependentlyMeasured => "independently_measured",
        TvRBasis::ManufacturerFreeField => "manufacturer_free_field",
        TvRBasis::Unknown => "unknown",
    }
}
fn quantity_name(value: FrequencyQuantity) -> &'static str {
    match value {
        FrequencyQuantity::OneThirdOctaveBandSpl => "one_third_octave_band_spl",
        FrequencyQuantity::PowerSpectralDensity => "power_spectral_density",
    }
}

fn validate_position(position: &Position, label: &str, errors: &mut Vec<String>) {
    if position.position_id.trim().is_empty()
        || !position.x_m.is_finite()
        || !position.y_m.is_finite()
        || !position.z_m.is_finite()
        || !finite_positive(position.expanded_position_uncertainty_m)
    {
        errors.push(format!("{label}: invalid position identity, coordinate or uncertainty"));
    }
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    if case.case_id.trim().is_empty()
        || case.source.dataset_id.trim().is_empty()
        || case.source.citation.trim().is_empty()
        || case.facility.facility_id.trim().is_empty()
        || case.facility.test_section_revision.trim().is_empty()
        || case.projector.instrument_id.trim().is_empty()
        || case.projector.projector_type.trim().is_empty()
        || case.projector.projector_revision.trim().is_empty()
        || case.projector.transmitter_chain_id.trim().is_empty()
        || case.receiver.hydrophone_instrument_id.trim().is_empty()
        || case.receiver.acquisition_chain_id.trim().is_empty()
        || case.facility.flow_state_id.trim().is_empty()
        || case.facility.free_surface_state_id.trim().is_empty()
    {
        errors.push("case, facility, instrument, signal-chain and state identities are required".to_string());
    }
    for (label, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("facility geometry", &case.facility.geometry_sha256),
        ("facility operating-condition", &case.facility.operating_condition_fingerprint_sha256),
        ("water-condition evidence", &case.facility.water_condition_evidence_sha256),
        ("transmitter-chain evidence", &case.projector.transmitter_chain_evidence_sha256),
        ("TVR/source-level evidence", &case.projector.tvr_or_reference_source_level_evidence_sha256),
        ("hydrophone calibration", &case.receiver.hydrophone_calibration_certificate_sha256),
        ("acquisition-chain evidence", &case.receiver.acquisition_chain_evidence_sha256),
    ] {
        if !digest_ok(digest) {
            errors.push(format!("{label} digest must be 64 hexadecimal characters"));
        }
    }
    if !finite_positive(case.repeatability_tolerance_db) {
        errors.push("repeatability_tolerance_db must be positive".to_string());
    }
    if case.samples.is_empty() {
        errors.push("at least one transfer-function sample is required".to_string());
    }
    let mut sample_ids = BTreeSet::new();
    let mut per_repeat = BTreeSet::new();
    let mut position_coordinates: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
    let mut source_trace_digests = BTreeSet::new();
    let mut received_trace_digests = BTreeSet::new();
    let mut band_metadata: BTreeMap<String, (f64, FrequencyQuantity, String)> = BTreeMap::new();
    for sample in &case.samples {
        if sample.sample_id.trim().is_empty() || !sample_ids.insert(sample.sample_id.clone()) {
            errors.push(format!("empty or duplicate sample_id {}", sample.sample_id));
        }
        if sample.repeat_id.trim().is_empty() || !per_repeat.insert((
            sample.source_position.position_id.clone(),
            sample.receiver_position.position_id.clone(),
            sample.frequency_band_id.clone(),
            sample.repeat_id.clone(),
        )) {
            errors.push(format!("empty or duplicate repeat_id in sample {}", sample.sample_id));
        }
        validate_position(&sample.source_position, "source position", &mut errors);
        validate_position(&sample.receiver_position, "receiver position", &mut errors);
        for (id, pos) in [
            (&sample.source_position.position_id, &sample.source_position),
            (&sample.receiver_position.position_id, &sample.receiver_position),
        ] {
            let coords = (pos.x_m, pos.y_m, pos.z_m);
            if let Some(existing) = position_coordinates.get(id) {
                if (existing.0 - coords.0).abs() > 1.0e-9
                    || (existing.1 - coords.1).abs() > 1.0e-9
                    || (existing.2 - coords.2).abs() > 1.0e-9
                {
                    errors.push(format!("position_id {} is reused with different coordinates", id));
                }
            } else {
                position_coordinates.insert(id.clone(), coords);
            }
        }
        if sample.frequency_band_id.trim().is_empty()
            || sample.bandwidth_id.trim().is_empty()
            || !finite_positive(sample.center_frequency_hz)
            || !sample.source_level_db.is_finite()
            || !finite_positive(sample.source_level_expanded_uncertainty_db)
            || !sample.received_spl_db.is_finite()
            || !finite_positive(sample.received_spl_expanded_uncertainty_db)
        {
            errors.push(format!("sample {} has invalid frequency, source/received levels or uncertainty", sample.sample_id));
        }
        if !digest_ok(&sample.source_trace_sha256) || !digest_ok(&sample.received_trace_sha256) {
            errors.push(format!("sample {} source/received trace digest is malformed", sample.sample_id));
        }
        if !source_trace_digests.insert(sample.source_trace_sha256.clone())
            || !received_trace_digests.insert(sample.received_trace_sha256.clone())
        {
            errors.push(format!("sample {} reuses a source or received trace digest", sample.sample_id));
        }
        let metadata = (sample.center_frequency_hz, sample.quantity, sample.bandwidth_id.clone());
        if let Some(existing) = band_metadata.get(&sample.frequency_band_id) {
            if (existing.0 - metadata.0).abs() > 1.0e-9
                || existing.1 != metadata.1
                || existing.2 != metadata.2
            {
                errors.push(format!(
                    "frequency band {} is pooled across different centre frequency, quantity or bandwidth",
                    sample.frequency_band_id
                ));
            }
        } else {
            band_metadata.insert(sample.frequency_band_id.clone(), metadata);
        }
        if let Some(additional) = sample.additional_source_basis_uncertainty_db
            && (!additional.is_finite() || additional <= 0.0)
        {
            errors.push(format!("sample {} additional source-basis uncertainty must be positive", sample.sample_id));
        }
    }
    errors
}

fn sample_transfer(case: &Case, sample: &Sample) -> (Option<f64>, Option<f64>, &'static str, Option<&'static str>) {
    if case.projector.tvr_basis == TvRBasis::Unknown {
        return (None, None, "provisional_source_basis", Some("unknown_tvr_or_source_level_basis"));
    }
    let extra = match case.projector.tvr_basis {
        TvRBasis::IndependentlyMeasured => 0.0,
        TvRBasis::ManufacturerFreeField if case.facility.is_closed_test_section => {
            match sample.additional_source_basis_uncertainty_db {
                Some(value) => value,
                None => return (None, None, "provisional_source_basis", Some("closed_facility_free_field_tvr_requires_additional_uncertainty")),
            }
        }
        TvRBasis::ManufacturerFreeField => 0.0,
        TvRBasis::Unknown => unreachable!(),
    };
    let tf = sample.received_spl_db - sample.source_level_db;
    let bound = sample.received_spl_expanded_uncertainty_db
        + sample.source_level_expanded_uncertainty_db
        + extra;
    (Some(tf), Some(bound), "measured_position_tf", None)
}

fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 { return None; }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_sq = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    Some((sum_sq / (values.len() - 1) as f64).sqrt())
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut grouped: BTreeMap<GroupKey, Vec<&Sample>> = BTreeMap::new();
    if validation_errors.is_empty() {
        for sample in &case.samples {
            grouped.entry(GroupKey {
                source_position_id: sample.source_position.position_id.clone(),
                receiver_position_id: sample.receiver_position.position_id.clone(),
                band_id: sample.frequency_band_id.clone(),
            }).or_default().push(sample);
        }
    }
    let mut position_band_results = Vec::new();
    let mut spatial_by_band: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut has_provisional_basis = false;
    let mut has_spatial_variation = false;
    let mut has_repeatability_failure = false;

    for (key, samples) in grouped {
        let first = samples[0];
        let mut tf_values = Vec::new();
        let mut receipts = Vec::new();
        let mut max_bound: f64 = 0.0;
        let mut position_disposition = "repeatability_unresolved";
        for sample in &samples {
            let (tf, bound, disposition, reason) = sample_transfer(case, sample);
            if let Some(value) = tf { tf_values.push(value); }
            if let Some(value) = bound { max_bound = max_bound.max(value); }
            if disposition == "provisional_source_basis" {
                has_provisional_basis = true;
                position_disposition = "provisional_source_basis";
            }
            receipts.push(SampleReceipt {
                sample_id: sample.sample_id.clone(),
                repeat_id: sample.repeat_id.clone(),
                transfer_function_db: tf,
                expanded_bound_db: bound,
                disposition,
                reason,
                source_trace_sha256: sample.source_trace_sha256.clone(),
                received_trace_sha256: sample.received_trace_sha256.clone(),
            });
        }
        let mean = if tf_values.is_empty() { None } else { Some(tf_values.iter().sum::<f64>() / tf_values.len() as f64) };
        let min = tf_values.iter().copied().reduce(f64::min);
        let max = tf_values.iter().copied().reduce(f64::max);
        let sd = sample_sd(&tf_values);
        let range = match (min, max) { (Some(lo), Some(hi)) => Some(hi - lo), _ => None };
        let repeatability_lower = range.map(|value| (value - 2.0 * max_bound).max(0.0));
        let repeatability_upper = range.map(|value| value + 2.0 * max_bound);
        if samples.len() >= 2 && tf_values.len() == samples.len() {
            match (repeatability_lower, repeatability_upper) {
                (_, Some(upper)) if upper <= case.repeatability_tolerance_db => {
                    position_disposition = "repeatable_position_tf";
                }
                (Some(lower), _) if lower > case.repeatability_tolerance_db => {
                    has_repeatability_failure = true;
                    position_disposition = "repeatability_outside_tolerance";
                }
                _ => {
                    position_disposition = "repeatability_uncertainty_overlaps_tolerance";
                }
            }
        } else if !has_provisional_basis {
            position_disposition = "incomplete_insufficient_repeats";
        }
        if let Some(value) = mean {
            spatial_by_band.entry(key.band_id.clone()).or_default().push((value, max_bound));
        }
        position_band_results.push(PositionBandResult {
            source_position_id: key.source_position_id,
            receiver_position_id: key.receiver_position_id,
            frequency_band_id: key.band_id,
            center_frequency_hz: first.center_frequency_hz,
            quantity: quantity_name(first.quantity),
            bandwidth_id: first.bandwidth_id.clone(),
            repeat_count: samples.len(),
            mean_transfer_function_db: mean,
            minimum_transfer_function_db: min,
            maximum_transfer_function_db: max,
            repeat_standard_deviation_db: sd,
            maximum_single_measurement_expanded_bound_db: if bound_is_zero(max_bound) { None } else { Some(max_bound) },
            conservative_mean_expanded_bound_db: mean.map(|_| max_bound),
            repeatability_tolerance_db: case.repeatability_tolerance_db,
            repeatability_disposition: position_disposition,
            source_position: clone_position(&first.source_position),
            receiver_position: clone_position(&first.receiver_position),
            samples: receipts,
        });
    }

    let mut spatial_band_results = Vec::new();
    for (band_id, values) in spatial_by_band {
        let min = values.iter().map(|(value, _)| *value).reduce(f64::min);
        let max = values.iter().map(|(value, _)| *value).reduce(f64::max);
        let max_bound = values.iter().map(|(_, bound)| *bound).fold(0.0_f64, f64::max);
        let range = match (min, max) { (Some(lo), Some(hi)) => Some(hi - lo), _ => None };
        let lower = range.map(|value| (value - 2.0 * max_bound).max(0.0));
        let upper = range.map(|value| value + 2.0 * max_bound);
        let disposition = if values.len() < 2 {
            "single_position_no_spatial_claim"
        } else if lower.is_some_and(|value| value > case.repeatability_tolerance_db) {
            has_spatial_variation = true;
            "spatially_variable_beyond_uncertainty"
        } else if upper.is_some_and(|value| value <= case.repeatability_tolerance_db) {
            "consistent_over_sampled_positions_only"
        } else {
            "spatial_uncertainty_overlaps_tolerance"
        };
        spatial_band_results.push(SpatialBandResult {
            frequency_band_id: band_id.clone(),
            center_frequency_hz: case.samples.iter().find(|sample| sample.frequency_band_id == band_id)
                .map(|sample| sample.center_frequency_hz).unwrap_or_default(),
            position_pair_count: values.len(),
            minimum_position_mean_tf_db: min,
            maximum_position_mean_tf_db: max,
            spatial_range_db: range,
            conservative_spatial_range_lower_db: lower,
            conservative_spatial_range_upper_db: upper,
            spatial_consistency_tolerance_db: case.repeatability_tolerance_db,
            spatial_consistency_disposition: disposition,
            extrapolation_performed: false,
        });
    }

    if case.projector.tvr_basis == TvRBasis::ManufacturerFreeField && case.facility.is_closed_test_section
        && case.samples.iter().any(|sample| sample.additional_source_basis_uncertainty_db.is_none())
    {
        reasons.push("manufacturer free-field TVR in a closed facility requires additional source-basis uncertainty".to_string());
    }
    if case.projector.tvr_basis == TvRBasis::Unknown {
        reasons.push("source-level/TVR basis is unknown; no measured transfer function can be computed".to_string());
    }
    if has_spatial_variation {
        reasons.push("facility transfer function varies across sampled source/receiver positions; no single representative factor is justified".to_string());
    }
    if has_repeatability_failure {
        reasons.push("same-position repeat scatter exceeds the declared tolerance; position transfer function is not repeatable".to_string());
    }
    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot qualify a measured facility transfer function".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted || !case.source.independently_reviewed {
        reasons.push("source rights or independent review are not eligible for a measured calibration claim".to_string());
    }

    let disposition = if !validation_errors.is_empty() {
        "INVALID"
    } else if has_spatial_variation || has_repeatability_failure {
        "SPATIALLY_VARIABLE"
    } else if has_provisional_basis {
        "PROVISIONAL_SOURCE_BASIS"
    } else if case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || position_band_results.iter().any(|x| x.repeat_count < 2)
        || position_band_results.iter().any(|x| x.repeatability_disposition != "repeatable_position_tf")
    {
        "INCOMPLETE"
    } else {
        "REPEATABLE_POSITION_TF"
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        facility_id: case.facility.facility_id.clone(),
        facility_revision: case.facility.test_section_revision.clone(),
        projector_id: case.projector.instrument_id.clone(),
        receiver_id: case.receiver.hydrophone_instrument_id.clone(),
        tvr_basis: tvr_name(case.projector.tvr_basis),
        disposition,
        claim_ceiling: "position_and_frequency_specific_transfer_function_from_declared_calibration_campaign_only_no_facility_wide_interpolation_free_field_correction_physical_cavitation_prediction_or_full_scale_qualification",
        position_band_results,
        spatial_band_results,
        facility_wide_representative_factor_established: false,
        reasons,
    }
}

fn bound_is_zero(value: f64) -> bool { value == 0.0 }

fn clone_position(position: &Position) -> Position {
    Position {
        position_id: position.position_id.clone(),
        x_m: position.x_m,
        y_m: position.y_m,
        z_m: position.z_m,
        expanded_position_uncertainty_m: position.expanded_position_uncertainty_m,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-acoustic-facility-tf-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-acoustic-facility-tf-audit <case.json>");
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
        "REPEATABLE_POSITION_TF" => {}
        "INVALID" | "SPATIALLY_VARIABLE" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String { std::iter::repeat_n(c, 64).collect() }

    fn position(id: &str, x: f64, y: f64, z: f64) -> Position {
        Position {
            position_id: id.to_string(),
            x_m: x,
            y_m: y,
            z_m: z,
            expanded_position_uncertainty_m: 0.001,
        }
    }

    fn sample(id: &str, repeat: &str, source_x: f64, receiver_y: f64, center_hz: f64, source_level: f64, received: f64) -> Sample {
        Sample {
            sample_id: id.to_string(),
            repeat_id: repeat.to_string(),
            source_position: position("source-position-A", source_x, 0.0, 0.0),
            receiver_position: position(if receiver_y == 0.0 { "receiver-position-A" } else { "receiver-position-B" }, 0.0, receiver_y, 0.0),
            frequency_band_id: format!("band-{center_hz:.0}Hz"),
            center_frequency_hz: center_hz,
            quantity: FrequencyQuantity::OneThirdOctaveBandSpl,
            bandwidth_id: "one-third-octave".to_string(),
            source_level_db: source_level,
            source_level_expanded_uncertainty_db: 0.1,
            received_spl_db: received,
            received_spl_expanded_uncertainty_db: 0.1,
            additional_source_basis_uncertainty_db: None,
            source_trace_sha256: digest(if repeat == "r1" { 'a' } else { 'b' }),
            received_trace_sha256: digest(if id.ends_with('1') { 'c' } else { 'd' }),
        }
    }

    fn case() -> Case {
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-facility-tf-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-facility-tf".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('e'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            facility: Facility {
                facility_id: "tunnel-A".to_string(),
                test_section_revision: "section-r2".to_string(),
                geometry_sha256: digest('1'),
                is_closed_test_section: true,
                operating_condition_fingerprint_sha256: digest('2'),
                water_condition_evidence_sha256: digest('3'),
                flow_state_id: "flow-off-reference".to_string(),
                free_surface_state_id: "no-free-surface".to_string(),
            },
            projector: Projector {
                instrument_id: "projector-P1".to_string(),
                projector_type: "underwater-projector".to_string(),
                projector_revision: "p-r1".to_string(),
                transmitter_chain_id: "drive-chain-1".to_string(),
                transmitter_chain_evidence_sha256: digest('4'),
                signal_type: SignalType::BroadbandNoise,
                tvr_basis: TvRBasis::IndependentlyMeasured,
                tvr_or_reference_source_level_evidence_sha256: digest('5'),
            },
            receiver: Receiver {
                hydrophone_instrument_id: "hydrophone-H1".to_string(),
                hydrophone_calibration_certificate_sha256: digest('6'),
                acquisition_chain_id: "daq-1".to_string(),
                acquisition_chain_evidence_sha256: digest('7'),
            },
            repeatability_tolerance_db: 1.5,
            samples: vec![
                sample("sample-1", "r1", 0.0, 0.0, 1000.0, 150.0, 120.0),
                sample("sample-2", "r2", 0.0, 0.0, 1000.0, 150.0, 121.0),
            ],
        }
    }

    #[test]
    fn independent_tvr_same_position_repeats_compute_known_tf() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        let report = audit(&input);
        assert_eq!(report.disposition, "REPEATABLE_POSITION_TF");
        assert_eq!(report.position_band_results[0].mean_transfer_function_db, Some(-29.5));
        assert_eq!(report.position_band_results[0].repeatability_disposition, "repeatable_position_tf");
        assert_eq!(report.position_band_results[0].repeat_count, 2);
        assert!(!report.facility_wide_representative_factor_established);
    }

    #[test]
    fn synthetic_fixture_never_becomes_measured_calibration() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn unknown_tvr_basis_never_emits_transfer_value() {
        let mut input = case();
        input.projector.tvr_basis = TvRBasis::Unknown;
        let report = audit(&input);
        assert_eq!(report.disposition, "PROVISIONAL_SOURCE_BASIS");
        assert!(report.position_band_results[0].mean_transfer_function_db.is_none());
    }

    #[test]
    fn closed_facility_free_field_tvr_requires_additional_uncertainty() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.projector.tvr_basis = TvRBasis::ManufacturerFreeField;
        let report = audit(&input);
        assert_eq!(report.disposition, "PROVISIONAL_SOURCE_BASIS");
        assert!(report.position_band_results[0].mean_transfer_function_db.is_none());
    }

    #[test]
    fn closed_facility_free_field_tvr_with_extra_uncertainty_is_provisional() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.projector.tvr_basis = TvRBasis::ManufacturerFreeField;
        for sample in &mut input.samples {
            sample.additional_source_basis_uncertainty_db = Some(1.5);
        }
        let report = audit(&input);
        assert_eq!(report.disposition, "PROVISIONAL_SOURCE_BASIS");
        assert!(report.position_band_results[0].mean_transfer_function_db.is_some());
    }

    #[test]
    fn changed_receiver_position_is_a_distinct_spatial_group() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        let mut third = input.samples[0].clone();
        third.sample_id = "sample-3".to_string();
        third.repeat_id = "r1".to_string();
        third.receiver_position = position("receiver-position-B", 0.0, 1.0, 0.0);
        third.received_spl_db = 110.0;
        third.received_trace_sha256 = digest('8');
        input.samples.push(third);
        let report = audit(&input);
        assert_eq!(report.spatial_band_results[0].position_pair_count, 2);
        assert_eq!(report.spatial_band_results[0].spatial_consistency_disposition, "spatially_variable");
        assert_eq!(report.disposition, "SPATIALLY_VARIABLE");
        assert_eq!(report.spatial_band_results[0].spatial_consistency_disposition, "spatially_variable_beyond_uncertainty");
    }

    #[test]
    fn duplicate_repeats_are_invalid() {
        let mut input = case();
        input.samples[1].repeat_id = "r1".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn different_signal_quantity_or_bandwidth_is_not_pooled() {
        let mut input = case();
        input.samples[1].quantity = FrequencyQuantity::PowerSpectralDensity;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn conflicting_frequency_metadata_in_one_band_is_invalid() {
        let mut input = case();
        input.samples[1].center_frequency_hz = 1250.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn same_position_repeat_scatter_above_tolerance_is_spatially_variable() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.samples[1].received_spl_db = 124.0;
        assert_eq!(audit(&input).disposition, "SPATIALLY_VARIABLE");
    }

    #[test]
    fn malformed_trace_or_calibration_digest_is_invalid() {
        let mut input = case();
        input.samples[1].source_trace_sha256 = "bad".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }
}
