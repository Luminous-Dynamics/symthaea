use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const AGGREGATION_METHOD: &str = "equal_weight_arithmetic_mean_db_across_positions";

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind { Synthetic, Measured }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus { PublicOrPermitted, Unknown, Restricted }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum Scope { WholePropellerDisc, RestrictedSector }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum SourceLevelBasis {
    IndependentlyMeasured,
    ManufacturerFreeFieldTvr,
    NominalUnverified,
    Unknown,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum WaterState {
    OperatingCondition,
    StillWaterUnpressurized,
    Other,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    campaign_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    facility: Facility,
    condition: CalibrationCondition,
    projector: Projector,
    receiver: Receiver,
    scope: Scope,
    aggregation_method_id: String,
    aggregation_method_sha256: String,
    aggregation_tolerance_db: f64,
    maximum_spatial_spread_db: f64,
    maximum_repeatability_sd_db: f64,
    minimum_repeats_per_position: usize,
    positions: Vec<Position>,
    bands: Vec<String>,
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
    revision_id: String,
    test_section_geometry_sha256: String,
    coordinate_frame_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationCondition {
    condition_id: String,
    condition_fingerprint_sha256: String,
    water_state: WaterState,
    temperature_c: f64,
    absolute_pressure_kpa: f64,
    gas_content_mg_l: Option<f64>,
    cavitation_susceptibility_alpha: Option<f64>,
    flow_speed_m_s: f64,
    free_surface_present: bool,
    free_surface_distance_m: Option<f64>,
    condition_evidence_sha256: String,
    condition_transfer_additional_u95_db: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Projector {
    instrument_id: String,
    serial_or_revision: String,
    projector_type: String,
    signal_id: String,
    signal_type: String,
    drive_chain_id: String,
    drive_chain_calibration_sha256: String,
    source_level_basis: SourceLevelBasis,
    source_level_basis_reference_sha256: String,
    source_basis_additional_u95_db: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receiver {
    hydrophone_instrument_id: String,
    hydrophone_calibration_certificate_sha256: String,
    acquisition_chain_id: String,
    acquisition_chain_calibration_sha256: String,
    x_m: f64,
    y_m: f64,
    z_m: f64,
    orientation_yaw_deg: f64,
    orientation_pitch_deg: f64,
    orientation_roll_deg: f64,
    position_uncertainty_m: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Position {
    position_id: String,
    x_m: f64,
    y_m: f64,
    z_m: f64,
    radial_position_ratio: f64,
    azimuth_deg: f64,
    orientation_yaw_deg: f64,
    orientation_pitch_deg: f64,
    orientation_roll_deg: f64,
    position_uncertainty_m: f64,
    repeats: Vec<Repeat>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Repeat {
    repeat_id: String,
    timebase_receipt_sha256: String,
    raw_input_trace_sha256: String,
    raw_output_trace_sha256: String,
    bands: Vec<BandMeasurement>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BandMeasurement {
    band_id: String,
    center_frequency_hz: f64,
    input_quantity_id: String,
    output_quantity_id: String,
    bandwidth_id: String,
    spectrum_representation_id: String,
    input_source_level_db: f64,
    input_source_level_u95_db: f64,
    output_received_spl_db: f64,
    output_received_spl_u95_db: f64,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    campaign_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    source_rights_status: &'static str,
    facility_id: String,
    facility_revision_id: String,
    test_section_geometry_sha256: String,
    coordinate_frame_id: String,
    condition_id: String,
    condition_fingerprint_sha256: String,
    condition_evidence_sha256: String,
    calibration_temperature_c: f64,
    calibration_absolute_pressure_kpa: f64,
    calibration_gas_content_mg_l: Option<f64>,
    calibration_cavitation_susceptibility_alpha: Option<f64>,
    calibration_flow_speed_m_s: f64,
    free_surface_present: bool,
    free_surface_distance_m: Option<f64>,
    condition_transfer_additional_u95_db: f64,
    source_projector_id: String,
    source_projector_serial_or_revision: String,
    source_projector_type: String,
    source_signal_id: String,
    source_signal_type: String,
    drive_chain_id: String,
    drive_chain_calibration_sha256: String,
    source_level_basis_reference_sha256: String,
    source_basis_additional_u95_db: f64,
    receiver_hydrophone_instrument_id: String,
    receiver_hydrophone_calibration_certificate_sha256: String,
    receiver_acquisition_chain_id: String,
    receiver_acquisition_chain_calibration_sha256: String,
    receiver_x_m: f64,
    receiver_y_m: f64,
    receiver_z_m: f64,
    receiver_orientation_yaw_deg: f64,
    receiver_orientation_pitch_deg: f64,
    receiver_orientation_roll_deg: f64,
    receiver_position_uncertainty_m: f64,
    source_level_basis: &'static str,
    calibration_water_state: &'static str,
    scope: &'static str,
    aggregation_method_id: String,
    position_count: usize,
    minimum_position_count_met: bool,
    distinct_azimuth_sectors: usize,
    outer_disc_position_count: usize,
    minimum_repeats_per_position: usize,
    aggregation_tolerance_db: f64,
    maximum_spatial_spread_db: f64,
    maximum_repeatability_sd_db: f64,
    disposition: &'static str,
    band_summaries: Vec<BandSummary>,
    position_summaries: Vec<PositionSummary>,
    reasons: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Debug, Serialize)]
struct BandSummary {
    band_id: String,
    center_frequency_hz: f64,
    input_quantity_id: String,
    output_quantity_id: String,
    bandwidth_id: String,
    spectrum_representation_id: String,
    position_count: usize,
    repeat_count: usize,
    aggregate_transfer_function_db: f64,
    minimum_position_mean_db: f64,
    maximum_position_mean_db: f64,
    spatial_range_db: f64,
    spatial_standard_deviation_db: Option<f64>,
    mean_within_position_repeat_standard_deviation_db: Option<f64>,
    maximum_pairwise_gradient_db_per_m: Option<f64>,
    conservative_expanded_uncertainty_db: f64,
    maximum_spatial_spread_tolerance_db: f64,
    maximum_repeatability_sd_tolerance_db: f64,
    spatially_representative_within_declared_tolerance: bool,
}

#[derive(Debug, Serialize)]
struct RepeatEvidence {
    repeat_id: String,
    timebase_receipt_sha256: String,
    raw_input_trace_sha256: String,
    raw_output_trace_sha256: String,
}

#[derive(Debug, Serialize)]
struct PositionSummary {
    position_id: String,
    x_m: f64,
    y_m: f64,
    z_m: f64,
    radial_position_ratio: f64,
    azimuth_deg: f64,
    repeat_count: usize,
    repeat_evidence: Vec<RepeatEvidence>,
    band_summaries: Vec<PositionBandSummary>,
}

#[derive(Debug, Serialize)]
struct PositionBandSummary {
    band_id: String,
    mean_transfer_function_db: f64,
    repeat_standard_deviation_db: Option<f64>,
    conservative_expanded_uncertainty_db: f64,
}

#[derive(Debug)]
struct PositionBandCalculation {
    mean_tf_db: f64,
    repeat_sd_db: Option<f64>,
    conservative_u95_db: f64,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 { return None; }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    Some((sum / (values.len() - 1) as f64).sqrt())
}

fn basis_name(value: SourceLevelBasis) -> &'static str {
    match value {
        SourceLevelBasis::IndependentlyMeasured => "independently_measured",
        SourceLevelBasis::ManufacturerFreeFieldTvr => "manufacturer_free_field_tvr",
        SourceLevelBasis::NominalUnverified => "nominal_unverified",
        SourceLevelBasis::Unknown => "unknown",
    }
}

fn water_state_name(value: WaterState) -> &'static str {
    match value {
        WaterState::OperatingCondition => "operating_condition",
        WaterState::StillWaterUnpressurized => "still_water_unpressurized",
        WaterState::Other => "other",
        WaterState::Unknown => "unknown",
    }
}

fn scope_name(value: Scope) -> &'static str {
    match value {
        Scope::WholePropellerDisc => "whole_propeller_disc",
        Scope::RestrictedSector => "restricted_sector",
    }
}

fn calculate_position_band(position: &Position, band_id: &str) -> Option<PositionBandCalculation> {
    let mut values = Vec::new();
    let mut uncertainties = Vec::new();
    for repeat in &position.repeats {
        let sample = repeat.bands.iter().find(|band| band.band_id == band_id)?;
        values.push(sample.output_received_spl_db - sample.input_source_level_db);
        uncertainties.push(
            sample.output_received_spl_u95_db
                + sample.input_source_level_u95_db
        );
    }
    if values.is_empty() { return None; }
    Some(PositionBandCalculation {
        mean_tf_db: values.iter().sum::<f64>() / values.len() as f64,
        repeat_sd_db: sample_sd(&values),
        // Averages the declared bounds without sqrt(N) reduction. This is a
        // conservative box bound and does not assume independent repeats.
        conservative_u95_db: uncertainties.iter().sum::<f64>() / uncertainties.len() as f64,
    })
}

fn maximum_pairwise_gradient(positions: &[&Position], means: &[f64]) -> Option<f64> {
    if positions.len() < 2 || positions.len() != means.len() { return None; }
    let mut maximum = 0.0_f64;
    for left in 0..positions.len() {
        for right in (left + 1)..positions.len() {
            let dx = positions[left].x_m - positions[right].x_m;
            let dy = positions[left].y_m - positions[right].y_m;
            let dz = positions[left].z_m - positions[right].z_m;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance > 0.0 {
                maximum = maximum.max((means[left] - means[right]).abs() / distance);
            }
        }
    }
    Some(maximum)
}

fn validate(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION { errors.push("unsupported schema_version".to_string()); }
    if case.campaign_id.trim().is_empty()
        || case.source.dataset_id.trim().is_empty()
        || case.source.citation.trim().is_empty()
        || case.facility.facility_id.trim().is_empty()
        || case.facility.revision_id.trim().is_empty()
        || case.facility.coordinate_frame_id.trim().is_empty()
        || case.condition.condition_id.trim().is_empty()
        || case.projector.instrument_id.trim().is_empty()
        || case.projector.serial_or_revision.trim().is_empty()
        || case.projector.projector_type.trim().is_empty()
        || case.projector.signal_id.trim().is_empty()
        || case.projector.signal_type.trim().is_empty()
        || case.projector.drive_chain_id.trim().is_empty()
        || case.receiver.hydrophone_instrument_id.trim().is_empty()
        || case.receiver.acquisition_chain_id.trim().is_empty()
    {
        errors.push("campaign/facility/condition/projector/receiver identity fields are required".to_string());
    }
    for (label, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("test-section geometry", &case.facility.test_section_geometry_sha256),
        ("condition fingerprint", &case.condition.condition_fingerprint_sha256),
        ("condition evidence", &case.condition.condition_evidence_sha256),
        ("drive-chain calibration", &case.projector.drive_chain_calibration_sha256),
        ("source-level basis evidence", &case.projector.source_level_basis_reference_sha256),
        ("hydrophone calibration", &case.receiver.hydrophone_calibration_certificate_sha256),
        ("acquisition-chain calibration", &case.receiver.acquisition_chain_calibration_sha256),
        ("aggregation method", &case.aggregation_method_sha256),
    ] {
        if !is_sha256(digest) { errors.push(format!("{label} digest must be SHA-256")); }
    }
    if case.aggregation_method_id != AGGREGATION_METHOD {
        errors.push("unsupported transfer-function aggregation method".to_string());
    }
    if !finite_positive(case.aggregation_tolerance_db)
        || !finite_positive(case.maximum_spatial_spread_db)
        || !finite_positive(case.maximum_repeatability_sd_db)
        || case.minimum_repeats_per_position < 2
    {
        errors.push("aggregation/spatial/repeatability tolerances or minimum repeat count are invalid".to_string());
    }
    if !case.condition.temperature_c.is_finite()
        || !finite_positive(case.condition.absolute_pressure_kpa)
        || !case.condition.flow_speed_m_s.is_finite()
        || case.condition.flow_speed_m_s < 0.0
        || !case.condition.condition_transfer_additional_u95_db.is_finite()
        || case.condition.condition_transfer_additional_u95_db < 0.0
    {
        errors.push("calibration water-condition measurements are invalid".to_string());
    }
    if case.condition.gas_content_mg_l.is_some_and(|value| !value.is_finite() || value < 0.0)
        || case.condition.cavitation_susceptibility_alpha.is_some_and(|value| !value.is_finite() || value < 0.0)
        || case.condition.free_surface_distance_m.is_some_and(|value| !finite_positive(value))
    {
        errors.push("optional gas, susceptibility or free-surface measurement is invalid".to_string());
    }
    if case.condition.gas_content_mg_l.is_none()
        && case.condition.cavitation_susceptibility_alpha.is_none()
    {
        errors.push("calibration condition requires gas-content or cavitation-susceptibility evidence".to_string());
    }
    if case.condition.gas_content_mg_l.is_none()
        && case.condition.cavitation_susceptibility_alpha.is_none()
    {
        errors.push("calibration requires gas-content or cavitation-susceptibility evidence".to_string());
    }
    if case.condition.free_surface_present != case.condition.free_surface_distance_m.is_some() {
        errors.push("free-surface presence and distance evidence must agree".to_string());
    }
    if !case.projector.source_basis_additional_u95_db.is_finite()
        || case.projector.source_basis_additional_u95_db < 0.0
    {
        errors.push("source-basis additional uncertainty is invalid".to_string());
    }
    if matches!(case.projector.source_level_basis, SourceLevelBasis::ManufacturerFreeFieldTvr)
        && case.projector.source_basis_additional_u95_db <= 0.0
    {
        errors.push("manufacturer free-field TVR in a closed facility requires additional source-basis uncertainty".to_string());
    }
    if matches!(case.condition.water_state, WaterState::StillWaterUnpressurized | WaterState::Other)
        && case.condition.condition_transfer_additional_u95_db <= 0.0
    {
        errors.push("still-water/other calibration conditions require additional transferability uncertainty".to_string());
    }
    for (label, value) in [
        ("receiver position uncertainty", case.receiver.position_uncertainty_m),
        ("projector source basis extra uncertainty", case.projector.source_basis_additional_u95_db.max(1.0e-12)),
    ] {
        if !finite_positive(value) { errors.push(format!("{label} must be positive")); }
    }
    if !case.receiver.x_m.is_finite() || !case.receiver.y_m.is_finite() || !case.receiver.z_m.is_finite()
        || !case.receiver.orientation_yaw_deg.is_finite()
        || !case.receiver.orientation_pitch_deg.is_finite()
        || !case.receiver.orientation_roll_deg.is_finite()
    {
        errors.push("receiver position/orientation is invalid".to_string());
    }
    if case.bands.is_empty() || case.bands.iter().any(|band| band.trim().is_empty()) {
        errors.push("at least one non-empty frequency-band ID is required".to_string());
    }
    let band_set: BTreeSet<String> = case.bands.iter().cloned().collect();
    if band_set.len() != case.bands.len() { errors.push("band IDs must be unique".to_string()); }
    let minimum_positions = match case.scope {
        Scope::WholePropellerDisc => 5,
        Scope::RestrictedSector => 3,
    };
    if case.positions.len() < minimum_positions {
        errors.push(format!("scope requires at least {minimum_positions} measured source positions"));
    }
    if case.scope == Scope::WholePropellerDisc {
        let sectors = case.positions.iter()
            .map(|position| (position.azimuth_deg.rem_euclid(360.0) / 90.0).floor() as u8)
            .collect::<BTreeSet<_>>()
            .len();
        let outer_count = case.positions.iter()
            .filter(|position| position.radial_position_ratio >= 0.7)
            .count();
        if sectors < 3 || outer_count < 3 {
            errors.push("whole-disc scope requires at least three azimuth sectors and three outer-disc positions".to_string());
        }
    }
    let mut position_ids = BTreeSet::new();
    for position in &case.positions {
        if position.position_id.trim().is_empty() || !position_ids.insert(position.position_id.clone()) {
            errors.push("source position IDs must be unique and non-empty".to_string());
        }
        if !position.x_m.is_finite() || !position.y_m.is_finite() || !position.z_m.is_finite()
            || !finite_positive(position.position_uncertainty_m)
            || !position.radial_position_ratio.is_finite()
            || position.radial_position_ratio < 0.0
            || position.radial_position_ratio > 1.05
            || !position.azimuth_deg.is_finite()
            || !position.orientation_yaw_deg.is_finite()
            || !position.orientation_pitch_deg.is_finite()
            || !position.orientation_roll_deg.is_finite()
        {
            errors.push(format!("position {} geometry/uncertainty is invalid", position.position_id));
        }
        if position.repeats.len() < case.minimum_repeats_per_position {
            errors.push(format!("position {} has fewer repeats than declared minimum", position.position_id));
        }
        let mut repeat_ids = BTreeSet::new();
        for repeat in &position.repeats {
            if repeat.repeat_id.trim().is_empty() || !repeat_ids.insert(repeat.repeat_id.clone()) {
                errors.push(format!("position {} repeat IDs must be unique and non-empty", position.position_id));
            }
            for (label, digest) in [
                ("repeat timebase", &repeat.timebase_receipt_sha256),
                ("raw source trace", &repeat.raw_input_trace_sha256),
                ("raw receiver trace", &repeat.raw_output_trace_sha256),
            ] {
                if !is_sha256(digest) {
                    errors.push(format!("position {} repeat {}: {label} digest is invalid", position.position_id, repeat.repeat_id));
                }
            }
            let mut repeat_bands = BTreeSet::new();
            for measurement in &repeat.bands {
                if !band_set.contains(&measurement.band_id) || !repeat_bands.insert(measurement.band_id.clone()) {
                    errors.push(format!("position {} repeat {} has unknown/duplicate band {}", position.position_id, repeat.repeat_id, measurement.band_id));
                }
                if !finite_positive(measurement.center_frequency_hz)
                    || measurement.input_quantity_id != "source_level_db_re_1upa2_m2"
                    || measurement.output_quantity_id != "received_spl_db_re_1upa2"
                    || measurement.bandwidth_id.trim().is_empty()
                    || measurement.spectrum_representation_id.trim().is_empty()
                    || !measurement.input_source_level_db.is_finite()
                    || !measurement.output_received_spl_db.is_finite()
                    || !finite_positive(measurement.input_source_level_u95_db)
                    || !finite_positive(measurement.output_received_spl_u95_db)
                {
                    errors.push(format!("position {} repeat {} band {} has invalid input/output level or uncertainty", position.position_id, repeat.repeat_id, measurement.band_id));
                }
            }
            if repeat_bands != band_set {
                errors.push(format!("position {} repeat {} does not cover the declared band set exactly", position.position_id, repeat.repeat_id));
            }
        }
    }
    for band in &case.bands {
        let mut frequencies = Vec::new();
        for position in &case.positions {
            for repeat in &position.repeats {
                if let Some(sample) = repeat.bands.iter().find(|sample| &sample.band_id == band) {
                    frequencies.push(sample.center_frequency_hz);
                }
            }
        }
        if let Some(first) = frequencies.first() {
            if frequencies.iter().any(|frequency| {
                (frequency - first).abs() / first > 1.0e-6
            }) {
                errors.push(format!("band {band} frequency changes across positions/repeats"));
            }
        }
        let mut semantic_signatures = BTreeSet::new();
        for position in &case.positions {
            for repeat in &position.repeats {
                if let Some(sample) = repeat.bands.iter().find(|sample| &sample.band_id == band) {
                    semantic_signatures.insert((
                        sample.input_quantity_id.clone(),
                        sample.output_quantity_id.clone(),
                        sample.bandwidth_id.clone(),
                        sample.spectrum_representation_id.clone(),
                    ));
                }
            }
        }
        if semantic_signatures.len() != 1 {
            errors.push(format!("band {band} mixes quantity, bandwidth or spectral-representation semantics"));
        }
    }
    errors
}

fn audit(case: &Case) -> AuditReport {
    let mut reasons = validate(case);
    let incomplete_validation = reasons.iter().any(|reason| {
        reason.contains("requires at least")
            || reason.contains("fewer repeats")
            || reason.contains("requires additional")
            || reason.contains("requires gas-content")
            || reason.contains("whole-disc scope requires")
    });
    let invalid_input = !reasons.is_empty() && !incomplete_validation;
    let mut position_summaries = Vec::new();
    let mut band_summaries = Vec::new();

    if reasons.is_empty() {
        for position in &case.positions {
            let mut position_bands = Vec::new();
            for band in &case.bands {
                let calc = calculate_position_band(position, band).expect("validated band exists in all repeats");
                position_bands.push(PositionBandSummary {
                    band_id: band.clone(),
                    mean_transfer_function_db: calc.mean_tf_db,
                    repeat_standard_deviation_db: calc.repeat_sd_db,
                    conservative_expanded_uncertainty_db: calc.conservative_u95_db
                        + case.projector.source_basis_additional_u95_db
                        + case.condition.condition_transfer_additional_u95_db,
                });
            }
            position_bands.sort_by(|left, right| left.band_id.cmp(&right.band_id));
            position_summaries.push(PositionSummary {
                position_id: position.position_id.clone(),
                x_m: position.x_m,
                y_m: position.y_m,
                z_m: position.z_m,
                radial_position_ratio: position.radial_position_ratio,
                azimuth_deg: position.azimuth_deg,
                repeat_count: position.repeats.len(),
                repeat_evidence: position.repeats.iter().map(|repeat| RepeatEvidence {
                    repeat_id: repeat.repeat_id.clone(),
                    timebase_receipt_sha256: repeat.timebase_receipt_sha256.clone(),
                    raw_input_trace_sha256: repeat.raw_input_trace_sha256.clone(),
                    raw_output_trace_sha256: repeat.raw_output_trace_sha256.clone(),
                }).collect(),
                band_summaries: position_bands,
            });
        }
        position_summaries.sort_by(|left, right| left.position_id.cmp(&right.position_id));

        for band in &case.bands {
            let calculations: Vec<_> = case.positions.iter()
                .map(|position| calculate_position_band(position, band).expect("validated position-band sample"))
                .collect();
            let position_means: Vec<f64> = calculations.iter().map(|calc| calc.mean_tf_db).collect();
            let minimum = position_means.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = position_means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let aggregate = position_means.iter().sum::<f64>() / position_means.len() as f64;
            let spatial_range = maximum - minimum;
            let mean_repeat_sd = {
                let values: Vec<f64> = calculations.iter().filter_map(|calc| calc.repeat_sd_db).collect();
                if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
            };
            let conservative_u95 = calculations.iter().map(|calc| calc.conservative_u95_db).sum::<f64>()
                / calculations.len() as f64
                + case.projector.source_basis_additional_u95_db
                + case.condition.condition_transfer_additional_u95_db;
            let positions: Vec<&Position> = case.positions.iter().collect();
            let gradient = maximum_pairwise_gradient(&positions, &position_means);
            let repeatable = mean_repeat_sd.is_some_and(|value| value <= case.maximum_repeatability_sd_db);
            let spatially_representative = spatial_range <= case.maximum_spatial_spread_db && repeatable;
            let repeat_count = case.positions.iter().map(|position| position.repeats.len()).sum();
            let first_measurement = case.positions[0].repeats[0].bands.iter()
                .find(|sample| &sample.band_id == band)
                .expect("band coverage validated");
            band_summaries.push(BandSummary {
                band_id: band.clone(),
                center_frequency_hz: first_measurement.center_frequency_hz,
                input_quantity_id: first_measurement.input_quantity_id.clone(),
                output_quantity_id: first_measurement.output_quantity_id.clone(),
                bandwidth_id: first_measurement.bandwidth_id.clone(),
                spectrum_representation_id: first_measurement.spectrum_representation_id.clone(),
                position_count: case.positions.len(),
                repeat_count,
                aggregate_transfer_function_db: aggregate,
                minimum_position_mean_db: minimum,
                maximum_position_mean_db: maximum,
                spatial_range_db: spatial_range,
                spatial_standard_deviation_db: sample_sd(&position_means),
                mean_within_position_repeat_standard_deviation_db: mean_repeat_sd,
                maximum_pairwise_gradient_db_per_m: gradient,
                conservative_expanded_uncertainty_db: conservative_u95,
                maximum_spatial_spread_tolerance_db: case.maximum_spatial_spread_db,
                maximum_repeatability_sd_tolerance_db: case.maximum_repeatability_sd_db,
                spatially_representative_within_declared_tolerance: spatially_representative,
            });
            if spatial_range > case.maximum_spatial_spread_db {
                reasons.push(format!("band {band}: spatial transfer range exceeds declared tolerance"));
            }
            if !repeatable {
                reasons.push(format!("band {band}: repeated same-position transfer measurements exceed declared repeatability tolerance or are insufficient"));
            }
        }
    }

    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot qualify a measured facility transfer function".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source dataset rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source evidence has not been independently reviewed".to_string());
    }

    let provisional_basis = matches!(
        case.projector.source_level_basis,
        SourceLevelBasis::ManufacturerFreeFieldTvr | SourceLevelBasis::NominalUnverified | SourceLevelBasis::Unknown
    );
    if provisional_basis {
        reasons.push("source-level basis is manufacturer free-field TVR, nominal or unknown; this profile is provisional".to_string());
    }
    let condition_mismatch = matches!(
        case.condition.water_state,
        WaterState::StillWaterUnpressurized | WaterState::Other | WaterState::Unknown
    );
    if condition_mismatch {
        reasons.push("calibration water/pressure/flow condition differs from or is not known to match the cavitation-test condition".to_string());
    }

    let spatial_fail = band_summaries.iter().any(|band| !band.spatially_representative_within_declared_tolerance);
    let minimum_positions = match case.scope {
        Scope::WholePropellerDisc => 5,
        Scope::RestrictedSector => 3,
    };
    let distinct_azimuth_sectors = case.positions.iter()
        .map(|position| (position.azimuth_deg.rem_euclid(360.0) / 90.0).floor() as u8)
        .collect::<BTreeSet<_>>()
        .len();
    let outer_disc_position_count = case.positions.iter()
        .filter(|position| position.radial_position_ratio >= 0.7)
        .count();
    let whole_disc_geometry_met = case.scope != Scope::WholePropellerDisc
        || (distinct_azimuth_sectors >= 3 && outer_disc_position_count >= 3);
    let minimum_met = case.positions.len() >= minimum_positions && whole_disc_geometry_met;

    let disposition = if invalid_input {
        "INVALID"
    } else if incomplete_validation
        || case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || provisional_basis
        || condition_mismatch
        || !minimum_met
    {
        "INCOMPLETE"
    } else if spatial_fail {
        "SPATIALLY_VARIABLE"
    } else {
        "REPEATABLE_POSITION_TF"
    };

    reasons.sort();
    reasons.dedup();
    AuditReport {
        schema_version: SCHEMA_VERSION,
        campaign_id: case.campaign_id.clone(),
        source_dataset_id: case.source.dataset_id.clone(),
        source_data_sha256: case.source.raw_data_sha256.clone(),
        source_rights_status: match case.source.rights_status {
            RightsStatus::PublicOrPermitted => "public_or_permitted",
            RightsStatus::Unknown => "unknown",
            RightsStatus::Restricted => "restricted",
        },
        facility_id: case.facility.facility_id.clone(),
        facility_revision_id: case.facility.revision_id.clone(),
        test_section_geometry_sha256: case.facility.test_section_geometry_sha256.clone(),
        coordinate_frame_id: case.facility.coordinate_frame_id.clone(),
        condition_id: case.condition.condition_id.clone(),
        condition_fingerprint_sha256: case.condition.condition_fingerprint_sha256.clone(),
        condition_evidence_sha256: case.condition.condition_evidence_sha256.clone(),
        calibration_temperature_c: case.condition.temperature_c,
        calibration_absolute_pressure_kpa: case.condition.absolute_pressure_kpa,
        calibration_gas_content_mg_l: case.condition.gas_content_mg_l,
        calibration_cavitation_susceptibility_alpha: case.condition.cavitation_susceptibility_alpha,
        calibration_flow_speed_m_s: case.condition.flow_speed_m_s,
        free_surface_present: case.condition.free_surface_present,
        free_surface_distance_m: case.condition.free_surface_distance_m,
        condition_transfer_additional_u95_db: case.condition.condition_transfer_additional_u95_db,
        source_projector_id: case.projector.instrument_id.clone(),
        source_projector_serial_or_revision: case.projector.serial_or_revision.clone(),
        source_projector_type: case.projector.projector_type.clone(),
        source_signal_id: case.projector.signal_id.clone(),
        source_signal_type: case.projector.signal_type.clone(),
        drive_chain_id: case.projector.drive_chain_id.clone(),
        drive_chain_calibration_sha256: case.projector.drive_chain_calibration_sha256.clone(),
        source_level_basis_reference_sha256: case.projector.source_level_basis_reference_sha256.clone(),
        source_basis_additional_u95_db: case.projector.source_basis_additional_u95_db,
        receiver_hydrophone_instrument_id: case.receiver.hydrophone_instrument_id.clone(),
        receiver_hydrophone_calibration_certificate_sha256: case.receiver.hydrophone_calibration_certificate_sha256.clone(),
        receiver_acquisition_chain_id: case.receiver.acquisition_chain_id.clone(),
        receiver_acquisition_chain_calibration_sha256: case.receiver.acquisition_chain_calibration_sha256.clone(),
        receiver_x_m: case.receiver.x_m,
        receiver_y_m: case.receiver.y_m,
        receiver_z_m: case.receiver.z_m,
        receiver_orientation_yaw_deg: case.receiver.orientation_yaw_deg,
        receiver_orientation_pitch_deg: case.receiver.orientation_pitch_deg,
        receiver_orientation_roll_deg: case.receiver.orientation_roll_deg,
        receiver_position_uncertainty_m: case.receiver.position_uncertainty_m,
        source_level_basis: basis_name(case.projector.source_level_basis),
        calibration_water_state: water_state_name(case.condition.water_state),
        scope: scope_name(case.scope),
        aggregation_method_id: case.aggregation_method_id.clone(),
        position_count: case.positions.len(),
        minimum_position_count_met: minimum_met,
        distinct_azimuth_sectors,
        outer_disc_position_count,
        minimum_repeats_per_position: case.minimum_repeats_per_position,
        aggregation_tolerance_db: case.aggregation_tolerance_db,
        maximum_spatial_spread_db: case.maximum_spatial_spread_db,
        maximum_repeatability_sd_db: case.maximum_repeatability_sd_db,
        disposition,
        band_summaries,
        position_summaries,
        reasons,
        claim_ceiling: "measured_facility_transfer_profile_for_declared_positions_and_conditions_only_no_free_field_or_full_scale_prediction_no_unmeasured_position_interpolation_no_propeller_noise_or_ecological_qualification",
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-facility-transfer-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-facility-transfer-audit <case.json>");
        process::exit(64);
    }
    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => { eprintln!("could not read {path}: {error}"); process::exit(64); }
    };
    let case: Case = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => { eprintln!("invalid case JSON: {error}"); process::exit(64); }
    };
    let report = audit(&case);
    println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition {
        "REPEATABLE_POSITION_TF" => {}
        "INVALID" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String { std::iter::repeat_n(c, 64).collect() }

    fn case() -> Case {
        let position_means = [19.8, 19.9, 20.0, 20.1, 20.2];
        let positions = position_means.iter().enumerate().map(|(index, transfer)| {
            let repeats = (0..2).map(|repeat| Repeat {
                repeat_id: format!("r{}-{}", index, repeat),
                timebase_receipt_sha256: digest('a'),
                raw_input_trace_sha256: format!("{:064x}", 1000 + index * 10 + repeat),
                raw_output_trace_sha256: format!("{:064x}", 2000 + index * 10 + repeat),
                bands: vec![BandMeasurement {
                    band_id: "band-1000hz".to_string(),
                    center_frequency_hz: 1000.0,
                    input_quantity_id: "source_level_db_re_1upa2_m2".to_string(),
                    output_quantity_id: "received_spl_db_re_1upa2".to_string(),
                    bandwidth_id: "one_third_octave".to_string(),
                    spectrum_representation_id: "power_spectrum".to_string(),
                    input_source_level_db: 100.0,
                    input_source_level_u95_db: 0.05,
                    output_received_spl_db: 100.0 + transfer,
                    output_received_spl_u95_db: 0.05,
                }],
            }).collect();
            Position {
                position_id: format!("position-{}", index + 1),
                x_m: (index as f64 - 2.0) * 0.05,
                y_m: 0.0,
                z_m: 0.175,
                radial_position_ratio: 0.8,
                azimuth_deg: index as f64 * 72.0,
                orientation_yaw_deg: 0.0,
                orientation_pitch_deg: 0.0,
                orientation_roll_deg: 0.0,
                position_uncertainty_m: 0.001,
                repeats,
            }
        }).collect();
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            campaign_id: "synthetic-calibration-campaign".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-facility-tf".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('b'),
                rights_status: RightsStatus::Unknown,
                independently_reviewed: false,
            },
            facility: Facility {
                facility_id: "synthetic-tunnel".to_string(),
                revision_id: "tunnel-rev-a".to_string(),
                test_section_geometry_sha256: digest('c'),
                coordinate_frame_id: "propeller-disc-frame".to_string(),
            },
            condition: CalibrationCondition {
                condition_id: "calibration-at-operating-condition".to_string(),
                condition_fingerprint_sha256: digest('d'),
                water_state: WaterState::OperatingCondition,
                temperature_c: 20.0,
                absolute_pressure_kpa: 130.0,
                gas_content_mg_l: Some(8.0),
                cavitation_susceptibility_alpha: Some(0.6),
                flow_speed_m_s: 5.0,
                free_surface_present: false,
                free_surface_distance_m: None,
                condition_evidence_sha256: digest('e'),
                condition_transfer_additional_u95_db: 0.0,
            },
            projector: Projector {
                instrument_id: "synthetic-projector".to_string(),
                serial_or_revision: "projector-rev-1".to_string(),
                projector_type: "broadband-omnidirectional".to_string(),
                signal_id: "sweep-001".to_string(),
                signal_type: "wideband-sweep".to_string(),
                drive_chain_id: "drive-chain-1".to_string(),
                drive_chain_calibration_sha256: digest('f'),
                source_level_basis: SourceLevelBasis::IndependentlyMeasured,
                source_level_basis_reference_sha256: digest('1'),
                source_basis_additional_u95_db: 0.0,
            },
            receiver: Receiver {
                hydrophone_instrument_id: "synthetic-hydrophone".to_string(),
                hydrophone_calibration_certificate_sha256: digest('2'),
                acquisition_chain_id: "acquisition-chain-1".to_string(),
                acquisition_chain_calibration_sha256: digest('3'),
                x_m: 0.0,
                y_m: -0.3,
                z_m: 0.0,
                orientation_yaw_deg: 0.0,
                orientation_pitch_deg: 0.0,
                orientation_roll_deg: 0.0,
                position_uncertainty_m: 0.001,
            },
            scope: Scope::WholePropellerDisc,
            aggregation_method_id: AGGREGATION_METHOD.to_string(),
            aggregation_method_sha256: digest('4'),
            aggregation_tolerance_db: 0.01,
            maximum_spatial_spread_db: 1.0,
            maximum_repeatability_sd_db: 0.2,
            minimum_repeats_per_position: 2,
            positions,
            bands: vec!["band-1000hz".to_string()],
        }
    }

    fn make_measured(case: &mut Case) {
        case.evidence_kind = EvidenceKind::Measured;
        case.source.rights_status = RightsStatus::PublicOrPermitted;
        case.source.independently_reviewed = true;
    }

    #[test]
    fn known_answer_transfer_is_output_spl_minus_source_level() {
        let report = audit(&case());
        let band = &report.band_summaries[0];
        assert!((band.aggregate_transfer_function_db - 20.0).abs() < 1e-12);
        assert!((band.spatial_range_db - 0.4).abs() < 1e-12);
        assert_eq!(band.position_count, 5);
        assert_eq!(band.repeat_count, 10);
    }

    #[test]
    fn five_to_ten_position_whole_disc_minimum_is_enforced() {
        let mut input = case();
        input.positions.truncate(4);
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn three_position_restricted_sector_is_accepted_for_synthetic_summary() {
        let mut input = case();
        input.scope = Scope::RestrictedSector;
        input.positions.truncate(3);
        let report = audit(&input);
        assert_eq!(report.position_count, 3);
        assert_eq!(report.minimum_position_count_met, true);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn repeated_same_position_scatter_is_reported() {
        let mut input = case();
        make_measured(&mut input);
        input.positions[0].repeats[1].bands[0].output_received_spl_db += 1.0;
        let report = audit(&input);
        assert!(report.position_summaries[0].band_summaries[0]
            .repeat_standard_deviation_db.unwrap() > 0.2);
        assert_eq!(report.disposition, "SPATIALLY_VARIABLE");
    }

    #[test]
    fn spatial_spread_beyond_tolerance_is_not_representative() {
        let mut input = case();
        make_measured(&mut input);
        input.positions[0].repeats.iter_mut().for_each(|repeat| {
            repeat.bands[0].output_received_spl_db -= 5.0;
        });
        let report = audit(&input);
        assert!(report.band_summaries[0].spatial_range_db > 1.0);
        assert_eq!(report.disposition, "SPATIALLY_VARIABLE");
    }

    #[test]
    fn free_field_manufacturer_tvr_is_provisional_even_with_extra_uncertainty() {
        let mut input = case();
        make_measured(&mut input);
        input.projector.source_level_basis = SourceLevelBasis::ManufacturerFreeFieldTvr;
        input.projector.source_basis_additional_u95_db = 1.0;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn manufacturer_tvr_without_additional_uncertainty_is_invalid() {
        let mut input = case();
        input.projector.source_level_basis = SourceLevelBasis::ManufacturerFreeFieldTvr;
        input.projector.source_basis_additional_u95_db = 0.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn still_water_calibration_requires_condition_uncertainty_and_is_provisional() {
        let mut input = case();
        make_measured(&mut input);
        input.condition.water_state = WaterState::StillWaterUnpressurized;
        input.condition.condition_transfer_additional_u95_db = 0.5;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn still_water_condition_without_additional_uncertainty_is_invalid() {
        let mut input = case();
        input.condition.water_state = WaterState::StillWaterUnpressurized;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn aggregate_tracks_the_equal_weight_position_profile() {
        let mut input = case();
        // Alter every reported position mean without altering the declared scalar.
        for position in &mut input.positions {
            for repeat in &mut position.repeats {
                repeat.bands[0].output_received_spl_db += 2.0;
            }
        }
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!((report.band_summaries[0].aggregate_transfer_function_db - 22.0).abs() < 1e-12);
    }

    #[test]
    fn repeat_requires_a_minimum_of_two_measurements() {
        let mut input = case();
        input.positions[0].repeats.truncate(1);
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn different_frequency_in_one_repeat_is_invalid() {
        let mut input = case();
        input.positions[0].repeats[0].bands[0].center_frequency_hz = 1200.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn missing_frequency_band_in_one_repeat_is_invalid() {
        let mut input = case();
        input.positions[0].repeats[0].bands.clear();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn facility_revision_identity_is_preserved() {
        let mut input = case();
        input.facility.revision_id = "tunnel-rev-b".to_string();
        let report = audit(&input);
        assert_eq!(report.facility_revision_id, "tunnel-rev-b");
    }

    #[test]
    fn malformed_raw_trace_or_calibration_digest_is_invalid() {
        let mut input = case();
        input.positions[0].repeats[0].raw_input_trace_sha256 = "bad".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn gradient_is_reported_without_interpolation() {
        let report = audit(&case());
        assert!(report.band_summaries[0].maximum_pairwise_gradient_db_per_m.is_some());
        assert!(report.claim_ceiling.contains("no_unmeasured_position_interpolation"));
    }

    #[test]
    fn source_and_receiver_measurement_lineage_is_preserved() {
        let report = audit(&case());
        assert_eq!(report.facility_revision_id, "tunnel-rev-a");
        assert_eq!(report.band_summaries[0].input_quantity_id, "source_level_db_re_1upa2_m2");
        assert_eq!(report.band_summaries[0].output_quantity_id, "received_spl_db_re_1upa2");
        assert_eq!(report.band_summaries[0].bandwidth_id, "one_third_octave");
        assert_eq!(report.source_projector_id, "synthetic-projector");
        assert_eq!(report.coordinate_frame_id, "propeller-disc-frame");
        assert_eq!(report.condition_fingerprint_sha256, digest('d'));
    }

    #[test]
    fn input_permutation_preserves_aggregate_transfer() {
        let mut input = case();
        let baseline = audit(&input).band_summaries[0].aggregate_transfer_function_db;
        input.positions.reverse();
        for position in &mut input.positions {
            position.repeats.reverse();
        }
        let permuted = audit(&input).band_summaries[0].aggregate_transfer_function_db;
        assert!((baseline - permuted).abs() < 1e-12);
    }

    #[test]
    fn synthetic_fixture_never_qualifies_as_measured_calibration() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("synthetic evidence")));
    }

    #[test]
    fn mismatched_power_spectral_density_and_band_power_are_not_mixed() {
        let mut input = case();
        input.positions[0].repeats[0].bands[0].spectrum_representation_id = "power_spectral_density".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn gas_or_cavitation_susceptibility_must_be_recorded() {
        let mut input = case();
        make_measured(&mut input);
        input.condition.gas_content_mg_l = None;
        input.condition.cavitation_susceptibility_alpha = None;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("gas-content")));
    }

    #[test]
    fn whole_disc_positions_must_cover_multiple_azimuth_sectors() {
        let mut input = case();
        for position in &mut input.positions {
            position.azimuth_deg = 10.0 + position.position_id[9..].parse::<f64>().unwrap();
        }
        let report = audit(&input);
        assert_eq!(report.distinct_azimuth_sectors, 1);
        assert_eq!(report.minimum_position_count_met, false);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn committed_schema_and_fixture_are_closed_and_synthetic() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-facility-transfer-v0.1.0.schema.json"
        )).expect("schema JSON valid");
        assert_eq!(schema["$schema"], "https://json-schema.org/draft/2020-12/schema");
        assert_eq!(schema["additionalProperties"], false);
        let raw = include_str!(
            "../../../docs/research/marine/fixtures/synthetic-marine-facility-transfer-v0.1.0.json"
        );
        let input: Case = serde_json::from_str(raw).expect("fixture matches Rust contract");
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.position_count, 5);
        assert_eq!(report.band_summaries[0].repeat_count, 10);
    }

    #[test]
    fn unknown_source_basis_is_never_accepted() {
        let mut input = case();
        make_measured(&mut input);
        input.projector.source_level_basis = SourceLevelBasis::Unknown;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("source-level basis")));
    }
}
