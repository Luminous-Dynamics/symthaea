use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process;

const SCHEMA_VERSION: &str = "0.1.0";
const ITTC_METHOD: &str = "ITTC-7.5-02-03-03.9-2024-Rev03";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    scaling: ScalingSelection,
    similarity: SimilarityEvidence,
    frequency_match_tolerance_fraction: f64,
    pairs: Vec<ScalePair>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Synthetic,
    Measured,
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

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus {
    PublicOrPermitted,
    Unknown,
    Restricted,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalingSelection {
    method_id: String,
    bandwidth: Bandwidth,
    representation: SpectrumRepresentation,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FrequencyRegime {
    HighFrequency,
    LowFrequency,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Bandwidth {
    Proportional,
    Constant,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SpectrumRepresentation {
    PowerSpectrum,
    PowerSpectralDensity,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimilarityEvidence {
    wake: SimilarityState,
    cavitation_pattern: SimilarityState,
    gas_nuclei_state: SimilarityState,
    reynolds_scale_review: ReviewState,
    facility_transfer_function: FacilityTransferState,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SimilarityState {
    Qualified,
    Dissimilar,
    Unknown,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ReviewState {
    Reviewed,
    NotReviewed,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FacilityTransferState {
    MeasuredAndApplied,
    Unknown,
    Rejected,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BackgroundMeasurementMethod {
    BareHub,
    PressureSuppressed,
    Unknown,
}

fn background_measurement_method_name(value: &BackgroundMeasurementMethod) -> &'static str {
    match value {
        BackgroundMeasurementMethod::BareHub => "bare_hub",
        BackgroundMeasurementMethod::PressureSuppressed => "pressure_suppressed",
        BackgroundMeasurementMethod::Unknown => "unknown",
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalePair {
    pair_id: String,
    model: ModelRun,
    full_scale: FullScaleRun,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRun {
    run_id: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    raw_trace_sha256: String,
    diameter_m: f64,
    rpm_hz: f64,
    rpm_u95_hz: f64,
    cavitation_number_n_d: f64,
    source_reference_distance_m: f64,
    background_measurement_method: BackgroundMeasurementMethod,
    background_trace_sha256: String,
    bands: Vec<ModelBand>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelBand {
    band_id: String,
    center_frequency_hz: f64,
    frequency_regime: FrequencyRegime,
    frequency_regime_basis: String,
    scale_transfer_u95_db: f64,
    total_spl_db: f64,
    background_spl_db: f64,
    facility_transfer_function_db: f64,
    total_u95_db: f64,
    background_u95_db: f64,
    facility_transfer_u95_db: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullScaleRun {
    run_id: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    raw_trace_sha256: String,
    diameter_m: f64,
    rpm_hz: f64,
    cavitation_number_n_d: f64,
    source_reference_distance_m: f64,
    source_level_method_id: String,
    bands: Vec<FullScaleBand>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullScaleBand {
    band_id: String,
    center_frequency_hz: f64,
    source_level_db: f64,
    source_level_u95_db: f64,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: &'static str,
    scaling_method_id: String,
    bandwidth_convention: &'static str,
    spectrum_representation: &'static str,
    disposition: &'static str,
    comparison_outcome: &'static str,
    unique_model_rotation_rates: usize,
    pair_reports: Vec<PairReport>,
    reasons: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Debug, Serialize)]
struct PairReport {
    pair_id: String,
    model_run_id: String,
    full_scale_run_id: String,
    background_measurement_method: &'static str,
    background_trace_sha256: String,
    model_timebase_receipt_sha256: String,
    full_scale_timebase_receipt_sha256: String,
    model_operating_point_receipt_sha256: String,
    model_raw_trace_sha256: String,
    full_scale_operating_point_receipt_sha256: String,
    full_scale_raw_trace_sha256: String,
    frequency_scale_factor: f64,
    band_reports: Vec<BandReport>,
}

#[derive(Debug, Serialize)]
struct BandReport {
    band_id: String,
    frequency_regime: &'static str,
    frequency_regime_basis: String,
    spl_scale_delta_db: f64,
    scale_transfer_u95_db: f64,
    model_source_level_db: Option<f64>,
    predicted_full_scale_source_level_db: Option<f64>,
    signed_error_db: Option<f64>,
    absolute_error_db: Option<f64>,
    predicted_interval_lower_db: Option<f64>,
    predicted_interval_upper_db: Option<f64>,
    mapped_frequency_hz: f64,
    observed_frequency_hz: Option<f64>,
    frequency_relative_error: Option<f64>,
    observed_full_scale_source_level_db: Option<f64>,
    observed_interval_lower_db: Option<f64>,
    observed_interval_upper_db: Option<f64>,
    comparison: &'static str,
}

fn evidence_kind_name(value: &EvidenceKind) -> &'static str {
    match value {
        EvidenceKind::Synthetic => "synthetic",
        EvidenceKind::Measured => "measured",
    }
}

fn bandwidth_name(value: &Bandwidth) -> &'static str {
    match value {
        Bandwidth::Proportional => "proportional",
        Bandwidth::Constant => "constant",
    }
}

fn frequency_regime_name(value: &FrequencyRegime) -> &'static str {
    match value {
        FrequencyRegime::HighFrequency => "high_frequency",
        FrequencyRegime::LowFrequency => "low_frequency",
    }
}

fn spectrum_representation_name(value: &SpectrumRepresentation) -> &'static str {
    match value {
        SpectrumRepresentation::PowerSpectrum => "power_spectrum",
        SpectrumRepresentation::PowerSpectralDensity => "power_spectral_density",
    }
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

// Stable linear-energy subtraction. total_db includes the background;
// the returned level is the background-corrected cavitation component.
fn background_corrected_spl(total_db: f64, background_db: f64) -> Option<f64> {
    if !total_db.is_finite() || !background_db.is_finite() || total_db <= background_db {
        return None;
    }
    let relative_background = 10.0_f64.powf((background_db - total_db) / 10.0);
    if !(0.0..1.0).contains(&relative_background) {
        return None;
    }
    let corrected = total_db + 10.0 * (1.0 - relative_background).log10();
    corrected.is_finite().then_some(corrected)
}

// ITTC Rev.03 §2.5.3: below 3 dB the signal is background-dominated;
// from 3 dB through 10 dB, subtract background in the energy domain;
// above 10 dB, no correction is required.
fn ittc_cavitation_spl(total_db: f64, background_db: f64) -> Option<f64> {
    if !total_db.is_finite() || !background_db.is_finite() || total_db <= background_db {
        return None;
    }
    let margin_db = total_db - background_db;
    if margin_db < 3.0 {
        None
    } else if margin_db > 10.0 {
        Some(total_db)
    } else {
        background_corrected_spl(total_db, background_db)
    }
}

fn scaling_exponents(
    regime: &FrequencyRegime,
    bandwidth: &Bandwidth,
) -> (f64, f64, f64, f64) {
    match (regime, bandwidth) {
        (FrequencyRegime::HighFrequency, Bandwidth::Proportional) => (0.5, 1.0, 1.5, 1.0),
        (FrequencyRegime::LowFrequency, Bandwidth::Proportional) => (1.0, 1.0, 2.0, 1.0),
        (FrequencyRegime::HighFrequency, Bandwidth::Constant) => (0.25, 1.0, 1.0, 1.5),
        (FrequencyRegime::LowFrequency, Bandwidth::Constant) => (0.75, 1.0, 1.5, 1.5),
    }
}

// ITTC Rev.03 Eq. 16. The logarithmic form avoids intermediate overflow.
fn spl_scale_delta_db(
    model: &ModelRun,
    full: &FullScaleRun,
    regime: &FrequencyRegime,
    bandwidth: &Bandwidth,
) -> f64 {
    let (w, x, y, z) = scaling_exponents(regime, bandwidth);
    let sigma_ratio = full.cavitation_number_n_d / model.cavitation_number_n_d;
    let distance_ratio =
        model.source_reference_distance_m / full.source_reference_distance_m;
    let rotational_diameter_ratio =
        (full.rpm_hz * full.diameter_m) / (model.rpm_hz * model.diameter_m);
    let diameter_ratio = full.diameter_m / model.diameter_m;
    20.0 / std::f64::consts::LN_10
        * (w * sigma_ratio.ln()
            + x * distance_ratio.ln()
            + y * rotational_diameter_ratio.ln()
            + z * diameter_ratio.ln())
}

// ITTC Rev.03 Eq. 15.
fn mapped_frequency_hz(model: &ModelRun, full: &FullScaleRun, model_frequency_hz: f64) -> f64 {
    model_frequency_hz
        * (full.rpm_hz / model.rpm_hz)
        * (full.cavitation_number_n_d / model.cavitation_number_n_d).sqrt()
}

fn distinct_model_rotation_rate_count(pairs: &[ScalePair]) -> usize {
    let mut intervals: Vec<(f64, f64)> = pairs
        .iter()
        .map(|pair| {
            (
                pair.model.rpm_hz - pair.model.rpm_u95_hz,
                pair.model.rpm_hz + pair.model.rpm_u95_hz,
            )
        })
        .collect();
    intervals.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut count = 0;
    let mut covered_upper = f64::NEG_INFINITY;
    for (lower, upper) in intervals {
        if lower > covered_upper {
            count += 1;
            covered_upper = upper;
        } else {
            covered_upper = covered_upper.max(upper);
        }
    }
    count
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "unsupported schema_version {}; expected {}",
            case.schema_version, SCHEMA_VERSION
        ));
    }
    if case.case_id.trim().is_empty() {
        errors.push("case_id is empty".to_string());
    }
    if case.source.dataset_id.trim().is_empty() || case.source.citation.trim().is_empty() {
        errors.push("source dataset_id and citation are required".to_string());
    }
    if !is_sha256(&case.source.raw_data_sha256) {
        errors.push("source raw_data_sha256 must be 64 hexadecimal characters".to_string());
    }
    if case.scaling.method_id != ITTC_METHOD {
        errors.push(format!("method_id must be {}", ITTC_METHOD));
    }
    if case.scaling.bandwidth == Bandwidth::Proportional
        && case.scaling.representation != SpectrumRepresentation::PowerSpectrum
    {
        errors.push(
            "ITTC proportional-bandwidth table requires power_spectrum representation"
                .to_string(),
        );
    }
    if !case.frequency_match_tolerance_fraction.is_finite()
        || case.frequency_match_tolerance_fraction <= 0.0
        || case.frequency_match_tolerance_fraction >= 1.0
    {
        errors.push("frequency_match_tolerance_fraction must be strictly between 0 and 1".to_string());
    }
    if case.pairs.is_empty() {
        errors.push("at least one matched model/full-scale pair is required".to_string());
    }

    let mut pair_ids = BTreeSet::new();
    for pair in &case.pairs {
        if pair.pair_id.trim().is_empty() || !pair_ids.insert(pair.pair_id.clone()) {
            errors.push(format!("empty or duplicate pair_id: {}", pair.pair_id));
        }
        for (label, value) in [
            ("model diameter", pair.model.diameter_m),
            ("model rpm_hz", pair.model.rpm_hz),
            ("model rpm_u95_hz", pair.model.rpm_u95_hz),
            ("model sigma_nD", pair.model.cavitation_number_n_d),
            ("model source reference distance", pair.model.source_reference_distance_m),
            ("full-scale diameter", pair.full_scale.diameter_m),
            ("full-scale rpm_hz", pair.full_scale.rpm_hz),
            ("full-scale sigma_nD", pair.full_scale.cavitation_number_n_d),
            ("full-scale source reference distance", pair.full_scale.source_reference_distance_m),
        ] {
            if !finite_positive(value) {
                errors.push(format!("{}: {} must be finite and positive", pair.pair_id, label));
            }
        }
        if pair.model.rpm_u95_hz >= pair.model.rpm_hz {
            errors.push(format!(
                "{}: model rpm_u95_hz must be smaller than rpm_hz",
                pair.pair_id
            ));
        }
        let reference_distance_delta =
            (pair.model.source_reference_distance_m - pair.full_scale.source_reference_distance_m).abs();
        if reference_distance_delta > 1e-6
            || (pair.model.source_reference_distance_m - 1.0).abs() > 1e-6
            || (pair.full_scale.source_reference_distance_m - 1.0).abs() > 1e-6
        {
            errors.push(format!(
                "{}: source levels must share the 1 m reference distance",
                pair.pair_id
            ));
        }

        for (label, digest) in [
            ("model timebase receipt", &pair.model.timebase_receipt_sha256),
            ("full-scale timebase receipt", &pair.full_scale.timebase_receipt_sha256),
            ("model operating-point receipt", &pair.model.operating_point_receipt_sha256),
            ("model raw trace", &pair.model.raw_trace_sha256),
            ("model background trace", &pair.model.background_trace_sha256),
            ("full-scale operating-point receipt", &pair.full_scale.operating_point_receipt_sha256),
            ("full-scale raw trace", &pair.full_scale.raw_trace_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("{}: {} digest must be 64 hexadecimal characters", pair.pair_id, label));
            }
        }
        if pair.full_scale.source_level_method_id.trim().is_empty() {
            errors.push(format!("{}: full-scale source-level method is required", pair.pair_id));
        }
        if pair.model.bands.is_empty() {
            errors.push(format!("{}: model spectrum must contain at least one band", pair.pair_id));
        }

        let mut model_band_ids = BTreeSet::new();
        for band in &pair.model.bands {
            if band.band_id.trim().is_empty() || !model_band_ids.insert(band.band_id.clone()) {
                errors.push(format!("{}: empty or duplicate model band_id {}", pair.pair_id, band.band_id));
            }
            if !finite_positive(band.center_frequency_hz) {
                errors.push(format!("{} / {}: model band frequency must be positive", pair.pair_id, band.band_id));
            }
            if band.frequency_regime_basis.trim().is_empty() {
                errors.push(format!("{} / {}: frequency_regime_basis is required", pair.pair_id, band.band_id));
            }
            if !band.total_spl_db.is_finite()
                || !band.background_spl_db.is_finite()
                || band.total_spl_db <= band.background_spl_db
            {
                errors.push(format!(
                    "{} / {}: total SPL must be finite and greater than background SPL",
                    pair.pair_id, band.band_id
                ));
            }
            for (label, value) in [
                ("total_u95_db", band.total_u95_db),
                ("background_u95_db", band.background_u95_db),
                ("facility_transfer_u95_db", band.facility_transfer_u95_db),
                ("scale_transfer_u95_db", band.scale_transfer_u95_db),
            ] {
                if !finite_positive(value) {
                    errors.push(format!("{} / {}: {} must be finite and positive; unknown uncertainty cannot be encoded as zero", pair.pair_id, band.band_id, label));
                }
            }
            if !band.facility_transfer_function_db.is_finite() {
                errors.push(format!("{} / {}: facility transfer function must be finite", pair.pair_id, band.band_id));
            }
        }

        let mut full_band_ids = BTreeSet::new();
        for band in &pair.full_scale.bands {
            if band.band_id.trim().is_empty() || !full_band_ids.insert(band.band_id.clone()) {
                errors.push(format!("{}: empty or duplicate full-scale band_id {}", pair.pair_id, band.band_id));
            }
            if !finite_positive(band.center_frequency_hz)
                || !band.source_level_db.is_finite()
                || !finite_positive(band.source_level_u95_db)
            {
                errors.push(format!("{} / {}: invalid full-scale band measurement or uncertainty", pair.pair_id, band.band_id));
            }
        }
        if !pair.full_scale.bands.is_empty() && model_band_ids != full_band_ids {
            errors.push(format!("{}: model and full-scale band_id sets must match exactly", pair.pair_id));
        }
    }
    errors
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let unique_model_rotation_rates = distinct_model_rotation_rate_count(&case.pairs);
    let has_unknown_background_method = case.pairs.iter().any(|pair| {
        pair.model.background_measurement_method == BackgroundMeasurementMethod::Unknown
    });

    if !validation_errors.is_empty() {
        return AuditReport {
            schema_version: SCHEMA_VERSION,
            case_id: case.case_id.clone(),
            source_dataset_id: case.source.dataset_id.clone(),
            source_data_sha256: case.source.raw_data_sha256.clone(),
            evidence_kind: evidence_kind_name(&case.evidence_kind),
            scaling_method_id: case.scaling.method_id.clone(),
            bandwidth_convention: bandwidth_name(&case.scaling.bandwidth),
            spectrum_representation: spectrum_representation_name(&case.scaling.representation),
            disposition: "invalid",
            comparison_outcome: "not_evaluated",
            unique_model_rotation_rates,
            pair_reports: Vec::new(),
            reasons: validation_errors,
            claim_ceiling: "arithmetic_and_declared_comparison_only_no_solver_or_physical_qualification",
        };
    }

    let mut reasons = Vec::new();
    let mut pair_reports = Vec::new();
    let has_dissimilarity = [
        &case.similarity.wake,
        &case.similarity.cavitation_pattern,
        &case.similarity.gas_nuclei_state,
    ]
    .iter()
    .any(|state| **state == SimilarityState::Dissimilar);

    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot support measured validation".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source reuse rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source evidence has not been independently reviewed".to_string());
    }
    for pair in &case.pairs {
        if pair.model.background_measurement_method == BackgroundMeasurementMethod::Unknown {
            reasons.push(format!("{}: background measurement method is unknown", pair.pair_id));
        }
    }
    for (name, state) in [
        ("wake similarity", &case.similarity.wake),
        ("cavitation-pattern similarity", &case.similarity.cavitation_pattern),
        ("gas/nuclei similarity", &case.similarity.gas_nuclei_state),
    ] {
        if *state == SimilarityState::Unknown {
            reasons.push(format!("{} is unknown", name));
        }
    }
    if case.similarity.reynolds_scale_review != ReviewState::Reviewed {
        reasons.push("Reynolds-scale effects have not been reviewed".to_string());
    }
    match &case.similarity.facility_transfer_function {
        FacilityTransferState::MeasuredAndApplied => {}
        FacilityTransferState::Unknown => reasons.push("facility transfer function is unknown".to_string()),
        FacilityTransferState::Rejected => reasons.push("facility transfer function was rejected".to_string()),
    }
    if unique_model_rotation_rates < 2 {
        reasons.push("at least two distinct model-scale propeller rotation rates are required".to_string());
    }

    let mut any_prediction_only = false;
    let mut any_unresolved_band = false;
    let mut any_disjoint_band = false;
    let mut any_frequency_mismatch = false;

    for pair in &case.pairs {
        let frequency_factor = (pair.full_scale.rpm_hz / pair.model.rpm_hz)
            * (pair.full_scale.cavitation_number_n_d / pair.model.cavitation_number_n_d).sqrt();
        let mut band_reports = Vec::new();

        for model_band in &pair.model.bands {
            let delta_db = spl_scale_delta_db(
                &pair.model,
                &pair.full_scale,
                &model_band.frequency_regime,
                &case.scaling.bandwidth,
            );
            let Some(corrected) = ittc_cavitation_spl(
                model_band.total_spl_db,
                model_band.background_spl_db,
            ) else {
                any_unresolved_band = true;
                band_reports.push(BandReport {
                    band_id: model_band.band_id.clone(),
                    frequency_regime: frequency_regime_name(&model_band.frequency_regime),
                    frequency_regime_basis: model_band.frequency_regime_basis.clone(),
                    spl_scale_delta_db: delta_db,
                    scale_transfer_u95_db: model_band.scale_transfer_u95_db,
                    model_source_level_db: None,
                    predicted_full_scale_source_level_db: None,
                    signed_error_db: None,
                    absolute_error_db: None,
                    predicted_interval_lower_db: None,
                    predicted_interval_upper_db: None,
                    mapped_frequency_hz: mapped_frequency_hz(
                        &pair.model,
                        &pair.full_scale,
                        model_band.center_frequency_hz,
                    ),
                    observed_frequency_hz: None,
                    frequency_relative_error: None,
                    observed_full_scale_source_level_db: None,
                    observed_interval_lower_db: None,
                    observed_interval_upper_db: None,
                    comparison: "background_dominated_unusable",
                });
                continue;
            };
            let model_source_level = corrected - model_band.facility_transfer_function_db;
            let predicted_full_scale = model_source_level + delta_db;
            let mapped_frequency = mapped_frequency_hz(
                &pair.model,
                &pair.full_scale,
                model_band.center_frequency_hz,
            );

            let Some(full_band) = pair.full_scale.bands.iter().find(|b| b.band_id == model_band.band_id) else {
                any_prediction_only = true;
                band_reports.push(BandReport {
                    band_id: model_band.band_id.clone(),
                    frequency_regime: frequency_regime_name(&model_band.frequency_regime),
                    frequency_regime_basis: model_band.frequency_regime_basis.clone(),
                    spl_scale_delta_db: delta_db,
                    scale_transfer_u95_db: model_band.scale_transfer_u95_db,
                    model_source_level_db: Some(model_source_level),
                    predicted_full_scale_source_level_db: Some(predicted_full_scale),
                    signed_error_db: None,
                    absolute_error_db: None,
                    predicted_interval_lower_db: None,
                    predicted_interval_upper_db: None,
                    mapped_frequency_hz: mapped_frequency,
                    observed_frequency_hz: None,
                    frequency_relative_error: None,
                    observed_full_scale_source_level_db: None,
                    observed_interval_lower_db: None,
                    observed_interval_upper_db: None,
                    comparison: "prediction_only",
                });
                continue;
            };

            let relative_frequency_error =
                (full_band.center_frequency_hz - mapped_frequency).abs() / mapped_frequency;
            if relative_frequency_error > case.frequency_match_tolerance_fraction {
                any_frequency_mismatch = true;
                band_reports.push(BandReport {
                    band_id: model_band.band_id.clone(),
                    frequency_regime: frequency_regime_name(&model_band.frequency_regime),
                    frequency_regime_basis: model_band.frequency_regime_basis.clone(),
                    spl_scale_delta_db: delta_db,
                    scale_transfer_u95_db: model_band.scale_transfer_u95_db,
                    model_source_level_db: Some(model_source_level),
                    predicted_full_scale_source_level_db: Some(predicted_full_scale),
                    signed_error_db: Some(predicted_full_scale - full_band.source_level_db),
                    absolute_error_db: Some((predicted_full_scale - full_band.source_level_db).abs()),
                    predicted_interval_lower_db: None,
                    predicted_interval_upper_db: None,
                    mapped_frequency_hz: mapped_frequency,
                    observed_frequency_hz: Some(full_band.center_frequency_hz),
                    frequency_relative_error: Some(relative_frequency_error),
                    observed_full_scale_source_level_db: Some(full_band.source_level_db),
                    observed_interval_lower_db: Some(full_band.source_level_db - full_band.source_level_u95_db),
                    observed_interval_upper_db: Some(full_band.source_level_db + full_band.source_level_u95_db),
                    comparison: "frequency_mismatch",
                });
                continue;
            }

            // Conservative endpoint propagation; no statistical independence assumption is made.
            let predicted_lower = ittc_cavitation_spl(
                model_band.total_spl_db - model_band.total_u95_db,
                model_band.background_spl_db + model_band.background_u95_db,
            );
            let predicted_upper = ittc_cavitation_spl(
                model_band.total_spl_db + model_band.total_u95_db,
                model_band.background_spl_db - model_band.background_u95_db,
            );
            let (Some(net_lower), Some(net_upper)) = (predicted_lower, predicted_upper) else {
                any_unresolved_band = true;
                band_reports.push(BandReport {
                    band_id: model_band.band_id.clone(),
                    frequency_regime: frequency_regime_name(&model_band.frequency_regime),
                    frequency_regime_basis: model_band.frequency_regime_basis.clone(),
                    spl_scale_delta_db: delta_db,
                    scale_transfer_u95_db: model_band.scale_transfer_u95_db,
                    model_source_level_db: Some(model_source_level),
                    predicted_full_scale_source_level_db: Some(predicted_full_scale),
                    signed_error_db: Some(predicted_full_scale - full_band.source_level_db),
                    absolute_error_db: Some((predicted_full_scale - full_band.source_level_db).abs()),
                    predicted_interval_lower_db: None,
                    predicted_interval_upper_db: None,
                    mapped_frequency_hz: mapped_frequency,
                    observed_frequency_hz: Some(full_band.center_frequency_hz),
                    frequency_relative_error: Some(relative_frequency_error),
                    observed_full_scale_source_level_db: Some(full_band.source_level_db),
                    observed_interval_lower_db: Some(full_band.source_level_db - full_band.source_level_u95_db),
                    observed_interval_upper_db: Some(full_band.source_level_db + full_band.source_level_u95_db),
                    comparison: "uncertainty_interval_unresolved",
                });
                continue;
            };

            let predicted_lower = net_lower
                - model_band.facility_transfer_function_db
                - model_band.facility_transfer_u95_db
                + delta_db
                - model_band.scale_transfer_u95_db;
            let predicted_upper = net_upper
                - model_band.facility_transfer_function_db
                + model_band.facility_transfer_u95_db
                + delta_db
                + model_band.scale_transfer_u95_db;
            let observed_lower = full_band.source_level_db - full_band.source_level_u95_db;
            let observed_upper = full_band.source_level_db + full_band.source_level_u95_db;
            let overlaps = predicted_lower <= observed_upper && observed_lower <= predicted_upper;
            if !overlaps {
                any_disjoint_band = true;
            }
            band_reports.push(BandReport {
                band_id: model_band.band_id.clone(),
                frequency_regime: frequency_regime_name(&model_band.frequency_regime),
                frequency_regime_basis: model_band.frequency_regime_basis.clone(),
                spl_scale_delta_db: delta_db,
                scale_transfer_u95_db: model_band.scale_transfer_u95_db,
                model_source_level_db: Some(model_source_level),
                predicted_full_scale_source_level_db: Some(predicted_full_scale),
                signed_error_db: Some(predicted_full_scale - full_band.source_level_db),
                absolute_error_db: Some((predicted_full_scale - full_band.source_level_db).abs()),
                predicted_interval_lower_db: Some(predicted_lower),
                predicted_interval_upper_db: Some(predicted_upper),
                mapped_frequency_hz: mapped_frequency,
                observed_frequency_hz: Some(full_band.center_frequency_hz),
                frequency_relative_error: Some(relative_frequency_error),
                observed_full_scale_source_level_db: Some(full_band.source_level_db),
                observed_interval_lower_db: Some(observed_lower),
                observed_interval_upper_db: Some(observed_upper),
                comparison: if overlaps { "intervals_overlap" } else { "intervals_disjoint" },
            });
        }

        pair_reports.push(PairReport {
            pair_id: pair.pair_id.clone(),
            model_run_id: pair.model.run_id.clone(),
            full_scale_run_id: pair.full_scale.run_id.clone(),
            background_measurement_method: background_measurement_method_name(
                &pair.model.background_measurement_method,
            ),
            background_trace_sha256: pair.model.background_trace_sha256.clone(),
            model_timebase_receipt_sha256: pair.model.timebase_receipt_sha256.clone(),
            full_scale_timebase_receipt_sha256: pair.full_scale.timebase_receipt_sha256.clone(),
            model_operating_point_receipt_sha256: pair.model.operating_point_receipt_sha256.clone(),
            model_raw_trace_sha256: pair.model.raw_trace_sha256.clone(),
            full_scale_operating_point_receipt_sha256: pair.full_scale.operating_point_receipt_sha256.clone(),
            full_scale_raw_trace_sha256: pair.full_scale.raw_trace_sha256.clone(),
            frequency_scale_factor: frequency_factor,
            band_reports,
        });
    }

    if any_frequency_mismatch {
        reasons.push("one or more full-scale bands do not match the ITTC-mapped model frequency".to_string());
    }
    if any_unresolved_band {
        reasons.push("one or more propagated prediction intervals are unresolved".to_string());
    }
    if any_prediction_only {
        reasons.push("one or more model-scale bands have no corresponding full-scale observation".to_string());
    }
    if any_disjoint_band {
        reasons.push("one or more measured full-scale bands are outside the propagated prediction interval".to_string());
    }

    let disposition = if has_dissimilarity
        || case.similarity.facility_transfer_function == FacilityTransferState::Rejected
    {
        "not_comparable"
    } else if case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || case.similarity.wake != SimilarityState::Qualified
        || case.similarity.cavitation_pattern != SimilarityState::Qualified
        || case.similarity.gas_nuclei_state != SimilarityState::Qualified
        || case.similarity.reynolds_scale_review != ReviewState::Reviewed
        || case.similarity.facility_transfer_function != FacilityTransferState::MeasuredAndApplied
        || has_unknown_background_method
        || unique_model_rotation_rates < 2
        || any_unresolved_band
        || any_frequency_mismatch
    {
        "incomplete"
    } else if any_prediction_only {
        "prediction_only"
    } else if any_disjoint_band {
        "comparison_outside_declared_uncertainty"
    } else {
        "computational_comparison_only"
    };

    let comparison_outcome = if any_frequency_mismatch {
        "frequency_mismatch"
    } else if any_unresolved_band {
        "uncertainty_unresolved"
    } else if any_prediction_only {
        "prediction_only"
    } else if any_disjoint_band {
        "at_least_one_band_disjoint"
    } else if pair_reports.iter().all(|p| !p.band_reports.is_empty()) {
        "all_declared_bands_overlap"
    } else {
        "not_evaluated"
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        source_dataset_id: case.source.dataset_id.clone(),
        source_data_sha256: case.source.raw_data_sha256.clone(),
        evidence_kind: evidence_kind_name(&case.evidence_kind),
        scaling_method_id: case.scaling.method_id.clone(),
        bandwidth_convention: bandwidth_name(&case.scaling.bandwidth),
        spectrum_representation: spectrum_representation_name(&case.scaling.representation),
        disposition,
        comparison_outcome,
        unique_model_rotation_rates,
        pair_reports,
        reasons,
        claim_ceiling: "arithmetic_and_declared_comparison_only_no_solver_or_physical_qualification",
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-acoustic-scale-transfer-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-acoustic-scale-transfer-audit <case.json>");
        process::exit(64);
    }

    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("could not read {}: {}", path, error);
            process::exit(64);
        }
    };
    let case: Case = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid case JSON: {}", error);
            process::exit(64);
        }
    };
    let report = audit(&case);
    println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));

    match report.disposition {
        "invalid" | "not_comparable" => process::exit(2),
        "incomplete" | "prediction_only" => process::exit(3),
        "comparison_outside_declared_uncertainty" => process::exit(4),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_case() -> Case {
        let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mut pairs = Vec::new();
        for (index, (model_rpm, full_rpm, model_frequency, full_frequency, total, background, measured)) in [
            (20.0, 1.0, 200.0, 10.0, 110.0, 100.0, 115.56242509439325),
            (22.0, 1.1, 220.0, 11.0, 111.0, 101.0, 116.56242509439325),
        ]
        .into_iter()
        .enumerate()
        {
            pairs.push(ScalePair {
                pair_id: format!("pair-{}", index + 1),
                model: ModelRun {
                    run_id: format!("model-{}", index + 1),
                    timebase_receipt_sha256: digest.to_string(),
                    operating_point_receipt_sha256: digest.to_string(),
                    raw_trace_sha256: digest.to_string(),
                    diameter_m: 0.25,
                    rpm_hz: model_rpm,
                    rpm_u95_hz: 0.01,
                    cavitation_number_n_d: 1.0,
                    source_reference_distance_m: 1.0,
                    background_measurement_method: BackgroundMeasurementMethod::BareHub,
                    background_trace_sha256: digest.to_string(),
                    bands: vec![ModelBand {
                        band_id: "bpf-fundamental".to_string(),
                        center_frequency_hz: model_frequency,
                        frequency_regime: FrequencyRegime::LowFrequency,
                        frequency_regime_basis: "Synthetic known-answer selection".to_string(),
                        scale_transfer_u95_db: 0.4,
                        total_spl_db: total,
                        background_spl_db: background,
                        facility_transfer_function_db: 20.0,
                        total_u95_db: 0.2,
                        background_u95_db: 0.2,
                        facility_transfer_u95_db: 0.1,
                    }],
                },
                full_scale: FullScaleRun {
                    run_id: format!("full-{}", index + 1),
                    timebase_receipt_sha256: digest.to_string(),
                    operating_point_receipt_sha256: digest.to_string(),
                    raw_trace_sha256: digest.to_string(),
                    diameter_m: 5.0,
                    rpm_hz: full_rpm,
                    cavitation_number_n_d: 1.0,
                    source_reference_distance_m: 1.0,
                    source_level_method_id: "declared-independent-source-level-method".to_string(),
                    bands: vec![FullScaleBand {
                        band_id: "bpf-fundamental".to_string(),
                        center_frequency_hz: full_frequency,
                        source_level_db: measured,
                        source_level_u95_db: 0.5,
                    }],
                },
            });
        }
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-known-answer-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-fixture".to_string(),
                citation: "Synthetic known-answer fixture; not measured data".to_string(),
                raw_data_sha256: digest.to_string(),
                rights_status: RightsStatus::Unknown,
                independently_reviewed: false,
            },
            scaling: ScalingSelection {
                method_id: ITTC_METHOD.to_string(),
                bandwidth: Bandwidth::Proportional,
                representation: SpectrumRepresentation::PowerSpectrum,
            },
            similarity: SimilarityEvidence {
                wake: SimilarityState::Qualified,
                cavitation_pattern: SimilarityState::Qualified,
                gas_nuclei_state: SimilarityState::Qualified,
                reynolds_scale_review: ReviewState::Reviewed,
                facility_transfer_function: FacilityTransferState::MeasuredAndApplied,
            },
            frequency_match_tolerance_fraction: 0.01,
            pairs,
        }
    }

    #[test]
    fn committed_synthetic_fixture_is_readable_and_never_qualified() {
        let raw = include_str!("../../../docs/research/marine/fixtures/synthetic-marine-acoustic-scale-transfer-v0.1.0.json");
        let case: Case = serde_json::from_str(raw).expect("committed fixture deserializes");
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert_eq!(report.evidence_kind, "synthetic");
        assert_eq!(report.source_dataset_id, "synthetic-fixture");
        assert_eq!(report.pair_reports.len(), 2);
        assert_eq!(
            report.pair_reports[0].model_timebase_receipt_sha256,
            case.pairs[0].model.timebase_receipt_sha256
        );
        assert_eq!(
            report.pair_reports[0].full_scale_timebase_receipt_sha256,
            case.pairs[0].full_scale.timebase_receipt_sha256
        );
        assert_eq!(
            report.pair_reports[0].band_reports[0].frequency_regime_basis,
            case.pairs[0].model.bands[0].frequency_regime_basis
        );
    }

    #[test]
    fn committed_schema_is_valid_json_and_declares_closed_objects() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-acoustic-scale-transfer-v0.1.0.schema.json"
        ))
        .expect("committed schema is valid JSON");
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["$defs"]["modelBand"]["additionalProperties"],
            false
        );
        assert!(schema["$defs"]["modelRun"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "background_measurement_method"));
        assert!(schema["$defs"]["modelRun"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "background_trace_sha256"));
    }

    #[test]
    fn model_rotation_rates_must_be_separated_beyond_measurement_uncertainty() {
        let mut case = fixture_case();
        case.pairs[0].model.rpm_hz = 20.0;
        case.pairs[0].model.rpm_u95_hz = 0.5;
        case.pairs[1].model.rpm_hz = 20.5;
        case.pairs[1].model.rpm_u95_hz = 0.5;
        let report = audit(&case);
        assert_eq!(report.unique_model_rotation_rates, 1);
        assert!(report.reasons.iter().any(|r| r.contains("two distinct")));
    }

    #[test]
    fn proportional_bandwidth_rejects_density_but_constant_bandwidth_allows_it() {
        let mut case = fixture_case();
        case.scaling.representation = SpectrumRepresentation::PowerSpectralDensity;
        assert_eq!(audit(&case).disposition, "invalid");

        case.scaling.bandwidth = Bandwidth::Constant;
        assert_eq!(audit(&case).disposition, "incomplete");
    }

    #[test]
    fn model_and_full_scale_timebases_are_separate_receipts() {
        let mut case = fixture_case();
        case.pairs[0].model.timebase_receipt_sha256 =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
        case.pairs[0].full_scale.timebase_receipt_sha256 =
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string();
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert_ne!(
            report.pair_reports[0].model_timebase_receipt_sha256,
            report.pair_reports[0].full_scale_timebase_receipt_sha256
        );
    }

    #[test]
    fn ittc_exponent_tables_are_distinct_for_regime_and_bandwidth() {
        assert_eq!(
            scaling_exponents(&FrequencyRegime::HighFrequency, &Bandwidth::Proportional),
            (0.5, 1.0, 1.5, 1.0)
        );
        assert_eq!(
            scaling_exponents(&FrequencyRegime::LowFrequency, &Bandwidth::Proportional),
            (1.0, 1.0, 2.0, 1.0)
        );
        assert_eq!(
            scaling_exponents(&FrequencyRegime::HighFrequency, &Bandwidth::Constant),
            (0.25, 1.0, 1.0, 1.5)
        );
        assert_eq!(
            scaling_exponents(&FrequencyRegime::LowFrequency, &Bandwidth::Constant),
            (0.75, 1.0, 1.5, 1.5)
        );
    }

    #[test]
    fn background_correction_uses_linear_energy_subtraction() {
        let corrected = ittc_cavitation_spl(110.0, 100.0).unwrap();
        assert!((corrected - 109.54242509439325).abs() < 1e-9);
        let at_three_db = ittc_cavitation_spl(103.0, 100.0).unwrap();
        assert!(at_three_db < 103.0);
        assert_eq!(ittc_cavitation_spl(110.0001, 100.0), Some(110.0001));
    }

    #[test]
    fn background_margin_below_three_db_is_not_scaled() {
        assert_eq!(ittc_cavitation_spl(102.9, 100.0), None);
        let mut case = fixture_case();
        case.pairs[0].model.bands[0].total_spl_db = 102.9;
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert_eq!(
            report.pair_reports[0].band_reports[0].comparison,
            "background_dominated_unusable"
        );
        assert_eq!(report.pair_reports[0].band_reports[0].model_source_level_db, None);
    }

    #[test]
    fn background_margin_above_ten_db_is_not_corrected() {
        assert_eq!(ittc_cavitation_spl(110.1, 100.0), Some(110.1));
        let mut case = fixture_case();
        case.pairs[0].model.bands[0].total_spl_db = 110.1;
        let report = audit(&case);
        assert!((report.pair_reports[0].band_reports[0].model_source_level_db.unwrap() - 90.1).abs() < 1e-12);
    }

    #[test]
    fn unknown_background_method_is_incomplete() {
        let mut case = fixture_case();
        case.pairs[0].model.background_measurement_method = BackgroundMeasurementMethod::Unknown;
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert!(report.reasons.iter().any(|reason| reason.contains("background measurement method")));
    }

    #[test]
    fn malformed_background_trace_digest_is_invalid() {
        let mut case = fixture_case();
        case.pairs[0].model.background_trace_sha256 = "missing".to_string();
        let report = audit(&case);
        assert_eq!(report.disposition, "invalid");
    }

    #[test]
    fn ittc_eq16_known_answer_for_common_source_reference_distance() {
        let case = fixture_case();
        let pair = &case.pairs[0];
        let delta = spl_scale_delta_db(
            &pair.model,
            &pair.full_scale,
            &pair.model.bands[0].frequency_regime,
            &case.scaling.bandwidth,
        );
        assert!((delta - 26.020599913279625).abs() < 1e-9);
    }

    #[test]
    fn ittc_eq15_maps_frequency_by_rotation_and_cavitation_number() {
        let case = fixture_case();
        let pair = &case.pairs[0];
        assert!((mapped_frequency_hz(&pair.model, &pair.full_scale, 200.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn fixture_arithmetic_never_promotes_synthetic_data_to_validation() {
        let report = audit(&fixture_case());
        assert_eq!(report.disposition, "incomplete");
        assert!(report.reasons.iter().any(|r| r.contains("synthetic evidence")));
        assert_eq!(report.pair_reports[0].band_reports[0].comparison, "intervals_overlap");
        assert!(report.claim_ceiling.contains("no_solver_or_physical_qualification"));
    }

    #[test]
    fn total_equal_to_background_is_invalid() {
        let mut case = fixture_case();
        case.pairs[0].model.bands[0].total_spl_db = 100.0;
        let report = audit(&case);
        assert_eq!(report.disposition, "invalid");
    }

    #[test]
    fn one_model_rotation_rate_remains_incomplete() {
        let mut case = fixture_case();
        case.pairs.pop();
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert!(report.reasons.iter().any(|r| r.contains("two distinct")));
    }

    #[test]
    fn mapped_frequency_mismatch_blocks_comparison() {
        let mut case = fixture_case();
        case.pairs[0].full_scale.bands[0].center_frequency_hz = 15.0;
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert_eq!(report.comparison_outcome, "frequency_mismatch");
    }

    #[test]
    fn unknown_similarity_cannot_become_a_pass() {
        let mut case = fixture_case();
        case.similarity.cavitation_pattern = SimilarityState::Unknown;
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
    }

    #[test]
    fn explicit_dissimilarity_is_not_comparable() {
        let mut case = fixture_case();
        case.similarity.wake = SimilarityState::Dissimilar;
        let report = audit(&case);
        assert_eq!(report.disposition, "not_comparable");
    }

    #[test]
    fn nonpositive_diameter_is_invalid() {
        let mut case = fixture_case();
        case.pairs[0].model.diameter_m = 0.0;
        let report = audit(&case);
        assert_eq!(report.disposition, "invalid");
    }

    #[test]
    fn disjoint_prediction_and_observation_intervals_are_reported() {
        let mut case = fixture_case();
        case.evidence_kind = EvidenceKind::Measured;
        case.source.rights_status = RightsStatus::PublicOrPermitted;
        case.source.independently_reviewed = true;
        case.pairs[0].full_scale.bands[0].source_level_db = 150.0;
        let report = audit(&case);
        assert_eq!(report.comparison_outcome, "at_least_one_band_disjoint");
        assert_eq!(report.disposition, "comparison_outside_declared_uncertainty");
        assert_eq!(report.pair_reports[0].band_reports[0].comparison, "intervals_disjoint");
        assert!(report.pair_reports[0].band_reports[0].signed_error_db.unwrap() < 0.0);
        assert_eq!(
            report.pair_reports[0].band_reports[0].absolute_error_db.unwrap(),
            report.pair_reports[0].band_reports[0].signed_error_db.unwrap().abs()
        );
        assert_eq!(
            report.pair_reports[0].band_reports[0].scale_transfer_u95_db,
            case.pairs[0].model.bands[0].scale_transfer_u95_db
        );
    }

    #[test]
    fn unresolved_background_uncertainty_is_not_silently_zeroed() {
        let mut case = fixture_case();
        case.pairs[0].model.bands[0].total_u95_db = 5.0;
        case.pairs[0].model.bands[0].background_u95_db = 5.0;
        let report = audit(&case);
        assert_eq!(report.disposition, "incomplete");
        assert_eq!(
            report.pair_reports[0].band_reports[0].comparison,
            "uncertainty_interval_unresolved"
        );
    }

    #[test]
    fn invalid_digest_is_rejected() {
        let mut case = fixture_case();
        case.pairs[0].model.operating_point_receipt_sha256 = "not-a-digest".to_string();
        let report = audit(&case);
        assert_eq!(report.disposition, "invalid");
    }

    #[test]
    fn source_levels_must_share_a_reference_distance() {
        let mut case = fixture_case();
        case.pairs[0].full_scale.source_reference_distance_m = 2.0;
        let report = audit(&case);
        assert_eq!(report.disposition, "invalid");
    }

    #[test]
    fn zero_uncertainty_cannot_masquerade_as_unknown() {
        let mut case = fixture_case();
        case.pairs[0].model.bands[0].total_u95_db = 0.0;
        let report = audit(&case);
        assert_eq!(report.disposition, "invalid");
    }
}
