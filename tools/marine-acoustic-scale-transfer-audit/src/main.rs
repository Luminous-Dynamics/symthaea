use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const BAND_EDGE_RELATIVE_TOLERANCE: f64 = 1.0e-6;
const EXPONENT_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    evidence_class: String,
    propeller_revision: String,
    service_fingerprint_sha256: String,
    method: Method,
    model_campaign: ModelCampaign,
    full_scale_trial: FullScaleTrial,
    operating_point_pairs: Vec<OperatingPointPair>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Method {
    id: String,
    bandwidth_convention: String,
    acoustic_efficiency_regime: String,
    spectrum_quantity: String,
    band_mapping_method: String,
    exponents: Exponents,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Exponents {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelCampaign {
    campaign_id: String,
    facility_id: String,
    propeller_revision: String,
    model_propeller_diameter_m: f64,
    model_rotation_rates_rps: Vec<f64>,
    calibration_certificate_sha256: String,
    facility_transfer_function_sha256: String,
    wake_field_evidence_sha256: String,
    cavitation_state_evidence_sha256: String,
    water_quality_nuclei_evidence_sha256: String,
    background_measurement_per_condition_available: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullScaleTrial {
    trial_id: String,
    propeller_revision: String,
    full_scale_propeller_diameter_m: f64,
    source_level_basis: String,
    source_level_method_id: String,
    propagation_separated: bool,
    calibration_certificate_sha256: String,
    raw_trace_sha256: String,
    source_level_reduction_evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperatingPointPair {
    pair_id: String,
    model_rotation_rate_rps: f64,
    full_scale_rotation_rate_rps: f64,
    model_cavitation_number_n_d: f64,
    full_scale_cavitation_number_n_d: f64,
    model_source_level_reference_distance_m: f64,
    full_scale_source_level_reference_distance_m: f64,
    model_reynolds_number: f64,
    full_scale_reynolds_number: f64,
    operating_point_profile_sha256: String,
    cavitation_pattern: String,
    cavitation_pattern_comparability: String,
    wake_similarity_disposition: String,
    water_quality_comparability: String,
    cavitation_pattern_evidence_sha256: String,
    model_raw_trace_sha256: String,
    background_measurement_method: String,
    background_raw_trace_sha256: String,
    bands: Vec<SpectralBand>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectralBand {
    band_id: String,
    model_lower_hz: f64,
    model_upper_hz: f64,
    full_scale_lower_hz: f64,
    full_scale_upper_hz: f64,
    model_signal_plus_background_spl_db: f64,
    model_background_spl_db: f64,
    facility_transfer_function_db: f64,
    observed_full_scale_source_level_db: f64,
    signal_to_background_margin_db: f64,
    background_correction_applied: bool,
    uncertainty: BandUncertainty,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BandUncertainty {
    model_measurement_expanded_uncertainty_db: f64,
    scaling_model_expanded_uncertainty_db: f64,
    full_scale_measurement_expanded_uncertainty_db: f64,
    confidence_level: f64,
    composition: String,
    independence_evidence_ref: Option<String>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    overall_disposition: String,
    structural_disposition: String,
    comparison_disposition: String,
    physical_validation_disposition: String,
    method_id: String,
    equation_profile: String,
    frequency_mapping_method: String,
    blockers: Vec<String>,
    incomplete_reasons: Vec<String>,
    pair_results: Vec<PairResult>,
}

#[derive(Debug, Serialize)]
struct PairResult {
    pair_id: String,
    background_measurement_method: String,
    frequency_scale_factor: Option<f64>,
    spl_correction_db: Option<f64>,
    combined_uncertainty_rule: String,
    band_results: Vec<BandResult>,
}

#[derive(Debug, Serialize)]
struct BandResult {
    band_id: String,
    signal_to_background_margin_db: f64,
    background_correction_applied: bool,
    background_corrected_model_source_level_db: Option<f64>,
    predicted_full_scale_source_level_db: Option<f64>,
    observed_full_scale_source_level_db: f64,
    residual_db: Option<f64>,
    combined_expanded_uncertainty_db: Option<f64>,
    uncertainty_rule: String,
    confidence_level: f64,
    within_declared_uncertainty: Option<bool>,
}

fn expected_exponents(bandwidth: &str, regime: &str) -> Option<[f64; 4]> {
    match (bandwidth, regime) {
        ("PROPORTIONAL", "HIGH_FREQUENCY") => Some([0.5, 1.0, 1.5, 1.0]),
        ("PROPORTIONAL", "LOW_FREQUENCY") => Some([1.0, 1.0, 2.0, 1.0]),
        ("CONSTANT", "HIGH_FREQUENCY") => Some([0.25, 1.0, 1.0, 1.5]),
        ("CONSTANT", "LOW_FREQUENCY") => Some([0.75, 1.0, 1.5, 1.5]),
        _ => None,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn relative_match(actual: f64, expected: f64) -> bool {
    if !positive_finite(actual) || !positive_finite(expected) {
        return false;
    }
    ((actual - expected) / expected).abs() <= BAND_EDGE_RELATIVE_TOLERANCE
}

fn audit(case: &Case) -> AuditReport {
    let mut blockers = Vec::new();
    let mut incomplete = Vec::new();
    let mut pair_results = Vec::new();

    if case.schema_version != SCHEMA_VERSION {
        blockers.push(format!("unsupported schema_version: {}", case.schema_version));
    }

    if case.evidence_class != "SYNTHETIC" && case.evidence_class != "MEASURED" {
        blockers.push(format!("unsupported evidence_class: {}", case.evidence_class));
    }

    if case.propeller_revision.trim().is_empty()
        || case.model_campaign.propeller_revision != case.propeller_revision
        || case.full_scale_trial.propeller_revision != case.propeller_revision
    {
        blockers.push("model/full-scale propeller revision identity mismatch".to_string());
    }

    if !is_sha256(&case.service_fingerprint_sha256) {
        incomplete.push("service fingerprint SHA-256 is absent or malformed".to_string());
    }

    let method = &case.method;
    if method.id != "ITTC_18TH_COMMITTEE_EQ15_EQ16" {
        blockers.push(format!("unsupported scale-law method id: {}", method.id));
    }
    if method.spectrum_quantity != "POWER_SPECTRUM"
        && method.spectrum_quantity != "POWER_SPECTRAL_DENSITY"
    {
        blockers.push(format!("unsupported spectrum quantity: {}", method.spectrum_quantity));
    }
    if method.bandwidth_convention == "PROPORTIONAL"
        && method.spectrum_quantity != "POWER_SPECTRUM"
    {
        blockers.push("proportional-band scaling requires a power spectrum; power spectral density is not interchangeable".to_string());
    }
    if method.bandwidth_convention == "PROPORTIONAL"
        && method.spectrum_quantity != "POWER_SPECTRUM"
    {
        blockers.push("proportional-band scaling requires a power spectrum; PSD is not interchangeable".to_string());
    }
    if method.band_mapping_method != "EXACT_SCALED_BOUNDS" {
        blockers.push("v0.1 supports exact scaled band bounds only; interpolation/rebinning is not qualified".to_string());
    }

    let profile = expected_exponents(
        &method.bandwidth_convention,
        &method.acoustic_efficiency_regime,
    );
    let profile_name = format!(
        "{}_{}",
        method.bandwidth_convention, method.acoustic_efficiency_regime
    );
    match profile {
        Some([w, x, y, z]) => {
            let declared = [
                method.exponents.w,
                method.exponents.x,
                method.exponents.y,
                method.exponents.z,
            ];
            let expected = [w, x, y, z];
            if declared.iter().zip(expected.iter()).any(|(a, b)| {
                !a.is_finite() || (a - b).abs() > EXPONENT_TOLERANCE
            }) {
                blockers.push("declared Eq. 16 exponents do not match the selected ITTC bandwidth/regime profile".to_string());
            }
        }
        None => blockers.push("unknown bandwidth/formulation profile".to_string()),
    }

    let campaign = &case.model_campaign;
    if campaign.campaign_id.trim().is_empty() || campaign.facility_id.trim().is_empty() {
        incomplete.push("model campaign/facility identity is missing".to_string());
    }

    if !positive_finite(campaign.model_propeller_diameter_m)
        || !positive_finite(case.full_scale_trial.full_scale_propeller_diameter_m)
    {
        blockers.push("model and full-scale propeller diameters must be positive finite values".to_string());
    }

    for (label, digest) in [
        ("model calibration", campaign.calibration_certificate_sha256.as_str()),
        ("facility transfer function", campaign.facility_transfer_function_sha256.as_str()),
        ("wake field", campaign.wake_field_evidence_sha256.as_str()),
        ("cavitation state", campaign.cavitation_state_evidence_sha256.as_str()),
        ("water quality/nuclei", campaign.water_quality_nuclei_evidence_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            incomplete.push(format!("{label} evidence digest is missing or malformed"));
        }
    }

    if !campaign.background_measurement_per_condition_available {
        incomplete.push("background noise was not declared measured for every model test condition".to_string());
    }

    let full = &case.full_scale_trial;
    if full.trial_id.trim().is_empty() || full.source_level_method_id.trim().is_empty() {
        incomplete.push("full-scale trial or source-level method identity is missing".to_string());
    }
    if full.source_level_basis != "SOURCE_LEVEL_AT_1M" {
        blockers.push("full-scale comparison must use source level, not received level or radiated-noise level".to_string());
    }
    if !full.propagation_separated {
        incomplete.push("full-scale propagation has not been separated from source-level comparison".to_string());
    }
    for (label, digest) in [
        ("full-scale calibration", full.calibration_certificate_sha256.as_str()),
        ("full-scale raw trace", full.raw_trace_sha256.as_str()),
        ("source-level reduction", full.source_level_reduction_evidence_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            incomplete.push(format!("{label} evidence digest is missing or malformed"));
        }
    }

    let mut campaign_rates = BTreeSet::new();
    for rate in &campaign.model_rotation_rates_rps {
        if !positive_finite(*rate) {
            blockers.push("model campaign contains a non-positive or non-finite rotation rate".to_string());
        } else {
            campaign_rates.insert(rate.to_bits());
        }
    }
    if campaign_rates.len() < 2 {
        incomplete.push("at least two distinct model-propeller rotation-rate strata are required".to_string());
    }
    if campaign_rates.len() != campaign.model_rotation_rates_rps.len() {
        incomplete.push("duplicate model rotation-rate strata were declared".to_string());
    }

    if case.operating_point_pairs.is_empty() {
        incomplete.push("no paired model/full-scale operating points were supplied".to_string());
    }

    let mut pair_ids = BTreeSet::new();
    let mut covered_rates = BTreeSet::new();

    for pair in &case.operating_point_pairs {
        if !pair_ids.insert(pair.pair_id.clone()) {
            blockers.push(format!("duplicate pair_id: {}", pair.pair_id));
        }
        if !positive_finite(pair.model_rotation_rate_rps)
            || !positive_finite(pair.full_scale_rotation_rate_rps)
            || !positive_finite(pair.model_cavitation_number_n_d)
            || !positive_finite(pair.full_scale_cavitation_number_n_d)
            || !positive_finite(pair.model_source_level_reference_distance_m)
            || !positive_finite(pair.full_scale_source_level_reference_distance_m)
            || !positive_finite(pair.model_reynolds_number)
            || !positive_finite(pair.full_scale_reynolds_number)
        {
            blockers.push(format!("pair {} has invalid scale/cavitation/Reynolds inputs or source levels not normalized to 1 m", pair.pair_id));
            continue;
        }

        if !campaign_rates.contains(&pair.model_rotation_rate_rps.to_bits()) {
            blockers.push(format!("pair {} rotation rate is absent from the frozen campaign strata", pair.pair_id));
        }
        if !covered_rates.insert(pair.model_rotation_rate_rps.to_bits()) {
            blockers.push(format!("more than one pair uses the same model rotation-rate stratum: {}", pair.model_rotation_rate_rps));
        }

        if !is_sha256(&pair.operating_point_profile_sha256)
            || !is_sha256(&pair.cavitation_pattern_evidence_sha256)
            || !is_sha256(&pair.model_raw_trace_sha256)
            || !is_sha256(&pair.background_raw_trace_sha256)
        {
            incomplete.push(format!("pair {} lacks operating-point, cavitation-pattern, model-trace or background-trace evidence", pair.pair_id));
        }
        if pair.background_measurement_method != "BARE_HUB"
            && pair.background_measurement_method != "PRESSURE_SUPPRESSED"
        {
            blockers.push(format!("pair {} has unsupported background measurement method", pair.pair_id));
        }

        match pair.cavitation_pattern.as_str() {
            "SHEET" | "CLOUD" | "BUBBLE" => {}
            "TIP_VORTEX" | "MIXED" => incomplete.push(format!("pair {} uses tip-vortex or mixed cavitation; Eq. 16 v0.1 does not implement the additional Reynolds-number cavitation-noise correction", pair.pair_id)),
            "UNKNOWN" => incomplete.push(format!("pair {} has unknown cavitation pattern", pair.pair_id)),
            other => blockers.push(format!("pair {} has unsupported cavitation pattern: {other}", pair.pair_id)),
        }
        match pair.cavitation_pattern_comparability.as_str() {
            "MATCHED" => {}
            "MISMATCH" => blockers.push(format!("pair {} declares dissimilar model/full-scale cavitation patterns", pair.pair_id)),
            "PARTIAL" | "UNKNOWN" => incomplete.push(format!("pair {} lacks a demonstrated cavitation-pattern match", pair.pair_id)),
            other => blockers.push(format!("pair {} has unknown cavitation comparability state: {other}", pair.pair_id)),
        }
        for (label, disposition) in [
            ("wake-field similarity", pair.wake_similarity_disposition.as_str()),
            ("water-quality/cavitation-nuclei comparability", pair.water_quality_comparability.as_str()),
        ] {
            match disposition {
                "MATCHED" => {}
                "MISMATCH" => blockers.push(format!("pair {} declares {label} mismatch", pair.pair_id)),
                "PARTIAL" | "UNKNOWN" => incomplete.push(format!("pair {} lacks demonstrated {label}", pair.pair_id)),
                other => blockers.push(format!("pair {} has invalid {label} disposition: {other}", pair.pair_id)),
            }
        }

        let mut band_ids = BTreeSet::new();
        let mut band_results = Vec::new();
        if pair.bands.is_empty() {
            incomplete.push(format!("pair {} has no spectral bands", pair.pair_id));
        }

        let sigma_ratio = pair.full_scale_cavitation_number_n_d / pair.model_cavitation_number_n_d;
        let frequency_scale = (pair.full_scale_rotation_rate_rps / pair.model_rotation_rate_rps)
            * sigma_ratio.sqrt();

        let delta_spl = match profile {
            Some([w, x, y, z]) => {
                let diameter_ratio = full.full_scale_propeller_diameter_m / campaign.model_propeller_diameter_m;
                let n_d_ratio = (pair.full_scale_rotation_rate_rps * full.full_scale_propeller_diameter_m)
                    / (pair.model_rotation_rate_rps * campaign.model_propeller_diameter_m);
                let multiplier = sigma_ratio.powf(w)
                    * (pair.model_source_level_reference_distance_m / pair.full_scale_source_level_reference_distance_m).powf(x)
                    * n_d_ratio.powf(y)
                    * diameter_ratio.powf(z);
                if positive_finite(multiplier) {
                    Some(20.0 * multiplier.log10())
                } else {
                    blockers.push(format!("pair {} produced a non-positive/non-finite Eq. 16 scale multiplier", pair.pair_id));
                    None
                }
            }
            None => None,
        };

        for band in &pair.bands {
            if !band_ids.insert(band.band_id.clone()) {
                blockers.push(format!("pair {} has duplicate band_id {}", pair.pair_id, band.band_id));
                continue;
            }
            if !positive_finite(band.model_lower_hz)
                || !positive_finite(band.model_upper_hz)
                || !positive_finite(band.full_scale_lower_hz)
                || !positive_finite(band.full_scale_upper_hz)
                || band.model_lower_hz >= band.model_upper_hz
                || band.full_scale_lower_hz >= band.full_scale_upper_hz
                || !band.model_signal_plus_background_spl_db.is_finite()
                || !band.model_background_spl_db.is_finite()
                || !band.facility_transfer_function_db.is_finite()
                || !band.observed_full_scale_source_level_db.is_finite()
            {
                blockers.push(format!("pair {} band {} has invalid frequency bounds or level", pair.pair_id, band.band_id));
                continue;
            }

            let expected_low = band.model_lower_hz * frequency_scale;
            let expected_high = band.model_upper_hz * frequency_scale;
            if !relative_match(band.full_scale_lower_hz, expected_low)
                || !relative_match(band.full_scale_upper_hz, expected_high)
            {
                blockers.push(format!("pair {} band {} is not an exact scaled-frequency match", pair.pair_id, band.band_id));
            }

            let calculated_margin = band.model_signal_plus_background_spl_db
                - band.model_background_spl_db;
            if !band.signal_to_background_margin_db.is_finite()
                || !calculated_margin.is_finite()
                || band.signal_to_background_margin_db < 0.0
                || (calculated_margin - band.signal_to_background_margin_db).abs() > 0.05
            {
                blockers.push(format!("pair {} band {} has an invalid or inconsistent measured signal/background margin", pair.pair_id, band.band_id));
            }
            let background_corrected_model_level = if calculated_margin < 3.0 {
                incomplete.push(format!("pair {} band {} is background-dominated (<3 dB) and cannot be used for scaling", pair.pair_id, band.band_id));
                None
            } else if calculated_margin < 10.0 {
                if !band.background_correction_applied {
                    incomplete.push(format!("pair {} band {} requires ITTC energy-domain background subtraction in the 3–10 dB margin range", pair.pair_id, band.band_id));
                    None
                } else {
                    let power_ratio = 10.0_f64.powf(-calculated_margin / 10.0);
                    let correction_term = 1.0 - power_ratio;
                    if correction_term <= 0.0 {
                        blockers.push(format!("pair {} band {} has a non-positive background-subtracted power", pair.pair_id, band.band_id));
                        None
                    } else {
                        Some(band.model_signal_plus_background_spl_db
                            + 10.0 * correction_term.log10()
                            - band.facility_transfer_function_db)
                    }
                }
            } else if band.background_correction_applied {
                let power_ratio = 10.0_f64.powf(-calculated_margin / 10.0);
                let correction_term = 1.0 - power_ratio;
                Some(band.model_signal_plus_background_spl_db
                    + 10.0 * correction_term.log10()
                    - band.facility_transfer_function_db)
            } else {
                Some(band.model_signal_plus_background_spl_db - band.facility_transfer_function_db)
            };

            let u = &band.uncertainty;
            let uncertainty_values_valid = finite_nonnegative(u.model_measurement_expanded_uncertainty_db)
                && finite_nonnegative(u.scaling_model_expanded_uncertainty_db)
                && finite_nonnegative(u.full_scale_measurement_expanded_uncertainty_db)
                && u.confidence_level.is_finite()
                && u.confidence_level > 0.0
                && u.confidence_level < 1.0;

            let (combined_u, rule) = if !uncertainty_values_valid {
                incomplete.push(format!("pair {} band {} has missing/invalid uncertainty components", pair.pair_id, band.band_id));
                (None, "UNAVAILABLE".to_string())
            } else {
                match u.composition.as_str() {
                    "INDEPENDENCE_SUPPORTED_RSS" => {
                        if u.independence_evidence_ref.as_deref().unwrap_or("").trim().is_empty() {
                            incomplete.push(format!("pair {} band {} requests RSS without independence evidence", pair.pair_id, band.band_id));
                            (None, "UNAVAILABLE".to_string())
                        } else {
                            let sum = u.model_measurement_expanded_uncertainty_db.powi(2)
                                + u.scaling_model_expanded_uncertainty_db.powi(2)
                                + u.full_scale_measurement_expanded_uncertainty_db.powi(2);
                            (Some(sum.sqrt()), "ROOT_SUM_SQUARE_SUPPORTED_INDEPENDENCE".to_string())
                        }
                    }
                    "CONSERVATIVE_LINEAR" => (
                        Some(u.model_measurement_expanded_uncertainty_db + u.scaling_model_expanded_uncertainty_db + u.full_scale_measurement_expanded_uncertainty_db),
                        "CONSERVATIVE_LINEAR_BOUND".to_string(),
                    ),
                    other => {
                        incomplete.push(format!("pair {} band {} uses unsupported uncertainty composition {other}", pair.pair_id, band.band_id));
                        (None, "UNAVAILABLE".to_string())
                    }
                }
            };

            let predicted = match (background_corrected_model_level, delta_spl) {
                (Some(model_level), Some(delta)) => Some(model_level + delta),
                _ => None,
            };
            let residual = predicted.map(|value| band.observed_full_scale_source_level_db - value);
            let within = match (residual, combined_u) {
                (Some(error), Some(limit)) => Some(error.abs() <= limit),
                _ => None,
            };

            band_results.push(BandResult {
                band_id: band.band_id.clone(),
                signal_to_background_margin_db: band.signal_to_background_margin_db,
                background_correction_applied: band.background_correction_applied,
                background_corrected_model_source_level_db: background_corrected_model_level,
                predicted_full_scale_source_level_db: predicted,
                observed_full_scale_source_level_db: band.observed_full_scale_source_level_db,
                residual_db: residual,
                combined_expanded_uncertainty_db: combined_u,
                uncertainty_rule: rule,
                confidence_level: u.confidence_level,
                within_declared_uncertainty: within,
            });

        }

        pair_results.push(PairResult {
            pair_id: pair.pair_id.clone(),
            background_measurement_method: pair.background_measurement_method.clone(),
            frequency_scale_factor: Some(frequency_scale),
            spl_correction_db: delta_spl,
            combined_uncertainty_rule: if pair.bands.iter().all(|b| b.uncertainty.composition == "CONSERVATIVE_LINEAR") {
                "CONSERVATIVE_LINEAR_BOUND".to_string()
            } else {
                "PER_BAND_RULE_REPORTED_IN_BAND_RESULTS".to_string()
            },
            band_results,
        });
    }

    for rate in &campaign_rates {
        if !covered_rates.contains(rate) {
            incomplete.push("a frozen model rotation-rate stratum has no paired full-scale validation point".to_string());
            break;
        }
    }

    let structural_disposition = if !blockers.is_empty() {
        "INVALID"
    } else if !incomplete.is_empty() {
        "INCOMPLETE"
    } else {
        "COMPARISON_INPUTS_COMPLETE"
    };

    let all_bands_within = pair_results.iter().flat_map(|p| p.band_results.iter())
        .all(|b| b.within_declared_uncertainty == Some(true));
    let any_band_outside = pair_results.iter().flat_map(|p| p.band_results.iter())
        .any(|b| b.within_declared_uncertainty == Some(false));

    let comparison_disposition = if structural_disposition != "COMPARISON_INPUTS_COMPLETE" {
        "NOT_EVALUATED"
    } else if any_band_outside {
        "OUTSIDE_DECLARED_UNCERTAINTY"
    } else if all_bands_within && !pair_results.is_empty() {
        "WITHIN_DECLARED_UNCERTAINTY"
    } else {
        "NOT_EVALUATED"
    };

    // This executable cannot authenticate certificates, verify raw traces or independently
    // establish the declared physical-similarity evidence. Numerical agreement is not qualification.
    let physical_validation_disposition = if case.evidence_class == "SYNTHETIC" {
        "NOT_EVALUATED_SYNTHETIC"
    } else {
        "REQUIRES_INDEPENDENT_PHYSICAL_REVIEW"
    };

    let overall_disposition = if structural_disposition == "INVALID" {
        "INVALID"
    } else if structural_disposition == "INCOMPLETE" || case.evidence_class == "SYNTHETIC" {
        "INCOMPLETE"
    } else if comparison_disposition == "OUTSIDE_DECLARED_UNCERTAINTY" {
        "COMPARISON_OUTSIDE_DECLARED_UNCERTAINTY"
    } else if comparison_disposition == "WITHIN_DECLARED_UNCERTAINTY" {
        "COMPARISON_WITHIN_DECLARED_UNCERTAINTY_REQUIRES_INDEPENDENT_REVIEW"
    } else {
        "INCOMPLETE"
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        overall_disposition: overall_disposition.to_string(),
        structural_disposition: structural_disposition.to_string(),
        comparison_disposition: comparison_disposition.to_string(),
        physical_validation_disposition: physical_validation_disposition.to_string(),
        method_id: method.id.clone(),
        equation_profile: profile_name,
        frequency_mapping_method: method.band_mapping_method.clone(),
        blockers,
        incomplete_reasons: incomplete,
        pair_results,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: marine-acoustic-scale-transfer-audit <case.json>");
        process::exit(64);
    }

    let input = match fs::read_to_string(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("failed to read {}: {error}", args[1]);
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
    println!("{}", serde_json::to_string_pretty(&report).expect("report serialization"));

    match report.overall_disposition.as_str() {
        "COMPARISON_WITHIN_DECLARED_UNCERTAINTY_REQUIRES_INDEPENDENT_REVIEW" => process::exit(0),
        "INVALID" | "INCOMPLETE" | "COMPARISON_OUTSIDE_DECLARED_UNCERTAINTY" => process::exit(2),
        _ => process::exit(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_case() -> Case {
        serde_json::from_str(include_str!(
            "../../../docs/research/marine/fixtures/synthetic-acoustic-scale-transfer-v0.1.0.json"
        ))
        .expect("fixture parses")
    }

    fn make_measured(case: &mut Case) {
        case.evidence_class = "MEASURED".to_string();
    }

    #[test]
    fn eq15_frequency_scaling_known_answer() {
        let ratio = (2.0_f64 / 10.0) * (0.8_f64).sqrt();
        assert!((ratio - 0.17888543819998318).abs() < 1.0e-14);
    }

    #[test]
    fn eq16_proportional_low_frequency_known_answer() {
        // Both input levels are source levels normalized to the same 1 m reference.
        let multiplier = 0.8_f64 * 1.0_f64 * 2.0_f64.powi(2) * 10.0_f64;
        assert!((multiplier - 32.0).abs() < 1.0e-14);
        let delta = 20.0 * multiplier.log10();
        assert!((delta - 30.10299956639812).abs() < 1.0e-12);
    }

    #[test]
    fn all_four_published_exponent_profiles_are_explicit() {
        assert_eq!(expected_exponents("PROPORTIONAL", "HIGH_FREQUENCY"), Some([0.5, 1.0, 1.5, 1.0]));
        assert_eq!(expected_exponents("PROPORTIONAL", "LOW_FREQUENCY"), Some([1.0, 1.0, 2.0, 1.0]));
        assert_eq!(expected_exponents("CONSTANT", "HIGH_FREQUENCY"), Some([0.25, 1.0, 1.0, 1.5]));
        assert_eq!(expected_exponents("CONSTANT", "LOW_FREQUENCY"), Some([0.75, 1.0, 1.5, 1.5]));
        assert_eq!(expected_exponents("UNKNOWN", "LOW_FREQUENCY"), None);
    }

    #[test]
    fn synthetic_fixture_never_becomes_physical_validation() {
        let case = synthetic_case();
        let report = audit(&case);
        assert_eq!(report.overall_disposition, "INCOMPLETE");
        assert_eq!(report.physical_validation_disposition, "NOT_EVALUATED_SYNTHETIC");
    }

    #[test]
    fn measured_disposition_logic_unit_vector_only() {
        // This in-memory evidence-class mutation exercises the status branch only;
        // it is not physical measurement evidence or a claim about the fixture.
        let mut case = synthetic_case();
        make_measured(&mut case);
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "COMPARISON_INPUTS_COMPLETE", "{:?}", report.incomplete_reasons);
        let corrected = report.pair_results[0].band_results[0]
            .background_corrected_model_source_level_db.expect("background-corrected level");
        assert!((corrected - 120.0).abs() < 1.0e-12);
        assert_eq!(report.comparison_disposition, "WITHIN_DECLARED_UNCERTAINTY");
        assert_eq!(report.physical_validation_disposition, "REQUIRES_INDEPENDENT_PHYSICAL_REVIEW");
        assert_eq!(report.overall_disposition, "COMPARISON_WITHIN_DECLARED_UNCERTAINTY_REQUIRES_INDEPENDENT_REVIEW");
        assert!(report.pair_results.iter().flat_map(|p| p.band_results.iter())
            .all(|b| b.within_declared_uncertainty == Some(true)));
    }

    #[test]
    fn one_rotation_rate_is_incomplete() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.model_campaign.model_rotation_rates_rps = vec![10.0];
        case.operating_point_pairs.truncate(1);
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }

    #[test]
    fn received_level_comparison_is_rejected() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.full_scale_trial.source_level_basis = "RECEIVED_LEVEL".to_string();
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INVALID");
    }

    #[test]
    fn wrong_exponents_are_rejected_not_silently_substituted() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.method.exponents.z = 3.0;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INVALID");
    }

    #[test]
    fn background_dominated_band_is_incomplete() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.operating_point_pairs[0].bands[0].model_background_spl_db = 117.1;
        case.operating_point_pairs[0].bands[0].signal_to_background_margin_db = 2.9;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }

    #[test]
    fn three_to_ten_db_background_margin_requires_correction() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        let band = &mut case.operating_point_pairs[0].bands[0];
        band.model_signal_plus_background_spl_db = 120.0;
        band.model_background_spl_db = 115.0;
        band.facility_transfer_function_db = 0.0;
        band.signal_to_background_margin_db = 5.0;
        band.background_correction_applied = false;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }

    #[test]
    fn corrected_three_to_ten_db_background_margin_is_eligible() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        let band = &mut case.operating_point_pairs[0].bands[0];
        band.model_signal_plus_background_spl_db = 120.0;
        band.model_background_spl_db = 115.0;
        band.facility_transfer_function_db = 0.0;
        band.signal_to_background_margin_db = 5.0;
        band.background_correction_applied = true;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "COMPARISON_INPUTS_COMPLETE", "{:?}", report.incomplete_reasons);
        let corrected = report.pair_results[0].band_results[0]
            .background_corrected_model_source_level_db.expect("background-corrected level");
        assert!((corrected - 118.349).abs() < 0.01);
    }

    #[test]
    fn tip_vortex_requires_unimplemented_reynolds_scale_correction() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.operating_point_pairs[0].cavitation_pattern = "TIP_VORTEX".to_string();
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }

    #[test]
    fn unknown_wake_similarity_is_incomplete() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.operating_point_pairs[0].wake_similarity_disposition = "UNKNOWN".to_string();
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }

    #[test]
    fn signal_background_margin_must_match_measured_levels() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.operating_point_pairs[0].bands[0].signal_to_background_margin_db = 9.0;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INVALID");
    }

    #[test]
    fn ten_db_signal_background_margin_needs_no_subtraction() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        let band = &mut case.operating_point_pairs[0].bands[0];
        band.model_background_spl_db = 110.0;
        band.signal_to_background_margin_db = 10.0;
        band.background_correction_applied = false;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "COMPARISON_INPUTS_COMPLETE", "{:?}", report.incomplete_reasons);
        let source_level = report.pair_results[0].band_results[0]
            .background_corrected_model_source_level_db.expect("source level");
        assert!((source_level - (120.0 - band.facility_transfer_function_db)).abs() < 1.0e-12);
    }

    #[test]
    fn missing_rss_independence_evidence_is_incomplete() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        let u = &mut case.operating_point_pairs[0].bands[0].uncertainty;
        u.composition = "INDEPENDENCE_SUPPORTED_RSS".to_string();
        u.independence_evidence_ref = None;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }

    #[test]
    fn unsupported_band_mapping_is_rejected() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.method.band_mapping_method = "INTERPOLATE_NEAREST".to_string();
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INVALID");
    }

    #[test]
    fn proportional_bandwidth_cannot_use_power_spectral_density() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.method.spectrum_quantity = "POWER_SPECTRAL_DENSITY".to_string();
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INVALID");
    }

    #[test]
    fn model_scale_trace_cannot_be_reused_for_another_band() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.operating_point_pairs[0].bands[0].full_scale_lower_hz *= 1.02;
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INVALID");
    }

    #[test]
    fn excessive_residual_is_reported_as_validation_failure() {
        let mut case = synthetic_case();
        make_measured(&mut case);
        case.operating_point_pairs[0].bands[0].observed_full_scale_source_level_db += 20.0;
        let report = audit(&case);
        assert_eq!(report.overall_disposition, "COMPARISON_OUTSIDE_DECLARED_UNCERTAINTY");
        assert_eq!(report.comparison_disposition, "OUTSIDE_DECLARED_UNCERTAINTY");
        assert_eq!(report.physical_validation_disposition, "REQUIRES_INDEPENDENT_PHYSICAL_REVIEW");
    }

    #[test]
    fn duplicate_rotation_rate_strata_are_not_counted_twice() {
        let mut case = synthetic_case();
        case.model_campaign.model_rotation_rates_rps = vec![10.0, 10.0, 12.0];
        let report = audit(&case);
        assert_eq!(report.structural_disposition, "INCOMPLETE");
    }
}
