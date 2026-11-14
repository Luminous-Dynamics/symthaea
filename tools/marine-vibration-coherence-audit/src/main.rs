use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const PRESSURE_PSD_UNIT: &str = "pa2_per_hz";
const ACCELERATION_PSD_UNIT: &str = "m2_per_s4_per_hz";
const CROSS_PSD_UNIT: &str = "pa_m_per_s2_per_hz";
const PRESSURE_REFERENCE_PA: f64 = 1.0e-6;
const PSD_MATRIX_RELATIVE_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
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

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum WelchWindow {
    Hann,
    Hamming,
    Blackman,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    vessel_id: String,
    propeller_revision: String,
    operating_condition_id: String,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_binding_receipt_sha256: String,
    synchronization_receipt_sha256: String,
    alignment_offset_applied_in_estimator: bool,
    residual_alignment_uncertainty_s: f64,
    maximum_phase_uncertainty_rad: f64,
    predeclared_coherence_interpretation_threshold: f64,
    cross_spectrum_convention: String,
    pressure_channel: ChannelEvidence,
    vibration_channel: ChannelEvidence,
    estimator: WelchEstimator,
    bands: Vec<BandEstimate>,
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
struct ChannelEvidence {
    instrument_id: String,
    channel_id: String,
    physical_location: String,
    orientation: String,
    calibration_certificate_sha256: String,
    raw_trace_sha256: String,
    timebase_id: String,
    sample_rate_hz: f64,
    sample_count: u64,
    window_start_s: f64,
    window_end_s: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WelchEstimator {
    estimator_id: String,
    estimator_version: String,
    window: WelchWindow,
    segment_length_samples: u64,
    overlap_fraction: f64,
    number_of_averages: u32,
    frequency_resolution_hz: f64,
    output_scaling: String,
    sidedness: String,
    averaging_method: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BandEstimate {
    band_id: String,
    frequency_hz: f64,
    pressure_psd: f64,
    pressure_psd_expanded_uncertainty: f64,
    pressure_psd_unit: String,
    acceleration_psd: f64,
    acceleration_psd_expanded_uncertainty: f64,
    acceleration_psd_unit: String,
    cross_psd_real: f64,
    cross_psd_real_expanded_uncertainty: f64,
    cross_psd_imag: f64,
    cross_psd_imag_expanded_uncertainty: f64,
    cross_psd_unit: String,
    uncertainty_coverage_level: f64,
    cross_spectrum_phase_expanded_uncertainty_rad: f64,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: &'static str,
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
    vibration_reference_location: String,
    vibration_reference_orientation: String,
    pressure_calibration_certificate_sha256: String,
    vibration_calibration_certificate_sha256: String,
    pressure_raw_trace_sha256: String,
    vibration_raw_trace_sha256: String,
    shared_timebase_id: String,
    sample_rate_hz: f64,
    sample_count: u64,
    pressure_window_start_s: f64,
    pressure_window_end_s: f64,
    estimator_id: String,
    estimator_version: String,
    cross_spectrum_convention: String,
    welch_window: &'static str,
    output_scaling: String,
    sidedness: String,
    averaging_method: String,
    number_of_averages: u32,
    overlap_fraction: f64,
    frequency_resolution_hz: f64,
    predeclared_coherence_interpretation_threshold: f64,
    disposition: &'static str,
    claim_ceiling: &'static str,
    band_reports: Vec<BandReport>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct BandReport {
    band_id: String,
    frequency_hz: f64,
    pressure_psd_pa2_per_hz: f64,
    acceleration_psd_m2_per_s4_per_hz: f64,
    cross_psd_real: f64,
    cross_psd_imag: f64,
    coherence_magnitude_squared: f64,
    coherence_lower_bound: Option<f64>,
    coherence_upper_bound: Option<f64>,
    coherence_threshold_met: bool,
    h1_real_pa_per_m_s2: f64,
    h1_imag_pa_per_m_s2: f64,
    h1_magnitude_pa_per_m_s2: f64,
    h1_phase_rad: Option<f64>,
    phase_uncertainty_bound_rad: f64,
    phase_status: &'static str,
    vibration_reference_coherent_pressure_psd_pa2_per_hz: f64,
    vibration_reference_coherent_pressure_psd_lower_bound: Option<f64>,
    vibration_reference_coherent_pressure_psd_upper_bound: Option<f64>,
    non_coherent_residual_pressure_psd_pa2_per_hz: f64,
    non_coherent_residual_pressure_psd_lower_bound: Option<f64>,
    non_coherent_residual_pressure_psd_upper_bound: Option<f64>,
    pressure_psd_level_db_re_1upa2_per_hz: f64,
    coherent_pressure_psd_level_db_re_1upa2_per_hz: Option<f64>,
    residual_pressure_psd_level_db_re_1upa2_per_hz: Option<f64>,
    uncertainty_status: &'static str,
    marginal_uncertainty_coverage_level: f64,
    uncertainty_interpretation: &'static str,
    interpretation: &'static str,
}

#[derive(Debug)]
struct ComputedBand {
    report: BandReport,
    uncertainty_unresolved: bool,
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

fn db_re_reference(value: f64, reference: f64) -> Option<f64> {
    if !finite_positive(value) || !finite_positive(reference) {
        return None;
    }
    let level = 10.0 * (value / reference).log10();
    level.is_finite().then_some(level)
}

fn validate_channel(channel: &ChannelEvidence, label: &str, errors: &mut Vec<String>) {
    for (field, value) in [
        ("instrument_id", channel.instrument_id.as_str()),
        ("channel_id", channel.channel_id.as_str()),
        ("physical_location", channel.physical_location.as_str()),
        ("orientation", channel.orientation.as_str()),
        ("timebase_id", channel.timebase_id.as_str()),
    ] {
        if value.trim().is_empty() {
            errors.push(format!("{label}: {field} is required"));
        }
    }
    for (field, digest) in [
        ("calibration_certificate_sha256", &channel.calibration_certificate_sha256),
        ("raw_trace_sha256", &channel.raw_trace_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label}: {field} must be a 64-character SHA-256 hex digest"));
        }
    }
    if !finite_positive(channel.sample_rate_hz)
        || channel.sample_count < 2
        || !channel.window_start_s.is_finite()
        || !channel.window_end_s.is_finite()
        || channel.window_end_s <= channel.window_start_s
    {
        errors.push(format!("{label}: invalid sampling rate, sample count or measurement window"));
    }
    if channel.sample_count >= 2
        && finite_positive(channel.sample_rate_hz)
        && channel.window_start_s.is_finite()
        && channel.window_end_s.is_finite()
        && channel.window_end_s > channel.window_start_s
    {
        let implied = channel.sample_count as f64 / channel.sample_rate_hz;
        let observed = channel.window_end_s - channel.window_start_s;
        let relative_error = (implied - observed).abs() / observed;
        if relative_error > 0.02 {
            errors.push(format!("{label}: sample_count / sample_rate does not match declared window duration within 2%"));
        }
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
        || case.vessel_id.trim().is_empty()
        || case.propeller_revision.trim().is_empty()
        || case.operating_condition_id.trim().is_empty()
    {
        errors.push("case, dataset, vessel, propeller revision and operating-condition identities are required".to_string());
    }
    for (field, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("operating-point receipt", &case.operating_point_receipt_sha256),
        ("service fingerprint", &case.service_fingerprint_sha256),
        ("environment fingerprint", &case.environment_fingerprint_sha256),
        ("timebase binding receipt", &case.timebase_binding_receipt_sha256),
        ("synchronization receipt", &case.synchronization_receipt_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{field} digest must be a 64-character SHA-256 hex string"));
        }
    }
    validate_channel(&case.pressure_channel, "pressure channel", &mut errors);
    validate_channel(&case.vibration_channel, "vibration channel", &mut errors);

    if case.pressure_channel.channel_id == case.vibration_channel.channel_id {
        errors.push("pressure and vibration channel IDs must be distinct".to_string());
    }
    if case.pressure_channel.instrument_id == case.vibration_channel.instrument_id {
        errors.push("pressure and vibration instrument IDs must be distinct in v0.1".to_string());
    }
    if case.pressure_channel.timebase_id != case.vibration_channel.timebase_id {
        errors.push("pressure and vibration channels do not share the same versioned timebase ID".to_string());
    }
    if (case.pressure_channel.sample_rate_hz - case.vibration_channel.sample_rate_hz).abs() > 1.0e-9 {
        errors.push("pressure and vibration sample rates differ".to_string());
    }
    if case.pressure_channel.sample_count != case.vibration_channel.sample_count {
        errors.push("pressure and vibration sample counts differ".to_string());
    }
    if (case.pressure_channel.window_start_s - case.vibration_channel.window_start_s).abs() > 1.0e-9
        || (case.pressure_channel.window_end_s - case.vibration_channel.window_end_s).abs() > 1.0e-9
    {
        errors.push("pressure and vibration measurement windows differ".to_string());
    }
    if !case.alignment_offset_applied_in_estimator {
        errors.push("declared sensor alignment offset has not been applied in the cross-spectral estimator".to_string());
    }
    if case.cross_spectrum_convention != "pressure_times_conjugate_acceleration" {
        errors.push("cross_spectrum_convention must be pressure_times_conjugate_acceleration".to_string());
    }
    if !finite_nonnegative(case.residual_alignment_uncertainty_s)
        || !finite_positive(case.maximum_phase_uncertainty_rad)
        || case.maximum_phase_uncertainty_rad > std::f64::consts::PI
    {
        errors.push("alignment uncertainty and phase uncertainty limit must be finite and within declared bounds".to_string());
    }
    if !case.predeclared_coherence_interpretation_threshold.is_finite()
        || case.predeclared_coherence_interpretation_threshold < 0.0
        || case.predeclared_coherence_interpretation_threshold > 1.0
    {
        errors.push("predeclared coherence interpretation threshold must be between 0 and 1".to_string());
    }

    let estimator = &case.estimator;
    if estimator.estimator_id.trim().is_empty()
        || estimator.estimator_version.trim().is_empty()
        || estimator.output_scaling != "density"
        || estimator.sidedness != "one_sided"
        || estimator.averaging_method != "arithmetic_mean_of_modified_periodograms"
    {
        errors.push("estimator ID/version, density scaling, one-sided spectra and arithmetic mean of modified periodograms are required".to_string());
    }
    if estimator.segment_length_samples < 8
        || estimator.segment_length_samples > case.pressure_channel.sample_count
        || estimator.number_of_averages < 1
        || !estimator.overlap_fraction.is_finite()
        || estimator.overlap_fraction < 0.0
        || estimator.overlap_fraction >= 1.0
        || !finite_positive(estimator.frequency_resolution_hz)
    {
        errors.push("invalid Welch segment, overlap, average count or frequency resolution".to_string());
    }
    if finite_positive(case.pressure_channel.sample_rate_hz)
        && finite_positive(estimator.frequency_resolution_hz)
        && estimator.segment_length_samples > 0
        && estimator.segment_length_samples <= case.pressure_channel.sample_count
    {
        let expected_resolution = case.pressure_channel.sample_rate_hz / estimator.segment_length_samples as f64;
        let relative_error = (estimator.frequency_resolution_hz - expected_resolution).abs() / expected_resolution;
        if relative_error > 1.0e-6 {
            errors.push("declared frequency resolution does not match sample_rate / segment_length".to_string());
        }
        let step = (estimator.segment_length_samples as f64 * (1.0 - estimator.overlap_fraction)).floor();
        if step >= 1.0 {
            let available = 1 + ((case.pressure_channel.sample_count - estimator.segment_length_samples) as f64 / step).floor() as u64;
            if estimator.number_of_averages as u64 > available {
                errors.push("declared Welch average count exceeds the available non-wrapping segments".to_string());
            }
        }
    }

    if case.bands.is_empty() {
        errors.push("at least one frequency-resolved cross-spectral estimate is required".to_string());
    }
    let mut band_ids = BTreeSet::new();
    let mut previous_frequency = 0.0;
    for band in &case.bands {
        if band.band_id.trim().is_empty() || !band_ids.insert(band.band_id.clone()) {
            errors.push(format!("empty or duplicate band_id {}", band.band_id));
        }
        if !finite_positive(band.frequency_hz)
            || band.frequency_hz <= previous_frequency
            || (finite_positive(case.pressure_channel.sample_rate_hz)
                && band.frequency_hz >= case.pressure_channel.sample_rate_hz / 2.0)
        {
            errors.push(format!("band {} frequency must be strictly increasing, positive and below Nyquist", band.band_id));
        } else if finite_positive(estimator.frequency_resolution_hz) {
            let grid_index = band.frequency_hz / estimator.frequency_resolution_hz;
            if (grid_index - grid_index.round()).abs() > 1.0e-6 {
                errors.push(format!("band {} frequency is not on the declared FFT frequency grid", band.band_id));
            }
        }
        previous_frequency = band.frequency_hz;
        if band.pressure_psd_unit != PRESSURE_PSD_UNIT
            || band.acceleration_psd_unit != ACCELERATION_PSD_UNIT
            || band.cross_psd_unit != CROSS_PSD_UNIT
        {
            errors.push(format!("band {} has unsupported PSD/cross-PSD units", band.band_id));
        }
        if !finite_positive(band.pressure_psd)
            || !finite_positive(band.acceleration_psd)
            || !band.cross_psd_real.is_finite()
            || !band.cross_psd_imag.is_finite()
            || !finite_positive(band.pressure_psd_expanded_uncertainty)
            || band.pressure_psd_expanded_uncertainty >= band.pressure_psd
            || !finite_positive(band.acceleration_psd_expanded_uncertainty)
            || band.acceleration_psd_expanded_uncertainty >= band.acceleration_psd
            || !finite_positive(band.cross_psd_real_expanded_uncertainty)
            || !finite_positive(band.cross_psd_imag_expanded_uncertainty)
            || !band.uncertainty_coverage_level.is_finite()
            || band.uncertainty_coverage_level <= 0.0
            || band.uncertainty_coverage_level >= 1.0
            || !finite_positive(band.cross_spectrum_phase_expanded_uncertainty_rad)
        {
            errors.push(format!("band {} has invalid spectral estimates or marginal uncertainty", band.band_id));
        }
        if band.cross_spectrum_phase_expanded_uncertainty_rad > std::f64::consts::PI {
            errors.push(format!("band {} cross-spectrum phase uncertainty exceeds pi radians", band.band_id));
        }
    }
    errors
}

fn calculate_band(
    case: &Case,
    input: &BandEstimate,
) -> Result<ComputedBand, String> {
    let p = input.pressure_psd;
    let a = input.acceleration_psd;
    let re = input.cross_psd_real;
    let im = input.cross_psd_imag;
    let cross_magnitude_squared = re.mul_add(re, im * im);
    let auto_product = p * a;
    if !cross_magnitude_squared.is_finite() || !auto_product.is_finite() || auto_product <= 0.0 {
        return Err(format!("band {} has non-finite spectral matrix entries", input.band_id));
    }
    let raw_coherence = cross_magnitude_squared / auto_product;
    if !raw_coherence.is_finite() || raw_coherence > 1.0 + PSD_MATRIX_RELATIVE_TOLERANCE {
        return Err(format!("band {} violates the cross-spectral positive-semidefinite constraint", input.band_id));
    }
    let coherence = raw_coherence.min(1.0);
    let h1_real = re / a;
    let h1_imag = im / a;
    let h1_magnitude = h1_real.hypot(h1_imag);
    let phase = im.atan2(re);
    let coherent_psd = cross_magnitude_squared / a;
    let mut residual_psd = p - coherent_psd;
    let residual_tolerance = p * PSD_MATRIX_RELATIVE_TOLERANCE;
    if residual_psd < -residual_tolerance {
        return Err(format!("band {} coherent projection exceeds pressure PSD beyond numerical tolerance", input.band_id));
    }
    if residual_psd < 0.0 {
        residual_psd = 0.0;
    }

    let cross_uncertainty_radius = input.cross_psd_real_expanded_uncertainty
        .hypot(input.cross_psd_imag_expanded_uncertainty);
    let cross_magnitude = re.hypot(im);
    let cross_lower = (cross_magnitude - cross_uncertainty_radius).max(0.0);
    let cross_upper = cross_magnitude + cross_uncertainty_radius;
    let p_lower = p - input.pressure_psd_expanded_uncertainty;
    let p_upper = p + input.pressure_psd_expanded_uncertainty;
    let a_lower = a - input.acceleration_psd_expanded_uncertainty;
    let a_upper = a + input.acceleration_psd_expanded_uncertainty;
    if !cross_uncertainty_radius.is_finite()
        || !cross_upper.is_finite()
        || !p_upper.is_finite()
        || !a_upper.is_finite()
    {
        return Err(format!("band {} uncertainty arithmetic overflowed", input.band_id));
    }
    let coherent_lower = cross_lower.powi(2) / a_upper;
    let coherent_upper = cross_upper.powi(2) / a_lower;
    let residual_lower = p_lower - coherent_upper;
    let residual_upper = p_upper - coherent_lower;
    let coherence_lower = (cross_lower.powi(2) / (p_upper * a_upper)).clamp(0.0, 1.0);
    let coherence_upper = (cross_upper.powi(2) / (p_lower * a_lower)).clamp(0.0, 1.0);

    let uncertainty_unresolved = residual_lower < 0.0
        || !coherent_lower.is_finite()
        || !coherent_upper.is_finite()
        || !residual_upper.is_finite();
    let (coherent_lower_bound, coherent_upper_bound, residual_lower_bound, residual_upper_bound) =
        if uncertainty_unresolved {
            (None, None, None, None)
        } else {
            (
                Some(coherent_lower),
                Some(coherent_upper),
                Some(residual_lower),
                Some(residual_upper),
            )
        };

    let phase_uncertainty_bound = input.cross_spectrum_phase_expanded_uncertainty_rad
        + 2.0 * std::f64::consts::PI * input.frequency_hz * case.residual_alignment_uncertainty_s;
    let phase_resolved = phase_uncertainty_bound <= case.maximum_phase_uncertainty_rad;
    let phase_status = if phase_resolved { "phase_resolved_within_declared_limit" } else { "phase_unresolved" };
    let uncertainty_status = if uncertainty_unresolved { "coherent_residual_interval_unresolved" } else { "marginal_interval_envelope_available" };

    let report = BandReport {
        band_id: input.band_id.clone(),
        frequency_hz: input.frequency_hz,
        pressure_psd_pa2_per_hz: p,
        acceleration_psd_m2_per_s4_per_hz: a,
        cross_psd_real: re,
        cross_psd_imag: im,
        coherence_magnitude_squared: coherence,
        coherence_lower_bound: Some(coherence_lower),
        coherence_upper_bound: Some(coherence_upper),
        coherence_threshold_met: coherence >= case.predeclared_coherence_interpretation_threshold,
        h1_real_pa_per_m_s2: h1_real,
        h1_imag_pa_per_m_s2: h1_imag,
        h1_magnitude_pa_per_m_s2: h1_magnitude,
        h1_phase_rad: if phase_resolved { Some(phase) } else { None },
        phase_uncertainty_bound_rad: phase_uncertainty_bound,
        phase_status,
        vibration_reference_coherent_pressure_psd_pa2_per_hz: coherent_psd,
        vibration_reference_coherent_pressure_psd_lower_bound: coherent_lower_bound,
        vibration_reference_coherent_pressure_psd_upper_bound: coherent_upper_bound,
        non_coherent_residual_pressure_psd_pa2_per_hz: residual_psd,
        non_coherent_residual_pressure_psd_lower_bound: residual_lower_bound,
        non_coherent_residual_pressure_psd_upper_bound: residual_upper_bound,
        pressure_psd_level_db_re_1upa2_per_hz: db_re_reference(p, PRESSURE_REFERENCE_PA.powi(2))
            .expect("validated positive pressure PSD"),
        coherent_pressure_psd_level_db_re_1upa2_per_hz: db_re_reference(coherent_psd, PRESSURE_REFERENCE_PA.powi(2)),
        residual_pressure_psd_level_db_re_1upa2_per_hz: db_re_reference(residual_psd, PRESSURE_REFERENCE_PA.powi(2)),
        uncertainty_status,
        marginal_uncertainty_coverage_level: input.uncertainty_coverage_level,
        uncertainty_interpretation: "conservative_marginal_interval_envelopes_only_no_joint_coverage_or_independence_assumption",
        interpretation: "linear_spectral_component_coherent_with_the_declared_accelerometer_reference_not_unique_physical_source_attribution",
    };
    Ok(ComputedBand {
        report,
        uncertainty_unresolved,
    })
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut computed = Vec::new();
    if validation_errors.is_empty() {
        for band in &case.bands {
            match calculate_band(case, band) {
                Ok(result) => computed.push(result),
                Err(reason) => reasons.push(reason),
            }
        }
    }
    let band_reports: Vec<BandReport> = computed.iter().map(|item| item.report.clone()).collect();
    let uncertainty_unresolved = computed.iter().any(|item| item.uncertainty_unresolved);
    let has_invalid_spectral_matrix = !validation_errors.is_empty()
        || computed.len() != case.bands.len()
        || reasons.iter().any(|reason| {
            reason.contains("positive-semidefinite constraint")
                || reason.contains("exceeds pressure PSD")
                || reason.contains("non-finite spectral matrix entries")
                || reason.contains("uncertainty arithmetic overflowed")
        });
    let phase_unresolved = band_reports.iter().any(|band| band.phase_status == "phase_unresolved");

    let disposition = if !validation_errors.is_empty() || has_invalid_spectral_matrix {
        "INVALID"
    } else if case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
    {
        "INCOMPLETE"
    } else if uncertainty_unresolved {
        "UNCERTAINTY_UNRESOLVED"
    } else if phase_unresolved {
        "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED"
    } else if band_reports.is_empty() {
        "INCOMPLETE"
    } else {
        "COHERENCE_ESTIMATE_ONLY"
    };

    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot establish physical source coherence".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source data rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source evidence has not been independently reviewed".to_string());
    }
    if uncertainty_unresolved {
        reasons.push("one or more bands do not have non-negative residual pressure PSD across the declared conservative uncertainty envelope".to_string());
    }
    if phase_unresolved {
        reasons.push("one or more transfer-function phase estimates exceed the declared time-alignment phase uncertainty limit".to_string());
    }

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        source_dataset_id: case.source.dataset_id.clone(),
        source_data_sha256: case.source.raw_data_sha256.clone(),
        evidence_kind: match case.evidence_kind {
            EvidenceKind::Synthetic => "synthetic",
            EvidenceKind::Measured => "measured",
        },
        vessel_id: case.vessel_id.clone(),
        propeller_revision: case.propeller_revision.clone(),
        operating_condition_id: case.operating_condition_id.clone(),
        operating_point_receipt_sha256: case.operating_point_receipt_sha256.clone(),
        service_fingerprint_sha256: case.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: case.environment_fingerprint_sha256.clone(),
        timebase_binding_receipt_sha256: case.timebase_binding_receipt_sha256.clone(),
        synchronization_receipt_sha256: case.synchronization_receipt_sha256.clone(),
        pressure_instrument_id: case.pressure_channel.instrument_id.clone(),
        pressure_channel_id: case.pressure_channel.channel_id.clone(),
        vibration_instrument_id: case.vibration_channel.instrument_id.clone(),
        vibration_channel_id: case.vibration_channel.channel_id.clone(),
        vibration_reference_location: case.vibration_channel.physical_location.clone(),
        vibration_reference_orientation: case.vibration_channel.orientation.clone(),
        pressure_calibration_certificate_sha256: case.pressure_channel.calibration_certificate_sha256.clone(),
        vibration_calibration_certificate_sha256: case.vibration_channel.calibration_certificate_sha256.clone(),
        pressure_raw_trace_sha256: case.pressure_channel.raw_trace_sha256.clone(),
        vibration_raw_trace_sha256: case.vibration_channel.raw_trace_sha256.clone(),
        shared_timebase_id: case.pressure_channel.timebase_id.clone(),
        sample_rate_hz: case.pressure_channel.sample_rate_hz,
        sample_count: case.pressure_channel.sample_count,
        pressure_window_start_s: case.pressure_channel.window_start_s,
        pressure_window_end_s: case.pressure_channel.window_end_s,
        estimator_id: case.estimator.estimator_id.clone(),
        estimator_version: case.estimator.estimator_version.clone(),
        cross_spectrum_convention: case.cross_spectrum_convention.clone(),
        welch_window: match case.estimator.window {
            WelchWindow::Hann => "hann",
            WelchWindow::Hamming => "hamming",
            WelchWindow::Blackman => "blackman",
        },
        number_of_averages: case.estimator.number_of_averages,
        overlap_fraction: case.estimator.overlap_fraction,
        output_scaling: case.estimator.output_scaling.clone(),
        sidedness: case.estimator.sidedness.clone(),
        averaging_method: case.estimator.averaging_method.clone(),
        frequency_resolution_hz: case.estimator.frequency_resolution_hz,
        predeclared_coherence_interpretation_threshold: case.predeclared_coherence_interpretation_threshold,
        disposition,
        claim_ceiling: "single_reference_linear_spectral_coherence_and_residual_only_no_unique_source_decomposition_causal_effect_noise_reduction_ecological_or_regulatory_qualification",
        band_reports,
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-vibration-coherence-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-vibration-coherence-audit <case.json>");
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
        "COHERENCE_ESTIMATE_ONLY" | "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED" => {}
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

    fn band(
        id: &str,
        frequency_hz: f64,
        p: f64,
        a: f64,
        re: f64,
        im: f64,
    ) -> BandEstimate {
        BandEstimate {
            band_id: id.to_string(),
            frequency_hz,
            pressure_psd: p,
            pressure_psd_expanded_uncertainty: 0.05,
            pressure_psd_unit: PRESSURE_PSD_UNIT.to_string(),
            acceleration_psd: a,
            acceleration_psd_expanded_uncertainty: 0.05,
            acceleration_psd_unit: ACCELERATION_PSD_UNIT.to_string(),
            cross_psd_real: re,
            cross_psd_real_expanded_uncertainty: 0.01,
            cross_psd_imag: im,
            cross_psd_imag_expanded_uncertainty: 0.01,
            cross_psd_unit: CROSS_PSD_UNIT.to_string(),
            uncertainty_coverage_level: 0.95,
            cross_spectrum_phase_expanded_uncertainty_rad: 0.01,
        }
    }

    fn case() -> Case {
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-vibration-coherence-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-csd".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('1'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            vessel_id: "vessel-001".to_string(),
            propeller_revision: "prop-001-r1".to_string(),
            operating_condition_id: "rpm-1200-load-60".to_string(),
            operating_point_receipt_sha256: digest('2'),
            service_fingerprint_sha256: digest('3'),
            environment_fingerprint_sha256: digest('4'),
            timebase_binding_receipt_sha256: digest('5'),
            synchronization_receipt_sha256: digest('6'),
            alignment_offset_applied_in_estimator: true,
            residual_alignment_uncertainty_s: 1.0e-6,
            maximum_phase_uncertainty_rad: 0.5,
            predeclared_coherence_interpretation_threshold: 0.7,
            cross_spectrum_convention: "pressure_times_conjugate_acceleration".to_string(),
            pressure_channel: ChannelEvidence {
                instrument_id: "hydrophone-H1".to_string(),
                channel_id: "pressure-channel-1".to_string(),
                physical_location: "tunnel-sidewall-station-1".to_string(),
                orientation: "omnidirectional".to_string(),
                calibration_certificate_sha256: digest('7'),
                raw_trace_sha256: digest('8'),
                timebase_id: "shared-clock-v1".to_string(),
                sample_rate_hz: 20_000.0,
                sample_count: 20_000,
                window_start_s: 0.0,
                window_end_s: 1.0,
            },
            vibration_channel: ChannelEvidence {
                instrument_id: "accelerometer-A1".to_string(),
                channel_id: "acceleration-channel-1".to_string(),
                physical_location: "shaft-bearing-housing".to_string(),
                orientation: "radial-x".to_string(),
                calibration_certificate_sha256: digest('9'),
                raw_trace_sha256: digest('a'),
                timebase_id: "shared-clock-v1".to_string(),
                sample_rate_hz: 20_000.0,
                sample_count: 20_000,
                window_start_s: 0.0,
                window_end_s: 1.0,
            },
            estimator: WelchEstimator {
                estimator_id: "welch-csd-rust-v1".to_string(),
                estimator_version: "1.0.0".to_string(),
                window: WelchWindow::Hann,
                segment_length_samples: 2_000,
                overlap_fraction: 0.5,
                number_of_averages: 19,
                frequency_resolution_hz: 10.0,
                output_scaling: "density".to_string(),
                sidedness: "one_sided".to_string(),
                averaging_method: "arithmetic_mean_of_modified_periodograms".to_string(),
            },
            bands: vec![
                band("band-100", 100.0, 4.0, 9.0, 3.0, 0.0),
                band("band-200", 200.0, 4.0, 9.0, 0.0, 3.0),
            ],
        }
    }

    #[test]
    fn report_receipt_preserves_exact_pressure_window_boundaries() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.pressure_channel.window_start_s = 12.5;
        input.pressure_channel.window_end_s = 13.5;
        input.vibration_channel.window_start_s = 12.5;
        input.vibration_channel.window_end_s = 13.5;
        let report = audit(&input);
        assert_eq!(report.pressure_window_start_s, 12.5);
        assert_eq!(report.pressure_window_end_s, 13.5);
        assert_eq!(report.pressure_raw_trace_sha256, input.pressure_channel.raw_trace_sha256);
    }

    #[test]
    fn cross_spectrum_sign_convention_is_explicit_and_enforced() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.cross_spectrum_convention = "acceleration_times_conjugate_pressure".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn welch_psd_scaling_and_sidedness_are_explicit() {
        let mut input = case();
        input.estimator.output_scaling = "spectrum".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
        let mut input = case();
        input.estimator.sidedness = "two_sided".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn committed_schema_and_fixture_preserve_the_declared_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-vibration-coherence-v0.1.0.schema.json"
        )).expect("schema JSON parses");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/fixtures/synthetic-marine-vibration-coherence-v0.1.0.json"
        )).expect("fixture JSON parses");
        assert_eq!(schema["$schema"], "https://json-schema.org/draft/2020-12/schema");
        for required in [
            "cross_spectrum_convention",
            "timebase_binding_receipt_sha256",
            "synchronization_receipt_sha256",
            "pressure_channel",
            "vibration_channel",
            "estimator",
            "bands",
        ] {
            assert!(schema["required"].as_array().unwrap().iter().any(|value| value == required));
        }
        assert_eq!(fixture["cross_spectrum_convention"], "pressure_times_conjugate_acceleration");
        assert_eq!(fixture["pressure_channel"]["timebase_id"], fixture["vibration_channel"]["timebase_id"]);
        assert_eq!(fixture["evidence_kind"], "synthetic");
    }

    #[test]
    fn a_failed_band_calculation_cannot_be_hidden_by_another_valid_band() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.bands[0].cross_psd_real = 7.0;
        let report = audit(&input);
        assert_eq!(report.disposition, "INVALID");
        assert_eq!(report.band_reports.len(), 1);
        assert!(report.reasons.iter().any(|reason| reason.contains("positive-semidefinite constraint")));
    }

    #[test]
    fn partially_coherent_known_answer() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.bands.truncate(1);
        let report = audit(&input);
        let result = &report.band_reports[0];
        assert!((result.coherence_magnitude_squared - 0.25).abs() < 1.0e-12);
        assert!((result.h1_real_pa_per_m_s2 - 1.0 / 3.0).abs() < 1.0e-12);
        assert!(result.h1_imag_pa_per_m_s2.abs() < 1.0e-12);
        assert!((result.vibration_reference_coherent_pressure_psd_pa2_per_hz - 1.0).abs() < 1.0e-12);
        assert!((result.non_coherent_residual_pressure_psd_pa2_per_hz - 3.0).abs() < 1.0e-12);
        assert_eq!(report.disposition, "COHERENCE_ESTIMATE_ONLY");
    }

    #[test]
    fn fully_coherent_known_answer_has_zero_residual() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.bands[0] = band("band-100", 100.0, 4.0, 9.0, 6.0, 0.0);
        let report = audit(&input);
        let result = &report.band_reports[0];
        assert!((result.coherence_magnitude_squared - 1.0).abs() < 1.0e-12);
        assert!((result.vibration_reference_coherent_pressure_psd_pa2_per_hz - 4.0).abs() < 1.0e-12);
        assert!(result.non_coherent_residual_pressure_psd_pa2_per_hz.abs() < 1.0e-12);
        assert_eq!(result.residual_pressure_psd_level_db_re_1upa2_per_hz, None);
    }

    #[test]
    fn quadrature_cross_spectrum_produces_complex_h1_phase() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.bands.truncate(1);
        input.bands[0] = band("band-200", 200.0, 4.0, 9.0, 0.0, 3.0);
        let result = &audit(&input).band_reports[0];
        assert!((result.coherence_magnitude_squared - 0.25).abs() < 1.0e-12);
        assert!((result.h1_phase_rad.unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }

    #[test]
    fn zero_cross_spectrum_is_fully_noncoherent() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.bands.truncate(1);
        input.bands[0] = band("band-100", 100.0, 4.0, 9.0, 0.0, 0.0);
        let result = &audit(&input).band_reports[0];
        assert_eq!(result.coherence_magnitude_squared, 0.0);
        assert_eq!(result.vibration_reference_coherent_pressure_psd_pa2_per_hz, 0.0);
        assert_eq!(result.non_coherent_residual_pressure_psd_pa2_per_hz, 4.0);
    }

    #[test]
    fn cauchy_schwarz_violation_is_invalid() {
        let mut input = case();
        input.bands.truncate(1);
        input.bands[0].cross_psd_real = 7.0;
        let report = audit(&input);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.reasons.iter().any(|reason| reason.contains("positive-semidefinite")));
    }

    #[test]
    fn phase_uncertainty_can_remove_phase_without_discarding_magnitude() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.residual_alignment_uncertainty_s = 1.0e-3;
        let report = audit(&input);
        assert_eq!(report.disposition, "MAGNITUDE_COHERENCE_ONLY_PHASE_UNRESOLVED");
        assert!(report.band_reports.iter().all(|band| band.h1_phase_rad.is_none()));
        assert!(report.band_reports.iter().all(|band| band.coherence_magnitude_squared > 0.0));
    }

    #[test]
    fn different_timebase_ids_are_invalid() {
        let mut input = case();
        input.vibration_channel.timebase_id = "separate-clock".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn sample_rate_or_window_mismatch_is_invalid() {
        let mut input = case();
        input.vibration_channel.sample_rate_hz = 10_000.0;
        assert_eq!(audit(&input).disposition, "INVALID");
        let mut input = case();
        input.vibration_channel.window_end_s = 0.9;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn synthetic_spectral_matrix_never_qualifies_measured_evidence() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("synthetic evidence")));
    }

    #[test]
    fn estimator_frequency_resolution_must_match_segment_length() {
        let mut input = case();
        input.estimator.frequency_resolution_hz = 5.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn declared_coherence_threshold_is_interpretive_not_universal() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.predeclared_coherence_interpretation_threshold = 0.3;
        let result = audit(&input);
        assert!(result.band_reports.iter().all(|band| !band.coherence_threshold_met));
        input.predeclared_coherence_interpretation_threshold = 0.2;
        let result = audit(&input);
        assert!(result.band_reports.iter().all(|band| band.coherence_threshold_met));
    }

    #[test]
    fn impossible_negative_residual_is_not_clamped_to_pass() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        input.bands[0].pressure_psd_expanded_uncertainty = 3.9;
        input.bands[0].acceleration_psd_expanded_uncertainty = 8.9;
        input.bands[0].cross_psd_real_expanded_uncertainty = 6.0;
        input.bands[0].cross_psd_imag_expanded_uncertainty = 6.0;
        let report = audit(&input);
        assert_eq!(report.disposition, "UNCERTAINTY_UNRESOLVED");
    }

    #[test]
    fn lower_frequency_phase_is_eligible_when_bound_is_below_limit() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::Measured;
        let report = audit(&input);
        assert!(report.band_reports.iter().all(|band| band.phase_status == "phase_resolved_within_declared_limit"));
    }
}
