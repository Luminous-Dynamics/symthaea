use serde::{Deserialize, Serialize};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const PRESSURE_PSD_REFERENCE_UPA2_PER_HZ: f64 = 1.0;
const PSD_MATRIX_RELATIVE_TOLERANCE: f64 = 1.0e-12;

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
enum ReferenceSignalKind {
    ShaftAcceleration,
    BearingAcceleration,
    GearboxAcceleration,
    PrimeMoverAcceleration,
    StructuralVibration,
    ShaftTorsion,
    HydrophonePressure,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    cross_spectral_convention: String,
    reference_channel: Channel,
    hydrophone_channel: Channel,
    acquisition: AcquisitionEvidence,
    operating_point_receipt_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    minimum_average_count: usize,
    minimum_coherence: f64,
    bins: Vec<SpectralBin>,
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
struct Channel {
    instrument_id: String,
    signal_kind: ReferenceSignalKind,
    unit: String,
    location_id: String,
    orientation: String,
    calibration_certificate_sha256: String,
    raw_trace_sha256: String,
    sample_rate_hz: f64,
    total_sample_count: u64,
    segment_length_samples: u64,
    fft_length_samples: u64,
    overlap_fraction: f64,
    window_function: String,
    average_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcquisitionEvidence {
    timebase_receipt_sha256: String,
    alignment_receipt_sha256: String,
    alignment_offset_s: f64,
    alignment_expanded_uncertainty_s: f64,
    maximum_alignment_uncertainty_s: f64,
    operating_condition_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectralBin {
    bin_id: String,
    fft_bin_index: u64,
    center_frequency_hz: f64,
    reference_auto_psd: f64,
    hydrophone_auto_psd_upa2_per_hz: f64,
    cross_spectral_real: f64,
    cross_spectral_imag: f64,
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
    cross_spectral_convention: &'static str,
    frequency_resolution_hz: f64,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_receipt_sha256: String,
    alignment_receipt_sha256: String,
    alignment_offset_s: f64,
    alignment_expanded_uncertainty_s: f64,
    reference_channel: ChannelReceipt,
    hydrophone_channel: ChannelReceipt,
    minimum_average_count: usize,
    minimum_coherence: f64,
    disposition: &'static str,
    claim_ceiling: &'static str,
    bins: Vec<BinReport>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ChannelReceipt {
    instrument_id: String,
    signal_kind: &'static str,
    unit: String,
    location_id: String,
    orientation: String,
    calibration_certificate_sha256: String,
    raw_trace_sha256: String,
    sample_rate_hz: f64,
    total_sample_count: u64,
    segment_length_samples: u64,
    fft_length_samples: u64,
    overlap_fraction: f64,
    window_function: String,
    average_count: usize,
}

#[derive(Debug, Serialize)]
struct BinReport {
    bin_id: String,
    fft_bin_index: u64,
    center_frequency_hz: f64,
    reference_auto_psd: f64,
    hydrophone_auto_psd_upa2_per_hz: f64,
    cross_spectral_real: f64,
    cross_spectral_imag: f64,
    cross_spectral_phase_rad: Option<f64>,
    magnitude_squared_coherence: f64,
    coherent_hydrophone_psd_upa2_per_hz: f64,
    residual_hydrophone_psd_upa2_per_hz: f64,
    coherent_fraction: f64,
    residual_fraction: f64,
    total_level_db_re_1upa2_per_hz: f64,
    coherent_level_db_re_1upa2_per_hz: Option<f64>,
    residual_level_db_re_1upa2_per_hz: Option<f64>,
    coherence_threshold: f64,
    disposition: &'static str,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn signal_kind_name(value: &ReferenceSignalKind) -> &'static str {
    match value {
        ReferenceSignalKind::ShaftAcceleration => "shaft_acceleration",
        ReferenceSignalKind::BearingAcceleration => "bearing_acceleration",
        ReferenceSignalKind::GearboxAcceleration => "gearbox_acceleration",
        ReferenceSignalKind::PrimeMoverAcceleration => "prime_mover_acceleration",
        ReferenceSignalKind::StructuralVibration => "structural_vibration",
        ReferenceSignalKind::ShaftTorsion => "shaft_torsion",
        ReferenceSignalKind::HydrophonePressure => "hydrophone_pressure",
    }
}

fn channel_receipt(channel: &Channel) -> ChannelReceipt {
    ChannelReceipt {
        instrument_id: channel.instrument_id.clone(),
        signal_kind: signal_kind_name(&channel.signal_kind),
        unit: channel.unit.clone(),
        location_id: channel.location_id.clone(),
        orientation: channel.orientation.clone(),
        calibration_certificate_sha256: channel.calibration_certificate_sha256.clone(),
        raw_trace_sha256: channel.raw_trace_sha256.clone(),
        sample_rate_hz: channel.sample_rate_hz,
        total_sample_count: channel.total_sample_count,
        segment_length_samples: channel.segment_length_samples,
        fft_length_samples: channel.fft_length_samples,
        overlap_fraction: channel.overlap_fraction,
        window_function: channel.window_function.clone(),
        average_count: channel.average_count,
    }
}

fn validate_channel(channel: &Channel, label: &str, errors: &mut Vec<String>) {
    if channel.instrument_id.trim().is_empty()
        || channel.unit.trim().is_empty()
        || channel.location_id.trim().is_empty()
        || channel.orientation.trim().is_empty()
        || channel.window_function.trim().is_empty()
    {
        errors.push(format!("{label}: instrument, unit, location, orientation and window function are required"));
    }
    if !is_sha256(&channel.calibration_certificate_sha256)
        || !is_sha256(&channel.raw_trace_sha256)
    {
        errors.push(format!("{label}: calibration and raw-trace digests must be SHA-256 hex"));
    }
    if !finite_positive(channel.sample_rate_hz)
        || channel.total_sample_count == 0
        || channel.segment_length_samples < 2
        || channel.fft_length_samples < channel.segment_length_samples
        || channel.segment_length_samples > channel.total_sample_count
        || !channel.overlap_fraction.is_finite()
        || channel.overlap_fraction < 0.0
        || channel.overlap_fraction >= 1.0
        || channel.average_count < 2
    {
        errors.push(format!("{label}: invalid sample rate, sample counts, FFT, overlap or average count"));
    }
    if channel.segment_length_samples > 0 && channel.total_sample_count >= channel.segment_length_samples {
        let step = (channel.segment_length_samples as f64 * (1.0 - channel.overlap_fraction)).floor();
        if step < 1.0 {
            errors.push(format!("{label}: overlap produces a zero-step segment schedule"));
        }
        let maximum_segments = if step >= 1.0 {
            1 + ((channel.total_sample_count - channel.segment_length_samples) as f64 / step).floor() as usize
        } else {
            0
        };
        if channel.average_count > maximum_segments {
            errors.push(format!("{label}: declared average_count exceeds the segment schedule"));
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
        || case.acquisition.operating_condition_id.trim().is_empty()
    {
        errors.push("case, dataset, citation and operating-condition IDs are required".to_string());
    }
    for (label, digest) in [
        ("source raw data", &case.source.raw_data_sha256),
        ("operating-point receipt", &case.operating_point_receipt_sha256),
        ("service fingerprint", &case.service_fingerprint_sha256),
        ("environment fingerprint", &case.environment_fingerprint_sha256),
        ("timebase receipt", &case.acquisition.timebase_receipt_sha256),
        ("alignment receipt", &case.acquisition.alignment_receipt_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label} digest must be 64 hexadecimal characters"));
        }
    }
    if !case.acquisition.alignment_offset_s.is_finite()
        || !finite_positive(case.acquisition.alignment_expanded_uncertainty_s)
        || !finite_positive(case.acquisition.maximum_alignment_uncertainty_s)
    {
        errors.push("channel alignment must have finite offset and positive finite uncertainty/tolerance".to_string());
    }
    if case.cross_spectral_convention != "reference_times_conjugate_hydrophone" {
        errors.push("cross_spectral_convention must be reference_times_conjugate_hydrophone".to_string());
    }
    if case.minimum_average_count < 2
        || !case.minimum_coherence.is_finite()
        || case.minimum_coherence <= 0.0
        || case.minimum_coherence > 1.0
    {
        errors.push("minimum_average_count must be >=2 and minimum_coherence must be in (0,1]".to_string());
    }
    validate_channel(&case.reference_channel, "reference channel", &mut errors);
    validate_channel(&case.hydrophone_channel, "hydrophone channel", &mut errors);

    let reference = &case.reference_channel;
    let hydrophone = &case.hydrophone_channel;
    if reference.instrument_id == hydrophone.instrument_id {
        errors.push("reference and hydrophone instrument IDs must be distinct".to_string());
    }
    if reference.signal_kind == ReferenceSignalKind::HydrophonePressure {
        errors.push("reference channel must be a vibration/shaft reference, not another hydrophone pressure channel".to_string());
    }
    if hydrophone.signal_kind != ReferenceSignalKind::HydrophonePressure {
        errors.push("hydrophone channel signal_kind must be hydrophone_pressure".to_string());
    }
    if reference.sample_rate_hz != hydrophone.sample_rate_hz
        || reference.total_sample_count != hydrophone.total_sample_count
        || reference.segment_length_samples != hydrophone.segment_length_samples
        || reference.fft_length_samples != hydrophone.fft_length_samples
        || reference.overlap_fraction != hydrophone.overlap_fraction
        || reference.window_function != hydrophone.window_function
        || reference.average_count != hydrophone.average_count
    {
        errors.push("reference/hydrophone sample rate, duration, segment, FFT, overlap, window and average configuration must match".to_string());
    }
    if reference.average_count < case.minimum_average_count
        || hydrophone.average_count < case.minimum_average_count
    {
        errors.push("reference/hydrophone average count is below the predeclared minimum".to_string());
    }

    if case.bins.is_empty() {
        errors.push("at least one frequency bin is required".to_string());
    }
    let mut bin_ids = std::collections::BTreeSet::new();
    let mut previous_frequency: Option<f64> = None;
    for bin in &case.bins {
        if bin.bin_id.trim().is_empty() || !bin_ids.insert(bin.bin_id.clone()) {
            errors.push("bin IDs must be non-empty and unique".to_string());
        }
        if bin.fft_bin_index == 0
            || bin.fft_bin_index > reference.fft_length_samples / 2
            || !finite_positive(bin.center_frequency_hz)
            || !finite_positive(bin.reference_auto_psd)
            || !finite_positive(bin.hydrophone_auto_psd_upa2_per_hz)
            || !bin.cross_spectral_real.is_finite()
            || !bin.cross_spectral_imag.is_finite()
        {
            errors.push(format!("bin {} has invalid frequency or spectral matrix values", bin.bin_id));
        }
        if let Some(previous) = previous_frequency
            && bin.center_frequency_hz <= previous
        {
            errors.push(format!("bin {} frequency is not strictly increasing", bin.bin_id));
        }
        previous_frequency = Some(bin.center_frequency_hz);
        if bin.center_frequency_hz > reference.sample_rate_hz / 2.0 {
            errors.push(format!("bin {} exceeds the Nyquist frequency", bin.bin_id));
        }
        let expected_frequency = bin.fft_bin_index as f64 * reference.sample_rate_hz
            / reference.fft_length_samples as f64;
        let frequency_tolerance = (reference.sample_rate_hz / reference.fft_length_samples as f64) * 1.0e-9;
        if (bin.center_frequency_hz - expected_frequency).abs() > frequency_tolerance {
            errors.push(format!("bin {} centre frequency does not match its declared FFT-bin index", bin.bin_id));
        }
    }
    errors
}

fn compute_bin(bin: &SpectralBin, threshold: f64) -> Result<BinReport, String> {
    let cross_magnitude_squared =
        bin.cross_spectral_real.powi(2) + bin.cross_spectral_imag.powi(2);
    let spectral_product = bin.reference_auto_psd * bin.hydrophone_auto_psd_upa2_per_hz;
    if !cross_magnitude_squared.is_finite()
        || !spectral_product.is_finite()
        || spectral_product <= 0.0
    {
        return Err(format!("bin {} spectral matrix overflows or is not positive", bin.bin_id));
    }
    let raw_coherence = cross_magnitude_squared / spectral_product;
    if !raw_coherence.is_finite()
        || raw_coherence > 1.0 + PSD_MATRIX_RELATIVE_TOLERANCE
    {
        return Err(format!("bin {} violates the positive-semidefinite spectral-matrix constraint", bin.bin_id));
    }
    let coherence = raw_coherence.clamp(0.0, 1.0);
    let coherent_psd = (cross_magnitude_squared / bin.reference_auto_psd)
        .min(bin.hydrophone_auto_psd_upa2_per_hz);
    let residual_psd = (bin.hydrophone_auto_psd_upa2_per_hz - coherent_psd).max(0.0);
    let coherent_fraction = coherent_psd / bin.hydrophone_auto_psd_upa2_per_hz;
    let residual_fraction = residual_psd / bin.hydrophone_auto_psd_upa2_per_hz;
    let to_level = |value: f64| {
        10.0 * (value / PRESSURE_PSD_REFERENCE_UPA2_PER_HZ).log10()
    };
    let disposition = if coherence >= threshold {
        "COHERENT_WITH_DECLARED_REFERENCE"
    } else {
        "BELOW_DECLARED_COHERENCE_THRESHOLD"
    };
    Ok(BinReport {
        bin_id: bin.bin_id.clone(),
        fft_bin_index: bin.fft_bin_index,
        center_frequency_hz: bin.center_frequency_hz,
        reference_auto_psd: bin.reference_auto_psd,
        hydrophone_auto_psd_upa2_per_hz: bin.hydrophone_auto_psd_upa2_per_hz,
        cross_spectral_real: bin.cross_spectral_real,
        cross_spectral_imag: bin.cross_spectral_imag,
        cross_spectral_phase_rad: if cross_magnitude_squared > 0.0 {
            Some(bin.cross_spectral_imag.atan2(bin.cross_spectral_real))
        } else {
            None
        },
        magnitude_squared_coherence: coherence,
        coherent_hydrophone_psd_upa2_per_hz: coherent_psd,
        residual_hydrophone_psd_upa2_per_hz: residual_psd,
        coherent_fraction,
        residual_fraction,
        total_level_db_re_1upa2_per_hz: to_level(bin.hydrophone_auto_psd_upa2_per_hz),
        coherent_level_db_re_1upa2_per_hz: (coherent_psd > 0.0).then(|| to_level(coherent_psd)),
        residual_level_db_re_1upa2_per_hz: (residual_psd > 0.0).then(|| to_level(residual_psd)),
        coherence_threshold: threshold,
        disposition,
    })
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut bin_reports = Vec::new();
    let mut spectral_matrix_invalid = false;
    if validation_errors.is_empty() {
        for bin in &case.bins {
            match compute_bin(bin, case.minimum_coherence) {
                Ok(report) => bin_reports.push(report),
                Err(reason) => {
                    spectral_matrix_invalid = true;
                    reasons.push(reason);
                }
            }
        }
    }

    let evidence_eligible = case.evidence_kind == EvidenceKind::Measured
        && case.source.rights_status == RightsStatus::PublicOrPermitted
        && case.source.independently_reviewed;
    let alignment_eligible = case.acquisition.alignment_expanded_uncertainty_s
        <= case.acquisition.maximum_alignment_uncertainty_s;
    if !alignment_eligible {
        reasons.push("channel alignment uncertainty exceeds the predeclared maximum".to_string());
    }
    let disposition = if !validation_errors.is_empty() || spectral_matrix_invalid {
        "INVALID"
    } else if !alignment_eligible || !evidence_eligible {
        "INCOMPLETE"
    } else {
        "COHERENCE_SCREENED"
    };
    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot qualify measured coherent-noise correction".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source dataset rights are not public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source metadata has not been independently reviewed".to_string());
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
        operating_condition_id: case.acquisition.operating_condition_id.clone(),
        operating_point_receipt_sha256: case.operating_point_receipt_sha256.clone(),
        cross_spectral_convention: "reference_times_conjugate_hydrophone",
        frequency_resolution_hz: case.reference_channel.sample_rate_hz
            / case.reference_channel.fft_length_samples as f64,
        service_fingerprint_sha256: case.service_fingerprint_sha256.clone(),
        environment_fingerprint_sha256: case.environment_fingerprint_sha256.clone(),
        timebase_receipt_sha256: case.acquisition.timebase_receipt_sha256.clone(),
        alignment_receipt_sha256: case.acquisition.alignment_receipt_sha256.clone(),
        alignment_offset_s: case.acquisition.alignment_offset_s,
        alignment_expanded_uncertainty_s: case.acquisition.alignment_expanded_uncertainty_s,
        reference_channel: channel_receipt(&case.reference_channel),
        hydrophone_channel: channel_receipt(&case.hydrophone_channel),
        minimum_average_count: case.minimum_average_count,
        minimum_coherence: case.minimum_coherence,
        disposition,
        claim_ceiling: "linear_component_coherent_with_declared_reference_only_not_unique_causal_source_attribution_or_ecological_or_regulatory_qualification",
        bins: bin_reports,
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-coherent-noise-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-coherent-noise-audit <case.json>");
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
        "COHERENCE_SCREENED" => {}
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

    fn channel(
        instrument: &str,
        signal_kind: ReferenceSignalKind,
        unit: &str,
        raw_digest: char,
    ) -> Channel {
        Channel {
            instrument_id: instrument.to_string(),
            signal_kind,
            unit: unit.to_string(),
            location_id: format!("location-{instrument}"),
            orientation: "longitudinal_axis".to_string(),
            calibration_certificate_sha256: digest('c'),
            raw_trace_sha256: digest(raw_digest),
            sample_rate_hz: 48_000.0,
            total_sample_count: 48_000,
            segment_length_samples: 4_096,
            fft_length_samples: 4_096,
            overlap_fraction: 0.5,
            window_function: "hann".to_string(),
            average_count: 16,
        }
    }

    fn case() -> Case {
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-coherence-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-cross-spectral-known-answer".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('a'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            cross_spectral_convention: "reference_times_conjugate_hydrophone".to_string(),
            reference_channel: channel(
                "shaft-accelerometer-1",
                ReferenceSignalKind::ShaftAcceleration,
                "m/s2",
                'd',
            ),
            hydrophone_channel: channel(
                "hydrophone-1",
                ReferenceSignalKind::HydrophonePressure,
                "uPa",
                'e',
            ),
            acquisition: AcquisitionEvidence {
                timebase_receipt_sha256: digest('f'),
                alignment_receipt_sha256: digest('1'),
                alignment_offset_s: 0.0,
                alignment_expanded_uncertainty_s: 0.0001,
                maximum_alignment_uncertainty_s: 0.001,
                operating_condition_id: "J0.8-sigma-nD1".to_string(),
            },
            operating_point_receipt_sha256: digest('2'),
            service_fingerprint_sha256: digest('3'),
            environment_fingerprint_sha256: digest('4'),
            minimum_average_count: 8,
            minimum_coherence: 0.8,
            bins: vec![
                SpectralBin {
                    bin_id: "100Hz".to_string(),
                    fft_bin_index: 9,
                    center_frequency_hz: 105.46875,
                    reference_auto_psd: 4.0,
                    hydrophone_auto_psd_upa2_per_hz: 19.0,
                    cross_spectral_real: 8.0,
                    cross_spectral_imag: 0.0,
                },
                SpectralBin {
                    bin_id: "200Hz".to_string(),
                    fft_bin_index: 17,
                    center_frequency_hz: 199.21875,
                    reference_auto_psd: 4.0,
                    hydrophone_auto_psd_upa2_per_hz: 10.0,
                    cross_spectral_real: 0.0,
                    cross_spectral_imag: 0.1,
                },
            ],
        }
    }

    #[test]
    fn coherent_component_known_answer_is_computed_in_linear_power() {
        let report = audit(&case());
        let bin = &report.bins[0];
        assert!((bin.magnitude_squared_coherence - 16.0 / 19.0).abs() < 1.0e-12);
        assert!((bin.coherent_hydrophone_psd_upa2_per_hz - 16.0).abs() < 1.0e-12);
        assert!((bin.residual_hydrophone_psd_upa2_per_hz - 3.0).abs() < 1.0e-12);
        assert_eq!(bin.disposition, "COHERENT_WITH_DECLARED_REFERENCE");
    }

    #[test]
    fn phase_rotation_changes_phase_not_coherence_or_power_partition() {
        let mut input = case();
        input.bins[0].cross_spectral_real = 0.0;
        input.bins[0].cross_spectral_imag = 8.0;
        let report = audit(&input);
        assert!((report.bins[0].magnitude_squared_coherence - 16.0 / 19.0).abs() < 1.0e-12);
        assert!((report.bins[0].coherent_hydrophone_psd_upa2_per_hz - 16.0).abs() < 1.0e-12);
        assert!((report.bins[0].cross_spectral_phase_rad.unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }

    #[test]
    fn independent_channel_is_below_declared_threshold() {
        let mut input = case();
        input.bins[0].cross_spectral_real = 0.1;
        input.bins[0].cross_spectral_imag = 0.0;
        let report = audit(&input);
        assert_eq!(report.bins[0].disposition, "BELOW_DECLARED_COHERENCE_THRESHOLD");
    }

    #[test]
    fn invalid_spectral_matrix_is_rejected() {
        let mut input = case();
        input.bins[0].cross_spectral_real = 100.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn zero_reference_auto_spectrum_is_invalid() {
        let mut input = case();
        input.bins[0].reference_auto_psd = 0.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn mismatched_channel_sampling_configuration_is_invalid() {
        let mut input = case();
        input.hydrophone_channel.fft_length_samples = 2_048;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn insufficient_averages_are_invalid() {
        let mut input = case();
        input.reference_channel.average_count = 4;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn synchronization_outside_the_declared_tolerance_is_incomplete() {
        let mut input = case();
        input.acquisition.alignment_expanded_uncertainty_s = 0.002;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("alignment uncertainty")));
    }

    #[test]
    fn zero_cross_spectrum_has_no_phase_or_coherent_level_and_serializes() {
        let mut input = case();
        input.bins[0].cross_spectral_real = 0.0;
        input.bins[0].cross_spectral_imag = 0.0;
        let report = audit(&input);
        assert_eq!(report.bins[0].cross_spectral_phase_rad, None);
        assert_eq!(report.bins[0].coherent_level_db_re_1upa2_per_hz, None);
        assert_eq!(report.bins[0].coherent_fraction, 0.0);
        assert!(serde_json::to_string(&report).is_ok());
    }

    #[test]
    fn alignment_beyond_tolerance_is_incomplete_not_invalid() {
        let mut input = case();
        input.acquisition.alignment_expanded_uncertainty_s = 0.002;
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn cross_spectral_convention_must_be_declared() {
        let mut input = case();
        input.cross_spectral_convention = "unknown".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn hydrophone_channel_must_be_a_pressure_channel() {
        let mut input = case();
        input.hydrophone_channel.signal_kind = ReferenceSignalKind::StructuralVibration;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn frequency_must_match_declared_fft_bin_index() {
        let mut input = case();
        input.bins[0].center_frequency_hz = 100.0;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn channel_average_count_cannot_exceed_available_segment_schedule() {
        let mut input = case();
        input.reference_channel.average_count = 40;
        input.hydrophone_channel.average_count = 40;
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn synthetic_result_is_never_measured_or_qualified() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.reasons.iter().any(|reason| reason.contains("synthetic evidence")));
        assert!(report.claim_ceiling.contains("not_unique_causal_source_attribution"));
    }
}
