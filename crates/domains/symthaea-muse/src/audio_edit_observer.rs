//! Independent WAV-domain gain-envelope observation for MUSE edits.
//!
//! This observer measures windowed RMS dBFS changes on a fixed, sample-aligned
//! timeline. It is deliberately not a BS.1770 loudness meter, source separator,
//! instrument recognizer, or proof of musical validity. It never time-warps:
//! mismatched sample rate, channels, or frame count is rejected.

use hound::{SampleFormat, WavReader};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::io::Cursor;

pub const ACOUSTIC_OBSERVATION_SCHEMA_V1: &str = "muse-acoustic-observation-v1";
pub const GAIN_ENVELOPE_CONTRACT_SCHEMA_V1: &str = "muse-gain-envelope-contract-v1";
pub const GAIN_ENVELOPE_OBSERVER_ID_V1: &str = "muse-windowed-rms-gain-observer";
pub const GAIN_ENVELOPE_OBSERVER_VERSION_V1: &str = "0.1.0";
const RECEIPT_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-acoustic-observation-v1";
const MIN_REPORTABLE_RMS_DBFS: f64 = -300.0;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRangeV1 {
    pub start_frame: u64,
    pub end_frame: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GainTargetWindowV1 {
    pub range: FrameRangeV1,
    pub expected_gain_db: f64,
    pub tolerance_db: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GainEnvelopeContractV1 {
    pub schema: String,
    /// full_mix supports a mix-level claim only. isolated_stem supports a claim
    /// only about the explicitly identified supplied stem or bus.
    pub scope_kind: String,
    pub scope_id: Option<String>,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_count: u64,
    /// Source windows quieter than this level are reported as abstentions.
    pub minimum_source_rms_dbfs: f64,
    /// Maximum fraction of candidate samples at or above 0.999 full scale.
    pub maximum_candidate_clipped_sample_fraction: f64,
    pub protected_max_abs_gain_delta_db: f64,
    pub target_windows: Vec<GainTargetWindowV1>,
    pub protected_windows: Vec<FrameRangeV1>,
}

#[derive(Clone, Debug)]
struct DecodedWav {
    sample_rate_hz: u32,
    channels: u16,
    frame_count: u64,
    sample_format: String,
    bits_per_sample: u16,
    samples: Vec<f64>,
    clipped_sample_fraction: f64,
}

#[derive(Clone, Debug)]
struct WindowObservation {
    range: FrameRangeV1,
    source_rms_dbfs: Option<Vec<f64>>,
    candidate_rms_dbfs: Option<Vec<f64>>,
    expected_gain_db: Option<f64>,
    observed_gain_db: Option<Vec<f64>>,
    absolute_error_db: Option<Vec<f64>>,
    passed: Option<bool>,
    abstention_reason: Option<String>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    fn normalize(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort_unstable();
                let mut sorted = Map::new();
                for key in keys {
                    let child = object.get(key).expect("key came from this object");
                    sorted.insert(key.clone(), normalize(child));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&normalize(value))
        .map_err(|error| format!("could not serialize canonical observation JSON: {error}"))
}

fn valid_range(range: &FrameRangeV1, frame_count: u64, label: &str) -> Result<(), String> {
    if range.start_frame >= range.end_frame || range.end_frame > frame_count {
        return Err(format!(
            "{label} must satisfy 0 <= start_frame < end_frame <= frame_count"
        ));
    }
    Ok(())
}

fn ranges_overlap(a: &FrameRangeV1, b: &FrameRangeV1) -> bool {
    a.start_frame < b.end_frame && b.start_frame < a.end_frame
}

fn validate_contract(contract: &GainEnvelopeContractV1) -> Result<(), String> {
    if contract.schema != GAIN_ENVELOPE_CONTRACT_SCHEMA_V1 {
        return Err("unsupported gain-envelope contract schema".into());
    }
    match contract.scope_kind.as_str() {
        "full_mix" if contract.scope_id.is_none() => {}
        "isolated_stem"
            if contract.scope_id.as_deref().is_some_and(|value| !value.trim().is_empty()) => {}
        "full_mix" => return Err("full_mix scope must not carry a scope_id".into()),
        "isolated_stem" => {
            return Err("isolated_stem scope requires a non-empty scope_id".into());
        }
        _ => return Err("scope_kind must be full_mix or isolated_stem".into()),
    }
    if contract.sample_rate_hz < 8_000
        || contract.sample_rate_hz > 384_000
        || contract.channels == 0
        || contract.channels > 64
        || contract.frame_count == 0
    {
        return Err("sample rate, channels, or frame count is outside supported bounds".into());
    }
    if !contract.minimum_source_rms_dbfs.is_finite()
        || !(-120.0..=0.0).contains(&contract.minimum_source_rms_dbfs)
        || !contract.maximum_candidate_clipped_sample_fraction.is_finite()
        || !(0.0..=1.0).contains(&contract.maximum_candidate_clipped_sample_fraction)
        || !contract.protected_max_abs_gain_delta_db.is_finite()
        || !(0.0..=24.0).contains(&contract.protected_max_abs_gain_delta_db)
    {
        return Err("gain-envelope thresholds are non-finite or outside supported bounds".into());
    }
    if contract.target_windows.is_empty() {
        return Err("at least one target gain window is required".into());
    }
    if contract.protected_windows.is_empty() {
        return Err("at least one protected audio window is required".into());
    }
    for (index, target) in contract.target_windows.iter().enumerate() {
        valid_range(&target.range, contract.frame_count, &format!("target_windows[{index}]"))?;
        if !target.expected_gain_db.is_finite()
            || !(-96.0..=24.0).contains(&target.expected_gain_db)
            || !target.tolerance_db.is_finite()
            || !(0.01..=24.0).contains(&target.tolerance_db)
        {
            return Err(format!("target_windows[{index}] has invalid expected gain or tolerance"));
        }
        for previous in &contract.target_windows[..index] {
            if ranges_overlap(&target.range, &previous.range) {
                return Err("target gain windows must not overlap".into());
            }
        }
        if contract.protected_windows.iter().any(|p| ranges_overlap(&target.range, p)) {
            return Err("target and protected audio windows must not overlap".into());
        }
    }
    for (index, protected) in contract.protected_windows.iter().enumerate() {
        valid_range(protected, contract.frame_count, &format!("protected_windows[{index}]"))?;
        if contract.protected_windows[..index]
            .iter()
            .any(|previous| ranges_overlap(protected, previous))
        {
            return Err("protected audio windows must not overlap".into());
        }
    }
    Ok(())
}

fn decode_wav(bytes: &[u8], label: &str) -> Result<DecodedWav, String> {
    let mut reader = WavReader::new(Cursor::new(bytes))
        .map_err(|error| format!("{label} WAV header is invalid: {error}"))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(format!("{label} WAV has zero channels"));
    }
    let frame_count = reader.duration() as u64;
    let expected_samples = frame_count
        .checked_mul(spec.channels as u64)
        .ok_or_else(|| format!("{label} WAV sample count overflows"))?;
    if expected_samples > MAX_INTERLEAVED_SAMPLES {
        return Err(format!(
            "{label} WAV exceeds the {} interleaved-sample memory limit",
            MAX_INTERLEAVED_SAMPLES
        ));
    }
    let capacity = usize::try_from(expected_samples)
        .map_err(|_| format!("{label} WAV is too large"))?;
    let mut samples = Vec::with_capacity(capacity);

    match spec.sample_format {
        SampleFormat::Int => {
            if !matches!(spec.bits_per_sample, 16 | 24 | 32) {
                return Err(format!(
                    "{label} WAV integer PCM must be 16, 24, or 32 bits; got {}",
                    spec.bits_per_sample
                ));
            }
            let divisor = 2_f64.powi(spec.bits_per_sample as i32 - 1);
            for sample in reader.samples::<i32>() {
                let raw = sample.map_err(|error| format!("{label} WAV PCM decode failed: {error}"))?;
                samples.push(raw as f64 / divisor);
            }
        }
        SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                return Err(format!(
                    "{label} WAV float format must be 32-bit; got {}",
                    spec.bits_per_sample
                ));
            }
            for sample in reader.samples::<f32>() {
                let raw = sample.map_err(|error| format!("{label} WAV float decode failed: {error}"))?;
                if !raw.is_finite() || raw.abs() > 15.0 {
                    return Err(format!("{label} WAV contains a non-finite or out-of-range float sample"));
                }
                samples.push(raw as f64);
            }
        }
    }
    if samples.len() as u64 != expected_samples {
        return Err(format!("{label} WAV sample count does not match its frame header"));
    }
    let clipped = samples.iter().filter(|sample| sample.abs() >= 0.999).count();
    let clipped_sample_fraction = if samples.is_empty() {
        0.0
    } else {
        clipped as f64 / samples.len() as f64
    };
    Ok(DecodedWav {
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        frame_count,
        sample_format: match spec.sample_format {
            SampleFormat::Int => "pcm_integer".to_string(),
            SampleFormat::Float => "ieee_float".to_string(),
        },
        bits_per_sample: spec.bits_per_sample,
        samples,
        clipped_sample_fraction,
    })
}

fn rms_dbfs_per_channel(audio: &DecodedWav, range: &FrameRangeV1) -> Vec<f64> {
    let channels = audio.channels as usize;
    let start = range.start_frame as usize;
    let end = range.end_frame as usize;
    (0..channels)
        .map(|channel| {
            let mut sum_squares = 0.0;
            let frame_count = (end - start) as f64;
            for frame in start..end {
                let sample = audio.samples[frame * channels + channel];
                sum_squares += sample * sample;
            }
            let rms = (sum_squares / frame_count).sqrt();
            if rms <= 0.0 {
                MIN_REPORTABLE_RMS_DBFS
            } else {
                (20.0 * rms.log10()).max(MIN_REPORTABLE_RMS_DBFS)
            }
        })
        .collect()
}

fn target_observation(
    source: &DecodedWav,
    candidate: &DecodedWav,
    target: &GainTargetWindowV1,
    minimum_source_rms_dbfs: f64,
) -> WindowObservation {
    let source_db = rms_dbfs_per_channel(source, &target.range);
    let candidate_db = rms_dbfs_per_channel(candidate, &target.range);
    if source_db.iter().any(|value| *value < minimum_source_rms_dbfs) {
        return WindowObservation {
            range: target.range.clone(),
            source_rms_dbfs: Some(source_db),
            candidate_rms_dbfs: Some(candidate_db),
            expected_gain_db: Some(target.expected_gain_db),
            observed_gain_db: None,
            absolute_error_db: None,
            passed: None,
            abstention_reason: Some("at least one source target channel is below the calibrated RMS floor".into()),
        };
    }
    let observed: Vec<f64> = candidate_db.iter().zip(source_db.iter())
        .map(|(candidate, source)| candidate - source)
        .collect();
    let errors: Vec<f64> = observed.iter()
        .map(|value| (value - target.expected_gain_db).abs())
        .collect();
    let passed = errors.iter().all(|error| *error <= target.tolerance_db);
    WindowObservation {
        range: target.range.clone(),
        source_rms_dbfs: Some(source_db),
        candidate_rms_dbfs: Some(candidate_db),
        expected_gain_db: Some(target.expected_gain_db),
        observed_gain_db: Some(observed),
        absolute_error_db: Some(errors),
        passed: Some(passed),
        abstention_reason: None,
    }
}

fn protected_observation(
    source: &DecodedWav,
    candidate: &DecodedWav,
    range: &FrameRangeV1,
    minimum_source_rms_dbfs: f64,
    tolerance_db: f64,
) -> WindowObservation {
    let source_db = rms_dbfs_per_channel(source, range);
    let candidate_db = rms_dbfs_per_channel(candidate, range);
    if source_db.iter().any(|value| *value < minimum_source_rms_dbfs) {
        return WindowObservation {
            range: range.clone(),
            source_rms_dbfs: Some(source_db),
            candidate_rms_dbfs: Some(candidate_db),
            expected_gain_db: Some(0.0),
            observed_gain_db: None,
            absolute_error_db: None,
            passed: None,
            abstention_reason: Some("at least one source protected channel is below the calibrated RMS floor".into()),
        };
    }
    let observed: Vec<f64> = candidate_db.iter().zip(source_db.iter())
        .map(|(candidate, source)| candidate - source)
        .collect();
    let errors: Vec<f64> = observed.iter().map(|value| value.abs()).collect();
    let passed = errors.iter().all(|error| *error <= tolerance_db);
    WindowObservation {
        range: range.clone(),
        source_rms_dbfs: Some(source_db),
        candidate_rms_dbfs: Some(candidate_db),
        expected_gain_db: Some(0.0),
        observed_gain_db: Some(observed),
        absolute_error_db: Some(errors),
        passed: Some(passed),
        abstention_reason: None,
    }
}

fn window_value(observation: &WindowObservation) -> Value {
    json!({
        "start_frame": observation.range.start_frame,
        "end_frame": observation.range.end_frame,
        "source_rms_dbfs": observation.source_rms_dbfs,
        "candidate_rms_dbfs": observation.candidate_rms_dbfs,
        "expected_gain_db": observation.expected_gain_db,
        "observed_gain_db": observation.observed_gain_db,
        "absolute_error_db": observation.absolute_error_db,
        "passed": observation.passed,
        "abstention_reason": observation.abstention_reason
    })
}

fn status(observations: &[WindowObservation]) -> &'static str {
    if observations.iter().any(|item| item.passed == Some(false)) {
        "fail"
    } else if observations.iter().any(|item| item.abstention_reason.is_some()) {
        "abstain"
    } else {
        "pass"
    }
}

fn observation_seal(value: &Value) -> Result<String, String> {
    let mut unsealed = value.clone();
    unsealed["receipt_sha256"] = Value::Null;
    let bytes = canonical_json_bytes(&unsealed)?;
    let mut hash = Sha256::new();
    hash.update(RECEIPT_DOMAIN_V1);
    hash.update([0]);
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

/// Independently measure a gain-envelope intervention on aligned WAV artifacts.
///
/// The contract supplies expected per-window gain changes in dB. The observer
/// does not infer an edit from free-form language. It rejects any sample-rate,
/// channel-count, or frame-count mismatch and performs no automatic alignment
/// or time warping. full_mix proves mix-level change only; instrument-local
/// claims require a supplied isolated_stem with an explicit scope ID.
pub fn observe_gain_envelope_v1(
    source_wav_bytes: &[u8],
    candidate_wav_bytes: &[u8],
    contract: &GainEnvelopeContractV1,
) -> Result<Value, String> {
    validate_contract(contract)?;
    let source = decode_wav(source_wav_bytes, "source")?;
    let candidate = decode_wav(candidate_wav_bytes, "candidate")?;
    if source.sample_rate_hz != candidate.sample_rate_hz
        || source.channels != candidate.channels
        || source.frame_count != candidate.frame_count
    {
        return Err("abstain: source and candidate WAV timelines must have identical sample rate, channels, and frame count; no time-warp alignment is performed".into());
    }
    if source.sample_format != candidate.sample_format
        || source.bits_per_sample != candidate.bits_per_sample
    {
        return Err("source and candidate WAV encodings must have identical sample format and bit depth".into());
    }
    if source.sample_rate_hz != contract.sample_rate_hz
        || source.channels != contract.channels
        || source.frame_count != contract.frame_count
    {
        return Err("WAV artifact format does not match the frozen gain-envelope contract".into());
    }

    let target_observations: Vec<WindowObservation> = contract
        .target_windows
        .iter()
        .map(|window| target_observation(
            &source,
            &candidate,
            window,
            contract.minimum_source_rms_dbfs,
        ))
        .collect();
    let protected_observations: Vec<WindowObservation> = contract
        .protected_windows
        .iter()
        .map(|range| protected_observation(
            &source,
            &candidate,
            range,
            contract.minimum_source_rms_dbfs,
            contract.protected_max_abs_gain_delta_db,
        ))
        .collect();
    let target_status = status(&target_observations);
    let preservation_status = status(&protected_observations);
    let clipping_passed =
        candidate.clipped_sample_fraction <= contract.maximum_candidate_clipped_sample_fraction;
    let overall_status = if target_status == "fail" || preservation_status == "fail" || !clipping_passed {
        "fail"
    } else if target_status == "abstain" || preservation_status == "abstain" {
        "abstain"
    } else {
        "pass"
    };

    let abstention_reasons: Vec<String> = target_observations
        .iter()
        .chain(protected_observations.iter())
        .filter_map(|item| item.abstention_reason.clone())
        .collect();
    let contract_value = serde_json::to_value(contract)
        .map_err(|error| format!("could not serialize gain-envelope contract: {error}"))?;
    let contract_sha256 = sha256(&canonical_json_bytes(&contract_value)?);
    let mut result = json!({
        "schema": ACOUSTIC_OBSERVATION_SCHEMA_V1,
        "observer": {
            "observer_id": GAIN_ENVELOPE_OBSERVER_ID_V1,
            "version": GAIN_ENVELOPE_OBSERVER_VERSION_V1,
            "measurement": "windowed_rms_dbfs_gain_delta"
        },
        "scope": {
            "kind": contract.scope_kind,
            "scope_id": contract.scope_id
        },
        "artifacts": {
            "contract_sha256": contract_sha256,
            "source_audio_sha256": sha256(source_wav_bytes),
            "candidate_audio_sha256": sha256(candidate_wav_bytes)
        },
        "audio": {
            "sample_rate_hz": source.sample_rate_hz,
            "channels": source.channels,
            "frame_count": source.frame_count,
            "source_sample_format": source.sample_format,
            "source_bits_per_sample": source.bits_per_sample,
            "candidate_sample_format": candidate.sample_format,
            "candidate_bits_per_sample": candidate.bits_per_sample,
            "source_clipped_sample_fraction": source.clipped_sample_fraction,
            "candidate_clipped_sample_fraction": candidate.clipped_sample_fraction
        },
        "target": {
            "status": target_status,
            "windows": target_observations.iter().map(window_value).collect::<Vec<_>>()
        },
        "preservation": {
            "status": preservation_status,
            "windows": protected_observations.iter().map(window_value).collect::<Vec<_>>()
        },
        "clipping": {
            "maximum_candidate_clipped_sample_fraction": contract.maximum_candidate_clipped_sample_fraction,
            "passed": clipping_passed
        },
        "outcome": {
            "acoustic_observation_status": overall_status,
            "abstention_reasons": abstention_reasons
        },
        "receipt_sha256": null
    });
    let seal = observation_seal(&result)?;
    result["receipt_sha256"] = Value::String(seal);
    verify_gain_envelope_observation_seal(&result)?;
    Ok(result)
}

/// Verify the domain-separated observation receipt seal.
pub fn verify_gain_envelope_observation_seal(result: &Value) -> Result<(), String> {
    let recorded = result
        .get("receipt_sha256")
        .and_then(Value::as_str)
        .ok_or("acoustic observation is missing receipt_sha256")?;
    let actual = observation_seal(result)?;
    if recorded.len() != 64
        || !recorded.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || recorded != actual.as_str()
    {
        return Err("acoustic observation receipt seal is invalid".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};

    fn wav(samples: &[f32], sample_rate_hz: u32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut bytes = Vec::new();
        {
            let cursor = Cursor::new(&mut bytes);
            let mut writer = WavWriter::new(cursor, spec).unwrap();
            for &sample in samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        bytes
    }

    fn int16_wav(samples: &[i16], sample_rate_hz: u32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut bytes = Vec::new();
        {
            let cursor = Cursor::new(&mut bytes);
            let mut writer = WavWriter::new(cursor, spec).unwrap();
            for &sample in samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        bytes
    }

    fn stereo_wav(samples: &[[f32; 2]], sample_rate_hz: u32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 2,
            sample_rate: sample_rate_hz,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut bytes = Vec::new();
        {
            let cursor = Cursor::new(&mut bytes);
            let mut writer = WavWriter::new(cursor, spec).unwrap();
            for frame in samples {
                writer.write_sample(frame[0]).unwrap();
                writer.write_sample(frame[1]).unwrap();
            }
            writer.finalize().unwrap();
        }
        bytes
    }

    fn contract() -> GainEnvelopeContractV1 {
        GainEnvelopeContractV1 {
            schema: GAIN_ENVELOPE_CONTRACT_SCHEMA_V1.into(),
            scope_kind: "isolated_stem".into(),
            scope_id: Some("melody-stem-01".into()),
            sample_rate_hz: 8_000,
            channels: 1,
            frame_count: 1_000,
            minimum_source_rms_dbfs: -75.0,
            maximum_candidate_clipped_sample_fraction: 0.001,
            protected_max_abs_gain_delta_db: 0.5,
            target_windows: vec![GainTargetWindowV1 {
                range: FrameRangeV1 { start_frame: 100, end_frame: 400 },
                expected_gain_db: -6.0,
                tolerance_db: 0.1,
            }],
            protected_windows: vec![FrameRangeV1 { start_frame: 600, end_frame: 900 }],
        }
    }

    fn source_and_candidate() -> (Vec<u8>, Vec<u8>) {
        let source_samples = vec![0.25_f32; 1_000];
        let mut candidate_samples = source_samples.clone();
        let gain = 10_f32.powf(-6.0 / 20.0);
        for sample in &mut candidate_samples[100..400] {
            *sample *= gain;
        }
        (wav(&source_samples, 8_000), wav(&candidate_samples, 8_000))
    }

    #[test]
    fn correct_gain_envelope_passes_target_and_protected_windows() {
        let (source, candidate) = source_and_candidate();
        let result = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap();
        assert_eq!(result["target"]["status"], "pass");
        assert_eq!(result["preservation"]["status"], "pass");
        assert_eq!(result["outcome"]["acoustic_observation_status"], "pass");
        verify_gain_envelope_observation_seal(&result).unwrap();
    }

    #[test]
    fn piecewise_gain_envelope_checks_each_temporal_segment() {
        let source_samples = vec![0.25_f32; 1_000];
        let mut candidate_samples = source_samples.clone();
        let gains = [
            (100..200, 10_f32.powf(-2.0 / 20.0)),
            (200..300, 10_f32.powf(-4.0 / 20.0)),
            (300..400, 10_f32.powf(-6.0 / 20.0)),
        ];
        for (range, gain) in gains {
            for sample in &mut candidate_samples[range] {
                *sample *= gain;
            }
        }
        let source = wav(&source_samples, 8_000);
        let candidate = wav(&candidate_samples, 8_000);
        let mut case = contract();
        case.target_windows = vec![
            GainTargetWindowV1 {
                range: FrameRangeV1 { start_frame: 100, end_frame: 200 },
                expected_gain_db: -2.0,
                tolerance_db: 0.1,
            },
            GainTargetWindowV1 {
                range: FrameRangeV1 { start_frame: 200, end_frame: 300 },
                expected_gain_db: -4.0,
                tolerance_db: 0.1,
            },
            GainTargetWindowV1 {
                range: FrameRangeV1 { start_frame: 300, end_frame: 400 },
                expected_gain_db: -6.0,
                tolerance_db: 0.1,
            },
        ];
        let result = observe_gain_envelope_v1(&source, &candidate, &case).unwrap();
        assert_eq!(result["target"]["status"], "pass");
        assert_eq!(result["target"]["windows"].as_array().unwrap().len(), 3);
        assert_eq!(result["outcome"]["acoustic_observation_status"], "pass");
    }

    #[test]
    fn integer_pcm16_gain_envelope_is_decoded_and_measured() {
        let source_samples = vec![8192_i16; 1_000];
        let mut candidate_samples = source_samples.clone();
        for sample in &mut candidate_samples[100..400] {
            *sample = 4096; // approximately -6.02 dB relative to 8192
        }
        let source = int16_wav(&source_samples, 8_000);
        let candidate = int16_wav(&candidate_samples, 8_000);
        let result = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap();
        assert_eq!(result["audio"]["source_sample_format"], "pcm_integer");
        assert_eq!(result["audio"]["source_bits_per_sample"], 16);
        assert_eq!(result["target"]["status"], "pass");
        assert_eq!(result["preservation"]["status"], "pass");
    }

    #[test]
    fn mixed_sample_encodings_are_rejected() {
        let source_samples = vec![0.25_f32; 1_000];
        let candidate_samples = vec![8192_i16; 1_000];
        let source = wav(&source_samples, 8_000);
        let candidate = int16_wav(&candidate_samples, 8_000);
        let error = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap_err();
        assert!(error.contains("identical sample format and bit depth"));
    }

    #[test]
    fn target_gain_must_pass_independently_on_each_channel() {
        let source_samples = vec![[0.25_f32, 0.25_f32]; 1_000];
        let mut candidate_samples = source_samples.clone();
        let gain = 10_f32.powf(-6.0 / 20.0);
        for frame in &mut candidate_samples[100..400] {
            frame[0] *= gain; // left channel changed as requested
            // right channel deliberately remains unchanged
        }
        let source = stereo_wav(&source_samples, 8_000);
        let candidate = stereo_wav(&candidate_samples, 8_000);
        let mut case = contract();
        case.channels = 2;
        let result = observe_gain_envelope_v1(&source, &candidate, &case).unwrap();
        assert_eq!(result["target"]["status"], "fail");
        assert_eq!(result["target"]["windows"][0]["passed"], false);
        assert_eq!(result["outcome"]["acoustic_observation_status"], "fail");
    }

    #[test]
    fn no_op_does_not_pass_a_requested_gain_edit() {
        let source_samples = vec![0.25_f32; 1_000];
        let source = wav(&source_samples, 8_000);
        let result = observe_gain_envelope_v1(&source, &source, &contract()).unwrap();
        assert_eq!(result["target"]["status"], "fail");
        assert_eq!(result["preservation"]["status"], "pass");
        assert_eq!(result["outcome"]["acoustic_observation_status"], "fail");
    }

    #[test]
    fn protected_window_mutation_fails_preservation() {
        let (source, _) = source_and_candidate();
        let mut samples = vec![0.25_f32; 1_000];
        let gain = 10_f32.powf(-6.0 / 20.0);
        for sample in &mut samples[100..400] {
            *sample *= gain;
        }
        for sample in &mut samples[600..900] {
            *sample *= gain;
        }
        let candidate = wav(&samples, 8_000);
        let result = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap();
        assert_eq!(result["target"]["status"], "pass");
        assert_eq!(result["preservation"]["status"], "fail");
        assert_eq!(result["outcome"]["acoustic_observation_status"], "fail");
    }

    #[test]
    fn protected_window_is_checked_per_channel() {
        let source_samples = vec![[0.25_f32, 0.25_f32]; 1_000];
        let mut candidate_samples = source_samples.clone();
        let gain = 10_f32.powf(-6.0 / 20.0);
        for frame in &mut candidate_samples[100..400] {
            frame[0] *= gain;
            frame[1] *= gain;
        }
        for frame in &mut candidate_samples[600..900] {
            frame[0] *= gain; // collateral change on left channel only
        }
        let source = stereo_wav(&source_samples, 8_000);
        let candidate = stereo_wav(&candidate_samples, 8_000);
        let mut case = contract();
        case.channels = 2;
        let result = observe_gain_envelope_v1(&source, &candidate, &case).unwrap();
        assert_eq!(result["target"]["status"], "pass");
        assert_eq!(result["preservation"]["status"], "fail");
        assert_eq!(result["preservation"]["windows"][0]["passed"], false);
    }

    #[test]
    fn non_finite_float_samples_are_rejected() {
        let mut source_samples = vec![0.25_f32; 1_000];
        source_samples[10] = f32::NAN;
        let candidate_samples = vec![0.25_f32; 1_000];
        let source = wav(&source_samples, 8_000);
        let candidate = wav(&candidate_samples, 8_000);
        let error = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap_err();
        assert!(error.contains("non-finite or out-of-range float sample"));
    }

    #[test]
    fn silent_source_target_window_abstains() {
        let mut source_samples = vec![0.25_f32; 1_000];
        let mut candidate_samples = source_samples.clone();
        source_samples[100..400].fill(0.0);
        candidate_samples[100..400].fill(0.0);
        let source = wav(&source_samples, 8_000);
        let candidate = wav(&candidate_samples, 8_000);
        let result = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap();
        assert_eq!(result["target"]["status"], "abstain");
        assert_eq!(result["outcome"]["acoustic_observation_status"], "abstain");
    }

    #[test]
    fn timeline_mismatch_is_rejected_without_time_warp() {
        let (source, _) = source_and_candidate();
        let candidate = wav(&vec![0.25_f32; 999], 8_000);
        let error = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap_err();
        assert!(error.contains("no time-warp alignment"));
    }

    #[test]
    fn target_and_protected_windows_must_not_overlap() {
        let mut case = contract();
        case.protected_windows[0] = FrameRangeV1 { start_frame: 300, end_frame: 900 };
        let (source, candidate) = source_and_candidate();
        let error = observe_gain_envelope_v1(&source, &candidate, &case).unwrap_err();
        assert!(error.contains("must not overlap"));
    }

    #[test]
    fn isolated_stem_scope_requires_an_explicit_identity() {
        let mut case = contract();
        case.scope_id = None;
        let (source, candidate) = source_and_candidate();
        let error = observe_gain_envelope_v1(&source, &candidate, &case).unwrap_err();
        assert!(error.contains("requires a non-empty scope_id"));
    }

    #[test]
    fn clipping_is_a_separate_failure_gate() {
        let source_samples = vec![0.25_f32; 1_000];
        let mut candidate_samples = source_samples.clone();
        let gain = 10_f32.powf(-6.0 / 20.0);
        for sample in &mut candidate_samples[100..400] {
            *sample *= gain;
        }
        candidate_samples[500] = 1.0;
        let source = wav(&source_samples, 8_000);
        let candidate = wav(&candidate_samples, 8_000);
        let mut case = contract();
        case.maximum_candidate_clipped_sample_fraction = 0.0;
        let result = observe_gain_envelope_v1(&source, &candidate, &case).unwrap();
        assert_eq!(result["target"]["status"], "pass");
        assert_eq!(result["clipping"]["passed"], false);
        assert_eq!(result["outcome"]["acoustic_observation_status"], "fail");
    }

    #[test]
    fn receipt_seal_detects_metric_tampering() {
        let (source, candidate) = source_and_candidate();
        let mut result = observe_gain_envelope_v1(&source, &candidate, &contract()).unwrap();
        result["target"]["windows"][0]["observed_gain_db"] = json!(0.0);
        assert!(verify_gain_envelope_observation_seal(&result).is_err());
    }
}
