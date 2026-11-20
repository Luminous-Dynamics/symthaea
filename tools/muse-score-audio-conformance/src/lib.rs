//! Independent pitch-conformance evidence for a narrow Score-to-audio slice.
//!
//! V1 accepts only one assigned, non-overlapping score part in a mono isolated
//! stem. It compares pitch on a stable, score-derived interior window using a
//! harmonic-salience spectrum. It performs no time alignment, tempo fitting,
//! source separation, instrument identification or polyphonic transcription.
//! Unsupported conditions abstain rather than being flattened into a plausible
//! but ungrounded event match.

use hound::{SampleFormat, WavReader};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{cmp::Ordering, io::Cursor};
use symthaea_music_theory::{
    pitch::{Pitch, PitchClass},
    rhythm::Duration,
    score::{Emphasis, PartId, Score, ScoreNote, VoiceRole},
    Key,
};

pub const SCORE_AUDIO_CONFORMANCE_SCHEMA_V1: &str = "muse-score-audio-conformance-v1";
pub const SCORE_AUDIO_CONTRACT_SCHEMA_V1: &str = "muse-score-audio-contract-v1";
pub const SCORE_AUDIO_OBSERVER_ID_V1: &str = "muse-score-audio-pitch-observer";
pub const SCORE_AUDIO_OBSERVER_VERSION_V1: &str = "0.1.0";
pub const SCORE_DIGEST_SCHEMA_V1: &str = "muse-score-serde-json-sha256-v1";
const RECEIPT_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-score-audio-conformance-v1";
const MAX_INTERLEAVED_SAMPLES: u64 = 32_000_000;
const MIN_REPORTABLE_DBFS: f64 = -300.0;
const MIN_MIDI_V1: u8 = 24;
const MAX_MIDI_V1: u8 = 108;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreAudioContractV1 {
    pub schema: String,
    /// The first observer accepts a mono, isolated stem only.
    pub scope_kind: String,
    pub scope_id: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    /// Power-of-two FFT window; also the minimum supported note duration.
    pub analysis_window_frames: u32,
    pub minimum_signal_rms_dbfs: f64,
    pub minimum_pitch_confidence: f64,
    pub pitch_tolerance_cents: f64,
}

#[derive(Clone, Debug)]
struct DecodedWav {
    sample_rate_hz: u32,
    channels: u16,
    frame_count: u64,
    samples: Vec<f64>,
    sample_format: String,
    bits_per_sample: u16,
    clipped_sample_fraction: f64,
}

#[derive(Clone, Debug)]
struct NoteObservation {
    score_note_ordinal: usize,
    expected_pitch_midi: u8,
    observed_pitch_midi: Option<u8>,
    estimated_frequency_hz: Option<f64>,
    pitch_error_cents: Option<f64>,
    pitch_confidence: Option<f64>,
    window_rms_dbfs: f64,
    window_start_frame: u64,
    window_end_frame: u64,
    status: &'static str,
    reason: Option<String>,
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
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update([0]);
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

fn exact_fields(value: &Value, fields: &[&str], label: &str) -> Result<(), String> {
    let object = value.as_object().ok_or_else(|| format!("{label} must be an object"))?;
    let mut actual: Vec<&str> = object.keys().map(String::as_str).collect();
    let mut expected = fields.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    if actual != expected {
        return Err(format!("{label} has an unsupported or missing field set"));
    }
    Ok(())
}

fn validate_score_wire_shape(value: &Value) -> Result<(), String> {
    exact_fields(value, &["key", "tempo_bpm", "meter", "notes", "total_beats"], "Score")?;
    exact_fields(value.get("key").ok_or("Score.key is missing")?, &["tonic", "tonality"], "Score.key")?;
    exact_fields(value.get("total_beats").ok_or("Score.total_beats is missing")?, &["num", "den"], "Score.total_beats")?;
    let notes = value.get("notes").and_then(Value::as_array).ok_or("Score.notes must be an array")?;
    for (index, note) in notes.iter().enumerate() {
        let label = format!("Score.notes[{index}]");
        exact_fields(note, &["part", "pitch", "onset", "duration", "velocity", "role", "emphasis", "section_intensity"], &label)?;
        if note.get("part").and_then(Value::as_u64).is_none_or(|part| part > u16::MAX as u64) {
            return Err(format!("{label}.part must be a u16 PartId"));
        }
        exact_fields(note.get("pitch").ok_or("Score note pitch is missing")?, &["midi"], &format!("{label}.pitch"))?;
        exact_fields(note.get("onset").ok_or("Score note onset is missing")?, &["num", "den"], &format!("{label}.onset"))?;
        exact_fields(note.get("duration").ok_or("Score note duration is missing")?, &["num", "den"], &format!("{label}.duration"))?;
    }
    Ok(())
}

fn validate_score(score: &Score) -> Result<(), String> {
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 || score.meter == 0 {
        return Err("Score tempo or meter is invalid".into());
    }
    if score.total_beats.den() <= 0 || score.total_beats.num() < 0 {
        return Err("Score total_beats rational is invalid".into());
    }
    for (index, note) in score.notes.iter().enumerate() {
        if note.onset.den() <= 0 || note.onset.num() < 0
            || note.duration.den() <= 0 || note.duration.num() <= 0
            || !note.velocity.is_finite() || !(0.0..=1.0).contains(&note.velocity)
            || !note.section_intensity.is_finite() || note.section_intensity < 0.0
        {
            return Err(format!("Score note {index} has invalid timing or dynamics"));
        }
        if (note.onset + note.duration).beats() > score.total_beats.beats() + 1e-9 {
            return Err(format!("Score note {index} ends beyond total_beats"));
        }
    }
    Ok(())
}

fn validate_contract(contract: &ScoreAudioContractV1) -> Result<(), String> {
    if contract.schema != SCORE_AUDIO_CONTRACT_SCHEMA_V1 {
        return Err("unsupported score-audio contract schema".into());
    }
    if contract.scope_kind != "isolated_stem" || contract.scope_id.trim().is_empty() {
        return Err("v1 score-audio observer requires an explicit isolated_stem scope ID".into());
    }
    if contract.channels != 1 {
        return Err("v1 score-audio observer requires mono isolated-stem WAV input".into());
    }
    if !(8_000..=192_000).contains(&contract.sample_rate_hz) {
        return Err("sample rate is outside supported bounds".into());
    }
    let window = contract.analysis_window_frames;
    if !(1_024..=16_384).contains(&window) || !window.is_power_of_two() {
        return Err("analysis_window_frames must be a power of two in 1024..=16384".into());
    }
    if !contract.minimum_signal_rms_dbfs.is_finite()
        || !(-120.0..=0.0).contains(&contract.minimum_signal_rms_dbfs)
        || !contract.minimum_pitch_confidence.is_finite()
        || !(0.0..=1.0).contains(&contract.minimum_pitch_confidence)
        || !contract.pitch_tolerance_cents.is_finite()
        || !(1.0..=100.0).contains(&contract.pitch_tolerance_cents)
    {
        return Err("signal, confidence or pitch tolerances are outside supported bounds".into());
    }
    Ok(())
}

fn decode_wav(bytes: &[u8]) -> Result<DecodedWav, String> {
    let mut reader = WavReader::new(Cursor::new(bytes))
        .map_err(|error| format!("WAV header is invalid: {error}"))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("v1 score-audio observer requires mono WAV".into());
    }
    let frames = reader.duration() as u64;
    if frames == 0 {
        return Err("WAV contains no audio frames".into());
    }
    let expected = frames.checked_mul(spec.channels as u64).ok_or("WAV sample count overflow")?;
    if expected > MAX_INTERLEAVED_SAMPLES {
        return Err("WAV exceeds the decoded-memory limit".into());
    }
    let mut samples = Vec::with_capacity(usize::try_from(expected).map_err(|_| "WAV too large")?);
    match spec.sample_format {
        SampleFormat::Int => {
            if !matches!(spec.bits_per_sample, 16 | 24 | 32) {
                return Err("integer WAV must be 16, 24 or 32 bits".into());
            }
            let divisor = 2_f64.powi(spec.bits_per_sample as i32 - 1);
            for sample in reader.samples::<i32>() {
                let value = sample.map_err(|error| format!("WAV PCM decode failed: {error}"))?;
                samples.push(value as f64 / divisor);
            }
        }
        SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                return Err("float WAV must be IEEE float32".into());
            }
            for sample in reader.samples::<f32>() {
                let value = sample.map_err(|error| format!("WAV float decode failed: {error}"))?;
                if !value.is_finite() || value.abs() > 16.0 {
                    return Err("WAV contains non-finite or grossly out-of-range float samples".into());
                }
                samples.push(value as f64);
            }
        }
    }
    if samples.len() as u64 != expected {
        return Err("WAV decoded sample count differs from header frame count".into());
    }
    let clipped = samples.iter().filter(|sample| sample.abs() >= 0.999).count();
    Ok(DecodedWav {
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        frame_count: frames,
        samples,
        sample_format: match spec.sample_format {
            SampleFormat::Int => "pcm_integer".into(),
            SampleFormat::Float => "ieee_float".into(),
        },
        bits_per_sample: spec.bits_per_sample,
        clipped_sample_fraction: clipped as f64 / expected as f64,
    })
}

fn rms_dbfs(samples: &[f64]) -> f64 {
    let mean_square = samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64;
    let rms = mean_square.sqrt();
    if rms <= 0.0 { MIN_REPORTABLE_DBFS } else { (20.0 * rms.log10()).max(MIN_REPORTABLE_DBFS) }
}

fn midi_frequency(midi: u8) -> f64 {
    440.0 * 2_f64.powf((midi as f64 - 69.0) / 12.0)
}

fn peak_near(magnitudes: &[f64], bin: f64) -> f64 {
    let center = bin.round() as isize;
    let mut peak = 0.0_f64;
    for offset in -1..=1 {
        let index = center + offset;
        if index > 0 && (index as usize) < magnitudes.len() {
            peak = peak.max(magnitudes[index as usize]);
        }
    }
    peak
}

fn pitch_salience(window: &[f64], sample_rate_hz: u32) -> (u8, f64, f64) {
    let n = window.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex<f64>> = window.iter().enumerate().map(|(index, sample)| {
        let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / (n - 1) as f64).cos();
        Complex::new(sample * hann, 0.0)
    }).collect();
    fft.process(&mut buffer);
    let magnitudes: Vec<f64> = buffer[..=n / 2].iter().map(|value| value.norm()).collect();
    let mut candidates = Vec::with_capacity((MAX_MIDI_V1 - MIN_MIDI_V1 + 1) as usize);
    for midi in MIN_MIDI_V1..=MAX_MIDI_V1 {
        let fundamental = midi_frequency(midi);
        let mut salience = 0.0;
        for harmonic in 1..=8 {
            let frequency = fundamental * harmonic as f64;
            if frequency >= sample_rate_hz as f64 / 2.0 { break; }
            salience += peak_near(&magnitudes, frequency * n as f64 / sample_rate_hz as f64) / (harmonic as f64).sqrt();
        }
        candidates.push((midi, salience));
    }
    candidates.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let (best_midi, best_score) = candidates[0];
    let second_score = candidates.get(1).map(|item| item.1).unwrap_or(0.0);
    let confidence = if best_score <= f64::EPSILON { 0.0 } else {
        ((best_score - second_score).max(0.0) / best_score).clamp(0.0, 1.0)
    };
    let fundamental_bin = midi_frequency(best_midi) * n as f64 / sample_rate_hz as f64;
    let center = fundamental_bin.round() as isize;
    let frequency = if center > 1 && (center as usize + 1) < magnitudes.len() {
        let left = magnitudes[(center - 1) as usize].max(f64::MIN_POSITIVE).ln();
        let middle = magnitudes[center as usize].max(f64::MIN_POSITIVE).ln();
        let right = magnitudes[(center + 1) as usize].max(f64::MIN_POSITIVE).ln();
        let denominator = left - 2.0 * middle + right;
        let delta = if denominator.abs() > 1e-12 { 0.5 * (left - right) / denominator } else { 0.0 };
        (center as f64 + delta.clamp(-1.0, 1.0)) * sample_rate_hz as f64 / n as f64
    } else {
        midi_frequency(best_midi)
    };
    (best_midi, frequency, confidence)
}

fn note_window(
    score: &Score,
    note: &ScoreNote,
    ordinal: usize,
    audio: &DecodedWav,
    contract: &ScoreAudioContractV1,
) -> NoteObservation {
    let onset_seconds = note.onset.seconds(score.tempo_bpm as f64);
    let duration_seconds = note.duration.seconds(score.tempo_bpm as f64);
    let start_float = onset_seconds * audio.sample_rate_hz as f64;
    let note_frames = duration_seconds * audio.sample_rate_hz as f64;
    let window_frames = contract.analysis_window_frames as f64;
    let abstain = |reason: String, start: u64, end: u64, rms: f64| NoteObservation {
        score_note_ordinal: ordinal,
        expected_pitch_midi: note.pitch.midi(),
        observed_pitch_midi: None,
        estimated_frequency_hz: None,
        pitch_error_cents: None,
        pitch_confidence: None,
        window_rms_dbfs: rms,
        window_start_frame: start,
        window_end_frame: end,
        status: "abstain",
        reason: Some(reason),
    };
    if !start_float.is_finite() || !note_frames.is_finite() || start_float < 0.0 {
        return abstain("score event has invalid timing".into(), 0, 0, MIN_REPORTABLE_DBFS);
    }
    if note_frames < window_frames {
        return abstain("note is shorter than the frozen analysis window".into(), 0, 0, MIN_REPORTABLE_DBFS);
    }
    let start = start_float.round() as u64;
    let stable_start = start.saturating_add(((note_frames - window_frames) / 2.0).round() as u64);
    let end = stable_start.saturating_add(contract.analysis_window_frames as u64);
    if end > audio.frame_count || end <= stable_start {
        return abstain("score-derived analysis window lies outside the WAV artifact".into(), stable_start, end, MIN_REPORTABLE_DBFS);
    }
    let samples = &audio.samples[stable_start as usize..end as usize];
    let level = rms_dbfs(samples);
    if level <= -120.0 {
        return NoteObservation {
            score_note_ordinal: ordinal,
            expected_pitch_midi: note.pitch.midi(),
            observed_pitch_midi: None,
            estimated_frequency_hz: None,
            pitch_error_cents: None,
            pitch_confidence: Some(0.0),
            window_rms_dbfs: level,
            window_start_frame: stable_start,
            window_end_frame: end,
            status: "fail",
            reason: Some("expected note window is silent".into()),
        };
    }
    if level < contract.minimum_signal_rms_dbfs {
        return abstain("expected note window is below the calibrated signal floor".into(), stable_start, end, level);
    }
    let (observed_midi, frequency, confidence) = pitch_salience(samples, audio.sample_rate_hz);
    let cents = 1200.0 * (frequency / midi_frequency(note.pitch.midi())).log2();
    let status = if confidence < contract.minimum_pitch_confidence {
        "abstain"
    } else if observed_midi != note.pitch.midi() || cents.abs() > contract.pitch_tolerance_cents {
        "fail"
    } else {
        "pass"
    };
    NoteObservation {
        score_note_ordinal: ordinal,
        expected_pitch_midi: note.pitch.midi(),
        observed_pitch_midi: Some(observed_midi),
        estimated_frequency_hz: Some(frequency),
        pitch_error_cents: Some(cents),
        pitch_confidence: Some(confidence),
        window_rms_dbfs: level,
        window_start_frame: stable_start,
        window_end_frame: end,
        status,
        reason: match status {
            "pass" => None,
            "fail" => Some("observed pitch does not match the score event".into()),
            _ => Some("pitch evidence is below the calibrated confidence threshold".into()),
        },
    }
}

fn outcome(observations: &[NoteObservation]) -> &'static str {
    if observations.iter().any(|item| item.status == "fail") {
        "fail"
    } else if observations.iter().any(|item| item.status == "abstain") {
        "abstain"
    } else {
        "pass"
    }
}

fn observation_value(item: &NoteObservation) -> Value {
    json!({
        "score_note_ordinal": item.score_note_ordinal,
        "expected_pitch_midi": item.expected_pitch_midi,
        "observed_pitch_midi": item.observed_pitch_midi,
        "estimated_frequency_hz": item.estimated_frequency_hz,
        "pitch_error_cents": item.pitch_error_cents,
        "pitch_confidence": item.pitch_confidence,
        "window_rms_dbfs": item.window_rms_dbfs,
        "window_start_frame": item.window_start_frame,
        "window_end_frame": item.window_end_frame,
        "status": item.status,
        "reason": item.reason
    })
}

fn base_result(
    score: &Score,
    wav_bytes: &[u8],
    audio: &DecodedWav,
    contract: &ScoreAudioContractV1,
    observations: Vec<NoteObservation>,
    status: &str,
    reasons: Vec<String>,
) -> Result<Value, String> {
    let score_bytes = serde_json::to_vec(score).map_err(|error| format!("typed Score serialization failed: {error}"))?;
    let contract_value = serde_json::to_value(contract).map_err(|error| format!("contract serialization failed: {error}"))?;
    Ok(json!({
        "schema": SCORE_AUDIO_CONFORMANCE_SCHEMA_V1,
        "observer": {"observer_id": SCORE_AUDIO_OBSERVER_ID_V1, "version": SCORE_AUDIO_OBSERVER_VERSION_V1, "method":"harmonic_salience_fft_pitch"},
        "scope": {"kind":contract.scope_kind,"scope_id":contract.scope_id},
        "artifacts": {
            "score_digest_schema": SCORE_DIGEST_SCHEMA_V1,
            "score_sha256": sha256(&score_bytes),
            "audio_sha256": sha256(wav_bytes),
            "contract_sha256": sha256(&canonical_json_bytes(&contract_value)?)
        },
        "alignment": {"mode":"fixed_score_tempo_no_auto_alignment","tempo_bpm":score.tempo_bpm,"offset_frames":0},
        "audio": {
            "sample_rate_hz":audio.sample_rate_hz,
            "channels":audio.channels,
            "frame_count":audio.frame_count,
            "sample_format":audio.sample_format,
            "bits_per_sample":audio.bits_per_sample,
            "clipped_sample_fraction":audio.clipped_sample_fraction
        },
        "events": observations.iter().map(observation_value).collect::<Vec<_>>(),
        "outcome": {"score_audio_status":status,"reasons":reasons},
        "receipt_sha256": null
    }))
}

fn seal_result(result: &mut Value) -> Result<(), String> {
    result["receipt_sha256"] = Value::Null;
    let digest = domain_digest(RECEIPT_DOMAIN_V1, &canonical_json_bytes(result)?);
    result["receipt_sha256"] = Value::String(digest);
    verify_score_audio_conformance_seal(result)
}

pub fn evaluate_score_audio_conformance_v1(
    score_json: &[u8],
    wav_bytes: &[u8],
    contract: &ScoreAudioContractV1,
) -> Result<Value, String> {
    validate_contract(contract)?;
    let score_value: Value = serde_json::from_slice(score_json).map_err(|error| format!("Score JSON invalid: {error}"))?;
    validate_score_wire_shape(&score_value)?;
    let score: Score = serde_json::from_value(score_value).map_err(|error| format!("typed Score invalid: {error}"))?;
    validate_score(&score)?;
    if score.notes.is_empty() {
        return Err("Score requires at least one note".into());
    }
    let audio = decode_wav(wav_bytes)?;
    if audio.sample_rate_hz != contract.sample_rate_hz || audio.channels != contract.channels {
        return Err("WAV sample rate/channel count differs from the frozen contract".into());
    }
    let score_bytes = serde_json::to_vec(&score).map_err(|error| format!("typed Score serialization failed: {error}"))?;
    let score_hash = sha256(&score_bytes);

    if !score.has_part_identity() {
        let mut result = base_result(&score, wav_bytes, &audio, contract, Vec::new(), "abstain", vec!["Score notes do not carry assigned PartId identity".into()])?;
        seal_result(&mut result)?;
        return Ok(result);
    }
    if score.parts().len() != 1 {
        let mut result = base_result(&score, wav_bytes, &audio, contract, Vec::new(), "abstain", vec!["v1 supports one isolated Score part; multi-part attribution is not implemented".into()])?;
        seal_result(&mut result)?;
        return Ok(result);
    }
    let mut ordered = score.notes.iter().enumerate().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.1.onset.beats().partial_cmp(&right.1.onset.beats()).unwrap_or(Ordering::Equal));
    for pair in ordered.windows(2) {
        let left_end = (pair[0].1.onset + pair[0].1.duration).beats();
        if left_end > pair[1].1.onset.beats() + 1e-9 {
            let mut result = base_result(&score, wav_bytes, &audio, contract, Vec::new(), "abstain", vec!["overlapping Score events are polyphonic and unsupported by this v1 observer".into()])?;
            seal_result(&mut result)?;
            return Ok(result);
        }
    }
    let observations = score.notes.iter().enumerate().map(|(ordinal, note)| {
        if !(MIN_MIDI_V1..=MAX_MIDI_V1).contains(&note.pitch.midi()) {
            NoteObservation {
                score_note_ordinal: ordinal, expected_pitch_midi: note.pitch.midi(),
                observed_pitch_midi: None, estimated_frequency_hz: None,
                pitch_error_cents: None, pitch_confidence: None,
                window_rms_dbfs: MIN_REPORTABLE_DBFS, window_start_frame: 0,
                window_end_frame: 0, status: "abstain",
                reason: Some("score pitch is outside the calibrated v1 MIDI range".into()),
            }
        } else {
            note_window(&score, note, ordinal, &audio, contract)
        }
    }).collect::<Vec<_>>();
    let status = outcome(&observations);
    let reasons = observations.iter().filter_map(|item| item.reason.clone()).collect::<Vec<_>>();
    let mut result = base_result(&score, wav_bytes, &audio, contract, observations, status, reasons)?;
    seal_result(&mut result)?;
    Ok(result)
}

pub fn verify_score_audio_conformance_seal(result: &Value) -> Result<(), String> {
    if result.get("schema").and_then(Value::as_str) != Some(SCORE_AUDIO_CONFORMANCE_SCHEMA_V1) {
        return Err("unsupported score-audio conformance result schema".into());
    }
    let recorded = result.get("receipt_sha256").and_then(Value::as_str).ok_or("score-audio result is missing receipt_sha256")?;
    if recorded.len() != 64 || !recorded.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("score-audio receipt seal is not lowercase SHA-256".into());
    }
    let mut unsealed = result.clone();
    unsealed["receipt_sha256"] = Value::Null;
    let actual = domain_digest(RECEIPT_DOMAIN_V1, &canonical_json_bytes(&unsealed)?);
    if recorded != actual {
        return Err("score-audio receipt seal mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};

    fn score(midi: u8, duration: Duration) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(ScoreNote {
            part: PartId(0), pitch: Pitch::from_midi(midi),
            onset: Duration::zero(), duration, velocity: 0.7,
            role: VoiceRole::Melody, emphasis: Emphasis::Normal, section_intensity: 1.0,
        });
        score
    }

    fn wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let spec = WavSpec { channels:1, sample_rate, bits_per_sample:32, sample_format:SampleFormat::Float };
        let mut bytes = Vec::new();
        let cursor = Cursor::new(&mut bytes);
        let mut writer = WavWriter::new(cursor, spec).unwrap();
        for sample in samples { writer.write_sample(*sample).unwrap(); }
        writer.finalize().unwrap();
        bytes
    }

    fn sine(midi: u8, frames: usize, sample_rate: u32) -> Vec<f32> {
        let frequency = midi_frequency(midi);
        (0..frames).map(|index| (0.25 * (2.0 * std::f64::consts::PI * frequency * index as f64 / sample_rate as f64).sin()) as f32).collect()
    }

    fn contract() -> ScoreAudioContractV1 {
        ScoreAudioContractV1 {
            schema:SCORE_AUDIO_CONTRACT_SCHEMA_V1.into(), scope_kind:"isolated_stem".into(),
            scope_id:"stem-melody-001".into(), sample_rate_hz:16_000, channels:1,
            analysis_window_frames:4096, minimum_signal_rms_dbfs:-65.0,
            minimum_pitch_confidence:0.15, pitch_tolerance_cents:35.0,
        }
    }

    #[test]
    fn exact_monophonic_score_pitch_passes() {
        let score = score(60, Duration::quarter());
        let score_bytes = serde_json::to_vec(&score).unwrap();
        let result = evaluate_score_audio_conformance_v1(&score_bytes, &wav(&sine(60,8_000,16_000),16_000), &contract()).unwrap();
        assert_eq!(result["outcome"]["score_audio_status"], "pass");
        assert_eq!(result["events"][0]["observed_pitch_midi"], 60);
        verify_score_audio_conformance_seal(&result).unwrap();
    }

    #[test]
    fn semitone_corruption_fails_score_audio_conformance() {
        let score = score(60, Duration::quarter());
        let result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(), &wav(&sine(61,8_000,16_000),16_000), &contract()).unwrap();
        assert_eq!(result["outcome"]["score_audio_status"], "fail");
        assert_eq!(result["events"][0]["observed_pitch_midi"], 61);
    }

    #[test]
    fn silent_expected_event_is_a_failure_not_a_pass() {
        let score = score(60, Duration::quarter());
        let result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(), &wav(&vec![0.0_f32;8_000],16_000), &contract()).unwrap();
        assert_eq!(result["outcome"]["score_audio_status"], "fail");
    }

    #[test]
    fn overlapping_score_events_abstain_instead_of_mixing_voices() {
        let mut score = score(60, Duration::whole());
        score.push(ScoreNote {
            part:PartId(0), pitch:Pitch::from_midi(67), onset:Duration::quarter(),
            duration:Duration::quarter(), velocity:0.7, role:VoiceRole::Harmony,
            emphasis:Emphasis::Normal, section_intensity:1.0
        });
        let result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(),&wav(&vec![0.0_f32;16_000],16_000),&contract()).unwrap();
        assert_eq!(result["outcome"]["score_audio_status"], "abstain");
    }

    #[test]
    fn unassigned_parts_abstain() {
        let mut score = score(60, Duration::quarter());
        score.notes[0].part = PartId::UNASSIGNED;
        let result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(),&wav(&sine(60,8_000,16_000),16_000),&contract()).unwrap();
        assert_eq!(result["outcome"]["score_audio_status"], "abstain");
    }

    #[test]
    fn short_note_abstains_when_stable_window_cannot_fit() {
        let score = score(60, Duration::eighth());
        let result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(),&wav(&sine(60,4_000,16_000),16_000),&contract()).unwrap();
        assert_eq!(result["outcome"]["score_audio_status"], "abstain");
    }

    #[test]
    fn unknown_score_fields_are_rejected() {
        let score = score(60, Duration::quarter());
        let mut value = serde_json::to_value(&score).unwrap();
        value["notes"][0]["unexpected_control"] = json!(true);
        let error = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&value).unwrap(), &wav(&sine(60,8_000,16_000),16_000), &contract()).unwrap_err();
        assert!(error.contains("unsupported or missing field set"));
    }

    #[test]
    fn result_tampering_is_detected() {
        let score = score(60, Duration::quarter());
        let mut result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(), &wav(&sine(60,8_000,16_000),16_000), &contract()).unwrap();
        result["events"][0]["observed_pitch_midi"] = json!(64);
        assert!(verify_score_audio_conformance_seal(&result).is_err());
    }

    #[test]
    fn score_digest_is_the_typed_score_serde_digest() {
        let score = score(60, Duration::quarter());
        let result = evaluate_score_audio_conformance_v1(&serde_json::to_vec(&score).unwrap(), &wav(&sine(60,8_000,16_000),16_000), &contract()).unwrap();
        assert_eq!(result["artifacts"]["score_digest_schema"], SCORE_DIGEST_SCHEMA_V1);
        assert_eq!(result["artifacts"]["score_sha256"], json!(sha256(&serde_json::to_vec(&score).unwrap())));
    }
}
