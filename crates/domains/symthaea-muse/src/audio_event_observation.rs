// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio-side pitch-energy observations for provenance-bound MIDI events.
//!
//! This is an observation probe, not an automatic transcription system. It
//! measures local harmonic energy near each expected MIDI pitch inside the
//! event's scheduled time interval. It does not attribute acoustic energy to
//! a PartId, prove note onset/release accuracy, identify an instrument, or
//! establish perceptual fidelity.

use crate::midi_performance::{
    MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1, MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1,
    MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1, MIDI_PERFORMANCE_SCHEMA_V1,
    MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1, MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1,
    MidiPerformanceArtifactV1,
};
use hound::{SampleFormat, WavReader};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

pub const AUDIO_EVENT_OBSERVATION_SCHEMA_V1: &str = "muse-audio-event-observation-v1";
pub const AUDIO_EVENT_OBSERVATION_MAX_FFT_FRAMES_V1: usize = 65_536;
pub const AUDIO_EVENT_OBSERVATION_MIN_FFT_FRAMES_V1: usize = 4_096;
pub const AUDIO_EVENT_OBSERVATION_HARMONICS_V1: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEventObservationStatusV1 {
    Measured,
    InsufficientDuration,
    TruncatedByAudio,
    FundamentalAboveNyquist,
    InsufficientFrequencyResolution,
    NoMeasurableSpectralEnergy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventPitchObservationV1 {
    pub source_note_ordinal: usize,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub pitch_midi: u8,
    pub expected_frequency_hz: f64,
    pub expected_start_frame: u64,
    pub expected_end_frame: u64,
    pub window_start_frame: Option<u64>,
    pub window_frame_count: Option<u32>,
    /// Mean target-harmonic peak / local sideband ratio in dB. This is a
    /// spectral salience descriptor, not a calibrated probability.
    pub harmonic_salience_db: Option<f64>,
    /// Fundamental-only peak / local sideband ratio in dB.
    pub fundamental_salience_db: Option<f64>,
    pub harmonics_measured: u8,
    pub status: AudioEventObservationStatusV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioEventObservationWavInfoV1 {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_format: String,
    pub frame_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventObservationReceiptV1 {
    pub schema: String,
    pub score_sha256: String,
    pub midi_sha256: String,
    pub midi_manifest_sha256: String,
    pub wav_sha256: String,
    pub wav_info: AudioEventObservationWavInfoV1,
    pub note_observations: Vec<AudioEventPitchObservationV1>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEventObservationIssueV1 {
    WrongMidiSchema,
    InvalidDigest { field: String },
    MidiBytesDigestMismatch,
    MidiManifestDigestMismatch,
    InvalidTempoOrResolution,
    EmptyNoteManifest,
    DuplicateSourceNoteOrdinal { ordinal: usize },
    InvalidNoteReceipt { ordinal: usize },
    MissingTrackReceipt { ordinal: usize },
    TrackReceiptMismatch { ordinal: usize },
    InvalidWav,
    NonFiniteAudioSample,
    SampleCountNotDivisibleByChannels,
    ReceiptDigestMismatch,
    ReceiptInputBindingMismatch,
    ReceiptNoteCoverageMismatch,
    ReceiptEventMismatch { ordinal: usize },
}

#[derive(Serialize)]
struct ReceiptCommitmentV1<'a> {
    schema: &'a str,
    score_sha256: &'a str,
    midi_sha256: &'a str,
    midi_manifest_sha256: &'a str,
    wav_sha256: &'a str,
    wav_info: &'a AudioEventObservationWavInfoV1,
    note_observations: &'a [AudioEventPitchObservationV1],
}

struct DecodedWavV1 {
    info: AudioEventObservationWavInfoV1,
    interleaved_samples: Vec<f32>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn domain_separated_sha256(schema: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn manifest_sha256(
    artifact: &MidiPerformanceArtifactV1,
) -> Result<String, AudioEventObservationIssueV1> {
    let bytes = serde_json::to_vec(&artifact.manifest)
        .map_err(|_| AudioEventObservationIssueV1::MidiManifestDigestMismatch)?;
    Ok(domain_separated_sha256(
        MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
        &bytes,
    ))
}

fn validate_midi_artifact(
    artifact: &MidiPerformanceArtifactV1,
) -> Vec<AudioEventObservationIssueV1> {
    let manifest = &artifact.manifest;
    let mut issues = Vec::new();
    if manifest.schema != MIDI_PERFORMANCE_SCHEMA_V1
        || manifest.score_digest_schema
            != symthaea_music_theory::orchestration_timeline::ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1
        || manifest.timeline_digest_schema != MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1
        || manifest.binding_digest_schema != MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1
        || manifest.capability_digest_schema != MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1
    {
        issues.push(AudioEventObservationIssueV1::WrongMidiSchema);
    }
    for (field, digest) in [
        ("score_sha256", manifest.score_sha256.as_str()),
        ("timeline_sha256", manifest.timeline_sha256.as_str()),
        ("binding_sha256", manifest.binding_sha256.as_str()),
        ("capability_sha256", manifest.capability_sha256.as_str()),
        ("midi_sha256", manifest.midi_sha256.as_str()),
        ("manifest_sha256", artifact.manifest_sha256.as_str()),
    ] {
        if !valid_sha256(digest) {
            issues.push(AudioEventObservationIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if manifest.ticks_per_quarter == 0
        || manifest.tempo_microseconds_per_quarter == 0
        || manifest.tempo_microseconds_per_quarter > 16_777_215
        || manifest.ticks_per_quarter != MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1
    {
        issues.push(AudioEventObservationIssueV1::InvalidTempoOrResolution);
    }
    if sha256_hex(&artifact.midi_bytes) != manifest.midi_sha256 {
        issues.push(AudioEventObservationIssueV1::MidiBytesDigestMismatch);
    }
    match manifest_sha256(artifact) {
        Ok(actual) if actual == artifact.manifest_sha256 => {}
        _ => issues.push(AudioEventObservationIssueV1::MidiManifestDigestMismatch),
    }
    if manifest.notes.is_empty() {
        issues.push(AudioEventObservationIssueV1::EmptyNoteManifest);
    }

    let mut ordinals = BTreeSet::new();
    for note in &manifest.notes {
        if !ordinals.insert(note.source_note_ordinal) {
            issues.push(AudioEventObservationIssueV1::DuplicateSourceNoteOrdinal {
                ordinal: note.source_note_ordinal,
            });
        }
        if note.pitch_midi > 127
            || note.end_tick <= note.start_tick
            || note.segment_id.trim().is_empty()
            || note.instrument_id.trim().is_empty()
        {
            issues.push(AudioEventObservationIssueV1::InvalidNoteReceipt {
                ordinal: note.source_note_ordinal,
            });
        }
        let Some(track) = manifest
            .tracks
            .iter()
            .find(|track| track.track_index == note.track_index)
        else {
            issues.push(AudioEventObservationIssueV1::MissingTrackReceipt {
                ordinal: note.source_note_ordinal,
            });
            continue;
        };
        if track.part_id != note.part_id
            || track.segment_id != note.segment_id
            || track.instrument_id != note.instrument_id
            || note.start_tick < track.segment_start_tick
            || note.end_tick > track.segment_end_tick
        {
            issues.push(AudioEventObservationIssueV1::TrackReceiptMismatch {
                ordinal: note.source_note_ordinal,
            });
        }
    }
    issues
}

fn decode_wav(wav_bytes: &[u8]) -> Result<DecodedWavV1, AudioEventObservationIssueV1> {
    let mut reader = WavReader::new(Cursor::new(wav_bytes))
        .map_err(|_| AudioEventObservationIssueV1::InvalidWav)?;
    let spec = reader.spec();
    if spec.channels == 0
        || spec.sample_rate == 0
        || spec.bits_per_sample == 0
        || spec.bits_per_sample > 32
        || (spec.sample_format == SampleFormat::Float && spec.bits_per_sample != 32)
    {
        return Err(AudioEventObservationIssueV1::InvalidWav);
    }
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| AudioEventObservationIssueV1::InvalidWav)?,
        SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| (f64::from(value) / scale) as f32)
                        .map_err(|_| AudioEventObservationIssueV1::InvalidWav)
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    if samples.is_empty() {
        return Err(AudioEventObservationIssueV1::InvalidWav);
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(AudioEventObservationIssueV1::NonFiniteAudioSample);
    }
    let channels = usize::from(spec.channels);
    if samples.len() % channels != 0 {
        return Err(AudioEventObservationIssueV1::SampleCountNotDivisibleByChannels);
    }
    let frame_count = samples.len() / channels;
    if frame_count == 0 {
        return Err(AudioEventObservationIssueV1::InvalidWav);
    }
    Ok(DecodedWavV1 {
        info: AudioEventObservationWavInfoV1 {
            sample_rate_hz: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            sample_format: match spec.sample_format {
                SampleFormat::Float => "float".into(),
                SampleFormat::Int => "integer".into(),
            },
            frame_count: frame_count as u64,
        },
        interleaved_samples: samples,
    })
}

fn expected_frame(tick: u64, tempo_us: u32, ppq: u16, sample_rate: u32) -> Option<u64> {
    let frame = (tick as f64 * f64::from(tempo_us) * f64::from(sample_rate))
        / (f64::from(ppq) * 1_000_000.0);
    (frame.is_finite() && frame >= 0.0 && frame <= u64::MAX as f64)
        .then_some(frame.round() as u64)
}

fn minimum_window_frames(sample_rate: u32, frequency_hz: f64) -> Option<usize> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return None;
    }
    // Require enough cycles to resolve a low fundamental, while retaining a
    // floor that gives useful bin spacing for ordinary orchestral pitches.
    let cycles = (9.0 * f64::from(sample_rate) / frequency_hz).ceil();
    let required = (AUDIO_EVENT_OBSERVATION_MIN_FFT_FRAMES_V1 as f64).max(cycles);
    if !required.is_finite() || required > AUDIO_EVENT_OBSERVATION_MAX_FFT_FRAMES_V1 as f64 {
        return None;
    }
    Some((required as usize).next_power_of_two())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        0.0
    } else if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

enum SpectralProbeResultV1 {
    Measured {
        harmonic_salience_db: f64,
        fundamental_salience_db: f64,
        harmonics_measured: u8,
    },
    NoMeasurableEnergy,
    NoResolvableHarmonics,
}

fn spectral_salience(
    wav: &DecodedWavV1,
    window_start: usize,
    window_frames: usize,
    frequency_hz: f64,
) -> SpectralProbeResultV1 {
    let sample_rate = f64::from(wav.info.sample_rate_hz);
    let channels = usize::from(wav.info.channels);
    let nyquist = sample_rate * 0.5;
    if frequency_hz >= nyquist {
        return SpectralProbeResultV1::NoResolvableHarmonics;
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_frames);
    let mut spectrum = vec![0.0f64; window_frames / 2 + 1];
    for channel in 0..channels {
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); window_frames];
        for (index, slot) in buffer.iter_mut().enumerate() {
            let frame = window_start + index;
            let sample = wav.interleaved_samples[frame * channels + channel];
            let hann = if window_frames <= 1 {
                1.0
            } else {
                0.5 - 0.5
                    * (std::f64::consts::TAU * index as f64 / (window_frames - 1) as f64).cos()
            };
            slot.re = sample * hann as f32;
        }
        fft.process(&mut buffer);
        for (bin, value) in spectrum.iter_mut().enumerate() {
            *value += f64::from(buffer[bin].norm()) / window_frames as f64 / channels as f64;
        }
    }
    let global_peak = spectrum.iter().copied().fold(0.0f64, f64::max);
    if global_peak <= 1e-12 {
        return SpectralProbeResultV1::NoMeasurableEnergy;
    }
    let noise_floor = (global_peak * 1e-8).max(1e-12);
    let mut saliences = Vec::new();
    for harmonic in 1..=AUDIO_EVENT_OBSERVATION_HARMONICS_V1 {
        let harmonic_hz = frequency_hz * harmonic as f64;
        if harmonic_hz >= nyquist * 0.98 {
            break;
        }
        let center = (harmonic_hz * window_frames as f64 / sample_rate).round() as usize;
        if center < 9 || center + 8 >= spectrum.len() {
            continue;
        }
        let peak = spectrum[center - 1..=center + 1]
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        let mut sidebands = Vec::with_capacity(12);
        sidebands.extend_from_slice(&spectrum[center - 8..center - 2]);
        sidebands.extend_from_slice(&spectrum[center + 3..=center + 8]);
        let local_floor = median(&mut sidebands).max(noise_floor);
        let ratio_db = 20.0 * ((peak + noise_floor) / (local_floor + noise_floor)).log10();
        saliences.push(ratio_db);
    }
    if saliences.is_empty() {
        return SpectralProbeResultV1::NoResolvableHarmonics;
    }
    let fundamental = saliences[0];
    let mean = saliences.iter().sum::<f64>() / saliences.len() as f64;
    SpectralProbeResultV1::Measured {
        harmonic_salience_db: mean,
        fundamental_salience_db: fundamental,
        harmonics_measured: saliences.len() as u8,
    }
}

fn observe_note(
    wav: &DecodedWavV1,
    note: &crate::midi_performance::MidiPerformanceNoteReceiptV1,
    tempo_us: u32,
    ppq: u16,
) -> AudioEventPitchObservationV1 {
    let frequency = 440.0 * 2.0f64.powf((f64::from(note.pitch_midi) - 69.0) / 12.0);
    let start = expected_frame(note.start_tick, tempo_us, ppq, wav.info.sample_rate_hz)
        .unwrap_or(u64::MAX);
    let end = expected_frame(note.end_tick, tempo_us, ppq, wav.info.sample_rate_hz)
        .unwrap_or(u64::MAX);
    let mut result = AudioEventPitchObservationV1 {
        source_note_ordinal: note.source_note_ordinal,
        part_id: note.part_id,
        segment_id: note.segment_id.clone(),
        instrument_id: note.instrument_id.clone(),
        pitch_midi: note.pitch_midi,
        expected_frequency_hz: frequency,
        expected_start_frame: start,
        expected_end_frame: end,
        window_start_frame: None,
        window_frame_count: None,
        harmonic_salience_db: None,
        fundamental_salience_db: None,
        harmonics_measured: 0,
        status: AudioEventObservationStatusV1::InsufficientDuration,
    };
    if end <= start {
        return result;
    }
    if end > wav.info.frame_count {
        result.status = AudioEventObservationStatusV1::TruncatedByAudio;
        return result;
    }
    if frequency >= f64::from(wav.info.sample_rate_hz) * 0.5 {
        result.status = AudioEventObservationStatusV1::FundamentalAboveNyquist;
        return result;
    }
    let Some(window_frames) = minimum_window_frames(wav.info.sample_rate_hz, frequency) else {
        result.status = AudioEventObservationStatusV1::InsufficientFrequencyResolution;
        return result;
    };
    let event_frames = (end - start) as usize;
    if event_frames < window_frames {
        result.status = AudioEventObservationStatusV1::InsufficientDuration;
        return result;
    }
    let window_start = start as usize + (event_frames - window_frames) / 2;
    result.window_start_frame = Some(window_start as u64);
    result.window_frame_count = Some(window_frames as u32);
    match spectral_salience(wav, window_start, window_frames, frequency) {
        SpectralProbeResultV1::Measured {
            harmonic_salience_db,
            fundamental_salience_db,
            harmonics_measured,
        } => {
            result.harmonic_salience_db = Some(harmonic_salience_db);
            result.fundamental_salience_db = Some(fundamental_salience_db);
            result.harmonics_measured = harmonics_measured;
            result.status = AudioEventObservationStatusV1::Measured;
        }
        SpectralProbeResultV1::NoMeasurableEnergy => {
            result.status = AudioEventObservationStatusV1::NoMeasurableSpectralEnergy;
        }
        SpectralProbeResultV1::NoResolvableHarmonics => {
            result.status = AudioEventObservationStatusV1::InsufficientFrequencyResolution;
        }
    }
    result
}

pub fn audio_event_observation_commitment_v1(
    receipt: &AudioEventObservationReceiptV1,
) -> Result<String, serde_json::Error> {
    let commitment = ReceiptCommitmentV1 {
        schema: &receipt.schema,
        score_sha256: &receipt.score_sha256,
        midi_sha256: &receipt.midi_sha256,
        midi_manifest_sha256: &receipt.midi_manifest_sha256,
        wav_sha256: &receipt.wav_sha256,
        wav_info: &receipt.wav_info,
        note_observations: &receipt.note_observations,
    };
    Ok(domain_separated_sha256(
        AUDIO_EVENT_OBSERVATION_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

/// Produce a pitch-energy observation receipt for the exact MIDI artifact and
/// WAV bytes supplied by the caller. Event timing is mapped from the MIDI
/// manifest's integer tempo/PPQ into sample frames. Notes too short for the
/// configured spectral resolution are explicitly marked as abstentions.
pub fn observe_rendered_audio_events_v1(
    wav_bytes: &[u8],
    midi: &MidiPerformanceArtifactV1,
) -> Result<AudioEventObservationReceiptV1, Vec<AudioEventObservationIssueV1>> {
    let issues = validate_midi_artifact(midi);
    if !issues.is_empty() {
        return Err(issues);
    }
    let wav = decode_wav(wav_bytes).map_err(|issue| vec![issue])?;
    let mut note_observations: Vec<_> = midi
        .manifest
        .notes
        .iter()
        .map(|note| {
            observe_note(
                &wav,
                note,
                midi.manifest.tempo_microseconds_per_quarter,
                midi.manifest.ticks_per_quarter,
            )
        })
        .collect();
    note_observations.sort_by_key(|note| note.source_note_ordinal);
    let mut receipt = AudioEventObservationReceiptV1 {
        schema: AUDIO_EVENT_OBSERVATION_SCHEMA_V1.into(),
        score_sha256: midi.manifest.score_sha256.clone(),
        midi_sha256: midi.manifest.midi_sha256.clone(),
        midi_manifest_sha256: midi.manifest_sha256.clone(),
        wav_sha256: sha256_hex(wav_bytes),
        wav_info: wav.info,
        note_observations,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = audio_event_observation_commitment_v1(&receipt)
        .map_err(|_| vec![AudioEventObservationIssueV1::ReceiptDigestMismatch])?;
    Ok(receipt)
}

/// Recheck receipt integrity and its bindings to the supplied exact inputs.
/// This validates commitments and note coverage; it does not re-run the FFT
/// or authenticate who produced the receipt.
pub fn validate_audio_event_observation_v1(
    receipt: &AudioEventObservationReceiptV1,
    wav_bytes: &[u8],
    midi: &MidiPerformanceArtifactV1,
) -> Vec<AudioEventObservationIssueV1> {
    let mut issues = validate_midi_artifact(midi);
    let decoded_wav = decode_wav(wav_bytes);
    if receipt.schema != AUDIO_EVENT_OBSERVATION_SCHEMA_V1
        || receipt.score_sha256 != midi.manifest.score_sha256
        || receipt.midi_sha256 != midi.manifest.midi_sha256
        || receipt.midi_manifest_sha256 != midi.manifest_sha256
        || receipt.wav_sha256 != sha256_hex(wav_bytes)
    {
        issues.push(AudioEventObservationIssueV1::ReceiptInputBindingMismatch);
    }
    match decoded_wav {
        Ok(decoded) if decoded.info == receipt.wav_info => {}
        _ => issues.push(AudioEventObservationIssueV1::ReceiptInputBindingMismatch),
    }
    let expected: BTreeMap<_, _> = midi
        .manifest
        .notes
        .iter()
        .map(|note| (note.source_note_ordinal, note))
        .collect();
    let observed_ordinals: BTreeSet<_> = receipt
        .note_observations
        .iter()
        .map(|note| note.source_note_ordinal)
        .collect();
    if expected.keys().copied().collect::<BTreeSet<_>>() != observed_ordinals
        || observed_ordinals.len() != receipt.note_observations.len()
    {
        issues.push(AudioEventObservationIssueV1::ReceiptNoteCoverageMismatch);
    }
    let wav_info = decoded_wav.as_ref().ok().map(|wav| &wav.info);
    for observation in &receipt.note_observations {
        let Some(source) = expected.get(&observation.source_note_ordinal) else {
            continue;
        };
        let frame_binding_matches = wav_info.is_some_and(|info| {
            expected_frame(
                source.start_tick,
                midi.manifest.tempo_microseconds_per_quarter,
                midi.manifest.ticks_per_quarter,
                info.sample_rate_hz,
            ) == Some(observation.expected_start_frame)
                && expected_frame(
                    source.end_tick,
                    midi.manifest.tempo_microseconds_per_quarter,
                    midi.manifest.ticks_per_quarter,
                    info.sample_rate_hz,
                ) == Some(observation.expected_end_frame)
        });
        let frequency = 440.0
            * 2.0f64.powf((f64::from(source.pitch_midi) - 69.0) / 12.0);
        let frequency_matches = observation.expected_frequency_hz.is_finite()
            && (observation.expected_frequency_hz - frequency).abs() <= 1e-9;
        let salience_is_finite = observation
            .harmonic_salience_db
            .into_iter()
            .chain(observation.fundamental_salience_db)
            .all(f64::is_finite);
        let window_is_well_formed = match (
            observation.window_start_frame,
            observation.window_frame_count,
        ) {
            (None, None) => true,
            (Some(start), Some(count)) => {
                count > 0
                    && start >= observation.expected_start_frame
                    && start.saturating_add(u64::from(count))
                        <= observation.expected_end_frame
                    && wav_info.is_some_and(|info| {
                        start.saturating_add(u64::from(count)) <= info.frame_count
                    })
            }
            _ => false,
        };
        if observation.part_id != source.part_id
            || observation.segment_id != source.segment_id
            || observation.instrument_id != source.instrument_id
            || observation.pitch_midi != source.pitch_midi
            || !frame_binding_matches
            || !frequency_matches
            || !salience_is_finite
            || !window_is_well_formed
            || observation.harmonics_measured > AUDIO_EVENT_OBSERVATION_HARMONICS_V1 as u8
            || (observation.status == AudioEventObservationStatusV1::Measured
                && (observation.harmonic_salience_db.is_none()
                    || observation.fundamental_salience_db.is_none()
                    || observation.window_start_frame.is_none()))
        {
            issues.push(AudioEventObservationIssueV1::ReceiptEventMismatch {
                ordinal: observation.source_note_ordinal,
            });
        }
    }
    match audio_event_observation_commitment_v1(receipt) {
        Ok(actual) if actual == receipt.receipt_sha256 => {}
        _ => issues.push(AudioEventObservationIssueV1::ReceiptDigestMismatch),
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_performance::{
        MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1, MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1, MIDI_PERFORMANCE_SCHEMA_V1,
        MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1, MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1,
        MidiPerformanceManifestV1, MidiPerformanceNoteReceiptV1, MidiPerformanceTrackReceiptV1,
    };

    fn midi_artifact(pitch: u8, end_tick: u64) -> MidiPerformanceArtifactV1 {
        let midi_bytes = b"deterministic-test-midi".to_vec();
        let mut manifest = MidiPerformanceManifestV1 {
            schema: MIDI_PERFORMANCE_SCHEMA_V1.into(),
            score_digest_schema: symthaea_music_theory::orchestration_timeline::ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: "a".repeat(64),
            timeline_digest_schema: MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1.into(),
            timeline_sha256: "b".repeat(64),
            binding_digest_schema: MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1.into(),
            binding_sha256: "c".repeat(64),
            capability_digest_schema: MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1.into(),
            capability_sha256: "d".repeat(64),
            midi_sha256: sha256_hex(&midi_bytes),
            ticks_per_quarter: MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1,
            tempo_microseconds_per_quarter: 500_000,
            meter_numerator: 4,
            meter_denominator: 4,
            tracks: vec![MidiPerformanceTrackReceiptV1 {
                track_index: 1,
                track_name: "Part 1 segment-a instrument-a".into(),
                part_id: 1,
                segment_id: "segment-a".into(),
                instrument_id: "instrument-a".into(),
                renderer_id: "test-renderer".into(),
                patch_id: "test-patch".into(),
                gm_program: 0,
                midi_channel: 0,
                segment_start_tick: 0,
                segment_end_tick: end_tick,
            }],
            notes: vec![MidiPerformanceNoteReceiptV1 {
                source_note_ordinal: 0,
                part_id: 1,
                segment_id: "segment-a".into(),
                instrument_id: "instrument-a".into(),
                track_index: 1,
                start_tick: 0,
                end_tick,
                pitch_midi: pitch,
                velocity_midi: 100,
            }],
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let manifest_sha256 =
            domain_separated_sha256(MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1, &bytes);
        MidiPerformanceArtifactV1 {
            midi_bytes,
            manifest,
            manifest_sha256,
        }
    }

    fn sine_wav(frequency_hz: f32, duration_secs: f32) -> Vec<u8> {
        let sample_rate = 40_960u32;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for frame in 0..(duration_secs * sample_rate as f32) as usize {
                let sample = 0.5
                    * (std::f32::consts::TAU * frequency_hz * frame as f32
                        / sample_rate as f32)
                        .sin();
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    fn silent_wav(duration_secs: f32) -> Vec<u8> {
        let sample_rate = 40_960u32;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for _ in 0..(duration_secs * sample_rate as f32) as usize {
                writer.write_sample(0.0f32).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn expected_pitch_produces_a_harmonic_energy_observation() {
        // At 40.96 kHz and a 4096-frame window, A4 is exactly FFT bin 44.
        let midi = midi_artifact(69, 1_920);
        let wav = sine_wav(440.0, 1.0);
        let receipt = observe_rendered_audio_events_v1(&wav, &midi).unwrap();
        assert_eq!(receipt.note_observations.len(), 1);
        let note = &receipt.note_observations[0];
        assert_eq!(note.status, AudioEventObservationStatusV1::Measured);
        assert!(note.harmonics_measured >= 1);
        assert!(
            note.fundamental_salience_db.unwrap() > 20.0,
            "expected a strong A4 spectral peak: {note:?}"
        );
        assert!(validate_audio_event_observation_v1(&receipt, &wav, &midi).is_empty());
    }

    #[test]
    fn off_pitch_audio_has_lower_target_fundamental_salience() {
        let midi = midi_artifact(69, 1_920);
        let correct = sine_wav(440.0, 1.0);
        let wrong = sine_wav(330.0, 1.0);
        let correct_receipt = observe_rendered_audio_events_v1(&correct, &midi).unwrap();
        let wrong_receipt = observe_rendered_audio_events_v1(&wrong, &midi).unwrap();
        let correct_db = correct_receipt.note_observations[0]
            .fundamental_salience_db
            .unwrap();
        let wrong_db = wrong_receipt.note_observations[0]
            .fundamental_salience_db
            .unwrap();
        assert!(
            correct_db > wrong_db + 20.0,
            "expected-pitch control={correct_db:.2} dB, off-pitch control={wrong_db:.2} dB"
        );
    }

    #[test]
    fn notes_shorter_than_the_required_window_abstain() {
        let midi = midi_artifact(69, 96); // 50 ms at 120 BPM.
        let wav = sine_wav(440.0, 1.0);
        let receipt = observe_rendered_audio_events_v1(&wav, &midi).unwrap();
        assert_eq!(
            receipt.note_observations[0].status,
            AudioEventObservationStatusV1::InsufficientDuration
        );
        assert!(receipt.note_observations[0].harmonic_salience_db.is_none());
    }

    #[test]
    fn midi_byte_tampering_is_rejected_before_observation() {
        let mut midi = midi_artifact(69, 1_920);
        midi.midi_bytes.push(0);
        let wav = sine_wav(440.0, 1.0);
        let issues = observe_rendered_audio_events_v1(&wav, &midi).unwrap_err();
        assert!(issues.contains(&AudioEventObservationIssueV1::MidiBytesDigestMismatch));
    }

    #[test]
    fn receipt_detects_wav_replacement_and_field_tampering() {
        let midi = midi_artifact(69, 1_920);
        let wav = sine_wav(440.0, 1.0);
        let mut receipt = observe_rendered_audio_events_v1(&wav, &midi).unwrap();
        let replacement = sine_wav(330.0, 1.0);
        assert!(
            validate_audio_event_observation_v1(&receipt, &replacement, &midi)
                .contains(&AudioEventObservationIssueV1::ReceiptInputBindingMismatch)
        );
        receipt.note_observations[0].pitch_midi = 70;
        assert!(
            validate_audio_event_observation_v1(&receipt, &wav, &midi)
                .contains(&AudioEventObservationIssueV1::ReceiptDigestMismatch)
        );
    }

    #[test]
    fn silent_audio_is_an_explicit_observation_abstention() {
        let midi = midi_artifact(69, 1_920);
        let silent = silent_wav(1.0);
        let receipt = observe_rendered_audio_events_v1(&silent, &midi).unwrap();
        assert_eq!(
            receipt.note_observations[0].status,
            AudioEventObservationStatusV1::NoMeasurableSpectralEnergy
        );
        assert!(receipt.note_observations[0].harmonic_salience_db.is_none());
    }

    #[test]
    fn non_finite_float_audio_is_rejected() {
        let mut cursor = Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 40_960,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            writer.write_sample(f32::NAN).unwrap();
            writer.finalize().unwrap();
        }
        let midi = midi_artifact(69, 1_920);
        assert_eq!(
            observe_rendered_audio_events_v1(&cursor.into_inner(), &midi).unwrap_err(),
            vec![AudioEventObservationIssueV1::NonFiniteAudioSample]
        );
    }
}
