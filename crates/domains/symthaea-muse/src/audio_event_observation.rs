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

pub const AUDIO_EVENT_BOUNDARY_SCHEMA_V1: &str =
    "muse-audio-event-boundary-observation-v1";
pub const AUDIO_EVENT_BOUNDARY_DEFAULT_SEARCH_MARGIN_MS_V1: u32 = 250;
pub const AUDIO_EVENT_BOUNDARY_MAX_SEARCH_MARGIN_MS_V1: u32 = 2_000;
pub const AUDIO_EVENT_BOUNDARY_ANALYSIS_WINDOW_MS_V1: u32 = 10;
pub const AUDIO_EVENT_BOUNDARY_ANALYSIS_HOP_MS_V1: u32 = 5;
pub const AUDIO_EVENT_BOUNDARY_MIN_CONTRAST_DB_V1: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEventBoundaryStatusV1 {
    Measured,
    InsufficientPreEventContext,
    InsufficientContrast,
    OnsetOutsideSearchWindow,
    ActivityEndOutsideSearchWindow,
    InsufficientOnsetEvidence,
    TruncatedByAudio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventBoundaryObservationV1 {
    pub source_note_ordinal: usize,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub expected_start_frame: u64,
    pub expected_end_frame: u64,
    pub search_start_frame: u64,
    pub search_end_frame: u64,
    pub observed_acoustic_activity_start_frame: Option<u64>,
    pub observed_acoustic_activity_end_frame: Option<u64>,
    /// Signed observed-start minus expected-MIDI-start, in sample frames.
    pub acoustic_onset_delta_frames: Option<i64>,
    /// Signed acoustic-activity-end minus expected-MIDI-end, not a note-off error.
    pub acoustic_activity_end_delta_frames: Option<i64>,
    pub baseline_rms: Option<f64>,
    pub peak_rms: Option<f64>,
    pub onset_contrast_db: Option<f64>,
    pub activity_end_contrast_db: Option<f64>,
    pub onset_status: AudioEventBoundaryStatusV1,
    pub activity_end_status: AudioEventBoundaryStatusV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventBoundaryObservationReceiptV1 {
    pub schema: String,
    pub detector_version: String,
    pub score_sha256: String,
    pub midi_sha256: String,
    pub midi_manifest_sha256: String,
    pub wav_sha256: String,
    pub wav_info: AudioEventObservationWavInfoV1,
    pub search_margin_ms: u32,
    pub analysis_window_frames: u32,
    pub analysis_hop_frames: u32,
    pub note_observations: Vec<AudioEventBoundaryObservationV1>,
    pub source_attribution_established: bool,
    pub limitations: Vec<String>,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct BoundaryReceiptCommitmentV1<'a> {
    schema: &'a str,
    detector_version: &'a str,
    score_sha256: &'a str,
    midi_sha256: &'a str,
    midi_manifest_sha256: &'a str,
    wav_sha256: &'a str,
    wav_info: &'a AudioEventObservationWavInfoV1,
    search_margin_ms: u32,
    analysis_window_frames: u32,
    analysis_hop_frames: u32,
    note_observations: &'a [AudioEventBoundaryObservationV1],
    source_attribution_established: bool,
    limitations: &'a [String],
}

#[derive(Debug, Clone, Copy)]
struct ActivityEnvelopeFrameV1 {
    center_frame: u64,
    rms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEventObservationIssueV1 {
    WrongMidiSchema,
    InvalidDigest { field: String },
    MidiBytesDigestMismatch,
    MidiManifestDigestMismatch,
    MidiBytesDoNotMatchManifest,
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
    InvalidBoundarySearchMargin,
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


fn midi_bytes_match_manifest(
    midi_bytes: &[u8],
    manifest: &crate::midi_performance::MidiPerformanceManifestV1,
) -> bool {
    use midly::{Format, MetaMessage, MidiMessage, Timing, TrackEventKind};

    let Ok(smf) = midly::Smf::parse(midi_bytes) else {
        return false;
    };
    if smf.header.format != Format::Parallel
        || !matches!(
            smf.header.timing,
            Timing::Metrical(ppq) if ppq.as_int() == manifest.ticks_per_quarter
        )
        || smf.tracks.len() != manifest.tracks.len() + 1
    {
        return false;
    }

    type NoteOnKey = (u16, u64, u8, u8, u8);
    type NoteOffKey = (u16, u64, u8, u8);
    type ProgramKey = (u16, u64, u8, u8);
    let mut expected_note_ons: BTreeMap<NoteOnKey, usize> = BTreeMap::new();
    let mut expected_note_offs: BTreeMap<NoteOffKey, usize> = BTreeMap::new();
    let mut expected_programs: BTreeMap<ProgramKey, usize> = BTreeMap::new();
    let track_receipts: BTreeMap<_, _> = manifest
        .tracks
        .iter()
        .map(|track| (track.track_index, track))
        .collect();
    if track_receipts.len() != manifest.tracks.len() {
        return false;
    }

    for track in &manifest.tracks {
        if track.track_index == 0
            || usize::from(track.track_index) >= smf.tracks.len()
            || track.midi_channel > 15
            || track.gm_program > 127
            || track.segment_end_tick <= track.segment_start_tick
        {
            return false;
        }
        *expected_programs
            .entry((
                track.track_index,
                track.segment_start_tick,
                track.midi_channel,
                track.gm_program,
            ))
            .or_insert(0) += 1;
    }

    for note in &manifest.notes {
        let Some(track) = track_receipts.get(&note.track_index) else {
            return false;
        };
        if track.part_id != note.part_id
            || track.segment_id != note.segment_id
            || track.instrument_id != note.instrument_id
            || note.start_tick < track.segment_start_tick
            || note.end_tick > track.segment_end_tick
            || note.end_tick <= note.start_tick
        {
            return false;
        }
        *expected_note_ons
            .entry((
                note.track_index,
                note.start_tick,
                track.midi_channel,
                note.pitch_midi,
                note.velocity_midi,
            ))
            .or_insert(0) += 1;
        *expected_note_offs
            .entry((
                note.track_index,
                note.end_tick,
                track.midi_channel,
                note.pitch_midi,
            ))
            .or_insert(0) += 1;
    }

    let mut actual_note_ons: BTreeMap<NoteOnKey, usize> = BTreeMap::new();
    let mut actual_note_offs: BTreeMap<NoteOffKey, usize> = BTreeMap::new();
    let mut actual_programs: BTreeMap<ProgramKey, usize> = BTreeMap::new();
    let mut tempo_events = 0usize;
    let mut meter_events = 0usize;

    for (index, events) in smf.tracks.iter().enumerate() {
        let Ok(track_index) = u16::try_from(index) else {
            return false;
        };
        let mut tick = 0u64;
        for event in events {
            let Some(next_tick) = tick.checked_add(u64::from(event.delta.as_int())) else {
                return false;
            };
            tick = next_tick;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    if track_index != 0
                        || tick != 0
                        || value.as_int() != manifest.tempo_microseconds_per_quarter
                    {
                        return false;
                    }
                    tempo_events += 1;
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(numerator, denominator, _, _)) => {
                    if track_index != 0
                        || tick != 0
                        || numerator != manifest.meter_numerator
                        || denominator != 2
                        || manifest.meter_denominator != 4
                    {
                        return false;
                    }
                    meter_events += 1;
                }
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            *actual_note_ons
                                .entry((
                                    track_index,
                                    tick,
                                    channel,
                                    key.as_int(),
                                    vel.as_int(),
                                ))
                                .or_insert(0) += 1;
                        }
                        MidiMessage::NoteOff { key, .. } => {
                            *actual_note_offs
                                .entry((track_index, tick, channel, key.as_int()))
                                .or_insert(0) += 1;
                        }
                        MidiMessage::NoteOn { key, .. } => {
                            *actual_note_offs
                                .entry((track_index, tick, channel, key.as_int()))
                                .or_insert(0) += 1;
                        }
                        MidiMessage::ProgramChange { program } => {
                            *actual_programs
                                .entry((track_index, tick, channel, program.as_int()))
                                .or_insert(0) += 1;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    tempo_events == 1
        && meter_events == 1
        && actual_note_ons == expected_note_ons
        && actual_note_offs == expected_note_offs
        && actual_programs == expected_programs
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
    if !midi_bytes_match_manifest(&artifact.midi_bytes, manifest) {
        issues.push(AudioEventObservationIssueV1::MidiBytesDoNotMatchManifest);
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
    if ppq == 0 {
        return None;
    }
    // Use checked integer arithmetic so long renders do not accumulate
    // floating-point timing drift. The final conversion rounds to the nearest
    // sample frame, with exact half-frame ties rounded upward.
    let numerator = u128::from(tick)
        .checked_mul(u128::from(tempo_us))?
        .checked_mul(u128::from(sample_rate))?;
    let denominator = u128::from(ppq).checked_mul(1_000_000)?;
    let rounded = numerator.checked_add(denominator / 2)? / denominator;
    u64::try_from(rounded).ok()
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

fn signed_frame_delta(observed: u64, expected: u64) -> Option<i64> {
    if observed >= expected {
        i64::try_from(observed - expected).ok()
    } else {
        i64::try_from(expected - observed).ok().map(|delta| -delta)
    }
}

fn activity_envelope_v1(
    wav: &DecodedWavV1,
    window_frames: usize,
    hop_frames: usize,
) -> Vec<ActivityEnvelopeFrameV1> {
    let channels = usize::from(wav.info.channels);
    let frame_count = wav.info.frame_count as usize;
    if channels == 0 || window_frames == 0 || hop_frames == 0 || frame_count < window_frames {
        return Vec::new();
    }

    let mut envelope = Vec::with_capacity((frame_count - window_frames) / hop_frames + 1);
    for start in (0..=frame_count - window_frames).step_by(hop_frames) {
        let mut energy = 0.0_f64;
        for frame in start..start + window_frames {
            let base = frame * channels;
            for channel in 0..channels {
                let sample = f64::from(wav.interleaved_samples[base + channel]);
                energy += sample * sample;
            }
        }
        let sample_count = (window_frames * channels) as f64;
        envelope.push(ActivityEnvelopeFrameV1 {
            center_frame: (start + window_frames / 2) as u64,
            rms: (energy / sample_count).sqrt(),
        });
    }
    envelope
}

fn median_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    })
}

fn amplitude_contrast_db(peak: f64, baseline: f64) -> f64 {
    20.0 * ((peak + 1e-9) / (baseline + 1e-9)).log10()
}

fn observe_activity_boundaries_v1(
    envelope: &[ActivityEnvelopeFrameV1],
    wav_frame_count: u64,
    note: &crate::midi_performance::MidiPerformanceNoteReceiptV1,
    tempo_us: u32,
    ppq: u16,
    sample_rate_hz: u32,
    search_margin_ms: u32,
    analysis_window_frames: u32,
    analysis_hop_frames: u32,
) -> AudioEventBoundaryObservationV1 {
    let expected_start = expected_frame(note.start_tick, tempo_us, ppq, sample_rate_hz)
        .unwrap_or(u64::MAX);
    let expected_end = expected_frame(note.end_tick, tempo_us, ppq, sample_rate_hz)
        .unwrap_or(u64::MAX);
    let margin_numerator = u64::from(sample_rate_hz) * u64::from(search_margin_ms);
    let margin_frames = margin_numerator.div_ceil(1_000);
    let half_margin = margin_frames / 2;
    let search_start = expected_start.saturating_sub(margin_frames);
    let search_end = expected_end.saturating_add(margin_frames);
    let observed_search_end = search_end.min(wav_frame_count);

    let mut result = AudioEventBoundaryObservationV1 {
        source_note_ordinal: note.source_note_ordinal,
        part_id: note.part_id,
        segment_id: note.segment_id.clone(),
        instrument_id: note.instrument_id.clone(),
        expected_start_frame: expected_start,
        expected_end_frame: expected_end,
        search_start_frame: search_start,
        search_end_frame: search_end,
        observed_acoustic_activity_start_frame: None,
        observed_acoustic_activity_end_frame: None,
        acoustic_onset_delta_frames: None,
        acoustic_activity_end_delta_frames: None,
        baseline_rms: None,
        peak_rms: None,
        onset_contrast_db: None,
        activity_end_contrast_db: None,
        onset_status: AudioEventBoundaryStatusV1::InsufficientPreEventContext,
        activity_end_status: AudioEventBoundaryStatusV1::InsufficientPreEventContext,
    };

    if expected_start == u64::MAX
        || expected_end == u64::MAX
        || expected_end <= expected_start
        || expected_end > wav_frame_count
    {
        result.onset_status = AudioEventBoundaryStatusV1::TruncatedByAudio;
        result.activity_end_status = AudioEventBoundaryStatusV1::TruncatedByAudio;
        return result;
    }

    // The first half of the bounded pre-event margin establishes a local
    // baseline. If there is not enough pre-event audio, do not invent an onset.
    if expected_start < margin_frames {
        result.onset_status = AudioEventBoundaryStatusV1::InsufficientPreEventContext;
        result.activity_end_status = AudioEventBoundaryStatusV1::InsufficientPreEventContext;
        return result;
    }
    let baseline_start = expected_start - margin_frames;
    let baseline_end = expected_start - half_margin;
    let baseline_values: Vec<_> = envelope
        .iter()
        .filter(|frame| {
            frame.center_frame >= baseline_start && frame.center_frame < baseline_end
        })
        .map(|frame| frame.rms)
        .collect();
    let Some(baseline) = median_f64(&baseline_values) else {
        return result;
    };
    if baseline_values.len() < 3 {
        return result;
    }
    result.baseline_rms = Some(baseline);

    let onset_search_start = expected_start.saturating_sub(half_margin);
    let onset_search_end = expected_start.saturating_add(half_margin);
    let region_peak = envelope
        .iter()
        .filter(|frame| {
            frame.center_frame >= onset_search_start
                && frame.center_frame <= observed_search_end
        })
        .map(|frame| frame.rms)
        .fold(0.0_f64, f64::max);
    result.peak_rms = Some(region_peak);
    let contrast_db = amplitude_contrast_db(region_peak, baseline);
    result.onset_contrast_db = Some(contrast_db);
    result.activity_end_contrast_db = Some(contrast_db);

    if contrast_db < AUDIO_EVENT_BOUNDARY_MIN_CONTRAST_DB_V1 {
        result.onset_status = AudioEventBoundaryStatusV1::InsufficientContrast;
        result.activity_end_status = AudioEventBoundaryStatusV1::InsufficientContrast;
        return result;
    }

    let threshold = baseline + 0.2 * (region_peak - baseline);
    let onset_candidates: Vec<_> = envelope
        .iter()
        .filter(|frame| {
            frame.center_frame >= onset_search_start
                && frame.center_frame <= onset_search_end
        })
        .collect();
    let mut onset_index = None;
    for index in 0..onset_candidates.len().saturating_sub(1) {
        if onset_candidates[index].rms >= threshold
            && onset_candidates[index + 1].rms >= threshold
        {
            onset_index = Some(index);
            break;
        }
    }

    let Some(onset_index) = onset_index else {
        result.onset_status = AudioEventBoundaryStatusV1::OnsetOutsideSearchWindow;
        result.activity_end_status = AudioEventBoundaryStatusV1::InsufficientOnsetEvidence;
        return result;
    };
    let observed_start = onset_candidates[onset_index].center_frame;
    result.observed_acoustic_activity_start_frame = Some(observed_start);
    result.acoustic_onset_delta_frames = signed_frame_delta(observed_start, expected_start);
    result.onset_status = AudioEventBoundaryStatusV1::Measured;

    // Acoustic activity end is the first sustained drop below the local
    // threshold. It may precede MIDI NoteOff for a naturally decaying sound,
    // or follow it due to release/reverb. It is not a symbolic note-off estimate.
    let end_candidates: Vec<_> = envelope
        .iter()
        .filter(|frame| {
            frame.center_frame >= observed_start
                && frame.center_frame <= observed_search_end
        })
        .collect();
    let mut end_index = None;
    for index in 0..end_candidates.len().saturating_sub(2) {
        if end_candidates[index].rms < threshold
            && end_candidates[index + 1].rms < threshold
            && end_candidates[index + 2].rms < threshold
        {
            end_index = Some(index);
            break;
        }
    }

    if let Some(index) = end_index {
        let observed_end = end_candidates[index].center_frame;
        result.observed_acoustic_activity_end_frame = Some(observed_end);
        result.acoustic_activity_end_delta_frames = signed_frame_delta(observed_end, expected_end);
        result.activity_end_status = AudioEventBoundaryStatusV1::Measured;
    } else if search_end > wav_frame_count {
        result.activity_end_status = AudioEventBoundaryStatusV1::TruncatedByAudio;
    } else {
        result.activity_end_status = AudioEventBoundaryStatusV1::ActivityEndOutsideSearchWindow;
    }
    let _ = analysis_window_frames;
    let _ = analysis_hop_frames;
    result
}

/// Domain-separated commitment for the independently recomputable boundary
/// observation. The unkeyed digest is a tamper-evident commitment, not a signature.
pub fn audio_event_boundary_observation_commitment_v1(
    receipt: &AudioEventBoundaryObservationReceiptV1,
) -> Result<String, serde_json::Error> {
    let commitment = BoundaryReceiptCommitmentV1 {
        schema: &receipt.schema,
        detector_version: &receipt.detector_version,
        score_sha256: &receipt.score_sha256,
        midi_sha256: &receipt.midi_sha256,
        midi_manifest_sha256: &receipt.midi_manifest_sha256,
        wav_sha256: &receipt.wav_sha256,
        wav_info: &receipt.wav_info,
        search_margin_ms: receipt.search_margin_ms,
        analysis_window_frames: receipt.analysis_window_frames,
        analysis_hop_frames: receipt.analysis_hop_frames,
        note_observations: &receipt.note_observations,
        source_attribution_established: receipt.source_attribution_established,
        limitations: &receipt.limitations,
    };
    Ok(domain_separated_sha256(
        AUDIO_EVENT_BOUNDARY_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

/// Observe acoustic-activity boundaries around each expected MIDI event.
/// This emits waveform-activity diagnostics only: it does not establish
/// source-note/PartId attribution or exact symbolic note-off time.
pub fn observe_rendered_audio_event_boundaries_v1(
    wav_bytes: &[u8],
    midi: &MidiPerformanceArtifactV1,
    search_margin_ms: u32,
) -> Result<AudioEventBoundaryObservationReceiptV1, Vec<AudioEventObservationIssueV1>> {
    if search_margin_ms == 0 || search_margin_ms > AUDIO_EVENT_BOUNDARY_MAX_SEARCH_MARGIN_MS_V1 {
        return Err(vec![AudioEventObservationIssueV1::InvalidBoundarySearchMargin]);
    }
    let midi_issues = validate_midi_artifact(midi);
    if !midi_issues.is_empty() {
        return Err(midi_issues);
    }
    let wav = decode_wav(wav_bytes).map_err(|issue| vec![issue])?;
    let window_frames = (u64::from(wav.info.sample_rate_hz)
        * u64::from(AUDIO_EVENT_BOUNDARY_ANALYSIS_WINDOW_MS_V1)
        / 1_000)
        .max(64) as usize;
    let hop_frames = (u64::from(wav.info.sample_rate_hz)
        * u64::from(AUDIO_EVENT_BOUNDARY_ANALYSIS_HOP_MS_V1)
        / 1_000)
        .max(1) as usize;
    let envelope = activity_envelope_v1(&wav, window_frames, hop_frames);

    let mut note_observations: Vec<_> = midi
        .manifest
        .notes
        .iter()
        .map(|note| {
            observe_activity_boundaries_v1(
                &envelope,
                wav.info.frame_count,
                note,
                midi.manifest.tempo_microseconds_per_quarter,
                midi.manifest.ticks_per_quarter,
                wav.info.sample_rate_hz,
                search_margin_ms,
                window_frames as u32,
                hop_frames as u32,
            )
        })
        .collect();
    note_observations.sort_by_key(|item| item.source_note_ordinal);

    let mut receipt = AudioEventBoundaryObservationReceiptV1 {
        schema: AUDIO_EVENT_BOUNDARY_SCHEMA_V1.into(),
        detector_version: "muse-acoustic-activity-envelope-v1".into(),
        score_sha256: midi.manifest.score_sha256.clone(),
        midi_sha256: midi.manifest.midi_sha256.clone(),
        midi_manifest_sha256: midi.manifest_sha256.clone(),
        wav_sha256: sha256_hex(wav_bytes),
        wav_info: wav.info,
        search_margin_ms,
        analysis_window_frames: window_frames as u32,
        analysis_hop_frames: hop_frames as u32,
        note_observations,
        source_attribution_established: false,
        limitations: vec![
            "This is an acoustic activity envelope, not an exact symbolic note-on or note-off detector.".into(),
            "A decaying/plucked sound may become inactive before MIDI NoteOff; a release/reverb tail may remain after MIDI NoteOff.".into(),
            "A mixed WAV can contain overlapping events from other parts; source-note and PartId attribution are not established.".into(),
            "The relative RMS threshold is a deterministic diagnostic heuristic, not a calibrated probability or universal audibility threshold.".into(),
        ],
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = audio_event_boundary_observation_commitment_v1(&receipt)
        .map_err(|_| vec![AudioEventObservationIssueV1::ReceiptDigestMismatch])?;
    Ok(receipt)
}

/// Recompute the complete boundary analysis from the exact WAV and MIDI
/// inputs. Receipt-only edits cannot be promoted by resealing the public hash.
pub fn validate_audio_event_boundary_observation_v1(
    receipt: &AudioEventBoundaryObservationReceiptV1,
    wav_bytes: &[u8],
    midi: &MidiPerformanceArtifactV1,
) -> Vec<AudioEventObservationIssueV1> {
    let mut issues = Vec::new();
    match audio_event_boundary_observation_commitment_v1(receipt) {
        Ok(commitment) if commitment == receipt.receipt_sha256 => {}
        _ => issues.push(AudioEventObservationIssueV1::ReceiptDigestMismatch),
    }
    match observe_rendered_audio_event_boundaries_v1(wav_bytes, midi, receipt.search_margin_ms) {
        Ok(expected) => {
            if expected.wav_sha256 != receipt.wav_sha256
                || expected.midi_sha256 != receipt.midi_sha256
                || expected.midi_manifest_sha256 != receipt.midi_manifest_sha256
                || expected.wav_info != receipt.wav_info
            {
                issues.push(AudioEventObservationIssueV1::ReceiptInputBindingMismatch);
            }
            if expected.note_observations != receipt.note_observations
                || expected.source_attribution_established != receipt.source_attribution_established
                || expected.limitations != receipt.limitations
                || expected.detector_version != receipt.detector_version
            {
                issues.push(AudioEventObservationIssueV1::ReceiptEventMismatch { ordinal: usize::MAX });
            }
        }
        Err(mut errors) => issues.append(&mut errors),
    }
    issues
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
    match &decoded_wav {
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

    fn variable_length_quantity(value: u64) -> Vec<u8> {
        let mut bytes = [0u8; 10];
        let mut index = bytes.len() - 1;
        bytes[index] = (value & 0x7f) as u8;
        let mut remaining = value >> 7;
        while remaining > 0 {
            index -= 1;
            bytes[index] = ((remaining & 0x7f) as u8) | 0x80;
            remaining >>= 7;
        }
        bytes[index..].to_vec()
    }

    fn test_midi_bytes(pitch: u8, end_tick: u64) -> Vec<u8> {
        let mut tempo_track = vec![
            0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20, // 500,000 us/quarter
            0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08, // 4/4
            0x00, 0xff, 0x2f, 0x00,
        ];
        let mut note_track = vec![
            0x00, 0xc0, 0x00, // GM program 0 on channel 0
            0x00, 0x90, pitch, 100, // NoteOn at tick zero
        ];
        note_track.extend(variable_length_quantity(end_tick));
        note_track.extend_from_slice(&[0x80, pitch, 0x00]); // NoteOff
        note_track.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes()); // MIDI Type 1
        bytes.extend_from_slice(&2u16.to_be_bytes()); // tempo + note tracks
        bytes.extend_from_slice(&960u16.to_be_bytes());
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(tempo_track.len() as u32).to_be_bytes());
        bytes.append(&mut tempo_track);
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(note_track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&note_track);
        bytes
    }

    fn midi_artifact(pitch: u8, end_tick: u64) -> MidiPerformanceArtifactV1 {
        let midi_bytes = test_midi_bytes(pitch, end_tick);
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
    fn midi_tick_to_sample_frame_mapping_uses_integer_rounding() {
        assert_eq!(expected_frame(960, 500_000, 960, 48_000), Some(24_000));
        assert_eq!(expected_frame(1, 500_000, 960, 44_100), Some(23));
        assert_eq!(expected_frame(1, 500_000, 0, 44_100), None);
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
    fn octave_harmonic_collision_can_mimic_expected_pitch_energy() {
        // C5 contains a strong second harmonic for C4. The probe can observe
        // harmonic energy at C4 even though the C4 fundamental is absent.
        // This is an intentional adversarial fixture: harmonic salience must
        // never be interpreted as proof that the expected note was played.
        let midi = midi_artifact(60, 1_920);
        let wav = sine_wav(523.2511, 1.0);
        let receipt = observe_rendered_audio_events_v1(&wav, &midi).unwrap();
        let note = &receipt.note_observations[0];
        assert_eq!(note.status, AudioEventObservationStatusV1::Measured);
        assert!(
            note.harmonic_salience_db.unwrap()
                > note.fundamental_salience_db.unwrap() + 6.0,
            "octave energy should raise harmonic salience without a C4 fundamental: {note:?}"
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
    fn midi_file_note_events_must_match_manifest_even_after_resealing_digests() {
        let mut midi = midi_artifact(69, 1_920);
        let marker = [0x90, 69, 100];
        let position = midi
            .midi_bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap();
        midi.midi_bytes[position + 1] = 70;
        midi.manifest.midi_sha256 = sha256_hex(&midi.midi_bytes);
        let manifest_bytes = serde_json::to_vec(&midi.manifest).unwrap();
        midi.manifest_sha256 = domain_separated_sha256(
            MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
            &manifest_bytes,
        );
        let wav = sine_wav(440.0, 1.0);
        let issues = observe_rendered_audio_events_v1(&wav, &midi).unwrap_err();
        assert!(issues.contains(&AudioEventObservationIssueV1::MidiBytesDoNotMatchManifest));
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
        receipt.receipt_sha256 = audio_event_observation_commitment_v1(&receipt).unwrap();
        assert!(
            validate_audio_event_observation_v1(&receipt, &wav, &midi)
                .contains(&AudioEventObservationIssueV1::ReceiptEventMismatch { ordinal: 0 })
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
