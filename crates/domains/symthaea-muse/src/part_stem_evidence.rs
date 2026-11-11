// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed admission of source-bound, part-isolated Muse audio stems.
//! This validates artifact structure and byte identity, not renderer execution
//! or acoustic/perceptual fidelity.

use crate::midi_performance::{
    MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1, MIDI_PERFORMANCE_SCHEMA_V1,
    MidiPerformanceArtifactV1, MidiPerformanceManifestV1, MidiPerformanceTrackReceiptV1,
};
use hound::{SampleFormat, WavReader};
use midly::{Format, MetaMessage, MidiMessage, Timing, TrackEventKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

pub const PART_STEM_EVIDENCE_SCHEMA_V1: &str = "muse-part-stem-evidence-v1";
pub const PART_STEM_BUNDLE_DIGEST_SCHEMA_V1: &str =
    "muse-part-stem-evidence-serde-json-sha256-v1";
pub const PART_STEM_TRACK_EVENT_SET_SCHEMA_V1: &str =
    "muse-midi-track-event-set-json-sha256-v1";
pub const PART_STEM_RECEIPT_DIGEST_SCHEMA_V1: &str =
    "muse-part-stem-validation-receipt-serde-json-sha256-v1";

pub const PART_STEM_MAX_BUNDLE_BYTES_V1: usize = 16 * 1024 * 1024;
pub const PART_STEM_MAX_STEMS_V1: usize = 256;
pub const PART_STEM_MAX_NOTES_V1: usize = 100_000;
pub const PART_STEM_MAX_WAV_BYTES_V1: usize = 256 * 1024 * 1024;
pub const PART_STEM_MAX_TOTAL_WAV_BYTES_V1: u64 = 512 * 1024 * 1024;
pub const PART_STEM_MAX_AUDIO_FRAMES_V1: u64 = 24_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PartStemSampleFormatV1 {
    PcmInteger,
    PcmFloat32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartStemAudioClockV1 {
    pub origin: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_format: PartStemSampleFormatV1,
    pub bits_per_sample: u16,
    pub frame_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartStemRecordV1 {
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub midi_track_index: u16,
    pub render_method: String,
    pub renderer_id: String,
    pub renderer_binary_sha256: String,
    pub patch_id: String,
    pub patch_asset_sha256: String,
    pub render_config_sha256: String,
    pub track_event_set_sha256: String,
    pub source_note_ordinals: Vec<usize>,
    pub wav_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartStemEvidenceBundleV1 {
    pub schema: String,
    pub score_sha256: String,
    pub midi_sha256: String,
    pub midi_manifest_sha256: String,
    pub audio_clock: PartStemAudioClockV1,
    pub stems: Vec<PartStemRecordV1>,
    pub bundle_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartStemWavInputV1 {
    pub wav_sha256: String,
    pub wav_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartStemEvidenceIssueV1 {
    BundleTooLarge,
    WrongBundleSchema,
    InvalidDigest { field: String },
    BundleCommitmentMismatch,
    TooManyStems,
    TooManyNotes,
    EmptyStemSet,
    InvalidAudioClock,
    InvalidIdentity { field: String },
    InvalidRenderMethod { track_index: u16 },
    DuplicateTrackIndex { track_index: u16 },
    DuplicateStemIdentity { track_index: u16 },
    DuplicatePartSegment { part_id: u16, segment_id: String },
    MissingStem { track_index: u16 },
    UnexpectedStem { track_index: u16 },
    StemIdentityMismatch { track_index: u16 },
    SourceOrdinalCoverageMismatch { track_index: u16 },
    InvalidSourceOrdinalOrder { track_index: u16 },
    TrackEventSetMismatch { track_index: u16 },
    MidiBytesDigestMismatch,
    MidiManifestDigestMismatch,
    WrongMidiSchema,
    InvalidMidiHeader,
    InvalidMidiTempoOrMeter,
    MidiTrackCountMismatch,
    InvalidMidiEvent { track_index: u16 },
    MidiNoteSetMismatch { track_index: u16 },
    MidiProgramSetMismatch { track_index: u16 },
    InvalidNoteReceipt { ordinal: usize },
    DuplicateSourceOrdinal { ordinal: usize },
    WavMissing { sha256: String },
    DuplicateWavInput { sha256: String },
    WavTooLarge { sha256: String },
    TotalWavBytesExceeded,
    WavDigestMismatch { sha256: String },
    InvalidWav { sha256: String },
    NonFiniteAudioSample { sha256: String },
    WavSampleCountMismatch { sha256: String },
    WavClockMismatch { sha256: String },
    TooManyAudioFrames { sha256: String },
    BundleSerialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartStemValidationStatusV1 {
    StructurallyConsistentUnattested,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendererExecutionEvidenceV1 {
    DeclaredIdentityOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartStemValidationReceiptV1 {
    pub schema: String,
    pub status: PartStemValidationStatusV1,
    pub renderer_execution_evidence: RendererExecutionEvidenceV1,
    pub bundle_sha256: String,
    pub score_sha256: String,
    pub midi_sha256: String,
    pub stem_count: usize,
    pub total_wav_bytes: u64,
    pub common_frame_count: u64,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct BundleCommitmentV1<'a> {
    schema: &'a str,
    score_sha256: &'a str,
    midi_sha256: &'a str,
    midi_manifest_sha256: &'a str,
    audio_clock: &'a PartStemAudioClockV1,
    stems: &'a [PartStemRecordV1],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct TrackEventNoteV1 {
    source_note_ordinal: usize,
    pitch_midi: u8,
    velocity_midi: u8,
    start_tick: u64,
    end_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct TrackEventProgramChangeV1 {
    tick: u64,
    gm_program: u8,
}

#[derive(Serialize)]
struct TrackEventSetCommitmentV1<'a> {
    schema: &'static str,
    score_sha256: &'a str,
    part_id: u16,
    segment_id: &'a str,
    instrument_id: &'a str,
    midi_track_index: u16,
    notes: &'a [TrackEventNoteV1],
    program_changes: &'a [TrackEventProgramChangeV1],
}

#[derive(Serialize)]
struct ReceiptCommitmentV1<'a> {
    schema: &'a str,
    status: PartStemValidationStatusV1,
    renderer_execution_evidence: RendererExecutionEvidenceV1,
    bundle_sha256: &'a str,
    score_sha256: &'a str,
    midi_sha256: &'a str,
    stem_count: usize,
    total_wav_bytes: u64,
    common_frame_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StemIdentityV1 {
    part_id: u16,
    segment_id: String,
    instrument_id: String,
    track_index: u16,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn domain_sha256(domain: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
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

fn non_blank(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 255
}

fn stem_identity(stem: &PartStemRecordV1) -> StemIdentityV1 {
    StemIdentityV1 {
        part_id: stem.part_id,
        segment_id: stem.segment_id.clone(),
        instrument_id: stem.instrument_id.clone(),
        track_index: stem.midi_track_index,
    }
}

fn track_identity(track: &MidiPerformanceTrackReceiptV1) -> StemIdentityV1 {
    StemIdentityV1 {
        part_id: track.part_id,
        segment_id: track.segment_id.clone(),
        instrument_id: track.instrument_id.clone(),
        track_index: track.track_index,
    }
}

/// Bundle commitment. Stem order is canonicalized by identity before hashing.
pub fn part_stem_bundle_commitment_v1(
    bundle: &PartStemEvidenceBundleV1,
) -> Result<String, PartStemEvidenceIssueV1> {
    let mut stems = bundle.stems.clone();
    stems.sort_by_key(stem_identity);
    let commitment = BundleCommitmentV1 {
        schema: &bundle.schema,
        score_sha256: &bundle.score_sha256,
        midi_sha256: &bundle.midi_sha256,
        midi_manifest_sha256: &bundle.midi_manifest_sha256,
        audio_clock: &bundle.audio_clock,
        stems: &stems,
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|_| PartStemEvidenceIssueV1::BundleSerialization)?;
    Ok(domain_sha256(PART_STEM_BUNDLE_DIGEST_SCHEMA_V1, &bytes))
}

/// Parse a bounded JSON manifest. All declared structures reject unknown fields.
pub fn parse_part_stem_bundle_v1(
    bytes: &[u8],
) -> Result<PartStemEvidenceBundleV1, PartStemEvidenceIssueV1> {
    if bytes.len() > PART_STEM_MAX_BUNDLE_BYTES_V1 {
        return Err(PartStemEvidenceIssueV1::BundleTooLarge);
    }
    serde_json::from_slice(bytes).map_err(|_| PartStemEvidenceIssueV1::BundleSerialization)
}

fn track_notes(
    manifest: &MidiPerformanceManifestV1,
    track_index: u16,
) -> Vec<TrackEventNoteV1> {
    let mut notes: Vec<_> = manifest.notes.iter()
        .filter(|note| note.track_index == track_index)
        .map(|note| TrackEventNoteV1 {
            source_note_ordinal: note.source_note_ordinal,
            pitch_midi: note.pitch_midi,
            velocity_midi: note.velocity_midi,
            start_tick: note.start_tick,
            end_tick: note.end_tick,
        })
        .collect();
    notes.sort();
    notes
}

fn verify_midi(
    midi: &MidiPerformanceArtifactV1,
) -> Result<BTreeMap<u16, Vec<TrackEventProgramChangeV1>>, Vec<PartStemEvidenceIssueV1>> {
    let manifest = &midi.manifest;
    let mut issues = Vec::new();
    if manifest.schema != MIDI_PERFORMANCE_SCHEMA_V1
        || manifest.ticks_per_quarter == 0
        || manifest.tempo_microseconds_per_quarter == 0
        || manifest.tempo_microseconds_per_quarter > 16_777_215
        || manifest.meter_numerator == 0
        || manifest.meter_denominator != 4
    {
        issues.push(PartStemEvidenceIssueV1::WrongMidiSchema);
    }
    for (field, value) in [
        ("score_sha256", manifest.score_sha256.as_str()),
        ("timeline_sha256", manifest.timeline_sha256.as_str()),
        ("binding_sha256", manifest.binding_sha256.as_str()),
        ("capability_sha256", manifest.capability_sha256.as_str()),
        ("midi_sha256", manifest.midi_sha256.as_str()),
        ("manifest_sha256", midi.manifest_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            issues.push(PartStemEvidenceIssueV1::InvalidDigest { field: field.into() });
        }
    }
    if sha256_hex(&midi.midi_bytes) != manifest.midi_sha256 {
        issues.push(PartStemEvidenceIssueV1::MidiBytesDigestMismatch);
    }
    let manifest_bytes = serde_json::to_vec(manifest)
        .map_err(|_| vec![PartStemEvidenceIssueV1::BundleSerialization])?;
    if domain_sha256(MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1, &manifest_bytes)
        != midi.manifest_sha256
    {
        issues.push(PartStemEvidenceIssueV1::MidiManifestDigestMismatch);
    }

    let smf = midly::Smf::parse(&midi.midi_bytes)
        .map_err(|_| vec![PartStemEvidenceIssueV1::InvalidMidiHeader])?;
    let ppq_matches = matches!(
        smf.header.timing,
        Timing::Metrical(ppq) if ppq.as_int() == manifest.ticks_per_quarter
    );
    if smf.header.format != Format::Parallel || !ppq_matches {
        issues.push(PartStemEvidenceIssueV1::InvalidMidiHeader);
    }
    if smf.tracks.len() != manifest.tracks.len() + 1 {
        issues.push(PartStemEvidenceIssueV1::MidiTrackCountMismatch);
    }

    let mut track_map = BTreeMap::new();
    let mut track_ids = BTreeSet::new();
    let mut part_segments = BTreeSet::new();
    for track in &manifest.tracks {
        if track.track_index == 0
            || usize::from(track.track_index) >= smf.tracks.len()
            || track.midi_channel > 15
            || track.gm_program > 127
            || track.segment_end_tick <= track.segment_start_tick
            || !non_blank(&track.segment_id)
            || !non_blank(&track.instrument_id)
        {
            issues.push(PartStemEvidenceIssueV1::InvalidNoteReceipt {
                ordinal: usize::from(track.track_index),
            });
        }
        if track_map.insert(track.track_index, track).is_some() {
            issues.push(PartStemEvidenceIssueV1::DuplicateTrackIndex {
                track_index: track.track_index,
            });
        }
        if !track_ids.insert(track_identity(track)) {
            issues.push(PartStemEvidenceIssueV1::DuplicateStemIdentity {
                track_index: track.track_index,
            });
        }
        if !part_segments.insert((track.part_id, track.segment_id.clone())) {
            issues.push(PartStemEvidenceIssueV1::DuplicatePartSegment {
                part_id: track.part_id,
                segment_id: track.segment_id.clone(),
            });
        }
    }
    if manifest.notes.is_empty() || manifest.notes.len() > PART_STEM_MAX_NOTES_V1 {
        issues.push(PartStemEvidenceIssueV1::TooManyNotes);
    }
    let mut ordinals = BTreeSet::new();
    let mut notes_by_track: BTreeMap<u16, Vec<(u64, u8, u8)>> = BTreeMap::new();
    let mut offs_by_track: BTreeMap<u16, Vec<(u64, u8)>> = BTreeMap::new();
    for note in &manifest.notes {
        if !ordinals.insert(note.source_note_ordinal) {
            issues.push(PartStemEvidenceIssueV1::DuplicateSourceOrdinal {
                ordinal: note.source_note_ordinal,
            });
        }
        if note.pitch_midi > 127
            || note.velocity_midi == 0
            || note.velocity_midi > 127
            || note.end_tick <= note.start_tick
            || !non_blank(&note.segment_id)
            || !non_blank(&note.instrument_id)
        {
            issues.push(PartStemEvidenceIssueV1::InvalidNoteReceipt {
                ordinal: note.source_note_ordinal,
            });
        }
        let Some(track) = track_map.get(&note.track_index) else {
            issues.push(PartStemEvidenceIssueV1::InvalidNoteReceipt {
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
            issues.push(PartStemEvidenceIssueV1::InvalidNoteReceipt {
                ordinal: note.source_note_ordinal,
            });
        }
        notes_by_track.entry(note.track_index).or_default()
            .push((note.start_tick, note.pitch_midi, note.velocity_midi));
        offs_by_track.entry(note.track_index).or_default()
            .push((note.end_tick, note.pitch_midi));
    }

    let mut tempo_events = 0usize;
    let mut meter_events = 0usize;
    if let Some(events) = smf.tracks.first() {
        let mut tick = 0u64;
        for event in events {
            let Some(next) = tick.checked_add(u64::from(event.delta.as_int())) else {
                issues.push(PartStemEvidenceIssueV1::InvalidMidiEvent { track_index: 0 });
                break;
            };
            tick = next;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value))
                    if tick == 0 && value.as_int() == manifest.tempo_microseconds_per_quarter =>
                {
                    tempo_events += 1;
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(numerator, denominator, _, _))
                    if tick == 0
                        && numerator == manifest.meter_numerator
                        && denominator == 2 =>
                {
                    meter_events += 1;
                }
                TrackEventKind::Meta(MetaMessage::TrackName(_))
                | TrackEventKind::Meta(MetaMessage::EndOfTrack) => {}
                _ => issues.push(PartStemEvidenceIssueV1::InvalidMidiEvent { track_index: 0 }),
            }
        }
    }
    if tempo_events != 1 || meter_events != 1 {
        issues.push(PartStemEvidenceIssueV1::InvalidMidiTempoOrMeter);
    }

    let mut program_map = BTreeMap::new();
    for (index, events) in smf.tracks.iter().enumerate().skip(1) {
        let Ok(track_index) = u16::try_from(index) else {
            issues.push(PartStemEvidenceIssueV1::InvalidMidiHeader);
            continue;
        };
        let Some(track_receipt) = track_map.get(&track_index) else {
            issues.push(PartStemEvidenceIssueV1::UnexpectedStem { track_index });
            continue;
        };
        let mut actual_on = Vec::new();
        let mut actual_off = Vec::new();
        let mut actual_programs = Vec::new();
        let mut tick = 0u64;
        for event in events {
            let Some(next) = tick.checked_add(u64::from(event.delta.as_int())) else {
                issues.push(PartStemEvidenceIssueV1::InvalidMidiEvent { track_index });
                break;
            };
            tick = next;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::TrackName(_))
                | TrackEventKind::Meta(MetaMessage::EndOfTrack) => {}
                TrackEventKind::Midi { channel, message } => {
                    if channel.as_int() != track_receipt.midi_channel {
                        issues.push(PartStemEvidenceIssueV1::InvalidMidiEvent { track_index });
                    }
                    match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            actual_on.push((tick, key.as_int(), vel.as_int()));
                        }
                        MidiMessage::NoteOff { key, .. }
                        | MidiMessage::NoteOn { key, vel: _ } => {
                            actual_off.push((tick, key.as_int()));
                        }
                        MidiMessage::ProgramChange { program } => {
                            actual_programs.push(TrackEventProgramChangeV1 {
                                tick,
                                gm_program: program.as_int(),
                            });
                        }
                        _ => issues.push(PartStemEvidenceIssueV1::InvalidMidiEvent { track_index }),
                    }
                }
                _ => issues.push(PartStemEvidenceIssueV1::InvalidMidiEvent { track_index }),
            }
        }
        actual_on.sort_unstable();
        actual_off.sort_unstable();
        let mut expected_on = notes_by_track.remove(&track_index).unwrap_or_default();
        let mut expected_off = offs_by_track.remove(&track_index).unwrap_or_default();
        expected_on.sort_unstable();
        expected_off.sort_unstable();
        actual_programs.sort();
        if actual_on != expected_on || actual_off != expected_off {
            issues.push(PartStemEvidenceIssueV1::MidiNoteSetMismatch { track_index });
        }
        let expected_programs = vec![TrackEventProgramChangeV1 {
            tick: track_receipt.segment_start_tick,
            gm_program: track_receipt.gm_program,
        }];
        if actual_programs != expected_programs {
            issues.push(PartStemEvidenceIssueV1::MidiProgramSetMismatch { track_index });
        }
        program_map.insert(track_index, actual_programs);
    }
    if issues.is_empty() {
        Ok(program_map)
    } else {
        Err(issues)
    }
}

fn track_event_set_sha256_v1(
    manifest: &MidiPerformanceManifestV1,
    track: &MidiPerformanceTrackReceiptV1,
    programs: &[TrackEventProgramChangeV1],
) -> Result<String, PartStemEvidenceIssueV1> {
    let notes = track_notes(manifest, track.track_index);
    let mut program_changes = programs.to_vec();
    program_changes.sort();
    let commitment = TrackEventSetCommitmentV1 {
        schema: "muse-midi-track-event-set-v1",
        score_sha256: &manifest.score_sha256,
        part_id: track.part_id,
        segment_id: &track.segment_id,
        instrument_id: &track.instrument_id,
        midi_track_index: track.track_index,
        notes: &notes,
        program_changes: &program_changes,
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|_| PartStemEvidenceIssueV1::BundleSerialization)?;
    Ok(domain_sha256(PART_STEM_TRACK_EVENT_SET_SCHEMA_V1, &bytes))
}

/// Calculate the exact track-event-set commitment for a validated MIDI track.
pub fn compute_midi_track_event_set_sha256_v1(
    midi: &MidiPerformanceArtifactV1,
    track_index: u16,
) -> Result<String, Vec<PartStemEvidenceIssueV1>> {
    let programs = verify_midi(midi)?;
    let Some(track) = midi.manifest.tracks.iter()
        .find(|track| track.track_index == track_index)
    else {
        return Err(vec![PartStemEvidenceIssueV1::UnexpectedStem { track_index }]);
    };
    let Some(track_programs) = programs.get(&track_index) else {
        return Err(vec![PartStemEvidenceIssueV1::MidiProgramSetMismatch { track_index }]);
    };
    track_event_set_sha256_v1(&midi.manifest, track, track_programs).map_err(|issue| vec![issue])
}

enum WavErrorV1 {
    Invalid,
    NonFinite,
    TooManyFrames,
    ClockMismatch,
    SampleCountMismatch,
}

fn validate_wav(
    bytes: &[u8],
    clock: &PartStemAudioClockV1,
) -> Result<(), WavErrorV1> {
    if bytes.is_empty() || bytes.len() > PART_STEM_MAX_WAV_BYTES_V1 {
        return Err(WavErrorV1::Invalid);
    }
    let mut reader = WavReader::new(Cursor::new(bytes)).map_err(|_| WavErrorV1::Invalid)?;
    let spec = reader.spec();
    if spec.channels != clock.channels
        || spec.sample_rate != clock.sample_rate_hz
        || spec.bits_per_sample != clock.bits_per_sample
        || (clock.sample_format == PartStemSampleFormatV1::PcmInteger
            && spec.sample_format != SampleFormat::Int)
        || (clock.sample_format == PartStemSampleFormatV1::PcmFloat32
            && (spec.sample_format != SampleFormat::Float || spec.bits_per_sample != 32))
    {
        return Err(WavErrorV1::ClockMismatch);
    }
    let channels = u64::from(spec.channels);
    if channels == 0 || clock.frame_count > PART_STEM_MAX_AUDIO_FRAMES_V1 {
        return Err(WavErrorV1::TooManyFrames);
    }
    let max_samples = PART_STEM_MAX_AUDIO_FRAMES_V1
        .checked_mul(channels).ok_or(WavErrorV1::TooManyFrames)?;
    let mut samples = 0u64;
    match spec.sample_format {
        SampleFormat::Float => {
            for item in reader.samples::<f32>() {
                if !item.map_err(|_| WavErrorV1::Invalid)?.is_finite() {
                    return Err(WavErrorV1::NonFinite);
                }
                samples += 1;
                if samples > max_samples {
                    return Err(WavErrorV1::TooManyFrames);
                }
            }
        }
        SampleFormat::Int => {
            for item in reader.samples::<i32>() {
                item.map_err(|_| WavErrorV1::Invalid)?;
                samples += 1;
                if samples > max_samples {
                    return Err(WavErrorV1::TooManyFrames);
                }
            }
        }
    }
    if samples % channels != 0 {
        return Err(WavErrorV1::SampleCountMismatch);
    }
    if samples / channels != clock.frame_count {
        return Err(WavErrorV1::SampleCountMismatch);
    }
    Ok(())
}

/// Validate the bundle against the exact MIDI artifact and full WAV byte set.
/// PASS here means structural consistency only; renderer execution is not
/// attested and the receipt explicitly records that limitation.
pub fn validate_part_stem_bundle_v1(
    bundle: &PartStemEvidenceBundleV1,
    midi: &MidiPerformanceArtifactV1,
    wav_inputs: &[PartStemWavInputV1],
) -> Result<PartStemValidationReceiptV1, Vec<PartStemEvidenceIssueV1>> {
    let mut issues = Vec::new();
    if bundle.schema != PART_STEM_EVIDENCE_SCHEMA_V1 {
        issues.push(PartStemEvidenceIssueV1::WrongBundleSchema);
    }
    if bundle.stems.is_empty() {
        issues.push(PartStemEvidenceIssueV1::EmptyStemSet);
    }
    if bundle.stems.len() > PART_STEM_MAX_STEMS_V1 {
        issues.push(PartStemEvidenceIssueV1::TooManyStems);
    }
    if bundle.stems.len() != midi.manifest.tracks.len() {
        issues.push(PartStemEvidenceIssueV1::MidiTrackCountMismatch);
    }
    if midi.manifest.notes.len() > PART_STEM_MAX_NOTES_V1 {
        issues.push(PartStemEvidenceIssueV1::TooManyNotes);
    }
    if bundle.score_sha256 != midi.manifest.score_sha256
        || bundle.midi_sha256 != midi.manifest.midi_sha256
        || bundle.midi_manifest_sha256 != midi.manifest_sha256
    {
        issues.push(PartStemEvidenceIssueV1::InvalidDigest {
            field: "source_binding".into(),
        });
    }
    for (field, digest) in [
        ("score_sha256", bundle.score_sha256.as_str()),
        ("midi_sha256", bundle.midi_sha256.as_str()),
        ("midi_manifest_sha256", bundle.midi_manifest_sha256.as_str()),
        ("bundle_sha256", bundle.bundle_sha256.as_str()),
    ] {
        if !valid_sha256(digest) {
            issues.push(PartStemEvidenceIssueV1::InvalidDigest { field: field.into() });
        }
    }
    if bundle.audio_clock.origin != "score_zero"
        || !(8_000..=192_000).contains(&bundle.audio_clock.sample_rate_hz)
        || bundle.audio_clock.channels == 0
        || bundle.audio_clock.channels > 8
        || bundle.audio_clock.frame_count == 0
        || bundle.audio_clock.frame_count > PART_STEM_MAX_AUDIO_FRAMES_V1
        || !matches!(bundle.audio_clock.bits_per_sample, 16 | 24 | 32)
        || (bundle.audio_clock.sample_format == PartStemSampleFormatV1::PcmFloat32
            && bundle.audio_clock.bits_per_sample != 32)
    {
        issues.push(PartStemEvidenceIssueV1::InvalidAudioClock);
    }
    match part_stem_bundle_commitment_v1(bundle) {
        Ok(value) if value == bundle.bundle_sha256 => {}
        _ => issues.push(PartStemEvidenceIssueV1::BundleCommitmentMismatch),
    }

    let program_map = match verify_midi(midi) {
        Ok(value) => value,
        Err(mut errors) => {
            issues.append(&mut errors);
            BTreeMap::new()
        }
    };
    let expected_tracks: BTreeMap<_, _> = midi.manifest.tracks.iter()
        .map(|track| (track.track_index, track))
        .collect();
    let mut expected_ordinals: BTreeMap<u16, Vec<usize>> = BTreeMap::new();
    for note in &midi.manifest.notes {
        expected_ordinals.entry(note.track_index).or_default()
            .push(note.source_note_ordinal);
    }
    for values in expected_ordinals.values_mut() {
        values.sort_unstable();
    }

    let mut wav_map = BTreeMap::new();
    for input in wav_inputs {
        if wav_map.insert(input.wav_sha256.as_str(), input).is_some() {
            issues.push(PartStemEvidenceIssueV1::DuplicateWavInput {
                sha256: input.wav_sha256.clone(),
            });
        }
    }
    let total_wav_bytes = wav_inputs.iter().try_fold(0u64, |sum, input| {
        sum.checked_add(input.wav_bytes.len() as u64)
    }).unwrap_or(u64::MAX);
    if total_wav_bytes > PART_STEM_MAX_TOTAL_WAV_BYTES_V1 {
        issues.push(PartStemEvidenceIssueV1::TotalWavBytesExceeded);
    }

    let mut seen_tracks = BTreeSet::new();
    let mut seen_identities = BTreeSet::new();
    for stem in &bundle.stems {
        let id = stem_identity(stem);
        if !seen_tracks.insert(stem.midi_track_index) || !seen_identities.insert(id.clone()) {
            issues.push(PartStemEvidenceIssueV1::DuplicateStemIdentity {
                track_index: stem.midi_track_index,
            });
        }
        if stem.render_method != "single_track_input" {
            issues.push(PartStemEvidenceIssueV1::InvalidRenderMethod {
                track_index: stem.midi_track_index,
            });
        }
        for (field, value) in [
            ("segment_id", stem.segment_id.as_str()),
            ("instrument_id", stem.instrument_id.as_str()),
            ("renderer_id", stem.renderer_id.as_str()),
            ("patch_id", stem.patch_id.as_str()),
        ] {
            if !non_blank(value) {
                issues.push(PartStemEvidenceIssueV1::InvalidIdentity { field: field.into() });
            }
        }
        for (field, value) in [
            ("renderer_binary_sha256", stem.renderer_binary_sha256.as_str()),
            ("patch_asset_sha256", stem.patch_asset_sha256.as_str()),
            ("render_config_sha256", stem.render_config_sha256.as_str()),
            ("track_event_set_sha256", stem.track_event_set_sha256.as_str()),
            ("wav_sha256", stem.wav_sha256.as_str()),
        ] {
            if !valid_sha256(value) {
                issues.push(PartStemEvidenceIssueV1::InvalidDigest { field: field.into() });
            }
        }
        let Some(track) = expected_tracks.get(&stem.midi_track_index) else {
            issues.push(PartStemEvidenceIssueV1::UnexpectedStem {
                track_index: stem.midi_track_index,
            });
            continue;
        };
        if id != track_identity(track) {
            issues.push(PartStemEvidenceIssueV1::StemIdentityMismatch {
                track_index: stem.midi_track_index,
            });
        }
        if stem.source_note_ordinals.windows(2).any(|pair| pair[0] >= pair[1]) {
            issues.push(PartStemEvidenceIssueV1::InvalidSourceOrdinalOrder {
                track_index: stem.midi_track_index,
            });
        }
        if stem.source_note_ordinals
            != expected_ordinals.get(&stem.midi_track_index).cloned().unwrap_or_default()
        {
            issues.push(PartStemEvidenceIssueV1::SourceOrdinalCoverageMismatch {
                track_index: stem.midi_track_index,
            });
        }
        if let Some(programs) = program_map.get(&stem.midi_track_index) {
            match track_event_set_sha256_v1(&midi.manifest, track, programs) {
                Ok(value) if value == stem.track_event_set_sha256 => {}
                _ => issues.push(PartStemEvidenceIssueV1::TrackEventSetMismatch {
                    track_index: stem.midi_track_index,
                }),
            }
        }

        let Some(wav) = wav_map.get(stem.wav_sha256.as_str()) else {
            issues.push(PartStemEvidenceIssueV1::WavMissing {
                sha256: stem.wav_sha256.clone(),
            });
            continue;
        };
        if wav.wav_bytes.len() > PART_STEM_MAX_WAV_BYTES_V1 {
            issues.push(PartStemEvidenceIssueV1::WavTooLarge {
                sha256: stem.wav_sha256.clone(),
            });
            continue;
        }
        if sha256_hex(&wav.wav_bytes) != stem.wav_sha256 || wav.wav_sha256 != stem.wav_sha256 {
            issues.push(PartStemEvidenceIssueV1::WavDigestMismatch {
                sha256: stem.wav_sha256.clone(),
            });
            continue;
        }
        if let Err(error) = validate_wav(&wav.wav_bytes, &bundle.audio_clock) {
            issues.push(match error {
                WavErrorV1::Invalid => PartStemEvidenceIssueV1::InvalidWav {
                    sha256: stem.wav_sha256.clone(),
                },
                WavErrorV1::NonFinite => PartStemEvidenceIssueV1::NonFiniteAudioSample {
                    sha256: stem.wav_sha256.clone(),
                },
                WavErrorV1::TooManyFrames => PartStemEvidenceIssueV1::TooManyAudioFrames {
                    sha256: stem.wav_sha256.clone(),
                },
                WavErrorV1::ClockMismatch => PartStemEvidenceIssueV1::WavClockMismatch {
                    sha256: stem.wav_sha256.clone(),
                },
                WavErrorV1::SampleCountMismatch => {
                    PartStemEvidenceIssueV1::WavSampleCountMismatch {
                        sha256: stem.wav_sha256.clone(),
                    }
                }
            });
        }
    }
    for track in &midi.manifest.tracks {
        if !seen_tracks.contains(&track.track_index) {
            issues.push(PartStemEvidenceIssueV1::MissingStem {
                track_index: track.track_index,
            });
        }
    }
    let declared_wavs: BTreeSet<_> = bundle.stems.iter()
        .map(|stem| stem.wav_sha256.as_str())
        .collect();
    for input in wav_inputs {
        if !declared_wavs.contains(input.wav_sha256.as_str()) {
            issues.push(PartStemEvidenceIssueV1::WavMissing {
                sha256: input.wav_sha256.clone(),
            });
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut receipt = PartStemValidationReceiptV1 {
        schema: PART_STEM_RECEIPT_DIGEST_SCHEMA_V1.into(),
        status: PartStemValidationStatusV1::StructurallyConsistentUnattested,
        renderer_execution_evidence: RendererExecutionEvidenceV1::DeclaredIdentityOnly,
        bundle_sha256: bundle.bundle_sha256.clone(),
        score_sha256: bundle.score_sha256.clone(),
        midi_sha256: bundle.midi_sha256.clone(),
        stem_count: bundle.stems.len(),
        total_wav_bytes,
        common_frame_count: bundle.audio_clock.frame_count,
        receipt_sha256: String::new(),
    };
    let commitment = ReceiptCommitmentV1 {
        schema: &receipt.schema,
        status: receipt.status,
        renderer_execution_evidence: receipt.renderer_execution_evidence.clone(),
        bundle_sha256: &receipt.bundle_sha256,
        score_sha256: &receipt.score_sha256,
        midi_sha256: &receipt.midi_sha256,
        stem_count: receipt.stem_count,
        total_wav_bytes: receipt.total_wav_bytes,
        common_frame_count: receipt.common_frame_count,
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|_| vec![PartStemEvidenceIssueV1::BundleSerialization])?;
    receipt.receipt_sha256 = domain_sha256(PART_STEM_RECEIPT_DIGEST_SCHEMA_V1, &bytes);
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_performance::{
        MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_SCHEMA_V1, MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1,
        MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1, MidiPerformanceManifestV1,
        MidiPerformanceNoteReceiptV1, MidiPerformanceTrackReceiptV1,
    };

    fn midi_bytes() -> Vec<u8> {
        let tempo = vec![
            0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
            0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let notes = vec![
            0x00, 0xc0, 0x00,
            0x00, 0x90, 60, 100,
            0x87, 0x40, 0x80, 60, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&2u16.to_be_bytes());
        bytes.extend_from_slice(&960u16.to_be_bytes());
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(tempo.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&tempo);
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(notes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&notes);
        bytes
    }

    fn pcm_wav(frames: u32) -> Vec<u8> {
        let data_len = frames * 2;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&48_000u32.to_le_bytes());
        bytes.extend_from_slice(&96_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        bytes.resize(44 + data_len as usize, 0);
        bytes
    }

    fn fixture() -> (
        PartStemEvidenceBundleV1,
        MidiPerformanceArtifactV1,
        Vec<PartStemWavInputV1>,
    ) {
        let midi_bytes = midi_bytes();
        let manifest = MidiPerformanceManifestV1 {
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
                track_name: "Part 0 segment-0".into(),
                part_id: 0,
                segment_id: "segment-0".into(),
                instrument_id: "instrument-0".into(),
                renderer_id: "fixture-renderer".into(),
                patch_id: "fixture-patch".into(),
                gm_program: 0,
                midi_channel: 0,
                segment_start_tick: 0,
                segment_end_tick: 960,
            }],
            notes: vec![MidiPerformanceNoteReceiptV1 {
                source_note_ordinal: 0,
                part_id: 0,
                segment_id: "segment-0".into(),
                instrument_id: "instrument-0".into(),
                track_index: 1,
                start_tick: 0,
                end_tick: 960,
                pitch_midi: 60,
                velocity_midi: 100,
            }],
        };
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let manifest_sha256 = domain_sha256(
            MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
            &manifest_bytes,
        );
        let midi = MidiPerformanceArtifactV1 {
            midi_bytes,
            manifest: manifest.clone(),
            manifest_sha256,
        };
        let wav_bytes = pcm_wav(24_000);
        let wav_sha256 = sha256_hex(&wav_bytes);
        let track_event_set_sha256 =
            compute_midi_track_event_set_sha256_v1(&midi, 1).unwrap();
        let mut bundle = PartStemEvidenceBundleV1 {
            schema: PART_STEM_EVIDENCE_SCHEMA_V1.into(),
            score_sha256: manifest.score_sha256.clone(),
            midi_sha256: manifest.midi_sha256.clone(),
            midi_manifest_sha256: midi.manifest_sha256.clone(),
            audio_clock: PartStemAudioClockV1 {
                origin: "score_zero".into(),
                sample_rate_hz: 48_000,
                channels: 1,
                sample_format: PartStemSampleFormatV1::PcmInteger,
                bits_per_sample: 16,
                frame_count: 24_000,
            },
            stems: vec![PartStemRecordV1 {
                part_id: 0,
                segment_id: "segment-0".into(),
                instrument_id: "instrument-0".into(),
                midi_track_index: 1,
                render_method: "single_track_input".into(),
                renderer_id: "fixture-renderer".into(),
                renderer_binary_sha256: "e".repeat(64),
                patch_id: "fixture-patch".into(),
                patch_asset_sha256: "f".repeat(64),
                render_config_sha256: "1".repeat(64),
                track_event_set_sha256,
                source_note_ordinals: vec![0],
                wav_sha256: wav_sha256.clone(),
            }],
            bundle_sha256: String::new(),
        };
        bundle.bundle_sha256 = part_stem_bundle_commitment_v1(&bundle).unwrap();
        (
            bundle,
            midi,
            vec![PartStemWavInputV1 { wav_sha256, wav_bytes }],
        )
    }

    #[test]
    fn exact_single_track_bundle_is_structurally_consistent() {
        let (bundle, midi, wavs) = fixture();
        let receipt = validate_part_stem_bundle_v1(&bundle, &midi, &wavs).unwrap();
        assert_eq!(
            receipt.status,
            PartStemValidationStatusV1::StructurallyConsistentUnattested
        );
        assert_eq!(
            receipt.renderer_execution_evidence,
            RendererExecutionEvidenceV1::DeclaredIdentityOnly
        );
        assert_eq!(receipt.stem_count, 1);
        assert!(valid_sha256(&receipt.receipt_sha256));
    }

    #[test]
    fn missing_stem_is_rejected() {
        let (mut bundle, midi, wavs) = fixture();
        bundle.stems.clear();
        bundle.bundle_sha256 = part_stem_bundle_commitment_v1(&bundle).unwrap();
        let issues = validate_part_stem_bundle_v1(&bundle, &midi, &wavs).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PartStemEvidenceIssueV1::EmptyStemSet | PartStemEvidenceIssueV1::MissingStem { .. }
        )));
    }

    #[test]
    fn stale_wav_digest_is_rejected() {
        let (mut bundle, midi, wavs) = fixture();
        bundle.stems[0].wav_sha256 = "9".repeat(64);
        bundle.bundle_sha256 = part_stem_bundle_commitment_v1(&bundle).unwrap();
        let issues = validate_part_stem_bundle_v1(&bundle, &midi, &wavs).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PartStemEvidenceIssueV1::WavMissing { .. }
        )));
    }

    #[test]
    fn wrong_common_clock_is_rejected() {
        let (mut bundle, midi, wavs) = fixture();
        bundle.audio_clock.sample_rate_hz = 44_100;
        bundle.bundle_sha256 = part_stem_bundle_commitment_v1(&bundle).unwrap();
        let issues = validate_part_stem_bundle_v1(&bundle, &midi, &wavs).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PartStemEvidenceIssueV1::WavClockMismatch { .. }
        )));
    }

    #[test]
    fn unknown_json_fields_are_rejected() {
        let bytes = br#"{
            "schema":"muse-part-stem-evidence-v1",
            "score_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "midi_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "midi_manifest_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "audio_clock":{"origin":"score_zero","sample_rate_hz":48000,"channels":1,
              "sample_format":"pcm-integer","bits_per_sample":16,"frame_count":24000},
            "stems":[],
            "bundle_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "unexpected":true
        }"#;
        assert!(parse_part_stem_bundle_v1(bytes).is_err());
    }
}
