// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-agnostic MIDI transcription evaluation against a part-aware reference.
//!
//! This module evaluates outputs from external audio-to-MIDI systems without
//! embedding model weights, Python runtimes, network APIs, or vendor code.
//! It reports instrument-agnostic onset/full-note metrics separately from
//! exact GM-program attribution. It follows the documented mir_eval defaults:
//! 50 ms onset tolerance, 50-cent pitch tolerance, and max(20% of reference
//! note duration, 50 ms) offset tolerance.
//!
//! Matching is one-to-one maximum-cardinality bipartite matching. The report
//! binds the exact reference manifest and predicted MIDI bytes. It does not
//! certify the acoustic source, the transcriber, or the identity of a rendered
//! instrument beyond what the MIDI prediction explicitly encodes.

use crate::midi_performance::{
    MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
    MIDI_PERFORMANCE_SCHEMA_V1,
    MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1,
    MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1,
    MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1,
    MidiPerformanceArtifactV1,
};
use midly::{Format, MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};

pub const MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1: &str =
    "muse-midi-transcription-evaluation-v1";
pub const MIDI_TRANSCRIPTION_EVALUATION_ALGORITHM_V1: &str =
    "mir-eval-default-tolerance-max-cardinality-bipartite-v1";
pub const MIDI_TRANSCRIPTION_DEFAULT_ONSET_TOLERANCE_MS_V1: u32 = 50;
pub const MIDI_TRANSCRIPTION_DEFAULT_PITCH_TOLERANCE_CENTS_V1: u16 = 50;
pub const MIDI_TRANSCRIPTION_DEFAULT_OFFSET_RATIO_BPS_V1: u16 = 2_000;
pub const MIDI_TRANSCRIPTION_DEFAULT_OFFSET_MIN_TOLERANCE_MS_V1: u32 = 50;
pub const MIDI_TRANSCRIPTION_MAX_NOTES_V1: usize = 20_000;
pub const MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1: usize = 2_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiTranscriptionEvaluationConfigV1 {
    pub onset_tolerance_ms: u32,
    pub pitch_tolerance_cents: u16,
    pub offset_ratio_basis_points: u16,
    pub offset_min_tolerance_ms: u32,
}

impl Default for MidiTranscriptionEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            onset_tolerance_ms: MIDI_TRANSCRIPTION_DEFAULT_ONSET_TOLERANCE_MS_V1,
            pitch_tolerance_cents: MIDI_TRANSCRIPTION_DEFAULT_PITCH_TOLERANCE_CENTS_V1,
            offset_ratio_basis_points: MIDI_TRANSCRIPTION_DEFAULT_OFFSET_RATIO_BPS_V1,
            offset_min_tolerance_ms: MIDI_TRANSCRIPTION_DEFAULT_OFFSET_MIN_TOLERANCE_MS_V1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MidiTranscribedNoteV1 {
    pub track_index: u16,
    pub midi_channel: u8,
    pub onset_tick: u64,
    pub offset_tick: Option<u64>,
    /// GM program captured at NoteOn; percussion channel 9 is retained.
    pub gm_program: u8,
    pub pitch_midi: u8,
    pub onset_seconds: f64,
    /// None means the prediction contained a NoteOn without a matching NoteOff.
    pub offset_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedMidiTranscriptionV1 {
    pub notes: Vec<MidiTranscribedNoteV1>,
    pub ticks_per_quarter: u16,
    pub tempo_point_count: u32,
    pub tempo_points: Vec<MidiTempoPointV1>,
    pub unclosed_note_count: u32,
    pub orphan_note_off_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTranscriptionMetricV1 {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTranscriptionTimingErrorSummaryV1 {
    pub sample_count: u32,
    pub mean_absolute_error_ms: Option<f64>,
    pub median_absolute_error_ms: Option<f64>,
    pub p95_absolute_error_ms: Option<f64>,
    pub p99_absolute_error_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTranscriptionEvaluationReportV1 {
    pub schema: String,
    pub algorithm: String,
    pub reference_score_sha256: String,
    pub reference_midi_sha256: String,
    pub reference_manifest_sha256: String,
    pub prediction_midi_sha256: String,
    pub config: MidiTranscriptionEvaluationConfigV1,
    pub reference_note_count: u32,
    pub prediction_note_count: u32,
    pub prediction_unclosed_note_count: u32,
    pub prediction_orphan_note_off_count: u32,
    /// Instrument-agnostic: pitch and onset must match.
    pub onset_pitch_metrics: MidiTranscriptionMetricV1,
    /// Instrument-agnostic: pitch, onset and offset must match.
    pub full_note_metrics: MidiTranscriptionMetricV1,
    /// Exact GM program, pitch and onset must match.
    pub exact_program_onset_metrics: MidiTranscriptionMetricV1,
    /// Exact GM program, pitch, onset and offset must match.
    pub exact_program_full_note_metrics: MidiTranscriptionMetricV1,
    pub onset_error_ms: MidiTranscriptionTimingErrorSummaryV1,
    pub offset_error_ms: MidiTranscriptionTimingErrorSummaryV1,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiTranscriptionEvaluationIssueV1 {
    InvalidConfig { field: String },
    ReferenceMidiDigestMismatch,
    ReferenceManifestDigestMismatch,
    ReferenceMidiDoesNotMatchManifest,
    InvalidReference,
    InvalidPredictionMidi,
    UnsupportedMidiFormat,
    UnsupportedSmpteTiming,
    InvalidTempo,
    MidiTickOverflow,
    TooManyPredictedNotes { count: usize, maximum: usize },
    InvalidReferenceTrack { track_index: u16 },
    InvalidReferenceNote { ordinal: usize },
    TooManyCandidatePairs { count: usize, maximum: usize },
    SerializationFailed,
}

#[derive(Debug, Clone, Copy)]
pub struct MidiTempoPointV1 {
    pub start_tick: u64,
    pub start_seconds: f64,
    pub microseconds_per_quarter: u32,
}

#[derive(Debug, Clone)]
struct EvaluationNoteV1 {
    pitch_midi: u8,
    gm_program: u8,
    onset_seconds: f64,
    offset_seconds: Option<f64>,
}

#[derive(Serialize)]
struct ReportCommitmentV1<'a> {
    schema: &'a str,
    algorithm: &'a str,
    reference_score_sha256: &'a str,
    reference_midi_sha256: &'a str,
    reference_manifest_sha256: &'a str,
    prediction_midi_sha256: &'a str,
    config: &'a MidiTranscriptionEvaluationConfigV1,
    reference_note_count: u32,
    prediction_note_count: u32,
    prediction_unclosed_note_count: u32,
    prediction_orphan_note_off_count: u32,
    onset_pitch_metrics: &'a MidiTranscriptionMetricV1,
    full_note_metrics: &'a MidiTranscriptionMetricV1,
    exact_program_onset_metrics: &'a MidiTranscriptionMetricV1,
    exact_program_full_note_metrics: &'a MidiTranscriptionMetricV1,
    onset_error_ms: &'a MidiTranscriptionTimingErrorSummaryV1,
    offset_error_ms: &'a MidiTranscriptionTimingErrorSummaryV1,
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

fn validate_config(
    config: &MidiTranscriptionEvaluationConfigV1,
) -> Result<(), MidiTranscriptionEvaluationIssueV1> {
    if config.onset_tolerance_ms == 0 || config.onset_tolerance_ms > 5_000 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidConfig {
            field: "onset_tolerance_ms".into(),
        });
    }
    if config.pitch_tolerance_cents > 1_200 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidConfig {
            field: "pitch_tolerance_cents".into(),
        });
    }
    if config.offset_ratio_basis_points == 0
        || config.offset_ratio_basis_points > 10_000
        || config.offset_min_tolerance_ms == 0
        || config.offset_min_tolerance_ms > 5_000
    {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidConfig {
            field: "offset_tolerance".into(),
        });
    }
    Ok(())
}

fn tempo_points_from_events(
    events: &[(u64, usize, usize, u32)],
    ticks_per_quarter: u16,
) -> Result<Vec<MidiTempoPointV1>, MidiTranscriptionEvaluationIssueV1> {
    if ticks_per_quarter == 0 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
    }
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|(tick, track, order, _)| (*tick, *track, *order));

    let mut points = vec![MidiTempoPointV1 {
        start_tick: 0,
        start_seconds: 0.0,
        microseconds_per_quarter: 500_000,
    }];

    for (tick, _, _, tempo) in sorted {
        if tempo == 0 {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidTempo);
        }
        let previous = *points.last().expect("initial tempo point is present");
        if tick < previous.start_tick {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
        }
        if tick == previous.start_tick {
            points.last_mut().expect("tempo point exists").microseconds_per_quarter = tempo;
            continue;
        }
        let delta_ticks = tick - previous.start_tick;
        let elapsed = delta_ticks as f64 * f64::from(previous.microseconds_per_quarter)
            / (f64::from(ticks_per_quarter) * 1_000_000.0);
        let start_seconds = previous.start_seconds + elapsed;
        if !start_seconds.is_finite() {
            return Err(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow);
        }
        points.push(MidiTempoPointV1 {
            start_tick: tick,
            start_seconds,
            microseconds_per_quarter: tempo,
        });
    }
    Ok(points)
}

fn tick_to_seconds(tick: u64, tempo_points: &[MidiTempoPointV1], ticks_per_quarter: u16) -> Option<f64> {
    let index = tempo_points.partition_point(|point| point.start_tick <= tick);
    let point = tempo_points.get(index.checked_sub(1)?)?;
    let delta_ticks = tick.checked_sub(point.start_tick)?;
    let seconds = point.start_seconds
        + delta_ticks as f64 * f64::from(point.microseconds_per_quarter)
            / (f64::from(ticks_per_quarter) * 1_000_000.0);
    seconds.is_finite().then_some(seconds)
}

/// Parse a predicted Standard MIDI file into seconds-domain note events.
/// Piecewise tempo is supported. Sequential format (independent patterns) and
/// SMPTE time division are rejected rather than guessed.
pub fn parse_transcribed_midi_v1(
    midi_bytes: &[u8],
) -> Result<ParsedMidiTranscriptionV1, MidiTranscriptionEvaluationIssueV1> {
    let smf = Smf::parse(midi_bytes)
        .map_err(|_| MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi)?;
    if !matches!(smf.header.format, Format::SingleTrack | Format::Parallel) {
        return Err(MidiTranscriptionEvaluationIssueV1::UnsupportedMidiFormat);
    }
    let ticks_per_quarter = match smf.header.timing {
        Timing::Metrical(value) => value.as_int(),
        Timing::Timecode(_, _) => {
            return Err(MidiTranscriptionEvaluationIssueV1::UnsupportedSmpteTiming);
        }
    };
    if ticks_per_quarter == 0 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
    }

    let mut tempo_events = Vec::new();
    let mut absolute_tracks = Vec::with_capacity(smf.tracks.len());
    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0u64;
        let mut absolute = Vec::with_capacity(track.len());
        for (event_index, event) in track.iter().enumerate() {
            tick = tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                tempo_events.push((tick, track_index, event_index, tempo.as_int()));
            }
            absolute.push((tick, event.kind));
        }
        absolute_tracks.push(absolute);
    }

    let tempo_points = tempo_points_from_events(&tempo_events, ticks_per_quarter)?;
    let mut notes = Vec::new();
    let mut unclosed_note_count = 0usize;
    let mut orphan_note_off_count = 0usize;

    for (track_index, events) in absolute_tracks.iter().enumerate() {
        let track_index_u16 = u16::try_from(track_index)
            .map_err(|_| MidiTranscriptionEvaluationIssueV1::UnsupportedMidiFormat)?;
        let mut programs = [0u8; 16];
        let mut pending: HashMap<(u8, u8), VecDeque<(u64, u8)>> = HashMap::new();
        for (tick, kind) in events {
            match *kind {
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    match message {
                        MidiMessage::ProgramChange { program } => {
                            programs[usize::from(channel)] = program.as_int();
                        }
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            pending
                                .entry((channel, key.as_int()))
                                .or_default()
                                .push_back((*tick, programs[usize::from(channel)]));
                        }
                        MidiMessage::NoteOff { key, .. }
                        | MidiMessage::NoteOn { key, .. } => {
                            if let Some(queue) = pending.get_mut(&(channel, key.as_int())) {
                                if let Some((start_tick, program)) = queue.pop_front() {
                                    let onset_seconds = tick_to_seconds(
                                        start_tick,
                                        &tempo_points,
                                        ticks_per_quarter,
                                    )
                                    .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
                                    let offset_seconds = tick_to_seconds(
                                        *tick,
                                        &tempo_points,
                                        ticks_per_quarter,
                                    )
                                    .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
                                    if offset_seconds <= onset_seconds {
                                        return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
                                    }
                                    notes.push(MidiTranscribedNoteV1 {
                                        track_index: track_index_u16,
                                        midi_channel: channel,
                                        onset_tick: start_tick,
                                        offset_tick: Some(*tick),
                                        gm_program: program,
                                        pitch_midi: key.as_int(),
                                        onset_seconds,
                                        offset_seconds: Some(offset_seconds),
                                    });
                                } else {
                                    orphan_note_off_count += 1;
                                }
                            } else {
                                orphan_note_off_count += 1;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        for ((channel, pitch), queue) in pending {
            for (start_tick, program) in queue {
                let onset_seconds = tick_to_seconds(
                    start_tick,
                    &tempo_points,
                    ticks_per_quarter,
                )
                .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
                unclosed_note_count += 1;
                notes.push(MidiTranscribedNoteV1 {
                    track_index: track_index_u16,
                    midi_channel: channel,
                    onset_tick: start_tick,
                    offset_tick: None,
                    gm_program: program,
                    pitch_midi: pitch,
                    onset_seconds,
                    offset_seconds: None,
                });
            }
        }
    }

    if notes.len() > MIDI_TRANSCRIPTION_MAX_NOTES_V1 {
        return Err(MidiTranscriptionEvaluationIssueV1::TooManyPredictedNotes {
            count: notes.len(),
            maximum: MIDI_TRANSCRIPTION_MAX_NOTES_V1,
        });
    }
    notes.sort_by(|left, right| {
        left.onset_seconds
            .total_cmp(&right.onset_seconds)
            .then_with(|| left.pitch_midi.cmp(&right.pitch_midi))
            .then_with(|| left.gm_program.cmp(&right.gm_program))
            .then_with(|| left.track_index.cmp(&right.track_index))
    });

    Ok(ParsedMidiTranscriptionV1 {
        notes,
        ticks_per_quarter,
        tempo_point_count: u32::try_from(tempo_points.len()).unwrap_or(u32::MAX),
        tempo_points,
        unclosed_note_count: u32::try_from(unclosed_note_count).unwrap_or(u32::MAX),
        orphan_note_off_count: u32::try_from(orphan_note_off_count).unwrap_or(u32::MAX),
    })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_reference_artifact_v1(
    reference: &MidiPerformanceArtifactV1,
) -> Result<(), MidiTranscriptionEvaluationIssueV1> {
    use midly::{MetaMessage, MidiMessage, TrackEventKind};

    let manifest = &reference.manifest;
    if manifest.schema != MIDI_PERFORMANCE_SCHEMA_V1
        || manifest.score_digest_schema
            != symthaea_music_theory::orchestration_timeline::ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1
        || manifest.timeline_digest_schema != MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1
        || manifest.binding_digest_schema != MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1
        || manifest.capability_digest_schema != MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1
        || manifest.ticks_per_quarter == 0
        || manifest.tempo_microseconds_per_quarter == 0
        || manifest.tempo_microseconds_per_quarter > 16_777_215
        || manifest.meter_numerator == 0
        || manifest.meter_denominator == 0
    {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidReference);
    }
    for digest in [
        manifest.score_sha256.as_str(),
        manifest.timeline_sha256.as_str(),
        manifest.binding_sha256.as_str(),
        manifest.capability_sha256.as_str(),
        manifest.midi_sha256.as_str(),
        reference.manifest_sha256.as_str(),
    ] {
        if !valid_sha256(digest) {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReference);
        }
    }
    if sha256_hex(&reference.midi_bytes) != manifest.midi_sha256 {
        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDigestMismatch);
    }
    let manifest_bytes = serde_json::to_vec(manifest)
        .map_err(|_| MidiTranscriptionEvaluationIssueV1::SerializationFailed)?;
    let actual_manifest_digest = domain_separated_sha256(
        MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
        &manifest_bytes,
    );
    if actual_manifest_digest != reference.manifest_sha256 {
        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceManifestDigestMismatch);
    }

    let smf = Smf::parse(&reference.midi_bytes)
        .map_err(|_| MidiTranscriptionEvaluationIssueV1::InvalidReference)?;
    if smf.header.format != Format::Parallel
        || !matches!(
            smf.header.timing,
            Timing::Metrical(ppq) if ppq.as_int() == manifest.ticks_per_quarter
        )
        || smf.tracks.len() != manifest.tracks.len() + 1
    {
        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest);
    }

    type NoteOnKey = (u16, u64, u8, u8, u8);
    type NoteOffKey = (u16, u64, u8, u8);
    type ProgramKey = (u16, u64, u8, u8);
    let track_receipts: BTreeMap<_, _> = manifest
        .tracks
        .iter()
        .map(|track| (track.track_index, track))
        .collect();
    if track_receipts.len() != manifest.tracks.len() {
        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest);
    }

    let mut expected_ons: BTreeMap<NoteOnKey, usize> = BTreeMap::new();
    let mut expected_offs: BTreeMap<NoteOffKey, usize> = BTreeMap::new();
    let mut expected_programs: BTreeMap<ProgramKey, usize> = BTreeMap::new();
    for track in &manifest.tracks {
        if track.track_index == 0
            || usize::from(track.track_index) >= smf.tracks.len()
            || track.midi_channel > 15
            || track.gm_program > 127
            || track.segment_end_tick <= track.segment_start_tick
        {
            return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest);
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
    let mut ordinals = std::collections::BTreeSet::new();
    for note in &manifest.notes {
        if !ordinals.insert(note.source_note_ordinal)
            || note.pitch_midi > 127
            || note.velocity_midi == 0
            || note.end_tick <= note.start_tick
        {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReferenceNote {
                ordinal: note.source_note_ordinal,
            });
        }
        let Some(track) = track_receipts.get(&note.track_index) else {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReferenceTrack {
                track_index: note.track_index,
            });
        };
        if track.part_id != note.part_id
            || track.segment_id != note.segment_id
            || track.instrument_id != note.instrument_id
            || note.start_tick < track.segment_start_tick
            || note.end_tick > track.segment_end_tick
        {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReferenceNote {
                ordinal: note.source_note_ordinal,
            });
        }
        *expected_ons
            .entry((
                note.track_index,
                note.start_tick,
                track.midi_channel,
                note.pitch_midi,
                note.velocity_midi,
            ))
            .or_insert(0) += 1;
        *expected_offs
            .entry((
                note.track_index,
                note.end_tick,
                track.midi_channel,
                note.pitch_midi,
            ))
            .or_insert(0) += 1;
    }
    let mut actual_ons: BTreeMap<NoteOnKey, usize> = BTreeMap::new();
    let mut actual_offs: BTreeMap<NoteOffKey, usize> = BTreeMap::new();
    let mut actual_programs: BTreeMap<ProgramKey, usize> = BTreeMap::new();
    let mut tempo_events = 0usize;
    let mut meter_events = 0usize;

    for (track_index, events) in smf.tracks.iter().enumerate() {
        let track_index = u16::try_from(track_index)
            .map_err(|_| MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest)?;
        let mut absolute_tick = 0u64;
        for event in events {
            absolute_tick = absolute_tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest)?;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    if track_index != 0
                        || absolute_tick != 0
                        || value.as_int() != manifest.tempo_microseconds_per_quarter
                    {
                        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest);
                    }
                    tempo_events += 1;
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(numerator, denominator, _, _)) => {
                    let expected_denominator_power =
                        manifest.meter_denominator.trailing_zeros() as u8;
                    if track_index != 0
                        || absolute_tick != 0
                        || numerator != manifest.meter_numerator
                        || manifest.meter_denominator.count_ones() != 1
                        || denominator != expected_denominator_power
                    {
                        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest);
                    }
                    meter_events += 1;
                }
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            *actual_ons
                                .entry((track_index, absolute_tick, channel, key.as_int(), vel.as_int()))
                                .or_insert(0) += 1;
                        }
                        MidiMessage::NoteOff { key, .. } => {
                            *actual_offs
                                .entry((track_index, absolute_tick, channel, key.as_int()))
                                .or_insert(0) += 1;
                        }
                        MidiMessage::NoteOn { key, .. } => {
                            *actual_offs
                                .entry((track_index, absolute_tick, channel, key.as_int()))
                                .or_insert(0) += 1;
                        }
                        MidiMessage::ProgramChange { program } => {
                            *actual_programs
                                .entry((track_index, absolute_tick, channel, program.as_int()))
                                .or_insert(0) += 1;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
    if tempo_events != 1
        || meter_events != 1
        || actual_ons != expected_ons
        || actual_offs != expected_offs
        || actual_programs != expected_programs
    {
        return Err(MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest);
    }
    Ok(())
}

fn reference_notes(
    reference: &MidiPerformanceArtifactV1,
) -> Result<Vec<EvaluationNoteV1>, MidiTranscriptionEvaluationIssueV1> {
    let programs: BTreeMap<_, _> = reference
        .manifest
        .tracks
        .iter()
        .map(|track| (track.track_index, track.gm_program))
        .collect();
    let mut notes = Vec::with_capacity(reference.manifest.notes.len());
    for note in &reference.manifest.notes {
        let Some(program) = programs.get(&note.track_index).copied() else {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReferenceTrack {
                track_index: note.track_index,
            });
        };
        if note.pitch_midi > 127 || note.end_tick <= note.start_tick {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReferenceNote {
                ordinal: note.source_note_ordinal,
            });
        }
        let onset_seconds = note.start_tick as f64
            * f64::from(reference.manifest.tempo_microseconds_per_quarter)
            / (f64::from(reference.manifest.ticks_per_quarter) * 1_000_000.0);
        let offset_seconds = note.end_tick as f64
            * f64::from(reference.manifest.tempo_microseconds_per_quarter)
            / (f64::from(reference.manifest.ticks_per_quarter) * 1_000_000.0);
        if !onset_seconds.is_finite()
            || !offset_seconds.is_finite()
            || offset_seconds <= onset_seconds
        {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidReferenceNote {
                ordinal: note.source_note_ordinal,
            });
        }
        notes.push(EvaluationNoteV1 {
            pitch_midi: note.pitch_midi,
            gm_program: program,
            onset_seconds,
            offset_seconds: Some(offset_seconds),
        });
    }
    Ok(notes)
}

fn maximum_cardinality_matching(
    adjacency: &[Vec<usize>],
    predicted_count: usize,
) -> Vec<(usize, usize)> {
    fn augment(
        reference_index: usize,
        adjacency: &[Vec<usize>],
        seen_predictions: &mut [bool],
        prediction_to_reference: &mut [Option<usize>],
    ) -> bool {
        for &prediction_index in &adjacency[reference_index] {
            if seen_predictions[prediction_index] {
                continue;
            }
            seen_predictions[prediction_index] = true;
            let previous_reference = prediction_to_reference[prediction_index];
            if previous_reference.is_none()
                || augment(
                    previous_reference.unwrap_or_default(),
                    adjacency,
                    seen_predictions,
                    prediction_to_reference,
                )
            {
                prediction_to_reference[prediction_index] = Some(reference_index);
                return true;
            }
        }
        false
    }

    let mut prediction_to_reference = vec![None; predicted_count];
    for reference_index in 0..adjacency.len() {
        let mut seen_predictions = vec![false; predicted_count];
        let _ = augment(
            reference_index,
            adjacency,
            &mut seen_predictions,
            &mut prediction_to_reference,
        );
    }
    let mut matching: Vec<_> = prediction_to_reference
        .iter()
        .enumerate()
        .filter_map(|(prediction_index, reference_index)| {
            reference_index.map(|reference_index| (reference_index, prediction_index))
        })
        .collect();
    matching.sort_unstable_by_key(|(reference_index, _)| *reference_index);
    matching
}

fn build_adjacency(
    reference: &[EvaluationNoteV1],
    predicted: &[MidiTranscribedNoteV1],
    config: &MidiTranscriptionEvaluationConfigV1,
    require_offset: bool,
    require_program: bool,
) -> Vec<Vec<usize>> {
    let onset_tolerance_seconds = f64::from(config.onset_tolerance_ms) / 1_000.0;
    let pitch_tolerance_semitones = f64::from(config.pitch_tolerance_cents) / 100.0;
    reference
        .iter()
        .map(|reference_note| {
            let reference_duration = reference_note
                .offset_seconds
                .unwrap_or(reference_note.onset_seconds)
                - reference_note.onset_seconds;
            let offset_tolerance = (reference_duration
                * f64::from(config.offset_ratio_basis_points)
                / 10_000.0)
                .max(f64::from(config.offset_min_tolerance_ms) / 1_000.0);
            predicted
                .iter()
                .enumerate()
                .filter_map(|(prediction_index, predicted_note)| {
                    let pitch_error = (i16::from(reference_note.pitch_midi)
                        - i16::from(predicted_note.pitch_midi))
                    .unsigned_abs() as f64;
                    if pitch_error > pitch_tolerance_semitones {
                        return None;
                    }
                    if (reference_note.onset_seconds - predicted_note.onset_seconds).abs()
                        > onset_tolerance_seconds
                    {
                        return None;
                    }
                    if require_program && reference_note.gm_program != predicted_note.gm_program {
                        return None;
                    }
                    if require_offset {
                        let Some(predicted_offset) = predicted_note.offset_seconds else {
                            return None;
                        };
                        let Some(reference_offset) = reference_note.offset_seconds else {
                            return None;
                        };
                        if (reference_offset - predicted_offset).abs() > offset_tolerance {
                            return None;
                        }
                    }
                    Some(prediction_index)
                })
                .collect()
        })
        .collect()
}

fn metric_from_matching(
    reference_count: usize,
    predicted_count: usize,
    true_positives: usize,
) -> MidiTranscriptionMetricV1 {
    let false_positives = predicted_count.saturating_sub(true_positives);
    let false_negatives = reference_count.saturating_sub(true_positives);
    let precision = if predicted_count == 0 {
        0.0
    } else {
        true_positives as f64 / predicted_count as f64
    };
    let recall = if reference_count == 0 {
        0.0
    } else {
        true_positives as f64 / reference_count as f64
    };
    let f1 = if precision + recall <= 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    MidiTranscriptionMetricV1 {
        true_positives: u32::try_from(true_positives).unwrap_or(u32::MAX),
        false_positives: u32::try_from(false_positives).unwrap_or(u32::MAX),
        false_negatives: u32::try_from(false_negatives).unwrap_or(u32::MAX),
        precision,
        recall,
        f1,
    }
}

fn timing_error_summary(values_seconds: &[f64]) -> MidiTranscriptionTimingErrorSummaryV1 {
    if values_seconds.is_empty() {
        return MidiTranscriptionTimingErrorSummaryV1 {
            sample_count: 0,
            mean_absolute_error_ms: None,
            median_absolute_error_ms: None,
            p95_absolute_error_ms: None,
            p99_absolute_error_ms: None,
        };
    }
    let mut values: Vec<f64> = values_seconds
        .iter()
        .map(|value| value.abs() * 1_000.0)
        .collect();
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let percentile = |percent: f64| {
        let index = ((values.len() as f64 * percent).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1);
        values[index]
    };
    let median = if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    };
    MidiTranscriptionTimingErrorSummaryV1 {
        sample_count: u32::try_from(values.len()).unwrap_or(u32::MAX),
        mean_absolute_error_ms: Some(mean),
        median_absolute_error_ms: Some(median),
        p95_absolute_error_ms: Some(percentile(0.95)),
        p99_absolute_error_ms: Some(percentile(0.99)),
    }
}

pub fn midi_transcription_evaluation_commitment_v1(
    report: &MidiTranscriptionEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    let commitment = ReportCommitmentV1 {
        schema: &report.schema,
        algorithm: &report.algorithm,
        reference_score_sha256: &report.reference_score_sha256,
        reference_midi_sha256: &report.reference_midi_sha256,
        reference_manifest_sha256: &report.reference_manifest_sha256,
        prediction_midi_sha256: &report.prediction_midi_sha256,
        config: &report.config,
        reference_note_count: report.reference_note_count,
        prediction_note_count: report.prediction_note_count,
        prediction_unclosed_note_count: report.prediction_unclosed_note_count,
        prediction_orphan_note_off_count: report.prediction_orphan_note_off_count,
        onset_pitch_metrics: &report.onset_pitch_metrics,
        full_note_metrics: &report.full_note_metrics,
        exact_program_onset_metrics: &report.exact_program_onset_metrics,
        exact_program_full_note_metrics: &report.exact_program_full_note_metrics,
        onset_error_ms: &report.onset_error_ms,
        offset_error_ms: &report.offset_error_ms,
    };
    Ok(domain_separated_sha256(
        MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

/// Evaluate a model-produced MIDI file against the exact part-aware MIDI
/// reference. If a validated clock profile is supplied, raw metrics remain
/// untouched and an additional global time correction is applied to the
/// predicted events only when the profile status is an unambiguous Candidate.
pub fn evaluate_midi_transcription_v1(
    prediction_midi_bytes: &[u8],
    reference: &MidiPerformanceArtifactV1,
    config: MidiTranscriptionEvaluationConfigV1,
) -> Result<MidiTranscriptionEvaluationReportV1, MidiTranscriptionEvaluationIssueV1> {
    validate_config(&config)?;
    validate_reference_artifact_v1(reference)?;

    let reference_notes = reference_notes(reference)?;
    let parsed = parse_transcribed_midi_v1(prediction_midi_bytes)?;
    if parsed.notes.len() > MIDI_TRANSCRIPTION_MAX_NOTES_V1 {
        return Err(MidiTranscriptionEvaluationIssueV1::TooManyPredictedNotes {
            count: parsed.notes.len(),
            maximum: MIDI_TRANSCRIPTION_MAX_NOTES_V1,
        });
    }

    let predicted_notes: Vec<_> = parsed
        .notes
        .iter()
        .map(|note| EvaluationNoteV1 {
            pitch_midi: note.pitch_midi,
            gm_program: note.gm_program,
            onset_seconds: note.onset_seconds,
            offset_seconds: note.offset_seconds,
        })
        .collect();

    let onset_adjacency =
        build_adjacency(&reference_notes, &predicted_notes, &config, false, false);
    let onset_matching =
        maximum_cardinality_matching(&onset_adjacency, predicted_notes.len());
    let full_adjacency =
        build_adjacency(&reference_notes, &predicted_notes, &config, true, false);
    let full_matching =
        maximum_cardinality_matching(&full_adjacency, predicted_notes.len());
    let program_onset_adjacency =
        build_adjacency(&reference_notes, &predicted_notes, &config, false, true);
    let program_onset_matching =
        maximum_cardinality_matching(&program_onset_adjacency, predicted_notes.len());
    let program_full_adjacency =
        build_adjacency(&reference_notes, &predicted_notes, &config, true, true);
    let program_full_matching =
        maximum_cardinality_matching(&program_full_adjacency, predicted_notes.len());

    let onset_errors: Vec<_> = onset_matching
        .iter()
        .map(|(reference_index, prediction_index)| {
            predicted_notes[*prediction_index].onset_seconds
                - reference_notes[*reference_index].onset_seconds
        })
        .collect();
    // Offset error is reported for onset-matched notes, even when their
    // offset misses the full-note tolerance. This avoids hiding bad releases
    // merely because they failed the full-note F1 match.
    let offset_errors: Vec<_> = onset_matching
        .iter()
        .filter_map(|(reference_index, prediction_index)| {
            Some(
                predicted_notes[*prediction_index].offset_seconds?
                    - reference_notes[*reference_index].offset_seconds?,
            )
        })
        .collect();

    let mut report = MidiTranscriptionEvaluationReportV1 {
        schema: MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1.into(),
        algorithm: MIDI_TRANSCRIPTION_EVALUATION_ALGORITHM_V1.into(),
        reference_score_sha256: reference.manifest.score_sha256.clone(),
        reference_midi_sha256: reference.manifest.midi_sha256.clone(),
        reference_manifest_sha256: reference.manifest_sha256.clone(),
        prediction_midi_sha256: sha256_hex(prediction_midi_bytes),
        config,
        reference_note_count: u32::try_from(reference_notes.len()).unwrap_or(u32::MAX),
        prediction_note_count: u32::try_from(predicted_notes.len()).unwrap_or(u32::MAX),
        prediction_unclosed_note_count: parsed.unclosed_note_count,
        prediction_orphan_note_off_count: parsed.orphan_note_off_count,
        onset_pitch_metrics: metric_from_matching(
            reference_notes.len(),
            predicted_notes.len(),
            onset_matching.len(),
        ),
        full_note_metrics: metric_from_matching(
            reference_notes.len(),
            predicted_notes.len(),
            full_matching.len(),
        ),
        exact_program_onset_metrics: metric_from_matching(
            reference_notes.len(),
            predicted_notes.len(),
            program_onset_matching.len(),
        ),
        exact_program_full_note_metrics: metric_from_matching(
            reference_notes.len(),
            predicted_notes.len(),
            program_full_matching.len(),
        ),
        onset_error_ms: timing_error_summary(&onset_errors),
        offset_error_ms: timing_error_summary(&offset_errors),
        report_sha256: String::new(),
    };
    report.report_sha256 = midi_transcription_evaluation_commitment_v1(&report)
        .map_err(|_| MidiTranscriptionEvaluationIssueV1::SerializationFailed)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_performance::{
        MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_SCHEMA_V1,
        MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1,
        MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1,
        MidiPerformanceManifestV1,
        MidiPerformanceNoteReceiptV1,
        MidiPerformanceTrackReceiptV1,
    };

    fn write_vlq(output: &mut Vec<u8>, mut value: u64) {
        let mut bytes = [0u8; 10];
        let mut index = bytes.len() - 1;
        bytes[index] = (value & 0x7f) as u8;
        value >>= 7;
        while value > 0 {
            index -= 1;
            bytes[index] = ((value & 0x7f) as u8) | 0x80;
            value >>= 7;
        }
        output.extend_from_slice(&bytes[index..]);
    }

    fn build_midi(
        notes: &[(u8, u64, u64)],
        program: u8,
        tempo_events: &[(u64, u32)],
    ) -> Vec<u8> {
        let mut note_events: Vec<(u64, u8, Vec<u8>)> =
            vec![(0, 0, vec![0xc0, program])];
        for (pitch, onset, offset) in notes {
            note_events.push((*onset, 2, vec![0x90, *pitch, 100]));
            note_events.push((*offset, 1, vec![0x80, *pitch, 0]));
        }
        note_events.sort_by_key(|event| (event.0, event.1));
        let mut note_track = Vec::new();
        let mut previous_tick = 0u64;
        for (tick, _, message) in note_events {
            write_vlq(&mut note_track, tick - previous_tick);
            note_track.extend_from_slice(&message);
            previous_tick = tick;
        }
        note_track.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);

        let mut tempo_track = Vec::new();
        let mut last_tick = 0u64;
        for (tick, tempo) in tempo_events {
            write_vlq(&mut tempo_track, *tick - last_tick);
            tempo_track.extend_from_slice(&[
                0xff,
                0x51,
                0x03,
                (tempo >> 16) as u8,
                (tempo >> 8) as u8,
                *tempo as u8,
            ]);
            last_tick = *tick;
        }
        if tempo_events.is_empty() {
            tempo_track.extend_from_slice(&[
                0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
            ]);
        }
        tempo_track.extend_from_slice(&[0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        tempo_track.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&2u16.to_be_bytes());
        bytes.extend_from_slice(&MIDI_TRANSCRIPTION_TEST_PPQ.to_be_bytes());
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(tempo_track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&tempo_track);
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(note_track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&note_track);
        bytes
    }

    const MIDI_TRANSCRIPTION_TEST_PPQ: u16 = MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1;

    fn reference_artifact(notes: &[(u8, u64, u64)], program: u8) -> MidiPerformanceArtifactV1 {
        let midi_bytes = build_midi(notes, program, &[(0, 500_000)]);
        let maximum_tick = notes.iter().map(|note| note.2).max().unwrap_or(1);
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
            ticks_per_quarter: MIDI_TRANSCRIPTION_TEST_PPQ,
            tempo_microseconds_per_quarter: 500_000,
            meter_numerator: 4,
            meter_denominator: 4,
            tracks: vec![MidiPerformanceTrackReceiptV1 {
                track_index: 1,
                track_name: "Reference".into(),
                part_id: 1,
                segment_id: "segment-1".into(),
                instrument_id: "instrument-1".into(),
                renderer_id: "fixture-renderer".into(),
                patch_id: "fixture-patch".into(),
                gm_program: program,
                midi_channel: 0,
                segment_start_tick: 0,
                segment_end_tick: maximum_tick + 1,
            }],
            notes: notes
                .iter()
                .enumerate()
                .map(|(ordinal, (pitch, onset, offset))| MidiPerformanceNoteReceiptV1 {
                    source_note_ordinal: ordinal,
                    part_id: 1,
                    segment_id: "segment-1".into(),
                    instrument_id: "instrument-1".into(),
                    track_index: 1,
                    start_tick: *onset,
                    end_tick: *offset,
                    pitch_midi: *pitch,
                    velocity_midi: 100,
                })
                .collect(),
        };
        let manifest_sha256 = domain_separated_sha256(
            MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
            &serde_json::to_vec(&manifest).unwrap(),
        );
        MidiPerformanceArtifactV1 {
            midi_bytes,
            manifest,
            manifest_sha256,
        }
    }

    fn default_config() -> MidiTranscriptionEvaluationConfigV1 {
        MidiTranscriptionEvaluationConfigV1::default()
    }

    #[test]
    fn exact_prediction_has_perfect_instrument_agnostic_and_program_metrics() {
        let notes = vec![(60, 0, 480), (64, 960, 1_440), (67, 1_920, 2_400)];
        let reference = reference_artifact(&notes, 40);
        let prediction = build_midi(&notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.f1, 1.0);
        assert_eq!(report.full_note_metrics.f1, 1.0);
        assert_eq!(report.exact_program_onset_metrics.f1, 1.0);
        assert_eq!(report.exact_program_full_note_metrics.f1, 1.0);
        assert_eq!(report.onset_error_ms.mean_absolute_error_ms, Some(0.0));
    }

    #[test]
    fn wrong_gm_program_changes_only_exact_program_metrics() {
        let notes = vec![(60, 0, 480), (64, 960, 1_440), (67, 1_920, 2_400)];
        let reference = reference_artifact(&notes, 40);
        let prediction = build_midi(&notes, 41, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.f1, 1.0);
        assert_eq!(report.full_note_metrics.f1, 1.0);
        assert_eq!(report.exact_program_onset_metrics.f1, 0.0);
        assert_eq!(report.exact_program_full_note_metrics.f1, 0.0);
    }

    #[test]
    fn onset_shift_beyond_tolerance_reduces_onset_and_full_note_f1() {
        let reference_notes = vec![(60, 0, 480), (64, 960, 1_440)];
        let prediction_notes = vec![(60, 100, 580), (64, 1_060, 1_540)];
        let reference = reference_artifact(&reference_notes, 40);
        let prediction = build_midi(&prediction_notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
            None,
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.true_positives, 0);
        assert_eq!(report.full_note_metrics.true_positives, 0);
    }

    #[test]
    fn offset_only_error_preserves_onset_f1_but_reduces_full_note_f1() {
        let reference_notes = vec![(60, 0, 480), (64, 960, 1_440)];
        let prediction_notes = vec![(60, 0, 800), (64, 960, 1_760)];
        let reference = reference_artifact(&reference_notes, 40);
        let prediction = build_midi(&prediction_notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
            None,
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.f1, 1.0);
        assert_eq!(report.full_note_metrics.true_positives, 0);
        assert_eq!(report.offset_error_ms.sample_count, 0);
    }

    #[test]
    fn tempo_map_is_integrated_when_transcription_ticks_are_converted_to_seconds() {
        // 960 ticks at 500,000 us/qn = 0.5s; then a 1,000,000 us/qn tempo.
        // The note from tick 960 to 1,440 therefore lasts 0.5s, from 0.5s to 1.0s.
        let prediction = build_midi(
            &[(60, 960, 1_440)],
            40,
            &[(0, 500_000), (960, 1_000_000)],
        );
        let parsed = parse_transcribed_midi_v1(&prediction).unwrap();
        assert_eq!(parsed.notes.len(), 1);
        assert!((parsed.notes[0].onset_seconds - 0.5).abs() < 1e-9);
        assert!((parsed.notes[0].offset_seconds.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(parsed.tempo_point_count, 2);
    }

    #[test]
    fn unclosed_note_is_kept_for_onset_metrics_but_cannot_match_full_note() {
        let notes = vec![(60, 0, 480)];
        let reference = reference_artifact(&notes, 40);
        // Use a valid type-0 prediction containing a NoteOn without NoteOff.
        let mut raw = Vec::new();
        raw.extend_from_slice(b"MThd");
        raw.extend_from_slice(&6u32.to_be_bytes());
        raw.extend_from_slice(&1u16.to_be_bytes());
        raw.extend_from_slice(&1u16.to_be_bytes());
        raw.extend_from_slice(&MIDI_TRANSCRIPTION_TEST_PPQ.to_be_bytes());
        raw.extend_from_slice(b"MTrk");
        let track = [0x00, 0xc0, 40, 0x00, 0x90, 60, 100, 0x00, 0xff, 0x2f, 0x00];
        raw.extend_from_slice(&(track.len() as u32).to_be_bytes());
        raw.extend_from_slice(&track);
        let report = evaluate_midi_transcription_v1(&raw, &reference, default_config()).unwrap();
        assert_eq!(report.prediction_unclosed_note_count, 1);
        assert_eq!(report.onset_pitch_metrics.true_positives, 1);
        assert_eq!(report.full_note_metrics.true_positives, 0);
    }
}
