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
use std::collections::{BTreeMap, VecDeque};

pub const MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1: &str =
    "muse-midi-transcription-evaluation-v1";
pub const MIDI_TRANSCRIPTION_EVALUATION_ALGORITHM_V1: &str =
    "mir-eval-0.8.2-hopcroft-karp-matching-thresholds-v1";
pub const MIDI_TRANSCRIPTION_DEFAULT_ONSET_TOLERANCE_MS_V1: u32 = 50;
pub const MIDI_TRANSCRIPTION_DEFAULT_PITCH_TOLERANCE_CENTS_V1: u16 = 50;
pub const MIDI_TRANSCRIPTION_DEFAULT_OFFSET_RATIO_BPS_V1: u16 = 2_000;
pub const MIDI_TRANSCRIPTION_DEFAULT_OFFSET_MIN_TOLERANCE_MS_V1: u32 = 50;
pub const MIDI_TRANSCRIPTION_DEFAULT_FRAME_STEP_MS_V1: u16 = 10;
pub const MIDI_TRANSCRIPTION_MAX_NOTES_V1: usize = 20_000;
pub const MIDI_TRANSCRIPTION_MAX_MIDI_BYTES_V1: usize = 64 * 1024 * 1024;
pub const MIDI_TRANSCRIPTION_MAX_MIDI_EVENTS_V1: usize = 1_000_000;
pub const MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATION_SCANS_V1: usize = 10_000_000;
pub const MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATIONS_V1: usize = 2_000_000;
pub const MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1: usize = 2_000_000;
pub const MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1: usize = 10_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiTranscriptionEvaluationConfigV1 {
    pub onset_tolerance_ms: u32,
    pub pitch_tolerance_cents: u16,
    pub offset_ratio_basis_points: u16,
    pub offset_min_tolerance_ms: u32,
    /// Enable only when the prediction pipeline actually estimates dynamics.
    /// Many AMT models serialize placeholder MIDI velocities.
    pub velocity_metrics_enabled: bool,
    pub velocity_tolerance: u8,
    pub frame_step_ms: u16,
}

impl Default for MidiTranscriptionEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            onset_tolerance_ms: MIDI_TRANSCRIPTION_DEFAULT_ONSET_TOLERANCE_MS_V1,
            pitch_tolerance_cents: MIDI_TRANSCRIPTION_DEFAULT_PITCH_TOLERANCE_CENTS_V1,
            offset_ratio_basis_points: MIDI_TRANSCRIPTION_DEFAULT_OFFSET_RATIO_BPS_V1,
            offset_min_tolerance_ms: MIDI_TRANSCRIPTION_DEFAULT_OFFSET_MIN_TOLERANCE_MS_V1,
            velocity_metrics_enabled: false,
            velocity_tolerance: 10,
            frame_step_ms: MIDI_TRANSCRIPTION_DEFAULT_FRAME_STEP_MS_V1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MidiTranscribedNoteV1 {
    pub track_index: u16,
    pub midi_channel: u8,
    pub onset_tick: u64,
    pub offset_tick: Option<u64>,
    /// Stable event order after the deterministic Type-1 track merge.
    pub source_event_ordinal: u32,
    /// GM program captured at NoteOn; percussion channel 9 is retained.
    pub gm_program: u8,
    pub pitch_midi: u8,
    pub velocity_midi: u8,
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
    pub overlapping_same_pitch_note_on_count: u32,
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
pub struct MidiTranscriptionValueErrorSummaryV1 {
    pub sample_count: u32,
    pub mean_absolute_error: Option<f64>,
    pub median_absolute_error: Option<f64>,
    pub p95_absolute_error: Option<f64>,
    pub p99_absolute_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTranscriptionTimingErrorSummaryV1 {
    pub sample_count: u32,
    pub mean_signed_error_ms: Option<f64>,
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
    pub prediction_tempo_point_count: u32,
    pub prediction_overlapping_same_pitch_note_on_count: u32,
    /// mir_eval-style onset-only metric; deliberately ignores pitch.
    pub onset_only_metrics: MidiTranscriptionMetricV1,
    /// mir_eval-style offset-only metric; deliberately ignores pitch and onset.
    pub offset_only_metrics: MidiTranscriptionMetricV1,
    /// Instrument-agnostic: pitch and onset must match.
    pub onset_pitch_metrics: MidiTranscriptionMetricV1,
    /// Instrument-agnostic: pitch, onset and offset must match.
    pub full_note_metrics: MidiTranscriptionMetricV1,
    /// Average interval overlap ratio over full-note matches; zero if none.
    pub average_overlap_ratio: f64,
    /// Exact GM program, pitch and onset must match.
    pub exact_program_onset_metrics: MidiTranscriptionMetricV1,
    /// Exact GM program, pitch, onset and offset must match.
    pub exact_program_full_note_metrics: MidiTranscriptionMetricV1,
    /// Full-note match plus MIDI velocity within the configured tolerance.
    pub velocity_aware_full_note_metrics: Option<MidiTranscriptionMetricV1>,
    /// Framewise pitch activity; simultaneous same-pitch notes collapse into one active cell.
    pub frame_pitch_metrics: MidiTranscriptionMetricV1,
    /// Framewise GM-program/pitch activity; not a persistent instrument identity metric.
    pub frame_program_pitch_metrics: MidiTranscriptionMetricV1,
    pub frame_step_ms: u16,
    pub velocity_error: Option<MidiTranscriptionValueErrorSummaryV1>,
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
    MidiTooLarge { count: usize, maximum: usize },
    TooManyMidiEvents { count: usize, maximum: usize },
    UnsupportedMidiFormat,
    UnsupportedSmpteTiming,
    InvalidTempo,
    ConflictingProgramChange { tick: u64, channel: u8 },
    MidiTickOverflow,
    TooManyPredictedNotes { count: usize, maximum: usize },
    InvalidReferenceTrack { track_index: u16 },
    InvalidReferenceNote { ordinal: usize },
    TooManyCandidatePairs { count: usize, maximum: usize },
    MatchingWorkBudgetExceeded { scans: usize, maximum: usize },
    SerializationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MidiTempoPointV1 {
    pub start_tick: u64,
    pub start_seconds: f64,
    pub microseconds_per_quarter: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct EvaluationNoteV1 {
    pub(crate) pitch_midi: u8,
    pub(crate) gm_program: u8,
    pub(crate) velocity_midi: u8,
    pub(crate) onset_seconds: f64,
    pub(crate) offset_seconds: Option<f64>,
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
    prediction_tempo_point_count: u32,
    prediction_overlapping_same_pitch_note_on_count: u32,
    onset_only_metrics: &'a MidiTranscriptionMetricV1,
    offset_only_metrics: &'a MidiTranscriptionMetricV1,
    onset_pitch_metrics: &'a MidiTranscriptionMetricV1,
    full_note_metrics: &'a MidiTranscriptionMetricV1,
    average_overlap_ratio: f64,
    exact_program_onset_metrics: &'a MidiTranscriptionMetricV1,
    exact_program_full_note_metrics: &'a MidiTranscriptionMetricV1,
    velocity_aware_full_note_metrics: &'a Option<MidiTranscriptionMetricV1>,
    frame_pitch_metrics: &'a MidiTranscriptionMetricV1,
    frame_program_pitch_metrics: &'a MidiTranscriptionMetricV1,
    frame_step_ms: u16,
    velocity_error: &'a Option<MidiTranscriptionValueErrorSummaryV1>,
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

pub(crate) fn validate_config(
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
    if config.velocity_tolerance > 127 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidConfig {
            field: "velocity_tolerance".into(),
        });
    }
    if config.frame_step_ms == 0 || config.frame_step_ms > 1_000 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidConfig {
            field: "frame_step_ms".into(),
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
    let mut explicit_tempo_at_zero_seen = false;

    for (tick, _, _, tempo) in sorted {
        if tempo == 0 {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidTempo);
        }
        let previous = *points.last().expect("initial tempo point is present");
        if tick < previous.start_tick {
            return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
        }
        if tick == 0 && !explicit_tempo_at_zero_seen {
            // The first explicit tempo at tick zero overrides the MIDI
            // default tempo. Subsequent conflicting explicit events at the
            // same tick remain invalid.
            points[0].microseconds_per_quarter = tempo;
            explicit_tempo_at_zero_seen = true;
            continue;
        }
        if tick == previous.start_tick {
            if tempo != previous.microseconds_per_quarter {
                // Conflicting tempo events at the same absolute tick across
                // tracks are ambiguous; do not let track ordering choose a
                // benchmark timebase silently.
                return Err(MidiTranscriptionEvaluationIssueV1::InvalidTempo);
            }
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
    if midi_bytes.len() > MIDI_TRANSCRIPTION_MAX_MIDI_BYTES_V1 {
        return Err(MidiTranscriptionEvaluationIssueV1::MidiTooLarge {
            count: midi_bytes.len(),
            maximum: MIDI_TRANSCRIPTION_MAX_MIDI_BYTES_V1,
        });
    }
    let smf = Smf::parse(midi_bytes)
        .map_err(|_| MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi)?;
    if !matches!(smf.header.format, Format::SingleTrack | Format::Parallel) {
        return Err(MidiTranscriptionEvaluationIssueV1::UnsupportedMidiFormat);
    }
    if smf.header.format == Format::SingleTrack && smf.tracks.len() != 1 {
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
    let mut absolute_events = Vec::new();
    let mut midi_event_count = 0usize;
    let mut program_at_tick: BTreeMap<(u64, u8), (usize, u8)> = BTreeMap::new();

    // Type-1 MIDI tracks are parallel parts on a shared channel namespace.
    // Merge their absolute events before applying channel program state or
    // pairing overlapping same-pitch NoteOn/NoteOff events.
    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0u64;
        for (event_index, event) in track.iter().enumerate() {
            midi_event_count = midi_event_count.saturating_add(1);
            if midi_event_count > MIDI_TRANSCRIPTION_MAX_MIDI_EVENTS_V1 {
                return Err(MidiTranscriptionEvaluationIssueV1::TooManyMidiEvents {
                    count: midi_event_count,
                    maximum: MIDI_TRANSCRIPTION_MAX_MIDI_EVENTS_V1,
                });
            }
            tick = tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                tempo_events.push((tick, track_index, event_index, tempo.as_int()));
            }
            if let TrackEventKind::Midi {
                channel,
                message: MidiMessage::ProgramChange { program },
            } = event.kind
            {
                let channel = channel.as_int();
                let program = program.as_int();
                if program_at_tick
                    .insert((tick, channel), (track_index, program))
                    .is_some_and(|(prior_track, prior_program)| {
                        prior_track != track_index && prior_program != program
                    })
                {
                    return Err(MidiTranscriptionEvaluationIssueV1::ConflictingProgramChange {
                        tick,
                        channel,
                    });
                }
            }
            absolute_events.push((tick, track_index, event_index, event.kind));
        }
    }

    let tempo_points = tempo_points_from_events(&tempo_events, ticks_per_quarter)?;
    // Preserve each track's event order at equal ticks. Cross-track tie
    // order is deterministic (track index, then event index); MIDI Type 1
    // does not define a stronger total order for simultaneous events on a
    // shared channel.
    absolute_events.sort_by_key(|(tick, track_index, event_index, _)| {
        (*tick, *track_index, *event_index)
    });

    let mut notes = Vec::new();
    let mut unclosed_note_count = 0usize;
    let mut orphan_note_off_count = 0usize;
    let mut overlapping_same_pitch_note_on_count = 0usize;
    let mut programs = [0u8; 16];
    let mut pending: BTreeMap<(u8, u8), VecDeque<(u64, u8, u8, u16, u32)>> = BTreeMap::new();
    let mut pending_note_count = 0usize;

    for (event_ordinal, (tick, track_index, _, kind)) in
        absolute_events.into_iter().enumerate()
    {
        let source_event_ordinal = u32::try_from(event_ordinal)
            .map_err(|_| MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
        match kind {
            TrackEventKind::Midi { channel, message } => {
                let channel = channel.as_int();
                match message {
                    MidiMessage::ProgramChange { program } => {
                        programs[usize::from(channel)] = program.as_int();
                    }
                    MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                        if notes.len().saturating_add(pending_note_count)
                            >= MIDI_TRANSCRIPTION_MAX_NOTES_V1
                        {
                            return Err(MidiTranscriptionEvaluationIssueV1::TooManyPredictedNotes {
                                count: notes.len().saturating_add(pending_note_count).saturating_add(1),
                                maximum: MIDI_TRANSCRIPTION_MAX_NOTES_V1,
                            });
                        }
                        let queue = pending.entry((channel, key.as_int())).or_default();
                        if !queue.is_empty() {
                            overlapping_same_pitch_note_on_count += 1;
                        }
                        let track_index_u16 = u16::try_from(track_index)
                            .map_err(|_| MidiTranscriptionEvaluationIssueV1::UnsupportedMidiFormat)?;
                        queue.push_back((
                            tick,
                            programs[usize::from(channel)],
                            vel.as_int(),
                            track_index_u16,
                            source_event_ordinal,
                        ));
                        pending_note_count += 1;
                    }
                    MidiMessage::NoteOff { key, .. }
                    | MidiMessage::NoteOn { key, .. } => {
                        if let Some(queue) = pending.get_mut(&(channel, key.as_int())) {
                            if let Some((
                                start_tick,
                                program,
                                velocity,
                                onset_track_index,
                                note_on_event_ordinal,
                            )) = queue.pop_front()
                            {
                                pending_note_count = pending_note_count.saturating_sub(1);
                                let onset_seconds = tick_to_seconds(
                                    start_tick,
                                    &tempo_points,
                                    ticks_per_quarter,
                                )
                                .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
                                let offset_seconds = tick_to_seconds(
                                    tick,
                                    &tempo_points,
                                    ticks_per_quarter,
                                )
                                .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
                                if offset_seconds <= onset_seconds {
                                    return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
                                }
                                notes.push(MidiTranscribedNoteV1 {
                                    track_index: onset_track_index,
                                    midi_channel: channel,
                                    onset_tick: start_tick,
                                    offset_tick: Some(tick),
                                    source_event_ordinal: note_on_event_ordinal,
                                    gm_program: program,
                                    pitch_midi: key.as_int(),
                                    velocity_midi: velocity,
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
        for (start_tick, program, velocity, onset_track_index, note_on_event_ordinal) in queue {
            let onset_seconds = tick_to_seconds(
                start_tick,
                &tempo_points,
                ticks_per_quarter,
            )
            .ok_or(MidiTranscriptionEvaluationIssueV1::MidiTickOverflow)?;
            unclosed_note_count += 1;
            notes.push(MidiTranscribedNoteV1 {
                track_index: onset_track_index,
                midi_channel: channel,
                onset_tick: start_tick,
                offset_tick: None,
                source_event_ordinal: note_on_event_ordinal,
                gm_program: program,
                pitch_midi: pitch,
                velocity_midi: velocity,
                onset_seconds,
                offset_seconds: None,
            });
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
            .then_with(|| left.source_event_ordinal.cmp(&right.source_event_ordinal))
    });

    Ok(ParsedMidiTranscriptionV1 {
        notes,
        ticks_per_quarter,
        tempo_point_count: u32::try_from(tempo_points.len()).unwrap_or(u32::MAX),
        tempo_points,
        unclosed_note_count: u32::try_from(unclosed_note_count).unwrap_or(u32::MAX),
        orphan_note_off_count: u32::try_from(orphan_note_off_count).unwrap_or(u32::MAX),
        overlapping_same_pitch_note_on_count:
            u32::try_from(overlapping_same_pitch_note_on_count).unwrap_or(u32::MAX),
    })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn validate_reference_artifact_v1(
    reference: &MidiPerformanceArtifactV1,
) -> Result<(), MidiTranscriptionEvaluationIssueV1> {
    use midly::{MetaMessage, MidiMessage, TrackEventKind};

    let manifest = &reference.manifest;
    if manifest.notes.len() > MIDI_TRANSCRIPTION_MAX_NOTES_V1 {
        return Err(MidiTranscriptionEvaluationIssueV1::InvalidReference);
    }
    if reference.midi_bytes.len() > MIDI_TRANSCRIPTION_MAX_MIDI_BYTES_V1 {
        return Err(MidiTranscriptionEvaluationIssueV1::MidiTooLarge {
            count: reference.midi_bytes.len(),
            maximum: MIDI_TRANSCRIPTION_MAX_MIDI_BYTES_V1,
        });
    }
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
    for (expected_ordinal, note) in manifest.notes.iter().enumerate() {
        if note.source_note_ordinal != expected_ordinal
            || !ordinals.insert(note.source_note_ordinal)
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
    let mut reference_event_count = 0usize;

    for (track_index, events) in smf.tracks.iter().enumerate() {
        let track_index = u16::try_from(track_index)
            .map_err(|_| MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest)?;
        let mut absolute_tick = 0u64;
        for event in events {
            reference_event_count = reference_event_count.saturating_add(1);
            if reference_event_count > MIDI_TRANSCRIPTION_MAX_MIDI_EVENTS_V1 {
                return Err(MidiTranscriptionEvaluationIssueV1::TooManyMidiEvents {
                    count: reference_event_count,
                    maximum: MIDI_TRANSCRIPTION_MAX_MIDI_EVENTS_V1,
                });
            }
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

pub(crate) fn reference_notes(
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
            velocity_midi: note.velocity_midi,
            onset_seconds,
            offset_seconds: Some(offset_seconds),
        });
    }
    Ok(notes)
}

pub(crate) fn maximum_cardinality_matching(
    adjacency: &[Vec<usize>],
    predicted_count: usize,
) -> Result<Vec<(usize, usize)>, MidiTranscriptionEvaluationIssueV1> {
    // Port of mir_eval.util._bipartite_match (Eppstein/Hopcroft-Karp) with
    // explicit stacks instead of recursive DFS. The graph orientation and
    // traversal order mirror mir_eval: predicted vertices in MIDI output
    // order, each connected to reference vertices in reference order.
    let reference_count = adjacency.len();
    let mut graph = vec![Vec::<usize>::new(); predicted_count];
    let mut prediction_order = Vec::new();
    let mut prediction_seen = vec![false; predicted_count];
    for (reference_index, predictions) in adjacency.iter().enumerate() {
        for &prediction_index in predictions {
            if prediction_index >= predicted_count {
                return Err(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi);
            }
            if !prediction_seen[prediction_index] {
                prediction_seen[prediction_index] = true;
                prediction_order.push(prediction_index);
            }
            graph[prediction_index].push(reference_index);
        }
    }

    let mut reference_to_prediction = vec![None; reference_count];
    let mut prediction_to_reference = vec![None; predicted_count];
    let mut edge_scans = 0usize;

    // Upstream greedy initialization.
    for &prediction_index in &prediction_order {
        for &reference_index in &graph[prediction_index] {
            edge_scans = edge_scans.saturating_add(1);
            if edge_scans > MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1 {
                return Err(MidiTranscriptionEvaluationIssueV1::MatchingWorkBudgetExceeded {
                    scans: edge_scans,
                    maximum: MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1,
                });
            }
            if reference_to_prediction[reference_index].is_none() {
                reference_to_prediction[reference_index] = Some(prediction_index);
                prediction_to_reference[prediction_index] = Some(reference_index);
                break;
            }
        }
    }

    // predecessor_of_prediction:
    //   None      => not in the current layered residual graph
    //   Some(None)=> unmatched root prediction vertex
    //   Some(Some(reference)) => reached through that matched reference vertex
    let mut predecessor_of_prediction = vec![None::<Option<usize>>; predicted_count];
    let mut predecessors_of_reference = vec![None::<Vec<usize>>; reference_count];
    let mut new_layer_generation = vec![0u32; reference_count];
    let mut new_layer_slot = vec![0usize; reference_count];
    let mut generation = 0u32;

    loop {
        predecessor_of_prediction.fill(None);
        for prediction_index in 0..predicted_count {
            if prediction_to_reference[prediction_index].is_none()
                && !graph[prediction_index].is_empty()
            {
                predecessor_of_prediction[prediction_index] = Some(None);
            }
        }

        let mut layer: Vec<usize> = prediction_order
            .iter()
            .copied()
            .filter(|&prediction_index| predecessor_of_prediction[prediction_index].is_some())
            .collect();
        predecessors_of_reference.fill(None);
        let mut unmatched_references = Vec::new();

        // Build Hopcroft-Karp alternating layers. Preserve first-encounter
        // order exactly as the upstream dictionary implementation does.
        while !layer.is_empty() && unmatched_references.is_empty() {
            generation = generation.wrapping_add(1);
            if generation == 0 {
                new_layer_generation.fill(0);
                generation = 1;
            }
            let mut new_layer: Vec<(usize, Vec<usize>)> = Vec::new();

            for &prediction_index in &layer {
                for &reference_index in &graph[prediction_index] {
                    edge_scans = edge_scans.saturating_add(1);
                    if edge_scans > MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1 {
                        return Err(MidiTranscriptionEvaluationIssueV1::MatchingWorkBudgetExceeded {
                            scans: edge_scans,
                            maximum: MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1,
                        });
                    }
                    if predecessors_of_reference[reference_index].is_some() {
                        continue;
                    }
                    let slot = if new_layer_generation[reference_index] != generation {
                        new_layer_generation[reference_index] = generation;
                        let slot = new_layer.len();
                        new_layer_slot[reference_index] = slot;
                        new_layer.push((reference_index, Vec::new()));
                        slot
                    } else {
                        new_layer_slot[reference_index]
                    };
                    new_layer[slot].1.push(prediction_index);
                }
            }

            layer.clear();
            for (reference_index, predecessor_predictions) in new_layer {
                predecessors_of_reference[reference_index] = Some(predecessor_predictions);
                if let Some(matched_prediction) = reference_to_prediction[reference_index] {
                    layer.push(matched_prediction);
                    predecessor_of_prediction[matched_prediction] = Some(Some(reference_index));
                } else {
                    unmatched_references.push(reference_index);
                }
            }
        }

        if unmatched_references.is_empty() {
            break;
        }

        // Iterative equivalent of mir_eval's recursive recurse(reference).
        // Each frame retains the predecessor prediction selected by its
        // parent so a successful path can be unwound without recursion.
        struct DfsFrame {
            reference_index: usize,
            neighbors: Vec<usize>,
            next_neighbor: usize,
            incoming_prediction: Option<usize>,
        }

        for root_reference in unmatched_references {
            let Some(root_neighbors) = predecessors_of_reference[root_reference].take() else {
                continue;
            };
            let mut stack = vec![DfsFrame {
                reference_index: root_reference,
                neighbors: root_neighbors,
                next_neighbor: 0,
                incoming_prediction: None,
            }];
            let mut augmented = false;

            while !stack.is_empty() && !augmented {
                let top = stack.len() - 1;
                let mut pushed = false;

                while stack[top].next_neighbor < stack[top].neighbors.len() {
                    let prediction_index = stack[top].neighbors[stack[top].next_neighbor];
                    stack[top].next_neighbor += 1;
                    edge_scans = edge_scans.saturating_add(1);
                    if edge_scans > MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1 {
                        return Err(MidiTranscriptionEvaluationIssueV1::MatchingWorkBudgetExceeded {
                            scans: edge_scans,
                            maximum: MIDI_TRANSCRIPTION_MAX_MATCHING_EDGE_SCANS_V1,
                        });
                    }

                    let Some(previous_reference) =
                        predecessor_of_prediction[prediction_index].take()
                    else {
                        continue;
                    };

                    match previous_reference {
                        None => {
                            let terminal_reference = stack[top].reference_index;
                            reference_to_prediction[terminal_reference] = Some(prediction_index);
                            prediction_to_reference[prediction_index] = Some(terminal_reference);

                            for stack_index in (0..top).rev() {
                                let parent_reference = stack[stack_index].reference_index;
                                let incoming_prediction = stack[stack_index + 1]
                                    .incoming_prediction
                                    .ok_or(MidiTranscriptionEvaluationIssueV1::InvalidPredictionMidi)?;
                                reference_to_prediction[parent_reference] =
                                    Some(incoming_prediction);
                                prediction_to_reference[incoming_prediction] =
                                    Some(parent_reference);
                            }
                            augmented = true;
                            break;
                        }
                        Some(previous_reference) => {
                            if let Some(neighbors) =
                                predecessors_of_reference[previous_reference].take()
                            {
                                stack.push(DfsFrame {
                                    reference_index: previous_reference,
                                    neighbors,
                                    next_neighbor: 0,
                                    incoming_prediction: Some(prediction_index),
                                });
                                pushed = true;
                                break;
                            }
                        }
                    }
                }

                if !augmented && !pushed {
                    stack.pop();
                }
            }
        }
    }

    let mut matching: Vec<_> = reference_to_prediction
        .iter()
        .enumerate()
        .filter_map(|(reference_index, prediction_index)| {
            prediction_index.map(|prediction_index| (reference_index, prediction_index))
        })
        .collect();
    matching.sort_unstable_by_key(|(reference_index, _)| *reference_index);
    Ok(matching)
}

fn mir_eval_round_distance_v1(distance: f64) -> f64 {
    // mir_eval 0.8.2 rounds onset/offset distances to N_DECIMALS = 4
    // before its inclusive threshold comparison.
    (distance * 10_000.0).round_ties_even() / 10_000.0
}

pub(crate) fn build_adjacency(
    reference: &[EvaluationNoteV1],
    predicted: &[MidiTranscribedNoteV1],
    config: &MidiTranscriptionEvaluationConfigV1,
    require_offset: bool,
    require_program: bool,
    require_velocity: bool,
) -> Result<Vec<Vec<usize>>, MidiTranscriptionEvaluationIssueV1> {
    // Prediction events are sorted by onset by parse_transcribed_midi_v1.
    // Use binary-search time windows and per-pitch candidates rather than the
    // O(reference_notes * predicted_notes) Cartesian scan.
    let onset_tolerance_seconds = f64::from(config.onset_tolerance_ms) / 1_000.0;
    let pitch_tolerance_cents = i32::from(config.pitch_tolerance_cents);
    let mut adjacency = Vec::with_capacity(reference.len());
    let mut edge_count = 0usize;

    for reference_note in reference {
        let lower_time = reference_note.onset_seconds - onset_tolerance_seconds - 0.00005;
        let upper_time = reference_note.onset_seconds + onset_tolerance_seconds + 0.00005;
        let start = predicted.partition_point(|note| note.onset_seconds < lower_time);
        let end = predicted.partition_point(|note| note.onset_seconds <= upper_time);
        let reference_duration = reference_note
            .offset_seconds
            .unwrap_or(reference_note.onset_seconds)
            - reference_note.onset_seconds;
        let offset_tolerance = (reference_duration
            * f64::from(config.offset_ratio_basis_points)
            / 10_000.0)
            .max(f64::from(config.offset_min_tolerance_ms) / 1_000.0);

        let mut candidates = Vec::new();
        for (prediction_index, predicted_note) in predicted.iter().enumerate().take(end).skip(start) {
            if mir_eval_round_distance_v1(
                (reference_note.onset_seconds - predicted_note.onset_seconds).abs(),
            ) > onset_tolerance_seconds
            {
                continue;
            }
            let pitch_error_cents = (i32::from(reference_note.pitch_midi)
                - i32::from(predicted_note.pitch_midi))
                .unsigned_abs() as i32
                * 100;
            if pitch_error_cents > pitch_tolerance_cents {
                continue;
            }
            if require_program && reference_note.gm_program != predicted_note.gm_program {
                continue;
            }
            if require_velocity
                && (i16::from(reference_note.velocity_midi)
                    - i16::from(predicted_note.velocity_midi))
                    .unsigned_abs() > u16::from(config.velocity_tolerance)
            {
                continue;
            }
            if require_offset {
                let Some(predicted_offset) = predicted_note.offset_seconds else {
                    continue;
                };
                let Some(reference_offset) = reference_note.offset_seconds else {
                    continue;
                };
                if mir_eval_round_distance_v1(
                    (reference_offset - predicted_offset).abs(),
                ) > offset_tolerance
                {
                    continue;
                }
            }
            candidates.push(prediction_index);
            edge_count = edge_count.saturating_add(1);
            if edge_count > MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1 {
                return Err(MidiTranscriptionEvaluationIssueV1::TooManyCandidatePairs {
                    count: edge_count,
                    maximum: MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1,
                });
            }
        }
        adjacency.push(candidates);
    }
    Ok(adjacency)
}

pub(crate) fn build_onset_only_adjacency(
    reference: &[EvaluationNoteV1],
    predicted: &[MidiTranscribedNoteV1],
    config: &MidiTranscriptionEvaluationConfigV1,
) -> Result<Vec<Vec<usize>>, MidiTranscriptionEvaluationIssueV1> {
    let tolerance = f64::from(config.onset_tolerance_ms) / 1_000.0;
    let mut adjacency = Vec::with_capacity(reference.len());
    let mut edge_count = 0usize;
    for note in reference {
        let lower = note.onset_seconds - tolerance - 0.00005;
        let upper = note.onset_seconds + tolerance + 0.00005;
        let start = predicted.partition_point(|predicted_note| predicted_note.onset_seconds < lower);
        let end = predicted.partition_point(|predicted_note| predicted_note.onset_seconds <= upper);
        let candidates: Vec<usize> = (start..end)
            .filter(|&prediction_index| {
                mir_eval_round_distance_v1(
                    (note.onset_seconds - predicted[prediction_index].onset_seconds).abs(),
                ) <= tolerance
            })
            .collect();
        edge_count = edge_count.saturating_add(candidates.len());
        if edge_count > MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1 {
            return Err(MidiTranscriptionEvaluationIssueV1::TooManyCandidatePairs {
                count: edge_count,
                maximum: MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1,
            });
        }
        adjacency.push(candidates);
    }
    Ok(adjacency)
}

pub(crate) fn build_offset_only_adjacency(
    reference: &[EvaluationNoteV1],
    predicted: &[MidiTranscribedNoteV1],
    config: &MidiTranscriptionEvaluationConfigV1,
) -> Result<Vec<Vec<usize>>, MidiTranscriptionEvaluationIssueV1> {
    let mut predicted_offsets: Vec<(f64, usize)> = predicted
        .iter()
        .enumerate()
        .filter_map(|(index, note)| note.offset_seconds.map(|offset| (offset, index)))
        .collect();
    predicted_offsets.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut adjacency = Vec::with_capacity(reference.len());
    let mut edge_count = 0usize;
    for note in reference {
        let Some(reference_offset) = note.offset_seconds else {
            adjacency.push(Vec::new());
            continue;
        };
        let duration = reference_offset - note.onset_seconds;
        let tolerance = (duration
            * f64::from(config.offset_ratio_basis_points)
            / 10_000.0)
            .max(f64::from(config.offset_min_tolerance_ms) / 1_000.0);
        let lower = reference_offset - tolerance - 0.00005;
        let upper = reference_offset + tolerance + 0.00005;
        let start = predicted_offsets.partition_point(|(offset, _)| *offset < lower);
        let end = predicted_offsets.partition_point(|(offset, _)| *offset <= upper);
        let mut candidates: Vec<usize> = predicted_offsets[start..end]
            .iter()
            .filter_map(|(offset, index)| {
                (mir_eval_round_distance_v1((reference_offset - *offset).abs()) <= tolerance)
                    .then_some(*index)
            })
            .collect();
        candidates.sort_unstable();
        edge_count = edge_count.saturating_add(candidates.len());
        if edge_count > MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1 {
            return Err(MidiTranscriptionEvaluationIssueV1::TooManyCandidatePairs {
                count: edge_count,
                maximum: MIDI_TRANSCRIPTION_MAX_CANDIDATE_PAIRS_V1,
            });
        }
        adjacency.push(candidates);
    }
    Ok(adjacency)
}

fn average_overlap_ratio_v1(
    reference: &[EvaluationNoteV1],
    predicted: &[MidiTranscribedNoteV1],
    matching: &[(usize, usize)],
) -> f64 {
    if matching.is_empty() {
        return 0.0;
    }
    let mut ratios = Vec::with_capacity(matching.len());
    for (reference_index, prediction_index) in matching {
        let Some(reference_note) = reference.get(*reference_index) else {
            return 0.0;
        };
        let Some(predicted_note) = predicted.get(*prediction_index) else {
            return 0.0;
        };
        let (Some(reference_offset), Some(predicted_offset)) = (
            reference_note.offset_seconds,
            predicted_note.offset_seconds,
        ) else {
            return 0.0;
        };
        let overlap = (reference_offset.min(predicted_offset)
            - reference_note.onset_seconds.max(predicted_note.onset_seconds))
            .max(0.0);
        let union_span = reference_offset.max(predicted_offset)
            - reference_note.onset_seconds.min(predicted_note.onset_seconds);
        if !union_span.is_finite() || union_span <= 0.0 {
            return 0.0;
        }
        ratios.push((overlap / union_span).clamp(0.0, 1.0));
    }
    ratios.iter().sum::<f64>() / ratios.len() as f64
}

pub(crate) fn metric_from_matching(
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

fn frame_activation_sets(
    notes: &[EvaluationNoteV1],
    frame_step_ms: u16,
) -> Result<
    (BTreeSet<(u8, u64)>, BTreeSet<(u8, u8, u64)>),
    MidiTranscriptionEvaluationIssueV1,
> {
    let frame_step_seconds = f64::from(frame_step_ms) / 1_000.0;
    let mut pitch_frames = BTreeSet::new();
    let mut program_pitch_frames = BTreeSet::new();
    let mut frame_scans = 0usize;

    for note in notes {
        let Some(offset_seconds) = note.offset_seconds else {
            // Unknown release time cannot contribute a defensible frame span.
            continue;
        };
        if !note.onset_seconds.is_finite()
            || !offset_seconds.is_finite()
            || note.onset_seconds < 0.0
            || offset_seconds <= note.onset_seconds
        {
            continue;
        }
        // Frames are half-open time cells. A note activates every cell with
        // non-zero interval overlap; repeated same-pitch activations collapse
        // to one pitch/frame for the instrument-agnostic view.
        let start_value = (note.onset_seconds / frame_step_seconds).floor();
        let end_value = (offset_seconds / frame_step_seconds).ceil();
        if !start_value.is_finite()
            || !end_value.is_finite()
            || start_value < 0.0
            || end_value > u64::MAX as f64
        {
            return Err(MidiTranscriptionEvaluationIssueV1::FrameActivationBudgetExceeded {
                scans: usize::MAX,
                maximum: MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATION_SCANS_V1,
            });
        }
        let start = start_value as u64;
        let end = (end_value as u64).max(start.saturating_add(1));
        for frame in start..end {
            frame_scans = frame_scans.saturating_add(1);
            if frame_scans > MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATION_SCANS_V1 {
                return Err(MidiTranscriptionEvaluationIssueV1::FrameActivationBudgetExceeded {
                    scans: frame_scans,
                    maximum: MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATION_SCANS_V1,
                });
            }
            pitch_frames.insert((note.pitch_midi, frame));
            program_pitch_frames.insert((note.gm_program, note.pitch_midi, frame));
            if pitch_frames.len() > MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATIONS_V1
                || program_pitch_frames.len() > MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATIONS_V1
            {
                return Err(MidiTranscriptionEvaluationIssueV1::FrameActivationBudgetExceeded {
                    scans: frame_scans,
                    maximum: MIDI_TRANSCRIPTION_MAX_FRAME_ACTIVATIONS_V1,
                });
            }
        }
    }
    Ok((pitch_frames, program_pitch_frames))
}

fn metric_from_set_counts(
    reference_count: usize,
    predicted_count: usize,
    true_positive_count: usize,
) -> MidiTranscriptionMetricV1 {
    metric_from_matching(reference_count, predicted_count, true_positive_count)
}

fn value_error_summary(values: &[f64]) -> MidiTranscriptionValueErrorSummaryV1 {
    if values.is_empty() {
        return MidiTranscriptionValueErrorSummaryV1 {
            sample_count: 0,
            mean_absolute_error: None,
            median_absolute_error: None,
            p95_absolute_error: None,
            p99_absolute_error: None,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let percentile = |fraction: f64| {
        let index = ((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        sorted[index]
    };
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    MidiTranscriptionValueErrorSummaryV1 {
        sample_count: u32::try_from(sorted.len()).unwrap_or(u32::MAX),
        mean_absolute_error: Some(mean),
        median_absolute_error: Some(median),
        p95_absolute_error: Some(percentile(0.95)),
        p99_absolute_error: Some(percentile(0.99)),
    }
}

fn timing_error_summary(values_seconds: &[f64]) -> MidiTranscriptionTimingErrorSummaryV1 {
    if values_seconds.is_empty() {
        return MidiTranscriptionTimingErrorSummaryV1 {
            sample_count: 0,
            mean_signed_error_ms: None,
            mean_absolute_error_ms: None,
            median_absolute_error_ms: None,
            p95_absolute_error_ms: None,
            p99_absolute_error_ms: None,
        };
    }
    let signed_values: Vec<f64> = values_seconds.iter().map(|value| value * 1_000.0).collect();
    let mean_signed = signed_values.iter().sum::<f64>() / signed_values.len() as f64;
    let mut values: Vec<f64> = signed_values.iter().map(|value| value.abs()).collect();
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
        mean_signed_error_ms: Some(mean_signed),
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
        prediction_tempo_point_count: report.prediction_tempo_point_count,
        prediction_overlapping_same_pitch_note_on_count:
            report.prediction_overlapping_same_pitch_note_on_count,
        onset_only_metrics: &report.onset_only_metrics,
        offset_only_metrics: &report.offset_only_metrics,
        onset_pitch_metrics: &report.onset_pitch_metrics,
        full_note_metrics: &report.full_note_metrics,
        average_overlap_ratio: report.average_overlap_ratio,
        exact_program_onset_metrics: &report.exact_program_onset_metrics,
        exact_program_full_note_metrics: &report.exact_program_full_note_metrics,
        velocity_aware_full_note_metrics: &report.velocity_aware_full_note_metrics,
        frame_pitch_metrics: &report.frame_pitch_metrics,
        frame_program_pitch_metrics: &report.frame_program_pitch_metrics,
        frame_step_ms: report.frame_step_ms,
        onset_error_ms: &report.onset_error_ms,
        velocity_error: &report.velocity_error,
        offset_error_ms: &report.offset_error_ms,
    };
    Ok(domain_separated_sha256(
        MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

/// Evaluate a model-produced MIDI file against the exact part-aware MIDI
/// reference. Prediction timestamps are interpreted in the reference score's
/// MIDI clock domain; use the separate clock-alignment stage before evaluation
/// when the source WAV has a known global render offset.
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
            velocity_midi: note.velocity_midi,
            onset_seconds: note.onset_seconds,
            offset_seconds: note.offset_seconds,
        })
        .collect();

    let onset_only_matching = {
        let adjacency =
            build_onset_only_adjacency(&reference_notes, &predicted_notes, &config)?;
        maximum_cardinality_matching(&adjacency, predicted_notes.len())?
    };
    let offset_only_matching = {
        let adjacency =
            build_offset_only_adjacency(&reference_notes, &predicted_notes, &config)?;
        maximum_cardinality_matching(&adjacency, predicted_notes.len())?
    };
    let onset_matching = {
        let adjacency =
            build_adjacency(&reference_notes, &predicted_notes, &config, false, false, false)?;
        maximum_cardinality_matching(&adjacency, predicted_notes.len())?
    };
    let full_matching = {
        let adjacency =
            build_adjacency(&reference_notes, &predicted_notes, &config, true, false, false)?;
        maximum_cardinality_matching(&adjacency, predicted_notes.len())?
    };
    let program_onset_matching = {
        let adjacency =
            build_adjacency(&reference_notes, &predicted_notes, &config, false, true, false)?;
        maximum_cardinality_matching(&adjacency, predicted_notes.len())?
    };
    let program_full_matching = {
        let adjacency =
            build_adjacency(&reference_notes, &predicted_notes, &config, true, true, false)?;
        maximum_cardinality_matching(&adjacency, predicted_notes.len())?
    };

    let velocity_matching = if config.velocity_metrics_enabled {
        let adjacency =
            build_adjacency(&reference_notes, &predicted_notes, &config, true, false, true)?;
        Some(maximum_cardinality_matching(&adjacency, predicted_notes.len())?)
    } else {
        None
    };

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

    let (reference_pitch_frames, reference_program_frames) =
        frame_activation_sets(&reference_notes, config.frame_step_ms)?;
    let (prediction_pitch_frames, prediction_program_frames) =
        frame_activation_sets(&predicted_notes, config.frame_step_ms)?;
    let frame_pitch_true_positives = reference_pitch_frames
        .intersection(&prediction_pitch_frames)
        .count();
    let frame_program_true_positives = reference_program_frames
        .intersection(&prediction_program_frames)
        .count();

    // Velocity MAE is measured on full-note matches, independently of the
    // stricter velocity-aware match F1, so the error does not disappear when
    // a transcriber gets the velocity wrong.
    let velocity_errors: Vec<_> = if config.velocity_metrics_enabled {
        full_matching
            .iter()
            .map(|(reference_index, prediction_index)| {
                (i16::from(predicted_notes[*prediction_index].velocity_midi)
                    - i16::from(reference_notes[*reference_index].velocity_midi))
                    .unsigned_abs() as f64
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut report = MidiTranscriptionEvaluationReportV1 {
        schema: MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1.into(),
        algorithm: MIDI_TRANSCRIPTION_EVALUATION_ALGORITHM_V1.into(),
        reference_score_sha256: reference.manifest.score_sha256.clone(),
        reference_midi_sha256: reference.manifest.midi_sha256.clone(),
        reference_manifest_sha256: reference.manifest_sha256.clone(),
        prediction_midi_sha256: sha256_hex(prediction_midi_bytes),
        config: config.clone(),
        reference_note_count: u32::try_from(reference_notes.len()).unwrap_or(u32::MAX),
        prediction_note_count: u32::try_from(predicted_notes.len()).unwrap_or(u32::MAX),
        prediction_unclosed_note_count: parsed.unclosed_note_count,
        prediction_orphan_note_off_count: parsed.orphan_note_off_count,
        prediction_tempo_point_count: parsed.tempo_point_count,
        prediction_overlapping_same_pitch_note_on_count:
            parsed.overlapping_same_pitch_note_on_count,
        onset_only_metrics: metric_from_matching(
            reference_notes.len(),
            predicted_notes.len(),
            onset_only_matching.len(),
        ),
        offset_only_metrics: metric_from_matching(
            reference_notes.len(),
            predicted_notes.len(),
            offset_only_matching.len(),
        ),
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
        average_overlap_ratio: average_overlap_ratio_v1(
            &reference_notes,
            &predicted_notes,
            &full_matching,
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
        velocity_aware_full_note_metrics: velocity_matching.map(|matching| {
            metric_from_matching(
                reference_notes.len(),
                predicted_notes.len(),
                matching.len(),
            )
        }),
        frame_pitch_metrics: metric_from_set_counts(
            reference_pitch_frames.len(),
            prediction_pitch_frames.len(),
            frame_pitch_true_positives,
        ),
        frame_program_pitch_metrics: metric_from_set_counts(
            reference_program_frames.len(),
            prediction_program_frames.len(),
            frame_program_true_positives,
        ),
        frame_step_ms: config.frame_step_ms,
        velocity_error: config
            .velocity_metrics_enabled
            .then(|| value_error_summary(&velocity_errors)),
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
        build_midi_layout(notes, program, tempo_events, 100, false)
    }

    fn build_midi_with_velocity(
        notes: &[(u8, u64, u64)],
        program: u8,
        tempo_events: &[(u64, u32)],
        velocity: u8,
    ) -> Vec<u8> {
        build_midi_layout(notes, program, tempo_events, velocity, false)
    }

    fn build_midi_with_program_in_conductor_track(
        notes: &[(u8, u64, u64)],
        program: u8,
    ) -> Vec<u8> {
        build_midi_layout(notes, program, &[(0, 500_000)], 100, true)
    }

    fn build_midi_layout(
        notes: &[(u8, u64, u64)],
        program: u8,
        tempo_events: &[(u64, u32)],
        velocity: u8,
        program_in_conductor_track: bool,
    ) -> Vec<u8> {
        let mut note_events: Vec<(u64, u8, Vec<u8>)> =
            if program_in_conductor_track {
                Vec::new()
            } else {
                vec![(0, 0, vec![0xc0, program])]
            };
        for (pitch, onset, offset) in notes {
            note_events.push((*onset, 2, vec![0x90, *pitch, velocity]));
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
        if program_in_conductor_track {
            tempo_track.extend_from_slice(&[0x00, 0xc0, program]);
        }
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
        reference_artifact_with_velocity(notes, program, 100)
    }

    fn reference_artifact_with_velocity(
        notes: &[(u8, u64, u64)],
        program: u8,
        velocity: u8,
    ) -> MidiPerformanceArtifactV1 {
        let midi_bytes = build_midi_with_velocity(notes, program, &[(0, 500_000)], velocity);
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
                    velocity_midi: velocity,
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
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.f1, 1.0);
        assert_eq!(report.full_note_metrics.true_positives, 0);
        assert_eq!(report.offset_error_ms.sample_count, 2);
        assert!(report.offset_error_ms.mean_absolute_error_ms.unwrap() > 150.0);
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
    fn velocity_error_and_velocity_aware_score_are_reported_separately() {
        let notes = vec![(60, 0, 480), (64, 960, 1_440)];
        let reference = reference_artifact_with_velocity(&notes, 40, 100);
        let prediction = build_midi_with_velocity(&notes, 40, &[(0, 500_000)], 40);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            MidiTranscriptionEvaluationConfigV1 {
                velocity_metrics_enabled: true,
                ..default_config()
            },
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.f1, 1.0);
        assert_eq!(report.full_note_metrics.f1, 1.0);
        assert_eq!(report.velocity_aware_full_note_metrics.as_ref().unwrap().f1, 0.0);
        assert_eq!(report.velocity_error.as_ref().unwrap().sample_count, 2);
        assert_eq!(
            report.velocity_error.as_ref().unwrap().mean_absolute_error,
            Some(60.0)
        );
    }

    #[test]
    fn frame_pitch_metrics_measure_time_occupancy_not_note_identity() {
        let notes = vec![(60, 0, 960), (60, 480, 1_440), (64, 1_920, 2_400)];
        let reference = reference_artifact(&notes, 40);
        let prediction = build_midi(&notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.frame_pitch_metrics.f1, 1.0);
        assert_eq!(report.frame_program_pitch_metrics.f1, 1.0);
        assert_eq!(report.frame_step_ms, 10);
        // Same-pitch overlapping note cells are unioned in frame metrics.
        assert!(report.frame_pitch_metrics.true_positives < 3 * 50);
    }

    #[test]
    fn reference_midi_byte_substitution_is_rejected_before_scoring() {
        let notes = vec![(60, 0, 480)];
        let mut reference = reference_artifact(&notes, 40);
        reference.midi_bytes.push(0);
        let prediction = build_midi(&notes, 40, &[(0, 500_000)]);
        let error = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            MidiTranscriptionEvaluationIssueV1::ReferenceMidiDigestMismatch
        );
    }

    #[test]
    fn re_sealed_reference_manifest_cannot_relabel_source_velocity() {
        let notes = vec![(60, 0, 480)];
        let mut reference = reference_artifact(&notes, 40);
        reference.manifest.notes[0].velocity_midi = 99;
        let bytes = serde_json::to_vec(&reference.manifest).unwrap();
        reference.manifest_sha256 = domain_separated_sha256(
            MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
            &bytes,
        );
        let prediction = build_midi(&notes, 40, &[(0, 500_000)]);
        let error = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            MidiTranscriptionEvaluationIssueV1::ReferenceMidiDoesNotMatchManifest
        );
    }

    #[test]
    fn maximum_cardinality_matching_does_not_double_count_duplicate_candidates() {
        let adjacency = vec![vec![0, 1], vec![0]];
        let matches = maximum_cardinality_matching(&adjacency, 2).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn matching_order_reproduces_mir_eval_hopcroft_karp_fixture() {
        // Port of mir_eval/tests/test_util.py::test_bipartite_match.
        // Predicted vertices p0..p9 connect to reference vertices with
        // progressively shorter prefixes. This forces multiple HK layers.
        let mut adjacency = vec![Vec::new(); 10];
        for reference_index in 0..10 {
            for prediction_index in 0..(10 - reference_index) {
                adjacency[reference_index].push(prediction_index);
            }
        }
        let matches = maximum_cardinality_matching(&adjacency, 10).unwrap();
        assert_eq!(
            matches,
            vec![
                (0, 9), (1, 8), (2, 7), (3, 6), (4, 5),
                (5, 4), (6, 3), (7, 2), (8, 1), (9, 0),
            ]
        );
    }

    #[test]
    fn conflicting_tempo_events_at_the_same_tick_are_rejected() {
        let midi = build_midi(&[(60, 0, 480)], 40, &[(0, 500_000), (0, 600_000)]);
        assert_eq!(
            parse_transcribed_midi_v1(&midi).unwrap_err(),
            MidiTranscriptionEvaluationIssueV1::InvalidTempo
        );
    }

    #[test]
    fn explicit_nondefault_tempo_at_tick_zero_overrides_midi_default() {
        let prediction = build_midi(
            &[(60, 960, 1_440)],
            40,
            &[(0, 1_000_000)],
        );
        let parsed = parse_transcribed_midi_v1(&prediction).unwrap();
        assert!((parsed.notes[0].onset_seconds - 1.0).abs() < 1e-9);
        assert!((parsed.notes[0].offset_seconds.unwrap() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn exactly_fifty_millisecond_onset_error_is_included_by_default() {
        let reference_notes = vec![(60, 0, 480)];
        let predicted_notes = vec![(60, 96, 576)]; // 50 ms at 500,000 us/qn and 960 PPQ.
        let reference = reference_artifact(&reference_notes, 40);
        let prediction = build_midi(&predicted_notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.onset_pitch_metrics.true_positives, 1);
        assert_eq!(report.full_note_metrics.true_positives, 1);
    }

    #[test]
    fn velocity_metrics_are_absent_by_default_without_an_explicit_claim() {
        let notes = vec![(60, 0, 480)];
        let reference = reference_artifact(&notes, 40);
        let prediction = build_midi_with_velocity(&notes, 40, &[(0, 500_000)], 1);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert!(report.velocity_aware_full_note_metrics.is_none());
        assert!(report.velocity_error.is_none());
    }

    #[test]
    fn onset_only_metric_can_match_wrong_pitch_while_pitch_aware_metric_rejects_it() {
        let reference_notes = vec![(60, 0, 480)];
        let prediction_notes = vec![(67, 0, 480)];
        let reference = reference_artifact(&reference_notes, 40);
        let prediction = build_midi(&prediction_notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.onset_only_metrics.f1, 1.0);
        assert_eq!(report.onset_pitch_metrics.f1, 0.0);
        assert_eq!(report.full_note_metrics.f1, 0.0);
    }

    #[test]
    fn offset_only_metric_can_match_wrong_pitch_and_wrong_onset() {
        let reference_notes = vec![(60, 0, 960)];
        let prediction_notes = vec![(67, 480, 960)];
        let reference = reference_artifact(&reference_notes, 40);
        let prediction = build_midi(&prediction_notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.offset_only_metrics.f1, 1.0);
        assert_eq!(report.onset_only_metrics.f1, 0.0);
        assert_eq!(report.onset_pitch_metrics.f1, 0.0);
        assert_eq!(report.full_note_metrics.f1, 0.0);
    }

    #[test]
    fn average_overlap_ratio_is_reported_for_full_note_matches() {
        let reference_notes = vec![(60, 0, 1_920)];
        let prediction_notes = vec![(60, 96, 1_824)];
        let reference = reference_artifact(&reference_notes, 40);
        let prediction = build_midi(&prediction_notes, 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.full_note_metrics.f1, 1.0);
        assert!(report.average_overlap_ratio < 1.0);
        assert!(report.average_overlap_ratio > 0.89);
    }

    #[test]
    fn program_change_in_conductor_track_applies_to_notes_in_part_track() {
        let notes = vec![(60, 0, 480), (64, 960, 1_440)];
        let reference = reference_artifact(&notes, 40);
        let prediction =
            build_midi_with_program_in_conductor_track(&notes, 40);
        let parsed = parse_transcribed_midi_v1(&prediction).unwrap();
        assert!(parsed.notes.iter().all(|note| note.gm_program == 40));
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.exact_program_onset_metrics.f1, 1.0);
        assert_eq!(report.exact_program_full_note_metrics.f1, 1.0);
    }

    #[test]
    fn type_one_note_on_and_note_off_can_be_paired_across_tracks() {
        let conductor = vec![
            0x00, 0xc0, 40,
            0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
            0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let note_on = vec![0x00, 0x90, 60, 100, 0x00, 0xff, 0x2f, 0x00];
        let note_off = vec![0x83, 0x60, 0x80, 60, 0x00, 0x00, 0xff, 0x2f, 0x00];
        let tracks = vec![conductor, note_on, note_off];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&3u16.to_be_bytes());
        bytes.extend_from_slice(&MIDI_TRANSCRIPTION_TEST_PPQ.to_be_bytes());
        for track in tracks {
            bytes.extend_from_slice(b"MTrk");
            bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&track);
        }
        let parsed = parse_transcribed_midi_v1(&bytes).unwrap();
        assert_eq!(parsed.notes.len(), 1);
        assert_eq!(parsed.unclosed_note_count, 0);
        assert_eq!(parsed.orphan_note_off_count, 0);
        assert_eq!(parsed.notes[0].track_index, 1);
        assert_eq!(parsed.notes[0].gm_program, 40);
        assert_eq!(parsed.notes[0].offset_tick, Some(480));
    }

    #[test]
    fn conflicting_same_tick_program_changes_on_shared_channel_are_rejected() {
        let conductor_a = vec![
            0x00, 0xc0, 40,
            0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
            0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let conductor_b = vec![0x00, 0xc0, 41, 0x00, 0xff, 0x2f, 0x00];
        let note_track = vec![0x00, 0x90, 60, 100, 0x83, 0x60, 0x80, 60, 0x00, 0x00, 0xff, 0x2f, 0x00];
        let tracks = vec![conductor_a, conductor_b, note_track];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&3u16.to_be_bytes());
        bytes.extend_from_slice(&MIDI_TRANSCRIPTION_TEST_PPQ.to_be_bytes());
        for track in tracks {
            bytes.extend_from_slice(b"MTrk");
            bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&track);
        }
        assert_eq!(
            parse_transcribed_midi_v1(&bytes).unwrap_err(),
            MidiTranscriptionEvaluationIssueV1::ConflictingProgramChange {
                tick: 0,
                channel: 0,
            }
        );
    }

    #[test]
    fn empty_reference_and_empty_prediction_are_scored_as_negative_examples() {
        let reference = reference_artifact(&[], 40);
        let prediction = build_midi(&[], 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.reference_note_count, 0);
        assert_eq!(report.prediction_note_count, 0);
        assert_eq!(report.onset_pitch_metrics.f1, 0.0);
        assert_eq!(report.full_note_metrics.f1, 0.0);
        assert_eq!(report.average_overlap_ratio, 0.0);
    }

    #[test]
    fn empty_reference_counts_transcribed_notes_as_false_positives() {
        let reference = reference_artifact(&[], 40);
        let prediction = build_midi(&[(60, 0, 480)], 40, &[(0, 500_000)]);
        let report = evaluate_midi_transcription_v1(
            &prediction,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.reference_note_count, 0);
        assert_eq!(report.prediction_note_count, 1);
        assert_eq!(report.onset_pitch_metrics.true_positives, 0);
        assert_eq!(report.onset_pitch_metrics.false_positives, 1);
        assert_eq!(report.onset_pitch_metrics.false_negatives, 0);
    }

    #[test]
    fn velocity_tolerance_above_midi_range_is_rejected() {
        let error = evaluate_midi_transcription_v1(
            &build_midi(&[], 40, &[(0, 500_000)]),
            &reference_artifact(&[(60, 0, 480)], 40),
            MidiTranscriptionEvaluationConfigV1 {
                velocity_tolerance: 128,
                velocity_metrics_enabled: true,
                ..default_config()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            MidiTranscriptionEvaluationIssueV1::InvalidConfig {
                field: "velocity_tolerance".into()
            }
        );
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
