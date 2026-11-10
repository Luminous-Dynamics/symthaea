// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Part-aware, provenance-carrying Standard MIDI realization.
//!
//! Separate from legacy role/frequency-based MIDI exporters. This module
//! consumes generator-authored PartIds and an exact score-digest-bound
//! orchestration timeline. It never reconstructs parts from VoiceRole,
//! pitch register, velocity, MIDI channel, or note similarity.
//!
//! The output is portable MIDI 1.0 plus a sidecar mapping digest-scoped source
//! note ordinals to exact MIDI ticks, tracks, parts, segments, and explicit
//! General MIDI program bindings. A GM program is not an acoustic renderer
//! qualification and does not guarantee identical sound across synthesizers.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use symthaea_music_theory::orchestration::InstrumentCapabilityV1;
use symthaea_music_theory::orchestration_timeline::{
    OrchestrationTimelineReportV1, OrchestrationTimelineV1,
    ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1, RationalBeatV1,
    validate_orchestration_timeline,
};
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::score::{Score, ScoreNote};

pub const MIDI_PERFORMANCE_SCHEMA_V1: &str = "muse-midi-performance-v1";
pub const MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1: &str =
    "muse-orchestration-timeline-serde-json-sha256-v1";
pub const MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1: &str =
    "muse-midi-instrument-bindings-serde-json-sha256-v1";
pub const MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1: &str =
    "muse-midi-performance-manifest-serde-json-sha256-v1";
pub const MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1: u16 = 960;
pub const MIDI_PERFORMANCE_MAX_VLQ_V1: u64 = 0x0fff_ffff;
pub const MIDI_PERFORMANCE_MAX_TEXT_BYTES_V1: usize = 255;

/// Explicit mapping from one orchestration instrument instance to a portable
/// MIDI program and a named external renderer/patch target. The program is an
/// export instruction, not proof that the sound bank or patch is qualified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiInstrumentBindingV1 {
    pub instrument_id: String,
    pub renderer_id: String,
    pub patch_id: String,
    /// Zero-based General MIDI program number.
    pub gm_program: u8,
    /// Optional external asset digest; this exporter does not inspect it.
    pub asset_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiPerformanceTrackReceiptV1 {
    pub track_index: u16,
    pub track_name: String,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub renderer_id: String,
    pub patch_id: String,
    pub gm_program: u8,
    pub midi_channel: u8,
    pub segment_start_tick: u64,
    pub segment_end_tick: u64,
}

/// Ordinals are scoped to score_sha256; they are not persistent EventIds.
/// Later edits must carry correspondence through the explicit event-remap
/// contract, never by reconstructing note similarity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiPerformanceNoteReceiptV1 {
    pub source_note_ordinal: usize,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub track_index: u16,
    pub start_tick: u64,
    pub end_tick: u64,
    pub pitch_midi: u8,
    pub velocity_midi: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiPerformanceManifestV1 {
    pub schema: String,
    pub score_digest_schema: String,
    pub score_sha256: String,
    pub timeline_digest_schema: String,
    pub timeline_sha256: String,
    pub binding_digest_schema: String,
    pub binding_sha256: String,
    pub midi_sha256: String,
    pub ticks_per_quarter: u16,
    pub tracks: Vec<MidiPerformanceTrackReceiptV1>,
    pub notes: Vec<MidiPerformanceNoteReceiptV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidiPerformanceArtifactV1 {
    pub midi_bytes: Vec<u8>,
    pub manifest: MidiPerformanceManifestV1,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiPerformanceErrorV1 {
    EmptyScore,
    InvalidTempo,
    InvalidMeter,
    ScoreSerialization(String),
    ExpectedScoreDigestMismatch,
    TimelineValidationFailed(OrchestrationTimelineReportV1),
    DuplicateInstrumentBinding(String),
    InvalidInstrumentBinding(String),
    MissingInstrumentBinding(String),
    UnexpectedInstrumentBinding(String),
    MidiChannelCapacityExceeded { instrument_count: usize, capacity: usize },
    InvalidRationalTime,
    NonIntegralMidiTick { note_index: Option<usize> },
    MidiTickOutOfRange { note_index: Option<usize> },
    MissingNoteSegmentAssignment { note_index: usize },
    AmbiguousNoteSegmentAssignment { note_index: usize },
    ZeroVelocityNote { note_index: usize },
    OversizedTextField,
    MidiEncoding(String),
    ManifestSerialization(String),
}

#[derive(Debug, Clone, Copy)]
struct RationalMoment {
    numerator: i128,
    denominator: i128,
}

impl RationalMoment {
    fn from_duration(value: Duration) -> Option<Self> {
        let denominator = i128::from(value.den());
        (denominator > 0).then_some(Self {
            numerator: i128::from(value.num()),
            denominator,
        })
    }

    fn from_wire(value: RationalBeatV1) -> Option<Self> {
        let denominator = i128::from(value.denominator);
        (denominator > 0).then_some(Self {
            numerator: i128::from(value.numerator),
            denominator,
        })
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        Some(Self {
            numerator: left.checked_add(right)?,
            denominator: self.denominator.checked_mul(other.denominator)?,
        })
    }

    fn compare(self, other: Self) -> Option<Ordering> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        Some(left.cmp(&right))
    }

    fn note_is_contained(
        self,
        note_end: Self,
        segment_start: Self,
        segment_end: Self,
    ) -> Option<bool> {
        Some(
            self.compare(segment_start)? != Ordering::Less
                && note_end.compare(segment_end)? != Ordering::Greater,
        )
    }
}

struct SegmentInfo<'a> {
    segment: &'a symthaea_music_theory::orchestration_timeline::OrchestrationSegmentV1,
    start: RationalMoment,
    end: RationalMoment,
    start_tick: u64,
    end_tick: u64,
}

#[derive(Debug, Clone)]
struct MidiEventV1 {
    tick: u64,
    order: u8,
    source_note_ordinal: usize,
    bytes: Vec<u8>,
}

struct EncodedTrackV1 {
    name: String,
    events: Vec<MidiEventV1>,
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

/// Same score digest recipe as the score-demand profile: SHA-256 of the exact
/// serde_json bytes. The schema identifier is metadata, not a hash prefix.
fn score_sha256_v1(score: &Score) -> Result<String, MidiPerformanceErrorV1> {
    let bytes = serde_json::to_vec(score)
        .map_err(|error| MidiPerformanceErrorV1::ScoreSerialization(error.to_string()))?;
    Ok(sha256_hex(&bytes))
}

fn rational_to_midi_ticks(
    numerator: i128,
    denominator: i128,
    note_index: Option<usize>,
) -> Result<u64, MidiPerformanceErrorV1> {
    if denominator <= 0 || numerator < 0 {
        return Err(MidiPerformanceErrorV1::InvalidRationalTime);
    }
    let scaled = numerator
        .checked_mul(i128::from(MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1))
        .ok_or(MidiPerformanceErrorV1::MidiTickOutOfRange { note_index })?;
    if scaled % denominator != 0 {
        return Err(MidiPerformanceErrorV1::NonIntegralMidiTick { note_index });
    }
    let ticks = scaled / denominator;
    if ticks < 0 || ticks > i128::from(MIDI_PERFORMANCE_MAX_VLQ_V1) {
        return Err(MidiPerformanceErrorV1::MidiTickOutOfRange { note_index });
    }
    Ok(ticks as u64)
}

fn duration_to_midi_ticks(
    value: Duration,
    note_index: Option<usize>,
) -> Result<u64, MidiPerformanceErrorV1> {
    let rational = RationalMoment::from_duration(value)
        .ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
    rational_to_midi_ticks(rational.numerator, rational.denominator, note_index)
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_bindings(
    timeline: &OrchestrationTimelineV1,
    bindings: &[MidiInstrumentBindingV1],
) -> Result<BTreeMap<String, MidiInstrumentBindingV1>, MidiPerformanceErrorV1> {
    let used: BTreeSet<String> = timeline
        .segments
        .iter()
        .map(|segment| segment.instrument_id.clone())
        .collect();
    let mut map = BTreeMap::new();
    for binding in bindings {
        if binding.instrument_id.trim().is_empty()
            || binding.renderer_id.trim().is_empty()
            || binding.patch_id.trim().is_empty()
            || binding.gm_program > 127
            || binding.asset_sha256.as_ref().is_some_and(|digest| !valid_sha256_hex(digest))
        {
            return Err(MidiPerformanceErrorV1::InvalidInstrumentBinding(
                binding.instrument_id.clone(),
            ));
        }
        if map.insert(binding.instrument_id.clone(), binding.clone()).is_some() {
            return Err(MidiPerformanceErrorV1::DuplicateInstrumentBinding(
                binding.instrument_id.clone(),
            ));
        }
    }
    for instrument_id in &used {
        if !map.contains_key(instrument_id) {
            return Err(MidiPerformanceErrorV1::MissingInstrumentBinding(
                instrument_id.clone(),
            ));
        }
    }
    for instrument_id in map.keys() {
        if !used.contains(instrument_id) {
            return Err(MidiPerformanceErrorV1::UnexpectedInstrumentBinding(
                instrument_id.clone(),
            ));
        }
    }
    Ok(map)
}

fn midi_channels_v1() -> [u8; 15] {
    // Reserve zero-based channel 9, conventionally used for GM percussion.
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
}

fn assign_channels_v1(
    segments: &[SegmentInfo<'_>],
) -> Result<BTreeMap<String, u8>, MidiPerformanceErrorV1> {
    let instruments: BTreeSet<String> = segments
        .iter()
        .map(|info| info.segment.instrument_id.clone())
        .collect();
    let channels = midi_channels_v1();
    if instruments.len() > channels.len() {
        return Err(MidiPerformanceErrorV1::MidiChannelCapacityExceeded {
            instrument_count: instruments.len(),
            capacity: channels.len(),
        });
    }
    Ok(instruments.into_iter().zip(channels).collect())
}

fn validate_tempo_and_meter(score: &Score) -> Result<u32, MidiPerformanceErrorV1> {
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 {
        return Err(MidiPerformanceErrorV1::InvalidTempo);
    }
    if score.meter == 0 {
        return Err(MidiPerformanceErrorV1::InvalidMeter);
    }
    let micros = (60_000_000.0f64 / f64::from(score.tempo_bpm)).round();
    if !micros.is_finite() || !(1.0..=16_777_215.0).contains(&micros) {
        return Err(MidiPerformanceErrorV1::InvalidTempo);
    }
    Ok(micros as u32)
}

fn write_vlq(output: &mut Vec<u8>, mut value: u64) -> Result<(), MidiPerformanceErrorV1> {
    if value > MIDI_PERFORMANCE_MAX_VLQ_V1 {
        return Err(MidiPerformanceErrorV1::MidiEncoding(
            "MIDI variable-length quantity exceeds the four-byte SMF limit".into(),
        ));
    }
    let mut bytes = [0u8; 4];
    let mut index = 3usize;
    bytes[index] = (value & 0x7f) as u8;
    value >>= 7;
    while value > 0 {
        index -= 1;
        bytes[index] = ((value & 0x7f) as u8) | 0x80;
        value >>= 7;
    }
    output.extend_from_slice(&bytes[index..]);
    Ok(())
}

fn push_meta_event(
    track: &mut Vec<u8>,
    delta: u64,
    meta_type: u8,
    payload: &[u8],
) -> Result<(), MidiPerformanceErrorV1> {
    write_vlq(track, delta)?;
    track.extend_from_slice(&[0xff, meta_type]);
    write_vlq(track, payload.len() as u64)?;
    track.extend_from_slice(payload);
    Ok(())
}

fn encode_track_v1(
    track: &EncodedTrackV1,
    is_tempo_track: bool,
    tempo_micros: u32,
    meter: u8,
) -> Result<Vec<u8>, MidiPerformanceErrorV1> {
    let mut bytes = Vec::new();
    if track.name.len() > MIDI_PERFORMANCE_MAX_TEXT_BYTES_V1 {
        return Err(MidiPerformanceErrorV1::OversizedTextField);
    }
    push_meta_event(&mut bytes, 0, 0x03, track.name.as_bytes())?;
    if is_tempo_track {
        let tempo = [
            ((tempo_micros >> 16) & 0xff) as u8,
            ((tempo_micros >> 8) & 0xff) as u8,
            (tempo_micros & 0xff) as u8,
        ];
        push_meta_event(&mut bytes, 0, 0x51, &tempo)?;
        // Score meter is quarter-note beats per bar: denominator 4.
        push_meta_event(&mut bytes, 0, 0x58, &[meter, 2, 24, 8])?;
    } else {
        let mut previous_tick = 0u64;
        for event in &track.events {
            let delta = event.tick.checked_sub(previous_tick).ok_or_else(|| {
                MidiPerformanceErrorV1::MidiEncoding(
                    "MIDI events are not monotonically ordered".into(),
                )
            })?;
            write_vlq(&mut bytes, delta)?;
            bytes.extend_from_slice(&event.bytes);
            previous_tick = event.tick;
        }
    }
    write_vlq(&mut bytes, 0)?;
    bytes.extend_from_slice(&[0xff, 0x2f, 0x00]);
    Ok(bytes)
}

fn score_note_end(note: &ScoreNote) -> Option<RationalMoment> {
    RationalMoment::from_duration(note.onset)?
        .checked_add(RationalMoment::from_duration(note.duration)?)
}

fn build_segments<'a>(
    timeline: &'a OrchestrationTimelineV1,
) -> Result<Vec<SegmentInfo<'a>>, MidiPerformanceErrorV1> {
    let mut result = Vec::with_capacity(timeline.segments.len());
    for segment in &timeline.segments {
        let start = RationalMoment::from_wire(segment.start)
            .ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
        let end = RationalMoment::from_wire(segment.end)
            .ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
        let start_tick = rational_to_midi_ticks(start.numerator, start.denominator, None)?;
        let end_tick = rational_to_midi_ticks(end.numerator, end.denominator, None)?;
        if end_tick <= start_tick {
            return Err(MidiPerformanceErrorV1::InvalidRationalTime);
        }
        result.push(SegmentInfo {
            segment,
            start,
            end,
            start_tick,
            end_tick,
        });
    }
    result.sort_by(|left, right| {
        left.segment
            .part_id
            .cmp(&right.segment.part_id)
            .then_with(|| left.start_tick.cmp(&right.start_tick))
            .then_with(|| left.segment.instrument_id.cmp(&right.segment.instrument_id))
            .then_with(|| left.segment.id.cmp(&right.segment.id))
    });
    Ok(result)
}

fn note_assignment(
    score: &Score,
    segments: &[SegmentInfo<'_>],
) -> Result<Vec<usize>, MidiPerformanceErrorV1> {
    let mut result = Vec::with_capacity(score.notes.len());
    for (note_index, note) in score.notes.iter().enumerate() {
        let start = RationalMoment::from_duration(note.onset)
            .ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
        let end = score_note_end(note).ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
        let mut matching = Vec::new();
        for (segment_index, segment) in segments.iter().enumerate() {
            if segment.segment.part_id != note.part.0 {
                continue;
            }
            let contained = start
                .note_is_contained(end, segment.start, segment.end)
                .ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
            if contained {
                matching.push(segment_index);
            }
        }
        match matching.as_slice() {
            [one] => result.push(*one),
            [] => return Err(MidiPerformanceErrorV1::MissingNoteSegmentAssignment { note_index }),
            _ => return Err(MidiPerformanceErrorV1::AmbiguousNoteSegmentAssignment { note_index }),
        }
    }
    Ok(result)
}

/// Export one exact symbolic Score through one exact orchestration timeline.
///
/// The caller supplies the expected score digest from its provenance
/// boundary. The exporter recomputes the canonical score hash, validates the
/// timeline and supplied structural instrument capabilities, and refuses any
/// note whose beat time cannot be represented exactly at 960 PPQ.
///
/// The instrument capability input is renderer-independent. Explicit MIDI
/// bindings are routing instructions only: this operation does not qualify
/// acoustic timbre, sample coverage, articulation, or player performance.
pub fn export_score_midi_performance_v1(
    score: &Score,
    timeline: &OrchestrationTimelineV1,
    instruments: &[InstrumentCapabilityV1],
    bindings: &[MidiInstrumentBindingV1],
    expected_score_sha256: &str,
) -> Result<MidiPerformanceArtifactV1, MidiPerformanceErrorV1> {
    if score.notes.is_empty() {
        return Err(MidiPerformanceErrorV1::EmptyScore);
    }
    let tempo_micros = validate_tempo_and_meter(score)?;
    let score_sha256 = score_sha256_v1(score)?;
    if !valid_sha256_hex(expected_score_sha256) || expected_score_sha256 != score_sha256 {
        return Err(MidiPerformanceErrorV1::ExpectedScoreDigestMismatch);
    }

    let timeline_report =
        validate_orchestration_timeline(score, timeline, instruments, &score_sha256);
    if !timeline_report.valid {
        return Err(MidiPerformanceErrorV1::TimelineValidationFailed(timeline_report));
    }

    let binding_map = validate_bindings(timeline, bindings)?;
    let segments = build_segments(timeline)?;
    let channels = assign_channels_v1(&segments)?;
    let assignments = note_assignment(score, &segments)?;

    let timeline_json = serde_json::to_vec(timeline)
        .map_err(|error| MidiPerformanceErrorV1::ManifestSerialization(error.to_string()))?;
    let timeline_sha256 = domain_separated_sha256(
        MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1,
        &timeline_json,
    );

    let canonical_bindings: Vec<MidiInstrumentBindingV1> =
        binding_map.values().cloned().collect();
    let binding_json = serde_json::to_vec(&canonical_bindings)
        .map_err(|error| MidiPerformanceErrorV1::ManifestSerialization(error.to_string()))?;
    let binding_sha256 = domain_separated_sha256(
        MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1,
        &binding_json,
    );

    let mut tracks = Vec::with_capacity(segments.len() + 1);
    tracks.push(EncodedTrackV1 {
        name: "Muse Tempo and Meter".into(),
        events: Vec::new(),
    });
    let mut track_receipts = Vec::with_capacity(segments.len());
    let mut segment_events: Vec<Vec<MidiEventV1>> =
        (0..segments.len()).map(|_| Vec::new()).collect();

    for (sorted_index, info) in segments.iter().enumerate() {
        if info.segment.id.len() > MIDI_PERFORMANCE_MAX_TEXT_BYTES_V1
            || info.segment.instrument_id.len() > MIDI_PERFORMANCE_MAX_TEXT_BYTES_V1
        {
            return Err(MidiPerformanceErrorV1::OversizedTextField);
        }
        let track_index = u16::try_from(sorted_index + 1).map_err(|_| {
            MidiPerformanceErrorV1::MidiEncoding(
                "MIDI track index exceeds the manifest representation".into(),
            )
        })?;
        let binding = binding_map.get(&info.segment.instrument_id).ok_or_else(|| {
            MidiPerformanceErrorV1::MissingInstrumentBinding(
                info.segment.instrument_id.clone(),
            )
        })?;
        let channel = *channels.get(&info.segment.instrument_id).ok_or_else(|| {
            MidiPerformanceErrorV1::MissingInstrumentBinding(
                info.segment.instrument_id.clone(),
            )
        })?;

        let track_name = format!(
            "Part {} {} {}",
            info.segment.part_id,
            info.segment.id,
            info.segment.instrument_id
        );
        if track_name.len() > MIDI_PERFORMANCE_MAX_TEXT_BYTES_V1 {
            return Err(MidiPerformanceErrorV1::OversizedTextField);
        }

        // Different instrument instances receive different MIDI channels.
        // The Program Change occurs at the exact segment start tick.
        segment_events[sorted_index].push(MidiEventV1 {
            tick: info.start_tick,
            order: 1,
            source_note_ordinal: 0,
            bytes: vec![0xc0 | (channel & 0x0f), binding.gm_program],
        });
        track_receipts.push(MidiPerformanceTrackReceiptV1 {
            track_index,
            track_name,
            part_id: info.segment.part_id,
            segment_id: info.segment.id.clone(),
            instrument_id: info.segment.instrument_id.clone(),
            renderer_id: binding.renderer_id.clone(),
            patch_id: binding.patch_id.clone(),
            gm_program: binding.gm_program,
            midi_channel: channel,
            segment_start_tick: info.start_tick,
            segment_end_tick: info.end_tick,
        });
    }

    let mut note_receipts = Vec::with_capacity(score.notes.len());
    for (note_index, note) in score.notes.iter().enumerate() {
        if note.velocity <= 0.0 {
            return Err(MidiPerformanceErrorV1::ZeroVelocityNote { note_index });
        }
        let segment_index = assignments[note_index];
        let info = &segments[segment_index];
        let track_index = u16::try_from(segment_index + 1).map_err(|_| {
            MidiPerformanceErrorV1::MidiEncoding(
                "MIDI track index exceeds the manifest representation".into(),
            )
        })?;
        let channel = *channels.get(&info.segment.instrument_id).ok_or_else(|| {
            MidiPerformanceErrorV1::MissingInstrumentBinding(
                info.segment.instrument_id.clone(),
            )
        })?;
        let start = duration_to_midi_ticks(note.onset, Some(note_index))?;
        let end_rational =
            score_note_end(note).ok_or(MidiPerformanceErrorV1::InvalidRationalTime)?;
        let end = rational_to_midi_ticks(
            end_rational.numerator,
            end_rational.denominator,
            Some(note_index),
        )?;
        if end <= start {
            return Err(MidiPerformanceErrorV1::InvalidRationalTime);
        }
        let velocity = (f64::from(note.velocity) * 127.0)
            .round()
            .clamp(1.0, 127.0) as u8;
        let pitch = note.pitch.midi();

        // Note-off sorts before Program Change and Note-on at equal ticks.
        segment_events[segment_index].push(MidiEventV1 {
            tick: end,
            order: 0,
            source_note_ordinal: note_index,
            bytes: vec![0x80 | (channel & 0x0f), pitch, 0],
        });
        segment_events[segment_index].push(MidiEventV1 {
            tick: start,
            order: 2,
            source_note_ordinal: note_index,
            bytes: vec![0x90 | (channel & 0x0f), pitch, velocity],
        });
        note_receipts.push(MidiPerformanceNoteReceiptV1 {
            source_note_ordinal: note_index,
            part_id: note.part.0,
            segment_id: info.segment.id.clone(),
            instrument_id: info.segment.instrument_id.clone(),
            track_index,
            start_tick: start,
            end_tick: end,
            pitch_midi: pitch,
            velocity_midi: velocity,
        });
    }

    for (index, events) in segment_events.iter_mut().enumerate() {
        events.sort_by_key(|event| {
            (event.tick, event.order, event.source_note_ordinal)
        });
        tracks.push(EncodedTrackV1 {
            name: track_receipts[index].track_name.clone(),
            events: std::mem::take(events),
        });
    }

    let mut midi_bytes = Vec::new();
    midi_bytes.extend_from_slice(b"MThd");
    midi_bytes.extend_from_slice(&6u32.to_be_bytes());
    midi_bytes.extend_from_slice(&1u16.to_be_bytes()); // Type 1.
    let track_count = u16::try_from(tracks.len()).map_err(|_| {
        MidiPerformanceErrorV1::MidiEncoding("MIDI track count exceeds SMF header".into())
    })?;
    midi_bytes.extend_from_slice(&track_count.to_be_bytes());
    midi_bytes.extend_from_slice(&MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1.to_be_bytes());

    for (index, track) in tracks.iter().enumerate() {
        let data = encode_track_v1(track, index == 0, tempo_micros, score.meter)?;
        let length = u32::try_from(data.len()).map_err(|_| {
            MidiPerformanceErrorV1::MidiEncoding(
                "MIDI track data exceeds the SMF chunk-size limit".into(),
            )
        })?;
        midi_bytes.extend_from_slice(b"MTrk");
        midi_bytes.extend_from_slice(&length.to_be_bytes());
        midi_bytes.extend_from_slice(&data);
    }

    note_receipts.sort_by_key(|receipt| receipt.source_note_ordinal);
    track_receipts.sort_by_key(|receipt| receipt.track_index);
    let midi_sha256 = sha256_hex(&midi_bytes);
    let manifest = MidiPerformanceManifestV1 {
        schema: MIDI_PERFORMANCE_SCHEMA_V1.into(),
        score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
        score_sha256,
        timeline_digest_schema: MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1.into(),
        timeline_sha256,
        binding_digest_schema: MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1.into(),
        binding_sha256,
        midi_sha256,
        ticks_per_quarter: MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1,
        tracks: track_receipts,
        notes: note_receipts,
    };
    let manifest_json = serde_json::to_vec(&manifest)
        .map_err(|error| MidiPerformanceErrorV1::ManifestSerialization(error.to_string()))?;
    let manifest_sha256 = domain_separated_sha256(
        MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
        &manifest_json,
    );

    Ok(MidiPerformanceArtifactV1 {
        midi_bytes,
        manifest,
        manifest_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::orchestration::{
        InstrumentKindV1, SoundingRangeV1,
    };
    use symthaea_music_theory::orchestration_timeline::{
        HandoffKindV1, OrchestrationSegmentV1, SegmentHandoffV1,
        ORCHESTRATION_TIMELINE_SCHEMA_V1,
    };
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::score::{Emphasis, PartId, VoiceRole};

    fn note(
        part: u16,
        pitch: u8,
        onset_num: i64,
        onset_den: i64,
        duration_num: i64,
        duration_den: i64,
        velocity: f32,
    ) -> ScoreNote {
        ScoreNote {
            part: PartId(part),
            pitch: Pitch::from_midi(pitch),
            onset: Duration::new(onset_num, onset_den),
            duration: Duration::new(duration_num, duration_den),
            velocity,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn score(notes: Vec<ScoreNote>) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for note in notes {
            score.push(note);
        }
        score
    }

    fn capability(id: &str, kind: InstrumentKindV1) -> InstrumentCapabilityV1 {
        InstrumentCapabilityV1 {
            id: id.into(),
            kind,
            sounding_range: SoundingRangeV1 {
                lowest_midi: 36,
                highest_midi: 96,
            },
            max_polyphony: 16,
        }
    }

    fn binding(id: &str, program: u8) -> MidiInstrumentBindingV1 {
        MidiInstrumentBindingV1 {
            instrument_id: id.into(),
            renderer_id: "test-midi-renderer-v1".into(),
            patch_id: format!("{id}-patch"),
            gm_program: program,
            asset_sha256: None,
        }
    }

    fn segment(
        id: &str,
        part_id: u16,
        instrument_id: &str,
        start: (i64, i64),
        end: (i64, i64),
        handoff: Option<SegmentHandoffV1>,
    ) -> OrchestrationSegmentV1 {
        OrchestrationSegmentV1 {
            id: id.into(),
            part_id,
            instrument_id: instrument_id.into(),
            start: RationalBeatV1 {
                numerator: start.0,
                denominator: start.1,
            },
            end: RationalBeatV1 {
                numerator: end.0,
                denominator: end.1,
            },
            musical_function: "authored_part_function".into(),
            handoff,
        }
    }

    fn timeline(score: &Score, segments: Vec<OrchestrationSegmentV1>) -> OrchestrationTimelineV1 {
        OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: score_sha256_v1(score).expect("score digest"),
            segments,
        }
    }

    fn export(
        score: &Score,
        timeline: &OrchestrationTimelineV1,
        instruments: &[InstrumentCapabilityV1],
        bindings: &[MidiInstrumentBindingV1],
    ) -> Result<MidiPerformanceArtifactV1, MidiPerformanceErrorV1> {
        let digest = score_sha256_v1(score).expect("score digest");
        export_score_midi_performance_v1(score, timeline, instruments, bindings, &digest)
    }

    #[test]
    fn exports_generator_authored_parts_not_voice_role_buckets() {
        let score = score(vec![
            note(10, 60, 0, 1, 1, 1, 0.8),
            note(20, 64, 0, 1, 1, 1, 0.7),
        ]);
        let timeline = timeline(
            &score,
            vec![
                segment("lead-segment", 10, "violin-a", (0, 1), (2, 1), None),
                segment("counter-segment", 20, "cello-b", (0, 1), (2, 1), None),
            ],
        );
        let instruments = vec![
            capability("violin-a", InstrumentKindV1::Violin),
            capability("cello-b", InstrumentKindV1::Cello),
        ];
        let artifact = export(
            &score,
            &timeline,
            &instruments,
            &[binding("violin-a", 40), binding("cello-b", 42)],
        )
        .expect("valid part-aware export");

        assert_eq!(&artifact.midi_bytes[0..4], b"MThd");
        assert_eq!(&artifact.midi_bytes[8..10], &1u16.to_be_bytes());
        assert_eq!(artifact.manifest.tracks.len(), 2);
        assert_eq!(artifact.manifest.notes.len(), 2);
        assert_eq!(artifact.manifest.notes[0].source_note_ordinal, 0);
        assert_eq!(artifact.manifest.notes[0].part_id, 10);
        assert_eq!(artifact.manifest.notes[1].part_id, 20);
        assert_ne!(
            artifact.manifest.tracks[0].midi_channel,
            artifact.manifest.tracks[1].midi_channel
        );
    }

    #[test]
    fn explicit_handoff_keeps_one_part_identity_across_instruments() {
        let score = score(vec![
            note(7, 60, 0, 1, 1, 1, 0.8),
            note(7, 62, 1, 1, 1, 1, 0.8),
        ]);
        let first = segment("opening-cello", 7, "cello-1", (0, 1), (1, 1), None);
        let second = segment(
            "answer-horn",
            7,
            "horn-1",
            (1, 1),
            (2, 1),
            Some(SegmentHandoffV1 {
                from_segment_id: first.id.clone(),
                kind: HandoffKindV1::Reattack,
            }),
        );
        let timeline = timeline(&score, vec![first, second]);
        let instruments = vec![
            capability("cello-1", InstrumentKindV1::Cello),
            capability("horn-1", InstrumentKindV1::FrenchHorn),
        ];
        let artifact = export(
            &score,
            &timeline,
            &instruments,
            &[binding("cello-1", 42), binding("horn-1", 60)],
        )
        .expect("valid handoff export");

        assert_eq!(artifact.manifest.tracks.len(), 2);
        assert!(artifact.manifest.notes.iter().all(|note| note.part_id == 7));
        assert_eq!(artifact.manifest.notes[0].segment_id, "opening-cello");
        assert_eq!(artifact.manifest.notes[1].segment_id, "answer-horn");
        assert_ne!(
            artifact.manifest.notes[0].instrument_id,
            artifact.manifest.notes[1].instrument_id
        );
    }

    #[test]
    fn overlapping_parts_receive_distinct_midi_channels() {
        let score = score(vec![
            note(1, 60, 0, 1, 2, 1, 0.8),
            note(2, 64, 0, 1, 2, 1, 0.8),
        ]);
        let timeline = timeline(
            &score,
            vec![
                segment("p1", 1, "inst-a", (0, 1), (2, 1), None),
                segment("p2", 2, "inst-b", (0, 1), (2, 1), None),
            ],
        );
        let instruments = vec![
            capability("inst-a", InstrumentKindV1::Violin),
            capability("inst-b", InstrumentKindV1::Cello),
        ];
        let artifact = export(
            &score,
            &timeline,
            &instruments,
            &[binding("inst-a", 40), binding("inst-b", 42)],
        )
        .expect("valid simultaneous export");
        assert_ne!(
            artifact.manifest.tracks[0].midi_channel,
            artifact.manifest.tracks[1].midi_channel
        );
    }

    #[test]
    fn stale_score_digest_is_rejected() {
        let score = score(vec![note(1, 60, 0, 1, 1, 1, 0.8)]);
        let mut timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (1, 1), None)],
        );
        timeline.score_sha256 = "a".repeat(64);
        let error = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[binding("inst-a", 40)],
        )
        .expect_err("stale timeline must fail");
        assert!(matches!(
            error,
            MidiPerformanceErrorV1::TimelineValidationFailed(_)
        ));
    }

    #[test]
    fn missing_instrument_binding_is_rejected() {
        let score = score(vec![note(1, 60, 0, 1, 1, 1, 0.8)]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (1, 1), None)],
        );
        let error = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[],
        )
        .expect_err("missing binding must fail");
        assert_eq!(
            error,
            MidiPerformanceErrorV1::MissingInstrumentBinding("inst-a".into())
        );
    }

    #[test]
    fn unassigned_part_identity_is_not_reconstructed() {
        let score = score(vec![note(u16::MAX, 60, 0, 1, 1, 1, 0.8)]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 0, "inst-a", (0, 1), (1, 1), None)],
        );
        let error = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[binding("inst-a", 40)],
        )
        .expect_err("unassigned identity must fail closed");
        assert!(matches!(
            error,
            MidiPerformanceErrorV1::TimelineValidationFailed(_)
        ));
    }

    #[test]
    fn non_integral_midi_tick_is_rejected_instead_of_rounded() {
        let score = score(vec![note(1, 60, 1, 961, 1, 1, 0.8)]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (2, 1), None)],
        );
        let error = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[binding("inst-a", 40)],
        )
        .expect_err("unsupported tick fraction must fail");
        assert_eq!(
            error,
            MidiPerformanceErrorV1::NonIntegralMidiTick { note_index: Some(0) }
        );
    }

    #[test]
    fn zero_velocity_note_is_not_silently_changed_into_a_note_on() {
        let score = score(vec![note(1, 60, 0, 1, 1, 1, 0.0)]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (1, 1), None)],
        );
        let error = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[binding("inst-a", 40)],
        )
        .expect_err("zero velocity is not an audible NoteOn");
        assert_eq!(error, MidiPerformanceErrorV1::ZeroVelocityNote { note_index: 0 });
    }

    #[test]
    fn frozen_inputs_produce_identical_midi_and_manifest_digests() {
        let score = score(vec![
            note(1, 60, 0, 1, 1, 1, 0.8),
            note(1, 62, 1, 1, 1, 1, 0.7),
        ]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (2, 1), None)],
        );
        let instruments = vec![capability("inst-a", InstrumentKindV1::Violin)];
        let bindings = vec![binding("inst-a", 40)];
        let first = export(&score, &timeline, &instruments, &bindings).expect("first export");
        let second = export(&score, &timeline, &instruments, &bindings).expect("second export");
        assert_eq!(first.midi_bytes, second.midi_bytes);
        assert_eq!(first.manifest, second.manifest);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
    }

    #[test]
    fn every_source_note_has_exactly_one_ordinal_receipt() {
        let score = score(vec![
            note(1, 60, 0, 1, 1, 1, 0.8),
            note(1, 62, 1, 1, 1, 1, 0.7),
            note(1, 64, 2, 1, 1, 1, 0.6),
        ]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (3, 1), None)],
        );
        let artifact = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[binding("inst-a", 40)],
        )
        .expect("valid export");
        assert_eq!(
            artifact.manifest.notes.iter()
                .map(|note| note.source_note_ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn rejects_general_midi_program_outside_zero_to_127() {
        let score = score(vec![note(1, 60, 0, 1, 1, 1, 0.8)]);
        let timeline = timeline(
            &score,
            vec![segment("p1", 1, "inst-a", (0, 1), (1, 1), None)],
        );
        let error = export(
            &score,
            &timeline,
            &[capability("inst-a", InstrumentKindV1::Violin)],
            &[binding("inst-a", 200)],
        )
        .expect_err("invalid GM program must fail");
        assert_eq!(
            error,
            MidiPerformanceErrorV1::InvalidInstrumentBinding("inst-a".into())
        );
    }

    #[test]
    fn refuses_more_than_fifteen_simultaneous_pitched_instrument_instances() {
        let mut notes = Vec::new();
        let mut segments = Vec::new();
        let mut instruments = Vec::new();
        let mut bindings = Vec::new();
        for index in 0..16u16 {
            let id = format!("instrument-{index:02}");
            notes.push(note(index, 60, 0, 1, 1, 1, 0.8));
            segments.push(segment(
                &format!("part-{index:02}"),
                index,
                &id,
                (0, 1),
                (1, 1),
                None,
            ));
            instruments.push(capability(&id, InstrumentKindV1::Violin));
            bindings.push(binding(&id, index as u8));
        }
        let score = score(notes);
        let timeline = timeline(&score, segments);
        let error = export(&score, &timeline, &instruments, &bindings)
            .expect_err("MIDI 1.0 channel capacity must be explicit");
        assert_eq!(
            error,
            MidiPerformanceErrorV1::MidiChannelCapacityExceeded {
                instrument_count: 16,
                capacity: 15,
            }
        );
    }

}
