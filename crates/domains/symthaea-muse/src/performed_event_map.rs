// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed score-to-performed-event lineage for symbolic audio editing.
//!
//! This module is the receipt/verifier boundary, not a MIDI parser or a claim
//! of trusted execution. The renderer adapter must derive these rows from the
//! exact performed MIDI bytes it is about to render, and bind the map to the
//! exact target Score and MIDI digests. A self-consistent caller-authored map
//! is not proof that its rows occur in the MIDI; independent parsing and the
//! renderer execution receipt remain required.
//!
//! Source IDs use the same global onset-ordered indices as Score::events()
//! and the full-score symbolic edit verifier: score_event_index:{index}.
//! A source event must have exactly one primary descendant (ordinal zero);
//! additional explicitly recorded descendants may represent performance
//! layers. Unmapped events, unknown source IDs, duplicated MIDI note-on
//! identities, timing/frame disagreement, and incomplete descendant ordinals
//! fail closed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const PERFORMED_EVENT_MAP_VERSION: &str = "mel-mr-eval-014-performed-event-map-v0.1";
pub const MAX_PERFORMED_EVENT_MAP_EVENTS: usize = 100_000;
const MAX_SAMPLE_RATE_HZ: u32 = 768_000;
const MIDI_LINEAGE_MARKER_MAGIC: &[u8; 3] = b"LDM";
const MIDI_LINEAGE_MARKER_VERSION: u8 = 1;

/// One performed MIDI note and its declared ancestry from a target-score event.
///
/// note_on_ordinal is the zero-based ordinal among note-on events in one MIDI
/// track, assigned by a parser of the exact MIDI artifact. It is not a
/// timestamp and does not rely on matching approximate onset times.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformedEventLineage {
    /// Canonical score_event_index:{index} in the target Score's event order.
    pub source_event_id: String,
    /// Zero for the primary note; subsequent values name additional descendants.
    pub descendant_ordinal: u32,
    pub midi_track_index: u16,
    pub note_on_ordinal: u32,
    pub midi_channel: u8,
    pub midi_note: u8,
    /// Onset and end in rounded microseconds, with a half-open interval.
    pub onset_micros: u64,
    pub end_micros: u64,
    /// Projected half-open PCM schedule interval, rounded from MIDI time.
    /// This is not a measured acoustic onset and excludes renderer-specific latency.
    pub onset_frame: u64,
    pub end_frame: u64,
}

/// Versioned identity and timing map between the exact target Score and MIDI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformedEventMap {
    pub map_version: String,
    pub target_score_sha256: String,
    /// Hash of the exact MIDI bytes delivered to the renderer.
    pub performed_midi_sha256: String,
    pub sample_rate_hz: u32,
    pub total_frames: u64,
    pub events: Vec<PerformedEventLineage>,
    /// Missing or ambiguous source mappings are recorded here, never inferred.
    #[serde(default)]
    pub unresolved_source_event_ids: Vec<String>,
    /// SHA-256 over the canonical map with this field blank.
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformedEventMapError {
    UnsupportedVersion,
    InvalidMidiBytes,
    UnsupportedMidiTiming,
    InvalidTempoMap,
    MidiTickOverflow,
    InvalidLineageMarker,
    DuplicateLineageMarker,
    OrphanLineageMarker,
    LineageMarkerTickMismatch,
    AmbiguousNotePairing,
    OrphanNoteOff,
    UnclosedMidiNote,
    InvalidSha256(&'static str),
    InvalidField(&'static str),
    TooManyEvents,
    InvalidSourceEventId,
    DuplicateSourceEventId,
    UnknownSourceEvent,
    MissingSourceEvent,
    UnresolvedSourceEvents,
    DuplicateMidiEventIdentity,
    DuplicateDescendantOrdinal,
    NonContiguousDescendants,
    InvalidMidiChannel,
    InvalidMidiNote,
    InvalidTimeInterval,
    InvalidFrameInterval,
    FrameProjectionMismatch,
    InvalidExpectedEventCount,
    ContextMismatch(&'static str),
    InvalidReceipt,
    SerializationFailure,
}

impl std::fmt::Display for PerformedEventMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "performed-event map validation failed: {self:?}")
    }
}

impl std::error::Error for PerformedEventMapError {}

/// Stable score-event identity shared with symbolic_audio_binding.
pub fn score_event_id(index: usize) -> String {
    format!("score_event_index:{index}")
}

/// Encode a sequencer-specific MIDI marker. The marker is written immediately
/// before its NoteOn in the same track and at the same tick, so parsing does
/// not need nearest-onset matching or a heuristic note correspondence.
pub(crate) fn encode_midi_lineage_marker_payload(
    source_event_index: usize,
    descendant_ordinal: u32,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(MIDI_LINEAGE_MARKER_MAGIC);
    payload.push(MIDI_LINEAGE_MARKER_VERSION);
    payload.extend_from_slice(&(source_event_index as u64).to_be_bytes());
    payload.extend_from_slice(&descendant_ordinal.to_be_bytes());
    payload
}

fn decode_midi_lineage_marker_payload(
    payload: &[u8],
) -> Result<Option<(usize, u32)>, PerformedEventMapError> {
    if !payload.starts_with(MIDI_LINEAGE_MARKER_MAGIC) {
        return Ok(None);
    }
    if payload.len() != 16 || payload[3] != MIDI_LINEAGE_MARKER_VERSION {
        return Err(PerformedEventMapError::InvalidLineageMarker);
    }
    let raw_index = u64::from_be_bytes(
        payload[4..12]
            .try_into()
            .map_err(|_| PerformedEventMapError::InvalidLineageMarker)?,
    );
    let source_index =
        usize::try_from(raw_index).map_err(|_| PerformedEventMapError::InvalidSourceEventId)?;
    let descendant_ordinal = u32::from_be_bytes(
        payload[12..16]
            .try_into()
            .map_err(|_| PerformedEventMapError::InvalidLineageMarker)?,
    );
    Ok(Some((source_index, descendant_ordinal)))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn parse_score_event_id(value: &str) -> Option<usize> {
    let suffix = value.strip_prefix("score_event_index:")?;
    if suffix.is_empty() || (suffix.len() > 1 && suffix.starts_with('0')) {
        return None;
    }
    if !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Convert microseconds to a PCM frame using nearest-frame, half-up rounding.
pub fn frame_for_time_micros(
    time_micros: u64,
    sample_rate_hz: u32,
) -> Result<u64, PerformedEventMapError> {
    if sample_rate_hz == 0 || sample_rate_hz > MAX_SAMPLE_RATE_HZ {
        return Err(PerformedEventMapError::InvalidField("sample_rate_hz"));
    }
    let scaled = time_micros
        .checked_mul(u64::from(sample_rate_hz))
        .and_then(|value| value.checked_add(500_000))
        .ok_or(PerformedEventMapError::InvalidFrameInterval)?;
    Ok(scaled / 1_000_000)
}

#[derive(Debug, Clone, Copy)]
struct PendingMidiNote {
    midi_track_index: u16,
    note_on_ordinal: u32,
    midi_channel: u8,
    midi_note: u8,
    onset_tick: u64,
    source: Option<(usize, u32)>,
}

fn midi_tick_to_micros(
    tick: u64,
    ticks_per_beat: u64,
    tempo_changes: &[(u64, u32)],
) -> Result<u64, PerformedEventMapError> {
    let mut previous_tick = 0u64;
    let mut tempo = 500_000u64;
    let mut scaled_micros = 0u128;
    for &(change_tick, usec_per_beat) in tempo_changes {
        if change_tick > tick {
            break;
        }
        if change_tick > previous_tick {
            let segment = u128::from(change_tick - previous_tick)
                .checked_mul(u128::from(tempo))
                .ok_or(PerformedEventMapError::MidiTickOverflow)?;
            scaled_micros = scaled_micros
                .checked_add(segment)
                .ok_or(PerformedEventMapError::MidiTickOverflow)?;
            previous_tick = change_tick;
        }
        tempo = u64::from(usec_per_beat);
    }
    if tick > previous_tick {
        let segment = u128::from(tick - previous_tick)
            .checked_mul(u128::from(tempo))
            .ok_or(PerformedEventMapError::MidiTickOverflow)?;
        scaled_micros = scaled_micros
            .checked_add(segment)
            .ok_or(PerformedEventMapError::MidiTickOverflow)?;
    }
    let rounded = scaled_micros
        .checked_add(u128::from(ticks_per_beat / 2))
        .ok_or(PerformedEventMapError::MidiTickOverflow)?
        / u128::from(ticks_per_beat);
    u64::try_from(rounded).map_err(|_| PerformedEventMapError::MidiTickOverflow)
}

fn close_midi_note(
    open_notes: &mut HashMap<(u8, u8), PendingMidiNote>,
    channel: u8,
    midi_note: u8,
    end_tick: u64,
    ticks_per_beat: u64,
    tempo_changes: &[(u64, u32)],
    sample_rate_hz: u32,
    output: &mut Vec<PerformedEventLineage>,
) -> Result<(), PerformedEventMapError> {
    let note = open_notes
        .remove(&(channel, midi_note))
        .ok_or(PerformedEventMapError::OrphanNoteOff)?;
    if end_tick <= note.onset_tick {
        return Err(PerformedEventMapError::InvalidTimeInterval);
    }
    if let Some((source_index, descendant_ordinal)) = note.source {
        let onset_micros =
            midi_tick_to_micros(note.onset_tick, ticks_per_beat, tempo_changes)?;
        let end_micros = midi_tick_to_micros(end_tick, ticks_per_beat, tempo_changes)?;
        output.push(PerformedEventLineage {
            source_event_id: score_event_id(source_index),
            descendant_ordinal,
            midi_track_index: note.midi_track_index,
            note_on_ordinal: note.note_on_ordinal,
            midi_channel: note.midi_channel,
            midi_note: note.midi_note,
            onset_frame: frame_for_time_micros(onset_micros, sample_rate_hz)?,
            end_frame: frame_for_time_micros(end_micros, sample_rate_hz)?,
            onset_micros,
            end_micros,
        });
    }
    Ok(())
}

impl PerformedEventMap {
    /// Build lineage from the exact serialized MIDI bytes. The caller supplies
    /// the target Score digest and the output WAV's sample rate/frame count.
    /// Only notes that contain a valid in-band lineage marker become mapped
    /// rows; missing source events are caught by verify_for_score.
    ///
    /// Markers are accepted only when immediately followed by a positive
    /// NoteOn at the same absolute tick. Note pairing ambiguity, malformed
    /// markers and incomplete MIDI note lifetimes fail closed.
    pub fn from_midi_bytes(
        midi_bytes: &[u8],
        target_score_sha256: &str,
        sample_rate_hz: u32,
        total_frames: u64,
    ) -> Result<Self, PerformedEventMapError> {
        if !is_lower_hex_sha256(target_score_sha256) {
            return Err(PerformedEventMapError::InvalidSha256("target_score_sha256"));
        }
        if sample_rate_hz == 0 || sample_rate_hz > MAX_SAMPLE_RATE_HZ || total_frames == 0 {
            return Err(PerformedEventMapError::InvalidField("audio_context"));
        }
        let smf = midly::Smf::parse(midi_bytes)
            .map_err(|_| PerformedEventMapError::InvalidMidiBytes)?;
        let ticks_per_beat = match &smf.header.timing {
            midly::Timing::Metrical(ticks) => u64::from(ticks.as_int()),
            midly::Timing::Timecode(_, _) => {
                return Err(PerformedEventMapError::UnsupportedMidiTiming);
            }
        };
        if ticks_per_beat == 0 {
            return Err(PerformedEventMapError::UnsupportedMidiTiming);
        }

        // MIDI tempo applies across tracks. Gather absolute-tick changes and
        // reject conflicting tempo values at one tick rather than guess order.
        let mut raw_tempos: Vec<(u64, u32, usize, usize)> = Vec::new();
        for (track_index, track) in smf.tracks.iter().enumerate() {
            let mut tick = 0u64;
            for (event_index, event) in track.iter().enumerate() {
                tick = tick
                    .checked_add(u64::from(event.delta.as_int()))
                    .ok_or(PerformedEventMapError::MidiTickOverflow)?;
                if let midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(value)) = &event.kind {
                    let tempo = value.as_int();
                    if tempo == 0 {
                        return Err(PerformedEventMapError::InvalidTempoMap);
                    }
                    raw_tempos.push((tick, tempo, track_index, event_index));
                }
            }
        }
        raw_tempos.sort_by_key(|(tick, _, track, event)| (*tick, *track, *event));
        let mut tempo_changes: Vec<(u64, u32)> = Vec::new();
        for (tick, tempo, _, _) in raw_tempos {
            if let Some((last_tick, last_tempo)) = tempo_changes.last() {
                if *last_tick == tick {
                    if *last_tempo != tempo {
                        return Err(PerformedEventMapError::InvalidTempoMap);
                    }
                    continue;
                }
            }
            tempo_changes.push((tick, tempo));
        }

        let mut events = Vec::new();
        for (raw_track_index, track) in smf.tracks.iter().enumerate() {
            let mut tick = 0u64;
            let mut next_note_on_ordinal = 0u32;
            let mut pending_marker: Option<(usize, u32, u64)> = None;
            let mut open_notes: HashMap<(u8, u8), PendingMidiNote> = HashMap::new();
            let track_index = u16::try_from(raw_track_index)
                .map_err(|_| PerformedEventMapError::InvalidField("midi_track_index"))?;

            for event in track {
                tick = tick
                    .checked_add(u64::from(event.delta.as_int()))
                    .ok_or(PerformedEventMapError::MidiTickOverflow)?;
                let is_marker = matches!(
                    &event.kind,
                    midly::TrackEventKind::Meta(midly::MetaMessage::SequencerSpecific(payload))
                        if payload.starts_with(MIDI_LINEAGE_MARKER_MAGIC)
                );
                let is_positive_note_on = matches!(
                    &event.kind,
                    midly::TrackEventKind::Midi {
                        message: midly::MidiMessage::NoteOn { vel, .. },
                        ..
                    } if vel.as_int() > 0
                );
                if let Some((_, _, marker_tick)) = pending_marker {
                    if is_marker {
                        return Err(PerformedEventMapError::DuplicateLineageMarker);
                    }
                    if !is_positive_note_on {
                        return Err(PerformedEventMapError::OrphanLineageMarker);
                    }
                    if tick != marker_tick {
                        return Err(PerformedEventMapError::LineageMarkerTickMismatch);
                    }
                }

                match &event.kind {
                    midly::TrackEventKind::Meta(
                        midly::MetaMessage::SequencerSpecific(payload),
                    ) if payload.starts_with(MIDI_LINEAGE_MARKER_MAGIC) => {
                        if pending_marker.is_some() {
                            return Err(PerformedEventMapError::DuplicateLineageMarker);
                        }
                        let (source_index, descendant_ordinal) =
                            decode_midi_lineage_marker_payload(payload)?
                                .ok_or(PerformedEventMapError::InvalidLineageMarker)?;
                        pending_marker = Some((source_index, descendant_ordinal, tick));
                    }
                    midly::TrackEventKind::Midi { channel, message } => {
                        let channel = channel.as_int();
                        match message {
                            midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                                let source = pending_marker
                                    .take()
                                    .map(|(index, ordinal, _)| (index, ordinal));
                                let note_on_ordinal = next_note_on_ordinal;
                                next_note_on_ordinal = next_note_on_ordinal
                                    .checked_add(1)
                                    .ok_or(PerformedEventMapError::MidiTickOverflow)?;
                                let midi_note = key.as_int();
                                if open_notes.contains_key(&(channel, midi_note)) {
                                    return Err(PerformedEventMapError::AmbiguousNotePairing);
                                }
                                open_notes.insert(
                                    (channel, midi_note),
                                    PendingMidiNote {
                                        midi_track_index: track_index,
                                        note_on_ordinal,
                                        midi_channel: channel,
                                        midi_note,
                                        onset_tick: tick,
                                        source,
                                    },
                                );
                            }
                            midly::MidiMessage::NoteOff { key, .. } => {
                                close_midi_note(
                                    &mut open_notes,
                                    channel,
                                    key.as_int(),
                                    tick,
                                    ticks_per_beat,
                                    &tempo_changes,
                                    sample_rate_hz,
                                    &mut events,
                                )?;
                            }
                            midly::MidiMessage::NoteOn { key, .. } => {
                                // MIDI permits NoteOn with velocity zero as
                                // a NoteOff encoding.
                                close_midi_note(
                                    &mut open_notes,
                                    channel,
                                    key.as_int(),
                                    tick,
                                    ticks_per_beat,
                                    &tempo_changes,
                                    sample_rate_hz,
                                    &mut events,
                                )?;
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            if pending_marker.is_some() {
                return Err(PerformedEventMapError::OrphanLineageMarker);
            }
            if !open_notes.is_empty() {
                return Err(PerformedEventMapError::UnclosedMidiNote);
            }
        }

        let mut map = Self {
            map_version: PERFORMED_EVENT_MAP_VERSION.into(),
            target_score_sha256: target_score_sha256.into(),
            performed_midi_sha256: sha256_hex(midi_bytes),
            sample_rate_hz,
            total_frames,
            events,
            unresolved_source_event_ids: Vec::new(),
            receipt_sha256: String::new(),
        };
        map.seal()?;
        Ok(map)
    }

    /// Canonicalize and seal a map. This checks structure and timing but does
    /// not establish that the caller's event rows actually occur in the MIDI.
    pub fn seal(&mut self) -> Result<(), PerformedEventMapError> {
        self.events.sort_by(|left, right| {
            left.source_event_id
                .cmp(&right.source_event_id)
                .then_with(|| left.descendant_ordinal.cmp(&right.descendant_ordinal))
                .then_with(|| left.midi_track_index.cmp(&right.midi_track_index))
                .then_with(|| left.note_on_ordinal.cmp(&right.note_on_ordinal))
        });
        self.unresolved_source_event_ids.sort();
        self.receipt_sha256.clear();
        self.validate_structure()?;
        self.receipt_sha256 = self.compute_sha256()?;
        Ok(())
    }

    /// Verify receipt integrity, artifact context, and complete source coverage.
    ///
    /// expected_target_event_count is the number of events in the bound target
    /// Score's Score::events() stream. Each event must have one primary
    /// performed event; extra descendants are allowed when explicitly
    /// represented. This method validates metadata, not the MIDI bytes; the
    /// adapter/parser must independently derive the metadata from those bytes.
    pub fn verify_for_score(
        &self,
        expected_target_score_sha256: &str,
        expected_performed_midi_sha256: &str,
        expected_sample_rate_hz: u32,
        expected_total_frames: u64,
        expected_target_event_count: usize,
    ) -> Result<(), PerformedEventMapError> {
        self.validate_structure()?;
        if self.compute_sha256()? != self.receipt_sha256 {
            return Err(PerformedEventMapError::InvalidReceipt);
        }
        if self.target_score_sha256 != expected_target_score_sha256 {
            return Err(PerformedEventMapError::ContextMismatch("target score digest"));
        }
        if self.performed_midi_sha256 != expected_performed_midi_sha256 {
            return Err(PerformedEventMapError::ContextMismatch("performed MIDI digest"));
        }
        if self.sample_rate_hz != expected_sample_rate_hz {
            return Err(PerformedEventMapError::ContextMismatch("sample rate"));
        }
        if self.total_frames != expected_total_frames {
            return Err(PerformedEventMapError::ContextMismatch("total frame count"));
        }
        if expected_target_event_count == 0
            || expected_target_event_count > MAX_PERFORMED_EVENT_MAP_EVENTS
        {
            return Err(PerformedEventMapError::InvalidExpectedEventCount);
        }
        if !self.unresolved_source_event_ids.is_empty() {
            return Err(PerformedEventMapError::UnresolvedSourceEvents);
        }

        let mut seen_source_ids = HashSet::with_capacity(expected_target_event_count);
        let mut descendants: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
        for event in &self.events {
            let Some(index) = parse_score_event_id(&event.source_event_id) else {
                return Err(PerformedEventMapError::InvalidSourceEventId);
            };
            if index >= expected_target_event_count {
                return Err(PerformedEventMapError::UnknownSourceEvent);
            }
            seen_source_ids.insert(event.source_event_id.as_str());
            descendants
                .entry(event.source_event_id.as_str())
                .or_default()
                .insert(event.descendant_ordinal);
        }

        for index in 0..expected_target_event_count {
            let id = score_event_id(index);
            if !seen_source_ids.contains(id.as_str()) {
                return Err(PerformedEventMapError::MissingSourceEvent);
            }
            let ordinals = descendants
                .get(id.as_str())
                .ok_or(PerformedEventMapError::MissingSourceEvent)?;
            if !ordinals.contains(&0) {
                return Err(PerformedEventMapError::MissingSourceEvent);
            }
            for (expected, actual) in ordinals.iter().enumerate() {
                if *actual != expected as u32 {
                    return Err(PerformedEventMapError::NonContiguousDescendants);
                }
            }
        }
        Ok(())
    }

    /// Verify the complete receipt/context gate and exact primary MIDI pitches.
    ///
    /// This is the public convenience entry point for acceptance paths; it
    /// prevents callers from treating pitch agreement alone as a verified map.
    pub fn verify_for_score_and_pitches(
        &self,
        expected_target_score_sha256: &str,
        expected_performed_midi_sha256: &str,
        expected_sample_rate_hz: u32,
        expected_total_frames: u64,
        expected_target_event_count: usize,
        expected_midi_by_score_index: &[u8],
    ) -> Result<(), PerformedEventMapError> {
        self.verify_for_score(
            expected_target_score_sha256,
            expected_performed_midi_sha256,
            expected_sample_rate_hz,
            expected_total_frames,
            expected_target_event_count,
        )?;
        self.verify_primary_pitches(expected_midi_by_score_index)
    }

    /// Check the primary MIDI note for each target-score event. This is
    /// private so external callers cannot bypass receipt/context validation.
    fn verify_primary_pitches(
        &self,
        expected_midi_by_score_index: &[u8],
    ) -> Result<(), PerformedEventMapError> {
        let mut primary = BTreeMap::new();
        for event in &self.events {
            if event.descendant_ordinal == 0 {
                if primary
                    .insert(event.source_event_id.as_str(), event.midi_note)
                    .is_some()
                {
                    return Err(PerformedEventMapError::DuplicateDescendantOrdinal);
                }
            }
        }
        if primary.len() != expected_midi_by_score_index.len() {
            return Err(PerformedEventMapError::MissingSourceEvent);
        }
        for (index, expected_pitch) in expected_midi_by_score_index.iter().enumerate() {
            if *expected_pitch > 127 {
                return Err(PerformedEventMapError::InvalidMidiNote);
            }
            if primary.get(score_event_id(index).as_str()) != Some(expected_pitch) {
                return Err(PerformedEventMapError::ContextMismatch("primary performed MIDI pitch"));
            }
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), PerformedEventMapError> {
        if self.map_version != PERFORMED_EVENT_MAP_VERSION {
            return Err(PerformedEventMapError::UnsupportedVersion);
        }
        if !is_lower_hex_sha256(&self.target_score_sha256) {
            return Err(PerformedEventMapError::InvalidSha256("target_score_sha256"));
        }
        if !is_lower_hex_sha256(&self.performed_midi_sha256) {
            return Err(PerformedEventMapError::InvalidSha256("performed_midi_sha256"));
        }
        if self.sample_rate_hz == 0 || self.sample_rate_hz > MAX_SAMPLE_RATE_HZ {
            return Err(PerformedEventMapError::InvalidField("sample_rate_hz"));
        }
        if self.total_frames == 0 {
            return Err(PerformedEventMapError::InvalidField("total_frames"));
        }
        if self.events.len() > MAX_PERFORMED_EVENT_MAP_EVENTS {
            return Err(PerformedEventMapError::TooManyEvents);
        }
        let mut midi_event_identities = HashSet::with_capacity(self.events.len());
        let mut ordinal_sets: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
        let mut mapped_source_ids = BTreeSet::new();
        for event in &self.events {
            if parse_score_event_id(&event.source_event_id).is_none() {
                return Err(PerformedEventMapError::InvalidSourceEventId);
            }
            if event.midi_channel > 15 {
                return Err(PerformedEventMapError::InvalidMidiChannel);
            }
            if event.midi_note > 127 {
                return Err(PerformedEventMapError::InvalidMidiNote);
            }
            if event.onset_micros >= event.end_micros {
                return Err(PerformedEventMapError::InvalidTimeInterval);
            }
            if event.onset_frame >= event.end_frame || event.end_frame > self.total_frames {
                return Err(PerformedEventMapError::InvalidFrameInterval);
            }
            if frame_for_time_micros(event.onset_micros, self.sample_rate_hz)? != event.onset_frame
                || frame_for_time_micros(event.end_micros, self.sample_rate_hz)? != event.end_frame
            {
                return Err(PerformedEventMapError::FrameProjectionMismatch);
            }
            if !midi_event_identities.insert((event.midi_track_index, event.note_on_ordinal)) {
                return Err(PerformedEventMapError::DuplicateMidiEventIdentity);
            }
            mapped_source_ids.insert(event.source_event_id.as_str());
            if !ordinal_sets
                .entry(event.source_event_id.as_str())
                .or_default()
                .insert(event.descendant_ordinal)
            {
                return Err(PerformedEventMapError::DuplicateDescendantOrdinal);
            }
        }
        let mut unresolved = BTreeSet::new();
        for source_id in &self.unresolved_source_event_ids {
            if parse_score_event_id(source_id).is_none() {
                return Err(PerformedEventMapError::InvalidSourceEventId);
            }
            if !unresolved.insert(source_id.as_str()) {
                return Err(PerformedEventMapError::DuplicateSourceEventId);
            }
            if mapped_source_ids.contains(source_id.as_str()) {
                return Err(PerformedEventMapError::DuplicateSourceEventId);
            }
        }
        Ok(())
    }

    fn compute_sha256(&self) -> Result<String, PerformedEventMapError> {
        let mut canonical = self.clone();
        canonical.receipt_sha256.clear();
        canonical.events.sort_by(|left, right| {
            left.source_event_id
                .cmp(&right.source_event_id)
                .then_with(|| left.descendant_ordinal.cmp(&right.descendant_ordinal))
                .then_with(|| left.midi_track_index.cmp(&right.midi_track_index))
                .then_with(|| left.note_on_ordinal.cmp(&right.note_on_ordinal))
        });
        canonical.unresolved_source_event_ids.sort();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|_| PerformedEventMapError::SerializationFailure)?;
        let mut hasher = Sha256::new();
        hasher.update(b"luminous-dynamics-performed-event-map-v1\0");
        hasher.update(bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    fn event(
        source_index: usize,
        descendant_ordinal: u32,
        track: u16,
        note_on_ordinal: u32,
        midi_note: u8,
        onset_micros: u64,
        end_micros: u64,
    ) -> PerformedEventLineage {
        PerformedEventLineage {
            source_event_id: score_event_id(source_index),
            descendant_ordinal,
            midi_track_index: track,
            note_on_ordinal,
            midi_channel: 0,
            midi_note,
            onset_micros,
            end_micros,
            onset_frame: frame_for_time_micros(onset_micros, 48_000).unwrap(),
            end_frame: frame_for_time_micros(end_micros, 48_000).unwrap(),
        }
    }

    fn fixture() -> PerformedEventMap {
        let mut map = PerformedEventMap {
            map_version: PERFORMED_EVENT_MAP_VERSION.into(),
            target_score_sha256: digest('a'),
            performed_midi_sha256: digest('b'),
            sample_rate_hz: 48_000,
            total_frames: 48_000,
            events: vec![
                event(1, 0, 0, 1, 64, 500_000, 900_000),
                event(0, 0, 0, 0, 60, 100_000, 500_000),
                event(1, 1, 1, 0, 52, 500_000, 900_000),
            ],
            unresolved_source_event_ids: vec![],
            receipt_sha256: String::new(),
        };
        map.seal().unwrap();
        map
    }

    fn minimal_marked_midi() -> Vec<u8> {
        let mut track = Vec::new();
        // Track name.
        track.extend_from_slice(&[0, 0xFF, 0x03, 4, b'T', b'e', b's', b't']);
        // 120 BPM tempo map.
        track.extend_from_slice(&[0, 0xFF, 0x51, 3, 0x07, 0xA1, 0x20]);
        let payload = encode_midi_lineage_marker_payload(0, 0);
        track.extend_from_slice(&[0, 0xFF, 0x7F, payload.len() as u8]);
        track.extend_from_slice(&payload);
        // C4 at tick 0, with a half-second gate (480 ticks at 120 BPM).
        track.extend_from_slice(&[0, 0x90, 60, 100]);
        track.extend_from_slice(&[0x83, 0x60, 0x80, 60, 0]);
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);

        let mut midi = Vec::new();
        midi.extend_from_slice(b"MThd");
        midi.extend_from_slice(&6u32.to_be_bytes());
        midi.extend_from_slice(&0u16.to_be_bytes());
        midi.extend_from_slice(&1u16.to_be_bytes());
        midi.extend_from_slice(&480u16.to_be_bytes());
        midi.extend_from_slice(b"MTrk");
        midi.extend_from_slice(&(track.len() as u32).to_be_bytes());
        midi.extend_from_slice(&track);
        midi
    }

    #[test]
    fn midi_parser_recovers_exact_source_note_identity_and_performed_time() {
        let midi = minimal_marked_midi();
        let midi_sha256 = sha256_hex(&midi);
        let map = PerformedEventMap::from_midi_bytes(
            &midi,
            &digest('a'),
            48_000,
            48_000,
        )
        .unwrap();
        assert_eq!(map.performed_midi_sha256, midi_sha256);
        assert_eq!(map.events.len(), 1);
        let event = &map.events[0];
        assert_eq!(event.source_event_id, score_event_id(0));
        assert_eq!(event.descendant_ordinal, 0);
        assert_eq!(event.midi_track_index, 0);
        assert_eq!(event.note_on_ordinal, 0);
        assert_eq!(event.midi_channel, 0);
        assert_eq!(event.midi_note, 60);
        assert_eq!(event.onset_micros, 0);
        assert_eq!(event.end_micros, 500_000);
        assert_eq!(event.onset_frame, 0);
        assert_eq!(event.end_frame, 24_000);
        map.verify_for_score_and_pitches(
            &digest('a'),
            &midi_sha256,
            48_000,
            48_000,
            1,
            &[60],
        )
        .unwrap();
    }

    #[test]
    fn midi_parser_rejects_an_orphaned_lineage_marker() {
        let mut midi = minimal_marked_midi();
        // Replace the four-byte NoteOn with a four-byte empty text meta event,
        // keeping the SMF track length unchanged while leaving the marker orphaned.
        let needle = [0x00, 0x90, 60, 100];
        let position = midi
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        let replacement = [0x00, 0xFF, 0x01, 0x00];
        midi[position..position + needle.len()].copy_from_slice(&replacement);
        assert_eq!(
            PerformedEventMap::from_midi_bytes(&midi, &digest('a'), 48_000, 48_000),
            Err(PerformedEventMapError::OrphanLineageMarker)
        );
    }

    #[test]
    fn sealed_map_binds_exact_score_midi_and_primary_pitch_context() {
        let map = fixture();
        map.verify_for_score_and_pitches(
            &digest('a'), &digest('b'), 48_000, 48_000, 2, &[60, 64],
        )
        .unwrap();
    }

    #[test]
    fn changed_score_or_midi_digest_is_rejected() {
        let map = fixture();
        assert_eq!(
            map.verify_for_score(&digest('c'), &digest('b'), 48_000, 48_000, 2),
            Err(PerformedEventMapError::ContextMismatch("target score digest"))
        );
        assert_eq!(
            map.verify_for_score(&digest('a'), &digest('c'), 48_000, 48_000, 2),
            Err(PerformedEventMapError::ContextMismatch("performed MIDI digest"))
        );
    }

    #[test]
    fn missing_source_event_fails_closed() {
        let mut map = fixture();
        map.events.retain(|event| event.source_event_id != score_event_id(1));
        map.seal().unwrap();
        assert_eq!(
            map.verify_for_score(&digest('a'), &digest('b'), 48_000, 48_000, 2),
            Err(PerformedEventMapError::MissingSourceEvent)
        );
    }

    #[test]
    fn unresolved_lineage_is_rejected_even_when_receipt_is_sealed() {
        let mut map = fixture();
        map.events.retain(|event| event.source_event_id != score_event_id(1));
        map.unresolved_source_event_ids = vec![score_event_id(1)];
        map.seal().unwrap();
        assert_eq!(
            map.verify_for_score(&digest('a'), &digest('b'), 48_000, 48_000, 2),
            Err(PerformedEventMapError::UnresolvedSourceEvents)
        );
    }

    #[test]
    fn duplicate_midi_event_identity_is_rejected() {
        let mut map = fixture();
        map.events[1].midi_track_index = map.events[0].midi_track_index;
        map.events[1].note_on_ordinal = map.events[0].note_on_ordinal;
        assert_eq!(
            map.seal(),
            Err(PerformedEventMapError::DuplicateMidiEventIdentity)
        );
    }

    #[test]
    fn wrong_frame_projection_is_rejected() {
        let mut map = fixture();
        map.events[0].onset_frame += 1;
        assert_eq!(
            map.seal(),
            Err(PerformedEventMapError::FrameProjectionMismatch)
        );
    }

    #[test]
    fn receipt_tampering_and_primary_pitch_substitution_are_rejected() {
        let mut map = fixture();
        map.events[0].midi_note = 65;
        assert_eq!(
            map.verify_for_score(&digest('a'), &digest('b'), 48_000, 48_000, 2),
            Err(PerformedEventMapError::InvalidReceipt)
        );
        map.seal().unwrap();
        assert_eq!(
            map.verify_for_score_and_pitches(
                &digest('a'), &digest('b'), 48_000, 48_000, 2, &[60, 64],
            ),
            Err(PerformedEventMapError::ContextMismatch("primary performed MIDI pitch"))
        );
    }

    #[test]
    fn descendant_ordinals_must_be_contiguous_from_zero() {
        let mut map = fixture();
        map.events
            .iter_mut()
            .find(|event| event.source_event_id == score_event_id(1) && event.descendant_ordinal == 1)
            .unwrap()
            .descendant_ordinal = 2;
        map.seal().unwrap();
        assert_eq!(
            map.verify_for_score(&digest('a'), &digest('b'), 48_000, 48_000, 2),
            Err(PerformedEventMapError::NonContiguousDescendants)
        );
    }
}
