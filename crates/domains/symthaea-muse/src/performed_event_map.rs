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
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub const PERFORMED_EVENT_MAP_VERSION: &str = "mel-mr-eval-014-performed-event-map-v0.1";
pub const MAX_PERFORMED_EVENT_MAP_EVENTS: usize = 100_000;
const MAX_SAMPLE_RATE_HZ: u32 = 768_000;

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
    /// Half-open PCM frame interval derived with nearest-frame, half-up rounding.
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

impl PerformedEventMap {
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
