// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Time-indexed orchestration: persistent parts can change instrument instances.
//!
//! The static orchestration plan binds one PartId to one instance for a whole
//! score. This timeline contract instead binds a PartId to an instrument over
//! half-open score-time intervals. A cello can introduce a line and a horn can
//! take it over without changing the line's identity.
//!
//! V1 supports sequential, rearticulated handoffs and entries after a rest.
//! Sustained cross-instrument ties, simultaneous doubling, written-pitch
//! notation, instrument technique constraints and renderer qualification are
//! deliberately not inferred here. A note must fit wholly inside exactly one
//! assignment segment. The validator checks supplied capability profiles; it
//! does not certify that those profiles are true or that a renderer exists.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::orchestration::{
    InstrumentCapabilityV1, MAX_INSTRUMENT_POLYPHONY, MAX_ORCHESTRATION_INSTRUMENTS,
    MAX_ORCHESTRATION_NOTES, MAX_ORCHESTRATION_PARTS,
};
use crate::rhythm::Duration;
use crate::score::Score;

pub const ORCHESTRATION_TIMELINE_SCHEMA_V1: &str = "orchestration-timeline-v1";
pub const MAX_ORCHESTRATION_SEGMENTS_V1: usize = 4_096;
const MAX_BEAT_NUMERATOR: i128 = 1_000_000_000;
const MAX_BEAT_DENOMINATOR: i128 = 1_000_000;

/// Exact non-negative score time in beats.
///
/// The interval represented by a segment is [start, end). Denominators must
/// be positive; numerator and denominator are bounded before arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RationalBeatV1 {
    pub numerator: i64,
    pub denominator: i64,
}

/// How a later instrument takes over a persistent part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffKindV1 {
    /// The next instrument begins exactly when the previous assignment ends.
    /// Notes crossing that boundary are rejected; a new attack is required.
    Reattack,
    /// The next instrument enters after a genuine notated rest in the part.
    AfterRest,
}

/// Explicit link from a new assignment segment to the previous segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentHandoffV1 {
    pub from_segment_id: String,
    pub kind: HandoffKindV1,
}

/// One instrument assignment over a half-open score-time interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationSegmentV1 {
    pub id: String,
    pub part_id: u16,
    pub instrument_id: String,
    pub start: RationalBeatV1,
    pub end: RationalBeatV1,
    /// Authored function such as "opening_theme" or "transformed_response".
    /// This is not inferred from VoiceRole or pitch register.
    pub musical_function: String,
    /// Required when this part changes to a different instrument instance.
    pub handoff: Option<SegmentHandoffV1>,
}

/// A complete time-indexed realization plan for one exact symbolic score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationTimelineV1 {
    pub schema: String,
    /// SHA-256 digest of the exact symbolic score, supplied by the score
    /// demand/provenance layer. This module validates its shape, not its origin.
    pub score_sha256: String,
    pub segments: Vec<OrchestrationSegmentV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationTimelineIssueCodeV1 {
    InvalidSchema,
    InvalidScoreDigest,
    EmptyScore,
    MissingPartIdentity,
    ResourceLimitExceeded,
    InvalidSegmentId,
    DuplicateSegmentId,
    InvalidPartId,
    InvalidInstrumentId,
    MissingInstrument,
    InvalidInstrumentProfile,
    InvalidSegmentTime,
    EmptySegment,
    EmptyMusicalFunction,
    OverlappingPartAssignments,
    MissingHandoff,
    InvalidHandoffReference,
    InvalidHandoffTiming,
    InvalidScoreTime,
    MissingNoteAssignment,
    NoteCrossesHandoff,
    AmbiguousNoteAssignment,
    NoteOutOfRange,
    PolyphonyExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationTimelineIssueV1 {
    pub code: OrchestrationTimelineIssueCodeV1,
    pub part_id: Option<u16>,
    pub segment_id: Option<String>,
    pub note_index: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentPeakPolyphonyV1 {
    pub instrument_id: String,
    pub peak_simultaneous_notes: u8,
    pub declared_budget: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationTimelineReportV1 {
    pub valid: bool,
    pub part_count: usize,
    pub segment_count: usize,
    pub used_instrument_count: usize,
    pub peak_polyphony: Vec<InstrumentPeakPolyphonyV1>,
    pub issues: Vec<OrchestrationTimelineIssueV1>,
}

impl OrchestrationTimelineReportV1 {
    fn new(part_count: usize, segment_count: usize) -> Self {
        Self {
            valid: true,
            part_count,
            segment_count,
            used_instrument_count: 0,
            peak_polyphony: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn push(
        &mut self,
        code: OrchestrationTimelineIssueCodeV1,
        part_id: Option<u16>,
        segment_id: Option<&str>,
        note_index: Option<usize>,
        message: impl Into<String>,
    ) {
        self.valid = false;
        self.issues.push(OrchestrationTimelineIssueV1 {
            code,
            part_id,
            segment_id: segment_id.map(str::to_owned),
            note_index,
            message: message.into(),
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct RationalMoment {
    numerator: i128,
    denominator: i128,
}

impl RationalMoment {
    fn from_pair(numerator: i128, denominator: i128, require_positive: bool) -> Option<Self> {
        if denominator <= 0
            || denominator > MAX_BEAT_DENOMINATOR
            || numerator < 0
            || numerator > MAX_BEAT_NUMERATOR
            || (require_positive && numerator == 0)
        {
            return None;
        }
        Some(Self {
            numerator,
            denominator,
        })
    }

    fn from_wire(value: RationalBeatV1) -> Option<Self> {
        Self::from_pair(
            i128::from(value.numerator),
            i128::from(value.denominator),
            false,
        )
    }

    fn from_duration(value: Duration, require_positive: bool) -> Option<Self> {
        Self::from_pair(
            i128::from(value.num()),
            i128::from(value.den()),
            require_positive,
        )
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        let numerator = left.checked_add(right)?;
        let denominator = self.denominator.checked_mul(other.denominator)?;
        Some(Self {
            numerator,
            denominator,
        })
    }

    fn compare(self, other: Self) -> Ordering {
        // Inputs are bounded above, and score-note ends are formed from two
        // bounded rationals. Their cross-products remain within i128.
        (self.numerator * other.denominator)
            .cmp(&(other.numerator * self.denominator))
    }

    fn strictly_before(self, other: Self) -> bool {
        self.compare(other) == Ordering::Less
    }

    fn at_or_after(self, other: Self) -> bool {
        self.compare(other) != Ordering::Less
    }

    fn at_or_before(self, other: Self) -> bool {
        self.compare(other) != Ordering::Greater
    }
}

struct ValidSegment<'a> {
    segment: &'a OrchestrationSegmentV1,
    start: RationalMoment,
    end: RationalMoment,
}

struct PolyphonyEvent {
    when: RationalMoment,
    /// -1 is note-off and +1 is note-on. Off sorts first at equal times.
    delta: i8,
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate a time-indexed orchestration plan against one symbolic score and
/// caller-supplied performer/instrument capability profiles.
///
/// Every sounding note must be covered by exactly one segment for its PartId.
/// Segments for one PartId may be sequential or separated by rests, but may not
/// overlap. A change of instrument instance requires an explicit handoff.
/// Polyphony is counted across all parts routed to the same instrument instance.
pub fn validate_orchestration_timeline(
    score: &Score,
    plan: &OrchestrationTimelineV1,
    instruments: &[InstrumentCapabilityV1],
) -> OrchestrationTimelineReportV1 {
    let part_count = score.parts().len();
    let mut report = OrchestrationTimelineReportV1::new(part_count, plan.segments.len());

    if plan.schema != ORCHESTRATION_TIMELINE_SCHEMA_V1 {
        report.push(
            OrchestrationTimelineIssueCodeV1::InvalidSchema,
            None,
            None,
            None,
            "timeline schema identifier is not supported",
        );
    }
    if !valid_sha256_hex(&plan.score_sha256) {
        report.push(
            OrchestrationTimelineIssueCodeV1::InvalidScoreDigest,
            None,
            None,
            None,
            "score_sha256 must be 64 lowercase hexadecimal characters",
        );
    }
    if score.notes.is_empty() {
        report.push(
            OrchestrationTimelineIssueCodeV1::EmptyScore,
            None,
            None,
            None,
            "cannot allocate an empty score",
        );
    } else if !score.has_part_identity() {
        report.push(
            OrchestrationTimelineIssueCodeV1::MissingPartIdentity,
            None,
            None,
            None,
            "every score note needs a generator-authored PartId; VoiceRole is not a substitute",
        );
    }
    if score.parts().len() > MAX_ORCHESTRATION_PARTS
        || score.notes.len() > MAX_ORCHESTRATION_NOTES
        || plan.segments.len() > MAX_ORCHESTRATION_SEGMENTS_V1
        || instruments.len() > MAX_ORCHESTRATION_INSTRUMENTS
    {
        report.push(
            OrchestrationTimelineIssueCodeV1::ResourceLimitExceeded,
            None,
            None,
            None,
            "score, segment or instrument count exceeds the bounded validation limits",
        );
        return report;
    }

    let mut instrument_map = BTreeMap::<String, &InstrumentCapabilityV1>::new();
    for instrument in instruments {
        if instrument.id.trim().is_empty() {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidInstrumentId,
                None,
                None,
                None,
                "instrument instance ID must not be empty",
            );
            continue;
        }
        if instrument_map
            .insert(instrument.id.clone(), instrument)
            .is_some()
        {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidInstrumentId,
                None,
                None,
                None,
                format!("duplicate instrument instance ID {}", instrument.id),
            );
        }
        if instrument.sounding_range.lowest_midi > instrument.sounding_range.highest_midi
            || instrument.sounding_range.highest_midi > 127
            || instrument.max_polyphony == 0
            || instrument.max_polyphony > MAX_INSTRUMENT_POLYPHONY
        {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidInstrumentProfile,
                None,
                None,
                None,
                format!("instrument instance {} has an invalid range or polyphony budget", instrument.id),
            );
        }
    }

    let mut seen_segment_ids = BTreeSet::<String>::new();
    let mut valid_segments = Vec::<ValidSegment<'_>>::new();
    let mut segments_by_part = BTreeMap::<u16, Vec<usize>>::new();

    for segment in &plan.segments {
        if segment.id.trim().is_empty() {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidSegmentId,
                Some(segment.part_id),
                None,
                None,
                "segment ID must not be empty",
            );
        } else if !seen_segment_ids.insert(segment.id.clone()) {
            report.push(
                OrchestrationTimelineIssueCodeV1::DuplicateSegmentId,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                "segment IDs must be unique",
            );
        }

        if segment.part_id == u16::MAX {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidPartId,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                "u16::MAX is the reserved unassigned PartId",
            );
        }
        if segment.instrument_id.trim().is_empty() {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidInstrumentId,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                "instrument instance ID must not be empty",
            );
        } else if !instrument_map.contains_key(&segment.instrument_id) {
            report.push(
                OrchestrationTimelineIssueCodeV1::MissingInstrument,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                format!("instrument instance {} is not declared", segment.instrument_id),
            );
        }
        if segment.musical_function.trim().is_empty() {
            report.push(
                OrchestrationTimelineIssueCodeV1::EmptyMusicalFunction,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                "musical_function must be authored and non-empty",
            );
        }

        let start = RationalMoment::from_wire(segment.start);
        let end = RationalMoment::from_wire(segment.end);
        let (Some(start), Some(end)) = (start, end) else {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidSegmentTime,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                "segment boundaries must be bounded non-negative rational beats",
            );
            continue;
        };
        if !start.strictly_before(end) {
            report.push(
                OrchestrationTimelineIssueCodeV1::EmptySegment,
                Some(segment.part_id),
                Some(&segment.id),
                None,
                "segment start must be strictly earlier than segment end",
            );
            continue;
        }

        let index = valid_segments.len();
        valid_segments.push(ValidSegment {
            segment,
            start,
            end,
        });
        segments_by_part
            .entry(segment.part_id)
            .or_default()
            .push(index);
    }

    // Sort each continuing part by exact start time and validate transitions.
    for (part_id, indices) in &mut segments_by_part {
        indices.sort_by(|left, right| {
            valid_segments[*left]
                .start
                .compare(valid_segments[*right].start)
                .then_with(|| valid_segments[*left].segment.id.cmp(&valid_segments[*right].segment.id))
        });

        for pair in indices.windows(2) {
            let previous = &valid_segments[pair[0]];
            let current = &valid_segments[pair[1]];

            if previous.end.strictly_before(current.start) == false
                && previous.end.compare(current.start) != Ordering::Equal
            {
                report.push(
                    OrchestrationTimelineIssueCodeV1::OverlappingPartAssignments,
                    Some(*part_id),
                    Some(&current.segment.id),
                    None,
                    "a persistent part cannot have overlapping primary instrument assignments in v1",
                );
            }

            if previous.segment.instrument_id != current.segment.instrument_id
                && current.segment.handoff.is_none()
            {
                report.push(
                    OrchestrationTimelineIssueCodeV1::MissingHandoff,
                    Some(*part_id),
                    Some(&current.segment.id),
                    None,
                    "changing instrument instances requires an explicit handoff from the previous segment",
                );
            }

            if let Some(handoff) = &current.segment.handoff {
                if handoff.from_segment_id != previous.segment.id {
                    report.push(
                        OrchestrationTimelineIssueCodeV1::InvalidHandoffReference,
                        Some(*part_id),
                        Some(&current.segment.id),
                        None,
                        "handoff must reference the immediately preceding segment for this PartId",
                    );
                    continue;
                }
                let timing_valid = match handoff.kind {
                    HandoffKindV1::Reattack => {
                        previous.end.compare(current.start) == Ordering::Equal
                    }
                    HandoffKindV1::AfterRest => previous.end.strictly_before(current.start),
                };
                if !timing_valid {
                    report.push(
                        OrchestrationTimelineIssueCodeV1::InvalidHandoffTiming,
                        Some(*part_id),
                        Some(&current.segment.id),
                        None,
                        "handoff kind does not match the exact boundary/rest between segments",
                    );
                }
            }
        }

        if let Some(first_index) = indices.first() {
            if valid_segments[*first_index].segment.handoff.is_some() {
                report.push(
                    OrchestrationTimelineIssueCodeV1::InvalidHandoffReference,
                    Some(*part_id),
                    Some(&valid_segments[*first_index].segment.id),
                    None,
                    "the first segment for a PartId cannot hand off from a previous segment",
                );
            }
        }
    }

    let mut polyphony_by_instrument = BTreeMap::<String, Vec<PolyphonyEvent>>::new();
    let mut used_instruments = BTreeSet::<String>::new();

    for (note_index, note) in score.notes.iter().enumerate() {
        if !note.part.is_assigned() {
            continue;
        }
        let Some(start) = RationalMoment::from_duration(note.onset, false) else {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidScoreTime,
                Some(note.part.0),
                None,
                Some(note_index),
                "note onset is negative or outside the bounded rational-time domain",
            );
            continue;
        };
        let Some(duration) = RationalMoment::from_duration(note.duration, true) else {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidScoreTime,
                Some(note.part.0),
                None,
                Some(note_index),
                "note duration must be positive and inside the bounded rational-time domain",
            );
            continue;
        };
        let Some(end) = start.checked_add(duration) else {
            report.push(
                OrchestrationTimelineIssueCodeV1::InvalidScoreTime,
                Some(note.part.0),
                None,
                Some(note_index),
                "note end cannot be represented safely in the bounded rational-time domain",
            );
            continue;
        };

        let part_segments = segments_by_part.get(&note.part.0);
        let covering: Vec<&ValidSegment<'_>> = part_segments
            .into_iter()
            .flat_map(|indices| indices.iter())
            .map(|index| &valid_segments[*index])
            .filter(|segment| {
                start.at_or_after(segment.start) && end.at_or_before(segment.end)
            })
            .collect();

        if covering.len() != 1 {
            let crosses_boundary = part_segments
                .into_iter()
                .flat_map(|indices| indices.iter())
                .map(|index| &valid_segments[*index])
                .any(|segment| {
                    start.strictly_before(segment.end) && segment.start.strictly_before(end)
                });
            if covering.len() > 1 {
                report.push(
                    OrchestrationTimelineIssueCodeV1::AmbiguousNoteAssignment,
                    Some(note.part.0),
                    None,
                    Some(note_index),
                    "note is covered by more than one instrument assignment segment",
                );
            } else if crosses_boundary {
                report.push(
                    OrchestrationTimelineIssueCodeV1::NoteCrossesHandoff,
                    Some(note.part.0),
                    None,
                    Some(note_index),
                    "note crosses an instrument handoff; split or rearticulate it explicitly",
                );
            } else {
                report.push(
                    OrchestrationTimelineIssueCodeV1::MissingNoteAssignment,
                    Some(note.part.0),
                    None,
                    Some(note_index),
                    "no single instrument assignment segment covers this note",
                );
            }
            continue;
        }

        let segment = covering[0].segment;
        let Some(instrument) = instrument_map.get(&segment.instrument_id) else {
            continue;
        };
        let midi = note.pitch.midi();
        if midi < instrument.sounding_range.lowest_midi
            || midi > instrument.sounding_range.highest_midi
        {
            report.push(
                OrchestrationTimelineIssueCodeV1::NoteOutOfRange,
                Some(note.part.0),
                Some(&segment.id),
                Some(note_index),
                format!(
                    "sounding MIDI note {midi} is outside the declared inclusive range {}..={}",
                    instrument.sounding_range.lowest_midi,
                    instrument.sounding_range.highest_midi
                ),
            );
        }

        used_instruments.insert(instrument.id.clone());
        let events = polyphony_by_instrument
            .entry(instrument.id.clone())
            .or_default();
        events.push(PolyphonyEvent {
            when: start,
            delta: 1,
        });
        events.push(PolyphonyEvent {
            when: end,
            delta: -1,
        });
    }

    report.used_instrument_count = used_instruments.len();
    for (instrument_id, events) in &mut polyphony_by_instrument {
        events.sort_by(|left, right| {
            left.when
                .compare(right.when)
                .then_with(|| left.delta.cmp(&right.delta))
        });
        let Some(instrument) = instrument_map.get(instrument_id) else {
            continue;
        };
        let mut active = 0i32;
        let mut peak = 0i32;
        for event in events {
            active += i32::from(event.delta);
            peak = peak.max(active);
        }
        report.peak_polyphony.push(InstrumentPeakPolyphonyV1 {
            instrument_id: instrument_id.clone(),
            peak_simultaneous_notes: peak.min(u8::MAX as i32) as u8,
            declared_budget: instrument.max_polyphony,
        });
        if peak > i32::from(instrument.max_polyphony) {
            report.push(
                OrchestrationTimelineIssueCodeV1::PolyphonyExceeded,
                None,
                None,
                None,
                format!(
                    "instrument instance {instrument_id} reaches {peak} simultaneous notes, over its declared budget {}",
                    instrument.max_polyphony
                ),
            );
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::orchestration::{InstrumentKindV1, SoundingRangeV1};
    use crate::pitch::{Pitch, PitchClass};
    use crate::score::{Emphasis, PartId, ScoreNote, VoiceRole};

    fn note(part: u16, midi: u8, onset: i64, duration: i64, role: VoiceRole) -> ScoreNote {
        ScoreNote {
            part: PartId(part),
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, 1),
            duration: Duration::new(duration, 1),
            velocity: 0.7,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn profile(id: &str, kind: InstrumentKindV1, low: u8, high: u8, polyphony: u8) -> InstrumentCapabilityV1 {
        InstrumentCapabilityV1 {
            id: id.into(),
            kind,
            sounding_range: SoundingRangeV1 {
                lowest_midi: low,
                highest_midi: high,
            },
            max_polyphony: polyphony,
        }
    }

    fn base_plan() -> OrchestrationTimelineV1 {
        OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_sha256: "a".repeat(64),
            segments: vec![
                OrchestrationSegmentV1 {
                    id: "cello-opening".into(),
                    part_id: 1,
                    instrument_id: "cello-1".into(),
                    start: RationalBeatV1 { numerator: 0, denominator: 1 },
                    end: RationalBeatV1 { numerator: 2, denominator: 1 },
                    musical_function: "opening-theme".into(),
                    handoff: None,
                },
                OrchestrationSegmentV1 {
                    id: "horn-transfer".into(),
                    part_id: 1,
                    instrument_id: "horn-1".into(),
                    start: RationalBeatV1 { numerator: 2, denominator: 1 },
                    end: RationalBeatV1 { numerator: 4, denominator: 1 },
                    musical_function: "transformed-theme".into(),
                    handoff: Some(SegmentHandoffV1 {
                        from_segment_id: "cello-opening".into(),
                        kind: HandoffKindV1::Reattack,
                    }),
                },
            ],
        }
    }

    fn base_instruments() -> Vec<InstrumentCapabilityV1> {
        vec![
            profile("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
            profile("horn-1", InstrumentKindV1::FrenchHorn, 34, 77, 1),
            profile("piano-1", InstrumentKindV1::Piano, 21, 108, 10),
        ]
    }

    fn base_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 96.0, 4);
        score.push(note(1, 48, 0, 2, VoiceRole::Melody));
        score.push(note(1, 60, 2, 2, VoiceRole::CounterMelody));
        score
    }

    #[test]
    fn accepts_a_persistent_part_transferred_from_cello_to_horn() {
        let report = validate_orchestration_timeline(
            &base_score(),
            &base_plan(),
            &base_instruments(),
        );
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.part_count, 1);
        assert_eq!(report.segment_count, 2);
        assert_eq!(report.used_instrument_count, 2);
    }

    #[test]
    fn rejects_a_change_of_instrument_without_an_explicit_handoff() {
        let mut plan = base_plan();
        plan.segments[1].handoff = None;
        let report = validate_orchestration_timeline(
            &base_score(),
            &plan,
            &base_instruments(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::MissingHandoff
        }));
    }

    #[test]
    fn rejects_notes_that_cross_an_instrument_handoff() {
        let mut score = Score::new(Key::major(PitchClass::C), 96.0, 4);
        score.push(note(1, 48, 0, 3, VoiceRole::Melody));
        score.push(note(1, 60, 3, 1, VoiceRole::Melody));
        let report = validate_orchestration_timeline(
            &score,
            &base_plan(),
            &base_instruments(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::NoteCrossesHandoff
        }));
    }

    #[test]
    fn rejects_overlapping_primary_assignments_for_one_part() {
        let mut plan = base_plan();
        plan.segments[1].start = RationalBeatV1 { numerator: 1, denominator: 1 };
        plan.segments[1].handoff = Some(SegmentHandoffV1 {
            from_segment_id: "cello-opening".into(),
            kind: HandoffKindV1::Reattack,
        });
        let report = validate_orchestration_timeline(
            &base_score(),
            &plan,
            &base_instruments(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::OverlappingPartAssignments
        }));
    }

    #[test]
    fn enforces_polyphony_across_multiple_parts_routed_to_one_instrument() {
        let mut score = Score::new(Key::major(PitchClass::C), 96.0, 4);
        score.push(note(1, 60, 0, 2, VoiceRole::Melody));
        score.push(note(2, 64, 0, 2, VoiceRole::Harmony));
        let piano = profile("piano-1", InstrumentKindV1::Piano, 21, 108, 1);
        let plan = OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_sha256: "b".repeat(64),
            segments: vec![
                OrchestrationSegmentV1 {
                    id: "piano-part-1".into(),
                    part_id: 1,
                    instrument_id: "piano-1".into(),
                    start: RationalBeatV1 { numerator: 0, denominator: 1 },
                    end: RationalBeatV1 { numerator: 2, denominator: 1 },
                    musical_function: "melody".into(),
                    handoff: None,
                },
                OrchestrationSegmentV1 {
                    id: "piano-part-2".into(),
                    part_id: 2,
                    instrument_id: "piano-1".into(),
                    start: RationalBeatV1 { numerator: 0, denominator: 1 },
                    end: RationalBeatV1 { numerator: 2, denominator: 1 },
                    musical_function: "harmony".into(),
                    handoff: None,
                },
            ],
        };
        let report = validate_orchestration_timeline(&score, &plan, &[piano]);
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::PolyphonyExceeded
        }));
    }

    #[test]
    fn allows_abutting_notes_at_a_handoff_boundary() {
        let report = validate_orchestration_timeline(
            &base_score(),
            &base_plan(),
            &base_instruments(),
        );
        assert!(!report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::PolyphonyExceeded
        }));
    }

    #[test]
    fn rejects_unassigned_part_identity() {
        let mut score = base_score();
        score.notes[0].part = PartId::UNASSIGNED;
        let report = validate_orchestration_timeline(
            &score,
            &base_plan(),
            &base_instruments(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::MissingPartIdentity
        }));
    }

    #[test]
    fn rejects_out_of_range_notes_for_the_time_local_instrument() {
        let mut score = base_score();
        score.notes[1].pitch = Pitch::from_midi(90);
        let report = validate_orchestration_timeline(
            &score,
            &base_plan(),
            &base_instruments(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::NoteOutOfRange
        }));
    }

    #[test]
    fn rejects_non_canonical_or_malformed_score_digest() {
        let mut plan = base_plan();
        plan.score_sha256 = "not-a-digest".into();
        let report = validate_orchestration_timeline(
            &base_score(),
            &plan,
            &base_instruments(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::InvalidScoreDigest
        }));
    }
}
