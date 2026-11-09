// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Instrument-aware orchestration contracts for symbolic scores.
//!
//! Composition roles and instrument identities are separate dimensions:
//! a PartId identifies a continuing musical line, while an assignment says
//! which performer/instrument instance realizes that line. Multiple parts may
//! intentionally use the same instrument profile only when the caller models
//! that as one player and supplies a sufficient polyphony limit. Ensemble
//! sections should normally use distinct instance IDs.
//!
//! MIDI pitches in Score are treated as sounding pitches. This module does
//! not yet serialize written transpositions, instrument-specific notation,
//! breath/bow constraints, idiomatic articulation, or renderer availability.
//! Every playable range and polyphony limit is supplied by the caller, rather
//! than silently importing one supposedly universal range table.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::rhythm::Duration;
use crate::score::{PartId, Score};

/// Wire-schema identifier for an instrument orchestration plan.
pub const ORCHESTRATION_PLAN_SCHEMA_V1: &str = "instrument-orchestration-plan-v1";

/// Resource ceilings for one validation pass.
pub const MAX_ORCHESTRATION_PARTS: usize = 256;
pub const MAX_ORCHESTRATION_INSTRUMENTS: usize = 128;
pub const MAX_ORCHESTRATION_NOTES: usize = 100_000;
pub const MAX_INSTRUMENT_POLYPHONY: u8 = 64;

/// Broad acoustic/electronic family. These are useful for coverage analysis,
/// not claims that instruments within a family are interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentFamilyV1 {
    Keyboard,
    BowedStrings,
    PluckedStrings,
    Woodwinds,
    Brass,
    Percussion,
    FreeReed,
    Voice,
    Electronic,
}

/// A concrete instrument kind. The list is an extensible vocabulary, not a
/// statement that every kind is currently rendered by Muse at production
/// quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentKindV1 {
    Piano,
    Harpsichord,
    Celesta,
    Organ,
    Violin,
    Viola,
    Cello,
    DoubleBass,
    UprightBass,
    Harp,
    ClassicalGuitar,
    AcousticGuitar,
    ElectricBass,
    Mandolin,
    Banjo,
    Sitar,
    Koto,
    Oud,
    Flute,
    Piccolo,
    Oboe,
    Clarinet,
    Bassoon,
    Saxophone,
    Ney,
    Duduk,
    FrenchHorn,
    Trumpet,
    Trombone,
    Tuba,
    Timpani,
    Marimba,
    Vibraphone,
    Xylophone,
    Accordion,
    SoloVoice,
    Choir,
    Synthesizer,
}

impl InstrumentKindV1 {
    /// Family classification for catalog grouping. It does not select an
    /// instrument's range, technique, or timbral behavior.
    pub const fn family(self) -> InstrumentFamilyV1 {
        match self {
            Self::Piano | Self::Harpsichord | Self::Celesta | Self::Organ => {
                InstrumentFamilyV1::Keyboard
            }
            Self::Violin | Self::Viola | Self::Cello | Self::DoubleBass => {
                InstrumentFamilyV1::BowedStrings
            }
            Self::Harp
            | Self::ClassicalGuitar
            | Self::AcousticGuitar
            | Self::ElectricBass
            | Self::UprightBass
            | Self::Mandolin
            | Self::Banjo
            | Self::Sitar
            | Self::Koto
            | Self::Oud => InstrumentFamilyV1::PluckedStrings,
            Self::Flute
            | Self::Piccolo
            | Self::Oboe
            | Self::Clarinet
            | Self::Bassoon
            | Self::Saxophone
            | Self::Ney
            | Self::Duduk => InstrumentFamilyV1::Woodwinds,
            Self::FrenchHorn | Self::Trumpet | Self::Trombone | Self::Tuba => {
                InstrumentFamilyV1::Brass
            }
            Self::Timpani | Self::Marimba | Self::Vibraphone | Self::Xylophone => {
                InstrumentFamilyV1::Percussion
            }
            Self::Accordion => InstrumentFamilyV1::FreeReed,
            Self::SoloVoice | Self::Choir => InstrumentFamilyV1::Voice,
            Self::Synthesizer => InstrumentFamilyV1::Electronic,
        }
    }
}

/// Caller-supplied sounding range, inclusive, in MIDI note numbers (0..=127).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoundingRangeV1 {
    pub lowest_midi: u8,
    pub highest_midi: u8,
}

/// Capabilities for one concrete performer/instrument instance.
///
/// The range is deliberately explicit. A real instrument's practical range
/// depends on model, performer, technique, dynamic, register and musical
/// context. max_polyphony is the simultaneous-note budget for this instance,
/// not a family-wide constant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentCapabilityV1 {
    /// Stable instance ID, e.g. "cello_solo" or "violin_section_1".
    pub id: String,
    pub kind: InstrumentKindV1,
    pub sounding_range: SoundingRangeV1,
    pub max_polyphony: u8,
}

/// Bind a persistent score part to one instrument instance.
///
/// musical_function is an authored, local description such as "theme",
/// "counterline", "bass_ground", "harmonic_pulse" or "coloristic_response".
/// It is not inferred from VoiceRole, register, or timbre.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentPartAssignmentV1 {
    pub part_id: u16,
    pub instrument_id: String,
    pub musical_function: String,
}

/// Versioned orchestration plan. Instruments represent performer instances;
/// two violins are two IDs when they are independent players.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentOrchestrationPlanV1 {
    pub schema: String,
    pub instruments: Vec<InstrumentCapabilityV1>,
    pub assignments: Vec<InstrumentPartAssignmentV1>,
}

/// Stable machine-readable issue categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationIssueCodeV1 {
    InvalidSchema,
    EmptyScore,
    MissingPartIdentity,
    ResourceLimitExceeded,
    InvalidInstrumentProfile,
    DuplicateInstrumentId,
    InvalidPartId,
    DuplicatePartAssignment,
    MissingInstrument,
    MissingMusicalFunction,
    MissingPartAssignment,
    UnusedPartAssignment,
    NoteOutOfRange,
    InvalidScoreTime,
    PolyphonyExceeded,
}

/// One validation finding. Optional fields identify the affected score part,
/// instrument instance, or note index without inventing an inferred cause.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationIssueV1 {
    pub code: OrchestrationIssueCodeV1,
    pub part_id: Option<u16>,
    pub instrument_id: Option<String>,
    pub note_index: Option<usize>,
    pub detail: String,
}

/// Summary of a validation run. valid is true only when there are no issues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationReportV1 {
    pub valid: bool,
    pub part_count: usize,
    pub assigned_instrument_count: usize,
    pub represented_family_count: usize,
    pub issues: Vec<OrchestrationIssueV1>,
}

impl OrchestrationReportV1 {
    fn new(part_count: usize) -> Self {
        Self {
            valid: false,
            part_count,
            assigned_instrument_count: 0,
            represented_family_count: 0,
            issues: Vec::new(),
        }
    }

    fn push(
        &mut self,
        code: OrchestrationIssueCodeV1,
        part_id: Option<u16>,
        instrument_id: Option<&str>,
        note_index: Option<usize>,
        detail: impl Into<String>,
    ) {
        self.issues.push(OrchestrationIssueV1 {
            code,
            part_id,
            instrument_id: instrument_id.map(str::to_owned),
            note_index,
            detail: detail.into(),
        });
    }

    fn finish(mut self) -> Self {
        self.valid = self.issues.is_empty();
        self
    }
}

/// Validate part identity, instrument references, sounding ranges, score-time
/// values, and simultaneous-note budgets.
///
/// This is a contract verifier, not an automatic orchestrator: it verifies a
/// proposed allocation and returns explicit failures. It never silently
/// octave-shifts, drops notes, changes instruments, or rewrites the score.
pub fn validate_instrument_orchestration(
    score: &Score,
    plan: &InstrumentOrchestrationPlanV1,
) -> OrchestrationReportV1 {
    let mut report = OrchestrationReportV1::new(0);

    if plan.schema != ORCHESTRATION_PLAN_SCHEMA_V1 {
        report.push(
            OrchestrationIssueCodeV1::InvalidSchema,
            None,
            None,
            None,
            "orchestration plan schema identifier is not supported",
        );
    }
    if score.notes.is_empty() {
        report.push(
            OrchestrationIssueCodeV1::EmptyScore,
            None,
            None,
            None,
            "an empty score cannot establish an instrument allocation",
        );
    }
    if score.notes.len() > MAX_ORCHESTRATION_NOTES
        || plan.assignments.len() > MAX_ORCHESTRATION_PARTS
        || plan.instruments.len() > MAX_ORCHESTRATION_INSTRUMENTS
    {
        report.push(
            OrchestrationIssueCodeV1::ResourceLimitExceeded,
            None,
            None,
            None,
            "score or plan exceeds a declared orchestration validation resource ceiling",
        );
        return report.finish();
    }

    // Build part identity in O(n log p) rather than Score::parts()'s
    // repeated Vec membership scan. The note-count ceiling is checked first.
    let score_parts: BTreeSet<u16> = score
        .notes
        .iter()
        .filter(|note| note.part.is_assigned())
        .map(|note| note.part.0)
        .collect();
    report.part_count = score_parts.len();

    if score_parts.len() > MAX_ORCHESTRATION_PARTS {
        report.push(
            OrchestrationIssueCodeV1::ResourceLimitExceeded,
            None,
            None,
            None,
            "score contains more distinct parts than the declared orchestration limit",
        );
        return report.finish();
    }
    if !score.notes.is_empty() && !score.has_part_identity() {
        report.push(
            OrchestrationIssueCodeV1::MissingPartIdentity,
            None,
            None,
            None,
            "every note must carry an assigned PartId before instrument allocation can be verified",
        );
    }

    let mut instrument_map: HashMap<&str, &InstrumentCapabilityV1> = HashMap::new();
    let mut valid_instrument_ids: HashSet<&str> = HashSet::new();
    for instrument in &plan.instruments {
        if instrument.id.trim().is_empty()
            || instrument.sounding_range.lowest_midi > instrument.sounding_range.highest_midi
            || instrument.sounding_range.highest_midi > 127
            || instrument.max_polyphony == 0
            || instrument.max_polyphony > MAX_INSTRUMENT_POLYPHONY
        {
            report.push(
                OrchestrationIssueCodeV1::InvalidInstrumentProfile,
                None,
                Some(&instrument.id),
                None,
                "instrument ID, inclusive sounding range, or polyphony limit is invalid",
            );
            continue;
        }
        if !valid_instrument_ids.insert(instrument.id.as_str()) {
            report.push(
                OrchestrationIssueCodeV1::DuplicateInstrumentId,
                None,
                Some(&instrument.id),
                None,
                "instrument instance IDs must be unique",
            );
            continue;
        }
        instrument_map.insert(instrument.id.as_str(), instrument);
    }

    let mut assignment_map: BTreeMap<u16, &InstrumentPartAssignmentV1> = BTreeMap::new();
    for assignment in &plan.assignments {
        if assignment.part_id == u16::MAX {
            report.push(
                OrchestrationIssueCodeV1::InvalidPartId,
                Some(assignment.part_id),
                Some(&assignment.instrument_id),
                None,
                "the reserved unassigned PartId cannot be orchestrated",
            );
            continue;
        }
        if assignment.musical_function.trim().is_empty() {
            report.push(
                OrchestrationIssueCodeV1::MissingMusicalFunction,
                Some(assignment.part_id),
                Some(&assignment.instrument_id),
                None,
                "each part needs an explicit musical-function description",
            );
        }
        if !instrument_map.contains_key(assignment.instrument_id.as_str()) {
            report.push(
                OrchestrationIssueCodeV1::MissingInstrument,
                Some(assignment.part_id),
                Some(&assignment.instrument_id),
                None,
                "assignment references no valid instrument instance",
            );
        }
        if assignment_map.contains_key(&assignment.part_id) {
            report.push(
                OrchestrationIssueCodeV1::DuplicatePartAssignment,
                Some(assignment.part_id),
                Some(&assignment.instrument_id),
                None,
                "a PartId must have exactly one instrument assignment in schema v1",
            );
            continue;
        }
        assignment_map.insert(assignment.part_id, assignment);
        if !score_parts.contains(&assignment.part_id) {
            report.push(
                OrchestrationIssueCodeV1::UnusedPartAssignment,
                Some(assignment.part_id),
                Some(&assignment.instrument_id),
                None,
                "assignment refers to a PartId that is not present in the score",
            );
        }
    }

    for part_id in &score_parts {
        if !assignment_map.contains_key(part_id) {
            report.push(
                OrchestrationIssueCodeV1::MissingPartAssignment,
                Some(*part_id),
                None,
                None,
                "score part has no instrument assignment",
            );
        }
    }

    let mut used_instruments: BTreeSet<String> = BTreeSet::new();
    let mut used_families: HashSet<InstrumentFamilyV1> = HashSet::new();
    let mut polyphony_events: BTreeMap<String, Vec<PolyphonyEvent>> = BTreeMap::new();

    for (note_index, note) in score.notes.iter().enumerate() {
        let part_id = note.part.0;
        if !note.part.is_assigned() {
            continue;
        }
        let Some(assignment) = assignment_map.get(&part_id) else {
            continue;
        };
        let Some(instrument) = instrument_map.get(assignment.instrument_id.as_str()) else {
            continue;
        };

        used_instruments.insert(instrument.id.clone());
        used_families.insert(instrument.kind.family());

        let midi = note.pitch.midi();
        if midi < instrument.sounding_range.lowest_midi
            || midi > instrument.sounding_range.highest_midi
        {
            report.push(
                OrchestrationIssueCodeV1::NoteOutOfRange,
                Some(part_id),
                Some(&instrument.id),
                Some(note_index),
                format!(
                    "sounding MIDI note {midi} is outside inclusive range {}..={}",
                    instrument.sounding_range.lowest_midi,
                    instrument.sounding_range.highest_midi
                ),
            );
        }

        let Some(start) = RationalMoment::from_duration(note.onset, false) else {
            report.push(
                OrchestrationIssueCodeV1::InvalidScoreTime,
                Some(part_id),
                Some(&instrument.id),
                Some(note_index),
                "note onset is negative or exceeds the bounded rational-time domain",
            );
            continue;
        };
        let Some(duration) = RationalMoment::from_duration(note.duration, true) else {
            report.push(
                OrchestrationIssueCodeV1::InvalidScoreTime,
                Some(part_id),
                Some(&instrument.id),
                Some(note_index),
                "note duration is not positive or exceeds the bounded rational-time domain",
            );
            continue;
        };
        let Some(end) = start.checked_add(duration) else {
            report.push(
                OrchestrationIssueCodeV1::InvalidScoreTime,
                Some(part_id),
                Some(&instrument.id),
                Some(note_index),
                "note end exceeds the bounded rational-time domain",
            );
            continue;
        };

        let events = polyphony_events.entry(instrument.id.clone()).or_default();
        events.push(PolyphonyEvent {
            when: start,
            delta: 1,
            note_index,
        });
        events.push(PolyphonyEvent {
            when: end,
            delta: -1,
            note_index,
        });
    }

    for (instrument_id, events) in &mut polyphony_events {
        events.sort_by(|left, right| {
            left.when
                .compare(right.when)
                .then_with(|| left.delta.cmp(&right.delta))
                .then_with(|| left.note_index.cmp(&right.note_index))
        });
        let Some(instrument) = instrument_map.get(instrument_id.as_str()) else {
            continue;
        };
        let mut active = 0i32;
        let mut maximum = 0i32;
        for event in events.iter() {
            active += event.delta as i32;
            maximum = maximum.max(active);
        }
        if maximum > instrument.max_polyphony as i32 {
            report.push(
                OrchestrationIssueCodeV1::PolyphonyExceeded,
                None,
                Some(instrument_id),
                None,
                format!(
                    "simultaneous note demand {maximum} exceeds instance budget {}",
                    instrument.max_polyphony
                ),
            );
        }
    }

    report.assigned_instrument_count = used_instruments.len();
    report.represented_family_count = used_families.len();
    report.finish()
}

/// Rational score-time coordinate used for exact overlap checks.
///
/// Inputs are bounded before construction. The largest cross-products are
/// therefore far below i128 limits; no floating-point comparisons are used.
#[derive(Debug, Clone, Copy)]
struct RationalMoment {
    numerator: i128,
    denominator: i128,
}

impl RationalMoment {
    const MAX_COMPONENT_NUMERATOR: i128 = 1_000_000_000;
    const MAX_COMPONENT_DENOMINATOR: i128 = 1_000_000;

    fn from_duration(value: Duration, require_positive: bool) -> Option<Self> {
        let numerator = value.num() as i128;
        let denominator = value.den() as i128;
        if denominator <= 0
            || denominator > Self::MAX_COMPONENT_DENOMINATOR
            || numerator.abs() > Self::MAX_COMPONENT_NUMERATOR
            || (require_positive && numerator <= 0)
            || (!require_positive && numerator < 0)
        {
            return None;
        }
        Some(Self {
            numerator,
            denominator,
        })
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

    fn compare(self, other: Self) -> std::cmp::Ordering {
        (self.numerator * other.denominator).cmp(&(other.numerator * self.denominator))
    }
}

struct PolyphonyEvent {
    when: RationalMoment,
    /// -1=end and +1=start; sorting -1 first makes abutting notes non-overlapping.
    delta: i8,
    note_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::score::{Emphasis, ScoreNote, VoiceRole};

    fn score_with_parts() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 96.0, 4);
        for (part, midi, role) in [
            (PartId(1), 48, VoiceRole::Bass),
            (PartId(2), 60, VoiceRole::Harmony),
            (PartId(3), 55, VoiceRole::CounterMelody),
        ] {
            score.push(ScoreNote {
                part,
                pitch: Pitch::from_midi(midi),
                onset: Duration::zero(),
                duration: Duration::whole(),
                velocity: 0.7,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn plan() -> InstrumentOrchestrationPlanV1 {
        InstrumentOrchestrationPlanV1 {
            schema: ORCHESTRATION_PLAN_SCHEMA_V1.into(),
            instruments: vec![
                InstrumentCapabilityV1 {
                    id: "cello_solo".into(),
                    kind: InstrumentKindV1::Cello,
                    sounding_range: SoundingRangeV1 {
                        lowest_midi: 36,
                        highest_midi: 84,
                    },
                    max_polyphony: 4,
                },
                InstrumentCapabilityV1 {
                    id: "grand_piano".into(),
                    kind: InstrumentKindV1::Piano,
                    sounding_range: SoundingRangeV1 {
                        lowest_midi: 21,
                        highest_midi: 108,
                    },
                    max_polyphony: 10,
                },
                InstrumentCapabilityV1 {
                    id: "horn_solo".into(),
                    kind: InstrumentKindV1::FrenchHorn,
                    sounding_range: SoundingRangeV1 {
                        lowest_midi: 34,
                        highest_midi: 77,
                    },
                    max_polyphony: 1,
                },
            ],
            assignments: vec![
                InstrumentPartAssignmentV1 {
                    part_id: 1,
                    instrument_id: "cello_solo".into(),
                    musical_function: "bass_ground".into(),
                },
                InstrumentPartAssignmentV1 {
                    part_id: 2,
                    instrument_id: "grand_piano".into(),
                    musical_function: "harmonic_motion".into(),
                },
                InstrumentPartAssignmentV1 {
                    part_id: 3,
                    instrument_id: "horn_solo".into(),
                    musical_function: "counter_theme".into(),
                },
            ],
        }
    }

    #[test]
    fn accepts_multi_family_cellopiano_french_horn_plan() {
        let report = validate_instrument_orchestration(&score_with_parts(), &plan());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.part_count, 3);
        assert_eq!(report.assigned_instrument_count, 3);
        assert_eq!(report.represented_family_count, 3);
    }

    #[test]
    fn rejects_unassigned_score_parts_instead_of_guessing_from_role() {
        let mut score = score_with_parts();
        score.notes[0].part = PartId::UNASSIGNED;
        let report = validate_instrument_orchestration(&score, &plan());
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::MissingPartIdentity
        }));
    }

    #[test]
    fn rejects_a_missing_part_assignment() {
        let mut value = plan();
        value.assignments.pop();
        let report = validate_instrument_orchestration(&score_with_parts(), &value);
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::MissingPartAssignment
        }));
    }

    #[test]
    fn rejects_an_out_of_range_note_without_transposing_it() {
        let mut score = score_with_parts();
        score.notes[0].pitch = Pitch::from_midi(30);
        let report = validate_instrument_orchestration(&score, &plan());
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::NoteOutOfRange
        }));
    }

    #[test]
    fn rejects_duplicate_part_assignments() {
        let mut value = plan();
        value.assignments.push(value.assignments[0].clone());
        let report = validate_instrument_orchestration(&score_with_parts(), &value);
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::DuplicatePartAssignment
        }));
    }

    #[test]
    fn rejects_a_reference_to_an_unknown_instrument_instance() {
        let mut value = plan();
        value.assignments[0].instrument_id = "missing_player".into();
        let report = validate_instrument_orchestration(&score_with_parts(), &value);
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::MissingInstrument
        }));
    }

    #[test]
    fn enforces_simultaneous_note_budget_across_parts_on_one_player() {
        let mut score = score_with_parts();
        score.notes[1].part = PartId(1);
        score.notes[1].pitch = Pitch::from_midi(60);
        let mut value = plan();
        value.instruments[0].max_polyphony = 1;
        value.assignments.retain(|assignment| assignment.part_id != 2);
        let report = validate_instrument_orchestration(&score, &value);
        assert!(report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::PolyphonyExceeded
        }));
    }

    #[test]
    fn end_and_start_at_same_rational_tick_do_not_consume_two_slots() {
        let mut score = score_with_parts();
        score.notes[1].part = PartId(1);
        score.notes[1].onset = Duration::whole();
        score.notes[1].duration = Duration::whole();
        score.notes[0].duration = Duration::whole();
        let mut value = plan();
        value.instruments[0].max_polyphony = 1;
        value.assignments.retain(|assignment| assignment.part_id != 2);
        let report = validate_instrument_orchestration(&score, &value);
        assert!(!report.issues.iter().any(|issue| {
            issue.code == OrchestrationIssueCodeV1::PolyphonyExceeded
        }));
    }

    #[test]
    fn instrument_kind_maps_to_distinct_families() {
        assert_eq!(
            InstrumentKindV1::FrenchHorn.family(),
            InstrumentFamilyV1::Brass
        );
        assert_eq!(
            InstrumentKindV1::Harp.family(),
            InstrumentFamilyV1::PluckedStrings
        );
        assert_eq!(
            InstrumentKindV1::Cello.family(),
            InstrumentFamilyV1::BowedStrings
        );
        assert_eq!(
            InstrumentKindV1::Saxophone.family(),
            InstrumentFamilyV1::Woodwinds
        );
        assert_eq!(
            InstrumentKindV1::UprightBass.family(),
            InstrumentFamilyV1::PluckedStrings
        );
    }

    #[test]
    fn rejects_an_invalid_range_or_zero_polyphony_profile() {
        let mut value = plan();
        value.instruments[0].sounding_range.lowest_midi = 90;
        value.instruments[1].max_polyphony = 0;
        let report = validate_instrument_orchestration(&score_with_parts(), &value);
        assert_eq!(
            report.issues.iter().filter(|issue| {
                issue.code == OrchestrationIssueCodeV1::InvalidInstrumentProfile
            }).count(),
            2
        );
    }

    #[test]
    fn rational_overlap_checks_accept_abutting_events() {
        let a = RationalMoment { numerator: 2, denominator: 3 };
        let b = RationalMoment { numerator: 4, denominator: 6 };
        assert_eq!(a.compare(b), std::cmp::Ordering::Equal);
        assert_eq!(
            a.checked_add(RationalMoment { numerator: 1, denominator: 3 })
                .unwrap()
                .compare(RationalMoment { numerator: 1, denominator: 1 }),
            std::cmp::Ordering::Equal
        );
    }
}
