// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-authored temporal relationships between persistent musical parts.
//!
//! This contract records why parts interact: doubling, imitation, canon,
//! call-and-response, counterpoint, support, interlock, or deliberate silence.
//! It validates identity, provenance binding, time windows and participant
//! coverage. It does not infer a relationship from pitch/role similarity and
//! does not certify that a declared relationship is perceptually convincing.

use serde::{Deserialize, Serialize};

use crate::rhythm::Duration;
use crate::score::{PartId, Score};

pub const ENSEMBLE_RELATION_SCHEMA_V1: &str = "ensemble-relations-v1";
/// Canonical SHA-256 of serialized Score bytes, shared with Muse genealogy.
pub const ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1: &str = "muse-score-serde-json-sha256-v1";
pub const MAX_ENSEMBLE_RELATIONS_V1: usize = 4_096;
const MAX_RELATION_BEAT_NUMERATOR: i128 = 1_000_000_000;
const MAX_RELATION_BEAT_DENOMINATOR: i128 = 1_000_000;

/// Exact non-negative score-time boundary in beats. Intervals are half-open:
/// [start, end). This mirrors the orchestration timeline exact-time policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationBeatV1 {
    pub numerator: i64,
    pub denominator: i64,
}

/// Musical relationship authored by the composition engine, not inferred
/// from register, VoiceRole, pitch similarity or note overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleRelationKindV1 {
    Doubling,
    Imitation,
    Canon,
    CallAndResponse,
    Counterpoint,
    DroneSupport,
    HarmonicSupport,
    RhythmicInterlock,
    /// Explicitly planned non-sounding interval for one part.
    PlannedSilence,
}

/// One relationship declaration over an exact score-time window.
///
/// part_a is the subject. part_b is the partner and is required for two-part
/// relationships. PlannedSilence is unary and requires part_b=None.
/// material_id optionally binds both participants to an authored motif or
/// other persistent musical material identifier; its semantics are owned by
/// the upstream composition/material-lineage layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleRelationV1 {
    pub id: String,
    pub kind: EnsembleRelationKindV1,
    pub part_a: u16,
    pub part_b: Option<u16>,
    pub start: RelationBeatV1,
    pub end: RelationBeatV1,
    pub material_id: Option<String>,
}

/// Complete relationship graph for one exact symbolic score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleRelationTimelineV1 {
    pub schema: String,
    /// Digest recipe/version used for score_sha256.
    pub score_digest_schema: String,
    /// SHA-256 of the exact symbolic Score, computed by the trusted upstream
    /// score-demand/provenance layer and passed separately to validation.
    pub score_sha256: String,
    pub relations: Vec<EnsembleRelationV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleRelationIssueCodeV1 {
    InvalidSchema,
    InvalidScoreDigestSchema,
    InvalidScoreDigest,
    InvalidExpectedScoreDigest,
    ScoreDigestMismatch,
    EmptyScore,
    MissingPartIdentity,
    ResourceLimitExceeded,
    InvalidRelationId,
    DuplicateRelationId,
    InvalidPartId,
    MissingSecondPart,
    UnexpectedSecondPart,
    SelfRelation,
    UnknownScorePart,
    InvalidTime,
    EmptyInterval,
    IntervalBeyondScore,
    EmptyMaterialId,
    MissingParticipantEvents,
    PlannedSilenceContainsEvents,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleRelationIssueV1 {
    pub code: EnsembleRelationIssueCodeV1,
    pub relation_id: Option<String>,
    pub part_id: Option<u16>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleRelationReportV1 {
    pub valid: bool,
    pub part_count: usize,
    pub relation_count: usize,
    pub issues: Vec<EnsembleRelationIssueV1>,
}

impl EnsembleRelationReportV1 {
    fn new(part_count: usize, relation_count: usize) -> Self {
        Self {
            valid: true,
            part_count,
            relation_count,
            issues: Vec::new(),
        }
    }

    fn push(
        &mut self,
        code: EnsembleRelationIssueCodeV1,
        relation_id: Option<&str>,
        part_id: Option<u16>,
        message: impl Into<String>,
    ) {
        self.valid = false;
        self.issues.push(EnsembleRelationIssueV1 {
            code,
            relation_id: relation_id.map(str::to_owned),
            part_id,
            message: message.into(),
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct Moment {
    num: i128,
    den: i128,
}

impl Moment {
    fn from_pair(num: i128, den: i128) -> Option<Self> {
        if den <= 0
            || den > MAX_RELATION_BEAT_DENOMINATOR
            || num < 0
            || num > MAX_RELATION_BEAT_NUMERATOR
        {
            return None;
        }
        Some(Self { num, den })
    }

    fn from_wire(value: RelationBeatV1) -> Option<Self> {
        Self::from_pair(i128::from(value.numerator), i128::from(value.denominator))
    }

    fn from_duration(value: Duration) -> Option<Self> {
        Self::from_pair(i128::from(value.num()), i128::from(value.den()))
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        let left = self.num.checked_mul(rhs.den)?;
        let right = rhs.num.checked_mul(self.den)?;
        let num = left.checked_add(right)?;
        let den = self.den.checked_mul(rhs.den)?;
        Self::from_pair(num, den)
    }

    fn less_than(self, rhs: Self) -> bool {
        self.num * rhs.den < rhs.num * self.den
    }

    fn greater_than(self, rhs: Self) -> bool {
        self.num * rhs.den > rhs.num * self.den
    }

    fn less_or_equal(self, rhs: Self) -> bool {
        !self.greater_than(rhs)
    }
}

fn is_valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn has_note_in_window(score: &Score, part: PartId, start: Moment, end: Moment) -> bool {
    score.notes.iter().any(|note| {
        if note.part != part {
            return false;
        }
        let Some(onset) = Moment::from_duration(note.onset) else {
            return false;
        };
        let Some(duration) = Moment::from_duration(note.duration) else {
            return false;
        };
        let Some(note_end) = onset.checked_add(duration) else {
            return false;
        };
        onset.less_than(end) && note_end.greater_than(start)
    })
}

/// Validate a source-authored relationship timeline against one exact Score.
///
/// This checks graph integrity, exact half-open time windows, part identity,
/// participant event coverage and explicit silence. It does not decide whether
/// an imitation, doubling, counterpoint or other relationship is artistically
/// successful; that requires musical analysis and/or expert listening.
pub fn validate_ensemble_relations(
    score: &Score,
    timeline: &EnsembleRelationTimelineV1,
    expected_score_sha256: &str,
) -> EnsembleRelationReportV1 {
    let part_count = score.parts().len();
    let mut report = EnsembleRelationReportV1::new(part_count, timeline.relations.len());

    if timeline.schema != ENSEMBLE_RELATION_SCHEMA_V1 {
        report.push(
            EnsembleRelationIssueCodeV1::InvalidSchema,
            None,
            None,
            "unsupported ensemble-relation schema",
        );
    }
    if timeline.score_digest_schema != ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1 {
        report.push(
            EnsembleRelationIssueCodeV1::InvalidScoreDigestSchema,
            None,
            None,
            "relationship timeline must use the exact score-demand digest schema",
        );
    }
    if timeline.score_digest_schema != ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1 {
        report.push(
            EnsembleRelationIssueCodeV1::InvalidScoreDigestSchema,
            None,
            None,
            "unsupported score digest schema; use the exact score-demand serialization contract",
        );
    }
    if !is_valid_digest(&timeline.score_sha256) {
        report.push(
            EnsembleRelationIssueCodeV1::InvalidScoreDigest,
            None,
            None,
            "timeline score_sha256 must be 64 lowercase hexadecimal characters",
        );
    }
    if !is_valid_digest(expected_score_sha256) {
        report.push(
            EnsembleRelationIssueCodeV1::InvalidExpectedScoreDigest,
            None,
            None,
            "expected score digest must come from the trusted score-demand/provenance layer",
        );
    } else if is_valid_digest(&timeline.score_sha256)
        && timeline.score_sha256 != expected_score_sha256
    {
        report.push(
            EnsembleRelationIssueCodeV1::ScoreDigestMismatch,
            None,
            None,
            "relationship timeline is bound to a different symbolic Score",
        );
    }
    if score.notes.is_empty() {
        report.push(
            EnsembleRelationIssueCodeV1::EmptyScore,
            None,
            None,
            "cannot validate relationships against an empty Score",
        );
    } else if !score.has_part_identity() {
        report.push(
            EnsembleRelationIssueCodeV1::MissingPartIdentity,
            None,
            None,
            "every note needs a generator-authored PartId; VoiceRole is not a substitute",
        );
    }
    if timeline.relations.len() > MAX_ENSEMBLE_RELATIONS_V1 {
        report.push(
            EnsembleRelationIssueCodeV1::ResourceLimitExceeded,
            None,
            None,
            "relationship count exceeds the v1 resource limit",
        );
        return report;
    }

    let known_parts: std::collections::BTreeSet<PartId> = score.parts().into_iter().collect();
    let mut seen_ids = std::collections::BTreeSet::new();
    let Some(score_end) = Moment::from_duration(score.total_beats) else {
        report.push(
            EnsembleRelationIssueCodeV1::InvalidTime,
            None,
            None,
            "score total_beats is not representable as a bounded non-negative rational",
        );
        return report;
    };

    for relation in &timeline.relations {
        if relation.id.trim().is_empty() {
            report.push(
                EnsembleRelationIssueCodeV1::InvalidRelationId,
                Some(&relation.id),
                None,
                "relation id must be non-empty",
            );
        } else if !seen_ids.insert(relation.id.as_str()) {
            report.push(
                EnsembleRelationIssueCodeV1::DuplicateRelationId,
                Some(&relation.id),
                None,
                "relation id must be unique within the timeline",
            );
        }
        if relation.part_a == PartId::UNASSIGNED.0 {
            report.push(
                EnsembleRelationIssueCodeV1::InvalidPartId,
                Some(&relation.id),
                Some(relation.part_a),
                "part_a must be a generator-authored PartId",
            );
            continue;
        }
        let a = PartId(relation.part_a);
        if !known_parts.contains(&a) {
            report.push(
                EnsembleRelationIssueCodeV1::UnknownScorePart,
                Some(&relation.id),
                Some(relation.part_a),
                "part_a is not present in the validated Score",
            );
            continue;
        }

        let unary = relation.kind == EnsembleRelationKindV1::PlannedSilence;
        match (unary, relation.part_b) {
            (true, None) => {}
            (true, Some(_)) => report.push(
                EnsembleRelationIssueCodeV1::UnexpectedSecondPart,
                Some(&relation.id),
                relation.part_b,
                "PlannedSilence is unary and must not declare part_b",
            ),
            (false, None) => report.push(
                EnsembleRelationIssueCodeV1::MissingSecondPart,
                Some(&relation.id),
                Some(relation.part_a),
                "this relationship kind requires a second persistent part",
            ),
            (false, Some(second)) if second == PartId::UNASSIGNED.0 => report.push(
                EnsembleRelationIssueCodeV1::InvalidPartId,
                Some(&relation.id),
                Some(second),
                "part_b must be a generator-authored PartId",
            ),
            (false, Some(second)) if second == relation.part_a => report.push(
                EnsembleRelationIssueCodeV1::SelfRelation,
                Some(&relation.id),
                Some(second),
                "a two-part relationship must reference two distinct parts",
            ),
            (false, Some(second)) if !known_parts.contains(&PartId(second)) => report.push(
                EnsembleRelationIssueCodeV1::UnknownScorePart,
                Some(&relation.id),
                Some(second),
                "part_b is not present in the validated Score",
            ),
            _ => {}
        }

        let (Some(start), Some(end)) = (
            Moment::from_wire(relation.start),
            Moment::from_wire(relation.end),
        ) else {
            report.push(
                EnsembleRelationIssueCodeV1::InvalidTime,
                Some(&relation.id),
                Some(relation.part_a),
                "relationship boundaries require bounded non-negative rational beats",
            );
            continue;
        };
        if !start.less_than(end) {
            report.push(
                EnsembleRelationIssueCodeV1::EmptyInterval,
                Some(&relation.id),
                Some(relation.part_a),
                "relationship window must have positive duration",
            );
            continue;
        }
        if !end.less_or_equal(score_end) {
            report.push(
                EnsembleRelationIssueCodeV1::IntervalBeyondScore,
                Some(&relation.id),
                Some(relation.part_a),
                "relationship window extends beyond the symbolic Score",
            );
            continue;
        }
        if relation.material_id.as_ref().is_some_and(|id| id.trim().is_empty()) {
            report.push(
                EnsembleRelationIssueCodeV1::EmptyMaterialId,
                Some(&relation.id),
                Some(relation.part_a),
                "material_id must be absent or non-empty",
            );
        }

        let a_has_events = has_note_in_window(score, a, start, end);
        if unary {
            if a_has_events {
                report.push(
                    EnsembleRelationIssueCodeV1::PlannedSilenceContainsEvents,
                    Some(&relation.id),
                    Some(relation.part_a),
                    "planned-silence window contains sounding events for part_a",
                );
            }
        } else if let Some(second) = relation.part_b {
            if !a_has_events {
                report.push(
                    EnsembleRelationIssueCodeV1::MissingParticipantEvents,
                    Some(&relation.id),
                    Some(relation.part_a),
                    "part_a has no sounding events in the declared relationship window",
                );
            }
            if second != PartId::UNASSIGNED.0
                && second != relation.part_a
                && known_parts.contains(&PartId(second))
                && !has_note_in_window(score, PartId(second), start, end)
            {
                report.push(
                    EnsembleRelationIssueCodeV1::MissingParticipantEvents,
                    Some(&relation.id),
                    Some(second),
                    "part_b has no sounding events in the declared relationship window",
                );
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::score::{Emphasis, ScoreNote, VoiceRole};

    fn score() -> Score {
        let key = Key::major(PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        for (part, midi, onset) in [(1, 60, 0), (2, 67, 0), (3, 72, 4)] {
            score.push(ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(2, 1),
                velocity: 0.7,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn digest() -> String {
        "a".repeat(64)
    }

    fn relation(kind: EnsembleRelationKindV1) -> EnsembleRelationV1 {
        EnsembleRelationV1 {
            id: "relation-1".into(),
            kind,
            part_a: 1,
            part_b: if kind == EnsembleRelationKindV1::PlannedSilence {
                None
            } else {
                Some(2)
            },
            start: RelationBeatV1 { numerator: 0, denominator: 1 },
            end: RelationBeatV1 { numerator: 2, denominator: 1 },
            material_id: Some("theme-alpha".into()),
        }
    }

    fn timeline(relations: Vec<EnsembleRelationV1>) -> EnsembleRelationTimelineV1 {
        EnsembleRelationTimelineV1 {
            schema: ENSEMBLE_RELATION_SCHEMA_V1.into(),
            score_digest_schema: ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: digest(),
            relations,
        }
    }

    #[test]
    fn accepts_a_source_authored_two_part_relation() {
        let report = validate_ensemble_relations(
            &score(),
            &timeline(vec![relation(EnsembleRelationKindV1::Doubling)]),
            &digest(),
        );
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.part_count, 3);
        assert_eq!(report.relation_count, 1);
    }

    #[test]
    fn planned_silence_is_explicit_and_checked() {
        let score = score();
        let mut silence = relation(EnsembleRelationKindV1::PlannedSilence);
        silence.part_a = 3;
        silence.start = RelationBeatV1 { numerator: 0, denominator: 1 };
        silence.end = RelationBeatV1 { numerator: 2, denominator: 1 };
        let accepted = validate_ensemble_relations(&score, &timeline(vec![silence.clone()]), &digest());
        assert!(accepted.valid, "{:?}", accepted.issues);

        silence.part_a = 1;
        let rejected = validate_ensemble_relations(&score, &timeline(vec![silence]), &digest());
        assert!(rejected.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::PlannedSilenceContainsEvents
        }));
    }

    #[test]
    fn rejects_unknown_or_unassigned_part_references() {
        let score = score();
        let mut unknown = relation(EnsembleRelationKindV1::Canon);
        unknown.part_b = Some(99);
        let mut unassigned = relation(EnsembleRelationKindV1::Imitation);
        unassigned.part_a = PartId::UNASSIGNED.0;
        let report = validate_ensemble_relations(&score, &timeline(vec![unknown, unassigned]), &digest());
        assert!(report.issues.iter().any(|issue| issue.code == EnsembleRelationIssueCodeV1::UnknownScorePart));
        assert!(report.issues.iter().any(|issue| issue.code == EnsembleRelationIssueCodeV1::InvalidPartId));
    }

    #[test]
    fn rejects_empty_windows_and_windows_beyond_score() {
        let score = score();
        let mut empty = relation(EnsembleRelationKindV1::Counterpoint);
        empty.end = empty.start;
        let mut beyond = relation(EnsembleRelationKindV1::Counterpoint);
        beyond.end = RelationBeatV1 { numerator: 99, denominator: 1 };
        let report = validate_ensemble_relations(&score, &timeline(vec![empty, beyond]), &digest());
        assert!(report.issues.iter().any(|issue| issue.code == EnsembleRelationIssueCodeV1::EmptyInterval));
        assert!(report.issues.iter().any(|issue| issue.code == EnsembleRelationIssueCodeV1::IntervalBeyondScore));
    }

    #[test]
    fn rejects_an_unknown_score_digest_schema() {
        let score = score();
        let mut relation_timeline = timeline(vec![relation(EnsembleRelationKindV1::Imitation)]);
        relation_timeline.score_digest_schema = "sha256-json-v0".into();
        let report = validate_ensemble_relations(&score, &relation_timeline, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::InvalidScoreDigestSchema
        }));
    }

    #[test]
    fn rejects_a_mismatched_score_digest_schema() {
        let mut timeline = timeline(vec![relation(EnsembleRelationKindV1::Imitation)]);
        timeline.score_digest_schema = "unknown-score-hash-v0".into();
        let report = validate_ensemble_relations(&score(), &timeline, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::InvalidScoreDigestSchema
        }));
    }

    #[test]
    fn rejects_a_relation_timeline_bound_to_another_score() {
        let report = validate_ensemble_relations(
            &score(),
            &timeline(vec![relation(EnsembleRelationKindV1::Imitation)]),
            &"b".repeat(64),
        );
        assert!(report.issues.iter().any(|issue| issue.code == EnsembleRelationIssueCodeV1::ScoreDigestMismatch));
    }

    #[test]
    fn exact_half_open_boundary_allows_silence_after_a_note_ends() {
        let score = score();
        let mut silence = relation(EnsembleRelationKindV1::PlannedSilence);
        silence.part_a = 1;
        silence.start = RelationBeatV1 { numerator: 2, denominator: 1 };
        silence.end = RelationBeatV1 { numerator: 4, denominator: 1 };
        let report = validate_ensemble_relations(&score, &timeline(vec![silence]), &digest());
        assert!(report.valid, "{:?}", report.issues);
    }

    #[test]
    fn rejects_missing_partner_and_self_relationships() {
        let score = score();
        let mut missing = relation(EnsembleRelationKindV1::Counterpoint);
        missing.part_b = None;
        let mut self_edge = relation(EnsembleRelationKindV1::Imitation);
        self_edge.part_b = Some(self_edge.part_a);
        self_edge.id = "self-edge".into();
        let report = validate_ensemble_relations(
            &score,
            &timeline(vec![missing, self_edge]),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::MissingSecondPart
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::SelfRelation
        }));
    }

    #[test]
    fn rejects_duplicate_relation_ids_and_empty_material_ids() {
        let score = score();
        let first = relation(EnsembleRelationKindV1::Doubling);
        let mut second = relation(EnsembleRelationKindV1::Canon);
        second.material_id = Some("   ".into());
        let report = validate_ensemble_relations(
            &score,
            &timeline(vec![first, second]),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::DuplicateRelationId
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == EnsembleRelationIssueCodeV1::EmptyMaterialId
        }));
    }

    #[test]
    fn rejects_a_relationship_that_claims_a_second_part_for_silence() {
        let mut silence = relation(EnsembleRelationKindV1::PlannedSilence);
        silence.part_b = Some(2);
        let report = validate_ensemble_relations(&score(), &timeline(vec![silence]), &digest());
        assert!(report.issues.iter().any(|issue| issue.code == EnsembleRelationIssueCodeV1::UnexpectedSecondPart));
    }
}
