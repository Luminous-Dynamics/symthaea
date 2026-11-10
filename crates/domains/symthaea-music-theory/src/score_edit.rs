// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Digest-bound, part-scoped and time-bounded symbolic score edits.
//!
//! V1 intentionally implements one safe edit class: scaling velocity for
//! generator-identified notes fully contained in an authorized score-time
//! window. It returns a candidate Score and an input-bound diff receipt;
//! it never mutates the source Score in place and never commits a candidate.
//!
//! Event indices in receipts are ordinals scoped to the exact input score
//! digest. They are not persistent EventIds and must not be reused after the
//! source Score changes. Persistent event identity remains a separate task.

use serde::{Deserialize, Serialize};

use crate::rhythm::Duration;
use crate::score::{PartId, Score};

pub const SCORE_EDIT_SCHEMA_V1: &str = "score-edit-plan-v1";
pub const MAX_SCORE_EDIT_OPERATIONS_V1: usize = 128;
pub const MAX_PROTECTED_WINDOWS_V1: usize = 512;
const MAX_EDIT_BEAT_NUMERATOR: i128 = 1_000_000_000;
const MAX_EDIT_BEAT_DENOMINATOR: i128 = 1_000_000;
const MAX_VELOCITY_SCALE_COMPONENT: u16 = 64;

/// Exact non-negative score time in beats. Scope intervals are half-open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditBeatV1 {
    pub numerator: i64,
    pub denominator: i64,
}

/// One explicit, localized score transformation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalePartVelocityV1 {
    pub id: String,
    pub part_id: u16,
    pub start: EditBeatV1,
    pub end: EditBeatV1,
    /// New velocity = old velocity * numerator / denominator.
    /// No implicit clamping is permitted.
    pub numerator: u16,
    pub denominator: u16,
}

/// A region whose source events must remain unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedPartWindowV1 {
    pub id: String,
    pub part_id: u16,
    pub start: EditBeatV1,
    pub end: EditBeatV1,
}

/// A versioned edit plan bound to one exact symbolic Score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEditPlanV1 {
    pub schema: String,
    /// SHA-256 of the exact input Score, supplied by trusted provenance.
    pub input_score_sha256: String,
    pub operations: Vec<ScalePartVelocityV1>,
    pub protected_windows: Vec<ProtectedPartWindowV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEditIssueCodeV1 {
    InvalidSchema,
    InvalidInputDigest,
    InvalidExpectedInputDigest,
    InputDigestMismatch,
    EmptyScore,
    MissingPartIdentity,
    ResourceLimitExceeded,
    InvalidOperationId,
    DuplicateOperationId,
    InvalidProtectedWindowId,
    DuplicateProtectedWindowId,
    InvalidPartId,
    UnknownScorePart,
    InvalidTime,
    EmptyInterval,
    IntervalBeyondScore,
    InvalidVelocityScale,
    EmptyTargetSelection,
    PartialNoteBoundary,
    VelocityOutOfRange,
    ProtectedScopeOverlap,
    OverlappingOperations,
    InvalidSourceVelocity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEditIssueV1 {
    pub code: ScoreEditIssueCodeV1,
    pub operation_id: Option<String>,
    pub part_id: Option<u16>,
    pub note_index: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEditReceiptV1 {
    pub schema: String,
    pub input_score_sha256: String,
    pub operation_ids: Vec<String>,
    /// Event ordinals scoped exclusively to input_score_sha256.
    pub changed_note_indices: Vec<usize>,
    pub protected_window_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreEditCandidateV1 {
    pub candidate: Score,
    pub receipt: ScoreEditReceiptV1,
}

#[derive(Debug, Clone, Copy)]
struct Moment {
    num: i128,
    den: i128,
}

impl Moment {
    fn from_pair(num: i128, den: i128) -> Option<Self> {
        if den <= 0
            || den > MAX_EDIT_BEAT_DENOMINATOR
            || num < 0
            || num > MAX_EDIT_BEAT_NUMERATOR
        {
            return None;
        }
        Some(Self { num, den })
    }

    fn from_wire(value: EditBeatV1) -> Option<Self> {
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

struct PreparedOperation<'a> {
    operation: &'a ScalePartVelocityV1,
    start: Moment,
    end: Moment,
    note_indices: Vec<usize>,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn intersects(start_a: Moment, end_a: Moment, start_b: Moment, end_b: Moment) -> bool {
    start_a.less_than(end_b) && end_a.greater_than(start_b)
}

fn validate_window(
    start_value: EditBeatV1,
    end_value: EditBeatV1,
    score_end: Moment,
) -> Result<(Moment, Moment), (ScoreEditIssueCodeV1, &'static str)> {
    let (Some(start), Some(end)) = (
        Moment::from_wire(start_value),
        Moment::from_wire(end_value),
    ) else {
        return Err((
            ScoreEditIssueCodeV1::InvalidTime,
            "time boundaries require bounded non-negative rational beats",
        ));
    };
    if !start.less_than(end) {
        return Err((
            ScoreEditIssueCodeV1::EmptyInterval,
            "edit window must have positive duration",
        ));
    }
    if !end.less_or_equal(score_end) {
        return Err((
            ScoreEditIssueCodeV1::IntervalBeyondScore,
            "edit window extends beyond the exact input Score",
        ));
    }
    Ok((start, end))
}

fn push_issue(
    issues: &mut Vec<ScoreEditIssueV1>,
    code: ScoreEditIssueCodeV1,
    operation_id: Option<&str>,
    part_id: Option<u16>,
    note_index: Option<usize>,
    message: impl Into<String>,
) {
    issues.push(ScoreEditIssueV1 {
        code,
        operation_id: operation_id.map(str::to_owned),
        part_id,
        note_index,
        message: message.into(),
    });
}

/// Preview a localized symbolic edit against one frozen Score.
///
/// The operation is all-or-nothing. A note that intersects but crosses the
/// edit-window boundary causes rejection; this v1 will not split or truncate
/// it. Protected regions and overlapping edits are rejected before mutation.
/// On success, the original Score remains untouched and the caller receives
/// a candidate plus a digest-scoped change receipt.
pub fn apply_score_edit_plan(
    source: &Score,
    plan: &ScoreEditPlanV1,
    expected_input_score_sha256: &str,
) -> Result<ScoreEditCandidateV1, Vec<ScoreEditIssueV1>> {
    let mut issues = Vec::new();

    if plan.schema != SCORE_EDIT_SCHEMA_V1 {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::InvalidSchema,
            None,
            None,
            None,
            "unsupported score-edit schema",
        );
    }
    if !valid_digest(&plan.input_score_sha256) {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::InvalidInputDigest,
            None,
            None,
            None,
            "input_score_sha256 must be 64 lowercase hexadecimal characters",
        );
    }
    if !valid_digest(expected_input_score_sha256) {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::InvalidExpectedInputDigest,
            None,
            None,
            None,
            "expected input digest must come from the trusted score-provenance layer",
        );
    } else if valid_digest(&plan.input_score_sha256)
        && plan.input_score_sha256 != expected_input_score_sha256
    {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::InputDigestMismatch,
            None,
            None,
            None,
            "edit plan is bound to a different source Score",
        );
    }
    if source.notes.is_empty() {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::EmptyScore,
            None,
            None,
            None,
            "cannot edit an empty Score",
        );
    } else if !source.has_part_identity() {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::MissingPartIdentity,
            None,
            None,
            None,
            "every note must have a generator-authored PartId; role-based inference is forbidden",
        );
    }
    if plan.operations.len() > MAX_SCORE_EDIT_OPERATIONS_V1
        || plan.protected_windows.len() > MAX_PROTECTED_WINDOWS_V1
    {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::ResourceLimitExceeded,
            None,
            None,
            None,
            "operation or protected-window count exceeds the v1 resource limit",
        );
        return Err(issues);
    }

    let known_parts: std::collections::BTreeSet<PartId> = source.parts().into_iter().collect();
    let Some(score_end) = Moment::from_duration(source.total_beats) else {
        push_issue(
            &mut issues,
            ScoreEditIssueCodeV1::InvalidTime,
            None,
            None,
            None,
            "source Score total_beats is not a bounded non-negative rational",
        );
        return Err(issues);
    };

    let mut operation_ids = std::collections::BTreeSet::new();
    let mut prepared = Vec::new();

    for operation in &plan.operations {
        if operation.id.trim().is_empty() {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::InvalidOperationId,
                Some(&operation.id),
                None,
                None,
                "operation id must be non-empty",
            );
        } else if !operation_ids.insert(operation.id.as_str()) {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::DuplicateOperationId,
                Some(&operation.id),
                None,
                None,
                "operation id must be unique",
            );
        }
        if operation.part_id == PartId::UNASSIGNED.0 {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::InvalidPartId,
                Some(&operation.id),
                Some(operation.part_id),
                None,
                "edit target requires a generator-authored PartId",
            );
            continue;
        }
        let part = PartId(operation.part_id);
        if !known_parts.contains(&part) {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::UnknownScorePart,
                Some(&operation.id),
                Some(operation.part_id),
                None,
                "target PartId is not present in the exact source Score",
            );
            continue;
        }
        if operation.numerator == 0
            || operation.denominator == 0
            || operation.numerator > MAX_VELOCITY_SCALE_COMPONENT
            || operation.denominator > MAX_VELOCITY_SCALE_COMPONENT
        {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::InvalidVelocityScale,
                Some(&operation.id),
                Some(operation.part_id),
                None,
                "velocity scale numerator and denominator must be in 1..=64",
            );
            continue;
        }
        let (start, end) = match validate_window(operation.start, operation.end, score_end) {
            Ok(window) => window,
            Err((code, message)) => {
                push_issue(
                    &mut issues,
                    code,
                    Some(&operation.id),
                    Some(operation.part_id),
                    None,
                    message,
                );
                continue;
            }
        };

        let mut note_indices = Vec::new();
        for (index, note) in source.notes.iter().enumerate() {
            if note.part != part {
                continue;
            }
            let (Some(onset), Some(duration)) = (
                Moment::from_duration(note.onset),
                Moment::from_duration(note.duration),
            ) else {
                push_issue(
                    &mut issues,
                    ScoreEditIssueCodeV1::InvalidSourceVelocity,
                    Some(&operation.id),
                    Some(operation.part_id),
                    Some(index),
                    "source note time is not representable as a bounded non-negative rational",
                );
                continue;
            };
            let Some(note_end) = onset.checked_add(duration) else {
                push_issue(
                    &mut issues,
                    ScoreEditIssueCodeV1::InvalidSourceVelocity,
                    Some(&operation.id),
                    Some(operation.part_id),
                    Some(index),
                    "source note end exceeds bounded rational arithmetic",
                );
                continue;
            };
            if intersects(onset, note_end, start, end) {
                if start.less_or_equal(onset) && note_end.less_or_equal(end) {
                    let scaled = note.velocity * f32::from(operation.numerator)
                        / f32::from(operation.denominator);
                    if !note.velocity.is_finite()
                        || !(0.0..=1.0).contains(&note.velocity)
                        || !scaled.is_finite()
                        || !(0.0..=1.0).contains(&scaled)
                    {
                        push_issue(
                            &mut issues,
                            ScoreEditIssueCodeV1::VelocityOutOfRange,
                            Some(&operation.id),
                            Some(operation.part_id),
                            Some(index),
                            "velocity scaling would create a non-finite or out-of-range value; clamping is forbidden",
                        );
                    } else {
                        note_indices.push(index);
                    }
                } else {
                    push_issue(
                        &mut issues,
                        ScoreEditIssueCodeV1::PartialNoteBoundary,
                        Some(&operation.id),
                        Some(operation.part_id),
                        Some(index),
                        "a sounding note crosses the edit-window boundary; v1 refuses to split it",
                    );
                }
            }
        }
        if note_indices.is_empty() {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::EmptyTargetSelection,
                Some(&operation.id),
                Some(operation.part_id),
                None,
                "operation selects no fully contained source notes",
            );
        }
        prepared.push(PreparedOperation {
            operation,
            start,
            end,
            note_indices,
        });
    }

    let mut protected_ids = std::collections::BTreeSet::new();
    let mut protected_windows = Vec::new();
    for protected in &plan.protected_windows {
        if protected.id.trim().is_empty() {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::InvalidProtectedWindowId,
                None,
                Some(protected.part_id),
                None,
                "protected-window id must be non-empty",
            );
        } else if !protected_ids.insert(protected.id.as_str()) {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::DuplicateProtectedWindowId,
                None,
                Some(protected.part_id),
                None,
                "protected-window id must be unique",
            );
        }
        if protected.part_id == PartId::UNASSIGNED.0 {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::InvalidPartId,
                None,
                Some(protected.part_id),
                None,
                "protected window requires a generator-authored PartId",
            );
            continue;
        }
        if !known_parts.contains(&PartId(protected.part_id)) {
            push_issue(
                &mut issues,
                ScoreEditIssueCodeV1::UnknownScorePart,
                None,
                Some(protected.part_id),
                None,
                "protected PartId is not present in the exact source Score",
            );
            continue;
        }
        match validate_window(protected.start, protected.end, score_end) {
            Ok((start, end)) => protected_windows.push((protected, start, end)),
            Err((code, message)) => push_issue(
                &mut issues,
                code,
                None,
                Some(protected.part_id),
                None,
                message,
            ),
        }
    }

    for (index, left) in prepared.iter().enumerate() {
        for right in prepared.iter().skip(index + 1) {
            if left.operation.part_id == right.operation.part_id
                && intersects(left.start, left.end, right.start, right.end)
            {
                push_issue(
                    &mut issues,
                    ScoreEditIssueCodeV1::OverlappingOperations,
                    Some(&right.operation.id),
                    Some(right.operation.part_id),
                    None,
                    "overlapping operations on the same part are rejected in v1",
                );
            }
        }
        for (protected, protected_start, protected_end) in &protected_windows {
            if left.operation.part_id == protected.part_id
                && intersects(left.start, left.end, *protected_start, *protected_end)
            {
                push_issue(
                    &mut issues,
                    ScoreEditIssueCodeV1::ProtectedScopeOverlap,
                    Some(&left.operation.id),
                    Some(left.operation.part_id),
                    None,
                    format!("operation overlaps protected window {}", protected.id),
                );
            }
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let mut candidate = source.clone();
    let mut changed_note_indices = Vec::new();
    for item in &prepared {
        for &index in &item.note_indices {
            let note = &mut candidate.notes[index];
            note.velocity = note.velocity * f32::from(item.operation.numerator)
                / f32::from(item.operation.denominator);
            changed_note_indices.push(index);
        }
    }
    changed_note_indices.sort_unstable();
    changed_note_indices.dedup();

    let receipt = ScoreEditReceiptV1 {
        schema: SCORE_EDIT_SCHEMA_V1.into(),
        input_score_sha256: plan.input_score_sha256.clone(),
        operation_ids: plan.operations.iter().map(|operation| operation.id.clone()).collect(),
        changed_note_indices,
        protected_window_ids: plan.protected_windows.iter().map(|window| window.id.clone()).collect(),
    };

    Ok(ScoreEditCandidateV1 { candidate, receipt })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::score::{Emphasis, ScoreNote, VoiceRole};

    fn source_score() -> Score {
        let key = Key::major(PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        for (part, midi, onset, velocity) in [
            (10u16, 60u8, 0i64, 0.8f32),
            (10, 64, 1, 0.4),
            (10, 67, 2, 0.7),
            (20, 48, 0, 0.6),
        ] {
            score.push(ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity,
                role: if part == 20 { VoiceRole::Bass } else { VoiceRole::Melody },
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn digest() -> String {
        "a".repeat(64)
    }

    fn operation(id: &str, start: i64, end: i64) -> ScalePartVelocityV1 {
        ScalePartVelocityV1 {
            id: id.into(),
            part_id: 10,
            start: EditBeatV1 { numerator: start, denominator: 1 },
            end: EditBeatV1 { numerator: end, denominator: 1 },
            numerator: 1,
            denominator: 2,
        }
    }

    fn plan(operations: Vec<ScalePartVelocityV1>) -> ScoreEditPlanV1 {
        ScoreEditPlanV1 {
            schema: SCORE_EDIT_SCHEMA_V1.into(),
            input_score_sha256: digest(),
            operations,
            protected_windows: Vec::new(),
        }
    }

    #[test]
    fn applies_velocity_edit_only_to_target_part_and_window() {
        let source = source_score();
        let original = source.clone();
        let result = apply_score_edit_plan(&source, &plan(vec![operation("soften-answer", 0, 2)]), &digest())
            .expect("valid local edit");
        assert_eq!(source, original, "preview must not mutate the source Score");
        assert_eq!(result.candidate.notes[0].velocity, 0.4);
        assert_eq!(result.candidate.notes[1].velocity, 0.2);
        assert_eq!(result.candidate.notes[2].velocity, 0.7);
        assert_eq!(result.candidate.notes[3].velocity, 0.6);
        assert_eq!(result.receipt.changed_note_indices, vec![0, 1]);
    }

    #[test]
    fn rejects_a_note_that_crosses_the_edit_window_boundary() {
        let source = source_score();
        let result = apply_score_edit_plan(&source, &plan(vec![operation("partial", 1, 2)]), &digest());
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|issue| {
            issue.code == ScoreEditIssueCodeV1::PartialNoteBoundary
        }));
    }

    #[test]
    fn rejects_a_protected_region_overlap() {
        let mut edit = plan(vec![operation("edit", 0, 2)]);
        edit.protected_windows.push(ProtectedPartWindowV1 {
            id: "opening-protected".into(),
            part_id: 10,
            start: EditBeatV1 { numerator: 0, denominator: 1 },
            end: EditBeatV1 { numerator: 1, denominator: 1 },
        });
        let result = apply_score_edit_plan(&source_score(), &edit, &digest());
        assert!(result.unwrap_err().iter().any(|issue| {
            issue.code == ScoreEditIssueCodeV1::ProtectedScopeOverlap
        }));
    }

    #[test]
    fn rejects_overlapping_operations_on_the_same_part() {
        let result = apply_score_edit_plan(
            &source_score(),
            &plan(vec![operation("first", 0, 2), operation("second", 1, 3)]),
            &digest(),
        );
        assert!(result.unwrap_err().iter().any(|issue| {
            issue.code == ScoreEditIssueCodeV1::OverlappingOperations
        }));
    }

    #[test]
    fn rejects_velocity_clipping_instead_of_silently_clamping() {
        let mut edit_op = operation("louder", 0, 1);
        edit_op.numerator = 2;
        edit_op.denominator = 1;
        let result = apply_score_edit_plan(&source_score(), &plan(vec![edit_op]), &digest());
        assert!(result.unwrap_err().iter().any(|issue| {
            issue.code == ScoreEditIssueCodeV1::VelocityOutOfRange
        }));
    }

    #[test]
    fn rejects_an_edit_plan_for_a_different_score_digest() {
        let result = apply_score_edit_plan(&source_score(), &plan(vec![operation("edit", 0, 1)]), &"b".repeat(64));
        assert!(result.unwrap_err().iter().any(|issue| {
            issue.code == ScoreEditIssueCodeV1::InputDigestMismatch
        }));
    }

    #[test]
    fn rejects_an_unassigned_or_unknown_part_target() {
        let mut unassigned = operation("unassigned", 0, 1);
        unassigned.part_id = PartId::UNASSIGNED.0;
        let mut unknown = operation("unknown", 0, 1);
        unknown.part_id = 99;
        let result = apply_score_edit_plan(&source_score(), &plan(vec![unassigned, unknown]), &digest());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|issue| issue.code == ScoreEditIssueCodeV1::InvalidPartId));
        assert!(errors.iter().any(|issue| issue.code == ScoreEditIssueCodeV1::UnknownScorePart));
    }
}
