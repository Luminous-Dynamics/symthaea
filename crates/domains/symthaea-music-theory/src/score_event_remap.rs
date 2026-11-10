// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit source-to-candidate event provenance for symbolic score edits.
//!
//! Event ordinals are scoped to exact source/candidate Score digests. This
//! module never reconstructs correspondence from pitch, onset, role, register
//! or acoustic similarity. The transformation must emit the map as it builds
//! the candidate. Graph validity does not establish musical correctness.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::score::{PartId, Score, ScoreNote};

pub const SCORE_EVENT_REMAP_SCHEMA_V1: &str = "score-event-remap-v1";
pub const SCORE_EVENT_REMAP_DIGEST_SCHEMA_V1: &str = "muse-score-serde-json-sha256-v1";
pub const MAX_SCORE_EVENT_REMAP_EVENTS_V1: usize = 1_000_000;

/// One source event's disposition in the candidate Score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ScoreEventDispositionV1 {
    Retained { candidate_index: usize },
    Modified {
        candidate_index: usize,
        changed_fields: Vec<ScoreEventFieldV1>,
    },
    Split { candidate_indices: Vec<usize> },
    Merged {
        candidate_index: usize,
        merge_group_id: String,
    },
    Deleted { reason: String },
}

/// Fields that can change while the persistent PartId remains unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEventFieldV1 {
    Pitch,
    Onset,
    Duration,
    Velocity,
    VoiceRole,
    Emphasis,
    SectionIntensity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventRemapEntryV1 {
    pub source_index: usize,
    pub disposition: ScoreEventDispositionV1,
}

/// Candidate event with no source-event parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventInsertionV1 {
    pub candidate_index: usize,
    pub part_id: u16,
    pub operation_id: String,
}

/// Complete event accounting for one frozen source-to-candidate transformation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventRemapV1 {
    pub schema: String,
    pub digest_schema: String,
    pub source_score_sha256: String,
    pub candidate_score_sha256: String,
    pub transformation_sha256: String,
    pub source_event_count: usize,
    pub candidate_event_count: usize,
    /// Exactly one disposition per source event, in increasing source order.
    pub source_entries: Vec<ScoreEventRemapEntryV1>,
    /// Every target with no source parent must be declared here exactly once.
    pub insertions: Vec<ScoreEventInsertionV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEventRemapIssueCodeV1 {
    InvalidSchema,
    InvalidDigestSchema,
    InvalidSourceDigest,
    InvalidCandidateDigest,
    InvalidTransformationDigest,
    InvalidExpectedSourceDigest,
    InvalidExpectedCandidateDigest,
    InvalidExpectedTransformationDigest,
    SourceDigestMismatch,
    CandidateDigestMismatch,
    TransformationDigestMismatch,
    ResourceLimitExceeded,
    SourceEventCountMismatch,
    CandidateEventCountMismatch,
    MissingSourceDisposition,
    SourceIndexOutOfRange,
    DuplicateSourceDisposition,
    NonCanonicalSourceOrder,
    CandidateIndexOutOfRange,
    SplitMustHaveMultipleTargets,
    DuplicateCandidateReference,
    CandidateReferenceConflict,
    MissingCandidateDisposition,
    DuplicateInsertion,
    InvalidInsertionOperationId,
    InvalidInsertionPartId,
    InsertionPartMismatch,
    InvalidDeletionReason,
    InvalidMergeGroupId,
    MergeGroupTargetMismatch,
    MergeGroupTooSmall,
    MergeSourcePartMismatch,
    PartIdentityChanged,
    RetainedEventChanged,
    ModifiedFieldDeclarationInvalid,
    ModifiedFieldMismatch,
    ModifiedEventHasNoChanges,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventRemapIssueV1 {
    pub code: ScoreEventRemapIssueCodeV1,
    pub source_index: Option<usize>,
    pub candidate_index: Option<usize>,
    pub merge_group_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventRemapReportV1 {
    pub valid: bool,
    pub source_event_count: usize,
    pub candidate_event_count: usize,
    pub retained_count: usize,
    pub modified_count: usize,
    pub split_source_count: usize,
    pub merged_source_count: usize,
    pub deleted_count: usize,
    pub inserted_count: usize,
    pub issues: Vec<ScoreEventRemapIssueV1>,
}

impl ScoreEventRemapReportV1 {
    fn new(source_event_count: usize, candidate_event_count: usize) -> Self {
        Self {
            valid: true,
            source_event_count,
            candidate_event_count,
            retained_count: 0,
            modified_count: 0,
            split_source_count: 0,
            merged_source_count: 0,
            deleted_count: 0,
            inserted_count: 0,
            issues: Vec::new(),
        }
    }

    fn push(
        &mut self,
        code: ScoreEventRemapIssueCodeV1,
        source_index: Option<usize>,
        candidate_index: Option<usize>,
        merge_group_id: Option<&str>,
        message: impl Into<String>,
    ) {
        self.valid = false;
        self.issues.push(ScoreEventRemapIssueV1 {
            code,
            source_index,
            candidate_index,
            merge_group_id: merge_group_id.map(str::to_owned),
            message: message.into(),
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CandidateOwner {
    Direct,
    Merge(String),
    Inserted,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn actual_changed_fields(source: &ScoreNote, target: &ScoreNote) -> BTreeSet<ScoreEventFieldV1> {
    let mut changed = BTreeSet::new();
    if source.pitch != target.pitch {
        changed.insert(ScoreEventFieldV1::Pitch);
    }
    if source.onset != target.onset {
        changed.insert(ScoreEventFieldV1::Onset);
    }
    if source.duration != target.duration {
        changed.insert(ScoreEventFieldV1::Duration);
    }
    if source.velocity.to_bits() != target.velocity.to_bits() {
        changed.insert(ScoreEventFieldV1::Velocity);
    }
    if source.role != target.role {
        changed.insert(ScoreEventFieldV1::VoiceRole);
    }
    if source.emphasis != target.emphasis {
        changed.insert(ScoreEventFieldV1::Emphasis);
    }
    if source.section_intensity.to_bits() != target.section_intensity.to_bits() {
        changed.insert(ScoreEventFieldV1::SectionIntensity);
    }
    changed
}

/// Validate an event remap against the exact source and candidate scores.
/// Expected digests must be computed by trusted score-demand/provenance code;
/// this module deliberately does not implement a second hashing algorithm.
pub fn validate_score_event_remap(
    source: &Score,
    candidate: &Score,
    remap: &ScoreEventRemapV1,
    expected_source_score_sha256: &str,
    expected_candidate_score_sha256: &str,
    expected_transformation_sha256: &str,
) -> ScoreEventRemapReportV1 {
    let mut report = ScoreEventRemapReportV1::new(source.notes.len(), candidate.notes.len());

    if remap.schema != SCORE_EVENT_REMAP_SCHEMA_V1 {
        report.push(
            ScoreEventRemapIssueCodeV1::InvalidSchema,
            None, None, None, "unsupported score-event-remap schema",
        );
    }
    if remap.digest_schema != SCORE_EVENT_REMAP_DIGEST_SCHEMA_V1 {
        report.push(
            ScoreEventRemapIssueCodeV1::InvalidDigestSchema,
            None, None, None, "remap must use the canonical Muse serialized-Score SHA-256 schema",
        );
    }

    let digest_bindings = [
        (
            remap.source_score_sha256.as_str(),
            expected_source_score_sha256,
            ScoreEventRemapIssueCodeV1::InvalidSourceDigest,
            ScoreEventRemapIssueCodeV1::InvalidExpectedSourceDigest,
            ScoreEventRemapIssueCodeV1::SourceDigestMismatch,
            "source Score",
        ),
        (
            remap.candidate_score_sha256.as_str(),
            expected_candidate_score_sha256,
            ScoreEventRemapIssueCodeV1::InvalidCandidateDigest,
            ScoreEventRemapIssueCodeV1::InvalidExpectedCandidateDigest,
            ScoreEventRemapIssueCodeV1::CandidateDigestMismatch,
            "candidate Score",
        ),
        (
            remap.transformation_sha256.as_str(),
            expected_transformation_sha256,
            ScoreEventRemapIssueCodeV1::InvalidTransformationDigest,
            ScoreEventRemapIssueCodeV1::InvalidExpectedTransformationDigest,
            ScoreEventRemapIssueCodeV1::TransformationDigestMismatch,
            "transformation plan",
        ),
    ];
    for (actual, expected, invalid, invalid_expected, mismatch, label) in digest_bindings {
        if !valid_digest(actual) {
            report.push(invalid, None, None, None, format!("{label} digest must be 64 lowercase hexadecimal characters"));
        }
        if !valid_digest(expected) {
            report.push(invalid_expected, None, None, None, format!("expected {label} digest must come from trusted provenance"));
        } else if valid_digest(actual) && actual != expected {
            report.push(mismatch, None, None, None, format!("remap is bound to a different {label} digest"));
        }
    }

    if remap.source_event_count != source.notes.len() {
        report.push(
            ScoreEventRemapIssueCodeV1::SourceEventCountMismatch,
            None, None, None, "declared source count differs from the source Score",
        );
    }
    if remap.candidate_event_count != candidate.notes.len() {
        report.push(
            ScoreEventRemapIssueCodeV1::CandidateEventCountMismatch,
            None, None, None, "declared candidate count differs from the candidate Score",
        );
    }
    if source.notes.len() > MAX_SCORE_EVENT_REMAP_EVENTS_V1
        || candidate.notes.len() > MAX_SCORE_EVENT_REMAP_EVENTS_V1
        || remap.source_entries.len() > MAX_SCORE_EVENT_REMAP_EVENTS_V1
        || remap.insertions.len() > MAX_SCORE_EVENT_REMAP_EVENTS_V1
    {
        report.push(
            ScoreEventRemapIssueCodeV1::ResourceLimitExceeded,
            None, None, None, "source, candidate or remap count exceeds the v1 resource limit",
        );
        return report;
    }
    if remap.source_entries.len() != source.notes.len() {
        report.push(
            ScoreEventRemapIssueCodeV1::MissingSourceDisposition,
            None, None, None, "there must be exactly one disposition entry for every source event",
        );
    }

    let mut source_seen = vec![false; source.notes.len()];
    let mut candidate_owners = vec![Vec::<CandidateOwner>::new(); candidate.notes.len()];
    let mut merge_groups = BTreeMap::<String, (usize, BTreeSet<usize>, u16)>::new();
    let mut last_source = None::<usize>;

    for entry in &remap.source_entries {
        if entry.source_index >= source.notes.len() {
            report.push(
                ScoreEventRemapIssueCodeV1::SourceIndexOutOfRange,
                Some(entry.source_index), None, None, "source ordinal is outside the frozen source Score",
            );
            continue;
        }
        if source_seen[entry.source_index] {
            report.push(
                ScoreEventRemapIssueCodeV1::DuplicateSourceDisposition,
                Some(entry.source_index), None, None, "source event has more than one disposition",
            );
        }
        source_seen[entry.source_index] = true;
        if let Some(previous) = last_source {
            if entry.source_index <= previous {
                report.push(
                    ScoreEventRemapIssueCodeV1::NonCanonicalSourceOrder,
                    Some(entry.source_index), None, None,
                    "source entries must be strictly increasing and unique",
                );
            }
        }
        last_source = Some(entry.source_index);

        let source_note = &source.notes[entry.source_index];
        match &entry.disposition {
            ScoreEventDispositionV1::Retained { candidate_index } => {
                report.retained_count += 1;
                if let Some(target) = candidate.notes.get(*candidate_index) {
                    if source_note.part != target.part {
                        report.push(
                            ScoreEventRemapIssueCodeV1::PartIdentityChanged,
                            Some(entry.source_index), Some(*candidate_index), None,
                            "retained event cannot change its persistent PartId",
                        );
                    }
                    if source_note != target {
                        report.push(
                            ScoreEventRemapIssueCodeV1::RetainedEventChanged,
                            Some(entry.source_index), Some(*candidate_index), None,
                            "retained disposition requires exact ScoreNote equality",
                        );
                    }
                    candidate_owners[*candidate_index].push(CandidateOwner::Direct);
                } else {
                    report.push(
                        ScoreEventRemapIssueCodeV1::CandidateIndexOutOfRange,
                        Some(entry.source_index), Some(*candidate_index), None,
                        "retained target ordinal is outside the candidate Score",
                    );
                }
            }
            ScoreEventDispositionV1::Modified { candidate_index, changed_fields } => {
                report.modified_count += 1;
                if changed_fields.is_empty() {
                    report.push(
                        ScoreEventRemapIssueCodeV1::ModifiedEventHasNoChanges,
                        Some(entry.source_index), Some(*candidate_index), None,
                        "modified disposition must declare at least one changed field",
                    );
                }
                let declared: BTreeSet<ScoreEventFieldV1> = changed_fields.iter().copied().collect();
                if declared.len() != changed_fields.len() {
                    report.push(
                        ScoreEventRemapIssueCodeV1::ModifiedFieldDeclarationInvalid,
                        Some(entry.source_index), Some(*candidate_index), None,
                        "changed field declaration contains duplicates",
                    );
                }
                if let Some(target) = candidate.notes.get(*candidate_index) {
                    if source_note.part != target.part {
                        report.push(
                            ScoreEventRemapIssueCodeV1::PartIdentityChanged,
                            Some(entry.source_index), Some(*candidate_index), None,
                            "modified event cannot change its persistent PartId",
                        );
                    }
                    if actual_changed_fields(source_note, target) != declared {
                        report.push(
                            ScoreEventRemapIssueCodeV1::ModifiedFieldMismatch,
                            Some(entry.source_index), Some(*candidate_index), None,
                            "declared changed fields do not exactly match the source/candidate ScoreNotes",
                        );
                    }
                    candidate_owners[*candidate_index].push(CandidateOwner::Direct);
                } else {
                    report.push(
                        ScoreEventRemapIssueCodeV1::CandidateIndexOutOfRange,
                        Some(entry.source_index), Some(*candidate_index), None,
                        "modified target ordinal is outside the candidate Score",
                    );
                }
            }
            ScoreEventDispositionV1::Split { candidate_indices } => {
                report.split_source_count += 1;
                if candidate_indices.len() < 2 {
                    report.push(
                        ScoreEventRemapIssueCodeV1::SplitMustHaveMultipleTargets,
                        Some(entry.source_index), None, None,
                        "split disposition must reference at least two candidate events",
                    );
                }
                let mut local = BTreeSet::new();
                for &target_index in candidate_indices {
                    if !local.insert(target_index) {
                        report.push(
                            ScoreEventRemapIssueCodeV1::DuplicateCandidateReference,
                            Some(entry.source_index), Some(target_index), None,
                            "one split cannot reference a candidate event twice",
                        );
                        continue;
                    }
                    if let Some(target) = candidate.notes.get(target_index) {
                        if source_note.part != target.part {
                            report.push(
                                ScoreEventRemapIssueCodeV1::PartIdentityChanged,
                                Some(entry.source_index), Some(target_index), None,
                                "split descendants must preserve the source PartId",
                            );
                        }
                        candidate_owners[target_index].push(CandidateOwner::Direct);
                    } else {
                        report.push(
                            ScoreEventRemapIssueCodeV1::CandidateIndexOutOfRange,
                            Some(entry.source_index), Some(target_index), None,
                            "split target ordinal is outside the candidate Score",
                        );
                    }
                }
            }
            ScoreEventDispositionV1::Merged { candidate_index, merge_group_id } => {
                report.merged_source_count += 1;
                if merge_group_id.trim().is_empty() {
                    report.push(
                        ScoreEventRemapIssueCodeV1::InvalidMergeGroupId,
                        Some(entry.source_index), Some(*candidate_index), Some(merge_group_id),
                        "merge group ID must be non-empty",
                    );
                }
                if let Some(target) = candidate.notes.get(*candidate_index) {
                    if source_note.part != target.part {
                        report.push(
                            ScoreEventRemapIssueCodeV1::PartIdentityChanged,
                            Some(entry.source_index), Some(*candidate_index), Some(merge_group_id),
                            "merged event must preserve the common source PartId",
                        );
                    }
                    let group = merge_groups.entry(merge_group_id.clone()).or_insert_with(|| {
                        (*candidate_index, BTreeSet::new(), source_note.part.0)
                    });
                    if group.0 != *candidate_index {
                        report.push(
                            ScoreEventRemapIssueCodeV1::MergeGroupTargetMismatch,
                            Some(entry.source_index), Some(*candidate_index), Some(merge_group_id),
                            "all sources in a merge group must map to one candidate event",
                        );
                    }
                    if group.2 != source_note.part.0 {
                        report.push(
                            ScoreEventRemapIssueCodeV1::MergeSourcePartMismatch,
                            Some(entry.source_index), Some(*candidate_index), Some(merge_group_id),
                            "source events in a merge group must share one persistent PartId",
                        );
                    }
                    group.1.insert(entry.source_index);
                    candidate_owners[*candidate_index].push(CandidateOwner::Merge(merge_group_id.clone()));
                } else {
                    report.push(
                        ScoreEventRemapIssueCodeV1::CandidateIndexOutOfRange,
                        Some(entry.source_index), Some(*candidate_index), Some(merge_group_id),
                        "merged target ordinal is outside the candidate Score",
                    );
                }
            }
            ScoreEventDispositionV1::Deleted { reason } => {
                report.deleted_count += 1;
                if reason.trim().is_empty() {
                    report.push(
                        ScoreEventRemapIssueCodeV1::InvalidDeletionReason,
                        Some(entry.source_index), None, None,
                        "deleted source event requires a non-empty reason",
                    );
                }
            }
        }
    }

    for (index, seen) in source_seen.iter().enumerate() {
        if !seen {
            report.push(
                ScoreEventRemapIssueCodeV1::MissingSourceDisposition,
                Some(index), None, None,
                "source event has no explicit disposition",
            );
        }
    }
    for (group_id, (_, sources, _)) in &merge_groups {
        if sources.len() < 2 {
            report.push(
                ScoreEventRemapIssueCodeV1::MergeGroupTooSmall,
                sources.iter().next().copied(), None, Some(group_id),
                "merge group must contain at least two distinct source events",
            );
        }
    }

    let mut insertion_seen = BTreeSet::new();
    for insertion in &remap.insertions {
        if insertion.candidate_index >= candidate.notes.len() {
            report.push(
                ScoreEventRemapIssueCodeV1::CandidateIndexOutOfRange,
                None, Some(insertion.candidate_index), None,
                "inserted event ordinal is outside the candidate Score",
            );
            continue;
        }
        if !insertion_seen.insert(insertion.candidate_index) {
            report.push(
                ScoreEventRemapIssueCodeV1::DuplicateInsertion,
                None, Some(insertion.candidate_index), None,
                "candidate event has more than one insertion declaration",
            );
        }
        if insertion.operation_id.trim().is_empty() {
            report.push(
                ScoreEventRemapIssueCodeV1::InvalidInsertionOperationId,
                None, Some(insertion.candidate_index), None,
                "inserted event requires a non-empty operation ID",
            );
        }
        if insertion.part_id == PartId::UNASSIGNED.0 {
            report.push(
                ScoreEventRemapIssueCodeV1::InvalidInsertionPartId,
                None, Some(insertion.candidate_index), None,
                "inserted event requires a generator-authored PartId",
            );
        }
        if candidate.notes[insertion.candidate_index].part.0 != insertion.part_id {
            report.push(
                ScoreEventRemapIssueCodeV1::InsertionPartMismatch,
                None, Some(insertion.candidate_index), None,
                "insertion PartId differs from the candidate event PartId",
            );
        }
        candidate_owners[insertion.candidate_index].push(CandidateOwner::Inserted);
        report.inserted_count += 1;
    }

    for (candidate_index, owners) in candidate_owners.iter().enumerate() {
        if owners.is_empty() {
            report.push(
                ScoreEventRemapIssueCodeV1::MissingCandidateDisposition,
                None, Some(candidate_index), None,
                "candidate event is neither mapped from a source nor declared inserted",
            );
            continue;
        }
        match owners.as_slice() {
            [CandidateOwner::Direct] | [CandidateOwner::Inserted] => {}
            [CandidateOwner::Merge(group_id), rest @ ..] => {
                let valid_merge = !rest.is_empty()
                    && rest.iter().all(|owner| {
                        matches!(owner, CandidateOwner::Merge(other) if other == group_id)
                    });
                if !valid_merge {
                    report.push(
                        ScoreEventRemapIssueCodeV1::CandidateReferenceConflict,
                        None, Some(candidate_index), Some(group_id),
                        "candidate event has conflicting direct, inserted or cross-group owners",
                    );
                }
            }
            _ => report.push(
                ScoreEventRemapIssueCodeV1::CandidateReferenceConflict,
                None, Some(candidate_index), None,
                "candidate event has multiple owners outside one declared merge group",
            ),
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::rhythm::Duration;
    use crate::score::{Emphasis, VoiceRole};

    fn note(part: u16, midi: u8, onset: i64, duration: i64, velocity: f32) -> ScoreNote {
        ScoreNote {
            part: PartId(part),
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, 1),
            duration: Duration::new(duration, 1),
            velocity,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn score(notes: Vec<ScoreNote>) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for note in notes {
            score.push(note);
        }
        score
    }

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn base_remap(source: &Score, candidate: &Score) -> ScoreEventRemapV1 {
        ScoreEventRemapV1 {
            schema: SCORE_EVENT_REMAP_SCHEMA_V1.into(),
            digest_schema: SCORE_EVENT_REMAP_DIGEST_SCHEMA_V1.into(),
            source_score_sha256: digest('a'),
            candidate_score_sha256: digest('b'),
            transformation_sha256: digest('c'),
            source_event_count: source.notes.len(),
            candidate_event_count: candidate.notes.len(),
            source_entries: Vec::new(),
            insertions: Vec::new(),
        }
    }

    fn validate(source: &Score, candidate: &Score, remap: &ScoreEventRemapV1) -> ScoreEventRemapReportV1 {
        validate_score_event_remap(
            source, candidate, remap, &digest('a'), &digest('b'), &digest('c'),
        )
    }

    #[test]
    fn accepts_complete_identity_ordinal_map() {
        let source = score(vec![note(1, 60, 0, 1, 0.7), note(2, 67, 0, 1, 0.5)]);
        let candidate = source.clone();
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![
            ScoreEventRemapEntryV1 {
                source_index: 0,
                disposition: ScoreEventDispositionV1::Retained { candidate_index: 0 },
            },
            ScoreEventRemapEntryV1 {
                source_index: 1,
                disposition: ScoreEventDispositionV1::Retained { candidate_index: 1 },
            },
        ];
        let report = validate(&source, &candidate, &remap);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.retained_count, 2);
    }

    #[test]
    fn accepts_exact_modified_velocity_mapping() {
        let source = score(vec![note(1, 60, 0, 1, 0.8)]);
        let candidate = score(vec![note(1, 60, 0, 1, 0.4)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Modified {
                candidate_index: 0,
                changed_fields: vec![ScoreEventFieldV1::Velocity],
            },
        }];
        let report = validate(&source, &candidate, &remap);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.modified_count, 1);
    }

    #[test]
    fn rejects_undeclared_or_false_modified_fields() {
        let source = score(vec![note(1, 60, 0, 1, 0.8)]);
        let candidate = score(vec![note(1, 62, 0, 1, 0.4)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Modified {
                candidate_index: 0,
                changed_fields: vec![ScoreEventFieldV1::Velocity],
            },
        }];
        let report = validate(&source, &candidate, &remap);
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::ModifiedFieldMismatch));
    }

    #[test]
    fn accepts_one_to_many_split_and_preserves_part_identity() {
        let source = score(vec![note(7, 60, 0, 2, 0.8)]);
        let candidate = score(vec![note(7, 60, 0, 1, 0.4), note(7, 62, 1, 1, 0.4)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Split { candidate_indices: vec![0, 1] },
        }];
        let report = validate(&source, &candidate, &remap);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.split_source_count, 1);
    }

    #[test]
    fn accepts_many_to_one_merge_with_one_group() {
        let source = score(vec![note(7, 60, 0, 1, 0.8), note(7, 60, 1, 1, 0.8)]);
        let candidate = score(vec![note(7, 60, 0, 2, 0.8)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![
            ScoreEventRemapEntryV1 {
                source_index: 0,
                disposition: ScoreEventDispositionV1::Merged {
                    candidate_index: 0, merge_group_id: "tie-1".into(),
                },
            },
            ScoreEventRemapEntryV1 {
                source_index: 1,
                disposition: ScoreEventDispositionV1::Merged {
                    candidate_index: 0, merge_group_id: "tie-1".into(),
                },
            },
        ];
        let report = validate(&source, &candidate, &remap);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.merged_source_count, 2);
    }

    #[test]
    fn accepts_explicit_deletion_and_inserted_candidate_event() {
        let source = score(vec![note(7, 60, 0, 1, 0.8)]);
        let candidate = score(vec![note(8, 72, 0, 1, 0.7)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Deleted {
                reason: "explicit motif removal".into(),
            },
        }];
        remap.insertions = vec![ScoreEventInsertionV1 {
            candidate_index: 0, part_id: 8, operation_id: "insert-new-line".into(),
        }];
        let report = validate(&source, &candidate, &remap);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.deleted_count, 1);
        assert_eq!(report.inserted_count, 1);
    }

    #[test]
    fn rejects_missing_source_or_candidate_dispositions() {
        let source = score(vec![note(1, 60, 0, 1, 0.8), note(1, 62, 1, 1, 0.8)]);
        let candidate = score(vec![note(1, 60, 0, 1, 0.8), note(1, 64, 1, 1, 0.8), note(1, 65, 2, 1, 0.8)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Retained { candidate_index: 0 },
        }];
        let report = validate(&source, &candidate, &remap);
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::MissingSourceDisposition));
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::MissingCandidateDisposition));
    }

    #[test]
    fn rejects_duplicate_or_conflicting_candidate_references() {
        let source = score(vec![note(1, 60, 0, 1, 0.8), note(1, 62, 1, 1, 0.8)]);
        let candidate = score(vec![note(1, 60, 0, 1, 0.8)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![
            ScoreEventRemapEntryV1 {
                source_index: 0,
                disposition: ScoreEventDispositionV1::Retained { candidate_index: 0 },
            },
            ScoreEventRemapEntryV1 {
                source_index: 1,
                disposition: ScoreEventDispositionV1::Retained { candidate_index: 0 },
            },
        ];
        let report = validate(&source, &candidate, &remap);
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::CandidateReferenceConflict));
    }

    #[test]
    fn rejects_part_identity_changes_and_cross_part_merges() {
        let source = score(vec![note(1, 60, 0, 1, 0.8)]);
        let candidate = score(vec![note(2, 60, 0, 1, 0.8)]);
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Modified {
                candidate_index: 0, changed_fields: vec![],
            },
        }];
        let report = validate(&source, &candidate, &remap);
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::PartIdentityChanged));
    }

    #[test]
    fn rejects_stale_source_candidate_or_transformation_digests() {
        let source = score(vec![note(1, 60, 0, 1, 0.8)]);
        let candidate = source.clone();
        let mut remap = base_remap(&source, &candidate);
        remap.source_entries = vec![ScoreEventRemapEntryV1 {
            source_index: 0,
            disposition: ScoreEventDispositionV1::Retained { candidate_index: 0 },
        }];
        let report = validate_score_event_remap(
            &source, &candidate, &remap, &digest('d'), &digest('e'), &digest('f'),
        );
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::SourceDigestMismatch));
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::CandidateDigestMismatch));
        assert!(report.issues.iter().any(|issue| issue.code == ScoreEventRemapIssueCodeV1::TransformationDigestMismatch));
    }
}
