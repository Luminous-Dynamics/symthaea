// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-bound lineage for persistent musical materials.
//!
//! A material family can have many instances, parts and time windows. Each
//! instance references exact note ordinals in one frozen Score. These ordinals
//! are digest-scoped, not persistent EventIds. Derivation edges are authored
//! claims; graph validity does not prove musical transformation correctness.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::rhythm::Duration;
use crate::score::{PartId, Score};

pub const MATERIAL_LINEAGE_SCHEMA_V1: &str = "material-lineage-v1";
pub const MAX_MATERIAL_INSTANCES_V1: usize = 4_096;
pub const MAX_MATERIAL_EDGES_V1: usize = 8_192;
pub const MAX_MATERIAL_EVENT_REFS_V1: usize = 1_000_000;
pub const MAX_MATERIAL_EVENTS_PER_INSTANCE_V1: usize = 65_536;
const MAX_BEAT_NUMERATOR: i128 = 1_000_000_000;
const MAX_BEAT_DENOMINATOR: i128 = 1_000_000;

/// Exact non-negative score-time boundary in beats. Windows are half-open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialBeatV1 {
    pub numerator: i64,
    pub denominator: i64,
}

/// One source-authored occurrence of a persistent musical material.
///
/// Event ordinals address the Score.notes array for this exact score digest.
/// They are not stable identifiers across score versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialInstanceV1 {
    pub instance_id: String,
    pub material_id: String,
    pub part_id: u16,
    pub start: MaterialBeatV1,
    pub end: MaterialBeatV1,
    /// Exact note ordinals in the frozen Score; strictly increasing.
    pub event_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialTransformKindV1 {
    Restatement,
    Transposition,
    Inversion,
    Retrograde,
    Augmentation,
    Diminution,
    Fragmentation,
    Ornamentation,
    Reharmonization,
    Revoicing,
    InstrumentHandoff,
    Doubling,
    CounterfactualVariant,
    Hybridization,
}

/// Provenance basis for a lineage edge. These declarations are not proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialEvidenceBasisV1 {
    SourceAuthoredIntent,
    MachineSuggestedUnverified,
    SymbolicallyChecked,
    IndependentlyHumanReviewed,
}

/// Directed derivation edge. Multiple parents are allowed for hybridization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLineageEdgeV1 {
    pub edge_id: String,
    pub source_instance_id: String,
    pub target_instance_id: String,
    pub transform: MaterialTransformKindV1,
    pub evidence_basis: MaterialEvidenceBasisV1,
    pub evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLineageTimelineV1 {
    pub schema: String,
    /// SHA-256 of the exact serialized Score, supplied by trusted provenance.
    pub score_sha256: String,
    pub instances: Vec<MaterialInstanceV1>,
    pub edges: Vec<MaterialLineageEdgeV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialLineageIssueCodeV1 {
    InvalidSchema,
    InvalidScoreDigest,
    InvalidExpectedScoreDigest,
    ScoreDigestMismatch,
    EmptyScore,
    MissingPartIdentity,
    ResourceLimitExceeded,
    InvalidInstanceId,
    DuplicateInstanceId,
    InvalidMaterialId,
    InvalidPartId,
    UnknownScorePart,
    InvalidTime,
    EmptyInterval,
    IntervalBeyondScore,
    EmptyEventSelection,
    TooManyEventsInInstance,
    TooManyTotalEventRefs,
    NonCanonicalEventOrder,
    DuplicateEventReference,
    EventIndexOutOfRange,
    EventPartMismatch,
    InvalidSourceEventTime,
    EventOutsideInstanceWindow,
    InvalidEdgeId,
    DuplicateEdgeId,
    InvalidEvidenceReference,
    UnknownSourceInstance,
    UnknownTargetInstance,
    SelfDerivation,
    LineageCycle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLineageIssueV1 {
    pub code: MaterialLineageIssueCodeV1,
    pub instance_id: Option<String>,
    pub edge_id: Option<String>,
    pub part_id: Option<u16>,
    pub event_index: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLineageReportV1 {
    pub valid: bool,
    pub instance_count: usize,
    pub edge_count: usize,
    pub unique_material_count: usize,
    pub referenced_event_count: usize,
    pub topological_instance_order: Vec<String>,
    pub issues: Vec<MaterialLineageIssueV1>,
}

impl MaterialLineageReportV1 {
    fn new(instance_count: usize, edge_count: usize) -> Self {
        Self {
            valid: true,
            instance_count,
            edge_count,
            unique_material_count: 0,
            referenced_event_count: 0,
            topological_instance_order: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn push(
        &mut self,
        code: MaterialLineageIssueCodeV1,
        instance_id: Option<&str>,
        edge_id: Option<&str>,
        part_id: Option<u16>,
        event_index: Option<usize>,
        message: impl Into<String>,
    ) {
        self.valid = false;
        self.issues.push(MaterialLineageIssueV1 {
            code,
            instance_id: instance_id.map(str::to_owned),
            edge_id: edge_id.map(str::to_owned),
            part_id,
            event_index,
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
            || den > MAX_BEAT_DENOMINATOR
            || num < 0
            || num > MAX_BEAT_NUMERATOR
        {
            return None;
        }
        Some(Self { num, den })
    }

    fn from_wire(value: MaterialBeatV1) -> Option<Self> {
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

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_window(
    start_value: MaterialBeatV1,
    end_value: MaterialBeatV1,
    score_end: Moment,
) -> Result<(Moment, Moment), (MaterialLineageIssueCodeV1, &'static str)> {
    let (Some(start), Some(end)) = (
        Moment::from_wire(start_value),
        Moment::from_wire(end_value),
    ) else {
        return Err((
            MaterialLineageIssueCodeV1::InvalidTime,
            "material boundaries require bounded non-negative rational beats",
        ));
    };
    if !start.less_than(end) {
        return Err((
            MaterialLineageIssueCodeV1::EmptyInterval,
            "material instance window must have positive duration",
        ));
    }
    if !end.less_or_equal(score_end) {
        return Err((
            MaterialLineageIssueCodeV1::IntervalBeyondScore,
            "material instance window extends beyond the exact source Score",
        ));
    }
    Ok((start, end))
}

/// Validate source-authored material instances, event references, derivation
/// references and DAG acyclicity against one frozen symbolic Score.
///
/// Lineage is never inferred from note similarity, VoiceRole, register, timing
/// or instrument labels. Musical correctness of a transformation remains a
/// separate semantic and/or human-review question.
pub fn validate_material_lineage(
    score: &Score,
    timeline: &MaterialLineageTimelineV1,
    expected_score_sha256: &str,
) -> MaterialLineageReportV1 {
    let mut report =
        MaterialLineageReportV1::new(timeline.instances.len(), timeline.edges.len());

    if timeline.schema != MATERIAL_LINEAGE_SCHEMA_V1 {
        report.push(
            MaterialLineageIssueCodeV1::InvalidSchema,
            None, None, None, None,
            "unsupported material-lineage schema",
        );
    }
    if !valid_digest(&timeline.score_sha256) {
        report.push(
            MaterialLineageIssueCodeV1::InvalidScoreDigest,
            None, None, None, None,
            "timeline score_sha256 must be 64 lowercase hexadecimal characters",
        );
    }
    if !valid_digest(expected_score_sha256) {
        report.push(
            MaterialLineageIssueCodeV1::InvalidExpectedScoreDigest,
            None, None, None, None,
            "expected Score digest must come from trusted score-demand/provenance",
        );
    } else if valid_digest(&timeline.score_sha256)
        && timeline.score_sha256 != expected_score_sha256
    {
        report.push(
            MaterialLineageIssueCodeV1::ScoreDigestMismatch,
            None, None, None, None,
            "material-lineage timeline is bound to a different symbolic Score",
        );
    }
    if score.notes.is_empty() {
        report.push(
            MaterialLineageIssueCodeV1::EmptyScore,
            None, None, None, None,
            "cannot validate material lineage against an empty Score",
        );
    } else if !score.has_part_identity() {
        report.push(
            MaterialLineageIssueCodeV1::MissingPartIdentity,
            None, None, None, None,
            "every referenced event must have a generator-authored PartId",
        );
    }
    if timeline.instances.len() > MAX_MATERIAL_INSTANCES_V1
        || timeline.edges.len() > MAX_MATERIAL_EDGES_V1
    {
        report.push(
            MaterialLineageIssueCodeV1::ResourceLimitExceeded,
            None, None, None, None,
            "material instance or edge count exceeds the v1 resource limit",
        );
        return report;
    }

    let Some(score_end) = Moment::from_duration(score.total_beats) else {
        report.push(
            MaterialLineageIssueCodeV1::InvalidTime,
            None, None, None, None,
            "source Score total_beats is not a bounded non-negative rational",
        );
        return report;
    };

    let known_parts: BTreeSet<PartId> = score.parts().into_iter().collect();
    let mut instance_by_id = BTreeMap::<String, usize>::new();
    let mut material_ids = BTreeSet::<String>::new();
    let mut total_event_refs = 0usize;

    for (instance_index, instance) in timeline.instances.iter().enumerate() {
        if instance.instance_id.trim().is_empty() {
            report.push(
                MaterialLineageIssueCodeV1::InvalidInstanceId,
                Some(&instance.instance_id), None, None, None,
                "instance_id must be non-empty",
            );
        } else if instance_by_id
            .insert(instance.instance_id.clone(), instance_index)
            .is_some()
        {
            report.push(
                MaterialLineageIssueCodeV1::DuplicateInstanceId,
                Some(&instance.instance_id), None, None, None,
                "instance_id must be unique within the timeline",
            );
        }
        if instance.material_id.trim().is_empty() {
            report.push(
                MaterialLineageIssueCodeV1::InvalidMaterialId,
                Some(&instance.instance_id), None, Some(instance.part_id), None,
                "material_id must be non-empty",
            );
        } else {
            material_ids.insert(instance.material_id.clone());
        }
        if instance.part_id == PartId::UNASSIGNED.0 {
            report.push(
                MaterialLineageIssueCodeV1::InvalidPartId,
                Some(&instance.instance_id), None, Some(instance.part_id), None,
                "material instance requires a generator-authored PartId",
            );
            continue;
        }
        if !known_parts.contains(&PartId(instance.part_id)) {
            report.push(
                MaterialLineageIssueCodeV1::UnknownScorePart,
                Some(&instance.instance_id), None, Some(instance.part_id), None,
                "material PartId is not present in the exact source Score",
            );
            continue;
        }
        if instance.event_indices.is_empty() {
            report.push(
                MaterialLineageIssueCodeV1::EmptyEventSelection,
                Some(&instance.instance_id), None, Some(instance.part_id), None,
                "material instance must reference at least one source Score event",
            );
        }
        if instance.event_indices.len() > MAX_MATERIAL_EVENTS_PER_INSTANCE_V1 {
            report.push(
                MaterialLineageIssueCodeV1::TooManyEventsInInstance,
                Some(&instance.instance_id), None, Some(instance.part_id), None,
                "material event count exceeds the per-instance resource limit",
            );
            continue;
        }
        total_event_refs = match total_event_refs.checked_add(instance.event_indices.len()) {
            Some(total) => total,
            None => {
                report.push(
                    MaterialLineageIssueCodeV1::TooManyTotalEventRefs,
                    Some(&instance.instance_id), None, Some(instance.part_id), None,
                    "total event reference count overflowed",
                );
                return report;
            }
        };
        if total_event_refs > MAX_MATERIAL_EVENT_REFS_V1 {
            report.push(
                MaterialLineageIssueCodeV1::TooManyTotalEventRefs,
                Some(&instance.instance_id), None, Some(instance.part_id), None,
                "total material event references exceed the v1 resource limit",
            );
            return report;
        }

        let (start, end) = match validate_window(instance.start, instance.end, score_end) {
            Ok(window) => window,
            Err((code, message)) => {
                report.push(
                    code, Some(&instance.instance_id), None, Some(instance.part_id), None,
                    message,
                );
                continue;
            }
        };

        let mut previous_index = None;
        for &event_index in &instance.event_indices {
            if let Some(previous) = previous_index {
                if event_index <= previous {
                    report.push(
                        if event_index == previous {
                            MaterialLineageIssueCodeV1::DuplicateEventReference
                        } else {
                            MaterialLineageIssueCodeV1::NonCanonicalEventOrder
                        },
                        Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                        "event indices must be unique and strictly increasing",
                    );
                }
            }
            previous_index = Some(event_index);

            let Some(note) = score.notes.get(event_index) else {
                report.push(
                    MaterialLineageIssueCodeV1::EventIndexOutOfRange,
                    Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                    "event index does not exist in the exact source Score",
                );
                continue;
            };
            if note.part.0 != instance.part_id {
                report.push(
                    MaterialLineageIssueCodeV1::EventPartMismatch,
                    Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                    "referenced event does not belong to the instance's declared PartId",
                );
                continue;
            }
            if note.duration.num() <= 0 {
                report.push(
                    MaterialLineageIssueCodeV1::InvalidSourceEventTime,
                    Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                    "referenced Score event must have a positive duration",
                );
                continue;
            }
            let (Some(onset), Some(duration)) = (
                Moment::from_duration(note.onset),
                Moment::from_duration(note.duration),
            ) else {
                report.push(
                    MaterialLineageIssueCodeV1::InvalidTime,
                    Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                    "referenced Score event has invalid rational timing",
                );
                continue;
            };
            let Some(note_end) = onset.checked_add(duration) else {
                report.push(
                    MaterialLineageIssueCodeV1::InvalidTime,
                    Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                    "referenced event end exceeds bounded rational arithmetic",
                );
                continue;
            };
            if onset.less_than(start) || note_end.greater_than(end) {
                report.push(
                    MaterialLineageIssueCodeV1::EventOutsideInstanceWindow,
                    Some(&instance.instance_id), None, Some(instance.part_id), Some(event_index),
                    "referenced event is not fully contained in the instance window",
                );
            }
            report.referenced_event_count = report.referenced_event_count.saturating_add(1);
        }
    }

    report.unique_material_count = material_ids.len();

    let mut edge_ids = BTreeSet::<String>::new();
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    let mut indegree = BTreeMap::<String, usize>::new();
    for instance_id in instance_by_id.keys() {
        adjacency.entry(instance_id.clone()).or_default();
        indegree.insert(instance_id.clone(), 0);
    }

    for edge in &timeline.edges {
        if edge.edge_id.trim().is_empty() {
            report.push(
                MaterialLineageIssueCodeV1::InvalidEdgeId,
                None, Some(&edge.edge_id), None, None,
                "edge_id must be non-empty",
            );
        } else if !edge_ids.insert(edge.edge_id.clone()) {
            report.push(
                MaterialLineageIssueCodeV1::DuplicateEdgeId,
                None, Some(&edge.edge_id), None, None,
                "edge_id must be unique within the timeline",
            );
        }
        if edge.evidence_ref.as_ref().is_some_and(|value| value.trim().is_empty()) {
            report.push(
                MaterialLineageIssueCodeV1::InvalidEvidenceReference,
                None, Some(&edge.edge_id), None, None,
                "evidence_ref must be absent or non-empty",
            );
        }
        let source_exists = instance_by_id.contains_key(&edge.source_instance_id);
        let target_exists = instance_by_id.contains_key(&edge.target_instance_id);
        if !source_exists {
            report.push(
                MaterialLineageIssueCodeV1::UnknownSourceInstance,
                None, Some(&edge.edge_id), None, None,
                "source_instance_id does not exist in this timeline",
            );
        }
        if !target_exists {
            report.push(
                MaterialLineageIssueCodeV1::UnknownTargetInstance,
                None, Some(&edge.edge_id), None, None,
                "target_instance_id does not exist in this timeline",
            );
        }
        if edge.source_instance_id == edge.target_instance_id {
            report.push(
                MaterialLineageIssueCodeV1::SelfDerivation,
                None, Some(&edge.edge_id), None, None,
                "a material instance cannot derive from itself",
            );
        }
        if source_exists && target_exists && edge.source_instance_id != edge.target_instance_id {
            let inserted = adjacency
                .entry(edge.source_instance_id.clone())
                .or_default()
                .insert(edge.target_instance_id.clone());
            if inserted {
                if let Some(degree) = indegree.get_mut(&edge.target_instance_id) {
                    *degree = degree.saturating_add(1);
                }
            }
        }
    }

    // Stable Kahn topological sort. Multiple authored claims for the same
    // source-target pair do not inflate indegree.
    let mut ready = BTreeSet::<String>::new();
    for (instance_id, degree) in &indegree {
        if *degree == 0 {
            ready.insert(instance_id.clone());
        }
    }
    while let Some(instance_id) = ready.pop_first() {
        report.topological_instance_order.push(instance_id.clone());
        if let Some(children) = adjacency.get(&instance_id) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }
    if report.topological_instance_order.len() != instance_by_id.len() {
        report.push(
            MaterialLineageIssueCodeV1::LineageCycle,
            None, None, None, None,
            "material derivation graph must be acyclic in v1",
        );
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
        for (part, midi, onset) in [
            (1u16, 60u8, 0i64),
            (1, 62, 1),
            (2, 67, 3),
            (3, 72, 4),
        ] {
            score.push(ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
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

    fn instance(
        id: &str,
        material: &str,
        part: u16,
        start: i64,
        end: i64,
        events: Vec<usize>,
    ) -> MaterialInstanceV1 {
        MaterialInstanceV1 {
            instance_id: id.into(),
            material_id: material.into(),
            part_id: part,
            start: MaterialBeatV1 { numerator: start, denominator: 1 },
            end: MaterialBeatV1 { numerator: end, denominator: 1 },
            event_indices: events,
        }
    }

    fn timeline(
        instances: Vec<MaterialInstanceV1>,
        edges: Vec<MaterialLineageEdgeV1>,
    ) -> MaterialLineageTimelineV1 {
        MaterialLineageTimelineV1 {
            schema: MATERIAL_LINEAGE_SCHEMA_V1.into(),
            score_sha256: digest(),
            instances,
            edges,
        }
    }

    fn edge(id: &str, source: &str, target: &str) -> MaterialLineageEdgeV1 {
        MaterialLineageEdgeV1 {
            edge_id: id.into(),
            source_instance_id: source.into(),
            target_instance_id: target.into(),
            transform: MaterialTransformKindV1::InstrumentHandoff,
            evidence_basis: MaterialEvidenceBasisV1::SourceAuthoredIntent,
            evidence_ref: None,
        }
    }

    #[test]
    fn accepts_source_authored_material_handoff_across_parts() {
        let timeline = timeline(
            vec![
                instance("theme-opening", "theme-alpha", 1, 0, 2, vec![0, 1]),
                instance("theme-return", "theme-alpha", 2, 3, 4, vec![2]),
            ],
            vec![edge("handoff-1", "theme-opening", "theme-return")],
        );
        let report = validate_material_lineage(&score(), &timeline, &digest());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.unique_material_count, 1);
        assert_eq!(report.referenced_event_count, 3);
        assert_eq!(
            report.topological_instance_order,
            vec!["theme-opening".to_owned(), "theme-return".to_owned()]
        );
    }

    #[test]
    fn allows_a_hybrid_material_with_two_authored_parents() {
        let timeline = timeline(
            vec![
                instance("theme-a", "a", 1, 0, 1, vec![0]),
                instance("theme-b", "b", 1, 1, 2, vec![1]),
                instance("hybrid", "a-plus-b", 2, 3, 4, vec![2]),
            ],
            vec![
                MaterialLineageEdgeV1 {
                    transform: MaterialTransformKindV1::Hybridization,
                    ..edge("merge-a", "theme-a", "hybrid")
                },
                MaterialLineageEdgeV1 {
                    transform: MaterialTransformKindV1::Hybridization,
                    ..edge("merge-b", "theme-b", "hybrid")
                },
            ],
        );
        let report = validate_material_lineage(&score(), &timeline, &digest());
        assert!(report.valid, "{:?}", report.issues);
    }

    #[test]
    fn rejects_a_lineage_cycle() {
        let timeline = timeline(
            vec![
                instance("a", "alpha", 1, 0, 1, vec![0]),
                instance("b", "alpha", 1, 1, 2, vec![1]),
            ],
            vec![edge("a-to-b", "a", "b"), edge("b-to-a", "b", "a")],
        );
        let report = validate_material_lineage(&score(), &timeline, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::LineageCycle
        }));
    }

    #[test]
    fn rejects_wrong_part_and_out_of_window_event_references() {
        let wrong_part = timeline(
            vec![instance("bad-part", "alpha", 2, 0, 1, vec![0])],
            Vec::new(),
        );
        let report = validate_material_lineage(&score(), &wrong_part, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::EventPartMismatch
        }));

        let wrong_window = timeline(
            vec![instance("bad-window", "alpha", 1, 1, 2, vec![0])],
            Vec::new(),
        );
        let report = validate_material_lineage(&score(), &wrong_window, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::EventOutsideInstanceWindow
        }));
    }

    #[test]
    fn rejects_zero_duration_source_events() {
        let mut source = score();
        source.notes[0].duration = Duration::zero();
        let timeline = timeline(
            vec![instance("zero-duration", "alpha", 1, 0, 1, vec![0])],
            Vec::new(),
        );
        let report = validate_material_lineage(&source, &timeline, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::InvalidSourceEventTime
        }));
    }

    #[test]
    fn rejects_duplicate_events_and_noncanonical_event_order() {
        let duplicate = timeline(
            vec![instance("duplicate", "alpha", 1, 0, 2, vec![0, 0])],
            Vec::new(),
        );
        let report = validate_material_lineage(&score(), &duplicate, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::DuplicateEventReference
        }));

        let unsorted = timeline(
            vec![instance("unsorted", "alpha", 1, 0, 2, vec![1, 0])],
            Vec::new(),
        );
        let report = validate_material_lineage(&score(), &unsorted, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::NonCanonicalEventOrder
        }));
    }

    #[test]
    fn rejects_duplicate_ids_unknown_edges_and_digest_mismatch() {
        let duplicated = timeline(
            vec![
                instance("same", "alpha", 1, 0, 1, vec![0]),
                instance("same", "beta", 1, 1, 2, vec![1]),
            ],
            vec![edge("unknown-edge", "same", "missing")],
        );
        let report = validate_material_lineage(&score(), &duplicated, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::DuplicateInstanceId
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::UnknownTargetInstance
        }));

        let mismatched = timeline(
            vec![instance("a", "alpha", 1, 0, 1, vec![0])],
            Vec::new(),
        );
        let report = validate_material_lineage(&score(), &mismatched, &"b".repeat(64));
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::ScoreDigestMismatch
        }));
    }

    #[test]
    fn rejects_bad_edge_shapes_and_resource_limits() {
        let bad_edges = timeline(
            vec![instance("a", "alpha", 1, 0, 1, vec![0])],
            vec![
                edge("self", "a", "a"),
                edge("", "missing", "a"),
            ],
        );
        let report = validate_material_lineage(&score(), &bad_edges, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::SelfDerivation
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::InvalidEdgeId
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::UnknownSourceInstance
        }));

        let oversized = MaterialLineageTimelineV1 {
            schema: MATERIAL_LINEAGE_SCHEMA_V1.into(),
            score_sha256: digest(),
            instances: vec![instance("a", "alpha", 1, 0, 1, vec![0]); MAX_MATERIAL_INSTANCES_V1 + 1],
            edges: Vec::new(),
        };
        let report = validate_material_lineage(&score(), &oversized, &digest());
        assert!(report.issues.iter().any(|issue| {
            issue.code == MaterialLineageIssueCodeV1::ResourceLimitExceeded
        }));
    }
}
