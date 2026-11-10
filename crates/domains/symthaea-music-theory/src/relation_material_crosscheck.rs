// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-validation between temporal part relationships and material lineage.
//!
//! This module proves that every relation's optional material-family reference
//! resolves inside the same frozen Score and, for two-part relationships, has
//! exactly one full-window source-authored instance for each participant.
//! PlannedSilence is unary: it requires a known material family, but not a
//! sounding instance in the silent interval.
//!
//! Graph/reference validity does not prove that imitation, canon, doubling or
//! counterpoint is musically convincing. No identity is inferred from pitch,
//! contour, register, VoiceRole or acoustic similarity.

use serde::{Deserialize, Serialize};

use crate::ensemble_relations::{
    ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1, EnsembleRelationKindV1,
    EnsembleRelationReportV1, EnsembleRelationTimelineV1,
    validate_ensemble_relations,
};
use crate::material_lineage::{
    MaterialInstanceV1, MaterialLineageReportV1, MaterialLineageTimelineV1,
    validate_material_lineage,
};
use crate::score::Score;

pub const RELATION_MATERIAL_CROSSCHECK_SCHEMA_V1: &str =
    "relation-material-crosscheck-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationMaterialIssueCodeV1 {
    BaseRelationTimelineInvalid,
    BaseMaterialTimelineInvalid,
    SidecarDigestMismatch,
    UnknownMaterialFamily,
    MissingPartMaterialInstance,
    AmbiguousPartMaterialInstance,
}

/// A cross-graph issue. The underlying component reports are retained
/// separately so callers can inspect the precise source validator failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationMaterialIssueV1 {
    pub code: RelationMaterialIssueCodeV1,
    pub relation_id: Option<String>,
    pub material_id: Option<String>,
    pub part_id: Option<u16>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationMaterialCrosscheckReportV1 {
    pub schema: String,
    pub valid: bool,
    pub relation_count: usize,
    pub material_instance_count: usize,
    pub unique_material_count: usize,
    pub relation_report: EnsembleRelationReportV1,
    pub material_report: MaterialLineageReportV1,
    pub issues: Vec<RelationMaterialIssueV1>,
}

impl RelationMaterialCrosscheckReportV1 {
    fn push(
        &mut self,
        code: RelationMaterialIssueCodeV1,
        relation_id: Option<&str>,
        material_id: Option<&str>,
        part_id: Option<u16>,
        message: impl Into<String>,
    ) {
        self.valid = false;
        self.issues.push(RelationMaterialIssueV1 {
            code,
            relation_id: relation_id.map(str::to_owned),
            material_id: material_id.map(str::to_owned),
            part_id,
            message: message.into(),
        });
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Whether a source-authored material instance covers the entire relation
/// interval. Both input timelines have already passed their own bounded-time
/// validators, so these cross products remain safely within i128.
fn instance_covers_relation(
    instance: &MaterialInstanceV1,
    relation_start_num: i64,
    relation_start_den: i64,
    relation_end_num: i64,
    relation_end_den: i64,
) -> bool {
    let instance_starts_no_later =
        i128::from(instance.start.numerator) * i128::from(relation_start_den)
            <= i128::from(relation_start_num) * i128::from(instance.start.denominator);
    let instance_ends_no_earlier =
        i128::from(instance.end.numerator) * i128::from(relation_end_den)
            >= i128::from(relation_end_num) * i128::from(instance.end.denominator);
    instance_starts_no_later && instance_ends_no_earlier
}

/// Validate relation and material sidecars against one exact symbolic Score.
///
/// Both component validators run first. Cross-validation proceeds only when
/// both reports are valid and both timeline digests equal the independently
/// supplied expected digest. For each relation with a material_id:
///
/// - the material family must exist;
/// - PlannedSilence only requires that family to exist;
/// - every two-part relation requires exactly one family instance per
///   participant PartId whose envelope covers the whole relation interval.
///
/// Multiple covering instances for one family/part are treated as ambiguous;
/// the validator does not choose one based on pitch or note similarity.
pub fn validate_relation_material_crosscheck(
    score: &Score,
    relations: &EnsembleRelationTimelineV1,
    materials: &MaterialLineageTimelineV1,
    expected_score_sha256: &str,
) -> RelationMaterialCrosscheckReportV1 {
    let relation_report =
        validate_ensemble_relations(score, relations, expected_score_sha256);
    let material_report =
        validate_material_lineage(score, materials, expected_score_sha256);

    let mut material_ids = std::collections::BTreeSet::new();
    for instance in &materials.instances {
        if !instance.material_id.trim().is_empty() {
            material_ids.insert(instance.material_id.as_str());
        }
    }

    let mut report = RelationMaterialCrosscheckReportV1 {
        schema: RELATION_MATERIAL_CROSSCHECK_SCHEMA_V1.into(),
        valid: relation_report.valid && material_report.valid,
        relation_count: relations.relations.len(),
        material_instance_count: materials.instances.len(),
        unique_material_count: material_ids.len(),
        relation_report,
        material_report,
        issues: Vec::new(),
    };

    if !report.relation_report.valid {
        report.push(
            RelationMaterialIssueCodeV1::BaseRelationTimelineInvalid,
            None,
            None,
            None,
            "relation timeline failed its independent score/part/time validator",
        );
    }
    if !report.material_report.valid {
        report.push(
            RelationMaterialIssueCodeV1::BaseMaterialTimelineInvalid,
            None,
            None,
            None,
            "material lineage failed its independent score/event/DAG validator",
        );
    }

    let digest_binding_valid = valid_digest(expected_score_sha256)
        && relations.score_sha256 == expected_score_sha256
        && materials.score_sha256 == expected_score_sha256
        && relations.score_sha256 == materials.score_sha256
        && relations.score_digest_schema == ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1;

    if !digest_binding_valid {
        report.push(
            RelationMaterialIssueCodeV1::SidecarDigestMismatch,
            None,
            None,
            None,
            "both sidecars must bind to the same trusted score digest and canonical digest schema",
        );
    }

    // Do not cross-validate invalid component graphs. Their detailed errors
    // remain visible in relation_report and material_report.
    if !report.relation_report.valid
        || !report.material_report.valid
        || !digest_binding_valid
    {
        return report;
    }

    for relation in &relations.relations {
        let Some(material_id) = relation.material_id.as_deref() else {
            continue;
        };

        if !material_ids.contains(material_id) {
            report.push(
                RelationMaterialIssueCodeV1::UnknownMaterialFamily,
                Some(&relation.id),
                Some(material_id),
                None,
                "relation material_id does not resolve to a material family in this Score",
            );
            continue;
        }

        // Silence is unary and intentionally has no sounding instance at
        // this time. A known material-family reference is sufficient.
        if relation.kind == EnsembleRelationKindV1::PlannedSilence {
            continue;
        }

        let Some(part_b) = relation.part_b else {
            // The base relation validator will already reject this shape.
            continue;
        };

        for part_id in [relation.part_a, part_b] {
            let matching: Vec<&MaterialInstanceV1> = materials
                .instances
                .iter()
                .filter(|instance| {
                    instance.material_id == material_id
                        && instance.part_id == part_id
                        && instance_covers_relation(
                            instance,
                            relation.start.numerator,
                            relation.start.denominator,
                            relation.end.numerator,
                            relation.end.denominator,
                        )
                })
                .collect();

            match matching.len() {
                0 => report.push(
                    RelationMaterialIssueCodeV1::MissingPartMaterialInstance,
                    Some(&relation.id),
                    Some(material_id),
                    Some(part_id),
                    "no instance of this material family for the participant PartId covers the full relation window",
                ),
                1 => {}
                _ => report.push(
                    RelationMaterialIssueCodeV1::AmbiguousPartMaterialInstance,
                    Some(&relation.id),
                    Some(material_id),
                    Some(part_id),
                    "multiple instances of this material family for the participant PartId cover the full relation window",
                ),
            }
        }
    }

    if !report.issues.is_empty() {
        report.valid = false;
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble_relations::{
        ENSEMBLE_RELATION_SCHEMA_V1, EnsembleRelationV1, RelationBeatV1,
    };
    use crate::harmony::Key;
    use crate::material_lineage::{
        MATERIAL_LINEAGE_SCHEMA_V1, MaterialBeatV1, MaterialInstanceV1,
    };
    use crate::pitch::{Pitch, PitchClass};
    use crate::rhythm::Duration;
    use crate::score::{Emphasis, PartId, ScoreNote, VoiceRole};

    fn digest() -> String {
        "a".repeat(64)
    }

    fn score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (part, midi, onset, duration) in [
            (1u16, 60u8, 0i64, 2i64),
            (2, 67, 0, 2),
            (3, 55, 4, 1),
        ] {
            score.push(ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(duration, 1),
                velocity: 0.6,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn relation(
        kind: EnsembleRelationKindV1,
        material_id: Option<&str>,
    ) -> EnsembleRelationV1 {
        EnsembleRelationV1 {
            id: "rel-1".into(),
            kind,
            part_a: if kind == EnsembleRelationKindV1::PlannedSilence { 3 } else { 1 },
            part_b: if kind == EnsembleRelationKindV1::PlannedSilence {
                None
            } else {
                Some(2)
            },
            start: RelationBeatV1 { numerator: 0, denominator: 1 },
            end: RelationBeatV1 { numerator: 2, denominator: 1 },
            material_id: material_id.map(str::to_owned),
        }
    }

    fn relation_timeline(relations: Vec<EnsembleRelationV1>) -> EnsembleRelationTimelineV1 {
        EnsembleRelationTimelineV1 {
            schema: ENSEMBLE_RELATION_SCHEMA_V1.into(),
            score_digest_schema: ENSEMBLE_RELATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: digest(),
            relations,
        }
    }

    fn instance(
        instance_id: &str,
        part_id: u16,
        event_index: usize,
        material_id: &str,
        start: i64,
        end: i64,
    ) -> MaterialInstanceV1 {
        MaterialInstanceV1 {
            instance_id: instance_id.into(),
            material_id: material_id.into(),
            part_id,
            start: MaterialBeatV1 { numerator: start, denominator: 1 },
            end: MaterialBeatV1 { numerator: end, denominator: 1 },
            event_indices: vec![event_index],
        }
    }

    fn material_timeline(instances: Vec<MaterialInstanceV1>) -> MaterialLineageTimelineV1 {
        MaterialLineageTimelineV1 {
            schema: MATERIAL_LINEAGE_SCHEMA_V1.into(),
            score_sha256: digest(),
            instances,
            edges: Vec::new(),
        }
    }

    fn valid_materials() -> MaterialLineageTimelineV1 {
        material_timeline(vec![
            instance("theme-cello", 1, 0, "theme-A", 0, 2),
            instance("theme-horn", 2, 1, "theme-A", 0, 2),
        ])
    }

    #[test]
    fn accepts_a_two_part_relation_with_source_instances_covering_the_window() {
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![relation(EnsembleRelationKindV1::Imitation, Some("theme-A"))]),
            &valid_materials(),
            &digest(),
        );
        assert!(report.valid, "{:?}", report.issues);
    }

    #[test]
    fn rejects_an_unknown_material_family() {
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![relation(EnsembleRelationKindV1::Doubling, Some("unknown-theme"))]),
            &valid_materials(),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::UnknownMaterialFamily
        }));
    }

    #[test]
    fn rejects_missing_part_material_coverage() {
        let materials = material_timeline(vec![
            instance("theme-cello", 1, 0, "theme-A", 0, 2),
        ]);
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![relation(EnsembleRelationKindV1::Canon, Some("theme-A"))]),
            &materials,
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::MissingPartMaterialInstance
                && issue.part_id == Some(2)
        }));
    }

    #[test]
    fn rejects_ambiguous_covering_instances_for_one_part() {
        let materials = material_timeline(vec![
            instance("theme-cello-a", 1, 0, "theme-A", 0, 2),
            instance("theme-cello-b", 1, 0, "theme-A", 0, 2),
            instance("theme-horn", 2, 1, "theme-A", 0, 2),
        ]);
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![relation(EnsembleRelationKindV1::Doubling, Some("theme-A"))]),
            &materials,
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::AmbiguousPartMaterialInstance
                && issue.part_id == Some(1)
        }));
    }

    #[test]
    fn requires_the_instance_envelope_to_cover_the_full_relation_window() {
        // Part 1 has a valid source-authored instance but its envelope ends at
        // beat 1, before the relationship window ends at beat 2.
        let materials = material_timeline(vec![
            instance("theme-cello", 1, 0, "theme-A", 0, 1),
            instance("theme-horn", 2, 1, "theme-A", 0, 2),
        ]);
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![relation(EnsembleRelationKindV1::Counterpoint, Some("theme-A"))]),
            &materials,
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::MissingPartMaterialInstance
                && issue.part_id == Some(1)
        }));
    }

    #[test]
    fn planned_silence_requires_only_a_known_material_family() {
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![relation(EnsembleRelationKindV1::PlannedSilence, Some("theme-A"))]),
            &valid_materials(),
            &digest(),
        );
        assert!(report.valid, "{:?}", report.issues);
    }

    #[test]
    fn rejects_a_material_or_relation_timeline_bound_to_another_score() {
        let mut relations = relation_timeline(vec![
            relation(EnsembleRelationKindV1::Imitation, Some("theme-A")),
        ]);
        relations.score_sha256 = "b".repeat(64);
        let report = validate_relation_material_crosscheck(
            &score(),
            &relations,
            &valid_materials(),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::BaseRelationTimelineInvalid
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::SidecarDigestMismatch
        }));
    }

    #[test]
    fn rejects_a_material_timeline_bound_to_another_score() {
        let mut materials = valid_materials();
        materials.score_sha256 = "b".repeat(64);
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![
                relation(EnsembleRelationKindV1::Doubling, Some("theme-A")),
            ]),
            &materials,
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::BaseMaterialTimelineInvalid
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::SidecarDigestMismatch
        }));
    }

    #[test]
    fn rejects_a_relation_timeline_using_an_unknown_digest_schema() {
        let mut relations = relation_timeline(vec![
            relation(EnsembleRelationKindV1::Canon, Some("theme-A")),
        ]);
        relations.score_digest_schema = "unknown-score-digest-v2".into();
        let report = validate_relation_material_crosscheck(
            &score(),
            &relations,
            &valid_materials(),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::BaseRelationTimelineInvalid
        }));
        assert!(!report.relation_report.valid);
    }

    #[test]
    fn planned_silence_still_rejects_an_unknown_material_family() {
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![
                relation(EnsembleRelationKindV1::PlannedSilence, Some("missing-family")),
            ]),
            &valid_materials(),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::UnknownMaterialFamily
        }));
    }

    #[test]
    fn reports_component_validator_failures_without_cross_graph_acceptance() {
        let mut materials = valid_materials();
        materials.instances[0].event_indices[0] = 99;
        let report = validate_relation_material_crosscheck(
            &score(),
            &relation_timeline(vec![
                relation(EnsembleRelationKindV1::Imitation, Some("theme-A")),
            ]),
            &materials,
            &digest(),
        );
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == RelationMaterialIssueCodeV1::BaseMaterialTimelineInvalid
        }));
        assert!(!report.material_report.valid);
    }
}
