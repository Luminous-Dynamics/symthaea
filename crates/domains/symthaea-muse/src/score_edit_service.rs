// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical Muse preview service for transactional symbolic score edits.
//!
//! The music-theory layer deliberately accepts an expected source digest from
//! its caller. This module is the trusted Muse adapter: it derives the source
//! and candidate digests using the existing score-demand analyzer, computes a
//! domain-separated edit-plan digest, and validates the complete source-to-
//! candidate event remap before returning any preview.
//!
//! Preview only: this module does not persist history, mutate the source,
//! invoke a renderer, export audio, or commit the candidate into a project.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use symthaea_music_theory::score::{Score};
use symthaea_music_theory::score_edit::{
    apply_score_edit_plan, build_velocity_edit_event_remap, ScoreEditCandidateV1,
    ScoreEditIssueV1, ScoreEditPlanV1, ScoreEventRemapIssueV1, ScoreEventRemapV1,
    SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1,
};

use crate::score_demand::{
    analyze_score_demand, ScoreDemandError, ScoreDemandProfile,
    SCORE_DEMAND_INPUT_DIGEST_SCHEMA, SCORE_DEMAND_PROFILE_SCHEMA,
};

pub const SCORE_EDIT_SERVICE_SCHEMA_V1: &str = "muse-score-edit-service-v1";
pub const SCORE_EDIT_TRANSFORMATION_DIGEST_SCHEMA_V1: &str =
    "muse-score-edit-plan-sha256-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum ScoreEditServiceErrorV1 {
    SourceDemand(String),
    UnsupportedSourceDigestSchema,
    LowLevelEditRejected(Vec<ScoreEditIssueV1>),
    CandidateDemand(String),
    CandidateDigestUnchanged,
    PlanDigestSerialization(String),
    EventRemapRejected(Vec<ScoreEventRemapIssueV1>),
}

/// Complete, non-persistent preview artifact. All digests use the canonical
/// Muse ScoreDemand serialization schema. The event map is validated against
/// both exact Scores before this value can be returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreEditPreviewV1 {
    pub schema: String,
    pub input_digest_schema: String,
    pub transformation_digest_schema: String,
    pub source_demand: ScoreDemandProfile,
    pub edit: ScoreEditCandidateV1,
    pub candidate_demand: ScoreDemandProfile,
    pub transformation_sha256: String,
    pub event_remap: ScoreEventRemapV1,
}

/// Domain-separated SHA-256 over the exact serialized edit-plan bytes.
///
/// The canonical input/candidate Score digests remain those emitted by
/// analyze_score_demand. This digest identifies the authored transformation
/// plan; it is not a substitute for either Score digest.
pub fn score_edit_plan_sha256_v1(
    plan: &ScoreEditPlanV1,
) -> Result<String, ScoreEditServiceErrorV1> {
    let bytes = serde_json::to_vec(plan)
        .map_err(|error| ScoreEditServiceErrorV1::PlanDigestSerialization(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(SCORE_EDIT_TRANSFORMATION_DIGEST_SCHEMA_V1.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Preview one score edit using the canonical Muse source and candidate
/// demand profiles. The original Score is borrowed immutably and never changed.
///
/// All-or-nothing contract:
/// 1. Analyze and digest the exact source Score.
/// 2. Apply the low-level edit against that trusted digest.
/// 3. Analyze and digest the candidate Score.
/// 4. Digest the exact edit plan.
/// 5. Build and validate the complete source-to-candidate event remap.
/// 6. Return all artifacts together, or return an error with no candidate.
pub fn preview_score_edit(
    source: &Score,
    plan: &ScoreEditPlanV1,
) -> Result<ScoreEditPreviewV1, ScoreEditServiceErrorV1> {
    let source_demand = analyze_score_demand(source)
        .map_err(|error: ScoreDemandError| ScoreEditServiceErrorV1::SourceDemand(format!("{error:?}")))?;

    if source_demand.schema != SCORE_DEMAND_PROFILE_SCHEMA
        || source_demand.input_digest_schema != SCORE_DEMAND_INPUT_DIGEST_SCHEMA
        || source_demand.input_digest_schema != SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1
    {
        return Err(ScoreEditServiceErrorV1::UnsupportedSourceDigestSchema);
    }

    let edit = apply_score_edit_plan(source, plan, &source_demand.score_sha256)
        .map_err(ScoreEditServiceErrorV1::LowLevelEditRejected)?;

    let candidate_demand = analyze_score_demand(&edit.candidate)
        .map_err(|error: ScoreDemandError| ScoreEditServiceErrorV1::CandidateDemand(format!("{error:?}")))?;

    if candidate_demand.schema != SCORE_DEMAND_PROFILE_SCHEMA
        || candidate_demand.input_digest_schema != source_demand.input_digest_schema
    {
        return Err(ScoreEditServiceErrorV1::UnsupportedSourceDigestSchema);
    }
    if candidate_demand.score_sha256 == source_demand.score_sha256 {
        return Err(ScoreEditServiceErrorV1::CandidateDigestUnchanged);
    }

    let transformation_sha256 = score_edit_plan_sha256_v1(plan)?;
    let event_remap = build_velocity_edit_event_remap(
        source,
        &edit,
        &source_demand.score_sha256,
        &candidate_demand.score_sha256,
        &transformation_sha256,
    )
    .map_err(ScoreEditServiceErrorV1::EventRemapRejected)?;

    Ok(ScoreEditPreviewV1 {
        schema: SCORE_EDIT_SERVICE_SCHEMA_V1.into(),
        input_digest_schema: source_demand.input_digest_schema.clone(),
        transformation_digest_schema: SCORE_EDIT_TRANSFORMATION_DIGEST_SCHEMA_V1.into(),
        source_demand,
        edit,
        candidate_demand,
        transformation_sha256,
        event_remap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::rhythm::Duration;
    use symthaea_music_theory::score::{Emphasis, PartId, ScoreNote, VoiceRole};
    use symthaea_music_theory::score_edit::{
        EditBeatV1, ProtectedPartWindowV1, ScalePartVelocityV1,
        SCORE_EDIT_SCHEMA_V1,
    };
    use symthaea_music_theory::score_event_remap::validate_score_event_remap;

    fn source_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        score.push(ScoreNote {
            part: PartId(10),
            pitch: Pitch::from_midi(60),
            onset: Duration::new(0, 1),
            duration: Duration::new(1, 1),
            velocity: 0.8,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        score.push(ScoreNote {
            part: PartId(10),
            pitch: Pitch::from_midi(64),
            onset: Duration::new(1, 1),
            duration: Duration::new(1, 1),
            velocity: 0.4,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        score.push(ScoreNote {
            part: PartId(20),
            pitch: Pitch::from_midi(48),
            onset: Duration::new(0, 1),
            duration: Duration::new(2, 1),
            velocity: 0.6,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        score
    }

    fn plan() -> ScoreEditPlanV1 {
        ScoreEditPlanV1 {
            schema: SCORE_EDIT_SCHEMA_V1.into(),
            input_digest_schema: SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1.into(),
            input_score_sha256: "a".repeat(64),
            operations: vec![ScalePartVelocityV1 {
                id: "soften-lead".into(),
                part_id: 10,
                start: EditBeatV1 { numerator: 0, denominator: 1 },
                end: EditBeatV1 { numerator: 2, denominator: 1 },
                numerator: 1,
                denominator: 2,
            }],
            protected_windows: vec![ProtectedPartWindowV1 {
                id: "bass-protected".into(),
                part_id: 20,
                start: EditBeatV1 { numerator: 0, denominator: 1 },
                end: EditBeatV1 { numerator: 2, denominator: 1 },
            }],
        }
    }

    #[test]
    fn derives_source_and_candidate_digests_and_validates_event_map() {
        let source = source_score();
        let preview = preview_score_edit(&source, &plan()).expect("valid localized edit");
        assert_ne!(preview.source_demand.score_sha256, preview.candidate_demand.score_sha256);
        assert_eq!(preview.event_remap.source_score_sha256, preview.source_demand.score_sha256);
        assert_eq!(preview.event_remap.candidate_score_sha256, preview.candidate_demand.score_sha256);
        assert_eq!(preview.event_remap.transformation_sha256, preview.transformation_sha256);
        assert_eq!(preview.event_remap.source_event_count, source.notes.len());
        assert_eq!(preview.event_remap.candidate_event_count, preview.edit.candidate.notes.len());
        assert!(validate_score_event_remap(
            &source,
            &preview.edit.candidate,
            &preview.event_remap,
            &preview.source_demand.score_sha256,
            &preview.candidate_demand.score_sha256,
            &preview.transformation_sha256,
        ).valid);
    }

    #[test]
    fn preview_never_mutates_source_score() {
        let source = source_score();
        let before = source.clone();
        let _preview = preview_score_edit(&source, &plan()).expect("valid localized edit");
        assert_eq!(source, before);
    }

    #[test]
    fn candidate_demand_metrics_are_unchanged_except_for_exact_score_digest() {
        let source = source_score();
        let preview = preview_score_edit(&source, &plan()).expect("valid localized edit");
        let mut source_profile = preview.source_demand.clone();
        source_profile.score_sha256 = preview.candidate_demand.score_sha256.clone();
        assert_eq!(source_profile, preview.candidate_demand);
    }

    #[test]
    fn plan_digest_is_deterministic_and_changes_with_plan_content() {
        let mut first = plan();
        first.input_score_sha256 = "c".repeat(64);
        let mut second = first.clone();
        let digest = score_edit_plan_sha256_v1(&first).unwrap();
        assert_eq!(digest, score_edit_plan_sha256_v1(&second).unwrap());
        second.operations[0].numerator = 2;
        assert_ne!(digest, score_edit_plan_sha256_v1(&second).unwrap());
    }

    #[test]
    fn mismatched_input_digest_is_rejected_before_candidate_output() {
        let mut invalid = plan();
        invalid.input_score_sha256 = "b".repeat(64);
        let result = preview_score_edit(&source_score(), &invalid);
        assert!(matches!(
            result,
            Err(ScoreEditServiceErrorV1::LowLevelEditRejected(_))
        ));
    }

    #[test]
    fn invalid_source_score_returns_no_partial_preview() {
        let mut invalid = source_score();
        invalid.notes[0].velocity = 1.2;
        let result = preview_score_edit(&invalid, &plan());
        assert!(matches!(
            result,
            Err(ScoreEditServiceErrorV1::SourceDemand(_))
        ));
    }

    #[test]
    fn invalid_candidate_edit_returns_no_partial_preview() {
        let mut invalid = plan();
        invalid.operations[0].numerator = 2;
        let result = preview_score_edit(&source_score(), &invalid);
        assert!(matches!(
            result,
            Err(ScoreEditServiceErrorV1::LowLevelEditRejected(_))
        ));
    }

    #[test]
    fn frozen_source_and_plan_produce_identical_preview_artifacts() {
        let source = source_score();
        let first = preview_score_edit(&source, &plan()).unwrap();
        let second = preview_score_edit(&source, &plan()).unwrap();
        assert_eq!(first, second);
    }
}
