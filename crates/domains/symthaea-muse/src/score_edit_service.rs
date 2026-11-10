// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trusted Muse-layer preview adapter for symbolic score edits.
//!
//! This service owns the digest wiring between score-demand analysis and the
//! lower-level music-theory edit contract. It returns a source profile,
//! candidate Score, candidate profile, edit receipt, and complete event remap
//! as one all-or-nothing preview. It does not write project history, render
//! audio, mutate the source Score, or commit the candidate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_music_theory::score::Score;
use symthaea_music_theory::score_edit::{
    ScoreEditCandidateV1, ScoreEditIssueV1, ScoreEditPlanV1,
    ScoreEditReceiptV1, apply_score_edit_plan, build_velocity_edit_event_remap,
};
use symthaea_music_theory::score_event_remap::{
    ScoreEventRemapIssueV1, ScoreEventRemapV1,
};

use crate::score_demand::{
    SCORE_DEMAND_INPUT_DIGEST_SCHEMA, ScoreDemandError, ScoreDemandProfile,
    analyze_score_demand,
};

pub const SCORE_EDIT_SERVICE_SCHEMA_V1: &str = "muse-score-edit-preview-service-v1";
/// Domain-separated hash contract for the exact serialized ScoreEditPlanV1.
pub const SCORE_EDIT_PLAN_DIGEST_SCHEMA_V1: &str =
    "muse-score-edit-plan-serde-json-sha256-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreEditServiceErrorV1 {
    SourceDemand(ScoreDemandError),
    EditRejected(Vec<ScoreEditIssueV1>),
    CandidateDemand(ScoreDemandError),
    PlanSerialization(String),
    EventRemapRejected(Vec<ScoreEventRemapIssueV1>),
}

/// Complete, in-memory preview artifact set. No component is persisted here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreEditPreviewV1 {
    pub schema: String,
    pub plan_digest_schema: String,
    pub transformation_sha256: String,
    pub source_demand: ScoreDemandProfile,
    pub candidate: Score,
    pub candidate_demand: ScoreDemandProfile,
    pub edit_receipt: ScoreEditReceiptV1,
    pub event_remap: ScoreEventRemapV1,
}

fn transformation_sha256(plan: &ScoreEditPlanV1) -> Result<String, ScoreEditServiceErrorV1> {
    let bytes = serde_json::to_vec(plan)
        .map_err(|error| ScoreEditServiceErrorV1::PlanSerialization(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(SCORE_EDIT_PLAN_DIGEST_SCHEMA_V1.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Preview an explicit score-edit plan using Muse's canonical score-demand
/// digests. The caller does not supply a source Score hash.
///
/// Sequence:
/// 1. Analyze the exact source Score and derive its trusted digest.
/// 2. Apply the low-level plan against that digest without mutating source.
/// 3. Analyze the candidate Score and derive its canonical candidate digest.
/// 4. Hash the exact serialized edit plan under a domain-separated schema.
/// 5. Emit and independently validate a complete source-to-candidate event map.
///
/// Any error returns without exposing a partial preview. This remains a
/// symbolic-only operation; rendering and persistent project approval are
/// separate downstream transactions.
pub fn preview_score_edit(
    source: &Score,
    plan: &ScoreEditPlanV1,
) -> Result<ScoreEditPreviewV1, ScoreEditServiceErrorV1> {
    let source_demand =
        analyze_score_demand(source).map_err(ScoreEditServiceErrorV1::SourceDemand)?;
    if source_demand.input_digest_schema != SCORE_DEMAND_INPUT_DIGEST_SCHEMA {
        // The score-demand analyzer is the authority for this value. Retain
        // this explicit guard so a future schema change cannot silently flow
        // into the edit planner.
        return Err(ScoreEditServiceErrorV1::EditRejected(vec![
            ScoreEditIssueV1 {
                code: symthaea_music_theory::score_edit::ScoreEditIssueCodeV1::InvalidInputDigestSchema,
                operation_id: None,
                part_id: None,
                note_index: None,
                message: "score-demand analyzer emitted an unsupported digest schema".into(),
            },
        ]));
    }

    let applied: ScoreEditCandidateV1 =
        apply_score_edit_plan(source, plan, &source_demand.score_sha256)
            .map_err(ScoreEditServiceErrorV1::EditRejected)?;

    let candidate_demand = analyze_score_demand(&applied.candidate)
        .map_err(ScoreEditServiceErrorV1::CandidateDemand)?;

    let transformation_sha256 = transformation_sha256(plan)?;
    let event_remap = build_velocity_edit_event_remap(
        source,
        &applied,
        &source_demand.score_sha256,
        &candidate_demand.score_sha256,
        &transformation_sha256,
    )
    .map_err(ScoreEditServiceErrorV1::EventRemapRejected)?;

    Ok(ScoreEditPreviewV1 {
        schema: SCORE_EDIT_SERVICE_SCHEMA_V1.into(),
        plan_digest_schema: SCORE_EDIT_PLAN_DIGEST_SCHEMA_V1.into(),
        transformation_sha256,
        source_demand,
        candidate: applied.candidate,
        candidate_demand,
        edit_receipt: applied.receipt,
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
        SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1, SCORE_EDIT_SCHEMA_V1,
        EditBeatV1, ScalePartVelocityV1, ScoreEditIssueCodeV1,
    };
    use symthaea_music_theory::score_event_remap::validate_score_event_remap;

    fn source_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (part, midi, onset, velocity, role) in [
            (1u16, 60u8, 0i64, 0.8f32, VoiceRole::Melody),
            (1, 64, 1, 0.4, VoiceRole::Melody),
            (2, 48, 0, 0.6, VoiceRole::Bass),
        ] {
            score.push(ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn plan(source: &Score) -> ScoreEditPlanV1 {
        let source_profile = analyze_score_demand(source).expect("valid source profile");
        ScoreEditPlanV1 {
            schema: SCORE_EDIT_SCHEMA_V1.into(),
            input_digest_schema: SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1.into(),
            input_score_sha256: source_profile.score_sha256,
            operations: vec![ScalePartVelocityV1 {
                id: "soften-opening".into(),
                part_id: 1,
                start: EditBeatV1 { numerator: 0, denominator: 2 },
                end: EditBeatV1 { numerator: 2, denominator: 1 },
                numerator: 1,
                denominator: 2,
            }],
            protected_windows: Vec::new(),
        }
    }

    #[test]
    fn preview_derives_both_digests_and_complete_event_remap() {
        let source = source_score();
        let original = source.clone();
        let preview = preview_score_edit(&source, &plan(&source)).expect("valid preview");

        assert_eq!(source, original, "preview must not mutate the source Score");
        assert_eq!(preview.schema, SCORE_EDIT_SERVICE_SCHEMA_V1);
        assert_eq!(preview.source_demand.score_sha256, preview.edit_receipt.input_score_sha256);
        assert_eq!(preview.source_demand.input_digest_schema, SCORE_DEMAND_INPUT_DIGEST_SCHEMA);
        assert_ne!(preview.source_demand.score_sha256, preview.candidate_demand.score_sha256);
        assert_eq!(preview.event_remap.source_score_sha256, preview.source_demand.score_sha256);
        assert_eq!(preview.event_remap.candidate_score_sha256, preview.candidate_demand.score_sha256);
        assert_eq!(preview.event_remap.transformation_sha256, preview.transformation_sha256);
        assert_eq!(preview.event_remap.source_event_count, source.notes.len());
        assert_eq!(preview.event_remap.candidate_event_count, source.notes.len());

        let remap_report = validate_score_event_remap(
            &source,
            &preview.candidate,
            &preview.event_remap,
            &preview.source_demand.score_sha256,
            &preview.candidate_demand.score_sha256,
            &preview.transformation_sha256,
        );
        assert!(remap_report.valid, "{:?}", remap_report.issues);
        assert_eq!(remap_report.modified_count, 2);
        assert_eq!(remap_report.retained_count, 1);
    }

    #[test]
    fn velocity_only_preview_preserves_non_velocity_demand_measurements() {
        let source = source_score();
        let preview = preview_score_edit(&source, &plan(&source)).expect("valid preview");

        assert_eq!(
            preview.source_demand.voice_demands.iter()
                .map(|demand| (demand.part_id, demand.note_count, demand.min_midi, demand.max_midi,
                    demand.distinct_onset_count, demand.peak_simultaneous_notes))
                .collect::<Vec<_>>(),
            preview.candidate_demand.voice_demands.iter()
                .map(|demand| (demand.part_id, demand.note_count, demand.min_midi, demand.max_midi,
                    demand.distinct_onset_count, demand.peak_simultaneous_notes))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn mismatched_plan_digest_is_rejected_without_candidate_output() {
        let source = source_score();
        let mut edit = plan(&source);
        edit.input_score_sha256 = "b".repeat(64);
        let error = preview_score_edit(&source, &edit).unwrap_err();
        assert!(matches!(
            error,
            ScoreEditServiceErrorV1::EditRejected(ref issues)
                if issues.iter().any(|issue| issue.code == ScoreEditIssueCodeV1::InputDigestMismatch)
        ));
    }

    #[test]
    fn malformed_source_demand_fails_before_any_candidate_is_exposed() {
        let mut source = source_score();
        source.notes[0].velocity = 1.5;
        let error = preview_score_edit(&source, &plan(&source)).unwrap_err();
        assert!(matches!(
            error,
            ScoreEditServiceErrorV1::SourceDemand(ScoreDemandError::VelocityOutOfRange { note_index: 0 })
        ));
    }

    #[test]
    fn repeated_preview_is_deterministic_for_frozen_inputs() {
        let source = source_score();
        let edit = plan(&source);
        let first = preview_score_edit(&source, &edit).expect("first preview");
        let second = preview_score_edit(&source, &edit).expect("second preview");
        assert_eq!(first, second);
    }

    #[test]
    fn preview_rejects_a_target_velocity_that_rounds_to_no_change() {
        let mut source = source_score();
        source.notes[0].velocity = 0.0;
        let error = preview_score_edit(&source, &plan(&source)).unwrap_err();
        assert!(matches!(
            error,
            ScoreEditServiceErrorV1::EditRejected(ref issues)
                if issues.iter().any(|issue| {
                    issue.code == ScoreEditIssueCodeV1::NoOpVelocityScale
                        && issue.note_index == Some(0)
                })
        ));
    }

    #[test]
    fn out_of_range_candidate_velocity_is_rejected_before_preview_is_returned() {
        // The low-level transaction rejects out-of-range candidate values
        // before the service can expose a partially successful preview.
        let source = source_score();
        let mut edit = plan(&source);
        edit.operations[0].numerator = 2;
        edit.operations[0].denominator = 1;
        let error = preview_score_edit(&source, &edit).unwrap_err();
        assert!(matches!(
            error,
            ScoreEditServiceErrorV1::EditRejected(ref issues)
                if issues.iter().any(|issue| issue.code == ScoreEditIssueCodeV1::VelocityOutOfRange)
        ));
    }
}
