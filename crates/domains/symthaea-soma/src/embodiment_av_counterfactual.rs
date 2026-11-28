// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired, counterfactual audiovisual evaluation contracts.
//!
//! This module validates evidence manifests and computes per-item metrics from
//! evaluator-supplied decisions. It does not load media, call a model, grade
//! natural-language responses, or claim that correlation proves visual use.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const AV_COUNTERFACTUAL_ITEM_SCHEMA_V1: &str = "symthaea.av-counterfactual-item.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvTaskFamily {
    TurnFloorState,
    Backchannel,
    AttentionGaze,
    GestureCue,
    SharedReferent,
    TemporalAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvConditionType {
    AvPositive,
    AvCounterfactual,
    AvIrrelevantControl,
    AudioOnly,
    VideoOnly,
    AvTemporallyShuffled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvVisualVariant {
    CuePresent,
    CueRemoved,
    CueSubstituted,
    IrrelevantAppearanceChange,
    IrrelevantBackgroundChange,
    None,
    TemporallyShuffled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvCueType {
    TurnYield,
    Backchannel,
    GazeShift,
    Gesture,
    SharedReferent,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvDecision {
    Wait,
    Respond,
    Continue,
    Yield,
    Acknowledge,
    ReferToSharedObject,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvPayload {
    Audio,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvTimingProfile {
    pub duration_ms: u32,
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub fps_numerator: u32,
    pub fps_denominator: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnotatedVisualCue {
    pub cue_type: AvCueType,
    pub start_ms: u32,
    pub end_ms: u32,
    pub task_relevance: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvCounterfactualCondition {
    pub condition_id: String,
    pub condition_type: AvConditionType,
    pub audio_digest: Option<String>,
    pub video_digest: Option<String>,
    pub timing_profile: Option<AvTimingProfile>,
    pub visual_variant: AvVisualVariant,
    pub annotated_visual_cue: Option<AnnotatedVisualCue>,
    pub expected_decision: AvDecision,
    pub model_visible_payloads: Vec<AvPayload>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvCounterfactualItemV1 {
    pub schema: String,
    pub item_id: String,
    pub pair_id: String,
    pub task_family: AvTaskFamily,
    pub task_requires_visual_evidence: bool,
    pub baseline_audio_digest: String,
    pub baseline_timing_profile: AvTimingProfile,
    pub gold_decision: AvDecision,
    pub gold_decision_when_visual_counterfactual: AvDecision,
    pub conditions: Vec<AvCounterfactualCondition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvCounterfactualError {
    WrongSchema,
    InvalidIdentifier,
    InvalidDigest,
    InvalidTimingProfile,
    InvalidConditionCount,
    DuplicateConditionId,
    MissingCondition(AvConditionType),
    DuplicateCondition(AvConditionType),
    AudioDigestMismatch,
    MissingAudioDigest,
    UnexpectedAudioDigest,
    MissingVideoDigest,
    UnexpectedVideoDigest,
    TimingProfileMismatch,
    MissingTimingProfile,
    UnexpectedTimingProfile,
    PayloadVisibilityMismatch,
    VisualVariantMismatch,
    MissingVisualCueAnnotation,
    UnexpectedVisualCueAnnotation,
    CueIntervalOutOfBounds,
    MissingDecisionRelevantVisualCue,
    CounterfactualDecisionNotContrasted,
    IncorrectExpectedDecision,
    MissingPrediction,
    DuplicatePrediction,
    UnexpectedPrediction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvConditionPrediction {
    pub condition_id: String,
    pub decision: AvDecision,
    /// Set only by the independent response grader, not the model under test.
    pub task_relevant_response: bool,
    /// True when the response primarily describes the person's appearance
    /// instead of addressing the interaction task.
    pub appearance_only_response: bool,
}

/// Per-item observations. Aggregate over preregistered items/speakers outside
/// this module and retain all denominators and confidence intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct AvGroundingItemMetrics {
    pub av_positive_correct: bool,
    pub audio_only_correct: bool,
    /// +1, 0, or -1 for this item; not an accuracy percentage.
    pub visual_causal_contribution_item: i8,
    pub counterfactual_sensitive_and_correct: bool,
    pub irrelevant_visual_invariant_and_correct: bool,
    pub temporal_shuffle_rejected: Option<bool>,
    pub captioning_collapse: bool,
    pub task_relevant_conditions: usize,
    pub task_relevant_conditions_graded: usize,
}

pub fn verify_av_counterfactual_item(
    item: &AvCounterfactualItemV1,
) -> Result<(), AvCounterfactualError> {
    if item.schema != AV_COUNTERFACTUAL_ITEM_SCHEMA_V1 {
        return Err(AvCounterfactualError::WrongSchema);
    }
    validate_identifier(&item.item_id)?;
    validate_identifier(&item.pair_id)?;
    validate_digest(&item.baseline_audio_digest)?;
    validate_timing(&item.baseline_timing_profile)?;
    if !(4..=8).contains(&item.conditions.len()) {
        return Err(AvCounterfactualError::InvalidConditionCount);
    }

    let mut ids = HashSet::with_capacity(item.conditions.len());
    let mut by_type: HashMap<AvConditionType, &AvCounterfactualCondition> =
        HashMap::with_capacity(item.conditions.len());

    for condition in &item.conditions {
        validate_identifier(&condition.condition_id)?;
        if !ids.insert(condition.condition_id.as_str()) {
            return Err(AvCounterfactualError::DuplicateConditionId);
        }
        if by_type.insert(condition.condition_type, condition).is_some() {
            return Err(AvCounterfactualError::DuplicateCondition(condition.condition_type));
        }
        verify_condition(item, condition)?;
    }

    for required in [
        AvConditionType::AvPositive,
        AvConditionType::AvCounterfactual,
        AvConditionType::AvIrrelevantControl,
        AvConditionType::AudioOnly,
    ] {
        if !by_type.contains_key(&required) {
            return Err(AvCounterfactualError::MissingCondition(required));
        }
    }

    let positive = by_type[&AvConditionType::AvPositive];
    let counterfactual = by_type[&AvConditionType::AvCounterfactual];
    let irrelevant = by_type[&AvConditionType::AvIrrelevantControl];
    let audio_only = by_type[&AvConditionType::AudioOnly];

    if positive.expected_decision != item.gold_decision
        || irrelevant.expected_decision != item.gold_decision
    {
        return Err(AvCounterfactualError::IncorrectExpectedDecision);
    }
    if counterfactual.expected_decision != item.gold_decision_when_visual_counterfactual {
        return Err(AvCounterfactualError::IncorrectExpectedDecision);
    }
    if item.task_requires_visual_evidence {
        if positive
            .annotated_visual_cue
            .as_ref()
            .is_none_or(|cue| !cue.task_relevance)
        {
            return Err(AvCounterfactualError::MissingDecisionRelevantVisualCue);
        }
        if counterfactual.visual_variant == AvVisualVariant::CueRemoved
            && counterfactual.annotated_visual_cue.is_some()
        {
            return Err(AvCounterfactualError::UnexpectedVisualCueAnnotation);
        }
        if item.gold_decision == item.gold_decision_when_visual_counterfactual {
            return Err(AvCounterfactualError::CounterfactualDecisionNotContrasted);
        }
        if audio_only.expected_decision != item.gold_decision_when_visual_counterfactual {
            return Err(AvCounterfactualError::IncorrectExpectedDecision);
        }
    }
    Ok(())
}

fn verify_condition(
    item: &AvCounterfactualItemV1,
    condition: &AvCounterfactualCondition,
) -> Result<(), AvCounterfactualError> {
    let audio_expected = item.baseline_audio_digest.as_str();
    match condition.condition_type {
        AvConditionType::AudioOnly => {
            let digest = condition
                .audio_digest
                .as_deref()
                .ok_or(AvCounterfactualError::MissingAudioDigest)?;
            validate_digest(digest)?;
            if digest != audio_expected {
                return Err(AvCounterfactualError::AudioDigestMismatch);
            }
            if condition.video_digest.is_some() {
                return Err(AvCounterfactualError::UnexpectedVideoDigest);
            }
            if condition.timing_profile.is_some() {
                return Err(AvCounterfactualError::UnexpectedTimingProfile);
            }
            if condition.visual_variant != AvVisualVariant::None {
                return Err(AvCounterfactualError::VisualVariantMismatch);
            }
            if condition.annotated_visual_cue.is_some() {
                return Err(AvCounterfactualError::UnexpectedVisualCueAnnotation);
            }
            if condition.model_visible_payloads.as_slice() != &[AvPayload::Audio] {
                return Err(AvCounterfactualError::PayloadVisibilityMismatch);
            }
        }
        AvConditionType::VideoOnly => {
            if condition.audio_digest.is_some() {
                return Err(AvCounterfactualError::UnexpectedAudioDigest);
            }
            let digest = condition
                .video_digest
                .as_deref()
                .ok_or(AvCounterfactualError::MissingVideoDigest)?;
            validate_digest(digest)?;
            verify_timing_profile(item, condition)?;
            if condition.model_visible_payloads.as_slice() != &[AvPayload::Video] {
                return Err(AvCounterfactualError::PayloadVisibilityMismatch);
            }
        }
        AvConditionType::AvPositive
        | AvConditionType::AvCounterfactual
        | AvConditionType::AvIrrelevantControl
        | AvConditionType::AvTemporallyShuffled => {
            let audio = condition
                .audio_digest
                .as_deref()
                .ok_or(AvCounterfactualError::MissingAudioDigest)?;
            validate_digest(audio)?;
            if audio != audio_expected {
                return Err(AvCounterfactualError::AudioDigestMismatch);
            }
            let video = condition
                .video_digest
                .as_deref()
                .ok_or(AvCounterfactualError::MissingVideoDigest)?;
            validate_digest(video)?;
            verify_timing_profile(item, condition)?;
            if condition.model_visible_payloads.as_slice() != &[AvPayload::Audio, AvPayload::Video] {
                return Err(AvCounterfactualError::PayloadVisibilityMismatch);
            }
            match condition.condition_type {
                AvConditionType::AvPositive => {
                    if condition.visual_variant != AvVisualVariant::CuePresent
                        || condition.annotated_visual_cue.is_none()
                    {
                        return Err(AvCounterfactualError::MissingVisualCueAnnotation);
                    }
                }
                AvConditionType::AvCounterfactual => {
                    if !matches!(
                        condition.visual_variant,
                        AvVisualVariant::CueRemoved | AvVisualVariant::CueSubstituted
                    ) {
                        return Err(AvCounterfactualError::VisualVariantMismatch);
                    }
                }
                AvConditionType::AvIrrelevantControl => {
                    if !matches!(
                        condition.visual_variant,
                        AvVisualVariant::IrrelevantAppearanceChange
                            | AvVisualVariant::IrrelevantBackgroundChange
                    ) {
                        return Err(AvCounterfactualError::VisualVariantMismatch);
                    }
                    if condition.annotated_visual_cue.is_none() {
                        return Err(AvCounterfactualError::MissingVisualCueAnnotation);
                    }
                }
                AvConditionType::AvTemporallyShuffled => {
                    if condition.visual_variant != AvVisualVariant::TemporallyShuffled {
                        return Err(AvCounterfactualError::VisualVariantMismatch);
                    }
                    if condition.annotated_visual_cue.as_ref().is_none_or(|cue| cue.task_relevance) {
                        return Err(AvCounterfactualError::UnexpectedVisualCueAnnotation);
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    if let Some(cue) = condition.annotated_visual_cue.as_ref() {
        let timing = condition
            .timing_profile
            .as_ref()
            .ok_or(AvCounterfactualError::MissingTimingProfile)?;
        if cue.start_ms >= cue.end_ms || cue.end_ms > timing.duration_ms {
            return Err(AvCounterfactualError::CueIntervalOutOfBounds);
        }
    }
    Ok(())
}

fn verify_timing_profile(
    item: &AvCounterfactualItemV1,
    condition: &AvCounterfactualCondition,
) -> Result<(), AvCounterfactualError> {
    let timing = condition
        .timing_profile
        .as_ref()
        .ok_or(AvCounterfactualError::MissingTimingProfile)?;
    validate_timing(timing)?;
    if timing != &item.baseline_timing_profile {
        return Err(AvCounterfactualError::TimingProfileMismatch);
    }
    Ok(())
}

fn validate_timing(timing: &AvTimingProfile) -> Result<(), AvCounterfactualError> {
    if timing.duration_ms == 0
        || timing.frame_count == 0
        || timing.width == 0
        || timing.height == 0
        || timing.fps_numerator == 0
        || timing.fps_denominator == 0
        || timing.duration_ms > 600_000
        || timing.frame_count > 1_000_000
        || timing.width > 8192
        || timing.height > 8192
    {
        return Err(AvCounterfactualError::InvalidTimingProfile);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), AvCounterfactualError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(AvCounterfactualError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), AvCounterfactualError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AvCounterfactualError::InvalidDigest);
    }
    Ok(())
}

/// Score one paired item using outputs adjudicated by an independent grader.
/// Missing/extra predictions fail rather than silently shrinking denominators.
pub fn evaluate_av_counterfactual_item(
    item: &AvCounterfactualItemV1,
    predictions: &[AvConditionPrediction],
) -> Result<AvGroundingItemMetrics, AvCounterfactualError> {
    verify_av_counterfactual_item(item)?;
    if predictions.len() != item.conditions.len() {
        return Err(AvCounterfactualError::MissingPrediction);
    }
    let mut prediction_by_id = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if prediction_by_id
            .insert(prediction.condition_id.as_str(), prediction)
            .is_some()
        {
            return Err(AvCounterfactualError::DuplicatePrediction);
        }
    }

    let mut condition_by_type = HashMap::with_capacity(item.conditions.len());
    for condition in &item.conditions {
        condition_by_type.insert(condition.condition_type, condition);
        if !prediction_by_id.contains_key(condition.condition_id.as_str()) {
            return Err(AvCounterfactualError::MissingPrediction);
        }
    }
    if prediction_by_id
        .keys()
        .any(|id| !item.conditions.iter().any(|condition| condition.condition_id == *id))
    {
        return Err(AvCounterfactualError::UnexpectedPrediction);
    }

    let prediction_for = |condition_type: AvConditionType| {
        let condition = condition_by_type[&condition_type];
        prediction_by_id[condition.condition_id.as_str()]
    };
    let av_positive = prediction_for(AvConditionType::AvPositive);
    let audio_only = prediction_for(AvConditionType::AudioOnly);
    let counterfactual = prediction_for(AvConditionType::AvCounterfactual);
    let irrelevant = prediction_for(AvConditionType::AvIrrelevantControl);
    let av_positive_correct = av_positive.decision == item.gold_decision;
    let audio_only_correct = audio_only.decision == item.conditions
        .iter()
        .find(|condition| condition.condition_type == AvConditionType::AudioOnly)
        .expect("required by verifier")
        .expected_decision;
    let counterfactual_sensitive_and_correct = counterfactual.decision
        == item.gold_decision_when_visual_counterfactual
        && counterfactual.decision != av_positive.decision;
    let irrelevant_visual_invariant_and_correct = irrelevant.decision == av_positive.decision
        && irrelevant.decision == item.gold_decision;
    let temporal_shuffle_rejected = condition_by_type
        .get(&AvConditionType::AvTemporallyShuffled)
        .map(|condition| {
            prediction_by_id[condition.condition_id.as_str()].decision
                == condition.expected_decision
        });
    let task_relevant: Vec<_> = item
        .conditions
        .iter()
        .filter(|condition| {
            condition
                .annotated_visual_cue
                .as_ref()
                .is_some_and(|cue| cue.task_relevance)
        })
        .collect();
    let task_relevant_conditions_graded = task_relevant.len();
    let captioning_collapse = task_relevant.iter().any(|condition| {
        prediction_by_id[condition.condition_id.as_str()].appearance_only_response
    });
    let task_relevant_responses = task_relevant
        .iter()
        .filter(|condition| {
            prediction_by_id[condition.condition_id.as_str()].task_relevant_response
        })
        .count();

    Ok(AvGroundingItemMetrics {
        av_positive_correct,
        audio_only_correct,
        visual_causal_contribution_item: (if av_positive_correct { 1 } else { 0 })
            - (if audio_only_correct { 1 } else { 0 }),
        counterfactual_sensitive_and_correct,
        irrelevant_visual_invariant_and_correct,
        temporal_shuffle_rejected,
        captioning_collapse,
        task_relevant_conditions: task_relevant_conditions_graded,
        task_relevant_conditions_graded: task_relevant_responses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timing() -> AvTimingProfile {
        AvTimingProfile {
            duration_ms: 5_000,
            frame_count: 150,
            width: 640,
            height: 360,
            fps_numerator: 30,
            fps_denominator: 1,
        }
    }

    fn cue(start_ms: u32, end_ms: u32, task_relevance: bool) -> AnnotatedVisualCue {
        AnnotatedVisualCue {
            cue_type: AvCueType::TurnYield,
            start_ms,
            end_ms,
            task_relevance,
        }
    }

    fn digest(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn condition(
        id: &str,
        condition_type: AvConditionType,
        audio_digest: Option<String>,
        video_digest: Option<String>,
        timing_profile: Option<AvTimingProfile>,
        visual_variant: AvVisualVariant,
        annotated_visual_cue: Option<AnnotatedVisualCue>,
        expected_decision: AvDecision,
        model_visible_payloads: Vec<AvPayload>,
    ) -> AvCounterfactualCondition {
        AvCounterfactualCondition {
            condition_id: id.to_owned(),
            condition_type,
            audio_digest,
            video_digest,
            timing_profile,
            visual_variant,
            annotated_visual_cue,
            expected_decision,
            model_visible_payloads,
        }
    }

    fn item() -> AvCounterfactualItemV1 {
        let timing = timing();
        let audio = digest('a');
        AvCounterfactualItemV1 {
            schema: AV_COUNTERFACTUAL_ITEM_SCHEMA_V1.to_owned(),
            item_id: "av-turn-yield-001".to_owned(),
            pair_id: "pair-turn-yield-001".to_owned(),
            task_family: AvTaskFamily::TurnFloorState,
            task_requires_visual_evidence: true,
            baseline_audio_digest: audio.clone(),
            baseline_timing_profile: timing.clone(),
            gold_decision: AvDecision::Yield,
            gold_decision_when_visual_counterfactual: AvDecision::Continue,
            conditions: vec![
                condition("av-positive",AvConditionType::AvPositive,Some(audio.clone()),Some(digest('b')),Some(timing.clone()),AvVisualVariant::CuePresent,Some(cue(1000,1400,true)),AvDecision::Yield,vec![AvPayload::Audio,AvPayload::Video]),
                condition("av-counterfactual",AvConditionType::AvCounterfactual,Some(audio.clone()),Some(digest('c')),Some(timing.clone()),AvVisualVariant::CueRemoved,None,AvDecision::Continue,vec![AvPayload::Audio,AvPayload::Video]),
                condition("av-irrelevant",AvConditionType::AvIrrelevantControl,Some(audio.clone()),Some(digest('d')),Some(timing.clone()),AvVisualVariant::IrrelevantAppearanceChange,Some(cue(1000,1400,true)),AvDecision::Yield,vec![AvPayload::Audio,AvPayload::Video]),
                condition("audio-only",AvConditionType::AudioOnly,Some(audio.clone()),None,None,AvVisualVariant::None,None,AvDecision::Continue,vec![AvPayload::Audio]),
                condition("video-only",AvConditionType::VideoOnly,None,Some(digest('b')),Some(timing.clone()),AvVisualVariant::CuePresent,Some(cue(1000,1400,true)),AvDecision::Yield,vec![AvPayload::Video]),
                condition("av-shuffled",AvConditionType::AvTemporallyShuffled,Some(audio.clone()),Some(digest('e')),Some(timing),AvVisualVariant::TemporallyShuffled,Some(cue(3500,3900,false)),AvDecision::Continue,vec![AvPayload::Audio,AvPayload::Video]),
            ],
        }
    }

    #[test]
    fn paired_manifest_preserves_identical_audio_and_timing() {
        assert_eq!(verify_av_counterfactual_item(&item()), Ok(()));
    }

    #[test]
    fn changed_audio_digest_is_rejected() {
        let mut value = item();
        value.conditions[1].audio_digest = Some(digest('f'));
        assert_eq!(
            verify_av_counterfactual_item(&value),
            Err(AvCounterfactualError::AudioDigestMismatch)
        );
    }

    #[test]
    fn changed_video_timing_is_rejected() {
        let mut value = item();
        value.conditions[1].timing_profile.as_mut().unwrap().frame_count -= 1;
        assert_eq!(
            verify_av_counterfactual_item(&value),
            Err(AvCounterfactualError::TimingProfileMismatch)
        );
    }

    #[test]
    fn out_of_bounds_cue_annotation_is_rejected() {
        let mut value = item();
        value.conditions[0].annotated_visual_cue.as_mut().unwrap().end_ms = 6_000;
        assert_eq!(
            verify_av_counterfactual_item(&value),
            Err(AvCounterfactualError::CueIntervalOutOfBounds)
        );
    }

    #[test]
    fn visual_required_item_needs_a_decision_relevant_positive_cue() {
        let mut value = item();
        value.conditions[0].annotated_visual_cue = Some(cue(1000,1400,false));
        assert_eq!(
            verify_av_counterfactual_item(&value),
            Err(AvCounterfactualError::MissingDecisionRelevantVisualCue)
        );
    }

    #[test]
    fn counterfactual_visual_condition_must_change_gold_decision() {
        let mut value = item();
        value.gold_decision_when_visual_counterfactual = AvDecision::Yield;
        value.conditions[1].expected_decision = AvDecision::Yield;
        value.conditions[3].expected_decision = AvDecision::Yield;
        assert_eq!(
            verify_av_counterfactual_item(&value),
            Err(AvCounterfactualError::CounterfactualDecisionNotContrasted)
        );
    }

    #[test]
    fn per_item_evaluation_reports_each_dimension_separately() {
        let value = item();
        let predictions = vec![
            AvConditionPrediction { condition_id:"av-positive".to_owned(),decision:AvDecision::Yield,task_relevant_response:true,appearance_only_response:false },
            AvConditionPrediction { condition_id:"av-counterfactual".to_owned(),decision:AvDecision::Continue,task_relevant_response:true,appearance_only_response:false },
            AvConditionPrediction { condition_id:"av-irrelevant".to_owned(),decision:AvDecision::Yield,task_relevant_response:true,appearance_only_response:false },
            AvConditionPrediction { condition_id:"audio-only".to_owned(),decision:AvDecision::Continue,task_relevant_response:true,appearance_only_response:false },
            AvConditionPrediction { condition_id:"video-only".to_owned(),decision:AvDecision::Yield,task_relevant_response:true,appearance_only_response:false },
            AvConditionPrediction { condition_id:"av-shuffled".to_owned(),decision:AvDecision::Continue,task_relevant_response:true,appearance_only_response:false },
        ];
        let metrics = evaluate_av_counterfactual_item(&value,&predictions).unwrap();
        assert!(metrics.av_positive_correct);
        assert!(metrics.counterfactual_sensitive_and_correct);
        assert!(metrics.irrelevant_visual_invariant_and_correct);
        assert_eq!(metrics.visual_causal_contribution_item,1);
        assert_eq!(metrics.temporal_shuffle_rejected,Some(true));
        assert!(!metrics.captioning_collapse);
    }
}
