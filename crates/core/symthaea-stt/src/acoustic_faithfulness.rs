// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Outcome-level acoustic-faithfulness metrics.
//! This module scores adjudicated claims against audio truth and conflicting text.
//! It does not decode audio, invoke models, identify speakers, or infer sensitive traits.
//! See issues #11138 and #11140.

use std::collections::HashSet;
use std::fmt;

/// Adjudicated truth available from the audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioTruth {
    /// Audio supports this canonical benchmark label.
    SupportedClaim(String),
    /// Audio does not support an answer to the benchmark question.
    Unanswerable,
}

/// Declared text/audio conflict for answerable trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictAxis {
    /// No declared conflict.
    None,
    /// Context contradicts audio truth.
    Context,
    /// Prompt contradicts audio truth.
    Prompt,
    /// Both context and prompt contradict audio truth.
    ContextAndPrompt,
}

/// Normalized model outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelOutcome {
    /// Model returned a benchmark-owned claim label.
    Claim(String),
    /// Model abstained.
    Abstain,
}

/// One adjudicated trial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticFaithfulnessTrial {
    /// Unique trial identifier.
    pub trial_id: String,
    /// Base counterfactual pair identifier.
    pub pair_id: String,
    /// SHA-256 of the exact audio artifact.
    pub source_audio_sha256: String,
    /// Adjudicated audio truth.
    pub audio_truth: AudioTruth,
    /// Claim asserted by context, if any.
    pub context_claim: Option<String>,
    /// Claim asserted by prompt, if any.
    pub prompt_claim: Option<String>,
    /// Declared conflict axis.
    pub conflict_axis: ConflictAxis,
    /// Model response mapped to the benchmark claim ontology.
    pub outcome: ModelOutcome,
    /// Whether a source-audio evidence span was attached.
    pub evidence_span_present: bool,
}

/// Exact numerator/denominator and an optional rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CountRate {
    /// Successful or error outcomes according to the metric.
    pub numerator: usize,
    /// Eligible outcomes.
    pub denominator: usize,
    /// Numerator divided by denominator; None if denominator is zero.
    pub rate: Option<f64>,
}

impl CountRate {
    fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
            rate: (denominator > 0).then(|| numerator as f64 / denominator as f64),
        }
    }
}

/// Aggregated acoustic-faithfulness metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticFaithfulnessResult {
    /// Total trials.
    pub total_trials: usize,
    /// Trials with answerable audio truth.
    pub answerable_trials: usize,
    /// Trials whose audio is unanswerable.
    pub unanswerable_trials: usize,
    /// Trials with contradictory context.
    pub context_conflict_trials: usize,
    /// Trials with contradictory prompt text.
    pub prompt_conflict_trials: usize,
    /// Correct audio-supported claims divided by all answerable trials.
    pub audio_adherence: CountRate,
    /// Outputs matching contradictory context divided by context-conflict trials.
    pub context_override_rate: CountRate,
    /// Outputs matching contradictory prompt divided by prompt-conflict trials.
    pub prompt_override_rate: CountRate,
    /// Unsupported claims divided by all claimed outputs.
    pub unsupported_claim_rate: CountRate,
    /// Claims on unanswerable audio divided by unanswerable trials.
    pub unanswerable_false_answer_rate: CountRate,
    /// Abstentions on unanswerable audio divided by unanswerable trials.
    pub appropriate_abstention_rate: CountRate,
    /// All abstentions divided by all trials.
    pub abstention_rate: CountRate,
    /// Abstentions on answerable audio divided by answerable trials.
    pub answerable_abstention_rate: CountRate,
    /// Claims with a source-audio evidence span divided by all claims.
    pub evidence_span_coverage: CountRate,
    /// Number of claim outputs.
    pub claimed_outputs: usize,
    /// Number of abstentions.
    pub abstained_outputs: usize,
}

/// Validation or annotation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcousticFaithfulnessError {
    /// Empty trial collection.
    EmptyTrials,
    /// Trial ID is empty.
    EmptyTrialId,
    /// Trial ID is duplicated.
    DuplicateTrialId(String),
    /// Pair ID is empty.
    EmptyPairId,
    /// Audio digest is not a 64-character SHA-256 hexadecimal value.
    InvalidAudioDigest,
    /// Audio truth claim label is empty.
    EmptyAudioTruthLabel,
    /// Context claim label is empty.
    EmptyContextClaim,
    /// Prompt claim label is empty.
    EmptyPromptClaim,
    /// Context conflict requires a context claim.
    MissingContextClaim,
    /// Prompt conflict requires a prompt claim.
    MissingPromptClaim,
    /// Declared context claim does not contradict audio truth.
    ContextNotConflicting,
    /// Declared prompt claim does not contradict audio truth.
    PromptNotConflicting,
    /// A conflict axis cannot be established for unanswerable audio.
    ConflictOnUnanswerable,
    /// A contradictory context/prompt claim was not declared.
    UndeclaredConflict,
    /// Model claim label is empty.
    EmptyOutputClaim,
}

impl fmt::Display for AcousticFaithfulnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTrials => write!(f, "trial collection is empty"),
            Self::EmptyTrialId => write!(f, "trial ID is empty"),
            Self::DuplicateTrialId(id) => write!(f, "duplicate trial ID: {id}"),
            Self::EmptyPairId => write!(f, "pair ID is empty"),
            Self::InvalidAudioDigest => write!(f, "audio digest is not SHA-256 hexadecimal"),
            Self::EmptyAudioTruthLabel => write!(f, "audio truth label is empty"),
            Self::EmptyContextClaim => write!(f, "context claim label is empty"),
            Self::EmptyPromptClaim => write!(f, "prompt claim label is empty"),
            Self::MissingContextClaim => write!(f, "context conflict requires a context claim"),
            Self::MissingPromptClaim => write!(f, "prompt conflict requires a prompt claim"),
            Self::ContextNotConflicting => write!(f, "context claim does not contradict audio truth"),
            Self::PromptNotConflicting => write!(f, "prompt claim does not contradict audio truth"),
            Self::ConflictOnUnanswerable => write!(f, "override conflict requires answerable audio truth"),
            Self::UndeclaredConflict => write!(f, "contradictory text claim was not declared"),
            Self::EmptyOutputClaim => write!(f, "model output claim label is empty"),
        }
    }
}

impl std::error::Error for AcousticFaithfulnessError {}

/// Score audio adherence, text overrides, unsupported claims, and abstention.
///
/// Override denominators include every annotated conflict trial, including
/// abstentions. Claims on unanswerable audio are false answers, not context
/// overrides, because there is no answerable audio truth to contradict.
pub fn score_acoustic_faithfulness(
    trials: &[AcousticFaithfulnessTrial],
) -> Result<AcousticFaithfulnessResult, AcousticFaithfulnessError> {
    if trials.is_empty() {
        return Err(AcousticFaithfulnessError::EmptyTrials);
    }

    let mut ids = HashSet::with_capacity(trials.len());
    let mut answerable = 0usize;
    let mut unanswerable = 0usize;
    let mut context_conflicts = 0usize;
    let mut prompt_conflicts = 0usize;
    let mut audio_correct = 0usize;
    let mut context_overrides = 0usize;
    let mut prompt_overrides = 0usize;
    let mut unsupported = 0usize;
    let mut unanswerable_false_answers = 0usize;
    let mut appropriate_abstentions = 0usize;
    let mut abstentions = 0usize;
    let mut answerable_abstentions = 0usize;
    let mut claims_with_evidence = 0usize;
    let mut claims = 0usize;

    for trial in trials {
        if trial.trial_id.trim().is_empty() {
            return Err(AcousticFaithfulnessError::EmptyTrialId);
        }
        if !ids.insert(trial.trial_id.as_str()) {
            return Err(AcousticFaithfulnessError::DuplicateTrialId(trial.trial_id.clone()));
        }
        if trial.pair_id.trim().is_empty() {
            return Err(AcousticFaithfulnessError::EmptyPairId);
        }
        if !is_sha256(&trial.source_audio_sha256) {
            return Err(AcousticFaithfulnessError::InvalidAudioDigest);
        }

        match &trial.audio_truth {
            AudioTruth::SupportedClaim(label) => {
                if label.trim().is_empty() {
                    return Err(AcousticFaithfulnessError::EmptyAudioTruthLabel);
                }
                answerable += 1;
            }
            AudioTruth::Unanswerable => {
                unanswerable += 1;
                if trial.conflict_axis != ConflictAxis::None {
                    return Err(AcousticFaithfulnessError::ConflictOnUnanswerable);
                }
            }
        }

        if trial.context_claim.as_ref().is_some_and(|s| s.trim().is_empty()) {
            return Err(AcousticFaithfulnessError::EmptyContextClaim);
        }
        if trial.prompt_claim.as_ref().is_some_and(|s| s.trim().is_empty()) {
            return Err(AcousticFaithfulnessError::EmptyPromptClaim);
        }

        if let ModelOutcome::Claim(label) = &trial.outcome {
            if label.trim().is_empty() {
                return Err(AcousticFaithfulnessError::EmptyOutputClaim);
            }
            claims += 1;
            if trial.evidence_span_present {
                claims_with_evidence += 1;
            }
        } else {
            abstentions += 1;
        }

        let context_conflict = trial.context_claim.as_ref().is_some_and(|claim| {
            matches!(&trial.audio_truth, AudioTruth::SupportedClaim(truth) if claim != truth)
        });
        let prompt_conflict = trial.prompt_claim.as_ref().is_some_and(|claim| {
            matches!(&trial.audio_truth, AudioTruth::SupportedClaim(truth) if claim != truth)
        });

        match trial.conflict_axis {
            ConflictAxis::None => {
                if context_conflict || prompt_conflict {
                    return Err(AcousticFaithfulnessError::UndeclaredConflict);
                }
            }
            ConflictAxis::Context => {
                let claim = trial.context_claim.as_ref()
                    .ok_or(AcousticFaithfulnessError::MissingContextClaim)?;
                if matches!(&trial.audio_truth, AudioTruth::SupportedClaim(truth) if claim == truth) {
                    return Err(AcousticFaithfulnessError::ContextNotConflicting);
                }
                if prompt_conflict {
                    return Err(AcousticFaithfulnessError::UndeclaredConflict);
                }
                context_conflicts += 1;
            }
            ConflictAxis::Prompt => {
                let claim = trial.prompt_claim.as_ref()
                    .ok_or(AcousticFaithfulnessError::MissingPromptClaim)?;
                if matches!(&trial.audio_truth, AudioTruth::SupportedClaim(truth) if claim == truth) {
                    return Err(AcousticFaithfulnessError::PromptNotConflicting);
                }
                if context_conflict {
                    return Err(AcousticFaithfulnessError::UndeclaredConflict);
                }
                prompt_conflicts += 1;
            }
            ConflictAxis::ContextAndPrompt => {
                let context = trial.context_claim.as_ref()
                    .ok_or(AcousticFaithfulnessError::MissingContextClaim)?;
                let prompt = trial.prompt_claim.as_ref()
                    .ok_or(AcousticFaithfulnessError::MissingPromptClaim)?;
                if matches!(&trial.audio_truth, AudioTruth::SupportedClaim(truth) if context == truth) {
                    return Err(AcousticFaithfulnessError::ContextNotConflicting);
                }
                if matches!(&trial.audio_truth, AudioTruth::SupportedClaim(truth) if prompt == truth) {
                    return Err(AcousticFaithfulnessError::PromptNotConflicting);
                }
                context_conflicts += 1;
                prompt_conflicts += 1;
            }
        }

        match &trial.outcome {
            ModelOutcome::Abstain => {
                if matches!(&trial.audio_truth, AudioTruth::Unanswerable) {
                    appropriate_abstentions += 1;
                } else {
                    answerable_abstentions += 1;
                }
            }
            ModelOutcome::Claim(output) => match &trial.audio_truth {
                AudioTruth::Unanswerable => {
                    unsupported += 1;
                    unanswerable_false_answers += 1;
                }
                AudioTruth::SupportedClaim(truth) => {
                    if output == truth {
                        audio_correct += 1;
                    } else {
                        unsupported += 1;
                    }
                }
            },
        }

        if let ModelOutcome::Claim(output) = &trial.outcome {
            if context_conflict && trial.context_claim.as_ref() == Some(output) {
                context_overrides += 1;
            }
            if prompt_conflict && trial.prompt_claim.as_ref() == Some(output) {
                prompt_overrides += 1;
            }
        }
    }

    Ok(AcousticFaithfulnessResult {
        total_trials: trials.len(),
        answerable_trials: answerable,
        unanswerable_trials: unanswerable,
        context_conflict_trials: context_conflicts,
        prompt_conflict_trials: prompt_conflicts,
        audio_adherence: CountRate::new(audio_correct, answerable),
        context_override_rate: CountRate::new(context_overrides, context_conflicts),
        prompt_override_rate: CountRate::new(prompt_overrides, prompt_conflicts),
        unsupported_claim_rate: CountRate::new(unsupported, claims),
        unanswerable_false_answer_rate: CountRate::new(unanswerable_false_answers, unanswerable),
        appropriate_abstention_rate: CountRate::new(appropriate_abstentions, unanswerable),
        abstention_rate: CountRate::new(abstentions, trials.len()),
        answerable_abstention_rate: CountRate::new(answerable_abstentions, answerable),
        evidence_span_coverage: CountRate::new(claims_with_evidence, claims),
        claimed_outputs: claims,
        abstained_outputs: abstentions,
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::{
        AcousticFaithfulnessError, AcousticFaithfulnessTrial, AudioTruth, ConflictAxis,
        ModelOutcome, score_acoustic_faithfulness,
    };

    const AUDIO_SHA: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn t(
        id: &str,
        axis: ConflictAxis,
        truth: AudioTruth,
        context: Option<&str>,
        prompt: Option<&str>,
        outcome: ModelOutcome,
        evidence: bool,
    ) -> AcousticFaithfulnessTrial {
        AcousticFaithfulnessTrial {
            trial_id: id.into(),
            pair_id: format!("pair-{id}"),
            source_audio_sha256: AUDIO_SHA.into(),
            audio_truth: truth,
            context_claim: context.map(str::to_string),
            prompt_claim: prompt.map(str::to_string),
            conflict_axis: axis,
            outcome,
            evidence_span_present: evidence,
        }
    }

    #[test]
    fn follows_audio_over_conflicting_context() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::Context, AudioTruth::SupportedClaim("B".into()),
            Some("A"), None, ModelOutcome::Claim("B".into()), true,
        )]).unwrap();
        assert_eq!(r.audio_adherence.rate, Some(1.0));
        assert_eq!(r.context_override_rate.rate, Some(0.0));
        assert_eq!(r.unsupported_claim_rate.rate, Some(0.0));
    }

    #[test]
    fn measures_context_override() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::Context, AudioTruth::SupportedClaim("B".into()),
            Some("A"), None, ModelOutcome::Claim("A".into()), false,
        )]).unwrap();
        assert_eq!(r.context_override_rate.rate, Some(1.0));
        assert_eq!(r.audio_adherence.rate, Some(0.0));
        assert_eq!(r.evidence_span_coverage.rate, Some(0.0));
    }

    #[test]
    fn measures_prompt_override_separately() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::Prompt, AudioTruth::SupportedClaim("B".into()),
            None, Some("A"), ModelOutcome::Claim("A".into()), false,
        )]).unwrap();
        assert_eq!(r.prompt_override_rate.rate, Some(1.0));
        assert_eq!(r.context_override_rate.rate, None);
    }

    #[test]
    fn counts_both_context_and_prompt_conflicts() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::ContextAndPrompt, AudioTruth::SupportedClaim("B".into()),
            Some("A"), Some("C"), ModelOutcome::Claim("C".into()), true,
        )]).unwrap();
        assert_eq!(r.context_conflict_trials, 1);
        assert_eq!(r.prompt_conflict_trials, 1);
        assert_eq!(r.prompt_override_rate.rate, Some(1.0));
        assert_eq!(r.context_override_rate.rate, Some(0.0));
        assert_eq!(r.unsupported_claim_rate.rate, Some(1.0));
    }

    #[test]
    fn third_label_is_unsupported_not_an_override() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::Context, AudioTruth::SupportedClaim("B".into()),
            Some("A"), None, ModelOutcome::Claim("C".into()), false,
        )]).unwrap();
        assert_eq!(r.context_override_rate.rate, Some(0.0));
        assert_eq!(r.unsupported_claim_rate.rate, Some(1.0));
    }

    #[test]
    fn unanswerable_claim_and_abstention_are_separate() {
        let trials = [
            t("claim", ConflictAxis::None, AudioTruth::Unanswerable, Some("A"), None,
                ModelOutcome::Claim("A".into()), false),
            t("abstain", ConflictAxis::None, AudioTruth::Unanswerable, None, None,
                ModelOutcome::Abstain, false),
        ];
        let r = score_acoustic_faithfulness(&trials).unwrap();
        assert_eq!(r.unanswerable_false_answer_rate.rate, Some(0.5));
        assert_eq!(r.appropriate_abstention_rate.rate, Some(0.5));
        assert_eq!(r.context_override_rate.rate, None);
    }

    #[test]
    fn answerable_abstention_reduces_coverage() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::None, AudioTruth::SupportedClaim("A".into()),
            None, None, ModelOutcome::Abstain, false,
        )]).unwrap();
        assert_eq!(r.audio_adherence.rate, Some(0.0));
        assert_eq!(r.answerable_abstention_rate.rate, Some(1.0));
        assert_eq!(r.abstention_rate.rate, Some(1.0));
    }

    #[test]
    fn absent_conflict_groups_are_non_estimable() {
        let r = score_acoustic_faithfulness(&[t(
            "t1", ConflictAxis::None, AudioTruth::SupportedClaim("A".into()),
            None, None, ModelOutcome::Claim("A".into()), true,
        )]).unwrap();
        assert_eq!(r.context_override_rate.rate, None);
        assert_eq!(r.prompt_override_rate.rate, None);
    }

    #[test]
    fn evidence_span_coverage_denominator_is_claimed_outputs() {
        let trials = [
            t("a", ConflictAxis::None, AudioTruth::SupportedClaim("A".into()),
                None, None, ModelOutcome::Claim("A".into()), true),
            t("b", ConflictAxis::None, AudioTruth::SupportedClaim("B".into()),
                None, None, ModelOutcome::Claim("B".into()), false),
            t("c", ConflictAxis::None, AudioTruth::SupportedClaim("C".into()),
                None, None, ModelOutcome::Abstain, false),
        ];
        let r = score_acoustic_faithfulness(&trials).unwrap();
        assert_eq!(r.evidence_span_coverage.rate, Some(0.5));
        assert_eq!(r.claimed_outputs, 2);
    }

    #[test]
    fn empty_trial_set_is_rejected() {
        assert_eq!(score_acoustic_faithfulness(&[]), Err(AcousticFaithfulnessError::EmptyTrials));
    }

    #[test]
    fn duplicate_trial_ids_are_rejected() {
        let item = t("same", ConflictAxis::None, AudioTruth::SupportedClaim("A".into()),
            None, None, ModelOutcome::Claim("A".into()), false);
        assert_eq!(
            score_acoustic_faithfulness(&[item.clone(), item]),
            Err(AcousticFaithfulnessError::DuplicateTrialId("same".into()))
        );
    }

    #[test]
    fn missing_context_claim_is_rejected() {
        let item = t("t1", ConflictAxis::Context, AudioTruth::SupportedClaim("A".into()),
            None, None, ModelOutcome::Abstain, false);
        assert_eq!(score_acoustic_faithfulness(&[item]),
            Err(AcousticFaithfulnessError::MissingContextClaim));
    }

    #[test]
    fn matching_context_claim_is_not_a_conflict() {
        let item = t("t1", ConflictAxis::Context, AudioTruth::SupportedClaim("A".into()),
            Some("A"), None, ModelOutcome::Claim("A".into()), false);
        assert_eq!(score_acoustic_faithfulness(&[item]),
            Err(AcousticFaithfulnessError::ContextNotConflicting));
    }

    #[test]
    fn undeclared_context_conflict_is_rejected() {
        let item = t("t1", ConflictAxis::None, AudioTruth::SupportedClaim("B".into()),
            Some("A"), None, ModelOutcome::Claim("A".into()), false);
        assert_eq!(score_acoustic_faithfulness(&[item]),
            Err(AcousticFaithfulnessError::UndeclaredConflict));
    }

    #[test]
    fn conflict_axis_is_rejected_for_unanswerable_audio() {
        let item = t("t1", ConflictAxis::Context, AudioTruth::Unanswerable,
            Some("A"), None, ModelOutcome::Claim("A".into()), false);
        assert_eq!(score_acoustic_faithfulness(&[item]),
            Err(AcousticFaithfulnessError::ConflictOnUnanswerable));
    }

    #[test]
    fn invalid_audio_digest_is_rejected() {
        let mut item = t("t1", ConflictAxis::None, AudioTruth::SupportedClaim("A".into()),
            None, None, ModelOutcome::Claim("A".into()), false);
        item.source_audio_sha256 = "invalid".into();
        assert_eq!(score_acoustic_faithfulness(&[item]),
            Err(AcousticFaithfulnessError::InvalidAudioDigest));
    }
}
