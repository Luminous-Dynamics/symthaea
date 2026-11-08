// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired counterfactual metrics for acoustic evidence invariance/sensitivity.
//! Reuses adjudicated trials from the acoustic-faithfulness scorer.
//! No audio decoding, model invocation, speaker identification, or sensitive inference.

use crate::acoustic_faithfulness::{AcousticFaithfulnessTrial, AudioTruth, ModelOutcome};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Controlled intervention between baseline and variant trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterfactualInterventionKind {
    /// Same audio and truth; only context/prompt conditions change.
    TextNuisance,
    /// Audio changes the adjudicated answer; non-audio conditions stay fixed.
    AudioRelevant,
    /// Audio changes through a nuisance transform; truth stays fixed.
    AudioNuisance,
}

/// Explicitly declared intervention dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangedDimension {
    ContextText,
    PromptText,
    LexicalContent,
    Negation,
    NumberOrEntity,
    EventPresence,
    TemporalOrder,
    AcousticNuisance,
}

/// Frozen relationship between two adjudicated trial records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterfactualPairManifest {
    pub pair_id: String,
    pub baseline_trial_id: String,
    pub variant_trial_id: String,
    pub intervention_kind: CounterfactualInterventionKind,
    pub baseline_condition_manifest_sha256: String,
    pub variant_condition_manifest_sha256: String,
    pub intervention_manifest_sha256: String,
    pub changed_dimensions: Vec<ChangedDimension>,
}

/// Exact numerator, denominator and optional rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairRate {
    pub numerator: usize,
    pub denominator: usize,
    pub rate: Option<f64>,
}

impl PairRate {
    fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
            rate: (denominator > 0).then(|| numerator as f64 / denominator as f64),
        }
    }
}

/// Pair-level nuisance-invariance and evidence-sensitivity results.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualPairResult {
    pub total_pairs: usize,
    pub text_nuisance_pairs: usize,
    /// Same normalized CLAIM/ABSTAIN outcome under text-only intervention.
    pub text_response_invariance: PairRate,
    /// Both outputs equal their own unchanged audio truth.
    pub text_joint_correct: PairRate,
    /// Both outputs abstain under text-only intervention.
    pub text_joint_abstention: PairRate,
    pub audio_relevant_pairs: usize,
    /// Both outputs equal their respective, different audio truths.
    pub audio_counterfactual_sensitivity: PairRate,
    /// Same non-abstaining claim despite a changed audio truth.
    pub audio_false_invariance: PairRate,
    /// Both outputs abstain despite a changed audio truth.
    pub audio_joint_abstention: PairRate,
    pub audio_nuisance_pairs: usize,
    /// Same claim among pairs where both sides claim.
    pub audio_nuisance_semantic_stability: PairRate,
    /// Both outputs equal the unchanged audio truth.
    pub audio_nuisance_joint_correct: PairRate,
    /// Exactly one side abstains.
    pub audio_nuisance_differential_abstention: PairRate,
    /// Both sides abstain under acoustic nuisance.
    pub audio_nuisance_joint_abstention: PairRate,
}

/// Invalid pair manifest, trial reference or intervention relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterfactualPairError {
    EmptyPairManifests,
    EmptyPairId,
    DuplicatePairId(String),
    InvalidTrialIndex,
    MissingTrial(String),
    SameTrialReference,
    ReusedTrial(String),
    PairIdMismatch,
    InvalidDigest,
    InvalidChangedDimensions,
    ChangedDimensionsMismatch,
    TextNuisanceRelationMismatch,
    TextNuisanceConditionsUnchanged,
    AudioRelevantAudioUnchanged,
    AudioRelevantTruthNotChanged,
    AudioRelevantConditionsChanged,
    AudioNuisanceAudioUnchanged,
    AudioNuisanceTruthChanged,
    AudioNuisanceConditionsChanged,
}

impl fmt::Display for CounterfactualPairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPairManifests => write!(f, "counterfactual pair manifest set is empty"),
            Self::EmptyPairId => write!(f, "counterfactual pair ID is empty"),
            Self::DuplicatePairId(id) => write!(f, "duplicate counterfactual pair ID: {id}"),
            Self::InvalidTrialIndex => write!(f, "trial set contains an empty or duplicate trial ID"),
            Self::MissingTrial(id) => write!(f, "missing trial: {id}"),
            Self::SameTrialReference => write!(f, "baseline and variant refer to the same trial"),
            Self::ReusedTrial(id) => write!(f, "trial reused by multiple pair roles: {id}"),
            Self::PairIdMismatch => write!(f, "trial pair ID differs from pair manifest"),
            Self::InvalidDigest => write!(f, "pair manifest contains invalid SHA-256"),
            Self::InvalidChangedDimensions => write!(f, "changed dimensions are empty or duplicated"),
            Self::ChangedDimensionsMismatch => write!(f, "changed dimensions do not match intervention kind"),
            Self::TextNuisanceRelationMismatch => write!(f, "text nuisance must preserve exact audio and truth"),
            Self::TextNuisanceConditionsUnchanged => write!(f, "text nuisance must change non-audio conditions"),
            Self::AudioRelevantAudioUnchanged => write!(f, "audio-relevant intervention must change audio"),
            Self::AudioRelevantTruthNotChanged => write!(f, "audio-relevant intervention requires distinct answerable truths"),
            Self::AudioRelevantConditionsChanged => write!(f, "audio-relevant intervention must hold non-audio conditions fixed"),
            Self::AudioNuisanceAudioUnchanged => write!(f, "audio nuisance must change audio"),
            Self::AudioNuisanceTruthChanged => write!(f, "audio nuisance must preserve truth"),
            Self::AudioNuisanceConditionsChanged => write!(f, "audio nuisance must hold non-audio conditions fixed"),
        }
    }
}

impl std::error::Error for CounterfactualPairError {}

/// Score nuisance invariance and causal evidence sensitivity.
///
/// Text-nuisance invariance is deliberately reported separately from joint
/// correctness: identical wrong answers are invariant but not grounded.
/// Audio-relevant sensitivity requires both outputs to match their distinct
/// adjudicated truths. Unanswerable truth transitions are excluded in v1.
pub fn score_counterfactual_pairs(
    trials: &[AcousticFaithfulnessTrial],
    pairs: &[CounterfactualPairManifest],
) -> Result<CounterfactualPairResult, CounterfactualPairError> {
    if pairs.is_empty() {
        return Err(CounterfactualPairError::EmptyPairManifests);
    }

    let mut index = HashMap::with_capacity(trials.len());
    for trial in trials {
        if trial.trial_id.trim().is_empty()
            || index.insert(trial.trial_id.as_str(), trial).is_some()
        {
            return Err(CounterfactualPairError::InvalidTrialIndex);
        }
    }

    let mut pair_ids = HashSet::with_capacity(pairs.len());
    let mut used_trials = HashSet::with_capacity(pairs.len() * 2);
    let mut text_n = 0;
    let mut text_invariant = 0;
    let mut text_correct = 0;
    let mut text_abstain = 0;
    let mut relevant_n = 0;
    let mut relevant_sensitive = 0;
    let mut relevant_false_invariant = 0;
    let mut relevant_both_claim = 0;
    let mut relevant_abstain = 0;
    let mut nuisance_n = 0;
    let mut nuisance_both_claim = 0;
    let mut nuisance_stable = 0;
    let mut nuisance_correct = 0;
    let mut nuisance_differential_abstain = 0;
    let mut nuisance_joint_abstain = 0;

    for pair in pairs {
        if pair.pair_id.trim().is_empty() {
            return Err(CounterfactualPairError::EmptyPairId);
        }
        if !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(CounterfactualPairError::DuplicatePairId(pair.pair_id.clone()));
        }
        if pair.baseline_trial_id == pair.variant_trial_id {
            return Err(CounterfactualPairError::SameTrialReference);
        }
        if !is_sha256(&pair.baseline_condition_manifest_sha256)
            || !is_sha256(&pair.variant_condition_manifest_sha256)
            || !is_sha256(&pair.intervention_manifest_sha256)
        {
            return Err(CounterfactualPairError::InvalidDigest);
        }
        if pair.changed_dimensions.is_empty()
            || pair.changed_dimensions.iter().copied().collect::<HashSet<_>>().len()
                != pair.changed_dimensions.len()
        {
            return Err(CounterfactualPairError::InvalidChangedDimensions);
        }

        let base = *index.get(pair.baseline_trial_id.as_str())
            .ok_or_else(|| CounterfactualPairError::MissingTrial(pair.baseline_trial_id.clone()))?;
        let variant = *index.get(pair.variant_trial_id.as_str())
            .ok_or_else(|| CounterfactualPairError::MissingTrial(pair.variant_trial_id.clone()))?;

        if base.pair_id != pair.pair_id || variant.pair_id != pair.pair_id {
            return Err(CounterfactualPairError::PairIdMismatch);
        }
        if !used_trials.insert(pair.baseline_trial_id.as_str()) {
            return Err(CounterfactualPairError::ReusedTrial(pair.baseline_trial_id.clone()));
        }
        if !used_trials.insert(pair.variant_trial_id.as_str()) {
            return Err(CounterfactualPairError::ReusedTrial(pair.variant_trial_id.clone()));
        }

        match pair.intervention_kind {
            CounterfactualInterventionKind::TextNuisance => {
                if pair.changed_dimensions.iter().any(|d| !matches!(d, ChangedDimension::ContextText | ChangedDimension::PromptText)) {
                    return Err(CounterfactualPairError::ChangedDimensionsMismatch);
                }
                if base.source_audio_sha256 != variant.source_audio_sha256 || base.audio_truth != variant.audio_truth {
                    return Err(CounterfactualPairError::TextNuisanceRelationMismatch);
                }
                if pair.baseline_condition_manifest_sha256 == pair.variant_condition_manifest_sha256 {
                    return Err(CounterfactualPairError::TextNuisanceConditionsUnchanged);
                }
                text_n += 1;
                if base.outcome == variant.outcome { text_invariant += 1; }
                if both_match_truth(base, variant) { text_correct += 1; }
                if matches!(base.outcome, ModelOutcome::Abstain) && matches!(variant.outcome, ModelOutcome::Abstain) {
                    text_abstain += 1;
                }
            }
            CounterfactualInterventionKind::AudioRelevant => {
                if pair.changed_dimensions.iter().any(|d| !matches!(
                    d,
                    ChangedDimension::LexicalContent | ChangedDimension::Negation
                        | ChangedDimension::NumberOrEntity | ChangedDimension::EventPresence
                        | ChangedDimension::TemporalOrder
                )) {
                    return Err(CounterfactualPairError::ChangedDimensionsMismatch);
                }
                if base.source_audio_sha256 == variant.source_audio_sha256 {
                    return Err(CounterfactualPairError::AudioRelevantAudioUnchanged);
                }
                let (AudioTruth::SupportedClaim(base_truth), AudioTruth::SupportedClaim(variant_truth)) =
                    (&base.audio_truth, &variant.audio_truth)
                else {
                    return Err(CounterfactualPairError::AudioRelevantTruthNotChanged);
                };
                if base_truth == variant_truth {
                    return Err(CounterfactualPairError::AudioRelevantTruthNotChanged);
                }
                if pair.baseline_condition_manifest_sha256 != pair.variant_condition_manifest_sha256 {
                    return Err(CounterfactualPairError::AudioRelevantConditionsChanged);
                }
                relevant_n += 1;
                if both_match_truth(base, variant) { relevant_sensitive += 1; }
                let both_claim = matches!(base.outcome, ModelOutcome::Claim(_))
                    && matches!(variant.outcome, ModelOutcome::Claim(_));
                if both_claim {
                    relevant_both_claim += 1;
                    if let (ModelOutcome::Claim(left), ModelOutcome::Claim(right)) = (&base.outcome, &variant.outcome) {
                        if left == right { relevant_false_invariant += 1; }
                    }
                }
                if matches!(base.outcome, ModelOutcome::Abstain) && matches!(variant.outcome, ModelOutcome::Abstain) {
                    relevant_abstain += 1;
                }
            }
            CounterfactualInterventionKind::AudioNuisance => {
                if pair.changed_dimensions.iter().any(|d| *d != ChangedDimension::AcousticNuisance) {
                    return Err(CounterfactualPairError::ChangedDimensionsMismatch);
                }
                if base.source_audio_sha256 == variant.source_audio_sha256 {
                    return Err(CounterfactualPairError::AudioNuisanceAudioUnchanged);
                }
                if base.audio_truth != variant.audio_truth {
                    return Err(CounterfactualPairError::AudioNuisanceTruthChanged);
                }
                if pair.baseline_condition_manifest_sha256 != pair.variant_condition_manifest_sha256 {
                    return Err(CounterfactualPairError::AudioNuisanceConditionsChanged);
                }
                nuisance_n += 1;
                let both_claim = matches!(base.outcome, ModelOutcome::Claim(_))
                    && matches!(variant.outcome, ModelOutcome::Claim(_));
                if both_claim {
                    nuisance_both_claim += 1;
                    if base.outcome == variant.outcome { nuisance_stable += 1; }
                }
                if both_match_truth(base, variant) { nuisance_correct += 1; }
                let base_abstain = matches!(base.outcome, ModelOutcome::Abstain);
                let variant_abstain = matches!(variant.outcome, ModelOutcome::Abstain);
                if base_abstain != variant_abstain {
                    nuisance_differential_abstain += 1;
                } else if base_abstain {
                    nuisance_joint_abstain += 1;
                }
            }
        }
    }

    Ok(CounterfactualPairResult {
        total_pairs: pairs.len(),
        text_nuisance_pairs: text_n,
        text_response_invariance: PairRate::new(text_invariant, text_n),
        text_joint_correct: PairRate::new(text_correct, text_n),
        text_joint_abstention: PairRate::new(text_abstain, text_n),
        audio_relevant_pairs: relevant_n,
        audio_counterfactual_sensitivity: PairRate::new(relevant_sensitive, relevant_n),
        audio_false_invariance: PairRate::new(relevant_false_invariant, relevant_both_claim),
        audio_joint_abstention: PairRate::new(relevant_abstain, relevant_n),
        audio_nuisance_pairs: nuisance_n,
        audio_nuisance_semantic_stability: PairRate::new(nuisance_stable, nuisance_both_claim),
        audio_nuisance_joint_correct: PairRate::new(nuisance_correct, nuisance_n),
        audio_nuisance_differential_abstention: PairRate::new(nuisance_differential_abstain, nuisance_n),
        audio_nuisance_joint_abstention: PairRate::new(nuisance_joint_abstain, nuisance_n),
    })
}

fn both_match_truth(
    baseline: &AcousticFaithfulnessTrial,
    variant: &AcousticFaithfulnessTrial,
) -> bool {
    matches!(
        (&baseline.audio_truth, &baseline.outcome),
        (AudioTruth::SupportedClaim(truth), ModelOutcome::Claim(output)) if truth == output
    ) && matches!(
        (&variant.audio_truth, &variant.outcome),
        (AudioTruth::SupportedClaim(truth), ModelOutcome::Claim(output)) if truth == output
    )
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::{
        ChangedDimension, CounterfactualInterventionKind, CounterfactualPairError,
        CounterfactualPairManifest, score_counterfactual_pairs,
    };
    use crate::acoustic_faithfulness::{AcousticFaithfulnessTrial, AudioTruth, ConflictAxis, ModelOutcome};

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn t(id: &str, pair: &str, audio: &str, truth: &str, outcome: ModelOutcome) -> AcousticFaithfulnessTrial {
        AcousticFaithfulnessTrial {
            trial_id: id.into(), pair_id: pair.into(), source_audio_sha256: audio.into(),
            audio_truth: AudioTruth::SupportedClaim(truth.into()),
            context_claim: None, prompt_claim: None, conflict_axis: ConflictAxis::None,
            outcome, evidence_span_present: true,
        }
    }

    fn p(id: &str, kind: CounterfactualInterventionKind, baseline: &str, variant: &str,
        base_condition: &str, variant_condition: &str, changed: Vec<ChangedDimension>) -> CounterfactualPairManifest {
        CounterfactualPairManifest {
            pair_id: id.into(), baseline_trial_id: baseline.into(), variant_trial_id: variant.into(),
            intervention_kind: kind, baseline_condition_manifest_sha256: base_condition.into(),
            variant_condition_manifest_sha256: variant_condition.into(),
            intervention_manifest_sha256: C.into(), changed_dimensions: changed,
        }
    }

    #[test]
    fn text_nuisance_same_correct_answer_is_invariant_and_correct() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",A,"B",ModelOutcome::Claim("B".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::ContextText])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.text_response_invariance.rate,Some(1.0));
        assert_eq!(r.text_joint_correct.rate,Some(1.0));
    }

    #[test]
    fn text_prompt_override_reduces_invariance_and_correctness() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",A,"B",ModelOutcome::Claim("A".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::PromptText])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.text_response_invariance.rate,Some(0.0));
        assert_eq!(r.text_joint_correct.rate,Some(0.0));
    }

    #[test]
    fn identical_wrong_answers_are_invariant_but_not_grounded() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("A".into())), t("b","p",A,"B",ModelOutcome::Claim("A".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::ContextText])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.text_response_invariance.rate,Some(1.0));
        assert_eq!(r.text_joint_correct.rate,Some(0.0));
    }

    #[test]
    fn paired_abstention_is_separate_from_joint_correctness() {
        let trials = [t("a","p",A,"B",ModelOutcome::Abstain), t("b","p",A,"B",ModelOutcome::Abstain)];
        let pairs = [p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::PromptText])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.text_response_invariance.rate,Some(1.0));
        assert_eq!(r.text_joint_abstention.rate,Some(1.0));
        assert_eq!(r.text_joint_correct.rate,Some(0.0));
    }

    #[test]
    fn audio_relevant_pair_requires_correct_distinct_responses() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",B,"C",ModelOutcome::Claim("C".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::AudioRelevant,"a","b",A,A,vec![ChangedDimension::Negation])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.audio_counterfactual_sensitivity.rate,Some(1.0));
        assert_eq!(r.audio_false_invariance.rate,Some(0.0));
    }

    #[test]
    fn same_wrong_claim_on_changed_audio_is_false_invariance() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("A".into())), t("b","p",B,"C",ModelOutcome::Claim("A".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::AudioRelevant,"a","b",A,A,vec![ChangedDimension::LexicalContent])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.audio_counterfactual_sensitivity.rate,Some(0.0));
        assert_eq!(r.audio_false_invariance.rate,Some(1.0));
    }

    #[test]
    fn both_abstain_on_audio_relevant_pair_is_not_sensitivity() {
        let trials = [t("a","p",A,"B",ModelOutcome::Abstain), t("b","p",B,"C",ModelOutcome::Abstain)];
        let pairs = [p("p",CounterfactualInterventionKind::AudioRelevant,"a","b",A,A,vec![ChangedDimension::EventPresence])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.audio_counterfactual_sensitivity.rate,Some(0.0));
        assert_eq!(r.audio_joint_abstention.rate,Some(1.0));
        assert_eq!(r.audio_false_invariance.rate,None);
    }

    #[test]
    fn acoustic_nuisance_stability_is_conditioned_on_two_claims() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",B,"B",ModelOutcome::Claim("B".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::AudioNuisance,"a","b",A,A,vec![ChangedDimension::AcousticNuisance])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.audio_nuisance_semantic_stability.rate,Some(1.0));
        assert_eq!(r.audio_nuisance_joint_correct.rate,Some(1.0));
    }

    #[test]
    fn differential_abstention_under_noise_is_reported() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",B,"B",ModelOutcome::Abstain)];
        let pairs = [p("p",CounterfactualInterventionKind::AudioNuisance,"a","b",A,A,vec![ChangedDimension::AcousticNuisance])];
        let r = score_counterfactual_pairs(&trials,&pairs).unwrap();
        assert_eq!(r.audio_nuisance_differential_abstention.rate,Some(1.0));
        assert_eq!(r.audio_nuisance_joint_correct.rate,Some(0.0));
    }

    #[test]
    fn rejects_text_nuisance_when_audio_or_truth_changes() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",B,"B",ModelOutcome::Claim("B".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::ContextText])];
        assert_eq!(score_counterfactual_pairs(&trials,&pairs),Err(CounterfactualPairError::TextNuisanceRelationMismatch));
    }

    #[test]
    fn rejects_audio_relevant_pair_if_condition_manifest_changes() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",B,"C",ModelOutcome::Claim("C".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::AudioRelevant,"a","b",A,B,vec![ChangedDimension::LexicalContent])];
        assert_eq!(score_counterfactual_pairs(&trials,&pairs),Err(CounterfactualPairError::AudioRelevantConditionsChanged));
    }

    #[test]
    fn rejects_audio_nuisance_pair_if_truth_changes() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",B,"C",ModelOutcome::Claim("C".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::AudioNuisance,"a","b",A,A,vec![ChangedDimension::AcousticNuisance])];
        assert_eq!(score_counterfactual_pairs(&trials,&pairs),Err(CounterfactualPairError::AudioNuisanceTruthChanged));
    }

    #[test]
    fn rejects_duplicate_pair_ids() {
        let trials = [
            t("a","p",A,"B",ModelOutcome::Claim("B".into())), t("b","p",A,"B",ModelOutcome::Claim("B".into())),
            t("c","q",A,"B",ModelOutcome::Claim("B".into())), t("d","q",A,"B",ModelOutcome::Claim("B".into()))
        ];
        let pairs = [
            p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::ContextText]),
            p("p",CounterfactualInterventionKind::TextNuisance,"c","d",A,B,vec![ChangedDimension::ContextText])
        ];
        assert_eq!(score_counterfactual_pairs(&trials,&pairs),Err(CounterfactualPairError::DuplicatePairId("p".into())));
    }

    #[test]
    fn rejects_reused_trial_across_pairs() {
        let trials = [
            t("a","p1",A,"B",ModelOutcome::Claim("B".into())), t("b","p1",A,"B",ModelOutcome::Claim("B".into())),
            t("c","p2",A,"B",ModelOutcome::Claim("B".into()))
        ];
        let pairs = [
            p("p1",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::ContextText]),
            p("p2",CounterfactualInterventionKind::TextNuisance,"a","c",A,B,vec![ChangedDimension::ContextText])
        ];
        assert_eq!(score_counterfactual_pairs(&trials,&pairs),Err(CounterfactualPairError::ReusedTrial("a".into())));
    }

    #[test]
    fn rejects_missing_trial() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into()))];
        let pairs = [p("p",CounterfactualInterventionKind::TextNuisance,"a","missing",A,B,vec![ChangedDimension::ContextText])];
        assert_eq!(score_counterfactual_pairs(&trials,&pairs),Err(CounterfactualPairError::MissingTrial("missing".into())));
    }

    #[test]
    fn rejects_bad_intervention_digest() {
        let trials = [t("a","p",A,"B",ModelOutcome::Claim("B".into())),t("b","p",A,"B",ModelOutcome::Claim("B".into()))];
        let mut pair = p("p",CounterfactualInterventionKind::TextNuisance,"a","b",A,B,vec![ChangedDimension::PromptText]);
        pair.intervention_manifest_sha256 = "bad".into();
        assert_eq!(score_counterfactual_pairs(&trials,&[pair]),Err(CounterfactualPairError::InvalidDigest));
    }

    #[test]
    fn rejects_unanswerable_audio_relevant_transition() {
        let mut baseline = t("a","p",A,"B",ModelOutcome::Claim("B".into()));
        baseline.audio_truth = AudioTruth::Unanswerable;
        let variant = t("b","p",B,"C",ModelOutcome::Claim("C".into()));
        let pairs = [p("p",CounterfactualInterventionKind::AudioRelevant,"a","b",A,A,vec![ChangedDimension::EventPresence])];
        assert_eq!(score_counterfactual_pairs(&[baseline,variant],&pairs),Err(CounterfactualPairError::AudioRelevantTruthNotChanged));
    }

    #[test]
    fn empty_manifest_set_is_rejected() {
        assert_eq!(score_counterfactual_pairs(&[],&[]),Err(CounterfactualPairError::EmptyPairManifests));
    }
}
