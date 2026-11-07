// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Multi-trial evaluation harness for the finite-hypothesis three-ball policy.
//!
//! This module repeatedly applies the existing two-step receding-horizon policy:
//!
//!   gaze -> observe one colour -> choose -> receive feedback.
//!
//! The joint rule/state posterior is preserved across trials, while the hidden
//! ball arrangement is reset from a fresh uniform prior for each trial. A
//! deterministic SplitMix64 stream supplies hidden scenes. The true rule and
//! hidden scenes are retained in evaluator receipts but are never passed to
//! the action-selection policy.
//!
//! This is a repeated two-step finite-hypothesis adaptation, NOT an exact
//! reproduction of the six-time-point SPM POMDP. It does not maintain the
//! continuous Dirichlet parameter posterior used by the source SPM agent.
//! Parameter-learning and source-faithful Dirichlet BMR remain separate paths.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::three_ball_partial_observation::{
    ThreeBallJointBelief, ThreeBallPartialObservationConfig,
    ThreeBallPartialObservationError, score_three_ball_gaze_policies,
    uniform_three_ball_state_prior,
};
use crate::three_ball_rule_space::{
    ThreeBallColour, ThreeBallFeedback, ThreeBallLocation, ThreeBallRuleSpace,
};

/// Model-family representation used for the sequential episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreeBallModelFamily {
    /// Preserve all 81 source-labelled rule hypotheses.
    SourceLabelled,
    /// Collapse identical feedback tensors and sum their prior mass.
    SemanticQuotient,
}

/// Configuration for one repeated two-step discovery episode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallSequentialDiscoveryConfig {
    /// Maximum number of independent hidden-scene trials.
    pub max_trials: usize,
    /// The source SPM Occam's-window criterion is a log odds threshold of 16 nats.
    pub commitment_log_odds_threshold_nats: f64,
    /// Preferences used while the agent is still gathering rule evidence.
    /// The source DEM_reasoning switches to neutral preferences before
    /// commitment, then restores [0, 2, -6] after commitment.
    pub discovery_feedback_log_preferences_nats: [f64; 3],
    /// Preferences restored after model commitment.
    pub post_commitment_feedback_log_preferences_nats: [f64; 3],
    /// FEP weights, costs, and other policy parameters.
    pub policy: ThreeBallPartialObservationConfig,
    /// Preserve all source labels or run over the semantic quotient.
    pub model_family: ThreeBallModelFamily,
}

impl Default for ThreeBallSequentialDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_trials: 64,
            commitment_log_odds_threshold_nats: 16.0,
            discovery_feedback_log_preferences_nats: [0.0; 3],
            post_commitment_feedback_log_preferences_nats: [0.0, 2.0, -6.0],
            policy: ThreeBallPartialObservationConfig::default(),
            model_family: ThreeBallModelFamily::SemanticQuotient,
        }
    }
}

/// One evaluator-visible receipt for a completed trial.
///
/// `hidden_state_for_audit` is environment ground truth, not an agent input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallSequentialTrialReceipt {
    pub trial_index: usize,
    pub hidden_state_for_audit: [ThreeBallColour; 3],
    pub selected_gaze: Option<ThreeBallLocation>,
    pub observed_colour: Option<ThreeBallColour>,
    pub selected_choice: Option<ThreeBallColour>,
    pub observed_feedback: ThreeBallFeedback,
    pub model_information_gain_nats: f64,
    pub conditional_state_information_gain_nats: f64,
    pub model_entropy_before_nats: f64,
    pub model_entropy_after_nats: f64,
    pub true_semantic_class_posterior_after: f64,
    pub winning_semantic_class_one_based_index_after: usize,
    pub winning_semantic_class_posterior_after: f64,
    /// None represents +infinity when the winning semantic class has posterior 1.
    pub winning_model_log_odds_after_nats: Option<f64>,
    pub cumulative_reward: i64,
    pub preferences_restored: bool,
    pub committed_after_trial: bool,
    pub correct_commitment_after_trial: bool,
}

/// Summary from one deterministic sequential discovery episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallSequentialDiscoveryResult {
    pub format_version: u32,
    pub seed: u64,
    pub true_rule_one_based_index: usize,
    pub true_semantic_class_one_based_index: usize,
    pub model_family: ThreeBallModelFamily,
    pub trial_receipts: Vec<ThreeBallSequentialTrialReceipt>,
    pub trials_to_model_commitment: Option<usize>,
    pub trials_to_correct_rule_commitment: Option<usize>,
    pub final_model_entropy_nats: f64,
    pub final_true_semantic_class_posterior: f64,
    pub final_winning_semantic_class_one_based_index: usize,
    pub final_winning_semantic_class_posterior: f64,
    pub mean_true_class_brier_score: f64,
    pub mean_true_class_negative_log_probability: f64,
    pub correct_feedback_count: usize,
    pub incorrect_feedback_count: usize,
    pub no_feedback_count: usize,
    pub gaze_count: usize,
    pub cumulative_reward: i64,
    /// Final model marginals in the selected model-family ordering.
    pub final_model_posterior: Vec<(String, f64)>,
}

/// Errors from the multi-trial evaluator.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ThreeBallSequentialDiscoveryError {
    #[error(transparent)]
    PartialObservation(#[from] ThreeBallPartialObservationError),
    #[error("max_trials must be greater than zero")]
    EmptyTrialBudget,
    #[error("commitment log-odds threshold must be finite and non-negative")]
    InvalidCommitmentThreshold,
    #[error("true rule one-based index {index} is outside 1..={rule_count}")]
    InvalidTrueRuleIndex { index: usize, rule_count: usize },
    #[error("model family has no posterior support for the true semantic rule")]
    TrueRuleMissingFromModelFamily,
    #[error("selected action index {index} is outside 0..{action_count}")]
    InvalidSelectedAction { index: usize, action_count: usize },
    #[error("selected gaze has no contingency for the observed colour")]
    MissingGazeContingency,
    #[error("semantic model posterior is empty or not normalized")]
    InvalidSemanticModelPosterior,
}

/// Run a seeded multi-trial discovery episode.
///
/// The true rule is specified using the source's one-based rule index. On each
/// trial the environment independently samples left/centre/right colours; the
/// agent sees only the selected gaze outcome, then receives feedback for its
/// chosen colour. The posterior over rules is carried into the next trial,
/// while the next hidden-scene prior is reset to uniform.
pub fn run_three_ball_sequential_discovery(
    space: &ThreeBallRuleSpace,
    true_rule_one_based_index: usize,
    seed: u64,
    config: ThreeBallSequentialDiscoveryConfig,
) -> Result<ThreeBallSequentialDiscoveryResult, ThreeBallSequentialDiscoveryError> {
    if config.max_trials == 0 {
        return Err(ThreeBallSequentialDiscoveryError::EmptyTrialBudget);
    }
    if !config.commitment_log_odds_threshold_nats.is_finite()
        || config.commitment_log_odds_threshold_nats < 0.0
    {
        return Err(ThreeBallSequentialDiscoveryError::InvalidCommitmentThreshold);
    }
    let true_rule_index = true_rule_one_based_index.checked_sub(1).ok_or(
        ThreeBallSequentialDiscoveryError::InvalidTrueRuleIndex {
            index: true_rule_one_based_index,
            rule_count: space.rules.len(),
        },
    )?;
    let true_rule = space.rules.get(true_rule_index).ok_or(
        ThreeBallSequentialDiscoveryError::InvalidTrueRuleIndex {
            index: true_rule_one_based_index,
            rule_count: space.rules.len(),
        },
    )?;
    let true_semantic_class_one_based_index = semantic_representative(true_rule);

    let mut belief = match config.model_family {
        ThreeBallModelFamily::SourceLabelled => {
            ThreeBallJointBelief::uniform_source_prior(space)?
        }
        ThreeBallModelFamily::SemanticQuotient => ThreeBallJointBelief::from_semantic_quotient_prior(
            space,
            &uniform_three_ball_state_prior(),
        )?,
    };
    let initial_semantic_posterior = semantic_model_posterior(space, &belief)?;
    if !initial_semantic_posterior.contains_key(&true_semantic_class_one_based_index) {
        return Err(ThreeBallSequentialDiscoveryError::TrueRuleMissingFromModelFamily);
    }

    let mut rng = SplitMix64::new(seed);
    let mut trial_receipts = Vec::with_capacity(config.max_trials);
    let mut trials_to_model_commitment = None;
    let mut trials_to_correct_rule_commitment = None;
    let mut correct_feedback_count = 0;
    let mut incorrect_feedback_count = 0;
    let mut no_feedback_count = 0;
    let mut gaze_count = 0;
    let mut cumulative_reward = 0_i64;
    let mut brier_total = 0.0;
    let mut negative_log_total = 0.0;
    let mut preferences_restored = false;

    for trial_index in 1..=config.max_trials {
        let hidden_state = [
            colour_from_index(rng.below(3)),
            colour_from_index(rng.below(3)),
            colour_from_index(rng.below(3)),
        ];

        let model_posterior_before = semantic_model_posterior(space, &belief)?;
        let model_entropy_before = entropy(&model_posterior_before);
        let (_, max_probability_before) = winning_semantic_class(&model_posterior_before)?;
        preferences_restored |=
            exceeds_log_odds_threshold(max_probability_before, config.commitment_log_odds_threshold_nats);

        let mut trial_policy = config.policy;
        trial_policy.feedback_log_preferences_nats = if preferences_restored {
            config.post_commitment_feedback_log_preferences_nats
        } else {
            config.discovery_feedback_log_preferences_nats
        };

        let decision = score_three_ball_gaze_policies(space, &belief, trial_policy)?;
        let action = decision
            .action_scores
            .get(decision.selected_action_index)
            .ok_or(ThreeBallSequentialDiscoveryError::InvalidSelectedAction {
                index: decision.selected_action_index,
                action_count: decision.action_scores.len(),
            })?;

        let selected_gaze = action.location;
        let (belief_after_gaze, observed_colour, selected_choice) =
            if let Some(location) = selected_gaze {
                gaze_count += 1;
                let colour = hidden_state[location as usize];
                let posterior = belief.condition_on_gaze(location, colour)?;
                let contingency = action
                    .contingencies
                    .iter()
                    .find(|entry| entry.observed_colour == colour)
                    .ok_or(ThreeBallSequentialDiscoveryError::MissingGazeContingency)?;
                (posterior, Some(colour), contingency.selected_choice)
            } else {
                (belief.clone(), None, action.immediate_choice)
            };

        let feedback = true_rule.feedback(hidden_state, selected_choice);
        match feedback {
            ThreeBallFeedback::Correct => {
                correct_feedback_count += 1;
                cumulative_reward += 1;
            }
            ThreeBallFeedback::Incorrect => {
                incorrect_feedback_count += 1;
                cumulative_reward -= 1;
            }
            ThreeBallFeedback::NoFeedback => no_feedback_count += 1,
        }

        let posterior_after_feedback =
            belief_after_gaze.condition_on_feedback(space, selected_choice, feedback)?;
        let model_posterior_after = semantic_model_posterior(space, &posterior_after_feedback)?;
        let model_entropy_after = entropy(&model_posterior_after);
        let true_probability = *model_posterior_after
            .get(&true_semantic_class_one_based_index)
            .ok_or(ThreeBallSequentialDiscoveryError::TrueRuleMissingFromModelFamily)?;
        let (winning_rule, winning_probability) =
            winning_semantic_class(&model_posterior_after)?;
        let winning_log_odds = finite_log_odds(winning_probability);
        let committed_after_trial = exceeds_log_odds_threshold(
            winning_probability,
            config.commitment_log_odds_threshold_nats,
        );
        let correct_commitment_after_trial = committed_after_trial
            && winning_rule == true_semantic_class_one_based_index;

        if committed_after_trial && trials_to_model_commitment.is_none() {
            trials_to_model_commitment = Some(trial_index);
        }
        if correct_commitment_after_trial && trials_to_correct_rule_commitment.is_none() {
            trials_to_correct_rule_commitment = Some(trial_index);
        }

        let class_count = model_posterior_after.len() as f64;
        let brier = model_posterior_after
            .iter()
            .map(|(class_index, probability)| {
                let expected = f64::from(*class_index == true_semantic_class_one_based_index);
                (probability - expected).powi(2)
            })
            .sum::<f64>();
        brier_total += brier;
        negative_log_total += -true_probability.max(f64::MIN_POSITIVE).ln();

        trial_receipts.push(ThreeBallSequentialTrialReceipt {
            trial_index,
            hidden_state_for_audit: hidden_state,
            selected_gaze,
            observed_colour,
            selected_choice,
            observed_feedback: feedback,
            model_information_gain_nats: action.model_information_gain_nats,
            conditional_state_information_gain_nats: action
                .conditional_state_information_gain_nats,
            model_entropy_before_nats: model_entropy_before,
            model_entropy_after_nats: model_entropy_after,
            true_semantic_class_posterior_after: true_probability,
            winning_semantic_class_one_based_index_after: winning_rule,
            winning_semantic_class_posterior_after: winning_probability,
            winning_model_log_odds_after_nats: winning_log_odds,
            cumulative_reward,
            preferences_restored,
            committed_after_trial,
            correct_commitment_after_trial,
        });

        // A new trial draws a new hidden scene. Preserve model beliefs but
        // remove state information specific to the previous scene.
        let model_marginals = posterior_after_feedback.model_marginals();
        belief = ThreeBallJointBelief::from_model_and_state_priors(
            space,
            posterior_after_feedback.model_ids.clone(),
            posterior_after_feedback.rule_indices.clone(),
            &model_marginals,
            &uniform_three_ball_state_prior(),
        )?;
    }

    let final_semantic_posterior = semantic_model_posterior(space, &belief)?;
    let final_model_entropy_nats = entropy(&final_semantic_posterior);
    let final_true_semantic_class_posterior = *final_semantic_posterior
        .get(&true_semantic_class_one_based_index)
        .ok_or(ThreeBallSequentialDiscoveryError::TrueRuleMissingFromModelFamily)?;
    let (final_winner, final_winner_probability) =
        winning_semantic_class(&final_semantic_posterior)?;

    Ok(ThreeBallSequentialDiscoveryResult {
        format_version: 1,
        seed,
        true_rule_one_based_index,
        true_semantic_class_one_based_index,
        model_family: config.model_family,
        trial_receipts,
        trials_to_model_commitment,
        trials_to_correct_rule_commitment,
        final_model_entropy_nats,
        final_true_semantic_class_posterior,
        final_winning_semantic_class_one_based_index: final_winner,
        final_winning_semantic_class_posterior: final_winner_probability,
        mean_true_class_brier_score: brier_total / config.max_trials as f64,
        mean_true_class_negative_log_probability: negative_log_total / config.max_trials as f64,
        correct_feedback_count,
        incorrect_feedback_count,
        no_feedback_count,
        gaze_count,
        cumulative_reward,
        final_model_posterior: final_semantic_posterior
            .into_iter()
            .map(|(class_index, probability)| {
                (format!("three-ball-semantic-rule-{class_index:02}"), probability)
            })
            .collect(),
    })
}

fn semantic_model_posterior(
    space: &ThreeBallRuleSpace,
    belief: &ThreeBallJointBelief,
) -> Result<BTreeMap<usize, f64>, ThreeBallSequentialDiscoveryError> {
    belief.validate()?;
    if belief.rule_indices.len() != belief.model_ids.len() {
        return Err(ThreeBallSequentialDiscoveryError::InvalidSemanticModelPosterior);
    }
    let marginals = belief.model_marginals();
    if marginals.len() != belief.rule_indices.len() {
        return Err(ThreeBallSequentialDiscoveryError::InvalidSemanticModelPosterior);
    }
    let mut posterior = BTreeMap::<usize, f64>::new();
    for (rule_index, probability) in belief.rule_indices.iter().copied().zip(marginals) {
        let rule = space.rules.get(rule_index).ok_or(
            ThreeBallSequentialDiscoveryError::InvalidTrueRuleIndex {
                index: rule_index + 1,
                rule_count: space.rules.len(),
            },
        )?;
        *posterior.entry(semantic_representative(rule)).or_default() += probability;
    }
    let total = posterior.values().sum::<f64>();
    if posterior.is_empty()
        || !total.is_finite()
        || (total - 1.0).abs() > 1e-9
        || posterior
            .values()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(ThreeBallSequentialDiscoveryError::InvalidSemanticModelPosterior);
    }
    Ok(posterior)
}

fn semantic_representative(rule: &crate::three_ball_rule_space::ThreeBallRule) -> usize {
    rule.duplicate_of_one_based_index
        .unwrap_or(rule.one_based_index)
}

fn winning_semantic_class(
    posterior: &BTreeMap<usize, f64>,
) -> Result<(usize, f64), ThreeBallSequentialDiscoveryError> {
    posterior
        .iter()
        .map(|(rule, probability)| (*rule, *probability))
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                // Deterministically prefer the lower source representative on ties.
                .then_with(|| right.0.cmp(&left.0))
        })
        .ok_or(ThreeBallSequentialDiscoveryError::InvalidSemanticModelPosterior)
}

fn entropy(posterior: &BTreeMap<usize, f64>) -> f64 {
    posterior
        .values()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn finite_log_odds(probability: f64) -> Option<f64> {
    if probability >= 1.0 {
        None
    } else if probability <= 0.0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((probability / (1.0 - probability)).ln())
    }
}

fn exceeds_log_odds_threshold(probability: f64, threshold: f64) -> bool {
    probability >= 1.0
        || finite_log_odds(probability).is_some_and(|log_odds| log_odds > threshold)
}

fn colour_from_index(index: usize) -> ThreeBallColour {
    match index {
        0 => ThreeBallColour::Red,
        1 => ThreeBallColour::Green,
        2 => ThreeBallColour::Blue,
        _ => unreachable!("three-ball colour index is in 0..3"),
    }
}

/// Small deterministic generator used to make this benchmark independent of
/// platform RNG implementation changes and external random-number crates.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_bound: u64) -> usize {
        debug_assert!(upper_bound > 0);
        // Rejection sampling avoids modulo bias for the small categorical domain.
        let threshold = upper_bound.wrapping_neg() % upper_bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % upper_bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::three_ball_rule_space::build_three_ball_rule_space;

    fn discovery_config(model_family: ThreeBallModelFamily) -> ThreeBallSequentialDiscoveryConfig {
        ThreeBallSequentialDiscoveryConfig {
            model_family,
            ..ThreeBallSequentialDiscoveryConfig::default()
        }
    }

    #[test]
    fn seeded_episode_is_bitwise_reproducible() {
        let space = build_three_ball_rule_space();
        let config = discovery_config(ThreeBallModelFamily::SemanticQuotient);
        let first =
            run_three_ball_sequential_discovery(&space, 65, 0x5eed, config).expect("first episode");
        let second =
            run_three_ball_sequential_discovery(&space, 65, 0x5eed, config).expect("second episode");
        assert_eq!(first, second);
        assert_eq!(first.trial_receipts.len(), config.max_trials);
        assert_eq!(first.trial_receipts[0].trial_index, 1);
        assert_eq!(first.trial_receipts.last().unwrap().trial_index, config.max_trials);
    }

    #[test]
    fn episode_preserves_model_beliefs_but_resets_each_hidden_scene() {
        let space = build_three_ball_rule_space();
        let config = ThreeBallSequentialDiscoveryConfig {
            max_trials: 8,
            ..discovery_config(ThreeBallModelFamily::SemanticQuotient)
        };
        let result =
            run_three_ball_sequential_discovery(&space, 65, 17, config).expect("episode");
        assert!(result.trial_receipts.iter().all(|trial| {
            trial.hidden_state_for_audit.iter().all(|colour| {
                matches!(
                    colour,
                    ThreeBallColour::Red | ThreeBallColour::Green | ThreeBallColour::Blue
                )
            })
        }));
        // A particular observation can increase posterior entropy; only
        // the policy's expected entropy change is constrained by mutual information.
        assert!(result.trial_receipts.iter().all(|trial| {
            trial.model_entropy_before_nats.is_finite()
                && trial.model_entropy_after_nats.is_finite()
        }));
        assert!(result.final_model_entropy_nats.is_finite());
    }

    #[test]
    fn source_labelled_and_semantic_quotient_episodes_preserve_truth_mass() {
        let space = build_three_ball_rule_space();
        let source = run_three_ball_sequential_discovery(
            &space,
            65,
            1234,
            discovery_config(ThreeBallModelFamily::SourceLabelled),
        )
        .expect("source-labelled episode");
        let quotient = run_three_ball_sequential_discovery(
            &space,
            65,
            1234,
            discovery_config(ThreeBallModelFamily::SemanticQuotient),
        )
        .expect("quotient episode");
        assert!(
            (source.final_true_semantic_class_posterior
                - quotient.final_true_semantic_class_posterior)
                .abs()
                < 1e-9
        );
        assert!(
            (source.final_model_entropy_nats - quotient.final_model_entropy_nats).abs() < 1e-9
        );
        assert_eq!(
            source.trials_to_correct_rule_commitment,
            quotient.trials_to_correct_rule_commitment
        );
    }

    #[test]
    fn neutral_discovery_preferences_and_model_eig_can_reduce_rule_uncertainty() {
        let space = build_three_ball_rule_space();
        let config = ThreeBallSequentialDiscoveryConfig {
            max_trials: 64,
            discovery_feedback_log_preferences_nats: [0.0; 3],
            model_family: ThreeBallModelFamily::SemanticQuotient,
            ..ThreeBallSequentialDiscoveryConfig::default()
        };
        let result =
            run_three_ball_sequential_discovery(&space, 65, 0x2026_1107, config)
                .expect("seeded rule discovery");
        let initial_entropy = (space.semantic_unique_rule_count as f64).ln();
        assert!(
            result.final_model_entropy_nats < initial_entropy,
            "rule posterior did not sharpen: final entropy={}, initial={}",
            result.final_model_entropy_nats,
            initial_entropy
        );
        assert!(result.mean_true_class_brier_score.is_finite());
        assert!(result.mean_true_class_negative_log_probability.is_finite());
    }

    #[test]
    fn invalid_episode_configuration_and_truth_fail_closed() {
        let space = build_three_ball_rule_space();
        assert!(matches!(
            run_three_ball_sequential_discovery(
                &space,
                65,
                1,
                ThreeBallSequentialDiscoveryConfig {
                    max_trials: 0,
                    ..Default::default()
                }
            ),
            Err(ThreeBallSequentialDiscoveryError::EmptyTrialBudget)
        ));
        assert!(matches!(
            run_three_ball_sequential_discovery(&space, 0, 1, Default::default()),
            Err(ThreeBallSequentialDiscoveryError::InvalidTrueRuleIndex { .. })
        ));
        assert!(matches!(
            run_three_ball_sequential_discovery(
                &space,
                65,
                1,
                ThreeBallSequentialDiscoveryConfig {
                    commitment_log_odds_threshold_nats: f64::NAN,
                    ..Default::default()
                }
            ),
            Err(ThreeBallSequentialDiscoveryError::InvalidCommitmentThreshold)
        ));
    }
}
