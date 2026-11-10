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
use std::collections::{BTreeMap, BTreeSet};

use crate::three_ball_partial_observation::{
    ThreeBallJointBelief, ThreeBallPartialObservationConfig, ThreeBallPartialObservationError,
    score_three_ball_gaze_policies, uniform_three_ball_state_prior,
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

/// Action-selection policy for the sequential discovery adaptation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreeBallSequentialPolicyKind {
    /// Select the lowest expected-free-energy gaze/choice action.
    ExpectedFreeEnergy,
    /// Remove model information value, retaining conditional state information.
    StateInformationOnly,
    /// Remove conditional state information value, retaining model information.
    ModelInformationOnly,
    /// Remove both epistemic terms; retain pragmatic preferences and costs.
    PragmaticOnly,
    /// Do not gaze: select the best immediate choice action.
    ImmediateChoiceOnly,
    /// Choose uniformly among the four legal action surfaces (immediate choice or one of three gazes)
    /// and uniformly among the four possible responses (three colours or no response).
    UniformRandomLegalAction,
    /// Force a gaze at one location; use the policy's best contingent response.
    FixedGaze(ThreeBallLocation),
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
    /// Policy arm used for this episode.
    pub policy_kind: ThreeBallSequentialPolicyKind,
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
            policy_kind: ThreeBallSequentialPolicyKind::ExpectedFreeEnergy,
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
    pub source_demo_commit: String,
    pub source_rule_generator_commit: String,
    pub parameterization_digest: String,
    pub true_rule_one_based_index: usize,
    pub true_semantic_class_one_based_index: usize,
    pub model_family: ThreeBallModelFamily,
    pub policy_kind: ThreeBallSequentialPolicyKind,
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

/// Raw paired episodes for one policy arm. All arms in one benchmark use
/// the same ordered seed list and the same hidden scenes for each seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallSequentialPolicyArm {
    pub policy_kind: ThreeBallSequentialPolicyKind,
    pub episodes: Vec<ThreeBallSequentialDiscoveryResult>,
    pub summary: ThreeBallSequentialPolicySummary,
}

/// Outcome summary for one paired policy arm. All means are over episodes,
/// with identical seed coverage across arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallSequentialPolicySummary {
    pub episode_count: usize,
    pub mean_final_true_semantic_class_posterior: f64,
    pub mean_final_model_entropy_nats: f64,
    pub mean_true_class_brier_score: f64,
    pub mean_true_class_negative_log_probability: f64,
    pub correct_semantic_class_commitment_rate: f64,
    pub mean_cumulative_reward: f64,
    pub mean_gaze_count: f64,
}

/// Percentile interval from deterministic paired bootstrap resampling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallBootstrapInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub replicates: usize,
}

/// Paired candidate-minus-reference contrasts. Positive and negative values
/// are interpreted according to the metric; raw per-seed episodes are kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallPairedPolicyContrast {
    pub reference_policy_kind: ThreeBallSequentialPolicyKind,
    pub candidate_policy_kind: ThreeBallSequentialPolicyKind,
    pub paired_seed_count: usize,
    pub mean_delta_final_true_semantic_class_posterior: f64,
    pub ci95_delta_final_true_semantic_class_posterior: Option<ThreeBallBootstrapInterval>,
    pub mean_delta_brier_score: f64,
    pub ci95_delta_brier_score: Option<ThreeBallBootstrapInterval>,
    pub mean_delta_correct_commitment_rate: f64,
    pub ci95_delta_correct_commitment_rate: Option<ThreeBallBootstrapInterval>,
    pub mean_delta_cumulative_reward: f64,
    pub ci95_delta_cumulative_reward: Option<ThreeBallBootstrapInterval>,
    pub mean_delta_gaze_count: f64,
    pub ci95_delta_gaze_count: Option<ThreeBallBootstrapInterval>,
}

/// Reproducible paired policy matrix. This stores raw per-seed episodes rather
/// than only aggregate scores so independent analyses can recompute paired
/// differences, bootstrap intervals, and calibration metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallPairedPolicyBenchmark {
    pub format_version: u32,
    pub source_demo_commit: String,
    pub source_rule_generator_commit: String,
    pub parameterization_digest: String,
    pub true_rule_one_based_index: usize,
    pub model_family: ThreeBallModelFamily,
    pub seeds: Vec<u64>,
    /// First supplied arm is the preregistered reference for paired contrasts.
    pub reference_policy_kind: ThreeBallSequentialPolicyKind,
    pub arms: Vec<ThreeBallSequentialPolicyArm>,
    pub paired_comparisons: Vec<ThreeBallPairedPolicyContrast>,
}

/// Run a paired policy-ablation matrix over identical environment seeds.
///
/// The environment and policy RNG streams are domain-separated inside each
/// episode. A random policy may consume extra policy draws without changing
/// the hidden-scene sequence used by another arm with the same seed.
pub fn run_three_ball_paired_policy_benchmark(
    space: &ThreeBallRuleSpace,
    true_rule_one_based_index: usize,
    seeds: &[u64],
    policy_kinds: &[ThreeBallSequentialPolicyKind],
    base_config: ThreeBallSequentialDiscoveryConfig,
) -> Result<ThreeBallPairedPolicyBenchmark, ThreeBallSequentialDiscoveryError> {
    if seeds.is_empty() {
        return Err(ThreeBallSequentialDiscoveryError::EmptySeedSet);
    }
    if policy_kinds.is_empty() {
        return Err(ThreeBallSequentialDiscoveryError::EmptyPolicySet);
    }
    let mut seen_seeds = BTreeSet::new();
    for seed in seeds.iter().copied() {
        if !seen_seeds.insert(seed) {
            return Err(ThreeBallSequentialDiscoveryError::DuplicateSeed(seed));
        }
    }
    let mut seen_policies = BTreeSet::new();
    for policy_kind in policy_kinds.iter().copied() {
        if !seen_policies.insert(policy_kind) {
            return Err(ThreeBallSequentialDiscoveryError::DuplicatePolicy(
                policy_kind,
            ));
        }
    }

    let mut arms = Vec::with_capacity(policy_kinds.len());
    for policy_kind in policy_kinds.iter().copied() {
        let mut episodes = Vec::with_capacity(seeds.len());
        for seed in seeds.iter().copied() {
            episodes.push(run_three_ball_sequential_discovery(
                space,
                true_rule_one_based_index,
                seed,
                ThreeBallSequentialDiscoveryConfig {
                    policy_kind,
                    ..base_config
                },
            )?);
        }
        let summary = summarize_policy_episodes(&episodes);
        arms.push(ThreeBallSequentialPolicyArm {
            policy_kind,
            episodes,
            summary,
        });
    }

    let reference_policy_kind = arms[0].policy_kind;
    let mut paired_comparisons = Vec::with_capacity(arms.len().saturating_sub(1));
    for (candidate_index, candidate) in arms.iter().enumerate().skip(1) {
        let reference = &arms[0].episodes;
        let candidate_episodes = &candidate.episodes;
        let deltas = paired_policy_metric_deltas(reference, candidate_episodes);
        let bootstrap_seed = 0xB007_57A9_2026_0000_u64 ^ candidate_index as u64;
        paired_comparisons.push(ThreeBallPairedPolicyContrast {
            reference_policy_kind,
            candidate_policy_kind: candidate.policy_kind,
            paired_seed_count: seeds.len(),
            mean_delta_final_true_semantic_class_posterior: mean(&deltas.final_true_posterior),
            ci95_delta_final_true_semantic_class_posterior: bootstrap_interval(
                &deltas.final_true_posterior,
                bootstrap_seed ^ 0x01,
            ),
            mean_delta_brier_score: mean(&deltas.brier),
            ci95_delta_brier_score: bootstrap_interval(&deltas.brier, bootstrap_seed ^ 0x02),
            mean_delta_correct_commitment_rate: mean(&deltas.correct_commitment),
            ci95_delta_correct_commitment_rate: bootstrap_interval(
                &deltas.correct_commitment,
                bootstrap_seed ^ 0x03,
            ),
            mean_delta_cumulative_reward: mean(&deltas.reward),
            ci95_delta_cumulative_reward: bootstrap_interval(&deltas.reward, bootstrap_seed ^ 0x04),
            mean_delta_gaze_count: mean(&deltas.gaze_count),
            ci95_delta_gaze_count: bootstrap_interval(&deltas.gaze_count, bootstrap_seed ^ 0x05),
        });
    }

    Ok(ThreeBallPairedPolicyBenchmark {
        format_version: 1,
        source_demo_commit: space.source_demo_commit.clone(),
        source_rule_generator_commit: space.source_rule_generator_commit.clone(),
        parameterization_digest: space.parameterization_digest.clone(),
        true_rule_one_based_index,
        model_family: base_config.model_family,
        seeds: seeds.to_vec(),
        reference_policy_kind,
        arms,
        paired_comparisons,
    })
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
    #[error("discovery or post-commitment preference schedule contains a non-finite value")]
    InvalidPreferenceSchedule,
    #[error("true rule one-based index {index} is outside 1..={rule_count}")]
    InvalidTrueRuleIndex { index: usize, rule_count: usize },
    #[error("model family has no posterior support for the true semantic rule")]
    TrueRuleMissingFromModelFamily,
    #[error("paired policy benchmark requires at least one seed")]
    EmptySeedSet,
    #[error("paired policy benchmark requires at least one policy arm")]
    EmptyPolicySet,
    #[error("paired policy benchmark contains duplicate seed {0}")]
    DuplicateSeed(u64),
    #[error("paired policy benchmark contains duplicate policy arm {0:?}")]
    DuplicatePolicy(ThreeBallSequentialPolicyKind),
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
    if config
        .discovery_feedback_log_preferences_nats
        .iter()
        .chain(config.post_commitment_feedback_log_preferences_nats.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ThreeBallSequentialDiscoveryError::InvalidPreferenceSchedule);
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
        ThreeBallModelFamily::SourceLabelled => ThreeBallJointBelief::uniform_source_prior(space)?,
        ThreeBallModelFamily::SemanticQuotient => {
            ThreeBallJointBelief::from_semantic_quotient_prior(
                space,
                &uniform_three_ball_state_prior(),
            )?
        }
    };
    let initial_semantic_posterior = semantic_model_posterior(space, &belief)?;
    if !initial_semantic_posterior.contains_key(&true_semantic_class_one_based_index) {
        return Err(ThreeBallSequentialDiscoveryError::TrueRuleMissingFromModelFamily);
    }

    // Environment and policy randomness are domain-separated. Paired policy
    // arms with the same seed therefore see identical hidden scenes even when
    // a stochastic policy consumes a different number of random draws.
    let mut environment_rng = SplitMix64::new(seed);
    let mut policy_rng = SplitMix64::new(seed ^ 0x11CE_5EED_5EED_0001);
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
            colour_from_index(environment_rng.below(3)),
            colour_from_index(environment_rng.below(3)),
            colour_from_index(environment_rng.below(3)),
        ];

        let model_posterior_before = semantic_model_posterior(space, &belief)?;
        let model_entropy_before = entropy(&model_posterior_before);
        let (_, max_probability_before) = winning_semantic_class(&model_posterior_before)?;
        preferences_restored |= exceeds_log_odds_threshold(
            max_probability_before,
            config.commitment_log_odds_threshold_nats,
        );

        let mut trial_policy = config.policy;
        trial_policy.feedback_log_preferences_nats = if preferences_restored {
            config.post_commitment_feedback_log_preferences_nats
        } else {
            config.discovery_feedback_log_preferences_nats
        };
        match config.policy_kind {
            ThreeBallSequentialPolicyKind::ExpectedFreeEnergy
            | ThreeBallSequentialPolicyKind::ImmediateChoiceOnly
            | ThreeBallSequentialPolicyKind::FixedGaze(_) => {}
            ThreeBallSequentialPolicyKind::StateInformationOnly => {
                trial_policy.model_information_weight = 0.0;
            }
            ThreeBallSequentialPolicyKind::ModelInformationOnly => {
                trial_policy.conditional_state_information_weight = 0.0;
            }
            ThreeBallSequentialPolicyKind::PragmaticOnly => {
                trial_policy.model_information_weight = 0.0;
                trial_policy.conditional_state_information_weight = 0.0;
            }
            ThreeBallSequentialPolicyKind::UniformRandomLegalAction => {}
        }

        let decision = score_three_ball_gaze_policies(space, &belief, trial_policy)?;
        let selected_action_index = match config.policy_kind {
            ThreeBallSequentialPolicyKind::ExpectedFreeEnergy
            | ThreeBallSequentialPolicyKind::StateInformationOnly
            | ThreeBallSequentialPolicyKind::ModelInformationOnly
            | ThreeBallSequentialPolicyKind::PragmaticOnly => decision.selected_action_index,
            ThreeBallSequentialPolicyKind::ImmediateChoiceOnly => 0,
            ThreeBallSequentialPolicyKind::UniformRandomLegalAction => {
                policy_rng.below(decision.action_scores.len() as u64)
            }
            ThreeBallSequentialPolicyKind::FixedGaze(location) => decision
                .action_scores
                .iter()
                .position(|score| score.location == Some(location))
                .ok_or(ThreeBallSequentialDiscoveryError::InvalidSelectedAction {
                    index: decision.action_scores.len(),
                    action_count: decision.action_scores.len(),
                })?,
        };
        let action = decision.action_scores.get(selected_action_index).ok_or(
            ThreeBallSequentialDiscoveryError::InvalidSelectedAction {
                index: decision.selected_action_index,
                action_count: decision.action_scores.len(),
            },
        )?;

        let selected_gaze = action.location;
        let (belief_after_gaze, observed_colour, selected_choice) = if let Some(location) =
            selected_gaze
        {
            gaze_count += 1;
            let colour = hidden_state[location as usize];
            let posterior = belief.condition_on_gaze(location, colour)?;
            let selected_choice =
                if config.policy_kind == ThreeBallSequentialPolicyKind::UniformRandomLegalAction {
                    random_response(&mut policy_rng)
                } else {
                    action
                        .contingencies
                        .iter()
                        .find(|entry| entry.observed_colour == colour)
                        .ok_or(ThreeBallSequentialDiscoveryError::MissingGazeContingency)?
                        .selected_choice
                };
            (posterior, Some(colour), selected_choice)
        } else {
            let selected_choice =
                if config.policy_kind == ThreeBallSequentialPolicyKind::UniformRandomLegalAction {
                    random_response(&mut policy_rng)
                } else {
                    action.immediate_choice
                };
            (belief.clone(), None, selected_choice)
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
        let (winning_rule, winning_probability) = winning_semantic_class(&model_posterior_after)?;
        let winning_log_odds = finite_log_odds(winning_probability);
        let committed_after_trial = exceeds_log_odds_threshold(
            winning_probability,
            config.commitment_log_odds_threshold_nats,
        );
        let correct_commitment_after_trial =
            committed_after_trial && winning_rule == true_semantic_class_one_based_index;

        if committed_after_trial && trials_to_model_commitment.is_none() {
            trials_to_model_commitment = Some(trial_index);
        }
        if correct_commitment_after_trial && trials_to_correct_rule_commitment.is_none() {
            trials_to_correct_rule_commitment = Some(trial_index);
        }

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
            conditional_state_information_gain_nats: action.conditional_state_information_gain_nats,
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
        source_demo_commit: space.source_demo_commit.clone(),
        source_rule_generator_commit: space.source_rule_generator_commit.clone(),
        parameterization_digest: space.parameterization_digest.clone(),
        true_rule_one_based_index,
        true_semantic_class_one_based_index,
        model_family: config.model_family,
        policy_kind: config.policy_kind,
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
                (
                    format!("three-ball-semantic-rule-{class_index:02}"),
                    probability,
                )
            })
            .collect(),
    })
}

#[derive(Debug)]
struct PairedPolicyMetricDeltas {
    final_true_posterior: Vec<f64>,
    brier: Vec<f64>,
    correct_commitment: Vec<f64>,
    reward: Vec<f64>,
    gaze_count: Vec<f64>,
}

fn summarize_policy_episodes(
    episodes: &[ThreeBallSequentialDiscoveryResult],
) -> ThreeBallSequentialPolicySummary {
    let count = episodes.len() as f64;
    ThreeBallSequentialPolicySummary {
        episode_count: episodes.len(),
        mean_final_true_semantic_class_posterior: episodes
            .iter()
            .map(|episode| episode.final_true_semantic_class_posterior)
            .sum::<f64>()
            / count,
        mean_final_model_entropy_nats: episodes
            .iter()
            .map(|episode| episode.final_model_entropy_nats)
            .sum::<f64>()
            / count,
        mean_true_class_brier_score: episodes
            .iter()
            .map(|episode| episode.mean_true_class_brier_score)
            .sum::<f64>()
            / count,
        mean_true_class_negative_log_probability: episodes
            .iter()
            .map(|episode| episode.mean_true_class_negative_log_probability)
            .sum::<f64>()
            / count,
        correct_semantic_class_commitment_rate: episodes
            .iter()
            .filter(|episode| episode.trials_to_correct_rule_commitment.is_some())
            .count() as f64
            / count,
        mean_cumulative_reward: episodes
            .iter()
            .map(|episode| episode.cumulative_reward as f64)
            .sum::<f64>()
            / count,
        mean_gaze_count: episodes
            .iter()
            .map(|episode| episode.gaze_count as f64)
            .sum::<f64>()
            / count,
    }
}

fn paired_policy_metric_deltas(
    reference: &[ThreeBallSequentialDiscoveryResult],
    candidate: &[ThreeBallSequentialDiscoveryResult],
) -> PairedPolicyMetricDeltas {
    debug_assert_eq!(reference.len(), candidate.len());
    PairedPolicyMetricDeltas {
        final_true_posterior: reference
            .iter()
            .zip(candidate)
            .map(|(left, right)| {
                right.final_true_semantic_class_posterior - left.final_true_semantic_class_posterior
            })
            .collect(),
        brier: reference
            .iter()
            .zip(candidate)
            .map(|(left, right)| {
                right.mean_true_class_brier_score - left.mean_true_class_brier_score
            })
            .collect(),
        correct_commitment: reference
            .iter()
            .zip(candidate)
            .map(|(left, right)| {
                (if right.trials_to_correct_rule_commitment.is_some() {
                    1.0
                } else {
                    0.0
                }) - (if left.trials_to_correct_rule_commitment.is_some() {
                    1.0
                } else {
                    0.0
                })
            })
            .collect(),
        reward: reference
            .iter()
            .zip(candidate)
            .map(|(left, right)| (right.cumulative_reward - left.cumulative_reward) as f64)
            .collect(),
        gaze_count: reference
            .iter()
            .zip(candidate)
            .map(|(left, right)| (right.gaze_count as f64) - (left.gaze_count as f64))
            .collect(),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Deterministic 95% percentile bootstrap over paired seed-level differences.
/// Fewer than two pairs do not support an interval and return None.
fn bootstrap_interval(values: &[f64], seed: u64) -> Option<ThreeBallBootstrapInterval> {
    if values.len() < 2 {
        return None;
    }
    const REPLICATES: usize = 2_000;
    let mut rng = SplitMix64::new(seed);
    let mut bootstrap_means = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        let total = (0..values.len())
            .map(|_| values[rng.below(values.len() as u64)])
            .sum::<f64>();
        bootstrap_means.push(total / values.len() as f64);
    }
    bootstrap_means.sort_by(f64::total_cmp);
    let lower_index = ((REPLICATES - 1) as f64 * 0.025).floor() as usize;
    let upper_index = ((REPLICATES - 1) as f64 * 0.975).ceil() as usize;
    Some(ThreeBallBootstrapInterval {
        lower: bootstrap_means[lower_index],
        upper: bootstrap_means[upper_index],
        confidence_level: 0.95,
        replicates: REPLICATES,
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
    probability >= 1.0 || finite_log_odds(probability).is_some_and(|log_odds| log_odds > threshold)
}

fn random_response(rng: &mut SplitMix64) -> Option<ThreeBallColour> {
    match rng.below(4) {
        0 => Some(ThreeBallColour::Red),
        1 => Some(ThreeBallColour::Green),
        2 => Some(ThreeBallColour::Blue),
        3 => None,
        _ => unreachable!("response index is in 0..4"),
    }
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
    fn paired_policy_matrix_preserves_seed_alignment_and_scene_identity() {
        let space = build_three_ball_rule_space();
        let seeds = [11, 23, 37];
        let policies = [
            ThreeBallSequentialPolicyKind::ExpectedFreeEnergy,
            ThreeBallSequentialPolicyKind::StateInformationOnly,
            ThreeBallSequentialPolicyKind::ModelInformationOnly,
            ThreeBallSequentialPolicyKind::PragmaticOnly,
            ThreeBallSequentialPolicyKind::ImmediateChoiceOnly,
            ThreeBallSequentialPolicyKind::FixedGaze(ThreeBallLocation::Centre),
            ThreeBallSequentialPolicyKind::UniformRandomLegalAction,
        ];
        let result = run_three_ball_paired_policy_benchmark(
            &space,
            65,
            &seeds,
            &policies,
            ThreeBallSequentialDiscoveryConfig {
                max_trials: 6,
                ..Default::default()
            },
        )
        .expect("paired policy matrix");
        assert_eq!(result.seeds, seeds);
        assert_eq!(result.arms.len(), policies.len());
        for (arm_index, arm) in result.arms.iter().enumerate() {
            assert_eq!(arm.policy_kind, policies[arm_index]);
            assert_eq!(arm.episodes.len(), seeds.len());
            for (seed_index, episode) in arm.episodes.iter().enumerate() {
                assert_eq!(episode.seed, seeds[seed_index]);
                assert_eq!(episode.policy_kind, policies[arm_index]);
            }
        }

        for seed_index in 0..seeds.len() {
            let reference_scenes = result.arms[0].episodes[seed_index]
                .trial_receipts
                .iter()
                .map(|trial| trial.hidden_state_for_audit)
                .collect::<Vec<_>>();
            for arm in &result.arms[1..] {
                let scenes = arm.episodes[seed_index]
                    .trial_receipts
                    .iter()
                    .map(|trial| trial.hidden_state_for_audit)
                    .collect::<Vec<_>>();
                assert_eq!(reference_scenes, scenes);
            }
        }
    }

    #[test]
    fn paired_policy_matrix_rejects_duplicate_seeds_and_policy_arms() {
        let space = build_three_ball_rule_space();
        let base = ThreeBallSequentialDiscoveryConfig {
            max_trials: 1,
            ..Default::default()
        };
        assert!(matches!(
            run_three_ball_paired_policy_benchmark(
                &space,
                65,
                &[1, 1],
                &[ThreeBallSequentialPolicyKind::ExpectedFreeEnergy],
                base,
            ),
            Err(ThreeBallSequentialDiscoveryError::DuplicateSeed(1))
        ));
        assert!(matches!(
            run_three_ball_paired_policy_benchmark(
                &space,
                65,
                &[1],
                &[
                    ThreeBallSequentialPolicyKind::ExpectedFreeEnergy,
                    ThreeBallSequentialPolicyKind::ExpectedFreeEnergy
                ],
                base,
            ),
            Err(ThreeBallSequentialDiscoveryError::DuplicatePolicy(
                ThreeBallSequentialPolicyKind::ExpectedFreeEnergy
            ))
        ));
    }

    #[test]
    fn paired_policy_arms_receive_identical_environment_scenes() {
        let space = build_three_ball_rule_space();
        let seed = 0x51de_2026;
        let base = ThreeBallSequentialDiscoveryConfig {
            max_trials: 12,
            ..Default::default()
        };
        let active = run_three_ball_sequential_discovery(
            &space,
            65,
            seed,
            ThreeBallSequentialDiscoveryConfig {
                policy_kind: ThreeBallSequentialPolicyKind::ExpectedFreeEnergy,
                ..base
            },
        )
        .expect("active inference episode");
        let random = run_three_ball_sequential_discovery(
            &space,
            65,
            seed,
            ThreeBallSequentialDiscoveryConfig {
                policy_kind: ThreeBallSequentialPolicyKind::UniformRandomLegalAction,
                ..base
            },
        )
        .expect("random policy episode");

        let active_scenes = active
            .trial_receipts
            .iter()
            .map(|trial| trial.hidden_state_for_audit)
            .collect::<Vec<_>>();
        let random_scenes = random
            .trial_receipts
            .iter()
            .map(|trial| trial.hidden_state_for_audit)
            .collect::<Vec<_>>();
        assert_eq!(active_scenes, random_scenes);
        assert_eq!(active.seed, random.seed);
        assert_ne!(active.policy_kind, random.policy_kind);
    }

    #[test]
    fn registered_policy_ablations_are_deterministic_and_finite() {
        let space = build_three_ball_rule_space();
        let kinds = [
            ThreeBallSequentialPolicyKind::ExpectedFreeEnergy,
            ThreeBallSequentialPolicyKind::StateInformationOnly,
            ThreeBallSequentialPolicyKind::ModelInformationOnly,
            ThreeBallSequentialPolicyKind::PragmaticOnly,
            ThreeBallSequentialPolicyKind::ImmediateChoiceOnly,
            ThreeBallSequentialPolicyKind::FixedGaze(ThreeBallLocation::Centre),
            ThreeBallSequentialPolicyKind::UniformRandomLegalAction,
        ];
        for policy_kind in kinds {
            let config = ThreeBallSequentialDiscoveryConfig {
                max_trials: 8,
                policy_kind,
                ..Default::default()
            };
            let first =
                run_three_ball_sequential_discovery(&space, 65, 77, config).expect("policy arm");
            let replay =
                run_three_ball_sequential_discovery(&space, 65, 77, config).expect("policy replay");
            assert_eq!(first, replay);
            assert_eq!(first.policy_kind, policy_kind);
            assert!(first.final_model_entropy_nats.is_finite());
            assert!(first.mean_true_class_brier_score.is_finite());
            assert!(
                first
                    .trial_receipts
                    .iter()
                    .all(|trial| trial.model_entropy_before_nats.is_finite()
                        && trial.model_entropy_after_nats.is_finite())
            );
        }
    }

    #[test]
    fn seeded_episode_is_bitwise_reproducible() {
        let space = build_three_ball_rule_space();
        let config = discovery_config(ThreeBallModelFamily::SemanticQuotient);
        let first =
            run_three_ball_sequential_discovery(&space, 65, 0x5eed, config).expect("first episode");
        let second = run_three_ball_sequential_discovery(&space, 65, 0x5eed, config)
            .expect("second episode");
        assert_eq!(first, second);
        assert_eq!(first.trial_receipts.len(), config.max_trials);
        assert_eq!(first.trial_receipts[0].trial_index, 1);
        assert_eq!(
            first.trial_receipts.last().unwrap().trial_index,
            config.max_trials
        );
    }

    #[test]
    fn episode_preserves_model_beliefs_but_resets_each_hidden_scene() {
        let space = build_three_ball_rule_space();
        let config = ThreeBallSequentialDiscoveryConfig {
            max_trials: 8,
            ..discovery_config(ThreeBallModelFamily::SemanticQuotient)
        };
        let result = run_three_ball_sequential_discovery(&space, 65, 17, config).expect("episode");
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
        assert!((source.final_model_entropy_nats - quotient.final_model_entropy_nats).abs() < 1e-9);
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
        let result = run_three_ball_sequential_discovery(&space, 65, 0x2026_1107, config)
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

        assert!(matches!(
            run_three_ball_sequential_discovery(
                &space,
                65,
                1,
                ThreeBallSequentialDiscoveryConfig {
                    post_commitment_feedback_log_preferences_nats: [0.0, f64::NAN, -6.0],
                    ..Default::default()
                }
            ),
            Err(ThreeBallSequentialDiscoveryError::InvalidPreferenceSchedule)
        ));
    }
}
