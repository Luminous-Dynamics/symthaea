// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reproducible three-ball rule-space construction.
//!
//! This module reconstructs the 3 × 3^3 categorical feedback-rule generator
//! used by SPM's public `spm_MDP_rules` implementation. It preserves the
//! original 81 indexed hypotheses and their model-average prior. It also
//! canonicalizes the actual likelihood tensors to audit semantic duplicates.
//!
//! IMPORTANT REPRODUCIBILITY NOTE:
//! The 2026 Nature Communications paper reports 81 candidate mappings and
//! 79 unique rules. The public SPM generator revision audited here yields 81
//! hypotheses but only 75 distinct feedback likelihood tensors: three
//! context-independent (constant criterion-location) mappings each appear
//! under all three context-factor labels. The benchmark reports this
//! discrepancy explicitly; it does not silently rewrite the source generator
//! or force the published count. Reconcile against the exact archived source
//! and author implementation before labelling numerical results an exact
//! reproduction.
//!
//! Source provenance:
//! - DEM_reasoning.m: SPM commit 28d1f103b021f729266cb7ea07baaf4d0285b22a
//! - spm_MDP_rules.m: SPM commit 5852cadb300a81fd3f30e7056ee329bca20f0efe

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::dirichlet_model_reduction::{
    DirichletActionPrediction, DirichletBayesianModelReductionError, DirichletConcentrationTensor,
    DirichletModelInformationGain, DirichletModelInformationGainError, DirichletPredictedOutcome,
    ReducedDirichletModelHypothesis, evaluate_dirichlet_model_information_gain,
    evaluate_dirichlet_model_information_gain_spm_compatible, reduce_dirichlet_models,
    reduce_dirichlet_models_spm_compatible,
};

pub const THREE_BALL_CONTEXT_RULE_HYPOTHESIS_COUNT: usize = 81;
pub const THREE_BALL_PAPER_REPORTED_UNIQUE_RULE_COUNT: usize = 79;
pub const THREE_BALL_PUBLIC_GENERATOR_UNIQUE_TENSOR_COUNT: usize = 75;
pub const THREE_BALL_PLANTED_RULE_ONE_BASED_INDEX: usize = 65;
pub const THREE_BALL_HIDDEN_STATE_COLUMN_COUNT: usize = 432;
pub const THREE_BALL_FEEDBACK_OUTCOME_COUNT: usize = 3;

/// Canonical, byte-exact descriptor hashed for the ordered likelihood tensor.
pub const THREE_BALL_PARAMETERIZATION_DESCRIPTOR: &str = "symthaea-three-ball-feedback-v1|outcomes=no-feedback,correct,incorrect|factors=left,centre,right,gaze,choice|left-fastest";

/// SHA-256 of the canonical descriptor above.
///
/// The digest was independently computed over the descriptor's UTF-8 bytes.
pub const THREE_BALL_PARAMETERIZATION_DIGEST: &str =
    "69d3c6e3c7ea7609efc79f6ba58a69fc0979733d59589f61cbbeafc68156a332";

pub const SPM_DEM_REASONING_COMMIT: &str = "28d1f103b021f729266cb7ea07baaf4d0285b22a";
pub const SPM_MDP_RULES_COMMIT: &str = "5852cadb300a81fd3f30e7056ee329bca20f0efe";

/// A ball's spatial location. Indices match the SPM feedback tensor factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(usize)]
pub enum ThreeBallLocation {
    Left = 0,
    Centre = 1,
    Right = 2,
}

/// A ball or choice colour. Indices match the SPM labelled state order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(usize)]
pub enum ThreeBallColour {
    Red = 0,
    Green = 1,
    Blue = 2,
}

/// Feedback modality order in the SPM model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum ThreeBallFeedback {
    NoFeedback = 0,
    Correct = 1,
    Incorrect = 2,
}

/// A rule: one context factor, and a context-colour → criterion-location map.
///
/// The selected criterion ball's colour is the correct choice. The gaze
/// factor is part of the likelihood tensor's state space but is not used by
/// the feedback rule itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallRule {
    /// One-based source enumeration index, 1..=81.
    pub one_based_index: usize,
    /// Which ball location supplies the context colour.
    pub context_factor: ThreeBallLocation,
    /// Criterion location for [red, green, blue] context-colour values.
    pub criterion_by_context_colour: [ThreeBallLocation; 3],
    /// First rule index with the same complete feedback likelihood tensor.
    /// None means this rule is the first occurrence of its semantic mapping.
    pub duplicate_of_one_based_index: Option<usize>,
    /// Source model identifier retained even for semantically duplicate rules.
    pub model_id: String,
    /// Equal prior mass over all 81 source hypotheses, including duplicates.
    pub prior_probability: f64,
    /// Prior-restricted Dirichlet likelihood model from the SPM rule generator.
    pub parameter_prior: DirichletConcentrationTensor,
}

/// Full rule family plus its Bayesian-model-average parent prior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallRuleSpace {
    pub format_version: u32,
    pub source_demo_commit: String,
    pub source_rule_generator_commit: String,
    pub parameterization_digest: String,
    pub rules: Vec<ThreeBallRule>,
    /// Per-column arithmetic mean of source hypotheses, used by the strict
    /// structural-zero research variant of Dirichlet BMR.
    pub full_model_average_prior: DirichletConcentrationTensor,
    /// Source-faithful SPM DEM_reasoning prior: the sum of the 81 likelihood
    /// tensors divided by 81 and then normalized globally by total tensor mass.
    pub spm_model_average_prior: DirichletConcentrationTensor,
    /// Number of distinct complete feedback likelihood tensors.
    pub semantic_unique_rule_count: usize,
    /// Groups of source indices that have identical feedback likelihood tensors.
    pub semantic_duplicate_groups: Vec<Vec<usize>>,
    /// The paper's stated unique count, retained as an external comparison.
    pub paper_reported_unique_rule_count: usize,
}

impl ThreeBallRule {
    /// Deterministic feedback under a fully specified hidden ball state.
    pub fn feedback(
        &self,
        ball_colours: [ThreeBallColour; 3],
        choice: Option<ThreeBallColour>,
    ) -> ThreeBallFeedback {
        let Some(choice) = choice else {
            return ThreeBallFeedback::NoFeedback;
        };
        let context_colour = ball_colours[self.context_factor as usize] as usize;
        let criterion_location = self.criterion_by_context_colour[context_colour] as usize;
        if choice == ball_colours[criterion_location] {
            ThreeBallFeedback::Correct
        } else {
            ThreeBallFeedback::Incorrect
        }
    }

    /// Visual outcome under one foveated gaze action.
    ///
    /// `None` represents the start location, where no ball is visible.
    pub fn visual_outcome(
        ball_colours: [ThreeBallColour; 3],
        gaze: Option<ThreeBallLocation>,
    ) -> Option<ThreeBallColour> {
        gaze.map(|location| ball_colours[location as usize])
    }
}

/// Reconstruct all 81 hypotheses in the exact SPM combination order.
///
/// The first combination dimension (context-factor identity) varies fastest;
/// the three criterion-location assignments follow in red, green, blue order.
/// This gives the planted centre-context mapping at source rule index 65.
pub fn build_three_ball_rule_space() -> ThreeBallRuleSpace {
    let mut rules = Vec::with_capacity(THREE_BALL_CONTEXT_RULE_HYPOTHESIS_COUNT);
    let mut duplicate_groups: Vec<Vec<usize>> = Vec::new();
    let mut first_seen = Vec::<(Vec<Vec<f64>>, Vec<Vec<bool>>, usize)>::new();

    // spm_combinations([context factor, criterion(red), criterion(green),
    // criterion(blue)]) varies the first dimension fastest.
    for blue_criterion in 0..3 {
        for green_criterion in 0..3 {
            for red_criterion in 0..3 {
                for context_factor in 0..3 {
                    let criterion_by_context_colour = [
                        location_from_index(red_criterion),
                        location_from_index(green_criterion),
                        location_from_index(blue_criterion),
                    ];
                    let parameter_prior =
                        build_rule_prior(context_factor, criterion_by_context_colour);
                    let existing = first_seen.iter().find(|(columns, support, _)| {
                        *columns == parameter_prior.columns
                            && *support == parameter_prior.active_support
                    });
                    let source_index = rules.len() + 1;
                    let duplicate_of_one_based_index =
                        existing.map(|(_, _, first_index)| *first_index);

                    if let Some((_, _, first_index)) = existing {
                        if let Some(group) = duplicate_groups
                            .iter_mut()
                            .find(|group| group[0] == *first_index)
                        {
                            group.push(source_index);
                        } else {
                            duplicate_groups.push(vec![*first_index, source_index]);
                        }
                    } else {
                        first_seen.push((
                            parameter_prior.columns.clone(),
                            parameter_prior.active_support.clone(),
                            source_index,
                        ));
                    }

                    rules.push(ThreeBallRule {
                        one_based_index: source_index,
                        context_factor: location_from_index(context_factor),
                        criterion_by_context_colour,
                        duplicate_of_one_based_index,
                        model_id: format!("three-ball-rule-{source_index:02}"),
                        prior_probability: 1.0 / THREE_BALL_CONTEXT_RULE_HYPOTHESIS_COUNT as f64,
                        parameter_prior,
                    });
                }
            }
        }
    }

    let semantic_unique_rule_count = first_seen.len();
    let full_model_average_prior = arithmetic_mean_prior(&rules);
    let spm_model_average_prior = globally_normalized_model_average_prior(&rules);

    ThreeBallRuleSpace {
        format_version: 1,
        source_demo_commit: SPM_DEM_REASONING_COMMIT.to_owned(),
        source_rule_generator_commit: SPM_MDP_RULES_COMMIT.to_owned(),
        parameterization_digest: THREE_BALL_PARAMETERIZATION_DIGEST.to_owned(),
        rules,
        full_model_average_prior,
        spm_model_average_prior,
        semantic_unique_rule_count,
        semantic_duplicate_groups: duplicate_groups,
        paper_reported_unique_rule_count: THREE_BALL_PAPER_REPORTED_UNIQUE_RULE_COUNT,
    }
}

impl ThreeBallRuleSpace {
    /// Return the planted paper rule (one-based rule 65).
    pub fn planted_rule(&self) -> &ThreeBallRule {
        &self.rules[THREE_BALL_PLANTED_RULE_ONE_BASED_INDEX - 1]
    }

    /// Build the reduced-model hypotheses expected by Dirichlet BMR.
    ///
    /// The source's 81 labelled hypotheses and equal prior masses are retained,
    /// including semantically duplicate likelihood tensors.
    pub fn reduced_model_hypotheses(&self) -> Vec<ReducedDirichletModelHypothesis> {
        self.rules
            .iter()
            .map(|rule| ReducedDirichletModelHypothesis {
                model_id: rule.model_id.clone(),
                prior_probability: rule.prior_probability,
                parameter_prior: rule.parameter_prior.clone(),
            })
            .collect()
    }

    /// Return the quotient model family over semantically identical likelihood tensors.
    ///
    /// Source hypotheses with identical complete feedback likelihood tensors are
    /// grouped under their first source index. Their prior probabilities are
    /// summed, preserving the Bayesian-model-average parent prior exactly.
    /// This removes label-only uncertainty without changing any prediction or
    /// model evidence available to the feedback modality.
    ///
    /// The uncompressed reduced_model_hypotheses remains available for
    /// source-faithful reproduction and comparisons against the paper's labelled
    /// hypothesis count.
    pub fn semantic_unique_model_hypotheses(&self) -> Vec<ReducedDirichletModelHypothesis> {
        let mut classes = BTreeMap::<usize, ReducedDirichletModelHypothesis>::new();
        for rule in &self.rules {
            let representative_index = rule
                .duplicate_of_one_based_index
                .unwrap_or(rule.one_based_index);
            let representative_id = format!("three-ball-semantic-rule-{representative_index:02}");
            let entry = classes.entry(representative_index).or_insert_with(|| {
                ReducedDirichletModelHypothesis {
                    model_id: representative_id,
                    prior_probability: 0.0,
                    parameter_prior: rule.parameter_prior.clone(),
                }
            });
            entry.prior_probability += rule.prior_probability;
        }
        classes.into_values().collect()
    }

    /// Whether the source paper's 79-unique count matches this source audit.
    pub fn unique_count_matches_paper(&self) -> bool {
        self.semantic_unique_rule_count == self.paper_reported_unique_rule_count
    }

    /// Number of context-independent models that are repeated under multiple
    /// context-factor labels by the public SPM rule generator.
    pub fn semantic_duplicate_count(&self) -> usize {
        self.rules
            .iter()
            .filter(|rule| rule.duplicate_of_one_based_index.is_some())
            .count()
    }
}

fn location_from_index(index: usize) -> ThreeBallLocation {
    match index {
        0 => ThreeBallLocation::Left,
        1 => ThreeBallLocation::Centre,
        2 => ThreeBallLocation::Right,
        _ => unreachable!("three-ball locations are exactly 0..3"),
    }
}

fn build_rule_prior(
    context_factor: usize,
    criterion_by_context_colour: [ThreeBallLocation; 3],
) -> DirichletConcentrationTensor {
    let mut columns = Vec::with_capacity(THREE_BALL_HIDDEN_STATE_COLUMN_COUNT);
    let mut active_support = Vec::with_capacity(THREE_BALL_HIDDEN_STATE_COLUMN_COUNT);

    // SPM state combination order: left varies fastest, then centre, right,
    // gaze, and choice. There are 3*3*3*4*4 = 432 columns.
    for choice in 0..4 {
        for _gaze in 0..4 {
            for right in 0..3 {
                for centre in 0..3 {
                    for left in 0..3 {
                        let ball_colours = [left, centre, right];
                        let outcome = if choice == 3 {
                            ThreeBallFeedback::NoFeedback as usize
                        } else {
                            let context_colour = ball_colours[context_factor];
                            let criterion_location =
                                criterion_by_context_colour[context_colour] as usize;
                            if choice == ball_colours[criterion_location] {
                                ThreeBallFeedback::Correct as usize
                            } else {
                                ThreeBallFeedback::Incorrect as usize
                            }
                        };

                        let mut concentration = vec![0.0; THREE_BALL_FEEDBACK_OUTCOME_COUNT];
                        let mut support = vec![false; THREE_BALL_FEEDBACK_OUTCOME_COUNT];
                        concentration[outcome] = 1.0;
                        support[outcome] = true;
                        columns.push(concentration);
                        active_support.push(support);
                    }
                }
            }
        }
    }

    debug_assert_eq!(columns.len(), THREE_BALL_HIDDEN_STATE_COLUMN_COUNT);
    DirichletConcentrationTensor {
        parameterization_digest: THREE_BALL_PARAMETERIZATION_DIGEST.to_owned(),
        columns,
        active_support,
    }
}

/// SPM DEM_reasoning computes A0 as the average of all rule likelihood
/// tensors and then divides the entire tensor by `sum(A0,'all')`. That is a
/// global normalization across all hidden-state columns, unlike the strict
/// per-column arithmetic mean retained by `full_model_average_prior`.
fn globally_normalized_model_average_prior(
    rules: &[ThreeBallRule],
) -> DirichletConcentrationTensor {
    let mut prior = arithmetic_mean_prior(rules);
    let total_mass = prior.columns.iter().flatten().sum::<f64>();
    if total_mass.is_finite() && total_mass > 0.0 {
        for column in &mut prior.columns {
            for concentration in column {
                *concentration /= total_mass;
            }
        }
    }
    prior
}

fn arithmetic_mean_prior(rules: &[ThreeBallRule]) -> DirichletConcentrationTensor {
    let column_count = rules[0].parameter_prior.columns.len();
    let outcome_count = rules[0].parameter_prior.columns[0].len();
    let mut columns = vec![vec![0.0; outcome_count]; column_count];
    let mut active_support = vec![vec![false; outcome_count]; column_count];

    for rule in rules {
        for column in 0..column_count {
            for outcome in 0..outcome_count {
                columns[column][outcome] +=
                    rule.parameter_prior.columns[column][outcome] * rule.prior_probability;
                active_support[column][outcome] |=
                    rule.parameter_prior.active_support[column][outcome];
            }
        }
    }

    DirichletConcentrationTensor {
        parameterization_digest: THREE_BALL_PARAMETERIZATION_DIGEST.to_owned(),
        columns,
        active_support,
    }
}

/// Log preferences over feedback in the order no-feedback, correct, incorrect.
pub const THREE_BALL_FEEDBACK_LOG_PREFERENCES_NATS: [f64; 3] = [0.0, 2.0, -6.0];

/// A legal feedback action in the known-state unit task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallFeedbackActionSpec {
    pub action_id: String,
    pub choice: Option<ThreeBallColour>,
    /// Non-negative action cost in nats, kept separate from outcome preferences.
    pub action_cost_nats: f64,
}

/// Explicit weights for the scoped known-state one-step decision objective.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallFeedbackDecisionConfig {
    /// Relative log preferences in [no-feedback, correct, incorrect] order.
    pub log_preferences_nats: [f64; THREE_BALL_FEEDBACK_OUTCOME_COUNT],
    /// Weight on prospective information about model identity.
    pub epistemic_weight: f64,
}

impl Default for ThreeBallFeedbackDecisionConfig {
    fn default() -> Self {
        Self {
            log_preferences_nats: THREE_BALL_FEEDBACK_LOG_PREFERENCES_NATS,
            epistemic_weight: 1.0,
        }
    }
}

/// Auditable decomposition of one known-state feedback action score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallFeedbackActionScore {
    pub action_id: String,
    pub choice: Option<ThreeBallColour>,
    /// KL[Q(feedback | action) || C(feedback)] in nats.
    pub pragmatic_risk_nats: f64,
    pub model_information_gain_nats: f64,
    pub epistemic_value_nats: f64,
    pub action_cost_nats: f64,
    /// Lower is better: pragmatic risk - epistemic value + action cost.
    pub expected_free_energy_nats: f64,
    pub predictive_feedback: Vec<(ThreeBallFeedback, f64)>,
    pub model_information_gain: DirichletModelInformationGain,
}

/// Result of evaluating a finite set of legal known-state feedback actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallFeedbackDecision {
    pub selected_action_index: usize,
    pub selected_action_id: String,
    pub action_scores: Vec<ThreeBallFeedbackActionScore>,
}

/// Errors from the known-state, feedback-only action evaluator.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ThreeBallFeedbackDecisionError {
    #[error(transparent)]
    BayesianModelReduction(#[from] DirichletBayesianModelReductionError),
    #[error(transparent)]
    ModelInformationGain(#[from] DirichletModelInformationGainError),
    #[error("the action set must not be empty")]
    EmptyActionSet,
    #[error("action {index} has an empty action ID")]
    EmptyActionId { index: usize },
    #[error("duplicate action ID: {0}")]
    DuplicateActionId(String),
    #[error("action {action_id} has invalid cost {cost}")]
    InvalidActionCost { action_id: String, cost: f64 },
    #[error("feedback decision configuration contains a non-finite preference")]
    NonFinitePreference,
    #[error("epistemic weight must be finite and non-negative, got {0}")]
    InvalidEpistemicWeight(f64),
    #[error("likelihood tensor column index {column_index} is outside 0..{column_count}")]
    InvalidLikelihoodColumn {
        column_index: usize,
        column_count: usize,
    },
    #[error(
        "model {model_id} has invalid posterior predictive concentration at outcome {outcome_index}"
    )]
    InvalidPosteriorPredictiveConcentration {
        model_id: String,
        outcome_index: usize,
    },
    #[error(
        "model {model_id} has no positive posterior predictive concentration in the selected column"
    )]
    EmptyPosteriorPredictiveColumn { model_id: String },
    #[error("posterior predictive distribution for model {model_id} sums to {sum}, expected 1")]
    ModelPosteriorPredictiveMassNotNormalized { model_id: String, sum: f64 },
    #[error("action {action_id} produced unknown feedback outcome ID {outcome_id}")]
    UnknownFeedbackOutcome {
        action_id: String,
        outcome_id: String,
    },
    #[error("pragmatic KL risk is numerically invalid: {0}")]
    InvalidPragmaticRisk(f64),
}

/// Build a model-bound action prediction for a known ball-colour state.
///
/// Per-model categorical predictions use the model-specific Dirichlet posterior
/// implied by the full posterior and the reduced prior:
///
///   a_m = a_full_posterior + a_m_prior - a_full_prior.
///
/// Models with zero current posterior probability have no evidence-supported
/// posterior; their normalized prior predictive is retained as a harmless
/// placeholder because their mixture weight is exactly zero. This function is
/// a known-state feedback probe, not a gaze/vision POMDP transition model.
/// Construct an action-conditioned feedback prediction using strict
/// structural-zero model semantics.
pub fn build_three_ball_feedback_action_prediction(
    space: &ThreeBallRuleSpace,
    full_posterior: &DirichletConcentrationTensor,
    models: &[ReducedDirichletModelHypothesis],
    ball_colours: [ThreeBallColour; 3],
    gaze: Option<ThreeBallLocation>,
    choice: Option<ThreeBallColour>,
    action_id: impl Into<String>,
) -> Result<DirichletActionPrediction, ThreeBallFeedbackDecisionError> {
    build_three_ball_feedback_action_prediction_with_mode(
        space,
        full_posterior,
        models,
        ball_colours,
        gaze,
        choice,
        action_id,
        false,
    )
}

/// Construct the same action prediction under the pinned SPM source convention:
/// globally normalized parent prior, posterior model odds from 1/32-smoothed
/// Dirichlet BMR, and model-conditional predictive columns from q + r - p.
pub fn build_three_ball_feedback_action_prediction_spm_compatible(
    space: &ThreeBallRuleSpace,
    full_posterior: &DirichletConcentrationTensor,
    models: &[ReducedDirichletModelHypothesis],
    ball_colours: [ThreeBallColour; 3],
    gaze: Option<ThreeBallLocation>,
    choice: Option<ThreeBallColour>,
    action_id: impl Into<String>,
) -> Result<DirichletActionPrediction, ThreeBallFeedbackDecisionError> {
    build_three_ball_feedback_action_prediction_with_mode(
        space,
        full_posterior,
        models,
        ball_colours,
        gaze,
        choice,
        action_id,
        true,
    )
}

fn build_three_ball_feedback_action_prediction_with_mode(
    space: &ThreeBallRuleSpace,
    full_posterior: &DirichletConcentrationTensor,
    models: &[ReducedDirichletModelHypothesis],
    ball_colours: [ThreeBallColour; 3],
    gaze: Option<ThreeBallLocation>,
    choice: Option<ThreeBallColour>,
    action_id: impl Into<String>,
    spm_compatible: bool,
) -> Result<DirichletActionPrediction, ThreeBallFeedbackDecisionError> {
    let action_id = action_id.into();
    let full_prior = if spm_compatible {
        &space.spm_model_average_prior
    } else {
        &space.full_model_average_prior
    };
    let current_belief = if spm_compatible {
        reduce_dirichlet_models_spm_compatible(full_posterior, full_prior, models)?
    } else {
        reduce_dirichlet_models(full_posterior, full_prior, models)?
    };
    let model_probabilities = current_belief
        .models
        .iter()
        .map(|model| (model.model_id.clone(), model.posterior_probability))
        .collect::<BTreeMap<_, _>>();

    let choice_index = choice.map(|value| value as usize).unwrap_or(3);
    let gaze_index = gaze.map(|value| 1 + value as usize).unwrap_or(0);
    let [left, centre, right] = ball_colours.map(|value| value as usize);
    let column_index = choice_index * 4 * 27 + gaze_index * 27 + right * 9 + centre * 3 + left;
    if column_index >= full_prior.columns.len() || column_index >= full_posterior.columns.len() {
        return Err(ThreeBallFeedbackDecisionError::InvalidLikelihoodColumn {
            column_index,
            column_count: full_prior.columns.len().min(full_posterior.columns.len()),
        });
    }

    let mut per_model_predictions = Vec::with_capacity(models.len());
    for model in models {
        let posterior_model_probability = model_probabilities
            .get(&model.model_id)
            .copied()
            .unwrap_or(0.0);
        // Zero-posterior hypotheses remain in BMR receipts but do not belong
        // to the posterior-predictive mixture or require undefined predictions.
        if posterior_model_probability <= 0.0 {
            continue;
        }
        let mut concentrations = Vec::with_capacity(THREE_BALL_FEEDBACK_OUTCOME_COUNT);
        for outcome_index in 0..THREE_BALL_FEEDBACK_OUTCOME_COUNT {
            let active = model.parameter_prior.active_support[column_index][outcome_index];
            let value = if posterior_model_probability > 0.0 {
                full_posterior.columns[column_index][outcome_index]
                    + model.parameter_prior.columns[column_index][outcome_index]
                    - full_prior.columns[column_index][outcome_index]
            } else {
                model.parameter_prior.columns[column_index][outcome_index]
            };
            if spm_compatible {
                // SPM's BMR smooths all coordinates for model evidence, then
                // removes that pseudocount before normalizing posterior means.
                // Thus data may support an outcome outside a sparse source rule.
                if !value.is_finite() || value < -1e-9 {
                    return Err(
                        ThreeBallFeedbackDecisionError::InvalidPosteriorPredictiveConcentration {
                            model_id: model.model_id.clone(),
                            outcome_index,
                        },
                    );
                }
                concentrations.push(value.max(0.0));
            } else {
                if active && (!value.is_finite() || value <= 0.0) {
                    return Err(
                        ThreeBallFeedbackDecisionError::InvalidPosteriorPredictiveConcentration {
                            model_id: model.model_id.clone(),
                            outcome_index,
                        },
                    );
                }
                if !active && value != 0.0 {
                    return Err(
                        ThreeBallFeedbackDecisionError::InvalidPosteriorPredictiveConcentration {
                            model_id: model.model_id.clone(),
                            outcome_index,
                        },
                    );
                }
                concentrations.push(if active { value } else { 0.0 });
            }
        }
        let total = concentrations.iter().sum::<f64>();
        let probabilities = if spm_compatible && total == 0.0 {
            vec![1.0 / THREE_BALL_FEEDBACK_OUTCOME_COUNT as f64; THREE_BALL_FEEDBACK_OUTCOME_COUNT]
        } else {
            if !total.is_finite() || total <= 0.0 {
                return Err(
                    ThreeBallFeedbackDecisionError::EmptyPosteriorPredictiveColumn {
                        model_id: model.model_id.clone(),
                    },
                );
            }
            concentrations
                .iter()
                .map(|value| value / total)
                .collect::<Vec<_>>()
        };
        let probability_mass = probabilities.iter().sum::<f64>();
        if !probability_mass.is_finite() || (probability_mass - 1.0).abs() > 1e-10 {
            return Err(
                ThreeBallFeedbackDecisionError::ModelPosteriorPredictiveMassNotNormalized {
                    model_id: model.model_id.clone(),
                    sum: probability_mass,
                },
            );
        }
        per_model_predictions.push((model.model_id.clone(), probabilities));
    }

    let mut mixture = [0.0_f64; THREE_BALL_FEEDBACK_OUTCOME_COUNT];
    for (model_id, probabilities) in &per_model_predictions {
        let model_probability = model_probabilities.get(model_id).copied().unwrap_or(0.0);
        for outcome_index in 0..THREE_BALL_FEEDBACK_OUTCOME_COUNT {
            mixture[outcome_index] += model_probability * probabilities[outcome_index];
        }
    }

    let mut outcomes = Vec::new();
    for outcome_index in 0..THREE_BALL_FEEDBACK_OUTCOME_COUNT {
        let mixture_probability = mixture[outcome_index];
        if mixture_probability <= 0.0 {
            continue;
        }
        let mut posterior_count_increment = vec![
            vec![0.0; THREE_BALL_FEEDBACK_OUTCOME_COUNT];
            THREE_BALL_HIDDEN_STATE_COLUMN_COUNT
        ];
        posterior_count_increment[column_index][outcome_index] = 1.0;
        let feedback_id = match outcome_index {
            0 => "no_feedback",
            1 => "correct",
            2 => "incorrect",
            _ => unreachable!("exactly three feedback outcomes"),
        };
        outcomes.push(DirichletPredictedOutcome {
            outcome_id: feedback_id.to_owned(),
            full_model_predictive_probability: mixture_probability,
            model_predictive_probabilities: per_model_predictions
                .iter()
                .map(|(model_id, probabilities)| (model_id.clone(), probabilities[outcome_index]))
                .collect(),
            posterior_count_increment,
        });
    }

    Ok(DirichletActionPrediction {
        action_id,
        outcomes,
    })
}

/// Score legal known-state feedback actions using a scoped one-step FEP objective.
///
/// This uses:
///
///   G(a) = KL[Q(feedback | a) || C(feedback)]
///          - epistemic_weight * I(model; feedback | a)
///          + action_cost.
///
/// The state is assumed known and gaze is fixed, so this is a feedback-only
/// one-step objective. It deliberately omits hidden-state information gain,
/// visual observation ambiguity, multi-step planning, and safety constraints;
/// it is not the full three-ball POMDP or the production ActiveInferenceAgent.
pub fn score_three_ball_known_state_feedback_actions(
    space: &ThreeBallRuleSpace,
    full_posterior: &DirichletConcentrationTensor,
    models: &[ReducedDirichletModelHypothesis],
    ball_colours: [ThreeBallColour; 3],
    gaze: Option<ThreeBallLocation>,
    actions: &[ThreeBallFeedbackActionSpec],
    config: ThreeBallFeedbackDecisionConfig,
) -> Result<ThreeBallFeedbackDecision, ThreeBallFeedbackDecisionError> {
    score_three_ball_known_state_feedback_actions_with_mode(
        space,
        full_posterior,
        models,
        ball_colours,
        gaze,
        actions,
        config,
        false,
    )
}

/// Score the same one-step known-state feedback decision using the pinned SPM
/// BMR semantics. This remains a scoped feedback-only evaluator, not the full POMDP.
pub fn score_three_ball_known_state_feedback_actions_spm_compatible(
    space: &ThreeBallRuleSpace,
    full_posterior: &DirichletConcentrationTensor,
    models: &[ReducedDirichletModelHypothesis],
    ball_colours: [ThreeBallColour; 3],
    gaze: Option<ThreeBallLocation>,
    actions: &[ThreeBallFeedbackActionSpec],
    config: ThreeBallFeedbackDecisionConfig,
) -> Result<ThreeBallFeedbackDecision, ThreeBallFeedbackDecisionError> {
    score_three_ball_known_state_feedback_actions_with_mode(
        space,
        full_posterior,
        models,
        ball_colours,
        gaze,
        actions,
        config,
        true,
    )
}

fn score_three_ball_known_state_feedback_actions_with_mode(
    space: &ThreeBallRuleSpace,
    full_posterior: &DirichletConcentrationTensor,
    models: &[ReducedDirichletModelHypothesis],
    ball_colours: [ThreeBallColour; 3],
    gaze: Option<ThreeBallLocation>,
    actions: &[ThreeBallFeedbackActionSpec],
    config: ThreeBallFeedbackDecisionConfig,
    spm_compatible: bool,
) -> Result<ThreeBallFeedbackDecision, ThreeBallFeedbackDecisionError> {
    if actions.is_empty() {
        return Err(ThreeBallFeedbackDecisionError::EmptyActionSet);
    }
    if !config.epistemic_weight.is_finite() || config.epistemic_weight < 0.0 {
        return Err(ThreeBallFeedbackDecisionError::InvalidEpistemicWeight(
            config.epistemic_weight,
        ));
    }
    if config
        .log_preferences_nats
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(ThreeBallFeedbackDecisionError::NonFinitePreference);
    }

    let max_log_preference = config
        .log_preferences_nats
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let log_normalizer = max_log_preference
        + config
            .log_preferences_nats
            .iter()
            .map(|value| (value - max_log_preference).exp())
            .sum::<f64>()
            .ln();
    let normalized_log_preferences = config
        .log_preferences_nats
        .map(|value| value - log_normalizer);

    let mut seen_action_ids = std::collections::BTreeSet::new();
    let mut action_scores = Vec::with_capacity(actions.len());
    for (action_index, action) in actions.iter().enumerate() {
        if action.action_id.trim().is_empty() {
            return Err(ThreeBallFeedbackDecisionError::EmptyActionId {
                index: action_index,
            });
        }
        if !seen_action_ids.insert(action.action_id.clone()) {
            return Err(ThreeBallFeedbackDecisionError::DuplicateActionId(
                action.action_id.clone(),
            ));
        }
        if !action.action_cost_nats.is_finite() || action.action_cost_nats < 0.0 {
            return Err(ThreeBallFeedbackDecisionError::InvalidActionCost {
                action_id: action.action_id.clone(),
                cost: action.action_cost_nats,
            });
        }

        let prediction = if spm_compatible {
            build_three_ball_feedback_action_prediction_spm_compatible(
                space,
                full_posterior,
                models,
                ball_colours,
                gaze,
                action.choice,
                action.action_id.clone(),
            )?
        } else {
            build_three_ball_feedback_action_prediction(
                space,
                full_posterior,
                models,
                ball_colours,
                gaze,
                action.choice,
                action.action_id.clone(),
            )?
        };
        let selected_full_prior = if spm_compatible {
            &space.spm_model_average_prior
        } else {
            &space.full_model_average_prior
        };
        let information_gain = if spm_compatible {
            evaluate_dirichlet_model_information_gain_spm_compatible(
                full_posterior,
                selected_full_prior,
                models,
                &prediction,
            )?
        } else {
            evaluate_dirichlet_model_information_gain(
                full_posterior,
                selected_full_prior,
                models,
                &prediction,
            )?
        };

        let mut predictive_feedback = Vec::with_capacity(information_gain.outcomes.len());
        let mut pragmatic_risk_nats = 0.0;
        for outcome in &information_gain.outcomes {
            let (feedback, outcome_index) = match outcome.outcome_id.as_str() {
                "no_feedback" => (ThreeBallFeedback::NoFeedback, 0),
                "correct" => (ThreeBallFeedback::Correct, 1),
                "incorrect" => (ThreeBallFeedback::Incorrect, 2),
                other => {
                    return Err(ThreeBallFeedbackDecisionError::UnknownFeedbackOutcome {
                        action_id: action.action_id.clone(),
                        outcome_id: other.to_owned(),
                    });
                }
            };
            let probability = outcome.predictive_probability;
            if probability > 0.0 {
                pragmatic_risk_nats +=
                    probability * (probability.ln() - normalized_log_preferences[outcome_index]);
            }
            predictive_feedback.push((feedback, probability));
        }
        if !pragmatic_risk_nats.is_finite() || pragmatic_risk_nats < -1e-10 {
            return Err(ThreeBallFeedbackDecisionError::InvalidPragmaticRisk(
                pragmatic_risk_nats,
            ));
        }
        let pragmatic_risk_nats = pragmatic_risk_nats.max(0.0);
        let epistemic_value_nats =
            config.epistemic_weight * information_gain.model_information_gain_nats;
        let expected_free_energy_nats =
            pragmatic_risk_nats - epistemic_value_nats + action.action_cost_nats;
        if !expected_free_energy_nats.is_finite() {
            return Err(ThreeBallFeedbackDecisionError::InvalidPragmaticRisk(
                expected_free_energy_nats,
            ));
        }

        action_scores.push(ThreeBallFeedbackActionScore {
            action_id: action.action_id.clone(),
            choice: action.choice,
            pragmatic_risk_nats,
            model_information_gain_nats: information_gain.model_information_gain_nats,
            epistemic_value_nats,
            action_cost_nats: action.action_cost_nats,
            expected_free_energy_nats,
            predictive_feedback,
            model_information_gain: information_gain,
        });
    }

    let selected_action_index = (0..action_scores.len())
        .min_by(|left, right| {
            action_scores[*left]
                .expected_free_energy_nats
                .total_cmp(&action_scores[*right].expected_free_energy_nats)
                .then_with(|| left.cmp(right))
        })
        .ok_or(ThreeBallFeedbackDecisionError::EmptyActionSet)?;

    Ok(ThreeBallFeedbackDecision {
        selected_action_index,
        selected_action_id: action_scores[selected_action_index].action_id.clone(),
        action_scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dirichlet_model_reduction::{
        DirichletActionPrediction, evaluate_dirichlet_model_information_gain,
        evaluate_dirichlet_model_information_gain_spm_compatible, reduce_dirichlet_models,
        reduce_dirichlet_models_spm_compatible,
    };

    #[test]
    fn reproduces_81_source_hypotheses_and_rule_65() {
        let space = build_three_ball_rule_space();
        assert_eq!(space.rules.len(), 81);
        assert_eq!(space.planted_rule().one_based_index, 65);
        assert_eq!(
            space.planted_rule().context_factor,
            ThreeBallLocation::Centre
        );
        assert_eq!(
            space.planted_rule().criterion_by_context_colour,
            [
                ThreeBallLocation::Left,
                ThreeBallLocation::Centre,
                ThreeBallLocation::Right
            ]
        );

        let rule = space.planted_rule();
        assert_eq!(
            rule.feedback(
                [
                    ThreeBallColour::Red,
                    ThreeBallColour::Red,
                    ThreeBallColour::Blue
                ],
                Some(ThreeBallColour::Red)
            ),
            ThreeBallFeedback::Correct
        );
        assert_eq!(
            rule.feedback(
                [
                    ThreeBallColour::Blue,
                    ThreeBallColour::Green,
                    ThreeBallColour::Red
                ],
                Some(ThreeBallColour::Green)
            ),
            ThreeBallFeedback::Correct
        );
        assert_eq!(
            rule.feedback(
                [
                    ThreeBallColour::Red,
                    ThreeBallColour::Blue,
                    ThreeBallColour::Green
                ],
                Some(ThreeBallColour::Green)
            ),
            ThreeBallFeedback::Incorrect
        );
        assert_eq!(
            rule.feedback(
                [
                    ThreeBallColour::Red,
                    ThreeBallColour::Green,
                    ThreeBallColour::Blue
                ],
                None
            ),
            ThreeBallFeedback::NoFeedback
        );
    }

    #[test]
    fn audits_public_generator_semantic_duplicates_without_forcing_paper_count() {
        let space = build_three_ball_rule_space();
        assert_eq!(space.semantic_unique_rule_count, 75);
        assert_eq!(space.semantic_duplicate_count(), 6);
        assert_eq!(space.semantic_duplicate_groups.len(), 3);
        assert!(
            space
                .semantic_duplicate_groups
                .iter()
                .all(|group| group.len() == 3)
        );
        assert_eq!(
            space.paper_reported_unique_rule_count,
            THREE_BALL_PAPER_REPORTED_UNIQUE_RULE_COUNT
        );
        assert!(!space.unique_count_matches_paper());
    }

    #[test]
    fn public_generator_duplicates_are_exactly_the_three_constant_criterion_mappings() {
        let space = build_three_ball_rule_space();
        assert_eq!(space.semantic_duplicate_groups.len(), 3);

        let mut duplicated_criterion_locations = std::collections::BTreeSet::new();
        for group in &space.semantic_duplicate_groups {
            assert_eq!(group.len(), 3);
            let rules = group
                .iter()
                .map(|one_based_index| &space.rules[one_based_index - 1])
                .collect::<Vec<_>>();

            let criterion_locations = rules[0].criterion_by_context_colour;
            assert!(
                criterion_locations
                    .iter()
                    .all(|location| *location == criterion_locations[0]),
                "only a context-independent criterion mapping may be a duplicate group"
            );
            assert!(duplicated_criterion_locations.insert(criterion_locations[0]));
            assert_eq!(
                rules
                    .iter()
                    .map(|rule| rule.context_factor)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                3,
                "the same constant mapping must appear under each context factor"
            );
            assert!(rules.iter().all(|rule| {
                rule.criterion_by_context_colour == criterion_locations
                    && rule
                        .parameter_prior
                        .columns
                        .iter()
                        .zip(rules[0].parameter_prior.columns.iter())
                        .all(|(left, right)| left == right)
            }));
        }

        assert_eq!(duplicated_criterion_locations.len(), 3);
        assert_eq!(
            duplicated_criterion_locations,
            [
                ThreeBallLocation::Left,
                ThreeBallLocation::Centre,
                ThreeBallLocation::Right,
            ]
            .into_iter()
            .collect()
        );

        for rule in &space.rules {
            let constant_mapping = rule
                .criterion_by_context_colour
                .iter()
                .all(|location| *location == rule.criterion_by_context_colour[0]);
            if !constant_mapping {
                assert!(
                    rule.duplicate_of_one_based_index.is_none(),
                    "non-constant mapping {} must have a distinct feedback tensor",
                    rule.one_based_index
                );
            }
        }
        assert_eq!(
            space.rules.len() - space.semantic_duplicate_count(),
            THREE_BALL_PUBLIC_GENERATOR_UNIQUE_TENSOR_COUNT
        );
    }

    #[test]
    fn semantic_quotient_preserves_the_model_average_prior() {
        let space = build_three_ball_rule_space();
        let compressed = space.semantic_unique_model_hypotheses();
        assert_eq!(
            compressed.len(),
            THREE_BALL_PUBLIC_GENERATOR_UNIQUE_TENSOR_COUNT
        );
        assert!(
            (compressed
                .iter()
                .map(|model| model.prior_probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );

        for column in 0..THREE_BALL_HIDDEN_STATE_COLUMN_COUNT {
            for outcome in 0..THREE_BALL_FEEDBACK_OUTCOME_COUNT {
                let compressed_average = compressed
                    .iter()
                    .map(|model| {
                        model.prior_probability * model.parameter_prior.columns[column][outcome]
                    })
                    .sum::<f64>();
                let original_average = space.full_model_average_prior.columns[column][outcome];
                assert!((compressed_average - original_average).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn semantic_quotient_preserves_bmr_evidence_and_aggregated_posterior() {
        let space = build_three_ball_rule_space();
        let original_models = space.reduced_model_hypotheses();
        let compressed_models = space.semantic_unique_model_hypotheses();

        let mut posterior = space.full_model_average_prior.clone();
        // A deterministic, model-independent observation increment over the
        // active coordinates of the BMA likelihood tensor.
        for column in 0..posterior.columns.len() {
            for outcome in 0..posterior.columns[column].len() {
                if posterior.active_support[column][outcome] {
                    posterior.columns[column][outcome] += 0.25;
                }
            }
        }

        let original_result = reduce_dirichlet_models(
            &posterior,
            &space.full_model_average_prior,
            &original_models,
        )
        .expect("uncompressed BMR");
        let compressed_result = reduce_dirichlet_models(
            &posterior,
            &space.full_model_average_prior,
            &compressed_models,
        )
        .expect("quotient BMR");

        let mut original_probabilities = std::collections::BTreeMap::<String, f64>::new();
        for model in original_result.models {
            original_probabilities.insert(model.model_id, model.posterior_probability);
        }
        for compressed in &compressed_result.models {
            let representative_index: usize = compressed
                .model_id
                .rsplit('-')
                .next()
                .expect("index suffix")
                .parse()
                .expect("numeric representative index");
            let source_group = space
                .rules
                .iter()
                .filter(|rule| {
                    rule.one_based_index == representative_index
                        || rule.duplicate_of_one_based_index == Some(representative_index)
                })
                .collect::<Vec<_>>();
            let aggregated_original = source_group
                .iter()
                .map(|rule| original_probabilities[&rule.model_id])
                .sum::<f64>();
            assert!(
                (aggregated_original - compressed.posterior_probability).abs() < 1e-9,
                "semantic class {representative_index} posterior changed under quotienting"
            );
        }
    }

    fn feedback_action_prediction(
        space: &ThreeBallRuleSpace,
        posterior: &DirichletConcentrationTensor,
        models: &[ReducedDirichletModelHypothesis],
        ball_colours: [ThreeBallColour; 3],
        gaze: Option<ThreeBallLocation>,
        choice: Option<ThreeBallColour>,
    ) -> DirichletActionPrediction {
        build_three_ball_feedback_action_prediction(
            space,
            posterior,
            models,
            ball_colours,
            gaze,
            choice,
            format!("gaze={gaze:?};choice={choice:?}"),
        )
        .expect("known-state feedback prediction")
    }

    #[test]
    fn three_ball_action_eig_and_class_posteriors_are_invariant_under_semantic_quotient() {
        let space = build_three_ball_rule_space();
        let labelled_models = space.reduced_model_hypotheses();
        let quotient_models = space.semantic_unique_model_hypotheses();
        let full_prior = &space.full_model_average_prior;
        let ball_colours = [
            ThreeBallColour::Red,
            ThreeBallColour::Green,
            ThreeBallColour::Blue,
        ];
        let choices = [
            Some(ThreeBallColour::Red),
            Some(ThreeBallColour::Green),
            Some(ThreeBallColour::Blue),
            None,
        ];

        let mut labelled_eig = Vec::with_capacity(choices.len());
        let mut quotient_eig = Vec::with_capacity(choices.len());
        for choice in choices {
            let labelled_action = feedback_action_prediction(
                &space,
                full_prior,
                &labelled_models,
                ball_colours,
                Some(ThreeBallLocation::Centre),
                choice,
            );
            let quotient_action = feedback_action_prediction(
                &space,
                full_prior,
                &quotient_models,
                ball_colours,
                Some(ThreeBallLocation::Centre),
                choice,
            );
            let source_result = evaluate_dirichlet_model_information_gain(
                full_prior,
                full_prior,
                &labelled_models,
                &labelled_action,
            )
            .expect("source-labelled action EIG");
            let quotient_result = evaluate_dirichlet_model_information_gain(
                full_prior,
                full_prior,
                &quotient_models,
                &quotient_action,
            )
            .expect("semantic quotient action EIG");

            assert!(
                (source_result.predictive_probability_mass
                    - quotient_result.predictive_probability_mass)
                    .abs()
                    < 1e-12
            );
            assert_eq!(source_result.outcomes.len(), quotient_result.outcomes.len());
            assert!(
                (source_result.model_information_gain_nats
                    - quotient_result.model_information_gain_nats)
                    .abs()
                    < 1e-10,
                "model EIG changed for action {}",
                labelled_action.action_id
            );

            for (source_outcome, quotient_outcome) in source_result
                .outcomes
                .iter()
                .zip(quotient_result.outcomes.iter())
            {
                assert_eq!(source_outcome.outcome_id, quotient_outcome.outcome_id);
                assert!(
                    (source_outcome.predictive_probability
                        - quotient_outcome.predictive_probability)
                        .abs()
                        < 1e-12
                );
                for quotient_model in &quotient_outcome.posterior_model_probabilities {
                    let representative_index: usize = quotient_model
                        .0
                        .rsplit('-')
                        .next()
                        .expect("semantic rule index")
                        .parse()
                        .expect("numeric rule index");
                    let aggregated_source_posterior = source_outcome
                        .posterior_model_probabilities
                        .iter()
                        .filter(|(model_id, _)| {
                            space.rules.iter().any(|rule| {
                                rule.model_id.as_str() == model_id.as_str()
                                    && (rule.one_based_index == representative_index
                                        || rule.duplicate_of_one_based_index
                                            == Some(representative_index))
                            })
                        })
                        .map(|(_, probability)| *probability)
                        .sum::<f64>();
                    assert!(
                        (aggregated_source_posterior - quotient_model.1).abs() < 1e-9,
                        "posterior mass differs for semantic rule {representative_index}"
                    );
                }
            }

            labelled_eig.push(source_result.model_information_gain_nats);
            quotient_eig.push(quotient_result.model_information_gain_nats);
        }

        assert!(
            labelled_eig
                .iter()
                .zip(quotient_eig.iter())
                .all(|(source, quotient)| (source - quotient).abs() < 1e-10)
        );
        let source_best = labelled_eig
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("non-empty action set");
        let quotient_best = quotient_eig
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("non-empty action set");
        assert_eq!(source_best, quotient_best);
    }

    #[test]
    fn known_state_feedback_action_evaluator_separates_risk_and_model_information() {
        let space = build_three_ball_rule_space();
        let models = space.reduced_model_hypotheses();
        let posterior = space.full_model_average_prior.clone();
        let ball_colours = [
            ThreeBallColour::Red,
            ThreeBallColour::Green,
            ThreeBallColour::Blue,
        ];
        let actions = vec![
            ThreeBallFeedbackActionSpec {
                action_id: "choose-red".into(),
                choice: Some(ThreeBallColour::Red),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "choose-green".into(),
                choice: Some(ThreeBallColour::Green),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "choose-blue".into(),
                choice: Some(ThreeBallColour::Blue),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "no-choice".into(),
                choice: None,
                action_cost_nats: 0.0,
            },
        ];
        let decision = score_three_ball_known_state_feedback_actions(
            &space,
            &posterior,
            &models,
            ball_colours,
            Some(ThreeBallLocation::Centre),
            &actions,
            ThreeBallFeedbackDecisionConfig::default(),
        )
        .expect("finite one-step known-state decision");

        assert_eq!(decision.action_scores.len(), 4);
        assert_eq!(
            decision.selected_action_id,
            decision.action_scores[decision.selected_action_index].action_id
        );
        for score in &decision.action_scores {
            assert!(score.pragmatic_risk_nats.is_finite());
            assert!(score.model_information_gain_nats >= 0.0);
            assert!(score.expected_free_energy_nats.is_finite());
            assert!(
                (score.expected_free_energy_nats
                    - (score.pragmatic_risk_nats - score.epistemic_value_nats
                        + score.action_cost_nats))
                    .abs()
                    < 1e-12
            );
            assert!(
                (score
                    .predictive_feedback
                    .iter()
                    .map(|(_, p)| p)
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1e-10
            );
        }

        let no_choice = decision
            .action_scores
            .iter()
            .find(|score| score.action_id == "no-choice")
            .expect("no-choice action");
        assert_eq!(no_choice.model_information_gain_nats, 0.0);
        assert_eq!(
            no_choice.predictive_feedback,
            vec![(ThreeBallFeedback::NoFeedback, 1.0)]
        );
    }

    #[test]
    fn pragmatic_known_state_policy_selects_planted_rule_choice_after_feedback_evidence() {
        let space = build_three_ball_rule_space();
        let models = space.reduced_model_hypotheses();
        let mut posterior = space.full_model_average_prior.clone();
        let ball_colours = [
            ThreeBallColour::Red,
            ThreeBallColour::Green,
            ThreeBallColour::Blue,
        ];
        // Rule 65 uses the centre ball as context; green therefore makes
        // the centre green ball the criterion and a green choice correct.
        // One observed correct-feedback count removes hypotheses that
        // assigned zero probability to this known-state event.
        let feedback_column = 1 * 4 * 27 + 2 * 27 + 2 * 9 + 1 * 3;
        posterior.columns[feedback_column][ThreeBallFeedback::Correct as usize] += 1.0;

        let actions = vec![
            ThreeBallFeedbackActionSpec {
                action_id: "choose-red".into(),
                choice: Some(ThreeBallColour::Red),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "choose-green".into(),
                choice: Some(ThreeBallColour::Green),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "choose-blue".into(),
                choice: Some(ThreeBallColour::Blue),
                action_cost_nats: 0.0,
            },
        ];
        let mut config = ThreeBallFeedbackDecisionConfig::default();
        config.epistemic_weight = 0.0;
        let decision = score_three_ball_known_state_feedback_actions(
            &space,
            &posterior,
            &models,
            ball_colours,
            Some(ThreeBallLocation::Centre),
            &actions,
            config,
        )
        .expect("pragmatic posterior-predictive action selection");

        assert_eq!(decision.selected_action_id, "choose-green");
        let green = decision
            .action_scores
            .iter()
            .find(|score| score.action_id == "choose-green")
            .expect("green choice");
        assert_eq!(
            green.predictive_feedback,
            vec![(ThreeBallFeedback::Correct, 1.0)]
        );
        assert_eq!(green.model_information_gain_nats, 0.0);
    }

    #[test]
    fn known_state_decision_rejects_negative_cost_and_invalid_epistemic_weight() {
        let space = build_three_ball_rule_space();
        let models = space.reduced_model_hypotheses();
        let posterior = space.full_model_average_prior.clone();
        let action = [ThreeBallFeedbackActionSpec {
            action_id: "choose-green".into(),
            choice: Some(ThreeBallColour::Green),
            action_cost_nats: -0.1,
        }];
        assert!(matches!(
            score_three_ball_known_state_feedback_actions(
                &space,
                &posterior,
                &models,
                [
                    ThreeBallColour::Red,
                    ThreeBallColour::Green,
                    ThreeBallColour::Blue
                ],
                Some(ThreeBallLocation::Centre),
                &action,
                ThreeBallFeedbackDecisionConfig::default(),
            ),
            Err(ThreeBallFeedbackDecisionError::InvalidActionCost { .. })
        ));

        let valid_action = [ThreeBallFeedbackActionSpec {
            action_id: "choose-green".into(),
            choice: Some(ThreeBallColour::Green),
            action_cost_nats: 0.0,
        }];
        let invalid_config = ThreeBallFeedbackDecisionConfig {
            epistemic_weight: f64::NAN,
            ..Default::default()
        };
        assert!(matches!(
            score_three_ball_known_state_feedback_actions(
                &space,
                &posterior,
                &models,
                [
                    ThreeBallColour::Red,
                    ThreeBallColour::Green,
                    ThreeBallColour::Blue
                ],
                Some(ThreeBallLocation::Centre),
                &valid_action,
                invalid_config,
            ),
            Err(ThreeBallFeedbackDecisionError::InvalidEpistemicWeight(_))
        ));
    }

    #[test]
    fn parameterization_digest_is_bound_to_a_canonical_descriptor() {
        assert_eq!(
            THREE_BALL_PARAMETERIZATION_DESCRIPTOR,
            "symthaea-three-ball-feedback-v1|outcomes=no-feedback,correct,incorrect|factors=left,centre,right,gaze,choice|left-fastest"
        );
        assert_eq!(THREE_BALL_PARAMETERIZATION_DIGEST.len(), 64);
        assert!(
            THREE_BALL_PARAMETERIZATION_DIGEST
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[test]
    fn likelihood_tensor_uses_spm_column_order_and_outcome_order() {
        let space = build_three_ball_rule_space();
        let rule = space.planted_rule();
        let columns = &rule.parameter_prior.columns;

        // SPM column zero is left=red, centre=red, right=red,
        // gaze=left, choice=red. Centre red selects left red: correct.
        assert_eq!(columns[0], vec![0.0, 1.0, 0.0]);

        // Column 3 changes centre to green while left and right stay red:
        // rule 65 selects centre green, so choosing red is incorrect.
        assert_eq!(columns[3], vec![0.0, 0.0, 1.0]);

        // First no-choice block: feedback is no-feedback for every scene.
        assert_eq!(columns[3 * 3 * 3 * 4 * 3], vec![1.0, 0.0, 0.0]);

        // State left=red, centre=green, right=blue, gaze=left, choice=red.
        // Rule 65 selects centre green, so choosing red is incorrect.
        let centre_green_state = 3 + 2 * 9;
        assert_eq!(columns[centre_green_state], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn model_average_prior_is_the_arithmetic_mean_of_all_81_reduced_priors() {
        let space = build_three_ball_rule_space();
        assert_eq!(space.full_model_average_prior.columns.len(), 432);
        assert_eq!(space.full_model_average_prior.active_support.len(), 432);
        for column in 0..432 {
            let mut column_mass = 0.0;
            for outcome in 0..3 {
                let expected = space
                    .rules
                    .iter()
                    .map(|rule| {
                        rule.parameter_prior.columns[column][outcome] * rule.prior_probability
                    })
                    .sum::<f64>();
                let actual = space.full_model_average_prior.columns[column][outcome];
                assert!((actual - expected).abs() < 1e-12);
                assert_eq!(
                    space.full_model_average_prior.active_support[column][outcome],
                    actual > 0.0
                );
                column_mass += actual;
            }
            assert!((column_mass - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn no_evidence_gives_equal_posterior_weight_to_all_81_labelled_hypotheses() {
        let space = build_three_ball_rule_space();
        let hypotheses = space.reduced_model_hypotheses();
        let posterior = space.full_model_average_prior.clone();
        let result =
            reduce_dirichlet_models(&posterior, &space.full_model_average_prior, &hypotheses)
                .expect("initial BMR");
        assert_eq!(result.models.len(), 81);
        for model in result.models {
            assert!((model.posterior_probability - 1.0 / 81.0).abs() < 1e-9);
            assert!(model.log_evidence_ratio.abs() < 1e-8);
        }
    }
    #[test]
    fn spm_full_model_average_prior_matches_global_tensor_normalization() {
        let space = build_three_ball_rule_space();
        let prior = &space.spm_model_average_prior;
        let raw_total = space
            .rules
            .iter()
            .map(|rule| {
                rule.prior_probability * rule.parameter_prior.columns.iter().flatten().sum::<f64>()
            })
            .sum::<f64>();
        assert!((prior.columns.iter().flatten().sum::<f64>() - 1.0).abs() < 1e-12);
        for column in 0..THREE_BALL_HIDDEN_STATE_COLUMN_COUNT {
            let column_mass = prior.columns[column].iter().sum::<f64>();
            assert!(
                (column_mass - 1.0 / THREE_BALL_HIDDEN_STATE_COLUMN_COUNT as f64).abs() < 1e-12
            );
            for outcome in 0..THREE_BALL_FEEDBACK_OUTCOME_COUNT {
                let raw = space
                    .rules
                    .iter()
                    .map(|rule| {
                        rule.prior_probability * rule.parameter_prior.columns[column][outcome]
                    })
                    .sum::<f64>();
                assert!((prior.columns[column][outcome] - raw / raw_total).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn spm_bmr_with_no_evidence_preserves_equal_prior_over_all_labelled_rules() {
        let space = build_three_ball_rule_space();
        let models = space.reduced_model_hypotheses();
        let prior = &space.spm_model_average_prior;
        let result = reduce_dirichlet_models_spm_compatible(prior, prior, &models)
            .expect("source-faithful BMR at zero evidence");
        assert_eq!(
            result.models.len(),
            THREE_BALL_CONTEXT_RULE_HYPOTHESIS_COUNT
        );
        for model in result.models {
            assert!((model.posterior_probability - 1.0 / 81.0).abs() < 1e-10);
            assert!(model.log_evidence_ratio.abs() < 1e-10);
        }
    }

    #[test]
    fn spm_pseudocount_mode_does_not_treat_a_reduced_model_zero_as_impossible() {
        let space = build_three_ball_rule_space();
        let models = space.reduced_model_hypotheses();
        let prior = &space.spm_model_average_prior;
        let mut posterior = prior.clone();
        let first_model = &models[0];
        let mut selected = None;
        'search: for column in 0..prior.columns.len() {
            for outcome in 0..prior.columns[column].len() {
                if prior.active_support[column][outcome]
                    && !first_model.parameter_prior.active_support[column][outcome]
                {
                    selected = Some((column, outcome));
                    break 'search;
                }
            }
        }
        let (column, outcome) =
            selected.expect("one source rule excludes outcomes supported by others");
        posterior.columns[column][outcome] += 1.0;

        let strict = reduce_dirichlet_models(&posterior, prior, &models)
            .expect("strict-support reduction retains other supported hypotheses");
        let source_compatible = reduce_dirichlet_models_spm_compatible(&posterior, prior, &models)
            .expect("SPM smoothing keeps each reduced likelihood column integrable");
        let strict_first = strict
            .models
            .iter()
            .find(|model| model.model_id == first_model.model_id)
            .expect("strict first model");
        let spm_first = source_compatible
            .models
            .iter()
            .find(|model| model.model_id == first_model.model_id)
            .expect("SPM first model");
        assert_eq!(strict_first.log_evidence_ratio, f64::NEG_INFINITY);
        assert!(spm_first.log_evidence_ratio.is_finite());
    }

    #[test]
    fn spm_three_ball_action_eig_is_invariant_under_semantic_quotient() {
        let space = build_three_ball_rule_space();
        let labelled_models = space.reduced_model_hypotheses();
        let quotient_models = space.semantic_unique_model_hypotheses();
        let prior = &space.spm_model_average_prior;
        let ball_colours = [
            ThreeBallColour::Red,
            ThreeBallColour::Green,
            ThreeBallColour::Blue,
        ];
        for choice in [
            Some(ThreeBallColour::Red),
            Some(ThreeBallColour::Green),
            Some(ThreeBallColour::Blue),
            None,
        ] {
            let labelled_action = build_three_ball_feedback_action_prediction_spm_compatible(
                &space,
                prior,
                &labelled_models,
                ball_colours,
                Some(ThreeBallLocation::Centre),
                choice,
                format!("spm-labelled-{choice:?}"),
            )
            .expect("labelled SPM action prediction");
            let quotient_action = build_three_ball_feedback_action_prediction_spm_compatible(
                &space,
                prior,
                &quotient_models,
                ball_colours,
                Some(ThreeBallLocation::Centre),
                choice,
                format!("spm-quotient-{choice:?}"),
            )
            .expect("quotient SPM action prediction");

            let labelled = evaluate_dirichlet_model_information_gain_spm_compatible(
                prior,
                prior,
                &labelled_models,
                &labelled_action,
            )
            .expect("labelled SPM model EIG");
            let quotient = evaluate_dirichlet_model_information_gain_spm_compatible(
                prior,
                prior,
                &quotient_models,
                &quotient_action,
            )
            .expect("quotient SPM model EIG");

            assert!(
                (labelled.model_information_gain_nats - quotient.model_information_gain_nats).abs()
                    < 1e-10
            );
            assert_eq!(labelled.outcomes.len(), quotient.outcomes.len());
            for (left, right) in labelled.outcomes.iter().zip(quotient.outcomes.iter()) {
                assert_eq!(left.outcome_id, right.outcome_id);
                assert!((left.predictive_probability - right.predictive_probability).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn spm_known_state_feedback_decision_uses_the_source_prior_and_bmr_path() {
        let space = build_three_ball_rule_space();
        let models = space.reduced_model_hypotheses();
        let posterior = space.spm_model_average_prior.clone();
        let actions = [
            ThreeBallFeedbackActionSpec {
                action_id: "green".into(),
                choice: Some(ThreeBallColour::Green),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "red".into(),
                choice: Some(ThreeBallColour::Red),
                action_cost_nats: 0.0,
            },
            ThreeBallFeedbackActionSpec {
                action_id: "blue".into(),
                choice: Some(ThreeBallColour::Blue),
                action_cost_nats: 0.0,
            },
        ];
        let decision = score_three_ball_known_state_feedback_actions_spm_compatible(
            &space,
            &posterior,
            &models,
            [
                ThreeBallColour::Red,
                ThreeBallColour::Green,
                ThreeBallColour::Blue,
            ],
            Some(ThreeBallLocation::Centre),
            &actions,
            ThreeBallFeedbackDecisionConfig::default(),
        )
        .expect("source-compatible one-step feedback decision");
        assert_eq!(decision.action_scores.len(), 3);
        assert!(decision.action_scores.iter().all(|score| {
            score.expected_free_energy_nats.is_finite() && score.model_information_gain_nats >= 0.0
        }));
    }
}
