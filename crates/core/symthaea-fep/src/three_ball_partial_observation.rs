// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-hypothesis active perception for the partially observed three-ball task.
//!
//! This module adds the missing gaze/state-belief layer to the known-state,
//! feedback-only evaluator in `three_ball_rule_space`. It maintains a joint
//! posterior over a discrete rule identity and the three hidden ball colours.
//! Gaze reveals one ball colour; a choice produces correct/incorrect/no-feedback
//! feedback according to the selected rule.
//!
//! The policy is explicitly a two-step receding-horizon approximation:
//!
//!   gaze -> observe one colour -> choose a colour (or no response).
//!
//! It scores the gaze step by its expected downstream choice free energy plus
//! model/state information value and action cost. It is not the full SPM POMDP:
//! continuous Dirichlet parameter uncertainty is not represented in this
//! finite-hypothesis belief, and no multi-trial policy search is performed.
//! The separate Dirichlet BMR module remains the parameter/model-evidence path.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::three_ball_rule_space::{
    THREE_BALL_FEEDBACK_LOG_PREFERENCES_NATS, ThreeBallColour, ThreeBallFeedback,
    ThreeBallLocation, ThreeBallRuleSpace,
};

pub const THREE_BALL_HIDDEN_STATE_COUNT: usize = 27;
pub const THREE_BALL_VISUAL_OUTCOME_COUNT: usize = 3;

/// One fully specified hidden state, in left/centre/right order.
pub type ThreeBallHiddenState = [ThreeBallColour; 3];

/// Enumerate all 27 hidden ball-colour arrangements with left varying fastest.
pub fn enumerate_three_ball_hidden_states() -> Vec<ThreeBallHiddenState> {
    let mut states = Vec::with_capacity(THREE_BALL_HIDDEN_STATE_COUNT);
    for right in 0..3 {
        for centre in 0..3 {
            for left in 0..3 {
                states.push([
                    colour_from_index(left),
                    colour_from_index(centre),
                    colour_from_index(right),
                ]);
            }
        }
    }
    states
}

/// Uniform prior over the 27 hidden ball-colour arrangements.
pub fn uniform_three_ball_state_prior() -> Vec<f64> {
    vec![1.0 / THREE_BALL_HIDDEN_STATE_COUNT as f64; THREE_BALL_HIDDEN_STATE_COUNT]
}

/// Joint posterior over a selected rule family and hidden ball-colour states.
///
/// `joint_probabilities[model][state]` is the posterior mass of that
/// model/state pair. Keeping the joint distribution (rather than separate
/// model and state marginals) preserves correlations induced by past feedback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallJointBelief {
    pub format_version: u32,
    /// External model IDs. They need not equal source IDs when using the
    /// semantic quotient; each entry maps to one rule index below.
    pub model_ids: Vec<String>,
    /// Index into `ThreeBallRuleSpace::rules` for each belief model.
    pub rule_indices: Vec<usize>,
    /// Deterministic 27-state order from `enumerate_three_ball_hidden_states`.
    pub hidden_states: Vec<ThreeBallHiddenState>,
    /// Joint posterior matrix, one row per model and one column per hidden state.
    pub joint_probabilities: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallInformationGain {
    pub outcome_probabilities: Vec<f64>,
    /// I(Model; observation).
    pub model_information_gain_nats: f64,
    /// I(HiddenState; observation | Model).
    pub conditional_state_information_gain_nats: f64,
    /// I((Model, HiddenState); observation).
    pub joint_information_gain_nats: f64,
    pub chain_rule_error_nats: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallPartialObservationConfig {
    pub feedback_log_preferences_nats: [f64; 3],
    pub model_information_weight: f64,
    pub conditional_state_information_weight: f64,
    pub gaze_action_cost_nats: [f64; 3],
    pub choice_action_cost_nats: f64,
}

impl Default for ThreeBallPartialObservationConfig {
    fn default() -> Self {
        Self {
            feedback_log_preferences_nats: THREE_BALL_FEEDBACK_LOG_PREFERENCES_NATS,
            model_information_weight: 1.0,
            conditional_state_information_weight: 1.0,
            gaze_action_cost_nats: [0.0; 3],
            choice_action_cost_nats: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallChoiceActionScore {
    pub choice: Option<ThreeBallColour>,
    pub predictive_feedback: Vec<(ThreeBallFeedback, f64)>,
    pub pragmatic_risk_nats: f64,
    pub model_information_gain_nats: f64,
    pub conditional_state_information_gain_nats: f64,
    pub epistemic_value_nats: f64,
    pub action_cost_nats: f64,
    /// Lower is better: pragmatic risk - epistemic value + cost.
    pub expected_free_energy_nats: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallGazeContingency {
    pub observed_colour: ThreeBallColour,
    pub observation_probability: f64,
    pub selected_choice: Option<ThreeBallColour>,
    pub downstream_choice_expected_free_energy_nats: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallGazeActionScore {
    /// None means do not take an additional gaze action; choose immediately.
    pub location: Option<ThreeBallLocation>,
    pub observation_probabilities: Vec<(ThreeBallColour, f64)>,
    pub model_information_gain_nats: f64,
    pub conditional_state_information_gain_nats: f64,
    pub epistemic_value_nats: f64,
    pub expected_downstream_choice_free_energy_nats: f64,
    pub action_cost_nats: f64,
    /// Lower is better. For gaze actions this includes the expected best
    /// feedback-choice score after each possible visual observation.
    pub expected_free_energy_nats: f64,
    pub contingencies: Vec<ThreeBallGazeContingency>,
    /// Choice selected immediately when `location == None`.
    pub immediate_choice: Option<ThreeBallColour>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeBallPartialObservationDecision {
    pub selected_action_index: usize,
    pub selected_gaze: Option<ThreeBallLocation>,
    pub selected_contingencies: Vec<ThreeBallGazeContingency>,
    /// Selected immediately when `selected_gaze == None`.
    pub selected_immediate_choice: Option<ThreeBallColour>,
    pub action_scores: Vec<ThreeBallGazeActionScore>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ThreeBallPartialObservationError {
    #[error("the selected model family must not be empty")]
    EmptyModelFamily,
    #[error("model ID at index {index} is empty")]
    EmptyModelId { index: usize },
    #[error("model ID is duplicated: {0}")]
    DuplicateModelId(String),
    #[error("rule index {rule_index} is outside 0..{rule_count}")]
    InvalidRuleIndex {
        rule_index: usize,
        rule_count: usize,
    },
    #[error("model posterior length {actual} does not match model family length {expected}")]
    ModelPosteriorLengthMismatch { expected: usize, actual: usize },
    #[error("model rule-index length {actual} does not match model family length {expected}")]
    ModelRuleIndexLengthMismatch { expected: usize, actual: usize },
    #[error("state prior must contain exactly 27 entries, got {0}")]
    InvalidStatePriorLength(usize),
    #[error("prior probability at {dimension} index {index} is invalid: {value}")]
    InvalidPriorProbability {
        dimension: &'static str,
        index: usize,
        value: f64,
    },
    #[error("{dimension} prior mass sums to {sum}, expected 1")]
    PriorMassNotNormalized { dimension: &'static str, sum: f64 },
    #[error("joint belief has an invalid shape or probability")]
    InvalidJointBelief,
    #[error("conditioning observation has zero or invalid probability")]
    ImpossibleObservation,
    #[error("decision configuration contains a non-finite or negative weight/cost")]
    InvalidConfiguration,
    #[error("feedback preferences must be finite")]
    InvalidPreferences,
    #[error("likelihood tensor dimensions or probability masses are invalid")]
    InvalidLikelihoods,
    #[error("no candidate action could be scored")]
    EmptyActionSet,
    #[error("model ID {0} is not present in the supplied rule space")]
    UnknownModelId(String),
    #[error("information calculation violated the mutual-information chain rule")]
    InformationChainRuleViolation,
}

impl ThreeBallJointBelief {
    /// Build the source-faithful belief over all 81 labelled source hypotheses.
    pub fn from_source_rule_prior(
        space: &ThreeBallRuleSpace,
        state_prior: &[f64],
    ) -> Result<Self, ThreeBallPartialObservationError> {
        let model_ids = space
            .rules
            .iter()
            .map(|rule| rule.model_id.clone())
            .collect();
        let rule_indices = (0..space.rules.len()).collect::<Vec<_>>();
        let model_prior = space
            .rules
            .iter()
            .map(|rule| rule.prior_probability)
            .collect::<Vec<_>>();
        Self::from_model_and_state_priors(space, model_ids, rule_indices, &model_prior, state_prior)
    }

    /// Build a belief over the quotient family of semantically unique rules.
    ///
    /// Duplicate source hypotheses are collapsed to their first representative
    /// and their model prior mass is summed, preserving observable predictions.
    pub fn from_semantic_quotient_prior(
        space: &ThreeBallRuleSpace,
        state_prior: &[f64],
    ) -> Result<Self, ThreeBallPartialObservationError> {
        let mut model_ids = Vec::new();
        let mut rule_indices = Vec::new();
        let mut model_prior = Vec::new();
        for rule in &space.rules {
            if rule.duplicate_of_one_based_index.is_some() {
                continue;
            }
            let representative = rule.one_based_index;
            let prior_mass = space
                .rules
                .iter()
                .filter(|candidate| {
                    candidate.one_based_index == representative
                        || candidate.duplicate_of_one_based_index == Some(representative)
                })
                .map(|candidate| candidate.prior_probability)
                .sum::<f64>();
            model_ids.push(format!("three-ball-semantic-rule-{representative:02}"));
            rule_indices.push(representative - 1);
            model_prior.push(prior_mass);
        }
        Self::from_model_and_state_priors(space, model_ids, rule_indices, &model_prior, state_prior)
    }

    /// Build a joint model/state prior with caller-supplied model probabilities.
    ///
    /// The initial joint distribution is factorized as P(model)P(state).
    /// Subsequent gaze and feedback observations update the full joint matrix.
    pub fn from_model_and_state_priors(
        space: &ThreeBallRuleSpace,
        model_ids: Vec<String>,
        rule_indices: Vec<usize>,
        model_prior: &[f64],
        state_prior: &[f64],
    ) -> Result<Self, ThreeBallPartialObservationError> {
        if model_ids.is_empty() {
            return Err(ThreeBallPartialObservationError::EmptyModelFamily);
        }
        if model_ids.len() != rule_indices.len() {
            return Err(
                ThreeBallPartialObservationError::ModelRuleIndexLengthMismatch {
                    expected: model_ids.len(),
                    actual: rule_indices.len(),
                },
            );
        }
        if model_ids.len() != model_prior.len() {
            return Err(
                ThreeBallPartialObservationError::ModelPosteriorLengthMismatch {
                    expected: model_ids.len(),
                    actual: model_prior.len(),
                },
            );
        }
        if state_prior.len() != THREE_BALL_HIDDEN_STATE_COUNT {
            return Err(ThreeBallPartialObservationError::InvalidStatePriorLength(
                state_prior.len(),
            ));
        }

        let mut seen = BTreeSet::new();
        for (index, (model_id, rule_index)) in model_ids.iter().zip(rule_indices.iter()).enumerate()
        {
            if model_id.trim().is_empty() {
                return Err(ThreeBallPartialObservationError::EmptyModelId { index });
            }
            if !seen.insert(model_id.clone()) {
                return Err(ThreeBallPartialObservationError::DuplicateModelId(
                    model_id.clone(),
                ));
            }
            if *rule_index >= space.rules.len() {
                return Err(ThreeBallPartialObservationError::InvalidRuleIndex {
                    rule_index: *rule_index,
                    rule_count: space.rules.len(),
                });
            }
        }

        validate_probability_vector("model", model_prior)?;
        validate_probability_vector("state", state_prior)?;
        let joint_probabilities = model_prior
            .iter()
            .map(|model_probability| {
                state_prior
                    .iter()
                    .map(|state_probability| model_probability * state_probability)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let belief = Self {
            format_version: 1,
            model_ids,
            rule_indices,
            hidden_states: enumerate_three_ball_hidden_states(),
            joint_probabilities,
        };
        belief.validate()?;
        Ok(belief)
    }

    /// Construct the uniform source-hypothesis / uniform hidden-state prior.
    pub fn uniform_source_prior(
        space: &ThreeBallRuleSpace,
    ) -> Result<Self, ThreeBallPartialObservationError> {
        Self::from_source_rule_prior(space, &uniform_three_ball_state_prior())
    }

    /// Validate dimensions, normalization, and source-family bindings.
    pub fn validate(&self) -> Result<(), ThreeBallPartialObservationError> {
        let mut seen_model_ids = BTreeSet::new();
        if self
            .model_ids
            .iter()
            .any(|id| id.trim().is_empty() || !seen_model_ids.insert(id))
        {
            return Err(ThreeBallPartialObservationError::InvalidJointBelief);
        }
        if self.format_version != 1
            || self.model_ids.is_empty()
            || self.model_ids.len() != self.rule_indices.len()
            || self.hidden_states != enumerate_three_ball_hidden_states()
            || self.joint_probabilities.len() != self.model_ids.len()
            || self
                .joint_probabilities
                .iter()
                .any(|row| row.len() != THREE_BALL_HIDDEN_STATE_COUNT)
        {
            return Err(ThreeBallPartialObservationError::InvalidJointBelief);
        }
        let mass = self.joint_probabilities.iter().flatten().sum::<f64>();
        if !mass.is_finite()
            || self
                .joint_probabilities
                .iter()
                .flatten()
                .any(|value| !value.is_finite() || *value < 0.0)
            || (mass - 1.0).abs() > 1e-9
        {
            return Err(ThreeBallPartialObservationError::InvalidJointBelief);
        }
        Ok(())
    }

    /// Posterior after seeing one ball's colour. The observation is deterministic
    /// given the hidden state, but may reveal model information through existing
    /// model/state posterior correlation.
    pub fn condition_on_gaze(
        &self,
        location: ThreeBallLocation,
        observed_colour: ThreeBallColour,
    ) -> Result<Self, ThreeBallPartialObservationError> {
        self.condition_on_likelihood(|state| {
            if state[location as usize] == observed_colour {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Posterior after observing feedback for a selected colour (or no response).
    pub fn condition_on_feedback(
        &self,
        space: &ThreeBallRuleSpace,
        choice: Option<ThreeBallColour>,
        observed_feedback: ThreeBallFeedback,
    ) -> Result<Self, ThreeBallPartialObservationError> {
        self.validate()?;
        let mut next = self.clone();
        let mut mass = 0.0;
        for model_index in 0..self.model_ids.len() {
            let rule_index = self.rule_indices[model_index];
            let rule = space.rules.get(rule_index).ok_or(
                ThreeBallPartialObservationError::InvalidRuleIndex {
                    rule_index,
                    rule_count: space.rules.len(),
                },
            )?;
            for (state_index, state) in self.hidden_states.iter().enumerate() {
                let likelihood = if rule.feedback(*state, choice) == observed_feedback {
                    1.0
                } else {
                    0.0
                };
                next.joint_probabilities[model_index][state_index] *= likelihood;
                mass += next.joint_probabilities[model_index][state_index];
            }
        }
        normalize_joint(&mut next.joint_probabilities, mass)?;
        Ok(next)
    }

    fn condition_on_likelihood(
        &self,
        likelihood: impl Fn(&ThreeBallHiddenState) -> f64,
    ) -> Result<Self, ThreeBallPartialObservationError> {
        self.validate()?;
        let mut next = self.clone();
        let mut mass = 0.0;
        for model_index in 0..self.model_ids.len() {
            for (state_index, state) in self.hidden_states.iter().enumerate() {
                let probability = likelihood(state);
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(ThreeBallPartialObservationError::InvalidLikelihoods);
                }
                next.joint_probabilities[model_index][state_index] *= probability;
                mass += next.joint_probabilities[model_index][state_index];
            }
        }
        normalize_joint(&mut next.joint_probabilities, mass)?;
        Ok(next)
    }

    pub fn model_marginals(&self) -> Vec<f64> {
        self.joint_probabilities
            .iter()
            .map(|row| row.iter().sum())
            .collect()
    }

    pub fn state_marginals(&self) -> Vec<f64> {
        (0..THREE_BALL_HIDDEN_STATE_COUNT)
            .map(|state| self.joint_probabilities.iter().map(|row| row[state]).sum())
            .collect()
    }
}

/// Score all four legal response actions: three colours and no response.
pub fn score_three_ball_choice_actions(
    space: &ThreeBallRuleSpace,
    belief: &ThreeBallJointBelief,
    config: ThreeBallPartialObservationConfig,
) -> Result<Vec<ThreeBallChoiceActionScore>, ThreeBallPartialObservationError> {
    belief.validate()?;
    validate_config(config)?;

    let preferences = normalized_log_preferences(config.feedback_log_preferences_nats);
    let choices = [
        Some(ThreeBallColour::Red),
        Some(ThreeBallColour::Green),
        Some(ThreeBallColour::Blue),
        None,
    ];
    choices
        .iter()
        .copied()
        .map(|choice| score_choice(space, belief, choice, &preferences, config))
        .collect()
}

/// Score gaze-then-choice policies against an immediate-choice baseline.
///
/// For each gaze location and each possible visual outcome, the joint model/state
/// belief is updated and the best feedback action is selected. The gaze score is
/// its action cost plus expected downstream choice free energy minus the
/// information value of the visual observation. The None action chooses now.
pub fn score_three_ball_gaze_policies(
    space: &ThreeBallRuleSpace,
    belief: &ThreeBallJointBelief,
    config: ThreeBallPartialObservationConfig,
) -> Result<ThreeBallPartialObservationDecision, ThreeBallPartialObservationError> {
    belief.validate()?;
    validate_config(config)?;
    let current_choices = score_three_ball_choice_actions(space, belief, config)?;
    let (current_choice, current_score) = best_choice(&current_choices)?;
    let mut action_scores = vec![ThreeBallGazeActionScore {
        location: None,
        observation_probabilities: Vec::new(),
        model_information_gain_nats: 0.0,
        conditional_state_information_gain_nats: 0.0,
        epistemic_value_nats: 0.0,
        expected_downstream_choice_free_energy_nats: current_score.expected_free_energy_nats,
        action_cost_nats: 0.0,
        expected_free_energy_nats: current_score.expected_free_energy_nats,
        contingencies: Vec::new(),
        immediate_choice: current_choice,
    }];

    for location in [
        ThreeBallLocation::Left,
        ThreeBallLocation::Centre,
        ThreeBallLocation::Right,
    ] {
        let likelihoods = gaze_likelihoods(belief, location);
        let information =
            evaluate_information_gain(belief, &likelihoods, THREE_BALL_VISUAL_OUTCOME_COUNT)?;
        let mut contingencies = Vec::new();
        let mut observation_probabilities = Vec::new();
        let mut expected_downstream = 0.0;

        for (colour_index, observation_probability) in information
            .outcome_probabilities
            .iter()
            .copied()
            .enumerate()
        {
            let colour = colour_from_index(colour_index);
            observation_probabilities.push((colour, observation_probability));
            if observation_probability <= 0.0 {
                continue;
            }
            let posterior = belief.condition_on_gaze(location, colour)?;
            let choices = score_three_ball_choice_actions(space, &posterior, config)?;
            let (selected_choice, selected_score) = best_choice(&choices)?;
            expected_downstream +=
                observation_probability * selected_score.expected_free_energy_nats;
            contingencies.push(ThreeBallGazeContingency {
                observed_colour: colour,
                observation_probability,
                selected_choice,
                downstream_choice_expected_free_energy_nats: selected_score
                    .expected_free_energy_nats,
            });
        }

        let action_cost = config.gaze_action_cost_nats[location as usize];
        let epistemic_value = config.model_information_weight
            * information.model_information_gain_nats
            + config.conditional_state_information_weight
                * information.conditional_state_information_gain_nats;
        let expected_free_energy = expected_downstream + action_cost - epistemic_value;

        action_scores.push(ThreeBallGazeActionScore {
            location: Some(location),
            observation_probabilities,
            model_information_gain_nats: information.model_information_gain_nats,
            conditional_state_information_gain_nats: information
                .conditional_state_information_gain_nats,
            epistemic_value_nats: epistemic_value,
            expected_downstream_choice_free_energy_nats: expected_downstream,
            action_cost_nats: action_cost,
            expected_free_energy_nats: expected_free_energy,
            contingencies,
            immediate_choice: None,
        });
    }

    let selected_action_index = (0..action_scores.len())
        .min_by(|left, right| {
            action_scores[*left]
                .expected_free_energy_nats
                .total_cmp(&action_scores[*right].expected_free_energy_nats)
                .then_with(|| left.cmp(right))
        })
        .ok_or(ThreeBallPartialObservationError::EmptyActionSet)?;
    let selected = &action_scores[selected_action_index];

    // Preserve the immediate-choice baseline as the None action's contingency;
    // gaze actions return their observation-conditioned choice map.
    let selected_contingencies = selected.contingencies.clone();

    Ok(ThreeBallPartialObservationDecision {
        selected_action_index,
        selected_gaze: selected.location,
        selected_contingencies,
        selected_immediate_choice: selected.immediate_choice,
        action_scores,
    })
}

fn score_choice(
    space: &ThreeBallRuleSpace,
    belief: &ThreeBallJointBelief,
    choice: Option<ThreeBallColour>,
    preferences: &[f64; 3],
    config: ThreeBallPartialObservationConfig,
) -> Result<ThreeBallChoiceActionScore, ThreeBallPartialObservationError> {
    let likelihoods = feedback_likelihoods(space, belief, choice)?;
    let information = evaluate_information_gain(belief, &likelihoods, 3)?;
    // Preferences are relative log probabilities (the source task supplies
    // [0, 2, -6]), not necessarily normalized log probabilities. Convert them
    // to log C(o) before computing D_KL[Q(o) || C(o)]. Omitting this
    // log-normalizer adds a negative constant and can make a valid KL appear
    // negative; clipping that value changes action rankings.
    let preference_max = preferences
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let log_preference_normalizer = preference_max
        + preferences
            .iter()
            .map(|value| (value - preference_max).exp())
            .sum::<f64>()
            .ln();
    if !log_preference_normalizer.is_finite() {
        return Err(ThreeBallPartialObservationError::InvalidPreferences);
    }
    let mut pragmatic_risk = 0.0;
    let feedback = [
        ThreeBallFeedback::NoFeedback,
        ThreeBallFeedback::Correct,
        ThreeBallFeedback::Incorrect,
    ];
    for (index, probability) in information
        .outcome_probabilities
        .iter()
        .copied()
        .enumerate()
    {
        if probability > 0.0 {
            let log_preference = preferences[index] - log_preference_normalizer;
            pragmatic_risk += probability * (probability.ln() - log_preference);
        }
    }
    if !pragmatic_risk.is_finite() || pragmatic_risk < -1e-9 {
        return Err(ThreeBallPartialObservationError::InvalidConfiguration);
    }
    // The exact KL is non-negative; tolerate and clamp only round-off scale
    // negatives, never an unnormalized-preference offset.
    let pragmatic_risk = pragmatic_risk.max(0.0);
    let epistemic_value = config.model_information_weight * information.model_information_gain_nats
        + config.conditional_state_information_weight
            * information.conditional_state_information_gain_nats;
    let action_cost = config.choice_action_cost_nats;
    let expected_free_energy = pragmatic_risk - epistemic_value + action_cost;

    Ok(ThreeBallChoiceActionScore {
        choice,
        predictive_feedback: feedback
            .iter()
            .copied()
            .zip(information.outcome_probabilities.iter().copied())
            .collect(),
        pragmatic_risk_nats: pragmatic_risk,
        model_information_gain_nats: information.model_information_gain_nats,
        conditional_state_information_gain_nats: information
            .conditional_state_information_gain_nats,
        epistemic_value_nats: epistemic_value,
        action_cost_nats: action_cost,
        expected_free_energy_nats: expected_free_energy,
    })
}

fn gaze_likelihoods(
    belief: &ThreeBallJointBelief,
    location: ThreeBallLocation,
) -> Vec<Vec<Vec<f64>>> {
    belief
        .rule_indices
        .iter()
        .map(|_| {
            belief
                .hidden_states
                .iter()
                .map(|state| {
                    let mut outcomes = vec![0.0; THREE_BALL_VISUAL_OUTCOME_COUNT];
                    outcomes[state[location as usize] as usize] = 1.0;
                    outcomes
                })
                .collect()
        })
        .collect()
}

fn feedback_likelihoods(
    space: &ThreeBallRuleSpace,
    belief: &ThreeBallJointBelief,
    choice: Option<ThreeBallColour>,
) -> Result<Vec<Vec<Vec<f64>>>, ThreeBallPartialObservationError> {
    belief
        .rule_indices
        .iter()
        .map(|rule_index| {
            let rule = space.rules.get(*rule_index).ok_or(
                ThreeBallPartialObservationError::InvalidRuleIndex {
                    rule_index: *rule_index,
                    rule_count: space.rules.len(),
                },
            )?;
            Ok(belief
                .hidden_states
                .iter()
                .map(|state| {
                    let feedback = rule.feedback(*state, choice) as usize;
                    let mut outcomes = vec![0.0; 3];
                    outcomes[feedback] = 1.0;
                    outcomes
                })
                .collect())
        })
        .collect()
}

fn evaluate_information_gain(
    belief: &ThreeBallJointBelief,
    likelihoods: &[Vec<Vec<f64>>],
    outcome_count: usize,
) -> Result<ThreeBallInformationGain, ThreeBallPartialObservationError> {
    if likelihoods.len() != belief.joint_probabilities.len()
        || likelihoods
            .iter()
            .any(|row| row.len() != THREE_BALL_HIDDEN_STATE_COUNT)
        || likelihoods
            .iter()
            .flatten()
            .any(|outcomes| outcomes.len() != outcome_count)
    {
        return Err(ThreeBallPartialObservationError::InvalidLikelihoods);
    }
    for row in likelihoods {
        for outcomes in row {
            if outcomes
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                || (outcomes.iter().sum::<f64>() - 1.0).abs() > 1e-9
            {
                return Err(ThreeBallPartialObservationError::InvalidLikelihoods);
            }
        }
    }

    let mut outcome_probabilities = vec![0.0; outcome_count];
    for model in 0..belief.model_ids.len() {
        for state in 0..THREE_BALL_HIDDEN_STATE_COUNT {
            let prior = belief.joint_probabilities[model][state];
            for outcome in 0..outcome_count {
                outcome_probabilities[outcome] += prior * likelihoods[model][state][outcome];
            }
        }
    }

    let prior_model = belief.model_marginals();
    let prior_joint_entropy = entropy(
        &belief
            .joint_probabilities
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );
    let prior_model_entropy = entropy(&prior_model);
    let prior_state_given_model_entropy = conditional_state_entropy(&belief.joint_probabilities);

    let mut expected_model_entropy = 0.0;
    let mut expected_state_given_model_entropy = 0.0;
    let mut expected_joint_entropy = 0.0;
    for (outcome, outcome_probability) in outcome_probabilities.iter().copied().enumerate() {
        if outcome_probability <= 0.0 {
            continue;
        }
        let mut posterior = belief.joint_probabilities.clone();
        for model in 0..belief.model_ids.len() {
            for state in 0..THREE_BALL_HIDDEN_STATE_COUNT {
                posterior[model][state] *= likelihoods[model][state][outcome] / outcome_probability;
            }
        }
        let model_marginal = posterior
            .iter()
            .map(|row| row.iter().sum())
            .collect::<Vec<_>>();
        expected_model_entropy += outcome_probability * entropy(&model_marginal);
        expected_state_given_model_entropy +=
            outcome_probability * conditional_state_entropy(&posterior);
        expected_joint_entropy +=
            outcome_probability * entropy(&posterior.iter().flatten().copied().collect::<Vec<_>>());
    }

    let model_information_gain_nats = prior_model_entropy - expected_model_entropy;
    let conditional_state_information_gain_nats =
        prior_state_given_model_entropy - expected_state_given_model_entropy;
    let joint_information_gain_nats = prior_joint_entropy - expected_joint_entropy;
    let chain_rule_error_nats = (joint_information_gain_nats
        - model_information_gain_nats
        - conditional_state_information_gain_nats)
        .abs();

    if !chain_rule_error_nats.is_finite()
        || chain_rule_error_nats > 1e-8
        || model_information_gain_nats < -1e-8
        || conditional_state_information_gain_nats < -1e-8
        || joint_information_gain_nats < -1e-8
    {
        return Err(ThreeBallPartialObservationError::InformationChainRuleViolation);
    }

    Ok(ThreeBallInformationGain {
        outcome_probabilities,
        model_information_gain_nats: model_information_gain_nats.max(0.0),
        conditional_state_information_gain_nats: conditional_state_information_gain_nats.max(0.0),
        joint_information_gain_nats: joint_information_gain_nats.max(0.0),
        chain_rule_error_nats,
    })
}

fn conditional_state_entropy(joint: &[Vec<f64>]) -> f64 {
    let model_marginals = joint
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let mut conditional = 0.0;
    for (model, row) in joint.iter().enumerate() {
        let model_probability = model_marginals[model];
        if model_probability > 0.0 {
            let state_posterior = row
                .iter()
                .map(|value| value / model_probability)
                .collect::<Vec<_>>();
            conditional += model_probability * entropy(&state_posterior);
        }
    }
    conditional
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn normalized_log_preferences(preferences: [f64; 3]) -> [f64; 3] {
    let max = preferences.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let log_normalizer = max
        + preferences
            .iter()
            .map(|value| (value - max).exp())
            .sum::<f64>()
            .ln();
    preferences.map(|value| value - log_normalizer)
}

fn validate_config(
    config: ThreeBallPartialObservationConfig,
) -> Result<(), ThreeBallPartialObservationError> {
    if config
        .feedback_log_preferences_nats
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(ThreeBallPartialObservationError::InvalidPreferences);
    }
    if !config.model_information_weight.is_finite()
        || config.model_information_weight < 0.0
        || !config.conditional_state_information_weight.is_finite()
        || config.conditional_state_information_weight < 0.0
        || !config.choice_action_cost_nats.is_finite()
        || config.choice_action_cost_nats < 0.0
        || config
            .gaze_action_cost_nats
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ThreeBallPartialObservationError::InvalidConfiguration);
    }
    Ok(())
}

fn validate_probability_vector(
    dimension: &'static str,
    probabilities: &[f64],
) -> Result<(), ThreeBallPartialObservationError> {
    let mut total = 0.0;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ThreeBallPartialObservationError::InvalidPriorProbability {
                dimension,
                index,
                value: probability,
            });
        }
        total += probability;
    }
    if (total - 1.0).abs() > 1e-9 {
        return Err(ThreeBallPartialObservationError::PriorMassNotNormalized {
            dimension,
            sum: total,
        });
    }
    Ok(())
}

fn normalize_joint(
    joint: &mut [Vec<f64>],
    total_mass: f64,
) -> Result<(), ThreeBallPartialObservationError> {
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(ThreeBallPartialObservationError::ImpossibleObservation);
    }
    for value in joint.iter_mut().flatten() {
        *value /= total_mass;
    }
    Ok(())
}

fn best_choice(
    scores: &[ThreeBallChoiceActionScore],
) -> Result<(Option<ThreeBallColour>, &ThreeBallChoiceActionScore), ThreeBallPartialObservationError>
{
    scores
        .iter()
        .enumerate()
        .min_by(|left, right| {
            left.1
                .expected_free_energy_nats
                .total_cmp(&right.1.expected_free_energy_nats)
                .then_with(|| left.0.cmp(&right.0))
        })
        .map(|(_, score)| (score.choice, score))
        .ok_or(ThreeBallPartialObservationError::EmptyActionSet)
}

fn colour_from_index(index: usize) -> ThreeBallColour {
    match index {
        0 => ThreeBallColour::Red,
        1 => ThreeBallColour::Green,
        2 => ThreeBallColour::Blue,
        _ => unreachable!("three-ball colour index is in 0..3"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::three_ball_rule_space::build_three_ball_rule_space;

    #[test]
    fn uniform_gaze_identifies_one_colour_with_ln_three_state_information() {
        let space = build_three_ball_rule_space();
        let belief = ThreeBallJointBelief::uniform_source_prior(&space).expect("prior");
        let likelihoods = gaze_likelihoods(&belief, ThreeBallLocation::Centre);
        let gain =
            evaluate_information_gain(&belief, &likelihoods, THREE_BALL_VISUAL_OUTCOME_COUNT)
                .expect("gaze information");

        assert_eq!(gain.model_information_gain_nats, 0.0);
        assert!((gain.conditional_state_information_gain_nats - 3.0_f64.ln()).abs() < 1e-10);
        assert!((gain.joint_information_gain_nats - 3.0_f64.ln()).abs() < 1e-10);
        assert!(gain.chain_rule_error_nats < 1e-12);
    }

    #[test]
    fn gaze_conditioning_reduces_hidden_state_belief_without_changing_independent_model_prior() {
        let space = build_three_ball_rule_space();
        let prior = ThreeBallJointBelief::uniform_source_prior(&space).expect("prior");
        let before_models = prior.model_marginals();
        let posterior = prior
            .condition_on_gaze(ThreeBallLocation::Centre, ThreeBallColour::Green)
            .expect("visual observation");
        let after_models = posterior.model_marginals();

        assert_eq!(
            posterior
                .state_marginals()
                .iter()
                .filter(|p| **p > 0.0)
                .count(),
            9
        );
        for (before, after) in before_models.iter().zip(after_models.iter()) {
            assert!((before - after).abs() < 1e-12);
        }
        assert!((posterior.joint_probabilities.iter().flatten().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correlated_model_state_prior_lets_gaze_disambiguate_model_identity() {
        let states = enumerate_three_ball_hidden_states();
        let mut joint = vec![vec![0.0; THREE_BALL_HIDDEN_STATE_COUNT]; 2];
        let per_state_mass = 0.5 / 9.0;
        for (state_index, state) in states.iter().enumerate() {
            match state[0] {
                ThreeBallColour::Red => joint[0][state_index] = per_state_mass,
                ThreeBallColour::Blue => joint[1][state_index] = per_state_mass,
                ThreeBallColour::Green => {}
            }
        }
        let belief = ThreeBallJointBelief {
            format_version: 1,
            model_ids: vec!["model-red-context".into(), "model-blue-context".into()],
            rule_indices: vec![0, 1],
            hidden_states: states,
            joint_probabilities: joint,
        };
        belief.validate().expect("correlated model/state prior");

        let likelihoods = gaze_likelihoods(&belief, ThreeBallLocation::Left);
        let information = evaluate_information_gain(&belief, &likelihoods, 3)
            .expect("model-disambiguating visual observation");

        assert!((information.model_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10);
        assert!(information.conditional_state_information_gain_nats.abs() < 1e-10);
        assert!((information.joint_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10);

        let posterior = belief
            .condition_on_gaze(ThreeBallLocation::Left, ThreeBallColour::Red)
            .expect("red visual observation");
        assert!((posterior.model_marginals()[0] - 1.0).abs() < 1e-12);
        assert!(posterior.model_marginals()[1] < 1e-12);
    }

    #[test]
    fn observed_feedback_updates_the_joint_model_state_posterior() {
        let space = build_three_ball_rule_space();
        let belief = ThreeBallJointBelief::uniform_source_prior(&space).expect("prior");
        let posterior = belief
            .condition_on_feedback(
                &space,
                Some(ThreeBallColour::Green),
                ThreeBallFeedback::Correct,
            )
            .expect("feedback observation");

        assert!((posterior.joint_probabilities.iter().flatten().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_ne!(belief.model_marginals(), posterior.model_marginals());
        assert_ne!(belief.state_marginals(), posterior.state_marginals());
    }

    #[test]
    fn gaze_then_choice_policy_reports_a_normalized_contingency_for_each_visible_colour() {
        let space = build_three_ball_rule_space();
        let belief = ThreeBallJointBelief::uniform_source_prior(&space).expect("prior");
        let decision = score_three_ball_gaze_policies(
            &space,
            &belief,
            ThreeBallPartialObservationConfig::default(),
        )
        .expect("two-step policy");

        assert!(decision.selected_action_index < decision.action_scores.len());
        assert_eq!(
            decision.selected_gaze,
            decision.action_scores[decision.selected_action_index].location
        );
        assert_eq!(
            decision.selected_immediate_choice,
            decision.action_scores[decision.selected_action_index].immediate_choice
        );
        assert_eq!(decision.action_scores.len(), 4);
        for action in &decision.action_scores {
            assert!(action.expected_free_energy_nats.is_finite());
            if action.location.is_some() {
                assert_eq!(action.observation_probabilities.len(), 3);
                assert!(
                    (action
                        .observation_probabilities
                        .iter()
                        .map(|(_, probability)| probability)
                        .sum::<f64>()
                        - 1.0)
                        .abs()
                        < 1e-10
                );
                assert_eq!(action.contingencies.len(), 3);
                for contingency in &action.contingencies {
                    assert!(
                        contingency
                            .downstream_choice_expected_free_energy_nats
                            .is_finite()
                    );
                }
            } else {
                assert!(action.contingencies.is_empty());
            }
        }
    }

    #[test]
    fn relative_log_preference_shift_does_not_change_kl_risk_or_policy_scores() {
        let space = build_three_ball_rule_space();
        let belief = ThreeBallJointBelief::uniform_source_prior(&space).expect("prior");
        let base = ThreeBallPartialObservationConfig::default();
        let shifted = ThreeBallPartialObservationConfig {
            feedback_log_preferences_nats: [
                base.feedback_log_preferences_nats[0] + 13.0,
                base.feedback_log_preferences_nats[1] + 13.0,
                base.feedback_log_preferences_nats[2] + 13.0,
            ],
            ..base
        };
        let base_scores = score_three_ball_choice_actions(&space, &belief, base)
            .expect("base relative preferences");
        let shifted_scores = score_three_ball_choice_actions(&space, &belief, shifted)
            .expect("shifted relative preferences");

        assert_eq!(base_scores.len(), shifted_scores.len());
        for (left, right) in base_scores.iter().zip(shifted_scores.iter()) {
            assert!(left.pragmatic_risk_nats >= 0.0);
            assert!((left.pragmatic_risk_nats - right.pragmatic_risk_nats).abs() < 1e-12);
            assert!(
                (left.expected_free_energy_nats - right.expected_free_energy_nats).abs() < 1e-12
            );
        }

        let base_policy =
            score_three_ball_gaze_policies(&space, &belief, base).expect("base policy");
        let shifted_policy =
            score_three_ball_gaze_policies(&space, &belief, shifted).expect("shifted policy");
        assert_eq!(
            base_policy.selected_action_index,
            shifted_policy.selected_action_index
        );
        for (left, right) in base_policy
            .action_scores
            .iter()
            .zip(shifted_policy.action_scores.iter())
        {
            assert!(
                (left.expected_free_energy_nats - right.expected_free_energy_nats).abs() < 1e-12
            );
        }
    }

    #[test]
    fn semantic_quotient_preserves_gaze_and_choice_scores() {
        let space = build_three_ball_rule_space();
        let source = ThreeBallJointBelief::uniform_source_prior(&space).expect("source prior");
        let quotient = ThreeBallJointBelief::from_semantic_quotient_prior(
            &space,
            &uniform_three_ball_state_prior(),
        )
        .expect("quotient prior");
        let source_choices = score_three_ball_choice_actions(
            &space,
            &source,
            ThreeBallPartialObservationConfig::default(),
        )
        .expect("source choices");
        let quotient_choices = score_three_ball_choice_actions(
            &space,
            &quotient,
            ThreeBallPartialObservationConfig::default(),
        )
        .expect("quotient choices");
        for (left, right) in source_choices.iter().zip(quotient_choices.iter()) {
            assert!(
                (left.expected_free_energy_nats - right.expected_free_energy_nats).abs() < 1e-10
            );
        }

        let source_policy = score_three_ball_gaze_policies(
            &space,
            &source,
            ThreeBallPartialObservationConfig::default(),
        )
        .expect("source gaze policy");
        let quotient_policy = score_three_ball_gaze_policies(
            &space,
            &quotient,
            ThreeBallPartialObservationConfig::default(),
        )
        .expect("quotient gaze policy");
        for (left, right) in source_policy
            .action_scores
            .iter()
            .zip(quotient_policy.action_scores.iter())
        {
            assert!(
                (left.expected_free_energy_nats - right.expected_free_energy_nats).abs() < 1e-10
            );
        }
    }

    #[test]
    fn impossible_observations_and_invalid_action_costs_fail_closed() {
        let space = build_three_ball_rule_space();
        let belief = ThreeBallJointBelief::uniform_source_prior(&space).expect("prior");
        let known_red = belief
            .condition_on_gaze(ThreeBallLocation::Left, ThreeBallColour::Red)
            .expect("red observation");
        assert!(matches!(
            known_red.condition_on_gaze(ThreeBallLocation::Left, ThreeBallColour::Blue),
            Err(ThreeBallPartialObservationError::ImpossibleObservation)
        ));

        let invalid = ThreeBallPartialObservationConfig {
            gaze_action_cost_nats: [0.0, f64::NAN, 0.0],
            ..Default::default()
        };
        assert!(matches!(
            score_three_ball_gaze_policies(&space, &belief, invalid),
            Err(ThreeBallPartialObservationError::InvalidConfiguration)
        ));
    }
}
