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
    DirichletConcentrationTensor, ReducedDirichletModelHypothesis,
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
    /// Arithmetic mean of all 81 source model priors, including duplicates.
    pub full_model_average_prior: DirichletConcentrationTensor,
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

    ThreeBallRuleSpace {
        format_version: 1,
        source_demo_commit: SPM_DEM_REASONING_COMMIT.to_owned(),
        source_rule_generator_commit: SPM_MDP_RULES_COMMIT.to_owned(),
        parameterization_digest: THREE_BALL_PARAMETERIZATION_DIGEST.to_owned(),
        rules,
        full_model_average_prior,
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
    pub fn semantic_unique_model_hypotheses(
        &self,
    ) -> Vec<ReducedDirichletModelHypothesis> {
        let mut classes = BTreeMap::<usize, ReducedDirichletModelHypothesis>::new();
        for rule in &self.rules {
            let representative_index = rule
                .duplicate_of_one_based_index
                .unwrap_or(rule.one_based_index);
            let representative_id = format!("three-ball-semantic-rule-{representative_index:02}");
            let entry = classes
                .entry(representative_index)
                .or_insert_with(|| ReducedDirichletModelHypothesis {
                    model_id: representative_id,
                    prior_probability: 0.0,
                    parameter_prior: rule.parameter_prior.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dirichlet_model_reduction::{
        evaluate_dirichlet_model_information_gain, reduce_dirichlet_models,
        DirichletActionPrediction, DirichletPredictedOutcome,
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
    fn semantic_quotient_preserves_the_model_average_prior() {
        let space = build_three_ball_rule_space();
        let compressed = space.semantic_unique_model_hypotheses();
        assert_eq!(compressed.len(), THREE_BALL_PUBLIC_GENERATOR_UNIQUE_TENSOR_COUNT);
        assert!((compressed.iter().map(|model| model.prior_probability).sum::<f64>() - 1.0).abs() < 1e-12);

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
        ball_colours: [ThreeBallColour; 3],
        gaze: Option<ThreeBallLocation>,
        choice: Option<ThreeBallColour>,
    ) -> DirichletActionPrediction {
        // Column order is choice (outer), gaze, right, centre, left (inner).
        let choice_index = choice.map(|value| value as usize).unwrap_or(3);
        let gaze_index = gaze.map(|value| value as usize).unwrap_or(3);
        let [left, centre, right] = ball_colours.map(|value| value as usize);
        let column_index =
            choice_index * 4 * 27 + gaze_index * 27 + right * 9 + centre * 3 + left * 1;

        let predictive = &space.full_model_average_prior.columns[column_index];
        let outcomes = predictive
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, probability)| *probability > 0.0)
            .map(|(outcome_index, probability)| {
                let mut posterior_count_increment =
                    vec![vec![0.0; THREE_BALL_FEEDBACK_OUTCOME_COUNT];
                         THREE_BALL_HIDDEN_STATE_COLUMN_COUNT];
                posterior_count_increment[column_index][outcome_index] = 1.0;
                DirichletPredictedOutcome {
                    outcome_id: format!("{:?}", match outcome_index {
                        0 => ThreeBallFeedback::NoFeedback,
                        1 => ThreeBallFeedback::Correct,
                        2 => ThreeBallFeedback::Incorrect,
                        _ => unreachable!("exactly three feedback outcomes"),
                    }),
                    full_model_predictive_probability: probability,
                    posterior_count_increment,
                }
            })
            .collect();

        DirichletActionPrediction {
            action_id: format!("gaze={gaze:?};choice={choice:?}"),
            outcomes,
        }
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
            let action = feedback_action_prediction(
                &space,
                ball_colours,
                Some(ThreeBallLocation::Centre),
                choice,
            );
            let source_result = evaluate_dirichlet_model_information_gain(
                full_prior,
                full_prior,
                &labelled_models,
                &action,
            )
            .expect("source-labelled action EIG");
            let quotient_result = evaluate_dirichlet_model_information_gain(
                full_prior,
                full_prior,
                &quotient_models,
                &action,
            )
            .expect("semantic quotient action EIG");

            assert!(
                (source_result.predictive_probability_mass
                    - quotient_result.predictive_probability_mass)
                    .abs()
                    < 1e-12
            );
            assert_eq!(
                source_result.outcomes.len(),
                quotient_result.outcomes.len()
            );
            assert!(
                (source_result.model_information_gain_nats
                    - quotient_result.model_information_gain_nats)
                    .abs()
                    < 1e-10,
                "model EIG changed for action {}",
                action.action_id
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

        assert!(labelled_eig
            .iter()
            .zip(quotient_eig.iter())
            .all(|(source, quotient)| (source - quotient).abs() < 1e-10));
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
}
