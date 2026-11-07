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

use crate::dirichlet_model_reduction::{
    DirichletConcentrationTensor, ReducedDirichletModelHypothesis,
};

pub const THREE_BALL_CONTEXT_RULE_HYPOTHESIS_COUNT: usize = 81;
pub const THREE_BALL_PAPER_REPORTED_UNIQUE_RULE_COUNT: usize = 79;
pub const THREE_BALL_PUBLIC_GENERATOR_UNIQUE_TENSOR_COUNT: usize = 75;
pub const THREE_BALL_PLANTED_RULE_ONE_BASED_INDEX: usize = 65;
pub const THREE_BALL_HIDDEN_STATE_COLUMN_COUNT: usize = 432;
pub const THREE_BALL_FEEDBACK_OUTCOME_COUNT: usize = 3;

/// SHA-256 of the ordered feedback-likelihood tensor parameterization:
/// outcomes [no-feedback, correct, incorrect], state factors
/// [left, centre, right, gaze, choice], with left varying fastest.
pub const THREE_BALL_PARAMETERIZATION_DIGEST: &str =
    "29cd312ceb72d317c95a873f9a5b24caeb2d9aabaf8f1aa354d36bad3129724e";

pub const SPM_DEM_REASONING_COMMIT: &str =
    "28d1f103b021f729266cb7ea07baaf4d0285b22a";
pub const SPM_MDP_RULES_COMMIT: &str =
    "5852cadb300a81fd3f30e7056ee329bca20f0efe";

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
                    let parameter_prior = build_rule_prior(
                        context_factor,
                        criterion_by_context_colour,
                    );
                    let existing = first_seen.iter().find(|(columns, support, _)| {
                        *columns == parameter_prior.columns
                            && *support == parameter_prior.active_support
                    });
                    let source_index = rules.len() + 1;
                    let duplicate_of_one_based_index = existing.map(|(_, _, first_index)| *first_index);

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
                        prior_probability: 1.0
                            / THREE_BALL_CONTEXT_RULE_HYPOTHESIS_COUNT as f64,
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
    use crate::dirichlet_model_reduction::reduce_dirichlet_models;

    #[test]
    fn reproduces_81_source_hypotheses_and_rule_65() {
        let space = build_three_ball_rule_space();
        assert_eq!(space.rules.len(), 81);
        assert_eq!(space.planted_rule().one_based_index, 65);
        assert_eq!(space.planted_rule().context_factor, ThreeBallLocation::Centre);
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
                [ThreeBallColour::Red, ThreeBallColour::Red, ThreeBallColour::Blue],
                Some(ThreeBallColour::Red)
            ),
            ThreeBallFeedback::Correct
        );
        assert_eq!(
            rule.feedback(
                [ThreeBallColour::Blue, ThreeBallColour::Green, ThreeBallColour::Red],
                Some(ThreeBallColour::Green)
            ),
            ThreeBallFeedback::Correct
        );
        assert_eq!(
            rule.feedback(
                [ThreeBallColour::Red, ThreeBallColour::Blue, ThreeBallColour::Green],
                Some(ThreeBallColour::Green)
            ),
            ThreeBallFeedback::Incorrect
        );
        assert_eq!(
            rule.feedback(
                [ThreeBallColour::Red, ThreeBallColour::Green, ThreeBallColour::Blue],
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
        assert!(space
            .semantic_duplicate_groups
            .iter()
            .all(|group| group.len() == 3));
        assert_eq!(
            space.paper_reported_unique_rule_count,
            THREE_BALL_PAPER_REPORTED_UNIQUE_RULE_COUNT
        );
        assert!(!space.unique_count_matches_paper());
    }

    #[test]
    fn model_average_prior_is_the_arithmetic_mean_of_all_81_reduced_priors() {
        let space = build_three_ball_rule_space();
        assert_eq!(space.full_model_average_prior.columns.len(), 432);
        assert_eq!(space.full_model_average_prior.active_support.len(), 432);
        for column in 0..432 {
            for outcome in 0..3 {
                let expected = space
                    .rules
                    .iter()
                    .map(|rule| {
                        rule.parameter_prior.columns[column][outcome] * rule.prior_probability
                    })
                    .sum::<f64>();
                assert!(
                    (space.full_model_average_prior.columns[column][outcome] - expected).abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn no_evidence_gives_equal_posterior_weight_to_all_81_labelled_hypotheses() {
        let space = build_three_ball_rule_space();
        let hypotheses = space.reduced_model_hypotheses();
        let posterior = space.full_model_average_prior.clone();
        let result = reduce_dirichlet_models(
            &posterior,
            &space.full_model_average_prior,
            &hypotheses,
        )
        .expect("initial BMR");
        assert_eq!(result.models.len(), 81);
        for model in result.models {
            assert!((model.posterior_probability - 1.0 / 81.0).abs() < 1e-9);
            assert!(model.log_evidence_ratio.abs() < 1e-8);
        }
    }
}
