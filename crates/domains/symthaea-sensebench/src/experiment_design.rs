// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cost-aware Bayesian experiment ranking by expected information gain.
//!
//! The selector is deliberately model-agnostic: upstream science code supplies
//! explicit priors and predicted outcome likelihoods for each candidate. This
//! module does not generate or validate the scientific hypotheses themselves.
//! It ranks experiments that are expected to distinguish those hypotheses.

use serde::{Deserialize, Serialize};

const PROBABILITY_TOLERANCE: f64 = 1.0e-6;

/// A hypothesis and its current prior probability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisPrior {
    pub id: String,
    pub probability: f64,
}

/// One experiment candidate.
///
/// likelihoods[h][o] is P(outcome o | hypothesis h). Every row must form a
/// probability distribution. Rows share an outcome ordering for this candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentCandidate {
    pub id: String,
    /// Relative resource cost. Must be finite and strictly positive.
    pub cost: f64,
    pub likelihoods: Vec<Vec<f64>>,
}

/// Candidate score; expected information gain is measured in bits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentScore {
    pub candidate_id: String,
    pub expected_information_gain_bits: f64,
    pub information_gain_per_cost: f64,
}

/// Posterior belief after observing one outcome of a candidate experiment.
/// One plausible likelihood model for the same candidate experiment set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLikelihoodScenario {
    /// Stable, non-empty model/scenario identifier.
    pub scenario_id: String,
    /// Relative prior weight for expected-gain aggregation; weights sum to one.
    pub weight: f64,
    /// Candidates supported by this model. IDs, costs, and outcome-index semantics
    /// must match across scenarios.
    pub candidates: Vec<ExperimentCandidate>,
}

/// Conservative score across a declared set of plausible likelihood models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustExperimentScore {
    pub candidate_id: String,
    /// Model-weighted expected information gain in bits.
    pub expected_information_gain_bits: f64,
    /// Minimum expected information gain across the declared model scenarios.
    pub worst_case_information_gain_bits: f64,
    /// Model-weighted expected information gain per unit cost.
    pub expected_information_gain_per_cost: f64,
    /// Minimum information gain per unit cost across declared model scenarios.
    pub worst_case_information_gain_per_cost: f64,
    /// Weighted standard deviation in expected information gain across model scenarios.
    pub model_sensitivity_stddev_bits: f64,
}

/// Posterior belief after observing one outcome of a candidate experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PosteriorUpdate {
    /// Updated probabilities, in the same hypothesis order as the priors.
    pub posterior: Vec<HypothesisPrior>,
    /// Predictive probability assigned to the observed outcome before observing it.
    pub outcome_probability: f64,
    /// Prior entropy minus posterior entropy, in bits.
    ///
    /// This realized value can be negative for a surprising observation; expected
    /// information gain over all outcomes is non-negative for a valid model.
    pub realized_information_gain_bits: f64,
}

/// Inputs were invalid or internally inconsistent.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExperimentDesignError {
    #[error("at least one hypothesis is required")]
    NoHypotheses,
    #[error("hypothesis identifiers must be non-empty and unique")]
    InvalidHypothesisIds,
    #[error("hypothesis priors must be finite, non-negative, and sum to one")]
    InvalidPriors,
    #[error("experiment identifiers must be non-empty and unique")]
    InvalidCandidateIds,
    #[error("experiment cost must be finite and strictly positive")]
    InvalidCost,
    #[error("each candidate must provide one likelihood row per hypothesis and at least one outcome")]
    InvalidLikelihoodShape,
    #[error("likelihoods must be finite probabilities in [0, 1]")]
    InvalidLikelihoodProbability,
    #[error("each hypothesis likelihood row must sum to one")]
    InvalidLikelihoodRow,
    #[error("observed outcome index is outside the candidate outcome range")]
    OutcomeIndexOutOfRange,
    #[error("observed outcome has zero probability under all supplied hypotheses")]
    ImpossibleObservedOutcome,
    #[error("at least one likelihood-model scenario is required")]
    NoModelScenarios,
    #[error("model scenario identifiers must be non-empty and unique")]
    InvalidModelScenarioIds,
    #[error("model scenario weights must be finite, positive, and sum to one")]
    InvalidModelWeights,
    #[error("candidate identifiers and costs must match across likelihood-model scenarios")]
    InconsistentCandidateSet,
}

/// Rank experiments by expected entropy reduction per cost unit.
///
/// Priors must sum to one within 1e-6. Each candidate must contain one
/// likelihood row per hypothesis; every row must be a probability distribution
/// over the same candidate-specific outcomes. Ties are resolved by candidate ID
/// so that output ordering remains deterministic.
pub fn rank_experiments(
    priors: &[HypothesisPrior],
    candidates: &[ExperimentCandidate],
) -> Result<Vec<ExperimentScore>, ExperimentDesignError> {
    if priors.is_empty() {
        return Err(ExperimentDesignError::NoHypotheses);
    }

    let mut hypothesis_ids: Vec<&str> = Vec::with_capacity(priors.len());
    let mut prior_probabilities = Vec::with_capacity(priors.len());
    for prior in priors {
        if prior.id.trim().is_empty() || hypothesis_ids.contains(&prior.id.as_str()) {
            return Err(ExperimentDesignError::InvalidHypothesisIds);
        }
        hypothesis_ids.push(prior.id.as_str());
        if !prior.probability.is_finite() || !(0.0..=1.0).contains(&prior.probability) {
            return Err(ExperimentDesignError::InvalidPriors);
        }
        prior_probabilities.push(prior.probability);
    }
    let prior_sum = prior_probabilities.iter().sum::<f64>();
    if prior_sum <= 0.0 || (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ExperimentDesignError::InvalidPriors);
    }
    // Remove tolerated floating-point summation error before entropy math.
    for probability in &mut prior_probabilities {
        *probability /= prior_sum;
    }

    let prior_entropy = entropy_bits(&prior_probabilities);
    let mut candidate_ids: Vec<&str> = Vec::with_capacity(candidates.len());
    let mut scores = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        if candidate.id.trim().is_empty() || candidate_ids.contains(&candidate.id.as_str()) {
            return Err(ExperimentDesignError::InvalidCandidateIds);
        }
        candidate_ids.push(candidate.id.as_str());

        if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
            return Err(ExperimentDesignError::InvalidCost);
        }
        if candidate.likelihoods.len() != priors.len()
            || candidate.likelihoods.first().is_none_or(Vec::is_empty)
        {
            return Err(ExperimentDesignError::InvalidLikelihoodShape);
        }
        let outcome_count = candidate.likelihoods[0].len();
        for row in &candidate.likelihoods {
            if row.len() != outcome_count {
                return Err(ExperimentDesignError::InvalidLikelihoodShape);
            }
            if row.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err(ExperimentDesignError::InvalidLikelihoodProbability);
            }
            if (row.iter().sum::<f64>() - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(ExperimentDesignError::InvalidLikelihoodRow);
            }
        }

        let mut expected_posterior_entropy = 0.0;
        for outcome_index in 0..outcome_count {
            let outcome_probability = prior_probabilities.iter()
                .zip(&candidate.likelihoods)
                .map(|(prior, row)| prior * row[outcome_index])
                .sum::<f64>();

            if outcome_probability <= f64::EPSILON {
                continue;
            }

            let posterior: Vec<f64> = prior_probabilities.iter()
                .zip(&candidate.likelihoods)
                .map(|(prior, row)| prior * row[outcome_index] / outcome_probability)
                .collect();
            expected_posterior_entropy +=
                outcome_probability * entropy_bits(&posterior);
        }

        let information_gain = (prior_entropy - expected_posterior_entropy)
            .clamp(0.0, prior_entropy);
        scores.push(ExperimentScore {
            candidate_id: candidate.id.clone(),
            expected_information_gain_bits: information_gain,
            information_gain_per_cost: information_gain / candidate.cost,
        });
    }

    scores.sort_by(|left, right| {
        right.information_gain_per_cost
            .total_cmp(&left.information_gain_per_cost)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    Ok(scores)
}

/// Rank experiments conservatively across an explicitly declared set of
/// plausible likelihood models.
///
/// Each model scenario supplies an alternative outcome-likelihood table for the
/// same candidate IDs and costs. Scenario weights must sum to one within
/// 1e-6; tolerated summation error is normalized. Candidates are ranked by
/// worst-case expected information gain per cost, with weighted expected gain
/// and stable IDs as tie-breakers. The weighted gain and model-sensitivity
/// deviation are also returned to make conservatism visible.
///
/// This is robust only to the supplied scenario set, not to every possible
/// misspecification. The model scenarios themselves must be justified and
/// documented by the science workflow.
pub fn rank_experiments_robust(
    priors: &[HypothesisPrior],
    scenarios: &[ModelLikelihoodScenario],
) -> Result<Vec<RobustExperimentScore>, ExperimentDesignError> {
    if scenarios.is_empty() {
        return Err(ExperimentDesignError::NoModelScenarios);
    }

    let mut scenario_ids: Vec<&str> = Vec::with_capacity(scenarios.len());
    let mut weight_sum = 0.0;
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty()
            || scenario_ids.contains(&scenario.scenario_id.as_str())
        {
            return Err(ExperimentDesignError::InvalidModelScenarioIds);
        }
        scenario_ids.push(scenario.scenario_id.as_str());
        if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
            return Err(ExperimentDesignError::InvalidModelWeights);
        }
        if scenario.candidates.is_empty() {
            return Err(ExperimentDesignError::InconsistentCandidateSet);
        }
        weight_sum += scenario.weight;
    }
    if (weight_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ExperimentDesignError::InvalidModelWeights);
    }

    let reference_candidates = &scenarios[0].candidates;
    let reference_ranking = rank_experiments(priors, reference_candidates)?;
    let mut rankings = vec![reference_ranking.clone()];
    for scenario in scenarios.iter().skip(1) {
        let ranking = rank_experiments(priors, &scenario.candidates)?;
        if scenario.candidates.len() != reference_candidates.len()
            || reference_candidates.iter().any(|reference| {
                scenario.candidates.iter().find(|candidate| candidate.id == reference.id)
                    .is_none_or(|candidate| {
                        let reference_outcomes =
                            reference.likelihoods.first().map(Vec::len);
                        let scenario_outcomes = candidate.likelihoods.first().map(Vec::len);
                        candidate.cost != reference.cost
                            || scenario_outcomes != reference_outcomes
                    })
            })
        {
            return Err(ExperimentDesignError::InconsistentCandidateSet);
        }
        // Ensure each scenario ranking covers the same candidate ID set. Ranking
        // order is expected to differ, so compare identity rather than index.
        if ranking.len() != reference_ranking.len()
            || reference_ranking.iter().any(|reference| {
                !ranking.iter().any(|candidate| candidate.candidate_id == reference.candidate_id)
            })
        {
            return Err(ExperimentDesignError::InconsistentCandidateSet);
        }
        rankings.push(ranking);
    }

    let mut output = Vec::with_capacity(reference_candidates.len());
    for reference in reference_candidates {
        let mut expected_gain = 0.0;
        let mut worst_case_gain = f64::INFINITY;
        let mut scenario_gains = Vec::with_capacity(scenarios.len());

        for (scenario_index, scenario) in scenarios.iter().enumerate() {
            let gain = rankings[scenario_index].iter()
                .find(|score| score.candidate_id == reference.id)
                .ok_or(ExperimentDesignError::InconsistentCandidateSet)?
                .expected_information_gain_bits;
            let normalized_weight = scenario.weight / weight_sum;
            expected_gain += normalized_weight * gain;
            worst_case_gain = worst_case_gain.min(gain);
            scenario_gains.push((normalized_weight, gain));
        }

        let weighted_variance = scenario_gains.iter()
            .map(|(weight, gain)| weight * (gain - expected_gain).powi(2))
            .sum::<f64>()
            .max(0.0);
        let expected_per_cost = expected_gain / reference.cost;
        let worst_case_per_cost = worst_case_gain / reference.cost;
        output.push(RobustExperimentScore {
            candidate_id: reference.id.clone(),
            expected_information_gain_bits: expected_gain,
            worst_case_information_gain_bits: worst_case_gain,
            expected_information_gain_per_cost: expected_per_cost,
            worst_case_information_gain_per_cost: worst_case_per_cost,
            model_sensitivity_stddev_bits: weighted_variance.sqrt(),
        });
    }

    output.sort_by(|left, right| {
        right.worst_case_information_gain_per_cost
            .total_cmp(&left.worst_case_information_gain_per_cost)
            .then_with(|| {
                right.expected_information_gain_per_cost
                    .total_cmp(&left.expected_information_gain_per_cost)
            })
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    Ok(output)
}

/// Update hypothesis probabilities after observing one candidate outcome.
///
/// This applies Bayes' rule using the same likelihood contract as
/// `rank_experiments`. The candidate is fully validated before updating.
/// Outcomes with zero predictive probability fail closed instead of producing
/// NaNs or silently retaining the prior.
pub fn update_posterior(
    priors: &[HypothesisPrior],
    candidate: &ExperimentCandidate,
    observed_outcome_index: usize,
) -> Result<PosteriorUpdate, ExperimentDesignError> {
    // Reuse the ranking path so priors, identifiers, cost, and every likelihood
    // row are validated consistently. Cost affects ranking, not Bayes' rule.
    rank_experiments(priors, std::slice::from_ref(candidate))?;

    let outcome_count = candidate.likelihoods[0].len();
    if observed_outcome_index >= outcome_count {
        return Err(ExperimentDesignError::OutcomeIndexOutOfRange);
    }

    let prior_sum = priors.iter().map(|prior| prior.probability).sum::<f64>();
    let normalized_priors: Vec<f64> = priors
        .iter()
        .map(|prior| prior.probability / prior_sum)
        .collect();
    let prior_entropy = entropy_bits(&normalized_priors);

    let outcome_probability = normalized_priors
        .iter()
        .zip(&candidate.likelihoods)
        .map(|(prior, row)| prior * row[observed_outcome_index])
        .sum::<f64>();

    if !outcome_probability.is_finite() || outcome_probability <= 0.0 {
        return Err(ExperimentDesignError::ImpossibleObservedOutcome);
    }

    let posterior_probabilities: Vec<f64> = normalized_priors
        .iter()
        .zip(&candidate.likelihoods)
        .map(|(prior, row)| prior * row[observed_outcome_index] / outcome_probability)
        .collect();
    let posterior = priors
        .iter()
        .zip(&posterior_probabilities)
        .map(|(prior, &probability)| HypothesisPrior {
            id: prior.id.clone(),
            probability,
        })
        .collect();
    let realized_information_gain_bits =
        prior_entropy - entropy_bits(&posterior_probabilities);

    Ok(PosteriorUpdate {
        posterior,
        outcome_probability,
        realized_information_gain_bits,
    })
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    -probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_priors() -> Vec<HypothesisPrior> {
        vec![
            HypothesisPrior { id: "H1".into(), probability: 0.5 },
            HypothesisPrior { id: "H2".into(), probability: 0.5 },
        ]
    }

    #[test]
    fn perfectly_discriminating_experiment_yields_one_bit() {
        let candidates = vec![ExperimentCandidate {
            id: "perfect".into(),
            cost: 1.0,
            likelihoods: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }];
        let scores = rank_experiments(&binary_priors(), &candidates).unwrap();
        assert!((scores[0].expected_information_gain_bits - 1.0).abs() < 1.0e-12);
        assert!((scores[0].information_gain_per_cost - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn uninformative_experiment_yields_zero_bits() {
        let candidates = vec![ExperimentCandidate {
            id: "uninformative".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        }];
        let scores = rank_experiments(&binary_priors(), &candidates).unwrap();
        assert_eq!(scores[0].expected_information_gain_bits, 0.0);
    }

    #[test]
    fn ranks_discriminating_experiment_above_uninformative_experiment() {
        let candidates = vec![
            ExperimentCandidate {
                id: "uninformative".into(),
                cost: 1.0,
                likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            },
            ExperimentCandidate {
                id: "discriminating".into(),
                cost: 2.0,
                likelihoods: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            },
        ];
        let scores = rank_experiments(&binary_priors(), &candidates).unwrap();
        assert_eq!(scores[0].candidate_id, "discriminating");
        assert_eq!(scores[1].candidate_id, "uninformative");
    }

    #[test]
    fn cost_can_reverse_experiment_priority() {
        let candidates = vec![
            ExperimentCandidate {
                id: "high-gain-expensive".into(),
                cost: 10.0,
                likelihoods: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            },
            ExperimentCandidate {
                id: "moderate-gain-cheap".into(),
                cost: 1.0,
                likelihoods: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            },
        ];
        let scores = rank_experiments(&binary_priors(), &candidates).unwrap();
        assert_eq!(scores[0].candidate_id, "moderate-gain-cheap");
    }

    #[test]
    fn invalid_probabilities_and_priors_fail_closed() {
        let mut priors = binary_priors();
        priors[0].probability = f64::NAN;
        assert_eq!(
            rank_experiments(&priors, &[]),
            Err(ExperimentDesignError::InvalidPriors)
        );

        let invalid = vec![ExperimentCandidate {
            id: "bad-row".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.9, 0.9], vec![0.0, 1.0]],
        }];
        assert_eq!(
            rank_experiments(&binary_priors(), &invalid),
            Err(ExperimentDesignError::InvalidLikelihoodRow)
        );
    }

    #[test]
    fn robust_ranker_prefers_consistently_informative_measurement() {
        let priors = binary_priors();
        let scenarios = vec![
            ModelLikelihoodScenario {
                scenario_id: "model-a".into(),
                weight: 0.5,
                candidates: vec![
                    ExperimentCandidate {
                        id: "brittle".into(),
                        cost: 1.0,
                        likelihoods: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    },
                    ExperimentCandidate {
                        id: "robust".into(),
                        cost: 1.0,
                        likelihoods: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
                    },
                ],
            },
            ModelLikelihoodScenario {
                scenario_id: "model-b".into(),
                weight: 0.5,
                candidates: vec![
                    ExperimentCandidate {
                        id: "brittle".into(),
                        cost: 1.0,
                        likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                    },
                    ExperimentCandidate {
                        id: "robust".into(),
                        cost: 1.0,
                        likelihoods: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
                    },
                ],
            },
        ];

        let scores = rank_experiments_robust(&priors, &scenarios).unwrap();
        assert_eq!(scores[0].candidate_id, "robust");
        assert_eq!(scores[1].candidate_id, "brittle");
        assert!(scores[0].worst_case_information_gain_bits > 0.0);
        assert_eq!(scores[0].model_sensitivity_stddev_bits, 0.0);
        assert_eq!(scores[1].worst_case_information_gain_bits, 0.0);
        assert!(scores[1].model_sensitivity_stddev_bits > 0.0);
    }

    #[test]
    fn robust_ranker_rejects_invalid_weights_and_mismatched_candidates() {
        let priors = binary_priors();
        assert_eq!(
            rank_experiments_robust(&priors, &[]),
            Err(ExperimentDesignError::NoModelScenarios)
        );
        let mut scenarios = vec![
            ModelLikelihoodScenario {
                scenario_id: "model-a".into(),
                weight: 0.5,
                candidates: vec![ExperimentCandidate {
                    id: "exp-a".into(),
                    cost: 1.0,
                    likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
                }],
            },
            ModelLikelihoodScenario {
                scenario_id: "model-b".into(),
                weight: 0.4,
                candidates: vec![ExperimentCandidate {
                    id: "exp-a".into(),
                    cost: 1.0,
                    likelihoods: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
                }],
            },
        ];
        assert_eq!(
            rank_experiments_robust(&priors, &scenarios),
            Err(ExperimentDesignError::InvalidModelWeights)
        );

        scenarios[1].weight = 0.5;
        scenarios[1].candidates[0].id = "different-id".into();
        assert_eq!(
            rank_experiments_robust(&priors, &scenarios),
            Err(ExperimentDesignError::InconsistentCandidateSet)
        );

        scenarios[1].candidates[0].id = "exp-a".into();
        scenarios[1].candidates[0].likelihoods =
            vec![vec![0.4, 0.4, 0.2], vec![0.2, 0.3, 0.5]];
        assert_eq!(
            rank_experiments_robust(&priors, &scenarios),
            Err(ExperimentDesignError::InconsistentCandidateSet)
        );
    }

    #[test]
    fn bayesian_update_moves_posterior_toward_likelihood_supported_hypothesis() {
        let priors = binary_priors();
        let candidate = ExperimentCandidate {
            id: "diagnostic".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.9, 0.1], vec![0.2, 0.8]],
        };
        let update = update_posterior(&priors, &candidate, 0).unwrap();
        assert!((update.outcome_probability - 0.55).abs() < 1.0e-12);
        assert!((update.posterior[0].probability - (0.45 / 0.55)).abs() < 1.0e-12);
        assert!((update.posterior[1].probability - (0.10 / 0.55)).abs() < 1.0e-12);
        assert!(update.realized_information_gain_bits > 0.0);
    }

    #[test]
    fn impossible_observation_and_bad_outcome_index_fail_closed() {
        let priors = binary_priors();
        let candidate = ExperimentCandidate {
            id: "deterministic-test".into(),
            cost: 1.0,
            likelihoods: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        };
        assert_eq!(
            update_posterior(&priors, &candidate, 1),
            Err(ExperimentDesignError::ImpossibleObservedOutcome)
        );
        assert_eq!(
            update_posterior(&priors, &candidate, 2),
            Err(ExperimentDesignError::OutcomeIndexOutOfRange)
        );
    }

    #[test]
    fn posterior_update_preserves_order_and_normalizes_small_prior_sum_tolerance() {
        let priors = vec![
            HypothesisPrior { id: "A".into(), probability: 0.5000001 },
            HypothesisPrior { id: "B".into(), probability: 0.5000001 },
        ];
        let candidate = ExperimentCandidate {
            id: "balanced".into(),
            cost: 2.0,
            likelihoods: vec![vec![0.7, 0.3], vec![0.3, 0.7]],
        };
        let update = update_posterior(&priors, &candidate, 0).unwrap();
        assert_eq!(update.posterior[0].id, "A");
        assert_eq!(update.posterior[1].id, "B");
        let sum = update.posterior.iter().map(|prior| prior.probability).sum::<f64>();
        assert!((sum - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn empty_candidate_list_is_valid_and_deterministic() {
        assert!(rank_experiments(&binary_priors(), &[]).unwrap().is_empty());
    }
}
