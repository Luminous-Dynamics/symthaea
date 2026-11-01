//! Dependency-free reference utilities for information-seeking connectome experiments.
//!
//! This crate implements exact discrete expected information gain (EIG) for a
//! finite hypothesis set and candidate experiments. It is a mathematical
//! selector, not a biological model: callers supply priors, outcome likelihoods,
//! and costs. The code does not estimate or calibrate them.
//!
//! For hypothesis H and candidate experiment e with outcome Y:
//!
//!     EIG(e) = H(H) - E_Y[H(H | Y, e)] = I(H; Y | e)
//!
//! Entropy and EIG are reported in bits. Cost-normalized EIG is bits per unit of
//! caller-defined cost. It is not interchangeable with downstream decision
//! utility or regret.

use core::fmt;
use std::collections::HashSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const ROUND_OFF_TOLERANCE: f64 = 1.0e-12;

/// Errors are explicit so malformed priors and likelihoods are never silently
/// normalized into plausible-looking experiment rankings.
#[derive(Debug, Clone, PartialEq)]
pub enum DesignError {
    EmptyHypothesisSet,
    PriorCountMismatch { expected: usize, actual: usize },
    InvalidPrior { index: usize, value: f64 },
    PriorMassNotOne { sum: f64 },
    EmptyExperimentId,
    DuplicateExperimentId { experiment_id: String },
    InvalidExperimentCost { cost: f64 },
    NoCandidateExperiments,
    EmptyOutcomeSet,
    DuplicateOutcomeLabel { label: String },
    LikelihoodRowCountMismatch { expected: usize, actual: usize },
    LikelihoodOutcomeCountMismatch {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidLikelihood {
        hypothesis_index: usize,
        outcome_index: usize,
        value: f64,
    },
    LikelihoodMassNotOne { hypothesis_index: usize, sum: f64 },
    OutcomeIndexOutOfRange { index: usize, outcome_count: usize },
    ImpossibleObservedOutcome { outcome_index: usize },
    SelectedExperimentNotFound { experiment_id: String },
    NumericalInstability,
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis set must not be empty"),
            Self::PriorCountMismatch { expected, actual } => {
                write!(f, "expected {expected} hypothesis priors, received {actual}")
            }
            Self::InvalidPrior { index, value } => write!(f, "prior {index} is invalid: {value}"),
            Self::PriorMassNotOne { sum } => {
                write!(f, "hypothesis priors must sum to one, received {sum}")
            }
            Self::EmptyExperimentId => write!(f, "experiment id must not be empty"),
            Self::DuplicateExperimentId { experiment_id } => {
                write!(f, "duplicate experiment id {experiment_id:?}")
            }
            Self::InvalidExperimentCost { cost } => {
                write!(f, "experiment cost must be finite and positive, received {cost}")
            }
            Self::NoCandidateExperiments => write!(f, "candidate experiment set must not be empty"),
            Self::EmptyOutcomeSet => write!(f, "candidate experiment must declare at least one outcome"),
            Self::DuplicateOutcomeLabel { label } => {
                write!(f, "candidate experiment has duplicate outcome label {label:?}")
            }
            Self::LikelihoodRowCountMismatch { expected, actual } => {
                write!(f, "expected {expected} likelihood rows, received {actual}")
            }
            Self::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has {actual} outcome likelihoods; expected {expected}"
            ),
            Self::InvalidLikelihood {
                hypothesis_index,
                outcome_index,
                value,
            } => write!(
                f,
                "likelihood for hypothesis {hypothesis_index}, outcome {outcome_index} is invalid: {value}"
            ),
            Self::LikelihoodMassNotOne { hypothesis_index, sum } => write!(
                f,
                "likelihoods for hypothesis {hypothesis_index} must sum to one, received {sum}"
            ),
            Self::OutcomeIndexOutOfRange { index, outcome_count } => {
                write!(f, "outcome index {index} is outside 0..{outcome_count}")
            }
            Self::ImpossibleObservedOutcome { outcome_index } => {
                write!(f, "outcome {outcome_index} has zero probability under every hypothesis")
            }
            Self::SelectedExperimentNotFound { experiment_id } => {
                write!(f, "selected experiment {experiment_id:?} is not in the candidate set")
            }
            Self::NumericalInstability => {
                write!(f, "entropy calculation exceeded its numerical tolerance")
            }
        }
    }
}

impl std::error::Error for DesignError {}

/// A candidate experiment's likelihood model.
///
/// likelihoods_by_hypothesis[h][y] is P(Y = outcome[y] | H = h, experiment).
/// Every row must be a normalized probability distribution. The outcome order
/// is shared across all hypotheses.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateExperiment {
    pub id: String,
    /// Caller-defined resource cost. Must be finite and strictly positive.
    pub cost: f64,
    pub outcomes: Vec<String>,
    pub likelihoods_by_hypothesis: Vec<Vec<f64>>,
}

/// Information score for one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationGainScore {
    pub prior_entropy_bits: f64,
    pub expected_posterior_entropy_bits: f64,
    pub information_gain_bits: f64,
    pub cost: f64,
    pub information_gain_bits_per_cost: f64,
}

/// Candidate and score after deterministic ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedExperiment {
    pub experiment_id: String,
    pub score: InformationGainScore,
}

/// Opportunity regret relative to the best candidate in a declared set.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRegret {
    pub selected_experiment_id: String,
    pub best_experiment_id: String,
    pub selected_utility_bits_per_cost: f64,
    pub best_utility_bits_per_cost: f64,
    pub regret_bits_per_cost: f64,
}

/// Shannon entropy in bits. Input probabilities must be non-negative and sum
/// to one; all production calls validate this before evaluating entropy.
pub fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|&probability| -probability * probability.log2())
        .sum()
}

fn validate_priors(priors: &[f64]) -> Result<(), DesignError> {
    if priors.is_empty() {
        return Err(DesignError::EmptyHypothesisSet);
    }

    let mut sum = 0.0;
    for (index, &prior) in priors.iter().enumerate() {
        if !prior.is_finite() || prior < 0.0 {
            return Err(DesignError::InvalidPrior { index, value: prior });
        }
        sum += prior;
    }

    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(DesignError::PriorMassNotOne { sum });
    }
    Ok(())
}

fn validate_candidate(
    priors: &[f64],
    candidate: &CandidateExperiment,
) -> Result<usize, DesignError> {
    validate_priors(priors)?;

    if candidate.id.trim().is_empty() {
        return Err(DesignError::EmptyExperimentId);
    }
    if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
        return Err(DesignError::InvalidExperimentCost { cost: candidate.cost });
    }
    if candidate.outcomes.is_empty() {
        return Err(DesignError::EmptyOutcomeSet);
    }

    let mut labels = HashSet::with_capacity(candidate.outcomes.len());
    for label in &candidate.outcomes {
        if !labels.insert(label.as_str()) {
            return Err(DesignError::DuplicateOutcomeLabel { label: label.clone() });
        }
    }

    if candidate.likelihoods_by_hypothesis.len() != priors.len() {
        return Err(DesignError::LikelihoodRowCountMismatch {
            expected: priors.len(),
            actual: candidate.likelihoods_by_hypothesis.len(),
        });
    }

    let outcome_count = candidate.outcomes.len();
    for (hypothesis_index, row) in candidate.likelihoods_by_hypothesis.iter().enumerate() {
        if row.len() != outcome_count {
            return Err(DesignError::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected: outcome_count,
                actual: row.len(),
            });
        }

        let mut row_sum = 0.0;
        for (outcome_index, &likelihood) in row.iter().enumerate() {
            if !likelihood.is_finite() || likelihood < 0.0 {
                return Err(DesignError::InvalidLikelihood {
                    hypothesis_index,
                    outcome_index,
                    value: likelihood,
                });
            }
            row_sum += likelihood;
        }

        if !row_sum.is_finite() || (row_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(DesignError::LikelihoodMassNotOne {
                hypothesis_index,
                sum: row_sum,
            });
        }
    }

    Ok(outcome_count)
}

/// Calculate EIG exactly for a finite discrete hypothesis and outcome space.
pub fn evaluate_candidate(
    priors: &[f64],
    candidate: &CandidateExperiment,
) -> Result<InformationGainScore, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    let prior_entropy = entropy_bits(priors);

    // P(y) = sum_h P(h) P(y | h, e).
    let mut outcome_marginals = vec![0.0; outcome_count];
    for (hypothesis_index, &prior) in priors.iter().enumerate() {
        for (outcome_index, &likelihood) in candidate.likelihoods_by_hypothesis[hypothesis_index]
            .iter()
            .enumerate()
        {
            outcome_marginals[outcome_index] += prior * likelihood;
        }
    }

    // E_y[H(H | y, e)] using Bayes' rule.
    let mut expected_posterior_entropy = 0.0;
    for (outcome_index, &marginal) in outcome_marginals.iter().enumerate() {
        if marginal <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = priors
            .iter()
            .enumerate()
            .map(|(hypothesis_index, &prior)| {
                prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                    / marginal
            })
            .collect();

        expected_posterior_entropy += marginal * entropy_bits(&posterior);
    }

    let mut information_gain = prior_entropy - expected_posterior_entropy;
    if information_gain < 0.0 && information_gain >= -ROUND_OFF_TOLERANCE {
        information_gain = 0.0;
    }
    if information_gain > prior_entropy
        && information_gain <= prior_entropy + ROUND_OFF_TOLERANCE
    {
        information_gain = prior_entropy;
    }
    if !information_gain.is_finite()
        || information_gain < 0.0
        || information_gain > prior_entropy + ROUND_OFF_TOLERANCE
    {
        return Err(DesignError::NumericalInstability);
    }

    Ok(InformationGainScore {
        prior_entropy_bits: prior_entropy,
        expected_posterior_entropy_bits: expected_posterior_entropy,
        information_gain_bits: information_gain,
        cost: candidate.cost,
        information_gain_bits_per_cost: information_gain / candidate.cost,
    })
}

/// Return P(Y = outcomes[outcome_index] | experiment).
pub fn outcome_probability(
    priors: &[f64],
    candidate: &CandidateExperiment,
    outcome_index: usize,
) -> Result<f64, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    if outcome_index >= outcome_count {
        return Err(DesignError::OutcomeIndexOutOfRange {
            index: outcome_index,
            outcome_count,
        });
    }

    Ok(priors
        .iter()
        .enumerate()
        .map(|(hypothesis_index, &prior)| {
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
        })
        .sum())
}

/// Update the hypothesis prior after observing one candidate outcome.
pub fn posterior_for_observed_outcome(
    priors: &[f64],
    candidate: &CandidateExperiment,
    outcome_index: usize,
) -> Result<Vec<f64>, DesignError> {
    let marginal = outcome_probability(priors, candidate, outcome_index)?;
    if marginal <= 0.0 || !marginal.is_finite() {
        return Err(DesignError::ImpossibleObservedOutcome { outcome_index });
    }

    Ok(priors
        .iter()
        .enumerate()
        .map(|(hypothesis_index, &prior)| {
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                / marginal
        })
        .collect())
}

/// Rank by EIG per cost, then raw EIG, then ascending ID. The final ID
/// tie-break makes output stable regardless of input order.
pub fn rank_candidates(
    priors: &[f64],
    candidates: &[CandidateExperiment],
) -> Result<Vec<RankedExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_candidate(priors, candidate)?,
        });
    }

    ranked.sort_by(|left, right| {
        right
            .score
            .information_gain_bits_per_cost
            .total_cmp(&left.score.information_gain_bits_per_cost)
            .then_with(|| {
                right
                    .score
                    .information_gain_bits
                    .total_cmp(&left.score.information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    Ok(ranked)
}

/// Compute opportunity regret under the EIG-per-cost objective. This is
/// regret relative to the best experiment in this finite candidate set, not
/// behavioural utility or real-world scientific value.
pub fn selection_regret(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    selected_experiment_id: &str,
) -> Result<SelectionRegret, DesignError> {
    let ranked = rank_candidates(priors, candidates)?;
    let best = ranked.first().ok_or(DesignError::NoCandidateExperiments)?;
    let selected = ranked
        .iter()
        .find(|candidate| candidate.experiment_id == selected_experiment_id)
        .ok_or_else(|| DesignError::SelectedExperimentNotFound {
            experiment_id: selected_experiment_id.to_owned(),
        })?;

    let best_utility = best.score.information_gain_bits_per_cost;
    let selected_utility = selected.score.information_gain_bits_per_cost;
    let regret = (best_utility - selected_utility).max(0.0);

    Ok(SelectionRegret {
        selected_experiment_id: selected.experiment_id.clone(),
        best_experiment_id: best.experiment_id.clone(),
        selected_utility_bits_per_cost: selected_utility,
        best_utility_bits_per_cost: best_utility,
        regret_bits_per_cost: regret,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_experiment(
        id: &str,
        cost: f64,
        response_given_h0: f64,
        response_given_h1: f64,
    ) -> CandidateExperiment {
        CandidateExperiment {
            id: id.to_owned(),
            cost,
            outcomes: vec!["response".to_owned(), "no_response".to_owned()],
            likelihoods_by_hypothesis: vec![
                vec![response_given_h0, 1.0 - response_given_h0],
                vec![response_given_h1, 1.0 - response_given_h1],
            ],
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15}, expected {expected:.15}, tolerance {tolerance}"
        );
    }

    #[test]
    fn uninformative_experiment_has_zero_eig() {
        let experiment = binary_experiment("uninformative", 1.0, 0.5, 0.5);
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.prior_entropy_bits, 1.0, 1.0e-12);
        close(score.expected_posterior_entropy_bits, 1.0, 1.0e-12);
        close(score.information_gain_bits, 0.0, 1.0e-12);
    }

    #[test]
    fn perfect_discrimination_yields_one_bit_for_equal_binary_prior() {
        let experiment = CandidateExperiment {
            id: "perfect".to_owned(),
            cost: 1.0,
            outcomes: vec!["left".to_owned(), "right".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.information_gain_bits, 1.0, 1.0e-12);
        close(score.expected_posterior_entropy_bits, 0.0, 1.0e-12);
    }

    #[test]
    fn noisy_ninety_ten_discriminator_matches_analytic_eig() {
        let experiment = binary_experiment("noisy", 1.0, 0.9, 0.1);
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.information_gain_bits, 0.531_004_406_410_718_8, 1.0e-12);
    }

    #[test]
    fn perfect_discrimination_reduces_asymmetric_prior_to_zero_entropy() {
        let experiment = CandidateExperiment {
            id: "asymmetric-perfect".to_owned(),
            cost: 1.0,
            outcomes: vec!["h0".to_owned(), "h1".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_candidate(&[0.8, 0.2], &experiment).unwrap();
        close(score.prior_entropy_bits, 0.721_928_094_887_362_3, 1.0e-12);
        close(score.information_gain_bits, score.prior_entropy_bits, 1.0e-12);
    }

    #[test]
    fn posterior_update_uses_bayes_rule() {
        let experiment = binary_experiment("bayes", 1.0, 0.8, 0.2);
        let posterior = posterior_for_observed_outcome(&[0.5, 0.5], &experiment, 0).unwrap();
        close(posterior[0], 0.8, 1.0e-12);
        close(posterior[1], 0.2, 1.0e-12);
    }

    #[test]
    fn cost_aware_ranking_can_prefer_less_raw_information() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let ranked = rank_candidates(&[0.5, 0.5], &[expensive, cheap]).unwrap();
        assert_eq!(ranked[0].experiment_id, "cheap");
        assert!(ranked[1].score.information_gain_bits > ranked[0].score.information_gain_bits);
        assert!(
            ranked[0].score.information_gain_bits_per_cost
                > ranked[1].score.information_gain_bits_per_cost
        );
    }

    #[test]
    fn candidate_order_does_not_change_ranked_tie_break() {
        let alpha = binary_experiment("alpha", 1.0, 0.9, 0.1);
        let zeta = binary_experiment("zeta", 1.0, 0.9, 0.1);
        let first = rank_candidates(&[0.5, 0.5], &[zeta.clone(), alpha.clone()]).unwrap();
        let second = rank_candidates(&[0.5, 0.5], &[alpha, zeta]).unwrap();
        assert_eq!(
            first.iter().map(|item| &item.experiment_id).collect::<Vec<_>>(),
            second.iter().map(|item| &item.experiment_id).collect::<Vec<_>>()
        );
        assert_eq!(first[0].experiment_id, "alpha");
    }

    #[test]
    fn selection_regret_is_zero_for_best_candidate_and_positive_otherwise() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let best = selection_regret(&[0.5, 0.5], &[expensive.clone(), cheap.clone()], "cheap")
            .unwrap();
        close(best.regret_bits_per_cost, 0.0, 1.0e-12);
        let regret = selection_regret(&[0.5, 0.5], &[expensive, cheap], "expensive").unwrap();
        assert!(regret.regret_bits_per_cost > 0.0);
        assert_eq!(regret.best_experiment_id, "cheap");
    }

    #[test]
    fn malformed_prior_fails_closed_without_renormalization() {
        let experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        assert_eq!(
            evaluate_candidate(&[0.4, 0.4], &experiment),
            Err(DesignError::PriorMassNotOne { sum: 0.8 })
        );
    }

    #[test]
    fn malformed_likelihood_fails_closed_without_renormalization() {
        let mut experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        experiment.likelihoods_by_hypothesis[0] = vec![0.9, 0.2];
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &experiment),
            Err(DesignError::LikelihoodMassNotOne { hypothesis_index: 0, .. })
        ));
    }

    #[test]
    fn non_finite_probability_and_non_positive_cost_are_rejected() {
        let mut experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        experiment.likelihoods_by_hypothesis[0][0] = f64::NAN;
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &experiment),
            Err(DesignError::InvalidLikelihood {
                hypothesis_index: 0,
                outcome_index: 0,
                ..
            })
        ));
        let bad_cost = binary_experiment("bad-cost", 0.0, 0.9, 0.1);
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &bad_cost),
            Err(DesignError::InvalidExperimentCost { .. })
        ));
    }

    #[test]
    fn impossible_observed_outcome_has_no_arbitrary_posterior() {
        let experiment = CandidateExperiment {
            id: "impossible".to_owned(),
            cost: 1.0,
            outcomes: vec!["only".to_owned(), "never".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        };
        assert_eq!(
            posterior_for_observed_outcome(&[0.5, 0.5], &experiment, 1),
            Err(DesignError::ImpossibleObservedOutcome { outcome_index: 1 })
        );
    }

    #[test]
    fn duplicate_outcomes_and_unknown_selection_fail_closed() {
        let mut duplicate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        duplicate.outcomes[1] = duplicate.outcomes[0].clone();
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &duplicate),
            Err(DesignError::DuplicateOutcomeLabel { .. })
        ));

        let valid = binary_experiment("valid", 1.0, 0.9, 0.1);
        assert!(matches!(
            selection_regret(&[0.5, 0.5], &[valid], "missing"),
            Err(DesignError::SelectedExperimentNotFound { .. })
        ));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let first = binary_experiment("same", 1.0, 0.9, 0.1);
        let second = binary_experiment("same", 2.0, 0.8, 0.2);
        assert!(matches!(
            rank_candidates(&[0.5, 0.5], &[first, second]),
            Err(DesignError::DuplicateExperimentId { .. })
        ));
    }

    #[test]
    fn prior_entropy_bounds_information_gain() {
        let experiment = binary_experiment("bounded", 1.0, 1.0, 0.0);
        let score = evaluate_candidate(&[0.9, 0.1], &experiment).unwrap();
        assert!(score.information_gain_bits <= score.prior_entropy_bits);
        assert!(score.information_gain_bits >= 0.0);
    }
}
