//! Dependency-free reference utilities for information-seeking connectome experiments.
//!
//! This crate implements exact discrete expected information gain (EIG) for a
//! finite hypothesis set and candidate experiments. It is a mathematical
//! selector, not a biological model: callers supply priors, outcome likelihoods,
//! and costs. The code does not estimate or calibrate them.
//!
//! For hypothesis H and candidate experiment e with outcome Y:
//!
//! ```text
//! EIG(e) = H(H) - E_Y[H(H | Y, e)] = I(H; Y | e)
//! ```
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
    PriorCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidPrior {
        index: usize,
        value: f64,
    },
    PriorMassNotOne {
        sum: f64,
    },
    EmptyExperimentId,
    DuplicateExperimentId {
        experiment_id: String,
    },
    InvalidExperimentCost {
        cost: f64,
    },
    NoCandidateExperiments,
    EmptyOutcomeSet,
    DuplicateOutcomeLabel {
        label: String,
    },
    LikelihoodRowCountMismatch {
        expected: usize,
        actual: usize,
    },
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
    LikelihoodMassNotOne {
        hypothesis_index: usize,
        sum: f64,
    },
    OutcomeIndexOutOfRange {
        index: usize,
        outcome_count: usize,
    },
    ImpossibleObservedOutcome {
        outcome_index: usize,
    },
    SelectedExperimentNotFound {
        experiment_id: String,
    },
    EmptyActionSet,
    EmptyActionLabel,
    DuplicateActionLabel {
        label: String,
    },
    DecisionUtilityRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    DecisionUtilityActionCountMismatch {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidDecisionUtility {
        hypothesis_index: usize,
        action_index: usize,
        value: f64,
    },
    EmptyParameterSet,
    FisherPriorRowLengthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    InvalidFisherPrior {
        row: usize,
        column: usize,
        value: f64,
    },
    NonSymmetricFisherPrior {
        row: usize,
        column: usize,
        left: f64,
        right: f64,
    },
    FisherPriorNotPositiveDefinite {
        pivot: usize,
    },
    FisherSensitivityRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    FisherSensitivityRowLengthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    InvalidFisherSensitivity {
        row: usize,
        column: usize,
        value: f64,
    },
    InvalidObservationVariance {
        row: usize,
        value: f64,
    },
    NumericalInstability,
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis set must not be empty"),
            Self::PriorCountMismatch { expected, actual } => {
                write!(
                    f,
                    "expected {expected} hypothesis priors, received {actual}"
                )
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
                write!(
                    f,
                    "experiment cost must be finite and positive, received {cost}"
                )
            }
            Self::NoCandidateExperiments => write!(f, "candidate experiment set must not be empty"),
            Self::EmptyOutcomeSet => {
                write!(f, "candidate experiment must declare at least one outcome")
            }
            Self::DuplicateOutcomeLabel { label } => {
                write!(
                    f,
                    "candidate experiment has duplicate outcome label {label:?}"
                )
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
            Self::LikelihoodMassNotOne {
                hypothesis_index,
                sum,
            } => write!(
                f,
                "likelihoods for hypothesis {hypothesis_index} must sum to one, received {sum}"
            ),
            Self::OutcomeIndexOutOfRange {
                index,
                outcome_count,
            } => {
                write!(f, "outcome index {index} is outside 0..{outcome_count}")
            }
            Self::ImpossibleObservedOutcome { outcome_index } => {
                write!(
                    f,
                    "outcome {outcome_index} has zero probability under every hypothesis"
                )
            }
            Self::SelectedExperimentNotFound { experiment_id } => {
                write!(
                    f,
                    "selected experiment {experiment_id:?} is not in the candidate set"
                )
            }
            Self::EmptyActionSet => write!(f, "decision action set must not be empty"),
            Self::EmptyActionLabel => write!(f, "decision action label must not be empty"),
            Self::DuplicateActionLabel { label } => {
                write!(f, "duplicate decision action label {label:?}")
            }
            Self::DecisionUtilityRowCountMismatch { expected, actual } => write!(
                f,
                "expected {expected} decision-utility rows, received {actual}"
            ),
            Self::DecisionUtilityActionCountMismatch {
                hypothesis_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has {actual} action utilities; expected {expected}"
            ),
            Self::InvalidDecisionUtility {
                hypothesis_index,
                action_index,
                value,
            } => write!(
                f,
                "utility for hypothesis {hypothesis_index}, action {action_index} is invalid: {value}"
            ),
            Self::EmptyParameterSet => write!(f, "Fisher design requires at least one parameter"),
            Self::FisherPriorRowLengthMismatch {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Fisher prior precision row {row} has {actual} values; expected {expected}"
            ),
            Self::InvalidFisherPrior { row, column, value } => write!(
                f,
                "Fisher prior precision entry ({row},{column}) is invalid: {value}"
            ),
            Self::NonSymmetricFisherPrior {
                row,
                column,
                left,
                right,
            } => write!(
                f,
                "Fisher prior precision is not symmetric at ({row},{column}): {left} versus {right}"
            ),
            Self::FisherPriorNotPositiveDefinite { pivot } => write!(
                f,
                "Fisher prior precision is not positive definite at Cholesky pivot {pivot}"
            ),
            Self::FisherSensitivityRowCountMismatch { expected, actual } => write!(
                f,
                "Fisher design has {actual} observation-variance entries; expected {expected} sensitivity rows"
            ),
            Self::FisherSensitivityRowLengthMismatch {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Fisher sensitivity row {row} has {actual} parameters; expected {expected}"
            ),
            Self::InvalidFisherSensitivity { row, column, value } => {
                write!(f, "Fisher sensitivity ({row},{column}) is invalid: {value}")
            }
            Self::InvalidObservationVariance { row, value } => write!(
                f,
                "observation variance {row} must be finite and positive, received {value}"
            ),
            Self::NumericalInstability => {
                write!(
                    f,
                    "information or decision-value calculation exceeded its numerical tolerance"
                )
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

/// Decision problem used to calculate expected value of sample information.
///
/// `utilities_by_hypothesis[h][a]` is the utility of selecting action `a`
/// when hypothesis `h` is true. Utilities are caller-defined, finite values;
/// they need not be probabilities or non-negative.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionProblem {
    pub actions: Vec<String>,
    pub utilities_by_hypothesis: Vec<Vec<f64>>,
}

/// Expected value of sample information (EVSI) for one candidate experiment.
///
/// EVSI is measured in the caller's utility units. It is deliberately distinct
/// from EIG (bits): an experiment can distinguish hypotheses that imply the
/// same action and therefore have high EIG but zero decision value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionValueScore {
    pub prior_best_expected_utility: f64,
    pub expected_best_utility_after_experiment: f64,
    pub expected_value_of_sample_information: f64,
    pub cost: f64,
    pub expected_value_per_cost: f64,
}

/// Candidate and decision-value score after deterministic ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedDecisionExperiment {
    pub experiment_id: String,
    pub score: DecisionValueScore,
}

/// Decision regret relative to the best candidate in the supplied finite pool.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionSelectionRegret {
    pub selected_experiment_id: String,
    pub best_experiment_id: String,
    pub selected_expected_value_per_cost: f64,
    pub best_expected_value_per_cost: f64,
    pub regret_per_cost: f64,
}

/// Candidate experiment represented by local output sensitivities.
///
/// Each row is the Jacobian of one scalar measured output with respect to the
/// declared parameters at the design point. Rows are weighted by independent
/// Gaussian observation variances. Correlated observations must be whitened by
/// the caller; full observation-covariance support is not implemented here.
#[derive(Debug, Clone, PartialEq)]
pub struct FisherInformationExperiment {
    pub id: String,
    pub cost: f64,
    pub sensitivities_by_observation: Vec<Vec<f64>>,
    pub observation_variances: Vec<f64>,
}

/// Parameter-identification objectives over a shared Fisher information matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FisherObjective {
    /// Half the log determinant ratio: Gaussian parameter-volume reduction in nats.
    DOptimal,
    /// Reduction in trace of posterior covariance (sum of marginal variances).
    AOptimal,
}

/// Fisher-information gains and costs for one candidate design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherInformationScore {
    pub parameter_count: usize,
    pub d_optimal_gain_nats: f64,
    pub a_optimal_variance_reduction: f64,
    pub cost: f64,
    pub d_optimal_gain_per_cost: f64,
    pub a_optimal_variance_reduction_per_cost: f64,
}

/// Candidate and Fisher-information score after deterministic ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedFisherExperiment {
    pub experiment_id: String,
    pub score: FisherInformationScore,
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
            return Err(DesignError::InvalidPrior {
                index,
                value: prior,
            });
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
        return Err(DesignError::InvalidExperimentCost {
            cost: candidate.cost,
        });
    }
    if candidate.outcomes.is_empty() {
        return Err(DesignError::EmptyOutcomeSet);
    }

    let mut labels = HashSet::with_capacity(candidate.outcomes.len());
    for label in &candidate.outcomes {
        if !labels.insert(label.as_str()) {
            return Err(DesignError::DuplicateOutcomeLabel {
                label: label.clone(),
            });
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
    if (-ROUND_OFF_TOLERANCE..0.0).contains(&information_gain) {
        information_gain = 0.0;
    }
    if information_gain > prior_entropy && information_gain <= prior_entropy + ROUND_OFF_TOLERANCE {
        information_gain = prior_entropy;
    }
    if !information_gain.is_finite()
        || information_gain < 0.0
        || information_gain > prior_entropy + ROUND_OFF_TOLERANCE
    {
        return Err(DesignError::NumericalInstability);
    }

    let information_gain_bits_per_cost = information_gain / candidate.cost;
    if !information_gain_bits_per_cost.is_finite() {
        return Err(DesignError::NumericalInstability);
    }
    Ok(InformationGainScore {
        prior_entropy_bits: prior_entropy,
        expected_posterior_entropy_bits: expected_posterior_entropy,
        information_gain_bits: information_gain,
        cost: candidate.cost,
        information_gain_bits_per_cost,
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
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index] / marginal
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

fn validate_decision_problem(priors: &[f64], problem: &DecisionProblem) -> Result<(), DesignError> {
    validate_priors(priors)?;
    if problem.actions.is_empty() {
        return Err(DesignError::EmptyActionSet);
    }
    let mut action_labels = HashSet::with_capacity(problem.actions.len());
    for label in &problem.actions {
        if label.trim().is_empty() {
            return Err(DesignError::EmptyActionLabel);
        }
        if !action_labels.insert(label.as_str()) {
            return Err(DesignError::DuplicateActionLabel {
                label: label.clone(),
            });
        }
    }
    if problem.utilities_by_hypothesis.len() != priors.len() {
        return Err(DesignError::DecisionUtilityRowCountMismatch {
            expected: priors.len(),
            actual: problem.utilities_by_hypothesis.len(),
        });
    }
    for (hypothesis_index, row) in problem.utilities_by_hypothesis.iter().enumerate() {
        if row.len() != problem.actions.len() {
            return Err(DesignError::DecisionUtilityActionCountMismatch {
                hypothesis_index,
                expected: problem.actions.len(),
                actual: row.len(),
            });
        }
        for (action_index, &utility) in row.iter().enumerate() {
            if !utility.is_finite() {
                return Err(DesignError::InvalidDecisionUtility {
                    hypothesis_index,
                    action_index,
                    value: utility,
                });
            }
        }
    }
    Ok(())
}

fn best_expected_utility(
    probabilities: &[f64],
    utilities_by_hypothesis: &[Vec<f64>],
    action_count: usize,
) -> Result<f64, DesignError> {
    let mut expected_by_action = vec![0.0; action_count];
    for (&probability, utilities) in probabilities.iter().zip(utilities_by_hypothesis) {
        for (expected, &utility) in expected_by_action.iter_mut().zip(utilities) {
            *expected += probability * utility;
        }
    }
    if expected_by_action
        .iter()
        .any(|expected| !expected.is_finite())
    {
        return Err(DesignError::NumericalInstability);
    }
    expected_by_action
        .into_iter()
        .reduce(f64::max)
        .ok_or(DesignError::NumericalInstability)
}

/// Evaluate the expected value of sample information for a candidate.
///
/// This computes the expected improvement in the best available action after
/// observing the candidate's outcome, compared with choosing an action before
/// running the experiment. It is not the same objective as EIG or EIG/cost.
pub fn evaluate_decision_value(
    priors: &[f64],
    candidate: &CandidateExperiment,
    problem: &DecisionProblem,
) -> Result<DecisionValueScore, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    validate_decision_problem(priors, problem)?;

    let prior_best = best_expected_utility(
        priors,
        &problem.utilities_by_hypothesis,
        problem.actions.len(),
    )?;

    let mut marginals = vec![0.0; outcome_count];
    for (h, &prior) in priors.iter().enumerate() {
        for (y, &likelihood) in candidate.likelihoods_by_hypothesis[h].iter().enumerate() {
            marginals[y] += prior * likelihood;
        }
    }

    let mut expected_post_experiment_best = 0.0;
    for (outcome_index, &marginal) in marginals.iter().enumerate() {
        if marginal <= 0.0 {
            continue;
        }
        let posterior: Vec<f64> = priors
            .iter()
            .enumerate()
            .map(|(h, &prior)| {
                prior * candidate.likelihoods_by_hypothesis[h][outcome_index] / marginal
            })
            .collect();
        let post_best = best_expected_utility(
            &posterior,
            &problem.utilities_by_hypothesis,
            problem.actions.len(),
        )?;
        expected_post_experiment_best += marginal * post_best;
    }

    let mut evsi = expected_post_experiment_best - prior_best;
    if (-ROUND_OFF_TOLERANCE..0.0).contains(&evsi) {
        evsi = 0.0;
    }
    if !evsi.is_finite() || evsi < 0.0 {
        return Err(DesignError::NumericalInstability);
    }

    let expected_value_per_cost = evsi / candidate.cost;
    if !expected_value_per_cost.is_finite() {
        return Err(DesignError::NumericalInstability);
    }
    Ok(DecisionValueScore {
        prior_best_expected_utility: prior_best,
        expected_best_utility_after_experiment: expected_post_experiment_best,
        expected_value_of_sample_information: evsi,
        cost: candidate.cost,
        expected_value_per_cost,
    })
}

/// Rank candidate experiments by EVSI per cost, then raw EVSI, then stable ID.
pub fn rank_candidates_by_decision_value(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    problem: &DecisionProblem,
) -> Result<Vec<RankedDecisionExperiment>, DesignError> {
    validate_priors(priors)?;
    validate_decision_problem(priors, problem)?;
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
        ranked.push(RankedDecisionExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_decision_value(priors, candidate, problem)?,
        });
    }
    ranked.sort_by(|left, right| {
        right
            .score
            .expected_value_per_cost
            .total_cmp(&left.score.expected_value_per_cost)
            .then_with(|| {
                right
                    .score
                    .expected_value_of_sample_information
                    .total_cmp(&left.score.expected_value_of_sample_information)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

/// Compute decision-value regret against the best candidate in the declared pool.
///
/// This is an EVSI-per-cost opportunity gap. It is distinct from EIG-selection
/// regret and should not be presented as realized biological utility.
pub fn decision_selection_regret(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    problem: &DecisionProblem,
    selected_experiment_id: &str,
) -> Result<DecisionSelectionRegret, DesignError> {
    let ranked = rank_candidates_by_decision_value(priors, candidates, problem)?;
    let best = ranked.first().ok_or(DesignError::NoCandidateExperiments)?;
    let selected = ranked
        .iter()
        .find(|candidate| candidate.experiment_id == selected_experiment_id)
        .ok_or_else(|| DesignError::SelectedExperimentNotFound {
            experiment_id: selected_experiment_id.to_owned(),
        })?;
    let best_utility = best.score.expected_value_per_cost;
    let selected_utility = selected.score.expected_value_per_cost;
    Ok(DecisionSelectionRegret {
        selected_experiment_id: selected.experiment_id.clone(),
        best_experiment_id: best.experiment_id.clone(),
        selected_expected_value_per_cost: selected_utility,
        best_expected_value_per_cost: best_utility,
        regret_per_cost: (best_utility - selected_utility).max(0.0),
    })
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, DesignError> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            if !value.is_finite() {
                return Err(DesignError::NumericalInstability);
            }
            if i == j {
                if value <= 0.0 {
                    return Err(DesignError::FisherPriorNotPositiveDefinite { pivot: i });
                }
                lower[i][j] = value.sqrt();
            } else {
                let pivot = lower[j][j];
                if pivot <= 0.0 || !pivot.is_finite() {
                    return Err(DesignError::FisherPriorNotPositiveDefinite { pivot: j });
                }
                lower[i][j] = value / pivot;
            }
        }
    }
    Ok(lower)
}

fn log_determinant_from_cholesky(lower: &[Vec<f64>]) -> Result<f64, DesignError> {
    let mut result = 0.0;
    for (index, row) in lower.iter().enumerate() {
        let diagonal = row[index];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(DesignError::NumericalInstability);
        }
        result += 2.0 * diagonal.ln();
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(DesignError::NumericalInstability)
    }
}

fn solve_spd_from_cholesky(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, DesignError> {
    let n = lower.len();
    if rhs.len() != n {
        return Err(DesignError::NumericalInstability);
    }
    let mut forward = vec![0.0; n];
    for i in 0..n {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[i][j] * forward[j];
        }
        forward[i] = value / lower[i][i];
        if !forward[i].is_finite() {
            return Err(DesignError::NumericalInstability);
        }
    }
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        let mut value = forward[i];
        for j in (i + 1)..n {
            value -= lower[j][i] * solution[j];
        }
        solution[i] = value / lower[i][i];
        if !solution[i].is_finite() {
            return Err(DesignError::NumericalInstability);
        }
    }
    Ok(solution)
}

fn trace_inverse_from_cholesky(lower: &[Vec<f64>]) -> Result<f64, DesignError> {
    let n = lower.len();
    let mut trace = 0.0;
    for column in 0..n {
        let mut basis = vec![0.0; n];
        basis[column] = 1.0;
        trace += solve_spd_from_cholesky(lower, &basis)?[column];
    }
    if trace.is_finite() && trace >= 0.0 {
        Ok(trace)
    } else {
        Err(DesignError::NumericalInstability)
    }
}

fn validate_fisher_prior(prior: &[Vec<f64>]) -> Result<usize, DesignError> {
    let n = prior.len();
    if n == 0 {
        return Err(DesignError::EmptyParameterSet);
    }
    if n > 4096 {
        return Err(DesignError::NumericalInstability);
    }
    for (row_index, row) in prior.iter().enumerate() {
        if row.len() != n {
            return Err(DesignError::FisherPriorRowLengthMismatch {
                row: row_index,
                expected: n,
                actual: row.len(),
            });
        }
    }
    for (row_index, row) in prior.iter().enumerate() {
        for (column, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(DesignError::InvalidFisherPrior {
                    row: row_index,
                    column,
                    value,
                });
            }
            let transpose = prior[column][row_index];
            if !transpose.is_finite() {
                return Err(DesignError::InvalidFisherPrior {
                    row: column,
                    column: row_index,
                    value: transpose,
                });
            }
            let tolerance = 1.0e-10 * value.abs().max(transpose.abs()).max(1.0);
            if (value - transpose).abs() > tolerance {
                return Err(DesignError::NonSymmetricFisherPrior {
                    row: row_index,
                    column,
                    left: value,
                    right: transpose,
                });
            }
        }
    }
    Ok(n)
}

fn symmetrized_prior(prior: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = prior.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = 0.5 * (prior[i][j] + prior[j][i]);
        }
    }
    result
}

fn validate_fisher_candidate(
    parameter_count: usize,
    candidate: &FisherInformationExperiment,
) -> Result<(), DesignError> {
    if candidate.id.trim().is_empty() {
        return Err(DesignError::EmptyExperimentId);
    }
    if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
        return Err(DesignError::InvalidExperimentCost {
            cost: candidate.cost,
        });
    }
    if candidate.sensitivities_by_observation.len() != candidate.observation_variances.len() {
        return Err(DesignError::FisherSensitivityRowCountMismatch {
            expected: candidate.sensitivities_by_observation.len(),
            actual: candidate.observation_variances.len(),
        });
    }
    for (row_index, row) in candidate.sensitivities_by_observation.iter().enumerate() {
        if row.len() != parameter_count {
            return Err(DesignError::FisherSensitivityRowLengthMismatch {
                row: row_index,
                expected: parameter_count,
                actual: row.len(),
            });
        }
        for (column, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(DesignError::InvalidFisherSensitivity {
                    row: row_index,
                    column,
                    value,
                });
            }
        }
        let variance = candidate.observation_variances[row_index];
        if !variance.is_finite() || variance <= 0.0 {
            return Err(DesignError::InvalidObservationVariance {
                row: row_index,
                value: variance,
            });
        }
    }
    Ok(())
}

/// F = Jᵀ R⁻¹ J for independent Gaussian observations.
fn fisher_posterior_precision(
    prior: &[Vec<f64>],
    candidate: &FisherInformationExperiment,
) -> Result<Vec<Vec<f64>>, DesignError> {
    let n = prior.len();
    let mut posterior = symmetrized_prior(prior);
    for (row_index, sensitivity) in candidate.sensitivities_by_observation.iter().enumerate() {
        let inverse_variance = 1.0 / candidate.observation_variances[row_index];
        if !inverse_variance.is_finite() {
            return Err(DesignError::NumericalInstability);
        }
        for i in 0..n {
            for j in 0..n {
                posterior[i][j] += sensitivity[i] * sensitivity[j] * inverse_variance;
                if !posterior[i][j].is_finite() {
                    return Err(DesignError::NumericalInstability);
                }
            }
        }
    }
    Ok(posterior)
}

/// Evaluate local Fisher-information gains for one candidate experiment.
///
/// D-optimal gain is 0.5 * ln(det(P_post) / det(P_prior)), the reduction in
/// Gaussian parameter uncertainty volume (nats). A-optimal gain is the
/// reduction in trace(P⁻¹), the sum of marginal parameter variances. This is a
/// local linearized criterion conditioned on caller-supplied sensitivities,
/// independent observation variances and prior precision.
pub fn evaluate_fisher_information_candidate(
    prior_precision: &[Vec<f64>],
    candidate: &FisherInformationExperiment,
) -> Result<FisherInformationScore, DesignError> {
    let parameter_count = validate_fisher_prior(prior_precision)?;
    validate_fisher_candidate(parameter_count, candidate)?;
    let canonical_prior = symmetrized_prior(prior_precision);
    let prior_cholesky = cholesky(&canonical_prior)?;
    let posterior_precision = fisher_posterior_precision(&canonical_prior, candidate)?;
    let posterior_cholesky = cholesky(&posterior_precision)?;
    let prior_logdet = log_determinant_from_cholesky(&prior_cholesky)?;
    let posterior_logdet = log_determinant_from_cholesky(&posterior_cholesky)?;
    let prior_trace_inverse = trace_inverse_from_cholesky(&prior_cholesky)?;
    let posterior_trace_inverse = trace_inverse_from_cholesky(&posterior_cholesky)?;

    let mut d_optimal_gain_nats = 0.5 * (posterior_logdet - prior_logdet);
    let mut a_optimal_variance_reduction = prior_trace_inverse - posterior_trace_inverse;
    if d_optimal_gain_nats < 0.0 && d_optimal_gain_nats >= -ROUND_OFF_TOLERANCE {
        d_optimal_gain_nats = 0.0;
    }
    if a_optimal_variance_reduction < 0.0 && a_optimal_variance_reduction >= -ROUND_OFF_TOLERANCE {
        a_optimal_variance_reduction = 0.0;
    }
    if !d_optimal_gain_nats.is_finite()
        || !a_optimal_variance_reduction.is_finite()
        || d_optimal_gain_nats < 0.0
        || a_optimal_variance_reduction < 0.0
    {
        return Err(DesignError::NumericalInstability);
    }
    let d_optimal_gain_per_cost = d_optimal_gain_nats / candidate.cost;
    let a_optimal_variance_reduction_per_cost = a_optimal_variance_reduction / candidate.cost;
    if !d_optimal_gain_per_cost.is_finite() || !a_optimal_variance_reduction_per_cost.is_finite() {
        return Err(DesignError::NumericalInstability);
    }
    Ok(FisherInformationScore {
        parameter_count,
        d_optimal_gain_nats,
        a_optimal_variance_reduction,
        cost: candidate.cost,
        d_optimal_gain_per_cost,
        a_optimal_variance_reduction_per_cost,
    })
}

/// Rank by D- or A-optimal gain per cost, then raw gain, then stable ID.
pub fn rank_fisher_candidates(
    prior_precision: &[Vec<f64>],
    candidates: &[FisherInformationExperiment],
    objective: FisherObjective,
) -> Result<Vec<RankedFisherExperiment>, DesignError> {
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }
    validate_fisher_prior(prior_precision)?;
    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedFisherExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_fisher_information_candidate(prior_precision, candidate)?,
        });
    }
    ranked.sort_by(|left, right| {
        let (left_cost, right_cost, left_raw, right_raw) = match objective {
            FisherObjective::DOptimal => (
                left.score.d_optimal_gain_per_cost,
                right.score.d_optimal_gain_per_cost,
                left.score.d_optimal_gain_nats,
                right.score.d_optimal_gain_nats,
            ),
            FisherObjective::AOptimal => (
                left.score.a_optimal_variance_reduction_per_cost,
                right.score.a_optimal_variance_reduction_per_cost,
                left.score.a_optimal_variance_reduction,
                right.score.a_optimal_variance_reduction,
            ),
        };
        right_cost
            .total_cmp(&left_cost)
            .then_with(|| right_raw.total_cmp(&left_raw))
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
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

    fn fisher_experiment(
        id: &str,
        cost: f64,
        sensitivities_by_observation: Vec<Vec<f64>>,
        observation_variances: Vec<f64>,
    ) -> FisherInformationExperiment {
        FisherInformationExperiment {
            id: id.to_owned(),
            cost,
            sensitivities_by_observation,
            observation_variances,
        }
    }

    #[test]
    fn fisher_known_answer_computes_d_and_a_optimal_gains() {
        let prior = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let experiment = fisher_experiment(
            "balanced",
            1.0,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![1.0, 1.0],
        );
        let score = evaluate_fisher_information_candidate(&prior, &experiment).unwrap();
        close(score.d_optimal_gain_nats, 2.0_f64.ln(), 1.0e-12);
        close(score.a_optimal_variance_reduction, 1.0, 1.0e-12);
        close(score.d_optimal_gain_per_cost, 2.0_f64.ln(), 1.0e-12);
        close(score.a_optimal_variance_reduction_per_cost, 1.0, 1.0e-12);
    }

    #[test]
    fn d_and_a_optimal_designs_can_choose_different_experiments() {
        let prior = vec![vec![100.0, 0.0], vec![0.0, 1.0]];
        let refine_known = fisher_experiment(
            "refine-already-known-parameter",
            1.0,
            vec![vec![300.0_f64.sqrt(), 0.0]],
            vec![1.0],
        );
        let resolve_uncertain = fisher_experiment(
            "resolve-uncertain-parameter",
            1.0,
            vec![vec![0.0, 1.0]],
            vec![1.0],
        );
        let d_ranked = rank_fisher_candidates(
            &prior,
            &[refine_known.clone(), resolve_uncertain.clone()],
            FisherObjective::DOptimal,
        )
        .unwrap();
        let a_ranked = rank_fisher_candidates(
            &prior,
            &[refine_known, resolve_uncertain],
            FisherObjective::AOptimal,
        )
        .unwrap();
        assert_eq!(d_ranked[0].experiment_id, "refine-already-known-parameter");
        assert_eq!(a_ranked[0].experiment_id, "resolve-uncertain-parameter");
    }

    #[test]
    fn observation_variance_scales_fisher_information() {
        let prior = vec![vec![1.0]];
        let precise = fisher_experiment("precise", 1.0, vec![vec![1.0]], vec![0.5]);
        let noisy = fisher_experiment("noisy", 1.0, vec![vec![1.0]], vec![2.0]);
        let precise_score = evaluate_fisher_information_candidate(&prior, &precise).unwrap();
        let noisy_score = evaluate_fisher_information_candidate(&prior, &noisy).unwrap();
        assert!(precise_score.d_optimal_gain_nats > noisy_score.d_optimal_gain_nats);
        assert!(
            precise_score.a_optimal_variance_reduction > noisy_score.a_optimal_variance_reduction
        );
    }

    #[test]
    fn fisher_validation_rejects_empty_nonsymmetric_and_non_spd_priors() {
        let candidate = fisher_experiment("candidate", 1.0, vec![vec![1.0, 0.0]], vec![1.0]);
        assert_eq!(
            evaluate_fisher_information_candidate(&[], &candidate),
            Err(DesignError::EmptyParameterSet)
        );
        assert!(matches!(
            evaluate_fisher_information_candidate(&[vec![1.0, 0.5], vec![0.0, 1.0]], &candidate),
            Err(DesignError::NonSymmetricFisherPrior { .. })
        ));
        assert!(matches!(
            evaluate_fisher_information_candidate(&[vec![1.0, 2.0], vec![2.0, 1.0]], &candidate),
            Err(DesignError::FisherPriorNotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn fisher_validation_rejects_bad_sensitivity_and_variance_shapes() {
        let prior = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bad_rows = fisher_experiment("rows", 1.0, vec![vec![1.0, 0.0]], vec![]);
        assert!(matches!(
            evaluate_fisher_information_candidate(&prior, &bad_rows),
            Err(DesignError::FisherSensitivityRowCountMismatch { .. })
        ));
        let bad_width = fisher_experiment("width", 1.0, vec![vec![1.0]], vec![1.0]);
        assert!(matches!(
            evaluate_fisher_information_candidate(&prior, &bad_width),
            Err(DesignError::FisherSensitivityRowLengthMismatch { .. })
        ));
        let bad_variance = fisher_experiment("variance", 1.0, vec![vec![1.0, 0.0]], vec![0.0]);
        assert!(matches!(
            evaluate_fisher_information_candidate(&prior, &bad_variance),
            Err(DesignError::InvalidObservationVariance { .. })
        ));
        let bad_sensitivity = fisher_experiment("nan", 1.0, vec![vec![f64::NAN, 0.0]], vec![1.0]);
        assert!(matches!(
            evaluate_fisher_information_candidate(&prior, &bad_sensitivity),
            Err(DesignError::InvalidFisherSensitivity { .. })
        ));
    }

    #[test]
    fn fisher_ranking_is_cost_aware_and_deterministic() {
        let prior = vec![vec![1.0]];
        let expensive = fisher_experiment("expensive", 10.0, vec![vec![3.0]], vec![1.0]);
        let cheap = fisher_experiment("cheap", 1.0, vec![vec![1.0]], vec![1.0]);
        let ranked =
            rank_fisher_candidates(&prior, &[expensive, cheap], FisherObjective::DOptimal).unwrap();
        assert_eq!(ranked[0].experiment_id, "cheap");
    }

    fn pairwise_and_triplet_response(stimulus_mask: u8, interaction_gain: f64, state: bool) -> f64 {
        let x1 = if stimulus_mask & 1 != 0 { 1.0 } else { 0.0 };
        let x2 = if stimulus_mask & 2 != 0 { 1.0 } else { 0.0 };
        let x3 = if stimulus_mask & 4 != 0 { 1.0 } else { 0.0 };
        x1 + 2.0 * x2 + 3.0 * x3 + 0.5 * x1 * x2 - 0.25 * x1 * x3
            + 0.75 * x2 * x3
            + if state {
                interaction_gain * x1 * x2 * x3
            } else {
                0.0
            }
    }

    fn deterministic_pairwise_vs_triplet_experiment(
        id: &str,
        stimulus_mask: u8,
        state: bool,
    ) -> CandidateExperiment {
        let pairwise = pairwise_and_triplet_response(stimulus_mask, 0.0, state);
        let triplet = pairwise_and_triplet_response(stimulus_mask, 2.0, state);
        let cost = f64::from(stimulus_mask.count_ones().max(1));
        if (pairwise - triplet).abs() <= f64::EPSILON {
            CandidateExperiment {
                id: id.to_owned(),
                cost,
                outcomes: vec!["same_response".to_owned()],
                likelihoods_by_hypothesis: vec![vec![1.0], vec![1.0]],
            }
        } else {
            CandidateExperiment {
                id: id.to_owned(),
                cost,
                outcomes: vec![
                    "pairwise_prediction".to_owned(),
                    "triplet_prediction".to_owned(),
                ],
                likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            }
        }
    }

    #[test]
    fn single_and_pairwise_stimuli_cannot_recover_a_pure_three_way_interaction() {
        let priors = [0.5, 0.5];
        let candidates: Vec<CandidateExperiment> = (1_u8..=6)
            .map(|mask| {
                deterministic_pairwise_vs_triplet_experiment(
                    &format!("subset-{mask:03b}"),
                    mask,
                    true,
                )
            })
            .collect();
        for candidate in &candidates {
            let score = evaluate_candidate(&priors, candidate).unwrap();
            close(score.information_gain_bits, 0.0, 1.0e-12);
        }

        let triple = deterministic_pairwise_vs_triplet_experiment("triple-111", 0b111, true);
        let score = evaluate_candidate(&priors, &triple).unwrap();
        close(score.information_gain_bits, 1.0, 1.0e-12);

        let mut all_candidates = candidates;
        all_candidates.push(triple);
        let ranked = rank_candidates(&priors, &all_candidates).unwrap();
        assert_eq!(ranked[0].experiment_id, "triple-111");
    }

    #[test]
    fn three_way_interaction_is_not_identifiable_outside_its_gating_state() {
        let priors = [0.5, 0.5];
        let inactive =
            deterministic_pairwise_vs_triplet_experiment("triple-state-off", 0b111, false);
        let active = deterministic_pairwise_vs_triplet_experiment("triple-state-on", 0b111, true);
        close(
            evaluate_candidate(&priors, &inactive)
                .unwrap()
                .information_gain_bits,
            0.0,
            1.0e-12,
        );
        close(
            evaluate_candidate(&priors, &active)
                .unwrap()
                .information_gain_bits,
            1.0,
            1.0e-12,
        );
    }

    #[test]
    fn eig_and_decision_value_are_distinct_experiment_objectives() {
        let priors = [0.25, 0.25, 0.25, 0.25];
        let problem = DecisionProblem {
            actions: vec!["choose_group_a".into(), "choose_group_b".into()],
            utilities_by_hypothesis: vec![
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
            ],
        };

        // This experiment reveals which member of each utility-equivalent
        // pair is present, but never reveals which decision group is correct.
        let within_group = CandidateExperiment {
            id: "within-group-identity".into(),
            cost: 1.0,
            outcomes: vec!["member_0".into(), "member_1".into()],
            likelihoods_by_hypothesis: vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
        };

        // This noisy experiment identifies the utility-relevant group with
        // 80% probability but has less total information about H.
        let group_signal = CandidateExperiment {
            id: "decision-relevant-group".into(),
            cost: 1.0,
            outcomes: vec!["group_a".into(), "group_b".into()],
            likelihoods_by_hypothesis: vec![
                vec![0.8, 0.2],
                vec![0.8, 0.2],
                vec![0.2, 0.8],
                vec![0.2, 0.8],
            ],
        };

        let eig_ranked =
            rank_candidates(&priors, &[within_group.clone(), group_signal.clone()]).unwrap();
        assert_eq!(eig_ranked[0].experiment_id, "within-group-identity");
        close(eig_ranked[0].score.information_gain_bits, 1.0, 1.0e-12);

        let within_score = evaluate_decision_value(&priors, &within_group, &problem).unwrap();
        let group_score = evaluate_decision_value(&priors, &group_signal, &problem).unwrap();
        close(
            within_score.expected_value_of_sample_information,
            0.0,
            1.0e-12,
        );
        close(
            group_score.expected_value_of_sample_information,
            0.3,
            1.0e-12,
        );

        let decision_ranked = rank_candidates_by_decision_value(
            &priors,
            &[within_group.clone(), group_signal.clone()],
            &problem,
        )
        .unwrap();
        assert_eq!(decision_ranked[0].experiment_id, "decision-relevant-group");

        let regret = decision_selection_regret(
            &priors,
            &[within_group, group_signal],
            &problem,
            "within-group-identity",
        )
        .unwrap();
        assert_eq!(regret.best_experiment_id, "decision-relevant-group");
        close(regret.regret_per_cost, 0.3, 1.0e-12);
    }

    #[test]
    fn empty_and_duplicate_decision_action_labels_reject() {
        let candidate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        let empty = DecisionProblem {
            actions: vec![" ".into()],
            utilities_by_hypothesis: vec![vec![1.0], vec![0.0]],
        };
        assert_eq!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &empty),
            Err(DesignError::EmptyActionLabel)
        );

        let duplicate = DecisionProblem {
            actions: vec!["same".into(), "same".into()],
            utilities_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        assert!(matches!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &duplicate),
            Err(DesignError::DuplicateActionLabel { .. })
        ));
    }

    #[test]
    fn cost_normalized_scores_reject_overflow() {
        let tiny_cost = f64::from_bits(1);
        let candidate = binary_experiment("tiny-cost", tiny_cost, 0.9, 0.1);
        assert_eq!(
            evaluate_candidate(&[0.5, 0.5], &candidate),
            Err(DesignError::NumericalInstability)
        );

        let problem = DecisionProblem {
            actions: vec!["a".into(), "b".into()],
            utilities_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        assert_eq!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &problem),
            Err(DesignError::NumericalInstability)
        );
    }

    #[test]
    fn invalid_decision_utility_shape_and_non_finite_values_reject() {
        let candidate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        let bad_shape = DecisionProblem {
            actions: vec!["a".into(), "b".into()],
            utilities_by_hypothesis: vec![vec![1.0, 0.0]],
        };
        assert!(matches!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &bad_shape),
            Err(DesignError::DecisionUtilityRowCountMismatch { .. })
        ));

        let bad_value = DecisionProblem {
            actions: vec!["a".into()],
            utilities_by_hypothesis: vec![vec![f64::NAN], vec![0.0]],
        };
        assert!(matches!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &bad_value),
            Err(DesignError::InvalidDecisionUtility { .. })
        ));
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
        close(
            score.information_gain_bits,
            0.531_004_406_410_718_8,
            1.0e-12,
        );
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
        close(
            score.information_gain_bits,
            score.prior_entropy_bits,
            1.0e-12,
        );
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
            first
                .iter()
                .map(|item| &item.experiment_id)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|item| &item.experiment_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(first[0].experiment_id, "alpha");
    }

    #[test]
    fn selection_regret_is_zero_for_best_candidate_and_positive_otherwise() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let best =
            selection_regret(&[0.5, 0.5], &[expensive.clone(), cheap.clone()], "cheap").unwrap();
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
            Err(DesignError::LikelihoodMassNotOne {
                hypothesis_index: 0,
                ..
            })
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
