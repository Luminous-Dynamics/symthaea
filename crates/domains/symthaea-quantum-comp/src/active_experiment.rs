//! Decision-focused active experiment selection.
//!
//! This module ranks measurements by expected downstream decision improvement
//! per unit measurement cost (EVSI/cost), not by information gain alone. The
//! model is finite and discrete: hypotheses have a prior, actions have a
//! declared utility in each hypothesis, and experiments define outcome
//! likelihoods conditioned on each hypothesis.
//!
//! This is an experiment-planning primitive, not a hardware controller or a
//! replacement for a calibrated physical likelihood model. Simulated
//! likelihoods must retain their provenance outside this mathematical core.

use std::error::Error;
use std::fmt;

const PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-9;
const NEGATIVE_ROUNDOFF_TOLERANCE: f64 = 1.0e-10;

/// A candidate experiment with explicitly conditioned outcome likelihoods.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionExperiment {
    id: String,
    measurement_cost: f64,
    outcomes: Vec<String>,
    /// Rows are hypotheses; columns are outcomes.
    likelihoods_by_hypothesis: Vec<Vec<f64>>,
}

impl DecisionExperiment {
    /// Construct and validate a candidate experiment.
    ///
    /// likelihoods_by_hypothesis[h][o] is P(outcome o | hypothesis h).
    /// Each row must sum to one within a tight floating-point tolerance.
    pub fn new(
        id: impl Into<String>,
        measurement_cost: f64,
        outcomes: Vec<String>,
        likelihoods_by_hypothesis: Vec<Vec<f64>>,
    ) -> Result<Self, ExperimentDesignError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ExperimentDesignError::EmptyIdentifier);
        }
        if !measurement_cost.is_finite() || measurement_cost <= 0.0 {
            return Err(ExperimentDesignError::InvalidMeasurementCost);
        }
        validate_unique_nonempty_labels(&outcomes)?;
        if likelihoods_by_hypothesis.is_empty() {
            return Err(ExperimentDesignError::EmptyLikelihoodMatrix);
        }

        for row in &likelihoods_by_hypothesis {
            if row.len() != outcomes.len() {
                return Err(ExperimentDesignError::LikelihoodDimensionMismatch);
            }
            validate_probability_distribution(row)?;
        }

        Ok(Self {
            id,
            measurement_cost,
            outcomes,
            likelihoods_by_hypothesis,
        })
    }

    /// Stable experiment identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Declared cost in the problem's common measurement-cost unit.
    pub fn measurement_cost(&self) -> f64 {
        self.measurement_cost
    }

    /// Declared outcome labels, in likelihood-matrix column order.
    pub fn outcomes(&self) -> &[String] {
        &self.outcomes
    }

    /// Per-hypothesis outcome probabilities, in hypothesis and outcome order.
    pub fn likelihoods_by_hypothesis(&self) -> &[Vec<f64>] {
        &self.likelihoods_by_hypothesis
    }
}

/// A finite decision problem and the experiments available to resolve it.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveExperimentProblem {
    hypotheses: Vec<String>,
    prior: Vec<f64>,
    actions: Vec<String>,
    /// Rows are actions; columns are hypotheses. Higher utility is better.
    utilities_by_action_and_hypothesis: Vec<Vec<f64>>,
    utility_unit: String,
    measurement_cost_unit: String,
    experiments: Vec<DecisionExperiment>,
}

impl ActiveExperimentProblem {
    /// Construct a validated decision-focused experiment problem.
    ///
    /// Utility must encode one predeclared downstream objective. Use separate
    /// problem instances for incomparable objectives such as time, energy,
    /// monetary cost, or scientific quality; do not silently collapse them
    /// into a private aggregate score.
    pub fn new(
        hypotheses: Vec<String>,
        prior: Vec<f64>,
        actions: Vec<String>,
        utilities_by_action_and_hypothesis: Vec<Vec<f64>>,
        utility_unit: impl Into<String>,
        measurement_cost_unit: impl Into<String>,
        experiments: Vec<DecisionExperiment>,
    ) -> Result<Self, ExperimentDesignError> {
        validate_unique_nonempty_labels(&hypotheses)?;
        validate_unique_nonempty_labels(&actions)?;
        let utility_unit = utility_unit.into();
        let measurement_cost_unit = measurement_cost_unit.into();
        if utility_unit.trim().is_empty() || measurement_cost_unit.trim().is_empty() {
            return Err(ExperimentDesignError::EmptyUnit);
        }
        if prior.len() != hypotheses.len() {
            return Err(ExperimentDesignError::PriorDimensionMismatch);
        }
        validate_probability_distribution(&prior)?;

        if utilities_by_action_and_hypothesis.len() != actions.len()
            || utilities_by_action_and_hypothesis
                .iter()
                .any(|row| row.len() != hypotheses.len())
        {
            return Err(ExperimentDesignError::UtilityDimensionMismatch);
        }
        if utilities_by_action_and_hypothesis
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(ExperimentDesignError::NonFiniteUtility);
        }
        if experiments.is_empty() {
            return Err(ExperimentDesignError::NoExperiments);
        }
        validate_unique_nonempty_labels(
            &experiments
                .iter()
                .map(|experiment| experiment.id.clone())
                .collect::<Vec<_>>(),
        )?;
        for experiment in &experiments {
            if experiment.likelihoods_by_hypothesis.len() != hypotheses.len() {
                return Err(ExperimentDesignError::LikelihoodDimensionMismatch);
            }
        }

        Ok(Self {
            hypotheses,
            prior,
            actions,
            utilities_by_action_and_hypothesis,
            utility_unit,
            measurement_cost_unit,
            experiments,
        })
    }

    /// Hypothesis labels in prior and utility-matrix column order.
    pub fn hypotheses(&self) -> &[String] {
        &self.hypotheses
    }

    /// Declared prior over hypotheses.
    pub fn prior(&self) -> &[f64] {
        &self.prior
    }

    /// Available decision actions in utility-matrix row order.
    pub fn actions(&self) -> &[String] {
        &self.actions
    }

    /// Action utilities, with rows as actions and columns as hypotheses.
    pub fn utilities_by_action_and_hypothesis(&self) -> &[Vec<f64>] {
        &self.utilities_by_action_and_hypothesis
    }

    /// Declared unit of action utility.
    pub fn utility_unit(&self) -> &str {
        &self.utility_unit
    }

    /// Declared measurement-cost unit shared by candidate experiments.
    pub fn measurement_cost_unit(&self) -> &str {
        &self.measurement_cost_unit
    }

    /// Candidate experiments in the order supplied to the constructor.
    pub fn experiments(&self) -> &[DecisionExperiment] {
        &self.experiments
    }

    /// Rank experiments from highest to lowest decision improvement per cost.
    pub fn rank_experiments(&self) -> Result<ExperimentRanking, ExperimentDesignError> {
        let (baseline_action, baseline_utility) = self.best_action_for_weights(&self.prior)?;
        let mut scores = Vec::with_capacity(self.experiments.len());

        for experiment in &self.experiments {
            let mut expected_utility_after_observation = 0.0;
            let mut outcome_probabilities = Vec::with_capacity(experiment.outcomes.len());
            let mut expected_posterior_entropy_bits = 0.0;

            for (outcome_index, outcome) in experiment.outcomes.iter().enumerate() {
                let unnormalized_posterior = self
                    .prior
                    .iter()
                    .enumerate()
                    .map(|(hypothesis_index, prior_probability)| {
                        prior_probability
                            * experiment.likelihoods_by_hypothesis[hypothesis_index]
                                [outcome_index]
                    })
                    .collect::<Vec<_>>();
                let outcome_probability = unnormalized_posterior.iter().sum::<f64>();
                if !outcome_probability.is_finite()
                    || !(-PROBABILITY_SUM_TOLERANCE..=1.0 + PROBABILITY_SUM_TOLERANCE)
                        .contains(&outcome_probability)
                {
                    return Err(ExperimentDesignError::InvalidDerivedProbability);
                }
                let outcome_probability = outcome_probability.clamp(0.0, 1.0);
                outcome_probabilities.push(OutcomeProbability {
                    outcome: outcome.clone(),
                    probability: outcome_probability,
                });
                if outcome_probability == 0.0 {
                    continue;
                }

                // P(outcome) * best posterior expected utility equals the
                // maximum unnormalized utility across actions. This avoids
                // division by small outcome probabilities in the decision score.
                expected_utility_after_observation +=
                    self.best_utility_for_weights(&unnormalized_posterior)?;

                let posterior = unnormalized_posterior
                    .iter()
                    .map(|weight| weight / outcome_probability)
                    .collect::<Vec<_>>();
                validate_probability_distribution(&posterior)?;
                expected_posterior_entropy_bits +=
                    outcome_probability * shannon_entropy_bits(&posterior);
            }

            let raw_evsi = expected_utility_after_observation - baseline_utility;
            let utility_scale = 1.0_f64
                .max(baseline_utility.abs())
                .max(expected_utility_after_observation.abs());
            if raw_evsi < -NEGATIVE_ROUNDOFF_TOLERANCE * utility_scale {
                return Err(ExperimentDesignError::NegativeDerivedValueOfInformation);
            }
            let expected_value_of_sample_information = raw_evsi.max(0.0);
            let expected_information_gain_bits =
                (shannon_entropy_bits(&self.prior) - expected_posterior_entropy_bits).max(0.0);
            let value_per_cost =
                expected_value_of_sample_information / experiment.measurement_cost;
            let information_gain_per_cost =
                expected_information_gain_bits / experiment.measurement_cost;
            if !value_per_cost.is_finite() || !information_gain_per_cost.is_finite() {
                return Err(ExperimentDesignError::DerivedValueOverflow);
            }

            scores.push(ExperimentScore {
                experiment_id: experiment.id.clone(),
                measurement_cost: experiment.measurement_cost,
                expected_value_of_sample_information,
                expected_information_gain_bits,
                value_per_cost,
                information_gain_per_cost,
                outcome_probabilities,
                expected_utility_after_observation,
            });
        }

        scores.sort_by(|left, right| {
            right
                .value_per_cost
                .total_cmp(&left.value_per_cost)
                .then_with(|| left.experiment_id.cmp(&right.experiment_id))
        });

        Ok(ExperimentRanking {
            baseline_best_action: baseline_action,
            baseline_best_expected_utility: baseline_utility,
            utility_unit: self.utility_unit.clone(),
            measurement_cost_unit: self.measurement_cost_unit.clone(),
            scores,
        })
    }

    fn best_action_for_weights(
        &self,
        weights: &[f64],
    ) -> Result<(String, f64), ExperimentDesignError> {
        let mut best_action: Option<&str> = None;
        let mut best_value = f64::NEG_INFINITY;

        for (action, utilities) in self.actions.iter().zip(&self.utilities_by_action_and_hypothesis) {
            let value = weighted_utility(utilities, weights)?;
            if value > best_value
                || (value == best_value
                    && best_action.is_none_or(|previous| action.as_str() < previous))
            {
                best_action = Some(action);
                best_value = value;
            }
        }

        match best_action {
            Some(action) => Ok((action.to_string(), best_value)),
            None => Err(ExperimentDesignError::NoActions),
        }
    }

    fn best_utility_for_weights(&self, weights: &[f64]) -> Result<f64, ExperimentDesignError> {
        self.actions
            .iter()
            .zip(&self.utilities_by_action_and_hypothesis)
            .try_fold(f64::NEG_INFINITY, |best, (_, utilities)| {
                Ok::<f64, ExperimentDesignError>(
                    best.max(weighted_utility(utilities, weights)?),
                )
            })
    }
}

/// Probability of observing one named outcome before conducting the experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeProbability {
    /// Outcome name.
    pub outcome: String,
    /// Prior-predictive probability of this outcome.
    pub probability: f64,
}

/// Scores and diagnostics for a single candidate experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentScore {
    /// Stable experiment identifier.
    pub experiment_id: String,
    /// Measurement cost in the declared common cost unit.
    pub measurement_cost: f64,
    /// Expected improvement in the downstream decision objective.
    pub expected_value_of_sample_information: f64,
    /// Expected entropy reduction in bits; diagnostic only.
    pub expected_information_gain_bits: f64,
    /// EVSI divided by measurement cost; the primary ranking score.
    pub value_per_cost: f64,
    /// Information gain divided by measurement cost; diagnostic comparison.
    pub information_gain_per_cost: f64,
    /// Marginal probability of each declared outcome.
    pub outcome_probabilities: Vec<OutcomeProbability>,
    /// Expected optimal action utility after observing the result.
    pub expected_utility_after_observation: f64,
}

/// A ranked list and the decision possible before any further experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentRanking {
    /// Best action under the current prior, with deterministic lexical tie-break.
    pub baseline_best_action: String,
    /// Expected utility of the best action under the current prior.
    pub baseline_best_expected_utility: f64,
    /// Declared utility unit.
    pub utility_unit: String,
    /// Declared measurement-cost unit.
    pub measurement_cost_unit: String,
    /// Experiments ranked by EVSI/cost, descending, with lexical tie-break.
    pub scores: Vec<ExperimentScore>,
}

/// Validation and calculation errors in decision-focused experiment design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentDesignError {
    /// Identifier or label is empty, blank, or duplicated.
    EmptyIdentifier,
    /// A unit string was empty.
    EmptyUnit,
    /// A probability is outside [0, 1], non-finite, or its distribution does not sum to one.
    InvalidProbabilityDistribution,
    /// A probability vector was expected to be non-empty.
    EmptyProbabilityDistribution,
    /// The prior length differs from the hypothesis count.
    PriorDimensionMismatch,
    /// The utility matrix does not match actions by hypotheses.
    UtilityDimensionMismatch,
    /// A likelihood matrix does not match outcomes by hypotheses.
    LikelihoodDimensionMismatch,
    /// A likelihood matrix contains no hypothesis rows.
    EmptyLikelihoodMatrix,
    /// There are no candidate experiments.
    NoExperiments,
    /// There are no candidate actions.
    NoActions,
    /// A measurement cost must be positive and finite.
    InvalidMeasurementCost,
    /// An action utility contains a non-finite number.
    NonFiniteUtility,
    /// The computed probability of an outcome was invalid.
    InvalidDerivedProbability,
    /// Floating-point arithmetic materially violated non-negative EVSI.
    NegativeDerivedValueOfInformation,
    /// A derived score overflowed.
    DerivedValueOverflow,
}

impl fmt::Display for ExperimentDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::EmptyIdentifier => "labels and identifiers must be unique and non-empty",
            Self::EmptyUnit => "utility and measurement-cost units must be non-empty",
            Self::InvalidProbabilityDistribution => {
                "probabilities must be finite, lie in [0, 1], and sum to one"
            }
            Self::EmptyProbabilityDistribution => "a probability distribution cannot be empty",
            Self::PriorDimensionMismatch => "prior length must equal hypothesis count",
            Self::UtilityDimensionMismatch => "utility matrix must match actions by hypotheses",
            Self::LikelihoodDimensionMismatch => {
                "likelihood matrix must match hypotheses by outcomes"
            }
            Self::EmptyLikelihoodMatrix => "likelihood matrix must contain hypothesis rows",
            Self::NoExperiments => "at least one candidate experiment is required",
            Self::NoActions => "at least one candidate action is required",
            Self::InvalidMeasurementCost => "measurement cost must be positive and finite",
            Self::NonFiniteUtility => "action utilities must be finite",
            Self::InvalidDerivedProbability => "derived outcome probability is invalid",
            Self::NegativeDerivedValueOfInformation => {
                "expected value of sample information became materially negative"
            }
            Self::DerivedValueOverflow => "derived experiment score overflowed",
        };
        f.write_str(description)
    }
}

impl Error for ExperimentDesignError {}

fn validate_unique_nonempty_labels(labels: &[String]) -> Result<(), ExperimentDesignError> {
    if labels.is_empty() {
        return Err(ExperimentDesignError::EmptyIdentifier);
    }
    for (index, label) in labels.iter().enumerate() {
        if label.trim().is_empty() || labels[..index].iter().any(|previous| previous == label) {
            return Err(ExperimentDesignError::EmptyIdentifier);
        }
    }
    Ok(())
}

fn validate_probability_distribution(values: &[f64]) -> Result<(), ExperimentDesignError> {
    if values.is_empty() {
        return Err(ExperimentDesignError::EmptyProbabilityDistribution);
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(ExperimentDesignError::InvalidProbabilityDistribution);
    }
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(ExperimentDesignError::InvalidProbabilityDistribution);
    }
    Ok(())
}

fn weighted_utility(utilities: &[f64], weights: &[f64]) -> Result<f64, ExperimentDesignError> {
    if utilities.len() != weights.len() {
        return Err(ExperimentDesignError::UtilityDimensionMismatch);
    }
    let result = utilities
        .iter()
        .zip(weights)
        .map(|(utility, weight)| utility * weight)
        .sum::<f64>();
    if !result.is_finite() {
        return Err(ExperimentDesignError::DerivedValueOverflow);
    }
    Ok(result)
}

fn shannon_entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn experiment(
        id: &str,
        cost: f64,
        outcomes: &[&str],
        likelihoods: &[&[f64]],
    ) -> DecisionExperiment {
        DecisionExperiment::new(
            id,
            cost,
            outcomes.iter().map(|outcome| (*outcome).to_string()).collect(),
            likelihoods.iter().map(|row| row.to_vec()).collect(),
        )
        .unwrap()
    }

    fn problem(
        experiments: Vec<DecisionExperiment>,
        utilities: Vec<Vec<f64>>,
    ) -> ActiveExperimentProblem {
        ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-A".into(), "choose-B".into()],
            utilities,
            "utility_points",
            "measurement_seconds",
            experiments,
        )
        .unwrap()
    }

    fn decision_utilities() -> Vec<Vec<f64>> {
        vec![vec![1.0, 0.0], vec![0.0, 1.0]]
    }

    #[test]
    fn perfect_decision_experiment_has_hand_computable_evsi() {
        let candidate = experiment("perfect", 1.0, &["a", "b"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let ranking = problem(vec![candidate], decision_utilities())
            .rank_experiments()
            .unwrap();
        let score = &ranking.scores[0];
        assert_eq!(ranking.baseline_best_expected_utility, 0.5);
        assert_eq!(score.expected_value_of_sample_information, 0.5);
        assert_eq!(score.expected_information_gain_bits, 1.0);
        assert_eq!(score.value_per_cost, 0.5);
        assert!((score.outcome_probabilities[0].probability - 0.5).abs() < 1e-12);
        assert!((score.outcome_probabilities[1].probability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn informative_but_decision_irrelevant_test_has_zero_evsi() {
        let candidate = experiment("irrelevant", 1.0, &["h0", "h1"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let same_action_wins = vec![vec![1.0, 1.0], vec![0.0, 0.0]];
        let ranking = problem(vec![candidate], same_action_wins)
            .rank_experiments()
            .unwrap();
        let score = &ranking.scores[0];
        assert_eq!(score.expected_value_of_sample_information, 0.0);
        assert_eq!(score.value_per_cost, 0.0);
        assert_eq!(score.expected_information_gain_bits, 1.0);
    }

    #[test]
    fn rank_by_decision_value_per_cost_not_information_gain_alone() {
        let perfect_cheap = experiment("a-cheap-perfect", 1.0, &["x", "y"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let perfect_expensive =
            experiment("z-expensive-perfect", 10.0, &["x", "y"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let irrelevant = experiment("b-irrelevant", 0.2, &["yes", "no"], &[&[0.5, 0.5], &[0.5, 0.5]]);
        let ranking = problem(
            vec![perfect_expensive, irrelevant, perfect_cheap],
            decision_utilities(),
        )
        .rank_experiments()
        .unwrap();
        assert_eq!(ranking.scores[0].experiment_id, "a-cheap-perfect");
        assert_eq!(ranking.scores[1].experiment_id, "z-expensive-perfect");
        assert_eq!(ranking.scores[2].experiment_id, "b-irrelevant");
    }

    #[test]
    fn uninformative_test_has_zero_gain_and_preserves_prior_decision() {
        let candidate = experiment("uninformative", 2.0, &["yes", "no"], &[&[0.4, 0.6], &[0.4, 0.6]]);
        let ranking = problem(vec![candidate], decision_utilities())
            .rank_experiments()
            .unwrap();
        assert_eq!(ranking.scores[0].expected_value_of_sample_information, 0.0);
        assert!(ranking.scores[0].expected_information_gain_bits.abs() < 1e-12);
        assert_eq!(ranking.baseline_best_action, "choose-A");
    }

    #[test]
    fn outcome_probabilities_sum_to_one() {
        let candidate = experiment("noisy", 1.0, &["0", "1", "2"], &[&[0.6, 0.3, 0.1], &[0.1, 0.3, 0.6]]);
        let ranking = problem(vec![candidate], decision_utilities())
            .rank_experiments()
            .unwrap();
        let sum: f64 = ranking.scores[0]
            .outcome_probabilities
            .iter()
            .map(|outcome| outcome.probability)
            .sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_probability_outcomes_are_safe() {
        let candidate = experiment("sparse", 1.0, &["certain", "impossible"], &[&[1.0, 0.0], &[1.0, 0.0]]);
        let ranking = problem(vec![candidate], decision_utilities())
            .rank_experiments()
            .unwrap();
        assert_eq!(ranking.scores[0].outcome_probabilities[1].probability, 0.0);
        assert!(ranking.scores[0].expected_value_of_sample_information.abs() < 1e-12);
    }

    #[test]
    fn all_negative_utilities_are_supported() {
        let candidate = experiment("perfect", 1.0, &["h0", "h1"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let utilities = vec![vec![-1.0, -5.0], vec![-5.0, -1.0]];
        let ranking = problem(vec![candidate], utilities).rank_experiments().unwrap();
        assert!((ranking.scores[0].expected_value_of_sample_information - 2.0).abs() < 1e-12);
    }

    #[test]
    fn duplicate_experiment_ids_are_rejected() {
        let first = experiment("same", 1.0, &["a", "b"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let second = first.clone();
        assert_eq!(
            ActiveExperimentProblem::new(
                vec!["H0".into(), "H1".into()],
                vec![0.5, 0.5],
                vec!["A".into()],
                vec![vec![1.0, 1.0]],
                "quality",
                "seconds",
                vec![first, second],
            ),
            Err(ExperimentDesignError::EmptyIdentifier)
        );
    }

    #[test]
    fn malformed_probability_distributions_are_rejected() {
        assert_eq!(
            DecisionExperiment::new(
                "bad-row",
                1.0,
                vec!["x".into(), "y".into()],
                vec![vec![0.3, 0.3], vec![0.5, 0.5]],
            ),
            Err(ExperimentDesignError::InvalidProbabilityDistribution)
        );
        assert_eq!(
            ActiveExperimentProblem::new(
                vec!["H0".into(), "H1".into()],
                vec![0.4, 0.4],
                vec!["A".into()],
                vec![vec![0.0, 0.0]],
                "quality",
                "seconds",
                vec![experiment("ok", 1.0, &["a", "b"], &[&[0.5, 0.5], &[0.5, 0.5]])],
            ),
            Err(ExperimentDesignError::InvalidProbabilityDistribution)
        );
    }

    #[test]
    fn mismatched_dimensions_and_invalid_costs_fail_closed() {
        assert_eq!(
            DecisionExperiment::new("bad-cost", 0.0, vec!["x".into()], vec![vec![1.0]]),
            Err(ExperimentDesignError::InvalidMeasurementCost)
        );
        let candidate = experiment("ok", 1.0, &["a", "b"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        assert_eq!(
            ActiveExperimentProblem::new(
                vec!["H0".into(), "H1".into()],
                vec![0.5, 0.5],
                vec!["A".into()],
                vec![vec![1.0]],
                "quality",
                "seconds",
                vec![candidate],
            ),
            Err(ExperimentDesignError::UtilityDimensionMismatch)
        );
    }

    #[test]
    fn tied_experiment_scores_have_deterministic_lexical_order() {
        let second = experiment("z", 1.0, &["x", "y"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let first = experiment("a", 1.0, &["x", "y"], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let ranking = problem(vec![second, first], decision_utilities())
            .rank_experiments()
            .unwrap();
        assert_eq!(ranking.scores[0].experiment_id, "a");
        assert_eq!(ranking.scores[1].experiment_id, "z");
    }

    #[test]
    fn non_finite_utilities_and_empty_units_are_rejected() {
        assert_eq!(
            ActiveExperimentProblem::new(
                vec!["H0".into()],
                vec![1.0],
                vec!["A".into()],
                vec![vec![f64::INFINITY]],
                "quality",
                "seconds",
                vec![experiment("ok", 1.0, &["yes"], &[&[1.0]])],
            ),
            Err(ExperimentDesignError::NonFiniteUtility)
        );
        assert_eq!(
            ActiveExperimentProblem::new(
                vec!["H0".into()],
                vec![1.0],
                vec!["A".into()],
                vec![vec![1.0]],
                " ",
                "seconds",
                vec![experiment("ok", 1.0, &["yes"], &[&[1.0]])],
            ),
            Err(ExperimentDesignError::EmptyUnit)
        );
    }
}
