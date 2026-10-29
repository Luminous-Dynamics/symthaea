//! Exact finite bake-off for experiment-selection policies.
//!
//! This harness measures downstream decision utility under a held-out,
//! declared ground-truth decision model. It compares nominal EVSI, nominal
//! information gain, robust worst-case EVSI, uniform random choice (evaluated
//! exactly, not by Monte Carlo), and a fixed-first baseline. All policies use
//! the same equal-weight model-averaged posterior to choose an action after an
//! observation, so the comparison isolates the experiment-selection policy.
//!
//! This is a decision-planner benchmark, not quantum hardware benchmarking.
//! The ground-truth model is known only to the evaluator and must never be
//! presented as information available to a deployed selector.

use crate::active_experiment::{ActiveExperimentProblem, ExperimentDesignError};
use crate::robust_experiment::{
    RobustExperimentError, RobustModelScenario, rank_robust_experiments,
};
use std::error::Error;
use std::fmt;

const PROBABILITY_TOLERANCE: f64 = 1.0e-10;
const UTILITY_TOLERANCE: f64 = 1.0e-9;

/// The experiment-selection strategy being benchmarked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentSelectionPolicy {
    /// Select by nominal expected value of sample information per cost.
    NominalEvsiPerCost,
    /// Select by nominal Shannon information gain per cost.
    NominalInformationGainPerCost,
    /// Select by worst-case EVSI per cost across the declared model ensemble.
    RobustWorstCaseEvsiPerCost,
    /// Uniform random selection, evaluated as its exact expectation.
    UniformRandom,
    /// Select the first experiment in the nominal problem's declared order.
    FixedFirst,
}

/// True expected result of using the declared planner posterior for one experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentTruthEvaluation {
    /// Candidate experiment ID.
    pub experiment_id: String,
    /// Resource cost for this one-measurement evaluation.
    pub measurement_cost: f64,
    /// Expected downstream action utility under held-out ground truth.
    pub expected_decision_utility: f64,
    /// Probability mass of true outcomes that no planning model supports.
    pub unsupported_outcome_probability: f64,
    /// Expected utility if the ground-truth model itself chose each action.
    /// This field is a reference for the evaluator, not a deployable policy.
    pub ground_truth_posterior_utility: f64,
}

/// Policy-level expected result under the held-out ground-truth model.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentPolicyEvaluation {
    /// Selection policy.
    pub policy: ExperimentSelectionPolicy,
    /// Selected experiment; None for the exact uniform-random expectation.
    pub selected_experiment_id: Option<String>,
    /// Expected decision utility under held-out ground truth.
    pub expected_decision_utility: f64,
    /// Oracle utility with the choice of experiment and posterior decision rule
    /// made using the ground-truth model.
    pub oracle_expected_utility: f64,
    /// Non-negative gap from the oracle reference.
    pub decision_regret: f64,
    /// Expected measurement cost. Costs are not folded into utility.
    pub expected_measurement_cost: f64,
    /// Probability mass unsupported by every planning model.
    pub unsupported_outcome_probability: f64,
}

/// Exact expected-utility bake-off report.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentPolicyBakeoffReport {
    /// Name/version of the evaluator's held-out ground-truth model.
    pub ground_truth_model_id: String,
    /// Declared planning scenarios.
    pub planning_scenario_ids: Vec<String>,
    /// Nominal scenario used by nominal policies and fixed-first ordering.
    pub nominal_scenario_id: String,
    /// Decision utility units.
    pub utility_unit: String,
    /// Measurement cost units.
    pub measurement_cost_unit: String,
    /// Expected utility of taking no measurement and choosing the best prior action.
    pub no_measurement_expected_utility: f64,
    /// Oracle expected utility, taking the best of no measurement or one experiment.
    pub oracle_expected_utility: f64,
    /// Experiment attaining oracle utility, if measuring is weakly better than abstaining.
    pub oracle_experiment_id: Option<String>,
    /// Held-out truth value for every candidate experiment.
    pub candidate_results: Vec<ExperimentTruthEvaluation>,
    /// Expected outcomes for each policy.
    pub policy_results: Vec<ExperimentPolicyEvaluation>,
}

/// Contract or arithmetic errors in an experiment-selection bake-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentPolicyBenchmarkError {
    /// Ground-truth model ID is empty.
    EmptyGroundTruthModelId,
    /// Planning scenario ensemble fails QARCH-006 validation.
    InvalidPlanningEnsemble,
    /// Ground truth disagrees with planning hypotheses.
    GroundTruthHypothesisMismatch,
    /// Ground truth disagrees with planning actions.
    GroundTruthActionMismatch,
    /// Ground truth changes decision utility semantics.
    GroundTruthUtilityMismatch,
    /// Ground truth uses a different measurement-cost unit.
    GroundTruthCostUnitMismatch,
    /// Ground truth does not have the same candidate experiment IDs.
    GroundTruthCandidateMismatch,
    /// Candidate outcomes or declared measurement costs differ.
    GroundTruthCandidateContractMismatch,
    /// A candidate is missing or there are no candidates.
    MissingCandidate,
    /// An expected value or probability became invalid or non-finite.
    InvalidDerivedValue,
}

impl fmt::Display for ExperimentPolicyBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::EmptyGroundTruthModelId => "ground-truth model ID must be non-empty",
            Self::InvalidPlanningEnsemble => "planning model ensemble failed validation",
            Self::GroundTruthHypothesisMismatch => "ground truth has different hypotheses",
            Self::GroundTruthActionMismatch => "ground truth has different decision actions",
            Self::GroundTruthUtilityMismatch => "ground truth has different action utilities or units",
            Self::GroundTruthCostUnitMismatch => "ground truth has a different cost unit",
            Self::GroundTruthCandidateMismatch => "ground truth has a different candidate set",
            Self::GroundTruthCandidateContractMismatch => "candidate outcomes/costs differ from ground truth",
            Self::MissingCandidate => "candidate experiment is missing or candidate set is empty",
            Self::InvalidDerivedValue => "derived utility/regret/probability is invalid",
        };
        f.write_str(description)
    }
}

impl Error for ExperimentPolicyBenchmarkError {}

impl From<RobustExperimentError> for ExperimentPolicyBenchmarkError {
    fn from(_: RobustExperimentError) -> Self {
        Self::InvalidPlanningEnsemble
    }
}

impl From<ExperimentDesignError> for ExperimentPolicyBenchmarkError {
    fn from(_: ExperimentDesignError) -> Self {
        Self::InvalidPlanningEnsemble
    }
}

/// Evaluate all declared policies against a known evaluator-only ground-truth model.
///
/// Model compatibility is validated before any score is produced. The true
/// model can differ in its prior and/or likelihoods, but must share the same
/// hypothesis/action semantics, utility matrix and units, candidate IDs,
/// outcome order, and costs. If ground truth assigns positive mass to an
/// outcome unsupported by every planning model, the evaluator records that
/// mass and the planner uses its best prior action for that outcome.
pub fn run_experiment_policy_bakeoff(
    ground_truth_model_id: impl Into<String>,
    ground_truth: &ActiveExperimentProblem,
    planning_scenarios: &[RobustModelScenario],
) -> Result<ExperimentPolicyBakeoffReport, ExperimentPolicyBenchmarkError> {
    let ground_truth_model_id = ground_truth_model_id.into();
    if ground_truth_model_id.trim().is_empty() {
        return Err(ExperimentPolicyBenchmarkError::EmptyGroundTruthModelId);
    }
    if planning_scenarios.len() < 2 {
        return Err(ExperimentPolicyBenchmarkError::InvalidPlanningEnsemble);
    }

    // Reuse QARCH-006's validation and scoring contract for the ambiguity set.
    let robust_report = rank_robust_experiments(planning_scenarios.to_vec())?;
    let nominal = planning_scenarios
        .iter()
        .find(|scenario| scenario.is_nominal())
        .ok_or(ExperimentPolicyBenchmarkError::InvalidPlanningEnsemble)?;
    let reference = nominal.problem();
    validate_ground_truth_contract(reference, ground_truth)?;

    let no_measurement_action = best_action(ground_truth, ground_truth.prior())?;
    let no_measurement_expected_utility = expected_action_utility(
        ground_truth,
        no_measurement_action,
        ground_truth.prior(),
    )?;

    let mut candidate_results = Vec::new();
    for candidate in reference.experiments() {
        let nominal_posterior_result =
            evaluate_experiment(ground_truth, planning_scenarios, candidate.id())?;
        let oracle_result = evaluate_experiment(ground_truth, &[], candidate.id())?;
        candidate_results.push(ExperimentTruthEvaluation {
            experiment_id: candidate.id().to_string(),
            measurement_cost: nominal_posterior_result.measurement_cost,
            expected_decision_utility: nominal_posterior_result.expected_decision_utility,
            unsupported_outcome_probability:
                nominal_posterior_result.unsupported_outcome_probability,
            ground_truth_posterior_utility: oracle_result.expected_decision_utility,
        });
    }

    let mut oracle_expected_utility = no_measurement_expected_utility;
    let mut oracle_experiment_id: Option<String> = None;
    for result in &candidate_results {
        let oracle_value = result.ground_truth_posterior_utility;
        if oracle_value > oracle_expected_utility + UTILITY_TOLERANCE
            || ((oracle_value - oracle_expected_utility).abs() <= UTILITY_TOLERANCE
                && oracle_experiment_id
                    .as_ref()
                    .is_some_and(|current| result.experiment_id < *current))
        {
            oracle_expected_utility = oracle_value;
            oracle_experiment_id = Some(result.experiment_id.clone());
        }
    }

    let nominal_rank = reference.rank_experiments()?;
    let nominal_evsi_id = nominal_rank
        .scores
        .first()
        .ok_or(ExperimentPolicyBenchmarkError::MissingCandidate)?
        .experiment_id
        .clone();
    let nominal_information_id = nominal_rank
        .scores
        .iter()
        .min_by(|left, right| {
            right
                .information_gain_per_cost
                .total_cmp(&left.information_gain_per_cost)
                .then_with(|| left.experiment_id.cmp(&right.experiment_id))
        })
        .ok_or(ExperimentPolicyBenchmarkError::MissingCandidate)?
        .experiment_id
        .clone();
    let robust_experiment_id = robust_report
        .scores
        .first()
        .ok_or(ExperimentPolicyBenchmarkError::MissingCandidate)?
        .experiment_id
        .clone();
    let fixed_first_id = reference
        .experiments()
        .first()
        .ok_or(ExperimentPolicyBenchmarkError::MissingCandidate)?
        .id()
        .to_string();

    let selections = [
        (ExperimentSelectionPolicy::NominalEvsiPerCost, Some(nominal_evsi_id)),
        (
            ExperimentSelectionPolicy::NominalInformationGainPerCost,
            Some(nominal_information_id),
        ),
        (
            ExperimentSelectionPolicy::RobustWorstCaseEvsiPerCost,
            Some(robust_experiment_id),
        ),
        (ExperimentSelectionPolicy::FixedFirst, Some(fixed_first_id)),
    ];
    let mut policy_results = Vec::new();
    for (policy, selected) in selections {
        let candidate = candidate_results
            .iter()
            .find(|result| Some(result.experiment_id.as_str()) == selected.as_deref())
            .ok_or(ExperimentPolicyBenchmarkError::MissingCandidate)?;
        policy_results.push(policy_result(
            policy,
            selected,
            candidate.expected_decision_utility,
            candidate.measurement_cost,
            candidate.unsupported_outcome_probability,
            oracle_expected_utility,
        )?);
    }

    // Exact expectation over uniformly sampled candidate experiments avoids
    // Monte Carlo variance in the random baseline.
    let count = candidate_results.len() as f64;
    let random_utility = candidate_results
        .iter()
        .map(|result| result.expected_decision_utility)
        .sum::<f64>()
        / count;
    let random_cost = candidate_results
        .iter()
        .map(|result| result.measurement_cost)
        .sum::<f64>()
        / count;
    let random_unsupported = candidate_results
        .iter()
        .map(|result| result.unsupported_outcome_probability)
        .sum::<f64>()
        / count;
    policy_results.push(policy_result(
        ExperimentSelectionPolicy::UniformRandom,
        None,
        random_utility,
        random_cost,
        random_unsupported,
        oracle_expected_utility,
    )?);

    Ok(ExperimentPolicyBakeoffReport {
        ground_truth_model_id,
        planning_scenario_ids: planning_scenarios
            .iter()
            .map(|scenario| scenario.id().to_string())
            .collect(),
        nominal_scenario_id: nominal.id().to_string(),
        utility_unit: reference.utility_unit().to_string(),
        measurement_cost_unit: reference.measurement_cost_unit().to_string(),
        no_measurement_expected_utility,
        oracle_expected_utility,
        oracle_experiment_id,
        candidate_results,
        policy_results,
    })
}

fn policy_result(
    policy: ExperimentSelectionPolicy,
    selected_experiment_id: Option<String>,
    expected_decision_utility: f64,
    expected_measurement_cost: f64,
    unsupported_outcome_probability: f64,
    oracle_expected_utility: f64,
) -> Result<ExperimentPolicyEvaluation, ExperimentPolicyBenchmarkError> {
    let regret = oracle_expected_utility - expected_decision_utility;
    if [
        expected_decision_utility,
        expected_measurement_cost,
        unsupported_outcome_probability,
        oracle_expected_utility,
        regret,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || expected_measurement_cost < 0.0
        || unsupported_outcome_probability < -PROBABILITY_TOLERANCE
        || unsupported_outcome_probability > 1.0 + PROBABILITY_TOLERANCE
        || regret < -UTILITY_TOLERANCE
    {
        return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
    }
    Ok(ExperimentPolicyEvaluation {
        policy,
        selected_experiment_id,
        expected_decision_utility,
        oracle_expected_utility,
        decision_regret: regret.max(0.0),
        expected_measurement_cost,
        unsupported_outcome_probability: unsupported_outcome_probability.clamp(0.0, 1.0),
    })
}

fn validate_ground_truth_contract(
    planning: &ActiveExperimentProblem,
    truth: &ActiveExperimentProblem,
) -> Result<(), ExperimentPolicyBenchmarkError> {
    if truth.hypotheses() != planning.hypotheses() {
        return Err(ExperimentPolicyBenchmarkError::GroundTruthHypothesisMismatch);
    }
    if truth.actions() != planning.actions() {
        return Err(ExperimentPolicyBenchmarkError::GroundTruthActionMismatch);
    }
    if truth.utilities_by_action_and_hypothesis() != planning.utilities_by_action_and_hypothesis()
        || truth.utility_unit() != planning.utility_unit()
    {
        return Err(ExperimentPolicyBenchmarkError::GroundTruthUtilityMismatch);
    }
    if truth.measurement_cost_unit() != planning.measurement_cost_unit() {
        return Err(ExperimentPolicyBenchmarkError::GroundTruthCostUnitMismatch);
    }
    if truth.experiments().len() != planning.experiments().len() {
        return Err(ExperimentPolicyBenchmarkError::GroundTruthCandidateMismatch);
    }
    for candidate in planning.experiments() {
        let other = truth
            .experiments()
            .iter()
            .find(|experiment| experiment.id() == candidate.id())
            .ok_or(ExperimentPolicyBenchmarkError::GroundTruthCandidateMismatch)?;
        if other.outcomes() != candidate.outcomes()
            || other.measurement_cost() != candidate.measurement_cost()
        {
            return Err(ExperimentPolicyBenchmarkError::GroundTruthCandidateContractMismatch);
        }
    }
    Ok(())
}

fn averaged_prior(scenarios: &[RobustModelScenario]) -> Vec<f64> {
    let count = scenarios.len() as f64;
    (0..scenarios[0].problem().prior().len())
        .map(|hypothesis| {
            scenarios
                .iter()
                .map(|scenario| scenario.problem().prior()[hypothesis])
                .sum::<f64>()
                / count
        })
        .collect()
}

fn best_action(
    reference: &ActiveExperimentProblem,
    hypothesis_weights: &[f64],
) -> Result<usize, ExperimentPolicyBenchmarkError> {
    if hypothesis_weights.len() != reference.hypotheses().len() {
        return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
    }
    let utilities = reference.utilities_by_action_and_hypothesis();
    let mut best: Option<(usize, f64)> = None;
    for (action_index, action) in reference.actions().iter().enumerate() {
        let score = utilities[action_index]
            .iter()
            .zip(hypothesis_weights)
            .map(|(utility, probability)| utility * probability)
            .sum::<f64>();
        if !score.is_finite() {
            return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
        }
        let replace = match best {
            None => true,
            Some((best_index, best_score)) => {
                score > best_score
                    || (score == best_score && action < &reference.actions()[best_index])
            }
        };
        if replace {
            best = Some((action_index, score));
        }
    }
    best.map(|(index, _)| index)
        .ok_or(ExperimentPolicyBenchmarkError::InvalidDerivedValue)
}

fn expected_action_utility(
    truth: &ActiveExperimentProblem,
    action_index: usize,
    hypothesis_weights: &[f64],
) -> Result<f64, ExperimentPolicyBenchmarkError> {
    if hypothesis_weights.len() != truth.hypotheses().len() {
        return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
    }
    let value = truth.utilities_by_action_and_hypothesis()[action_index]
        .iter()
        .zip(hypothesis_weights)
        .map(|(utility, weight)| utility * weight)
        .sum::<f64>();
    if !value.is_finite() {
        return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
    }
    Ok(value)
}

struct SingleExperimentEvaluation {
    measurement_cost: f64,
    expected_decision_utility: f64,
    unsupported_outcome_probability: f64,
}

fn evaluate_experiment(
    truth: &ActiveExperimentProblem,
    posterior_models: &[RobustModelScenario],
    experiment_id: &str,
) -> Result<SingleExperimentEvaluation, ExperimentPolicyBenchmarkError> {
    let true_experiment = truth
        .experiments()
        .iter()
        .find(|experiment| experiment.id() == experiment_id)
        .ok_or(ExperimentPolicyBenchmarkError::MissingCandidate)?;
    let cost = true_experiment.measurement_cost();
    let baseline_action = if posterior_models.is_empty() {
        best_action(truth, truth.prior())?
    } else {
        best_action(truth, &averaged_prior(posterior_models))?
    };
    let mut expected_utility = 0.0;
    let mut unsupported_mass = 0.0;
    let outcome_count = true_experiment.outcomes().len();

    for outcome_index in 0..outcome_count {
        let true_outcome_mass = truth
            .prior()
            .iter()
            .enumerate()
            .map(|(hypothesis, prior)| {
                prior * true_experiment.likelihoods_by_hypothesis()[hypothesis][outcome_index]
            })
            .sum::<f64>();
        if !true_outcome_mass.is_finite() || true_outcome_mass < 0.0 {
            return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
        }
        if true_outcome_mass <= 0.0 {
            continue;
        }

        let mut posterior_weights = vec![0.0; truth.hypotheses().len()];
        if !posterior_models.is_empty() {
            let model_count = posterior_models.len() as f64;
            for (hypothesis, weight) in posterior_weights.iter_mut().enumerate() {
                *weight = posterior_models
                    .iter()
                    .map(|scenario| {
                        let model = scenario.problem();
                        let experiment = model
                            .experiments()
                            .iter()
                            .find(|candidate| candidate.id() == experiment_id);
                        match experiment {
                            Some(experiment) => {
                                model.prior()[hypothesis]
                                    * experiment.likelihoods_by_hypothesis()[hypothesis]
                                        [outcome_index]
                            }
                            None => 0.0,
                        }
                    })
                    .sum::<f64>()
                    / model_count;
            }
        } else {
            for (hypothesis, weight) in posterior_weights.iter_mut().enumerate() {
                *weight = truth.prior()[hypothesis]
                    * true_experiment.likelihoods_by_hypothesis()[hypothesis][outcome_index];
            }
        }

        let predicted_mass = posterior_weights.iter().sum::<f64>();
        let action = if !predicted_mass.is_finite() || predicted_mass < 0.0 {
            return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
        } else if predicted_mass <= 0.0 {
            unsupported_mass += true_outcome_mass;
            baseline_action
        } else {
            best_action(truth, &posterior_weights)?
        };

        for hypothesis in 0..truth.hypotheses().len() {
            let joint = truth.prior()[hypothesis]
                * true_experiment.likelihoods_by_hypothesis()[hypothesis][outcome_index];
            expected_utility += joint
                * truth.utilities_by_action_and_hypothesis()[action][hypothesis];
        }
    }

    if !expected_utility.is_finite()
        || !unsupported_mass.is_finite()
        || unsupported_mass > 1.0 + PROBABILITY_TOLERANCE
    {
        return Err(ExperimentPolicyBenchmarkError::InvalidDerivedValue);
    }
    Ok(SingleExperimentEvaluation {
        measurement_cost: cost,
        expected_decision_utility: expected_utility,
        unsupported_outcome_probability: unsupported_mass.min(1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_experiment::DecisionExperiment;

    fn experiment(id: &str, cost: f64, likelihoods: Vec<Vec<f64>>) -> DecisionExperiment {
        DecisionExperiment::new(
            id,
            cost,
            vec!["outcome-0".into(), "outcome-1".into()],
            likelihoods,
        )
        .unwrap()
    }

    fn problem(
        likelihood_a: Vec<Vec<f64>>,
        likelihood_b: Vec<Vec<f64>>,
    ) -> ActiveExperimentProblem {
        ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            "utility_points",
            "measurement_seconds",
            vec![
                experiment("A-nominal-perfect", 1.0, likelihood_a),
                experiment("B-moderate", 1.0, likelihood_b),
            ],
        )
        .unwrap()
    }

    fn scenario(
        id: &str,
        nominal: bool,
        likelihood_a: Vec<Vec<f64>>,
    ) -> RobustModelScenario {
        RobustModelScenario::new(
            id,
            nominal,
            problem(
                likelihood_a,
                vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            ),
        )
        .unwrap()
    }

    fn fixture_models() -> (Vec<RobustModelScenario>, ActiveExperimentProblem) {
        let nominal = scenario(
            "nominal",
            true,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let alternative = scenario(
            "plausible-alternative",
            false,
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        let truth = problem(
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![vec![0.8, 0.2], vec![0.2, 0.8]],
        );
        (vec![nominal, alternative], truth)
    }

    fn policy<'a>(
        report: &'a ExperimentPolicyBakeoffReport,
        policy: ExperimentSelectionPolicy,
    ) -> &'a ExperimentPolicyEvaluation {
        report.policy_results.iter().find(|result| result.policy == policy).unwrap()
    }

    #[test]
    fn held_out_counterexample_proves_robust_choice_reduces_regret() {
        let (models, truth) = fixture_models();
        let report = run_experiment_policy_bakeoff("heldout-alternative", &truth, &models).unwrap();

        assert!((report.oracle_expected_utility - 0.8).abs() < 1e-12);
        let nominal = policy(&report, ExperimentSelectionPolicy::NominalEvsiPerCost);
        assert_eq!(nominal.selected_experiment_id.as_deref(), Some("A-nominal-perfect"));
        assert!((nominal.expected_decision_utility - 0.5).abs() < 1e-12);
        assert!((nominal.decision_regret - 0.3).abs() < 1e-12);

        let entropy = policy(&report, ExperimentSelectionPolicy::NominalInformationGainPerCost);
        assert_eq!(entropy.selected_experiment_id.as_deref(), Some("A-nominal-perfect"));
        assert!((entropy.decision_regret - 0.3).abs() < 1e-12);

        let robust = policy(&report, ExperimentSelectionPolicy::RobustWorstCaseEvsiPerCost);
        assert_eq!(robust.selected_experiment_id.as_deref(), Some("B-moderate"));
        assert!((robust.expected_decision_utility - 0.8).abs() < 1e-12);
        assert!(robust.decision_regret.abs() < 1e-12);
    }

    #[test]
    fn random_baseline_is_exact_expected_value_not_monte_carlo() {
        let (models, truth) = fixture_models();
        let report = run_experiment_policy_bakeoff("heldout-alternative", &truth, &models).unwrap();
        let random = policy(&report, ExperimentSelectionPolicy::UniformRandom);
        assert_eq!(random.selected_experiment_id, None);
        assert!((random.expected_decision_utility - 0.65).abs() < 1e-12);
        assert!((random.decision_regret - 0.15).abs() < 1e-12);
        assert_eq!(random.expected_measurement_cost, 1.0);
    }

    #[test]
    fn fixed_first_baseline_uses_declared_nominal_order() {
        let (models, truth) = fixture_models();
        let report = run_experiment_policy_bakeoff("heldout-alternative", &truth, &models).unwrap();
        let fixed = policy(&report, ExperimentSelectionPolicy::FixedFirst);
        assert_eq!(fixed.selected_experiment_id.as_deref(), Some("A-nominal-perfect"));
        assert!((fixed.decision_regret - 0.3).abs() < 1e-12);
    }

    #[test]
    fn truth_with_uncovered_outcome_mass_is_explicit() {
        let nominal = scenario(
            "nominal",
            true,
            vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        );
        let alternative = scenario(
            "alternative",
            false,
            vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        );
        let truth = problem(
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let report = run_experiment_policy_bakeoff("truth-with-unseen-outcome", &truth, &[nominal, alternative]).unwrap();
        let candidate = report.candidate_results.iter()
            .find(|result| result.experiment_id == "A-nominal-perfect").unwrap();
        assert!((candidate.unsupported_outcome_probability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ground_truth_candidate_mismatch_is_rejected() {
        let (models, truth) = fixture_models();
        truth = ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            "utility_points",
            "measurement_seconds",
            vec![experiment(
                "different-id",
                1.0,
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )],
        ).unwrap();
        assert_eq!(
            run_experiment_policy_bakeoff("truth", &truth, &models),
            Err(ExperimentPolicyBenchmarkError::GroundTruthCandidateMismatch)
        );
    }
}
