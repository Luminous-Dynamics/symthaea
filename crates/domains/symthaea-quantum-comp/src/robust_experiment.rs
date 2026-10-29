//! Robust decision-focused experiment selection under model misspecification.
//!
//! A finite ambiguity set contains multiple plausible priors and/or outcome
//! likelihood models for the same decision problem. Each candidate experiment is
//! scored by QARCH-005 under every model; this module ranks candidates by their
//! worst-case expected value of sample information (EVSI) per cost, while
//! preserving nominal and ensemble diagnostics.
//!
//! This is robust only with respect to the explicit finite ensemble. It does
//! not guarantee safety against unrepresented models and does not calibrate the
//! physical likelihoods supplied by the caller.

use crate::active_experiment::{
    ActiveExperimentProblem, ExperimentDesignError, ExperimentScore,
};
use std::error::Error;
use std::fmt;

/// One plausible model of the same underlying decision problem.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustModelScenario {
    id: String,
    nominal: bool,
    problem: ActiveExperimentProblem,
}

impl RobustModelScenario {
    /// Create a named model scenario.
    ///
    /// Exactly one scenario in a robust ensemble must be designated nominal.
    /// Other scenarios may represent alternative priors, likelihoods, calibrated
    /// model variants, or predeclared perturbation regimes. Action utility and
    /// decision semantics must remain identical across all scenarios.
    pub fn new(
        id: impl Into<String>,
        nominal: bool,
        problem: ActiveExperimentProblem,
    ) -> Result<Self, RobustExperimentError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(RobustExperimentError::InvalidScenarioIdentifier);
        }
        Ok(Self {
            id,
            nominal,
            problem,
        })
    }

    /// Stable identifier for this plausible model.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Whether this is the declared nominal model.
    pub fn is_nominal(&self) -> bool {
        self.nominal
    }

    /// The validated decision and likelihood model for this scenario.
    pub fn problem(&self) -> &ActiveExperimentProblem {
        &self.problem
    }
}

/// Candidate experiment scores across a finite model ambiguity set.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustExperimentScore {
    /// Candidate experiment ID, shared by every model in the ensemble.
    pub experiment_id: String,
    /// Shared measurement cost.
    pub measurement_cost: f64,
    /// Nominal EVSI/cost.
    pub nominal_value_per_cost: f64,
    /// Minimum EVSI/cost over the declared model ensemble; primary ranking key.
    pub worst_case_value_per_cost: f64,
    /// Maximum EVSI/cost over the declared model ensemble.
    pub best_case_value_per_cost: f64,
    /// Arithmetic mean EVSI/cost over the declared model ensemble.
    pub mean_value_per_cost: f64,
    /// Maximum minus minimum EVSI/cost, exposing model sensitivity.
    pub value_per_cost_spread: f64,
    /// Every model's EVSI/cost, in report scenario order.
    pub scenario_values_per_cost: Vec<ScenarioValue>,
    /// Nominal rank under ordinary QARCH-005 selection (1 is best).
    pub nominal_rank: usize,
    /// Robust rank by worst-case EVSI/cost (1 is best).
    pub robust_rank: usize,
}

/// A candidate's EVSI/cost under one specific declared model.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioValue {
    /// Scenario ID.
    pub scenario_id: String,
    /// Whether this is the nominal model.
    pub nominal: bool,
    /// Expected value of sample information in utility units.
    pub evsi: f64,
    /// Expected value of sample information per cost unit.
    pub value_per_cost: f64,
}

/// Complete nominal and robust rankings for the same decision problem.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustExperimentRanking {
    /// Model scenario IDs in supplied report order.
    pub scenario_ids: Vec<String>,
    /// ID of the declared nominal scenario.
    pub nominal_scenario_id: String,
    /// Utility unit shared across every model.
    pub utility_unit: String,
    /// Measurement-cost unit shared across every model.
    pub measurement_cost_unit: String,
    /// Experiments sorted robust-first (worst-case EVSI/cost descending).
    pub scores: Vec<RobustExperimentScore>,
}

/// Validation errors for robust model ensembles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustExperimentError {
    /// Scenario identifier is empty.
    InvalidScenarioIdentifier,
    /// At least two plausible scenarios are required.
    TooFewScenarios,
    /// Scenario identifiers must be unique.
    DuplicateScenarioIdentifier,
    /// Exactly one scenario must be nominal.
    InvalidNominalScenarioCount,
    /// Scenario sets define different hypotheses.
    HypothesisSetMismatch,
    /// Scenario sets define different actions.
    ActionSetMismatch,
    /// Scenario utilities or their units differ.
    DecisionUtilityMismatch,
    /// Scenario measurement-cost units differ.
    MeasurementCostUnitMismatch,
    /// Candidate experiment sets differ.
    CandidateSetMismatch,
    /// Matching experiment IDs have different cost or outcomes.
    CandidateContractMismatch,
    /// Candidate model scoring failed validation or computation.
    ModelEvaluationFailed,
    /// Aggregated score became non-finite.
    DerivedScoreOverflow,
}

impl fmt::Display for RobustExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::InvalidScenarioIdentifier => "scenario identifiers must be non-empty",
            Self::TooFewScenarios => "at least two plausible scenarios are required",
            Self::DuplicateScenarioIdentifier => "scenario identifiers must be unique",
            Self::InvalidNominalScenarioCount => "exactly one nominal scenario is required",
            Self::HypothesisSetMismatch => "scenarios do not share the same hypotheses",
            Self::ActionSetMismatch => "scenarios do not share the same decision actions",
            Self::DecisionUtilityMismatch => {
                "scenarios must share identical decision utilities and utility units"
            }
            Self::MeasurementCostUnitMismatch => {
                "scenarios must share a common measurement-cost unit"
            }
            Self::CandidateSetMismatch => "scenarios do not share the same candidate experiments",
            Self::CandidateContractMismatch => {
                "a candidate has different costs or outcome labels across scenarios"
            }
            Self::ModelEvaluationFailed => "one scenario could not be scored",
            Self::DerivedScoreOverflow => "robust aggregation produced a non-finite score",
        };
        f.write_str(description)
    }
}

impl Error for RobustExperimentError {}

impl From<ExperimentDesignError> for RobustExperimentError {
    fn from(_: ExperimentDesignError) -> Self {
        Self::ModelEvaluationFailed
    }
}

/// Rank candidate experiments by worst-case EVSI/cost over explicit plausible models.
///
/// Candidate IDs, outcomes, costs, hypothesis/action labels, utilities and unit
/// semantics must agree across scenarios. Priors and likelihoods may differ.
/// Sorting is deterministic: descending worst-case score, descending nominal
/// score as a tie-break, then lexical candidate ID.
pub fn rank_robust_experiments(
    scenarios: Vec<RobustModelScenario>,
) -> Result<RobustExperimentRanking, RobustExperimentError> {
    if scenarios.len() < 2 {
        return Err(RobustExperimentError::TooFewScenarios);
    }

    let scenario_ids = scenarios
        .iter()
        .map(|scenario| scenario.id.clone())
        .collect::<Vec<_>>();
    ensure_unique_ids(&scenario_ids)?;

    let nominal_indices = scenarios
        .iter()
        .enumerate()
        .filter_map(|(index, scenario)| scenario.nominal.then_some(index))
        .collect::<Vec<_>>();
    if nominal_indices.len() != 1 {
        return Err(RobustExperimentError::InvalidNominalScenarioCount);
    }
    let nominal_index = nominal_indices[0];
    let reference = &scenarios[0].problem;
    validate_compatible_models(reference, &scenarios)?;

    let ranked = scenarios
        .iter()
        .map(|scenario| {
            scenario
                .problem
                .rank_experiments()
                .map_err(|_| RobustExperimentError::ModelEvaluationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let nominal_ranks = ranked[nominal_index]
        .scores
        .iter()
        .enumerate()
        .map(|(index, score)| (score.experiment_id.clone(), index + 1))
        .collect::<Vec<_>>();

    let mut robust_scores = Vec::with_capacity(reference.experiments().len());
    for reference_candidate in reference.experiments() {
        let candidate_id = reference_candidate.id();
        let mut scenario_values = Vec::with_capacity(scenarios.len());
        let mut costs = Vec::with_capacity(scenarios.len());

        for (scenario, ranking) in scenarios.iter().zip(&ranked) {
            let score = ranking
                .scores
                .iter()
                .find(|score| score.experiment_id == candidate_id)
                .ok_or(RobustExperimentError::CandidateSetMismatch)?;
            scenario_values.push(ScenarioValue {
                scenario_id: scenario.id.clone(),
                nominal: scenario.nominal,
                evsi: score.expected_value_of_sample_information,
                value_per_cost: score.value_per_cost,
            });
            costs.push(score.measurement_cost);
        }

        let measurement_cost = costs[0];
        if costs.iter().any(|cost| *cost != measurement_cost) {
            return Err(RobustExperimentError::CandidateContractMismatch);
        }
        let values = scenario_values
            .iter()
            .map(|value| value.value_per_cost)
            .collect::<Vec<_>>();
        let worst_case_value_per_cost = values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let best_case_value_per_cost = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_value_per_cost = values.iter().sum::<f64>() / values.len() as f64;
        let value_per_cost_spread = best_case_value_per_cost - worst_case_value_per_cost;
        if [
            worst_case_value_per_cost,
            best_case_value_per_cost,
            mean_value_per_cost,
            value_per_cost_spread,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(RobustExperimentError::DerivedScoreOverflow);
        }
        let nominal_value_per_cost = scenario_values[nominal_index].value_per_cost;
        let nominal_rank = nominal_ranks
            .iter()
            .find(|(id, _)| id == candidate_id)
            .map(|(_, rank)| *rank)
            .ok_or(RobustExperimentError::CandidateSetMismatch)?;

        robust_scores.push(RobustExperimentScore {
            experiment_id: candidate_id.to_string(),
            measurement_cost,
            nominal_value_per_cost,
            worst_case_value_per_cost,
            best_case_value_per_cost,
            mean_value_per_cost,
            value_per_cost_spread,
            scenario_values_per_cost: scenario_values,
            nominal_rank,
            robust_rank: 0,
        });
    }

    robust_scores.sort_by(|left, right| {
        right
            .worst_case_value_per_cost
            .total_cmp(&left.worst_case_value_per_cost)
            .then_with(|| {
                right
                    .nominal_value_per_cost
                    .total_cmp(&left.nominal_value_per_cost)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    for (index, score) in robust_scores.iter_mut().enumerate() {
        score.robust_rank = index + 1;
    }

    Ok(RobustExperimentRanking {
        scenario_ids,
        nominal_scenario_id: scenarios[nominal_index].id.clone(),
        utility_unit: reference.utility_unit().to_string(),
        measurement_cost_unit: reference.measurement_cost_unit().to_string(),
        scores: robust_scores,
    })
}

fn ensure_unique_ids(ids: &[String]) -> Result<(), RobustExperimentError> {
    for (index, id) in ids.iter().enumerate() {
        if id.trim().is_empty() {
            return Err(RobustExperimentError::InvalidScenarioIdentifier);
        }
        if ids[..index].iter().any(|previous| previous == id) {
            return Err(RobustExperimentError::DuplicateScenarioIdentifier);
        }
    }
    Ok(())
}

fn validate_compatible_models(
    reference: &ActiveExperimentProblem,
    scenarios: &[RobustModelScenario],
) -> Result<(), RobustExperimentError> {
    for scenario in scenarios.iter().skip(1) {
        let candidate = &scenario.problem;
        if candidate.hypotheses() != reference.hypotheses() {
            return Err(RobustExperimentError::HypothesisSetMismatch);
        }
        if candidate.actions() != reference.actions() {
            return Err(RobustExperimentError::ActionSetMismatch);
        }
        if candidate.utilities_by_action_and_hypothesis()
            != reference.utilities_by_action_and_hypothesis()
            || candidate.utility_unit() != reference.utility_unit()
        {
            return Err(RobustExperimentError::DecisionUtilityMismatch);
        }
        if candidate.measurement_cost_unit() != reference.measurement_cost_unit() {
            return Err(RobustExperimentError::MeasurementCostUnitMismatch);
        }
        if candidate.experiments().len() != reference.experiments().len() {
            return Err(RobustExperimentError::CandidateSetMismatch);
        }
        for reference_experiment in reference.experiments() {
            let other = candidate
                .experiments()
                .iter()
                .find(|experiment| experiment.id() == reference_experiment.id())
                .ok_or(RobustExperimentError::CandidateSetMismatch)?;
            if other.outcomes() != reference_experiment.outcomes()
                || other.measurement_cost() != reference_experiment.measurement_cost()
            {
                return Err(RobustExperimentError::CandidateContractMismatch);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_experiment::DecisionExperiment;

    fn experiment(
        id: &str,
        cost: f64,
        likelihoods: Vec<Vec<f64>>,
    ) -> DecisionExperiment {
        DecisionExperiment::new(
            id,
            cost,
            vec!["outcome-0".into(), "outcome-1".into()],
            likelihoods,
        )
        .unwrap()
    }

    fn problem(
        perfect_likelihoods: Vec<Vec<f64>>,
        add_second: bool,
    ) -> ActiveExperimentProblem {
        let mut experiments = vec![
            experiment(
                "A-perfect-nominal",
                1.0,
                perfect_likelihoods,
            ),
            experiment(
                "B-robust-moderate",
                1.0,
                vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            ),
        ];
        if !add_second {
            experiments.truncate(1);
        }
        ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            "utility_points",
            "measurement_seconds",
            experiments,
        )
        .unwrap()
    }

    #[test]
    fn robust_ranking_rejects_a_nominal_winner_that_collapses() {
        let nominal = RobustModelScenario::new(
            "nominal",
            true,
            problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], true),
        )
        .unwrap();
        let alternative = RobustModelScenario::new(
            "plausible-uninformative",
            false,
            problem(vec![vec![0.5, 0.5], vec![0.5, 0.5]], true),
        )
        .unwrap();

        let nominal_rank = nominal.problem.rank_experiments().unwrap();
        assert_eq!(nominal_rank.scores[0].experiment_id, "A-perfect-nominal");
        assert!((nominal_rank.scores[0].value_per_cost - 0.5).abs() < 1e-12);
        assert!((nominal_rank.scores[1].value_per_cost - 0.3).abs() < 1e-12);

        let report = rank_robust_experiments(vec![nominal, alternative]).unwrap();
        assert_eq!(report.scores[0].experiment_id, "B-robust-moderate");
        assert_eq!(report.scores[0].nominal_rank, 2);
        assert_eq!(report.scores[0].robust_rank, 1);
        assert!((report.scores[0].worst_case_value_per_cost - 0.3).abs() < 1e-12);
        assert_eq!(report.scores[1].experiment_id, "A-perfect-nominal");
        assert_eq!(report.scores[1].nominal_rank, 1);
        assert_eq!(report.scores[1].robust_rank, 2);
        assert_eq!(report.scores[1].worst_case_value_per_cost, 0.0);
    }

    #[test]
    fn sensitivity_spread_exposes_misspecification_risk() {
        let nominal = RobustModelScenario::new(
            "nominal",
            true,
            problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], true),
        )
        .unwrap();
        let alternative = RobustModelScenario::new(
            "alternative",
            false,
            problem(vec![vec![0.5, 0.5], vec![0.5, 0.5]], true),
        )
        .unwrap();
        let report = rank_robust_experiments(vec![nominal, alternative]).unwrap();
        let fragile = report
            .scores
            .iter()
            .find(|score| score.experiment_id == "A-perfect-nominal")
            .unwrap();
        assert!((fragile.value_per_cost_spread - 0.5).abs() < 1e-12);
        let robust = report
            .scores
            .iter()
            .find(|score| score.experiment_id == "B-robust-moderate")
            .unwrap();
        assert!(robust.value_per_cost_spread.abs() < 1e-12);
    }

    #[test]
    fn duplicate_scenario_ids_and_missing_nominal_are_rejected() {
        let p = problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], true);
        let first = RobustModelScenario::new("same", true, p.clone()).unwrap();
        let second = RobustModelScenario::new("same", false, p.clone()).unwrap();
        assert_eq!(
            rank_robust_experiments(vec![first, second]),
            Err(RobustExperimentError::DuplicateScenarioIdentifier)
        );
        let first = RobustModelScenario::new("first", false, p.clone()).unwrap();
        let second = RobustModelScenario::new("second", false, p).unwrap();
        assert_eq!(
            rank_robust_experiments(vec![first, second]),
            Err(RobustExperimentError::InvalidNominalScenarioCount)
        );
    }

    #[test]
    fn candidate_contract_mismatch_is_rejected() {
        let nominal = RobustModelScenario::new(
            "nominal",
            true,
            problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], true),
        )
        .unwrap();
        let altered = ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            "utility_points",
            "measurement_seconds",
            vec![
                experiment("A-perfect-nominal", 2.0, vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
                experiment("B-robust-moderate", 1.0, vec![vec![0.8, 0.2], vec![0.2, 0.8]]),
            ],
        ).unwrap();
        let scenario = RobustModelScenario::new("changed-cost", false, altered).unwrap();
        assert_eq!(
            rank_robust_experiments(vec![nominal, scenario]),
            Err(RobustExperimentError::CandidateContractMismatch)
        );
    }

    #[test]
    fn utility_semantics_must_match_across_scenarios() {
        let nominal = RobustModelScenario::new(
            "nominal",
            true,
            problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], true),
        ).unwrap();
        let altered = ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![10.0, 0.0], vec![0.0, 1.0]],
            "utility_points",
            "measurement_seconds",
            vec![
                experiment("A-perfect-nominal", 1.0, vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
                experiment("B-robust-moderate", 1.0, vec![vec![0.8, 0.2], vec![0.2, 0.8]]),
            ],
        ).unwrap();
        let scenario = RobustModelScenario::new("altered-utility", false, altered).unwrap();
        assert_eq!(
            rank_robust_experiments(vec![nominal, scenario]),
            Err(RobustExperimentError::DecisionUtilityMismatch)
        );
    }
}
