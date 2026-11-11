// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic stress fixture for active-experiment selection under likelihood misspecification.
//!
//! The policy sees only the declared likelihood-model scenarios. Evaluator-only
//! hidden-world likelihoods score the decision against the information-gain oracle.
//! One held-out world is intentionally absent from the policy model set. This is
//! a transparent synthetic fixture, not a sealed evaluator or physical evidence.

use serde::{Deserialize, Serialize};

use crate::experiment_design::{
    ExperimentCandidate, ExperimentDesignError, HypothesisPrior, ModelLikelihoodScenario,
    rank_experiments, rank_experiments_robust,
};

const H_DIRECT: &str = "direct_cause";
const H_SHARED: &str = "shared_cause";
const FIXED_DESIGN_ID: &str = "coarse_spatial_comparator";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustnessPolicy {
    /// Optimistically optimize information gain under the first declared model only.
    NominalExpectedGain,
    /// Maximize variance in predicted outcome probability across hypotheses under
    /// the first declared model, divided by cost. This simple baseline can be brittle.
    NominalPredictiveVariance,
    /// Optimize weighted expected gain across declared model scenarios.
    WeightedExpectedGain,
    /// Maximize worst-case gain across the declared model-scenario set.
    WorstCaseRobust,
    /// A predeclared fixed design, independent of the hidden world.
    FixedDesign,
    /// Exact expected score under uniform random selection from the candidate set.
    UniformRandomExpectation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldPolicyResult {
    pub world_id: String,
    /// True when this world is excluded from the models available to the selector.
    pub held_out: bool,
    pub policy: RobustnessPolicy,
    /// None for the analytically averaged uniform-random baseline.
    pub selected_candidate_id: Option<String>,
    pub oracle_candidate_id: String,
    /// Expected information gain per cost under evaluator-only hidden-world likelihoods.
    /// This is not an observed or realized gain from sampled outcomes.
    pub selected_hidden_world_expected_information_gain_per_cost: f64,
    /// Expected information gain per cost for the evaluator's oracle candidate.
    pub oracle_hidden_world_expected_information_gain_per_cost: f64,
    /// Regret in expected information gain per cost versus the evaluator-world oracle.
    pub regret_bits_per_cost: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyRegretSummary {
    pub policy: RobustnessPolicy,
    pub worlds: usize,
    pub held_out_worlds: usize,
    pub mean_regret_bits_per_cost: f64,
    pub max_regret_bits_per_cost: f64,
    /// Fractional reduction in maximum regret relative to nominal single-model ranking.
    /// Negative values indicate worse worst-case regret; None means the reference
    /// policy has zero maximum regret, so the ratio is undefined.
    pub max_regret_reduction_vs_nominal: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMisspecificationBenchmarkReport {
    pub schema_version: u16,
    pub fixture_version: String,
    pub declared_model_scenarios: Vec<String>,
    pub hidden_worlds: usize,
    pub held_out_world_ids: Vec<String>,
    pub per_world: Vec<WorldPolicyResult>,
    pub summaries: Vec<PolicyRegretSummary>,
}

/// Malformed benchmark fixture or a scoring invariant violation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelMisspecificationBenchmarkError {
    #[error(transparent)]
    InvalidExperimentDesign(#[from] ExperimentDesignError),
    #[error("the declared fixed-design candidate is absent")]
    MissingFixedDesign,
    #[error("a policy candidate is missing from the evaluator's score table")]
    MissingCandidateScore,
    #[error("benchmark candidate library is empty")]
    EmptyCandidateLibrary,
}

/// Run a deterministic, closed-form selection-regret fixture.
///
/// The first declared model favors a targeted measurement; the second treats it as
/// uninformative. Both declared models agree that the coarse comparator is
/// moderately informative. A third evaluator-only world holds out a different
/// likelihood table. Likelihoods are hand-authored synthetic inputs only.
///
/// This fixture does not claim that robust selection beats every fixed design: in
/// this particular small fixture, the fixed coarse comparator is equally strong
/// on worst-case regret.
pub fn run_model_misspecification_benchmark(
) -> Result<ModelMisspecificationBenchmarkReport, ModelMisspecificationBenchmarkError> {
    let priors = vec![
        HypothesisPrior { id: H_DIRECT.into(), probability: 0.5 },
        HypothesisPrior { id: H_SHARED.into(), probability: 0.5 },
    ];

    // These are the only likelihood tables available to selector policies.
    let model_scenarios = vec![
        ModelLikelihoodScenario {
            scenario_id: "declared_model_a".into(),
            weight: 0.5,
            candidates: vec![
                binary_candidate("targeted_brittle_measurement", 0.805),
                binary_candidate(FIXED_DESIGN_ID, 0.80),
                binary_candidate("negative_control", 0.50),
            ],
        },
        ModelLikelihoodScenario {
            scenario_id: "declared_model_b".into(),
            weight: 0.5,
            candidates: vec![
                binary_candidate("targeted_brittle_measurement", 0.50),
                binary_candidate(FIXED_DESIGN_ID, 0.80),
                binary_candidate("negative_control", 0.50),
            ],
        },
    ];

    // These tables are evaluator-only and are never passed into the selector APIs.
    let worlds = vec![
        HiddenWorld {
            id: "world_a".into(),
            held_out: false,
            candidates: vec![
                binary_candidate("targeted_brittle_measurement", 0.805),
                binary_candidate(FIXED_DESIGN_ID, 0.80),
                binary_candidate("negative_control", 0.50),
            ],
        },
        HiddenWorld {
            id: "world_b".into(),
            held_out: false,
            candidates: vec![
                binary_candidate("targeted_brittle_measurement", 0.50),
                binary_candidate(FIXED_DESIGN_ID, 0.80),
                binary_candidate("negative_control", 0.50),
            ],
        },
        HiddenWorld {
            id: "held_out_world_c".into(),
            held_out: true,
            candidates: vec![
                binary_candidate("targeted_brittle_measurement", 0.55),
                binary_candidate(FIXED_DESIGN_ID, 0.75),
                binary_candidate("negative_control", 0.50),
            ],
        },
    ];

    let policies = [
        RobustnessPolicy::NominalExpectedGain,
        RobustnessPolicy::NominalPredictiveVariance,
        RobustnessPolicy::WeightedExpectedGain,
        RobustnessPolicy::WorstCaseRobust,
        RobustnessPolicy::FixedDesign,
        RobustnessPolicy::UniformRandomExpectation,
    ];

    let mut per_world = Vec::with_capacity(worlds.len() * policies.len());
    for world in &worlds {
        // Only the evaluator sees this oracle ranking, after a policy choice.
        let oracle_ranking = rank_experiments(&priors, &world.candidates)?;
        let oracle = oracle_ranking.first()
            .ok_or(ModelMisspecificationBenchmarkError::EmptyCandidateLibrary)?;
        let oracle_score = oracle.information_gain_per_cost;
        let oracle_id = oracle.candidate_id.clone();

        for policy in policies {
            let selected = choose_policy(policy, &priors, &model_scenarios)?;
            let selected_score = match selected.as_deref() {
                Some(candidate_id) => oracle_ranking.iter()
                    .find(|score| score.candidate_id == candidate_id)
                    .map(|score| score.information_gain_per_cost)
                    .ok_or(ModelMisspecificationBenchmarkError::MissingCandidateScore)?,
                None => {
                    // Exact expectation of uniform random choice; there is no Monte
                    // Carlo variance and no arbitrary random seed for this fixture.
                    oracle_ranking.iter()
                        .map(|score| score.information_gain_per_cost)
                        .sum::<f64>() / oracle_ranking.len() as f64
                }
            };
            per_world.push(WorldPolicyResult {
                world_id: world.id.clone(),
                held_out: world.held_out,
                policy,
                selected_candidate_id: selected,
                oracle_candidate_id: oracle_id.clone(),
                selected_hidden_world_expected_information_gain_per_cost: selected_score,
                oracle_hidden_world_expected_information_gain_per_cost: oracle_score,
                regret_bits_per_cost: (oracle_score - selected_score).max(0.0),
            });
        }
    }

    let nominal_max_regret = per_world.iter()
        .filter(|row| row.policy == RobustnessPolicy::NominalExpectedGain)
        .map(|row| row.regret_bits_per_cost)
        .fold(0.0, f64::max);

    let summaries = policies.iter().copied().map(|policy| {
        let rows: Vec<&WorldPolicyResult> = per_world.iter()
            .filter(|row| row.policy == policy)
            .collect();
        let max_regret = rows.iter()
            .map(|row| row.regret_bits_per_cost)
            .fold(0.0, f64::max);
        let mean_regret = rows.iter()
            .map(|row| row.regret_bits_per_cost)
            .sum::<f64>() / rows.len() as f64;
        let held_out_worlds = rows.iter().filter(|row| row.held_out).count();
        PolicyRegretSummary {
            policy,
            worlds: rows.len(),
            held_out_worlds,
            mean_regret_bits_per_cost: mean_regret,
            max_regret_bits_per_cost: max_regret,
            max_regret_reduction_vs_nominal: (nominal_max_regret > f64::EPSILON)
                .then_some(1.0 - max_regret / nominal_max_regret),
        }
    }).collect();

    Ok(ModelMisspecificationBenchmarkReport {
        schema_version: 1,
        fixture_version: "sensebench-misspec-0.1.0".into(),
        declared_model_scenarios: model_scenarios.iter()
            .map(|scenario| scenario.scenario_id.clone())
            .collect(),
        hidden_worlds: worlds.len(),
        held_out_world_ids: worlds.iter()
            .filter(|world| world.held_out)
            .map(|world| world.id.clone())
            .collect(),
        per_world,
        summaries,
    })
}

#[derive(Debug, Clone)]
struct HiddenWorld {
    id: String,
    held_out: bool,
    candidates: Vec<ExperimentCandidate>,
}

/// Return a candidate ID, or None for the analytically averaged random policy.
/// The hidden-world object is intentionally not an argument to this selector.
fn choose_policy(
    policy: RobustnessPolicy,
    priors: &[HypothesisPrior],
    model_scenarios: &[ModelLikelihoodScenario],
) -> Result<Option<String>, ModelMisspecificationBenchmarkError> {
    match policy {
        RobustnessPolicy::NominalExpectedGain => {
            let ranking = rank_experiments(priors, &model_scenarios[0].candidates)?;
            ranking.first()
                .map(|score| Some(score.candidate_id.clone()))
                .ok_or(ModelMisspecificationBenchmarkError::EmptyCandidateLibrary)
        }
        RobustnessPolicy::NominalPredictiveVariance => {
            let candidates = &model_scenarios.first()
                .ok_or(ModelMisspecificationBenchmarkError::EmptyCandidateLibrary)?
                .candidates;
            let candidate = candidates.iter()
                .max_by(|left, right| {
                    (predictive_variance(left, priors) / left.cost)
                        .total_cmp(&(predictive_variance(right, priors) / right.cost))
                        .then_with(|| right.id.cmp(&left.id))
                })
                .ok_or(ModelMisspecificationBenchmarkError::EmptyCandidateLibrary)?;
            Ok(Some(candidate.id.clone()))
        }
        RobustnessPolicy::WeightedExpectedGain => {
            let scores = rank_experiments_robust(priors, model_scenarios)?;
            scores.into_iter()
                .max_by(|left, right| {
                    left.expected_information_gain_per_cost
                        .total_cmp(&right.expected_information_gain_per_cost)
                        .then_with(|| right.candidate_id.cmp(&left.candidate_id))
                })
                .map(|score| Some(score.candidate_id))
                .ok_or(ModelMisspecificationBenchmarkError::EmptyCandidateLibrary)
        }
        RobustnessPolicy::WorstCaseRobust => {
            let ranking = rank_experiments_robust(priors, model_scenarios)?;
            ranking.first()
                .map(|score| Some(score.candidate_id.clone()))
                .ok_or(ModelMisspecificationBenchmarkError::EmptyCandidateLibrary)
        }
        RobustnessPolicy::FixedDesign => {
            let present = model_scenarios.first()
                .is_some_and(|scenario| scenario.candidates.iter()
                    .any(|candidate| candidate.id == FIXED_DESIGN_ID));
            if !present {
                return Err(ModelMisspecificationBenchmarkError::MissingFixedDesign);
            }
            Ok(Some(FIXED_DESIGN_ID.into()))
        }
        RobustnessPolicy::UniformRandomExpectation => Ok(None),
    }
}

/// Variance across hypothesis-conditional probabilities of outcome 0.
/// This is a simple acquisition baseline, not expected information gain.
fn predictive_variance(
    candidate: &ExperimentCandidate,
    priors: &[HypothesisPrior],
) -> f64 {
    let mean = priors.iter()
        .zip(&candidate.likelihoods)
        .map(|(prior, row)| prior.probability * row[0])
        .sum::<f64>();
    priors.iter()
        .zip(&candidate.likelihoods)
        .map(|(prior, row)| prior.probability * (row[0] - mean).powi(2))
        .sum::<f64>()
}

/// Build a symmetric binary-outcome likelihood table.
///
/// Under the first hypothesis, P(outcome 0) = p; under the second it is 1-p.
/// With equal priors, p = 0.5 is uninformative and larger deviations are more
/// discriminating. Values here are illustrative synthetic inputs only.
fn binary_candidate(id: &str, p: f64) -> ExperimentCandidate {
    ExperimentCandidate {
        id: id.into(),
        cost: 1.0,
        likelihoods: vec![vec![p, 1.0 - p], vec![1.0 - p, p]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(report: &ModelMisspecificationBenchmarkReport, policy: RobustnessPolicy) -> &PolicyRegretSummary {
        report.summaries.iter().find(|entry| entry.policy == policy).unwrap()
    }

    fn row<'a>(
        report: &'a ModelMisspecificationBenchmarkReport,
        world: &str,
        policy: RobustnessPolicy,
    ) -> &'a WorldPolicyResult {
        report.per_world.iter()
            .find(|entry| entry.world_id == world && entry.policy == policy)
            .unwrap()
    }

    #[test]
    fn nominal_model_can_choose_a_brittle_measurement_and_regret_is_reported() {
        let report = run_model_misspecification_benchmark().unwrap();
        assert_eq!(
            row(&report, "world_a", RobustnessPolicy::NominalExpectedGain)
                .selected_candidate_id.as_deref(),
            Some("targeted_brittle_measurement")
        );
        assert_eq!(
            row(&report, "world_b", RobustnessPolicy::NominalExpectedGain)
                .selected_candidate_id.as_deref(),
            Some("targeted_brittle_measurement")
        );
        assert!(row(&report, "world_b", RobustnessPolicy::NominalExpectedGain)
            .regret_bits_per_cost > 0.0);
    }

    #[test]
    fn robust_selection_reduces_worst_case_regret_on_this_fixture() {
        let report = run_model_misspecification_benchmark().unwrap();
        let nominal = summary(&report, RobustnessPolicy::NominalExpectedGain);
        let robust = summary(&report, RobustnessPolicy::WorstCaseRobust);
        assert_eq!(
            row(&report, "held_out_world_c", RobustnessPolicy::WorstCaseRobust)
                .selected_candidate_id.as_deref(),
            Some(FIXED_DESIGN_ID)
        );
        let correctly_specified = row(
            &report,
            "world_a",
            RobustnessPolicy::WorstCaseRobust,
        );
        assert!(
            correctly_specified.selected_hidden_world_expected_information_gain_per_cost
                >= correctly_specified.oracle_hidden_world_expected_information_gain_per_cost * 0.90,
            "robust gain loss on the correctly specified control must be <= 10%",
        );
        assert!(
            robust.max_regret_bits_per_cost < nominal.max_regret_bits_per_cost * 0.45,
            "robust worst-case regret {:?} should be < 45% of nominal {:?}",
            robust.max_regret_bits_per_cost,
            nominal.max_regret_bits_per_cost,
        );
        assert!(robust.max_regret_reduction_vs_nominal.unwrap() > 0.55);
    }

    #[test]
    fn nominal_predictive_variance_can_also_be_brittle_under_model_shift() {
        let report = run_model_misspecification_benchmark().unwrap();
        assert_eq!(
            row(&report, "world_a", RobustnessPolicy::NominalPredictiveVariance)
                .selected_candidate_id.as_deref(),
            Some("targeted_brittle_measurement")
        );
        assert!(
            row(&report, "world_b", RobustnessPolicy::NominalPredictiveVariance)
                .regret_bits_per_cost > 0.0
        );
    }

    #[test]
    fn fixed_design_is_not_outscored_by_robust_policy_on_this_small_fixture() {
        // Keep this negative/control result explicit: a strong fixed measurement
        // can match the robust selector, so this test does not claim general active
        // selection superiority.
        let report = run_model_misspecification_benchmark().unwrap();
        let robust = summary(&report, RobustnessPolicy::WorstCaseRobust);
        let fixed = summary(&report, RobustnessPolicy::FixedDesign);
        assert!(
            (robust.max_regret_bits_per_cost - fixed.max_regret_bits_per_cost).abs() < 1.0e-12
        );
    }

    #[test]
    fn held_out_world_is_reported_and_repeated_runs_are_deterministic() {
        let first = run_model_misspecification_benchmark().unwrap();
        let second = run_model_misspecification_benchmark().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.held_out_world_ids, vec!["held_out_world_c"]);
        assert_eq!(first.per_world.len(), 18);
    }

    #[test]
    fn uniform_random_baseline_is_analytically_averaged() {
        let report = run_model_misspecification_benchmark().unwrap();
        let row = row(
            &report,
            "held_out_world_c",
            RobustnessPolicy::UniformRandomExpectation,
        );
        assert_eq!(row.selected_candidate_id, None);
        assert!(row.selected_hidden_world_expected_information_gain_per_cost > 0.0);
        assert!(row.selected_hidden_world_expected_information_gain_per_cost
            < row.oracle_hidden_world_expected_information_gain_per_cost);
    }
}
