// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reproducible benchmark harness for diagnostic policies.
//!
//! Belief likelihoods and ground-truth outcome distributions are stored
//! separately so that well-specified and deliberately misspecified agents can
//! be evaluated without silently sampling from their own predictions.
//! The harness compares no-probe, pure model-EIG, targeted decision-EIG,
//! cost-aware EVSI, and always-inspect policies on paired deterministic episodes.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionDiagnosticError, DecisionDiagnosticEvaluation,
    DecisionHypothesis, DiagnosticAction, evaluate_decision_focused_diagnostics,
    posterior_model_probabilities,
};
use super::hierarchical_information_gain::{
    HierarchicalHypothesis, compute_hierarchical_information_gains,
};

const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// The policy being evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticPolicy {
    /// Make the best decision under the prior, without a diagnostic.
    NoProbe,
    /// Maximize model information gain, ignoring probe cost.
    PureModelInformationGain,
    /// Maximize information gain about which downstream decision is optimal.
    /// This is the targeted-active-learning baseline, not an EVSI utility criterion.
    OptimalDecisionInformationGain,
    /// Minimize posterior Bayes risk plus configured cost and safety penalties.
    CostAwareEvsi,
    /// Always perform the first safety-eligible diagnostic.
    AlwaysInspect,
}

/// One diagnostic decision problem with a separate ground-truth observation model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticScenario {
    pub scenario_id: String,
    pub hypotheses: Vec<DecisionHypothesis>,
    pub actions: Vec<DiagnosticAction>,
    /// Truth-side probabilities indexed by [hypothesis][action][outcome].
    /// These may intentionally differ from the agent's likelihoods to model
    /// calibration error, while preserving common outcome support.
    pub true_outcome_probabilities_by_hypothesis_action: Vec<Vec<Vec<f64>>>,
}

/// Aggregate benchmark metrics for one policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticPolicyMetrics {
    pub scenario_id: String,
    pub policy: DiagnosticPolicy,
    pub episodes: usize,
    pub probes_taken: usize,
    pub probes_declined: usize,
    pub selected_action_counts: Vec<usize>,
    pub mean_task_loss: f64,
    pub mean_oracle_task_loss: f64,
    pub mean_decision_regret: f64,
    pub mean_total_cost: f64,
    pub mean_observation_cost: f64,
    pub mean_safety_risk: f64,
    pub mean_predicted_model_eig_nats: f64,
    pub mean_predicted_decision_eig_nats: f64,
    /// Can be negative under model misspecification.
    pub mean_realized_model_entropy_reduction_nats: f64,
    pub mean_model_brier_score: f64,
}

/// Paired outcome contrast for a candidate policy against a reference policy.
///
/// Differences are candidate minus reference; negative values favor the candidate.
/// The normal interval is descriptive. The Hoeffding interval is a conservative,
/// finite-sample bound for independent paired episodes with the declared range.
/// Neither interval replaces independent scenario-level replication or a
/// cluster-aware analysis when multiple scenarios are aggregated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticPolicyContrast {
    pub reference_policy: DiagnosticPolicy,
    pub candidate_policy: DiagnosticPolicy,
    pub episodes: usize,
    pub mean_task_loss_difference: f64,
    /// Descriptive paired normal-approximation interval for task loss.
    pub task_loss_difference_ci95: Option<[f64; 2]>,
    /// Distribution-free paired Hoeffding interval; paired task loss lies in [-1, 1].
    pub task_loss_difference_hoeffding_ci95: Option<[f64; 2]>,
    /// Simultaneous family-wise 95% Hoeffding interval, Bonferroni-adjusted across all contrasts and outcomes.
    pub task_loss_difference_familywise_ci95: Option<[f64; 2]>,
    pub mean_decision_regret_difference: f64,
    /// Descriptive paired normal-approximation interval for decision regret.
    pub decision_regret_difference_ci95: Option<[f64; 2]>,
    /// Distribution-free paired Hoeffding interval; paired regret lies in [-1, 1].
    pub decision_regret_difference_hoeffding_ci95: Option<[f64; 2]>,
    /// Simultaneous family-wise 95% Hoeffding interval for decision regret.
    pub decision_regret_difference_familywise_ci95: Option<[f64; 2]>,
    pub mean_total_cost_difference: f64,
    /// Descriptive paired normal-approximation interval for total cost.
    pub total_cost_difference_ci95: Option<[f64; 2]>,
    /// Distribution-free paired Hoeffding interval using the scenario's known cost bound.
    pub total_cost_difference_hoeffding_ci95: Option<[f64; 2]>,
    /// Simultaneous family-wise 95% Hoeffding interval for total cost.
    pub total_cost_difference_familywise_ci95: Option<[f64; 2]>,
    /// Number of simultaneous claims in the Bonferroni family.
    pub familywise_comparison_count: usize,
    pub confidence_interval_method: String,
}

/// Results for the fixed policy comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticPolicyBenchmark {
    pub scenario_id: String,
    pub seed: u64,
    pub policies: Vec<DiagnosticPolicyMetrics>,
    pub paired_evsi_contrasts: Vec<DiagnosticPolicyContrast>,
}

#[derive(Debug, Clone, Copy)]
struct DiagnosticEpisodeMetrics {
    task_loss: f64,
    decision_regret: f64,
    total_cost: f64,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DiagnosticBenchmarkError {
    #[error(transparent)]
    Decision(#[from] DecisionDiagnosticError),
    #[error("scenario identifier must not be empty")]
    EmptyScenarioId,
    #[error("at least one episode is required")]
    EmptyEpisodeSet,
    #[error("ground-truth hypothesis count {actual} does not match {expected}")]
    TruthHypothesisCount { expected: usize, actual: usize },
    #[error(
        "ground-truth action count for hypothesis {hypothesis_index}: {actual}, expected {expected}"
    )]
    TruthActionCount {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "ground-truth outcome count for hypothesis {hypothesis_index}, action {action_index}: {actual}, expected {expected}"
    )]
    TruthOutcomeCount {
        hypothesis_index: usize,
        action_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "ground-truth distribution is invalid for hypothesis {hypothesis_index}, action {action_index}"
    )]
    InvalidTruthDistribution {
        hypothesis_index: usize,
        action_index: usize,
    },
    #[error(
        "ground-truth model assigns mass outside the agent's likelihood support for hypothesis {hypothesis_index}, action {action_index}, outcome {outcome_index}"
    )]
    TruthOutsideModelSupport {
        hypothesis_index: usize,
        action_index: usize,
        outcome_index: usize,
    },
    #[error("scenario identifier does not match")]
    ScenarioIdMismatch,
    #[error("paired policy episode counts differ: reference {reference}, candidate {candidate}")]
    PairedEpisodeCountMismatch { reference: usize, candidate: usize },
}

/// Run the fixed five-policy suite using paired deterministic episode seeds.
///
/// Each policy sees the same sequence of hidden hypotheses, while observations
/// are sampled from the scenario's truth-side model. This is a diagnostic
/// benchmark, not a trained RL comparison or an empirical deployment claim.
pub fn run_diagnostic_policy_benchmark(
    scenario: &DiagnosticScenario,
    episodes: usize,
    seed: u64,
    config: DecisionDiagnosticConfig,
) -> Result<DiagnosticPolicyBenchmark, DiagnosticBenchmarkError> {
    if episodes == 0 {
        return Err(DiagnosticBenchmarkError::EmptyEpisodeSet);
    }
    validate_scenario(scenario, config)?;

    let policies = [
        DiagnosticPolicy::NoProbe,
        DiagnosticPolicy::PureModelInformationGain,
        DiagnosticPolicy::OptimalDecisionInformationGain,
        DiagnosticPolicy::CostAwareEvsi,
        DiagnosticPolicy::AlwaysInspect,
    ];
    let runs = policies
        .into_iter()
        .map(|policy| run_policy(scenario, policy, episodes, seed, config))
        .collect::<Result<Vec<_>, _>>()?;

    let policy_metrics = runs.iter().map(|(metrics, _)| metrics.clone()).collect();
    let candidate = runs
        .iter()
        .find(|(metrics, _)| metrics.policy == DiagnosticPolicy::CostAwareEvsi)
        .expect("the fixed policy suite includes cost-aware EVSI");
    // Each episode's total cost is bounded by task loss (<= 1), the maximum
    // weighted observation cost, and the maximum weighted safety penalty.
    let total_cost_absolute_bound = 1.0
        + config.observation_cost_weight
            * scenario
                .actions
                .iter()
                .map(|action| action.observation_cost)
                .fold(0.0, f64::max)
        + config.safety_risk_weight
            * scenario
                .actions
                .iter()
                .map(|action| action.safety_risk)
                .fold(0.0, f64::max);
    let reference_policies = [
        DiagnosticPolicy::NoProbe,
        DiagnosticPolicy::PureModelInformationGain,
        DiagnosticPolicy::OptimalDecisionInformationGain,
        DiagnosticPolicy::AlwaysInspect,
    ];
    let paired_evsi_contrasts = reference_policies
        .into_iter()
        .map(|reference_policy| {
            let reference = runs
                .iter()
                .find(|(metrics, _)| metrics.policy == reference_policy)
                .expect("the fixed policy suite includes every registered reference");
            paired_policy_contrast(
                reference_policy,
                &reference.1,
                DiagnosticPolicy::CostAwareEvsi,
                &candidate.1,
                total_cost_absolute_bound,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(DiagnosticPolicyBenchmark {
        scenario_id: scenario.scenario_id.clone(),
        seed,
        policies: policy_metrics,
        paired_evsi_contrasts,
    })
}

fn validate_scenario(
    scenario: &DiagnosticScenario,
    config: DecisionDiagnosticConfig,
) -> Result<(), DiagnosticBenchmarkError> {
    if scenario.scenario_id.trim().is_empty() {
        return Err(DiagnosticBenchmarkError::EmptyScenarioId);
    }
    evaluate_decision_focused_diagnostics(&scenario.hypotheses, &scenario.actions, config)?;

    let expected_hypotheses = scenario.hypotheses.len();
    if scenario
        .true_outcome_probabilities_by_hypothesis_action
        .len()
        != expected_hypotheses
    {
        return Err(DiagnosticBenchmarkError::TruthHypothesisCount {
            expected: expected_hypotheses,
            actual: scenario
                .true_outcome_probabilities_by_hypothesis_action
                .len(),
        });
    }

    for (hypothesis_index, (hypothesis, truth_actions)) in scenario
        .hypotheses
        .iter()
        .zip(&scenario.true_outcome_probabilities_by_hypothesis_action)
        .enumerate()
    {
        if truth_actions.len() != scenario.actions.len() {
            return Err(DiagnosticBenchmarkError::TruthActionCount {
                hypothesis_index,
                expected: scenario.actions.len(),
                actual: truth_actions.len(),
            });
        }
        for (action_index, truth_outcomes) in truth_actions.iter().enumerate() {
            let model_outcomes = &hypothesis.outcome_probabilities_by_action[action_index];
            if truth_outcomes.len() != model_outcomes.len() {
                return Err(DiagnosticBenchmarkError::TruthOutcomeCount {
                    hypothesis_index,
                    action_index,
                    expected: model_outcomes.len(),
                    actual: truth_outcomes.len(),
                });
            }
            let sum = truth_outcomes.iter().sum::<f64>();
            if truth_outcomes
                .iter()
                .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
                || (sum - 1.0).abs() > PROBABILITY_TOLERANCE
            {
                return Err(DiagnosticBenchmarkError::InvalidTruthDistribution {
                    hypothesis_index,
                    action_index,
                });
            }
            for (outcome_index, (truth, model)) in
                truth_outcomes.iter().zip(model_outcomes).enumerate()
            {
                if *truth > 0.0 && *model == 0.0 {
                    return Err(DiagnosticBenchmarkError::TruthOutsideModelSupport {
                        hypothesis_index,
                        action_index,
                        outcome_index,
                    });
                }
            }
        }
    }
    Ok(())
}

fn run_policy(
    scenario: &DiagnosticScenario,
    policy: DiagnosticPolicy,
    episodes: usize,
    seed: u64,
    config: DecisionDiagnosticConfig,
) -> Result<(DiagnosticPolicyMetrics, Vec<DiagnosticEpisodeMetrics>), DiagnosticBenchmarkError> {
    let joint = to_joint_hypotheses(&scenario.hypotheses);
    let information = compute_hierarchical_information_gains(&joint)?;
    let decision_evaluation =
        evaluate_decision_focused_diagnostics(&scenario.hypotheses, &scenario.actions, config)?;
    let decision_count = scenario.hypotheses[0].loss_by_final_decision.len();
    let prior = scenario
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();

    let selected_action =
        select_action(scenario, &information, &decision_evaluation, policy, config)?;
    let mut probes_taken = 0;
    let mut selected_action_counts = vec![0; scenario.actions.len()];
    let mut task_loss_sum = 0.0;
    let mut oracle_loss_sum = 0.0;
    let mut regret_sum = 0.0;
    let mut total_cost_sum = 0.0;
    let mut observation_cost_sum = 0.0;
    let mut safety_risk_sum = 0.0;
    let mut predicted_eig_sum = 0.0;
    let mut predicted_decision_eig_sum = 0.0;
    let mut realized_entropy_reduction_sum = 0.0;
    let mut brier_sum = 0.0;
    let mut episode_metrics = Vec::with_capacity(episodes);

    for episode_index in 0..episodes {
        // Reinitialize per episode so every policy sees the same hidden model
        // and the same outcome quantile, independent of earlier probe choices.
        let episode_seed =
            seed.wrapping_add((episode_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = DeterministicRng::new(episode_seed);
        let true_hypothesis = sample_categorical(&prior, &mut rng);
        let mut posterior = prior.clone();
        let mut action_cost = 0.0;
        let mut action_risk = 0.0;

        if let Some(action_index) = selected_action {
            let action = &scenario.actions[action_index];
            let predicted = &information[action_index];
            let predicted_decision = &decision_evaluation.actions[action_index];
            let truth = &scenario.true_outcome_probabilities_by_hypothesis_action[true_hypothesis]
                [action_index];
            let outcome_index = sample_categorical(truth, &mut rng);
            posterior =
                posterior_model_probabilities(&scenario.hypotheses, action_index, outcome_index)?;
            action_cost = action.observation_cost;
            action_risk = action.safety_risk;
            predicted_eig_sum += predicted.model_information_gain_nats;
            predicted_decision_eig_sum += predicted_decision.decision_information_gain_nats;
            selected_action_counts[action_index] += 1;
            probes_taken += 1;
        }

        realized_entropy_reduction_sum +=
            categorical_entropy(&prior) - categorical_entropy(&posterior);
        brier_sum += multiclass_brier_score(&posterior, true_hypothesis);

        let selected_decision = best_decision(&posterior, &scenario.hypotheses, decision_count);
        let task_loss =
            scenario.hypotheses[true_hypothesis].loss_by_final_decision[selected_decision];
        let oracle_loss = scenario.hypotheses[true_hypothesis]
            .loss_by_final_decision
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let regret = (task_loss - oracle_loss).max(0.0);
        let total_cost = task_loss
            + config.observation_cost_weight * action_cost
            + config.safety_risk_weight * action_risk;

        task_loss_sum += task_loss;
        oracle_loss_sum += oracle_loss;
        regret_sum += regret;
        total_cost_sum += total_cost;
        observation_cost_sum += action_cost;
        safety_risk_sum += action_risk;
        episode_metrics.push(DiagnosticEpisodeMetrics {
            task_loss,
            decision_regret: regret,
            total_cost,
        });
    }

    let metrics = DiagnosticPolicyMetrics {
        scenario_id: scenario.scenario_id.clone(),
        policy,
        episodes,
        probes_taken,
        probes_declined: episodes - probes_taken,
        selected_action_counts,
        mean_task_loss: task_loss_sum / episodes as f64,
        mean_oracle_task_loss: oracle_loss_sum / episodes as f64,
        mean_decision_regret: regret_sum / episodes as f64,
        mean_total_cost: total_cost_sum / episodes as f64,
        mean_observation_cost: observation_cost_sum / episodes as f64,
        mean_safety_risk: safety_risk_sum / episodes as f64,
        mean_predicted_model_eig_nats: predicted_eig_sum / episodes as f64,
        mean_predicted_decision_eig_nats: predicted_decision_eig_sum / episodes as f64,
        mean_realized_model_entropy_reduction_nats: realized_entropy_reduction_sum
            / episodes as f64,
        mean_model_brier_score: brier_sum / episodes as f64,
    };
    Ok((metrics, episode_metrics))
}

fn paired_policy_contrast(
    reference_policy: DiagnosticPolicy,
    reference: &[DiagnosticEpisodeMetrics],
    candidate_policy: DiagnosticPolicy,
    candidate: &[DiagnosticEpisodeMetrics],
    total_cost_absolute_bound: f64,
) -> Result<DiagnosticPolicyContrast, DiagnosticBenchmarkError> {
    if reference.len() != candidate.len() {
        return Err(DiagnosticBenchmarkError::PairedEpisodeCountMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }
    let episodes = reference.len();
    let task_loss_differences = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| candidate.task_loss - reference.task_loss)
        .collect::<Vec<_>>();
    let regret_differences = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| candidate.decision_regret - reference.decision_regret)
        .collect::<Vec<_>>();
    let cost_differences = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| candidate.total_cost - reference.total_cost)
        .collect::<Vec<_>>();
    let (mean_task_loss_difference, task_loss_difference_ci95) =
        paired_mean_ci95(&task_loss_differences);
    let (mean_decision_regret_difference, decision_regret_difference_ci95) =
        paired_mean_ci95(&regret_differences);
    let (mean_total_cost_difference, total_cost_difference_ci95) =
        paired_mean_ci95(&cost_differences);
    let familywise_alpha = 0.05 / 12.0; // 4 reference policies × 3 co-reported outcomes.
    let task_loss_difference_hoeffding_ci95 =
        paired_hoeffding_ci95(&task_loss_differences, 1.0, 0.05);
    let task_loss_difference_familywise_ci95 =
        paired_hoeffding_ci95(&task_loss_differences, 1.0, familywise_alpha);
    let decision_regret_difference_hoeffding_ci95 =
        paired_hoeffding_ci95(&regret_differences, 1.0, 0.05);
    let decision_regret_difference_familywise_ci95 =
        paired_hoeffding_ci95(&regret_differences, 1.0, familywise_alpha);
    let total_cost_difference_hoeffding_ci95 =
        paired_hoeffding_ci95(&cost_differences, total_cost_absolute_bound, 0.05);
    let total_cost_difference_familywise_ci95 =
        paired_hoeffding_ci95(&cost_differences, total_cost_absolute_bound, familywise_alpha);

    Ok(DiagnosticPolicyContrast {
        reference_policy,
        candidate_policy,
        episodes,
        mean_task_loss_difference,
        task_loss_difference_ci95,
        task_loss_difference_hoeffding_ci95,
        task_loss_difference_familywise_ci95,
        mean_decision_regret_difference,
        decision_regret_difference_ci95,
        decision_regret_difference_hoeffding_ci95,
        decision_regret_difference_familywise_ci95,
        mean_total_cost_difference,
        total_cost_difference_ci95,
        total_cost_difference_hoeffding_ci95,
        total_cost_difference_familywise_ci95,
        familywise_comparison_count: 12,
        confidence_interval_method:
            "paired_normal_descriptive_plus_95pct_hoeffding_and_95pct_bonferroni_12_family"
                .to_owned(),
    })
}

/// Mean and descriptive normal-approximation 95% interval for paired episode differences.
/// For one episode the normal interval is absent rather than falsely reporting zero width.
fn paired_mean_ci95(values: &[f64]) -> (f64, Option<[f64; 2]>) {
    if values.is_empty() {
        return (0.0, None);
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    if values.len() < 2 {
        return (mean, None);
    }
    let sample_variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (count - 1.0);
    let standard_error = (sample_variance / count).sqrt();
    let half_width = 1.959_963_984_540_054 * standard_error;
    (mean, Some([mean - half_width, mean + half_width]))
}

/// Distribution-free, two-sided Hoeffding interval for independent paired
/// observations known to lie in [-absolute_bound, absolute_bound].
///
/// For X in an interval of width 2B, Hoeffding gives
/// P(|mean(X)-E[X]| >= r) <= 2 exp(-n r^2 / (2 B^2)).
/// The confidence interval is clipped to the known support of the mean.
fn paired_hoeffding_ci95(
    values: &[f64],
    absolute_bound: f64,
    alpha: f64,
) -> Option<[f64; 2]> {
    if values.is_empty()
        || !absolute_bound.is_finite()
        || absolute_bound <= 0.0
        || !alpha.is_finite()
        || !(0.0..1.0).contains(&alpha)
    {
        return None;
    }
    if values.iter().any(|value| !value.is_finite() || value.abs() > absolute_bound) {
        return None;
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let radius = absolute_bound * (2.0 * (2.0_f64 / alpha).ln() / count).sqrt();
    Some([
        (mean - radius).max(-absolute_bound),
        (mean + radius).min(absolute_bound),
    ])
}

fn select_action(
    scenario: &DiagnosticScenario,
    information: &[super::hierarchical_information_gain::HierarchicalActionInformationGain],
    decision_evaluation: &DecisionDiagnosticEvaluation,
    policy: DiagnosticPolicy,
    config: DecisionDiagnosticConfig,
) -> Result<Option<usize>, DecisionDiagnosticError> {
    match policy {
        DiagnosticPolicy::NoProbe => Ok(None),
        DiagnosticPolicy::PureModelInformationGain => {
            let mut selected: Option<(usize, f64)> = None;
            for (index, action) in scenario.actions.iter().enumerate() {
                if action.safety_risk > config.maximum_safety_risk {
                    continue;
                }
                let eig = information[index].model_information_gain_nats;
                if selected.is_none_or(|(_, best)| eig > best) {
                    selected = Some((index, eig));
                }
            }
            Ok(selected
                .filter(|(_, eig)| *eig > 0.0)
                .map(|(index, _)| index))
        }
        DiagnosticPolicy::OptimalDecisionInformationGain => {
            let mut selected: Option<(usize, f64)> = None;
            for (index, action) in scenario.actions.iter().enumerate() {
                if action.safety_risk > config.maximum_safety_risk {
                    continue;
                }
                let information = decision_evaluation.actions[index].decision_information_gain_nats;
                if selected.is_none_or(|(_, best)| information > best) {
                    selected = Some((index, information));
                }
            }
            Ok(selected
                .filter(|(_, information)| *information > 0.0)
                .map(|(index, _)| index))
        }
        DiagnosticPolicy::CostAwareEvsi => Ok(decision_evaluation.selected_action_index),
        DiagnosticPolicy::AlwaysInspect => Ok(scenario
            .actions
            .iter()
            .position(|action| action.safety_risk <= config.maximum_safety_risk)),
    }
}

fn to_joint_hypotheses(hypotheses: &[DecisionHypothesis]) -> Vec<HierarchicalHypothesis> {
    hypotheses
        .iter()
        .map(|hypothesis| HierarchicalHypothesis {
            model_id: hypothesis.id.clone(),
            parameter_id: "benchmark-parameter".to_owned(),
            state_id: "benchmark-state".to_owned(),
            prior_probability: hypothesis.prior_probability,
            outcome_probabilities_by_action: hypothesis.outcome_probabilities_by_action.clone(),
        })
        .collect()
}

fn best_decision(
    posterior: &[f64],
    hypotheses: &[DecisionHypothesis],
    decision_count: usize,
) -> usize {
    (0..decision_count)
        .min_by(|left, right| {
            expected_decision_loss(*left, posterior, hypotheses)
                .total_cmp(&expected_decision_loss(*right, posterior, hypotheses))
        })
        .unwrap_or(0)
}

fn expected_decision_loss(
    decision_index: usize,
    posterior: &[f64],
    hypotheses: &[DecisionHypothesis],
) -> f64 {
    hypotheses
        .iter()
        .zip(posterior)
        .map(|(hypothesis, probability)| {
            probability * hypothesis.loss_by_final_decision[decision_index]
        })
        .sum()
}

fn categorical_entropy(probabilities: &[f64]) -> f64 {
    -probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f64>()
}

fn multiclass_brier_score(probabilities: &[f64], true_index: usize) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, probability)| {
            let target = if index == true_index { 1.0 } else { 0.0 };
            (probability - target).powi(2)
        })
        .sum()
}

fn sample_categorical(probabilities: &[f64], rng: &mut DeterministicRng) -> usize {
    let draw = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irrelevant_vs_relevant_scenario() -> DiagnosticScenario {
        let mut hypotheses = Vec::new();
        let mut truth = Vec::new();
        for index in 0..4 {
            let fault_a = index < 2;
            let nuisance_zero = index % 2 == 0;
            let nuisance_distribution = if nuisance_zero {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
            let fault_distribution = if fault_a {
                vec![0.7, 0.3]
            } else {
                vec![0.3, 0.7]
            };
            hypotheses.push(DecisionHypothesis {
                id: format!("model-{index}"),
                prior_probability: 0.25,
                outcome_probabilities_by_action: vec![
                    nuisance_distribution.clone(),
                    fault_distribution.clone(),
                ],
                loss_by_final_decision: if fault_a {
                    vec![0.0, 1.0]
                } else {
                    vec![1.0, 0.0]
                },
            });
            truth.push(vec![nuisance_distribution, fault_distribution]);
        }

        DiagnosticScenario {
            scenario_id: "irrelevant-high-eig-vs-relevant-low-eig".to_owned(),
            hypotheses,
            actions: vec![
                DiagnosticAction {
                    id: "inspect-irrelevant-nuisance".into(),
                    observation_cost: 0.0,
                    safety_risk: 0.0,
                },
                DiagnosticAction {
                    id: "inspect-decision-relevant-fault".into(),
                    observation_cost: 0.05,
                    safety_risk: 0.0,
                },
            ],
            true_outcome_probabilities_by_hypothesis_action: truth,
        }
    }

    fn config() -> DecisionDiagnosticConfig {
        DecisionDiagnosticConfig {
            observation_cost_weight: 1.0,
            safety_risk_weight: 0.0,
            maximum_safety_risk: 1.0,
        }
    }

    #[test]
    fn evsi_beats_pure_eig_when_more_information_is_irrelevant() {
        let result = run_diagnostic_policy_benchmark(
            &irrelevant_vs_relevant_scenario(),
            10_000,
            0x5EED,
            config(),
        )
        .expect("benchmark should run");

        let no_probe = &result.policies[0];
        let pure_eig = &result.policies[1];
        let decision_eig = &result.policies[2];
        let evsi = &result.policies[3];
        let always = &result.policies[4];

        assert_eq!(no_probe.probes_taken, 0);
        assert_eq!(pure_eig.selected_action_counts, vec![10_000, 0]);
        assert_eq!(decision_eig.selected_action_counts, vec![0, 10_000]);
        assert_eq!(evsi.selected_action_counts, vec![0, 10_000]);
        assert_eq!(always.selected_action_counts, vec![10_000, 0]);
        assert!(
            pure_eig.mean_predicted_model_eig_nats > decision_eig.mean_predicted_model_eig_nats
        );
        assert!(decision_eig.mean_predicted_decision_eig_nats > 0.0);
        assert!(evsi.mean_predicted_decision_eig_nats > 0.0);
        assert!(evsi.mean_task_loss < pure_eig.mean_task_loss - 0.15);
        assert!(evsi.mean_total_cost < pure_eig.mean_total_cost - 0.10);
        assert!(evsi.mean_decision_regret < pure_eig.mean_decision_regret);
        assert!(evsi.mean_realized_model_entropy_reduction_nats > 0.0);
        assert!((no_probe.mean_task_loss - 0.5).abs() < 0.03);
        assert_eq!(result.paired_evsi_contrasts.len(), 4);
        let evsi_vs_pure_eig = &result.paired_evsi_contrasts[1];
        assert_eq!(evsi_vs_pure_eig.episodes, 10_000);
        assert!(evsi_vs_pure_eig.mean_decision_regret_difference < -0.15);
        let regret_interval = evsi_vs_pure_eig
            .decision_regret_difference_ci95
            .expect("paired interval with 10,000 episodes");
        assert!(regret_interval[1] < 0.0);
        let distribution_free_interval = evsi_vs_pure_eig
            .decision_regret_difference_hoeffding_ci95
            .expect("finite-sample paired interval with 10,000 episodes");
        assert!(distribution_free_interval[0] <= evsi_vs_pure_eig.mean_decision_regret_difference);
        assert!(distribution_free_interval[1] >= evsi_vs_pure_eig.mean_decision_regret_difference);
        assert!(distribution_free_interval[1] < 0.0);
    }

    #[test]
    fn paired_hoeffding_interval_is_finite_sample_and_shrinks_with_more_episodes() {
        let small = vec![-0.2, -0.1, 0.0, 0.1];
        let large = small.repeat(100);
        let small_interval =
            paired_hoeffding_ci95(&small, 1.0, 0.05).expect("small interval");
        let large_interval =
            paired_hoeffding_ci95(&large, 1.0, 0.05).expect("large interval");
        let familywise_interval =
            paired_hoeffding_ci95(&large, 1.0, 0.05 / 12.0).expect("family-wise interval");
        assert!(small_interval[0] <= 0.0 && small_interval[1] >= 0.0);
        assert!(large_interval[0] <= 0.0 && large_interval[1] >= 0.0);
        let small_width = small_interval[1] - small_interval[0];
        let large_width = large_interval[1] - large_interval[0];
        assert!(large_width < small_width);
        assert!(familywise_interval[0] <= 0.0 && familywise_interval[1] >= 0.0);
        assert!(
            familywise_interval[1] - familywise_interval[0] >= large_width
        );
        assert!(paired_hoeffding_ci95(&[], 1.0, 0.05).is_none());
        assert!(paired_hoeffding_ci95(&[1.1], 1.0, 0.05).is_none());
        assert!(paired_hoeffding_ci95(&[0.0], f64::NAN, 0.05).is_none());
        assert!(paired_hoeffding_ci95(&[0.0], 1.0, 0.0).is_none());
    }

    #[test]
    fn targeted_decision_information_is_not_a_substitute_for_cost_aware_evsi() {
        let mut scenario = irrelevant_vs_relevant_scenario();
        scenario.actions[1].observation_cost = 0.25;
        let result = run_diagnostic_policy_benchmark(&scenario, 10_000, 0xC057, config())
            .expect("cost-sensitive comparison should run");

        let pure_eig = &result.policies[1];
        let decision_eig = &result.policies[2];
        let evsi = &result.policies[3];

        assert_eq!(pure_eig.selected_action_counts, vec![10_000, 0]);
        assert_eq!(decision_eig.selected_action_counts, vec![0, 10_000]);
        assert_eq!(evsi.selected_action_counts, vec![0, 0]);
        assert_eq!(evsi.probes_declined, 10_000);
        assert!(decision_eig.mean_predicted_decision_eig_nats > 0.0);
        assert!(evsi.mean_predicted_decision_eig_nats == 0.0);
        assert!((evsi.mean_task_loss - 0.5).abs() < 0.03);
    }

    #[test]
    fn model_misspecification_is_visible_in_decision_and_brier_metrics() {
        let matched = irrelevant_vs_relevant_scenario();
        let mut misspecified = irrelevant_vs_relevant_scenario();
        for (index, truth_actions) in misspecified
            .true_outcome_probabilities_by_hypothesis_action
            .iter_mut()
            .enumerate()
        {
            truth_actions[1] = if index < 2 {
                vec![0.55, 0.45]
            } else {
                vec![0.45, 0.55]
            };
        }

        let matched_result = run_diagnostic_policy_benchmark(&matched, 20_000, 0xCA11B, config())
            .expect("matched likelihood benchmark");
        let misspecified_result =
            run_diagnostic_policy_benchmark(&misspecified, 20_000, 0xCA11B, config())
                .expect("misspecified likelihood benchmark");
        let matched_evsi = &matched_result.policies[3];
        let misspecified_evsi = &misspecified_result.policies[3];

        assert_eq!(matched_evsi.selected_action_counts, vec![0, 20_000]);
        assert_eq!(misspecified_evsi.selected_action_counts, vec![0, 20_000]);
        assert!(misspecified_evsi.mean_task_loss > matched_evsi.mean_task_loss + 0.10);
        assert!(misspecified_evsi.mean_model_brier_score > matched_evsi.mean_model_brier_score);
    }

    #[test]
    fn paired_policy_results_are_reproducible() {
        let scenario = irrelevant_vs_relevant_scenario();
        let first =
            run_diagnostic_policy_benchmark(&scenario, 512, 42, config()).expect("first benchmark");
        let second = run_diagnostic_policy_benchmark(&scenario, 512, 42, config())
            .expect("second benchmark");
        assert_eq!(first, second);
    }

    #[test]
    fn scenario_rejects_truth_outside_model_support() {
        let mut scenario = irrelevant_vs_relevant_scenario();
        scenario.true_outcome_probabilities_by_hypothesis_action[0][0] = vec![0.0, 1.0];
        assert!(matches!(
            run_diagnostic_policy_benchmark(&scenario, 1, 1, config()),
            Err(DiagnosticBenchmarkError::TruthOutsideModelSupport {
                hypothesis_index: 0,
                action_index: 0,
                outcome_index: 1
            })
        ));
    }
}
