// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired-episode benchmark for myopic versus sequential diagnostic policies.
//!
//! Ground-truth outcome likelihoods are separate from the agent's model
//! likelihoods. Each policy sees the same hidden hypothesis and action-specific
//! random stream for each episode. The sequential policy follows a bounded
//! contingent plan; the myopic policy greedily recomputes one-step EVSI after
//! each observation.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::decision_diagnostic_benchmark::DiagnosticScenario;
use super::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionDiagnosticError, DecisionHypothesis, DiagnosticAction,
    evaluate_decision_focused_diagnostics, posterior_model_probabilities,
};
use super::sequential_diagnostic_planning::{
    SequentialDiagnosticConfig, SequentialDiagnosticError, evaluate_sequential_decision_diagnostics,
};

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const SEED_MIX: u64 = 0x9E3779B97F4A7C15;

/// Policies compared in the sequential diagnostic benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequentialDiagnosticPolicy {
    /// Make the best downstream decision without collecting observations.
    NoProbe,
    /// Recompute and take the best one-step EVSI action after each observation.
    GreedyEvsi,
    /// Follow the bounded finite-horizon EVSI decision tree.
    SequentialEvsi,
    /// Inspect the first safety-eligible remaining action, regardless of value.
    AlwaysInspect,
}

/// Aggregate metrics for one sequential policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticPolicyMetrics {
    pub scenario_id: String,
    pub policy: SequentialDiagnosticPolicy,
    pub episodes: usize,
    pub probes_taken: usize,
    pub mean_probes_per_episode: f64,
    pub selected_action_counts: Vec<usize>,
    pub mean_task_loss: f64,
    pub mean_oracle_task_loss: f64,
    pub mean_decision_regret: f64,
    pub mean_total_cost: f64,
    pub mean_observation_cost: f64,
    pub mean_safety_risk: f64,
    pub mean_final_model_entropy_reduction_nats: f64,
    pub mean_final_model_brier_score: f64,
}

/// Paired comparison; all differences are candidate minus reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticPolicyContrast {
    pub reference_policy: SequentialDiagnosticPolicy,
    pub candidate_policy: SequentialDiagnosticPolicy,
    pub episodes: usize,
    pub mean_task_loss_difference: f64,
    pub task_loss_difference_ci95: Option<[f64; 2]>,
    pub mean_decision_regret_difference: f64,
    pub decision_regret_difference_ci95: Option<[f64; 2]>,
    pub mean_total_cost_difference: f64,
    pub total_cost_difference_ci95: Option<[f64; 2]>,
    pub confidence_interval_method: String,
}

/// Full paired benchmark result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticPolicyBenchmark {
    pub scenario_id: String,
    pub episodes: usize,
    pub seed: u64,
    pub policies: Vec<SequentialDiagnosticPolicyMetrics>,
    pub paired_sequential_evsi_contrasts: Vec<SequentialDiagnosticPolicyContrast>,
}

#[derive(Debug, Clone, Copy)]
struct EpisodeMetrics {
    task_loss: f64,
    decision_regret: f64,
    total_cost: f64,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SequentialDiagnosticBenchmarkError {
    #[error(transparent)]
    Decision(#[from] DecisionDiagnosticError),
    #[error(transparent)]
    Planner(#[from] SequentialDiagnosticError),
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
        "ground-truth outcome distribution is invalid for hypothesis {hypothesis_index}, action {action_index}"
    )]
    InvalidTruthDistribution {
        hypothesis_index: usize,
        action_index: usize,
    },
    #[error(
        "ground-truth probability has no support in the agent model for hypothesis {hypothesis_index}, action {action_index}, outcome {outcome_index}"
    )]
    TruthOutsideModelSupport {
        hypothesis_index: usize,
        action_index: usize,
        outcome_index: usize,
    },
    #[error(
        "selected plan has no continuation branch for observed outcome {outcome_index} on action {action_index}"
    )]
    MissingPlanBranch {
        action_index: usize,
        outcome_index: usize,
    },
    #[error("paired policy episode counts differ: reference {reference}, candidate {candidate}")]
    PairedEpisodeCountMismatch { reference: usize, candidate: usize },
}

/// Run no-probe, greedy EVSI, bounded sequential EVSI, and always-inspect policies.
///
/// All policies use paired hidden-hypothesis draws. Outcome random streams are
/// keyed by episode and action, so policies that choose the same action receive
/// the same stochastic outcome quantile even when their preceding paths differ.
pub fn run_sequential_diagnostic_policy_benchmark(
    scenario: &DiagnosticScenario,
    episodes: usize,
    seed: u64,
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
) -> Result<SequentialDiagnosticPolicyBenchmark, SequentialDiagnosticBenchmarkError> {
    if episodes == 0 {
        return Err(SequentialDiagnosticBenchmarkError::EmptyEpisodeSet);
    }
    validate_scenario(scenario, decision_config)?;

    let policies = [
        SequentialDiagnosticPolicy::NoProbe,
        SequentialDiagnosticPolicy::GreedyEvsi,
        SequentialDiagnosticPolicy::SequentialEvsi,
        SequentialDiagnosticPolicy::AlwaysInspect,
    ];
    let runs = policies
        .into_iter()
        .map(|policy| {
            run_policy(
                scenario,
                policy,
                episodes,
                seed,
                decision_config,
                planning_config,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let policy_metrics = runs
        .iter()
        .map(|(metrics, _)| metrics.clone())
        .collect::<Vec<_>>();
    let sequential = runs
        .iter()
        .find(|(metrics, _)| metrics.policy == SequentialDiagnosticPolicy::SequentialEvsi)
        .expect("the policy suite contains sequential EVSI");
    let reference_policies = [
        SequentialDiagnosticPolicy::NoProbe,
        SequentialDiagnosticPolicy::GreedyEvsi,
        SequentialDiagnosticPolicy::AlwaysInspect,
    ];
    let contrasts = reference_policies
        .into_iter()
        .map(|reference_policy| {
            let reference = runs
                .iter()
                .find(|(metrics, _)| metrics.policy == reference_policy)
                .expect("the policy suite contains every registered reference");
            paired_contrast(reference_policy, &reference.1, &sequential.1)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(SequentialDiagnosticPolicyBenchmark {
        scenario_id: scenario.scenario_id.clone(),
        episodes,
        seed,
        policies: policy_metrics,
        paired_sequential_evsi_contrasts: contrasts,
    })
}

fn validate_scenario(
    scenario: &DiagnosticScenario,
    decision_config: DecisionDiagnosticConfig,
) -> Result<(), SequentialDiagnosticBenchmarkError> {
    if scenario.scenario_id.trim().is_empty() {
        return Err(SequentialDiagnosticBenchmarkError::EmptyScenarioId);
    }
    evaluate_decision_focused_diagnostics(
        &scenario.hypotheses,
        &scenario.actions,
        decision_config,
    )?;

    if scenario
        .true_outcome_probabilities_by_hypothesis_action
        .len()
        != scenario.hypotheses.len()
    {
        return Err(SequentialDiagnosticBenchmarkError::TruthHypothesisCount {
            expected: scenario.hypotheses.len(),
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
            return Err(SequentialDiagnosticBenchmarkError::TruthActionCount {
                hypothesis_index,
                expected: scenario.actions.len(),
                actual: truth_actions.len(),
            });
        }
        for (action_index, truth_outcomes) in truth_actions.iter().enumerate() {
            let model_outcomes = &hypothesis.outcome_probabilities_by_action[action_index];
            if truth_outcomes.len() != model_outcomes.len() {
                return Err(SequentialDiagnosticBenchmarkError::TruthOutcomeCount {
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
                return Err(
                    SequentialDiagnosticBenchmarkError::InvalidTruthDistribution {
                        hypothesis_index,
                        action_index,
                    },
                );
            }
            for (outcome_index, (truth, model)) in
                truth_outcomes.iter().zip(model_outcomes).enumerate()
            {
                if *truth > 0.0 && *model == 0.0 {
                    return Err(
                        SequentialDiagnosticBenchmarkError::TruthOutsideModelSupport {
                            hypothesis_index,
                            action_index,
                            outcome_index,
                        },
                    );
                }
            }
        }
    }
    Ok(())
}

fn run_policy(
    scenario: &DiagnosticScenario,
    policy: SequentialDiagnosticPolicy,
    episodes: usize,
    seed: u64,
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
) -> Result<
    (SequentialDiagnosticPolicyMetrics, Vec<EpisodeMetrics>),
    SequentialDiagnosticBenchmarkError,
> {
    let decision_count = scenario.hypotheses[0].loss_by_final_decision.len();
    let prior = scenario
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();
    let sequential_plan = if policy == SequentialDiagnosticPolicy::SequentialEvsi {
        Some(evaluate_sequential_decision_diagnostics(
            &scenario.hypotheses,
            &scenario.actions,
            decision_config,
            planning_config,
        )?)
    } else {
        None
    };

    let mut probes_taken = 0;
    let mut selected_action_counts = vec![0; scenario.actions.len()];
    let mut task_loss_sum = 0.0;
    let mut oracle_task_loss_sum = 0.0;
    let mut regret_sum = 0.0;
    let mut total_cost_sum = 0.0;
    let mut observation_cost_sum = 0.0;
    let mut safety_risk_sum = 0.0;
    let mut entropy_reduction_sum = 0.0;
    let mut brier_sum = 0.0;
    let mut episode_metrics = Vec::with_capacity(episodes);

    for episode_index in 0..episodes {
        let episode_seed = seed.wrapping_add((episode_index as u64).wrapping_mul(SEED_MIX));
        let mut hidden_rng = DeterministicRng::new(episode_seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        let true_hypothesis = sample_categorical(&prior, &mut hidden_rng);
        let mut posterior = prior.clone();
        let mut observation_cost = 0.0;
        let mut safety_risk = 0.0;
        let mut episode_probes = 0;

        match policy {
            SequentialDiagnosticPolicy::NoProbe => {}
            SequentialDiagnosticPolicy::SequentialEvsi => {
                let plan = sequential_plan
                    .as_ref()
                    .expect("sequential policy has a plan");
                let mut node = &plan.plan;
                while let Some(action_index) = node.selected_action_index {
                    let current_hypotheses =
                        hypotheses_with_priors(&scenario.hypotheses, &posterior);
                    let outcome_index =
                        sample_outcome(scenario, true_hypothesis, action_index, episode_seed);
                    posterior = posterior_model_probabilities(
                        &current_hypotheses,
                        action_index,
                        outcome_index,
                    )?;

                    let action = &scenario.actions[action_index];
                    observation_cost += action.observation_cost;
                    safety_risk += action.safety_risk;
                    selected_action_counts[action_index] += 1;
                    probes_taken += 1;
                    episode_probes += 1;

                    let branch = node
                        .branches
                        .iter()
                        .find(|branch| branch.outcome_index == outcome_index)
                        .ok_or(SequentialDiagnosticBenchmarkError::MissingPlanBranch {
                            action_index,
                            outcome_index,
                        })?;
                    node = &branch.continuation;
                }
            }
            SequentialDiagnosticPolicy::GreedyEvsi => {
                let mut remaining = (0..scenario.actions.len()).collect::<Vec<_>>();
                for _ in 0..planning_config.horizon {
                    if remaining.is_empty() {
                        break;
                    }
                    let current_hypotheses =
                        subset_hypotheses(&scenario.hypotheses, &posterior, &remaining);
                    let current_actions = remaining
                        .iter()
                        .map(|index| scenario.actions[*index].clone())
                        .collect::<Vec<_>>();
                    let evaluation = evaluate_decision_focused_diagnostics(
                        &current_hypotheses,
                        &current_actions,
                        decision_config,
                    )?;
                    let Some(local_index) = evaluation.selected_action_index else {
                        break;
                    };
                    let action_index = remaining.remove(local_index);
                    let outcome_index =
                        sample_outcome(scenario, true_hypothesis, action_index, episode_seed);
                    posterior = posterior_model_probabilities(
                        &current_hypotheses,
                        local_index,
                        outcome_index,
                    )?;

                    let action = &scenario.actions[action_index];
                    observation_cost += action.observation_cost;
                    safety_risk += action.safety_risk;
                    selected_action_counts[action_index] += 1;
                    probes_taken += 1;
                    episode_probes += 1;
                }
            }
            SequentialDiagnosticPolicy::AlwaysInspect => {
                let mut remaining = (0..scenario.actions.len()).collect::<Vec<_>>();
                for _ in 0..planning_config.horizon {
                    let Some(position) = remaining.iter().position(|index| {
                        scenario.actions[*index].safety_risk <= decision_config.maximum_safety_risk
                    }) else {
                        break;
                    };
                    let action_index = remaining.remove(position);
                    let current_hypotheses =
                        hypotheses_with_priors(&scenario.hypotheses, &posterior);
                    let outcome_index =
                        sample_outcome(scenario, true_hypothesis, action_index, episode_seed);
                    posterior = posterior_model_probabilities(
                        &current_hypotheses,
                        action_index,
                        outcome_index,
                    )?;

                    let action = &scenario.actions[action_index];
                    observation_cost += action.observation_cost;
                    safety_risk += action.safety_risk;
                    selected_action_counts[action_index] += 1;
                    probes_taken += 1;
                    episode_probes += 1;
                }
            }
        }

        let selected_decision = best_decision(&posterior, &scenario.hypotheses, decision_count);
        let task_loss =
            scenario.hypotheses[true_hypothesis].loss_by_final_decision[selected_decision];
        let oracle_task_loss = scenario.hypotheses[true_hypothesis]
            .loss_by_final_decision
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let regret = (task_loss - oracle_task_loss).max(0.0);
        let total_cost = task_loss
            + decision_config.observation_cost_weight * observation_cost
            + decision_config.safety_risk_weight * safety_risk;

        task_loss_sum += task_loss;
        oracle_task_loss_sum += oracle_task_loss;
        regret_sum += regret;
        total_cost_sum += total_cost;
        observation_cost_sum += observation_cost;
        safety_risk_sum += safety_risk;
        entropy_reduction_sum += categorical_entropy(&prior) - categorical_entropy(&posterior);
        brier_sum += multiclass_brier_score(&posterior, true_hypothesis);
        episode_metrics.push(EpisodeMetrics {
            task_loss,
            decision_regret: regret,
            total_cost,
        });

        debug_assert!(episode_probes <= planning_config.horizon);
    }

    Ok((
        SequentialDiagnosticPolicyMetrics {
            scenario_id: scenario.scenario_id.clone(),
            policy,
            episodes,
            probes_taken,
            mean_probes_per_episode: probes_taken as f64 / episodes as f64,
            selected_action_counts,
            mean_task_loss: task_loss_sum / episodes as f64,
            mean_oracle_task_loss: oracle_task_loss_sum / episodes as f64,
            mean_decision_regret: regret_sum / episodes as f64,
            mean_total_cost: total_cost_sum / episodes as f64,
            mean_observation_cost: observation_cost_sum / episodes as f64,
            mean_safety_risk: safety_risk_sum / episodes as f64,
            mean_final_model_entropy_reduction_nats: entropy_reduction_sum / episodes as f64,
            mean_final_model_brier_score: brier_sum / episodes as f64,
        },
        episode_metrics,
    ))
}

fn hypotheses_with_priors(
    hypotheses: &[DecisionHypothesis],
    posterior: &[f64],
) -> Vec<DecisionHypothesis> {
    hypotheses
        .iter()
        .zip(posterior)
        .map(|(hypothesis, probability)| {
            let mut updated = hypothesis.clone();
            updated.prior_probability = *probability;
            updated
        })
        .collect()
}

fn subset_hypotheses(
    hypotheses: &[DecisionHypothesis],
    posterior: &[f64],
    remaining_actions: &[usize],
) -> Vec<DecisionHypothesis> {
    hypotheses
        .iter()
        .zip(posterior)
        .map(|(hypothesis, probability)| {
            let mut updated = hypothesis.clone();
            updated.prior_probability = *probability;
            updated.outcome_probabilities_by_action = remaining_actions
                .iter()
                .map(|action_index| {
                    hypothesis.outcome_probabilities_by_action[*action_index].clone()
                })
                .collect();
            updated
        })
        .collect()
}

fn sample_outcome(
    scenario: &DiagnosticScenario,
    true_hypothesis: usize,
    action_index: usize,
    episode_seed: u64,
) -> usize {
    let action_seed =
        episode_seed ^ (action_index as u64).wrapping_mul(SEED_MIX) ^ 0x5A5A_5A5A_5A5A_5A5A;
    let mut rng = DeterministicRng::new(action_seed);
    sample_categorical(
        &scenario.true_outcome_probabilities_by_hypothesis_action[true_hypothesis][action_index],
        &mut rng,
    )
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

fn paired_contrast(
    reference_policy: SequentialDiagnosticPolicy,
    reference: &[EpisodeMetrics],
    candidate: &[EpisodeMetrics],
) -> Result<SequentialDiagnosticPolicyContrast, SequentialDiagnosticBenchmarkError> {
    if reference.len() != candidate.len() {
        return Err(
            SequentialDiagnosticBenchmarkError::PairedEpisodeCountMismatch {
                reference: reference.len(),
                candidate: candidate.len(),
            },
        );
    }
    let task_differences = reference
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
        paired_mean_ci95(&task_differences);
    let (mean_decision_regret_difference, decision_regret_difference_ci95) =
        paired_mean_ci95(&regret_differences);
    let (mean_total_cost_difference, total_cost_difference_ci95) =
        paired_mean_ci95(&cost_differences);

    Ok(SequentialDiagnosticPolicyContrast {
        reference_policy,
        candidate_policy: SequentialDiagnosticPolicy::SequentialEvsi,
        episodes: reference.len(),
        mean_task_loss_difference,
        task_loss_difference_ci95,
        mean_decision_regret_difference,
        decision_regret_difference_ci95,
        mean_total_cost_difference,
        total_cost_difference_ci95,
        confidence_interval_method: "paired_normal_approximation_95".to_owned(),
    })
}

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

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { SEED_MIX } else { seed },
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

    fn complementary_probe_scenario() -> DiagnosticScenario {
        let mut hypotheses = Vec::new();
        let mut truth = Vec::new();
        for nuisance in 0..2 {
            for first_bit in 0..2 {
                for second_bit in 0..2 {
                    let xor = first_bit ^ second_bit;
                    let nuisance_outcomes = if nuisance == 0 {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    };
                    let first_outcomes = if first_bit == 0 {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    };
                    let second_outcomes = if second_bit == 0 {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    };
                    let action_outcomes = vec![
                        nuisance_outcomes.clone(),
                        first_outcomes.clone(),
                        second_outcomes.clone(),
                    ];
                    hypotheses.push(DecisionHypothesis {
                        id: format!("fault-{nuisance}-{first_bit}-{second_bit}"),
                        prior_probability: 0.125,
                        outcome_probabilities_by_action: action_outcomes,
                        loss_by_final_decision: if xor == 0 {
                            vec![0.0, 1.0]
                        } else {
                            vec![1.0, 0.0]
                        },
                    });
                    truth.push(vec![nuisance_outcomes, first_outcomes, second_outcomes]);
                }
            }
        }

        DiagnosticScenario {
            scenario_id: "complementary-probes-with-irrelevant-high-eig-nuisance".into(),
            hypotheses,
            actions: vec![
                DiagnosticAction {
                    id: "inspect-irrelevant-nuisance".into(),
                    observation_cost: 0.05,
                    safety_risk: 0.0,
                },
                DiagnosticAction {
                    id: "reveal-first-fault-bit".into(),
                    observation_cost: 0.05,
                    safety_risk: 0.0,
                },
                DiagnosticAction {
                    id: "reveal-second-fault-bit".into(),
                    observation_cost: 0.05,
                    safety_risk: 0.0,
                },
            ],
            true_outcome_probabilities_by_hypothesis_action: truth,
        }
    }

    fn decision_config() -> DecisionDiagnosticConfig {
        DecisionDiagnosticConfig {
            observation_cost_weight: 1.0,
            safety_risk_weight: 0.0,
            maximum_safety_risk: 1.0,
        }
    }

    #[test]
    fn sequential_evsi_beats_myopic_and_always_inspect_on_complementary_probes() {
        let result = run_sequential_diagnostic_policy_benchmark(
            &complementary_probe_scenario(),
            4096,
            0x5EED,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 4096,
            },
        )
        .expect("paired sequential policy benchmark");

        let no_probe = &result.policies[0];
        let greedy = &result.policies[1];
        let sequential = &result.policies[2];
        let always = &result.policies[3];

        assert_eq!(no_probe.probes_taken, 0);
        assert_eq!(greedy.probes_taken, 0);
        assert_eq!(sequential.probes_taken, 8192);
        assert_eq!(always.probes_taken, 8192);

        assert_eq!(greedy.selected_action_counts, vec![0, 0, 0]);
        assert_eq!(sequential.selected_action_counts, vec![0, 4096, 4096]);
        assert_eq!(always.selected_action_counts, vec![4096, 4096, 0]);

        assert!(sequential.mean_task_loss < 1e-12);
        assert!(sequential.mean_task_loss < greedy.mean_task_loss - 0.4);
        assert!(sequential.mean_task_loss < no_probe.mean_task_loss - 0.4);
        assert!(sequential.mean_task_loss < always.mean_task_loss - 0.4);
        assert!(sequential.mean_total_cost < always.mean_total_cost - 0.4);
        assert_eq!(result.paired_sequential_evsi_contrasts.len(), 3);
        assert!(
            result.paired_sequential_evsi_contrasts[0]
                .task_loss_difference_ci95
                .expect("paired interval")
                .iter()
                .all(|bound| *bound < 0.0)
        );
    }
}
