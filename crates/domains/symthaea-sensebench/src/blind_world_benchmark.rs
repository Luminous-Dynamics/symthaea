// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Outcome-generating evaluation of active measurement selection under hidden
//! likelihood misspecification.
//!
//! The policy receives declared likelihood scenarios and its evolving posterior.
//! A separate evaluator path uses hidden-world likelihoods to sample outcomes and
//! score decisions. "Held out" means excluded from policy model construction; this
//! in-process benchmark is not a cryptographic sandbox or physical evidence.

use serde::{Deserialize, Serialize};

use crate::experiment_design::{
    ExperimentCandidate, ExperimentDesignError, HypothesisPrior, ModelLikelihoodScenario,
    rank_experiments, rank_experiments_robust, update_posterior,
};

const HYPOTHESIS_IDS: [&str; 3] = [
    "direct_cause",
    "shared_cause_confounder",
    "no_direct_causal_effect",
];
const MAX_TRIALS_PER_WORLD: usize = 100_000;
const MAX_RESOURCE_BUDGET: usize = 256;
const WORLD_VARIANTS_PER_REGIME: usize = 4;
const MODEL_ADEQUACY_QUANTILE: f64 = 0.99;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlindWorldPolicy {
    NominalExpectedGain,
    NominalPredictiveVariance,
    ModelWeightedExpectedGain,
    WorstCaseRobust,
    DevelopmentFixedDesign,
    UniformRandom,
    PassiveOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlindWorldRegime {
    NominalCalibration,
    ConfounderCalibration,
    InterventionArtifact,
    SensorDegradation,
    EffectReversal,
    NullNegativeControl,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindWorldPolicySummary {
    pub world_id: String,
    pub regime: BlindWorldRegime,
    pub held_out_from_policy_models: bool,
    pub policy: BlindWorldPolicy,
    pub trials: usize,
    pub resource_budget_units: usize,
    pub mean_brier_score: f64,
    pub brier_score_ci95: [f64; 2],
    pub mean_log_loss: f64,
    pub log_loss_ci95: [f64; 2],
    /// Mean prequential negative log likelihood assigned to observed outcomes
    /// by the declared likelihood-model ensemble, before updating the posterior.
    pub mean_prequential_predictive_log_loss: f64,
    pub prequential_predictive_log_loss_ci95: [f64; 2],
    /// Rate of trials whose prequential predictive loss exceeds the cutoff
    /// calibrated from independent-seed simulations under declared model scenarios.
    pub model_adequacy_alarm_rate: f64,
    pub model_adequacy_alarm_wilson_ci95: [f64; 2],
    /// Fraction of trials not abstained after the model-adequacy check.
    pub accepted_trial_fraction_after_adequacy_gate: f64,
    /// Null-truth unsupported causal claims suppressed by the adequacy gate,
    /// divided by all null-truth trials.
    pub adequacy_gated_unsupported_causal_claim_rate_on_null_truth: f64,
    /// False causal claim fraction among null-truth trials that passed the gate.
    pub false_causal_claim_rate_among_accepted_null_trials: Option<f64>,
    pub model_adequacy_threshold_mean_nll: f64,
    pub top_hypothesis_accuracy: f64,
    pub top_hypothesis_accuracy_wilson_ci95: [f64; 2],
    /// Expected gain predicted by the declared model for the chosen actions.
    pub mean_expected_information_gain_bits_per_cost: f64,
    /// Evaluator-world expected gain for chosen actions, based on hidden likelihoods.
    pub mean_hidden_world_expected_information_gain_bits_per_cost: f64,
    /// Per-selected-action regret versus the best currently affordable candidate
    /// under the evaluator-world likelihood table.
    pub mean_acquisition_regret_bits_per_selection: f64,
    /// Can be negative: a surprising outcome can increase posterior entropy.
    pub mean_realized_entropy_reduction_bits_per_cost: f64,
    pub mean_calibration_error_10_bins: f64,
    pub threshold_reached_fraction: f64,
    pub threshold_reached_wilson_ci95: [f64; 2],
    pub mean_experiments_used: f64,
    pub mean_cost_spent: f64,
    pub mean_experiments_to_threshold: Option<f64>,
    pub mean_cost_to_threshold: Option<f64>,
    /// Among trials where the true explanation is no direct causal effect,
    /// fraction where a direct/shared causal explanation crosses the threshold.
    pub unsupported_causal_claim_rate_on_null_truth: f64,
    pub unsupported_causal_claim_rate_wilson_ci95: [f64; 2],
    pub null_truth_trials: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAdequacyThreshold {
    pub policy: BlindWorldPolicy,
    pub calibration_quantile: f64,
    pub threshold_mean_prequential_nll: f64,
    pub independent_calibration_trials: usize,
    pub calibration_source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindWorldBenchmarkReport {
    pub schema_version: u16,
    pub generator_version: String,
    pub seed: u64,
    pub trials_per_world: usize,
    pub resource_budget_units: usize,
    pub world_variants_per_regime: usize,
    pub posterior_threshold: f64,
    pub model_adequacy_thresholds: Vec<ModelAdequacyThreshold>,
    pub declared_model_scenarios: Vec<String>,
    pub development_fixed_candidate_id: String,
    pub world_count: usize,
    pub held_out_world_count: usize,
    pub summaries: Vec<BlindWorldPolicySummary>,
    pub interpretation_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BlindWorldBenchmarkError {
    #[error("trials per world must be between 3 and MAX_TRIALS_PER_WORLD")]
    InvalidTrialCount,
    #[error("resource budget must be between 1 and MAX_RESOURCE_BUDGET units")]
    InvalidResourceBudget,
    #[error("posterior threshold must be finite and in (1/3, 1]")]
    InvalidPosteriorThreshold,
    #[error("benchmark candidate library is empty")]
    EmptyCandidateLibrary,
    #[error("a selected candidate is absent from a declared or hidden candidate set")]
    MissingCandidate,
    #[error(transparent)]
    InvalidExperimentDesign(#[from] ExperimentDesignError),
}

/// Evaluate seven measurement-selection strategies over two development regimes
/// and four held-out regimes. All strategies receive the same prior, candidate
/// identifiers/costs, posterior update rule, and total resource budget. Truth is
/// balanced as closely as possible across the three hypotheses in every regime.
///
/// This benchmark samples synthetic outcomes; unlike the closed-form robustness
/// fixture, it measures realized posterior quality, calibration, cost-to-threshold,
/// and the gap between predicted information gain and realized entropy reduction.
/// Its hidden tables are code-visible and must not be described as securely sealed.
pub fn run_blind_world_benchmark(
    trials_per_world: usize,
    resource_budget_units: usize,
    seed: u64,
    posterior_threshold: f64,
) -> Result<BlindWorldBenchmarkReport, BlindWorldBenchmarkError> {
    if !(3..=MAX_TRIALS_PER_WORLD).contains(&trials_per_world) {
        return Err(BlindWorldBenchmarkError::InvalidTrialCount);
    }
    if !(1..=MAX_RESOURCE_BUDGET).contains(&resource_budget_units) {
        return Err(BlindWorldBenchmarkError::InvalidResourceBudget);
    }
    if !posterior_threshold.is_finite()
        || posterior_threshold <= 1.0 / 3.0
        || posterior_threshold > 1.0
    {
        return Err(BlindWorldBenchmarkError::InvalidPosteriorThreshold);
    }

    let priors = uniform_priors();
    let model_scenarios = declared_model_scenarios();
    let worlds = hidden_worlds(seed);
    let development_worlds: Vec<HiddenWorld> = worlds.iter()
        .filter(|world| !world.held_out)
        .cloned()
        .collect();
    let fixed_candidate_id = choose_development_fixed_design(&priors, &development_worlds)?;
    let policies = [
        BlindWorldPolicy::NominalExpectedGain,
        BlindWorldPolicy::NominalPredictiveVariance,
        BlindWorldPolicy::ModelWeightedExpectedGain,
        BlindWorldPolicy::WorstCaseRobust,
        BlindWorldPolicy::DevelopmentFixedDesign,
        BlindWorldPolicy::UniformRandom,
        BlindWorldPolicy::PassiveOnly,
    ];
    let model_adequacy_thresholds = calibrate_model_adequacy_thresholds(
        trials_per_world,
        resource_budget_units,
        seed,
        posterior_threshold,
        &priors,
        &model_scenarios,
        &fixed_candidate_id,
        &policies,
    )?;

    let mut summaries = Vec::with_capacity(worlds.len() * policies.len());
    for (world_index, world) in worlds.iter().enumerate() {
        for policy in policies {
            let mut trials = Vec::with_capacity(trials_per_world);
            for trial_index in 0..trials_per_world {
                let trial_seed = seed
                    ^ (world_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ (trial_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                trials.push(simulate_trial(
                    policy,
                    &priors,
                    &model_scenarios,
                    &world.candidates,
                    &fixed_candidate_id,
                    resource_budget_units,
                    posterior_threshold,
                    trial_index % HYPOTHESIS_IDS.len(),
                    trial_seed,
                )?);
            }
            let adequacy_threshold = model_adequacy_thresholds.iter()
                .find(|entry| entry.policy == policy)
                .map(|entry| entry.threshold_mean_prequential_nll)
                .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
            summaries.push(summarize_world_policy(
                world,
                policy,
                resource_budget_units,
                posterior_threshold,
                adequacy_threshold,
                trials,
            ));
        }
    }

    Ok(BlindWorldBenchmarkReport {
        schema_version: 2,
        generator_version: "sensebench-blind-worlds-0.2.0".into(),
        seed,
        trials_per_world,
        resource_budget_units,
        world_variants_per_regime: WORLD_VARIANTS_PER_REGIME,
        posterior_threshold,
        model_adequacy_thresholds,
        declared_model_scenarios: model_scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.clone())
            .collect(),
        development_fixed_candidate_id: fixed_candidate_id,
        world_count: worlds.len(),
        held_out_world_count: worlds.iter().filter(|world| world.held_out).count(),
        summaries,
        interpretation_boundary: concat!(
            "Synthetic software evaluation only. Held-out regimes are excluded from ",
            "policy model construction, but their definitions remain visible in source; ",
            "this is not cryptographic blinding, calibrated physical evidence, or proof ",
            "of generalization to real instruments."
        ).into(),
    })
}

/// Calibrate a per-policy lack-of-fit cutoff using independent-seed outcomes
/// generated by the declared model scenarios only. This is a synthetic
/// prequential-score threshold, not a universal goodness-of-fit test.
fn calibrate_model_adequacy_thresholds(
    trials_per_scenario: usize,
    resource_budget_units: usize,
    seed: u64,
    posterior_threshold: f64,
    priors: &[HypothesisPrior],
    model_scenarios: &[ModelLikelihoodScenario],
    fixed_candidate_id: &str,
    policies: &[BlindWorldPolicy],
) -> Result<Vec<ModelAdequacyThreshold>, BlindWorldBenchmarkError> {
    let mut thresholds = Vec::with_capacity(policies.len());
    for policy in policies.iter().copied() {
        let mut calibration_scores = Vec::with_capacity(
            trials_per_scenario.saturating_mul(model_scenarios.len()),
        );
        for (scenario_index, scenario) in model_scenarios.iter().enumerate() {
            for trial_index in 0..trials_per_scenario {
                // A disjoint stream from the evaluated hidden worlds. The same
                // calibration seed tuple is reused across policies for pairing.
                let calibration_seed = seed
                    ^ 0xCA11_BA7E_2026_9919
                    ^ (scenario_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ (trial_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let result = simulate_trial(
                    policy,
                    priors,
                    model_scenarios,
                    &scenario.candidates,
                    fixed_candidate_id,
                    resource_budget_units,
                    posterior_threshold,
                    trial_index % priors.len(),
                    calibration_seed,
                )?;
                calibration_scores.push(result.mean_prequential_predictive_nll);
            }
        }
        calibration_scores.sort_by(f64::total_cmp);
        let quantile_index = ((calibration_scores.len() as f64
            * MODEL_ADEQUACY_QUANTILE).ceil() as usize)
            .saturating_sub(1)
            .min(calibration_scores.len().saturating_sub(1));
        let cutoff = calibration_scores.get(quantile_index).copied()
            .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
        thresholds.push(ModelAdequacyThreshold {
            policy,
            calibration_quantile: MODEL_ADEQUACY_QUANTILE,
            threshold_mean_prequential_nll: cutoff,
            independent_calibration_trials: calibration_scores.len(),
            calibration_source: "declared_model_scenarios_independent_seed".into(),
        });
    }
    Ok(thresholds)
}

#[derive(Debug, Clone, PartialEq)]
struct HiddenWorld {
    id: String,
    regime: BlindWorldRegime,
    held_out: bool,
    candidates: Vec<ExperimentCandidate>,
}

#[derive(Debug, Clone)]
struct TrialResult {
    brier: f64,
    log_loss: f64,
    top_correct: bool,
    top_confidence: f64,
    threshold_reached: bool,
    experiments_used: usize,
    cost_spent: f64,
    steps_to_threshold: Option<usize>,
    cost_to_threshold: Option<f64>,
    expected_gain_bits_per_cost: f64,
    hidden_world_gain_bits_per_cost: f64,
    acquisition_regret_bits_per_selection: f64,
    realized_entropy_reduction_bits_per_cost: f64,
    mean_prequential_predictive_nll: f64,
    unsupported_causal_claim: bool,
    null_truth: bool,
}

#[allow(clippy::too_many_arguments)]
fn simulate_trial(
    policy: BlindWorldPolicy,
    priors: &[HypothesisPrior],
    model_scenarios: &[ModelLikelihoodScenario],
    hidden_candidates: &[ExperimentCandidate],
    fixed_candidate_id: &str,
    resource_budget_units: usize,
    threshold: f64,
    true_hypothesis_index: usize,
    seed: u64,
) -> Result<TrialResult, BlindWorldBenchmarkError> {
    let mut posterior = priors.to_vec();
    let mut outcome_rng = BenchmarkRng::new(seed ^ 0x0B5E_7A71);
    let mut selection_rng = BenchmarkRng::new(seed ^ 0x51EC_7100);
    let mut cost_spent = 0.0;
    let mut experiments_used = 0usize;
    let mut expected_gain_sum = 0.0;
    let mut hidden_world_gain_sum = 0.0;
    let mut acquisition_regret_sum = 0.0;
    let mut realized_gain_sum = 0.0;
    let mut predictive_nll_sum = 0.0;
    let mut gain_cost_sum = 0.0;
    let mut steps_to_threshold = None;
    let mut cost_to_threshold = None;

    for step in 1..=resource_budget_units {
        let remaining = resource_budget_units as f64 - cost_spent;
        let affordable_models: Vec<ModelLikelihoodScenario> = model_scenarios
            .iter()
            .map(|scenario| ModelLikelihoodScenario {
                scenario_id: scenario.scenario_id.clone(),
                weight: scenario.weight,
                candidates: scenario.candidates.iter()
                    .filter(|candidate| candidate.cost <= remaining + 1.0e-9)
                    .cloned()
                    .collect(),
            })
            .collect();
        if affordable_models.first().is_none_or(|scenario| scenario.candidates.is_empty()) {
            break;
        }

        // The selector API receives no hidden-world identifier, likelihood table,
        // or true hypothesis index.
        let selected_id = choose_candidate(
            policy,
            &posterior,
            &affordable_models,
            fixed_candidate_id,
            &mut selection_rng,
        )?;
        let Some(selected_id) = selected_id else {
            break;
        };
        let model_candidate = affordable_models[0].candidates.iter()
            .find(|candidate| candidate.id == selected_id)
            .ok_or(BlindWorldBenchmarkError::MissingCandidate)?;
        let affordable_hidden_candidates: Vec<ExperimentCandidate> = hidden_candidates.iter()
            .filter(|candidate| candidate.cost <= remaining + 1.0e-9)
            .cloned()
            .collect();
        let hidden_candidate = affordable_hidden_candidates.iter()
            .find(|candidate| candidate.id == selected_id)
            .ok_or(BlindWorldBenchmarkError::MissingCandidate)?;

        let expected_score = rank_experiments(
            &posterior,
            std::slice::from_ref(model_candidate),
        )?.into_iter().next()
            .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
        let selected_hidden_score = rank_experiments(
            &posterior,
            std::slice::from_ref(hidden_candidate),
        )?.into_iter().next()
            .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
        let hidden_oracle = rank_experiments(
            &posterior,
            &affordable_hidden_candidates,
        )?.into_iter().next()
            .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
        acquisition_regret_sum += (
            hidden_oracle.information_gain_per_cost
                - selected_hidden_score.information_gain_per_cost
        ).max(0.0);
        hidden_world_gain_sum += selected_hidden_score.information_gain_per_cost;
        let likelihood_row = &hidden_candidate.likelihoods[true_hypothesis_index];
        let outcome_index = if outcome_rng.unit_f64() < likelihood_row[0] { 0 } else { 1 };
        // Every strategy updates with the same declared nominal model. This keeps
        // the experiment-choice comparison interpretable; the robust policies are
        // not silently granted a separate posterior-inference advantage.
        let update = update_posterior(&posterior, model_candidate, outcome_index)?;
        let ensemble_outcome_probability = ensemble_predictive_outcome_probability(
            &posterior,
            &affordable_models,
            &selected_id,
            outcome_index,
        )?;
        predictive_nll_sum += -ensemble_outcome_probability.max(1.0e-12).ln();
        posterior = update.posterior;

        experiments_used += 1;
        cost_spent += model_candidate.cost;
        expected_gain_sum += expected_score.expected_information_gain_bits;
        realized_gain_sum += update.realized_information_gain_bits;
        gain_cost_sum += model_candidate.cost;

        if top_confidence(&posterior) >= threshold {
            steps_to_threshold = Some(step);
            cost_to_threshold = Some(cost_spent);
            break;
        }
    }

    let true_probabilities: Vec<f64> = posterior.iter()
        .map(|hypothesis| hypothesis.probability)
        .collect();
    let mut brier = 0.0;
    for (index, probability) in true_probabilities.iter().enumerate() {
        let target = if index == true_hypothesis_index { 1.0 } else { 0.0 };
        brier += (probability - target).powi(2);
    }
    let true_probability = true_probabilities[true_hypothesis_index].clamp(1.0e-12, 1.0);
    let most_likely = top_index(&posterior);
    let confidence = posterior[most_likely].probability;
    let unsupported_causal_claim = true_hypothesis_index == 2
        && confidence >= threshold
        && (most_likely == 0 || most_likely == 1);

    Ok(TrialResult {
        brier,
        log_loss: -true_probability.ln(),
        top_correct: most_likely == true_hypothesis_index,
        top_confidence: confidence,
        threshold_reached: steps_to_threshold.is_some(),
        experiments_used,
        cost_spent,
        steps_to_threshold,
        cost_to_threshold,
        expected_gain_bits_per_cost: if gain_cost_sum > 0.0 {
            expected_gain_sum / gain_cost_sum
        } else {
            0.0
        },
        hidden_world_gain_bits_per_cost: if experiments_used > 0 {
            hidden_world_gain_sum / experiments_used as f64
        } else {
            0.0
        },
        acquisition_regret_bits_per_selection: if experiments_used > 0 {
            acquisition_regret_sum / experiments_used as f64
        } else {
            0.0
        },
        realized_entropy_reduction_bits_per_cost: if gain_cost_sum > 0.0 {
            realized_gain_sum / gain_cost_sum
        } else {
            0.0
        },
        mean_prequential_predictive_nll: if experiments_used > 0 {
            predictive_nll_sum / experiments_used as f64
        } else {
            0.0
        },
        unsupported_causal_claim,
        null_truth: true_hypothesis_index == 2,
    })
}

fn ensemble_predictive_outcome_probability(
    posterior: &[HypothesisPrior],
    model_scenarios: &[ModelLikelihoodScenario],
    candidate_id: &str,
    outcome_index: usize,
) -> Result<f64, BlindWorldBenchmarkError> {
    let total_weight = model_scenarios.iter().map(|scenario| scenario.weight).sum::<f64>();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(BlindWorldBenchmarkError::EmptyCandidateLibrary);
    }
    let mut mixture_probability = 0.0;
    for scenario in model_scenarios {
        let candidate = scenario.candidates.iter()
            .find(|candidate| candidate.id == candidate_id)
            .ok_or(BlindWorldBenchmarkError::MissingCandidate)?;
        let outcome_count = candidate.likelihoods.first()
            .map(Vec::len)
            .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
        if outcome_index >= outcome_count {
            return Err(BlindWorldBenchmarkError::MissingCandidate);
        }
        let scenario_probability = posterior.iter()
            .zip(&candidate.likelihoods)
            .map(|(hypothesis, row)| hypothesis.probability * row[outcome_index])
            .sum::<f64>();
        mixture_probability += (scenario.weight / total_weight) * scenario_probability;
    }
    Ok(mixture_probability.clamp(0.0, 1.0))
}

fn choose_candidate(
    policy: BlindWorldPolicy,
    posterior: &[HypothesisPrior],
    model_scenarios: &[ModelLikelihoodScenario],
    fixed_candidate_id: &str,
    selection_rng: &mut BenchmarkRng,
) -> Result<Option<String>, BlindWorldBenchmarkError> {
    let nominal = model_scenarios.first()
        .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
    if nominal.candidates.is_empty() {
        return Ok(None);
    }
    let selected = match policy {
        BlindWorldPolicy::NominalExpectedGain => rank_experiments(
            posterior, &nominal.candidates,
        )?.first().map(|score| score.candidate_id.clone()),
        BlindWorldPolicy::NominalPredictiveVariance => nominal.candidates.iter()
            .max_by(|left, right| {
                (predictive_variance(left, posterior) / left.cost)
                    .total_cmp(&(predictive_variance(right, posterior) / right.cost))
                    .then_with(|| right.id.cmp(&left.id))
            })
            .map(|candidate| candidate.id.clone()),
        BlindWorldPolicy::ModelWeightedExpectedGain => {
            let scores = rank_experiments_robust(posterior, model_scenarios)?;
            scores.into_iter().max_by(|left, right| {
                left.expected_information_gain_per_cost
                    .total_cmp(&right.expected_information_gain_per_cost)
                    .then_with(|| right.candidate_id.cmp(&left.candidate_id))
            }).map(|score| score.candidate_id)
        }
        BlindWorldPolicy::WorstCaseRobust => rank_experiments_robust(
            posterior, model_scenarios,
        )?.first().map(|score| score.candidate_id.clone()),
        BlindWorldPolicy::DevelopmentFixedDesign => nominal.candidates.iter()
            .find(|candidate| candidate.id == fixed_candidate_id)
            .map(|candidate| candidate.id.clone()),
        BlindWorldPolicy::UniformRandom => {
            let index = selection_rng.index(nominal.candidates.len());
            nominal.candidates.get(index).map(|candidate| candidate.id.clone())
        }
        BlindWorldPolicy::PassiveOnly => nominal.candidates.iter()
            .find(|candidate| candidate.id == "passive_correlation")
            .map(|candidate| candidate.id.clone()),
    };
    Ok(selected)
}

fn predictive_variance(candidate: &ExperimentCandidate, priors: &[HypothesisPrior]) -> f64 {
    let mean = priors.iter()
        .zip(&candidate.likelihoods)
        .map(|(prior, row)| prior.probability * row[0])
        .sum::<f64>();
    priors.iter()
        .zip(&candidate.likelihoods)
        .map(|(prior, row)| prior.probability * (row[0] - mean).powi(2))
        .sum::<f64>()
}

fn choose_development_fixed_design(
    priors: &[HypothesisPrior],
    development_worlds: &[HiddenWorld],
) -> Result<String, BlindWorldBenchmarkError> {
    let first = development_worlds.first()
        .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)?;
    let mut totals: Vec<(String, f64)> = first.candidates.iter()
        .map(|candidate| (candidate.id.clone(), 0.0))
        .collect();
    for world in development_worlds {
        let ranking = rank_experiments(priors, &world.candidates)?;
        for (candidate_id, total) in &mut totals {
            let score = ranking.iter().find(|score| score.candidate_id == *candidate_id)
                .ok_or(BlindWorldBenchmarkError::MissingCandidate)?;
            *total += score.information_gain_per_cost / development_worlds.len() as f64;
        }
    }
    totals.into_iter()
        .max_by(|(left_id, left_score), (right_id, right_score)| {
            left_score.total_cmp(right_score).then_with(|| right_id.cmp(left_id))
        })
        .map(|(candidate_id, _)| candidate_id)
        .ok_or(BlindWorldBenchmarkError::EmptyCandidateLibrary)
}

fn summarize_world_policy(
    world: &HiddenWorld,
    policy: BlindWorldPolicy,
    resource_budget_units: usize,
    _threshold: f64,
    adequacy_threshold: f64,
    trials: Vec<TrialResult>,
) -> BlindWorldPolicySummary {
    let brier: Vec<f64> = trials.iter().map(|trial| trial.brier).collect();
    let log_loss: Vec<f64> = trials.iter().map(|trial| trial.log_loss).collect();
    let predictive_nll: Vec<f64> = trials.iter()
        .map(|trial| trial.mean_prequential_predictive_nll).collect();
    let count = trials.len();
    let correct = trials.iter().filter(|trial| trial.top_correct).count();
    let reached = trials.iter().filter(|trial| trial.threshold_reached).count();
    let experiments_to_threshold: Vec<f64> = trials.iter()
        .filter_map(|trial| trial.steps_to_threshold.map(|steps| steps as f64))
        .collect();
    let cost_to_threshold: Vec<f64> = trials.iter()
        .filter_map(|trial| trial.cost_to_threshold)
        .collect();
    let null_truth_trials: Vec<&TrialResult> = trials.iter()
        .filter(|trial| trial.null_truth)
        .collect();
    let unsupported = null_truth_trials.iter()
        .filter(|trial| trial.unsupported_causal_claim)
        .count();
    let alarms = trials.iter()
        .filter(|trial| trial.mean_prequential_predictive_nll > adequacy_threshold)
        .count();
    let accepted = count.saturating_sub(alarms);
    let accepted_null_trials: Vec<&TrialResult> = null_truth_trials.iter().copied()
        .filter(|trial| trial.mean_prequential_predictive_nll <= adequacy_threshold)
        .collect();
    let accepted_false_claims = accepted_null_trials.iter()
        .filter(|trial| trial.unsupported_causal_claim)
        .count();
    let ece = calibration_error_10_bins(&trials);
    let mean = |values: &[f64]| -> f64 {
        if values.is_empty() { 0.0 } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };

    BlindWorldPolicySummary {
        world_id: world.id.clone(),
        regime: world.regime,
        held_out_from_policy_models: world.held_out,
        policy,
        trials: count,
        resource_budget_units,
        mean_brier_score: mean(&brier),
        brier_score_ci95: mean_ci95(&brier, 0.0, 2.0),
        mean_log_loss: mean(&log_loss),
        log_loss_ci95: mean_ci95(&log_loss, 0.0, 30.0),
        mean_prequential_predictive_log_loss: mean(&predictive_nll),
        prequential_predictive_log_loss_ci95: mean_ci95(&predictive_nll, 0.0, 30.0),
        model_adequacy_alarm_rate: alarms as f64 / count as f64,
        model_adequacy_alarm_wilson_ci95: wilson_ci95(alarms, count),
        accepted_trial_fraction_after_adequacy_gate: accepted as f64 / count as f64,
        adequacy_gated_unsupported_causal_claim_rate_on_null_truth: if null_truth_trials.is_empty() {
            0.0
        } else {
            accepted_false_claims as f64 / null_truth_trials.len() as f64
        },
        false_causal_claim_rate_among_accepted_null_trials: (!accepted_null_trials.is_empty())
            .then_some(accepted_false_claims as f64 / accepted_null_trials.len() as f64),
        model_adequacy_threshold_mean_nll: adequacy_threshold,
        top_hypothesis_accuracy: correct as f64 / count as f64,
        top_hypothesis_accuracy_wilson_ci95: wilson_ci95(correct, count),
        mean_expected_information_gain_bits_per_cost: mean(&trials.iter()
            .map(|trial| trial.expected_gain_bits_per_cost).collect::<Vec<_>>()),
        mean_hidden_world_expected_information_gain_bits_per_cost: mean(&trials.iter()
            .map(|trial| trial.hidden_world_gain_bits_per_cost).collect::<Vec<_>>()),
        mean_acquisition_regret_bits_per_selection: mean(&trials.iter()
            .map(|trial| trial.acquisition_regret_bits_per_selection).collect::<Vec<_>>()),
        mean_realized_entropy_reduction_bits_per_cost: mean(&trials.iter()
            .map(|trial| trial.realized_entropy_reduction_bits_per_cost).collect::<Vec<_>>()),
        mean_calibration_error_10_bins: ece,
        threshold_reached_fraction: reached as f64 / count as f64,
        threshold_reached_wilson_ci95: wilson_ci95(reached, count),
        mean_experiments_used: mean(&trials.iter()
            .map(|trial| trial.experiments_used as f64).collect::<Vec<_>>()),
        mean_cost_spent: mean(&trials.iter().map(|trial| trial.cost_spent).collect::<Vec<_>>()),
        mean_experiments_to_threshold: (!experiments_to_threshold.is_empty())
            .then(|| mean(&experiments_to_threshold)),
        mean_cost_to_threshold: (!cost_to_threshold.is_empty()).then(|| mean(&cost_to_threshold)),
        unsupported_causal_claim_rate_on_null_truth: unsupported as f64
            / null_truth_trials.len() as f64,
        unsupported_causal_claim_rate_wilson_ci95: wilson_ci95(unsupported, null_truth_trials.len()),
        null_truth_trials: null_truth_trials.len(),
    }
}

fn calibration_error_10_bins(trials: &[TrialResult]) -> f64 {
    let mut counts = [0usize; 10];
    let mut confidence_sums = [0.0_f64; 10];
    let mut correct_sums = [0usize; 10];
    for trial in trials {
        let bin = ((trial.top_confidence * 10.0).floor() as usize).min(9);
        counts[bin] += 1;
        confidence_sums[bin] += trial.top_confidence;
        if trial.top_correct {
            correct_sums[bin] += 1;
        }
    }
    let n = trials.len() as f64;
    counts.iter().enumerate().map(|(bin, count)| {
        if *count == 0 { 0.0 } else {
            (*count as f64 / n)
                * ((confidence_sums[bin] / *count as f64)
                    - (correct_sums[bin] as f64 / *count as f64)).abs()
        }
    }).sum()
}

fn mean_ci95(values: &[f64], minimum: f64, maximum: f64) -> [f64; 2] {
    if values.len() < 2 {
        return [minimum, maximum];
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / (n - 1.0);
    let half_width = 1.96 * (variance / n).sqrt();
    [
        (mean - half_width).clamp(minimum, maximum),
        (mean + half_width).clamp(minimum, maximum),
    ]
}

fn wilson_ci95(successes: usize, count: usize) -> [f64; 2] {
    if count == 0 {
        return [0.0, 1.0];
    }
    let n = count as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    [(center - half_width).clamp(0.0, 1.0), (center + half_width).clamp(0.0, 1.0)]
}

fn uniform_priors() -> Vec<HypothesisPrior> {
    HYPOTHESIS_IDS.iter().map(|id| HypothesisPrior {
        id: (*id).into(),
        probability: 1.0 / HYPOTHESIS_IDS.len() as f64,
    }).collect()
}

fn binary_candidate(id: &str, cost: f64, probabilities: [f64; 3]) -> ExperimentCandidate {
    ExperimentCandidate {
        id: id.into(),
        cost,
        likelihoods: probabilities.iter().map(|p| vec![*p, 1.0 - *p]).collect(),
    }
}

fn candidate_set(
    passive: [f64; 3],
    intervention: [f64; 3],
    confounder: [f64; 3],
    negative: [f64; 3],
    delayed: [f64; 3],
) -> Vec<ExperimentCandidate> {
    vec![
        binary_candidate("passive_correlation", 1.0, passive),
        binary_candidate("intervene_on_x", 1.5, intervention),
        binary_candidate("measure_confounder", 1.25, confounder),
        binary_candidate("negative_control", 1.0, negative),
        binary_candidate("repeat_after_delay", 1.0, delayed),
    ]
}

fn declared_model_scenarios() -> Vec<ModelLikelihoodScenario> {
    vec![
        ModelLikelihoodScenario {
            scenario_id: "nominal_model_a".into(),
            weight: 0.5,
            candidates: candidate_set(
                [0.82, 0.80, 0.52],
                [0.90, 0.30, 0.50],
                [0.38, 0.84, 0.51],
                [0.50, 0.50, 0.50],
                [0.72, 0.64, 0.55],
            ),
        },
        ModelLikelihoodScenario {
            scenario_id: "alternative_model_b".into(),
            weight: 0.5,
            candidates: candidate_set(
                [0.78, 0.80, 0.58],
                [0.67, 0.48, 0.52],
                [0.45, 0.72, 0.55],
                [0.50, 0.50, 0.50],
                [0.65, 0.69, 0.58],
            ),
        },
    ]
}

fn hidden_worlds(seed: u64) -> Vec<HiddenWorld> {
    let templates = hidden_world_templates();
    let mut generated = Vec::with_capacity(templates.len() * WORLD_VARIANTS_PER_REGIME);
    for (world_index, template) in templates.iter().enumerate() {
        for replicate in 0..WORLD_VARIANTS_PER_REGIME {
            let mut world = template.clone();
            world.id = format!("{}-variant-{replicate:02}", template.id);
            let world_seed = seed
                ^ (world_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                ^ (replicate as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            perturb_world_likelihoods(&mut world, world_seed);
            generated.push(world);
        }
    }
    generated
}

/// Perturb each regime with a versioned deterministic stream while preserving its
/// defining control: null likelihoods remain equal across hypotheses, the
/// intervention-artifact action remains uninformative, and effect reversal remains
/// reversed. Independent world variants enable world-level resampling across runs.
fn perturb_world_likelihoods(world: &mut HiddenWorld, seed: u64) {
    let mut rng = BenchmarkRng::new(seed ^ 0xB11D_2026);
    for candidate in &mut world.candidates {
        if candidate.id == "negative_control"
            || world.regime == BlindWorldRegime::NullNegativeControl
            || (world.regime == BlindWorldRegime::InterventionArtifact
                && candidate.id == "intervene_on_x")
        {
            let base = if candidate.id == "negative_control" {
                0.5
            } else if world.regime == BlindWorldRegime::NullNegativeControl {
                candidate.likelihoods[0][0]
            } else {
                0.58
            };
            let probability = (base + (rng.unit_f64() - 0.5) * 0.08).clamp(0.05, 0.95);
            for row in &mut candidate.likelihoods {
                row[0] = probability;
                row[1] = 1.0 - probability;
            }
            continue;
        }

        let amplitude = match world.regime {
            BlindWorldRegime::SensorDegradation => 0.012,
            BlindWorldRegime::EffectReversal => 0.02,
            _ => 0.035,
        };
        for row in &mut candidate.likelihoods {
            let delta = (rng.unit_f64() - 0.5) * 2.0 * amplitude;
            let probability = (row[0] + delta).clamp(0.05, 0.95);
            row[0] = probability;
            row[1] = 1.0 - probability;
        }
    }
}

fn hidden_world_templates() -> Vec<HiddenWorld> {
    let nominal = candidate_set(
        [0.82, 0.80, 0.52],
        [0.90, 0.30, 0.50],
        [0.38, 0.84, 0.51],
        [0.50, 0.50, 0.50],
        [0.72, 0.64, 0.55],
    );
    let confounder = candidate_set(
        [0.82, 0.82, 0.54],
        [0.82, 0.52, 0.50],
        [0.36, 0.88, 0.52],
        [0.50, 0.50, 0.50],
        [0.71, 0.68, 0.55],
    );
    let intervention_artifact = candidate_set(
        [0.82, 0.80, 0.52],
        [0.58, 0.58, 0.57],
        [0.36, 0.86, 0.52],
        [0.50, 0.50, 0.50],
        [0.71, 0.65, 0.55],
    );
    let base = candidate_set(
        [0.82, 0.80, 0.52],
        [0.90, 0.30, 0.50],
        [0.38, 0.84, 0.51],
        [0.50, 0.50, 0.50],
        [0.72, 0.64, 0.55],
    );
    let degraded = base.iter().map(|candidate| ExperimentCandidate {
        id: candidate.id.clone(),
        cost: candidate.cost,
        likelihoods: candidate.likelihoods.iter().map(|row| {
            let p = 0.5 + (row[0] - 0.5) * 0.35;
            vec![p, 1.0 - p]
        }).collect(),
    }).collect();
    let reversal = candidate_set(
        [0.78, 0.80, 0.53],
        [0.32, 0.90, 0.50],
        [0.86, 0.32, 0.51],
        [0.50, 0.50, 0.50],
        [0.61, 0.70, 0.55],
    );
    let null = candidate_set(
        [0.55, 0.55, 0.55],
        [0.57, 0.57, 0.57],
        [0.53, 0.53, 0.53],
        [0.50, 0.50, 0.50],
        [0.54, 0.54, 0.54],
    );

    vec![
        HiddenWorld {
            id: "dev_nominal_calibration".into(),
            regime: BlindWorldRegime::NominalCalibration,
            held_out: false,
            candidates: nominal,
        },
        HiddenWorld {
            id: "dev_confounder_calibration".into(),
            regime: BlindWorldRegime::ConfounderCalibration,
            held_out: false,
            candidates: confounder,
        },
        HiddenWorld {
            id: "holdout_intervention_artifact".into(),
            regime: BlindWorldRegime::InterventionArtifact,
            held_out: true,
            candidates: intervention_artifact,
        },
        HiddenWorld {
            id: "holdout_sensor_degradation".into(),
            regime: BlindWorldRegime::SensorDegradation,
            held_out: true,
            candidates: degraded,
        },
        HiddenWorld {
            id: "holdout_effect_reversal".into(),
            regime: BlindWorldRegime::EffectReversal,
            held_out: true,
            candidates: reversal,
        },
        HiddenWorld {
            id: "holdout_null_negative_control".into(),
            regime: BlindWorldRegime::NullNegativeControl,
            held_out: true,
            candidates: null,
        },
    ]
}

fn top_index(posterior: &[HypothesisPrior]) -> usize {
    let mut best_index = 0;
    for index in 1..posterior.len() {
        if posterior[index].probability > posterior[best_index].probability {
            best_index = index;
        }
    }
    best_index
}

fn top_confidence(posterior: &[HypothesisPrior]) -> f64 {
    posterior[top_index(posterior)].probability
}

struct BenchmarkRng {
    state: u64,
}

impl BenchmarkRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sample without modulo bias for a bound that is strictly positive.
    fn index(&mut self, upper_bound: usize) -> usize {
        assert!(upper_bound > 0, "index bound must be positive");
        let bound = upper_bound as u64;
        let zone = (u64::MAX / bound) * bound;
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % bound) as usize;
            }
        }
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_runs_are_byte_stable_for_identical_configuration() {
        let first = run_blind_world_benchmark(12, 8, 0x51A5, 0.95).unwrap();
        let second = run_blind_world_benchmark(12, 8, 0x51A5, 0.95).unwrap();
        assert_eq!(first, second);
        assert_eq!(serde_json::to_vec(&first).unwrap(), serde_json::to_vec(&second).unwrap());
    }

    #[test]
    fn suite_contains_development_and_held_out_world_regimes() {
        let report = run_blind_world_benchmark(6, 6, 42, 0.95).unwrap();
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.generator_version, "sensebench-blind-worlds-0.2.0");
        assert_eq!(report.world_count, 24);
        assert_eq!(report.held_out_world_count, 16);
        assert_eq!(report.world_variants_per_regime, WORLD_VARIANTS_PER_REGIME);
        assert_eq!(report.summaries.len(), 24 * 7);
        assert!(report.summaries.iter().any(|row| {
            row.regime == BlindWorldRegime::NullNegativeControl
                && row.held_out_from_policy_models
        }));
        assert!(report.summaries.iter().any(|row| {
            row.regime == BlindWorldRegime::EffectReversal
                && row.held_out_from_policy_models
        }));
    }

    #[test]
    fn scoring_metrics_stay_in_declared_ranges() {
        let report = run_blind_world_benchmark(30, 8, 123, 0.95).unwrap();
        for row in report.summaries {
            assert!((0.0..=2.0).contains(&row.mean_brier_score));
            assert!((0.0..=30.0).contains(&row.mean_log_loss));
            assert!((0.0..=30.0).contains(&row.mean_prequential_predictive_log_loss));
            assert!((0.0..=1.0).contains(&row.model_adequacy_alarm_rate));
            assert!((0.0..=1.0).contains(&row.accepted_trial_fraction_after_adequacy_gate));
            assert!((0.0..=1.0).contains(&row.adequacy_gated_unsupported_causal_claim_rate_on_null_truth));
            assert!((row.model_adequacy_alarm_rate
                + row.accepted_trial_fraction_after_adequacy_gate - 1.0).abs() < 1.0e-12);
            if row.regime == BlindWorldRegime::NullNegativeControl {
                assert!(
                    row.adequacy_gated_unsupported_causal_claim_rate_on_null_truth
                        <= row.unsupported_causal_claim_rate_on_null_truth + 1.0e-12
                );
            }
            assert!((0.0..=2.0).contains(&row.mean_hidden_world_expected_information_gain_bits_per_cost));
            assert!((0.0..=2.0).contains(&row.mean_acquisition_regret_bits_per_selection));
            assert!((0.0..=1.0).contains(&row.top_hypothesis_accuracy));
            assert!((0.0..=1.0).contains(&row.mean_calibration_error_10_bins));
            assert!((0.0..=1.0).contains(&row.threshold_reached_fraction));
            assert!((0.0..=1.0).contains(&row.unsupported_causal_claim_rate_on_null_truth));
        }
    }

    #[test]
    fn adequacy_cutoffs_are_reported_from_independent_model_calibration() {
        let report = run_blind_world_benchmark(6, 6, 777, 0.95).unwrap();
        assert_eq!(report.model_adequacy_thresholds.len(), 7);
        for threshold in &report.model_adequacy_thresholds {
            assert_eq!(threshold.calibration_quantile, MODEL_ADEQUACY_QUANTILE);
            assert_eq!(threshold.independent_calibration_trials, 12);
            assert_eq!(
                threshold.calibration_source,
                "declared_model_scenarios_independent_seed"
            );
            assert!(threshold.threshold_mean_prequential_nll.is_finite());
            assert!(threshold.threshold_mean_prequential_nll >= 0.0);
        }
    }

    #[test]
    fn ensemble_predictive_probability_uses_declared_model_weights() {
        let priors = vec![
            HypothesisPrior { id: "h0".into(), probability: 0.5 },
            HypothesisPrior { id: "h1".into(), probability: 0.5 },
        ];
        let model_a = ModelLikelihoodScenario {
            scenario_id: "a".into(),
            weight: 0.25,
            candidates: vec![ExperimentCandidate {
                id: "measure".into(),
                cost: 1.0,
                likelihoods: vec![vec![0.9, 0.1], vec![0.7, 0.3]],
            }],
        };
        let model_b = ModelLikelihoodScenario {
            scenario_id: "b".into(),
            weight: 0.75,
            candidates: vec![ExperimentCandidate {
                id: "measure".into(),
                cost: 1.0,
                likelihoods: vec![vec![0.8, 0.2], vec![0.6, 0.4]],
            }],
        };
        let p = ensemble_predictive_outcome_probability(
            &priors, &[model_a, model_b], "measure", 0
        ).unwrap();
        // Model A assigns 0.8 and B assigns 0.7 under the equal prior:
        // 0.25 * 0.8 + 0.75 * 0.7 = 0.725.
        assert!((p - 0.725).abs() < 1.0e-12);
    }

    #[test]
    fn adequacy_gate_never_increases_null_false_claim_rate() {
        let report = run_blind_world_benchmark(12, 8, 0xBADA55, 0.95).unwrap();
        for row in report.summaries.iter()
            .filter(|row| row.regime == BlindWorldRegime::NullNegativeControl)
        {
            assert!(
                row.adequacy_gated_unsupported_causal_claim_rate_on_null_truth
                    <= row.unsupported_causal_claim_rate_on_null_truth + 1.0e-12
            );
        }
    }

    #[test]
    fn invalid_configuration_fails_closed() {
        assert_eq!(
            run_blind_world_benchmark(2, 8, 1, 0.95),
            Err(BlindWorldBenchmarkError::InvalidTrialCount)
        );
        assert_eq!(
            run_blind_world_benchmark(3, 0, 1, 0.95),
            Err(BlindWorldBenchmarkError::InvalidResourceBudget)
        );
        assert_eq!(
            run_blind_world_benchmark(3, 8, 1, 1.0 / 3.0),
            Err(BlindWorldBenchmarkError::InvalidPosteriorThreshold)
        );
    }

    #[test]
    fn parameterized_worlds_replay_and_change_with_seed() {
        let first = hidden_worlds(101);
        let replay = hidden_worlds(101);
        let other_seed = hidden_worlds(102);
        assert_eq!(first, replay);
        assert_ne!(first, other_seed);
        assert_eq!(first.len(), 6 * WORLD_VARIANTS_PER_REGIME);
    }

    #[test]
    fn generated_world_variants_preserve_declared_control_invariants() {
        let worlds = hidden_worlds(2026);
        for world in &worlds {
            let negative_control = world.candidates.iter()
                .find(|candidate| candidate.id == "negative_control").unwrap();
            let negative_probability = negative_control.likelihoods[0][0];
            assert!(negative_control.likelihoods.iter()
                .all(|row| (row[0] - negative_probability).abs() < 1.0e-12));
            match world.regime {
                BlindWorldRegime::NullNegativeControl => {
                    for candidate in &world.candidates {
                        let first = candidate.likelihoods[0][0];
                        assert!(candidate.likelihoods.iter()
                            .all(|row| (row[0] - first).abs() < 1.0e-12));
                    }
                }
                BlindWorldRegime::InterventionArtifact => {
                    let intervention = world.candidates.iter()
                        .find(|candidate| candidate.id == "intervene_on_x").unwrap();
                    let first = intervention.likelihoods[0][0];
                    assert!(intervention.likelihoods.iter()
                        .all(|row| (row[0] - first).abs() < 1.0e-12));
                }
                BlindWorldRegime::EffectReversal => {
                    let intervention = world.candidates.iter()
                        .find(|candidate| candidate.id == "intervene_on_x").unwrap();
                    assert!(intervention.likelihoods[0][0] < intervention.likelihoods[1][0]);
                    let confounder = world.candidates.iter()
                        .find(|candidate| candidate.id == "measure_confounder").unwrap();
                    assert!(confounder.likelihoods[0][0] > confounder.likelihoods[1][0]);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn fixed_design_is_selected_from_development_regimes_only() {
        let priors = uniform_priors();
        let worlds = hidden_worlds(42);
        let development: Vec<HiddenWorld> = worlds.iter()
            .filter(|world| !world.held_out)
            .cloned()
            .collect();
        let chosen = choose_development_fixed_design(&priors, &development).unwrap();
        assert!(worlds.iter().all(|world| {
            world.candidates.iter().any(|candidate| candidate.id == chosen)
        }));
    }
}
