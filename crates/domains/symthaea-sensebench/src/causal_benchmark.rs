// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Controlled active-selection benchmark using two competing causal explanations.
//!
//! This is a deliberately small software benchmark, not a model of a specific
//! physical system. It checks whether the selector follows informative evidence
//! rather than merely taking more measurements. Truth stays inside the evaluator;
//! the selection policy receives only priors and candidate likelihood models.

use serde::{Deserialize, Serialize};

use crate::experiment_design::{
    ExperimentCandidate, HypothesisPrior, rank_experiments, update_posterior,
};

const H_DIRECT: &str = "direct_cause";
const H_SHARED: &str = "shared_cause";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    /// Re-rank candidates under the current posterior at each decision.
    ActiveInformationGain,
    /// Choose a candidate uniformly from the available candidates.
    Random,
    /// Use a passive observation that is intentionally non-discriminating.
    PassiveOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionPolicySummary {
    pub policy: SelectionPolicy,
    pub trials: usize,
    pub experiment_budget: usize,
    pub mean_true_hypothesis_probability: f64,
    /// Normal-approximation 95% interval across simulated trials, clipped to [0, 1].
    pub mean_true_hypothesis_probability_ci95: [f64; 2],
    /// Multiclass Brier score (sum over the two hypothesis outcomes; lower is better).
    pub mean_brier_score: f64,
    /// Normal-approximation 95% interval across simulated trials, clipped to [0, 2].
    pub mean_brier_score_ci95: [f64; 2],
    pub final_top_hypothesis_accuracy: f64,
    /// 95% Wilson interval for the top-hypothesis accuracy proportion.
    pub final_top_hypothesis_accuracy_wilson_ci95: [f64; 2],
    pub posterior_threshold: f64,
    pub posterior_threshold_reached_fraction: f64,
    /// 95% Wilson interval for the fraction of trials reaching the posterior threshold.
    pub posterior_threshold_reached_wilson_ci95: [f64; 2],
    /// Mean number of experiments among trials that reached the threshold.
    pub mean_experiments_to_threshold: Option<f64>,
    pub mean_informative_experiments: f64,
}

/// Invalid simulation/evaluation settings.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CausalBenchmarkError {
    #[error("trials must be > 0")]
    NoTrials,
    #[error("trial count must be even to balance the ground-truth hypotheses")]
    OddTrialCount,
    #[error("experiment budget must be > 0")]
    NoExperimentBudget,
    #[error("posterior threshold must be finite and in (0.5, 1]")]
    InvalidPosteriorThreshold,
}

/// Run controlled, paired synthetic trials for active, random, and passive-only policies.
///
/// Ground truth is balanced across the two hypotheses and is not passed to the
/// policy. Every policy receives the same candidate library and its current
/// posterior. A per-trial random stream is shared across policies for outcome
/// generation (common random numbers); random action choice uses a separate stream.
/// The active policy can choose an intervention whose outcome likelihood differs
/// under each hypothesis; passive candidates have identical likelihood rows and
/// therefore zero expected information gain.
///
/// An even trial count is required to balance the two ground-truth hypotheses.
/// Results are deterministic for a fixed tuple of (trials, budget, seed, threshold).
pub fn run_causal_selection_benchmark(
    trials: usize,
    experiment_budget: usize,
    seed: u64,
    posterior_threshold: f64,
) -> Result<Vec<SelectionPolicySummary>, CausalBenchmarkError> {
    if trials == 0 {
        return Err(CausalBenchmarkError::NoTrials);
    }
    if trials % 2 != 0 {
        return Err(CausalBenchmarkError::OddTrialCount);
    }
    if experiment_budget == 0 {
        return Err(CausalBenchmarkError::NoExperimentBudget);
    }
    if !posterior_threshold.is_finite()
        || posterior_threshold <= 0.5
        || posterior_threshold > 1.0
    {
        return Err(CausalBenchmarkError::InvalidPosteriorThreshold);
    }

    let candidates = candidate_library();
    let policies = [
        SelectionPolicy::ActiveInformationGain,
        SelectionPolicy::Random,
        SelectionPolicy::PassiveOnly,
    ];
    let mut summaries = Vec::with_capacity(policies.len());

    for policy in policies {
        let mut probability_sum = 0.0;
        let mut probability_square_sum = 0.0;
        let mut brier_sum = 0.0;
        let mut brier_square_sum = 0.0;
        let mut correct_count = 0usize;
        let mut reached_count = 0usize;
        let mut steps_to_threshold_sum = 0usize;
        let mut informative_count = 0usize;

        for trial in 0..trials {
            // Balanced evaluation: neither hypothesis can dominate by prevalence.
            let true_hypothesis_index = trial % 2;
            let trial_seed =
                seed ^ (trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let result = simulate_trial(
                policy,
                true_hypothesis_index,
                trial_seed,
                experiment_budget,
                posterior_threshold,
                &candidates,
            );

            let true_probability = result.posterior[true_hypothesis_index].probability;
            probability_sum += true_probability;
            probability_square_sum += true_probability * true_probability;
            let brier_score = result.posterior.iter()
                .enumerate()
                .map(|(index, hypothesis)| {
                    let target = if index == true_hypothesis_index { 1.0 } else { 0.0 };
                    (hypothesis.probability - target).powi(2)
                })
                .sum::<f64>();
            brier_sum += brier_score;
            brier_square_sum += brier_score * brier_score;

            let top_index = if result.posterior[0].probability >= result.posterior[1].probability {
                0
            } else {
                1
            };
            if top_index == true_hypothesis_index {
                correct_count += 1;
            }
            if let Some(step) = result.steps_to_threshold {
                reached_count += 1;
                steps_to_threshold_sum += step;
            }
            informative_count += result.informative_experiments;
        }

        summaries.push(SelectionPolicySummary {
            policy,
            trials,
            experiment_budget,
            mean_true_hypothesis_probability: probability_sum / trials as f64,
            mean_true_hypothesis_probability_ci95: mean_ci95(
                probability_sum,
                probability_square_sum,
                trials,
                0.0,
                1.0,
            ),
            mean_brier_score: brier_sum / trials as f64,
            mean_brier_score_ci95: mean_ci95(
                brier_sum,
                brier_square_sum,
                trials,
                0.0,
                2.0,
            ),
            final_top_hypothesis_accuracy: correct_count as f64 / trials as f64,
            final_top_hypothesis_accuracy_wilson_ci95: wilson_ci95(correct_count, trials),
            posterior_threshold,
            posterior_threshold_reached_fraction: reached_count as f64 / trials as f64,
            posterior_threshold_reached_wilson_ci95: wilson_ci95(reached_count, trials),
            mean_experiments_to_threshold: (reached_count > 0).then_some(
                steps_to_threshold_sum as f64 / reached_count as f64
            ),
            mean_informative_experiments: informative_count as f64 / trials as f64,
        });
    }

    Ok(summaries)
}

/// Normal-approximation interval for a bounded mean, using empirical sample
/// variance across trials. This describes Monte Carlo variation conditional on
/// the fixed synthetic generator and its supplied likelihood model; it does not
/// quantify model misspecification or real-world generalization error.
fn mean_ci95(sum: f64, square_sum: f64, count: usize, min: f64, max: f64) -> [f64; 2] {
    let mean = sum / count as f64;
    if count < 2 {
        // Do not imply zero uncertainty when one trial is insufficient to
        // estimate sample variance.
        return [min, max];
    }
    let sample_variance =
        ((square_sum - sum * sum / count as f64) / (count - 1) as f64).max(0.0);
    let standard_error = (sample_variance / count as f64).sqrt();
    [
        (mean - 1.96 * standard_error).clamp(min, max),
        (mean + 1.96 * standard_error).clamp(min, max),
    ]
}

/// Wilson score interval for a binomial proportion; avoids degenerate zero-width
/// intervals at observed rates of exactly zero or one.
fn wilson_ci95(successes: usize, count: usize) -> [f64; 2] {
    let n = count as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z
        * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    [(center - half_width).clamp(0.0, 1.0), (center + half_width).clamp(0.0, 1.0)]
}

struct TrialResult {
    posterior: Vec<HypothesisPrior>,
    steps_to_threshold: Option<usize>,
    informative_experiments: usize,
}

fn simulate_trial(
    policy: SelectionPolicy,
    true_hypothesis_index: usize,
    seed: u64,
    budget: usize,
    threshold: f64,
    candidates: &[ExperimentCandidate],
) -> TrialResult {
    let mut outcome_rng = BenchmarkRng::new(seed ^ 0xDEC0_DE00);
    let mut selection_rng = BenchmarkRng::new(seed ^ 0x51EC_7100);
    let mut posterior = vec![
        HypothesisPrior { id: H_DIRECT.into(), probability: 0.5 },
        HypothesisPrior { id: H_SHARED.into(), probability: 0.5 },
    ];
    let mut steps_to_threshold = None;
    let mut informative_experiments = 0usize;

    for step in 1..=budget {
        let candidate_index = match policy {
            SelectionPolicy::ActiveInformationGain => {
                let ranking = rank_experiments(&posterior, candidates)
                    .expect("benchmark candidate model must remain valid");
                candidates.iter()
                    .position(|candidate| candidate.id == ranking[0].candidate_id)
                    .expect("ranked candidate must exist")
            }
            SelectionPolicy::Random => {
                (selection_rng.next_u64() % candidates.len() as u64) as usize
            },
            SelectionPolicy::PassiveOnly => 0,
        };
        let candidate = &candidates[candidate_index];
        if candidate.id == "intervene_on_x" {
            informative_experiments += 1;
        }

        // The evaluator uses the true hypothesis solely to produce an outcome.
        // The policy only receives that outcome and its candidate likelihoods.
        let true_likelihoods = &candidate.likelihoods[true_hypothesis_index];
        let outcome_index = if outcome_rng.unit_f64() < true_likelihoods[0] {
            0
        } else {
            1
        };

        let update = update_posterior(&posterior, candidate, outcome_index)
            .expect("simulated outcome must be supported by the chosen candidate model");
        posterior = update.posterior;

        let maximum_probability = posterior.iter()
            .map(|hypothesis| hypothesis.probability)
            .fold(0.0, f64::max);
        if maximum_probability >= threshold {
            steps_to_threshold = Some(step);
            break;
        }
    }

    TrialResult { posterior, steps_to_threshold, informative_experiments }
}

fn candidate_library() -> Vec<ExperimentCandidate> {
    vec![
        // Passive observation: the observed variables correlate under both models,
        // so the outcome distribution cannot distinguish the hypotheses.
        ExperimentCandidate {
            id: "passive_observation".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.8, 0.2], vec![0.8, 0.2]],
        },
        // Intervention: if X directly causes Y, the outcome distribution differs
        // from the shared-cause model after intervening on X.
        ExperimentCandidate {
            id: "intervene_on_x".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.9, 0.1], vec![0.2, 0.8]],
        },
        ExperimentCandidate {
            id: "negative_control".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        },
        ExperimentCandidate {
            id: "alternate_passive_measurement".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.65, 0.35], vec![0.65, 0.35]],
        },
    ]
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

    fn unit_f64(&mut self) -> f64 {
        // Use the top 53 bits so the result is in [0, 1) exactly as an f64 fraction.
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_policy_beats_random_and_passive_on_blind_causal_trials() {
        let summaries = run_causal_selection_benchmark(512, 8, 0x5E115E, 0.95).unwrap();
        assert_eq!(summaries.len(), 3);
        let active = &summaries[0];
        let random = &summaries[1];
        let passive = &summaries[2];

        assert_eq!(active.policy, SelectionPolicy::ActiveInformationGain);
        assert!(active.mean_informative_experiments > 0.0);
        assert!(active.mean_informative_experiments <= 8.0);
        assert_eq!(random.policy, SelectionPolicy::Random);
        assert_eq!(passive.policy, SelectionPolicy::PassiveOnly);
        assert_eq!(passive.mean_informative_experiments, 0.0);
        assert!(active.mean_informative_experiments > random.mean_informative_experiments);

        assert!(active.mean_brier_score < random.mean_brier_score);
        assert!(active.mean_brier_score < passive.mean_brier_score);
        assert!(active.final_top_hypothesis_accuracy >= random.final_top_hypothesis_accuracy);
        assert!(active.posterior_threshold_reached_fraction >= random.posterior_threshold_reached_fraction);
    }

    #[test]
    fn uncertainty_intervals_contain_point_estimates_and_remain_bounded() {
        let summaries = run_causal_selection_benchmark(256, 8, 0xBEEF, 0.95).unwrap();
        for summary in summaries {
            let probability_ci = summary.mean_true_hypothesis_probability_ci95;
            assert!(probability_ci[0] <= summary.mean_true_hypothesis_probability);
            assert!(summary.mean_true_hypothesis_probability <= probability_ci[1]);
            assert!(probability_ci[0] >= 0.0 && probability_ci[1] <= 1.0);

            let brier_ci = summary.mean_brier_score_ci95;
            assert!(brier_ci[0] <= summary.mean_brier_score);
            assert!(summary.mean_brier_score <= brier_ci[1]);
            assert!(brier_ci[0] >= 0.0 && brier_ci[1] <= 2.0);

            let accuracy_ci = summary.final_top_hypothesis_accuracy_wilson_ci95;
            assert!(accuracy_ci[0] <= summary.final_top_hypothesis_accuracy);
            assert!(summary.final_top_hypothesis_accuracy <= accuracy_ci[1]);
            assert!(accuracy_ci[0] >= 0.0 && accuracy_ci[1] <= 1.0);
        }
    }

    #[test]
    fn identical_inputs_produce_identical_summaries() {
        let a = run_causal_selection_benchmark(64, 6, 12345, 0.95).unwrap();
        let b = run_causal_selection_benchmark(64, 6, 12345, 0.95).unwrap();
        assert_eq!(a, b);
        assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&b).unwrap());
    }

    #[test]
    fn benchmark_rejects_invalid_configuration() {
        assert_eq!(
            run_causal_selection_benchmark(0, 8, 1, 0.95),
            Err(CausalBenchmarkError::NoTrials)
        );
        assert_eq!(
            run_causal_selection_benchmark(9, 8, 1, 0.95),
            Err(CausalBenchmarkError::OddTrialCount)
        );
        assert_eq!(
            run_causal_selection_benchmark(10, 0, 1, 0.95),
            Err(CausalBenchmarkError::NoExperimentBudget)
        );
        assert_eq!(
            run_causal_selection_benchmark(10, 2, 1, 0.5),
            Err(CausalBenchmarkError::InvalidPosteriorThreshold)
        );
    }
}
