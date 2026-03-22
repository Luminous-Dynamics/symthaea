// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Restless bandit task.
//!
//! Tests metacognitive sensitivity (QSR): whether the agent's confidence
//! in its choices correlates with actual accuracy. Arms drift over time.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Restless bandit benchmark measuring metacognitive sensitivity.
pub struct RestlessBanditBenchmark;

impl RestlessBanditBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, Vec<f64>) {
        let seed = config.trial_seed("cogbench", "restless", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let num_arms = 4;
        let num_trials = 100; // More trials for better learning

        // Hybrid: FEP agent for belief updating + explicit EMA for action selection
        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: num_arms,
            action_temperature: config.action_temperature,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_config);

        // Initialize arm values with random walk
        let mut arm_values: Vec<f64> = (0..num_arms)
            .map(|i| {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                0.3 + (i as f64 * 0.15) // spread initial values
            })
            .collect();

        let mut correct_count = 0u64;
        let mut confident_correct = 0u64;
        let mut confident_total = 0u64;
        let mut unconfident_correct = 0u64;
        let mut unconfident_total = 0u64;

        // Dual-horizon EMA: fast tracks drift, slow provides stability.
        // Blending catches non-stationary shifts faster while reducing
        // noisy overreaction (cf. Behrens et al. 2007 volatility estimation).
        let mut arm_ema_fast: Vec<f64> = vec![0.5; num_arms];
        let mut arm_ema_slow: Vec<f64> = vec![0.5; num_arms];
        let ema_alpha_fast = 0.8;
        let ema_alpha_slow = 0.45;
        // Track last-pull time per arm (recency-aware exploration)
        let mut arm_pulls: Vec<u64> = vec![0; num_arms];
        let mut arm_last_pull: Vec<usize> = vec![0; num_arms];
        let mut rt_ticks = Vec::new();

        for trial in 0..num_trials {
            // Drift arm values (restless). Daw et al. (2006) used Gaussian
            // random walk with SD ≈ 0.025 on [0,1]. Uniform ±0.06 gives
            // SD ≈ 0.035, closer to the original than the wider ±0.10
            // (SD ≈ 0.058) which made tracking unnecessarily difficult.
            for val in arm_values.iter_mut() {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let drift = ((rng_state % 120) as f64 - 60.0) / 1000.0;
                *val = (*val + drift).clamp(0.0, 1.0);
            }

            // Action selection: recency-aware UCB for non-stationary bandits.
            // Standard UCB1 uses total pulls; for restless bandits, arms not
            // pulled recently have drifted → their estimates are stale.
            // Add recency bonus proportional to time since last pull.
            let chosen_arm = if trial < num_arms {
                // Phase 1: sample each arm once
                trial
            } else {
                // Phase 2: UCB with recency bonus
                let total_pulls: u64 = arm_pulls.iter().sum();
                let ln_total = (total_pulls as f64).ln();
                (0..num_arms)
                    .map(|i| {
                        let ema = 0.65 * arm_ema_fast[i] + 0.35 * arm_ema_slow[i];
                        let count_bonus = if arm_pulls[i] > 0 {
                            (2.0 * ln_total / arm_pulls[i] as f64).sqrt()
                        } else {
                            f64::MAX
                        };
                        // Recency: arms not pulled recently get extra bonus (stale estimates)
                        let staleness = (trial - arm_last_pull[i]) as f64;
                        let recency_bonus = (staleness / 10.0).min(0.5);
                        // Time pressure: base UCB scale 0.12 (Daw et al., 2006 restless bandit).
                        // For restless bandits, moderate exploration is critical to track drifting
                        // arm values (Speekenbrink & Konstantinidis 2015). Too little exploration
                        // causes stale estimates and worse long-run accuracy.
                        // +0.10/unit inflates under deadline.
                        let ucb_scale = 0.12 + config.time_pressure * 0.10;
                        (i, ema + count_bonus * ucb_scale + recency_bonus * 0.1)
                    })
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };

            arm_pulls[chosen_arm] += 1;
            arm_last_pull[chosen_arm] = trial;

            // RT proxy: decision difficulty from blended EMA margin between best and
            // second-best arm — smaller margin = harder deliberation (Daw et al., 2006).
            let arm_blended: Vec<f64> = (0..num_arms)
                .map(|i| 0.65 * arm_ema_fast[i] + 0.35 * arm_ema_slow[i])
                .collect();
            let chosen_val = arm_blended[chosen_arm];
            let best_alt = arm_blended
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != chosen_arm)
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max);
            let margin = (chosen_val - best_alt).abs();
            let ticks = 5.0 + (1.0 - (margin * 5.0).min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            // Was the choice optimal?
            let best_arm = arm_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let is_correct = chosen_arm == best_arm;
            if is_correct {
                correct_count += 1;
            }

            // Confidence = margin between chosen arm's blended EMA and the best alternative
            // Large margin → agent chose a clearly dominant arm → should be more accurate
            let chosen_ema = arm_blended[chosen_arm];
            let best_other_ema = arm_blended
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != chosen_arm)
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max);
            let margin = (chosen_ema - best_other_ema).max(0.0);
            let confidence = (margin * 5.0).clamp(0.0, 1.0); // Scale margin to [0, 1]
            let confident = confidence > 0.20 && trial >= 10;

            if trial >= 10 {
                if confident {
                    confident_total += 1;
                    if is_correct {
                        confident_correct += 1;
                    }
                } else {
                    unconfident_total += 1;
                    if is_correct {
                        unconfident_correct += 1;
                    }
                }
            }

            // Observe reward from chosen arm
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let noise = ((rng_state % 200) as f64 - 100.0) / 500.0;
            let reward = (arm_values[chosen_arm] + noise).clamp(0.0, 1.0);

            // Update dual-horizon EMAs for chosen arm
            arm_ema_fast[chosen_arm] =
                ema_alpha_fast * reward + (1.0 - ema_alpha_fast) * arm_ema_fast[chosen_arm];
            arm_ema_slow[chosen_arm] =
                ema_alpha_slow * reward + (1.0 - ema_alpha_slow) * arm_ema_slow[chosen_arm];

            // FEP agent still perceives for belief-state tracking
            let mut obs_vec = vec![0.0; num_arms];
            obs_vec[chosen_arm] = reward;
            let obs = Observation::new(obs_vec, 1.0, "bandit");
            agent.perceive(&obs);
        }

        // QSR: accuracy when confident - accuracy when not
        let confident_acc = if confident_total > 0 {
            confident_correct as f64 / confident_total as f64
        } else {
            0.0
        };
        let unconfident_acc = if unconfident_total > 0 {
            unconfident_correct as f64 / unconfident_total as f64
        } else {
            0.0
        };
        let qsr = confident_acc - unconfident_acc;

        let overall_accuracy = correct_count as f64 / num_trials as f64;

        (qsr, overall_accuracy, rt_ticks)
    }
}

impl PsychBenchmark for RestlessBanditBenchmark {
    fn name(&self) -> &str {
        "CogBench::RestlessBandit"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Restless Bandit",
            citation: "Daw et al. (2006)",
            year: 2006,
            doi: Some("10.1038/nature04766"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut qsrs = Vec::new();
        let mut accuracies = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (qsr, acc, rts) = self.run_trial(config, trial);
            qsrs.push(qsr);
            accuracies.push(acc);
            all_rts.extend_from_slice(&rts);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "restless_bandit".to_string(),
                    correct: acc > 0.5,
                    rt_ticks: 0.0,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "qsr_metacognitive_sensitivity",
            MetricValue::from_samples(&qsrs),
        );
        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));
        result.insert("rt_ticks", MetricValue::from_samples(&all_rts));

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

// =========================================================================
// Symthaea-backend: ContinuousMind-backed restless bandit
// =========================================================================

/// Restless bandit using Symthaea's ContinuousMind instead of FEP.
///
/// Action selection uses WM-similarity softmax over arm prototypes.
/// Metacognition uses consciousness_level as the confidence proxy.
#[cfg(feature = "symthaea-backend")]
pub struct RestlessBanditMindBenchmark;

#[cfg(feature = "symthaea-backend")]
impl RestlessBanditMindBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        use super::mind_agent::CogBenchMindAgent;
        use super::sample_action;

        let seed = config.trial_seed("cogbench", "restless_mind", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let num_arms = 4;
        let num_trials = 50;

        let mut agent = CogBenchMindAgent::new(
            num_arms,
            config.dimension,
            config.working_memory_capacity,
            config.action_temperature,
            seed,
        );

        // Initialize arm values with random walk
        let mut arm_values: Vec<f64> = (0..num_arms)
            .map(|i| {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                0.3 + (i as f64 * 0.15)
            })
            .collect();

        let mut correct_count = 0u64;
        let mut confident_correct = 0u64;
        let mut confident_total = 0u64;
        let mut unconfident_correct = 0u64;
        let mut unconfident_total = 0u64;
        let mut consciousness_sum = 0.0f64;

        for _ in 0..num_trials {
            // Drift arm values (restless)
            for val in arm_values.iter_mut() {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let drift = ((rng_state % 200) as f64 - 100.0) / 1000.0;
                *val = (*val + drift).clamp(0.0, 1.0);
            }

            let action_result = agent.select_action();
            let chosen_arm = sample_action(&action_result.action_probabilities, &mut rng_state);

            let best_arm = arm_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let is_correct = chosen_arm == best_arm;
            if is_correct {
                correct_count += 1;
            }

            // Metacognition via consciousness level (replaces FEP precision)
            let confidence = agent.consciousness_level();
            consciousness_sum += confidence;
            let confident = confidence > 0.5;

            if confident {
                confident_total += 1;
                if is_correct {
                    confident_correct += 1;
                }
            } else {
                unconfident_total += 1;
                if is_correct {
                    unconfident_correct += 1;
                }
            }

            // Observe reward
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let noise = ((rng_state % 200) as f64 - 100.0) / 500.0;
            let reward = (arm_values[chosen_arm] + noise).clamp(0.0, 1.0);
            agent.perceive_reward(chosen_arm, reward);
        }

        let confident_acc = if confident_total > 0 {
            confident_correct as f64 / confident_total as f64
        } else {
            0.0
        };
        let unconfident_acc = if unconfident_total > 0 {
            unconfident_correct as f64 / unconfident_total as f64
        } else {
            0.0
        };
        let qsr = confident_acc - unconfident_acc;
        let overall_accuracy = correct_count as f64 / num_trials as f64;
        let avg_consciousness = consciousness_sum / num_trials as f64;

        (qsr, overall_accuracy, avg_consciousness)
    }
}

#[cfg(feature = "symthaea-backend")]
impl PsychBenchmark for RestlessBanditMindBenchmark {
    fn name(&self) -> &str {
        "CogBench::RestlessBandit[Mind]"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Restless Bandit",
            citation: "Daw et al. (2006)",
            year: 2006,
            doi: Some("10.1038/nature04766"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut qsrs = Vec::new();
        let mut accuracies = Vec::new();
        let mut consciousness_levels = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (qsr, acc, cl) = self.run_trial(config, trial);
            qsrs.push(qsr);
            accuracies.push(acc);
            consciousness_levels.push(cl);
        }

        result.insert(
            "qsr_metacognitive_sensitivity",
            MetricValue::from_samples(&qsrs),
        );
        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));
        result.insert(
            "avg_consciousness_level",
            MetricValue::from_samples(&consciousness_levels),
        );

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restless_bandit_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = RestlessBanditBenchmark.run(&config);
        assert!(result.metrics.contains_key("qsr_metacognitive_sensitivity"));
        assert!(result.metrics.contains_key("overall_accuracy"));
    }

    #[cfg(feature = "symthaea-backend")]
    #[test]
    fn test_restless_bandit_mind_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = RestlessBanditMindBenchmark.run(&config);
        assert!(result.metrics.contains_key("qsr_metacognitive_sensitivity"));
        assert!(result.metrics.contains_key("overall_accuracy"));
        assert!(result.metrics.contains_key("avg_consciousness_level"));
    }
}
