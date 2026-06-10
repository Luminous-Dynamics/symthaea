// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Horizon task: exploration vs exploitation.
//!
//! Measures directed exploration (choosing informative actions) and
//! random exploration (entropy of action selection) under different
//! planning horizons (1 vs 6 remaining choices).
//!
//! Model: Bayesian value tracker with horizon-scaled exploration bonus
//! (UCB-like). Arm means update via EMA; exploration bonus decays with
//! observation count and scales with remaining horizon (Wilson et al. 2014
//! — information has value proportional to remaining opportunities).
//!
//! Forced-choice phase teaches arm 0 = good (0.8), arm 1 = bad (0.3)
//! with asymmetric exposure (arm 1 less known). Free-choice phase uses
//! softmax over score = mean + horizon-scaled exploration bonus.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Horizon task benchmark.
pub struct HorizonBenchmark;

impl HorizonBenchmark {
    fn run_trial(
        &self,
        horizon: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, f64, Vec<f64>) {
        let seed = config.trial_seed("cogbench", &format!("horizon_{}", horizon), trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        let good_arm_value = 0.8;
        let bad_arm_value = 0.3;
        // Higher LR helps build stronger arm-value distinctions during the
        // short forced-choice phase, enabling more informed exploration
        // decisions in the free-choice phase.
        let learning_rate = 0.35;

        // Bayesian arm tracking: mean value + observation count
        let mut arm_mean = [0.5f64; 2]; // uninformative prior
        let mut arm_count = [0u64; 2];

        // Forced-choice phase: asymmetric exposure (4 arm 0, 1 arm 1)
        let forced_arms = [0, 0, 1, 0, 0];
        for &forced_arm in &forced_arms {
            let val = if forced_arm == 0 {
                good_arm_value
            } else {
                bad_arm_value
            };
            arm_mean[forced_arm] += learning_rate * (val - arm_mean[forced_arm]);
            arm_count[forced_arm] += 1;
        }

        // Free-choice phase: softmax over score = mean + exploration bonus
        let mut directed_exploration_count = 0u64;
        let mut total_entropy = 0.0f64;
        let mut good_arm_choices = 0u64;
        let mut rt_ticks = Vec::new();
        let num_choices = horizon.max(1);

        for choice_idx in 0..num_choices {
            let remaining = (num_choices - choice_idx) as f64;
            // Exploration bonus: UCB-like term scaled by remaining horizon.
            // With longer horizon, information is more valuable (Wilson et al. 2014).
            // Coefficient 0.55 provides an information bonus large enough to
            // overcome the learned value gap (~0.35) on early free-choice trials
            // when the less-observed arm has low count (1-2 observations).
            // Human directed exploration at horizon 6 is ~35% (Wilson et al. 2014),
            // driven by the "information bonus" — participants choose the less-known
            // arm even when its expected value is lower. Thompson sampling analyses
            // (Chapelle & Li, 2011) show that optimal exploration rate increases
            // with the uncertainty-to-value-gap ratio.
            let exploration_bonus = |count: u64| -> f64 {
                let info_value = 0.55 / (count as f64 + 1.0).sqrt();
                info_value * (remaining / 6.0).min(1.0)
            };

            let score0 = arm_mean[0] + exploration_bonus(arm_count[0]);
            let score1 = arm_mean[1] + exploration_bonus(arm_count[1]);

            // Softmax action selection; time pressure: +0.10/unit adds exploration noise, modeling
            // reduced deliberation in explore-exploit tradeoffs under deadline (Wilson et al., 2014 horizon task).
            // Temperature 0.45 (up from 0.35) produces noisier action selection,
            // matching the substantial stochasticity in human explore-exploit decisions
            // (Wilson et al. 2014 — entropy of choice distributions is high even
            // when value differences are large, suggesting noise in the decision rule).
            let temp = config.action_temperature.max(0.1) * 0.45 + config.time_pressure * 0.10;
            let max_s = score0.max(score1);
            let e0 = ((score0 - max_s) / temp).exp();
            let e1 = ((score1 - max_s) / temp).exp();
            let total = e0 + e1;
            let p1 = e1 / total;

            // RT proxy: decision difficulty from score similarity — closer scores
            // mean harder deliberation (Wilson et al., 2014 horizon task).
            let score_diff = (score0 - score1).abs();
            let max_score_range = 1.0;
            let ticks = 5.0 + (1.0 - (score_diff / max_score_range).min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            // Stochastic choice
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let roll = (rng_state % 10000) as f64 / 10000.0;
            let chosen_arm = if roll < p1 { 1 } else { 0 };

            // Directed exploration: choosing the less-observed arm
            let less_observed = if arm_count[0] <= arm_count[1] { 0 } else { 1 };
            if chosen_arm == less_observed {
                directed_exploration_count += 1;
            }

            if chosen_arm == 0 {
                good_arm_choices += 1;
            }

            // Random exploration: entropy of action distribution
            let p0 = e0 / total;
            let entropy = if p0 > 0.0 && p1 > 0.0 {
                -p0 * p0.ln() - p1 * p1.ln()
            } else {
                0.0
            };
            total_entropy += entropy;

            // Update arm mean from outcome
            arm_count[chosen_arm] += 1;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let obs_val = if chosen_arm == 0 {
                good_arm_value + (rng_state as f64 % 20.0 - 10.0) / 100.0
            } else {
                bad_arm_value + (rng_state as f64 % 20.0 - 10.0) / 100.0
            };
            let obs_val = obs_val.clamp(0.0, 1.0);
            arm_mean[chosen_arm] += learning_rate * (obs_val - arm_mean[chosen_arm]);
        }

        let directed_rate = directed_exploration_count as f64 / num_choices as f64;
        let avg_entropy = total_entropy / num_choices as f64;
        let exploit_rate = good_arm_choices as f64 / num_choices as f64;

        (directed_rate, avg_entropy, exploit_rate, rt_ticks)
    }
}

impl PsychBenchmark for HorizonBenchmark {
    fn name(&self) -> &str {
        "CogBench::HorizonTask"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Horizon Task (Explore-Exploit)",
            citation: "Wilson et al. (2014)",
            year: 2014,
            doi: Some("10.1037/a0038199"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        for horizon in [1, 6] {
            let mut directed = Vec::new();
            let mut random = Vec::new();
            let mut exploit = Vec::new();
            let mut all_rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (d, r, e, rts) = self.run_trial(horizon, config, trial);
                directed.push(d);
                random.push(r);
                exploit.push(e);
                all_rts.extend_from_slice(&rts);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("horizon_{}", horizon),
                        correct: e > 0.5,
                        rt_ticks: if rts.is_empty() {
                            0.0
                        } else {
                            rts.iter().sum::<f64>() / rts.len() as f64
                        },
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }

            result.insert(
                format!("horizon_{}::directed_exploration", horizon),
                MetricValue::from_samples(&directed),
            );
            result.insert(
                format!("horizon_{}::random_exploration", horizon),
                MetricValue::from_samples(&random),
            );
            result.insert(
                format!("horizon_{}::exploitation_rate", horizon),
                MetricValue::from_samples(&exploit),
            );
            result.insert(
                format!("horizon_{}::rt_ticks", horizon),
                MetricValue::from_samples(&all_rts),
            );
        }

        result.conditions = 2;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizon_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = HorizonBenchmark.run(&config);
        assert!(
            result
                .metrics
                .contains_key("horizon_1::directed_exploration")
        );
        assert!(
            result
                .metrics
                .contains_key("horizon_6::directed_exploration")
        );
        assert!(result.metrics.contains_key("horizon_1::exploitation_rate"));
    }
}
