// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal discounting task.
//!
//! Tests preference between smaller-sooner vs larger-later rewards.
//! Maps to EFE temporal weighting over the planning horizon.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

use super::sample_action;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Temporal discounting benchmark.
pub struct TemporalDiscountingBenchmark;

/// A discounting choice: smaller-sooner vs larger-later.
struct DiscountingChoice {
    small_amount: f64,
    small_delay: usize,
    large_amount: f64,
    large_delay: usize,
}

impl TemporalDiscountingBenchmark {
    /// Standard choice set (inspired by Kirby MCQ).
    fn choice_set() -> Vec<DiscountingChoice> {
        vec![
            DiscountingChoice {
                small_amount: 0.54,
                small_delay: 1,
                large_amount: 0.55,
                large_delay: 4,
            },
            DiscountingChoice {
                small_amount: 0.47,
                small_delay: 1,
                large_amount: 0.50,
                large_delay: 4,
            },
            DiscountingChoice {
                small_amount: 0.34,
                small_delay: 1,
                large_amount: 0.50,
                large_delay: 6,
            },
            DiscountingChoice {
                small_amount: 0.27,
                small_delay: 1,
                large_amount: 0.50,
                large_delay: 8,
            },
            DiscountingChoice {
                small_amount: 0.20,
                small_delay: 1,
                large_amount: 0.50,
                large_delay: 10,
            },
            DiscountingChoice {
                small_amount: 0.40,
                small_delay: 1,
                large_amount: 0.80,
                large_delay: 3,
            },
            DiscountingChoice {
                small_amount: 0.30,
                small_delay: 1,
                large_amount: 0.80,
                large_delay: 6,
            },
            DiscountingChoice {
                small_amount: 0.20,
                small_delay: 1,
                large_amount: 0.80,
                large_delay: 12,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, Vec<f64>) {
        let choices = Self::choice_set();
        let mut patient_count = 0u64;
        let seed = config.trial_seed("cogbench", "discounting", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let mut rt_ticks = Vec::new();

        for choice in choices.iter() {
            let agent_config = ActiveInferenceAgentConfig {
                state_dim: 4,
                obs_dim: 4,
                num_actions: 2, // 0 = smaller-sooner, 1 = larger-later
                planning_horizon: config.planning_horizon,
                // Time pressure: +0.15/unit raises choice temperature, modeling impulsive preference
                // for smaller-sooner rewards under deadline (Bickel et al., 2007; Wickelgren, 1977 SAT).
                action_temperature: config.action_temperature + config.time_pressure * 0.15,
                ..Default::default()
            };
            let mut agent = ActiveInferenceAgent::new(agent_config);

            // Set goal preferences: prefer higher value
            agent.set_goals(vec![0.8; 4], 2.0);

            // Present the choice context
            let context = Observation::new(
                vec![
                    choice.small_amount,
                    choice.small_delay as f64 / 12.0,
                    choice.large_amount,
                    choice.large_delay as f64 / 12.0,
                ],
                1.0,
                "discounting",
            );
            agent.perceive(&context);

            let action_result = agent.select_action();

            // RT proxy: choice difficulty from subjective value similarity — when
            // discounted values are close, deliberation takes longer (Bickel et al., 2007).
            let small_sv = choice.small_amount / (1.0 + 0.1 * choice.small_delay as f64);
            let large_sv = choice.large_amount / (1.0 + 0.1 * choice.large_delay as f64);
            let sv_diff = (small_sv - large_sv).abs();
            let max_sv = choice.large_amount; // upper bound for normalization
            let ticks = 5.0 + (1.0 - (sv_diff / max_sv).min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            // Stochastic sampling; action 1 = patient (larger-later)
            let chosen = sample_action(&action_result.action_probabilities, &mut rng_state);
            if chosen == 1 {
                patient_count += 1;
            }
        }

        // Discounting score S: proportion of patient choices
        (patient_count as f64 / choices.len() as f64, rt_ticks)
    }
}

impl PsychBenchmark for TemporalDiscountingBenchmark {
    fn name(&self) -> &str {
        "CogBench::TemporalDiscounting"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Temporal Discounting",
            citation: "Green & Myerson (2004)",
            year: 2004,
            doi: Some("10.1037/0033-2909.130.5.769"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut scores = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (s, rts) = self.run_trial(config, trial);
            scores.push(s);
            all_rts.extend_from_slice(&rts);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "discounting".to_string(),
                    correct: s > 0.0,
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

        result.insert("discounting_score_S", MetricValue::from_samples(&scores));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_discounting_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = TemporalDiscountingBenchmark.run(&config);
        assert!(result.metrics.contains_key("discounting_score_S"));
        let s = &result.metrics["discounting_score_S"];
        assert!(s.mean >= 0.0 && s.mean <= 1.0);
    }
}
