//! Temporal discounting task.
//!
//! Tests preference between smaller-sooner vs larger-later rewards.
//! Maps to EFE temporal weighting over the planning horizon.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

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

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let choices = Self::choice_set();
        let mut patient_count = 0u64;
        let seed = config.trial_seed("cogbench", "discounting", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        for choice in choices.iter() {
            let agent_config = ActiveInferenceAgentConfig {
                state_dim: 4,
                obs_dim: 4,
                num_actions: 2, // 0 = smaller-sooner, 1 = larger-later
                planning_horizon: config.planning_horizon,
                action_temperature: config.action_temperature,
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

            // Stochastic sampling; action 1 = patient (larger-later)
            let chosen = sample_action(&action_result.action_probabilities, &mut rng_state);
            if chosen == 1 {
                patient_count += 1;
            }
        }

        // Discounting score S: proportion of patient choices
        patient_count as f64 / choices.len() as f64
    }
}

impl PsychBenchmark for TemporalDiscountingBenchmark {
    fn name(&self) -> &str {
        "CogBench::TemporalDiscounting"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut scores = Vec::new();

        for trial in 0..config.trials_per_condition {
            let s = self.run_trial(config, trial);
            scores.push(s);
        }

        result.insert("discounting_score_S", MetricValue::from_samples(&scores));

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
