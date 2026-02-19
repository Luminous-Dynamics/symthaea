//! Horizon task: exploration vs exploitation.
//!
//! Measures directed exploration (choosing informative actions) and
//! random exploration (entropy of action selection) under different
//! planning horizons (1 vs 6 remaining choices).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Horizon task benchmark.
pub struct HorizonBenchmark;

impl HorizonBenchmark {
    fn run_trial(
        &self,
        horizon: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64) {
        let seed = config.trial_seed("cogbench", &format!("horizon_{}", horizon), trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: 2, // Two bandits
            planning_horizon: horizon,
            action_temperature: config.action_temperature,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_config);

        // Forced-choice phase: show one arm is clearly better (4 forced trials)
        let good_arm_value = 0.8;
        let bad_arm_value = 0.3;
        for i in 0..4 {
            let val = if i % 2 == 0 { good_arm_value } else { bad_arm_value };
            let obs = Observation::new(vec![val; 4], 1.0, "bandit");
            agent.perceive(&obs);
        }

        // Free-choice phase: measure exploration behavior
        let mut directed_exploration_count = 0u64;
        let mut total_entropy = 0.0f64;
        let num_choices = horizon.max(1);

        for _ in 0..num_choices {
            let action_result = agent.select_action();

            // Directed exploration: choosing the less-known arm
            if action_result.is_exploratory {
                directed_exploration_count += 1;
            }

            // Random exploration: entropy of action distribution
            let entropy: f64 = action_result
                .action_probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            total_entropy += entropy;

            // Simulate observation after action
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let obs_val = if action_result.action == 0 {
                good_arm_value + (rng_state as f64 % 20.0 - 10.0) / 100.0
            } else {
                bad_arm_value + (rng_state as f64 % 20.0 - 10.0) / 100.0
            };
            let obs = Observation::new(vec![obs_val.clamp(0.0, 1.0); 4], 1.0, "bandit");
            agent.perceive(&obs);
        }

        let directed_rate = directed_exploration_count as f64 / num_choices as f64;
        let avg_entropy = total_entropy / num_choices as f64;

        (directed_rate, avg_entropy)
    }
}

impl PsychBenchmark for HorizonBenchmark {
    fn name(&self) -> &str {
        "CogBench::HorizonTask"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        for horizon in [1, 6] {
            let mut directed = Vec::new();
            let mut random = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (d, r) = self.run_trial(horizon, config, trial);
                directed.push(d);
                random.push(r);
            }

            result.insert(
                format!("horizon_{}::directed_exploration", horizon),
                MetricValue::from_samples(&directed),
            );
            result.insert(
                format!("horizon_{}::random_exploration", horizon),
                MetricValue::from_samples(&random),
            );
        }

        result.conditions = 2;
        result.trials_per_condition = config.trials_per_condition;
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
        assert!(result.metrics.contains_key("horizon_1::directed_exploration"));
        assert!(result.metrics.contains_key("horizon_6::directed_exploration"));
    }
}
