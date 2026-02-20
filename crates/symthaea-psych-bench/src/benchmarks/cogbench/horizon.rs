//! Horizon task: exploration vs exploitation.
//!
//! Measures directed exploration (choosing informative actions) and
//! random exploration (entropy of action selection) under different
//! planning horizons (1 vs 6 remaining choices).
//!
//! The forced-choice phase teaches the agent that arm 0 gives high reward
//! and arm 1 gives low reward via `learn_from_outcome()`. The free-choice
//! phase uses stochastic sampling from the agent's softmax distribution
//! (matching the standard active inference formulation) rather than greedy
//! argmax, enabling genuine exploration behavior.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

use super::sample_action;

/// Horizon task benchmark.
pub struct HorizonBenchmark;

impl HorizonBenchmark {
    fn run_trial(
        &self,
        horizon: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, f64) {
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
        // Set goals: prefer high-value observations (drives exploitation)
        agent.set_goals(vec![0.8, 0.8, 0.8, 0.8], 2.0);

        // Forced-choice phase: teach that arm 0 is good (0.8) and arm 1 is bad (0.3)
        // Asymmetric exposure: arm 0 gets more trials (better known)
        let good_arm_value = 0.8;
        let bad_arm_value = 0.3;
        // 4 forced on arm 0, 1 forced on arm 1 → arm 1 is less known
        let forced_arms = [0, 0, 1, 0, 0];
        for &forced_arm in &forced_arms {
            let val = if forced_arm == 0 { good_arm_value } else { bad_arm_value };
            let arm0_signal = if forced_arm == 0 { 1.0 } else { 0.0 };
            let arm1_signal = if forced_arm == 1 { 1.0 } else { 0.0 };
            let obs = Observation::new(
                vec![val, arm0_signal, arm1_signal, val],
                1.0,
                "bandit",
            );
            agent.perceive(&obs);
            let _action = agent.select_action();
            let outcome = Observation::new(
                vec![val, arm0_signal, arm1_signal, val],
                1.0,
                "bandit_outcome",
            );
            agent.learn_from_outcome(forced_arm, &outcome);
        }

        // Free-choice phase: measure exploration behavior
        let mut directed_exploration_count = 0u64;
        let mut total_entropy = 0.0f64;
        let mut good_arm_choices = 0u64;
        let num_choices = horizon.max(1);

        // Track arm observation counts for directed exploration metric
        let mut arm_observations = [4u64, 1u64]; // From forced phase

        for _ in 0..num_choices {
            let action_result = agent.select_action();

            // Stochastic action selection from softmax distribution
            let chosen_arm = sample_action(&action_result.action_probabilities, &mut rng_state);

            // Directed exploration: choosing the less-observed arm
            let less_observed = if arm_observations[0] <= arm_observations[1] { 0 } else { 1 };
            if chosen_arm == less_observed {
                directed_exploration_count += 1;
            }

            if chosen_arm == 0 {
                good_arm_choices += 1;
            }

            // Random exploration: entropy of action distribution
            let entropy: f64 = action_result
                .action_probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            total_entropy += entropy;

            // Simulate observation after action with arm-specific encoding
            arm_observations[chosen_arm] += 1;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let obs_val = if chosen_arm == 0 {
                good_arm_value + (rng_state as f64 % 20.0 - 10.0) / 100.0
            } else {
                bad_arm_value + (rng_state as f64 % 20.0 - 10.0) / 100.0
            };
            let obs_val = obs_val.clamp(0.0, 1.0);
            let arm0_sig = if chosen_arm == 0 { 1.0 } else { 0.0 };
            let arm1_sig = if chosen_arm == 1 { 1.0 } else { 0.0 };
            let outcome = Observation::new(
                vec![obs_val, arm0_sig, arm1_sig, obs_val],
                1.0,
                "bandit_outcome",
            );
            agent.learn_from_outcome(chosen_arm, &outcome);
        }

        let directed_rate = directed_exploration_count as f64 / num_choices as f64;
        let avg_entropy = total_entropy / num_choices as f64;
        let exploit_rate = good_arm_choices as f64 / num_choices as f64;

        (directed_rate, avg_entropy, exploit_rate)
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
            let mut exploit = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (d, r, e) = self.run_trial(horizon, config, trial);
                directed.push(d);
                random.push(r);
                exploit.push(e);
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
        assert!(result.metrics.contains_key("horizon_1::exploitation_rate"));
    }
}
