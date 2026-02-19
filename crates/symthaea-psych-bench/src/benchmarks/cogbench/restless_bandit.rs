//! Restless bandit task.
//!
//! Tests metacognitive sensitivity (QSR): whether the agent's confidence
//! in its choices correlates with actual accuracy. Arms drift over time.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Restless bandit benchmark measuring metacognitive sensitivity.
pub struct RestlessBanditBenchmark;

impl RestlessBanditBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64) {
        let seed = config.trial_seed("cogbench", "restless", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let num_arms = 4;
        let num_trials = 50;

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
            let chosen_arm = action_result.action % num_arms;

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

            // Confidence from precision estimator
            let confidence = agent.precision.perceptual_precision();
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

            // Observe reward from chosen arm
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let noise = ((rng_state % 200) as f64 - 100.0) / 500.0;
            let reward = (arm_values[chosen_arm] + noise).clamp(0.0, 1.0);

            let obs = Observation::new(vec![reward; 4], 1.0, "bandit");
            agent.perceive(&obs);
        }

        // QSR (quality of subjective report): accuracy when confident - accuracy when not
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

        (qsr, overall_accuracy)
    }
}

impl PsychBenchmark for RestlessBanditBenchmark {
    fn name(&self) -> &str {
        "CogBench::RestlessBandit"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut qsrs = Vec::new();
        let mut accuracies = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (qsr, acc) = self.run_trial(config, trial);
            qsrs.push(qsr);
            accuracies.push(acc);
        }

        result.insert("qsr_metacognitive_sensitivity", MetricValue::from_samples(&qsrs));
        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));

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
}
