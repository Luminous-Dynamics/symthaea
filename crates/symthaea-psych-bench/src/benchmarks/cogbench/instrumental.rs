//! Instrumental learning task.
//!
//! Tests learning rate and optimism bias by presenting positive and
//! negative outcomes separately, measuring asymmetric learning.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Instrumental learning benchmark.
pub struct InstrumentalLearningBenchmark;

impl InstrumentalLearningBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, f64) {
        let seed = config.trial_seed("cogbench", "instrumental", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: 2,
            action_temperature: config.action_temperature,
            enable_td_learning: config.enable_fep,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_config);

        // Phase 1: Win condition (action 0 = 80% reward, action 1 = 20% reward)
        let mut win_errors = Vec::new();
        for _ in 0..20 {
            let action_result = agent.select_action();
            let chosen = action_result.action % 2;

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let reward = if chosen == 0 {
                if rng_state % 100 < 80 { 0.9 } else { 0.1 }
            } else {
                if rng_state % 100 < 20 { 0.9 } else { 0.1 }
            };

            let obs = Observation::new(vec![reward; 4], 1.0, "reward");
            let result = agent.perceive(&obs);
            win_errors.push(result.free_energy.prediction_error);
        }

        // Phase 2: Loss condition (action 0 = 20% loss, action 1 = 80% loss)
        let mut loss_errors = Vec::new();
        for _ in 0..20 {
            let action_result = agent.select_action();
            let chosen = action_result.action % 2;

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let reward = if chosen == 0 {
                if rng_state % 100 < 80 { 0.5 } else { 0.1 }
            } else {
                if rng_state % 100 < 20 { 0.5 } else { 0.1 }
            };

            let obs = Observation::new(vec![reward; 4], 1.0, "reward");
            let result = agent.perceive(&obs);
            loss_errors.push(result.free_energy.prediction_error);
        }

        // Learning rate: average prediction error reduction over trials
        let win_lr = if win_errors.len() >= 4 {
            let early: f64 = win_errors[..4].iter().sum::<f64>() / 4.0;
            let late: f64 = win_errors[win_errors.len() - 4..].iter().sum::<f64>() / 4.0;
            (early - late).max(0.0)
        } else {
            0.0
        };

        let loss_lr = if loss_errors.len() >= 4 {
            let early: f64 = loss_errors[..4].iter().sum::<f64>() / 4.0;
            let late: f64 = loss_errors[loss_errors.len() - 4..].iter().sum::<f64>() / 4.0;
            (early - late).max(0.0)
        } else {
            0.0
        };

        // Optimism bias: learning faster from wins than losses
        let overall_lr = (win_lr + loss_lr) / 2.0;
        let optimism_bias = win_lr - loss_lr;

        (overall_lr, optimism_bias, agent.stats.exploration_rate)
    }
}

impl PsychBenchmark for InstrumentalLearningBenchmark {
    fn name(&self) -> &str {
        "CogBench::InstrumentalLearning"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut lrs = Vec::new();
        let mut biases = Vec::new();
        let mut exploration_rates = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (lr, bias, er) = self.run_trial(config, trial);
            lrs.push(lr);
            biases.push(bias);
            exploration_rates.push(er);
        }

        result.insert("learning_rate", MetricValue::from_samples(&lrs));
        result.insert("optimism_bias", MetricValue::from_samples(&biases));
        result.insert("exploration_rate", MetricValue::from_samples(&exploration_rates));

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
    fn test_instrumental_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = InstrumentalLearningBenchmark.run(&config);
        assert!(result.metrics.contains_key("learning_rate"));
        assert!(result.metrics.contains_key("optimism_bias"));
    }
}
