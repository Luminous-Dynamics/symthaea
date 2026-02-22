//! Probabilistic reasoning task.
//!
//! Tests how the agent integrates prior beliefs with new evidence.
//! Measures beta1 (prior weight) and beta2 (likelihood weight) from
//! belief update strength after perceiving observations.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Probabilistic reasoning benchmark.
pub struct ProbabilisticReasoningBenchmark;

impl ProbabilisticReasoningBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let seed = config.trial_seed("cogbench", "probabilistic", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: 4,
            belief_learning_rate: 0.15,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_config);

        // Phase 1: Establish prior with consistent observations
        let prior_obs_value = 0.7;
        for _ in 0..5 {
            let obs = Observation::new(vec![prior_obs_value; 4], 1.0, "cognitive");
            agent.perceive(&obs);
        }
        let belief_after_prior = agent.belief.mean.clone();

        // Phase 2: Present conflicting evidence
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let _rng_state = rng_state;
        let conflicting_value = 0.3;
        let obs = Observation::new(vec![conflicting_value; 4], 1.0, "cognitive");
        let result = agent.perceive(&obs);

        // beta1 (prior weight): how much belief stayed near the prior
        let prior_mean: f64 =
            belief_after_prior.iter().sum::<f64>() / belief_after_prior.len() as f64;
        let new_mean: f64 = result.updated_belief.mean.iter().sum::<f64>()
            / result.updated_belief.mean.len() as f64;

        // If belief barely moved toward new evidence, prior weight is high
        let total_range = (prior_obs_value - conflicting_value).abs();
        let belief_shift = (new_mean - prior_mean).abs();
        let beta1 = if total_range > 0.001 {
            1.0 - (belief_shift / total_range).min(1.0)
        } else {
            0.5
        };

        // beta2 (likelihood weight): how much new evidence influenced belief
        let beta2 = if total_range > 0.001 {
            (belief_shift / total_range).min(1.0)
        } else {
            0.5
        };

        (beta1, beta2)
    }
}

impl PsychBenchmark for ProbabilisticReasoningBenchmark {
    fn name(&self) -> &str {
        "CogBench::ProbabilisticReasoning"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut beta1s = Vec::new();
        let mut beta2s = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (b1, b2) = self.run_trial(config, trial);
            beta1s.push(b1);
            beta2s.push(b2);
        }

        result.insert("beta1_prior_weight", MetricValue::from_samples(&beta1s));
        result.insert(
            "beta2_likelihood_weight",
            MetricValue::from_samples(&beta2s),
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
    fn test_probabilistic_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = ProbabilisticReasoningBenchmark.run(&config);
        assert!(result.metrics.contains_key("beta1_prior_weight"));
        assert!(result.metrics.contains_key("beta2_likelihood_weight"));
        for (_, val) in &result.metrics {
            assert!(val.mean.is_finite());
            assert!(val.mean >= 0.0 && val.mean <= 1.0);
        }
    }
}
