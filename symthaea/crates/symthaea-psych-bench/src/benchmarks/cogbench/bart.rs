//! Balloon Analogue Risk Task (BART).
//!
//! Tests risk-taking behavior: pump a balloon for increasing reward,
//! but it may pop (losing all). Measures average pumps and pop rate.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// BART benchmark measuring risk-taking.
pub struct BartBenchmark;

impl BartBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        let seed = config.trial_seed("cogbench", "bart", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let num_balloons = 15;
        let max_pumps = 64;

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: 2, // 0 = pump, 1 = cash out
            action_temperature: config.action_temperature,
            planning_horizon: config.planning_horizon,
            ..Default::default()
        };

        let mut total_pumps = 0u64;
        let mut total_earnings = 0.0f64;
        let mut pops = 0u64;

        for _ in 0..num_balloons {
            let mut agent = ActiveInferenceAgent::new(agent_config.clone());

            // Each balloon has a random pop threshold
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pop_threshold = (rng_state % (max_pumps as u64 - 5)) as usize + 5;

            let mut pumps = 0usize;
            let mut popped = false;

            loop {
                // Present current state (pumps/max as normalized value)
                let state_val = pumps as f64 / max_pumps as f64;
                let obs = Observation::new(
                    vec![state_val, 1.0 - state_val, pumps as f64 * 0.01, 0.5],
                    1.0,
                    "bart",
                );
                agent.perceive(&obs);

                let action_result = agent.select_action();

                if action_result.action % 2 == 1 || pumps >= max_pumps {
                    // Cash out
                    total_earnings += pumps as f64 * 0.05;
                    break;
                }

                pumps += 1;
                if pumps >= pop_threshold {
                    popped = true;
                    pops += 1;
                    // Pop feedback
                    let pop_obs = Observation::new(vec![0.0; 4], 1.0, "bart_pop");
                    agent.perceive(&pop_obs);
                    break;
                }
            }

            if !popped {
                total_pumps += pumps as u64;
            }
        }

        let avg_pumps = total_pumps as f64 / (num_balloons - pops as usize).max(1) as f64;
        let pop_rate = pops as f64 / num_balloons as f64;
        let avg_earnings = total_earnings / num_balloons as f64;

        (avg_pumps, pop_rate, avg_earnings)
    }
}

impl PsychBenchmark for BartBenchmark {
    fn name(&self) -> &str {
        "CogBench::BART"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut pumps = Vec::new();
        let mut pop_rates = Vec::new();
        let mut earnings = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (p, pr, e) = self.run_trial(config, trial);
            pumps.push(p);
            pop_rates.push(pr);
            earnings.push(e);
        }

        result.insert("average_pumps", MetricValue::from_samples(&pumps));
        result.insert("pop_rate", MetricValue::from_samples(&pop_rates));
        result.insert("average_earnings", MetricValue::from_samples(&earnings));

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
    fn test_bart_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = BartBenchmark.run(&config);
        assert!(result.metrics.contains_key("average_pumps"));
        assert!(result.metrics.contains_key("pop_rate"));
    }
}
