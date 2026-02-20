//! Balloon Analogue Risk Task (BART).
//!
//! Tests risk-taking behavior: pump a balloon for increasing reward,
//! but it may pop (losing all). Measures average pumps and pop rate.
//!
//! The FEP agent learns across balloons: pumping yields reward (observations
//! move toward preferences), but over-pumping risks a pop (observations crash
//! to zero). The agent's action probabilities are sampled stochastically,
//! matching the softmax-sampling formulation of active inference.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

use super::sample_action;

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

        // Persistent agent across balloons — learns from experience
        let mut agent = ActiveInferenceAgent::new(agent_config);
        // Higher reward preference, lower risk aversion → encourages pumping
        agent.set_goals(vec![0.5, 0.9, 0.5, 0.1], 1.0);

        // Warm-up: 8 ascending reward steps showing successful pumping
        for step in 0..8 {
            let pump_level = step as f64 / 12.0;
            let reward_level = step as f64 * 0.15;
            let reward_norm = (reward_level / 2.0).min(1.0);
            let obs = Observation::new(
                vec![pump_level, reward_norm, 1.0 - pump_level, pump_level * 0.5],
                1.0,
                "bart_warmup",
            );
            agent.perceive(&obs);
            let _ = agent.select_action();
            let next_level = (step + 1) as f64 / 12.0;
            let next_reward = ((step + 1) as f64 * 0.15 / 2.0).min(1.0);
            let outcome = Observation::new(
                vec![next_level, next_reward, 1.0 - next_level, next_level * 0.5],
                1.0,
                "bart_warmup_outcome",
            );
            agent.learn_from_outcome(0, &outcome); // pump → reward continues
        }
        // Show a pop event (only after many pumps)
        let pop_obs = Observation::new(vec![0.0, 0.0, 0.0, 1.0], 1.0, "bart_warmup_pop");
        agent.learn_from_outcome(0, &pop_obs);

        let mut total_pumps = 0u64;
        let mut total_earnings = 0.0f64;
        let mut pops = 0u64;

        for _ in 0..num_balloons {
            // Each balloon has a random pop threshold
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pop_threshold = (rng_state % (max_pumps as u64 - 5)) as usize + 5;

            let mut pumps = 0usize;
            let mut popped = false;

            loop {
                // Present current state: [inflation, reward_norm, headroom, risk]
                let inflation = pumps as f64 / max_pumps as f64;
                let accumulated = pumps as f64 * 0.15;
                let reward_norm = (accumulated / 2.0).min(1.0);
                let risk_signal = inflation * 0.5; // linear, half-magnitude
                let obs = Observation::new(
                    vec![inflation, reward_norm, 1.0 - inflation, risk_signal],
                    1.0,
                    "bart",
                );
                agent.perceive(&obs);

                let action_result = agent.select_action();

                // Stochastic action selection from softmax distribution
                let chosen = sample_action(&action_result.action_probabilities, &mut rng_state);

                if chosen == 1 || pumps >= max_pumps {
                    // Cash out
                    total_earnings += accumulated;
                    let cashout_obs = Observation::new(
                        vec![0.0, reward_norm, 1.0, 0.0],
                        1.0,
                        "bart_cashout",
                    );
                    agent.learn_from_outcome(1, &cashout_obs);
                    break;
                }

                // Pump
                pumps += 1;
                let new_inflation = pumps as f64 / max_pumps as f64;
                let new_reward = (pumps as f64 * 0.15 / 2.0).min(1.0);
                let pump_obs = Observation::new(
                    vec![new_inflation, new_reward, 1.0 - new_inflation, new_inflation * 0.5],
                    1.0,
                    "bart_pump",
                );
                agent.learn_from_outcome(0, &pump_obs);

                if pumps >= pop_threshold {
                    popped = true;
                    pops += 1;
                    let pop_obs = Observation::new(vec![0.0, 0.0, 0.0, 1.0], 1.0, "bart_pop");
                    agent.learn_from_outcome(0, &pop_obs);
                    break;
                }
            }

            if !popped {
                total_pumps += pumps as u64;
            }
        }

        let cashed_out = (num_balloons - pops as usize).max(1);
        let avg_pumps = total_pumps as f64 / cashed_out as f64;
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
