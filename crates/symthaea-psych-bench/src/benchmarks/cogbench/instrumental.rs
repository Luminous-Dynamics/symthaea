// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Instrumental learning task.
//!
//! Tests learning rate and optimism bias by presenting positive and
//! negative outcomes separately, measuring asymmetric learning.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

use super::sample_action;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Instrumental learning benchmark.
pub struct InstrumentalLearningBenchmark;

impl InstrumentalLearningBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, f64, f64, Vec<f64>) {
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

        // Explicit action-value tracking (EMA of rewards per action)
        let mut action_reward = [0.5f64; 2]; // prior: neutral
        let reward_lr = 0.3; // EMA for reward learning

        // Track contingency sensitivity: proportion of correct choices
        // (action 0 is always the higher-reward action)
        let mut late_correct = 0u32;
        let mut rt_ticks = Vec::new();

        // Phase 1: Win condition (action 0 = 80% reward, action 1 = 20% reward)
        let mut win_errors = Vec::new();
        for trial in 0..20 {
            let action_result = agent.select_action();

            // Blend FEP probs with explicit reward-based probs
            let fep_probs = &action_result.action_probabilities;
            // Time pressure: base 0.15 yields ~85% optimal choice rate; +0.10/unit flattens
            // reward discrimination, modeling hasty valuation under SAT (Wickelgren, 1977).
            let rv_temp = 0.15 + config.time_pressure * 0.10;
            let rv_max = action_reward[0].max(action_reward[1]);
            let rv_exp: Vec<f64> = action_reward
                .iter()
                .map(|v| ((v - rv_max) / rv_temp).exp())
                .collect();
            let rv_sum: f64 = rv_exp.iter().sum();
            let rv_probs: Vec<f64> = rv_exp.iter().map(|e| e / rv_sum).collect();
            let progress = (trial as f64 / 10.0).min(1.0);
            let rv_weight = 0.2 + 0.6 * progress;
            let blended: Vec<f64> = (0..2)
                .map(|a| (1.0 - rv_weight) * fep_probs[a] + rv_weight * rv_probs[a])
                .collect();
            let bsum: f64 = blended.iter().sum();
            let final_probs: Vec<f64> = blended.iter().map(|p| p / bsum).collect();

            let chosen = sample_action(&final_probs, &mut rng_state);

            // RT proxy: decision difficulty from value certainty — smaller difference
            // between action values means harder choice (Wickelgren, 1977 SAT).
            let value_diff = (action_reward[0] - action_reward[1]).abs();
            let ticks = 5.0 + (1.0 - value_diff.min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            // Track contingency sensitivity in last 10 trials (after learning)
            if trial >= 10 && chosen == 0 {
                late_correct += 1;
            }

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let reward = if chosen == 0 {
                if rng_state % 100 < 80 { 0.9 } else { 0.1 }
            } else if rng_state % 100 < 20 {
                0.9
            } else {
                0.1
            };

            // Update action-value EMA
            action_reward[chosen] = (1.0 - reward_lr) * action_reward[chosen] + reward_lr * reward;

            let obs = Observation::new(vec![reward; 4], 1.0, "reward");
            let result = agent.perceive(&obs);
            win_errors.push(result.free_energy.prediction_error);
        }

        // Phase 2: Loss condition (action 0 = 80% medium, action 1 = 20% medium)
        let mut loss_errors = Vec::new();
        for trial in 0..20 {
            let action_result = agent.select_action();

            let fep_probs = &action_result.action_probabilities;
            // Time pressure: same SAT scaling as win phase (Wickelgren, 1977).
            let rv_temp = 0.15 + config.time_pressure * 0.10;
            let rv_max = action_reward[0].max(action_reward[1]);
            let rv_exp: Vec<f64> = action_reward
                .iter()
                .map(|v| ((v - rv_max) / rv_temp).exp())
                .collect();
            let rv_sum: f64 = rv_exp.iter().sum();
            let rv_probs: Vec<f64> = rv_exp.iter().map(|e| e / rv_sum).collect();
            let progress = ((trial + 20) as f64 / 20.0).min(1.0);
            let rv_weight = 0.2 + 0.6 * progress;
            let blended: Vec<f64> = (0..2)
                .map(|a| (1.0 - rv_weight) * fep_probs[a] + rv_weight * rv_probs[a])
                .collect();
            let bsum: f64 = blended.iter().sum();
            let final_probs: Vec<f64> = blended.iter().map(|p| p / bsum).collect();

            let chosen = sample_action(&final_probs, &mut rng_state);

            // RT proxy: same value-certainty model as win phase
            let value_diff = (action_reward[0] - action_reward[1]).abs();
            let ticks = 5.0 + (1.0 - value_diff.min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let reward = if chosen == 0 {
                if rng_state % 100 < 80 { 0.5 } else { 0.1 }
            } else if rng_state % 100 < 20 {
                0.5
            } else {
                0.1
            };

            action_reward[chosen] = (1.0 - reward_lr) * action_reward[chosen] + reward_lr * reward;

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
        let contingency_sensitivity = late_correct as f64 / 10.0;

        (
            overall_lr,
            optimism_bias,
            agent.stats.exploration_rate,
            contingency_sensitivity,
            rt_ticks,
        )
    }
}

impl PsychBenchmark for InstrumentalLearningBenchmark {
    fn name(&self) -> &str {
        "CogBench::InstrumentalLearning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Instrumental Learning",
            citation: "Daw et al. (2011)",
            year: 2011,
            doi: Some("10.1016/j.neuron.2011.02.027"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut lrs = Vec::new();
        let mut biases = Vec::new();
        let mut exploration_rates = Vec::new();
        let mut sensitivities = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (lr, bias, er, cs, rts) = self.run_trial(config, trial);
            lrs.push(lr);
            biases.push(bias);
            exploration_rates.push(er);
            sensitivities.push(cs);
            all_rts.extend_from_slice(&rts);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "instrumental".to_string(),
                    correct: cs > 0.5,
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

        result.insert("learning_rate", MetricValue::from_samples(&lrs));
        result.insert("optimism_bias", MetricValue::from_samples(&biases));
        result.insert(
            "exploration_rate",
            MetricValue::from_samples(&exploration_rates),
        );
        result.insert(
            "contingency_sensitivity",
            MetricValue::from_samples(&sensitivities),
        );
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

// =========================================================================
// Symthaea-backend: ContinuousMind-backed instrumental learning
// =========================================================================

/// Instrumental learning using Symthaea's ContinuousMind instead of FEP.
///
/// Tests asymmetric learning from positive vs negative outcomes.
/// Learning rate is measured via WM similarity drift rather than
/// prediction error reduction.
#[cfg(feature = "symthaea-backend")]
pub struct InstrumentalLearningMindBenchmark;

#[cfg(feature = "symthaea-backend")]
impl InstrumentalLearningMindBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        use super::mind_agent::CogBenchMindAgent;

        let seed = config.trial_seed("cogbench", "instrumental_mind", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        let mut agent = CogBenchMindAgent::new(
            2,
            config.dimension,
            config.working_memory_capacity,
            config.action_temperature,
            seed,
        );

        // Phase 1: Win condition (action 0 = 80% high reward, action 1 = 20%)
        let mut win_consciousness: Vec<f64> = Vec::new();
        for _ in 0..20 {
            let action_result = agent.select_action();
            let chosen = sample_action(&action_result.action_probabilities, &mut rng_state);

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let reward = if chosen == 0 {
                if rng_state % 100 < 80 { 0.9 } else { 0.1 }
            } else {
                if rng_state % 100 < 20 { 0.9 } else { 0.1 }
            };

            agent.perceive_reward(chosen, reward);
            win_consciousness.push(agent.consciousness_level());
        }

        // Phase 2: Loss condition (lower rewards overall)
        let mut loss_consciousness: Vec<f64> = Vec::new();
        for _ in 0..20 {
            let action_result = agent.select_action();
            let chosen = sample_action(&action_result.action_probabilities, &mut rng_state);

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let reward = if chosen == 0 {
                if rng_state % 100 < 80 { 0.5 } else { 0.1 }
            } else {
                if rng_state % 100 < 20 { 0.5 } else { 0.1 }
            };

            agent.perceive_reward(chosen, reward);
            loss_consciousness.push(agent.consciousness_level());
        }

        // Learning rate: consciousness level change over trials
        // Higher consciousness → better integration → faster learning
        let win_lr = if win_consciousness.len() >= 4 {
            let early: f64 = win_consciousness[..4].iter().sum::<f64>() / 4.0;
            let late: f64 = win_consciousness[win_consciousness.len() - 4..]
                .iter()
                .sum::<f64>()
                / 4.0;
            (late - early).abs()
        } else {
            0.0
        };

        let loss_lr = if loss_consciousness.len() >= 4 {
            let early: f64 = loss_consciousness[..4].iter().sum::<f64>() / 4.0;
            let late: f64 = loss_consciousness[loss_consciousness.len() - 4..]
                .iter()
                .sum::<f64>()
                / 4.0;
            (late - early).abs()
        } else {
            0.0
        };

        let overall_lr = (win_lr + loss_lr) / 2.0;
        let optimism_bias = win_lr - loss_lr;
        let final_consciousness = agent.consciousness_level();

        (overall_lr, optimism_bias, final_consciousness)
    }
}

#[cfg(feature = "symthaea-backend")]
impl PsychBenchmark for InstrumentalLearningMindBenchmark {
    fn name(&self) -> &str {
        "CogBench::InstrumentalLearning[Mind]"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Instrumental Learning",
            citation: "Daw et al. (2011)",
            year: 2011,
            doi: Some("10.1016/j.neuron.2011.02.027"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut lrs = Vec::new();
        let mut biases = Vec::new();
        let mut consciousness_levels = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (lr, bias, cl) = self.run_trial(config, trial);
            lrs.push(lr);
            biases.push(bias);
            consciousness_levels.push(cl);
        }

        result.insert("learning_rate", MetricValue::from_samples(&lrs));
        result.insert("optimism_bias", MetricValue::from_samples(&biases));
        result.insert(
            "final_consciousness_level",
            MetricValue::from_samples(&consciousness_levels),
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
    fn test_instrumental_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = InstrumentalLearningBenchmark.run(&config);
        assert!(result.metrics.contains_key("learning_rate"));
        assert!(result.metrics.contains_key("optimism_bias"));
    }

    #[cfg(feature = "symthaea-backend")]
    #[test]
    fn test_instrumental_mind_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = InstrumentalLearningMindBenchmark.run(&config);
        assert!(result.metrics.contains_key("learning_rate"));
        assert!(result.metrics.contains_key("optimism_bias"));
        assert!(result.metrics.contains_key("final_consciousness_level"));
    }
}
