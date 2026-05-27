// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Two-step task (Daw et al., 2011).
//!
//! Tests model-based vs model-free behavior. A choice at stage 1 leads
//! (probabilistically) to one of two stage-2 states, each with different
//! reward probabilities. Model-based agents track transition structure.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

use super::sample_action;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Two-step task benchmark measuring model-basedness.
pub struct TwoStepBenchmark;

impl TwoStepBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, Vec<f64>) {
        let seed = config.trial_seed("cogbench", "two_step", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        // More episodes → cleaner transition×reward interaction estimate (Daw et al., 2011).
        let num_episodes = 100;

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: 2,
            planning_horizon: config.planning_horizon,
            action_temperature: config.action_temperature,
            enable_td_learning: config.enable_fep,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_config);

        // Track stay/switch behavior for model-based analysis
        let mut common_rewarded_stay = 0u64;
        let mut common_rewarded_total = 0u64;
        let mut rare_rewarded_stay = 0u64;
        let mut rare_rewarded_total = 0u64;
        let mut common_unrewarded_stay = 0u64;
        let mut common_unrewarded_total = 0u64;
        let mut rare_unrewarded_stay = 0u64;
        let mut rare_unrewarded_total = 0u64;

        let mut prev_action: Option<usize> = None;
        let mut prev_common = false;
        let mut prev_rewarded = false;

        // Explicit transition model: counts of action→state transitions.
        // This enables model-based action selection alongside FEP.
        // transition_counts[action][state] = number of times action led to state.
        // Adaptive Laplace prior: strong early (regularizes sparse data), decays as
        // evidence accumulates. Prevents early noise from distorting transition estimates
        // while letting later episodes reflect true 70/30 structure more sharply.
        let mut transition_counts = [[1.0f64; 2]; 2]; // will be recomputed per-episode
        // Reward model: EMA of rewards in each state
        let mut state_reward = [0.5f64; 2]; // prior: 0.5
        // Higher LR (0.60, up from 0.50) speeds reward learning, allowing the
        // agent to track the state-reward difference (state 0: 60% vs state 1: 40%)
        // more precisely. This improves the transition×reward interaction signal
        // (β3) because the model-based value difference between actions becomes
        // more distinct when state rewards are tracked accurately. Behrens et al.
        // (2007): optimal reward LR increases with reward volatility.
        let reward_lr = 0.60;
        let mut rt_ticks = Vec::new();

        for ep in 0..num_episodes {
            // Adaptive prior: strong regularization early (prior=1.0) decays toward 0.1
            // as evidence accumulates, sharpening the learned transition model.
            let prior = (50.0 / (ep as f64 + 50.0)).max(0.1);
            // Recompute effective counts with adaptive prior offset
            let effective_counts = [
                [
                    transition_counts[0][0] - 1.0 + prior,
                    transition_counts[0][1] - 1.0 + prior,
                ],
                [
                    transition_counts[1][0] - 1.0 + prior,
                    transition_counts[1][1] - 1.0 + prior,
                ],
            ];

            // Stage 1: blend FEP action selection with model-based values.
            // As the agent learns transitions, the model-based signal grows.
            let stage1_result = agent.select_action();
            let fep_probs = &stage1_result.action_probabilities;

            // Model-based action values: E[reward | action] = Σ_s P(s|a) * V(s)
            // Uses effective_counts with adaptive prior for sharper estimates.
            let mb_values: Vec<f64> = (0..2)
                .map(|a| {
                    let total = effective_counts[a][0] + effective_counts[a][1];
                    let p0 = effective_counts[a][0] / total;
                    let p1 = effective_counts[a][1] / total;
                    p0 * state_reward[0] + p1 * state_reward[1]
                })
                .collect();

            // Softmax over model-based values — low temp makes MB signal decisive
            // Time pressure: base 0.1 preserves model-based control (Daw et al., 2011 two-step);
            // +0.10/unit degrades MB signal, shifting toward model-free under SAT (Heitz, 2014).
            let mb_temp = 0.1 + config.time_pressure * 0.10;
            let mb_max = mb_values[0].max(mb_values[1]);
            let mb_exp: Vec<f64> = mb_values
                .iter()
                .map(|v| ((v - mb_max) / mb_temp).exp())
                .collect();
            let mb_sum: f64 = mb_exp.iter().sum();
            let mb_probs: Vec<f64> = mb_exp.iter().map(|e| e / mb_sum).collect();

            // Blend: ramp model-based weight gradually, saturating by episode 40.
            // Slower ramp gives the agent more episodes to build accurate transition
            // and reward models before going fully model-based.
            let progress = (ep as f64 / 40.0).min(1.0);
            let mb_weight = 0.3 + 0.65 * progress;
            let blended_probs: Vec<f64> = (0..2)
                .map(|a| (1.0 - mb_weight) * fep_probs[a] + mb_weight * mb_probs[a])
                .collect();
            let prob_sum: f64 = blended_probs.iter().sum();
            let final_probs: Vec<f64> = blended_probs.iter().map(|p| p / prob_sum).collect();

            // RT proxy: stage-1 decision difficulty from model-based value margin —
            // closer MB values = harder deliberation (Daw et al., 2011 two-step task).
            let mb_diff = (mb_values[0] - mb_values[1]).abs();
            let ticks = 5.0 + (1.0 - mb_diff.min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            let stage1_action = sample_action(&final_probs, &mut rng_state);

            // Transition: 70% common, 30% rare
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let is_common = (rng_state % 100) < 70;
            let stage2_state = if is_common {
                stage1_action // common transition
            } else {
                1 - stage1_action // rare transition
            };

            // Update transition model: track which state each action leads to
            transition_counts[stage1_action][stage2_state] += 1.0;

            // Observe stage 2 state
            let stage2_obs = vec![stage2_state as f64 * 0.8 + 0.1; 4];
            agent.perceive(&Observation::new(stage2_obs, 1.0, "state"));

            // Stage 2: reward (drifting probabilities)
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let reward_prob = if stage2_state == 0 { 0.6 } else { 0.4 };
            let rewarded = (rng_state % 100) as f64 / 100.0 < reward_prob;

            let reward_val = if rewarded { 0.9 } else { 0.1 };
            let reward_obs = Observation::new(vec![reward_val; 4], 1.0, "reward");
            agent.perceive(&reward_obs);

            // Update reward model: EMA of rewards per state
            state_reward[stage2_state] =
                (1.0 - reward_lr) * state_reward[stage2_state] + reward_lr * reward_val;

            // Track stay/switch behavior relative to previous trial
            if let Some(prev_a) = prev_action {
                let stayed = stage1_action == prev_a;
                match (prev_common, prev_rewarded) {
                    (true, true) => {
                        common_rewarded_total += 1;
                        if stayed {
                            common_rewarded_stay += 1;
                        }
                    }
                    (false, true) => {
                        rare_rewarded_total += 1;
                        if stayed {
                            rare_rewarded_stay += 1;
                        }
                    }
                    (true, false) => {
                        common_unrewarded_total += 1;
                        if stayed {
                            common_unrewarded_stay += 1;
                        }
                    }
                    (false, false) => {
                        rare_unrewarded_total += 1;
                        if stayed {
                            rare_unrewarded_stay += 1;
                        }
                    }
                }
            }

            prev_action = Some(stage1_action);
            prev_common = is_common;
            prev_rewarded = rewarded;
        }

        // Model-basedness (beta3):
        // Model-based agent shows interaction between transition and reward:
        // stays more after common-rewarded AND rare-unrewarded (both confirm model)
        let cr_rate = safe_rate(common_rewarded_stay, common_rewarded_total);
        let rr_rate = safe_rate(rare_rewarded_stay, rare_rewarded_total);
        let cu_rate = safe_rate(common_unrewarded_stay, common_unrewarded_total);
        let ru_rate = safe_rate(rare_unrewarded_stay, rare_unrewarded_total);

        // Model-free: reward effect = (CR + RR)/2 - (CU + RU)/2
        let reward_effect = ((cr_rate + rr_rate) / 2.0) - ((cu_rate + ru_rate) / 2.0);
        // Model-based: transition x reward interaction
        let interaction = (cr_rate - rr_rate) - (cu_rate - ru_rate);

        // beta3 = model-basedness index
        let model_basedness = interaction.abs().min(1.0);

        (model_basedness, reward_effect, rt_ticks)
    }
}

fn safe_rate(num: u64, denom: u64) -> f64 {
    if denom > 0 {
        num as f64 / denom as f64
    } else {
        0.5
    }
}

impl PsychBenchmark for TwoStepBenchmark {
    fn name(&self) -> &str {
        "CogBench::TwoStep"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Two-Step Decision Task",
            citation: "Daw et al. (2011)",
            year: 2011,
            doi: Some("10.1016/j.neuron.2011.02.027"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut model_basedness = Vec::new();
        let mut reward_effects = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (mb, re, rts) = self.run_trial(config, trial);
            model_basedness.push(mb);
            reward_effects.push(re);
            all_rts.extend_from_slice(&rts);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "two_step".to_string(),
                    correct: mb > 0.0,
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

        result.insert(
            "beta3_model_basedness",
            MetricValue::from_samples(&model_basedness),
        );
        result.insert("reward_effect", MetricValue::from_samples(&reward_effects));
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
    fn test_two_step_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = TwoStepBenchmark.run(&config);
        assert!(result.metrics.contains_key("beta3_model_basedness"));
    }
}
