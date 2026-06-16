// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Balloon Analogue Risk Task (BART).
//!
//! Tests risk-taking behavior: pump a balloon for increasing reward,
//! but it may pop (losing all). Measures average pumps and pop rate.
//!
//! Uses a target-pump model: the agent computes an optimal stopping point
//! from expected value (marginal gain vs marginal risk), modulated by the
//! FEP agent's risk sensitivity. This avoids the geometric decay problem
//! of per-step stochastic pump/cash-out decisions.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// BART benchmark measuring risk-taking.
pub struct BartBenchmark;

impl BartBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64, Vec<f64>) {
        let seed = config.trial_seed("cogbench", "bart", trial_idx);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let num_balloons = 20;
        let max_pumps = 64;

        // Hybrid: FEP agent for risk modulation + explicit EV for target pumps
        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: 2,
            action_temperature: config.action_temperature.max(5.0),
            planning_horizon: 1,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_config);
        agent.set_goals(vec![0.0, 1.0, 0.0, 0.0], 0.5);

        // Warmup: teach reward growth pattern
        for step in 0..15 {
            let reward_norm = (step as f64 * 0.1).min(1.0);
            let obs = Observation::new(
                vec![
                    step as f64 / 30.0,
                    reward_norm,
                    1.0 - step as f64 / 30.0,
                    0.0,
                ],
                1.0,
                "bart_warmup",
            );
            agent.perceive(&obs);
            let _ = agent.select_action();
            let next_r = ((step + 1) as f64 * 0.1).min(1.0);
            let outcome = Observation::new(
                vec![
                    (step + 1) as f64 / 30.0,
                    next_r,
                    1.0 - (step + 1) as f64 / 30.0,
                    0.0,
                ],
                1.0,
                "bart_warmup_outcome",
            );
            agent.learn_from_outcome(0, &outcome);
        }
        let pop_obs = Observation::new(vec![0.0, 0.0, 0.0, 1.0], 1.0, "bart_warmup_pop");
        agent.learn_from_outcome(0, &pop_obs);

        // Track experienced pop rate across balloons
        let mut experienced_pops = 0u64;
        let mut experienced_balloons = 0u64;
        let mut total_pumps = 0u64;
        let mut total_earnings = 0.0f64;
        let mut pops = 0u64;
        let mut rt_ticks = Vec::new();

        #[allow(clippy::explicit_counter_loop)]
        // experienced_balloons tracks cumulative state across pops, not just loop iteration
        for _ in 0..num_balloons {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pop_threshold = (rng_state % (max_pumps as u64 - 5)) as usize + 5;

            // Target-pump model: compute optimal stop via total EV.
            //
            // Pop threshold is uniform in [min_pop, max_pumps]. If we pump to T:
            //   P(survive) = max(0, max_pumps - T) / (max_pumps - min_pop)
            //   EV(T) = T * reward_per_pump * P(survive)
            // Optimal T maximizes EV(T) = T * (max_pumps - T) / range.
            // This gives T_opt = max_pumps / 2 (≈32 for max_pumps=64).
            //
            // The agent learns from experience: track observed pop thresholds
            // and estimate the min/max range adaptively.
            let reward_per_pump = 0.15;
            let min_pop_estimate = 5.0; // prior: balloons survive at least 5 pumps
            let max_pop_estimate = if experienced_balloons > 3 {
                // After pops, estimate max range from pop rate
                let avg_pumped = if experienced_balloons > experienced_pops {
                    total_pumps as f64 / (experienced_balloons - experienced_pops).max(1) as f64
                } else {
                    20.0
                };
                // Estimate max: roughly 2 * avg_pumped (since pop is ~uniform)
                (avg_pumped * 2.0).min(max_pumps as f64).max(20.0)
            } else {
                max_pumps as f64 // prior: full range
            };

            let range = (max_pop_estimate - min_pop_estimate).max(1.0);

            // Find pump count that maximizes total EV
            let mut best_ev = 0.0;
            let mut target_pumps = 1usize;
            for t in 1..max_pumps {
                let survive_prob = ((max_pop_estimate - t as f64) / range).clamp(0.0, 1.0);
                let ev = t as f64 * reward_per_pump * survive_prob;
                if ev > best_ev {
                    best_ev = ev;
                    target_pumps = t;
                }
            }

            // FEP agent modulates target
            let inflation_at_target = target_pumps as f64 / max_pumps as f64;
            let pop_rate_so_far = if experienced_balloons > 0 {
                experienced_pops as f64 / experienced_balloons as f64
            } else {
                0.3
            };
            let obs = Observation::new(
                vec![
                    inflation_at_target,
                    (target_pumps as f64 * reward_per_pump / 3.0).min(1.0),
                    1.0 - inflation_at_target,
                    pop_rate_so_far.min(1.0),
                ],
                1.0,
                "bart",
            );
            agent.perceive(&obs);
            let action_result = agent.select_action();
            let fep_pump_prob = action_result
                .action_probabilities
                .first()
                .copied()
                .unwrap_or(0.5);

            // FEP modulates target by +/- 12%
            let fep_adjustment = (fep_pump_prob - 0.5) * 0.24;
            let adjusted_target = ((target_pumps as f64 * (1.0 + fep_adjustment)).round() as usize)
                .max(1)
                .min(max_pumps);

            // Small noise (+/- 3 pumps); time pressure widens noise band
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            // Time pressure: base noise 4 pumps; +6/unit widens decision jitter, modeling impulsive
            // pump-to-target estimation under urgency (Lejuez et al., 2002 BART; Heitz, 2014 SAT).
            let noise_range = 6 + (config.time_pressure * 6.0) as u64;
            let noise = (rng_state % noise_range) as i64 - (noise_range as i64 / 2);
            let final_target = (adjusted_target as i64 + noise)
                .max(1)
                .min(max_pumps as i64) as usize;

            // Execute pumps to target (may pop before reaching it)
            let mut pumps = 0usize;
            let mut popped = false;
            for _ in 0..final_target {
                pumps += 1;
                if pumps >= pop_threshold {
                    popped = true;
                    pops += 1;
                    experienced_pops += 1;
                    let pop_obs = Observation::new(vec![0.0, 0.0, 0.0, 1.0], 1.0, "bart_pop");
                    agent.learn_from_outcome(0, &pop_obs);
                    break;
                }
            }

            // RT proxy: pumps per balloon reflect deliberation length —
            // more pumps = longer risk-assessment process (Lejuez et al., 2002).
            rt_ticks.push(pumps as f64);

            experienced_balloons += 1;
            if !popped {
                let accumulated = pumps as f64 * 0.15;
                total_earnings += accumulated;
                total_pumps += pumps as u64;
                let cashout_obs = Observation::new(
                    vec![0.0, (accumulated / 3.0).min(1.0), 1.0, 0.0],
                    1.0,
                    "bart_cashout",
                );
                agent.learn_from_outcome(1, &cashout_obs);
            }
        }

        let cashed_out = (num_balloons - pops as usize).max(1);
        let avg_pumps = total_pumps as f64 / cashed_out as f64;
        let pop_rate = pops as f64 / num_balloons as f64;
        let avg_earnings = total_earnings / num_balloons as f64;

        (avg_pumps, pop_rate, avg_earnings, rt_ticks)
    }
}

impl PsychBenchmark for BartBenchmark {
    fn name(&self) -> &str {
        "CogBench::BART"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Balloon Analogue Risk Task",
            citation: "Lejuez et al. (2002)",
            year: 2002,
            doi: Some("10.1037/1076-898X.8.2.75"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut pumps = Vec::new();
        let mut pop_rates = Vec::new();
        let mut earnings = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (p, pr, e, rts) = self.run_trial(config, trial);
            pumps.push(p);
            pop_rates.push(pr);
            earnings.push(e);
            all_rts.extend_from_slice(&rts);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "bart".to_string(),
                    correct: pr < 0.5,
                    rt_ticks: p,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("average_pumps", MetricValue::from_samples(&pumps));
        result.insert("pop_rate", MetricValue::from_samples(&pop_rates));
        result.insert("average_earnings", MetricValue::from_samples(&earnings));
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
