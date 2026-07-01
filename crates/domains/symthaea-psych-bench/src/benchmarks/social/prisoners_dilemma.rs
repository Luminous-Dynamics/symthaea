// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prisoner's Dilemma.
//!
//! Two players simultaneously choose to cooperate or defect. Mutual cooperation
//! yields a good outcome; mutual defection yields a poor one. Unilateral defection
//! tempts with the highest payoff but punishes the cooperator.
//!
//! HDC implementation: Cooperate and defect prototypes encode the two strategies.
//! Temptation-to-defect (T/R ratio) varies across conditions. Social cognition
//! (moral reasoning, reciprocity) increases cooperation rate beyond Nash equilibrium.
//!
//! Human baselines (Sally, 1995 meta-analysis; Rapoport & Chammah, 1965):
//! - cooperation_rate: 0.47 (SD~0.15) — one-shot cooperation
//! - mutual_cooperation_rate: 0.30 (SD~0.12) — fraction mutual-C outcomes
//! - payoff_efficiency: 0.65 (SD~0.10) — actual / mutual-C payoff

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Prisoner's Dilemma benchmark.
pub struct PrisonersDilemmaBenchmark;

struct TrialResult {
    cooperation_rate: f64,
    mutual_cooperation_rate: f64,
    exploitation_asymmetry: f64,
    payoff_efficiency: f64,
    rt_ticks: f64,
}

impl PrisonersDilemmaBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("social", "prisoners_dilemma", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Prototypes: cooperate and defect
        let cooperate_proto = ContinuousHV::random(dim, seed.wrapping_add(1));
        let defect_proto = ContinuousHV::random(dim, seed.wrapping_add(2));

        let noise_level: f32 = (0.15 + config.time_pressure as f32 * 0.15)
            * diff_model.interference_multiplier(config.difficulty) as f32;

        // Social cognition: moral reasoning and reciprocity norms boost cooperation
        let social_bonus: f32 = if config.enable_social { 0.15 } else { 0.0 };

        // Temptation-to-reward ratios: higher T/R makes defection more tempting
        let tr_ratios: [f32; 5] = [1.2, 1.5, 2.0, 3.0, 5.0];
        let reward = 1.0f32; // R normalized to 1.0
        let trials_per_condition = 8;

        let mut cooperations_by_condition = [0u32; 5];
        let mut total_by_condition = [0u32; 5];
        let mut total_cooperations = 0u32;
        let mut total_decisions = 0u32;
        let mut rt_sum = 0.0f64;
        let mut payoff_sum = 0.0f64;

        for (cond_idx, &temptation) in tr_ratios.iter().enumerate() {
            for _ in 0..trials_per_condition {
                // Encode the payoff landscape as a blend of cooperate/defect prototypes
                let coop_weight = reward / (reward + temptation);
                let defect_weight = temptation / (reward + temptation);
                let scenario_hv = ContinuousHV::weighted_bundle(
                    &[&cooperate_proto, &defect_proto],
                    &[coop_weight, defect_weight],
                );

                // Decision: compare scenario similarity to cooperate prototype
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let coop_sim =
                    scenario_hv.similarity(&cooperate_proto) * (1.0 - noise_degrade) + social_bonus;
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;

                // Cooperate if similarity to cooperate prototype exceeds threshold
                let threshold = 0.50 - config.time_pressure as f32 * 0.08;
                let cooperate = (coop_sim + noise) >= threshold;

                if cooperate {
                    cooperations_by_condition[cond_idx] += 1;
                    total_cooperations += 1;
                    // Cooperative payoff: R=1.0 (assuming mutual cooperation)
                    payoff_sum += reward as f64;
                } else {
                    // Defect payoff: average of T (exploit) and P (mutual defect)
                    let punishment = 0.5f32; // P < R
                    payoff_sum += ((temptation + punishment) / 2.0) as f64;
                }
                total_by_condition[cond_idx] += 1;
                total_decisions += 1;

                // RT: higher temptation = more conflict = slower
                let conflict = (temptation - 1.0).min(4.0) / 4.0;
                let base_rt = 4.0 + conflict as f64 * 5.0;
                let tp_speedup = config.time_pressure * 1.5;
                rt_sum += (base_rt - tp_speedup).max(1.0);
            }
        }

        let cooperation_rate = if total_decisions > 0 {
            total_cooperations as f64 / total_decisions as f64
        } else {
            0.0
        };

        // Mutual cooperation: P(both cooperate) ≈ P(C)^2
        let mutual_cooperation_rate = cooperation_rate * cooperation_rate;

        // Exploitation asymmetry: variance in cooperation rate across conditions
        let mut coop_rates = [0.0f64; 5];
        for (i, &total) in total_by_condition.iter().enumerate() {
            coop_rates[i] = if total > 0 {
                cooperations_by_condition[i] as f64 / total as f64
            } else {
                0.0
            };
        }
        let mean_coop: f64 = coop_rates.iter().sum::<f64>() / 5.0;
        let variance: f64 = coop_rates
            .iter()
            .map(|r| (r - mean_coop).powi(2))
            .sum::<f64>()
            / 5.0;
        let exploitation_asymmetry = variance.sqrt().clamp(0.0, 1.0);

        // Payoff efficiency: actual payoff / mutual-cooperation payoff
        let max_payoff = total_decisions as f64 * reward as f64;
        let payoff_efficiency = if max_payoff > 0.0 {
            (payoff_sum / max_payoff).clamp(0.0, 2.0)
        } else {
            0.0
        };

        TrialResult {
            cooperation_rate,
            mutual_cooperation_rate,
            exploitation_asymmetry,
            payoff_efficiency,
            rt_ticks: rt_sum / total_decisions.max(1) as f64,
        }
    }
}

impl PsychBenchmark for PrisonersDilemmaBenchmark {
    fn name(&self) -> &str {
        "Social::PrisonersDilemma"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Prisoner's Dilemma",
            citation: "Rapoport & Chammah (1965); Sally (1995)",
            year: 1965,
            doi: Some("10.1037/h0049469"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut coop_rates = Vec::new();
        let mut mutual_rates = Vec::new();
        let mut asymmetries = Vec::new();
        let mut efficiencies = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            coop_rates.push(r.cooperation_rate);
            mutual_rates.push(r.mutual_cooperation_rate);
            asymmetries.push(r.exploitation_asymmetry);
            efficiencies.push(r.payoff_efficiency);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "prisoners_dilemma".to_string(),
                    correct: r.cooperation_rate > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("cooperation_rate", MetricValue::from_samples(&coop_rates));
        result.insert(
            "mutual_cooperation_rate",
            MetricValue::from_samples(&mutual_rates),
        );
        result.insert(
            "exploitation_asymmetry",
            MetricValue::from_samples(&asymmetries),
        );
        result.insert(
            "payoff_efficiency",
            MetricValue::from_samples(&efficiencies),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 5; // 5 T/R ratio conditions
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
    fn test_prisoners_dilemma_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = PrisonersDilemmaBenchmark.run(&config);
        assert!(result.metrics.contains_key("cooperation_rate"));
        assert!(result.metrics.contains_key("mutual_cooperation_rate"));
        assert!(result.metrics.contains_key("payoff_efficiency"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_prisoners_dilemma_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = PrisonersDilemmaBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_prisoners_dilemma_cooperation_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PrisonersDilemmaBenchmark.run(&config);
        let rate = result.metrics["cooperation_rate"].mean;
        assert!(
            rate >= 0.0 && rate <= 1.0,
            "cooperation rate ({:.3}) out of bounds",
            rate
        );
    }

    #[test]
    fn test_prisoners_dilemma_efficiency_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PrisonersDilemmaBenchmark.run(&config);
        let eff = result.metrics["payoff_efficiency"].mean;
        assert!(eff >= 0.0, "payoff efficiency ({:.3}) should be >= 0", eff);
    }

    #[test]
    fn test_prisoners_dilemma_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = PrisonersDilemmaBenchmark.run(&base);
        let r_press = PrisonersDilemmaBenchmark.run(&pressed);
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
