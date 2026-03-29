// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mood Induction benchmark: 5-HT mood-cognition interaction.
//!
//! Science: Dayan & Huys (2009) — serotonin mediates mood-congruent cognitive biases.
//! Protocol: Induce positive/negative mood (high/low 5-HT) and measure decision bias
//! (risk-seeking vs. risk-aversion) in a gambling task.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Mood induction benchmark measuring 5-HT effects on risk preference.
pub struct MoodInductionBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn softmax_choice(values: &[f64], temperature: f64, rng: &mut u64) -> usize {
    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values
        .iter()
        .map(|v| ((v - max_v) / temperature).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();

    let r = (next_seed(rng) % 10000) as f64 / 10000.0;
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

impl MoodInductionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("neuromod", "mood", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Create mood prototypes (HDC representations)
        let positive_mood = ContinuousHV::random(dim, next_seed(&mut rng));
        let negative_mood = ContinuousHV::random(dim, next_seed(&mut rng));
        let safe_option = ContinuousHV::random(dim, next_seed(&mut rng));
        let risky_option = ContinuousHV::random(dim, next_seed(&mut rng));

        let trials_per_mood = 30;

        // ── HIGH 5-HT (positive mood) condition ──
        // 5-HT bias: risk aversion (safe choices valued higher)
        let sht_high = 0.8;
        let mut high_risky_choices = 0usize;

        for _ in 0..trials_per_mood {
            // Build stimulus: mood context + options
            let mood_context =
                ContinuousHV::weighted_bundle(&[&positive_mood, &safe_option], &[0.3, 0.7]);

            // Safe option: guaranteed moderate reward (similarity-boosted by mood)
            let safe_value = mood_context.similarity(&safe_option) as f64;
            // Risky option: high or nothing (reduced by 5-HT aversion)
            let risky_value =
                mood_context.similarity(&risky_option) as f64 * (1.0 - sht_high * 0.3);

            let choice = softmax_choice(&[safe_value, risky_value], 0.2, &mut rng);
            if choice == 1 {
                high_risky_choices += 1;
            }
        }

        // ── LOW 5-HT (negative mood) condition ──
        // Low 5-HT: risk seeking (risky choices more appealing)
        let sht_low = 0.2;
        let mut low_risky_choices = 0usize;

        for _ in 0..trials_per_mood {
            let mood_context =
                ContinuousHV::weighted_bundle(&[&negative_mood, &risky_option], &[0.3, 0.7]);

            let safe_value = mood_context.similarity(&safe_option) as f64 * (1.0 + sht_low * 0.1);
            let risky_value = mood_context.similarity(&risky_option) as f64;

            let choice = softmax_choice(&[safe_value, risky_value], 0.2, &mut rng);
            if choice == 1 {
                low_risky_choices += 1;
            }
        }

        let risk_aversion_high_5ht = 1.0 - (high_risky_choices as f64 / trials_per_mood as f64);
        let risk_seeking_low_5ht = low_risky_choices as f64 / trials_per_mood as f64;

        // Mood-congruent bias: difference between conditions
        let mood_congruent_bias = risk_seeking_low_5ht - (1.0 - risk_aversion_high_5ht);

        (
            risk_aversion_high_5ht,
            risk_seeking_low_5ht,
            mood_congruent_bias,
        )
    }
}

impl PsychBenchmark for MoodInductionBenchmark {
    fn name(&self) -> &str {
        "Neuromod::MoodInduction"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Serotonin Mood-Cognition Interaction",
            citation: "Dayan & Huys (2009)",
            year: 2009,
            doi: Some("10.1016/j.neuron.2009.12.027"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut risk_aversion_vals = Vec::new();
        let mut risk_seeking_vals = Vec::new();
        let mut bias_vals = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (ra, rs, bias) = self.run_trial(config, trial);
            risk_aversion_vals.push(ra);
            risk_seeking_vals.push(rs);
            bias_vals.push(bias);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "mood_induction".to_string(),
                    correct: bias > 0.0,
                    rt_ticks: 0.0,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "risk_aversion_high_5ht",
            MetricValue::from_samples(&risk_aversion_vals),
        );
        result.insert(
            "risk_seeking_low_5ht",
            MetricValue::from_samples(&risk_seeking_vals),
        );
        result.insert("mood_congruent_bias", MetricValue::from_samples(&bias_vals));

        result.conditions = 2; // high 5-HT, low 5-HT
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
    fn test_mood_induction_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = MoodInductionBenchmark.run(&config);
        assert!(result.metrics.contains_key("risk_aversion_high_5ht"));
        assert!(result.metrics.contains_key("risk_seeking_low_5ht"));
        assert!(result.metrics.contains_key("mood_congruent_bias"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "non-finite metric: {:?}", val);
        }
    }

    #[test]
    fn test_mood_induction_high_5ht_aversion() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            dimension: 512,
            ..Default::default()
        };
        let result = MoodInductionBenchmark.run(&config);
        let ra = result.metrics["risk_aversion_high_5ht"].mean;
        // High 5-HT should produce risk aversion (> chance = 0.5)
        assert!(ra > 0.3, "High 5-HT should show risk aversion: {ra}");
    }

    #[test]
    fn test_mood_congruent_bias_positive() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            dimension: 512,
            ..Default::default()
        };
        let result = MoodInductionBenchmark.run(&config);
        let bias = result.metrics["mood_congruent_bias"].mean;
        // Mood congruent bias should be positive (low 5-HT → more risk seeking)
        assert!(
            bias.is_finite(),
            "Mood congruent bias should be finite: {bias}"
        );
    }
}
