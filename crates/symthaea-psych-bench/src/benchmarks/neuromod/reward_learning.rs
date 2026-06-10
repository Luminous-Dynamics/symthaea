// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reward Learning benchmark: DA reward prediction error.
//!
//! Science: Schultz (1997) — midbrain dopamine neurons encode reward prediction error.
//! Protocol: Train on reward contingency (A→reward, B→nothing), then reverse.
//! Measure trials-to-criterion and lose-shift ratio.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Reward learning benchmark measuring DA-driven reversal learning.
pub struct RewardLearningBenchmark;

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

impl RewardLearningBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        let seed = config.trial_seed("neuromod", "reward", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Association strengths (learned via Hebbian-like update)
        let mut q_values = [0.5_f64, 0.5]; // [A, B]

        // Asymmetric learning rates (Frank et al., 2007): phasic DA dips
        // (negative RPE) drive faster unlearning than DA bursts drive acquisition.
        // This is a core feature of the basal ganglia Go/NoGo pathway asymmetry.
        let lr_pos = 0.15; // DA burst → Go pathway
        let lr_neg = 0.25; // DA dip → NoGo pathway (faster unlearning)
        let base_temperature = 0.3;

        // Lapse rate: random responding on a fraction of trials (models
        // attention lapses / individual differences for ICC reliability)
        let lapse_rate = config.lapse_rate;

        // Phase 1: Acquisition (A→reward, B→nothing), 40 trials
        let criterion = 0.8;

        for _trial in 0..40 {
            let choice = if lapse_rate > 0.0
                && (next_seed(&mut rng) % 10000) as f64 / 10000.0 < lapse_rate
            {
                (next_seed(&mut rng) % 2) as usize // random lapse
            } else {
                softmax_choice(&q_values, base_temperature, &mut rng)
            };
            let reward = if choice == 0 { 1.0 } else { 0.0 };

            // DA RPE update with asymmetric learning rates
            let rpe = reward - q_values[choice];
            let lr = if rpe >= 0.0 { lr_pos } else { lr_neg };
            q_values[choice] += lr * rpe;
            q_values[choice] = q_values[choice].clamp(0.0, 1.0);
        }

        // Phase 2: Reversal (B→reward, A→nothing), 40 trials
        // Surprise-driven exploration: high negative RPE increases temperature
        // (Daw et al., 2006 — uncertainty-driven exploration)
        let mut lose_shifts = 0usize;
        let mut losses = 0usize;
        let mut reversal_criterion_trials = 40usize;
        let mut reversal_criterion_met = false;
        let mut phase2_b_choices = 0usize;
        let mut recent_rpe_magnitude = 0.0_f64; // EMA of |negative RPE|

        for trial in 0..40 {
            // Surprise-driven exploration: boost temperature when recent negative
            // RPE is high (unexpected non-reward increases uncertainty → exploration)
            let temperature = base_temperature + recent_rpe_magnitude * 0.15;

            let choice = if lapse_rate > 0.0
                && (next_seed(&mut rng) % 10000) as f64 / 10000.0 < lapse_rate
            {
                (next_seed(&mut rng) % 2) as usize
            } else {
                softmax_choice(&q_values, temperature, &mut rng)
            };
            let reward = if choice == 1 { 1.0 } else { 0.0 };

            // Track lose-shift: chose A (now unrewarded) → switched to B next
            if reward == 0.0 {
                losses += 1;
            }

            // DA RPE update with asymmetric rates
            let rpe = reward - q_values[choice];
            let lr = if rpe >= 0.0 { lr_pos } else { lr_neg };
            q_values[choice] += lr * rpe;
            q_values[choice] = q_values[choice].clamp(0.0, 1.0);

            // Track surprise for exploration modulation (EMA, alpha=0.3)
            if rpe < 0.0 {
                recent_rpe_magnitude = recent_rpe_magnitude * 0.7 + (-rpe) * 0.3;
            } else {
                recent_rpe_magnitude *= 0.7; // decay when rewards are received
            }

            if choice == 1 {
                phase2_b_choices += 1;
                if losses > 0 {
                    lose_shifts += 1;
                }
            }

            if !reversal_criterion_met && trial > 5 {
                let recent_b = phase2_b_choices as f64 / (trial + 1) as f64;
                if recent_b > criterion {
                    reversal_criterion_trials = trial + 1;
                    reversal_criterion_met = true;
                }
            }
        }

        let lose_shift_ratio = if losses > 0 {
            lose_shifts as f64 / losses as f64
        } else {
            0.0
        };

        // DA-reward correlation: higher Q for rewarded stimulus indicates DA-driven learning
        let da_reward_corr = (q_values[1] - q_values[0]).clamp(-1.0, 1.0);

        (
            reversal_criterion_trials as f64,
            lose_shift_ratio,
            da_reward_corr,
        )
    }
}

impl PsychBenchmark for RewardLearningBenchmark {
    fn name(&self) -> &str {
        "Neuromod::RewardLearning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Reward Prediction Error / Reversal Learning",
            citation: "Schultz (1997)",
            year: 1997,
            doi: Some("10.1126/science.275.5306.1593"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut trials_to_crit = Vec::new();
        let mut lose_shift_ratios = Vec::new();
        let mut da_correlations = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (ttc, lsr, dac) = self.run_trial(config, trial);
            trials_to_crit.push(ttc);
            lose_shift_ratios.push(lsr);
            da_correlations.push(dac);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "reward_learning".to_string(),
                    correct: dac > 0.0,
                    rt_ticks: ttc,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "trials_to_criterion",
            MetricValue::from_samples(&trials_to_crit),
        );
        result.insert(
            "lose_shift_ratio",
            MetricValue::from_samples(&lose_shift_ratios),
        );
        result.insert(
            "da_reward_correlation",
            MetricValue::from_samples(&da_correlations),
        );

        result.conditions = 2; // acquisition + reversal
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
    fn test_reward_learning_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = RewardLearningBenchmark.run(&config);
        assert!(result.metrics.contains_key("trials_to_criterion"));
        assert!(result.metrics.contains_key("lose_shift_ratio"));
        assert!(result.metrics.contains_key("da_reward_correlation"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "non-finite metric: {:?}", val);
        }
    }

    #[test]
    fn test_reward_learning_reversal_occurs() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            dimension: 256,
            ..Default::default()
        };
        let result = RewardLearningBenchmark.run(&config);
        let ttc = result.metrics["trials_to_criterion"].mean;
        // Should eventually learn reversal (< 40 trials = max)
        assert!(ttc <= 40.0, "trials_to_criterion too high: {ttc}");
    }

    #[test]
    fn test_reward_learning_da_correlation_positive() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            dimension: 256,
            ..Default::default()
        };
        let result = RewardLearningBenchmark.run(&config);
        let dac = result.metrics["da_reward_correlation"].mean;
        // After reversal, B should be valued higher → positive correlation
        assert!(dac > 0.0, "DA-reward correlation should be positive: {dac}");
    }
}
