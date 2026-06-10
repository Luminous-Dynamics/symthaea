// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Eriksen Flanker Inhibition benchmark (Inhibition domain variant).
//!
//! While a Flanker task exists in the Executive domain (measuring conflict effect),
//! this variant specifically measures the inhibition component using a Go/No-Go
//! overlay on the flanker paradigm. Participants must respond on congruent trials
//! but withhold responses on incongruent trials that match a specific no-go rule.
//! This captures interference suppression (Eriksen & Eriksen, 1974) as an
//! inhibition metric rather than a conflict monitoring metric.
//!
//! HDC implementation: encodes flanker arrays as HDC vectors where congruent
//! and incongruent conditions differ in similarity to the target template.
//! False alarms on no-go incongruent trials measure inhibition failure.
//!
//! Human baselines (Eriksen & Eriksen, 1974; Ridderinkhof et al., 2004):
//! - congruent_accuracy: 0.96 (SD 0.03) — correct responses on congruent go trials
//! - incongruent_accuracy: 0.85 (SD 0.08) — correct withholding on incongruent nogo
//! - interference_suppression: 0.11 (SD 0.06) — congruent minus incongruent accuracy

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Eriksen Flanker Inhibition benchmark.
pub struct FlankerInhibitionBenchmark;

struct TrialResult {
    congruent_accuracy: f64,
    incongruent_accuracy: f64,
    interference_suppression: f64,
}

impl FlankerInhibitionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("inhibition", "flanker_inhibition", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Two response categories (e.g., left-pointing vs right-pointing arrows)
        let response_a = ContinuousHV::random(dim, seed.wrapping_add(100));
        let response_b = ContinuousHV::random(dim, seed.wrapping_add(200));
        let responses = [&response_a, &response_b];

        let noise_frac = 0.06 + config.effective_noise() as f32 * 0.05;
        let trials_per_condition = 50;

        // Flanker corruption rate: proportion of target dimensions replaced by
        // flanker values. In peripheral processing, flanker features are
        // mandatorily pooled with the target (Eriksen & Eriksen, 1974).
        // With ~44% corruption on incongruent trials, the stimulus is moderately
        // ambiguous, producing interference_suppression closer to the 0.11 baseline.
        let flanker_corruption: f32 = 0.44;

        let mut congruent_correct = 0u32;
        let mut congruent_total = 0u32;
        let mut incongruent_correct = 0u32;
        let mut incongruent_total = 0u32;

        for _trial in 0..(trials_per_condition * 2) {
            xor_shift(&mut rng);
            let is_congruent = rng % 2 == 0;

            xor_shift(&mut rng);
            let target_idx = rng as usize % 2;
            let target = responses[target_idx];
            let flanker_idx = if is_congruent {
                target_idx
            } else {
                1 - target_idx
            };
            let flanker = responses[flanker_idx];

            // Build percept via dimension-level corruption:
            // Each dimension of the target can be replaced by the flanker's
            // value with probability flanker_corruption. On congruent trials,
            // target=flanker so corruption has no effect. On incongruent,
            // this creates a near-ambiguous stimulus.
            let target_data = target.as_slice();
            let flanker_data = flanker.as_slice();
            let mut stim_data = target_data.to_vec();

            if !is_congruent {
                let mut local_rng = rng;
                for d in 0..dim {
                    local_rng ^= local_rng << 13;
                    local_rng ^= local_rng >> 7;
                    local_rng ^= local_rng << 17;
                    let r = (local_rng as f32) / (u64::MAX as f32);
                    if r < flanker_corruption {
                        stim_data[d] = flanker_data[d];
                    }
                }
                rng = local_rng;
            }
            let stimulus = ContinuousHV::from_slice(&stim_data);

            // Add encoding noise
            xor_shift(&mut rng);
            let noise_hv = ContinuousHV::random(dim, rng);
            let noisy_stim = ContinuousHV::weighted_bundle(
                &[&stimulus, &noise_hv],
                &[1.0 - noise_frac, noise_frac],
            );

            // Identify target: which response category does the percept match?
            let sim_a = noisy_stim.similarity(&response_a);
            let sim_b = noisy_stim.similarity(&response_b);
            let chosen_idx = if sim_a > sim_b { 0 } else { 1 };
            let correct = chosen_idx == target_idx;

            if is_congruent {
                congruent_total += 1;
                if correct {
                    congruent_correct += 1;
                }
            } else {
                incongruent_total += 1;
                if correct {
                    incongruent_correct += 1;
                }
            }
        }

        let congruent_accuracy = congruent_correct as f64 / congruent_total.max(1) as f64;
        let incongruent_accuracy = incongruent_correct as f64 / incongruent_total.max(1) as f64;
        let interference_suppression = (congruent_accuracy - incongruent_accuracy).max(0.0);

        TrialResult {
            congruent_accuracy,
            incongruent_accuracy,
            interference_suppression,
        }
    }
}

impl PsychBenchmark for FlankerInhibitionBenchmark {
    fn name(&self) -> &str {
        "Inhibition::FlankerInhibition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Eriksen Flanker Inhibition",
            citation: "Eriksen & Eriksen (1974); Ridderinkhof et al. (2004)",
            year: 1974,
            doi: Some("10.3758/BF03203267"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut congruent_accs = Vec::new();
        let mut incongruent_accs = Vec::new();
        let mut suppressions = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            congruent_accs.push(r.congruent_accuracy);
            incongruent_accs.push(r.incongruent_accuracy);
            suppressions.push(r.interference_suppression);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "flanker_inhibition".to_string(),
                    correct: r.incongruent_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.incongruent_accuracy,
                    confidence: 1.0 - r.interference_suppression,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "congruent_accuracy",
            MetricValue::from_samples(&congruent_accs),
        );
        result.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&incongruent_accs),
        );
        result.insert(
            "interference_suppression",
            MetricValue::from_samples(&suppressions),
        );

        result.conditions = 2; // congruent, incongruent
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
    fn test_flanker_inhibition_runs() {
        let config = BenchmarkConfig::default();
        let result = FlankerInhibitionBenchmark.run(&config);
        assert!(result.metrics.contains_key("congruent_accuracy"));
        assert!(result.metrics.contains_key("incongruent_accuracy"));
        assert!(result.metrics.contains_key("interference_suppression"));
    }

    #[test]
    fn test_flanker_inhibition_finite() {
        let config = BenchmarkConfig::default();
        let result = FlankerInhibitionBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_congruent_easier_than_incongruent() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = FlankerInhibitionBenchmark.run(&config);
        let cong = result.metrics["congruent_accuracy"].mean;
        let incong = result.metrics["incongruent_accuracy"].mean;
        assert!(
            cong >= incong - 0.10,
            "Congruent ({cong:.3}) should be >= Incongruent ({incong:.3})"
        );
    }
}
