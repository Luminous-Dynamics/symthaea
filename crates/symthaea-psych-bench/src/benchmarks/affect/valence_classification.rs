// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Valence classification task.
//!
//! Creates 3 valence prototypes (positive/negative/neutral) and generates
//! stimuli as weighted bundles of object + valence. Tests whether the system
//! can correctly classify stimuli by their affective valence using HDC
//! similarity.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Valence classification benchmark.
pub struct ValenceClassificationBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

impl ValenceClassificationBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> ([f64; 4], f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("affect", "valence", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Create 3 valence prototypes
        let positive_proto = ContinuousHV::random(dim, next_seed(&mut rng));
        let negative_proto = ContinuousHV::random(dim, next_seed(&mut rng));
        let neutral_proto = ContinuousHV::random(dim, next_seed(&mut rng));

        let protos = [&positive_proto, &negative_proto, &neutral_proto];

        // Generate 30 stimuli (10 per valence)
        let mut correct = [0usize; 3]; // positive, negative, neutral
        let mut total = [0usize; 3];
        let mut trial_rts = Vec::new();

        for valence in 0..3 {
            for _item in 0..10 {
                total[valence] += 1;

                // Create object HV
                let object_hv = ContinuousHV::random(dim, next_seed(&mut rng));

                // Blend: 60% object + 40% valence prototype.
                // Time pressure: 0.15/unit shifts weight from valence to object, modeling degraded
                // affective encoding under speed emphasis (Pessoa, 2009 dual-competition model).
                let noise_weight = config.time_pressure * 0.15;
                let valence_weight = (0.4 - noise_weight) as f32;
                let object_weight = (0.6 + noise_weight) as f32;
                let stimulus = ContinuousHV::weighted_bundle(
                    &[&object_hv, protos[valence]],
                    &[object_weight, valence_weight],
                );

                // Classify by max similarity to prototypes
                let sims: Vec<f32> = protos.iter().map(|p| stimulus.similarity(p)).collect();
                let predicted = sims
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == valence {
                    correct[valence] += 1;
                }

                // RT proxy: margin between best and second-best similarity
                let mut sorted_sims = sims.clone();
                sorted_sims.sort_by(|a, b| b.total_cmp(a));
                let margin = (sorted_sims[0] - sorted_sims[1]) as f64;
                let base = 3.0;
                let range = 6.0;
                let rt = base + (1.0 - margin.min(1.0)) * range;
                trial_rts.push(rt);
            }
        }

        let pos_acc = correct[0] as f64 / total[0] as f64;
        let neg_acc = correct[1] as f64 / total[1] as f64;
        let neu_acc = correct[2] as f64 / total[2] as f64;
        let overall =
            (correct[0] + correct[1] + correct[2]) as f64 / (total[0] + total[1] + total[2]) as f64;

        let mean_rt = trial_rts.iter().sum::<f64>() / trial_rts.len().max(1) as f64;
        ([overall, pos_acc, neg_acc, neu_acc], mean_rt)
    }
}

impl PsychBenchmark for ValenceClassificationBenchmark {
    fn name(&self) -> &str {
        "Affect::ValenceClassification"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Affective Valence Classification",
            citation: "Russell (1980)",
            year: 1980,
            doi: Some("10.1037/h0077714"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut overall_accs = Vec::new();
        let mut pos_accs = Vec::new();
        let mut neg_accs = Vec::new();
        let mut neu_accs = Vec::new();
        let mut rt_ticks = Vec::new();

        for trial in 0..config.trials_per_condition {
            let ([overall, pos, neg, neu], rt) = self.run_trial(config, trial);
            overall_accs.push(overall);
            pos_accs.push(pos);
            neg_accs.push(neg);
            neu_accs.push(neu);
            rt_ticks.push(rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "valence".to_string(),
                    correct: overall > 0.5,
                    rt_ticks: rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("valence_accuracy", MetricValue::from_samples(&overall_accs));
        result.insert("positive_accuracy", MetricValue::from_samples(&pos_accs));
        result.insert("negative_accuracy", MetricValue::from_samples(&neg_accs));
        result.insert("neutral_accuracy", MetricValue::from_samples(&neu_accs));
        result.insert("rt_ticks", MetricValue::from_samples(&rt_ticks));

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
    fn test_valence_classification_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = ValenceClassificationBenchmark.run(&config);
        assert!(result.metrics.contains_key("valence_accuracy"));
        assert!(result.metrics.contains_key("positive_accuracy"));
        assert!(result.metrics.contains_key("negative_accuracy"));
        assert!(result.metrics.contains_key("neutral_accuracy"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite());
        }
    }
}
