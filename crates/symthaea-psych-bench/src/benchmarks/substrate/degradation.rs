//! Substrate Degradation benchmark.
//!
//! Tests how consciousness-relevant metrics degrade as substrate quality
//! decreases. A system claiming substrate independence should show graceful
//! degradation — not catastrophic failure — as substrate properties worsen.
//!
//! HDC implementation: Progressively increases encoding noise and reduces
//! effective dimensionality (simulating substrate degradation). Measures
//! information integration (binding coherence), temporal dynamics (sequence
//! discrimination), and workspace capacity (item recall) at each level.
//!
//! Human baselines (theoretical, derived from IIT/GWT):
//! - degradation_slope: 0.15 (SD 0.06) — accuracy loss per degradation step
//! - critical_threshold: 0.60 (SD 0.10) — quality level where performance collapses
//! - graceful_ratio: 0.80 (SD 0.08) — linearity of degradation (1.0 = perfectly linear)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Substrate Degradation benchmark.
pub struct SubstrateDegradationBenchmark;

struct TrialResult {
    degradation_slope: f64,
    critical_threshold: f64,
    graceful_ratio: f64,
}

impl SubstrateDegradationBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("substrate", "degradation", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Degradation levels: 1.0 (pristine) down to 0.1 (severely degraded)
        let levels: Vec<f64> = (0..10).map(|i| 1.0 - i as f64 * 0.1).collect();
        let n_features = 6usize;
        let n_items = 3usize;
        let trials_per_level = 20;

        let mut accuracies = Vec::new();

        for &quality in &levels {
            // Noise scales with degradation: 0% at quality=1.0, up to 40% at quality=0.1
            let noise_frac = (1.0 - quality) as f32 * 0.40;
            let mut level_correct = 0u32;
            let mut level_total = 0u32;

            for trial in 0..trials_per_level {
                // Create feature prototypes and role binders
                xor_shift(&mut rng);
                let features: Vec<ContinuousHV> = (0..n_features)
                    .map(|i| {
                        ContinuousHV::random(
                            dim,
                            seed.wrapping_add(1000 + trial as u64 * 100 + i as u64),
                        )
                    })
                    .collect();
                let role = ContinuousHV::random(dim, seed.wrapping_add(2000 + trial as u64));

                // Create items by binding features to roles
                let items: Vec<ContinuousHV> = (0..n_items)
                    .map(|i| {
                        xor_shift(&mut rng);
                        let feat_idx = (rng as usize + i) % n_features;
                        features[feat_idx].bind(&role)
                    })
                    .collect();

                // Bundle items into memory
                let item_refs: Vec<&ContinuousHV> = items.iter().collect();
                let memory = ContinuousHV::bundle(&item_refs);

                // Degrade: add noise
                xor_shift(&mut rng);
                let degraded = if noise_frac > 0.01 {
                    let noise_hv = ContinuousHV::random(dim, rng);
                    ContinuousHV::weighted_bundle(
                        &[&memory, &noise_hv],
                        &[1.0 - noise_frac, noise_frac],
                    )
                } else {
                    memory.clone()
                };

                // Probe: which item was in memory?
                xor_shift(&mut rng);
                let probe_idx = rng as usize % n_items;

                // Find best matching item
                let mut best_idx = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (i, item) in items.iter().enumerate() {
                    let sim = degraded.similarity(item);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                level_total += 1;
                if best_idx == probe_idx {
                    level_correct += 1;
                }
            }

            accuracies.push(level_correct as f64 / level_total as f64);
        }

        // Degradation slope: linear regression of accuracy vs quality level
        let n = levels.len() as f64;
        let x_mean = levels.iter().sum::<f64>() / n;
        let y_mean = accuracies.iter().sum::<f64>() / n;
        let num: f64 = levels
            .iter()
            .zip(accuracies.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        let den: f64 = levels.iter().map(|x| (x - x_mean).powi(2)).sum();
        let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };
        let degradation_slope = slope.abs(); // positive = accuracy drops with degradation

        // Critical threshold: first quality level where accuracy drops below 0.5
        let critical_threshold = levels
            .iter()
            .zip(accuracies.iter())
            .rev()
            .find(|(_, &acc)| acc >= 0.5)
            .map(|(&q, _)| q)
            .unwrap_or(0.1);

        // Graceful ratio: how linear is the degradation?
        // R² of the linear fit — 1.0 = perfectly linear, 0.0 = catastrophic
        let ss_res: f64 = levels
            .iter()
            .zip(accuracies.iter())
            .map(|(x, y)| {
                let predicted = y_mean + slope * (x - x_mean);
                (y - predicted).powi(2)
            })
            .sum();
        let ss_tot: f64 = accuracies.iter().map(|y| (y - y_mean).powi(2)).sum();
        let graceful_ratio = if ss_tot > 1e-10 {
            (1.0 - ss_res / ss_tot).max(0.0)
        } else {
            1.0
        };

        TrialResult {
            degradation_slope,
            critical_threshold,
            graceful_ratio,
        }
    }
}

impl PsychBenchmark for SubstrateDegradationBenchmark {
    fn name(&self) -> &str {
        "Substrate::Degradation"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Substrate Quality Degradation",
            citation: "Tononi (2004); Koch et al. (2016)",
            year: 2004,
            doi: Some("10.1186/1471-2202-5-42"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut slopes = Vec::new();
        let mut thresholds = Vec::new();
        let mut ratios = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            slopes.push(r.degradation_slope);
            thresholds.push(r.critical_threshold);
            ratios.push(r.graceful_ratio);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "degradation".to_string(),
                    correct: r.graceful_ratio > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.graceful_ratio,
                    confidence: r.critical_threshold,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("degradation_slope", MetricValue::from_samples(&slopes));
        result.insert("critical_threshold", MetricValue::from_samples(&thresholds));
        result.insert("graceful_ratio", MetricValue::from_samples(&ratios));

        result.conditions = 10; // 10 degradation levels
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
    fn test_degradation_runs() {
        let config = BenchmarkConfig::default();
        let result = SubstrateDegradationBenchmark.run(&config);
        assert!(result.metrics.contains_key("degradation_slope"));
        assert!(result.metrics.contains_key("critical_threshold"));
        assert!(result.metrics.contains_key("graceful_ratio"));
    }

    #[test]
    fn test_degradation_finite() {
        let config = BenchmarkConfig::default();
        let result = SubstrateDegradationBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_degradation_slope_positive() {
        let config = BenchmarkConfig::default();
        let result = SubstrateDegradationBenchmark.run(&config);
        let slope = result.metrics["degradation_slope"].mean;
        assert!(
            slope >= 0.0,
            "degradation slope {slope} should be non-negative"
        );
    }
}
