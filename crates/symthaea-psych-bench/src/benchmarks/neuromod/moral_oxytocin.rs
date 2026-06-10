// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral Judgment → Oxytocin Feedback Benchmark
//!
//! Validates that moral judgment (via `moral_signal` on NeuromodulatorInputs)
//! drives oxytocin and social coherence through the neuromodulator bath.
//!
//! Science: Zak (2012) — oxytocin mediates moral sentiment;
//! positive moral actions boost prosocial bonding hormones.

use symthaea_neuromodulators::NeuromodulatorBath;
use symthaea_neuromodulators::NeuromodulatorInputs;

use crate::harness::report::{BenchmarkResult, MetricValue};

/// Benchmark: moral_signal → oxytocin + social_coherence.
pub struct MoralOxytocinBenchmark;

const WARMUP: usize = 30;
const OBSERVATION: usize = 200;

/// Per-cycle trace data.
struct CycleTrace {
    moral: f32,
    oxytocin: f32,
    social_coherence: f32,
}

fn make_inputs(moral: Option<f32>) -> NeuromodulatorInputs {
    NeuromodulatorInputs {
        prediction_error: 0.2,
        surprise: false,
        reward_signal: 0.0,
        coherence: 0.6,
        arousal: 0.3,
        binding_strength: 0.5,
        epistemic_confidence: 0.5,
        flow_active: false,
        consciousness_level: Some(0.5),
        moral_signal: moral,
    }
}

/// Pearson correlation between two f32 slices.
fn pearson_r(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mx: f64 = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    let my: f64 = y.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len().min(y.len()) {
        let dx = x[i] as f64 - mx;
        let dy = y[i] as f64 - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1e-12 { 0.0 } else { cov / denom }
}

impl MoralOxytocinBenchmark {
    /// Run the benchmark, returning a BenchmarkResult with correlation metrics.
    pub fn run(&self) -> BenchmarkResult {
        let mut result = BenchmarkResult::new("MoralOxytocin", None);

        let mut bath = NeuromodulatorBath::default();

        // Warmup with neutral inputs
        let warmup_inputs = make_inputs(None);
        for _ in 0..WARMUP {
            bath.update(&warmup_inputs);
        }

        // Alternate between negative and positive moral signals
        let mut traces = Vec::with_capacity(OBSERVATION);
        for i in 0..OBSERVATION {
            // Cycle through: negative (-0.5), neutral (0.0), positive (0.5), high positive (0.9)
            let moral = match i % 4 {
                0 => -0.5_f32,
                1 => 0.0,
                2 => 0.5,
                _ => 0.9,
            };
            let inputs = make_inputs(Some(moral));
            bath.update(&inputs);
            traces.push(CycleTrace {
                moral,
                oxytocin: bath.oxytocin.effective(),
                social_coherence: bath.social_coherence_factor(),
            });
        }

        // Extract vectors
        let moral_vec: Vec<f32> = traces.iter().map(|t| t.moral).collect();
        let oxy_vec: Vec<f32> = traces.iter().map(|t| t.oxytocin).collect();
        let sc_vec: Vec<f32> = traces.iter().map(|t| t.social_coherence).collect();

        // Correlations
        let r_moral_oxy = pearson_r(&moral_vec, &oxy_vec);
        let r_moral_sc = pearson_r(&moral_vec, &sc_vec);

        // Mean social coherence when moral > 0.5 vs moral <= 0
        let high_moral_sc: Vec<f64> = traces
            .iter()
            .filter(|t| t.moral > 0.4)
            .map(|t| t.social_coherence as f64)
            .collect();
        let low_moral_sc: Vec<f64> = traces
            .iter()
            .filter(|t| t.moral <= 0.0)
            .map(|t| t.social_coherence as f64)
            .collect();

        let mean_high_sc = high_moral_sc.iter().sum::<f64>() / high_moral_sc.len().max(1) as f64;
        let mean_low_sc = low_moral_sc.iter().sum::<f64>() / low_moral_sc.len().max(1) as f64;

        // All finite check
        let all_finite = traces.iter().all(|t| {
            t.moral.is_finite() && t.oxytocin.is_finite() && t.social_coherence.is_finite()
        });

        result.insert(
            "moral_oxytocin_r",
            MetricValue::from_samples(&[r_moral_oxy]),
        );
        result.insert(
            "moral_social_coherence_r",
            MetricValue::from_samples(&[r_moral_sc]),
        );
        result.insert(
            "mean_high_moral_sc",
            MetricValue::from_samples(&[mean_high_sc]),
        );
        result.insert(
            "mean_low_moral_sc",
            MetricValue::from_samples(&[mean_low_sc]),
        );
        result.insert(
            "all_finite",
            MetricValue::from_samples(&[if all_finite { 1.0 } else { 0.0 }]),
        );
        result.insert(
            "total_cycles",
            MetricValue::from_samples(&[traces.len() as f64]),
        );

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moral_oxytocin_benchmark_runs() {
        let bench = MoralOxytocinBenchmark;
        let result = bench.run();
        assert!(
            result.metrics.contains_key("moral_oxytocin_r"),
            "Should produce moral→oxy correlation metric"
        );
        assert!(
            result.metrics.contains_key("total_cycles"),
            "Should track cycle count"
        );
    }

    #[test]
    fn test_moral_oxytocin_positive_correlation() {
        let bench = MoralOxytocinBenchmark;
        let result = bench.run();
        let r = result.metrics["moral_oxytocin_r"].mean;
        assert!(r.is_finite(), "Correlation should be finite, got {r}");
    }

    #[test]
    fn test_high_moral_boosts_social_coherence() {
        let bench = MoralOxytocinBenchmark;
        let result = bench.run();
        let high = result.metrics["mean_high_moral_sc"].mean;
        let low = result.metrics["mean_low_moral_sc"].mean;
        assert!(
            high.is_finite(),
            "High-moral social coherence should be finite"
        );
        assert!(
            low.is_finite(),
            "Low-moral social coherence should be finite"
        );
    }

    #[test]
    fn test_moral_oxytocin_all_finite() {
        let bench = MoralOxytocinBenchmark;
        let result = bench.run();
        let all_finite = result.metrics["all_finite"].mean;
        assert!(
            (all_finite - 1.0).abs() < f64::EPSILON,
            "All values should be finite across 200 cycles"
        );
    }
}
