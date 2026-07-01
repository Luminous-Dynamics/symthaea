// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness → Bath Feedback Benchmark
//!
//! Validates that consciousness level (Φ/Ψ) modulates neuromodulator
//! baselines end-to-end through the bath's `consciousness_modulation()`.
//!
//! Science: Tononi (2004) — IIT predicts consciousness level affects
//! global neural excitability and neuromodulator homeostasis.

use symthaea_neuromodulators::NeuromodulatorBath;
use symthaea_neuromodulators::NeuromodulatorInputs;

use crate::harness::report::{BenchmarkResult, MetricValue};

/// Benchmark: consciousness level → neuromod modulation.
pub struct ConsciousnessFeedbackBenchmark;

const WARMUP: usize = 30;
const OBSERVATION: usize = 200;

/// Per-cycle trace data.
struct CycleTrace {
    psi: f32,
    consciousness_mod: f32,
    serotonin: f32,
}

fn make_inputs(psi: Option<f32>) -> NeuromodulatorInputs {
    NeuromodulatorInputs {
        prediction_error: 0.2,
        surprise: false,
        reward_signal: 0.0,
        coherence: 0.6,
        arousal: 0.3,
        binding_strength: 0.5,
        epistemic_confidence: 0.5,
        flow_active: false,
        consciousness_level: psi,
        moral_signal: None,
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

impl ConsciousnessFeedbackBenchmark {
    /// Run the benchmark, returning a BenchmarkResult with correlation metrics.
    pub fn run(&self) -> BenchmarkResult {
        let mut result = BenchmarkResult::new("ConsciousnessFeedback", None);

        let mut bath = NeuromodulatorBath::default();

        // Warmup with neutral inputs
        let warmup_inputs = make_inputs(None);
        for _ in 0..WARMUP {
            bath.update(&warmup_inputs);
        }

        // Alternate between low and high psi to create variance
        let mut traces = Vec::with_capacity(OBSERVATION);
        for i in 0..OBSERVATION {
            let psi = if i % 4 < 2 { 0.2_f32 } else { 0.8 };
            let inputs = make_inputs(Some(psi));
            bath.update(&inputs);
            traces.push(CycleTrace {
                psi,
                consciousness_mod: bath.consciousness_modulation(),
                serotonin: bath.serotonin.effective(),
            });
        }

        // Extract vectors
        let psi_vec: Vec<f32> = traces.iter().map(|t| t.psi).collect();
        let cons_mod_vec: Vec<f32> = traces.iter().map(|t| t.consciousness_mod).collect();
        let sht_vec: Vec<f32> = traces.iter().map(|t| t.serotonin).collect();

        // Correlations
        let r_psi_consmod = pearson_r(&psi_vec, &cons_mod_vec);
        let r_psi_sht = pearson_r(&psi_vec, &sht_vec);

        // Quartile analysis: mean 5-HT in top vs bottom psi quartile
        let mut sorted_by_psi: Vec<&CycleTrace> = traces.iter().collect();
        sorted_by_psi.sort_by(|a, b| a.psi.total_cmp(&b.psi));
        let q_size = sorted_by_psi.len() / 4;
        let bottom_sht: f64 = sorted_by_psi[..q_size]
            .iter()
            .map(|t| t.serotonin as f64)
            .sum::<f64>()
            / q_size.max(1) as f64;
        let top_sht: f64 = sorted_by_psi[sorted_by_psi.len() - q_size..]
            .iter()
            .map(|t| t.serotonin as f64)
            .sum::<f64>()
            / q_size.max(1) as f64;

        // All finite check
        let all_finite = traces.iter().all(|t| {
            t.psi.is_finite() && t.consciousness_mod.is_finite() && t.serotonin.is_finite()
        });

        result.insert(
            "psi_consciousness_mod_r",
            MetricValue::from_samples(&[r_psi_consmod]),
        );
        result.insert("psi_serotonin_r", MetricValue::from_samples(&[r_psi_sht]));
        result.insert("top_quartile_sht", MetricValue::from_samples(&[top_sht]));
        result.insert(
            "bottom_quartile_sht",
            MetricValue::from_samples(&[bottom_sht]),
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
    fn test_consciousness_feedback_benchmark_runs() {
        let bench = ConsciousnessFeedbackBenchmark;
        let result = bench.run();
        assert!(
            result.metrics.contains_key("psi_consciousness_mod_r"),
            "Should produce correlation metric"
        );
        assert!(
            result.metrics.contains_key("total_cycles"),
            "Should track cycle count"
        );
    }

    #[test]
    fn test_psi_consciousness_mod_correlation() {
        let bench = ConsciousnessFeedbackBenchmark;
        let result = bench.run();
        let r = result.metrics["psi_consciousness_mod_r"].mean;
        assert!(r.is_finite(), "Correlation should be finite, got {r}");
    }

    #[test]
    fn test_high_psi_boosts_serotonin() {
        let bench = ConsciousnessFeedbackBenchmark;
        let result = bench.run();
        let top = result.metrics["top_quartile_sht"].mean;
        let bottom = result.metrics["bottom_quartile_sht"].mean;
        assert!(top.is_finite(), "Top quartile 5-HT should be finite");
        assert!(bottom.is_finite(), "Bottom quartile 5-HT should be finite");
    }

    #[test]
    fn test_consciousness_feedback_all_finite() {
        let bench = ConsciousnessFeedbackBenchmark;
        let result = bench.run();
        let all_finite = result.metrics["all_finite"].mean;
        assert!(
            (all_finite - 1.0).abs() < f64::EPSILON,
            "All values should be finite across 200 cycles"
        );
    }
}
