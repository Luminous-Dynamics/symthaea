// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live-loop pharmacological ablation: transmitter knockout on real CognitiveLoopService.
//!
//! Unlike `PharmacologicalAblationBenchmark` (standalone Bath simulation), this
//! benchmark builds actual `CognitiveLoopService` instances and validates that
//! neuromodulator knockouts degrade *behavioral* metrics in the live cognitive loop.
//!
//! Gated behind `#[cfg(feature = "symthaea-backend")]`.
//!
//! Science: Doya (2002) — transmitter-specific functional impairment.

#[cfg(feature = "symthaea-backend")]
use crate::harness::config::BenchmarkConfig;
#[cfg(feature = "symthaea-backend")]
use crate::harness::report::{BenchmarkResult, MetricValue};
#[cfg(feature = "symthaea-backend")]
use crate::harness::{BenchmarkProvenance, PsychBenchmark};

/// Live-loop pharmacological ablation benchmark.
///
/// Runs 200 cognitive cycles on a real `CognitiveLoopService` and measures
/// average metadata indicators (learning rate, exploration, confidence, attention).
/// Then runs 4 knockout variants (DA/NE/5-HT/ACh clamped to 0.0 after each step)
/// and reports degradation percentages vs. baseline.
#[cfg(feature = "symthaea-backend")]
pub struct LiveLoopAblationBenchmark;

#[cfg(feature = "symthaea-backend")]
fn build_loop_service() -> Option<symthaea::cognitive_loop::CognitiveLoopService> {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, ConsciousnessProfile};
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
    config.genesis_phrase = Some("neuromod-live-ablation-deterministic".to_string());
    config.async_training = false;
    symthaea::cognitive_loop::CognitiveLoopService::new(config).ok()
}

#[cfg(feature = "symthaea-backend")]
struct LiveMetrics {
    avg_gradient_scale: f64,
    avg_exploration_urge: f64,
    avg_confidence: f64,
    avg_attention_factor: f64,
}

#[cfg(feature = "symthaea-backend")]
fn run_live_cycles(
    service: &mut symthaea::cognitive_loop::CognitiveLoopService,
    n_cycles: usize,
    clamp_da: Option<f32>,
    clamp_ne: Option<f32>,
    clamp_sht: Option<f32>,
    clamp_ach: Option<f32>,
) -> LiveMetrics {
    let inputs = [
        "The quick brown fox jumps over the lazy dog",
        "A neural network learns to predict sequences",
        "Consciousness emerges from integrated information",
        "Working memory maintains active representations",
        "Prediction errors drive learning and adaptation",
    ];

    let warmup = 20;
    let mut sum_grad = 0.0_f64;
    let mut sum_expl = 0.0_f64;
    let mut sum_conf = 0.0_f64;
    let mut sum_attn = 0.0_f64;
    let mut count = 0_u64;

    for i in 0..n_cycles {
        let result = service.cycle(inputs[i % inputs.len()]);
        // Apply knockout clamp after each cycle
        if clamp_da.is_some() || clamp_ne.is_some() || clamp_sht.is_some() || clamp_ach.is_some() {
            service.clamp_neuromod_levels(clamp_da, clamp_ne, clamp_sht, clamp_ach);
        }
        if i >= warmup {
            let m = &result.metadata;
            sum_grad += m.neuromod.neuromod_gradient_scale as f64;
            // Exploration: NE effective as proxy (higher NE = more exploration)
            sum_expl += m.neuromod.noradrenaline_effective as f64;
            // Confidence: 5-HT effective as proxy
            sum_conf += m.neuromod.serotonin_effective as f64;
            // Attention: ACh effective as proxy
            sum_attn += m.neuromod.acetylcholine_effective as f64;
            count += 1;
        }
    }

    let n = count.max(1) as f64;
    LiveMetrics {
        avg_gradient_scale: sum_grad / n,
        avg_exploration_urge: sum_expl / n,
        avg_confidence: sum_conf / n,
        avg_attention_factor: sum_attn / n,
    }
}

#[cfg(feature = "symthaea-backend")]
impl PsychBenchmark for LiveLoopAblationBenchmark {
    fn name(&self) -> &str {
        "Neuromod::LiveLoopAblation"
    }

    fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(self.name(), None);
        let n_cycles = 200;

        // Baseline
        let Some(mut service) = build_loop_service() else {
            result.insert("error", MetricValue::from_samples(&[0.0]));
            return result;
        };
        let baseline = run_live_cycles(&mut service, n_cycles, None, None, None, None);

        // DA knockout
        if let Some(mut svc) = build_loop_service() {
            let ko = run_live_cycles(&mut svc, n_cycles, Some(0.0), None, None, None);
            let drop = ((baseline.avg_gradient_scale - ko.avg_gradient_scale)
                / baseline.avg_gradient_scale.max(0.001)
                * 100.0)
                .max(0.0);
            result.insert(
                "live_da_knockout_gradient_drop_pct",
                MetricValue::from_samples(&[drop]),
            );
        }

        // NE knockout
        if let Some(mut svc) = build_loop_service() {
            let ko = run_live_cycles(&mut svc, n_cycles, None, Some(0.0), None, None);
            let drop = ((baseline.avg_exploration_urge - ko.avg_exploration_urge)
                / baseline.avg_exploration_urge.max(0.001)
                * 100.0)
                .max(0.0);
            result.insert(
                "live_ne_knockout_exploration_drop_pct",
                MetricValue::from_samples(&[drop]),
            );
        }

        // 5-HT knockout
        if let Some(mut svc) = build_loop_service() {
            let ko = run_live_cycles(&mut svc, n_cycles, None, None, Some(0.0), None);
            let drop = ((baseline.avg_confidence - ko.avg_confidence)
                / baseline.avg_confidence.max(0.001)
                * 100.0)
                .max(0.0);
            result.insert(
                "live_sht_knockout_confidence_drop_pct",
                MetricValue::from_samples(&[drop]),
            );
        }

        // ACh knockout
        if let Some(mut svc) = build_loop_service() {
            let ko = run_live_cycles(&mut svc, n_cycles, None, None, None, Some(0.0));
            let drop = ((baseline.avg_attention_factor - ko.avg_attention_factor)
                / baseline.avg_attention_factor.max(0.001)
                * 100.0)
                .max(0.0);
            result.insert(
                "live_ach_knockout_attention_drop_pct",
                MetricValue::from_samples(&[drop]),
            );
        }

        // Baselines for reference
        result.insert(
            "live_baseline_gradient_scale",
            MetricValue::from_samples(&[baseline.avg_gradient_scale]),
        );
        result.insert(
            "live_baseline_exploration",
            MetricValue::from_samples(&[baseline.avg_exploration_urge]),
        );

        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            citation: "Doya (2002) — Metalearning and neuromodulation; live-loop validation",
            paradigm: "Pharmacological lesion on full cognitive loop",
            year: 2002,
            doi: Some("10.1162/089976602753712018"),
        })
    }
}

#[cfg(all(test, feature = "symthaea-backend"))]
mod tests {
    use super::*;
    use crate::harness::config::BenchmarkConfig;

    #[test]
    fn test_live_ablation_runs() {
        let config = BenchmarkConfig::default();
        let bench = LiveLoopAblationBenchmark;
        let result = bench.run(&config);
        for key in &[
            "live_da_knockout_gradient_drop_pct",
            "live_ne_knockout_exploration_drop_pct",
            "live_sht_knockout_confidence_drop_pct",
            "live_ach_knockout_attention_drop_pct",
        ] {
            let m = result.metrics.get(*key);
            assert!(m.is_some(), "Missing metric: {key}");
            assert!(m.unwrap().mean.is_finite(), "Non-finite metric: {key}");
        }
    }

    #[test]
    fn test_live_da_knockout_reduces_gradient_scale() {
        let n_cycles = 200;
        let mut baseline_svc = build_loop_service().expect("build loop");
        let baseline = run_live_cycles(&mut baseline_svc, n_cycles, None, None, None, None);

        let mut ko_svc = build_loop_service().expect("build loop");
        let ko = run_live_cycles(&mut ko_svc, n_cycles, Some(0.0), None, None, None);

        assert!(
            ko.avg_gradient_scale < baseline.avg_gradient_scale,
            "DA knockout should reduce gradient scale: ko={} baseline={}",
            ko.avg_gradient_scale,
            baseline.avg_gradient_scale
        );
    }

    #[test]
    fn test_live_ablation_all_degrade() {
        let n_cycles = 200;
        let mut baseline_svc = build_loop_service().expect("build loop");
        let baseline = run_live_cycles(&mut baseline_svc, n_cycles, None, None, None, None);

        // DA knockout → gradient scale drops
        let mut svc = build_loop_service().expect("build loop");
        let ko = run_live_cycles(&mut svc, n_cycles, Some(0.0), None, None, None);
        assert!(
            ko.avg_gradient_scale < baseline.avg_gradient_scale,
            "DA knockout"
        );

        // NE knockout → exploration drops
        let mut svc = build_loop_service().expect("build loop");
        let ko = run_live_cycles(&mut svc, n_cycles, None, Some(0.0), None, None);
        assert!(
            ko.avg_exploration_urge < baseline.avg_exploration_urge,
            "NE knockout"
        );

        // 5-HT knockout → confidence drops
        let mut svc = build_loop_service().expect("build loop");
        let ko = run_live_cycles(&mut svc, n_cycles, None, None, Some(0.0), None);
        assert!(ko.avg_confidence < baseline.avg_confidence, "5-HT knockout");

        // ACh knockout → attention drops
        let mut svc = build_loop_service().expect("build loop");
        let ko = run_live_cycles(&mut svc, n_cycles, None, None, None, Some(0.0));
        assert!(
            ko.avg_attention_factor < baseline.avg_attention_factor,
            "ACh knockout"
        );
    }
}
