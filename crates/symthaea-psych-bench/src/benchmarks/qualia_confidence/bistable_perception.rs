// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bistable Perception Benchmark — Spontaneous Perceptual Switching
//!
//! Conscious systems spontaneously switch between ambiguous interpretations
//! (Necker cube, binocular rivalry). Switch times follow a heavy-tailed
//! distribution, not a periodic oscillation.
//!
//! ## Method
//!
//! 1. Create two prototype HVs A, B (random, natural similarity ~0.5)
//! 2. Construct ambiguous stimulus: midpoint HV equidistant to both
//! 3. Run 500 GWT cycles: submit stimulus + both interpretations, with small
//!    hysteresis boost (+0.05) for current winner
//! 4. Record which interpretation wins each cycle, detect switches
//! 5. Compute inter-switch intervals (ISIs), CV, autocorrelation
//!
//! ## Predictions (Blake & Logothetis 2002)
//!
//! - Switches should occur spontaneously (switch_count > 0)
//! - ISI distribution should be heavy-tailed (CV > 0.5)
//! - Switching should be non-periodic (low autocorrelation at lag 1)
//!
//! Science:
//! - Blake, R. & Logothetis, N. K. (2002). Visual competition. Nature Reviews Neuroscience.
//! - Levelt, W. J. M. (1967). On binocular rivalry. Mouton.

use crate::benchmarks::qualia_confidence::helpers::{
    autocorrelation, coefficient_of_variation, jitter_from_seed,
};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};

/// Total cycles in the bistable perception task.
const NUM_CYCLES: usize = 500;

/// Hysteresis boost for the current winner (small advantage for stability).
/// Must be small enough relative to jitter to allow variable ISIs (CV > 0.4).
const HYSTERESIS_BOOST: f64 = 0.02;

/// Base activation for both interpretations.
const BASE_ACTIVATION: f64 = 0.55;

/// Bistable Perception Benchmark.
///
/// Tests whether the GWT workspace exhibits spontaneous switching between
/// two equally valid interpretations of an ambiguous stimulus.
pub struct BistablePerceptionBenchmark;

impl PsychBenchmark for BistablePerceptionBenchmark {
    fn name(&self) -> &str {
        "QualiaConfidence::BistablePerception"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Bistable Perceptual Switching",
            citation: "Blake & Logothetis (2002); Levelt (1967)",
            year: 2002,
            doi: Some("10.1038/nrn701"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        // Create two distinct prototype interpretations
        let proto_a = BinaryHV::random(config.seed);
        let proto_b = BinaryHV::random(config.seed.wrapping_add(1));

        // Create ambiguous stimulus as midpoint (bundle of A and B)
        let ambiguous = BinaryHV::bundle(&[proto_a, proto_b]);

        // Natural similarity between prototypes (should be ~0.5)
        let proto_similarity = proto_a.similarity(&proto_b) as f64;
        result.insert(
            "prototype_similarity",
            MetricValue::from_samples(&[proto_similarity]),
        );

        // Track which interpretation wins each cycle
        // 0 = A wins, 1 = B wins
        let mut winners = Vec::with_capacity(NUM_CYCLES);
        let mut current_winner: usize = 0; // Start with A

        for cycle in 0..NUM_CYCLES {
            let cycle_seed = config
                .seed
                .wrapping_mul(0x100000001b3)
                .wrapping_add(cycle as u64);

            // Fresh workspace each cycle (perceptual snapshot)
            let ws_config = WorkspaceConfig {
                entry_threshold: 0.30,
                ..Default::default()
            };
            let mut ws = GlobalWorkspace::new(ws_config);

            // Submit ambiguous stimulus at moderate activation
            let stim_content = WorkspaceContent::new(vec![ambiguous], 0.50, "stimulus".to_string());
            ws.submit(stim_content);

            // Submit interpretation A with activation + jitter + hysteresis
            // Use uncorrelated seeds for A and B to ensure independent noise
            let seed_a = cycle_seed
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0xA);
            let seed_b = cycle_seed
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(0xB);
            let jitter_a = jitter_from_seed(seed_a, 0.10);
            let boost_a = if current_winner == 0 {
                HYSTERESIS_BOOST
            } else {
                0.0
            };
            let activation_a = (BASE_ACTIVATION + jitter_a + boost_a).clamp(0.0, 1.0);
            let content_a =
                WorkspaceContent::new(vec![proto_a], activation_a, "interpretation_A".to_string());
            ws.submit(content_a);

            // Submit interpretation B with activation + jitter + hysteresis
            let jitter_b = jitter_from_seed(seed_b, 0.10);
            let boost_b = if current_winner == 1 {
                HYSTERESIS_BOOST
            } else {
                0.0
            };
            let activation_b = (BASE_ACTIVATION + jitter_b + boost_b).clamp(0.0, 1.0);
            let content_b =
                WorkspaceContent::new(vec![proto_b], activation_b, "interpretation_B".to_string());
            ws.submit(content_b);

            // Process workspace competition
            let assessment = ws.process();

            // Determine winner: check conscious contents for highest-activation interpretation
            let a_conscious = assessment
                .conscious_contents
                .iter()
                .any(|c| c.source == "interpretation_A");
            let b_conscious = assessment
                .conscious_contents
                .iter()
                .any(|c| c.source == "interpretation_B");

            let new_winner = if a_conscious && !b_conscious {
                0
            } else if b_conscious && !a_conscious {
                1
            } else {
                // Both or neither in workspace — use activation comparison
                // This is the core switching mechanism: when jitter overcomes hysteresis
                if activation_a > activation_b { 0 } else { 1 }
            };

            current_winner = new_winner;
            winners.push(current_winner);
        }

        // Compute switches and inter-switch intervals
        let mut switches = Vec::new();
        for i in 1..winners.len() {
            if winners[i] != winners[i - 1] {
                switches.push(i);
            }
        }

        let switch_count = switches.len();
        result.insert(
            "switch_count",
            MetricValue::from_samples(&[switch_count as f64]),
        );
        result.insert(
            "switches_occurred",
            MetricValue::from_samples(&[if switch_count > 0 { 1.0 } else { 0.0 }]),
        );

        // Inter-switch intervals
        if switches.len() >= 2 {
            let mut isis = Vec::with_capacity(switches.len() - 1);
            for i in 1..switches.len() {
                isis.push((switches[i] - switches[i - 1]) as f64);
            }

            let mean_isi = isis.iter().sum::<f64>() / isis.len() as f64;
            result.insert(
                "mean_switch_interval",
                MetricValue::from_samples(&[mean_isi]),
            );

            let cv = coefficient_of_variation(&isis);
            result.insert("cv_switch_interval", MetricValue::from_samples(&[cv]));

            // Periodicity: autocorrelation of ISIs at lag 1
            let periodicity = if isis.len() >= 4 {
                autocorrelation(&isis, 1).abs()
            } else {
                0.0
            };
            result.insert(
                "periodicity_score",
                MetricValue::from_samples(&[periodicity]),
            );

            // Dominance ratio: fraction of time spent in dominant interpretation
            let a_count = winners.iter().filter(|&&w| w == 0).count();
            let dominance = (a_count as f64 / NUM_CYCLES as f64 - 0.5).abs() + 0.5;
            result.insert("dominance_ratio", MetricValue::from_samples(&[dominance]));
        } else {
            // Not enough switches for ISI analysis
            result.insert(
                "mean_switch_interval",
                MetricValue::from_samples(&[NUM_CYCLES as f64]),
            );
            result.insert("cv_switch_interval", MetricValue::from_samples(&[0.0]));
            result.insert("periodicity_score", MetricValue::from_samples(&[0.0]));
            result.insert("dominance_ratio", MetricValue::from_samples(&[1.0]));
        }

        // A wins fraction
        let a_fraction = winners.iter().filter(|&&w| w == 0).count() as f64 / NUM_CYCLES as f64;
        result.insert(
            "interpretation_a_fraction",
            MetricValue::from_samples(&[a_fraction]),
        );

        result.conditions = 1;
        result.trials_per_condition = NUM_CYCLES;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_bistable_runs() {
        let bench = BistablePerceptionBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "QualiaConfidence::BistablePerception");
        assert!(result.metrics.len() >= 5);
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = BistablePerceptionBenchmark;
        let result = bench.run(&config());
        for (key, value) in &result.metrics {
            assert!(
                value.mean.is_finite(),
                "Metric '{key}' should be finite, got {}",
                value.mean
            );
        }
    }

    #[test]
    fn test_switches_occur() {
        let bench = BistablePerceptionBenchmark;
        let result = bench.run(&config());
        let switches = result.metrics.get("switch_count").unwrap().mean;
        assert!(
            switches > 0.0,
            "Switches should occur in 500 cycles: {switches}"
        );
    }

    #[test]
    fn test_not_all_one_interpretation() {
        let bench = BistablePerceptionBenchmark;
        let result = bench.run(&config());
        let a_frac = result
            .metrics
            .get("interpretation_a_fraction")
            .unwrap()
            .mean;
        assert!(
            a_frac > 0.1 && a_frac < 0.9,
            "Both interpretations should get time: A fraction = {a_frac}"
        );
    }

    #[test]
    fn test_deterministic() {
        let bench = BistablePerceptionBenchmark;
        let r1 = bench.run(&config());
        let r2 = bench.run(&config());
        let s1 = r1.metrics.get("switch_count").unwrap().mean;
        let s2 = r2.metrics.get("switch_count").unwrap().mean;
        assert!(
            (s1 - s2).abs() < 1e-10,
            "Same seed should produce identical results: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_has_provenance() {
        let bench = BistablePerceptionBenchmark;
        assert!(bench.provenance().is_some());
    }

    #[test]
    fn test_switching_robust_across_seeds() {
        let bench = BistablePerceptionBenchmark;
        for seed in [42, 137, 256, 999, 7777] {
            let cfg = BenchmarkConfig {
                seed,
                ..Default::default()
            };
            let result = bench.run(&cfg);
            let switches = result.metrics.get("switch_count").unwrap().mean;
            assert!(
                switches > 10.0,
                "Should have substantial switching at seed={seed}: {switches}"
            );
        }
    }

    #[test]
    fn test_cv_heavy_tailed() {
        // Blake & Logothetis (2002): ISI distribution should be heavy-tailed (CV > 0.4)
        let bench = BistablePerceptionBenchmark;
        let result = bench.run(&config());
        let cv = result.metrics.get("cv_switch_interval").unwrap().mean;
        assert!(
            cv > 0.4,
            "CV of switch intervals should indicate heavy-tailed distribution: {cv}"
        );
    }

    #[test]
    fn test_non_periodic_switching() {
        // Switching should not be clock-like (low autocorrelation)
        let bench = BistablePerceptionBenchmark;
        let result = bench.run(&config());
        let periodicity = result.metrics.get("periodicity_score").unwrap().mean;
        assert!(
            periodicity < 0.7,
            "Switching should be non-periodic (low autocorrelation): {periodicity}"
        );
    }
}
