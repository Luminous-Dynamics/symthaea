// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Drift-Consciousness Coupling Benchmark
//!
//! Tests whether an abrupt shift in moral trajectory produces measurable
//! consciousness degradation (cognitive dissonance), followed by recovery
//! as the new moral stance stabilizes.
//!
//! ## Method
//!
//! 1. Generate 40 HDC vectors clustered around a "harmony A" moral prototype
//! 2. Abruptly switch to 20 vectors clustered around a conflicting "harmony B" prototype
//! 3. Follow with 40 more "harmony B" vectors (stabilization)
//! 4. At each step, compute moral drift as L2 distance between first-half and
//!    second-half means of the last 20 harmony-projected vectors
//! 5. Simulate consciousness coupling: consciousness = base × (1 - drift × attenuation)
//! 6. Measure: (a) consciousness drop during transition, (b) recovery after stabilization
//!
//! ## Prediction
//!
//! - Consciousness drops > 5% during the transition window (cycles 40-60)
//! - Consciousness recovers to within 2% of baseline after stabilization (cycles 80-100)
//!
//! Science:
//! - Festinger, L. (1957). A Theory of Cognitive Dissonance.
//! - Kruglanski, A. (1989). Lay Epistemics and Human Knowledge.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Number of cycles in phase A (coherent moral stance).
const PHASE_A_CYCLES: usize = 40;

/// Number of cycles in transition (conflicting moral stance).
const TRANSITION_CYCLES: usize = 20;

/// Number of cycles in phase B (stabilized new stance).
const PHASE_B_CYCLES: usize = 40;

/// Total cycles.
const TOTAL_CYCLES: usize = PHASE_A_CYCLES + TRANSITION_CYCLES + PHASE_B_CYCLES;

/// Lookback window for drift computation.
const DRIFT_LOOKBACK: usize = 20;

/// Moral drift → consciousness attenuation strength.
const ATTENUATION_STRENGTH: f64 = 0.3;

/// Number of harmony axes (Eight Harmonies).
const N_HARMONIES: usize = 8;

/// Drift-Consciousness Coupling Benchmark.
pub struct DriftCouplingBenchmark;

impl DriftCouplingBenchmark {
    /// Project a ContinuousHV onto N_HARMONIES dimensions via stride-based averaging.
    fn harmony_projection(hv: &ContinuousHV) -> [f64; N_HARMONIES] {
        let data = &hv.values;
        let stride = data.len() / N_HARMONIES;
        let mut proj = [0.0f64; N_HARMONIES];
        for (h, p) in proj.iter_mut().enumerate() {
            let start = h * stride;
            let end = ((h + 1) * stride).min(data.len());
            let sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            *p = sum / (end - start) as f64;
        }
        // Softmax normalization
        let max = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = proj.iter().map(|&v| (v - max).exp()).sum();
        for p in proj.iter_mut() {
            *p = (*p - max).exp() / exp_sum;
        }
        proj
    }

    /// Compute moral drift as L2 distance between first-half and second-half means.
    fn moral_drift(trajectory: &[[f64; N_HARMONIES]], lookback: usize) -> f64 {
        let window: Vec<_> = trajectory.iter().rev().take(lookback).collect();
        if window.len() < 4 {
            return 0.0;
        }
        let mid = window.len() / 2;
        let (first_half, second_half) = (&window[..mid], &window[mid..]);

        let mut mean_first = [0.0f64; N_HARMONIES];
        let mut mean_second = [0.0f64; N_HARMONIES];
        for p in first_half {
            for (i, &v) in p.iter().enumerate() {
                mean_first[i] += v;
            }
        }
        for p in second_half {
            for (i, &v) in p.iter().enumerate() {
                mean_second[i] += v;
            }
        }
        for v in mean_first.iter_mut() {
            *v /= first_half.len() as f64;
        }
        for v in mean_second.iter_mut() {
            *v /= second_half.len() as f64;
        }

        // L2 distance
        mean_first
            .iter()
            .zip(mean_second.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Generate a moral prototype HV biased toward a specific harmony axis.
    fn make_prototype(dim: usize, dominant_harmony: usize, seed: u64) -> ContinuousHV {
        let mut hv = ContinuousHV::random(dim, seed);
        let stride = dim / N_HARMONIES;
        let start = dominant_harmony * stride;
        let end = ((dominant_harmony + 1) * stride).min(dim);
        // Bias the dominant harmony region toward positive values
        let data = &mut hv.values;
        for v in data[start..end].iter_mut() {
            *v = (*v + 0.5).clamp(-1.0, 1.0);
        }
        hv
    }

    /// Generate a scenario HV near a prototype with controlled noise.
    fn near_prototype(prototype: &ContinuousHV, noise: f32, seed: u64) -> ContinuousHV {
        let noise_hv = ContinuousHV::random(prototype.dim(), seed);
        let mut result = Vec::with_capacity(prototype.dim());
        for (p, n) in prototype.values.iter().zip(noise_hv.values.iter()) {
            result.push(p * (1.0 - noise) + n * noise);
        }
        ContinuousHV::from_values(result)
    }
}

impl PsychBenchmark for DriftCouplingBenchmark {
    fn name(&self) -> &str {
        "NormativeIntegration::DriftCoupling"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Moral Drift-Consciousness Coupling",
            citation: "Festinger (1957); Damasio (1994)",
            year: 1957,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let dim = config.dimension.max(256);
        let seed = config.seed;

        // Create two conflicting moral prototypes
        let prototype_a = Self::make_prototype(dim, 0, seed); // Harmony 0 dominant
        let prototype_b = Self::make_prototype(dim, 3, seed + 999); // Harmony 3 dominant

        let mut trajectory: Vec<[f64; N_HARMONIES]> = Vec::with_capacity(TOTAL_CYCLES);
        let mut consciousness_trace: Vec<f64> = Vec::with_capacity(TOTAL_CYCLES);
        let mut drift_trace: Vec<f64> = Vec::with_capacity(TOTAL_CYCLES);
        let mut trace = Vec::new();

        let base_consciousness = 0.8;

        for cycle in 0..TOTAL_CYCLES {
            // Generate scenario near the appropriate prototype
            let hv = if cycle < PHASE_A_CYCLES {
                Self::near_prototype(&prototype_a, 0.15, seed + cycle as u64 * 7)
            } else if cycle < PHASE_A_CYCLES + TRANSITION_CYCLES {
                // During transition: mix of both (simulates conflicting inputs)
                let t = (cycle - PHASE_A_CYCLES) as f32 / TRANSITION_CYCLES as f32;
                let a = Self::near_prototype(&prototype_a, 0.15, seed + cycle as u64 * 7);
                let b = Self::near_prototype(&prototype_b, 0.15, seed + cycle as u64 * 7 + 500);
                // Interpolate
                let mixed: Vec<f32> = a
                    .values
                    .iter()
                    .zip(b.values.iter())
                    .map(|(av, bv)| av * (1.0 - t) + bv * t)
                    .collect();
                ContinuousHV::from_values(mixed)
            } else {
                Self::near_prototype(&prototype_b, 0.15, seed + cycle as u64 * 7)
            };

            let proj = Self::harmony_projection(&hv);
            trajectory.push(proj);

            let drift = Self::moral_drift(&trajectory, DRIFT_LOOKBACK);
            drift_trace.push(drift);

            // Simulate consciousness coupling
            let consciousness =
                (base_consciousness * (1.0 - drift * ATTENUATION_STRENGTH)).clamp(0.0, 1.0);
            consciousness_trace.push(consciousness);

            if config.trial_trace {
                let phase = if cycle < PHASE_A_CYCLES {
                    "coherent_A"
                } else if cycle < PHASE_A_CYCLES + TRANSITION_CYCLES {
                    "transition"
                } else {
                    "stabilized_B"
                };
                let mut extra = BTreeMap::new();
                extra.insert("drift".into(), drift);
                extra.insert(
                    "phase_idx".into(),
                    if cycle < PHASE_A_CYCLES {
                        0.0
                    } else if cycle < PHASE_A_CYCLES + TRANSITION_CYCLES {
                        1.0
                    } else {
                        2.0
                    },
                );
                trace.push(TrialOutcome {
                    trial_idx: cycle,
                    condition: phase.to_string(),
                    correct: true,
                    rt_ticks: 0.0,
                    similarity: consciousness,
                    confidence: drift,
                    response_idx: 0,
                    extra,
                });
            }
        }

        // Compute metrics
        let baseline_consciousness: f64 =
            consciousness_trace[..PHASE_A_CYCLES].iter().sum::<f64>() / PHASE_A_CYCLES as f64;

        let transition_consciousness: f64 = consciousness_trace
            [PHASE_A_CYCLES..PHASE_A_CYCLES + TRANSITION_CYCLES]
            .iter()
            .sum::<f64>()
            / TRANSITION_CYCLES as f64;

        let stabilized_consciousness: f64 = consciousness_trace
            [PHASE_A_CYCLES + TRANSITION_CYCLES..]
            .iter()
            .sum::<f64>()
            / PHASE_B_CYCLES as f64;

        let consciousness_drop = baseline_consciousness - transition_consciousness;
        let consciousness_drop_pct = consciousness_drop / baseline_consciousness * 100.0;

        let recovery = (stabilized_consciousness - transition_consciousness).max(0.0);
        let recovery_pct = recovery / baseline_consciousness * 100.0;

        let peak_drift = drift_trace
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Prediction: consciousness should drop > 5% during transition
        let prediction_met = consciousness_drop_pct > 5.0;

        result.insert(
            "baseline_consciousness",
            MetricValue::from_samples(&[baseline_consciousness]),
        );
        result.insert(
            "transition_consciousness",
            MetricValue::from_samples(&[transition_consciousness]),
        );
        result.insert(
            "stabilized_consciousness",
            MetricValue::from_samples(&[stabilized_consciousness]),
        );
        result.insert(
            "consciousness_drop_pct",
            MetricValue::from_samples(&[consciousness_drop_pct]),
        );
        result.insert("recovery_pct", MetricValue::from_samples(&[recovery_pct]));
        result.insert("peak_drift", MetricValue::from_samples(&[peak_drift]));
        result.insert(
            "prediction_met",
            MetricValue::from_samples(&[if prediction_met { 1.0 } else { 0.0 }]),
        );

        result.conditions = 3; // coherent_A, transition, stabilized_B
        result.trials_per_condition = TOTAL_CYCLES / 3;
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
    fn test_drift_coupling_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = DriftCouplingBenchmark.run(&config);
        assert!(result.metrics.contains_key("consciousness_drop_pct"));
        assert!(result.metrics.contains_key("peak_drift"));
    }

    #[test]
    fn test_drift_coupling_produces_measurable_drop() {
        let config = BenchmarkConfig {
            dimension: 512,
            ..Default::default()
        };
        let result = DriftCouplingBenchmark.run(&config);
        let drop = result.metrics["consciousness_drop_pct"].mean;
        assert!(
            drop > 0.0,
            "Moral trajectory shift should produce nonzero consciousness drop, got {:.4}%",
            drop
        );
    }

    #[test]
    fn test_harmony_projection_sums_to_one() {
        let hv = ContinuousHV::random(256, 42);
        let proj = DriftCouplingBenchmark::harmony_projection(&hv);
        let sum: f64 = proj.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Softmax projection should sum to 1.0, got {:.6}",
            sum
        );
    }

    #[test]
    fn test_moral_drift_zero_for_constant_trajectory() {
        let hv = ContinuousHV::random(256, 42);
        let proj = DriftCouplingBenchmark::harmony_projection(&hv);
        let trajectory = vec![proj; 20];
        let drift = DriftCouplingBenchmark::moral_drift(&trajectory, 20);
        assert!(
            drift < 1e-10,
            "Constant trajectory should have zero drift, got {:.6}",
            drift
        );
    }

    #[test]
    fn test_moral_drift_nonzero_for_shifted_trajectory() {
        let proto_a = DriftCouplingBenchmark::make_prototype(256, 0, 42);
        let proto_b = DriftCouplingBenchmark::make_prototype(256, 3, 99);
        let mut trajectory = Vec::new();
        for i in 0..10 {
            let hv = DriftCouplingBenchmark::near_prototype(&proto_a, 0.1, i as u64);
            trajectory.push(DriftCouplingBenchmark::harmony_projection(&hv));
        }
        for i in 10..20 {
            let hv = DriftCouplingBenchmark::near_prototype(&proto_b, 0.1, i as u64);
            trajectory.push(DriftCouplingBenchmark::harmony_projection(&hv));
        }
        let drift = DriftCouplingBenchmark::moral_drift(&trajectory, 20);
        assert!(
            drift > 0.01,
            "Shifted trajectory should have nonzero drift, got {:.6}",
            drift
        );
    }

    #[test]
    fn test_near_prototype_preserves_dimension() {
        let proto = DriftCouplingBenchmark::make_prototype(256, 2, 42);
        let near = DriftCouplingBenchmark::near_prototype(&proto, 0.3, 99);
        assert_eq!(near.dim(), 256);
    }

    #[test]
    fn test_prediction_met_flag() {
        let config = BenchmarkConfig {
            dimension: 512,
            ..Default::default()
        };
        let result = DriftCouplingBenchmark.run(&config);
        // prediction_met should be present as a metric
        assert!(result.metrics.contains_key("prediction_met"));
    }
}
