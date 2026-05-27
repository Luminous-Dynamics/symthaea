// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Stability Benchmark
//!
//! Measures how much noise each institutional axiom can absorb before its
//! nearest-neighbor identity changes. This per-axiom noise tolerance tests
//! whether the HDC encoding creates robust, well-separated institutional
//! representations.
//!
//! ## Key Claims Tested
//!
//! 1. **Mean Stability**: Average noise ceiling across all 12 axioms,
//!    normalized to [0, 1] (noise_ceiling / 0.50).
//! 2. **Min Stability**: Worst-case axiom — the weakest link in the
//!    institutional vocabulary.
//! 3. **Stability Variance**: How uniformly stable are the axioms?
//!    Low variance means all axioms are equally robust.
//!
//! ## References
//!
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.
//! - Plate, T. (2003). *Holographic Reduced Representations*. CSLI.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem};

/// Institutional Reasoning: Stability benchmark.
pub struct InstitutionalStabilityBenchmark;

struct TrialResult {
    mean_stability: f64,
    min_stability: f64,
    stability_variance: f64,
}

/// All 18 institutional axiom names.
const AXIOM_NAMES: &[&str] = &[
    "REVOLUTION",
    "FAILED_STATE",
    "BORDER_DISPUTE",
    "LEGITIMATE_GOVERNANCE",
    "REGULATORY_CAPTURE",
    "TRADE_AGREEMENT",
    "ECONOMIC_SANCTION",
    "CIVIL_DISOBEDIENCE",
    "DEMOCRATIC_ELECTION",
    "ARMS_EMBARGO",
    "SOCIAL_CONTRACT",
    "CORRUPTION",
    "PEACE_TREATY",
    "FEDERATION",
    "CONSTITUTIONAL_CRISIS",
    "IMPEACHMENT",
    "DIPLOMACY",
    "COLONIALISM",
];

/// Noise levels to sweep (fine-grained near the ceiling).
const NOISE_SWEEP: &[f32] = &[
    0.10, 0.15, 0.20, 0.25, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.49,
];

impl InstitutionalStabilityBenchmark {
    /// Find the noise ceiling for a single axiom: highest noise level at which
    /// the noisy HV's nearest neighbor is still the original axiom.
    fn axiom_noise_ceiling(
        algebra: &CompositionAlgebra,
        system: &PrimitiveSystem,
        axiom: &str,
        seed: u64,
    ) -> f32 {
        let hv = match algebra.get_encoding(axiom, system) {
            Some(hv) => hv,
            None => return 0.0,
        };

        let mut ceiling: f32 = 0.0;
        for &noise in NOISE_SWEEP {
            let noise_seed = seed.wrapping_mul((noise * 1000.0) as u64 + 1);
            let noisy = hv.add_noise(noise, noise_seed);

            let mut best_name = "";
            let mut best_sim: f32 = -1.0;
            for &candidate in AXIOM_NAMES {
                if let Some(c_hv) = algebra.get_encoding(candidate, system) {
                    let sim = noisy.similarity(&c_hv);
                    if sim > best_sim {
                        best_sim = sim;
                        best_name = candidate;
                    }
                }
            }
            if best_name == axiom {
                ceiling = noise;
            } else {
                break;
            }
        }
        ceiling
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("institutional", "stability", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        let mut ceilings = Vec::new();
        for (idx, &axiom) in AXIOM_NAMES.iter().enumerate() {
            let axiom_seed = seed.wrapping_add(idx as u64 * 7919);
            let ceiling = Self::axiom_noise_ceiling(&algebra, &system, axiom, axiom_seed);
            // Normalize: ceiling / 0.50
            ceilings.push((ceiling / 0.50) as f64);
        }

        let mean_stability = ceilings.iter().sum::<f64>() / ceilings.len() as f64;
        let min_stability = ceilings
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let variance = if ceilings.len() > 1 {
            let n = ceilings.len() as f64;
            ceilings
                .iter()
                .map(|x| (x - mean_stability).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };

        TrialResult {
            mean_stability,
            min_stability,
            stability_variance: variance,
        }
    }
}

impl PsychBenchmark for InstitutionalStabilityBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::Stability"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "HDC Noise Tolerance / Institutional Robustness",
            citation: "Kanerva, P. (2009). Hyperdimensional computing.",
            year: 2009,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut stabilities = Vec::new();
        let mut min_stabilities = Vec::new();
        let mut variances = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            stabilities.push(r.mean_stability);
            min_stabilities.push(r.min_stability);
            variances.push(r.stability_variance);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("min_stability".to_string(), r.min_stability);
                extra.insert("stability_variance".to_string(), r.stability_variance);
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "stability".to_string(),
                    correct: r.mean_stability > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.mean_stability,
                    confidence: r.min_stability,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "institutional_stability",
            MetricValue::from_samples(&stabilities),
        );
        result.insert(
            "institutional_min_stability",
            MetricValue::from_samples(&min_stabilities),
        );
        result.insert(
            "institutional_stability_variance",
            MetricValue::from_samples(&variances),
        );

        result.conditions = 3;
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
    fn test_stability_runs() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalStabilityBenchmark.run(&config);
        assert!(result.metrics.contains_key("institutional_stability"));
        assert!(result.metrics.contains_key("institutional_min_stability"));
        assert!(
            result
                .metrics
                .contains_key("institutional_stability_variance")
        );
    }

    #[test]
    fn test_stability_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalStabilityBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_print_stability_scores() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalStabilityBenchmark.run(&config);
        eprintln!("\n═══ Institutional Stability Benchmark Scores ═══");
        for (key, val) in &result.metrics {
            let short = key.strip_prefix("institutional_").unwrap_or(key);
            eprintln!("  {short:<35} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }

        // Per-axiom stability details
        let seed = config.trial_seed("institutional", "stability", 0);
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        eprintln!("\n  ── Per-axiom noise ceilings ──");
        for (idx, &axiom) in AXIOM_NAMES.iter().enumerate() {
            let axiom_seed = seed.wrapping_add(idx as u64 * 7919);
            let ceiling = InstitutionalStabilityBenchmark::axiom_noise_ceiling(
                &algebra, &system, axiom, axiom_seed,
            );
            let normalized = ceiling / 0.50;
            eprintln!("  {axiom:<25} ceiling={ceiling:.2}  norm={normalized:.4}");
        }
        eprintln!("══════════════════════════════════════════════════\n");
    }
}
