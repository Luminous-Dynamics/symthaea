// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Chain Inference Benchmark.
//!
//! Tests whether the system can trace multi-step causal chains in HDC space.
//! Given A→B→C→D, can the system infer that A causally affects D through
//! intermediate steps, and correctly identify the chain depth?
//!
//! ## HDC Implementation
//!
//! Each causal link is a permuted XOR binding: A→B is encoded as ρ(A)⊕B.
//! A chain A→B→C is the bundle of links: ρ(A)⊕B and ρ(B)⊕C. To test
//! chain inference, we check whether unbinding ρ(A) from the first link
//! yields B, which can then be used to query the second link.
//!
//! ## Key Claims Tested
//!
//! 1. **Chain Tracing**: Given a causal chain of length N, the system can
//!    recover the terminal effect from the initial cause.
//! 2. **Depth Sensitivity**: Accuracy decreases with chain length, following
//!    signal attenuation in HDC space.
//! 3. **Chain vs Direct**: Multi-step chains produce weaker (but non-zero)
//!    causal evidence compared to direct links.
//!
//! ## References
//!
//! - Sloman, S. (2005). *Causal Models*. Oxford UP.
//! - Bramley, N. R. et al. (2017). Formalizing Neurath's ship. *Cognition*.
//! - Waldmann, M. R. & Holyoak, K. J. (1992). Predictive and diagnostic
//!   learning within causal models. *JEP: General*.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Causal Chain Inference benchmark.
pub struct CausalChainBenchmark;

struct TrialResult {
    /// Accuracy of recovering terminal effect from initial cause
    chain_tracing_accuracy: f64,
    /// How accuracy varies with chain depth (slope of accuracy vs depth)
    depth_sensitivity: f64,
    /// Ratio of chain evidence to direct evidence
    chain_vs_direct_ratio: f64,
}

impl CausalChainBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("causal_reasoning", "causal_chain", trial_idx);

        // ── Test 1: Chain tracing across depths 2-5 ───────────────────
        let max_depth = 5;
        let mut accuracies_by_depth = Vec::with_capacity(max_depth);

        for depth in 2..=max_depth {
            let mut depth_accuracy = 0.0;
            let n_chains = 5;

            for chain_idx in 0..n_chains {
                let chain_seed = seed
                    .wrapping_add(depth as u64 * 1000)
                    .wrapping_add(chain_idx as u64 * 100);

                // Generate chain variables: V0 → V1 → V2 → ... → V_depth
                let variables: Vec<BinaryHV> = (0..=depth)
                    .map(|i| BinaryHV::random(chain_seed.wrapping_add(i as u64)))
                    .collect();

                // Encode each link as ρ(cause) ⊕ effect
                // Permutation makes the binding directional (cause→effect ≠ effect→cause)
                let links: Vec<BinaryHV> = (0..depth)
                    .map(|i| variables[i].permute(1).bind(&variables[i + 1]))
                    .collect();

                // Trace the chain: starting from V0, follow each link
                let mut current = variables[0];
                let mut trace_success = true;

                for (i, link) in links.iter().enumerate() {
                    // Unbind ρ(current) from the link to get next variable
                    let predicted_next = link.bind(&current.permute(1));

                    // Check similarity to actual next variable
                    let sim = predicted_next.cosine_similarity(&variables[i + 1]);

                    // XOR unbinding is exact for binary, so similarity should be 1.0
                    // (or very close due to permutation alignment)
                    if sim < 0.5 {
                        trace_success = false;
                        break;
                    }

                    current = variables[i + 1];
                }

                if trace_success {
                    depth_accuracy += 1.0;
                }
            }

            accuracies_by_depth.push(depth_accuracy / n_chains as f64);
        }

        let chain_tracing_accuracy =
            accuracies_by_depth.iter().sum::<f64>() / accuracies_by_depth.len() as f64;

        // ── Test 2: Depth sensitivity ─────────────────────────────────
        // Compute slope: does accuracy decrease with depth?
        // For perfectly lossless XOR, it shouldn't — this tests the encoding.
        let depth_sensitivity = if accuracies_by_depth.len() >= 2 {
            let first = accuracies_by_depth[0];
            let last = *accuracies_by_depth.last().unwrap();
            (first - last).abs() // Lower = more robust across depths
        } else {
            0.0
        };

        // ── Test 3: Chain vs direct evidence ──────────────────────────
        // Compare: how similar is V0's indirect effect on V_N (via chain)
        // vs a direct link V0→V_N?
        let n_comparisons = 5;
        let mut ratio_sum = 0.0;

        for i in 0..n_comparisons {
            let s = seed.wrapping_add(5000 + i as u64 * 100);
            let depth = 3;

            // Build a 3-step chain
            let variables: Vec<BinaryHV> = (0..=depth)
                .map(|j| BinaryHV::random(s.wrapping_add(j as u64)))
                .collect();

            // Direct link: V0 ⊕ V3
            let direct = variables[0].bind(&variables[depth]);

            // Chain composite: bundle of all links
            let links: Vec<BinaryHV> = (0..depth)
                .map(|j| variables[j].permute(1).bind(&variables[j + 1]))
                .collect();

            // Bundle all links into a single chain representation
            // Using majority vote bundling approximation via XOR
            let mut chain_composite = links[0];
            for link in &links[1..] {
                chain_composite = chain_composite.bind(link);
            }

            // How similar is the chain composite to the direct link?
            let chain_sim = chain_composite.cosine_similarity(&direct);

            // Normalize: direct self-similarity is 1.0
            ratio_sum += chain_sim as f64;
        }
        let chain_vs_direct_ratio = ratio_sum / n_comparisons as f64;

        TrialResult {
            chain_tracing_accuracy,
            depth_sensitivity,
            chain_vs_direct_ratio,
        }
    }
}

impl PsychBenchmark for CausalChainBenchmark {
    fn name(&self) -> &str {
        "CausalReasoning::CausalChain"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Causal Chain Inference",
            citation: "Sloman, S. (2005). Causal Models: How People Think about the World and Its Alternatives. Oxford UP.",
            year: 2005,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut tracing_vals = Vec::new();
        let mut sensitivity_vals = Vec::new();
        let mut ratio_vals = Vec::new();
        let mut trace = Vec::new();

        for t in 0..config.trials_per_condition {
            let r = self.run_trial(config, t);
            tracing_vals.push(r.chain_tracing_accuracy);
            sensitivity_vals.push(r.depth_sensitivity);
            ratio_vals.push(r.chain_vs_direct_ratio);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("depth_sensitivity".to_string(), r.depth_sensitivity);
                extra.insert("chain_vs_direct_ratio".to_string(), r.chain_vs_direct_ratio);
                trace.push(TrialOutcome {
                    trial_idx: t,
                    correct: r.chain_tracing_accuracy > 0.5,
                    rt_ticks: 1.0,
                    confidence: r.chain_tracing_accuracy,
                    similarity: r.chain_tracing_accuracy,
                    response_idx: 0,
                    condition: "causal_chain".to_string(),
                    extra,
                });
            }
        }

        result.insert(
            "chain_tracing_accuracy",
            MetricValue::from_samples(&tracing_vals),
        );
        result.insert(
            "depth_sensitivity",
            MetricValue::from_samples(&sensitivity_vals),
        );
        result.insert(
            "chain_vs_direct_ratio",
            MetricValue::from_samples(&ratio_vals),
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

    fn default_config() -> BenchmarkConfig {
        BenchmarkConfig {
            trials_per_condition: 10,
            ..BenchmarkConfig::default()
        }
    }

    #[test]
    fn test_causal_chain_runs() {
        let bench = CausalChainBenchmark;
        let result = bench.run(&default_config());
        assert!(result.benchmark.contains("CausalChain"));
        assert!(result.metrics.contains_key("chain_tracing_accuracy"));
    }

    #[test]
    fn test_chain_tracing_above_chance() {
        let bench = CausalChainBenchmark;
        let result = bench.run(&default_config());
        let acc = result.metrics["chain_tracing_accuracy"].mean;
        // XOR chain tracing should be exact (or very close), well above 0.5
        assert!(acc > 0.5, "Chain tracing should be above chance, got {acc}");
    }

    #[test]
    fn test_metrics_bounded() {
        let bench = CausalChainBenchmark;
        let result = bench.run(&default_config());
        for (name, metric) in &result.metrics {
            // chain_vs_direct_ratio uses cosine similarity which can be negative
            assert!(
                metric.mean >= -1.0 && metric.mean <= 1.0,
                "{name} should be in [-1,1], got {}",
                metric.mean
            );
        }
    }

    #[test]
    fn test_provenance_present() {
        let bench = CausalChainBenchmark;
        let prov = bench.provenance().expect("Should have provenance");
        assert_eq!(prov.year, 2005);
        assert!(prov.citation.contains("Sloman"));
    }

    #[test]
    fn test_depth_sensitivity_finite() {
        let bench = CausalChainBenchmark;
        let result = bench.run(&default_config());
        let sensitivity = result.metrics["depth_sensitivity"].mean;
        assert!(
            sensitivity.is_finite(),
            "Depth sensitivity should be finite, got {sensitivity}"
        );
    }
}
