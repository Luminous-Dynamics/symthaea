// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Isomorphism Benchmark
//!
//! Tests whether the HDC composition algebra can detect structural equivalence
//! between institutions. Two institutions are "isomorphic" if they share the
//! same primitive components (regardless of surface labeling) — this is the
//! HDC analog of recognizing that different organizations with the same power
//! structures are functionally identical.
//!
//! ## Key Claims Tested
//!
//! 1. **Self-isomorphism**: Every axiom is maximally similar to itself (identity).
//! 2. **Overlap sensitivity**: Axioms sharing more primitives show higher similarity
//!    than axioms sharing fewer.
//! 3. **Isomorphism discrimination**: The system can rank pairs by structural
//!    overlap, producing a positive correlation between shared-component count
//!    and similarity.
//!
//! ## References
//!
//! - DiMaggio & Powell (1983). The iron cage revisited: Institutional isomorphism.
//!   *American Sociological Review*.
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem};

/// Institutional Reasoning: Isomorphism benchmark.
pub struct InstitutionalIsomorphismBenchmark;

struct TrialResult {
    /// Mean self-similarity across all axioms (should be ~1.0)
    self_similarity: f64,
    /// Gamma correlation between shared-component count and encoding similarity
    overlap_correlation: f64,
    /// Mean similarity of high-overlap pairs (>=2 shared) vs low-overlap (0 shared)
    discrimination_gap: f64,
    /// Fraction of pair comparisons where higher overlap => higher similarity
    monotonicity: f64,
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

impl InstitutionalIsomorphismBenchmark {
    fn run_trial(&self, _config: &BenchmarkConfig, _trial_idx: usize) -> TrialResult {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();
        let loaded = algebra.load_institutional_axioms(&system);
        assert!(loaded >= 18);

        // 1. Self-similarity: each axiom vs itself
        let mut self_sims = Vec::new();
        for &name in AXIOM_NAMES {
            if let Some(comp) = algebra.compositions().get(name) {
                let sim = comp.encoding.similarity(&comp.encoding);
                self_sims.push(sim as f64);
            }
        }
        let self_similarity = if self_sims.is_empty() {
            0.0
        } else {
            self_sims.iter().sum::<f64>() / self_sims.len() as f64
        };

        // 2. Pairwise: compute similarity and shared-component count for all pairs
        let mut overlaps = Vec::new(); // (shared_count, similarity)
        for (i, &name_a) in AXIOM_NAMES.iter().enumerate() {
            let comp_a = match algebra.compositions().get(name_a) {
                Some(c) => c,
                None => continue,
            };
            for &name_b in AXIOM_NAMES.iter().skip(i + 1) {
                let comp_b = match algebra.compositions().get(name_b) {
                    Some(c) => c,
                    None => continue,
                };
                let sim = comp_a.encoding.similarity(&comp_b.encoding) as f64;
                let shared = comp_a
                    .sources
                    .iter()
                    .filter(|s| comp_b.sources.contains(s))
                    .count();
                overlaps.push((shared, sim));
            }
        }

        // 3. Overlap correlation (Goodman-Kruskal gamma)
        let mut concordant = 0u64;
        let mut discordant = 0u64;
        for i in 0..overlaps.len() {
            for j in (i + 1)..overlaps.len() {
                let overlap_diff = overlaps[i].0 as f64 - overlaps[j].0 as f64;
                let sim_diff = overlaps[i].1 - overlaps[j].1;
                let product = overlap_diff * sim_diff;
                if product > 0.0 {
                    concordant += 1;
                } else if product < 0.0 {
                    discordant += 1;
                }
            }
        }
        let overlap_correlation = if concordant + discordant > 0 {
            (concordant as f64 - discordant as f64) / (concordant as f64 + discordant as f64)
        } else {
            0.0
        };

        // 4. Discrimination gap: high-overlap vs zero-overlap mean similarity
        let high_overlap: Vec<f64> = overlaps
            .iter()
            .filter(|(s, _)| *s >= 2)
            .map(|(_, sim)| *sim)
            .collect();
        let zero_overlap: Vec<f64> = overlaps
            .iter()
            .filter(|(s, _)| *s == 0)
            .map(|(_, sim)| *sim)
            .collect();
        let mean_high = if high_overlap.is_empty() {
            0.0
        } else {
            high_overlap.iter().sum::<f64>() / high_overlap.len() as f64
        };
        let mean_zero = if zero_overlap.is_empty() {
            0.0
        } else {
            zero_overlap.iter().sum::<f64>() / zero_overlap.len() as f64
        };
        let discrimination_gap = (mean_high - mean_zero).max(0.0);

        // 5. Monotonicity: fraction of pairs where higher overlap => higher sim
        let mut mono_correct = 0u32;
        let mut mono_total = 0u32;
        for i in 0..overlaps.len() {
            for j in (i + 1)..overlaps.len() {
                if overlaps[i].0 != overlaps[j].0 {
                    mono_total += 1;
                    let overlap_diff = overlaps[i].0 as f64 - overlaps[j].0 as f64;
                    let sim_diff = overlaps[i].1 - overlaps[j].1;
                    if overlap_diff * sim_diff > 0.0 {
                        mono_correct += 1;
                    }
                }
            }
        }
        let monotonicity = if mono_total > 0 {
            mono_correct as f64 / mono_total as f64
        } else {
            0.0
        };

        TrialResult {
            self_similarity,
            overlap_correlation,
            discrimination_gap,
            monotonicity,
        }
    }
}

impl PsychBenchmark for InstitutionalIsomorphismBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::Isomorphism"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Isomorphism (HDC Composition Algebra)",
            citation: "DiMaggio & Powell (1983). The iron cage revisited. ASR.",
            year: 1983,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut self_sims = Vec::new();
        let mut correlations = Vec::new();
        let mut gaps = Vec::new();
        let mut monotonicities = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            self_sims.push(r.self_similarity);
            correlations.push(r.overlap_correlation);
            gaps.push(r.discrimination_gap);
            monotonicities.push(r.monotonicity);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("overlap_correlation".to_string(), r.overlap_correlation);
                extra.insert("discrimination_gap".to_string(), r.discrimination_gap);
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "isomorphism".to_string(),
                    correct: r.overlap_correlation > 0.0,
                    rt_ticks: 0.0,
                    similarity: r.self_similarity,
                    confidence: r.monotonicity,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "isomorphism_self_similarity",
            MetricValue::from_samples(&self_sims),
        );
        result.insert(
            "isomorphism_overlap_correlation",
            MetricValue::from_samples(&correlations),
        );
        result.insert(
            "isomorphism_discrimination_gap",
            MetricValue::from_samples(&gaps),
        );
        result.insert(
            "isomorphism_monotonicity",
            MetricValue::from_samples(&monotonicities),
        );

        result.conditions = 4;
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
    fn test_isomorphism_runs() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalIsomorphismBenchmark.run(&config);
        assert!(result.metrics.contains_key("isomorphism_self_similarity"));
        assert!(
            result
                .metrics
                .contains_key("isomorphism_overlap_correlation")
        );
        assert!(
            result
                .metrics
                .contains_key("isomorphism_discrimination_gap")
        );
        assert!(result.metrics.contains_key("isomorphism_monotonicity"));
    }

    #[test]
    fn test_isomorphism_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalIsomorphismBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_self_similarity_near_one() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalIsomorphismBenchmark.run(&config);
        let ss = result.metrics["isomorphism_self_similarity"].mean;
        assert!(ss > 0.99, "self-similarity should be ~1.0, got {:.4}", ss);
    }

    #[test]
    fn test_overlap_correlation_positive() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalIsomorphismBenchmark.run(&config);
        let corr = result.metrics["isomorphism_overlap_correlation"].mean;
        assert!(
            corr > 0.0,
            "overlap correlation should be positive, got {:.4}",
            corr
        );
    }

    #[test]
    fn test_print_isomorphism_scores() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalIsomorphismBenchmark.run(&config);
        eprintln!("\n═══ Institutional Isomorphism Benchmark ═══");
        for (key, val) in &result.metrics {
            let short = key.strip_prefix("isomorphism_").unwrap_or(key);
            eprintln!("  {short:<35} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }
        eprintln!("═══════════════════════════════════════════\n");
    }
}
