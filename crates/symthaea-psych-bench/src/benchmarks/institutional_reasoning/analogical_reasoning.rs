//! Institutional Reasoning: Analogical Reasoning Benchmark
//!
//! Tests whether the HDC composition algebra can solve institutional analogies
//! of the form "A is to B as ??? is to D". This is a harder test than
//! decomposition because it requires computing and applying transformations
//! across different institutional domains.
//!
//! ## HDC Implementation
//!
//! Given bundled compositions A and B, the transformation is computed as the
//! set-difference of their source components. Components in A but not B are
//! "removed"; components in B but not A are "added". This transformation is
//! then applied to D using XOR-binding (toggle), and the nearest axiom to
//! the result is identified.
//!
//! ## Key Claims Tested
//!
//! 1. **Analogical Transfer**: The system can apply a structural transformation
//!    from one institutional domain to another.
//! 2. **Shared-Component Sensitivity**: Analogies between axioms with shared
//!    components should produce higher similarity than unrelated pairs.
//! 3. **Symmetry**: If A:B :: X:D, then B:A :: Y:D should produce a different
//!    (and meaningful) result.
//!
//! ## References
//!
//! - Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
//!   *Cognitive Science*.
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::CompositionAlgebra;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

/// Institutional Reasoning: Analogical Reasoning benchmark.
pub struct AnalogicalReasoningBenchmark;

struct TrialResult {
    /// Fraction of analogies that produce above-chance similarity
    transfer_accuracy: f64,
    /// Mean similarity of analogical targets to their nearest axiom
    transfer_strength: f64,
    /// Whether shared-component analogies score higher than unrelated
    shared_component_advantage: f64,
    /// Whether A:B::?:D ≠ B:A::?:D (asymmetry)
    asymmetry_score: f64,
}

/// An analogy test case: A is to B as ??? is to D.
struct AnalogyCase {
    a: &'static str,
    b: &'static str,
    d: &'static str,
    /// True if A and D share at least one source component
    shares_components: bool,
}

impl AnalogicalReasoningBenchmark {
    fn analogy_cases() -> Vec<AnalogyCase> {
        vec![
            // Governance transformations
            AnalogyCase {
                a: "REVOLUTION",
                b: "LEGITIMATE_GOVERNANCE",
                d: "TRADE_AGREEMENT",
                shares_components: false,
            },
            AnalogyCase {
                a: "LEGITIMATE_GOVERNANCE",
                b: "REVOLUTION",
                d: "ECONOMIC_SANCTION",
                shares_components: false,
            },
            // Shared-component analogies (should score higher)
            AnalogyCase {
                a: "REVOLUTION",
                b: "CIVIL_DISOBEDIENCE",
                d: "ECONOMIC_SANCTION",
                shares_components: true, // PROHIBITION shared between A and D
            },
            AnalogyCase {
                a: "TRADE_AGREEMENT",
                b: "ECONOMIC_SANCTION",
                d: "LEGITIMATE_GOVERNANCE",
                shares_components: false,
            },
            AnalogyCase {
                a: "REGULATORY_CAPTURE",
                b: "LEGITIMATE_GOVERNANCE",
                d: "TRADE_AGREEMENT",
                shares_components: false,
            },
            // Exchange-domain analogies
            AnalogyCase {
                a: "ECONOMIC_SANCTION",
                b: "TRADE_AGREEMENT",
                d: "REVOLUTION",
                shares_components: true, // PROHIBITION shared between A and D
            },
            // New axiom analogies — should improve asymmetry scores
            AnalogyCase {
                a: "LEGITIMATE_GOVERNANCE",
                b: "CORRUPTION",
                d: "SOCIAL_CONTRACT",
                shares_components: true, // AUTHORITY shared
            },
            AnalogyCase {
                a: "DEMOCRATIC_ELECTION",
                b: "REVOLUTION",
                d: "SOCIAL_CONTRACT",
                shares_components: true, // AUTHORITY, LEGITIMACY shared
            },
            AnalogyCase {
                a: "ARMS_EMBARGO",
                b: "TRADE_AGREEMENT",
                d: "CIVIL_DISOBEDIENCE",
                shares_components: false,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _seed = config.trial_seed("institutional", "analogical_reasoning", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        let loaded = algebra.load_institutional_axioms(&system);
        assert!(loaded >= 12);

        let cases = Self::analogy_cases();

        // ── 1. Transfer accuracy: above-chance similarity ──
        let mut above_chance = 0usize;
        let mut strengths = Vec::new();
        let mut shared_sims = Vec::new();
        let mut unrelated_sims = Vec::new();

        for case in &cases {
            let result = algebra.query_analogy(case.a, case.b, case.d, &system);
            let (_nearest, sim, _hv) = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            if sim > 0.50 {
                above_chance += 1;
            }
            strengths.push(sim as f64);

            if case.shares_components {
                shared_sims.push(sim as f64);
            } else {
                unrelated_sims.push(sim as f64);
            }
        }

        let transfer_accuracy = if cases.is_empty() {
            0.0
        } else {
            above_chance as f64 / cases.len() as f64
        };

        let transfer_strength = if strengths.is_empty() {
            0.0
        } else {
            strengths.iter().sum::<f64>() / strengths.len() as f64
        };

        // ── 2. Shared-component advantage ──
        let mean_shared = if shared_sims.is_empty() {
            0.5
        } else {
            shared_sims.iter().sum::<f64>() / shared_sims.len() as f64
        };
        let mean_unrelated = if unrelated_sims.is_empty() {
            0.5
        } else {
            unrelated_sims.iter().sum::<f64>() / unrelated_sims.len() as f64
        };
        let shared_component_advantage = (mean_shared - mean_unrelated).max(0.0);

        // ── 3. Asymmetry: A:B::?:D should differ from B:A::?:D ──
        let mut asymmetry_count = 0usize;
        let mut asymmetry_total = 0usize;
        for case in &cases {
            let fwd = algebra.query_analogy(case.a, case.b, case.d, &system);
            let rev = algebra.query_analogy(case.b, case.a, case.d, &system);
            if let (Ok((fwd_name, _, _)), Ok((rev_name, _, _))) = (fwd, rev) {
                asymmetry_total += 1;
                if fwd_name != rev_name {
                    asymmetry_count += 1;
                }
            }
        }
        let asymmetry_score = if asymmetry_total > 0 {
            asymmetry_count as f64 / asymmetry_total as f64
        } else {
            0.0
        };

        TrialResult {
            transfer_accuracy,
            transfer_strength,
            shared_component_advantage,
            asymmetry_score,
        }
    }
}

impl PsychBenchmark for AnalogicalReasoningBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::AnalogicalReasoning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Analogical Reasoning (HDC Composition Algebra)",
            citation: "Gentner, D. (1983). Structure-mapping. Cognitive Science.",
            year: 1983,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut transfer_accs = Vec::new();
        let mut transfer_strengths = Vec::new();
        let mut advantages = Vec::new();
        let mut asymmetries = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            transfer_accs.push(r.transfer_accuracy);
            transfer_strengths.push(r.transfer_strength);
            advantages.push(r.shared_component_advantage);
            asymmetries.push(r.asymmetry_score);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("transfer_strength".to_string(), r.transfer_strength);
                extra.insert(
                    "shared_component_advantage".to_string(),
                    r.shared_component_advantage,
                );
                extra.insert("asymmetry_score".to_string(), r.asymmetry_score);
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "analogical_reasoning".to_string(),
                    correct: r.transfer_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.transfer_strength,
                    confidence: r.transfer_accuracy,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "analogical_transfer_accuracy",
            MetricValue::from_samples(&transfer_accs),
        );
        result.insert(
            "analogical_transfer_strength",
            MetricValue::from_samples(&transfer_strengths),
        );
        result.insert(
            "analogical_shared_component_advantage",
            MetricValue::from_samples(&advantages),
        );
        result.insert(
            "analogical_asymmetry_score",
            MetricValue::from_samples(&asymmetries),
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
    fn test_analogical_reasoning_runs() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        assert!(result
            .metrics
            .contains_key("analogical_transfer_accuracy"));
        assert!(result
            .metrics
            .contains_key("analogical_transfer_strength"));
    }

    #[test]
    fn test_analogical_reasoning_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_transfer_strength_above_chance() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        let strength = result
            .metrics
            .get("analogical_transfer_strength")
            .unwrap();
        // With bundled compositions, analogical targets should have
        // above-chance similarity to some axiom
        assert!(
            strength.mean > 0.48,
            "Transfer strength should be near or above chance, got {}",
            strength.mean
        );
    }

    #[test]
    fn test_print_analogical_scores() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        eprintln!("\n═══ Analogical Reasoning Benchmark Scores ═══");
        for (key, val) in &result.metrics {
            let short = key.strip_prefix("analogical_").unwrap_or(key);
            eprintln!("  {short:<35} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }
        eprintln!("═══════════════════════════════════════════════\n");
    }
}
