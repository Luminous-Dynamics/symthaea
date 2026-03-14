//! Institutional Reasoning: Causal Decomposition Benchmark
//!
//! Tests whether the HDC composition algebra can correctly answer counterfactual
//! institutional questions using bundled compositions and `query_decomposition`.
//!
//! ## Key Claims Tested
//!
//! 1. **Causal Decomposition Accuracy**: Removing a component from a bundled
//!    composite via re-bundling should produce a residual closer to the
//!    semantically expected axiom than to an unrelated one.
//!
//! 2. **Axiom Discrimination**: Each institutional axiom should occupy a distinct
//!    region of the 16,384D hypervector space.
//!
//! 3. **Component Similarity**: Bundled composites should remain similar to their
//!    source components (unlike XOR binding which produces orthogonal composites).
//!
//! 4. **Cross-Domain Coherence**: Composites that share components should be
//!    more similar than composites that share no components.
//!
//! ## References
//!
//! - Putnam, H. (1967). Psychological predicates. In *Art, Mind, and Religion*.
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.
//! - Ostrom, E. (1990). *Governing the Commons*. Cambridge UP.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::CompositionAlgebra;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

/// Institutional Reasoning: Causal Decomposition benchmark.
pub struct InstitutionalReasoningBenchmark;

struct TrialResult {
    decomposition_accuracy: f64,
    axiom_discrimination: f64,
    component_similarity: f64,
    cross_domain_coherence: f64,
}

/// A decomposition test case: remove `removed` from `composite`, expect the
/// residual to be closer to `expected_near` than to `expected_far`.
struct DecompositionCase {
    composite: &'static str,
    removed: &'static str,
    expected_near: &'static str,
    expected_far: &'static str,
}

/// All 12 institutional axiom names loaded by `load_institutional_axioms`.
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
];

impl InstitutionalReasoningBenchmark {
    fn decomposition_cases() -> Vec<DecompositionCase> {
        vec![
            DecompositionCase {
                composite: "LEGITIMATE_GOVERNANCE",
                removed: "TRUST",
                expected_near: "REVOLUTION",
                expected_far: "TRADE_AGREEMENT",
            },
            DecompositionCase {
                composite: "LEGITIMATE_GOVERNANCE",
                removed: "LEGITIMACY",
                expected_near: "REVOLUTION",
                expected_far: "TRADE_AGREEMENT",
            },
            DecompositionCase {
                composite: "TRADE_AGREEMENT",
                removed: "RECIPROCATE",
                expected_near: "ECONOMIC_SANCTION",
                expected_far: "CIVIL_DISOBEDIENCE",
            },
            DecompositionCase {
                composite: "ECONOMIC_SANCTION",
                removed: "PROHIBITION",
                expected_near: "TRADE_AGREEMENT",
                expected_far: "FAILED_STATE",
            },
            DecompositionCase {
                composite: "DEMOCRATIC_ELECTION",
                removed: "COOPERATE",
                expected_near: "LEGITIMATE_GOVERNANCE",
                expected_far: "FAILED_STATE",
            },
            DecompositionCase {
                composite: "SOCIAL_CONTRACT",
                removed: "COOPERATE",
                expected_near: "BORDER_DISPUTE",
                expected_far: "TRADE_AGREEMENT",
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("institutional", "causal_decomposition", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        let loaded = algebra.load_institutional_axioms(&system);
        assert!(
            loaded >= 12,
            "expected at least 12 institutional axioms, got {loaded}"
        );

        // ── 1. Decomposition accuracy (using query_decomposition) ──
        let cases = Self::decomposition_cases();
        let mut correct = 0usize;
        let total = cases.len();

        for case in &cases {
            let near_hv = match algebra.get_encoding(case.expected_near, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let far_hv = match algebra.get_encoding(case.expected_far, &system) {
                Some(hv) => hv,
                None => continue,
            };

            let residual = match algebra.query_decomposition(case.composite, case.removed, &system)
            {
                Ok((_name, _sim, hv)) => hv,
                Err(_) => continue,
            };

            let sim_near = residual.similarity(&near_hv);
            let sim_far = residual.similarity(&far_hv);

            if sim_near > sim_far {
                correct += 1;
            }
        }

        let decomposition_accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        // ── 2. Axiom discrimination ──
        let mut discrimination_scores = Vec::new();
        for &name_a in AXIOM_NAMES {
            let hv_a = match algebra.get_encoding(name_a, &system) {
                Some(hv) => hv,
                None => continue,
            };

            let mut max_cross = f32::NEG_INFINITY;
            for &name_b in AXIOM_NAMES {
                if name_a == name_b {
                    continue;
                }
                let hv_b = match algebra.get_encoding(name_b, &system) {
                    Some(hv) => hv,
                    None => continue,
                };
                let sim = hv_a.similarity(&hv_b);
                if sim > max_cross {
                    max_cross = sim;
                }
            }

            if max_cross.is_finite() {
                discrimination_scores.push((1.0 - max_cross as f64).max(0.0));
            }
        }

        let axiom_discrimination = if discrimination_scores.is_empty() {
            0.0
        } else {
            discrimination_scores.iter().sum::<f64>() / discrimination_scores.len() as f64
        };

        // ── 3. Component similarity ──
        // With bundling, composites should be similar to their source components.
        let mut comp_sims = Vec::new();
        for &name in AXIOM_NAMES {
            let comp = match algebra.get(name) {
                Some(c) => c,
                None => continue,
            };
            let comp_hv = comp.encoding;
            for source in &comp.sources {
                if let Some(src_hv) = algebra.get_encoding(source, &system) {
                    comp_sims.push(src_hv.similarity(&comp_hv) as f64);
                }
            }
        }

        let component_similarity = if comp_sims.is_empty() {
            0.0
        } else {
            comp_sims.iter().sum::<f64>() / comp_sims.len() as f64
        };

        // ── 4. Cross-domain coherence ──
        let _ = seed;

        let shared_pairs: &[(&str, &str)] = &[
            ("REVOLUTION", "LEGITIMATE_GOVERNANCE"),    // share AUTHORITY
            ("TRADE_AGREEMENT", "ECONOMIC_SANCTION"),   // share EXCHANGE
            ("REVOLUTION", "CIVIL_DISOBEDIENCE"),       // share PROHIBITION
            ("REVOLUTION", "ARMS_EMBARGO"),             // share ENFORCEMENT, PROHIBITION
            ("DEMOCRATIC_ELECTION", "SOCIAL_CONTRACT"), // share LEGITIMACY, COOPERATE
            ("CORRUPTION", "REGULATORY_CAPTURE"),       // share DEFECT
        ];

        let mut shared_sims = Vec::new();
        for &(a, b) in shared_pairs {
            let hv_a = match algebra.get_encoding(a, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let hv_b = match algebra.get_encoding(b, &system) {
                Some(hv) => hv,
                None => continue,
            };
            shared_sims.push(hv_a.similarity(&hv_b) as f64);
        }

        let mut all_sims = Vec::new();
        for (i, &name_a) in AXIOM_NAMES.iter().enumerate() {
            for &name_b in &AXIOM_NAMES[i + 1..] {
                let hv_a = match algebra.get_encoding(name_a, &system) {
                    Some(hv) => hv,
                    None => continue,
                };
                let hv_b = match algebra.get_encoding(name_b, &system) {
                    Some(hv) => hv,
                    None => continue,
                };
                all_sims.push(hv_a.similarity(&hv_b) as f64);
            }
        }

        let mean_shared = if shared_sims.is_empty() {
            0.0
        } else {
            shared_sims.iter().sum::<f64>() / shared_sims.len() as f64
        };
        let mean_all = if all_sims.is_empty() {
            0.0
        } else {
            all_sims.iter().sum::<f64>() / all_sims.len() as f64
        };

        let cross_domain_coherence = if (1.0 - mean_all).abs() > 1e-10 {
            ((mean_shared - mean_all) / (1.0 - mean_all)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        TrialResult {
            decomposition_accuracy,
            axiom_discrimination,
            component_similarity,
            cross_domain_coherence,
        }
    }
}

impl PsychBenchmark for InstitutionalReasoningBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::CausalDecomposition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Causal Decomposition (HDC Composition Algebra)",
            citation: "Putnam (1967); Kanerva (2009)",
            year: 2009,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut decomposition_accs = Vec::new();
        let mut axiom_discs = Vec::new();
        let mut comp_sims = Vec::new();
        let mut coherences = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            decomposition_accs.push(r.decomposition_accuracy);
            axiom_discs.push(r.axiom_discrimination);
            comp_sims.push(r.component_similarity);
            coherences.push(r.cross_domain_coherence);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("axiom_discrimination".to_string(), r.axiom_discrimination);
                extra.insert("component_similarity".to_string(), r.component_similarity);
                extra.insert(
                    "cross_domain_coherence".to_string(),
                    r.cross_domain_coherence,
                );
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "causal_decomposition".to_string(),
                    correct: r.decomposition_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.decomposition_accuracy,
                    confidence: r.component_similarity,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "institutional_decomposition_accuracy",
            MetricValue::from_samples(&decomposition_accs),
        );
        result.insert(
            "institutional_axiom_discrimination",
            MetricValue::from_samples(&axiom_discs),
        );
        result.insert(
            "institutional_recovery_fidelity",
            MetricValue::from_samples(&comp_sims),
        );
        result.insert(
            "institutional_cross_domain_coherence",
            MetricValue::from_samples(&coherences),
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
    fn test_institutional_reasoning_runs() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        assert!(result
            .metrics
            .contains_key("institutional_decomposition_accuracy"));
        assert!(result
            .metrics
            .contains_key("institutional_axiom_discrimination"));
        assert!(result
            .metrics
            .contains_key("institutional_recovery_fidelity"));
        assert!(result
            .metrics
            .contains_key("institutional_cross_domain_coherence"));
    }

    #[test]
    fn test_institutional_reasoning_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_component_similarity_high() {
        // Bundled composites should be similar to their source components
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        let comp_sim = result
            .metrics
            .get("institutional_recovery_fidelity")
            .unwrap();
        assert!(
            comp_sim.mean > 0.55,
            "Component similarity should be > 0.55, got {}",
            comp_sim.mean
        );
    }

    #[test]
    fn test_axiom_discrimination_positive() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        let disc = result
            .metrics
            .get("institutional_axiom_discrimination")
            .unwrap();
        assert!(
            disc.mean > 0.0,
            "axiom discrimination should be positive, got {}",
            disc.mean
        );
    }
}
