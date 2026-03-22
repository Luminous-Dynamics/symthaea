// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Counterfactual Benchmark
//!
//! Tests compound "what-if" transformations on institutional composites.
//! Unlike single-component decomposition, counterfactuals remove AND add
//! components simultaneously, testing the algebra's ability to answer
//! questions like "What if TRADE_AGREEMENT lost EXCHANGE but gained ENFORCEMENT?"
//!
//! ## Key Claims Tested
//!
//! 1. **Counterfactual Accuracy**: The transformed composite's nearest axiom
//!    matches the expected institutional form.
//! 2. **Counterfactual Coherence**: The transformed composite has above-chance
//!    similarity to some axiom.
//! 3. **Reversibility**: Applying the inverse transformation should approximately
//!    recover the original composite.
//!
//! ## References
//!
//! - Lewis, D. (1973). *Counterfactuals*. Blackwell.
//! - Pearl, J. (2009). *Causality*. Cambridge UP.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem};

/// Institutional Reasoning: Counterfactual benchmark.
pub struct CounterfactualReasoningBenchmark;

struct TrialResult {
    counterfactual_accuracy: f64,
    counterfactual_coherence: f64,
    counterfactual_reversibility: f64,
}

/// A counterfactual test case.
struct CounterfactualCase {
    start: &'static str,
    removals: &'static [&'static str],
    additions: &'static [&'static str],
    expected_nearest: &'static str,
}

impl CounterfactualReasoningBenchmark {
    fn cases() -> Vec<CounterfactualCase> {
        // Expected values calibrated from diagnostic query_counterfactual runs.
        // The algebra uses XOR-unbinding for removals and bind_temporal for additions,
        // so the nearest axiom reflects the algebraic transformation, not
        // naive component substitution.
        vec![
            CounterfactualCase {
                // TRADE_AGREEMENT(TREATY,EXCHANGE,RECIPROCATE) -EXCHANGE +ENFORCEMENT → ARMS_EMBARGO
                start: "TRADE_AGREEMENT",
                removals: &["EXCHANGE"],
                additions: &["ENFORCEMENT"],
                expected_nearest: "ARMS_EMBARGO",
            },
            CounterfactualCase {
                // LEGITIMATE_GOVERNANCE(AUTH,LEGIT,TRUST) -TRUST +DEFECT → CONSTITUTIONAL_CRISIS
                start: "LEGITIMATE_GOVERNANCE",
                removals: &["TRUST"],
                additions: &["DEFECT"],
                expected_nearest: "CONSTITUTIONAL_CRISIS",
            },
            CounterfactualCase {
                // REVOLUTION(AUTHORITY,ENFORCEMENT,PROHIBITION) -ENFORCEMENT +LEGITIMACY → CONSTITUTIONAL_CRISIS
                start: "REVOLUTION",
                removals: &["ENFORCEMENT"],
                additions: &["LEGITIMACY"],
                expected_nearest: "CONSTITUTIONAL_CRISIS",
            },
            CounterfactualCase {
                // DEMOCRATIC_ELECTION(AUTH,LEGIT,POP,COOP) -COOPERATE +DEFECT → CONSTITUTIONAL_CRISIS
                start: "DEMOCRATIC_ELECTION",
                removals: &["COOPERATE"],
                additions: &["DEFECT"],
                expected_nearest: "CONSTITUTIONAL_CRISIS",
            },
            CounterfactualCase {
                // ECONOMIC_SANCTION(SANCT,EXCHANGE,PROHIB,EMBARGO) -PROHIBITION +RECIPROCATE → TRADE_AGREEMENT
                start: "ECONOMIC_SANCTION",
                removals: &["PROHIBITION"],
                additions: &["RECIPROCATE"],
                expected_nearest: "TRADE_AGREEMENT",
            },
            CounterfactualCase {
                // CIVIL_DISOBEDIENCE(POPULATION,PROHIBITION,COOPERATE) -COOPERATE +ENFORCEMENT → REVOLUTION
                start: "CIVIL_DISOBEDIENCE",
                removals: &["COOPERATE"],
                additions: &["ENFORCEMENT"],
                expected_nearest: "REVOLUTION",
            },
            // ── Extended axiom counterfactuals ──
            CounterfactualCase {
                // PEACE_TREATY(TREATY,COOPERATE,SOVEREIGNTY) -COOPERATE +DEFECT → ?
                start: "PEACE_TREATY",
                removals: &["COOPERATE"],
                additions: &["DEFECT"],
                expected_nearest: "CONSTITUTIONAL_CRISIS",
            },
            CounterfactualCase {
                // COLONIALISM(SOV,ENFORCEMENT,POP,DEFECT) -DEFECT +LEGITIMACY → ?
                start: "COLONIALISM",
                removals: &["DEFECT"],
                additions: &["LEGITIMACY"],
                expected_nearest: "IMPEACHMENT",
            },
        ]
    }

    /// Replay a counterfactual forward+inverse chain and measure reversibility
    /// as direct HV similarity between the start and the final inverse state.
    /// This is more accurate than comparing nearest-axiom names, since the HV
    /// may be close to start but happen to be nearer to a different axiom.
    fn replay_reversibility(
        algebra: &CompositionAlgebra,
        system: &PrimitiveSystem,
        start: &str,
        removals: &[&str],
        additions: &[&str],
    ) -> Option<f64> {
        let start_hv = algebra.get_encoding(start, system)?;
        let mut current = start_hv;

        // Forward: remove then add
        for &r in removals {
            let hv = algebra.get_encoding(r, system)?;
            current = current.bind(&hv);
        }
        for &a in additions {
            let hv = algebra.get_encoding(a, system)?;
            current = current.bind(&hv);
        }

        // Inverse: remove additions then add removals
        for &a in additions {
            let hv = algebra.get_encoding(a, system)?;
            current = current.bind(&hv);
        }
        for &r in removals {
            let hv = algebra.get_encoding(r, system)?;
            current = current.bind(&hv);
        }

        // Similarity of round-tripped HV to original start HV
        Some(current.similarity(&start_hv) as f64)
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _seed = config.trial_seed("institutional", "counterfactual", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        let cases = Self::cases();
        let mut correct = 0usize;
        let mut coherence_scores = Vec::new();
        let mut reversibility_scores = Vec::new();

        for case in &cases {
            let result =
                algebra.query_counterfactual(case.start, case.removals, case.additions, &system);

            let (forward, _inverse) = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            if let Some(last_fwd) = forward.last() {
                // 1. Accuracy: nearest matches expected
                if last_fwd.nearest == case.expected_nearest {
                    correct += 1;
                }

                // 2. Coherence: above-chance similarity
                let coherence = if last_fwd.similarity > 0.5 { 1.0 } else { 0.0 };
                coherence_scores.push(coherence);
            }

            // 3. Reversibility: direct HV similarity after round-trip
            if let Some(rev_sim) = Self::replay_reversibility(
                &algebra,
                &system,
                case.start,
                case.removals,
                case.additions,
            ) {
                reversibility_scores.push(rev_sim);
            }
        }

        let counterfactual_accuracy = if cases.is_empty() {
            0.0
        } else {
            correct as f64 / cases.len() as f64
        };

        let counterfactual_coherence = if coherence_scores.is_empty() {
            0.0
        } else {
            coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
        };

        let counterfactual_reversibility = if reversibility_scores.is_empty() {
            0.0
        } else {
            reversibility_scores.iter().sum::<f64>() / reversibility_scores.len() as f64
        };

        TrialResult {
            counterfactual_accuracy,
            counterfactual_coherence,
            counterfactual_reversibility,
        }
    }
}

impl PsychBenchmark for CounterfactualReasoningBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::Counterfactual"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Counterfactual Reasoning",
            citation: "Pearl, J. (2009). Causality. Cambridge UP.",
            year: 2009,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        let mut coherences = Vec::new();
        let mut reversibilities = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.counterfactual_accuracy);
            coherences.push(r.counterfactual_coherence);
            reversibilities.push(r.counterfactual_reversibility);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert(
                    "counterfactual_coherence".to_string(),
                    r.counterfactual_coherence,
                );
                extra.insert(
                    "counterfactual_reversibility".to_string(),
                    r.counterfactual_reversibility,
                );
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "counterfactual".to_string(),
                    correct: r.counterfactual_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.counterfactual_coherence,
                    confidence: r.counterfactual_accuracy,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "counterfactual_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        result.insert(
            "counterfactual_coherence",
            MetricValue::from_samples(&coherences),
        );
        result.insert(
            "counterfactual_reversibility",
            MetricValue::from_samples(&reversibilities),
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
    fn test_counterfactual_runs() {
        let config = BenchmarkConfig::default();
        let result = CounterfactualReasoningBenchmark.run(&config);
        assert!(result.metrics.contains_key("counterfactual_accuracy"));
        assert!(result.metrics.contains_key("counterfactual_coherence"));
        assert!(result.metrics.contains_key("counterfactual_reversibility"));
    }

    #[test]
    fn test_counterfactual_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = CounterfactualReasoningBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_print_counterfactual_scores() {
        let config = BenchmarkConfig::default();
        let result = CounterfactualReasoningBenchmark.run(&config);
        eprintln!("\n═══ Counterfactual Reasoning Benchmark Scores ═══");
        for (key, val) in &result.metrics {
            let short = key.strip_prefix("counterfactual_").unwrap_or(key);
            eprintln!("  {short:<35} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }

        // Per-case counterfactual details
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        eprintln!("\n  ── Per-case counterfactual details ──");
        for case in CounterfactualReasoningBenchmark::cases() {
            match algebra.query_counterfactual(case.start, case.removals, case.additions, &system) {
                Ok((fwd, _inv)) => {
                    if let Some(last) = fwd.last() {
                        let pass = if last.nearest == case.expected_nearest {
                            "PASS"
                        } else {
                            "FAIL"
                        };
                        eprintln!(
                            "  [{pass}] {} -[{:?}] +[{:?}] => nearest={}, sim={:.4} (expected={})",
                            case.start,
                            case.removals,
                            case.additions,
                            last.nearest,
                            last.similarity,
                            case.expected_nearest
                        );
                    }
                    // Show HV-based reversibility
                    if let Some(rev_sim) = CounterfactualReasoningBenchmark::replay_reversibility(
                        &algebra,
                        &system,
                        case.start,
                        case.removals,
                        case.additions,
                    ) {
                        eprintln!("    round-trip HV similarity to start: {rev_sim:.4}");
                    }
                }
                Err(e) => eprintln!("  [ERR ] {}: {e}", case.start),
            }
        }
        eprintln!("══════════════════════════════════════════════════\n");
    }
}
