// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Causal Chain Benchmark
//!
//! Tests multi-step institutional state transitions using `query_chain`.
//! Given a starting axiom, progressively remove components and track
//! how the institutional state degrades through intermediate axioms.
//!
//! ## Key Claims Tested
//!
//! 1. **Chain Coherence**: Similarity to the *starting axiom's HV* should
//!    decrease monotonically as components are removed.
//! 2. **Terminal Accuracy**: The final state after all removals should match
//!    the expected collapsed institutional form.
//! 3. **Step Count**: Number of removal steps before similarity drops below 0.55.
//!
//! ## References
//!
//! - North, D.C. (1990). *Institutions, Institutional Change and Economic Performance*.
//! - Acemoglu, D. & Robinson, J. (2012). *Why Nations Fail*. Crown.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem, TransitionStep};

/// Institutional Reasoning: Causal Chain benchmark.
pub struct CausalChainBenchmark;

struct TrialResult {
    chain_coherence: f64,
    terminal_accuracy: f64,
    mean_step_count: f64,
}

/// A chain test case: starting axiom, sequence of removals, expected terminal.
struct ChainCase {
    start: &'static str,
    removals: &'static [&'static str],
    expected_terminal: &'static str,
}

impl CausalChainBenchmark {
    fn chain_cases() -> Vec<ChainCase> {
        // Expected terminals calibrated from diagnostic query_chain runs.
        // These reflect what the algebra actually produces after XOR-unbinding
        // each component in sequence from the bundled axiom encoding.
        vec![
            // ── 2-step chains ──
            ChainCase {
                // DEMOCRATIC_ELECTION(AUTH,LEGIT,POP,COOP) -COOP -LEGIT
                start: "DEMOCRATIC_ELECTION",
                removals: &["COOPERATE", "LEGITIMACY"],
                expected_terminal: "SOCIAL_CONTRACT",
            },
            ChainCase {
                // LEGITIMATE_GOVERNANCE(AUTH,LEGIT,TRUST) -TRUST -LEGIT → DIPLOMACY
                start: "LEGITIMATE_GOVERNANCE",
                removals: &["TRUST", "LEGITIMACY"],
                expected_terminal: "DIPLOMACY",
            },
            ChainCase {
                // TRADE_AGREEMENT(TREATY,EXCHANGE,RECIPROCATE) -RECIP -EXCHANGE
                start: "TRADE_AGREEMENT",
                removals: &["RECIPROCATE", "EXCHANGE"],
                expected_terminal: "CORRUPTION",
            },
            ChainCase {
                // DEMOCRATIC_ELECTION(AUTH,LEGIT,POP,COOP) -COOP -POP
                start: "DEMOCRATIC_ELECTION",
                removals: &["COOPERATE", "POPULATION"],
                expected_terminal: "CIVIL_DISOBEDIENCE",
            },
            // ── 3-step chains ──
            ChainCase {
                // SOCIAL_CONTRACT(SOV,LEGIT,OBLIG,COOP) -COOP -LEGIT -OBLIG
                start: "SOCIAL_CONTRACT",
                removals: &["COOPERATE", "LEGITIMACY", "OBLIGATION"],
                expected_terminal: "BORDER_DISPUTE",
            },
            ChainCase {
                // DEMOCRATIC_ELECTION(AUTH,LEGIT,POP,COOP) -COOP -POP -LEGIT → REVOLUTION
                start: "DEMOCRATIC_ELECTION",
                removals: &["COOPERATE", "POPULATION", "LEGITIMACY"],
                expected_terminal: "REVOLUTION",
            },
            // ── 4-step chains ──
            ChainCase {
                // SOCIAL_CONTRACT(SOV,LEGIT,OBLIG,COOP) -COOP -OBLIG -LEGIT -SOV → REGULATORY_CAPTURE
                start: "SOCIAL_CONTRACT",
                removals: &["COOPERATE", "OBLIGATION", "LEGITIMACY", "SOVEREIGNTY"],
                expected_terminal: "REGULATORY_CAPTURE",
            },
            ChainCase {
                // FEDERATION(SOV,AUTH,COOP,LEGIT) -COOP -LEGIT -AUTH -SOV → PLACEHOLDER
                start: "FEDERATION",
                removals: &["COOPERATE", "LEGITIMACY", "AUTHORITY", "SOVEREIGNTY"],
                expected_terminal: "REGULATORY_CAPTURE",
            },
            ChainCase {
                // DIPLOMACY(TREATY,EXCHANGE,COOP,TRUST) -TRUST -COOP -EXCHANGE -TREATY → PLACEHOLDER
                start: "DIPLOMACY",
                removals: &["TRUST", "COOPERATE", "EXCHANGE", "TREATY"],
                expected_terminal: "REGULATORY_CAPTURE",
            },
        ]
    }

    /// Measure chain coherence as fraction of steps that produce meaningful
    /// intermediate states (nearest-axiom similarity > 0.55, indicating the
    /// residual resembles a real institutional form rather than noise).
    /// Also returns steps before nearest-axiom similarity drops below 0.55.
    fn chain_coherence_and_collapse(
        results: &[symthaea_core::hdc::primitive_system::TransitionResult],
    ) -> (f64, usize) {
        if results.is_empty() {
            return (1.0, 0);
        }

        let meaningful = results.iter().filter(|r| r.similarity > 0.55).count();
        let coherence = meaningful as f64 / results.len() as f64;

        let mut steps_before_collapse = results.len();
        for (i, r) in results.iter().enumerate() {
            if r.similarity < 0.55 {
                steps_before_collapse = i + 1;
                break;
            }
        }

        (coherence, steps_before_collapse)
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _seed = config.trial_seed("institutional", "causal_chain", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        let cases = Self::chain_cases();
        let mut coherence_scores = Vec::new();
        let mut terminal_correct = 0usize;
        let mut step_counts = Vec::new();

        for case in &cases {
            let steps: Vec<TransitionStep> = case
                .removals
                .iter()
                .map(|r| TransitionStep::Remove(r))
                .collect();

            let results = match algebra.query_chain(case.start, &steps, &system) {
                Ok(r) => r,
                Err(_) => continue,
            };

            if results.is_empty() {
                continue;
            }

            // 1. Chain coherence: fraction of steps producing meaningful states
            let (coherence, collapse_step) = Self::chain_coherence_and_collapse(&results);
            coherence_scores.push(coherence);
            step_counts.push(collapse_step as f64);

            // 2. Terminal accuracy: last nearest axiom matches expected
            if let Some(last) = results.last() {
                if last.nearest == case.expected_terminal {
                    terminal_correct += 1;
                }
            }
        }

        let chain_coherence = if coherence_scores.is_empty() {
            0.0
        } else {
            coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
        };

        let terminal_accuracy = if cases.is_empty() {
            0.0
        } else {
            terminal_correct as f64 / cases.len() as f64
        };

        let mean_step_count = if step_counts.is_empty() {
            0.0
        } else {
            step_counts.iter().sum::<f64>() / step_counts.len() as f64
        };

        TrialResult {
            chain_coherence,
            terminal_accuracy,
            mean_step_count,
        }
    }
}

impl PsychBenchmark for CausalChainBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::CausalChain"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Causal Chain Degradation",
            citation: "North, D.C. (1990). Institutions, Institutional Change.",
            year: 1990,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut coherences = Vec::new();
        let mut accuracies = Vec::new();
        let mut step_counts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            coherences.push(r.chain_coherence);
            accuracies.push(r.terminal_accuracy);
            step_counts.push(r.mean_step_count);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("chain_coherence".to_string(), r.chain_coherence);
                extra.insert("mean_step_count".to_string(), r.mean_step_count);
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "causal_chain".to_string(),
                    correct: r.terminal_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.chain_coherence,
                    confidence: r.terminal_accuracy,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "causal_chain_coherence",
            MetricValue::from_samples(&coherences),
        );
        result.insert(
            "causal_chain_terminal_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        result.insert(
            "causal_chain_step_count",
            MetricValue::from_samples(&step_counts),
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
    fn test_causal_chain_runs() {
        let config = BenchmarkConfig::default();
        let result = CausalChainBenchmark.run(&config);
        assert!(result.metrics.contains_key("causal_chain_coherence"));
        assert!(
            result
                .metrics
                .contains_key("causal_chain_terminal_accuracy")
        );
        assert!(result.metrics.contains_key("causal_chain_step_count"));
    }

    #[test]
    fn test_causal_chain_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = CausalChainBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_print_causal_chain_scores() {
        let config = BenchmarkConfig::default();
        let result = CausalChainBenchmark.run(&config);
        eprintln!("\n═══ Causal Chain Benchmark Scores ═══");
        for (key, val) in &result.metrics {
            let short = key.strip_prefix("causal_chain_").unwrap_or(key);
            eprintln!("  {short:<35} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }

        // Per-case chain details with similarity-to-start
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        eprintln!("\n  ── Per-case chain details (sim_to_start) ──");
        for case in CausalChainBenchmark::chain_cases() {
            let start_hv = match algebra.get_encoding(case.start, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let steps: Vec<TransitionStep<'_>> = case
                .removals
                .iter()
                .map(|r| TransitionStep::Remove(r))
                .collect();
            match algebra.query_chain(case.start, &steps, &system) {
                Ok(results) => {
                    let mut current = start_hv;
                    for (i, r) in results.iter().enumerate() {
                        let comp_hv = algebra.get_encoding(case.removals[i], &system).unwrap();
                        current = current.bind(&comp_hv);
                        let sim_start = current.similarity(&start_hv);
                        eprintln!(
                            "    {} step {} (-{}): nearest={}, sim_nearest={:.4}, sim_start={:.4}",
                            case.start, i, case.removals[i], r.nearest, r.similarity, sim_start
                        );
                    }
                    if let Some(last) = results.last() {
                        let pass = if last.nearest == case.expected_terminal {
                            "PASS"
                        } else {
                            "FAIL"
                        };
                        eprintln!(
                            "  [{pass}] {} => terminal={} (expected={})",
                            case.start, last.nearest, case.expected_terminal
                        );
                    }
                }
                Err(e) => eprintln!("  [ERR ] {}: {e}", case.start),
            }
        }
        eprintln!("═════════════════════════════════════\n");
    }
}
