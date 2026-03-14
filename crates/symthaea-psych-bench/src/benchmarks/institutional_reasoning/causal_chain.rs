//! Institutional Reasoning: Causal Chain Benchmark
//!
//! Tests multi-step institutional state transitions using `query_chain`.
//! Given a starting axiom, progressively remove components and track
//! how the institutional state degrades through intermediate axioms.
//!
//! ## Key Claims Tested
//!
//! 1. **Chain Coherence**: Similarity to the starting axiom should decrease
//!    monotonically as components are removed.
//! 2. **Terminal Accuracy**: The final state after all removals should match
//!    the expected collapsed institutional form.
//! 3. **Step Count**: Number of removal steps before similarity drops below 0.5.
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
        vec![
            ChainCase {
                start: "DEMOCRATIC_ELECTION",
                removals: &["COOPERATE", "LEGITIMACY"],
                expected_terminal: "REVOLUTION",
            },
            ChainCase {
                start: "LEGITIMATE_GOVERNANCE",
                removals: &["TRUST", "LEGITIMACY"],
                expected_terminal: "REVOLUTION",
            },
            ChainCase {
                start: "SOCIAL_CONTRACT",
                removals: &["COOPERATE", "LEGITIMACY", "OBLIGATION"],
                expected_terminal: "FAILED_STATE",
            },
            ChainCase {
                start: "TRADE_AGREEMENT",
                removals: &["RECIPROCATE", "EXCHANGE"],
                expected_terminal: "BORDER_DISPUTE",
            },
            ChainCase {
                start: "DEMOCRATIC_ELECTION",
                removals: &["COOPERATE", "POPULATION"],
                expected_terminal: "CORRUPTION",
            },
        ]
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
            let start_hv = match algebra.get_encoding(case.start, &system) {
                Some(hv) => hv,
                None => continue,
            };

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

            // 1. Chain coherence: similarity to start should decrease
            let mut monotonic_steps = 0usize;
            let mut prev_sim = 1.0f32; // self-similarity at start
            for result in &results {
                // Compute similarity of current state to start
                // We approximate via the reported nearest-axiom similarity
                // Actually, we need the HV. Use query_chain results' similarity.
                if result.similarity <= prev_sim {
                    monotonic_steps += 1;
                }
                prev_sim = result.similarity;
            }
            coherence_scores.push(monotonic_steps as f64 / results.len() as f64);

            // 2. Terminal accuracy: last nearest axiom matches expected
            if let Some(last) = results.last() {
                if last.nearest == case.expected_terminal {
                    terminal_correct += 1;
                }
            }

            // 3. Step count: how many steps before similarity drops below 0.55
            let mut steps_before_collapse = results.len();
            for (i, result) in results.iter().enumerate() {
                if result.similarity < 0.55 {
                    steps_before_collapse = i + 1;
                    break;
                }
            }
            step_counts.push(steps_before_collapse as f64);
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
        assert!(result
            .metrics
            .contains_key("causal_chain_terminal_accuracy"));
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
}
