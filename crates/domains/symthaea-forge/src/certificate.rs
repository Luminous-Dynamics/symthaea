// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human-reviewable report for a Forge candidate mutation.
//!
//! `ForgeCertificate` is deliberately a review artifact, not a generic algorithm-evidence or
//! promotion receipt. The evidence-first algorithm stack may later bind its exact candidate bytes,
//! transformations, gates, and benchmark runs into stronger typed receipts.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvidence {
    pub gate: String,
    pub passed: bool,
    pub duration_ms: u128,
    pub output_tail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEvidence {
    pub metric_name: String,
    /// Exact finite score from the baseline benchmark harness. Lower is better.
    pub baseline_score: f64,
    /// Exact finite score from the candidate benchmark harness. Lower is better.
    pub candidate_score: f64,
    /// Baseline minus candidate. Positive means the candidate's lower-is-better score improved.
    pub absolute_improvement: f64,
    /// Relative improvement against `abs(baseline)`. `None` when the baseline is exactly zero,
    /// because a relative fraction would be undefined rather than zero or infinity.
    pub improvement_fraction: Option<f64>,
}

/// One accepted mutation in a candidate's ordered derivation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    pub generation: usize,
    pub operator: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeCertificate {
    pub generated_at_unix_ms: u128,
    pub target_file: PathBuf,
    pub target_function: String,
    pub package: String,
    pub git_sha: Option<String>,
    pub generation: usize,
    /// Convenience fields mirroring the final mutation-history entry.
    pub mutation_operator: String,
    pub mutation_detail: String,
    /// Every generation-winning mutation compounded into `after_source`, in order.
    pub mutation_history: Vec<MutationRecord>,
    pub gates: Vec<GateEvidence>,
    pub benchmark: Option<BenchmarkEvidence>,
    /// Original containing function only.
    pub before_source: String,
    /// Candidate containing function after the full ordered mutation history.
    pub after_source: String,
}

impl ForgeCertificate {
    pub fn all_gates_passed(&self) -> bool {
        self.gates.iter().all(|gate| gate.passed)
    }

    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Human-readable summary. A benchmark delta is an observed search score, not replicated
    /// evidence of production superiority.
    pub fn summary(&self) -> String {
        let bench_line = match &self.benchmark {
            Some(benchmark) => match benchmark.improvement_fraction {
                Some(fraction) => format!(
                    "benchmark: {} {:.1} -> {:.1} (Δ={:+.1}, {:+.2}%)",
                    benchmark.metric_name,
                    benchmark.baseline_score,
                    benchmark.candidate_score,
                    benchmark.absolute_improvement,
                    fraction * 100.0
                ),
                None => format!(
                    "benchmark: {} {:.1} -> {:.1} (Δ={:+.1}; relative change undefined for zero baseline)",
                    benchmark.metric_name,
                    benchmark.baseline_score,
                    benchmark.candidate_score,
                    benchmark.absolute_improvement,
                ),
            },
            None => "benchmark: not configured (correctness-only search)".to_string(),
        };
        let lineage_note = if self.mutation_history.len() > 1 {
            format!(
                "  ⚠ this diff compounds {} mutations across generations 0-{} -- see mutation_history, not just the line below\n",
                self.mutation_history.len(),
                self.generation
            )
        } else {
            String::new()
        };
        format!(
            "[{status}] {op} on {func} in {file}\n{lineage_note}  {detail}\n  {bench_line}",
            status = if self.all_gates_passed() { "PASS" } else { "FAIL" },
            op = self.mutation_operator,
            func = self.target_function,
            file = self.target_file.display(),
            detail = self.mutation_detail,
        )
    }
}

pub fn gate_result_to_evidence(gate: &crate::fitness::GateResult) -> GateEvidence {
    GateEvidence {
        gate: gate.gate.label().to_string(),
        passed: gate.passed,
        duration_ms: gate.duration.as_millis(),
        output_tail: gate.output_tail.clone(),
    }
}

pub fn unix_millis_now(elapsed_since_epoch: Duration) -> u128 {
    elapsed_since_epoch.as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::{Gate, GateResult};
    use std::time::Duration;

    fn sample_certificate(all_pass: bool) -> ForgeCertificate {
        ForgeCertificate {
            generated_at_unix_ms: 0,
            target_file: PathBuf::from("src/entropy.rs"),
            target_function: "entropy_histogram".to_string(),
            package: "symthaea-core".to_string(),
            git_sha: Some("deadbeef".to_string()),
            generation: 2,
            mutation_operator: "NumericLiteralPerturb".to_string(),
            mutation_detail: "0.9999 -> 0.9995".to_string(),
            mutation_history: vec![MutationRecord {
                generation: 2,
                operator: "NumericLiteralPerturb".to_string(),
                detail: "0.9999 -> 0.9995".to_string(),
            }],
            gates: vec![
                GateEvidence {
                    gate: "compile".to_string(),
                    passed: true,
                    duration_ms: 500,
                    output_tail: String::new(),
                },
                GateEvidence {
                    gate: "test".to_string(),
                    passed: all_pass,
                    duration_ms: 300,
                    output_tail: String::new(),
                },
            ],
            benchmark: Some(BenchmarkEvidence {
                metric_name: "median_ns".to_string(),
                baseline_score: 1000.0,
                candidate_score: 900.0,
                absolute_improvement: 100.0,
                improvement_fraction: Some(0.1),
            }),
            before_source: "fn entropy_histogram(&self, hv: &ContinuousHV) -> f64 { 0 }".into(),
            after_source: "fn entropy_histogram(&self, hv: &ContinuousHV) -> f64 { 1 }".into(),
        }
    }

    #[test]
    fn all_gates_passed_reflects_gate_evidence() {
        assert!(sample_certificate(true).all_gates_passed());
        assert!(!sample_certificate(false).all_gates_passed());
    }

    #[test]
    fn json_roundtrips_optional_relative_improvement() {
        let cert = sample_certificate(true);
        let json = cert.to_json_pretty().unwrap();
        let back: ForgeCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.target_function, cert.target_function);
        assert_eq!(
            back.benchmark.unwrap().improvement_fraction,
            Some(0.1)
        );
    }

    #[test]
    fn summary_reports_pass_and_benchmark_delta() {
        let summary = sample_certificate(true).summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("10.00%"));
        assert!(summary.contains("Δ=+100.0"));
    }

    #[test]
    fn zero_baseline_has_no_fake_relative_percentage() {
        let mut cert = sample_certificate(true);
        let benchmark = cert.benchmark.as_mut().unwrap();
        benchmark.baseline_score = 0.0;
        benchmark.candidate_score = -1.0;
        benchmark.absolute_improvement = 1.0;
        benchmark.improvement_fraction = None;
        assert!(cert.summary().contains("relative change undefined"));
    }

    #[test]
    fn summary_warns_when_multiple_mutations_are_compounded() {
        let mut cert = sample_certificate(true);
        cert.mutation_history.push(MutationRecord {
            generation: 1,
            operator: "ArithmeticOperatorSwap".to_string(),
            detail: "- -> +".to_string(),
        });
        assert!(cert.summary().contains("compounds 2 mutations"));
    }

    #[test]
    fn summary_reports_fail_when_a_gate_fails() {
        assert!(sample_certificate(false).summary().contains("FAIL"));
    }

    #[test]
    fn gate_result_to_evidence_preserves_fields() {
        let result = GateResult {
            gate: Gate::Compile,
            passed: true,
            output_tail: "ok".into(),
            duration: Duration::from_millis(42),
        };
        let evidence = gate_result_to_evidence(&result);
        assert_eq!(evidence.gate, "compile");
        assert!(evidence.passed);
        assert_eq!(evidence.duration_ms, 42);
    }
}