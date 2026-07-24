// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human-reviewable evidence record for a candidate mutation.
//!
//! A certificate is written to disk for the best surviving candidate of a
//! search run -- it is never auto-applied. Per the self-mod safety rules
//! ("human stays at every candidate->live boundary"), the certificate's
//! job is to give a human everything needed to decide whether to actually
//! copy the candidate over the real source file: what changed, why it was
//! judged better, and the exact evidence (gate pass/fail + benchmark
//! numbers) rather than a bare claim.

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
    pub baseline_score: f64,
    pub candidate_score: f64,
    /// Positive = improvement (candidate score is lower, since fitness.rs's
    /// convention is "lower wins").
    pub improvement_fraction: f64,
}

/// One accepted mutation in a candidate's lineage. A winning candidate from
/// generation N carries the full chain of every generation-winning mutation
/// from 0..=N, not just its own -- `search.rs`'s elitism means each
/// generation mutates the *previous* generation's winner, so a generation-2
/// certificate's `after_source` reflects TWO compounded mutations, not one.
/// (Found the hard way: a first real search campaign against
/// `chi_squared_test` produced a certificate labeled with a single
/// `ArithmeticOperatorSwap`, but the before/after diff silently also
/// contained an earlier generation's separate accepted mutation -- see
/// `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md` Tier 2 status.)
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
    /// Convenience fields mirroring the LAST entry of `mutation_history` --
    /// kept for simple display, but never read these alone to understand
    /// what changed if `mutation_history.len() > 1`.
    pub mutation_operator: String,
    pub mutation_detail: String,
    /// Every generation-winning mutation from generation 0 up to and
    /// including this candidate's own, in order. Length 1 means
    /// `before_source`/`after_source` differ by exactly this one mutation;
    /// length > 1 means they differ by all of them compounded.
    pub mutation_history: Vec<MutationRecord>,
    pub gates: Vec<GateEvidence>,
    pub benchmark: Option<BenchmarkEvidence>,
    /// Full before/after source of the *containing function only* (not the
    /// whole file) so a human can read the diff without opening the repo.
    /// `before_source` is always the pristine original; `after_source`
    /// reflects the full `mutation_history` applied in order, not just the
    /// latest entry.
    pub before_source: String,
    pub after_source: String,
}

impl ForgeCertificate {
    pub fn all_gates_passed(&self) -> bool {
        self.gates.iter().all(|g| g.passed)
    }

    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// A short human-readable summary, suitable for a terminal or a
    /// `report.md` header. Explicitly flags when the diff reflects more
    /// than one compounded mutation, since that's easy to miss.
    pub fn summary(&self) -> String {
        let bench_line = match &self.benchmark {
            Some(b) => format!(
                "benchmark: {} {:.1} -> {:.1} ({:+.2}%)",
                b.metric_name,
                b.baseline_score,
                b.candidate_score,
                b.improvement_fraction * 100.0
            ),
            None => "benchmark: not run".to_string(),
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
            status = if self.all_gates_passed() {
                "PASS"
            } else {
                "FAIL"
            },
            op = self.mutation_operator,
            func = self.target_function,
            file = self.target_file.display(),
            detail = self.mutation_detail,
        )
    }
}

pub fn gate_result_to_evidence(g: &crate::fitness::GateResult) -> GateEvidence {
    GateEvidence {
        gate: g.gate.label().to_string(),
        passed: g.passed,
        duration_ms: g.duration.as_millis(),
        output_tail: g.output_tail.clone(),
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
                improvement_fraction: 0.1,
            }),
            before_source: "fn entropy_histogram(&self, hv: &ContinuousHV) -> f64 { 0 }"
                .to_string(),
            after_source: "fn entropy_histogram(&self, hv: &ContinuousHV) -> f64 { 1 }".to_string(),
        }
    }

    #[test]
    fn all_gates_passed_reflects_gate_evidence() {
        assert!(sample_certificate(true).all_gates_passed());
        assert!(!sample_certificate(false).all_gates_passed());
    }

    #[test]
    fn json_roundtrips() {
        let cert = sample_certificate(true);
        let json = cert.to_json_pretty().unwrap();
        let back: ForgeCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.target_function, cert.target_function);
        assert_eq!(back.mutation_detail, cert.mutation_detail);
    }

    #[test]
    fn summary_reports_pass_and_benchmark_delta() {
        let cert = sample_certificate(true);
        let s = cert.summary();
        assert!(s.contains("PASS"));
        assert!(s.contains("10.00%") || s.contains("+10.00%"));
    }

    #[test]
    fn summary_warns_when_multiple_mutations_are_compounded() {
        let mut cert = sample_certificate(true);
        assert!(
            !cert.summary().contains("compounds"),
            "a single-mutation history should not trigger the warning"
        );
        cert.mutation_history.push(MutationRecord {
            generation: 1,
            operator: "ArithmeticOperatorSwap".to_string(),
            detail: "- -> +".to_string(),
        });
        let s = cert.summary();
        assert!(
            s.contains("compounds 2 mutations"),
            "a multi-entry history must warn a reviewer it's not just the labeled mutation, got: {s}"
        );
    }

    #[test]
    fn summary_reports_fail_when_a_gate_fails() {
        let cert = sample_certificate(false);
        assert!(cert.summary().contains("FAIL"));
    }

    #[test]
    fn gate_result_to_evidence_preserves_fields() {
        let g = GateResult {
            gate: Gate::Compile,
            passed: true,
            output_tail: "ok".to_string(),
            duration: Duration::from_millis(42),
        };
        let e = gate_result_to_evidence(&g);
        assert_eq!(e.gate, "compile");
        assert!(e.passed);
        assert_eq!(e.duration_ms, 42);
    }
}
