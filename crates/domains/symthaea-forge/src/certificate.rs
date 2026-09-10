// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human-reviewable, content-addressed report for a Forge candidate mutation.
//!
//! `ForgeCertificate` remains a search/review artifact rather than a performance or promotion
//! receipt. It does, however, bind the exact full source file evaluated by Forge and the ordered
//! parent->child artifact chain of every accepted mutation. That gives the evidence-first
//! algorithm stack a trustworthy candidate identity without treating Forge's local benchmark as
//! replicated evidence.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use symthaea_algorithms::{ContentId, TransformationId};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CertificateError {
    #[error("Forge certificate must contain at least one accepted mutation")]
    EmptyMutationHistory,
    #[error("Forge survivor certificate contains a failed correctness gate")]
    CorrectnessGateFailed,
    #[error("mutation history generation order is not strictly increasing")]
    NonIncreasingGeneration,
    #[error("mutation artifact chain is discontinuous")]
    ArtifactChainMismatch,
    #[error("certificate convenience fields do not match the final mutation")]
    FinalMutationMismatch,
    #[error("benchmark evidence is non-finite or arithmetically inconsistent")]
    InvalidBenchmarkEvidence,
    #[error("full candidate source does not match its certificate artifact identity")]
    CandidateSourceMismatch,
}

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
    /// Exact finite score from the parent benchmark harness. Lower is better.
    pub baseline_score: f64,
    /// Exact finite score from the candidate benchmark harness. Lower is better.
    pub candidate_score: f64,
    /// Baseline minus candidate. Positive means the candidate's lower-is-better score improved.
    pub absolute_improvement: f64,
    /// Relative improvement against `abs(baseline)`. `None` when the baseline is exactly zero,
    /// because a relative fraction would be undefined rather than zero or infinity.
    pub improvement_fraction: Option<f64>,
}

impl BenchmarkEvidence {
    pub fn validate(&self) -> Result<(), CertificateError> {
        if self.metric_name.trim().is_empty()
            || !self.baseline_score.is_finite()
            || !self.candidate_score.is_finite()
            || !self.absolute_improvement.is_finite()
            || self.improvement_fraction.is_some_and(|value| !value.is_finite())
        {
            return Err(CertificateError::InvalidBenchmarkEvidence);
        }

        let expected_absolute = self.baseline_score - self.candidate_score;
        if expected_absolute.to_bits() != self.absolute_improvement.to_bits() {
            return Err(CertificateError::InvalidBenchmarkEvidence);
        }
        let expected_relative = if self.baseline_score == 0.0 {
            None
        } else {
            Some(expected_absolute / self.baseline_score.abs())
        };
        let same_relative = match (expected_relative, self.improvement_fraction) {
            (None, None) => true,
            (Some(expected), Some(observed)) => expected.to_bits() == observed.to_bits(),
            _ => false,
        };
        if !same_relative {
            return Err(CertificateError::InvalidBenchmarkEvidence);
        }
        Ok(())
    }
}

/// Stable content identity for one exact full UTF-8 Rust source file as seen by Forge.
pub fn full_source_artifact_id(source: &str) -> ContentId {
    ContentId::derive("symthaea.forge-full-source.v1", [source.as_bytes()])
}

/// One accepted mutation in a candidate's ordered derivation history.
///
/// Parent/candidate artifact identities make the textual mutation description auditable as a
/// concrete transition rather than a free-floating claim. `transformation_id` commits the exact
/// generation, operator, detail and parent/child content identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    pub generation: usize,
    pub operator: String,
    pub detail: String,
    pub parent_artifact_id: ContentId,
    pub candidate_artifact_id: ContentId,
    pub transformation_id: TransformationId,
}

impl MutationRecord {
    pub fn new(
        generation: usize,
        operator: impl Into<String>,
        detail: impl Into<String>,
        parent_artifact_id: ContentId,
        candidate_artifact_id: ContentId,
    ) -> Self {
        let operator = operator.into();
        let detail = detail.into();
        let generation_bytes = (generation as u128).to_be_bytes();
        let transformation_id = TransformationId(ContentId::derive(
            "symthaea.forge-transformation.v1",
            [
                generation_bytes.as_slice(),
                operator.as_bytes(),
                detail.as_bytes(),
                parent_artifact_id.as_str().as_bytes(),
                candidate_artifact_id.as_str().as_bytes(),
            ],
        ));
        Self {
            generation,
            operator,
            detail,
            parent_artifact_id,
            candidate_artifact_id,
            transformation_id,
        }
    }

    pub fn validate(&self) -> Result<(), CertificateError> {
        let rebuilt = Self::new(
            self.generation,
            self.operator.clone(),
            self.detail.clone(),
            self.parent_artifact_id.clone(),
            self.candidate_artifact_id.clone(),
        );
        if rebuilt.transformation_id != self.transformation_id {
            return Err(CertificateError::ArtifactChainMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeCertificate {
    pub generated_at_unix_ms: u128,
    pub target_file: PathBuf,
    pub target_function: String,
    pub package: String,
    pub git_sha: Option<String>,
    pub generation: usize,
    /// Exact pristine full-file identity before Forge applied any accepted mutation.
    pub baseline_artifact_id: ContentId,
    /// Exact full-file identity of the final survivor that passed Forge's configured gates.
    pub candidate_artifact_id: ContentId,
    /// Convenience fields mirroring the final mutation-history entry.
    pub mutation_operator: String,
    pub mutation_detail: String,
    /// Every accepted mutation compounded into the final candidate, in execution order.
    pub mutation_history: Vec<MutationRecord>,
    pub gates: Vec<GateEvidence>,
    pub benchmark: Option<BenchmarkEvidence>,
    /// Original containing function only, for human review.
    pub before_source: String,
    /// Candidate containing function after the full ordered mutation history.
    pub after_source: String,
}

impl ForgeCertificate {
    pub fn all_gates_passed(&self) -> bool {
        !self.gates.is_empty() && self.gates.iter().all(|gate| gate.passed)
    }

    /// Validate the certificate's self-contained structural claims.
    ///
    /// This does not prove that Cargo commands actually ran or that the benchmark is repeatable;
    /// those stronger claims belong to later capsule/evaluation evidence.
    pub fn validate(&self) -> Result<(), CertificateError> {
        if self.mutation_history.is_empty() {
            return Err(CertificateError::EmptyMutationHistory);
        }
        if !self.all_gates_passed() {
            return Err(CertificateError::CorrectnessGateFailed);
        }
        if let Some(benchmark) = &self.benchmark {
            benchmark.validate()?;
        }

        let mut previous_generation = None;
        let mut expected_parent = &self.baseline_artifact_id;
        for mutation in &self.mutation_history {
            mutation.validate()?;
            if previous_generation.is_some_and(|previous| mutation.generation <= previous) {
                return Err(CertificateError::NonIncreasingGeneration);
            }
            if &mutation.parent_artifact_id != expected_parent {
                return Err(CertificateError::ArtifactChainMismatch);
            }
            previous_generation = Some(mutation.generation);
            expected_parent = &mutation.candidate_artifact_id;
        }

        let final_mutation = self
            .mutation_history
            .last()
            .ok_or(CertificateError::EmptyMutationHistory)?;
        if self.generation != final_mutation.generation
            || self.mutation_operator != final_mutation.operator
            || self.mutation_detail != final_mutation.detail
        {
            return Err(CertificateError::FinalMutationMismatch);
        }
        if self.candidate_artifact_id != final_mutation.candidate_artifact_id {
            return Err(CertificateError::ArtifactChainMismatch);
        }
        Ok(())
    }

    pub fn transformation_ids(&self) -> Vec<TransformationId> {
        self.mutation_history
            .iter()
            .map(|mutation| mutation.transformation_id.clone())
            .collect()
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
                "  ⚠ this diff compounds {} mutations across accepted generations -- see mutation_history, not just the line below\n",
                self.mutation_history.len()
            )
        } else {
            String::new()
        };
        format!(
            "[{status}] {op} on {func} in {file}\n{lineage_note}  {detail}\n  artifact: {artifact}\n  {bench_line}",
            status = if self.all_gates_passed() { "PASS" } else { "FAIL" },
            op = self.mutation_operator,
            func = self.target_function,
            file = self.target_file.display(),
            detail = self.mutation_detail,
            artifact = self.candidate_artifact_id,
        )
    }
}

/// Exact survivor bytes paired with the certificate that names those bytes.
#[derive(Debug, Clone)]
pub struct ForgeCandidate {
    certificate: ForgeCertificate,
    full_source: String,
}

impl ForgeCandidate {
    pub fn new(
        certificate: ForgeCertificate,
        full_source: String,
    ) -> Result<Self, CertificateError> {
        certificate.validate()?;
        if full_source_artifact_id(&full_source) != certificate.candidate_artifact_id {
            return Err(CertificateError::CandidateSourceMismatch);
        }
        Ok(Self {
            certificate,
            full_source,
        })
    }

    pub fn validate(&self) -> Result<(), CertificateError> {
        self.certificate.validate()?;
        if full_source_artifact_id(&self.full_source) != self.certificate.candidate_artifact_id {
            return Err(CertificateError::CandidateSourceMismatch);
        }
        Ok(())
    }

    pub fn certificate(&self) -> &ForgeCertificate {
        &self.certificate
    }

    pub fn full_source(&self) -> &str {
        &self.full_source
    }

    pub fn artifact_id(&self) -> &ContentId {
        &self.certificate.candidate_artifact_id
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

    fn sample_certificate(all_pass: bool) -> (ForgeCertificate, String) {
        let baseline_source = "fn entropy_histogram() -> f64 { 0.0 }\n";
        let candidate_source = "fn entropy_histogram() -> f64 { 1.0 }\n".to_string();
        let baseline_artifact_id = full_source_artifact_id(baseline_source);
        let candidate_artifact_id = full_source_artifact_id(&candidate_source);
        let mutation = MutationRecord::new(
            2,
            "NumericLiteralPerturb",
            "0.0 -> 1.0",
            baseline_artifact_id.clone(),
            candidate_artifact_id.clone(),
        );
        (
            ForgeCertificate {
                generated_at_unix_ms: 0,
                target_file: PathBuf::from("src/entropy.rs"),
                target_function: "entropy_histogram".to_string(),
                package: "symthaea-core".to_string(),
                git_sha: Some("deadbeef".to_string()),
                generation: 2,
                baseline_artifact_id,
                candidate_artifact_id,
                mutation_operator: mutation.operator.clone(),
                mutation_detail: mutation.detail.clone(),
                mutation_history: vec![mutation],
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
                before_source: "fn entropy_histogram() -> f64 { 0.0 }".into(),
                after_source: "fn entropy_histogram() -> f64 { 1.0 }".into(),
            },
            candidate_source,
        )
    }

    #[test]
    fn validation_requires_passing_gates() {
        assert!(sample_certificate(true).0.validate().is_ok());
        assert_eq!(
            sample_certificate(false).0.validate().unwrap_err(),
            CertificateError::CorrectnessGateFailed
        );
    }

    #[test]
    fn artifact_chain_detects_substitution() {
        let (mut cert, _) = sample_certificate(true);
        cert.mutation_history[0].candidate_artifact_id =
            full_source_artifact_id("fn entropy_histogram() -> f64 { 2.0 }\n");
        assert!(cert.validate().is_err());
    }

    #[test]
    fn candidate_binds_exact_full_source() {
        let (cert, source) = sample_certificate(true);
        let candidate = ForgeCandidate::new(cert.clone(), source).unwrap();
        assert!(candidate.validate().is_ok());
        assert_eq!(candidate.artifact_id(), &cert.candidate_artifact_id);
        assert_eq!(
            ForgeCandidate::new(cert, "different bytes\n".into()).unwrap_err(),
            CertificateError::CandidateSourceMismatch
        );
    }

    #[test]
    fn json_roundtrip_preserves_content_addressed_lineage() {
        let (cert, _) = sample_certificate(true);
        let json = cert.to_json_pretty().unwrap();
        let back: ForgeCertificate = serde_json::from_str(&json).unwrap();
        assert!(back.validate().is_ok());
        assert_eq!(back.candidate_artifact_id, cert.candidate_artifact_id);
        assert_eq!(back.transformation_ids(), cert.transformation_ids());
    }

    #[test]
    fn benchmark_arithmetic_is_revalidated() {
        let (mut cert, _) = sample_certificate(true);
        cert.benchmark.as_mut().unwrap().absolute_improvement = 99.0;
        assert_eq!(
            cert.validate().unwrap_err(),
            CertificateError::InvalidBenchmarkEvidence
        );
    }

    #[test]
    fn summary_reports_pass_and_artifact() {
        let (cert, _) = sample_certificate(true);
        let summary = cert.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("10.00%"));
        assert!(summary.contains(cert.candidate_artifact_id.as_str()));
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