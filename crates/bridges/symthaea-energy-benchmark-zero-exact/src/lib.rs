// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-candidate-universe wrapper for Energy Discovery Benchmark Zero.
//!
//! The generic Benchmark Zero evaluator permits the truth set to contain
//! candidates that were not rankable. That is useful generally, but unsafe for
//! the Matbench leakage-controlled path where exposed-training compositions are
//! deliberately excluded from the screening universe.
//!
//! This adapter fails closed unless the ranked candidate ids and truth candidate
//! ids are exactly equal before delegating to the existing measurement math.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use symthaea_energy_benchmark_zero::{
    BandgapTruthSet, BenchmarkError, BenchmarkReceipt, ScreeningRun, evaluate,
};
use thiserror::Error;

pub const RECEIPT_SCHEMA: &str = "symthaea.energy-benchmark-zero.exact-universe.v1";
pub const CAPABILITY_CLASSIFICATION: &str =
    "EXACT-UNIVERSE MEASUREMENT WRAPPER ONLY -- not a screening method, holdout-cleanliness certificate, material certification, or promotion authority.";

const CANDIDATE_UNIVERSE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.exact-candidate-universe.v1\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.exact-universe-receipt.v1\0";

/// Replayable evidence that generic Benchmark Zero measurement was performed
/// only after exact candidate-universe equality was established.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactUniverseBenchmarkReceipt {
    pub schema: String,
    pub capability_classification: String,
    /// SHA-256 over canonical sorted candidate ids. This digest is independent
    /// of ranking order and experimental values.
    pub candidate_universe_sha256: String,
    pub candidate_count: usize,
    pub benchmark: BenchmarkReceipt,
}

impl ExactUniverseBenchmarkReceipt {
    pub fn validate(&self) -> Result<(), ExactUniverseError> {
        if self.schema != RECEIPT_SCHEMA
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(ExactUniverseError::InvalidReceipt(
                "schema or capability classification was altered".into(),
            ));
        }
        validate_sha256(&self.candidate_universe_sha256)?;
        if self.candidate_count == 0 {
            return Err(ExactUniverseError::InvalidReceipt(
                "candidate universe cannot be empty".into(),
            ));
        }
        if self.benchmark.metrics.ranked_candidate_count != self.candidate_count
            || self.benchmark.metrics.truth_candidate_count != self.candidate_count
        {
            return Err(ExactUniverseError::InvalidReceipt(
                "benchmark ranked/truth counts do not equal exact candidate count".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ExactUniverseError> {
        self.validate()?;
        domain_separated_sha256(RECEIPT_DIGEST_DOMAIN, self)
    }
}

/// Evaluate one already-ranked run only when its candidate ids exactly equal
/// the truth candidate ids.
///
/// This prevents excluded/non-rankable truth rows from affecting qualifying
/// denominators, global-best target error, recall, or target regret.
pub fn evaluate_exact_universe(
    run: &ScreeningRun,
    truth: &BandgapTruthSet,
    k: usize,
) -> Result<ExactUniverseBenchmarkReceipt, ExactUniverseError> {
    run.validate()?;
    truth.validate()?;

    let ranked_ids: BTreeSet<&str> = run
        .ranked
        .iter()
        .map(|record| record.candidate_id.as_str())
        .collect();
    let truth_ids: BTreeSet<&str> = truth
        .experimental_gap_ev
        .keys()
        .map(String::as_str)
        .collect();

    if ranked_ids != truth_ids {
        let ranked_only = ranked_ids.difference(&truth_ids).count();
        let truth_only = truth_ids.difference(&ranked_ids).count();
        return Err(ExactUniverseError::CandidateUniverseMismatch {
            ranked_count: ranked_ids.len(),
            truth_count: truth_ids.len(),
            ranked_only,
            truth_only,
        });
    }

    let candidate_universe_sha256 = candidate_universe_sha256(ranked_ids.iter().copied())?;
    let benchmark = evaluate(run, truth, k)?;
    let receipt = ExactUniverseBenchmarkReceipt {
        schema: RECEIPT_SCHEMA.into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        candidate_universe_sha256,
        candidate_count: ranked_ids.len(),
        benchmark,
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Replay exact-universe validation and measurement from the supplied run and
/// truth, then require byte-equivalent semantic receipt equality.
pub fn verify_exact_universe_receipt(
    run: &ScreeningRun,
    truth: &BandgapTruthSet,
    expected: &ExactUniverseBenchmarkReceipt,
) -> Result<(), ExactUniverseError> {
    expected.validate()?;
    let observed = evaluate_exact_universe(run, truth, expected.benchmark.metrics.k)?;
    if &observed != expected {
        return Err(ExactUniverseError::ReplayMismatch);
    }
    Ok(())
}

pub fn candidate_universe_sha256<'a>(
    candidate_ids: impl IntoIterator<Item = &'a str>,
) -> Result<String, ExactUniverseError> {
    let ids: BTreeSet<&str> = candidate_ids.into_iter().collect();
    if ids.is_empty() || ids.iter().any(|id| id.trim().is_empty()) {
        return Err(ExactUniverseError::InvalidCandidateUniverse);
    }
    domain_separated_sha256(CANDIDATE_UNIVERSE_DIGEST_DOMAIN, &ids)
}

fn validate_sha256(value: &str) -> Result<(), ExactUniverseError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ExactUniverseError::InvalidReceipt(
            "candidate universe SHA-256 must be 64 hexadecimal characters".into(),
        ));
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, ExactUniverseError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum ExactUniverseError {
    #[error("Benchmark Zero rejected input: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error(
        "ranked/truth candidate universes differ: ranked={ranked_count}, truth={truth_count}, ranked_only={ranked_only}, truth_only={truth_only}"
    )]
    CandidateUniverseMismatch {
        ranked_count: usize,
        truth_count: usize,
        ranked_only: usize,
        truth_only: usize,
    },
    #[error("candidate universe must contain unique non-empty ids")]
    InvalidCandidateUniverse,
    #[error("invalid exact-universe receipt: {0}")]
    InvalidReceipt(String),
    #[error("exact-universe receipt replay differs from supplied receipt")]
    ReplayMismatch,
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use symthaea_energy_benchmark_zero::{
        BandgapTarget, DataSliceId, DatasetProvenance, ScreeningMethodProvenance,
        ScreeningRecord,
    };

    fn run() -> ScreeningRun {
        ScreeningRun {
            method: ScreeningMethodProvenance {
                method_id: "fixture".into(),
                version: "v0".into(),
                training_slices: BTreeSet::new(),
            },
            target: BandgapTarget::new(1.0, 1.5).unwrap(),
            ranked: vec![
                ScreeningRecord {
                    candidate_id: "A".into(),
                    predicted_gap_ev: 1.2,
                    uncertainty_ev: None,
                },
                ScreeningRecord {
                    candidate_id: "B".into(),
                    predicted_gap_ev: 1.8,
                    uncertainty_ev: None,
                },
            ],
        }
    }

    fn truth() -> BandgapTruthSet {
        BandgapTruthSet {
            provenance: DatasetProvenance {
                slice: DataSliceId::new("fixture", "truth").unwrap(),
                source_uri: "https://example.invalid/fixture".into(),
                content_digest: "sha256:fixture".into(),
                license: "fixture".into(),
            },
            experimental_gap_ev: BTreeMap::from([("A".into(), 1.1), ("B".into(), 1.9)]),
        }
    }

    #[test]
    fn exact_universe_delegates_measurement_and_binds_digest() {
        let receipt = evaluate_exact_universe(&run(), &truth(), 1).unwrap();
        assert_eq!(receipt.candidate_count, 2);
        assert_eq!(receipt.benchmark.metrics.ranked_candidate_count, 2);
        assert_eq!(receipt.benchmark.metrics.truth_candidate_count, 2);
        verify_exact_universe_receipt(&run(), &truth(), &receipt).unwrap();
    }

    #[test]
    fn extra_truth_candidate_fails_before_measurement() {
        let mut truth = truth();
        truth.experimental_gap_ev.insert("EXCLUDED".into(), 1.25);
        let error = evaluate_exact_universe(&run(), &truth, 1).unwrap_err();
        assert!(matches!(
            error,
            ExactUniverseError::CandidateUniverseMismatch {
                ranked_count: 2,
                truth_count: 3,
                ranked_only: 0,
                truth_only: 1,
            }
        ));
    }

    #[test]
    fn missing_ranked_truth_candidate_fails_as_universe_mismatch() {
        let mut truth = truth();
        truth.experimental_gap_ev.remove("B");
        let error = evaluate_exact_universe(&run(), &truth, 1).unwrap_err();
        assert!(matches!(
            error,
            ExactUniverseError::CandidateUniverseMismatch {
                ranked_count: 2,
                truth_count: 1,
                ranked_only: 1,
                truth_only: 0,
            }
        ));
    }

    #[test]
    fn universe_digest_is_order_independent() {
        let first = candidate_universe_sha256(["A", "B"]).unwrap();
        let second = candidate_universe_sha256(["B", "A"]).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn tampered_universe_digest_fails_structural_validation() {
        let mut receipt = evaluate_exact_universe(&run(), &truth(), 1).unwrap();
        receipt.candidate_universe_sha256 = "0".repeat(64);
        let observed = evaluate_exact_universe(&run(), &truth(), 1).unwrap();
        assert_ne!(receipt, observed);
        assert!(verify_exact_universe_receipt(&run(), &truth(), &receipt).is_err());
    }
}
