// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition-only baseline runner for Energy Discovery Benchmark Zero.
//!
//! The prediction/ranking core in this crate never receives experimental gap
//! values. The outer evaluation path first obtains a leakage-qualified Matbench
//! truth slice, reconstructs only the retained candidate roster/compositions,
//! freezes the baseline ranking, and only then passes that ranking plus truth to
//! the measurement-only Benchmark Zero protocol.
//!
//! This is a baseline measurement path, not a material discovery, certification,
//! novelty, experiment, procurement, manufacturing, deployment, or actuation
//! authority path.

#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use symthaea_bandgap::bandgap_baseline::electronegativity_bandgap;
use symthaea_energy_benchmark_zero::{
    evaluate, BandgapTarget, BenchmarkError, BenchmarkReceipt, ScreeningMethodProvenance,
    ScreeningRecord, ScreeningRun,
};
use symthaea_matbench_folds::{
    build_leakage_clean_test_fold_from_official_bytes, FoldError, FoldIndex,
};
use symthaea_matbench_gap::{
    parse_official_matbench_expt_gap, AdapterError, CompositionFingerprint, MatbenchGapDataset,
};
use thiserror::Error;

pub const BASELINE_METHOD_ID: &str =
    "symthaea-bandgap/electronegativity-composition-baseline";
pub const BASELINE_METHOD_VERSION: &str =
    "symthaea-bandgap.electronegativity-composition-baseline.v0@blob:37aa1e7d399d263bd31573f004e6d7da180c7cb6";
pub const BASELINE_METHOD_SOURCE_PATH: &str =
    "crates/domains/symthaea-bandgap/src/bandgap_baseline.rs";
pub const BASELINE_METHOD_SOURCE_BLOB_SHA1: &str =
    "37aa1e7d399d263bd31573f004e6d7da180c7cb6";

/// This disclosure is deliberately stronger than an empty `training_slices`
/// collection. The baseline has no machine-readable fit slice available to the
/// benchmark protocol, but its coefficients were historically hand-chosen to
/// roughly reproduce familiar semiconductor gaps. Therefore an empty formal
/// training-slice set must not be interpreted as scientific blindness.
pub const BASELINE_METHOD_DISCLOSURE: &str =
    "Composition-only electronegativity baseline. No Matbench truth value is supplied to prediction or ranking. The benchmark method declares no exact machine-readable training slice because the baseline coefficients were historically hand-chosen/fitted against familiar semiconductor behavior; this is not evidence of historical blindness or independence from public semiconductor knowledge.";

pub const CAPABILITY_CLASSIFICATION: &str =
    "MEASUREMENT-ONLY BENCHMARK BASELINE -- not a material certification, novelty claim, experiment authorization, procurement decision, manufacturing approval, deployment decision, or physical authority.";

const COMBINED_RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.baseline-receipt.v0\0";

/// One candidate roster entry presented to the pure composition-only ranking
/// path. Experimental/reference gap values are structurally absent.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineCandidate {
    pub candidate_id: String,
    pub composition: Vec<(u8, f64)>,
}

/// Review artifact binding the leakage qualification and Benchmark Zero result
/// to one fixed, explicitly identified baseline method.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BaselineBenchmarkReceipt {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub fold: u8,
    pub method_id: &'static str,
    pub method_version: &'static str,
    pub method_source_path: &'static str,
    pub method_source_blob_sha1: &'static str,
    pub method_disclosure: &'static str,
    pub fold_derivation_disclosure: &'static str,
    pub index_identity_disclosure: &'static str,
    pub residual_leakage_disclosure: &'static str,
    pub leakage_qualification_sha256: String,
    pub benchmark_receipt_blake3: String,
    pub combined_sha256: String,
    pub benchmark: BenchmarkReceipt,
}

impl BaselineBenchmarkReceipt {
    pub fn to_json_pretty(&self) -> Result<String, BaselineError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Error)]
pub enum BaselineError {
    #[error("baseline candidate roster is empty")]
    EmptyCandidateRoster,
    #[error("duplicate baseline candidate id {0:?}")]
    DuplicateCandidate(String),
    #[error("invalid baseline candidate {candidate_id:?}: {reason}")]
    InvalidCandidate { candidate_id: String, reason: String },
    #[error("leakage-qualified candidate {0:?} is missing from the pinned source artifact")]
    MissingQualifiedCandidate(String),
    #[error(
        "candidate roster size mismatch: leakage qualification retained {expected}, reconstructed {actual}"
    )]
    CandidateRosterSizeMismatch { expected: usize, actual: usize },
    #[error("source artifact identity mismatch across qualification and prediction parsing")]
    SourceArtifactIdentityMismatch,
    #[error(transparent)]
    Fold(#[from] FoldError),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(transparent)]
    Benchmark(#[from] BenchmarkError),
    #[error("baseline receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Build the fixed composition-only baseline ranking without receiving any
/// experimental/reference gap values.
///
/// Ranking semantics are preregistered here:
/// 1. minimum predicted distance to the caller's target interval;
/// 2. minimum predicted distance to target midpoint;
/// 3. lexical candidate id for deterministic tie-breaking.
///
/// No uncertainty is invented for this hand-built baseline.
pub fn build_baseline_screening_run(
    candidates: &[BaselineCandidate],
    target: BandgapTarget,
) -> Result<ScreeningRun, BaselineError> {
    target.validate()?;
    if candidates.is_empty() {
        return Err(BaselineError::EmptyCandidateRoster);
    }

    let mut seen = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        validate_candidate(candidate)?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(BaselineError::DuplicateCandidate(
                candidate.candidate_id.clone(),
            ));
        }

        let predicted_gap_ev = electronegativity_bandgap(&candidate.composition);
        if !predicted_gap_ev.is_finite() || predicted_gap_ev < 0.0 {
            return Err(BaselineError::InvalidCandidate {
                candidate_id: candidate.candidate_id.clone(),
                reason: "baseline prediction must be finite and non-negative".into(),
            });
        }

        ranked.push(ScreeningRecord {
            candidate_id: candidate.candidate_id.clone(),
            predicted_gap_ev,
            uncertainty_ev: None,
        });
    }

    ranked.sort_by(|left, right| {
        let left_window = distance_to_target_window(target, left.predicted_gap_ev);
        let right_window = distance_to_target_window(target, right.predicted_gap_ev);
        left_window
            .total_cmp(&right_window)
            .then_with(|| {
                target
                    .midpoint_error(left.predicted_gap_ev)
                    .total_cmp(&target.midpoint_error(right.predicted_gap_ev))
            })
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let run = ScreeningRun {
        method: ScreeningMethodProvenance {
            method_id: BASELINE_METHOD_ID.to_owned(),
            version: BASELINE_METHOD_VERSION.to_owned(),
            // Intentionally empty: there is no exact machine-readable fit slice
            // to declare for the hand-tuned baseline. See METHOD_DISCLOSURE;
            // empty does not mean independent or historically blind.
            training_slices: BTreeSet::new(),
        },
        target,
        ranked,
    };
    run.validate()?;
    Ok(run)
}

/// Execute one leakage-qualified Benchmark Zero baseline measurement from the
/// exact pinned `matbench_expt_gap` compressed artifact bytes.
///
/// The ranking is frozen before `evaluate` receives experimental truth.
pub fn run_baseline_benchmark_from_official_bytes(
    compressed_bytes: &[u8],
    fold: FoldIndex,
    target: BandgapTarget,
    top_k: usize,
) -> Result<BaselineBenchmarkReceipt, BaselineError> {
    let qualification =
        build_leakage_clean_test_fold_from_official_bytes(compressed_bytes, fold)?;
    let dataset = parse_official_matbench_expt_gap(compressed_bytes)?;

    if !dataset
        .compressed_sha256
        .eq_ignore_ascii_case(&qualification.source_artifact_sha256)
    {
        return Err(BaselineError::SourceArtifactIdentityMismatch);
    }

    // Candidate membership comes from the qualified fold/mask. Gap values are
    // not read by the ranking path; only the retained candidate IDs are used.
    let retained_ids: BTreeSet<String> = qualification
        .truth
        .experimental_gap_ev
        .keys()
        .cloned()
        .collect();
    let candidates = candidate_roster(&dataset, &retained_ids)?;
    if candidates.len() != qualification.retained_test_count {
        return Err(BaselineError::CandidateRosterSizeMismatch {
            expected: qualification.retained_test_count,
            actual: candidates.len(),
        });
    }

    let screening_run = build_baseline_screening_run(&candidates, target)?;
    let benchmark = evaluate(&screening_run, &qualification.truth, top_k)?;
    let benchmark_receipt_blake3 = benchmark.digest()?;
    let combined_sha256 = combined_receipt_sha256(
        fold,
        &qualification.qualification_sha256,
        &benchmark_receipt_blake3,
    );

    Ok(BaselineBenchmarkReceipt {
        schema: "symthaea.energy-benchmark-zero.baseline-receipt.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        fold: fold.value(),
        method_id: BASELINE_METHOD_ID,
        method_version: BASELINE_METHOD_VERSION,
        method_source_path: BASELINE_METHOD_SOURCE_PATH,
        method_source_blob_sha1: BASELINE_METHOD_SOURCE_BLOB_SHA1,
        method_disclosure: BASELINE_METHOD_DISCLOSURE,
        fold_derivation_disclosure: qualification.fold_derivation_disclosure,
        index_identity_disclosure: qualification.index_identity_disclosure,
        residual_leakage_disclosure: qualification.residual_leakage_disclosure,
        leakage_qualification_sha256: qualification.qualification_sha256,
        benchmark_receipt_blake3,
        combined_sha256,
        benchmark,
    })
}

fn candidate_roster(
    dataset: &MatbenchGapDataset,
    retained_ids: &BTreeSet<String>,
) -> Result<Vec<BaselineCandidate>, BaselineError> {
    let mut candidates = Vec::with_capacity(retained_ids.len());
    let mut found = BTreeSet::new();

    for record in &dataset.ordered_records {
        if retained_ids.contains(&record.candidate_id) {
            found.insert(record.candidate_id.clone());
            candidates.push(BaselineCandidate {
                candidate_id: record.candidate_id.clone(),
                composition: composition_from_fingerprint(&record.composition)?,
            });
        }
    }

    for candidate_id in retained_ids {
        if !found.contains(candidate_id) {
            return Err(BaselineError::MissingQualifiedCandidate(
                candidate_id.clone(),
            ));
        }
    }

    Ok(candidates)
}

fn composition_from_fingerprint(
    fingerprint: &CompositionFingerprint,
) -> Result<Vec<(u8, f64)>, BaselineError> {
    if fingerprint.0.is_empty() {
        return Err(BaselineError::InvalidCandidate {
            candidate_id: "<fingerprint>".into(),
            reason: "composition fingerprint cannot be empty".into(),
        });
    }

    let total: u128 = fingerprint.0.iter().map(|(_, amount)| u128::from(*amount)).sum();
    if total == 0 {
        return Err(BaselineError::InvalidCandidate {
            candidate_id: "<fingerprint>".into(),
            reason: "composition fingerprint has zero total amount".into(),
        });
    }
    let total_f64 = total as f64;

    let mut composition = Vec::with_capacity(fingerprint.0.len());
    for &(atomic_number, amount) in &fingerprint.0 {
        if atomic_number == 0 || amount == 0 {
            return Err(BaselineError::InvalidCandidate {
                candidate_id: "<fingerprint>".into(),
                reason: "composition fingerprint contains zero atomic number or amount".into(),
            });
        }
        composition.push((atomic_number, amount as f64 / total_f64));
    }
    Ok(composition)
}

fn validate_candidate(candidate: &BaselineCandidate) -> Result<(), BaselineError> {
    if candidate.candidate_id.trim().is_empty() {
        return Err(BaselineError::InvalidCandidate {
            candidate_id: candidate.candidate_id.clone(),
            reason: "candidate id cannot be empty".into(),
        });
    }
    if candidate.composition.is_empty() {
        return Err(BaselineError::InvalidCandidate {
            candidate_id: candidate.candidate_id.clone(),
            reason: "composition cannot be empty".into(),
        });
    }

    let mut total = 0.0f64;
    let mut elements = BTreeSet::new();
    for &(atomic_number, fraction) in &candidate.composition {
        if atomic_number == 0 || !fraction.is_finite() || fraction <= 0.0 {
            return Err(BaselineError::InvalidCandidate {
                candidate_id: candidate.candidate_id.clone(),
                reason: "composition requires nonzero atomic numbers and finite positive fractions"
                    .into(),
            });
        }
        if !elements.insert(atomic_number) {
            return Err(BaselineError::InvalidCandidate {
                candidate_id: candidate.candidate_id.clone(),
                reason: format!("duplicate atomic number {atomic_number}"),
            });
        }
        total += fraction;
    }

    if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
        return Err(BaselineError::InvalidCandidate {
            candidate_id: candidate.candidate_id.clone(),
            reason: format!("composition fractions must sum to one; got {total}"),
        });
    }
    Ok(())
}

fn distance_to_target_window(target: BandgapTarget, predicted_gap_ev: f64) -> f64 {
    if predicted_gap_ev < target.min_ev {
        target.min_ev - predicted_gap_ev
    } else if predicted_gap_ev > target.max_ev {
        predicted_gap_ev - target.max_ev
    } else {
        0.0
    }
}

fn combined_receipt_sha256(
    fold: FoldIndex,
    qualification_sha256: &str,
    benchmark_receipt_blake3: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(COMBINED_RECEIPT_DIGEST_DOMAIN);
    hasher.update([fold.value()]);
    update_text(&mut hasher, BASELINE_METHOD_ID);
    update_text(&mut hasher, BASELINE_METHOD_VERSION);
    update_text(&mut hasher, BASELINE_METHOD_SOURCE_PATH);
    update_text(&mut hasher, BASELINE_METHOD_SOURCE_BLOB_SHA1);
    update_text(&mut hasher, qualification_sha256);
    update_text(&mut hasher, benchmark_receipt_blake3);
    hex_lower(&hasher.finalize())
}

fn update_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, composition: &[(u8, f64)]) -> BaselineCandidate {
        BaselineCandidate {
            candidate_id: id.into(),
            composition: composition.to_vec(),
        }
    }

    #[test]
    fn fingerprint_conversion_renormalizes_quantized_amounts() {
        let fingerprint = CompositionFingerprint(vec![(31, 500_000_001), (33, 499_999_999)]);
        let composition = composition_from_fingerprint(&fingerprint).unwrap();
        assert!((composition.iter().map(|(_, value)| value).sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(composition[0].0, 31);
        assert_eq!(composition[1].0, 33);
    }

    #[test]
    fn pure_ranking_prefers_target_and_carries_no_fake_uncertainty() {
        let candidates = vec![
            candidate("diamond", &[(6, 1.0)]),
            candidate("silicon", &[(14, 1.0)]),
            candidate("nacl", &[(11, 0.5), (17, 0.5)]),
        ];
        let target = BandgapTarget::new(1.0, 1.3).unwrap();
        let run = build_baseline_screening_run(&candidates, target).unwrap();

        assert_eq!(run.ranked[0].candidate_id, "silicon");
        assert!(run.ranked.iter().all(|record| record.uncertainty_ev.is_none()));
        assert!(run.method.training_slices.is_empty());
    }

    #[test]
    fn equal_predictions_use_candidate_id_only_as_final_tie_break() {
        let candidates = vec![
            candidate("zeta", &[(14, 1.0)]),
            candidate("alpha", &[(14, 1.0)]),
        ];
        let target = BandgapTarget::new(1.0, 1.3).unwrap();
        let run = build_baseline_screening_run(&candidates, target).unwrap();
        assert_eq!(run.ranked[0].candidate_id, "alpha");
        assert_eq!(run.ranked[1].candidate_id, "zeta");
    }

    #[test]
    fn method_identity_and_historical_fit_disclosure_are_explicit() {
        assert!(BASELINE_METHOD_VERSION.contains(BASELINE_METHOD_SOURCE_BLOB_SHA1));
        assert!(BASELINE_METHOD_DISCLOSURE.contains("hand-chosen/fitted"));
        assert!(BASELINE_METHOD_DISCLOSURE.contains("not evidence of historical blindness"));
    }

    #[test]
    fn malformed_rosters_fail_closed() {
        let target = BandgapTarget::new(1.0, 1.3).unwrap();
        assert!(build_baseline_screening_run(&[], target).is_err());

        let duplicates = vec![
            candidate("same", &[(14, 1.0)]),
            candidate("same", &[(6, 1.0)]),
        ];
        assert!(matches!(
            build_baseline_screening_run(&duplicates, target),
            Err(BaselineError::DuplicateCandidate(id)) if id == "same"
        ));
    }
}
