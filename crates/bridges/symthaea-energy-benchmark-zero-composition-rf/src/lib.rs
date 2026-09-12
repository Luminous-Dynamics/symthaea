// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leakage-qualified composition-only RF adapter for Energy Benchmark Zero.
//!
//! The ranking path receives only candidate ids/compositions plus an explicit
//! target interval. Experimental Matbench gaps remain outside the ranking type
//! until the existing measurement-only benchmark evaluator is invoked.

#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use symthaea_bandgap::ml_bandgap::CompositionBandgapPredictor;
use symthaea_energy_benchmark_zero::{
    evaluate, BandgapTarget, BenchmarkError, BenchmarkReceipt, DataSliceId,
    ScreeningMethodProvenance, ScreeningRecord, ScreeningRun,
};
use symthaea_matbench_folds::{
    build_leakage_clean_test_fold_from_official_bytes, symthaea_training_table_sha256, FoldError,
    FoldIndex,
};
use symthaea_matbench_gap::{
    parse_official_matbench_expt_gap, AdapterError, CompositionFingerprint, MatbenchGapDataset,
};
use thiserror::Error;

pub const COMPOSITION_RF_METHOD_ID: &str =
    "symthaea-bandgap/crystal-ablated-composition-rf";
pub const COMPOSITION_RF_MODEL_SOURCE_PATH: &str =
    "crates/domains/symthaea-bandgap/src/ml_bandgap.rs";
pub const COMPOSITION_RF_MODEL_SOURCE_BLOB_SHA1: &str =
    "9f5d0e5dc81e346c7558dca0a29e9e05925d88f9";
pub const CRYSTAL_ABLATION_POLICY: &str =
    "feature[16] is CrystalSystem::Unknown for every training and inference row; the constant sentinel carries no discriminative crystal information";
pub const UNCERTAINTY_DISCLOSURE: &str =
    "The underlying RF returns inter-tree standard deviation, but no calibration theorem currently establishes it as predictive uncertainty. Benchmark ScreeningRecord.uncertainty_ev is therefore None.";
pub const TRAINING_DISCLOSURE: &str =
    "The model trains on Symthaea's curated experimental band-gap table. The exact table identity is declared as a training slice and exact normalized-composition overlaps are removed from external evaluation by the upstream leakage qualification. This does not establish historical blindness or source-family independence from public semiconductor literature.";
pub const CAPABILITY_CLASSIFICATION: &str =
    "MEASUREMENT-ONLY LEARNED-MODEL BENCHMARK -- not calibrated uncertainty, material certification, novelty, experiment authorization, procurement, manufacturing, deployment, or physical authority.";

const COMBINED_RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.composition-rf-receipt.v0\0";

#[derive(Debug, Clone, PartialEq)]
pub struct CompositionRfCandidate {
    pub candidate_id: String,
    pub composition: Vec<(u8, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompositionRfBenchmarkReceipt {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub fold: u8,
    pub method_id: &'static str,
    pub method_version: String,
    pub model_source_path: &'static str,
    pub model_source_blob_sha1: &'static str,
    pub crystal_ablation_policy: &'static str,
    pub training_table_sha256: String,
    pub training_entry_count: usize,
    pub training_disclosure: &'static str,
    pub uncertainty_disclosure: &'static str,
    pub leakage_qualification_sha256: String,
    pub benchmark_receipt_blake3: String,
    pub combined_sha256: String,
    pub benchmark: BenchmarkReceipt,
}

impl CompositionRfBenchmarkReceipt {
    pub fn to_json_pretty(&self) -> Result<String, CompositionRfError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Error)]
pub enum CompositionRfError {
    #[error("composition-RF candidate roster is empty")]
    EmptyCandidateRoster,
    #[error("duplicate composition-RF candidate id {0:?}")]
    DuplicateCandidate(String),
    #[error("invalid composition-RF candidate {candidate_id:?}: {reason}")]
    InvalidCandidate { candidate_id: String, reason: String },
    #[error("leakage-qualified candidate {0:?} is missing from the pinned source artifact")]
    MissingQualifiedCandidate(String),
    #[error(
        "candidate roster size mismatch: leakage qualification retained {expected}, reconstructed {actual}"
    )]
    CandidateRosterSizeMismatch { expected: usize, actual: usize },
    #[error("source artifact identity mismatch across qualification and prediction parsing")]
    SourceArtifactIdentityMismatch,
    #[error("training table identity mismatch: qualification {qualified}, runtime {runtime}")]
    TrainingTableIdentityMismatch { qualified: String, runtime: String },
    #[error(transparent)]
    Fold(#[from] FoldError),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(transparent)]
    Benchmark(#[from] BenchmarkError),
    #[error("composition-RF receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub fn composition_rf_method_version(training_table_sha256: &str) -> String {
    format!(
        "symthaea-bandgap.crystal-ablated-composition-rf.v0@blob:{}@trainsha256:{}",
        COMPOSITION_RF_MODEL_SOURCE_BLOB_SHA1, training_table_sha256
    )
}

pub fn composition_rf_training_slice(
    training_table_sha256: &str,
) -> Result<DataSliceId, CompositionRfError> {
    Ok(DataSliceId::new(
        "symthaea-bandgap-curated-training",
        format!("full@sha256:{training_table_sha256}"),
    )?)
}

/// Build a deterministic composition-only RF ranking without receiving external
/// experimental truth.
///
/// Ranking semantics intentionally match the fixed physics baseline:
/// 1. minimum predicted distance to the target interval;
/// 2. minimum predicted distance to target midpoint;
/// 3. lexical candidate id as the final deterministic tie-break.
pub fn build_composition_rf_screening_run(
    candidates: &[CompositionRfCandidate],
    target: BandgapTarget,
    training_table_sha256: &str,
) -> Result<ScreeningRun, CompositionRfError> {
    target.validate()?;
    validate_sha256(training_table_sha256, "training table SHA-256")?;
    if candidates.is_empty() {
        return Err(CompositionRfError::EmptyCandidateRoster);
    }

    let predictor = CompositionBandgapPredictor::new();
    let mut seen = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        validate_candidate(candidate)?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(CompositionRfError::DuplicateCandidate(
                candidate.candidate_id.clone(),
            ));
        }

        let prediction = predictor.predict(&candidate.composition);
        if !prediction.bandgap.is_finite() || prediction.bandgap < 0.0 {
            return Err(CompositionRfError::InvalidCandidate {
                candidate_id: candidate.candidate_id.clone(),
                reason: "RF prediction must be finite and non-negative".into(),
            });
        }

        // Inter-tree spread is intentionally not promoted to calibrated
        // predictive uncertainty. See UNCERTAINTY_DISCLOSURE.
        ranked.push(ScreeningRecord {
            candidate_id: candidate.candidate_id.clone(),
            predicted_gap_ev: prediction.bandgap,
            uncertainty_ev: None,
        });
    }

    ranked.sort_by(|left, right| {
        distance_to_target_window(target, left.predicted_gap_ev)
            .total_cmp(&distance_to_target_window(target, right.predicted_gap_ev))
            .then_with(|| {
                target
                    .midpoint_error(left.predicted_gap_ev)
                    .total_cmp(&target.midpoint_error(right.predicted_gap_ev))
            })
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let run = ScreeningRun {
        method: ScreeningMethodProvenance {
            method_id: COMPOSITION_RF_METHOD_ID.to_owned(),
            version: composition_rf_method_version(training_table_sha256),
            training_slices: BTreeSet::from([composition_rf_training_slice(
                training_table_sha256,
            )?]),
        },
        target,
        ranked,
    };
    run.validate()?;
    Ok(run)
}

pub fn run_composition_rf_benchmark_from_official_bytes(
    compressed_bytes: &[u8],
    fold: FoldIndex,
    target: BandgapTarget,
    top_k: usize,
) -> Result<CompositionRfBenchmarkReceipt, CompositionRfError> {
    let qualification =
        build_leakage_clean_test_fold_from_official_bytes(compressed_bytes, fold)?;
    let dataset = parse_official_matbench_expt_gap(compressed_bytes)?;

    if !dataset
        .compressed_sha256
        .eq_ignore_ascii_case(&qualification.source_artifact_sha256)
    {
        return Err(CompositionRfError::SourceArtifactIdentityMismatch);
    }

    let runtime_training_sha256 = symthaea_training_table_sha256();
    if runtime_training_sha256 != qualification.symthaea_training_table_sha256 {
        return Err(CompositionRfError::TrainingTableIdentityMismatch {
            qualified: qualification.symthaea_training_table_sha256,
            runtime: runtime_training_sha256,
        });
    }

    let retained_ids: BTreeSet<String> = qualification
        .truth
        .experimental_gap_ev
        .keys()
        .cloned()
        .collect();
    let candidates = candidate_roster(&dataset, &retained_ids)?;
    if candidates.len() != qualification.retained_test_count {
        return Err(CompositionRfError::CandidateRosterSizeMismatch {
            expected: qualification.retained_test_count,
            actual: candidates.len(),
        });
    }

    let method_version = composition_rf_method_version(&runtime_training_sha256);
    let screening_run =
        build_composition_rf_screening_run(&candidates, target, &runtime_training_sha256)?;
    if screening_run.method.version != method_version {
        return Err(CompositionRfError::InvalidCandidate {
            candidate_id: "<method>".into(),
            reason: "method-version reconstruction mismatch".into(),
        });
    }

    let benchmark = evaluate(&screening_run, &qualification.truth, top_k)?;
    let benchmark_receipt_blake3 = benchmark.digest()?;
    let combined_sha256 = combined_receipt_sha256(
        fold,
        &method_version,
        &runtime_training_sha256,
        &qualification.qualification_sha256,
        &benchmark_receipt_blake3,
    );

    Ok(CompositionRfBenchmarkReceipt {
        schema: "symthaea.energy-benchmark-zero.composition-rf-receipt.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        fold: fold.value(),
        method_id: COMPOSITION_RF_METHOD_ID,
        method_version,
        model_source_path: COMPOSITION_RF_MODEL_SOURCE_PATH,
        model_source_blob_sha1: COMPOSITION_RF_MODEL_SOURCE_BLOB_SHA1,
        crystal_ablation_policy: CRYSTAL_ABLATION_POLICY,
        training_table_sha256: runtime_training_sha256,
        training_entry_count: qualification.symthaea_training_entry_count,
        training_disclosure: TRAINING_DISCLOSURE,
        uncertainty_disclosure: UNCERTAINTY_DISCLOSURE,
        leakage_qualification_sha256: qualification.qualification_sha256,
        benchmark_receipt_blake3,
        combined_sha256,
        benchmark,
    })
}

fn candidate_roster(
    dataset: &MatbenchGapDataset,
    retained_ids: &BTreeSet<String>,
) -> Result<Vec<CompositionRfCandidate>, CompositionRfError> {
    let mut candidates = Vec::with_capacity(retained_ids.len());
    let mut found = BTreeSet::new();

    for record in &dataset.ordered_records {
        if retained_ids.contains(&record.candidate_id) {
            found.insert(record.candidate_id.clone());
            candidates.push(CompositionRfCandidate {
                candidate_id: record.candidate_id.clone(),
                composition: composition_from_fingerprint(&record.composition)?,
            });
        }
    }

    for candidate_id in retained_ids {
        if !found.contains(candidate_id) {
            return Err(CompositionRfError::MissingQualifiedCandidate(
                candidate_id.clone(),
            ));
        }
    }
    Ok(candidates)
}

fn composition_from_fingerprint(
    fingerprint: &CompositionFingerprint,
) -> Result<Vec<(u8, f64)>, CompositionRfError> {
    if fingerprint.0.is_empty() {
        return Err(CompositionRfError::InvalidCandidate {
            candidate_id: "<fingerprint>".into(),
            reason: "composition fingerprint cannot be empty".into(),
        });
    }
    let total: u128 = fingerprint.0.iter().map(|(_, amount)| u128::from(*amount)).sum();
    if total == 0 {
        return Err(CompositionRfError::InvalidCandidate {
            candidate_id: "<fingerprint>".into(),
            reason: "composition fingerprint has zero total amount".into(),
        });
    }
    let total_f64 = total as f64;
    Ok(fingerprint
        .0
        .iter()
        .map(|&(atomic_number, amount)| (atomic_number, amount as f64 / total_f64))
        .collect())
}

fn validate_candidate(candidate: &CompositionRfCandidate) -> Result<(), CompositionRfError> {
    if candidate.candidate_id.trim().is_empty() || candidate.composition.is_empty() {
        return Err(CompositionRfError::InvalidCandidate {
            candidate_id: candidate.candidate_id.clone(),
            reason: "candidate id and composition must be non-empty".into(),
        });
    }

    let mut total = 0.0f64;
    let mut elements = BTreeSet::new();
    for &(atomic_number, fraction) in &candidate.composition {
        if atomic_number == 0 || !fraction.is_finite() || fraction <= 0.0 {
            return Err(CompositionRfError::InvalidCandidate {
                candidate_id: candidate.candidate_id.clone(),
                reason: "composition requires nonzero atomic numbers and finite positive fractions"
                    .into(),
            });
        }
        if !elements.insert(atomic_number) {
            return Err(CompositionRfError::InvalidCandidate {
                candidate_id: candidate.candidate_id.clone(),
                reason: format!("duplicate atomic number {atomic_number}"),
            });
        }
        total += fraction;
    }
    if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
        return Err(CompositionRfError::InvalidCandidate {
            candidate_id: candidate.candidate_id.clone(),
            reason: format!("composition fractions must sum to one; got {total}"),
        });
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<(), CompositionRfError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CompositionRfError::InvalidCandidate {
            candidate_id: "<method>".into(),
            reason: format!("{name} must be exactly 64 hexadecimal characters"),
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
    method_version: &str,
    training_table_sha256: &str,
    qualification_sha256: &str,
    benchmark_receipt_blake3: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(COMBINED_RECEIPT_DIGEST_DOMAIN);
    hasher.update([fold.value()]);
    update_text(&mut hasher, COMPOSITION_RF_METHOD_ID);
    update_text(&mut hasher, method_version);
    update_text(&mut hasher, COMPOSITION_RF_MODEL_SOURCE_PATH);
    update_text(&mut hasher, COMPOSITION_RF_MODEL_SOURCE_BLOB_SHA1);
    update_text(&mut hasher, CRYSTAL_ABLATION_POLICY);
    update_text(&mut hasher, training_table_sha256);
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

    fn candidate(id: &str, composition: &[(u8, f64)]) -> CompositionRfCandidate {
        CompositionRfCandidate {
            candidate_id: id.into(),
            composition: composition.to_vec(),
        }
    }

    #[test]
    fn learned_ranking_is_deterministic_and_has_no_calibrated_uncertainty_claim() {
        let candidates = vec![
            candidate("si", &[(14, 1.0)]),
            candidate("gaas", &[(31, 0.5), (33, 0.5)]),
            candidate("zno", &[(30, 0.5), (8, 0.5)]),
        ];
        let target = BandgapTarget::new(1.0, 1.6).unwrap();
        let training_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let first = build_composition_rf_screening_run(&candidates, target, training_sha).unwrap();
        let second = build_composition_rf_screening_run(&candidates, target, training_sha).unwrap();
        assert_eq!(first.ranking_digest().unwrap(), second.ranking_digest().unwrap());
        assert!(first.ranked.iter().all(|record| record.uncertainty_ev.is_none()));
    }

    #[test]
    fn exact_training_slice_and_model_identity_are_declared() {
        let candidates = vec![candidate("si", &[(14, 1.0)])];
        let target = BandgapTarget::new(1.0, 1.6).unwrap();
        let training_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let run = build_composition_rf_screening_run(&candidates, target, training_sha).unwrap();
        assert_eq!(run.method.training_slices.len(), 1);
        assert!(run.method.version.contains(COMPOSITION_RF_MODEL_SOURCE_BLOB_SHA1));
        assert!(run.method.version.contains(training_sha));
    }

    #[test]
    fn malformed_candidates_and_training_identity_fail_closed() {
        let target = BandgapTarget::new(1.0, 1.6).unwrap();
        let bad_sha = "not-a-sha";
        assert!(build_composition_rf_screening_run(
            &[candidate("si", &[(14, 1.0)])],
            target,
            bad_sha,
        )
        .is_err());

        let duplicates = vec![
            candidate("same", &[(14, 1.0)]),
            candidate("same", &[(6, 1.0)]),
        ];
        let good_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(matches!(
            build_composition_rf_screening_run(&duplicates, target, good_sha),
            Err(CompositionRfError::DuplicateCandidate(id)) if id == "same"
        ));
    }

    #[test]
    fn uncertainty_and_training_disclosures_are_explicit() {
        assert!(UNCERTAINTY_DISCLOSURE.contains("not"));
        assert!(UNCERTAINTY_DISCLOSURE.contains("calibration"));
        assert!(TRAINING_DISCLOSURE.contains("curated experimental"));
        assert!(TRAINING_DISCLOSURE.contains("does not establish historical blindness"));
    }
}
