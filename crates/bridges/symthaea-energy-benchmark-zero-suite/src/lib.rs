// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Five-fold evidence suite for Energy Discovery Benchmark Zero.
//!
//! The Matbench folds are treated as disjoint evaluation partitions of one
//! dataset, not as five independent scientific replications. The suite binds all
//! fold plans, leakage qualifications, baseline/control receipts, candidate
//! partition identity, and aggregate metrics into one review artifact.

#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_energy_benchmark_zero::BandgapTarget;
use symthaea_energy_benchmark_zero_baseline::BASELINE_METHOD_VERSION;
use symthaea_energy_benchmark_zero_controls::{
    run_controlled_benchmark_from_official_bytes, BenchmarkPlan, ControlError,
    ControlledBenchmarkReceipt, BLIND_CONTROL_METHOD_VERSION, BLIND_CONTROL_REPLICATES,
};
use symthaea_matbench_folds::{
    build_leakage_clean_test_fold_from_official_bytes, canonical_fold_counts, FoldError,
    FoldIndex,
};
use thiserror::Error;

pub const PARTITION_DISCLOSURE: &str =
    "The five Matbench folds are disjoint evaluation partitions of one source dataset. They are not five independent physical experiments or independent scientific replications, and fold-level variation must not be converted into a replication-count or significance claim.";

pub const CAPABILITY_CLASSIFICATION: &str =
    "MEASUREMENT-ONLY FIVE-FOLD BENCHMARK SUITE -- not independent replication, statistical significance, material certification, novelty, experiment authorization, procurement, manufacturing, deployment, or physical authority.";

const SUITE_PLAN_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-benchmark-zero.suite-plan.v0\0";
const CANDIDATE_PARTITION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.candidate-partition.v0\0";
const AGGREGATE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.suite-aggregate.v0\0";
const SUITE_RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.suite-receipt.v0\0";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkSuitePlan {
    pub schema: &'static str,
    pub target: BandgapTarget,
    pub top_k: usize,
    pub baseline_method_version: &'static str,
    pub blind_control_method_version: &'static str,
    pub blind_control_replicates: u16,
    pub fold_plan_sha256: Vec<String>,
}

impl BenchmarkSuitePlan {
    pub fn new(target: BandgapTarget, top_k: usize) -> Result<Self, SuiteError> {
        target.validate().map_err(ControlError::from)?;
        if top_k == 0 {
            return Err(SuiteError::InvalidPlan("top-k must be positive".into()));
        }

        let mut fold_plan_sha256 = Vec::with_capacity(5);
        for fold in 0u8..5 {
            let plan = BenchmarkPlan::new(FoldIndex::new(fold)?, target, top_k)?;
            fold_plan_sha256.push(plan.sha256()?);
        }

        let plan = Self {
            schema: "symthaea.energy-benchmark-zero.suite-plan.v0",
            target,
            top_k,
            baseline_method_version: BASELINE_METHOD_VERSION,
            blind_control_method_version: BLIND_CONTROL_METHOD_VERSION,
            blind_control_replicates: BLIND_CONTROL_REPLICATES,
            fold_plan_sha256,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), SuiteError> {
        self.target.validate().map_err(ControlError::from)?;
        if self.top_k == 0 {
            return Err(SuiteError::InvalidPlan("top-k must be positive".into()));
        }
        if self.baseline_method_version != BASELINE_METHOD_VERSION
            || self.blind_control_method_version != BLIND_CONTROL_METHOD_VERSION
            || self.blind_control_replicates != BLIND_CONTROL_REPLICATES
            || self.fold_plan_sha256.len() != 5
        {
            return Err(SuiteError::InvalidPlan(
                "suite method/control identity or fold-plan count does not match executable contract"
                    .into(),
            ));
        }

        for fold in 0u8..5 {
            let expected = BenchmarkPlan::new(FoldIndex::new(fold)?, self.target, self.top_k)?
                .sha256()?;
            if self.fold_plan_sha256[usize::from(fold)] != expected {
                return Err(SuiteError::InvalidPlan(format!(
                    "fold {fold} plan digest does not match suite policy"
                )));
            }
        }
        Ok(())
    }

    pub fn fold_plan(&self, fold: u8) -> Result<BenchmarkPlan, SuiteError> {
        self.validate()?;
        let plan = BenchmarkPlan::new(FoldIndex::new(fold)?, self.target, self.top_k)?;
        let digest = plan.sha256()?;
        if self.fold_plan_sha256[usize::from(fold)] != digest {
            return Err(SuiteError::InvalidPlan(format!(
                "fold {fold} plan digest mismatch"
            )));
        }
        Ok(plan)
    }

    pub fn sha256(&self) -> Result<String, SuiteError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(SUITE_PLAN_DIGEST_DOMAIN);
        hasher.update(self.target.min_ev.to_bits().to_le_bytes());
        hasher.update(self.target.max_ev.to_bits().to_le_bytes());
        hasher.update((self.top_k as u64).to_le_bytes());
        update_text(&mut hasher, self.baseline_method_version);
        update_text(&mut hasher, self.blind_control_method_version);
        hasher.update(self.blind_control_replicates.to_le_bytes());
        for digest in &self.fold_plan_sha256 {
            update_text(&mut hasher, digest);
        }
        Ok(hex_lower(&hasher.finalize()))
    }

    pub fn to_json_pretty(&self) -> Result<String, SuiteError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SuiteMetrics {
    pub fold_count: usize,
    pub retained_candidate_count: usize,
    pub excluded_training_overlap_count: usize,
    pub qualifying_truth_count: usize,
    pub total_top_k_slots: usize,
    pub baseline_top_k_hits: usize,
    pub baseline_micro_top_k_precision: f64,
    pub baseline_micro_top_k_recall: f64,
    pub baseline_weighted_mae_ev: f64,
    pub baseline_macro_target_regret_ev: f64,
    pub baseline_max_target_regret_ev: f64,
    pub blind_expected_top_k_hits: f64,
    pub blind_expected_micro_top_k_precision: f64,
    pub baseline_hit_lift_over_blind_mean: f64,
    pub baseline_precision_lift_over_blind_mean: f64,
    pub mean_fraction_controls_hits_le_baseline: f64,
    pub mean_fraction_controls_regret_ge_baseline: f64,
    pub folds_baseline_hits_ge_blind_mean: usize,
    pub folds_baseline_regret_le_blind_median: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkSuiteReceipt {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub partition_disclosure: &'static str,
    pub suite_plan: BenchmarkSuitePlan,
    pub suite_plan_sha256: String,
    pub registration_evidence_ref: Option<String>,
    pub candidate_partition_sha256: String,
    pub aggregate_sha256: String,
    pub metrics: SuiteMetrics,
    pub fold_receipts: Vec<ControlledBenchmarkReceipt>,
    pub suite_sha256: String,
}

impl BenchmarkSuiteReceipt {
    pub fn to_json_pretty(&self) -> Result<String, SuiteError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Error)]
pub enum SuiteError {
    #[error("invalid five-fold suite plan: {0}")]
    InvalidPlan(String),
    #[error("registration evidence reference cannot be blank")]
    BlankRegistrationEvidenceRef,
    #[error("candidate {0:?} appears in more than one cleaned fold")]
    DuplicateCandidateAcrossFolds(String),
    #[error(
        "canonical fold population mismatch: expected {expected} total rows, observed {actual}"
    )]
    CanonicalPopulationMismatch { expected: usize, actual: usize },
    #[error("fold {fold} leakage qualification identity differs from controlled receipt")]
    QualificationMismatch { fold: u8 },
    #[error(transparent)]
    Fold(#[from] FoldError),
    #[error(transparent)]
    Control(#[from] ControlError),
    #[error("suite receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub fn run_five_fold_suite_from_official_bytes(
    compressed_bytes: &[u8],
    suite_plan: BenchmarkSuitePlan,
    registration_evidence_ref: Option<String>,
) -> Result<BenchmarkSuiteReceipt, SuiteError> {
    suite_plan.validate()?;
    let registration_evidence_ref = normalize_registration_ref(registration_evidence_ref)?;
    let suite_plan_sha256 = suite_plan.sha256()?;

    let mut seen_candidates = BTreeSet::new();
    let mut partition_hasher = Sha256::new();
    partition_hasher.update(CANDIDATE_PARTITION_DIGEST_DOMAIN);
    update_text(&mut partition_hasher, &suite_plan_sha256);

    let expected_fold_counts = canonical_fold_counts();
    let expected_canonical_total: usize = expected_fold_counts.iter().sum();
    let mut observed_canonical_total = 0usize;
    let mut total_retained_candidates = 0usize;
    let mut total_exclusions = 0usize;
    let mut receipts = Vec::with_capacity(5);

    for fold_value in 0u8..5 {
        let fold = FoldIndex::new(fold_value)?;
        let qualification =
            build_leakage_clean_test_fold_from_official_bytes(compressed_bytes, fold)?;
        if qualification.canonical_test_count != expected_fold_counts[usize::from(fold_value)] {
            return Err(SuiteError::CanonicalPopulationMismatch {
                expected: expected_fold_counts[usize::from(fold_value)],
                actual: qualification.canonical_test_count,
            });
        }
        observed_canonical_total += qualification.canonical_test_count;
        total_retained_candidates += qualification.retained_test_count;
        total_exclusions += qualification.excluded_training_overlap.len();

        partition_hasher.update([fold_value]);
        update_text(&mut partition_hasher, &qualification.qualification_sha256);
        partition_hasher.update((qualification.retained_test_count as u64).to_le_bytes());
        for candidate_id in qualification.truth.experimental_gap_ev.keys() {
            if !seen_candidates.insert(candidate_id.clone()) {
                return Err(SuiteError::DuplicateCandidateAcrossFolds(
                    candidate_id.clone(),
                ));
            }
            update_text(&mut partition_hasher, candidate_id);
        }

        let controlled = run_controlled_benchmark_from_official_bytes(
            compressed_bytes,
            suite_plan.fold_plan(fold_value)?,
            registration_evidence_ref.clone(),
        )?;
        if controlled.leakage_qualification_sha256 != qualification.qualification_sha256 {
            return Err(SuiteError::QualificationMismatch { fold: fold_value });
        }
        receipts.push(controlled);
    }

    if observed_canonical_total != expected_canonical_total {
        return Err(SuiteError::CanonicalPopulationMismatch {
            expected: expected_canonical_total,
            actual: observed_canonical_total,
        });
    }

    let candidate_partition_sha256 = hex_lower(&partition_hasher.finalize());
    let metrics = aggregate_metrics(
        &receipts,
        total_retained_candidates,
        total_exclusions,
        suite_plan.top_k,
    );
    let aggregate_sha256 = serialized_sha256(AGGREGATE_DIGEST_DOMAIN, &metrics)?;
    let suite_sha256 = suite_receipt_sha256(
        &suite_plan_sha256,
        registration_evidence_ref.as_deref(),
        &candidate_partition_sha256,
        &aggregate_sha256,
        &receipts,
    );

    Ok(BenchmarkSuiteReceipt {
        schema: "symthaea.energy-benchmark-zero.suite-receipt.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        partition_disclosure: PARTITION_DISCLOSURE,
        suite_plan,
        suite_plan_sha256,
        registration_evidence_ref,
        candidate_partition_sha256,
        aggregate_sha256,
        metrics,
        fold_receipts: receipts,
        suite_sha256,
    })
}

fn aggregate_metrics(
    receipts: &[ControlledBenchmarkReceipt],
    retained_candidate_count: usize,
    excluded_training_overlap_count: usize,
    top_k: usize,
) -> SuiteMetrics {
    debug_assert_eq!(receipts.len(), 5);

    let fold_count = receipts.len();
    let baseline_top_k_hits: usize = receipts
        .iter()
        .map(|receipt| receipt.baseline.benchmark.metrics.top_k_hits)
        .sum();
    let qualifying_truth_count: usize = receipts
        .iter()
        .map(|receipt| receipt.baseline.benchmark.metrics.qualifying_truth_count)
        .sum();
    let total_top_k_slots = fold_count * top_k;

    let weighted_mae_numerator: f64 = receipts
        .iter()
        .map(|receipt| {
            let metrics = &receipt.baseline.benchmark.metrics;
            metrics.mean_abs_prediction_error_ev * metrics.ranked_candidate_count as f64
        })
        .sum();
    let baseline_weighted_mae_ev = weighted_mae_numerator / retained_candidate_count as f64;
    let baseline_macro_target_regret_ev = receipts
        .iter()
        .map(|receipt| receipt.baseline.benchmark.metrics.target_regret_ev)
        .sum::<f64>()
        / fold_count as f64;
    let baseline_max_target_regret_ev = receipts
        .iter()
        .map(|receipt| receipt.baseline.benchmark.metrics.target_regret_ev)
        .fold(f64::NEG_INFINITY, f64::max);

    let blind_expected_top_k_hits: f64 = receipts
        .iter()
        .map(|receipt| receipt.blind_control.mean_top_k_hits)
        .sum();
    let blind_expected_micro_top_k_precision =
        blind_expected_top_k_hits / total_top_k_slots as f64;
    let baseline_micro_top_k_precision = baseline_top_k_hits as f64 / total_top_k_slots as f64;
    let baseline_micro_top_k_recall = baseline_top_k_hits as f64 / qualifying_truth_count as f64;

    let mean_fraction_controls_hits_le_baseline = receipts
        .iter()
        .map(|receipt| receipt.blind_control.fraction_controls_hits_le_baseline)
        .sum::<f64>()
        / fold_count as f64;
    let mean_fraction_controls_regret_ge_baseline = receipts
        .iter()
        .map(|receipt| receipt.blind_control.fraction_controls_regret_ge_baseline)
        .sum::<f64>()
        / fold_count as f64;
    let folds_baseline_hits_ge_blind_mean = receipts
        .iter()
        .filter(|receipt| {
            receipt.baseline.benchmark.metrics.top_k_hits as f64
                >= receipt.blind_control.mean_top_k_hits
        })
        .count();
    let folds_baseline_regret_le_blind_median = receipts
        .iter()
        .filter(|receipt| {
            receipt.baseline.benchmark.metrics.target_regret_ev
                <= receipt.blind_control.median_target_regret_ev
        })
        .count();

    SuiteMetrics {
        fold_count,
        retained_candidate_count,
        excluded_training_overlap_count,
        qualifying_truth_count,
        total_top_k_slots,
        baseline_top_k_hits,
        baseline_micro_top_k_precision,
        baseline_micro_top_k_recall,
        baseline_weighted_mae_ev,
        baseline_macro_target_regret_ev,
        baseline_max_target_regret_ev,
        blind_expected_top_k_hits,
        blind_expected_micro_top_k_precision,
        baseline_hit_lift_over_blind_mean: baseline_top_k_hits as f64
            - blind_expected_top_k_hits,
        baseline_precision_lift_over_blind_mean: baseline_micro_top_k_precision
            - blind_expected_micro_top_k_precision,
        mean_fraction_controls_hits_le_baseline,
        mean_fraction_controls_regret_ge_baseline,
        folds_baseline_hits_ge_blind_mean,
        folds_baseline_regret_le_blind_median,
    }
}

fn normalize_registration_ref(value: Option<String>) -> Result<Option<String>, SuiteError> {
    match value {
        Some(value) if value.trim().is_empty() => Err(SuiteError::BlankRegistrationEvidenceRef),
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn serialized_sha256<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<String, SuiteError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    Ok(hex_lower(&hasher.finalize()))
}

fn suite_receipt_sha256(
    suite_plan_sha256: &str,
    registration_evidence_ref: Option<&str>,
    candidate_partition_sha256: &str,
    aggregate_sha256: &str,
    receipts: &[ControlledBenchmarkReceipt],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SUITE_RECEIPT_DIGEST_DOMAIN);
    update_text(&mut hasher, suite_plan_sha256);
    match registration_evidence_ref {
        Some(value) => {
            hasher.update([1]);
            update_text(&mut hasher, value);
        }
        None => hasher.update([0]),
    }
    update_text(&mut hasher, candidate_partition_sha256);
    update_text(&mut hasher, aggregate_sha256);
    for receipt in receipts {
        update_text(&mut hasher, &receipt.comparison_sha256);
    }
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

    #[test]
    fn suite_plan_binds_all_five_fold_plans() {
        let plan = BenchmarkSuitePlan::new(BandgapTarget::new(1.0, 1.5).unwrap(), 20).unwrap();
        assert_eq!(plan.fold_plan_sha256.len(), 5);
        let unique: BTreeSet<&String> = plan.fold_plan_sha256.iter().collect();
        assert_eq!(unique.len(), 5);
        plan.validate().unwrap();
    }

    #[test]
    fn suite_plan_digest_changes_with_policy() {
        let a = BenchmarkSuitePlan::new(BandgapTarget::new(1.0, 1.5).unwrap(), 20).unwrap();
        let b = BenchmarkSuitePlan::new(BandgapTarget::new(1.1, 1.5).unwrap(), 20).unwrap();
        let c = BenchmarkSuitePlan::new(BandgapTarget::new(1.0, 1.5).unwrap(), 25).unwrap();
        assert_ne!(a.sha256().unwrap(), b.sha256().unwrap());
        assert_ne!(a.sha256().unwrap(), c.sha256().unwrap());
    }

    #[test]
    fn fold_partition_is_explicitly_not_replication() {
        assert!(PARTITION_DISCLOSURE.contains("not as five independent"));
        assert!(PARTITION_DISCLOSURE.contains("not be converted into a replication-count"));
    }

    #[test]
    fn blank_registration_reference_is_rejected() {
        assert!(normalize_registration_ref(Some("  ".into())).is_err());
        assert_eq!(normalize_registration_ref(None).unwrap(), None);
    }
}
