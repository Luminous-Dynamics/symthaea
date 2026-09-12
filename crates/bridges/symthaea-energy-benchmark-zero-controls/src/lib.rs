// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistrable, property-blind controls for Energy Discovery Benchmark Zero.
//!
//! This crate compares the fixed composition-only physics baseline to a frozen
//! ensemble of deterministic SHA-256 candidate permutations. Blind rankings use
//! candidate identity only. In this benchmark those ids are composition-derived,
//! but the control treats them as opaque bytes and never parses element/fraction
//! structure, predicted property, target, or truth into ranking construction.
//!
//! The resulting fractions are deterministic reference-ensemble comparisons,
//! not p-values, confidence levels, or claims that SHA-256 replicates are
//! independent physical experiments.

#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_energy_benchmark_zero::{
    BandgapTarget, BandgapTruthSet, BenchmarkError, BenchmarkMetrics,
};
use symthaea_energy_benchmark_zero_baseline::{
    run_baseline_benchmark_from_official_bytes, BaselineBenchmarkReceipt, BaselineError,
    BASELINE_METHOD_VERSION,
};
use symthaea_matbench_folds::{
    build_leakage_clean_test_fold_from_official_bytes, FoldError, FoldIndex,
};
use thiserror::Error;

pub const BLIND_CONTROL_METHOD_ID: &str =
    "symthaea-energy-benchmark-zero/property-blind-sha256-permutation";
pub const BLIND_CONTROL_METHOD_VERSION: &str =
    "property-blind-sha256-permutation.v0/256-replicates";
pub const BLIND_CONTROL_REPLICATES: u16 = 256;

pub const BLIND_CONTROL_DISCLOSURE: &str =
    "The control ensemble is a fixed deterministic set of SHA-256 permutations over candidate ids. Those ids are composition-derived in this benchmark, but the control treats each id as an opaque byte string and extracts no parsed composition or material-property features. Ranking receives no target, prediction, or truth value. Fractions against the 256 controls are descriptive reference-ensemble comparisons, not p-values, confidence intervals, or proof of random/independent sampling.";

pub const PLAN_CHRONOLOGY_DISCLOSURE: &str =
    "The benchmark-plan SHA-256 is a content identity, not proof that the plan existed before results were observed. A chronology claim requires a separately immutable/timestamped registration evidence reference created before execution.";

pub const CAPABILITY_CLASSIFICATION: &str =
    "MEASUREMENT-ONLY BENCHMARK CONTROL -- not statistical significance, material certification, novelty, experiment authorization, procurement, manufacturing, deployment, or physical authority.";

const PLAN_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-benchmark-zero.plan.v0\0";
const BLIND_RANK_DOMAIN: &[u8] = b"symthaea.energy-benchmark-zero.blind-rank.v0\0";
const BLIND_ENSEMBLE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.blind-ensemble.v0\0";
const CONTROL_SUMMARY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.blind-summary.v0\0";
const COMPARISON_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.controlled-comparison.v0\0";

/// Benchmark choices that should be frozen before execution when chronology
/// matters. The digest itself does not prove when the plan was created.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkPlan {
    pub schema: &'static str,
    pub fold: u8,
    pub target: BandgapTarget,
    pub top_k: usize,
    pub baseline_method_version: &'static str,
    pub blind_control_method_version: &'static str,
    pub blind_control_replicates: u16,
}

impl BenchmarkPlan {
    pub fn new(fold: FoldIndex, target: BandgapTarget, top_k: usize) -> Result<Self, ControlError> {
        let plan = Self {
            schema: "symthaea.energy-benchmark-zero.plan.v0",
            fold: fold.value(),
            target,
            top_k,
            baseline_method_version: BASELINE_METHOD_VERSION,
            blind_control_method_version: BLIND_CONTROL_METHOD_VERSION,
            blind_control_replicates: BLIND_CONTROL_REPLICATES,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), ControlError> {
        FoldIndex::new(self.fold)?;
        self.target.validate()?;
        if self.top_k == 0 {
            return Err(ControlError::InvalidPlan(
                "top-k must be positive".into(),
            ));
        }
        if self.baseline_method_version != BASELINE_METHOD_VERSION
            || self.blind_control_method_version != BLIND_CONTROL_METHOD_VERSION
            || self.blind_control_replicates != BLIND_CONTROL_REPLICATES
        {
            return Err(ControlError::InvalidPlan(
                "plan method/control identity does not match this executable contract".into(),
            ));
        }
        Ok(())
    }

    pub fn fold_index(&self) -> Result<FoldIndex, ControlError> {
        Ok(FoldIndex::new(self.fold)?)
    }

    pub fn sha256(&self) -> Result<String, ControlError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(PLAN_DIGEST_DOMAIN);
        hasher.update([self.fold]);
        hasher.update(self.target.min_ev.to_bits().to_le_bytes());
        hasher.update(self.target.max_ev.to_bits().to_le_bytes());
        hasher.update((self.top_k as u64).to_le_bytes());
        update_text(&mut hasher, self.baseline_method_version);
        update_text(&mut hasher, self.blind_control_method_version);
        hasher.update(self.blind_control_replicates.to_le_bytes());
        Ok(hex_lower(&hasher.finalize()))
    }

    pub fn to_json_pretty(&self) -> Result<String, ControlError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Ranking-only metrics for the fixed property-blind control ensemble.
/// Prediction MAE is intentionally absent: the control makes no property
/// prediction and should not be forced through a fake midpoint predictor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BlindControlSummary {
    pub method_id: &'static str,
    pub method_version: &'static str,
    pub disclosure: &'static str,
    pub replicate_count: u16,
    pub candidate_count: usize,
    pub top_k: usize,
    pub qualifying_truth_count: usize,
    pub ensemble_sha256: String,
    pub mean_top_k_hits: f64,
    pub min_top_k_hits: usize,
    pub max_top_k_hits: usize,
    pub mean_top_k_precision: f64,
    pub mean_top_k_recall: f64,
    pub mean_target_regret_ev: f64,
    pub median_target_regret_ev: f64,
    pub min_target_regret_ev: f64,
    pub max_target_regret_ev: f64,
    /// Fraction of fixed blind controls whose top-k hit count is less than or
    /// equal to the physics baseline. Descriptive only; not a p-value.
    pub fraction_controls_hits_le_baseline: f64,
    /// Fraction of fixed blind controls whose target regret is greater than or
    /// equal to the physics baseline (lower regret is better). Descriptive only.
    pub fraction_controls_regret_ge_baseline: f64,
    /// Fraction of controls not better than baseline on both hit count and
    /// target regret simultaneously.
    pub fraction_controls_not_better_on_both: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControlledBenchmarkReceipt {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub plan: BenchmarkPlan,
    pub plan_sha256: String,
    pub registration_evidence_ref: Option<String>,
    pub chronology_disclosure: &'static str,
    pub leakage_qualification_sha256: String,
    pub baseline: BaselineBenchmarkReceipt,
    pub blind_control: BlindControlSummary,
    pub blind_control_summary_sha256: String,
    pub comparison_sha256: String,
}

impl ControlledBenchmarkReceipt {
    pub fn to_json_pretty(&self) -> Result<String, ControlError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("invalid benchmark plan: {0}")]
    InvalidPlan(String),
    #[error("registration evidence reference cannot be blank")]
    BlankRegistrationEvidenceRef,
    #[error("control candidate set is empty")]
    EmptyCandidateSet,
    #[error("control candidate set contains duplicate id {0:?}")]
    DuplicateCandidate(String),
    #[error("truth is missing control candidate {0:?}")]
    MissingTruth(String),
    #[error(
        "control candidate count {candidate_count} does not equal truth count {truth_count}"
    )]
    CandidateTruthCountMismatch {
        candidate_count: usize,
        truth_count: usize,
    },
    #[error("top-k {requested} exceeds control candidate count {available}")]
    InvalidTopK { requested: usize, available: usize },
    #[error("truth set has no candidates inside the requested target window")]
    NoQualifyingTruth,
    #[error("baseline and control leakage qualification identities differ")]
    LeakageQualificationMismatch,
    #[error("baseline receipt does not match the frozen benchmark plan")]
    BaselinePlanMismatch,
    #[error(transparent)]
    Fold(#[from] FoldError),
    #[error(transparent)]
    Baseline(#[from] BaselineError),
    #[error(transparent)]
    Benchmark(#[from] BenchmarkError),
    #[error("control receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Run the frozen baseline plus 256 fixed property-blind ranking controls.
///
/// When `registration_evidence_ref` is supplied it is bound into the final
/// comparison identity, but this crate does not authenticate its timestamp or
/// prove that registration occurred before result observation.
pub fn run_controlled_benchmark_from_official_bytes(
    compressed_bytes: &[u8],
    plan: BenchmarkPlan,
    registration_evidence_ref: Option<String>,
) -> Result<ControlledBenchmarkReceipt, ControlError> {
    plan.validate()?;
    let registration_evidence_ref = normalize_registration_ref(registration_evidence_ref)?;
    let plan_sha256 = plan.sha256()?;
    let fold = plan.fold_index()?;

    let qualification =
        build_leakage_clean_test_fold_from_official_bytes(compressed_bytes, fold)?;
    if plan.top_k > qualification.retained_test_count {
        return Err(ControlError::InvalidTopK {
            requested: plan.top_k,
            available: qualification.retained_test_count,
        });
    }

    let baseline = run_baseline_benchmark_from_official_bytes(
        compressed_bytes,
        fold,
        plan.target,
        plan.top_k,
    )?;

    if baseline.leakage_qualification_sha256 != qualification.qualification_sha256 {
        return Err(ControlError::LeakageQualificationMismatch);
    }
    if baseline.fold != plan.fold
        || baseline.benchmark.target != plan.target
        || baseline.benchmark.metrics.k != plan.top_k
        || baseline.method_version != plan.baseline_method_version
    {
        return Err(ControlError::BaselinePlanMismatch);
    }

    let candidate_ids: Vec<String> = qualification
        .truth
        .experimental_gap_ev
        .keys()
        .cloned()
        .collect();
    let blind_control = evaluate_blind_control_ensemble(
        &candidate_ids,
        &qualification.truth,
        plan.target,
        plan.top_k,
        &baseline.benchmark.metrics,
    )?;
    let blind_control_summary_sha256 =
        serialized_sha256(CONTROL_SUMMARY_DIGEST_DOMAIN, &blind_control)?;
    let comparison_sha256 = comparison_sha256(
        &plan_sha256,
        registration_evidence_ref.as_deref(),
        &qualification.qualification_sha256,
        &baseline.combined_sha256,
        &blind_control_summary_sha256,
    );

    Ok(ControlledBenchmarkReceipt {
        schema: "symthaea.energy-benchmark-zero.controlled-comparison.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        plan,
        plan_sha256,
        registration_evidence_ref,
        chronology_disclosure: PLAN_CHRONOLOGY_DISCLOSURE,
        leakage_qualification_sha256: qualification.qualification_sha256,
        baseline,
        blind_control,
        blind_control_summary_sha256,
        comparison_sha256,
    })
}

/// Evaluate a fixed property-blind reference ensemble against one truth slice.
/// Blind ranking itself depends only on candidate ids and replicate index.
pub fn evaluate_blind_control_ensemble(
    candidate_ids: &[String],
    truth: &BandgapTruthSet,
    target: BandgapTarget,
    top_k: usize,
    baseline_metrics: &BenchmarkMetrics,
) -> Result<BlindControlSummary, ControlError> {
    target.validate()?;
    truth.validate()?;
    validate_candidate_ids(candidate_ids, truth)?;
    if top_k == 0 || top_k > candidate_ids.len() {
        return Err(ControlError::InvalidTopK {
            requested: top_k,
            available: candidate_ids.len(),
        });
    }

    let qualifying_truth_count = truth
        .experimental_gap_ev
        .values()
        .filter(|&&gap| target.qualifies(gap))
        .count();
    if qualifying_truth_count == 0 {
        return Err(ControlError::NoQualifyingTruth);
    }
    let global_best_target_error_ev = truth
        .experimental_gap_ev
        .values()
        .map(|&gap| target.midpoint_error(gap))
        .fold(f64::INFINITY, f64::min);

    let mut ensemble_hasher = Sha256::new();
    ensemble_hasher.update(BLIND_ENSEMBLE_DIGEST_DOMAIN);
    ensemble_hasher.update(BLIND_CONTROL_METHOD_VERSION.as_bytes());
    ensemble_hasher.update((candidate_ids.len() as u64).to_le_bytes());
    ensemble_hasher.update((top_k as u64).to_le_bytes());

    let mut hit_counts = Vec::with_capacity(usize::from(BLIND_CONTROL_REPLICATES));
    let mut regrets = Vec::with_capacity(usize::from(BLIND_CONTROL_REPLICATES));

    for replicate in 0..BLIND_CONTROL_REPLICATES {
        let ordered = blind_order(candidate_ids, replicate);
        ensemble_hasher.update(replicate.to_le_bytes());
        for candidate_id in &ordered {
            update_text(&mut ensemble_hasher, candidate_id);
        }

        let top = &ordered[..top_k];
        let hits = top
            .iter()
            .filter(|candidate_id| target.qualifies(truth.experimental_gap_ev[*candidate_id]))
            .count();
        let best_selected_target_error_ev = top
            .iter()
            .map(|candidate_id| target.midpoint_error(truth.experimental_gap_ev[*candidate_id]))
            .fold(f64::INFINITY, f64::min);
        let regret =
            (best_selected_target_error_ev - global_best_target_error_ev).max(0.0);
        hit_counts.push(hits);
        regrets.push(regret);
    }

    let replicate_count = f64::from(BLIND_CONTROL_REPLICATES);
    let hit_sum: usize = hit_counts.iter().sum();
    let mean_top_k_hits = hit_sum as f64 / replicate_count;
    let mean_top_k_precision = mean_top_k_hits / top_k as f64;
    let mean_top_k_recall = mean_top_k_hits / qualifying_truth_count as f64;
    let mean_target_regret_ev = regrets.iter().sum::<f64>() / replicate_count;

    let mut sorted_regrets = regrets.clone();
    sorted_regrets.sort_by(|left, right| left.total_cmp(right));
    let median_target_regret_ev = 0.5
        * (sorted_regrets[sorted_regrets.len() / 2 - 1]
            + sorted_regrets[sorted_regrets.len() / 2]);

    let baseline_hits = baseline_metrics.top_k_hits;
    let baseline_regret = baseline_metrics.target_regret_ev;
    let hits_le = hit_counts
        .iter()
        .filter(|&&hits| hits <= baseline_hits)
        .count();
    let regret_ge = regrets
        .iter()
        .filter(|&&regret| regret >= baseline_regret)
        .count();
    let not_better_both = hit_counts
        .iter()
        .zip(regrets.iter())
        .filter(|(hits, regret)| **hits <= baseline_hits && **regret >= baseline_regret)
        .count();

    Ok(BlindControlSummary {
        method_id: BLIND_CONTROL_METHOD_ID,
        method_version: BLIND_CONTROL_METHOD_VERSION,
        disclosure: BLIND_CONTROL_DISCLOSURE,
        replicate_count: BLIND_CONTROL_REPLICATES,
        candidate_count: candidate_ids.len(),
        top_k,
        qualifying_truth_count,
        ensemble_sha256: hex_lower(&ensemble_hasher.finalize()),
        mean_top_k_hits,
        min_top_k_hits: *hit_counts.iter().min().expect("non-empty fixed ensemble"),
        max_top_k_hits: *hit_counts.iter().max().expect("non-empty fixed ensemble"),
        mean_top_k_precision,
        mean_top_k_recall,
        mean_target_regret_ev,
        median_target_regret_ev,
        min_target_regret_ev: regrets.iter().copied().fold(f64::INFINITY, f64::min),
        max_target_regret_ev: regrets
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        fraction_controls_hits_le_baseline: hits_le as f64 / replicate_count,
        fraction_controls_regret_ge_baseline: regret_ge as f64 / replicate_count,
        fraction_controls_not_better_on_both: not_better_both as f64 / replicate_count,
    })
}

fn blind_order<'a>(candidate_ids: &'a [String], replicate: u16) -> Vec<&'a str> {
    let mut scored: Vec<([u8; 32], &str)> = candidate_ids
        .iter()
        .map(|candidate_id| {
            let mut hasher = Sha256::new();
            hasher.update(BLIND_RANK_DOMAIN);
            hasher.update(replicate.to_le_bytes());
            update_text(&mut hasher, candidate_id);
            let score: [u8; 32] = hasher.finalize().into();
            (score, candidate_id.as_str())
        })
        .collect();
    scored.sort_by(|(left_score, left_id), (right_score, right_id)| {
        left_score
            .cmp(right_score)
            .then_with(|| left_id.cmp(right_id))
    });
    scored.into_iter().map(|(_, id)| id).collect()
}

fn validate_candidate_ids(
    candidate_ids: &[String],
    truth: &BandgapTruthSet,
) -> Result<(), ControlError> {
    if candidate_ids.is_empty() {
        return Err(ControlError::EmptyCandidateSet);
    }
    if candidate_ids.len() != truth.experimental_gap_ev.len() {
        return Err(ControlError::CandidateTruthCountMismatch {
            candidate_count: candidate_ids.len(),
            truth_count: truth.experimental_gap_ev.len(),
        });
    }

    let mut seen = BTreeSet::new();
    for candidate_id in candidate_ids {
        if !seen.insert(candidate_id.as_str()) {
            return Err(ControlError::DuplicateCandidate(candidate_id.clone()));
        }
        if !truth.experimental_gap_ev.contains_key(candidate_id) {
            return Err(ControlError::MissingTruth(candidate_id.clone()));
        }
    }
    Ok(())
}

fn normalize_registration_ref(value: Option<String>) -> Result<Option<String>, ControlError> {
    match value {
        Some(value) if value.trim().is_empty() => Err(ControlError::BlankRegistrationEvidenceRef),
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn serialized_sha256<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<String, ControlError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    Ok(hex_lower(&hasher.finalize()))
}

fn comparison_sha256(
    plan_sha256: &str,
    registration_evidence_ref: Option<&str>,
    qualification_sha256: &str,
    baseline_combined_sha256: &str,
    blind_control_summary_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(COMPARISON_DIGEST_DOMAIN);
    update_text(&mut hasher, plan_sha256);
    match registration_evidence_ref {
        Some(value) => {
            hasher.update([1]);
            update_text(&mut hasher, value);
        }
        None => hasher.update([0]),
    }
    update_text(&mut hasher, qualification_sha256);
    update_text(&mut hasher, baseline_combined_sha256);
    update_text(&mut hasher, blind_control_summary_sha256);
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
    use std::collections::BTreeMap;
    use symthaea_energy_benchmark_zero::{DataSliceId, DatasetProvenance};

    fn truth() -> BandgapTruthSet {
        BandgapTruthSet {
            provenance: DatasetProvenance {
                slice: DataSliceId::new("fixture", "test").unwrap(),
                source_uri: "https://example.invalid/fixture".into(),
                content_digest: "sha256:fixture".into(),
                license: "test-fixture".into(),
            },
            experimental_gap_ev: BTreeMap::from([
                ("A".into(), 1.10),
                ("B".into(), 1.20),
                ("C".into(), 1.40),
                ("D".into(), 1.80),
                ("E".into(), 2.20),
                ("F".into(), 0.70),
            ]),
        }
    }

    fn baseline_metrics() -> BenchmarkMetrics {
        BenchmarkMetrics {
            k: 2,
            ranked_candidate_count: 6,
            truth_candidate_count: 6,
            qualifying_truth_count: 3,
            top_k_hits: 2,
            top_k_precision: 1.0,
            top_k_recall: 2.0 / 3.0,
            mean_abs_prediction_error_ev: 0.5,
            best_selected_target_error_ev: 0.0,
            global_best_target_error_ev: 0.0,
            target_regret_ev: 0.0,
        }
    }

    #[test]
    fn plan_digest_binds_policy_choices() {
        let fold = FoldIndex::new(0).unwrap();
        let a = BenchmarkPlan::new(fold, BandgapTarget::new(1.0, 1.5).unwrap(), 10).unwrap();
        let b = BenchmarkPlan::new(fold, BandgapTarget::new(1.1, 1.5).unwrap(), 10).unwrap();
        let c = BenchmarkPlan::new(fold, BandgapTarget::new(1.0, 1.5).unwrap(), 11).unwrap();
        assert_ne!(a.sha256().unwrap(), b.sha256().unwrap());
        assert_ne!(a.sha256().unwrap(), c.sha256().unwrap());
        assert!(PLAN_CHRONOLOGY_DISCLOSURE.contains("not proof"));
    }

    #[test]
    fn blind_order_is_deterministic_and_replicate_specific() {
        let ids = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        assert_eq!(blind_order(&ids, 7), blind_order(&ids, 7));
        assert_ne!(blind_order(&ids, 7), blind_order(&ids, 8));
    }

    #[test]
    fn blind_control_summary_is_deterministic() {
        let truth = truth();
        let ids: Vec<String> = truth.experimental_gap_ev.keys().cloned().collect();
        let target = BandgapTarget::new(1.0, 1.5).unwrap();
        let first = evaluate_blind_control_ensemble(
            &ids,
            &truth,
            target,
            2,
            &baseline_metrics(),
        )
        .unwrap();
        let second = evaluate_blind_control_ensemble(
            &ids,
            &truth,
            target,
            2,
            &baseline_metrics(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.replicate_count, 256);
        assert_eq!(first.ensemble_sha256.len(), 64);
    }

    #[test]
    fn control_is_ranking_only_and_does_not_claim_significance() {
        assert!(BLIND_CONTROL_DISCLOSURE.contains("opaque byte string"));
        assert!(BLIND_CONTROL_DISCLOSURE.contains("not p-values"));
    }

    #[test]
    fn candidate_truth_mismatch_fails_closed() {
        let truth = truth();
        let ids = vec!["A".into(), "B".into()];
        assert!(matches!(
            evaluate_blind_control_ensemble(
                &ids,
                &truth,
                BandgapTarget::new(1.0, 1.5).unwrap(),
                1,
                &baseline_metrics(),
            ),
            Err(ControlError::CandidateTruthCountMismatch { .. })
        ));
    }

    #[test]
    fn blank_registration_reference_is_rejected() {
        assert!(normalize_registration_ref(Some("   ".into())).is_err());
        assert_eq!(normalize_registration_ref(None).unwrap(), None);
    }
}
