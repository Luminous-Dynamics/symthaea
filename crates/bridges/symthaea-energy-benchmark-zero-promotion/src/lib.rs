// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Declared-criteria promotion gate for Energy Discovery Benchmark Zero.
//!
//! This crate does not define universal scientific success thresholds. A caller
//! supplies explicit criteria, which are content-addressed before execution.
//! The gate compares the learned composition-only RF against the fixed physics
//! baseline on the same five leakage-qualified folds and retains contrary
//! fold-level evidence rather than allowing aggregate averages to hide it.

#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_energy_benchmark_zero::BandgapTarget;
use symthaea_energy_benchmark_zero_composition_rf::{
    composition_rf_method_version, run_composition_rf_benchmark_from_official_bytes,
    CompositionRfBenchmarkReceipt, CompositionRfError,
};
use symthaea_energy_benchmark_zero_suite::{
    run_five_fold_suite_from_official_bytes, BenchmarkSuitePlan, BenchmarkSuiteReceipt,
    SuiteError,
};
use symthaea_matbench_folds::{symthaea_training_table_sha256, FoldError, FoldIndex};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "DECLARED-CRITERIA BENCHMARK PROMOTION ASSESSMENT -- not scientific validation, material certification, novelty, experiment authorization, procurement, manufacturing, deployment, or physical authority.";

pub const CRITERIA_DISCLOSURE: &str =
    "Promotion criteria are caller-declared policy thresholds. Meeting them establishes only that the measured benchmark receipts satisfy those declared thresholds; it does not establish universal scientific validity, novelty, experimental truth, or deployment fitness.";

pub const CHRONOLOGY_DISCLOSURE: &str =
    "The promotion-plan SHA-256 is a content identity, not proof that criteria were fixed before results were observed. A chronology claim requires a separately immutable/timestamped registration evidence reference created before execution.";

pub const PARTITION_DISCLOSURE: &str =
    "All comparisons use five disjoint Matbench evaluation partitions from one source dataset. They are not five independent scientific replications and must not be converted into a replication count or significance claim.";

const CRITERIA_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.promotion-criteria.v0\0";
const PLAN_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-benchmark-zero.promotion-plan.v0\0";
const AGGREGATE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.promotion-aggregate.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-benchmark-zero.promotion-receipt.v0\0";

/// Explicit benchmark policy supplied by the caller.
///
/// No `Default` implementation exists deliberately: promotion thresholds must
/// be chosen explicitly rather than silently inheriting scientific authority
/// from this crate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PromotionCriteria {
    /// Learned micro top-k precision minus baseline precision must be >= this.
    pub min_micro_precision_delta: f64,
    /// Learned micro top-k recall minus baseline recall must be >= this.
    pub min_micro_recall_delta: f64,
    /// Learned weighted MAE minus baseline weighted MAE must be <= this.
    /// Negative values require an improvement.
    pub max_weighted_mae_delta_ev: f64,
    /// Learned macro target regret minus baseline macro regret must be <= this.
    /// Negative values require an improvement.
    pub max_macro_regret_delta_ev: f64,
    /// Per fold, learned top-k hits may fall below baseline by at most this many.
    pub max_per_fold_top_k_hit_drop: usize,
    /// Per fold, learned MAE minus baseline MAE must be <= this.
    pub max_per_fold_mae_increase_ev: f64,
    /// Per fold, learned target regret minus baseline regret must be <= this.
    pub max_per_fold_regret_increase_ev: f64,
    /// Number of folds where learned hits must meet/exceed the blind mean.
    pub min_folds_hits_ge_blind_mean: usize,
    /// Number of folds where learned regret must be <= blind median regret.
    pub min_folds_regret_le_blind_median: usize,
}

impl PromotionCriteria {
    pub fn validate(&self, top_k: usize) -> Result<(), PromotionError> {
        if top_k == 0 {
            return Err(PromotionError::InvalidCriteria(
                "top-k must be positive before criteria can be validated".into(),
            ));
        }
        for (name, value) in [
            ("min_micro_precision_delta", self.min_micro_precision_delta),
            ("min_micro_recall_delta", self.min_micro_recall_delta),
            ("max_weighted_mae_delta_ev", self.max_weighted_mae_delta_ev),
            ("max_macro_regret_delta_ev", self.max_macro_regret_delta_ev),
            (
                "max_per_fold_mae_increase_ev",
                self.max_per_fold_mae_increase_ev,
            ),
            (
                "max_per_fold_regret_increase_ev",
                self.max_per_fold_regret_increase_ev,
            ),
        ] {
            if !value.is_finite() {
                return Err(PromotionError::InvalidCriteria(format!(
                    "{name} must be finite"
                )));
            }
        }
        if !(-1.0..=1.0).contains(&self.min_micro_precision_delta)
            || !(-1.0..=1.0).contains(&self.min_micro_recall_delta)
        {
            return Err(PromotionError::InvalidCriteria(
                "micro precision/recall deltas must lie in [-1, 1]".into(),
            ));
        }
        if self.max_per_fold_top_k_hit_drop > top_k {
            return Err(PromotionError::InvalidCriteria(format!(
                "per-fold hit drop {} exceeds top-k {top_k}",
                self.max_per_fold_top_k_hit_drop
            )));
        }
        if self.min_folds_hits_ge_blind_mean > 5
            || self.min_folds_regret_le_blind_median > 5
        {
            return Err(PromotionError::InvalidCriteria(
                "fold-count criteria must lie in 0..=5".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self, top_k: usize) -> Result<String, PromotionError> {
        self.validate(top_k)?;
        let mut hasher = Sha256::new();
        hasher.update(CRITERIA_DIGEST_DOMAIN);
        hasher.update(self.min_micro_precision_delta.to_bits().to_le_bytes());
        hasher.update(self.min_micro_recall_delta.to_bits().to_le_bytes());
        hasher.update(self.max_weighted_mae_delta_ev.to_bits().to_le_bytes());
        hasher.update(self.max_macro_regret_delta_ev.to_bits().to_le_bytes());
        hasher.update((self.max_per_fold_top_k_hit_drop as u64).to_le_bytes());
        hasher.update(self.max_per_fold_mae_increase_ev.to_bits().to_le_bytes());
        hasher.update(self.max_per_fold_regret_increase_ev.to_bits().to_le_bytes());
        hasher.update((self.min_folds_hits_ge_blind_mean as u64).to_le_bytes());
        hasher.update((self.min_folds_regret_le_blind_median as u64).to_le_bytes());
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromotionPlan {
    pub schema: &'static str,
    pub suite_plan: BenchmarkSuitePlan,
    pub suite_plan_sha256: String,
    pub learned_method_version: String,
    pub training_table_sha256: String,
    pub criteria: PromotionCriteria,
    pub criteria_sha256: String,
    pub criteria_disclosure: &'static str,
    pub chronology_disclosure: &'static str,
}

impl PromotionPlan {
    pub fn new(
        target: BandgapTarget,
        top_k: usize,
        criteria: PromotionCriteria,
    ) -> Result<Self, PromotionError> {
        criteria.validate(top_k)?;
        let suite_plan = BenchmarkSuitePlan::new(target, top_k)?;
        let suite_plan_sha256 = suite_plan.sha256()?;
        let training_table_sha256 = symthaea_training_table_sha256();
        let learned_method_version = composition_rf_method_version(&training_table_sha256);
        let criteria_sha256 = criteria.sha256(top_k)?;
        let plan = Self {
            schema: "symthaea.energy-benchmark-zero.promotion-plan.v0",
            suite_plan,
            suite_plan_sha256,
            learned_method_version,
            training_table_sha256,
            criteria,
            criteria_sha256,
            criteria_disclosure: CRITERIA_DISCLOSURE,
            chronology_disclosure: CHRONOLOGY_DISCLOSURE,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), PromotionError> {
        self.suite_plan.validate()?;
        self.criteria.validate(self.suite_plan.top_k)?;
        let suite_digest = self.suite_plan.sha256()?;
        if self.suite_plan_sha256 != suite_digest {
            return Err(PromotionError::InvalidPlan(
                "suite-plan digest does not match embedded suite plan".into(),
            ));
        }
        let runtime_training_sha = symthaea_training_table_sha256();
        if self.training_table_sha256 != runtime_training_sha {
            return Err(PromotionError::InvalidPlan(
                "training-table identity does not match current executable contract".into(),
            ));
        }
        if self.learned_method_version != composition_rf_method_version(&runtime_training_sha) {
            return Err(PromotionError::InvalidPlan(
                "learned-model version does not match current executable contract".into(),
            ));
        }
        if self.criteria_sha256 != self.criteria.sha256(self.suite_plan.top_k)? {
            return Err(PromotionError::InvalidPlan(
                "criteria digest does not match embedded criteria".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, PromotionError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(PLAN_DIGEST_DOMAIN);
        update_text(&mut hasher, &self.suite_plan_sha256);
        update_text(&mut hasher, &self.learned_method_version);
        update_text(&mut hasher, &self.training_table_sha256);
        update_text(&mut hasher, &self.criteria_sha256);
        Ok(hex_lower(&hasher.finalize()))
    }

    pub fn to_json_pretty(&self) -> Result<String, PromotionError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PromotionOutcome {
    MeetsAllDeclaredCriteria,
    ViolatesAtLeastOneDeclaredCriterion,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FoldComparison {
    pub fold: u8,
    pub leakage_qualification_sha256: String,
    pub baseline_combined_sha256: String,
    pub learned_combined_sha256: String,
    pub candidate_count: usize,
    pub qualifying_truth_count: usize,
    pub baseline_top_k_hits: usize,
    pub learned_top_k_hits: usize,
    pub top_k_hit_delta: isize,
    pub baseline_mae_ev: f64,
    pub learned_mae_ev: f64,
    pub mae_delta_ev: f64,
    pub baseline_target_regret_ev: f64,
    pub learned_target_regret_ev: f64,
    pub target_regret_delta_ev: f64,
    pub blind_mean_top_k_hits: f64,
    pub blind_median_target_regret_ev: f64,
    pub learned_hits_ge_blind_mean: bool,
    pub learned_regret_le_blind_median: bool,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromotionAggregate {
    pub fold_count: usize,
    pub candidate_count: usize,
    pub qualifying_truth_count: usize,
    pub total_top_k_slots: usize,
    pub baseline_top_k_hits: usize,
    pub learned_top_k_hits: usize,
    pub baseline_micro_top_k_precision: f64,
    pub learned_micro_top_k_precision: f64,
    pub micro_precision_delta: f64,
    pub baseline_micro_top_k_recall: f64,
    pub learned_micro_top_k_recall: f64,
    pub micro_recall_delta: f64,
    pub baseline_weighted_mae_ev: f64,
    pub learned_weighted_mae_ev: f64,
    pub weighted_mae_delta_ev: f64,
    pub baseline_macro_target_regret_ev: f64,
    pub learned_macro_target_regret_ev: f64,
    pub macro_target_regret_delta_ev: f64,
    pub learned_max_target_regret_ev: f64,
    pub folds_learned_hits_ge_blind_mean: usize,
    pub folds_learned_regret_le_blind_median: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromotionReceipt {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub partition_disclosure: &'static str,
    pub plan: PromotionPlan,
    pub plan_sha256: String,
    pub registration_evidence_ref: Option<String>,
    pub chronology_disclosure: &'static str,
    pub baseline_suite: BenchmarkSuiteReceipt,
    pub learned_fold_receipts: Vec<CompositionRfBenchmarkReceipt>,
    pub fold_comparisons: Vec<FoldComparison>,
    pub aggregate: PromotionAggregate,
    pub aggregate_sha256: String,
    pub violations: Vec<String>,
    pub outcome: PromotionOutcome,
    pub receipt_sha256: String,
}

impl PromotionReceipt {
    pub fn to_json_pretty(&self) -> Result<String, PromotionError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Error)]
pub enum PromotionError {
    #[error("invalid promotion criteria: {0}")]
    InvalidCriteria(String),
    #[error("invalid promotion plan: {0}")]
    InvalidPlan(String),
    #[error("registration evidence reference cannot be blank")]
    BlankRegistrationEvidenceRef,
    #[error("fold {fold} baseline/learned leakage qualification identities differ")]
    QualificationMismatch { fold: u8 },
    #[error("fold {fold} baseline/learned benchmark policy differs")]
    PolicyMismatch { fold: u8 },
    #[error("fold {fold} baseline/learned evaluated populations differ")]
    PopulationMismatch { fold: u8 },
    #[error(transparent)]
    Suite(#[from] SuiteError),
    #[error(transparent)]
    Learned(#[from] CompositionRfError),
    #[error(transparent)]
    Fold(#[from] FoldError),
    #[error("promotion receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub fn run_promotion_assessment_from_official_bytes(
    compressed_bytes: &[u8],
    plan: PromotionPlan,
    registration_evidence_ref: Option<String>,
) -> Result<PromotionReceipt, PromotionError> {
    plan.validate()?;
    let registration_evidence_ref = normalize_registration_ref(registration_evidence_ref)?;
    let plan_sha256 = plan.sha256()?;

    // Promotion chronology is bound at this layer. The baseline suite is run
    // without claiming a separate suite-registration chronology.
    let baseline_suite =
        run_five_fold_suite_from_official_bytes(compressed_bytes, plan.suite_plan.clone(), None)?;

    let mut learned_fold_receipts = Vec::with_capacity(5);
    let mut fold_comparisons = Vec::with_capacity(5);

    let mut learned_hit_total = 0usize;
    let mut learned_weighted_abs_error = 0.0f64;
    let mut learned_regret_sum = 0.0f64;
    let mut learned_max_regret = f64::NEG_INFINITY;
    let mut learned_candidate_total = 0usize;
    let mut learned_qualifying_total = 0usize;
    let mut folds_hits_ge_blind_mean = 0usize;
    let mut folds_regret_le_blind_median = 0usize;

    for fold_value in 0u8..5 {
        let fold = FoldIndex::new(fold_value)?;
        let learned = run_composition_rf_benchmark_from_official_bytes(
            compressed_bytes,
            fold,
            plan.suite_plan.target,
            plan.suite_plan.top_k,
        )?;
        let controlled = &baseline_suite.fold_receipts[usize::from(fold_value)];

        if controlled.plan.fold != fold_value || learned.fold != fold_value {
            return Err(PromotionError::PolicyMismatch { fold: fold_value });
        }
        if controlled.leakage_qualification_sha256 != learned.leakage_qualification_sha256 {
            return Err(PromotionError::QualificationMismatch { fold: fold_value });
        }
        if controlled.baseline.benchmark.target != learned.benchmark.target
            || controlled.baseline.benchmark.target != plan.suite_plan.target
            || controlled.baseline.benchmark.metrics.k != learned.benchmark.metrics.k
            || learned.benchmark.metrics.k != plan.suite_plan.top_k
        {
            return Err(PromotionError::PolicyMismatch { fold: fold_value });
        }

        let baseline_metrics = &controlled.baseline.benchmark.metrics;
        let learned_metrics = &learned.benchmark.metrics;
        if baseline_metrics.ranked_candidate_count != learned_metrics.ranked_candidate_count
            || baseline_metrics.truth_candidate_count != learned_metrics.truth_candidate_count
            || baseline_metrics.qualifying_truth_count != learned_metrics.qualifying_truth_count
        {
            return Err(PromotionError::PopulationMismatch { fold: fold_value });
        }

        let hit_delta = learned_metrics.top_k_hits as isize - baseline_metrics.top_k_hits as isize;
        let mae_delta =
            learned_metrics.mean_abs_prediction_error_ev - baseline_metrics.mean_abs_prediction_error_ev;
        let regret_delta = learned_metrics.target_regret_ev - baseline_metrics.target_regret_ev;
        let learned_hits_ge_blind_mean =
            learned_metrics.top_k_hits as f64 >= controlled.blind_control.mean_top_k_hits;
        let learned_regret_le_blind_median =
            learned_metrics.target_regret_ev <= controlled.blind_control.median_target_regret_ev;

        if learned_hits_ge_blind_mean {
            folds_hits_ge_blind_mean += 1;
        }
        if learned_regret_le_blind_median {
            folds_regret_le_blind_median += 1;
        }

        let fold_comparison = FoldComparison {
            fold: fold_value,
            leakage_qualification_sha256: learned.leakage_qualification_sha256.clone(),
            baseline_combined_sha256: controlled.baseline.combined_sha256.clone(),
            learned_combined_sha256: learned.combined_sha256.clone(),
            candidate_count: learned_metrics.ranked_candidate_count,
            qualifying_truth_count: learned_metrics.qualifying_truth_count,
            baseline_top_k_hits: baseline_metrics.top_k_hits,
            learned_top_k_hits: learned_metrics.top_k_hits,
            top_k_hit_delta: hit_delta,
            baseline_mae_ev: baseline_metrics.mean_abs_prediction_error_ev,
            learned_mae_ev: learned_metrics.mean_abs_prediction_error_ev,
            mae_delta_ev: mae_delta,
            baseline_target_regret_ev: baseline_metrics.target_regret_ev,
            learned_target_regret_ev: learned_metrics.target_regret_ev,
            target_regret_delta_ev: regret_delta,
            blind_mean_top_k_hits: controlled.blind_control.mean_top_k_hits,
            blind_median_target_regret_ev: controlled.blind_control.median_target_regret_ev,
            learned_hits_ge_blind_mean,
            learned_regret_le_blind_median,
            violations: Vec::new(),
        };

        learned_hit_total += learned_metrics.top_k_hits;
        learned_weighted_abs_error += learned_metrics.mean_abs_prediction_error_ev
            * learned_metrics.ranked_candidate_count as f64;
        learned_regret_sum += learned_metrics.target_regret_ev;
        learned_max_regret = learned_max_regret.max(learned_metrics.target_regret_ev);
        learned_candidate_total += learned_metrics.ranked_candidate_count;
        learned_qualifying_total += learned_metrics.qualifying_truth_count;

        fold_comparisons.push(fold_comparison);
        learned_fold_receipts.push(learned);
    }

    if learned_candidate_total != baseline_suite.metrics.retained_candidate_count
        || learned_qualifying_total != baseline_suite.metrics.qualifying_truth_count
    {
        return Err(PromotionError::InvalidPlan(
            "learned aggregate population differs from baseline suite population".into(),
        ));
    }

    let total_top_k_slots = 5 * plan.suite_plan.top_k;
    let learned_micro_precision = learned_hit_total as f64 / total_top_k_slots as f64;
    let learned_micro_recall = learned_hit_total as f64 / learned_qualifying_total as f64;
    let learned_weighted_mae = learned_weighted_abs_error / learned_candidate_total as f64;
    let learned_macro_regret = learned_regret_sum / 5.0;

    let aggregate = PromotionAggregate {
        fold_count: 5,
        candidate_count: learned_candidate_total,
        qualifying_truth_count: learned_qualifying_total,
        total_top_k_slots,
        baseline_top_k_hits: baseline_suite.metrics.baseline_top_k_hits,
        learned_top_k_hits: learned_hit_total,
        baseline_micro_top_k_precision: baseline_suite.metrics.baseline_micro_top_k_precision,
        learned_micro_top_k_precision: learned_micro_precision,
        micro_precision_delta: learned_micro_precision
            - baseline_suite.metrics.baseline_micro_top_k_precision,
        baseline_micro_top_k_recall: baseline_suite.metrics.baseline_micro_top_k_recall,
        learned_micro_top_k_recall: learned_micro_recall,
        micro_recall_delta: learned_micro_recall - baseline_suite.metrics.baseline_micro_top_k_recall,
        baseline_weighted_mae_ev: baseline_suite.metrics.baseline_weighted_mae_ev,
        learned_weighted_mae_ev: learned_weighted_mae,
        weighted_mae_delta_ev: learned_weighted_mae
            - baseline_suite.metrics.baseline_weighted_mae_ev,
        baseline_macro_target_regret_ev: baseline_suite.metrics.baseline_macro_target_regret_ev,
        learned_macro_target_regret_ev: learned_macro_regret,
        macro_target_regret_delta_ev: learned_macro_regret
            - baseline_suite.metrics.baseline_macro_target_regret_ev,
        learned_max_target_regret_ev: learned_max_regret,
        folds_learned_hits_ge_blind_mean: folds_hits_ge_blind_mean,
        folds_learned_regret_le_blind_median: folds_regret_le_blind_median,
    };

    let mut violations = evaluate_declared_criteria(
        &plan.criteria,
        plan.suite_plan.top_k,
        &aggregate,
        &mut fold_comparisons,
    )?;
    // Stable ordering makes the receipt deterministic and review-friendly.
    violations.sort();
    for fold in &mut fold_comparisons {
        fold.violations.sort();
    }

    let outcome = if violations.is_empty() {
        PromotionOutcome::MeetsAllDeclaredCriteria
    } else {
        PromotionOutcome::ViolatesAtLeastOneDeclaredCriterion
    };
    let aggregate_sha256 = serialized_sha256(AGGREGATE_DIGEST_DOMAIN, &aggregate)?;
    let receipt_sha256 = promotion_receipt_sha256(
        &plan_sha256,
        registration_evidence_ref.as_deref(),
        &baseline_suite.suite_sha256,
        &learned_fold_receipts,
        &aggregate_sha256,
        &violations,
        outcome,
    );

    Ok(PromotionReceipt {
        schema: "symthaea.energy-benchmark-zero.promotion-receipt.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        partition_disclosure: PARTITION_DISCLOSURE,
        plan,
        plan_sha256,
        registration_evidence_ref,
        chronology_disclosure: CHRONOLOGY_DISCLOSURE,
        baseline_suite,
        learned_fold_receipts,
        fold_comparisons,
        aggregate,
        aggregate_sha256,
        violations,
        outcome,
        receipt_sha256,
    })
}

pub fn evaluate_declared_criteria(
    criteria: &PromotionCriteria,
    top_k: usize,
    aggregate: &PromotionAggregate,
    fold_comparisons: &mut [FoldComparison],
) -> Result<Vec<String>, PromotionError> {
    criteria.validate(top_k)?;
    if fold_comparisons.len() != 5 || aggregate.fold_count != 5 {
        return Err(PromotionError::InvalidPlan(
            "promotion assessment requires exactly five fold comparisons".into(),
        ));
    }

    let mut violations = Vec::new();
    if aggregate.micro_precision_delta < criteria.min_micro_precision_delta {
        violations.push(format!(
            "aggregate micro precision delta {} < declared minimum {}",
            aggregate.micro_precision_delta, criteria.min_micro_precision_delta
        ));
    }
    if aggregate.micro_recall_delta < criteria.min_micro_recall_delta {
        violations.push(format!(
            "aggregate micro recall delta {} < declared minimum {}",
            aggregate.micro_recall_delta, criteria.min_micro_recall_delta
        ));
    }
    if aggregate.weighted_mae_delta_ev > criteria.max_weighted_mae_delta_ev {
        violations.push(format!(
            "aggregate weighted MAE delta {} eV > declared maximum {} eV",
            aggregate.weighted_mae_delta_ev, criteria.max_weighted_mae_delta_ev
        ));
    }
    if aggregate.macro_target_regret_delta_ev > criteria.max_macro_regret_delta_ev {
        violations.push(format!(
            "aggregate macro target-regret delta {} eV > declared maximum {} eV",
            aggregate.macro_target_regret_delta_ev, criteria.max_macro_regret_delta_ev
        ));
    }
    if aggregate.folds_learned_hits_ge_blind_mean < criteria.min_folds_hits_ge_blind_mean {
        violations.push(format!(
            "folds meeting/exceeding blind mean hits {} < declared minimum {}",
            aggregate.folds_learned_hits_ge_blind_mean, criteria.min_folds_hits_ge_blind_mean
        ));
    }
    if aggregate.folds_learned_regret_le_blind_median
        < criteria.min_folds_regret_le_blind_median
    {
        violations.push(format!(
            "folds at/below blind median regret {} < declared minimum {}",
            aggregate.folds_learned_regret_le_blind_median,
            criteria.min_folds_regret_le_blind_median
        ));
    }

    for fold in fold_comparisons {
        let hit_drop = fold.baseline_top_k_hits.saturating_sub(fold.learned_top_k_hits);
        if hit_drop > criteria.max_per_fold_top_k_hit_drop {
            fold.violations.push(format!(
                "fold {} top-k hit drop {} > declared maximum {}",
                fold.fold, hit_drop, criteria.max_per_fold_top_k_hit_drop
            ));
        }
        if fold.mae_delta_ev > criteria.max_per_fold_mae_increase_ev {
            fold.violations.push(format!(
                "fold {} MAE delta {} eV > declared maximum {} eV",
                fold.fold, fold.mae_delta_ev, criteria.max_per_fold_mae_increase_ev
            ));
        }
        if fold.target_regret_delta_ev > criteria.max_per_fold_regret_increase_ev {
            fold.violations.push(format!(
                "fold {} target-regret delta {} eV > declared maximum {} eV",
                fold.fold,
                fold.target_regret_delta_ev,
                criteria.max_per_fold_regret_increase_ev
            ));
        }
        violations.extend(fold.violations.iter().cloned());
    }

    Ok(violations)
}

fn normalize_registration_ref(value: Option<String>) -> Result<Option<String>, PromotionError> {
    match value {
        Some(value) if value.trim().is_empty() => Err(PromotionError::BlankRegistrationEvidenceRef),
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn serialized_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, PromotionError> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    Ok(hex_lower(&hasher.finalize()))
}

fn promotion_receipt_sha256(
    plan_sha256: &str,
    registration_evidence_ref: Option<&str>,
    baseline_suite_sha256: &str,
    learned_fold_receipts: &[CompositionRfBenchmarkReceipt],
    aggregate_sha256: &str,
    violations: &[String],
    outcome: PromotionOutcome,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DIGEST_DOMAIN);
    update_text(&mut hasher, plan_sha256);
    match registration_evidence_ref {
        Some(value) => {
            hasher.update([1]);
            update_text(&mut hasher, value);
        }
        None => hasher.update([0]),
    }
    update_text(&mut hasher, baseline_suite_sha256);
    for receipt in learned_fold_receipts {
        update_text(&mut hasher, &receipt.combined_sha256);
    }
    update_text(&mut hasher, aggregate_sha256);
    for violation in violations {
        update_text(&mut hasher, violation);
    }
    hasher.update([match outcome {
        PromotionOutcome::MeetsAllDeclaredCriteria => 1,
        PromotionOutcome::ViolatesAtLeastOneDeclaredCriterion => 0,
    }]);
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

    fn permissive_criteria() -> PromotionCriteria {
        PromotionCriteria {
            min_micro_precision_delta: -1.0,
            min_micro_recall_delta: -1.0,
            max_weighted_mae_delta_ev: 10.0,
            max_macro_regret_delta_ev: 10.0,
            max_per_fold_top_k_hit_drop: 2,
            max_per_fold_mae_increase_ev: 10.0,
            max_per_fold_regret_increase_ev: 10.0,
            min_folds_hits_ge_blind_mean: 0,
            min_folds_regret_le_blind_median: 0,
        }
    }

    fn aggregate() -> PromotionAggregate {
        PromotionAggregate {
            fold_count: 5,
            candidate_count: 100,
            qualifying_truth_count: 20,
            total_top_k_slots: 10,
            baseline_top_k_hits: 4,
            learned_top_k_hits: 6,
            baseline_micro_top_k_precision: 0.4,
            learned_micro_top_k_precision: 0.6,
            micro_precision_delta: 0.2,
            baseline_micro_top_k_recall: 0.2,
            learned_micro_top_k_recall: 0.3,
            micro_recall_delta: 0.1,
            baseline_weighted_mae_ev: 1.0,
            learned_weighted_mae_ev: 0.8,
            weighted_mae_delta_ev: -0.2,
            baseline_macro_target_regret_ev: 0.2,
            learned_macro_target_regret_ev: 0.1,
            macro_target_regret_delta_ev: -0.1,
            learned_max_target_regret_ev: 0.2,
            folds_learned_hits_ge_blind_mean: 5,
            folds_learned_regret_le_blind_median: 5,
        }
    }

    fn folds() -> Vec<FoldComparison> {
        (0u8..5)
            .map(|fold| FoldComparison {
                fold,
                leakage_qualification_sha256: format!("q{fold}"),
                baseline_combined_sha256: format!("b{fold}"),
                learned_combined_sha256: format!("l{fold}"),
                candidate_count: 20,
                qualifying_truth_count: 4,
                baseline_top_k_hits: 1,
                learned_top_k_hits: 1,
                top_k_hit_delta: 0,
                baseline_mae_ev: 1.0,
                learned_mae_ev: 0.8,
                mae_delta_ev: -0.2,
                baseline_target_regret_ev: 0.2,
                learned_target_regret_ev: 0.1,
                target_regret_delta_ev: -0.1,
                blind_mean_top_k_hits: 0.5,
                blind_median_target_regret_ev: 0.3,
                learned_hits_ge_blind_mean: true,
                learned_regret_le_blind_median: true,
                violations: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn criteria_have_no_default_and_digest_binds_thresholds() {
        let a = permissive_criteria();
        let mut b = a;
        b.max_weighted_mae_delta_ev = 9.0;
        assert_ne!(a.sha256(2).unwrap(), b.sha256(2).unwrap());
        assert!(CRITERIA_DISCLOSURE.contains("caller-declared"));
    }

    #[test]
    fn permissive_declared_criteria_can_be_met_without_global_validation_label() {
        let mut fold_values = folds();
        let violations = evaluate_declared_criteria(
            &permissive_criteria(),
            2,
            &aggregate(),
            &mut fold_values,
        )
        .unwrap();
        assert!(violations.is_empty());
        let outcome = if violations.is_empty() {
            PromotionOutcome::MeetsAllDeclaredCriteria
        } else {
            PromotionOutcome::ViolatesAtLeastOneDeclaredCriterion
        };
        assert_eq!(outcome, PromotionOutcome::MeetsAllDeclaredCriteria);
        assert!(!CAPABILITY_CLASSIFICATION.contains("VALIDATED"));
    }

    #[test]
    fn bad_fold_is_not_hidden_by_good_aggregate() {
        let mut criteria = permissive_criteria();
        criteria.max_per_fold_top_k_hit_drop = 0;
        let mut fold_values = folds();
        fold_values[3].baseline_top_k_hits = 2;
        fold_values[3].learned_top_k_hits = 0;
        fold_values[3].top_k_hit_delta = -2;

        let violations = evaluate_declared_criteria(
            &criteria,
            2,
            &aggregate(),
            &mut fold_values,
        )
        .unwrap();
        assert!(violations.iter().any(|value| value.contains("fold 3 top-k hit drop")));
        assert!(!fold_values[3].violations.is_empty());
    }

    #[test]
    fn criteria_validation_rejects_nonfinite_and_impossible_fold_counts() {
        let mut criteria = permissive_criteria();
        criteria.max_weighted_mae_delta_ev = f64::NAN;
        assert!(criteria.validate(2).is_err());

        let mut criteria = permissive_criteria();
        criteria.min_folds_hits_ge_blind_mean = 6;
        assert!(criteria.validate(2).is_err());
    }

    #[test]
    fn promotion_plan_digest_binds_criteria_and_model_identity() {
        let target = BandgapTarget::new(1.0, 1.5).unwrap();
        let a = PromotionPlan::new(target, 2, permissive_criteria()).unwrap();
        let mut changed = permissive_criteria();
        changed.max_macro_regret_delta_ev = 9.0;
        let b = PromotionPlan::new(target, 2, changed).unwrap();
        assert_ne!(a.sha256().unwrap(), b.sha256().unwrap());
        assert!(a.learned_method_version.contains(&a.training_table_sha256));
        assert!(CHRONOLOGY_DISCLOSURE.contains("not proof"));
    }
}
