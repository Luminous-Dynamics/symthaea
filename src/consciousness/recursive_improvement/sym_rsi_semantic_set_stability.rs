// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic calibration-stability audit for SYM-RSI-SEM-003.
//!
//! SEM-003S never rewrites the canonical SEM-003 result. It asks whether the
//! canonical result survives eight preregistered leave-one-fold-out calibration
//! perturbations under the exact same primary criteria.

use std::collections::HashSet;

use super::semantic_context::{ContextIdentity, SemanticContextError};
use super::semantic_null_calibration::SemanticNullCalibration;
use super::sym_rsi_semantic_set_experiment::{
    SYM_RSI_SEM_003_AMENDMENT_1_SHA, SYM_RSI_SEM_003_DIMENSIONS,
    SYM_RSI_SEM_003_PREREGISTRATION_SHA, Sem3AmbiguousQuery, Sem3Corpus,
    Sem3DimensionMetrics, Sem3Disposition, Sem3ExperimentError, Sem3LabelledQuery,
    Sem3Receipt, Sem3Threshold, Sem3UnlabelledQuery, ambiguous_query_digest,
    build_semantic_null, calibrate_sem3_thresholds, candidate_bank_digest, disposition_tag,
    evaluate_sem3_with_thresholds, labelled_query_digest, prepare_sem3_corpus,
    run_sym_rsi_sem_003, sem3_candidate_set_identities, summarize_sem3_primary, threshold_for,
    unlabelled_query_digest,
};

pub const SYM_RSI_SEM_003S_SCHEMA: &str =
    "symthaea.sym-rsi-sem-003s.calibration-stability.v1";
pub const SYM_RSI_SEM_003S_PREREGISTRATION_SHA: &str =
    "a8640c13c5cb3e1c56d7006750eb21ea85c200ca";
pub const SYM_RSI_SEM_003S_AMENDMENT_1_SHA: &str =
    "e829e281ae8683e5a26d7911b8e0d8fdd890bbba";
pub const SYM_RSI_SEM_003_PRIMARY_IMPLEMENTATION_SHA: &str =
    "dd533164f8aff636f06e49a161fa75b4e70359d1";
pub const SYM_RSI_SEM_003S_FOLD_COUNT: usize = 8;
pub const SYM_RSI_SEM_003S_OMITTED_PER_DIMENSION: usize = 4;
pub const SYM_RSI_SEM_003S_RETAINED_PER_DIMENSION: usize = 28;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sem3sDisposition {
    StablePass,
    PrimaryPassCalibrationFragile,
    PrimaryNotPass,
    /// Reserved for serialized/adjudicated receipts. Live construction fails first.
    IntegrityFailure,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3sOmittedCalibrationDiagnostics {
    pub context_dimension: usize,
    pub fold_id: usize,
    pub included_count: usize,
    pub omitted_count: usize,
    pub threshold: f32,
    pub omitted_target_coverage: f64,
    pub min_nonconformity: f32,
    pub max_nonconformity: f32,
    pub mean_nonconformity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3sDriftSummary {
    pub query_count: usize,
    pub exact_set_agreement_rate: f64,
    pub mean_symmetric_difference_size: f64,
    pub max_symmetric_difference_size: usize,
    pub mean_absolute_set_size_difference: f64,
    pub max_absolute_set_size_difference: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3sThresholdStabilitySummary {
    pub context_dimension: usize,
    pub canonical_threshold: f32,
    pub min_fold_threshold: f32,
    pub max_fold_threshold: f32,
    pub mean_fold_threshold: f64,
    pub absolute_threshold_range: f32,
    pub max_absolute_deviation_from_canonical: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sem3sFoldReceipt {
    pub fold_id: usize,
    pub fold_membership_digest: [u8; 32],
    pub thresholds: Vec<Sem3Threshold>,
    pub dimension_metrics: Vec<Sem3DimensionMetrics>,
    pub clean_target_coverage: f64,
    pub clean_mean_set_size: f64,
    pub ambiguous_at_least_one_parent_inclusion: f64,
    pub ambiguous_dual_parent_inclusion: f64,
    pub ambiguous_singleton_forced_choice_rate: f64,
    pub unrelated_nonempty_set_rate: f64,
    pub ood_support_rejection_rate: f64,
    pub clean_coverage_passed: bool,
    pub per_dimension_clean_coverage_passed: bool,
    pub clean_set_size_guard_passed: bool,
    pub ambiguous_parent_guard_passed: bool,
    pub ambiguous_dual_parent_guard_passed: bool,
    pub ambiguous_singleton_guard_passed: bool,
    pub unrelated_guard_passed: bool,
    pub ood_support_guard_passed: bool,
    pub disposition: Sem3Disposition,
    pub clean_drift: Sem3sDriftSummary,
    pub ambiguous_drift: Sem3sDriftSummary,
    pub unrelated_drift: Sem3sDriftSummary,
    pub omitted_calibration: Vec<Sem3sOmittedCalibrationDiagnostics>,
    pub evidence_digest: [u8; 32],
}

impl Sem3sFoldReceipt {
    pub fn validate_self(&self) -> bool {
        self.fold_id < SYM_RSI_SEM_003S_FOLD_COUNT
            && self.thresholds.len() == SYM_RSI_SEM_003_DIMENSIONS.len()
            && self.omitted_calibration.len() == SYM_RSI_SEM_003_DIMENSIONS.len()
            && self.evidence_digest == fold_receipt_digest(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sem3sReceipt {
    pub schema: &'static str,
    pub preregistration_sha: &'static str,
    pub amendment_1_sha: &'static str,
    pub primary_preregistration_sha: &'static str,
    pub primary_amendment_1_sha: &'static str,
    pub primary_implementation_sha: &'static str,
    pub stability_implementation_sha: String,
    pub primary_receipt_evidence_digest: [u8; 32],
    pub candidate_bank_digest: [u8; 32],
    pub calibration_query_digest: [u8; 32],
    pub clean_query_digest: [u8; 32],
    pub ambiguous_query_digest: [u8; 32],
    pub unrelated_query_digest: [u8; 32],
    pub ood_query_digest: [u8; 32],
    pub folds: Vec<Sem3sFoldReceipt>,
    pub threshold_stability: Vec<Sem3sThresholdStabilitySummary>,
    pub disposition: Sem3sDisposition,
    pub evidence_digest: [u8; 32],
}

impl Sem3sReceipt {
    pub fn evidence_digest_hex(&self) -> String {
        hex::encode(self.evidence_digest)
    }

    pub fn validate_self(&self) -> bool {
        self.schema == SYM_RSI_SEM_003S_SCHEMA
            && self.preregistration_sha == SYM_RSI_SEM_003S_PREREGISTRATION_SHA
            && self.amendment_1_sha == SYM_RSI_SEM_003S_AMENDMENT_1_SHA
            && self.primary_preregistration_sha == SYM_RSI_SEM_003_PREREGISTRATION_SHA
            && self.primary_amendment_1_sha == SYM_RSI_SEM_003_AMENDMENT_1_SHA
            && self.primary_implementation_sha == SYM_RSI_SEM_003_PRIMARY_IMPLEMENTATION_SHA
            && is_valid_sha(&self.stability_implementation_sha)
            && self.folds.len() == SYM_RSI_SEM_003S_FOLD_COUNT
            && self.threshold_stability.len() == SYM_RSI_SEM_003_DIMENSIONS.len()
            && self.folds.iter().all(Sem3sFoldReceipt::validate_self)
            && self.evidence_digest == stability_receipt_digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sem3sError {
    Primary(Sem3ExperimentError),
    InvalidStabilitySubject,
    PrimaryReceiptInvalid,
    PrimarySubjectMismatch,
    PrimaryReceiptReconstructionMismatch,
    CorpusDigestMismatch,
    FoldCountMismatch,
    FoldBalanceMismatch,
    MissingCandidate,
}

impl From<Sem3ExperimentError> for Sem3sError {
    fn from(value: Sem3ExperimentError) -> Self {
        Self::Primary(value)
    }
}

impl From<SemanticContextError> for Sem3sError {
    fn from(value: SemanticContextError) -> Self {
        Self::Primary(Sem3ExperimentError::Context(value))
    }
}

/// Run the preregistered SEM-003S deterministic calibration-stability audit.
///
/// This consumes only the synthetic SEM-003 corpus. It does not touch protected
/// SYM-RSI experiment partitions and grants no confidence/evidence authority.
pub fn run_sym_rsi_sem_003s(
    stability_implementation_sha: &str,
    primary_receipt: &Sem3Receipt,
) -> Result<Sem3sReceipt, Sem3sError> {
    if !is_valid_sha(stability_implementation_sha) {
        return Err(Sem3sError::InvalidStabilitySubject);
    }
    validate_primary_receipt(primary_receipt)?;

    let corpus = prepare_sem3_corpus()?;
    validate_corpus_binding(&corpus, primary_receipt)?;

    let reconstructed = run_sym_rsi_sem_003(SYM_RSI_SEM_003_PRIMARY_IMPLEMENTATION_SHA)?;
    if reconstructed != *primary_receipt {
        return Err(Sem3sError::PrimaryReceiptReconstructionMismatch);
    }

    let semantic_null = build_semantic_null(&corpus)?;
    let mut folds = Vec::with_capacity(SYM_RSI_SEM_003S_FOLD_COUNT);
    for fold_id in 0..SYM_RSI_SEM_003S_FOLD_COUNT {
        folds.push(run_fold(
            &corpus,
            &semantic_null,
            primary_receipt,
            fold_id,
        )?);
    }

    let threshold_stability = threshold_stability(primary_receipt, &folds)?;
    let fold_dispositions = folds.iter().map(|fold| fold.disposition).collect::<Vec<_>>();
    let disposition = classify_stability(primary_receipt.disposition, &fold_dispositions);

    let mut receipt = Sem3sReceipt {
        schema: SYM_RSI_SEM_003S_SCHEMA,
        preregistration_sha: SYM_RSI_SEM_003S_PREREGISTRATION_SHA,
        amendment_1_sha: SYM_RSI_SEM_003S_AMENDMENT_1_SHA,
        primary_preregistration_sha: SYM_RSI_SEM_003_PREREGISTRATION_SHA,
        primary_amendment_1_sha: SYM_RSI_SEM_003_AMENDMENT_1_SHA,
        primary_implementation_sha: SYM_RSI_SEM_003_PRIMARY_IMPLEMENTATION_SHA,
        stability_implementation_sha: stability_implementation_sha.to_owned(),
        primary_receipt_evidence_digest: primary_receipt.evidence_digest,
        candidate_bank_digest: primary_receipt.candidate_bank_digest,
        calibration_query_digest: primary_receipt.calibration_query_digest,
        clean_query_digest: primary_receipt.clean_query_digest,
        ambiguous_query_digest: primary_receipt.ambiguous_query_digest,
        unrelated_query_digest: primary_receipt.unrelated_query_digest,
        ood_query_digest: primary_receipt.ood_query_digest,
        folds,
        threshold_stability,
        disposition,
        evidence_digest: [0; 32],
    };
    receipt.evidence_digest = stability_receipt_digest(&receipt);
    Ok(receipt)
}

fn validate_primary_receipt(primary: &Sem3Receipt) -> Result<(), Sem3sError> {
    if !primary.validate_self() {
        return Err(Sem3sError::PrimaryReceiptInvalid);
    }
    if primary.subject_sha != SYM_RSI_SEM_003_PRIMARY_IMPLEMENTATION_SHA {
        return Err(Sem3sError::PrimarySubjectMismatch);
    }
    Ok(())
}

fn validate_corpus_binding(corpus: &Sem3Corpus, primary: &Sem3Receipt) -> Result<(), Sem3sError> {
    let matches = candidate_bank_digest(&corpus.candidates) == primary.candidate_bank_digest
        && labelled_query_digest("calibration", &corpus.calibration)
            == primary.calibration_query_digest
        && labelled_query_digest("clean", &corpus.clean) == primary.clean_query_digest
        && ambiguous_query_digest(&corpus.ambiguous) == primary.ambiguous_query_digest
        && unlabelled_query_digest("unrelated", &corpus.unrelated)
            == primary.unrelated_query_digest
        && unlabelled_query_digest("ood", &corpus.ood) == primary.ood_query_digest;
    if matches {
        Ok(())
    } else {
        Err(Sem3sError::CorpusDigestMismatch)
    }
}

fn run_fold(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    primary: &Sem3Receipt,
    fold_id: usize,
) -> Result<Sem3sFoldReceipt, Sem3sError> {
    let (retained, omitted, fold_membership_digest) = split_calibration(corpus, fold_id)?;
    let thresholds = calibrate_sem3_thresholds(corpus, &retained)?;
    if thresholds
        .iter()
        .any(|threshold| threshold.calibration_count != SYM_RSI_SEM_003S_RETAINED_PER_DIMENSION)
    {
        return Err(Sem3sError::FoldBalanceMismatch);
    }

    let evaluation = evaluate_sem3_with_thresholds(corpus, semantic_null, &thresholds)?;
    let summary = summarize_sem3_primary(&evaluation);
    let omitted_calibration = omitted_diagnostics(corpus, &omitted, &thresholds, fold_id)?;
    let clean_drift = drift_for_labelled(
        corpus,
        semantic_null,
        &primary.thresholds,
        &thresholds,
        &corpus.clean,
    )?;
    let ambiguous_drift = drift_for_ambiguous(
        corpus,
        semantic_null,
        &primary.thresholds,
        &thresholds,
        &corpus.ambiguous,
    )?;
    let unrelated_drift = drift_for_unlabelled(
        corpus,
        semantic_null,
        &primary.thresholds,
        &thresholds,
        &corpus.unrelated,
    )?;

    let mut receipt = Sem3sFoldReceipt {
        fold_id,
        fold_membership_digest,
        thresholds,
        dimension_metrics: evaluation.dimension_metrics,
        clean_target_coverage: summary.clean_target_coverage,
        clean_mean_set_size: summary.clean_mean_set_size,
        ambiguous_at_least_one_parent_inclusion: summary.ambiguous_at_least_one_parent_inclusion,
        ambiguous_dual_parent_inclusion: summary.ambiguous_dual_parent_inclusion,
        ambiguous_singleton_forced_choice_rate: summary.ambiguous_singleton_forced_choice_rate,
        unrelated_nonempty_set_rate: summary.unrelated_nonempty_set_rate,
        ood_support_rejection_rate: summary.ood_support_rejection_rate,
        clean_coverage_passed: summary.clean_coverage_passed,
        per_dimension_clean_coverage_passed: summary.per_dimension_clean_coverage_passed,
        clean_set_size_guard_passed: summary.clean_set_size_guard_passed,
        ambiguous_parent_guard_passed: summary.ambiguous_parent_guard_passed,
        ambiguous_dual_parent_guard_passed: summary.ambiguous_dual_parent_guard_passed,
        ambiguous_singleton_guard_passed: summary.ambiguous_singleton_guard_passed,
        unrelated_guard_passed: summary.unrelated_guard_passed,
        ood_support_guard_passed: summary.ood_support_guard_passed,
        disposition: summary.disposition,
        clean_drift,
        ambiguous_drift,
        unrelated_drift,
        omitted_calibration,
        evidence_digest: [0; 32],
    };
    receipt.evidence_digest = fold_receipt_digest(&receipt);
    Ok(receipt)
}

fn split_calibration(
    corpus: &Sem3Corpus,
    fold_id: usize,
) -> Result<(Vec<Sem3LabelledQuery>, Vec<Sem3LabelledQuery>, [u8; 32]), Sem3sError> {
    if fold_id >= SYM_RSI_SEM_003S_FOLD_COUNT {
        return Err(Sem3sError::FoldCountMismatch);
    }

    let mut retained = Vec::new();
    let mut omitted = Vec::new();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003s.fold-membership.v1");
    hasher.update(&(fold_id as u64).to_le_bytes());

    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let mut queries = Vec::new();
        for query in corpus
            .calibration
            .iter()
            .filter(|query| query.dimension == dimension)
        {
            let identity = ContextIdentity::from_context(&query.context)?;
            queries.push((identity, query.clone()));
        }
        queries.sort_by(|left, right| left.0.exact_digest.cmp(&right.0.exact_digest));
        if queries.len()
            != SYM_RSI_SEM_003S_FOLD_COUNT * SYM_RSI_SEM_003S_OMITTED_PER_DIMENSION
        {
            return Err(Sem3sError::FoldBalanceMismatch);
        }

        let mut omitted_in_dimension = 0usize;
        for (position, (identity, query)) in queries.into_iter().enumerate() {
            if position % SYM_RSI_SEM_003S_FOLD_COUNT == fold_id {
                hasher.update(&(dimension as u64).to_le_bytes());
                hasher.update(&identity.exact_digest);
                hasher.update(&query.target.exact_digest);
                omitted.push(query);
                omitted_in_dimension += 1;
            } else {
                retained.push(query);
            }
        }
        if omitted_in_dimension != SYM_RSI_SEM_003S_OMITTED_PER_DIMENSION {
            return Err(Sem3sError::FoldBalanceMismatch);
        }
    }

    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let retained_count = retained
            .iter()
            .filter(|query| query.dimension == dimension)
            .count();
        let omitted_count = omitted
            .iter()
            .filter(|query| query.dimension == dimension)
            .count();
        if retained_count != SYM_RSI_SEM_003S_RETAINED_PER_DIMENSION
            || omitted_count != SYM_RSI_SEM_003S_OMITTED_PER_DIMENSION
        {
            return Err(Sem3sError::FoldBalanceMismatch);
        }
    }

    Ok((retained, omitted, *hasher.finalize().as_bytes()))
}

fn omitted_diagnostics(
    corpus: &Sem3Corpus,
    omitted: &[Sem3LabelledQuery],
    thresholds: &[Sem3Threshold],
    fold_id: usize,
) -> Result<Vec<Sem3sOmittedCalibrationDiagnostics>, Sem3sError> {
    let mut diagnostics = Vec::with_capacity(SYM_RSI_SEM_003_DIMENSIONS.len());
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let threshold = threshold_for(dimension, thresholds)?;
        let dim_omitted = omitted
            .iter()
            .filter(|query| query.dimension == dimension)
            .collect::<Vec<_>>();
        if dim_omitted.len() != SYM_RSI_SEM_003S_OMITTED_PER_DIMENSION {
            return Err(Sem3sError::FoldBalanceMismatch);
        }

        let mut scores = Vec::with_capacity(dim_omitted.len());
        for query in dim_omitted {
            let target = corpus
                .candidates
                .iter()
                .find(|candidate| candidate.identity == query.target)
                .ok_or(Sem3sError::MissingCandidate)?;
            let query_key = corpus.encoder.encode(&query.context)?;
            let similarity = query_key.similarity(&target.semantic_key);
            scores.push((1.0 - similarity).clamp(0.0, 1.0));
        }
        let covered = scores.iter().filter(|score| **score <= threshold).count();
        diagnostics.push(Sem3sOmittedCalibrationDiagnostics {
            context_dimension: dimension,
            fold_id,
            included_count: SYM_RSI_SEM_003S_RETAINED_PER_DIMENSION,
            omitted_count: scores.len(),
            threshold,
            omitted_target_coverage: rate(covered, scores.len()),
            min_nonconformity: min_f32(&scores),
            max_nonconformity: max_f32(&scores),
            mean_nonconformity: mean_f32(&scores),
        });
    }
    Ok(diagnostics)
}

fn drift_for_labelled(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    canonical: &[Sem3Threshold],
    fold: &[Sem3Threshold],
    queries: &[Sem3LabelledQuery],
) -> Result<Sem3sDriftSummary, Sem3sError> {
    let contexts = queries
        .iter()
        .map(|query| (query.dimension, query.context.as_slice()))
        .collect::<Vec<_>>();
    drift_for_contexts(corpus, semantic_null, canonical, fold, &contexts)
}

fn drift_for_ambiguous(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    canonical: &[Sem3Threshold],
    fold: &[Sem3Threshold],
    queries: &[Sem3AmbiguousQuery],
) -> Result<Sem3sDriftSummary, Sem3sError> {
    let contexts = queries
        .iter()
        .map(|query| (query.dimension, query.context.as_slice()))
        .collect::<Vec<_>>();
    drift_for_contexts(corpus, semantic_null, canonical, fold, &contexts)
}

fn drift_for_unlabelled(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    canonical: &[Sem3Threshold],
    fold: &[Sem3Threshold],
    queries: &[Sem3UnlabelledQuery],
) -> Result<Sem3sDriftSummary, Sem3sError> {
    let contexts = queries
        .iter()
        .map(|query| (query.dimension, query.context.as_slice()))
        .collect::<Vec<_>>();
    drift_for_contexts(corpus, semantic_null, canonical, fold, &contexts)
}

fn drift_for_contexts(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    canonical: &[Sem3Threshold],
    fold: &[Sem3Threshold],
    contexts: &[(usize, &[f32])],
) -> Result<Sem3sDriftSummary, Sem3sError> {
    let mut exact_agreements = 0usize;
    let mut symmetric_differences = Vec::with_capacity(contexts.len());
    let mut size_differences = Vec::with_capacity(contexts.len());

    for (dimension, context) in contexts {
        let canonical_threshold = threshold_for(*dimension, canonical)?;
        let fold_threshold = threshold_for(*dimension, fold)?;
        let canonical_set = sem3_candidate_set_identities(
            corpus,
            semantic_null,
            *dimension,
            context,
            canonical_threshold,
        )?;
        let fold_set = sem3_candidate_set_identities(
            corpus,
            semantic_null,
            *dimension,
            context,
            fold_threshold,
        )?;
        if canonical_set == fold_set {
            exact_agreements += 1;
        }
        symmetric_differences.push(symmetric_difference_size(&canonical_set, &fold_set));
        size_differences.push(canonical_set.len().abs_diff(fold_set.len()));
    }

    Ok(Sem3sDriftSummary {
        query_count: contexts.len(),
        exact_set_agreement_rate: rate(exact_agreements, contexts.len()),
        mean_symmetric_difference_size: mean_usize(&symmetric_differences),
        max_symmetric_difference_size: symmetric_differences
            .iter()
            .copied()
            .max()
            .unwrap_or_default(),
        mean_absolute_set_size_difference: mean_usize(&size_differences),
        max_absolute_set_size_difference: size_differences
            .iter()
            .copied()
            .max()
            .unwrap_or_default(),
    })
}

fn symmetric_difference_size(
    left: &HashSet<ContextIdentity>,
    right: &HashSet<ContextIdentity>,
) -> usize {
    left.symmetric_difference(right).count()
}

fn threshold_stability(
    primary: &Sem3Receipt,
    folds: &[Sem3sFoldReceipt],
) -> Result<Vec<Sem3sThresholdStabilitySummary>, Sem3sError> {
    if folds.len() != SYM_RSI_SEM_003S_FOLD_COUNT {
        return Err(Sem3sError::FoldCountMismatch);
    }
    let mut summaries = Vec::with_capacity(SYM_RSI_SEM_003_DIMENSIONS.len());
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let canonical_threshold = threshold_for(dimension, &primary.thresholds)?;
        let fold_thresholds = folds
            .iter()
            .map(|fold| threshold_for(dimension, &fold.thresholds))
            .collect::<Result<Vec<_>, _>>()?;
        let min_fold_threshold = min_f32(&fold_thresholds);
        let max_fold_threshold = max_f32(&fold_thresholds);
        let max_absolute_deviation_from_canonical = fold_thresholds
            .iter()
            .map(|threshold| (*threshold - canonical_threshold).abs())
            .fold(0.0_f32, f32::max);
        summaries.push(Sem3sThresholdStabilitySummary {
            context_dimension: dimension,
            canonical_threshold,
            min_fold_threshold,
            max_fold_threshold,
            mean_fold_threshold: mean_f32(&fold_thresholds),
            absolute_threshold_range: max_fold_threshold - min_fold_threshold,
            max_absolute_deviation_from_canonical,
        });
    }
    Ok(summaries)
}

fn classify_stability(
    primary: Sem3Disposition,
    fold_dispositions: &[Sem3Disposition],
) -> Sem3sDisposition {
    if primary != Sem3Disposition::Pass {
        Sem3sDisposition::PrimaryNotPass
    } else if fold_dispositions.len() == SYM_RSI_SEM_003S_FOLD_COUNT
        && fold_dispositions
            .iter()
            .all(|disposition| *disposition == Sem3Disposition::Pass)
    {
        Sem3sDisposition::StablePass
    } else {
        Sem3sDisposition::PrimaryPassCalibrationFragile
    }
}

fn fold_receipt_digest(receipt: &Sem3sFoldReceipt) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003s.fold-receipt.v1");
    hasher.update(&(receipt.fold_id as u64).to_le_bytes());
    hasher.update(&receipt.fold_membership_digest);
    for threshold in &receipt.thresholds {
        hash_threshold(&mut hasher, *threshold);
    }
    for metrics in &receipt.dimension_metrics {
        hash_dimension_metrics(&mut hasher, *metrics);
    }
    for value in [
        receipt.clean_target_coverage,
        receipt.clean_mean_set_size,
        receipt.ambiguous_at_least_one_parent_inclusion,
        receipt.ambiguous_dual_parent_inclusion,
        receipt.ambiguous_singleton_forced_choice_rate,
        receipt.unrelated_nonempty_set_rate,
        receipt.ood_support_rejection_rate,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&[
        receipt.clean_coverage_passed as u8,
        receipt.per_dimension_clean_coverage_passed as u8,
        receipt.clean_set_size_guard_passed as u8,
        receipt.ambiguous_parent_guard_passed as u8,
        receipt.ambiguous_dual_parent_guard_passed as u8,
        receipt.ambiguous_singleton_guard_passed as u8,
        receipt.unrelated_guard_passed as u8,
        receipt.ood_support_guard_passed as u8,
        disposition_tag(receipt.disposition),
    ]);
    hash_drift(&mut hasher, receipt.clean_drift);
    hash_drift(&mut hasher, receipt.ambiguous_drift);
    hash_drift(&mut hasher, receipt.unrelated_drift);
    hasher.update(&(receipt.omitted_calibration.len() as u64).to_le_bytes());
    for diagnostics in &receipt.omitted_calibration {
        hash_omitted(&mut hasher, *diagnostics);
    }
    *hasher.finalize().as_bytes()
}

fn stability_receipt_digest(receipt: &Sem3sReceipt) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003s.receipt-integrity.v1");
    hasher.update(receipt.schema.as_bytes());
    hasher.update(receipt.preregistration_sha.as_bytes());
    hasher.update(receipt.amendment_1_sha.as_bytes());
    hasher.update(receipt.primary_preregistration_sha.as_bytes());
    hasher.update(receipt.primary_amendment_1_sha.as_bytes());
    hasher.update(receipt.primary_implementation_sha.as_bytes());
    hash_string(&mut hasher, &receipt.stability_implementation_sha);
    hasher.update(&receipt.primary_receipt_evidence_digest);
    for digest in [
        receipt.candidate_bank_digest,
        receipt.calibration_query_digest,
        receipt.clean_query_digest,
        receipt.ambiguous_query_digest,
        receipt.unrelated_query_digest,
        receipt.ood_query_digest,
    ] {
        hasher.update(&digest);
    }
    hasher.update(&(receipt.folds.len() as u64).to_le_bytes());
    for fold in &receipt.folds {
        hasher.update(&fold.evidence_digest);
    }
    hasher.update(&(receipt.threshold_stability.len() as u64).to_le_bytes());
    for summary in &receipt.threshold_stability {
        hash_threshold_stability(&mut hasher, *summary);
    }
    hasher.update(&[stability_disposition_tag(receipt.disposition)]);
    *hasher.finalize().as_bytes()
}

fn hash_threshold(hasher: &mut blake3::Hasher, threshold: Sem3Threshold) {
    hasher.update(&(threshold.context_dimension as u64).to_le_bytes());
    hasher.update(&(threshold.calibration_count as u64).to_le_bytes());
    hasher.update(&threshold.nonconformity_threshold.to_bits().to_le_bytes());
}

fn hash_dimension_metrics(hasher: &mut blake3::Hasher, metrics: Sem3DimensionMetrics) {
    for value in [
        metrics.context_dimension,
        metrics.clean_count,
        metrics.ambiguous_count,
        metrics.unrelated_count,
        metrics.ood_count,
        metrics.clean_p95_set_size,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    for value in [
        metrics.clean_target_coverage,
        metrics.clean_singleton_correct_rate,
        metrics.clean_mean_set_size,
        metrics.ambiguous_at_least_one_parent_inclusion,
        metrics.ambiguous_dual_parent_inclusion,
        metrics.ambiguous_singleton_forced_choice_rate,
        metrics.ambiguous_mean_set_size,
        metrics.unrelated_nonempty_set_rate,
        metrics.ood_support_rejection_rate,
        metrics.mean_clean_margin,
        metrics.mean_ambiguous_margin,
        metrics.mean_unrelated_margin,
        metrics.mean_ood_raw_similarity,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_drift(hasher: &mut blake3::Hasher, drift: Sem3sDriftSummary) {
    hasher.update(&(drift.query_count as u64).to_le_bytes());
    hasher.update(&drift.exact_set_agreement_rate.to_bits().to_le_bytes());
    hasher.update(&drift.mean_symmetric_difference_size.to_bits().to_le_bytes());
    hasher.update(&(drift.max_symmetric_difference_size as u64).to_le_bytes());
    hasher.update(&drift.mean_absolute_set_size_difference.to_bits().to_le_bytes());
    hasher.update(&(drift.max_absolute_set_size_difference as u64).to_le_bytes());
}

fn hash_omitted(hasher: &mut blake3::Hasher, diagnostics: Sem3sOmittedCalibrationDiagnostics) {
    hasher.update(&(diagnostics.context_dimension as u64).to_le_bytes());
    hasher.update(&(diagnostics.fold_id as u64).to_le_bytes());
    hasher.update(&(diagnostics.included_count as u64).to_le_bytes());
    hasher.update(&(diagnostics.omitted_count as u64).to_le_bytes());
    hasher.update(&diagnostics.threshold.to_bits().to_le_bytes());
    hasher.update(&diagnostics.omitted_target_coverage.to_bits().to_le_bytes());
    hasher.update(&diagnostics.min_nonconformity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.max_nonconformity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.mean_nonconformity.to_bits().to_le_bytes());
}

fn hash_threshold_stability(
    hasher: &mut blake3::Hasher,
    summary: Sem3sThresholdStabilitySummary,
) {
    hasher.update(&(summary.context_dimension as u64).to_le_bytes());
    hasher.update(&summary.canonical_threshold.to_bits().to_le_bytes());
    hasher.update(&summary.min_fold_threshold.to_bits().to_le_bytes());
    hasher.update(&summary.max_fold_threshold.to_bits().to_le_bytes());
    hasher.update(&summary.mean_fold_threshold.to_bits().to_le_bytes());
    hasher.update(&summary.absolute_threshold_range.to_bits().to_le_bytes());
    hasher.update(
        &summary
            .max_absolute_deviation_from_canonical
            .to_bits()
            .to_le_bytes(),
    );
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn stability_disposition_tag(disposition: Sem3sDisposition) -> u8 {
    match disposition {
        Sem3sDisposition::StablePass => 1,
        Sem3sDisposition::PrimaryPassCalibrationFragile => 2,
        Sem3sDisposition::PrimaryNotPass => 3,
        Sem3sDisposition::IntegrityFailure => 4,
    }
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f64 / values.len() as f64
    }
}

fn mean_f32(values: &[f32]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
    }
}

fn min_f32(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn max_f32(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn is_valid_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_eight_passes_are_required_for_stable_pass() {
        let all_pass = vec![Sem3Disposition::Pass; SYM_RSI_SEM_003S_FOLD_COUNT];
        assert_eq!(
            classify_stability(Sem3Disposition::Pass, &all_pass),
            Sem3sDisposition::StablePass
        );

        let mut one_fragile = all_pass;
        one_fragile[3] = Sem3Disposition::Tradeoff;
        assert_eq!(
            classify_stability(Sem3Disposition::Pass, &one_fragile),
            Sem3sDisposition::PrimaryPassCalibrationFragile
        );
    }

    #[test]
    fn primary_failure_cannot_be_rescued_by_stability_folds() {
        let all_pass = vec![Sem3Disposition::Pass; SYM_RSI_SEM_003S_FOLD_COUNT];
        assert_eq!(
            classify_stability(Sem3Disposition::NotEstablished, &all_pass),
            Sem3sDisposition::PrimaryNotPass
        );
    }

    #[test]
    fn fold_assignment_is_balanced_without_semantic_scores() {
        let corpus = prepare_sem3_corpus().unwrap();
        for fold_id in 0..SYM_RSI_SEM_003S_FOLD_COUNT {
            let (retained, omitted, _) = split_calibration(&corpus, fold_id).unwrap();
            for dimension in SYM_RSI_SEM_003_DIMENSIONS {
                assert_eq!(
                    retained
                        .iter()
                        .filter(|query| query.dimension == dimension)
                        .count(),
                    SYM_RSI_SEM_003S_RETAINED_PER_DIMENSION
                );
                assert_eq!(
                    omitted
                        .iter()
                        .filter(|query| query.dimension == dimension)
                        .count(),
                    SYM_RSI_SEM_003S_OMITTED_PER_DIMENSION
                );
            }
        }
    }

    #[test]
    fn fold_membership_digests_are_distinct() {
        let corpus = prepare_sem3_corpus().unwrap();
        let mut digests = HashSet::new();
        for fold_id in 0..SYM_RSI_SEM_003S_FOLD_COUNT {
            let (_, _, digest) = split_calibration(&corpus, fold_id).unwrap();
            assert!(digests.insert(digest));
        }
        assert_eq!(digests.len(), SYM_RSI_SEM_003S_FOLD_COUNT);
    }

    #[test]
    fn malformed_stability_subject_is_rejected() {
        assert_eq!(
            run_sym_rsi_sem_003s(
                "not-a-sha",
                &run_sym_rsi_sem_003(SYM_RSI_SEM_003_PRIMARY_IMPLEMENTATION_SHA).unwrap()
            )
            .unwrap_err(),
            Sem3sError::InvalidStabilitySubject
        );
    }
}