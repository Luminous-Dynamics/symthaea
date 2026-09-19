// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered SYM-RSI-SEM-003 set-valued semantic-retrieval benchmark.
//!
//! Candidate sets are retrieval/search objects only. Membership grants no evidence,
//! validation, probability, confidence, or authority.

use std::collections::{BTreeMap, HashSet};

use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};
use super::semantic_null_calibration::{
    SemanticNullCalibration, SemanticNullCalibrationError, SemanticNullConfig,
};
use super::semantic_support::{
    SemanticContextSupport, SemanticSupportError, assess_semantic_context_support,
};
use symthaea_core::hdc::BinaryHV;

pub const SYM_RSI_SEM_003_SCHEMA: &str = "symthaea.sym-rsi-sem-003.v1";
pub const SYM_RSI_SEM_003_GENERATOR_VERSION: &str = "semantic-set-generator.v1";
pub const SYM_RSI_SEM_003_PREREGISTRATION_SHA: &str =
    "af8162162e13a2b3c95fc70a0af9b62a3183ac4f";
pub const SYM_RSI_SEM_003_AMENDMENT_1_SHA: &str =
    "418f057387eacf2747edba6e8efaffca05d4b04c";
pub const SYM_RSI_SEM_003_DIMENSIONS: [usize; 3] = [4, 8, 16];
pub const SYM_RSI_SEM_003_CANDIDATES_PER_DIMENSION: usize = 64;
pub const SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION: usize = 32;
pub const SYM_RSI_SEM_003_CLEAN_QUERIES_PER_DIMENSION: usize = 32;
pub const SYM_RSI_SEM_003_AMBIGUOUS_QUERIES_PER_DIMENSION: usize = 24;
pub const SYM_RSI_SEM_003_UNRELATED_QUERIES_PER_DIMENSION: usize = 24;
pub const SYM_RSI_SEM_003_OOD_QUERIES_PER_DIMENSION: usize = 16;
pub const SYM_RSI_SEM_003_ALPHA: f64 = 0.10;
const RATE_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3Threshold {
    pub context_dimension: usize,
    pub calibration_count: usize,
    pub nonconformity_threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sem3SetSizeBin {
    pub set_size: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3RegimeDiagnostics {
    pub query_count: usize,
    pub mean_margin: f64,
    pub p95_margin: f32,
    pub exact_tie_count: usize,
    pub mean_specificity: f64,
    pub p05_specificity: f32,
    pub p50_specificity: f32,
    pub p95_specificity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3OodDiagnostics {
    pub query_count: usize,
    pub support_rejection_count: usize,
    pub support_rejection_rate: f64,
    pub mean_top1_similarity: f64,
    pub max_top1_similarity: f32,
    pub mean_specificity: f64,
    pub max_specificity: f32,
    pub exact_tie_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem3DimensionMetrics {
    pub context_dimension: usize,
    pub clean_count: usize,
    pub ambiguous_count: usize,
    pub unrelated_count: usize,
    pub ood_count: usize,
    pub clean_target_coverage: f64,
    pub clean_singleton_correct_rate: f64,
    pub clean_mean_set_size: f64,
    pub clean_p95_set_size: usize,
    pub ambiguous_at_least_one_parent_inclusion: f64,
    pub ambiguous_dual_parent_inclusion: f64,
    pub ambiguous_singleton_forced_choice_rate: f64,
    pub ambiguous_mean_set_size: f64,
    pub unrelated_nonempty_set_rate: f64,
    pub ood_support_rejection_rate: f64,
    pub mean_clean_margin: f64,
    pub mean_ambiguous_margin: f64,
    pub mean_unrelated_margin: f64,
    pub mean_ood_raw_similarity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sem3Disposition {
    Pass,
    Tradeoff,
    NotEstablished,
    FailClosed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sem3Receipt {
    pub schema: &'static str,
    pub generator_version: &'static str,
    pub preregistration_sha: &'static str,
    pub amendment_1_sha: &'static str,
    pub subject_sha: String,
    pub candidate_bank_digest: [u8; 32],
    pub calibration_query_digest: [u8; 32],
    pub clean_query_digest: [u8; 32],
    pub ambiguous_query_digest: [u8; 32],
    pub unrelated_query_digest: [u8; 32],
    pub ood_query_digest: [u8; 32],
    pub semantic_null_calibration_digest: [u8; 32],
    pub thresholds: Vec<Sem3Threshold>,
    pub dimension_metrics: Vec<Sem3DimensionMetrics>,
    pub clean_set_size_histogram: Vec<Sem3SetSizeBin>,
    pub ambiguous_set_size_histogram: Vec<Sem3SetSizeBin>,
    pub unrelated_set_size_histogram: Vec<Sem3SetSizeBin>,
    pub clean_diagnostics: Sem3RegimeDiagnostics,
    pub ambiguous_diagnostics: Sem3RegimeDiagnostics,
    pub unrelated_diagnostics: Sem3RegimeDiagnostics,
    pub ood_diagnostics: Sem3OodDiagnostics,
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
    pub evidence_digest: [u8; 32],
}

impl Sem3Receipt {
    pub fn evidence_digest_hex(&self) -> String {
        hex::encode(self.evidence_digest)
    }

    pub fn validate_self(&self) -> bool {
        self.schema == SYM_RSI_SEM_003_SCHEMA
            && self.generator_version == SYM_RSI_SEM_003_GENERATOR_VERSION
            && self.preregistration_sha == SYM_RSI_SEM_003_PREREGISTRATION_SHA
            && self.amendment_1_sha == SYM_RSI_SEM_003_AMENDMENT_1_SHA
            && self.evidence_digest == receipt_digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sem3ExperimentError {
    Context(SemanticContextError),
    Support(SemanticSupportError),
    NullCalibration(SemanticNullCalibrationError),
    InvalidSubjectSha,
    MissingThreshold { context_dimension: usize },
    MissingCandidate,
    CorpusIdentityOverlap,
    CorpusCountMismatch,
    CandidateOutsideNominalSupport,
    QueryOutsideNominalSupport,
    CalibrationTargetMismatch,
    InvalidCalibration,
    InsufficientCandidates,
}

impl From<SemanticContextError> for Sem3ExperimentError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

impl From<SemanticSupportError> for Sem3ExperimentError {
    fn from(value: SemanticSupportError) -> Self {
        Self::Support(value)
    }
}

impl From<SemanticNullCalibrationError> for Sem3ExperimentError {
    fn from(value: SemanticNullCalibrationError) -> Self {
        Self::NullCalibration(value)
    }
}

#[derive(Clone)]
pub(super) struct Sem3Candidate {
    pub(super) dimension: usize,
    pub(super) context: Vec<f32>,
    pub(super) identity: ContextIdentity,
    pub(super) semantic_key: BinaryHV,
    pub(super) support: SemanticContextSupport,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3LabelledQuery {
    pub(super) dimension: usize,
    pub(super) context: Vec<f32>,
    pub(super) target: ContextIdentity,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3AmbiguousQuery {
    pub(super) dimension: usize,
    pub(super) context: Vec<f32>,
    pub(super) left_parent: ContextIdentity,
    pub(super) right_parent: ContextIdentity,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3UnlabelledQuery {
    pub(super) dimension: usize,
    pub(super) context: Vec<f32>,
}

#[derive(Clone)]
pub(super) struct Sem3Corpus {
    pub(super) encoder: SemanticContextEncoder,
    pub(super) candidates: Vec<Sem3Candidate>,
    pub(super) calibration: Vec<Sem3LabelledQuery>,
    pub(super) clean: Vec<Sem3LabelledQuery>,
    pub(super) ambiguous: Vec<Sem3AmbiguousQuery>,
    pub(super) unrelated: Vec<Sem3UnlabelledQuery>,
    pub(super) ood: Vec<Sem3UnlabelledQuery>,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3SetObservation {
    pub(super) identities: HashSet<ContextIdentity>,
    pub(super) margin: f32,
    pub(super) exact_tie: bool,
    pub(super) specificity: f32,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3CleanObservation {
    pub(super) base: Sem3SetObservation,
    pub(super) target_included: bool,
    pub(super) singleton_correct: bool,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3AmbiguousObservation {
    pub(super) base: Sem3SetObservation,
    pub(super) at_least_one_parent: bool,
    pub(super) both_parents: bool,
    pub(super) singleton_forced_choice: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Sem3OodObservation {
    pub(super) support_rejected: bool,
    pub(super) top1_similarity: f32,
    pub(super) exact_tie: bool,
    pub(super) specificity: f32,
}

#[derive(Debug, Clone)]
pub(super) struct Sem3Evaluation {
    pub(super) dimension_metrics: Vec<Sem3DimensionMetrics>,
    pub(super) clean: Vec<Sem3CleanObservation>,
    pub(super) ambiguous: Vec<Sem3AmbiguousObservation>,
    pub(super) unrelated: Vec<Sem3SetObservation>,
    pub(super) ood: Vec<Sem3OodObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Sem3PrimarySummary {
    pub(super) clean_target_coverage: f64,
    pub(super) clean_mean_set_size: f64,
    pub(super) ambiguous_at_least_one_parent_inclusion: f64,
    pub(super) ambiguous_dual_parent_inclusion: f64,
    pub(super) ambiguous_singleton_forced_choice_rate: f64,
    pub(super) unrelated_nonempty_set_rate: f64,
    pub(super) ood_support_rejection_rate: f64,
    pub(super) clean_coverage_passed: bool,
    pub(super) per_dimension_clean_coverage_passed: bool,
    pub(super) clean_set_size_guard_passed: bool,
    pub(super) ambiguous_parent_guard_passed: bool,
    pub(super) ambiguous_dual_parent_guard_passed: bool,
    pub(super) ambiguous_singleton_guard_passed: bool,
    pub(super) unrelated_guard_passed: bool,
    pub(super) ood_support_guard_passed: bool,
    pub(super) disposition: Sem3Disposition,
}

/// Execute the preregistered SEM-003 synthetic benchmark.
///
/// A returned receipt is benchmark measurement only. It is not executable
/// qualification of `subject_sha` and grants no evidence/confidence authority.
pub fn run_sym_rsi_sem_003(subject_sha: &str) -> Result<Sem3Receipt, Sem3ExperimentError> {
    validate_subject_sha(subject_sha)?;
    let corpus = prepare_sem3_corpus()?;
    let thresholds = calibrate_sem3_thresholds(&corpus, &corpus.calibration)?;
    let semantic_null = build_semantic_null(&corpus)?;
    let evaluation = evaluate_sem3_with_thresholds(&corpus, &semantic_null, &thresholds)?;
    let primary = summarize_sem3_primary(&evaluation);

    let clean_set_size_histogram = set_size_histogram(
        &evaluation
            .clean
            .iter()
            .map(|observation| observation.base.identities.len())
            .collect::<Vec<_>>(),
    );
    let ambiguous_set_size_histogram = set_size_histogram(
        &evaluation
            .ambiguous
            .iter()
            .map(|observation| observation.base.identities.len())
            .collect::<Vec<_>>(),
    );
    let unrelated_set_size_histogram = set_size_histogram(
        &evaluation
            .unrelated
            .iter()
            .map(|observation| observation.identities.len())
            .collect::<Vec<_>>(),
    );

    let clean_diagnostics = regime_diagnostics(
        &evaluation
            .clean
            .iter()
            .map(|observation| &observation.base)
            .collect::<Vec<_>>(),
    );
    let ambiguous_diagnostics = regime_diagnostics(
        &evaluation
            .ambiguous
            .iter()
            .map(|observation| &observation.base)
            .collect::<Vec<_>>(),
    );
    let unrelated_refs = evaluation.unrelated.iter().collect::<Vec<_>>();
    let unrelated_diagnostics = regime_diagnostics(&unrelated_refs);
    let ood_diagnostics = ood_diagnostics(&evaluation.ood);

    let mut receipt = Sem3Receipt {
        schema: SYM_RSI_SEM_003_SCHEMA,
        generator_version: SYM_RSI_SEM_003_GENERATOR_VERSION,
        preregistration_sha: SYM_RSI_SEM_003_PREREGISTRATION_SHA,
        amendment_1_sha: SYM_RSI_SEM_003_AMENDMENT_1_SHA,
        subject_sha: subject_sha.to_owned(),
        candidate_bank_digest: candidate_bank_digest(&corpus.candidates),
        calibration_query_digest: labelled_query_digest("calibration", &corpus.calibration),
        clean_query_digest: labelled_query_digest("clean", &corpus.clean),
        ambiguous_query_digest: ambiguous_query_digest(&corpus.ambiguous),
        unrelated_query_digest: unlabelled_query_digest("unrelated", &corpus.unrelated),
        ood_query_digest: unlabelled_query_digest("ood", &corpus.ood),
        semantic_null_calibration_digest: semantic_null.identity().digest,
        thresholds,
        dimension_metrics: evaluation.dimension_metrics,
        clean_set_size_histogram,
        ambiguous_set_size_histogram,
        unrelated_set_size_histogram,
        clean_diagnostics,
        ambiguous_diagnostics,
        unrelated_diagnostics,
        ood_diagnostics,
        clean_target_coverage: primary.clean_target_coverage,
        clean_mean_set_size: primary.clean_mean_set_size,
        ambiguous_at_least_one_parent_inclusion: primary.ambiguous_at_least_one_parent_inclusion,
        ambiguous_dual_parent_inclusion: primary.ambiguous_dual_parent_inclusion,
        ambiguous_singleton_forced_choice_rate: primary.ambiguous_singleton_forced_choice_rate,
        unrelated_nonempty_set_rate: primary.unrelated_nonempty_set_rate,
        ood_support_rejection_rate: primary.ood_support_rejection_rate,
        clean_coverage_passed: primary.clean_coverage_passed,
        per_dimension_clean_coverage_passed: primary.per_dimension_clean_coverage_passed,
        clean_set_size_guard_passed: primary.clean_set_size_guard_passed,
        ambiguous_parent_guard_passed: primary.ambiguous_parent_guard_passed,
        ambiguous_dual_parent_guard_passed: primary.ambiguous_dual_parent_guard_passed,
        ambiguous_singleton_guard_passed: primary.ambiguous_singleton_guard_passed,
        unrelated_guard_passed: primary.unrelated_guard_passed,
        ood_support_guard_passed: primary.ood_support_guard_passed,
        disposition: primary.disposition,
        evidence_digest: [0; 32],
    };
    receipt.evidence_digest = receipt_digest(&receipt);
    Ok(receipt)
}

pub(super) fn prepare_sem3_corpus() -> Result<Sem3Corpus, Sem3ExperimentError> {
    let encoder = SemanticContextEncoder::default();
    let candidates = candidate_bank(encoder)?;
    let calibration = calibration_queries(&candidates)?;
    let clean = clean_queries(&candidates)?;
    let ambiguous = ambiguous_queries(&candidates)?;
    let unrelated = unrelated_queries();
    let ood = ood_queries(&candidates)?;

    validate_expected_counts(
        &candidates,
        &calibration,
        &clean,
        &ambiguous,
        &unrelated,
        &ood,
    )?;
    validate_exact_disjointness(
        &candidates,
        &calibration,
        &clean,
        &ambiguous,
        &unrelated,
        &ood,
    )?;

    Ok(Sem3Corpus {
        encoder,
        candidates,
        calibration,
        clean,
        ambiguous,
        unrelated,
        ood,
    })
}

pub(super) fn calibrate_sem3_thresholds(
    corpus: &Sem3Corpus,
    queries: &[Sem3LabelledQuery],
) -> Result<Vec<Sem3Threshold>, Sem3ExperimentError> {
    let mut thresholds = Vec::with_capacity(SYM_RSI_SEM_003_DIMENSIONS.len());
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let bank = corpus
            .candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();
        let dim_queries = queries
            .iter()
            .filter(|query| query.dimension == dimension)
            .collect::<Vec<_>>();
        if dim_queries.is_empty() {
            return Err(Sem3ExperimentError::InvalidCalibration);
        }

        let mut scores = Vec::with_capacity(dim_queries.len());
        for query in dim_queries {
            require_query_support(corpus.encoder, &query.context)?;
            let target = bank
                .iter()
                .find(|candidate| candidate.identity == query.target)
                .ok_or(Sem3ExperimentError::CalibrationTargetMismatch)?;
            if !target.support.within_nominal_support() {
                return Err(Sem3ExperimentError::CandidateOutsideNominalSupport);
            }
            let query_key = corpus.encoder.encode(&query.context)?;
            let similarity = query_key.similarity(&target.semantic_key);
            scores.push((1.0 - similarity).clamp(0.0, 1.0));
        }

        scores.sort_by(|left, right| left.total_cmp(right));
        let n = scores.len();
        let rank = (((n + 1) as f64 * (1.0 - SYM_RSI_SEM_003_ALPHA)).ceil() as usize)
            .clamp(1, n);
        thresholds.push(Sem3Threshold {
            context_dimension: dimension,
            calibration_count: n,
            nonconformity_threshold: scores[rank - 1],
        });
    }
    Ok(thresholds)
}

pub(super) fn build_semantic_null(
    corpus: &Sem3Corpus,
) -> Result<SemanticNullCalibration, Sem3ExperimentError> {
    let contexts = corpus
        .candidates
        .iter()
        .map(|candidate| candidate.context.clone())
        .collect::<Vec<_>>();
    Ok(SemanticNullCalibration::build(
        &contexts,
        corpus.encoder,
        SemanticNullConfig {
            max_reference_contexts_per_dimension: SYM_RSI_SEM_003_CANDIDATES_PER_DIMENSION,
            min_null_pairs_per_dimension: 128,
        },
    )?)
}

pub(super) fn evaluate_sem3_with_thresholds(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    thresholds: &[Sem3Threshold],
) -> Result<Sem3Evaluation, Sem3ExperimentError> {
    let mut dimension_metrics = Vec::with_capacity(SYM_RSI_SEM_003_DIMENSIONS.len());
    let mut all_clean = Vec::new();
    let mut all_ambiguous = Vec::new();
    let mut all_unrelated = Vec::new();
    let mut all_ood = Vec::new();

    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let threshold = threshold_for(dimension, thresholds)?;
        let bank = corpus
            .candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();

        let clean = corpus
            .clean
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| evaluate_clean_query(corpus.encoder, semantic_null, &bank, query, threshold))
            .collect::<Result<Vec<_>, _>>()?;
        let ambiguous = corpus
            .ambiguous
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| {
                evaluate_ambiguous_query(corpus.encoder, semantic_null, &bank, query, threshold)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unrelated = corpus
            .unrelated
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| {
                evaluate_unrelated_query(corpus.encoder, semantic_null, &bank, query, threshold)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ood = corpus
            .ood
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| evaluate_ood_query(corpus.encoder, semantic_null, &bank, query))
            .collect::<Result<Vec<_>, _>>()?;

        dimension_metrics.push(build_dimension_metrics(
            dimension,
            &clean,
            &ambiguous,
            &unrelated,
            &ood,
        ));

        all_clean.extend(clean);
        all_ambiguous.extend(ambiguous);
        all_unrelated.extend(unrelated);
        all_ood.extend(ood);
    }

    Ok(Sem3Evaluation {
        dimension_metrics,
        clean: all_clean,
        ambiguous: all_ambiguous,
        unrelated: all_unrelated,
        ood: all_ood,
    })
}

pub(super) fn summarize_sem3_primary(evaluation: &Sem3Evaluation) -> Sem3PrimarySummary {
    let clean_target_coverage = rate(
        evaluation
            .clean
            .iter()
            .filter(|observation| observation.target_included)
            .count(),
        evaluation.clean.len(),
    );
    let clean_mean_set_size = mean_usize(
        &evaluation
            .clean
            .iter()
            .map(|observation| observation.base.identities.len())
            .collect::<Vec<_>>(),
    );
    let ambiguous_at_least_one_parent_inclusion = rate(
        evaluation
            .ambiguous
            .iter()
            .filter(|observation| observation.at_least_one_parent)
            .count(),
        evaluation.ambiguous.len(),
    );
    let ambiguous_dual_parent_inclusion = rate(
        evaluation
            .ambiguous
            .iter()
            .filter(|observation| observation.both_parents)
            .count(),
        evaluation.ambiguous.len(),
    );
    let ambiguous_singleton_forced_choice_rate = rate(
        evaluation
            .ambiguous
            .iter()
            .filter(|observation| observation.singleton_forced_choice)
            .count(),
        evaluation.ambiguous.len(),
    );
    let unrelated_nonempty_set_rate = rate(
        evaluation
            .unrelated
            .iter()
            .filter(|observation| !observation.identities.is_empty())
            .count(),
        evaluation.unrelated.len(),
    );
    let ood_support_rejection_rate = rate(
        evaluation
            .ood
            .iter()
            .filter(|observation| observation.support_rejected)
            .count(),
        evaluation.ood.len(),
    );

    let clean_coverage_passed = clean_target_coverage + RATE_EPSILON >= 0.90;
    let per_dimension_clean_coverage_passed = evaluation
        .dimension_metrics
        .iter()
        .all(|metrics| metrics.clean_target_coverage + RATE_EPSILON >= 0.85);
    let clean_set_size_guard_passed = clean_mean_set_size <= 2.0 + RATE_EPSILON;
    let ambiguous_parent_guard_passed =
        ambiguous_at_least_one_parent_inclusion + RATE_EPSILON >= 0.90;
    let ambiguous_dual_parent_guard_passed =
        ambiguous_dual_parent_inclusion + RATE_EPSILON >= 0.70;
    let ambiguous_singleton_guard_passed =
        ambiguous_singleton_forced_choice_rate <= 0.20 + RATE_EPSILON;
    let unrelated_guard_passed = unrelated_nonempty_set_rate <= 0.10 + RATE_EPSILON;
    let ood_support_guard_passed = (ood_support_rejection_rate - 1.0).abs() <= RATE_EPSILON;

    let non_ood_guards_passed = clean_coverage_passed
        && per_dimension_clean_coverage_passed
        && clean_set_size_guard_passed
        && ambiguous_parent_guard_passed
        && ambiguous_dual_parent_guard_passed
        && ambiguous_singleton_guard_passed
        && unrelated_guard_passed;

    let disposition = if !ood_support_guard_passed {
        Sem3Disposition::FailClosed
    } else if non_ood_guards_passed {
        Sem3Disposition::Pass
    } else if clean_coverage_passed {
        Sem3Disposition::Tradeoff
    } else {
        Sem3Disposition::NotEstablished
    };

    Sem3PrimarySummary {
        clean_target_coverage,
        clean_mean_set_size,
        ambiguous_at_least_one_parent_inclusion,
        ambiguous_dual_parent_inclusion,
        ambiguous_singleton_forced_choice_rate,
        unrelated_nonempty_set_rate,
        ood_support_rejection_rate,
        clean_coverage_passed,
        per_dimension_clean_coverage_passed,
        clean_set_size_guard_passed,
        ambiguous_parent_guard_passed,
        ambiguous_dual_parent_guard_passed,
        ambiguous_singleton_guard_passed,
        unrelated_guard_passed,
        ood_support_guard_passed,
        disposition,
    }
}

fn candidate_bank(encoder: SemanticContextEncoder) -> Result<Vec<Sem3Candidate>, Sem3ExperimentError> {
    let mut candidates = Vec::with_capacity(
        SYM_RSI_SEM_003_DIMENSIONS.len() * SYM_RSI_SEM_003_CANDIDATES_PER_DIMENSION,
    );
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        for index in 0..SYM_RSI_SEM_003_CANDIDATES_PER_DIMENSION {
            let context = candidate_context(dimension, index);
            let support = assess_semantic_context_support(encoder, &context)?;
            if !support.within_nominal_support() {
                return Err(Sem3ExperimentError::CandidateOutsideNominalSupport);
            }
            candidates.push(Sem3Candidate {
                dimension,
                identity: ContextIdentity::from_context(&context)?,
                semantic_key: encoder.encode(&context)?,
                support,
                context,
            });
        }
    }
    Ok(candidates)
}

fn calibration_queries(
    candidates: &[Sem3Candidate],
) -> Result<Vec<Sem3LabelledQuery>, Sem3ExperimentError> {
    labelled_perturbation_queries(
        candidates,
        0,
        SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION,
        0.025,
        3,
    )
}

fn clean_queries(
    candidates: &[Sem3Candidate],
) -> Result<Vec<Sem3LabelledQuery>, Sem3ExperimentError> {
    labelled_perturbation_queries(
        candidates,
        SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION,
        SYM_RSI_SEM_003_CLEAN_QUERIES_PER_DIMENSION,
        0.045,
        11,
    )
}

fn labelled_perturbation_queries(
    candidates: &[Sem3Candidate],
    skip: usize,
    take: usize,
    magnitude: f32,
    salt: usize,
) -> Result<Vec<Sem3LabelledQuery>, Sem3ExperimentError> {
    let mut queries = Vec::new();
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let bank = candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();
        for target in bank.iter().skip(skip).take(take) {
            let target_index = candidate_index(&bank, target.identity)?;
            queries.push(Sem3LabelledQuery {
                dimension,
                context: perturb_context(&target.context, target_index, magnitude, salt),
                target: target.identity,
            });
        }
    }
    Ok(queries)
}

fn ambiguous_queries(
    candidates: &[Sem3Candidate],
) -> Result<Vec<Sem3AmbiguousQuery>, Sem3ExperimentError> {
    let mut queries = Vec::new();
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let bank = candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();
        for family in 0..SYM_RSI_SEM_003_AMBIGUOUS_QUERIES_PER_DIMENSION {
            let left = bank
                .get(family * 2)
                .ok_or(Sem3ExperimentError::MissingCandidate)?;
            let right = bank
                .get(family * 2 + 1)
                .ok_or(Sem3ExperimentError::MissingCandidate)?;
            let context = left
                .context
                .iter()
                .zip(&right.context)
                .map(|(left_value, right_value)| (left_value + right_value) * 0.5)
                .collect::<Vec<_>>();
            queries.push(Sem3AmbiguousQuery {
                dimension,
                context,
                left_parent: left.identity,
                right_parent: right.identity,
            });
        }
    }
    Ok(queries)
}

fn unrelated_queries() -> Vec<Sem3UnlabelledQuery> {
    let mut queries = Vec::new();
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        for index in 0..SYM_RSI_SEM_003_UNRELATED_QUERIES_PER_DIMENSION {
            queries.push(Sem3UnlabelledQuery {
                dimension,
                context: unrelated_context(dimension, index),
            });
        }
    }
    queries
}

fn ood_queries(
    candidates: &[Sem3Candidate],
) -> Result<Vec<Sem3UnlabelledQuery>, Sem3ExperimentError> {
    let mut queries = Vec::new();
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let bank = candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();
        for offset in 0..SYM_RSI_SEM_003_OOD_QUERIES_PER_DIMENSION {
            let source = bank
                .get(48 + offset)
                .ok_or(Sem3ExperimentError::MissingCandidate)?;
            let mut context = source.context.clone();
            let coordinate = (offset * 7 + dimension) % dimension;
            context[coordinate] = if source.context[coordinate] >= 0.0 {
                1.5 + source.context[coordinate].abs()
            } else {
                -1.5 - source.context[coordinate].abs()
            };
            queries.push(Sem3UnlabelledQuery { dimension, context });
        }
    }
    Ok(queries)
}

fn evaluate_clean_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Sem3Candidate],
    query: &Sem3LabelledQuery,
    threshold: f32,
) -> Result<Sem3CleanObservation, Sem3ExperimentError> {
    require_query_support(encoder, &query.context)?;
    let ranked = candidate_set_and_diagnostics(encoder, semantic_null, bank, &query.context, threshold)?;
    let target_included = ranked.identities.contains(&query.target);
    Ok(Sem3CleanObservation {
        singleton_correct: target_included && ranked.identities.len() == 1,
        target_included,
        base: ranked,
    })
}

fn evaluate_ambiguous_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Sem3Candidate],
    query: &Sem3AmbiguousQuery,
    threshold: f32,
) -> Result<Sem3AmbiguousObservation, Sem3ExperimentError> {
    require_query_support(encoder, &query.context)?;
    let ranked = candidate_set_and_diagnostics(encoder, semantic_null, bank, &query.context, threshold)?;
    let left = ranked.identities.contains(&query.left_parent);
    let right = ranked.identities.contains(&query.right_parent);
    Ok(Sem3AmbiguousObservation {
        singleton_forced_choice: ranked.identities.len() == 1,
        at_least_one_parent: left || right,
        both_parents: left && right,
        base: ranked,
    })
}

fn evaluate_unrelated_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Sem3Candidate],
    query: &Sem3UnlabelledQuery,
    threshold: f32,
) -> Result<Sem3SetObservation, Sem3ExperimentError> {
    require_query_support(encoder, &query.context)?;
    candidate_set_and_diagnostics(encoder, semantic_null, bank, &query.context, threshold)
}

fn evaluate_ood_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Sem3Candidate],
    query: &Sem3UnlabelledQuery,
) -> Result<Sem3OodObservation, Sem3ExperimentError> {
    let support = assess_semantic_context_support(encoder, &query.context)?;
    let query_key = encoder.encode(&query.context)?;
    let (top1, _, exact_tie) = top_two_similarities(&query_key, bank)?;
    let specificity = semantic_null
        .assess(query.dimension, top1)?
        .retrieval_specificity;
    Ok(Sem3OodObservation {
        support_rejected: !support.within_nominal_support(),
        top1_similarity: top1,
        exact_tie,
        specificity,
    })
}

pub(super) fn sem3_candidate_set_identities(
    corpus: &Sem3Corpus,
    semantic_null: &SemanticNullCalibration,
    dimension: usize,
    context: &[f32],
    threshold: f32,
) -> Result<HashSet<ContextIdentity>, Sem3ExperimentError> {
    require_query_support(corpus.encoder, context)?;
    let bank = corpus
        .candidates
        .iter()
        .filter(|candidate| candidate.dimension == dimension)
        .collect::<Vec<_>>();
    Ok(candidate_set_and_diagnostics(
        corpus.encoder,
        semantic_null,
        &bank,
        context,
        threshold,
    )?
    .identities)
}

fn candidate_set_and_diagnostics(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Sem3Candidate],
    query: &[f32],
    threshold: f32,
) -> Result<Sem3SetObservation, Sem3ExperimentError> {
    let query_key = encoder.encode(query)?;
    let mut identities = HashSet::new();
    for candidate in bank {
        if !candidate.support.within_nominal_support() {
            return Err(Sem3ExperimentError::CandidateOutsideNominalSupport);
        }
        let similarity = query_key.similarity(&candidate.semantic_key);
        if 1.0 - similarity <= threshold {
            identities.insert(candidate.identity);
        }
    }
    let (top1, runner_up, exact_tie) = top_two_similarities(&query_key, bank)?;
    let specificity = semantic_null
        .assess(query.len(), top1)?
        .retrieval_specificity;
    Ok(Sem3SetObservation {
        identities,
        margin: (top1 - runner_up).max(0.0),
        exact_tie,
        specificity,
    })
}

fn top_two_similarities(
    query_key: &BinaryHV,
    bank: &[&Sem3Candidate],
) -> Result<(f32, f32, bool), Sem3ExperimentError> {
    if bank.len() < 2 {
        return Err(Sem3ExperimentError::InsufficientCandidates);
    }
    let mut similarities = bank
        .iter()
        .map(|candidate| query_key.similarity(&candidate.semantic_key))
        .collect::<Vec<_>>();
    similarities.sort_by(|left, right| right.total_cmp(left));
    let top1 = similarities[0];
    let runner_up = similarities[1];
    Ok((top1, runner_up, top1.to_bits() == runner_up.to_bits()))
}

fn require_query_support(
    encoder: SemanticContextEncoder,
    context: &[f32],
) -> Result<(), Sem3ExperimentError> {
    if !assess_semantic_context_support(encoder, context)?.within_nominal_support() {
        return Err(Sem3ExperimentError::QueryOutsideNominalSupport);
    }
    Ok(())
}

fn build_dimension_metrics(
    dimension: usize,
    clean: &[Sem3CleanObservation],
    ambiguous: &[Sem3AmbiguousObservation],
    unrelated: &[Sem3SetObservation],
    ood: &[Sem3OodObservation],
) -> Sem3DimensionMetrics {
    let clean_sizes = clean
        .iter()
        .map(|observation| observation.base.identities.len())
        .collect::<Vec<_>>();
    Sem3DimensionMetrics {
        context_dimension: dimension,
        clean_count: clean.len(),
        ambiguous_count: ambiguous.len(),
        unrelated_count: unrelated.len(),
        ood_count: ood.len(),
        clean_target_coverage: rate(
            clean.iter().filter(|observation| observation.target_included).count(),
            clean.len(),
        ),
        clean_singleton_correct_rate: rate(
            clean
                .iter()
                .filter(|observation| observation.singleton_correct)
                .count(),
            clean.len(),
        ),
        clean_mean_set_size: mean_usize(&clean_sizes),
        clean_p95_set_size: percentile_usize(&clean_sizes, 0.95),
        ambiguous_at_least_one_parent_inclusion: rate(
            ambiguous
                .iter()
                .filter(|observation| observation.at_least_one_parent)
                .count(),
            ambiguous.len(),
        ),
        ambiguous_dual_parent_inclusion: rate(
            ambiguous
                .iter()
                .filter(|observation| observation.both_parents)
                .count(),
            ambiguous.len(),
        ),
        ambiguous_singleton_forced_choice_rate: rate(
            ambiguous
                .iter()
                .filter(|observation| observation.singleton_forced_choice)
                .count(),
            ambiguous.len(),
        ),
        ambiguous_mean_set_size: mean_usize(
            &ambiguous
                .iter()
                .map(|observation| observation.base.identities.len())
                .collect::<Vec<_>>(),
        ),
        unrelated_nonempty_set_rate: rate(
            unrelated
                .iter()
                .filter(|observation| !observation.identities.is_empty())
                .count(),
            unrelated.len(),
        ),
        ood_support_rejection_rate: rate(
            ood.iter().filter(|observation| observation.support_rejected).count(),
            ood.len(),
        ),
        mean_clean_margin: mean_f32(
            &clean.iter().map(|observation| observation.base.margin).collect::<Vec<_>>(),
        ),
        mean_ambiguous_margin: mean_f32(
            &ambiguous
                .iter()
                .map(|observation| observation.base.margin)
                .collect::<Vec<_>>(),
        ),
        mean_unrelated_margin: mean_f32(
            &unrelated.iter().map(|observation| observation.margin).collect::<Vec<_>>(),
        ),
        mean_ood_raw_similarity: mean_f32(
            &ood.iter().map(|observation| observation.top1_similarity).collect::<Vec<_>>(),
        ),
    }
}

fn regime_diagnostics(observations: &[&Sem3SetObservation]) -> Sem3RegimeDiagnostics {
    let margins = observations.iter().map(|observation| observation.margin).collect::<Vec<_>>();
    let specificities = observations
        .iter()
        .map(|observation| observation.specificity)
        .collect::<Vec<_>>();
    Sem3RegimeDiagnostics {
        query_count: observations.len(),
        mean_margin: mean_f32(&margins),
        p95_margin: percentile_f32(&margins, 0.95),
        exact_tie_count: observations.iter().filter(|observation| observation.exact_tie).count(),
        mean_specificity: mean_f32(&specificities),
        p05_specificity: percentile_f32(&specificities, 0.05),
        p50_specificity: percentile_f32(&specificities, 0.50),
        p95_specificity: percentile_f32(&specificities, 0.95),
    }
}

fn ood_diagnostics(observations: &[Sem3OodObservation]) -> Sem3OodDiagnostics {
    let similarities = observations
        .iter()
        .map(|observation| observation.top1_similarity)
        .collect::<Vec<_>>();
    let specificities = observations
        .iter()
        .map(|observation| observation.specificity)
        .collect::<Vec<_>>();
    let support_rejection_count = observations
        .iter()
        .filter(|observation| observation.support_rejected)
        .count();
    Sem3OodDiagnostics {
        query_count: observations.len(),
        support_rejection_count,
        support_rejection_rate: rate(support_rejection_count, observations.len()),
        mean_top1_similarity: mean_f32(&similarities),
        max_top1_similarity: max_f32(&similarities),
        mean_specificity: mean_f32(&specificities),
        max_specificity: max_f32(&specificities),
        exact_tie_count: observations.iter().filter(|observation| observation.exact_tie).count(),
    }
}

fn set_size_histogram(values: &[usize]) -> Vec<Sem3SetSizeBin> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for &value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(set_size, count)| Sem3SetSizeBin { set_size, count })
        .collect()
}

pub(super) fn threshold_for(
    dimension: usize,
    thresholds: &[Sem3Threshold],
) -> Result<f32, Sem3ExperimentError> {
    thresholds
        .iter()
        .find(|threshold| threshold.context_dimension == dimension)
        .map(|threshold| threshold.nonconformity_threshold)
        .ok_or(Sem3ExperimentError::MissingThreshold { context_dimension: dimension })
}

fn candidate_index(
    bank: &[&Sem3Candidate],
    identity: ContextIdentity,
) -> Result<usize, Sem3ExperimentError> {
    bank.iter()
        .position(|candidate| candidate.identity == identity)
        .ok_or(Sem3ExperimentError::MissingCandidate)
}

fn candidate_context(dimension: usize, index: usize) -> Vec<f32> {
    let family = index / 2;
    let side = index % 2;
    let mut context = family_center(dimension, family);
    let coordinate = (family * 5 + dimension) % dimension;
    context[coordinate] += if side == 0 { -0.08 } else { 0.08 };
    context
}

fn family_center(dimension: usize, family: usize) -> Vec<f32> {
    (0..dimension)
        .map(|coordinate| {
            let mixed = dimension
                .wrapping_mul(131)
                .wrapping_add(family.wrapping_mul(47))
                .wrapping_add(coordinate.wrapping_mul(31))
                .wrapping_add(coordinate.wrapping_mul(coordinate).wrapping_mul(7));
            (mixed % 1201) as f32 / 1000.0 - 0.6
        })
        .collect()
}

fn unrelated_context(dimension: usize, index: usize) -> Vec<f32> {
    (0..dimension)
        .map(|coordinate| {
            let mixed = 0x9e37usize
                .wrapping_add(dimension.wrapping_mul(173))
                .wrapping_add(index.wrapping_mul(89))
                .wrapping_add(coordinate.wrapping_mul(61))
                .wrapping_add(coordinate.wrapping_mul(coordinate).wrapping_mul(13));
            (mixed % 1201) as f32 / 1000.0 - 0.6
        })
        .collect()
}

fn perturb_context(source: &[f32], index: usize, magnitude: f32, salt: usize) -> Vec<f32> {
    let mut query = source.to_vec();
    let coordinate = (index.wrapping_mul(7).wrapping_add(salt)) % query.len();
    let direction = if (index + salt) % 2 == 0 { 1.0 } else { -1.0 };
    query[coordinate] = (query[coordinate] + direction * magnitude).clamp(-0.95, 0.95);
    query
}

fn validate_expected_counts(
    candidates: &[Sem3Candidate],
    calibration: &[Sem3LabelledQuery],
    clean: &[Sem3LabelledQuery],
    ambiguous: &[Sem3AmbiguousQuery],
    unrelated: &[Sem3UnlabelledQuery],
    ood: &[Sem3UnlabelledQuery],
) -> Result<(), Sem3ExperimentError> {
    let dimensions = SYM_RSI_SEM_003_DIMENSIONS.len();
    let valid = candidates.len() == dimensions * SYM_RSI_SEM_003_CANDIDATES_PER_DIMENSION
        && calibration.len() == dimensions * SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION
        && clean.len() == dimensions * SYM_RSI_SEM_003_CLEAN_QUERIES_PER_DIMENSION
        && ambiguous.len() == dimensions * SYM_RSI_SEM_003_AMBIGUOUS_QUERIES_PER_DIMENSION
        && unrelated.len() == dimensions * SYM_RSI_SEM_003_UNRELATED_QUERIES_PER_DIMENSION
        && ood.len() == dimensions * SYM_RSI_SEM_003_OOD_QUERIES_PER_DIMENSION;
    if valid {
        Ok(())
    } else {
        Err(Sem3ExperimentError::CorpusCountMismatch)
    }
}

fn validate_exact_disjointness(
    candidates: &[Sem3Candidate],
    calibration: &[Sem3LabelledQuery],
    clean: &[Sem3LabelledQuery],
    ambiguous: &[Sem3AmbiguousQuery],
    unrelated: &[Sem3UnlabelledQuery],
    ood: &[Sem3UnlabelledQuery],
) -> Result<(), Sem3ExperimentError> {
    let mut seen = HashSet::<[u8; 32]>::new();
    for candidate in candidates {
        if !seen.insert(candidate.identity.exact_digest) {
            return Err(Sem3ExperimentError::CorpusIdentityOverlap);
        }
    }
    for query in calibration {
        insert_query_identity(&mut seen, &query.context)?;
    }
    for query in clean {
        insert_query_identity(&mut seen, &query.context)?;
    }
    for query in ambiguous {
        insert_query_identity(&mut seen, &query.context)?;
    }
    for query in unrelated {
        insert_query_identity(&mut seen, &query.context)?;
    }
    for query in ood {
        insert_query_identity(&mut seen, &query.context)?;
    }
    Ok(())
}

fn insert_query_identity(
    seen: &mut HashSet<[u8; 32]>,
    context: &[f32],
) -> Result<(), Sem3ExperimentError> {
    let identity = ContextIdentity::from_context(context)?;
    if !seen.insert(identity.exact_digest) {
        return Err(Sem3ExperimentError::CorpusIdentityOverlap);
    }
    Ok(())
}

pub(super) fn candidate_bank_digest(candidates: &[Sem3Candidate]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003.candidate-bank.v1");
    for candidate in candidates {
        hasher.update(&(candidate.dimension as u64).to_le_bytes());
        hash_context(&mut hasher, &candidate.context);
        hasher.update(&candidate.identity.exact_digest);
        hash_support(&mut hasher, candidate.support);
    }
    *hasher.finalize().as_bytes()
}

pub(super) fn labelled_query_digest(domain: &str, queries: &[Sem3LabelledQuery]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003.labelled-queries.v1");
    hash_string(&mut hasher, domain);
    for query in queries {
        hasher.update(&(query.dimension as u64).to_le_bytes());
        hash_context(&mut hasher, &query.context);
        hasher.update(&query.target.exact_digest);
    }
    *hasher.finalize().as_bytes()
}

pub(super) fn ambiguous_query_digest(queries: &[Sem3AmbiguousQuery]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003.ambiguous-queries.v1");
    for query in queries {
        hasher.update(&(query.dimension as u64).to_le_bytes());
        hash_context(&mut hasher, &query.context);
        hasher.update(&query.left_parent.exact_digest);
        hasher.update(&query.right_parent.exact_digest);
    }
    *hasher.finalize().as_bytes()
}

pub(super) fn unlabelled_query_digest(
    domain: &str,
    queries: &[Sem3UnlabelledQuery],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003.unlabelled-queries.v1");
    hash_string(&mut hasher, domain);
    for query in queries {
        hasher.update(&(query.dimension as u64).to_le_bytes());
        hash_context(&mut hasher, &query.context);
    }
    *hasher.finalize().as_bytes()
}

fn receipt_digest(receipt: &Sem3Receipt) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-003.receipt-integrity.v1");
    hasher.update(receipt.schema.as_bytes());
    hasher.update(receipt.generator_version.as_bytes());
    hasher.update(receipt.preregistration_sha.as_bytes());
    hasher.update(receipt.amendment_1_sha.as_bytes());
    hasher.update(receipt.subject_sha.as_bytes());
    for digest in [
        receipt.candidate_bank_digest,
        receipt.calibration_query_digest,
        receipt.clean_query_digest,
        receipt.ambiguous_query_digest,
        receipt.unrelated_query_digest,
        receipt.ood_query_digest,
        receipt.semantic_null_calibration_digest,
    ] {
        hasher.update(&digest);
    }
    for threshold in &receipt.thresholds {
        hash_threshold(&mut hasher, *threshold);
    }
    for metrics in &receipt.dimension_metrics {
        hash_dimension_metrics(&mut hasher, *metrics);
    }
    hash_histogram(&mut hasher, &receipt.clean_set_size_histogram);
    hash_histogram(&mut hasher, &receipt.ambiguous_set_size_histogram);
    hash_histogram(&mut hasher, &receipt.unrelated_set_size_histogram);
    hash_regime_diagnostics(&mut hasher, receipt.clean_diagnostics);
    hash_regime_diagnostics(&mut hasher, receipt.ambiguous_diagnostics);
    hash_regime_diagnostics(&mut hasher, receipt.unrelated_diagnostics);
    hash_ood_diagnostics(&mut hasher, receipt.ood_diagnostics);
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

fn hash_histogram(hasher: &mut blake3::Hasher, bins: &[Sem3SetSizeBin]) {
    hasher.update(&(bins.len() as u64).to_le_bytes());
    for bin in bins {
        hasher.update(&(bin.set_size as u64).to_le_bytes());
        hasher.update(&(bin.count as u64).to_le_bytes());
    }
}

fn hash_regime_diagnostics(hasher: &mut blake3::Hasher, diagnostics: Sem3RegimeDiagnostics) {
    hasher.update(&(diagnostics.query_count as u64).to_le_bytes());
    hasher.update(&diagnostics.mean_margin.to_bits().to_le_bytes());
    hasher.update(&diagnostics.p95_margin.to_bits().to_le_bytes());
    hasher.update(&(diagnostics.exact_tie_count as u64).to_le_bytes());
    hasher.update(&diagnostics.mean_specificity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.p05_specificity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.p50_specificity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.p95_specificity.to_bits().to_le_bytes());
}

fn hash_ood_diagnostics(hasher: &mut blake3::Hasher, diagnostics: Sem3OodDiagnostics) {
    hasher.update(&(diagnostics.query_count as u64).to_le_bytes());
    hasher.update(&(diagnostics.support_rejection_count as u64).to_le_bytes());
    hasher.update(&diagnostics.support_rejection_rate.to_bits().to_le_bytes());
    hasher.update(&diagnostics.mean_top1_similarity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.max_top1_similarity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.mean_specificity.to_bits().to_le_bytes());
    hasher.update(&diagnostics.max_specificity.to_bits().to_le_bytes());
    hasher.update(&(diagnostics.exact_tie_count as u64).to_le_bytes());
}

fn hash_support(hasher: &mut blake3::Hasher, support: SemanticContextSupport) {
    hasher.update(&(support.context_dimension as u64).to_le_bytes());
    hasher.update(&support.clamp_abs.to_bits().to_le_bytes());
    hasher.update(&support.max_abs_input.to_bits().to_le_bytes());
    hasher.update(&(support.clipped_value_count as u64).to_le_bytes());
    hasher.update(&support.clipped_fraction.to_bits().to_le_bytes());
    hasher.update(&support.max_overflow_abs.to_bits().to_le_bytes());
    hasher.update(&support.mean_overflow_abs.to_bits().to_le_bytes());
}

pub(super) fn hash_context(hasher: &mut blake3::Hasher, context: &[f32]) {
    hasher.update(&(context.len() as u64).to_le_bytes());
    for value in context {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

pub(super) fn disposition_tag(disposition: Sem3Disposition) -> u8 {
    match disposition {
        Sem3Disposition::Pass => 1,
        Sem3Disposition::Tradeoff => 2,
        Sem3Disposition::NotEstablished => 3,
        Sem3Disposition::FailClosed => 4,
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
        values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64
    }
}

fn max_f32(values: &[f32]) -> f32 {
    values.iter().copied().fold(0.0_f32, f32::max)
}

fn percentile_usize(values: &[usize], probability: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((sorted.len() as f64 * probability).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn percentile_f32(values: &[f32], probability: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let rank = ((sorted.len() as f64 * probability).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn validate_subject_sha(subject_sha: &str) -> Result<(), Sem3ExperimentError> {
    if subject_sha.len() != 40 || !subject_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Sem3ExperimentError::InvalidSubjectSha);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUBJECT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn sem3_receipt_binds_protocol_and_self_validates() {
        let receipt = run_sym_rsi_sem_003(SUBJECT).unwrap();
        assert_eq!(receipt.preregistration_sha, SYM_RSI_SEM_003_PREREGISTRATION_SHA);
        assert_eq!(receipt.amendment_1_sha, SYM_RSI_SEM_003_AMENDMENT_1_SHA);
        assert!(receipt.validate_self());
        assert_eq!(receipt.thresholds.len(), SYM_RSI_SEM_003_DIMENSIONS.len());
        assert_eq!(receipt.dimension_metrics.len(), SYM_RSI_SEM_003_DIMENSIONS.len());
    }

    #[test]
    fn sem3_histograms_reconcile_to_frozen_query_counts() {
        let receipt = run_sym_rsi_sem_003(SUBJECT).unwrap();
        let clean_count = receipt
            .clean_set_size_histogram
            .iter()
            .map(|bin| bin.count)
            .sum::<usize>();
        let ambiguous_count = receipt
            .ambiguous_set_size_histogram
            .iter()
            .map(|bin| bin.count)
            .sum::<usize>();
        let unrelated_count = receipt
            .unrelated_set_size_histogram
            .iter()
            .map(|bin| bin.count)
            .sum::<usize>();
        assert_eq!(
            clean_count,
            SYM_RSI_SEM_003_DIMENSIONS.len() * SYM_RSI_SEM_003_CLEAN_QUERIES_PER_DIMENSION
        );
        assert_eq!(
            ambiguous_count,
            SYM_RSI_SEM_003_DIMENSIONS.len() * SYM_RSI_SEM_003_AMBIGUOUS_QUERIES_PER_DIMENSION
        );
        assert_eq!(
            unrelated_count,
            SYM_RSI_SEM_003_DIMENSIONS.len() * SYM_RSI_SEM_003_UNRELATED_QUERIES_PER_DIMENSION
        );
    }

    #[test]
    fn sem3_ood_queries_are_always_support_rejected() {
        let receipt = run_sym_rsi_sem_003(SUBJECT).unwrap();
        assert!((receipt.ood_support_rejection_rate - 1.0).abs() <= RATE_EPSILON);
        assert!(receipt.ood_support_guard_passed);
        assert_eq!(
            receipt.ood_diagnostics.support_rejection_count,
            SYM_RSI_SEM_003_DIMENSIONS.len() * SYM_RSI_SEM_003_OOD_QUERIES_PER_DIMENSION
        );
    }

    #[test]
    fn sem3_tampered_metric_breaks_receipt_integrity() {
        let mut receipt = run_sym_rsi_sem_003(SUBJECT).unwrap();
        receipt.clean_target_coverage =
            f64::from_bits(receipt.clean_target_coverage.to_bits() ^ 1);
        assert!(!receipt.validate_self());
    }

    #[test]
    fn sem3_calibration_queries_pass_support_gate() {
        let corpus = prepare_sem3_corpus().unwrap();
        for query in &corpus.calibration {
            require_query_support(corpus.encoder, &query.context).unwrap();
        }
        let thresholds = calibrate_sem3_thresholds(&corpus, &corpus.calibration).unwrap();
        assert!(thresholds.iter().all(|threshold| {
            threshold.calibration_count == SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION
        }));
    }

    #[test]
    fn sem3_primary_summary_matches_receipt_surface() {
        let corpus = prepare_sem3_corpus().unwrap();
        let thresholds = calibrate_sem3_thresholds(&corpus, &corpus.calibration).unwrap();
        let semantic_null = build_semantic_null(&corpus).unwrap();
        let evaluation = evaluate_sem3_with_thresholds(&corpus, &semantic_null, &thresholds).unwrap();
        let primary = summarize_sem3_primary(&evaluation);
        let receipt = run_sym_rsi_sem_003(SUBJECT).unwrap();
        assert_eq!(primary.clean_target_coverage, receipt.clean_target_coverage);
        assert_eq!(primary.clean_mean_set_size, receipt.clean_mean_set_size);
        assert_eq!(primary.disposition, receipt.disposition);
    }
}