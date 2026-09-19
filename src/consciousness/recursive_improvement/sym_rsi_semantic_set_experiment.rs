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
struct Candidate {
    dimension: usize,
    context: Vec<f32>,
    identity: ContextIdentity,
    semantic_key: BinaryHV,
    support: SemanticContextSupport,
}

#[derive(Debug, Clone)]
struct LabelledQuery {
    dimension: usize,
    context: Vec<f32>,
    target: ContextIdentity,
}

#[derive(Debug, Clone)]
struct AmbiguousQuery {
    dimension: usize,
    context: Vec<f32>,
    left_parent: ContextIdentity,
    right_parent: ContextIdentity,
}

#[derive(Debug, Clone)]
struct UnlabelledQuery {
    dimension: usize,
    context: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct SetObservation {
    set_size: usize,
    margin: f32,
    exact_tie: bool,
    specificity: f32,
}

#[derive(Debug, Clone, Copy)]
struct CleanObservation {
    base: SetObservation,
    target_included: bool,
    singleton_correct: bool,
}

#[derive(Debug, Clone, Copy)]
struct AmbiguousObservation {
    base: SetObservation,
    at_least_one_parent: bool,
    both_parents: bool,
    singleton_forced_choice: bool,
}

#[derive(Debug, Clone, Copy)]
struct OodObservation {
    support_rejected: bool,
    top1_similarity: f32,
    exact_tie: bool,
    specificity: f32,
}

#[derive(Debug)]
struct RankedSet {
    identities: HashSet<ContextIdentity>,
    margin: f32,
    exact_tie: bool,
    specificity: f32,
}

/// Execute the preregistered SEM-003 synthetic benchmark.
///
/// A returned receipt is benchmark measurement only. It is not executable
/// qualification of `subject_sha` and grants no evidence/confidence authority.
pub fn run_sym_rsi_sem_003(subject_sha: &str) -> Result<Sem3Receipt, Sem3ExperimentError> {
    validate_subject_sha(subject_sha)?;
    let encoder = SemanticContextEncoder::default();

    let candidates = candidate_bank(encoder)?;
    let calibration = calibration_queries(&candidates)?;
    let clean = clean_queries(&candidates)?;
    let ambiguous = ambiguous_queries(&candidates)?;
    let unrelated = unrelated_queries();
    let ood = ood_queries(&candidates)?;

    validate_exact_disjointness(
        &candidates,
        &calibration,
        &clean,
        &ambiguous,
        &unrelated,
        &ood,
    )?;

    let thresholds = calibrate_thresholds(encoder, &candidates, &calibration)?;
    let semantic_null = build_semantic_null(encoder, &candidates)?;

    let mut dimension_metrics = Vec::with_capacity(SYM_RSI_SEM_003_DIMENSIONS.len());
    let mut all_clean = Vec::new();
    let mut all_ambiguous = Vec::new();
    let mut all_unrelated = Vec::new();
    let mut all_ood = Vec::new();

    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let threshold = threshold_for(dimension, &thresholds)?;
        let bank = candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();

        let clean_observations = clean
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| evaluate_clean_query(encoder, &semantic_null, &bank, query, threshold))
            .collect::<Result<Vec<_>, _>>()?;
        let ambiguous_observations = ambiguous
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| evaluate_ambiguous_query(encoder, &semantic_null, &bank, query, threshold))
            .collect::<Result<Vec<_>, _>>()?;
        let unrelated_observations = unrelated
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| evaluate_unrelated_query(encoder, &semantic_null, &bank, query, threshold))
            .collect::<Result<Vec<_>, _>>()?;
        let ood_observations = ood
            .iter()
            .filter(|query| query.dimension == dimension)
            .map(|query| evaluate_ood_query(encoder, &semantic_null, &bank, query))
            .collect::<Result<Vec<_>, _>>()?;

        dimension_metrics.push(build_dimension_metrics(
            dimension,
            &clean_observations,
            &ambiguous_observations,
            &unrelated_observations,
            &ood_observations,
        ));

        all_clean.extend(clean_observations);
        all_ambiguous.extend(ambiguous_observations);
        all_unrelated.extend(unrelated_observations);
        all_ood.extend(ood_observations);
    }

    let clean_target_coverage = rate(
        all_clean
            .iter()
            .filter(|observation| observation.target_included)
            .count(),
        all_clean.len(),
    );
    let clean_mean_set_size = mean_usize(
        &all_clean
            .iter()
            .map(|observation| observation.base.set_size)
            .collect::<Vec<_>>(),
    );
    let ambiguous_at_least_one_parent_inclusion = rate(
        all_ambiguous
            .iter()
            .filter(|observation| observation.at_least_one_parent)
            .count(),
        all_ambiguous.len(),
    );
    let ambiguous_dual_parent_inclusion = rate(
        all_ambiguous
            .iter()
            .filter(|observation| observation.both_parents)
            .count(),
        all_ambiguous.len(),
    );
    let ambiguous_singleton_forced_choice_rate = rate(
        all_ambiguous
            .iter()
            .filter(|observation| observation.singleton_forced_choice)
            .count(),
        all_ambiguous.len(),
    );
    let unrelated_nonempty_set_rate = rate(
        all_unrelated
            .iter()
            .filter(|observation| observation.set_size > 0)
            .count(),
        all_unrelated.len(),
    );
    let ood_support_rejection_rate = rate(
        all_ood
            .iter()
            .filter(|observation| observation.support_rejected)
            .count(),
        all_ood.len(),
    );

    let clean_coverage_passed = clean_target_coverage + RATE_EPSILON >= 0.90;
    let per_dimension_clean_coverage_passed = dimension_metrics
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

    let clean_set_size_histogram = set_size_histogram(
        &all_clean
            .iter()
            .map(|observation| observation.base.set_size)
            .collect::<Vec<_>>(),
    );
    let ambiguous_set_size_histogram = set_size_histogram(
        &all_ambiguous
            .iter()
            .map(|observation| observation.base.set_size)
            .collect::<Vec<_>>(),
    );
    let unrelated_set_size_histogram = set_size_histogram(
        &all_unrelated
            .iter()
            .map(|observation| observation.set_size)
            .collect::<Vec<_>>(),
    );

    let clean_diagnostics = regime_diagnostics(
        &all_clean
            .iter()
            .map(|observation| observation.base)
            .collect::<Vec<_>>(),
    );
    let ambiguous_diagnostics = regime_diagnostics(
        &all_ambiguous
            .iter()
            .map(|observation| observation.base)
            .collect::<Vec<_>>(),
    );
    let unrelated_diagnostics = regime_diagnostics(&all_unrelated);
    let ood_diagnostics = ood_diagnostics(&all_ood);

    let mut receipt = Sem3Receipt {
        schema: SYM_RSI_SEM_003_SCHEMA,
        generator_version: SYM_RSI_SEM_003_GENERATOR_VERSION,
        preregistration_sha: SYM_RSI_SEM_003_PREREGISTRATION_SHA,
        amendment_1_sha: SYM_RSI_SEM_003_AMENDMENT_1_SHA,
        subject_sha: subject_sha.to_owned(),
        candidate_bank_digest: candidate_bank_digest(&candidates),
        calibration_query_digest: labelled_query_digest("calibration", &calibration),
        clean_query_digest: labelled_query_digest("clean", &clean),
        ambiguous_query_digest: ambiguous_query_digest(&ambiguous),
        unrelated_query_digest: unlabelled_query_digest("unrelated", &unrelated),
        ood_query_digest: unlabelled_query_digest("ood", &ood),
        semantic_null_calibration_digest: semantic_null.identity().digest,
        thresholds,
        dimension_metrics,
        clean_set_size_histogram,
        ambiguous_set_size_histogram,
        unrelated_set_size_histogram,
        clean_diagnostics,
        ambiguous_diagnostics,
        unrelated_diagnostics,
        ood_diagnostics,
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
        evidence_digest: [0; 32],
    };
    receipt.evidence_digest = receipt_digest(&receipt);
    Ok(receipt)
}

fn candidate_bank(encoder: SemanticContextEncoder) -> Result<Vec<Candidate>, Sem3ExperimentError> {
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
            candidates.push(Candidate {
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

fn build_semantic_null(
    encoder: SemanticContextEncoder,
    candidates: &[Candidate],
) -> Result<SemanticNullCalibration, Sem3ExperimentError> {
    let contexts = candidates
        .iter()
        .map(|candidate| candidate.context.clone())
        .collect::<Vec<_>>();
    Ok(SemanticNullCalibration::build(
        &contexts,
        encoder,
        SemanticNullConfig {
            max_reference_contexts_per_dimension: SYM_RSI_SEM_003_CANDIDATES_PER_DIMENSION,
            min_null_pairs_per_dimension: 128,
        },
    )?)
}

fn calibration_queries(candidates: &[Candidate]) -> Result<Vec<LabelledQuery>, Sem3ExperimentError> {
    labelled_perturbation_queries(
        candidates,
        0,
        SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION,
        0.025,
        3,
    )
}

fn clean_queries(candidates: &[Candidate]) -> Result<Vec<LabelledQuery>, Sem3ExperimentError> {
    labelled_perturbation_queries(
        candidates,
        SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION,
        SYM_RSI_SEM_003_CLEAN_QUERIES_PER_DIMENSION,
        0.045,
        11,
    )
}

fn labelled_perturbation_queries(
    candidates: &[Candidate],
    skip: usize,
    take: usize,
    magnitude: f32,
    salt: usize,
) -> Result<Vec<LabelledQuery>, Sem3ExperimentError> {
    let mut queries = Vec::new();
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let bank = candidates
            .iter()
            .filter(|candidate| candidate.dimension == dimension)
            .collect::<Vec<_>>();
        for target in bank.iter().skip(skip).take(take) {
            let target_index = candidate_index(&bank, target.identity)?;
            queries.push(LabelledQuery {
                dimension,
                context: perturb_context(&target.context, target_index, magnitude, salt),
                target: target.identity,
            });
        }
    }
    Ok(queries)
}

fn ambiguous_queries(candidates: &[Candidate]) -> Result<Vec<AmbiguousQuery>, Sem3ExperimentError> {
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
            queries.push(AmbiguousQuery {
                dimension,
                context,
                left_parent: left.identity,
                right_parent: right.identity,
            });
        }
    }
    Ok(queries)
}

fn unrelated_queries() -> Vec<UnlabelledQuery> {
    let mut queries = Vec::new();
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        for index in 0..SYM_RSI_SEM_003_UNRELATED_QUERIES_PER_DIMENSION {
            queries.push(UnlabelledQuery {
                dimension,
                context: unrelated_context(dimension, index),
            });
        }
    }
    queries
}

fn ood_queries(candidates: &[Candidate]) -> Result<Vec<UnlabelledQuery>, Sem3ExperimentError> {
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
            queries.push(UnlabelledQuery { dimension, context });
        }
    }
    Ok(queries)
}

fn calibrate_thresholds(
    encoder: SemanticContextEncoder,
    candidates: &[Candidate],
    queries: &[LabelledQuery],
) -> Result<Vec<Sem3Threshold>, Sem3ExperimentError> {
    let mut thresholds = Vec::with_capacity(SYM_RSI_SEM_003_DIMENSIONS.len());
    for dimension in SYM_RSI_SEM_003_DIMENSIONS {
        let bank = candidates
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
            require_query_support(encoder, &query.context)?;
            let target = bank
                .iter()
                .find(|candidate| candidate.identity == query.target)
                .ok_or(Sem3ExperimentError::CalibrationTargetMismatch)?;
            if !target.support.within_nominal_support() {
                return Err(Sem3ExperimentError::CandidateOutsideNominalSupport);
            }
            let query_key = encoder.encode(&query.context)?;
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

fn evaluate_clean_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Candidate],
    query: &LabelledQuery,
    threshold: f32,
) -> Result<CleanObservation, Sem3ExperimentError> {
    require_query_support(encoder, &query.context)?;
    let ranked = candidate_set_and_diagnostics(encoder, semantic_null, bank, &query.context, threshold)?;
    let target_included = ranked.identities.contains(&query.target);
    Ok(CleanObservation {
        base: SetObservation::from_ranked(&ranked),
        target_included,
        singleton_correct: target_included && ranked.identities.len() == 1,
    })
}

fn evaluate_ambiguous_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Candidate],
    query: &AmbiguousQuery,
    threshold: f32,
) -> Result<AmbiguousObservation, Sem3ExperimentError> {
    require_query_support(encoder, &query.context)?;
    let ranked = candidate_set_and_diagnostics(encoder, semantic_null, bank, &query.context, threshold)?;
    let left = ranked.identities.contains(&query.left_parent);
    let right = ranked.identities.contains(&query.right_parent);
    Ok(AmbiguousObservation {
        base: SetObservation::from_ranked(&ranked),
        at_least_one_parent: left || right,
        both_parents: left && right,
        singleton_forced_choice: ranked.identities.len() == 1,
    })
}

fn evaluate_unrelated_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Candidate],
    query: &UnlabelledQuery,
    threshold: f32,
) -> Result<SetObservation, Sem3ExperimentError> {
    require_query_support(encoder, &query.context)?;
    let ranked = candidate_set_and_diagnostics(encoder, semantic_null, bank, &query.context, threshold)?;
    Ok(SetObservation::from_ranked(&ranked))
}

fn evaluate_ood_query(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Candidate],
    query: &UnlabelledQuery,
) -> Result<OodObservation, Sem3ExperimentError> {
    let support = assess_semantic_context_support(encoder, &query.context)?;
    let query_key = encoder.encode(&query.context)?;
    let (top1, _, exact_tie) = top_two_similarities(&query_key, bank)?;
    let specificity = semantic_null
        .assess(query.dimension, top1)?
        .retrieval_specificity;
    Ok(OodObservation {
        support_rejected: !support.within_nominal_support(),
        top1_similarity: top1,
        exact_tie,
        specificity,
    })
}

impl SetObservation {
    fn from_ranked(ranked: &RankedSet) -> Self {
        Self {
            set_size: ranked.identities.len(),
            margin: ranked.margin,
            exact_tie: ranked.exact_tie,
            specificity: ranked.specificity,
        }
    }
}

fn candidate_set_and_diagnostics(
    encoder: SemanticContextEncoder,
    semantic_null: &SemanticNullCalibration,
    bank: &[&Candidate],
    query: &[f32],
    threshold: f32,
) -> Result<RankedSet, Sem3ExperimentError> {
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
    Ok(RankedSet {
        identities,
        margin: (top1 - runner_up).max(0.0),
        exact_tie,
        specificity,
    })
}

fn top_two_similarities(
    query_key: &BinaryHV,
    bank: &[&Candidate],
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
    clean: &[CleanObservation],
    ambiguous: &[AmbiguousObservation],
    unrelated: &[SetObservation],
    ood: &[OodObservation],
) -> Sem3DimensionMetrics {
    let clean_sizes = clean
        .iter()
        .map(|observation| observation.base.set_size)
        .collect::<Vec<_>>();
    Sem3DimensionMetrics {
        context_dimension: dimension,
        clean_count: clean.len(),
        ambiguous_count: ambiguous.len(),
        unrelated_count: unrelated.len(),
        ood_count: ood.len(),
        clean_target_coverage: rate(
            clean
                .iter()
                .filter(|observation| observation.target_included)
                .count(),
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
                .map(|observation| observation.base.set_size)
                .collect::<Vec<_>>(),
        ),
        unrelated_nonempty_set_rate: rate(
            unrelated
                .iter()
                .filter(|observation| observation.set_size > 0)
                .count(),
            unrelated.len(),
        ),
        ood_support_rejection_rate: rate(
            ood.iter()
                .filter(|observation| observation.support_rejected)
                .count(),
            ood.len(),
        ),
        mean_clean_margin: mean_f32(
            &clean
                .iter()
                .map(|observation| observation.base.margin)
                .collect::<Vec<_>>(),
        ),
        mean_ambiguous_margin: mean_f32(
            &ambiguous
                .iter()
                .map(|observation| observation.base.margin)
                .collect::<Vec<_>>(),
        ),
        mean_unrelated_margin: mean_f32(
            &unrelated
                .iter()
                .map(|observation| observation.margin)
                .collect::<Vec<_>>(),
        ),
        mean_ood_raw_similarity: mean_f32(
            &ood.iter()
                .map(|observation| observation.top1_similarity)
                .collect::<Vec<_>>(),
        ),
    }
}

fn regime_diagnostics(observations: &[SetObservation]) -> Sem3RegimeDiagnostics {
    let margins = observations
        .iter()
        .map(|observation| observation.margin)
        .collect::<Vec<_>>();
    let specificities = observations
        .iter()
        .map(|observation| observation.specificity)
        .collect::<Vec<_>>();
    Sem3RegimeDiagnostics {
        query_count: observations.len(),
        mean_margin: mean_f32(&margins),
        p95_margin: percentile_f32(&margins, 0.95),
        exact_tie_count: observations
            .iter()
            .filter(|observation| observation.exact_tie)
            .count(),
        mean_specificity: mean_f32(&specificities),
        p05_specificity: percentile_f32(&specificities, 0.05),
        p50_specificity: percentile_f32(&specificities, 0.50),
        p95_specificity: percentile_f32(&specificities, 0.95),
    }
}

fn ood_diagnostics(observations: &[OodObservation]) -> Sem3OodDiagnostics {
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
        exact_tie_count: observations
            .iter()
            .filter(|observation| observation.exact_tie)
            .count(),
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

fn threshold_for(
    dimension: usize,
    thresholds: &[Sem3Threshold],
) -> Result<f32, Sem3ExperimentError> {
    thresholds
        .iter()
        .find(|threshold| threshold.context_dimension == dimension)
        .map(|threshold| threshold.nonconformity_threshold)
        .ok_or(Sem3ExperimentError::MissingThreshold {
            context_dimension: dimension,
        })
}

fn candidate_index(
    bank: &[&Candidate],
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
    let delta = if side == 0 { -0.08 } else { 0.08 };
    context[coordinate] += delta;
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

fn validate_exact_disjointness(
    candidates: &[Candidate],
    calibration: &[LabelledQuery],
    clean: &[LabelledQuery],
    ambiguous: &[AmbiguousQuery],
    unrelated: &[UnlabelledQuery],
    ood: &[UnlabelledQuery],
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

fn candidate_bank_digest(candidates: &[Candidate]) -> [u8; 32] {
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

fn labelled_query_digest(domain: &str, queries: &[LabelledQuery]) -> [u8; 32] {
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

fn ambiguous_query_digest(queries: &[AmbiguousQuery]) -> [u8; 32] {
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

fn unlabelled_query_digest(domain: &str, queries: &[UnlabelledQuery]) -> [u8; 32] {
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

fn hash_context(hasher: &mut blake3::Hasher, context: &[f32]) {
    hasher.update(&(context.len() as u64).to_le_bytes());
    for value in context {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn disposition_tag(disposition: Sem3Disposition) -> u8 {
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

fn candidate_index(
    bank: &[&Candidate],
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
    let delta = if side == 0 { -0.08 } else { 0.08 };
    context[coordinate] += delta;
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

fn validate_exact_disjointness(
    candidates: &[Candidate],
    calibration: &[LabelledQuery],
    clean: &[LabelledQuery],
    ambiguous: &[AmbiguousQuery],
    unrelated: &[UnlabelledQuery],
    ood: &[UnlabelledQuery],
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
        let encoder = SemanticContextEncoder::default();
        let candidates = candidate_bank(encoder).unwrap();
        let calibration = calibration_queries(&candidates).unwrap();
        for query in &calibration {
            require_query_support(encoder, &query.context).unwrap();
        }
        let thresholds = calibrate_thresholds(encoder, &candidates, &calibration).unwrap();
        assert!(thresholds.iter().all(|threshold| {
            threshold.calibration_count == SYM_RSI_SEM_003_CALIBRATION_QUERIES_PER_DIMENSION
        }));
    }
}
