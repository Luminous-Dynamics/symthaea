// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered SYM-RSI-SEM-002 ambiguity-aware semantic-admission benchmark.
//!
//! This benchmark owns deterministic synthetic contexts. It does not call protected
//! SYM-RSI fixtures, consume protected seeds, or grant evidence/confidence authority.

use std::collections::{HashMap, HashSet};

use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};
use super::semantic_null_calibration::{
    SemanticNullCalibration, SemanticNullCalibrationError, SemanticNullConfig,
};
use super::semantic_retrieval_ambiguity::{
    SemanticRetrievalAmbiguity, SemanticRetrievalAmbiguityError,
};
use super::semantic_support::{SemanticSupportError, assess_semantic_context_support};
use symthaea_core::hdc::BinaryHV;

pub const SYM_RSI_SEM_002_SCHEMA: &str = "symthaea.sym-rsi-sem-002.v1";
pub const SYM_RSI_SEM_002_GENERATOR_VERSION: &str = "semantic-ambiguity-generator.v1";
pub const SYM_RSI_SEM_002_DIMENSIONS: [usize; 3] = [4, 8, 16];
pub const SYM_RSI_SEM_002_REFERENCE_COUNT_PER_DIMENSION: usize = 48;
pub const SYM_RSI_SEM_002_NULL_MARGIN_QUERY_COUNT_PER_DIMENSION: usize = 24;
pub const SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION: usize = 24;
pub const SYM_RSI_SEM_002_OOD_COUNT_PER_DIMENSION: usize = 16;
pub const SYM_RSI_SEM_002_SPECIFICITY_THRESHOLD: f32 = 0.95;
pub const SYM_RSI_SEM_002_PRIMARY_AMBIGUITY_REDUCTION: f64 = 0.10;
pub const SYM_RSI_SEM_002_CLEAN_RETENTION_TOLERANCE: f64 = 0.10;
pub const SYM_RSI_SEM_002_DIMENSION_RETENTION_TOLERANCE: f64 = 0.15;
const RATE_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sem2Regime {
    CleanRelated,
    Ambiguous,
    Unrelated,
    OodClampSaturation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sem2Disposition {
    Pass,
    Tradeoff,
    NotEstablished,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem2AdmissionRates {
    pub raw: f64,
    pub specificity: f64,
    pub margin: f64,
    pub specificity_margin: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem2MarginThreshold {
    pub context_dimension: usize,
    pub sample_count: usize,
    pub median_margin: f32,
    pub p95_margin: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sem2DimensionMetrics {
    pub context_dimension: usize,
    pub clean_count: usize,
    pub ambiguous_count: usize,
    pub unrelated_count: usize,
    pub ood_count: usize,
    pub clean_top1_target_accuracy: f64,
    pub clean_correct_activation: Sem2AdmissionRates,
    pub ambiguous_activation: Sem2AdmissionRates,
    pub unrelated_activation: Sem2AdmissionRates,
    pub ood_raw_activation: f64,
    pub ood_support_rejection_rate: f64,
    pub mean_clean_similarity: f64,
    pub mean_clean_margin: f64,
    pub mean_clean_specificity: f64,
    pub mean_ambiguous_similarity: f64,
    pub mean_ambiguous_margin: f64,
    pub mean_ambiguous_specificity: f64,
    pub mean_unrelated_similarity: f64,
    pub mean_unrelated_margin: f64,
    pub mean_unrelated_specificity: f64,
    pub mean_ood_similarity: f64,
    pub mean_ood_margin: f64,
    pub mean_ood_specificity: f64,
    pub exact_tie_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sem2Receipt {
    pub schema: &'static str,
    pub generator_version: &'static str,
    pub subject_sha: String,
    pub semantic_calibration_digest: [u8; 32],
    pub margin_calibration_digest: [u8; 32],
    pub reference_memory_digest: [u8; 32],
    pub null_margin_query_digest: [u8; 32],
    pub evaluation_memory_digest: [u8; 32],
    pub clean_query_digest: [u8; 32],
    pub ambiguous_query_digest: [u8; 32],
    pub unrelated_query_digest: [u8; 32],
    pub ood_query_digest: [u8; 32],
    pub raw_global_similarity_threshold: f32,
    pub specificity_threshold: f32,
    pub margin_thresholds: Vec<Sem2MarginThreshold>,
    pub dimension_metrics: Vec<Sem2DimensionMetrics>,
    pub clean_correct_activation: Sem2AdmissionRates,
    pub ambiguous_activation: Sem2AdmissionRates,
    pub unrelated_activation: Sem2AdmissionRates,
    pub clean_top1_target_accuracy: f64,
    pub ambiguous_activation_reduction_specificity_to_combined: f64,
    pub primary_ambiguity_reduction_passed: bool,
    pub clean_retention_guard_passed: bool,
    pub dimension_retention_guard_passed: bool,
    pub unrelated_guard_passed: bool,
    pub disposition: Sem2Disposition,
    pub evidence_digest: [u8; 32],
}

impl Sem2Receipt {
    pub fn evidence_digest_hex(&self) -> String {
        hex::encode(self.evidence_digest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sem2ExperimentError {
    Context(SemanticContextError),
    Calibration(SemanticNullCalibrationError),
    Ambiguity(SemanticRetrievalAmbiguityError),
    Support(SemanticSupportError),
    InvalidSubjectSha,
    MissingCalibrationSummary { context_dimension: usize },
    MissingMarginThreshold { context_dimension: usize },
    InsufficientCandidates { context_dimension: usize },
    DuplicateIdentity { partition: &'static str },
    PartitionOverlap { left: &'static str, right: &'static str },
    CleanQueryIdentityCollision,
    AmbiguousQueryIdentityCollision,
    OodQueryIdentityCollision,
}

impl From<SemanticContextError> for Sem2ExperimentError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

impl From<SemanticNullCalibrationError> for Sem2ExperimentError {
    fn from(value: SemanticNullCalibrationError) -> Self {
        Self::Calibration(value)
    }
}

impl From<SemanticRetrievalAmbiguityError> for Sem2ExperimentError {
    fn from(value: SemanticRetrievalAmbiguityError) -> Self {
        Self::Ambiguity(value)
    }
}

impl From<SemanticSupportError> for Sem2ExperimentError {
    fn from(value: SemanticSupportError) -> Self {
        Self::Support(value)
    }
}

#[derive(Debug, Clone)]
struct EvalCase {
    dimension: usize,
    regime: Sem2Regime,
    query: Vec<f32>,
    intended_target: Option<ContextIdentity>,
    source_a: Option<ContextIdentity>,
    source_b: Option<ContextIdentity>,
}

#[derive(Debug, Clone, Copy)]
struct AdmissionFlags {
    raw: bool,
    specificity: bool,
    margin: bool,
    specificity_margin: bool,
}

#[derive(Debug, Clone)]
struct Observation {
    dimension: usize,
    regime: Sem2Regime,
    intended_target: Option<ContextIdentity>,
    top_identity: ContextIdentity,
    best_similarity: f32,
    top_two_margin: f32,
    retrieval_specificity: f32,
    admission: AdmissionFlags,
    query_within_nominal_support: bool,
}

#[derive(Clone)]
struct EncodedCandidate {
    identity: ContextIdentity,
    hypervector: BinaryHV,
}

#[derive(Debug, Clone, Copy)]
struct RankedQuery {
    top_identity: ContextIdentity,
    best_similarity: f32,
    top_two_margin: f32,
}

struct ReceiptDigestInput<'a> {
    subject_sha: &'a str,
    semantic_calibration_digest: [u8; 32],
    margin_calibration_digest: [u8; 32],
    reference_memory_digest: [u8; 32],
    null_margin_query_digest: [u8; 32],
    evaluation_memory_digest: [u8; 32],
    clean_query_digest: [u8; 32],
    ambiguous_query_digest: [u8; 32],
    unrelated_query_digest: [u8; 32],
    ood_query_digest: [u8; 32],
    raw_global_similarity_threshold: f32,
    margin_thresholds: &'a [Sem2MarginThreshold],
    dimension_metrics: &'a [Sem2DimensionMetrics],
    clean_correct_activation: Sem2AdmissionRates,
    ambiguous_activation: Sem2AdmissionRates,
    unrelated_activation: Sem2AdmissionRates,
    clean_top1_target_accuracy: f64,
    ambiguous_reduction: f64,
    primary_passed: bool,
    clean_guard: bool,
    dimension_guard: bool,
    unrelated_guard: bool,
    disposition: Sem2Disposition,
}

/// Execute the preregistered SEM-002 synthetic benchmark.
///
/// The returned receipt is benchmark measurement only. It is not executable
/// qualification of this Rust subject and grants no evidence/confidence authority.
pub fn run_sym_rsi_sem_002(subject_sha: &str) -> Result<Sem2Receipt, Sem2ExperimentError> {
    validate_subject_sha(subject_sha)?;

    let encoder = SemanticContextEncoder::default();
    let reference_memory = generated_contexts(0, SYM_RSI_SEM_002_REFERENCE_COUNT_PER_DIMENSION);
    let null_margin_queries = generated_contexts(
        60,
        SYM_RSI_SEM_002_NULL_MARGIN_QUERY_COUNT_PER_DIMENSION,
    );
    let evaluation_memory = generated_contexts(100, SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION);
    let unrelated_query_sources =
        generated_contexts(1000, SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION);
    let ood_anchors = generated_contexts(2000, SYM_RSI_SEM_002_OOD_COUNT_PER_DIMENSION);

    validate_source_partitions(&[
        ("reference-memory", reference_memory.as_slice()),
        ("null-margin-query", null_margin_queries.as_slice()),
        ("evaluation-memory", evaluation_memory.as_slice()),
        ("unrelated-query-source", unrelated_query_sources.as_slice()),
        ("ood-anchor", ood_anchors.as_slice()),
    ])?;

    let semantic_calibration = SemanticNullCalibration::build(
        &reference_memory,
        encoder,
        SemanticNullConfig {
            max_reference_contexts_per_dimension: SYM_RSI_SEM_002_REFERENCE_COUNT_PER_DIMENSION,
            min_null_pairs_per_dimension: 128,
        },
    )?;
    let raw_global_similarity_threshold = global_raw_threshold(&semantic_calibration)?;
    let margin_thresholds =
        build_margin_thresholds(encoder, &reference_memory, &null_margin_queries)?;

    let clean_cases = clean_related_cases(&evaluation_memory)?;
    let ambiguous_cases = ambiguous_cases(&evaluation_memory)?;
    let unrelated_cases = unrelated_cases(&unrelated_query_sources);
    let ood_cases = ood_cases(&ood_anchors)?;

    validate_evaluation_cases(
        &reference_memory,
        &null_margin_queries,
        &evaluation_memory,
        &clean_cases,
        &ambiguous_cases,
        &unrelated_cases,
        &ood_cases,
    )?;

    let mut observations = Vec::new();
    for dimension in SYM_RSI_SEM_002_DIMENSIONS {
        let bank_contexts = contexts_for_dimension(&evaluation_memory, dimension);
        let bank = encode_bank(encoder, &bank_contexts)?;
        let margin_threshold = margin_threshold_for(&margin_thresholds, dimension)?;

        for case in clean_cases
            .iter()
            .chain(ambiguous_cases.iter())
            .chain(unrelated_cases.iter())
            .chain(ood_cases.iter())
            .filter(|case| case.dimension == dimension)
        {
            observations.push(evaluate_case(
                encoder,
                &semantic_calibration,
                &bank,
                case,
                raw_global_similarity_threshold,
                margin_threshold.p95_margin,
            )?);
        }
    }

    let dimension_metrics = SYM_RSI_SEM_002_DIMENSIONS
        .iter()
        .copied()
        .map(|dimension| dimension_metrics(dimension, &observations))
        .collect::<Vec<_>>();
    let clean_correct_activation = correct_clean_activation_rates(&observations);
    let ambiguous_activation = activation_rates(&observations, Sem2Regime::Ambiguous);
    let unrelated_activation = activation_rates(&observations, Sem2Regime::Unrelated);
    let clean_top1_target_accuracy = clean_top1_target_accuracy(&observations);

    let ambiguous_reduction =
        ambiguous_activation.specificity - ambiguous_activation.specificity_margin;
    let primary_passed = ambiguous_reduction + RATE_EPSILON
        >= SYM_RSI_SEM_002_PRIMARY_AMBIGUITY_REDUCTION;
    let clean_guard = clean_correct_activation.specificity_margin
        + SYM_RSI_SEM_002_CLEAN_RETENTION_TOLERANCE
        >= clean_correct_activation.specificity;
    let dimension_guard = dimension_metrics.iter().all(|metrics| {
        metrics.clean_correct_activation.specificity_margin
            + SYM_RSI_SEM_002_DIMENSION_RETENTION_TOLERANCE
            >= metrics.clean_correct_activation.specificity
    });
    let unrelated_guard = unrelated_activation.specificity_margin
        <= unrelated_activation.specificity + RATE_EPSILON;

    let disposition = if !primary_passed {
        Sem2Disposition::NotEstablished
    } else if clean_guard && dimension_guard && unrelated_guard {
        Sem2Disposition::Pass
    } else {
        Sem2Disposition::Tradeoff
    };

    let semantic_calibration_digest = semantic_calibration.identity().digest;
    let margin_calibration_digest = margin_calibration_digest(
        &reference_memory,
        &null_margin_queries,
        &margin_thresholds,
    );
    let reference_memory_digest = context_corpus_digest("reference-memory", &reference_memory);
    let null_margin_query_digest = context_corpus_digest("null-margin-query", &null_margin_queries);
    let evaluation_memory_digest = context_corpus_digest("evaluation-memory", &evaluation_memory);
    let clean_query_digest = case_corpus_digest("clean-related", &clean_cases);
    let ambiguous_query_digest = case_corpus_digest("ambiguous", &ambiguous_cases);
    let unrelated_query_digest = case_corpus_digest("unrelated", &unrelated_cases);
    let ood_query_digest = case_corpus_digest("ood", &ood_cases);

    let evidence_digest = receipt_digest(&ReceiptDigestInput {
        subject_sha,
        semantic_calibration_digest,
        margin_calibration_digest,
        reference_memory_digest,
        null_margin_query_digest,
        evaluation_memory_digest,
        clean_query_digest,
        ambiguous_query_digest,
        unrelated_query_digest,
        ood_query_digest,
        raw_global_similarity_threshold,
        margin_thresholds: &margin_thresholds,
        dimension_metrics: &dimension_metrics,
        clean_correct_activation,
        ambiguous_activation,
        unrelated_activation,
        clean_top1_target_accuracy,
        ambiguous_reduction,
        primary_passed,
        clean_guard,
        dimension_guard,
        unrelated_guard,
        disposition,
    });

    Ok(Sem2Receipt {
        schema: SYM_RSI_SEM_002_SCHEMA,
        generator_version: SYM_RSI_SEM_002_GENERATOR_VERSION,
        subject_sha: subject_sha.to_owned(),
        semantic_calibration_digest,
        margin_calibration_digest,
        reference_memory_digest,
        null_margin_query_digest,
        evaluation_memory_digest,
        clean_query_digest,
        ambiguous_query_digest,
        unrelated_query_digest,
        ood_query_digest,
        raw_global_similarity_threshold,
        specificity_threshold: SYM_RSI_SEM_002_SPECIFICITY_THRESHOLD,
        margin_thresholds,
        dimension_metrics,
        clean_correct_activation,
        ambiguous_activation,
        unrelated_activation,
        clean_top1_target_accuracy,
        ambiguous_activation_reduction_specificity_to_combined: ambiguous_reduction,
        primary_ambiguity_reduction_passed: primary_passed,
        clean_retention_guard_passed: clean_guard,
        dimension_retention_guard_passed: dimension_guard,
        unrelated_guard_passed: unrelated_guard,
        disposition,
        evidence_digest,
    })
}

fn generated_contexts(start: usize, count: usize) -> Vec<Vec<f32>> {
    let mut contexts = Vec::with_capacity(SYM_RSI_SEM_002_DIMENSIONS.len() * count);
    for dimension in SYM_RSI_SEM_002_DIMENSIONS {
        for offset in 0..count {
            contexts.push(base_context(dimension, start + offset));
        }
    }
    contexts
}

fn base_context(dimension: usize, index: usize) -> Vec<f32> {
    (0..dimension)
        .map(|coordinate| {
            let mixed = 211usize
                .wrapping_add(dimension.wrapping_mul(131))
                .wrapping_add(index.wrapping_mul(67))
                .wrapping_add(coordinate.wrapping_mul(37))
                .wrapping_add(coordinate.wrapping_mul(coordinate).wrapping_mul(17));
            (mixed % 1801) as f32 / 900.0 - 1.0
        })
        .collect()
}

fn contexts_for_dimension(contexts: &[Vec<f32>], dimension: usize) -> Vec<Vec<f32>> {
    contexts
        .iter()
        .filter(|context| context.len() == dimension)
        .cloned()
        .collect()
}

fn clean_related_cases(
    evaluation_memory: &[Vec<f32>],
) -> Result<Vec<EvalCase>, Sem2ExperimentError> {
    let mut cases = Vec::new();
    for context in evaluation_memory {
        let identity = ContextIdentity::from_context(context)?;
        for (variant, magnitude) in [0.04_f32, 0.08_f32].into_iter().enumerate() {
            cases.push(EvalCase {
                dimension: context.len(),
                regime: Sem2Regime::CleanRelated,
                query: local_perturbation(context, identity.fast_hash, variant, magnitude),
                intended_target: Some(identity),
                source_a: Some(identity),
                source_b: None,
            });
        }
    }
    Ok(cases)
}

fn ambiguous_cases(
    evaluation_memory: &[Vec<f32>],
) -> Result<Vec<EvalCase>, Sem2ExperimentError> {
    let mut cases = Vec::new();
    for dimension in SYM_RSI_SEM_002_DIMENSIONS {
        let bank = contexts_for_dimension(evaluation_memory, dimension);
        for pair in bank.chunks_exact(2) {
            let left_identity = ContextIdentity::from_context(&pair[0])?;
            let right_identity = ContextIdentity::from_context(&pair[1])?;
            for (left_weight, right_weight) in [(0.5_f32, 0.5_f32), (0.52_f32, 0.48_f32)] {
                let query = pair[0]
                    .iter()
                    .zip(&pair[1])
                    .map(|(left, right)| {
                        (left_weight * left + right_weight * right).clamp(-1.0, 1.0)
                    })
                    .collect::<Vec<_>>();
                cases.push(EvalCase {
                    dimension,
                    regime: Sem2Regime::Ambiguous,
                    query,
                    intended_target: None,
                    source_a: Some(left_identity),
                    source_b: Some(right_identity),
                });
            }
        }
    }
    Ok(cases)
}

fn unrelated_cases(unrelated_queries: &[Vec<f32>]) -> Vec<EvalCase> {
    unrelated_queries
        .iter()
        .map(|query| EvalCase {
            dimension: query.len(),
            regime: Sem2Regime::Unrelated,
            query: query.clone(),
            intended_target: None,
            source_a: None,
            source_b: None,
        })
        .collect()
}

fn ood_cases(ood_anchors: &[Vec<f32>]) -> Result<Vec<EvalCase>, Sem2ExperimentError> {
    let mut cases = Vec::with_capacity(ood_anchors.len());
    for (index, anchor) in ood_anchors.iter().enumerate() {
        let source_identity = ContextIdentity::from_context(anchor)?;
        let mut query = anchor.clone();
        let coordinate = (index * 7 + anchor.len()) % anchor.len();
        query[coordinate] = if anchor[coordinate] >= 0.0 {
            1.5 + anchor[coordinate].abs()
        } else {
            -1.5 - anchor[coordinate].abs()
        };
        cases.push(EvalCase {
            dimension: anchor.len(),
            regime: Sem2Regime::OodClampSaturation,
            query,
            intended_target: None,
            source_a: Some(source_identity),
            source_b: None,
        });
    }
    Ok(cases)
}

fn local_perturbation(
    anchor: &[f32],
    selector: u64,
    variant: usize,
    magnitude: f32,
) -> Vec<f32> {
    let mut query = anchor.to_vec();
    let base_coordinate = (selector % query.len() as u64) as usize;
    let coordinate = (base_coordinate + variant * 3) % query.len();
    if query[coordinate] >= 0.0 {
        query[coordinate] = (query[coordinate] - magnitude).max(-1.0);
    } else {
        query[coordinate] = (query[coordinate] + magnitude).min(1.0);
    }
    query
}

fn global_raw_threshold(calibration: &SemanticNullCalibration) -> Result<f32, Sem2ExperimentError> {
    let summaries = calibration.summaries();
    let mut sum = 0.0_f32;
    for dimension in SYM_RSI_SEM_002_DIMENSIONS {
        let summary = summaries
            .iter()
            .find(|summary| summary.context_dimension == dimension)
            .ok_or(Sem2ExperimentError::MissingCalibrationSummary {
                context_dimension: dimension,
            })?;
        sum += summary.p95_similarity;
    }
    Ok(sum / SYM_RSI_SEM_002_DIMENSIONS.len() as f32)
}

fn build_margin_thresholds(
    encoder: SemanticContextEncoder,
    reference_memory: &[Vec<f32>],
    null_queries: &[Vec<f32>],
) -> Result<Vec<Sem2MarginThreshold>, Sem2ExperimentError> {
    let mut thresholds = Vec::new();
    for dimension in SYM_RSI_SEM_002_DIMENSIONS {
        let bank = encode_bank(encoder, &contexts_for_dimension(reference_memory, dimension))?;
        let queries = contexts_for_dimension(null_queries, dimension);
        let mut margins = Vec::with_capacity(queries.len());
        for query in &queries {
            margins.push(rank_query(encoder, &bank, query)?.top_two_margin);
        }
        margins.sort_by(f32::total_cmp);
        thresholds.push(Sem2MarginThreshold {
            context_dimension: dimension,
            sample_count: margins.len(),
            median_margin: median(&margins),
            p95_margin: nearest_rank_quantile(&margins, 0.95),
        });
    }
    Ok(thresholds)
}

fn margin_threshold_for(
    thresholds: &[Sem2MarginThreshold],
    dimension: usize,
) -> Result<Sem2MarginThreshold, Sem2ExperimentError> {
    thresholds
        .iter()
        .copied()
        .find(|threshold| threshold.context_dimension == dimension)
        .ok_or(Sem2ExperimentError::MissingMarginThreshold {
            context_dimension: dimension,
        })
}

fn encode_bank(
    encoder: SemanticContextEncoder,
    contexts: &[Vec<f32>],
) -> Result<Vec<EncodedCandidate>, Sem2ExperimentError> {
    contexts
        .iter()
        .map(|context| {
            Ok(EncodedCandidate {
                identity: ContextIdentity::from_context(context)?,
                hypervector: encoder.encode(context)?,
            })
        })
        .collect()
}

fn rank_query(
    encoder: SemanticContextEncoder,
    bank: &[EncodedCandidate],
    query: &[f32],
) -> Result<RankedQuery, Sem2ExperimentError> {
    if bank.len() < 2 {
        return Err(Sem2ExperimentError::InsufficientCandidates {
            context_dimension: query.len(),
        });
    }

    let query_hv = encoder.encode(query)?;
    let mut scores = bank
        .iter()
        .map(|candidate| (candidate.identity, query_hv.similarity(&candidate.hypervector)))
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.exact_digest.cmp(&right.0.exact_digest))
            .then_with(|| left.0.fast_hash.cmp(&right.0.fast_hash))
    });

    let ambiguity = SemanticRetrievalAmbiguity::from_similarities(
        scores.iter().map(|(_, similarity)| *similarity),
    )?;
    let best_similarity = ambiguity
        .best_similarity
        .ok_or(Sem2ExperimentError::InsufficientCandidates {
            context_dimension: query.len(),
        })?;
    let top_two_margin = ambiguity
        .top_two_margin
        .ok_or(Sem2ExperimentError::InsufficientCandidates {
            context_dimension: query.len(),
        })?;

    Ok(RankedQuery {
        top_identity: scores[0].0,
        best_similarity,
        top_two_margin,
    })
}

fn evaluate_case(
    encoder: SemanticContextEncoder,
    calibration: &SemanticNullCalibration,
    bank: &[EncodedCandidate],
    case: &EvalCase,
    raw_threshold: f32,
    margin_threshold: f32,
) -> Result<Observation, Sem2ExperimentError> {
    let ranked = rank_query(encoder, bank, &case.query)?;
    let retrieval_specificity = calibration
        .assess(case.dimension, ranked.best_similarity)?
        .retrieval_specificity;
    let specificity = retrieval_specificity >= SYM_RSI_SEM_002_SPECIFICITY_THRESHOLD;
    let margin = ranked.top_two_margin >= margin_threshold;
    let support = assess_semantic_context_support(encoder, &case.query)?;

    Ok(Observation {
        dimension: case.dimension,
        regime: case.regime,
        intended_target: case.intended_target,
        top_identity: ranked.top_identity,
        best_similarity: ranked.best_similarity,
        top_two_margin: ranked.top_two_margin,
        retrieval_specificity,
        admission: AdmissionFlags {
            raw: ranked.best_similarity >= raw_threshold,
            specificity,
            margin,
            specificity_margin: specificity && margin,
        },
        query_within_nominal_support: support.within_nominal_support(),
    })
}

fn dimension_metrics(dimension: usize, observations: &[Observation]) -> Sem2DimensionMetrics {
    let clean = select(observations, dimension, Sem2Regime::CleanRelated);
    let ambiguous = select(observations, dimension, Sem2Regime::Ambiguous);
    let unrelated = select(observations, dimension, Sem2Regime::Unrelated);
    let ood = select(observations, dimension, Sem2Regime::OodClampSaturation);

    Sem2DimensionMetrics {
        context_dimension: dimension,
        clean_count: clean.len(),
        ambiguous_count: ambiguous.len(),
        unrelated_count: unrelated.len(),
        ood_count: ood.len(),
        clean_top1_target_accuracy: clean_top1_target_accuracy_refs(&clean),
        clean_correct_activation: correct_clean_activation_rates_refs(&clean),
        ambiguous_activation: admission_rates_refs(&ambiguous),
        unrelated_activation: admission_rates_refs(&unrelated),
        ood_raw_activation: rate(
            ood.iter().filter(|observation| observation.admission.raw).count(),
            ood.len(),
        ),
        ood_support_rejection_rate: rate(
            ood.iter()
                .filter(|observation| !observation.query_within_nominal_support)
                .count(),
            ood.len(),
        ),
        mean_clean_similarity: mean_field(&clean, |observation| observation.best_similarity),
        mean_clean_margin: mean_field(&clean, |observation| observation.top_two_margin),
        mean_clean_specificity: mean_field(&clean, |observation| observation.retrieval_specificity),
        mean_ambiguous_similarity: mean_field(&ambiguous, |observation| observation.best_similarity),
        mean_ambiguous_margin: mean_field(&ambiguous, |observation| observation.top_two_margin),
        mean_ambiguous_specificity: mean_field(&ambiguous, |observation| {
            observation.retrieval_specificity
        }),
        mean_unrelated_similarity: mean_field(&unrelated, |observation| observation.best_similarity),
        mean_unrelated_margin: mean_field(&unrelated, |observation| observation.top_two_margin),
        mean_unrelated_specificity: mean_field(&unrelated, |observation| {
            observation.retrieval_specificity
        }),
        mean_ood_similarity: mean_field(&ood, |observation| observation.best_similarity),
        mean_ood_margin: mean_field(&ood, |observation| observation.top_two_margin),
        mean_ood_specificity: mean_field(&ood, |observation| observation.retrieval_specificity),
        exact_tie_count: clean
            .iter()
            .chain(ambiguous.iter())
            .chain(unrelated.iter())
            .chain(ood.iter())
            .filter(|observation| observation.top_two_margin == 0.0)
            .count(),
    }
}

fn select(
    observations: &[Observation],
    dimension: usize,
    regime: Sem2Regime,
) -> Vec<&Observation> {
    observations
        .iter()
        .filter(|observation| observation.dimension == dimension && observation.regime == regime)
        .collect()
}

fn admission_rates(observations: &[Observation], regime: Sem2Regime) -> Sem2AdmissionRates {
    let selected = observations
        .iter()
        .filter(|observation| observation.regime == regime)
        .collect::<Vec<_>>();
    admission_rates_refs(&selected)
}

fn admission_rates_refs(observations: &[&Observation]) -> Sem2AdmissionRates {
    Sem2AdmissionRates {
        raw: admission_rate(observations, |flags| flags.raw),
        specificity: admission_rate(observations, |flags| flags.specificity),
        margin: admission_rate(observations, |flags| flags.margin),
        specificity_margin: admission_rate(observations, |flags| flags.specificity_margin),
    }
}

fn admission_rate(
    observations: &[&Observation],
    selector: impl Fn(AdmissionFlags) -> bool,
) -> f64 {
    rate(
        observations
            .iter()
            .filter(|observation| selector(observation.admission))
            .count(),
        observations.len(),
    )
}

fn correct_clean_activation_rates(observations: &[Observation]) -> Sem2AdmissionRates {
    let clean = observations
        .iter()
        .filter(|observation| observation.regime == Sem2Regime::CleanRelated)
        .collect::<Vec<_>>();
    correct_clean_activation_rates_refs(&clean)
}

fn correct_clean_activation_rates_refs(observations: &[&Observation]) -> Sem2AdmissionRates {
    Sem2AdmissionRates {
        raw: correct_activation_rate(observations, |flags| flags.raw),
        specificity: correct_activation_rate(observations, |flags| flags.specificity),
        margin: correct_activation_rate(observations, |flags| flags.margin),
        specificity_margin: correct_activation_rate(observations, |flags| flags.specificity_margin),
    }
}

fn correct_activation_rate(
    observations: &[&Observation],
    selector: impl Fn(AdmissionFlags) -> bool,
) -> f64 {
    rate(
        observations
            .iter()
            .filter(|observation| {
                observation.intended_target == Some(observation.top_identity)
                    && selector(observation.admission)
            })
            .count(),
        observations.len(),
    )
}

fn clean_top1_target_accuracy(observations: &[Observation]) -> f64 {
    let clean = observations
        .iter()
        .filter(|observation| observation.regime == Sem2Regime::CleanRelated)
        .collect::<Vec<_>>();
    clean_top1_target_accuracy_refs(&clean)
}

fn clean_top1_target_accuracy_refs(observations: &[&Observation]) -> f64 {
    rate(
        observations
            .iter()
            .filter(|observation| observation.intended_target == Some(observation.top_identity))
            .count(),
        observations.len(),
    )
}

fn mean_field(observations: &[&Observation], selector: impl Fn(&Observation) -> f32) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    observations
        .iter()
        .map(|observation| selector(observation) as f64)
        .sum::<f64>()
        / observations.len() as f64
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn median(sorted: &[f32]) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let midpoint = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    }
}

fn nearest_rank_quantile(sorted: &[f32], quantile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (quantile * sorted.len() as f32).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn validate_source_partitions(
    partitions: &[(&'static str, &[Vec<f32>])],
) -> Result<(), Sem2ExperimentError> {
    let mut sets: HashMap<&'static str, HashSet<ContextIdentity>> = HashMap::new();
    for &(name, contexts) in partitions {
        let identities = identity_set(contexts)?;
        if identities.len() != contexts.len() {
            return Err(Sem2ExperimentError::DuplicateIdentity { partition: name });
        }
        sets.insert(name, identities);
    }
    for left_index in 0..partitions.len() {
        for right_index in (left_index + 1)..partitions.len() {
            let left = partitions[left_index].0;
            let right = partitions[right_index].0;
            let left_set = sets.get(left).expect("validated partition must exist");
            let right_set = sets.get(right).expect("validated partition must exist");
            if !left_set.is_disjoint(right_set) {
                return Err(Sem2ExperimentError::PartitionOverlap { left, right });
            }
        }
    }
    Ok(())
}

fn validate_evaluation_cases(
    reference_memory: &[Vec<f32>],
    null_margin_queries: &[Vec<f32>],
    evaluation_memory: &[Vec<f32>],
    clean: &[EvalCase],
    ambiguous: &[EvalCase],
    unrelated: &[EvalCase],
    ood: &[EvalCase],
) -> Result<(), Sem2ExperimentError> {
    let source_partitions = [
        ("reference-memory", identity_set(reference_memory)?),
        ("null-margin-query", identity_set(null_margin_queries)?),
        ("evaluation-memory", identity_set(evaluation_memory)?),
    ];
    let query_partitions = [
        ("clean-related", case_identity_set(clean)?, clean.len()),
        ("ambiguous", case_identity_set(ambiguous)?, ambiguous.len()),
        ("unrelated", case_identity_set(unrelated)?, unrelated.len()),
        ("ood", case_identity_set(ood)?, ood.len()),
    ];

    for (query_name, query_set, expected_len) in &query_partitions {
        if query_set.len() != *expected_len {
            return Err(Sem2ExperimentError::DuplicateIdentity {
                partition: *query_name,
            });
        }
        for (source_name, source_set) in &source_partitions {
            if !query_set.is_disjoint(source_set) {
                return Err(Sem2ExperimentError::PartitionOverlap {
                    left: *source_name,
                    right: *query_name,
                });
            }
        }
    }
    for left_index in 0..query_partitions.len() {
        for right_index in (left_index + 1)..query_partitions.len() {
            let (left_name, left_set, _) = &query_partitions[left_index];
            let (right_name, right_set, _) = &query_partitions[right_index];
            if !left_set.is_disjoint(right_set) {
                return Err(Sem2ExperimentError::PartitionOverlap {
                    left: *left_name,
                    right: *right_name,
                });
            }
        }
    }

    for case in clean {
        let query_identity = ContextIdentity::from_context(&case.query)?;
        if case.intended_target == Some(query_identity) {
            return Err(Sem2ExperimentError::CleanQueryIdentityCollision);
        }
    }
    for case in ambiguous {
        let query_identity = ContextIdentity::from_context(&case.query)?;
        if case.source_a == Some(query_identity) || case.source_b == Some(query_identity) {
            return Err(Sem2ExperimentError::AmbiguousQueryIdentityCollision);
        }
    }
    for case in ood {
        let query_identity = ContextIdentity::from_context(&case.query)?;
        if case.source_a == Some(query_identity) {
            return Err(Sem2ExperimentError::OodQueryIdentityCollision);
        }
    }
    Ok(())
}

fn identity_set(contexts: &[Vec<f32>]) -> Result<HashSet<ContextIdentity>, Sem2ExperimentError> {
    contexts
        .iter()
        .map(|context| ContextIdentity::from_context(context).map_err(Sem2ExperimentError::from))
        .collect()
}

fn case_identity_set(cases: &[EvalCase]) -> Result<HashSet<ContextIdentity>, Sem2ExperimentError> {
    cases
        .iter()
        .map(|case| ContextIdentity::from_context(&case.query).map_err(Sem2ExperimentError::from))
        .collect()
}

fn context_corpus_digest(label: &str, contexts: &[Vec<f32>]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_002_SCHEMA.as_bytes());
    hasher.update(label.as_bytes());
    hasher.update(&(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_context(&mut hasher, context);
    }
    *hasher.finalize().as_bytes()
}

fn case_corpus_digest(label: &str, cases: &[EvalCase]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_002_SCHEMA.as_bytes());
    hasher.update(label.as_bytes());
    hasher.update(&(cases.len() as u64).to_le_bytes());
    for case in cases {
        hasher.update(&(case.dimension as u64).to_le_bytes());
        hasher.update(&[regime_tag(case.regime)]);
        hash_optional_identity(&mut hasher, case.intended_target);
        hash_optional_identity(&mut hasher, case.source_a);
        hash_optional_identity(&mut hasher, case.source_b);
        hash_context(&mut hasher, &case.query);
    }
    *hasher.finalize().as_bytes()
}

fn margin_calibration_digest(
    reference_memory: &[Vec<f32>],
    null_queries: &[Vec<f32>],
    thresholds: &[Sem2MarginThreshold],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-sem-002.margin-calibration.v1");
    hasher.update(&context_corpus_digest("margin-reference", reference_memory));
    hasher.update(&context_corpus_digest("margin-null-query", null_queries));
    for threshold in thresholds {
        hash_margin_threshold(&mut hasher, *threshold);
    }
    *hasher.finalize().as_bytes()
}

fn receipt_digest(input: &ReceiptDigestInput<'_>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_002_SCHEMA.as_bytes());
    hasher.update(input.subject_sha.as_bytes());
    hasher.update(&input.semantic_calibration_digest);
    hasher.update(&input.margin_calibration_digest);
    hasher.update(&input.reference_memory_digest);
    hasher.update(&input.null_margin_query_digest);
    hasher.update(&input.evaluation_memory_digest);
    hasher.update(&input.clean_query_digest);
    hasher.update(&input.ambiguous_query_digest);
    hasher.update(&input.unrelated_query_digest);
    hasher.update(&input.ood_query_digest);
    hasher.update(&input.raw_global_similarity_threshold.to_bits().to_le_bytes());
    hasher.update(&SYM_RSI_SEM_002_SPECIFICITY_THRESHOLD.to_bits().to_le_bytes());
    for threshold in input.margin_thresholds {
        hash_margin_threshold(&mut hasher, *threshold);
    }
    for metrics in input.dimension_metrics {
        hash_dimension_metrics(&mut hasher, *metrics);
    }
    hash_admission_rates(&mut hasher, input.clean_correct_activation);
    hash_admission_rates(&mut hasher, input.ambiguous_activation);
    hash_admission_rates(&mut hasher, input.unrelated_activation);
    hasher.update(&input.clean_top1_target_accuracy.to_bits().to_le_bytes());
    hasher.update(&input.ambiguous_reduction.to_bits().to_le_bytes());
    hasher.update(&[
        input.primary_passed as u8,
        input.clean_guard as u8,
        input.dimension_guard as u8,
        input.unrelated_guard as u8,
        disposition_tag(input.disposition),
    ]);
    *hasher.finalize().as_bytes()
}

fn hash_dimension_metrics(hasher: &mut blake3::Hasher, metrics: Sem2DimensionMetrics) {
    for value in [
        metrics.context_dimension,
        metrics.clean_count,
        metrics.ambiguous_count,
        metrics.unrelated_count,
        metrics.ood_count,
        metrics.exact_tie_count,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    for value in [
        metrics.clean_top1_target_accuracy,
        metrics.ood_raw_activation,
        metrics.ood_support_rejection_rate,
        metrics.mean_clean_similarity,
        metrics.mean_clean_margin,
        metrics.mean_clean_specificity,
        metrics.mean_ambiguous_similarity,
        metrics.mean_ambiguous_margin,
        metrics.mean_ambiguous_specificity,
        metrics.mean_unrelated_similarity,
        metrics.mean_unrelated_margin,
        metrics.mean_unrelated_specificity,
        metrics.mean_ood_similarity,
        metrics.mean_ood_margin,
        metrics.mean_ood_specificity,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hash_admission_rates(hasher, metrics.clean_correct_activation);
    hash_admission_rates(hasher, metrics.ambiguous_activation);
    hash_admission_rates(hasher, metrics.unrelated_activation);
}

fn hash_admission_rates(hasher: &mut blake3::Hasher, rates: Sem2AdmissionRates) {
    for value in [
        rates.raw,
        rates.specificity,
        rates.margin,
        rates.specificity_margin,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_margin_threshold(hasher: &mut blake3::Hasher, threshold: Sem2MarginThreshold) {
    hasher.update(&(threshold.context_dimension as u64).to_le_bytes());
    hasher.update(&(threshold.sample_count as u64).to_le_bytes());
    hasher.update(&threshold.median_margin.to_bits().to_le_bytes());
    hasher.update(&threshold.p95_margin.to_bits().to_le_bytes());
}

fn hash_optional_identity(hasher: &mut blake3::Hasher, identity: Option<ContextIdentity>) {
    match identity {
        Some(identity) => {
            hasher.update(&[1]);
            hasher.update(&identity.fast_hash.to_le_bytes());
            hasher.update(&identity.exact_digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_context(hasher: &mut blake3::Hasher, context: &[f32]) {
    hasher.update(&(context.len() as u64).to_le_bytes());
    for value in context {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn regime_tag(regime: Sem2Regime) -> u8 {
    match regime {
        Sem2Regime::CleanRelated => 1,
        Sem2Regime::Ambiguous => 2,
        Sem2Regime::Unrelated => 3,
        Sem2Regime::OodClampSaturation => 4,
    }
}

fn disposition_tag(disposition: Sem2Disposition) -> u8 {
    match disposition {
        Sem2Disposition::Pass => 1,
        Sem2Disposition::Tradeoff => 2,
        Sem2Disposition::NotEstablished => 3,
    }
}

fn validate_subject_sha(subject_sha: &str) -> Result<(), Sem2ExperimentError> {
    if subject_sha.len() == 40 && subject_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(Sem2ExperimentError::InvalidSubjectSha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUBJECT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn source_partitions_are_exact_disjoint() {
        let reference = generated_contexts(0, SYM_RSI_SEM_002_REFERENCE_COUNT_PER_DIMENSION);
        let null_queries = generated_contexts(
            60,
            SYM_RSI_SEM_002_NULL_MARGIN_QUERY_COUNT_PER_DIMENSION,
        );
        let evaluation = generated_contexts(100, SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION);
        let unrelated = generated_contexts(1000, SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION);
        let ood = generated_contexts(2000, SYM_RSI_SEM_002_OOD_COUNT_PER_DIMENSION);
        validate_source_partitions(&[
            ("reference", reference.as_slice()),
            ("null", null_queries.as_slice()),
            ("evaluation", evaluation.as_slice()),
            ("unrelated", unrelated.as_slice()),
            ("ood", ood.as_slice()),
        ])
        .unwrap();
    }

    #[test]
    fn all_evaluation_queries_pass_anti_leakage_checks() {
        let reference = generated_contexts(0, SYM_RSI_SEM_002_REFERENCE_COUNT_PER_DIMENSION);
        let null_queries = generated_contexts(
            60,
            SYM_RSI_SEM_002_NULL_MARGIN_QUERY_COUNT_PER_DIMENSION,
        );
        let evaluation = generated_contexts(100, SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION);
        let unrelated_sources =
            generated_contexts(1000, SYM_RSI_SEM_002_EVAL_BANK_COUNT_PER_DIMENSION);
        let ood_anchors = generated_contexts(2000, SYM_RSI_SEM_002_OOD_COUNT_PER_DIMENSION);
        let clean = clean_related_cases(&evaluation).unwrap();
        let ambiguous = ambiguous_cases(&evaluation).unwrap();
        let unrelated = unrelated_cases(&unrelated_sources);
        let ood = ood_cases(&ood_anchors).unwrap();
        validate_evaluation_cases(
            &reference,
            &null_queries,
            &evaluation,
            &clean,
            &ambiguous,
            &unrelated,
            &ood,
        )
        .unwrap();
    }

    #[test]
    fn benchmark_is_deterministic_for_same_subject() {
        let left = run_sym_rsi_sem_002(SUBJECT).unwrap();
        let right = run_sym_rsi_sem_002(SUBJECT).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.margin_thresholds.len(), 3);
        assert_eq!(left.dimension_metrics.len(), 3);
        assert!(
            left.margin_thresholds
                .iter()
                .all(|threshold| threshold.sample_count == 24)
        );
    }

    #[test]
    fn subject_changes_receipt_identity_not_geometry() {
        let left = run_sym_rsi_sem_002(SUBJECT).unwrap();
        let right = run_sym_rsi_sem_002("89abcdef0123456789abcdef0123456789abcdef").unwrap();
        assert_eq!(left.margin_thresholds, right.margin_thresholds);
        assert_eq!(left.dimension_metrics, right.dimension_metrics);
        assert_ne!(left.evidence_digest, right.evidence_digest);
    }

    #[test]
    fn combined_rule_is_never_more_permissive_than_components() {
        let receipt = run_sym_rsi_sem_002(SUBJECT).unwrap();
        for metrics in &receipt.dimension_metrics {
            assert!(
                metrics.ambiguous_activation.specificity_margin
                    <= metrics.ambiguous_activation.specificity + RATE_EPSILON
            );
            assert!(
                metrics.ambiguous_activation.specificity_margin
                    <= metrics.ambiguous_activation.margin + RATE_EPSILON
            );
            assert!(
                metrics.unrelated_activation.specificity_margin
                    <= metrics.unrelated_activation.specificity + RATE_EPSILON
            );
        }
    }

    #[test]
    fn all_ood_queries_are_outside_nominal_support() {
        let encoder = SemanticContextEncoder::default();
        let anchors = generated_contexts(2000, SYM_RSI_SEM_002_OOD_COUNT_PER_DIMENSION);
        for case in ood_cases(&anchors).unwrap() {
            assert!(
                !assess_semantic_context_support(encoder, &case.query)
                    .unwrap()
                    .within_nominal_support()
            );
        }
    }

    #[test]
    fn invalid_subject_sha_fails_closed() {
        assert_eq!(
            run_sym_rsi_sem_002("not-a-sha"),
            Err(Sem2ExperimentError::InvalidSubjectSha)
        );
    }
}
