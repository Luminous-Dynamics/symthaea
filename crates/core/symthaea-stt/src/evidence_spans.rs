// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound temporal evidence citation scoring for speech claims.
//!
//! This module evaluates emitted claim IDs and their cited half-open sample
//! intervals against independently adjudicated reference support intervals.
//! It does not inspect audio, establish annotation truth, or prove that a model
//! internally relied on a citation. All interval lengths are computed after
//! per-claim interval union so overlapping spans cannot inflate a score.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Half-open sample interval [start_sample, end_sample).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleInterval {
    pub start_sample: u64,
    pub end_sample: u64,
}

impl SampleInterval {
    pub fn new(start_sample: u64, end_sample: u64) -> Self {
        Self { start_sample, end_sample }
    }

    pub fn sample_count(self) -> u64 {
        self.end_sample - self.start_sample
    }
}

/// Reference claim and independently adjudicated source support intervals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceClaimEvidence {
    pub claim_id: String,
    pub support_intervals: Vec<SampleInterval>,
}

/// One model-emitted citation bound to the exact source audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceCitation {
    pub evaluation_audio_sha256: String,
    pub interval: SampleInterval,
}

/// One model-emitted claim and zero or more supporting citations.
///
/// An empty citation vector is valid input: it represents a claim for which the
/// system supplied no evidence. The scorer will not count it as evidence-backed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictedClaimEvidence {
    pub claim_id: String,
    pub citations: Vec<EvidenceCitation>,
}

/// One evaluation item with a frozen source identity and sample coordinate system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceSpanTrial {
    pub trial_id: String,
    pub evaluation_audio_sha256: String,
    pub sample_rate_hz: u32,
    pub sample_count: u64,
    pub reference_claims: Vec<ReferenceClaimEvidence>,
    pub predicted_claims: Vec<PredictedClaimEvidence>,
}

/// Integer numerator/denominator and optional rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceRate {
    pub numerator: u128,
    pub denominator: u128,
    pub rate: Option<f64>,
}

impl EvidenceRate {
    fn new(numerator: u128, denominator: u128) -> Self {
        Self {
            numerator,
            denominator,
            rate: (denominator > 0).then(|| numerator as f64 / denominator as f64),
        }
    }
}

/// Claim correctness and temporal evidence-localization results.
///
/// Counts and interval lengths are claim-conditioned: the same source samples
/// may legitimately support two distinct claims and are counted once per claim.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceSpanResult {
    /// Exact trial and audio coordinate identity used by this result.
    pub trial_id: String,
    pub evaluation_audio_sha256: String,
    pub sample_rate_hz: u32,
    pub sample_count: u64,
    pub total_reference_claims: usize,
    pub total_predicted_claims: usize,
    pub matched_claims: usize,
    pub unsupported_predicted_claims: usize,
    pub predicted_claims_without_citations: usize,
    pub reference_claims_with_supported_citations: usize,
    pub citation_count: usize,
    pub reference_support_interval_count: usize,
    pub predicted_citation_samples: u128,
    pub reference_support_samples: u128,
    pub overlapping_support_samples: u128,
    /// Correct predicted claim IDs divided by all predicted claim IDs.
    pub claim_precision: EvidenceRate,
    /// Correct predicted claim IDs divided by all reference claim IDs.
    pub claim_recall: EvidenceRate,
    /// Reference claims with at least one intersecting citation divided by references.
    pub reference_claim_evidence_coverage: EvidenceRate,
    /// Intersecting cited samples divided by all cited samples.
    pub citation_sample_precision: EvidenceRate,
    /// Intersecting reference samples divided by all reference support samples.
    pub reference_sample_recall: EvidenceRate,
    /// Intersection divided by the union of predicted and reference support samples.
    pub sample_iou: EvidenceRate,
    /// Predicted claim IDs absent from the reference ontology divided by predictions.
    pub unsupported_predicted_claim_rate: EvidenceRate,
    /// Cited samples that do not support the predicted claim divided by all cited samples.
    pub unsupported_citation_sample_rate: EvidenceRate,
}

/// Invalid source identity, claim annotation, citation, or sample interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceSpanError {
    EmptyTrialId,
    InvalidEvaluationAudioDigest,
    InvalidSampleRate,
    EmptyAudio,
    EmptyReferenceClaims,
    EmptyClaimId,
    DuplicateReferenceClaim(String),
    DuplicatePredictedClaim(String),
    EmptyReferenceSupport(String),
    InvalidInterval {
        claim_id: String,
        start_sample: u64,
        end_sample: u64,
        sample_count: u64,
    },
    CitationEvaluationDigestMismatch {
        claim_id: String,
        expected: String,
        actual: String,
    },
}

impl fmt::Display for EvidenceSpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for EvidenceSpanError {}

/// Score emitted claims and source-bound citations against reference support.
///
/// Claim-ID correctness and citation localization are independent outputs. A
/// correct claim with a missing or misplaced citation can have perfect claim
/// precision/recall while its evidence coverage or sample precision is zero.
/// An unsupported claim with citations contributes to the citation denominator
/// but cannot contribute to the overlap numerator.
///
/// Intervals are half-open sample ranges. Overlaps and duplicates are unioned
/// independently for each claim before lengths and intersections are measured.
pub fn score_evidence_spans(
    trial: &EvidenceSpanTrial,
) -> Result<EvidenceSpanResult, EvidenceSpanError> {
    if trial.trial_id.trim().is_empty() {
        return Err(EvidenceSpanError::EmptyTrialId);
    }
    if !is_sha256(&trial.evaluation_audio_sha256) {
        return Err(EvidenceSpanError::InvalidEvaluationAudioDigest);
    }
    if trial.sample_rate_hz == 0 {
        return Err(EvidenceSpanError::InvalidSampleRate);
    }
    if trial.sample_count == 0 {
        return Err(EvidenceSpanError::EmptyAudio);
    }
    if trial.reference_claims.is_empty() {
        return Err(EvidenceSpanError::EmptyReferenceClaims);
    }

    let mut references: HashMap<&str, Vec<SampleInterval>> =
        HashMap::with_capacity(trial.reference_claims.len());
    let mut reference_support_samples = 0_u128;
    let mut reference_support_interval_count = 0_usize;

    for claim in &trial.reference_claims {
        if claim.claim_id.trim().is_empty() {
            return Err(EvidenceSpanError::EmptyClaimId);
        }
        if claim.support_intervals.is_empty() {
            return Err(EvidenceSpanError::EmptyReferenceSupport(claim.claim_id.clone()));
        }
        validate_intervals(&claim.claim_id, &claim.support_intervals, trial.sample_count)?;
        let merged = union_intervals(&claim.support_intervals);
        let samples = total_interval_samples(&merged);
        reference_support_samples += samples;
        reference_support_interval_count += claim.support_intervals.len();
        if references.insert(claim.claim_id.as_str(), merged).is_some() {
            return Err(EvidenceSpanError::DuplicateReferenceClaim(claim.claim_id.clone()));
        }
    }

    let mut predicted_ids = HashSet::with_capacity(trial.predicted_claims.len());
    let mut predicted_citation_samples = 0_u128;
    let mut citation_count = 0_usize;
    let mut predicted_by_claim: HashMap<&str, Vec<SampleInterval>> =
        HashMap::with_capacity(trial.predicted_claims.len());

    for claim in &trial.predicted_claims {
        if claim.claim_id.trim().is_empty() {
            return Err(EvidenceSpanError::EmptyClaimId);
        }
        if !predicted_ids.insert(claim.claim_id.as_str()) {
            return Err(EvidenceSpanError::DuplicatePredictedClaim(claim.claim_id.clone()));
        }

        let mut intervals = Vec::with_capacity(claim.citations.len());
        for citation in &claim.citations {
            if !citation.evaluation_audio_sha256.eq_ignore_ascii_case(&trial.evaluation_audio_sha256) {
                return Err(EvidenceSpanError::CitationEvaluationDigestMismatch {
                    claim_id: claim.claim_id.clone(),
                    expected: trial.evaluation_audio_sha256.clone(),
                    actual: citation.evaluation_audio_sha256.clone(),
                });
            }
            validate_intervals(
                &claim.claim_id,
                std::slice::from_ref(&citation.interval),
                trial.sample_count,
            )?;
            intervals.push(citation.interval);
            citation_count += 1;
        }

        let merged = union_intervals(&intervals);
        predicted_citation_samples += total_interval_samples(&merged);
        predicted_by_claim.insert(claim.claim_id.as_str(), merged);
    }

    let mut matched_claims = 0_usize;
    let mut unsupported_predicted_claims = 0_usize;
    let mut predicted_claims_without_citations = 0_usize;
    let mut reference_claims_with_supported_citations = 0_usize;
    let mut overlapping_support_samples = 0_u128;

    for claim in &trial.predicted_claims {
        let Some(reference_intervals) = references.get(claim.claim_id.as_str()) else {
            unsupported_predicted_claims += 1;
            continue;
        };
        matched_claims += 1;

        let predicted_intervals = predicted_by_claim
            .get(claim.claim_id.as_str())
            .expect("every predicted claim has a citation entry");
        if predicted_intervals.is_empty() {
            predicted_claims_without_citations += 1;
            continue;
        }

        let overlap = intersection_sample_count(reference_intervals, predicted_intervals);
        overlapping_support_samples += overlap;
        if overlap > 0 {
            reference_claims_with_supported_citations += 1;
        }
    }

    let predicted_claim_count = trial.predicted_claims.len() as u128;
    let reference_claim_count = trial.reference_claims.len() as u128;
    let iou_denominator = predicted_citation_samples
        .saturating_add(reference_support_samples)
        .saturating_sub(overlapping_support_samples);

    Ok(EvidenceSpanResult {
        trial_id: trial.trial_id.clone(),
        evaluation_audio_sha256: trial.evaluation_audio_sha256.clone(),
        sample_rate_hz: trial.sample_rate_hz,
        sample_count: trial.sample_count,
        total_reference_claims: trial.reference_claims.len(),
        total_predicted_claims: trial.predicted_claims.len(),
        matched_claims,
        unsupported_predicted_claims,
        predicted_claims_without_citations,
        reference_claims_with_supported_citations,
        citation_count,
        reference_support_interval_count,
        predicted_citation_samples,
        reference_support_samples,
        overlapping_support_samples,
        claim_precision: EvidenceRate::new(matched_claims as u128, predicted_claim_count),
        claim_recall: EvidenceRate::new(matched_claims as u128, reference_claim_count),
        reference_claim_evidence_coverage: EvidenceRate::new(
            reference_claims_with_supported_citations as u128,
            reference_claim_count,
        ),
        citation_sample_precision: EvidenceRate::new(
            overlapping_support_samples,
            predicted_citation_samples,
        ),
        reference_sample_recall: EvidenceRate::new(
            overlapping_support_samples,
            reference_support_samples,
        ),
        sample_iou: EvidenceRate::new(overlapping_support_samples, iou_denominator),
        unsupported_predicted_claim_rate: EvidenceRate::new(
            unsupported_predicted_claims as u128,
            predicted_claim_count,
        ),
        unsupported_citation_sample_rate: EvidenceRate::new(
            predicted_citation_samples.saturating_sub(overlapping_support_samples),
            predicted_citation_samples,
        ),
    })
}

fn validate_intervals(
    claim_id: &str,
    intervals: &[SampleInterval],
    sample_count: u64,
) -> Result<(), EvidenceSpanError> {
    for interval in intervals {
        if interval.start_sample >= interval.end_sample || interval.end_sample > sample_count {
            return Err(EvidenceSpanError::InvalidInterval {
                claim_id: claim_id.to_string(),
                start_sample: interval.start_sample,
                end_sample: interval.end_sample,
                sample_count,
            });
        }
    }
    Ok(())
}

/// Sort and union intervals, merging overlap and touching half-open ranges.
fn union_intervals(intervals: &[SampleInterval]) -> Vec<SampleInterval> {
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|interval| (interval.start_sample, interval.end_sample));
    let mut merged: Vec<SampleInterval> = Vec::with_capacity(sorted.len());

    for interval in sorted {
        if let Some(last) = merged.last_mut() {
            if interval.start_sample <= last.end_sample {
                last.end_sample = last.end_sample.max(interval.end_sample);
                continue;
            }
        }
        merged.push(interval);
    }

    merged
}

fn total_interval_samples(intervals: &[SampleInterval]) -> u128 {
    intervals.iter().map(|interval| interval.sample_count() as u128).sum()
}

fn intersection_sample_count(
    left: &[SampleInterval],
    right: &[SampleInterval],
) -> u128 {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut total = 0_u128;

    while left_index < left.len() && right_index < right.len() {
        let a = left[left_index];
        let b = right[right_index];
        let start = a.start_sample.max(b.start_sample);
        let end = a.end_sample.min(b.end_sample);
        if start < end {
            total += (end - start) as u128;
        }

        if a.end_sample <= b.end_sample {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }

    total
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::{
        EvidenceCitation, EvidenceSpanError, EvidenceSpanTrial, PredictedClaimEvidence,
        ReferenceClaimEvidence, SampleInterval, score_evidence_spans,
    };

    const AUDIO: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_AUDIO: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn interval(start: u64, end: u64) -> SampleInterval {
        SampleInterval::new(start, end)
    }

    fn reference(claim_id: &str, spans: Vec<SampleInterval>) -> ReferenceClaimEvidence {
        ReferenceClaimEvidence {
            claim_id: claim_id.into(),
            support_intervals: spans,
        }
    }

    fn prediction(claim_id: &str, spans: Vec<SampleInterval>) -> PredictedClaimEvidence {
        PredictedClaimEvidence {
            claim_id: claim_id.into(),
            citations: spans
                .into_iter()
                .map(|interval| EvidenceCitation {
                    evaluation_audio_sha256: AUDIO.into(),
                    interval,
                })
                .collect(),
        }
    }

    fn trial(
        reference_claims: Vec<ReferenceClaimEvidence>,
        predicted_claims: Vec<PredictedClaimEvidence>,
    ) -> EvidenceSpanTrial {
        EvidenceSpanTrial {
            trial_id: "trial-1".into(),
            evaluation_audio_sha256: AUDIO.into(),
            sample_rate_hz: 16_000,
            sample_count: 100,
            reference_claims,
            predicted_claims,
        }
    }

    #[test]
    fn exact_claim_and_citation_score_perfect() {
        let result = score_evidence_spans(&trial(
            vec![reference("door-open", vec![interval(10, 20)])],
            vec![prediction("door-open", vec![interval(10, 20)])],
        )).unwrap();
        assert_eq!(result.claim_precision.rate, Some(1.0));
        assert_eq!(result.claim_recall.rate, Some(1.0));
        assert_eq!(result.reference_claim_evidence_coverage.rate, Some(1.0));
        assert_eq!(result.citation_sample_precision.rate, Some(1.0));
        assert_eq!(result.reference_sample_recall.rate, Some(1.0));
        assert_eq!(result.sample_iou.rate, Some(1.0));
    }

    #[test]
    fn broad_citation_is_penalized_by_sample_precision() {
        let result = score_evidence_spans(&trial(
            vec![reference("door-open", vec![interval(10, 20)])],
            vec![prediction("door-open", vec![interval(0, 30)])],
        )).unwrap();
        assert_eq!(result.citation_sample_precision.rate, Some(1.0 / 3.0));
        assert_eq!(result.reference_sample_recall.rate, Some(1.0));
        assert_eq!(result.sample_iou.rate, Some(1.0 / 3.0));
    }

    #[test]
    fn partial_temporal_overlap_is_not_full_evidence_support() {
        let result = score_evidence_spans(&trial(
            vec![reference("quantity", vec![interval(10, 20)])],
            vec![prediction("quantity", vec![interval(15, 25)])],
        )).unwrap();
        assert_eq!(result.overlapping_support_samples, 5);
        assert_eq!(result.citation_sample_precision.rate, Some(0.5));
        assert_eq!(result.reference_sample_recall.rate, Some(0.5));
        assert_eq!(result.sample_iou.rate, Some(1.0 / 3.0));
    }

    #[test]
    fn wrong_claim_id_cannot_borrow_another_claims_evidence() {
        let result = score_evidence_spans(&trial(
            vec![reference("heard-go", vec![interval(10, 20)])],
            vec![prediction("heard-stop", vec![interval(10, 20)])],
        )).unwrap();
        assert_eq!(result.claim_precision.rate, Some(0.0));
        assert_eq!(result.claim_recall.rate, Some(0.0));
        assert_eq!(result.reference_claim_evidence_coverage.rate, Some(0.0));
        assert_eq!(result.citation_sample_precision.rate, Some(0.0));
        assert_eq!(result.unsupported_predicted_claim_rate.rate, Some(1.0));
    }

    #[test]
    fn correct_claim_without_citation_is_not_evidence_covered() {
        let result = score_evidence_spans(&trial(
            vec![reference("negation", vec![interval(30, 35)])],
            vec![prediction("negation", vec![])],
        )).unwrap();
        assert_eq!(result.claim_precision.rate, Some(1.0));
        assert_eq!(result.claim_recall.rate, Some(1.0));
        assert_eq!(result.reference_claim_evidence_coverage.rate, Some(0.0));
        assert_eq!(result.citation_sample_precision.rate, None);
        assert_eq!(result.reference_sample_recall.rate, Some(0.0));
        assert_eq!(result.predicted_claims_without_citations, 1);
    }

    #[test]
    fn overlapping_prediction_intervals_are_unioned_before_scoring() {
        let result = score_evidence_spans(&trial(
            vec![reference("event", vec![interval(10, 20)])],
            vec![prediction("event", vec![interval(10, 18), interval(15, 22)])],
        )).unwrap();
        assert_eq!(result.predicted_citation_samples, 12);
        assert_eq!(result.overlapping_support_samples, 10);
        assert_eq!(result.citation_sample_precision.rate, Some(10.0 / 12.0));
    }

    #[test]
    fn multiple_reference_support_regions_are_scored_as_a_union() {
        let result = score_evidence_spans(&trial(
            vec![reference("two-words", vec![interval(10, 15), interval(20, 25)])],
            vec![prediction("two-words", vec![interval(12, 22)])],
        )).unwrap();
        assert_eq!(result.reference_support_samples, 10);
        assert_eq!(result.predicted_citation_samples, 10);
        assert_eq!(result.overlapping_support_samples, 5);
        assert_eq!(result.citation_sample_precision.rate, Some(0.5));
        assert_eq!(result.reference_sample_recall.rate, Some(0.5));
    }

    #[test]
    fn unsupported_claims_and_their_citations_remain_in_denominators() {
        let result = score_evidence_spans(&trial(
            vec![reference("answer", vec![interval(10, 20)])],
            vec![
                prediction("answer", vec![interval(10, 20)]),
                prediction("invented-answer", vec![interval(40, 50)]),
            ],
        )).unwrap();
        assert_eq!(result.total_predicted_claims, 2);
        assert_eq!(result.unsupported_predicted_claims, 1);
        assert_eq!(result.claim_precision.rate, Some(0.5));
        assert_eq!(result.citation_sample_precision.rate, Some(0.5));
        assert_eq!(result.unsupported_citation_sample_rate.rate, Some(0.5));
    }

    #[test]
    fn empty_prediction_set_is_reported_without_false_success() {
        let result = score_evidence_spans(&trial(
            vec![reference("answer", vec![interval(10, 20)])],
            vec![],
        )).unwrap();
        assert_eq!(result.claim_precision.rate, None);
        assert_eq!(result.claim_recall.rate, Some(0.0));
        assert_eq!(result.reference_claim_evidence_coverage.rate, Some(0.0));
        assert_eq!(result.citation_sample_precision.rate, None);
        assert_eq!(result.reference_sample_recall.rate, Some(0.0));
    }

    #[test]
    fn rejects_citation_bound_to_different_audio_digest() {
        let mut bad = prediction("answer", vec![interval(10, 20)]);
        bad.citations[0].evaluation_audio_sha256 = OTHER_AUDIO.into();
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![reference("answer", vec![interval(10, 20)])],
                vec![bad],
            )),
            Err(EvidenceSpanError::CitationEvaluationDigestMismatch { .. })
        ));
    }

    #[test]
    fn rejects_out_of_bounds_reference_interval() {
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![reference("answer", vec![interval(90, 101)])],
                vec![],
            )),
            Err(EvidenceSpanError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn rejects_out_of_bounds_prediction_interval() {
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![reference("answer", vec![interval(10, 20)])],
                vec![prediction("answer", vec![interval(90, 101)])],
            )),
            Err(EvidenceSpanError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn rejects_zero_length_interval() {
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![reference("answer", vec![interval(10, 10)])],
                vec![],
            )),
            Err(EvidenceSpanError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_reference_claim_ids() {
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![
                    reference("answer", vec![interval(10, 20)]),
                    reference("answer", vec![interval(20, 30)]),
                ],
                vec![],
            )),
            Err(EvidenceSpanError::DuplicateReferenceClaim(_))
        ));
    }

    #[test]
    fn rejects_duplicate_predicted_claim_ids() {
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![reference("answer", vec![interval(10, 20)])],
                vec![
                    prediction("answer", vec![interval(10, 20)]),
                    prediction("answer", vec![interval(20, 30)]),
                ],
            )),
            Err(EvidenceSpanError::DuplicatePredictedClaim(_))
        ));
    }

    #[test]
    fn rejects_empty_reference_support() {
        assert!(matches!(
            score_evidence_spans(&trial(
                vec![reference("answer", vec![])],
                vec![],
            )),
            Err(EvidenceSpanError::EmptyReferenceSupport(_))
        ));
    }

    #[test]
    fn rejects_invalid_trial_metadata() {
        let mut bad = trial(vec![reference("answer", vec![interval(10, 20)])], vec![]);
        bad.sample_rate_hz = 0;
        assert_eq!(score_evidence_spans(&bad), Err(EvidenceSpanError::InvalidSampleRate));
        bad.sample_rate_hz = 16_000;
        bad.sample_count = 0;
        assert_eq!(score_evidence_spans(&bad), Err(EvidenceSpanError::EmptyAudio));
        bad.sample_count = 100;
        bad.evaluation_audio_sha256 = "not-a-digest".into();
        assert_eq!(score_evidence_spans(&bad), Err(EvidenceSpanError::InvalidEvaluationAudioDigest));
    }

    #[test]
    fn rejects_empty_trial_and_claim_identifiers() {
        let mut bad = trial(vec![reference("answer", vec![interval(10, 20)])], vec![]);
        bad.trial_id.clear();
        assert_eq!(score_evidence_spans(&bad), Err(EvidenceSpanError::EmptyTrialId));
        bad.trial_id = "trial-1".into();
        bad.reference_claims[0].claim_id.clear();
        assert_eq!(score_evidence_spans(&bad), Err(EvidenceSpanError::EmptyClaimId));
    }

    #[test]
    fn touching_half_open_intervals_are_merged_without_changing_length() {
        let result = score_evidence_spans(&trial(
            vec![reference("answer", vec![interval(10, 15), interval(15, 20)])],
            vec![prediction("answer", vec![interval(10, 20)])],
        )).unwrap();
        assert_eq!(result.reference_support_samples, 10);
        assert_eq!(result.overlapping_support_samples, 10);
        assert_eq!(result.sample_iou.rate, Some(1.0));
    }

    #[test]
    fn zero_denominators_are_null_not_zero() {
        let result = score_evidence_spans(&trial(
            vec![reference("answer", vec![interval(10, 20)])],
            vec![prediction("answer", vec![])],
        )).unwrap();
        assert_eq!(result.citation_sample_precision.numerator, 0);
        assert_eq!(result.citation_sample_precision.denominator, 0);
        assert_eq!(result.citation_sample_precision.rate, None);
        assert_eq!(result.unsupported_citation_sample_rate.rate, None);
    }
}
