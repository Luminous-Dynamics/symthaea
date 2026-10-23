//! Risk-coverage analysis and fixed-sequence certification for selective retrieval.
//!
//! The descriptive curve is useful for development-time policy analysis. Its
//! data-derived score breakpoints must not be used as if they were a locked-test
//! certificate. For certification, pass a predeclared, strictly descending
//! threshold sequence to certify_fixed_threshold_sequence and evaluate it on
//! qualification data not used to tune those thresholds.
//!
//! The certification routine uses the Clopper-Pearson upper bound from the
//! parent crate and stops at the first candidate threshold that cannot be
//! certified. Under the documented binomial assumptions and a threshold
//! sequence fixed independently of evaluation labels, this is a fixed-sequence
//! multiple-testing protocol. It does not guarantee reliability under arbitrary
//! distribution shift or remove the need for group-aware sampling.

use std::collections::HashSet;

use crate::{RiskBudget, one_sided_clopper_pearson_upper};

/// One scored query for selective retrieval evaluation.
///
/// Larger acceptance_score means the frozen routing policy is more willing to
/// return approximate results. unsafe_outcome must be assigned by a
/// preregistered evaluator: Some(true) is unsafe, Some(false) is safe, and
/// None means the outcome is unknown/unlabelled. An unknown accepted outcome
/// is never silently treated as safe for certification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskObservation {
    pub query_id: u64,
    pub acceptance_score: f64,
    pub unsafe_outcome: Option<bool>,
}

/// Input validation failures for the risk-coverage routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCoverageError {
    EmptyInput,
    NonFiniteAcceptanceScore,
    DuplicateQueryId,
    EmptyThresholdSequence,
    NonFiniteThreshold,
    ThresholdsNotStrictlyDescending,
}

impl std::fmt::Display for RiskCoverageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "at least one observation is required"),
            Self::NonFiniteAcceptanceScore => {
                write!(formatter, "acceptance scores must be finite")
            }
            Self::DuplicateQueryId => write!(formatter, "query IDs must be unique"),
            Self::EmptyThresholdSequence => {
                write!(formatter, "at least one threshold is required")
            }
            Self::NonFiniteThreshold => write!(formatter, "thresholds must be finite"),
            Self::ThresholdsNotStrictlyDescending => {
                write!(formatter, "thresholds must be strictly descending")
            }
        }
    }
}

impl std::error::Error for RiskCoverageError {}

/// Descriptive point on a development-set risk-coverage curve.
///
/// observed_label_risk uses only accepted cases with known labels.
/// optimistic_risk assigns every unknown accepted label as safe;
/// pessimistic_risk assigns every unknown accepted label as unsafe. These two
/// values are deterministic partial-label bounds, not confidence intervals.
///
/// No confidence bound is attached to this data-derived curve: use a fixed,
/// predeclared threshold sequence or a single locked-test threshold to qualify
/// risk.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskCoveragePoint {
    /// None denotes the zero-coverage origin; otherwise this score threshold is
    /// inclusive (accept if score >= threshold).
    pub threshold: Option<f64>,
    pub total_count: usize,
    pub accepted_count: usize,
    pub labelled_accepted_count: usize,
    pub unknown_accepted_count: usize,
    pub unsafe_labelled_count: usize,
    /// Accepted queries divided by all queries.
    pub coverage: f64,
    /// Labelled accepted queries divided by all queries.
    pub labelled_coverage: f64,
    /// Labelled accepted queries divided by accepted queries, if any accepted.
    pub accepted_label_completeness: Option<f64>,
    /// Unsafe labelled / labelled accepted, when at least one label exists.
    pub observed_label_risk: Option<f64>,
    /// Unsafe labelled / all accepted (unknown labels optimistically safe).
    pub optimistic_risk: Option<f64>,
    /// (unsafe labelled + unknown accepted) / all accepted.
    pub pessimistic_risk: Option<f64>,
}

/// Build a descriptive risk-coverage curve from development data.
///
/// The curve starts at zero coverage, then adds every distinct observed score
/// from highest to lowest. All cases with exactly equal scores enter together;
/// arbitrary row ordering never splits a tie. This curve is descriptive only:
/// selecting a favorable point from it and reporting a confidence guarantee on
/// the same data would reuse evaluation labels for policy selection.
pub fn risk_coverage_curve(
    observations: &[RiskObservation],
) -> Result<Vec<RiskCoveragePoint>, RiskCoverageError> {
    validate_observations(observations)?;

    let mut scores: Vec<f64> = observations
        .iter()
        .map(|observation| observation.acceptance_score)
        .collect();
    scores.sort_by(|left, right| right.total_cmp(left));
    scores.dedup_by(|left, right| *left == *right);

    let mut curve = Vec::with_capacity(scores.len() + 1);
    curve.push(summarize_at_threshold(observations, None));

    for threshold in scores {
        curve.push(summarize_at_threshold(observations, Some(threshold)));
    }
    Ok(curve)
}

/// Status of one fixed-sequence risk-certification candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdCertificationStatus {
    /// The one-sided upper bound is at or below the risk budget.
    Certified,
    /// Too few accepted cases exist; no hypothesis test was performed.
    InsufficientEvidence,
    /// At least one accepted case has an unknown outcome; certification stops.
    UnknownAcceptedLabels,
    /// The upper confidence bound exceeds the risk budget; sequence stops.
    RiskBudgetExceeded,
    /// Not tested because an earlier eligible threshold failed certification.
    NotTestedAfterStop,
}

/// Auditable result for one preregistered threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdCertificationPoint {
    pub threshold: f64,
    pub accepted_count: usize,
    pub labelled_accepted_count: usize,
    pub unknown_accepted_count: usize,
    pub unsafe_labelled_count: usize,
    /// None if evidence is insufficient or accepted outcomes include unknown labels.
    pub upper_confidence_bound: Option<f64>,
    pub status: ThresholdCertificationStatus,
}

/// Results of fixed-sequence threshold certification.
///
/// Thresholds must be specified independently of evaluation labels and supplied
/// from most selective (highest threshold) to least selective (lowest threshold).
/// Certification continues after a threshold with too few accepted cases, but
/// stops at the first threshold containing unknown accepted labels or exceeding
/// the risk budget. Later thresholds are explicitly marked untested.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedSequenceRiskReport {
    pub confidence_level: f64,
    pub maximum_unsafe_rate: f64,
    pub minimum_accepted: usize,
    /// Number of thresholds actually certified.
    pub certified_threshold_count: usize,
    /// Threshold at which certification stopped, if any.
    pub stopped_at_threshold: Option<f64>,
    pub points: Vec<ThresholdCertificationPoint>,
}

/// Certify a predeclared ordered threshold sequence on held-out data.
///
/// This implements a fixed-sequence testing pattern: test candidate thresholds
/// from highest acceptance score to lowest; continue only while each eligible
/// threshold is certified. Thresholds must be selected before viewing evaluation
/// labels. Do not pass data-derived thresholds from risk_coverage_curve into this
/// function after inspecting labels and call the result an independent guarantee.
pub fn certify_fixed_threshold_sequence(
    observations: &[RiskObservation],
    thresholds: &[f64],
    budget: RiskBudget,
) -> Result<FixedSequenceRiskReport, RiskCoverageError> {
    validate_observations(observations)?;
    if thresholds.is_empty() {
        return Err(RiskCoverageError::EmptyThresholdSequence);
    }
    if thresholds.iter().any(|threshold| !threshold.is_finite()) {
        return Err(RiskCoverageError::NonFiniteThreshold);
    }
    if thresholds.windows(2).any(|pair| pair[0] <= pair[1]) {
        return Err(RiskCoverageError::ThresholdsNotStrictlyDescending);
    }

    let mut points = Vec::with_capacity(thresholds.len());
    let mut stopped_at_threshold = None;
    let mut stopped = false;
    let mut certified_threshold_count = 0;

    for &threshold in thresholds {
        if stopped {
            points.push(ThresholdCertificationPoint {
                threshold,
                accepted_count: observations
                    .iter()
                    .filter(|observation| observation.acceptance_score >= threshold)
                    .count(),
                labelled_accepted_count: 0,
                unknown_accepted_count: 0,
                unsafe_labelled_count: 0,
                upper_confidence_bound: None,
                status: ThresholdCertificationStatus::NotTestedAfterStop,
            });
            continue;
        }

        let point = summarize_at_threshold(observations, Some(threshold));
        let mut certified_point = ThresholdCertificationPoint {
            threshold,
            accepted_count: point.accepted_count,
            labelled_accepted_count: point.labelled_accepted_count,
            unknown_accepted_count: point.unknown_accepted_count,
            unsafe_labelled_count: point.unsafe_labelled_count,
            upper_confidence_bound: None,
            status: ThresholdCertificationStatus::InsufficientEvidence,
        };

        if point.accepted_count < budget.minimum_accepted() {
            // This candidate is not tested; lower thresholds may have enough cases.
            certified_point.status = ThresholdCertificationStatus::InsufficientEvidence;
        } else if point.unknown_accepted_count > 0 {
            certified_point.status = ThresholdCertificationStatus::UnknownAcceptedLabels;
            stopped = true;
            stopped_at_threshold = Some(threshold);
        } else {
            // With no unknown accepted labels, the binomial denominator is complete.
            let upper = one_sided_clopper_pearson_upper(
                point.unsafe_labelled_count,
                point.accepted_count,
                budget.confidence_level(),
            )
            .map_err(|_| RiskCoverageError::EmptyInput)?;
            certified_point.upper_confidence_bound = Some(upper);
            if upper <= budget.maximum_unsafe_rate() {
                certified_point.status = ThresholdCertificationStatus::Certified;
                certified_threshold_count += 1;
            } else {
                certified_point.status = ThresholdCertificationStatus::RiskBudgetExceeded;
                stopped = true;
                stopped_at_threshold = Some(threshold);
            }
        }

        points.push(certified_point);
    }

    Ok(FixedSequenceRiskReport {
        confidence_level: budget.confidence_level(),
        maximum_unsafe_rate: budget.maximum_unsafe_rate(),
        minimum_accepted: budget.minimum_accepted(),
        certified_threshold_count,
        stopped_at_threshold,
        points,
    })
}

fn validate_observations(
    observations: &[RiskObservation],
) -> Result<(), RiskCoverageError> {
    if observations.is_empty() {
        return Err(RiskCoverageError::EmptyInput);
    }
    let mut ids = HashSet::with_capacity(observations.len());
    for observation in observations {
        if !observation.acceptance_score.is_finite() {
            return Err(RiskCoverageError::NonFiniteAcceptanceScore);
        }
        if !ids.insert(observation.query_id) {
            return Err(RiskCoverageError::DuplicateQueryId);
        }
    }
    Ok(())
}

fn summarize_at_threshold(
    observations: &[RiskObservation],
    threshold: Option<f64>,
) -> RiskCoveragePoint {
    let accepted: Vec<&RiskObservation> = observations
        .iter()
        .filter(|observation| {
            threshold.is_some_and(|value| observation.acceptance_score >= value)
        })
        .collect();
    let accepted_count = accepted.len();
    let labelled_accepted_count = accepted
        .iter()
        .filter(|observation| observation.unsafe_outcome.is_some())
        .count();
    let unknown_accepted_count = accepted_count - labelled_accepted_count;
    let unsafe_labelled_count = accepted
        .iter()
        .filter(|observation| observation.unsafe_outcome == Some(true))
        .count();

    let total_count = observations.len();
    let coverage = accepted_count as f64 / total_count as f64;
    let labelled_coverage = labelled_accepted_count as f64 / total_count as f64;
    let accepted_label_completeness = (accepted_count > 0)
        .then_some(labelled_accepted_count as f64 / accepted_count as f64);
    let observed_label_risk = (labelled_accepted_count > 0)
        .then_some(unsafe_labelled_count as f64 / labelled_accepted_count as f64);
    let optimistic_risk =
        (accepted_count > 0).then_some(unsafe_labelled_count as f64 / accepted_count as f64);
    let pessimistic_risk = (accepted_count > 0)
        .then_some((unsafe_labelled_count + unknown_accepted_count) as f64 / accepted_count as f64);

    RiskCoveragePoint {
        threshold,
        total_count,
        accepted_count,
        labelled_accepted_count,
        unknown_accepted_count,
        unsafe_labelled_count,
        coverage,
        labelled_coverage,
        accepted_label_completeness,
        observed_label_risk,
        optimistic_risk,
        pessimistic_risk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(id: u64, score: f64, unsafe_outcome: Option<bool>) -> RiskObservation {
        RiskObservation { query_id: id, acceptance_score: score, unsafe_outcome }
    }

    #[test]
    fn curve_starts_at_zero_and_adds_unique_scores_highest_first() {
        let observations = [
            obs(1, 0.8, Some(false)),
            obs(2, 0.6, Some(true)),
            obs(3, 0.8, Some(true)),
            obs(4, 0.2, Some(false)),
        ];
        let curve = risk_coverage_curve(&observations).unwrap();
        assert_eq!(curve.len(), 4);
        assert_eq!(curve[0].accepted_count, 0);
        assert_eq!(curve[1].threshold, Some(0.8));
        assert_eq!(curve[1].accepted_count, 2);
        assert_eq!(curve[2].threshold, Some(0.6));
        assert_eq!(curve[2].accepted_count, 3);
        assert_eq!(curve[3].threshold, Some(0.2));
        assert_eq!(curve[3].accepted_count, 4);
    }

    #[test]
    fn tied_scores_enter_together_regardless_of_row_order() {
        let first = [
            obs(1, 0.5, Some(false)),
            obs(2, 0.5, Some(true)),
            obs(3, 0.4, Some(false)),
        ];
        let shuffled = [first[2], first[1], first[0]];
        let left = risk_coverage_curve(&first).unwrap();
        let right = risk_coverage_curve(&shuffled).unwrap();
        assert_eq!(left[1].accepted_count, 2);
        assert_eq!(left[1].unsafe_labelled_count, 1);
        assert_eq!(left, right);
    }

    #[test]
    fn unknown_labels_are_visible_and_bounded_not_silently_dropped() {
        let observations = [
            obs(1, 0.9, Some(false)),
            obs(2, 0.9, None),
            obs(3, 0.7, Some(true)),
        ];
        let curve = risk_coverage_curve(&observations).unwrap();
        let top = &curve[1];
        assert_eq!(top.accepted_count, 2);
        assert_eq!(top.labelled_accepted_count, 1);
        assert_eq!(top.unknown_accepted_count, 1);
        assert_eq!(top.observed_label_risk, Some(0.0));
        assert_eq!(top.optimistic_risk, Some(0.0));
        assert_eq!(top.pessimistic_risk, Some(0.5));
        assert_eq!(top.accepted_label_completeness, Some(0.5));
    }

    #[test]
    fn curve_validation_fails_closed() {
        assert_eq!(risk_coverage_curve(&[]).unwrap_err(), RiskCoverageError::EmptyInput);
        assert_eq!(
            risk_coverage_curve(&[obs(1, f64::NAN, Some(false))]).unwrap_err(),
            RiskCoverageError::NonFiniteAcceptanceScore
        );
        assert_eq!(
            risk_coverage_curve(&[obs(1, 0.2, Some(false)), obs(1, 0.1, Some(true))]).unwrap_err(),
            RiskCoverageError::DuplicateQueryId
        );
    }

    #[test]
    fn fixed_sequence_certifies_prefix_then_stops_at_first_failure() {
        let mut observations = Vec::new();
        for id in 0..20 {
            observations.push(obs(id, 0.9, Some(false)));
        }
        observations.push(obs(20, 0.8, Some(true)));
        observations.push(obs(21, 0.7, Some(false)));

        let budget = RiskBudget::new(0.20, 0.95, 10).unwrap();
        let report = certify_fixed_threshold_sequence(
            &observations,
            &[0.9, 0.8, 0.7],
            budget,
        )
        .unwrap();

        assert_eq!(report.points[0].status, ThresholdCertificationStatus::Certified);
        assert_eq!(report.points[1].status, ThresholdCertificationStatus::RiskBudgetExceeded);
        assert_eq!(report.points[2].status, ThresholdCertificationStatus::NotTestedAfterStop);
        assert_eq!(report.certified_threshold_count, 1);
        assert_eq!(report.stopped_at_threshold, Some(0.8));
    }

    #[test]
    fn unknown_accepted_label_stops_certification_and_does_not_pass() {
        let observations = [
            obs(1, 0.9, Some(false)),
            obs(2, 0.8, None),
            obs(3, 0.7, Some(false)),
        ];
        let report = certify_fixed_threshold_sequence(
            &observations,
            &[0.9, 0.8, 0.7],
            RiskBudget::new(0.5, 0.95, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(report.points[0].status, ThresholdCertificationStatus::Certified);
        assert_eq!(report.points[1].status, ThresholdCertificationStatus::UnknownAcceptedLabels);
        assert_eq!(report.points[2].status, ThresholdCertificationStatus::NotTestedAfterStop);
    }

    #[test]
    fn insufficient_early_threshold_does_not_block_larger_sample_threshold() {
        let observations = [
            obs(1, 0.9, Some(false)),
            obs(2, 0.8, Some(false)),
            obs(3, 0.7, Some(false)),
        ];
        let report = certify_fixed_threshold_sequence(
            &observations,
            &[0.9, 0.8, 0.7],
            RiskBudget::new(0.99, 0.95, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(report.points[0].status, ThresholdCertificationStatus::InsufficientEvidence);
        assert_eq!(report.points[1].status, ThresholdCertificationStatus::Certified);
        assert_eq!(report.points[2].status, ThresholdCertificationStatus::Certified);
    }

    #[test]
    fn bad_threshold_sequences_are_rejected() {
        let observations = [obs(1, 0.9, Some(false))];
        let budget = RiskBudget::new(0.99, 0.95, 1).unwrap();
        assert_eq!(
            certify_fixed_threshold_sequence(&observations, &[], budget).unwrap_err(),
            RiskCoverageError::EmptyThresholdSequence
        );
        assert_eq!(
            certify_fixed_threshold_sequence(&observations, &[0.5, 0.5], budget).unwrap_err(),
            RiskCoverageError::ThresholdsNotStrictlyDescending
        );
        assert_eq!(
            certify_fixed_threshold_sequence(&observations, &[0.5, f64::NAN], budget).unwrap_err(),
            RiskCoverageError::NonFiniteThreshold
        );
    }
}
