//! Deterministic, dependency-free selective-prediction metrics for CRISPR evidence triage.
//!
//! Compile and run without Cargo or network access:
//!   rustc --edition=2024 --test scripts/crispr_selective_risk.rs -o /tmp/crispr-selective-tests
//!   /tmp/crispr-selective-tests
//!
//! This is an offline metric kernel, not a biological model or a safety certification.
//! Calibration is measured only on Positive / AssayDefinedNegative outcomes; unknown
//! outcomes are counted explicitly. Risk bounds assign each unknown accepted case either
//! the correct or incorrect outcome and are not confidence intervals or clinical risk.
//! Exact confidence ties enter or leave the accepted set together.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeLabel {
    Positive,
    AssayDefinedNegative,
    UnknownUnassayed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredCase {
    pub id: String,
    /// A declared group such as guide/Cas9, assay, cell context, or cohort.
    pub context: String,
    /// Probability of a positive outcome; must be finite and within [0, 1].
    pub probability_positive: f64,
    /// Policy confidence used for abstention; higher means the policy is more willing
    /// to act. This may be derived from uncertainty, but its derivation must be frozen.
    pub confidence: f64,
    pub label: OutcomeLabel,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SelectiveMetricError {
    NoCases,
    EmptyIdentity { case_id: String },
    DuplicateCaseId(String),
    InvalidProbability { case_id: String, value: f64 },
    InvalidConfidence { case_id: String, value: f64 },
    ZeroCalibrationBins,
    NoObservedLabels,
}

impl fmt::Display for SelectiveMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCases => write!(f, "at least one scored case is required"),
            Self::EmptyIdentity { case_id } => {
                write!(f, "case {case_id:?} must have a non-empty id and context")
            }
            Self::DuplicateCaseId(id) => write!(f, "duplicate scored-case id: {id}"),
            Self::InvalidProbability { case_id, value } => write!(
                f,
                "case {case_id} has invalid positive probability {value}; expected finite [0, 1]"
            ),
            Self::InvalidConfidence { case_id, value } => write!(
                f,
                "case {case_id} has invalid confidence {value}; expected finite [0, 1]"
            ),
            Self::ZeroCalibrationBins => write!(f, "calibration requires at least one fixed bin"),
            Self::NoObservedLabels => {
                write!(f, "calibration is not estimable without measured positive/negative labels")
            }
        }
    }
}

impl std::error::Error for SelectiveMetricError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationBin {
    pub lower_inclusive: f64,
    pub upper_exclusive: f64,
    pub count: usize,
    pub mean_predicted_probability: Option<f64>,
    pub observed_positive_rate: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationSummary {
    pub total_cases: usize,
    pub observed_label_count: usize,
    pub unknown_label_count: usize,
    pub brier_score: f64,
    /// May be +infinity if a probability of exactly 0 or 1 contradicts an observed label.
    /// Endpoints are not silently clipped to hide overconfident errors.
    pub log_loss: f64,
    /// Fixed-bin, count-weighted absolute calibration gap. Report with Brier/log loss
    /// and reliability bins; ECE alone is not a sufficient calibration certificate.
    pub expected_calibration_error: f64,
    pub bins: Vec<CalibrationBin>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorRateBounds {
    /// All unknown accepted outcomes assigned a correct prediction.
    pub lower: f64,
    /// All unknown accepted outcomes assigned an incorrect prediction.
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiskCoveragePoint {
    /// None is the zero-coverage origin. Otherwise this inclusive threshold accepts all
    /// cases at exactly the same confidence together.
    pub confidence_threshold: Option<f64>,
    pub accepted_count: usize,
    /// Accepted cases divided by every case in the declared context universe.
    pub coverage: f64,
    pub observed_label_accepted: usize,
    pub unknown_accepted: usize,
    /// Observed errors divided by known-label accepted cases; conditional on ascertainment.
    pub observed_label_risk: Option<f64>,
    /// None at zero coverage; otherwise empirical identification bounds, not confidence intervals.
    pub error_rate_bounds: Option<ErrorRateBounds>,
    /// Known-label accepted cases divided by the full universe.
    pub observed_label_coverage: f64,
    /// Known-label accepted cases divided by all accepted cases; None at zero coverage.
    pub accepted_label_completeness: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextSummary {
    pub context: String,
    pub total_cases: usize,
    pub unknown_label_count: usize,
    /// None means no measured labels support calibration in this context.
    pub calibration: Option<CalibrationSummary>,
    pub risk_coverage: Vec<RiskCoveragePoint>,
}

fn validate_cases(cases: &[ScoredCase]) -> Result<(), SelectiveMetricError> {
    if cases.is_empty() {
        return Err(SelectiveMetricError::NoCases);
    }
    let mut ids = HashSet::with_capacity(cases.len());
    for case in cases {
        if case.id.trim().is_empty() || case.context.trim().is_empty() {
            return Err(SelectiveMetricError::EmptyIdentity {
                case_id: case.id.clone(),
            });
        }
        if !ids.insert(case.id.as_str()) {
            return Err(SelectiveMetricError::DuplicateCaseId(case.id.clone()));
        }
        if !case.probability_positive.is_finite()
            || !(0.0..=1.0).contains(&case.probability_positive)
        {
            return Err(SelectiveMetricError::InvalidProbability {
                case_id: case.id.clone(),
                value: case.probability_positive,
            });
        }
        if !case.confidence.is_finite() || !(0.0..=1.0).contains(&case.confidence) {
            return Err(SelectiveMetricError::InvalidConfidence {
                case_id: case.id.clone(),
                value: case.confidence,
            });
        }
    }
    Ok(())
}

fn observed_binary_label(label: OutcomeLabel) -> Option<bool> {
    match label {
        OutcomeLabel::Positive => Some(true),
        OutcomeLabel::AssayDefinedNegative => Some(false),
        OutcomeLabel::UnknownUnassayed => None,
    }
}

/// Compute calibration only over measured labels with declared positive/negative semantics.
/// Unknown outcomes are excluded from the loss denominator, but explicitly counted so the
/// report cannot mistake this conditional score for a full-cohort calibration result.
/// Bins are fixed-width and fixed before evaluation; p = 1 belongs to the final bin.
pub fn calibration_summary(
    cases: &[ScoredCase],
    bin_count: usize,
) -> Result<CalibrationSummary, SelectiveMetricError> {
    validate_cases(cases)?;
    if bin_count == 0 {
        return Err(SelectiveMetricError::ZeroCalibrationBins);
    }

    let mut count = vec![0_usize; bin_count];
    let mut probability_sum = vec![0.0_f64; bin_count];
    let mut positive_sum = vec![0_usize; bin_count];
    let mut brier_sum = 0.0_f64;
    let mut log_loss_sum = 0.0_f64;
    let mut observed_label_count = 0_usize;
    let unknown_label_count = cases
        .iter()
        .filter(|case| case.label == OutcomeLabel::UnknownUnassayed)
        .count();

    for case in cases {
        let Some(actual) = observed_binary_label(case.label) else {
            continue;
        };
        let p = case.probability_positive;
        let y = if actual { 1.0 } else { 0.0 };
        observed_label_count += 1;
        brier_sum += (p - y) * (p - y);
        log_loss_sum += if actual { -p.ln() } else { -(1.0 - p).ln() };

        let index = ((p * bin_count as f64).floor() as usize).min(bin_count - 1);
        count[index] += 1;
        probability_sum[index] += p;
        positive_sum[index] += if actual { 1 } else { 0 };
    }

    if observed_label_count == 0 {
        return Err(SelectiveMetricError::NoObservedLabels);
    }

    let mut bins = Vec::with_capacity(bin_count);
    let mut expected_calibration_error = 0.0_f64;
    for index in 0..bin_count {
        let lower = index as f64 / bin_count as f64;
        let upper = (index + 1) as f64 / bin_count as f64;
        let n = count[index];
        let (mean_predicted_probability, observed_positive_rate) = if n == 0 {
            (None, None)
        } else {
            let mean_p = probability_sum[index] / n as f64;
            let rate = positive_sum[index] as f64 / n as f64;
            expected_calibration_error +=
                (n as f64 / observed_label_count as f64) * (mean_p - rate).abs();
            (Some(mean_p), Some(rate))
        };
        bins.push(CalibrationBin {
            lower_inclusive: lower,
            upper_exclusive: upper,
            count: n,
            mean_predicted_probability,
            observed_positive_rate,
        });
    }

    Ok(CalibrationSummary {
        total_cases: cases.len(),
        observed_label_count,
        unknown_label_count,
        brier_score: brier_sum / observed_label_count as f64,
        log_loss: log_loss_sum / observed_label_count as f64,
        expected_calibration_error,
        bins,
    })
}

/// Build a deterministic risk–coverage curve by accepting highest-confidence cases first.
/// The curve begins with an explicit zero-coverage origin whose undefined rates are None.
/// The predicted class is positive iff p > 0.5 (ties predict negative). Cases with exactly
/// equal confidence enter as one tier, avoiding an arbitrary within-tie threshold.
/// Every non-empty point reports observed-label risk and worst/best empirical assignments
/// of unknown labels; these bounds are not confidence intervals.
pub fn risk_coverage_curve(
    cases: &[ScoredCase],
) -> Result<Vec<RiskCoveragePoint>, SelectiveMetricError> {
    validate_cases(cases)?;
    let mut ranked: Vec<&ScoredCase> = cases.iter().collect();
    ranked.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| a.id.cmp(&b.id))
    });

    let total = ranked.len();
    let mut accepted = 0_usize;
    let mut observed_label_accepted = 0_usize;
    let mut unknown_accepted = 0_usize;
    let mut errors = 0_usize;
    let mut points = vec![RiskCoveragePoint {
        confidence_threshold: None,
        accepted_count: 0,
        coverage: 0.0,
        observed_label_accepted: 0,
        unknown_accepted: 0,
        observed_label_risk: None,
        error_rate_bounds: None,
        observed_label_coverage: 0.0,
        accepted_label_completeness: None,
    }];
    let mut index = 0_usize;

    while index < ranked.len() {
        let threshold = ranked[index].confidence;
        let mut end = index + 1;
        while end < ranked.len() && ranked[end].confidence == threshold {
            end += 1;
        }

        for case in &ranked[index..end] {
            accepted += 1;
            match observed_binary_label(case.label) {
                None => unknown_accepted += 1,
                Some(actual) => {
                    observed_label_accepted += 1;
                    let predicted = case.probability_positive > 0.5;
                    if predicted != actual {
                        errors += 1;
                    }
                }
            }
        }

        let accepted_f = accepted as f64;
        let known_f = observed_label_accepted as f64;
        points.push(RiskCoveragePoint {
            confidence_threshold: Some(threshold),
            accepted_count: accepted,
            coverage: accepted_f / total as f64,
            observed_label_accepted,
            unknown_accepted,
            observed_label_risk: (observed_label_accepted > 0)
                .then(|| errors as f64 / known_f),
            error_rate_bounds: Some(ErrorRateBounds {
                lower: errors as f64 / accepted_f,
                upper: (errors + unknown_accepted) as f64 / accepted_f,
            }),
            observed_label_coverage: known_f / total as f64,
            accepted_label_completeness: Some(known_f / accepted_f),
        });
        index = end;
    }

    Ok(points)
}

/// Produce independently reported metrics per declared context. No pooled score can hide a
/// context's abstention, missing-label rate, calibration failure, or observed error pattern.
pub fn context_stratified_summary(
    cases: &[ScoredCase],
    bin_count: usize,
) -> Result<Vec<ContextSummary>, SelectiveMetricError> {
    validate_cases(cases)?;
    if bin_count == 0 {
        return Err(SelectiveMetricError::ZeroCalibrationBins);
    }

    let mut by_context: BTreeMap<&str, Vec<ScoredCase>> = BTreeMap::new();
    for case in cases {
        by_context
            .entry(case.context.as_str())
            .or_default()
            .push(case.clone());
    }

    by_context
        .into_iter()
        .map(|(context, members)| {
            let unknown_label_count = members
                .iter()
                .filter(|case| case.label == OutcomeLabel::UnknownUnassayed)
                .count();
            let calibration = match calibration_summary(&members, bin_count) {
                Ok(summary) => Some(summary),
                Err(SelectiveMetricError::NoObservedLabels) => None,
                Err(error) => return Err(error),
            };
            let risk_coverage = risk_coverage_curve(&members)?;
            Ok(ContextSummary {
                context: context.to_owned(),
                total_cases: members.len(),
                unknown_label_count,
                calibration,
                risk_coverage,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(
        id: &str,
        context: &str,
        p: f64,
        confidence: f64,
        label: OutcomeLabel,
    ) -> ScoredCase {
        ScoredCase {
            id: id.to_owned(),
            context: context.to_owned(),
            probability_positive: p,
            confidence,
            label,
        }
    }

    #[test]
    fn calibration_scores_only_measured_labels_and_counts_unknowns() {
        let cases = vec![
            case("a", "guide-a", 0.9, 0.9, OutcomeLabel::Positive),
            case("b", "guide-a", 0.2, 0.8, OutcomeLabel::AssayDefinedNegative),
            case("c", "guide-a", 0.8, 0.7, OutcomeLabel::UnknownUnassayed),
        ];
        let summary = calibration_summary(&cases, 5).unwrap();
        assert_eq!(summary.total_cases, 3);
        assert_eq!(summary.observed_label_count, 2);
        assert_eq!(summary.unknown_label_count, 1);
        assert!((summary.brier_score - 0.025).abs() < 1e-12);
        assert!((summary.expected_calibration_error - 0.15).abs() < 1e-12);
    }

    #[test]
    fn probability_one_is_included_in_last_reliability_bin() {
        let cases = vec![case("a", "g", 1.0, 1.0, OutcomeLabel::Positive)];
        let summary = calibration_summary(&cases, 4).unwrap();
        assert_eq!(summary.bins.len(), 4);
        assert_eq!(summary.bins[3].count, 1);
    }

    #[test]
    fn overconfident_wrong_endpoint_is_not_clipped() {
        let cases = vec![case("a", "g", 1.0, 1.0, OutcomeLabel::AssayDefinedNegative)];
        let summary = calibration_summary(&cases, 2).unwrap();
        assert!(summary.log_loss.is_infinite());
    }

    #[test]
    fn calibration_fails_closed_without_observed_labels() {
        let cases = vec![case("a", "g", 0.8, 0.2, OutcomeLabel::UnknownUnassayed)];
        assert_eq!(
            calibration_summary(&cases, 5),
            Err(SelectiveMetricError::NoObservedLabels)
        );
    }

    #[test]
    fn zero_bins_are_rejected() {
        let cases = vec![case("a", "g", 0.8, 0.2, OutcomeLabel::Positive)];
        assert_eq!(
            calibration_summary(&cases, 0),
            Err(SelectiveMetricError::ZeroCalibrationBins)
        );
    }

    #[test]
    fn probability_out_of_range_is_rejected() {
        let cases = vec![case("a", "g", 1.01, 0.2, OutcomeLabel::Positive)];
        assert!(matches!(
            risk_coverage_curve(&cases),
            Err(SelectiveMetricError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn non_finite_confidence_is_rejected() {
        let cases = vec![case("a", "g", 0.8, f64::NAN, OutcomeLabel::Positive)];
        assert!(matches!(
            risk_coverage_curve(&cases),
            Err(SelectiveMetricError::InvalidConfidence { .. })
        ));
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let cases = vec![
            case("same", "g", 0.8, 0.8, OutcomeLabel::Positive),
            case("same", "h", 0.2, 0.7, OutcomeLabel::AssayDefinedNegative),
        ];
        assert_eq!(
            risk_coverage_curve(&cases),
            Err(SelectiveMetricError::DuplicateCaseId("same".to_owned()))
        );
    }

    #[test]
    fn confidence_ties_enter_as_a_single_tier() {
        let cases = vec![
            case("a", "g", 0.9, 0.7, OutcomeLabel::Positive),
            case("b", "g", 0.1, 0.7, OutcomeLabel::AssayDefinedNegative),
            case("c", "g", 0.9, 0.5, OutcomeLabel::Positive),
        ];
        let curve = risk_coverage_curve(&cases).unwrap();
        assert_eq!(curve.len(), 3);
        assert_eq!(curve[0].confidence_threshold, None);
        assert_eq!(curve[1].confidence_threshold, Some(0.7));
        assert_eq!(curve[1].accepted_count, 2);
        assert!((curve[1].coverage - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn unknown_accepted_outcome_is_bounded_not_counted_as_negative() {
        let cases = vec![
            case("correct", "g", 0.9, 0.9, OutcomeLabel::Positive),
            case("unknown", "g", 0.2, 0.8, OutcomeLabel::UnknownUnassayed),
            case("wrong", "g", 0.8, 0.1, OutcomeLabel::AssayDefinedNegative),
        ];
        let curve = risk_coverage_curve(&cases).unwrap();
        let all = curve.last().unwrap();
        assert_eq!(all.accepted_count, 3);
        assert_eq!(all.unknown_accepted, 1);
        let bounds = all.error_rate_bounds.unwrap();
        assert!((bounds.lower - 1.0 / 3.0).abs() < 1e-12);
        assert!((bounds.upper - 2.0 / 3.0).abs() < 1e-12);
        assert!((all.observed_label_risk.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn all_unknown_risk_bounds_span_zero_to_one() {
        let cases = vec![
            case("a", "g", 0.9, 0.9, OutcomeLabel::UnknownUnassayed),
            case("b", "g", 0.1, 0.8, OutcomeLabel::UnknownUnassayed),
        ];
        let curve = risk_coverage_curve(&cases).unwrap();
        let all = curve.last().unwrap();
        assert_eq!(all.observed_label_risk, None);
        assert_eq!(all.error_rate_bounds, Some(ErrorRateBounds { lower: 0.0, upper: 1.0 }));
    }

    #[test]
    fn contexts_are_sorted_and_all_unknown_context_has_no_calibration() {
        let cases = vec![
            case("a", "z-context", 0.9, 0.9, OutcomeLabel::Positive),
            case("b", "a-context", 0.3, 0.7, OutcomeLabel::UnknownUnassayed),
        ];
        let summary = context_stratified_summary(&cases, 4).unwrap();
        assert_eq!(summary[0].context, "a-context");
        assert!(summary[0].calibration.is_none());
        assert_eq!(summary[0].risk_coverage.len(), 2);
        assert_eq!(summary[1].context, "z-context");
        assert!(summary[1].calibration.is_some());
    }

    #[test]
    fn zero_coverage_origin_has_undefined_rates() {
        let cases = vec![case("a", "g", 0.9, 0.9, OutcomeLabel::Positive)];
        let curve = risk_coverage_curve(&cases).unwrap();
        let origin = &curve[0];
        assert_eq!(origin.confidence_threshold, None);
        assert_eq!(origin.accepted_count, 0);
        assert_eq!(origin.coverage, 0.0);
        assert_eq!(origin.observed_label_risk, None);
        assert_eq!(origin.error_rate_bounds, None);
        assert_eq!(origin.accepted_label_completeness, None);
    }

    #[test]
    fn empty_cases_are_rejected() {
        assert_eq!(
            risk_coverage_curve(&[]),
            Err(SelectiveMetricError::NoCases)
        );
    }

    #[test]
    fn prediction_tie_at_half_is_negative() {
        let cases = vec![case("a", "g", 0.5, 1.0, OutcomeLabel::Positive)];
        let curve = risk_coverage_curve(&cases).unwrap();
        assert_eq!(curve[1].observed_label_risk, Some(1.0));
    }
}
