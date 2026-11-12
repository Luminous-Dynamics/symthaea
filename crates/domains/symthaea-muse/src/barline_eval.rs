// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact, evidence-gated evaluation of notated measure structure.
//!
//! This module deliberately does not infer barlines from a time signature,
//! note density, beat grid, or the legacy `Score::meter` field. Callers must
//! provide explicit measure spans on both sides. Coordinates are exact
//! quarter-note beats; MIDI ticks, audio seconds, and renderer frames must be
//! converted by a provenance-aware adapter before reaching this API.
//!
//! Missing or unsupported measure structure yields `Unavailable`, never a
//! fabricated zero or perfect score.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const BARLINE_EVALUATOR_VERSION: &str = "muse-mir-009-barline-evaluator-v1";

/// A non-negative, reduced rational coordinate in quarter-note beats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RationalBeat {
    num: i64,
    den: i64,
}

impl RationalBeat {
    pub fn new(num: i64, den: i64) -> Result<Self, EvaluationError> {
        if den <= 0 {
            return Err(EvaluationError::NonPositiveDenominator { den });
        }
        if num < 0 {
            return Err(EvaluationError::NegativeCoordinate { num });
        }
        let divisor = gcd(num, den);
        Ok(Self {
            num: num / divisor,
            den: den / divisor,
        })
    }

    pub const fn numerator(self) -> i64 {
        self.num
    }

    pub const fn denominator(self) -> i64 {
        self.den
    }

    pub fn as_beats(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    fn cmp_exact(self, other: Self) -> Ordering {
        (i128::from(self.num) * i128::from(other.den))
            .cmp(&(i128::from(other.num) * i128::from(self.den)))
    }

    fn within(self, other: Self, tolerance: Self) -> Result<bool, EvaluationError> {
        let left = i128::from(self.num)
            .checked_mul(i128::from(other.den))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        let right = i128::from(other.num)
            .checked_mul(i128::from(self.den))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        let difference = (left - right).abs();
        let lhs = difference
            .checked_mul(i128::from(tolerance.den))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        let rhs = i128::from(tolerance.num)
            .checked_mul(i128::from(self.den))
            .and_then(|value| value.checked_mul(i128::from(other.den)))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        Ok(lhs <= rhs)
    }

    fn absolute_difference(self, other: Self) -> Result<Self, EvaluationError> {
        let left = i128::from(self.num)
            .checked_mul(i128::from(other.den))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        let right = i128::from(other.num)
            .checked_mul(i128::from(self.den))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        let numerator = (left - right).abs();
        let denominator = i128::from(self.den)
            .checked_mul(i128::from(other.den))
            .ok_or(EvaluationError::ArithmeticOverflow)?;
        let divisor = gcd_i128(numerator, denominator);
        let reduced_num = numerator / divisor;
        let reduced_den = denominator / divisor;
        let num = i64::try_from(reduced_num).map_err(|_| EvaluationError::ArithmeticOverflow)?;
        let den = i64::try_from(reduced_den).map_err(|_| EvaluationError::ArithmeticOverflow)?;
        Self::new(num, den)
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

/// How much structural evidence the source actually provides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasureMapStatus {
    /// Explicit, provenance-bound measure spans are available.
    Supported,
    /// The source format or adapter cannot represent the required structure.
    Unsupported { reason: String },
    /// The source may contain structure, but it has not been established.
    Unknown { reason: String },
}

/// A single notated measure. Start is inclusive; end is exclusive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasureSpan {
    pub start: RationalBeat,
    pub end: RationalBeat,
    pub kind: MeasureKind,
}

/// Measure class is evaluated separately from boundary location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasureKind {
    Regular,
    Pickup,
    Incomplete,
    Irregular,
}

/// Explicit measure sequence supplied by a parser or annotation adapter.
///
/// `source_id` should identify the source/annotation artifact, preferably by
/// immutable digest. This API does not treat the identifier as authentication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasureMap {
    pub source_id: String,
    pub status: MeasureMapStatus,
    pub measures: Vec<MeasureSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationError {
    EmptySourceId,
    NoMeasures,
    NonPositiveDenominator { den: i64 },
    NegativeCoordinate { num: i64 },
    NonReducedRational,
    EmptyOrReversedMeasure { index: usize },
    NonContiguousMeasures { left_index: usize },
    ArithmeticOverflow,
    InvalidTolerance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchCounts {
    pub matched: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionRecallF1 {
    /// None means there were no predicted positives; it is not silently 0 or 1.
    pub precision: Option<f64>,
    /// None means there were no gold positives; it is not silently 0 or 1.
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarlineEvaluationReport {
    pub evaluator_version: String,
    pub gold_source_id: String,
    pub prediction_source_id: String,
    pub tolerance: RationalBeat,
    pub gold_internal_barline_count: usize,
    pub predicted_internal_barline_count: usize,
    pub barline_counts: MatchCounts,
    pub barline_metrics: PrecisionRecallF1,
    /// Exact absolute displacement for each one-to-one matched boundary.
    pub matched_boundary_displacements: Vec<RationalBeat>,
    pub gold_measure_count: usize,
    pub predicted_measure_count: usize,
    pub measure_counts: MatchCounts,
    pub measure_metrics: PrecisionRecallF1,
    pub matched_measure_class_agreements: usize,
    pub matched_measure_class_disagreements: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnavailableReason {
    GoldStructureUnknown { reason: String },
    GoldStructureUnsupported { reason: String },
    PredictionStructureUnknown { reason: String },
    PredictionStructureUnsupported { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationOutcome {
    Evaluated(BarlineEvaluationReport),
    Unavailable(UnavailableReason),
}

/// Evaluate barlines and measure spans without inferring missing structure.
///
/// Boundary matching is one-to-one and order-preserving. Its tolerance is in
/// quarter-note beats, not seconds, MIDI ticks, or denominator-note units.
pub fn evaluate_measure_maps(
    gold: &MeasureMap,
    prediction: &MeasureMap,
    tolerance: RationalBeat,
) -> Result<EvaluationOutcome, EvaluationError> {
    validate_rational(tolerance)?;

    let unavailable = match (&gold.status, &prediction.status) {
        (MeasureMapStatus::Unknown { reason }, _) => {
            Some(UnavailableReason::GoldStructureUnknown {
                reason: reason.clone(),
            })
        }
        (MeasureMapStatus::Unsupported { reason }, _) => {
            Some(UnavailableReason::GoldStructureUnsupported {
                reason: reason.clone(),
            })
        }
        (_, MeasureMapStatus::Unknown { reason }) => {
            Some(UnavailableReason::PredictionStructureUnknown {
                reason: reason.clone(),
            })
        }
        (_, MeasureMapStatus::Unsupported { reason }) => {
            Some(UnavailableReason::PredictionStructureUnsupported {
                reason: reason.clone(),
            })
        }
        (MeasureMapStatus::Supported, MeasureMapStatus::Supported) => None,
    };
    if let Some(reason) = unavailable {
        return Ok(EvaluationOutcome::Unavailable(reason));
    }

    validate_map(gold)?;
    validate_map(prediction)?;

    let gold_boundaries = internal_boundaries(gold);
    let predicted_boundaries = internal_boundaries(prediction);
    let (barline_counts, matched_boundary_displacements) =
        match_boundaries(&gold_boundaries, &predicted_boundaries, tolerance)?;
    let barline_metrics = metrics(&barline_counts);

    let (measure_counts, class_agreements, class_disagreements) =
        match_measures(&gold.measures, &prediction.measures, tolerance)?;
    let measure_metrics = metrics(&measure_counts);

    Ok(EvaluationOutcome::Evaluated(BarlineEvaluationReport {
        evaluator_version: BARLINE_EVALUATOR_VERSION.to_string(),
        gold_source_id: gold.source_id.clone(),
        prediction_source_id: prediction.source_id.clone(),
        tolerance,
        gold_internal_barline_count: gold_boundaries.len(),
        predicted_internal_barline_count: predicted_boundaries.len(),
        barline_counts,
        barline_metrics,
        matched_boundary_displacements,
        gold_measure_count: gold.measures.len(),
        predicted_measure_count: prediction.measures.len(),
        measure_counts,
        measure_metrics,
        matched_measure_class_agreements: class_agreements,
        matched_measure_class_disagreements: class_disagreements,
    }))
}

fn validate_map(map: &MeasureMap) -> Result<(), EvaluationError> {
    if map.source_id.trim().is_empty() {
        return Err(EvaluationError::EmptySourceId);
    }
    if map.measures.is_empty() {
        return Err(EvaluationError::NoMeasures);
    }
    for (index, measure) in map.measures.iter().enumerate() {
        validate_rational(measure.start)?;
        validate_rational(measure.end)?;
        if measure.start.cmp_exact(measure.end) != Ordering::Less {
            return Err(EvaluationError::EmptyOrReversedMeasure { index });
        }
        if index > 0 {
            let previous = &map.measures[index - 1];
            if previous.end != measure.start {
                return Err(EvaluationError::NonContiguousMeasures {
                    left_index: index - 1,
                });
            }
        }
    }
    Ok(())
}

fn validate_rational(value: RationalBeat) -> Result<(), EvaluationError> {
    if value.den <= 0 {
        return Err(EvaluationError::NonPositiveDenominator { den: value.den });
    }
    if value.num < 0 {
        return Err(EvaluationError::NegativeCoordinate { num: value.num });
    }
    if gcd(value.num, value.den) != 1 {
        return Err(EvaluationError::NonReducedRational);
    }
    Ok(())
}

fn internal_boundaries(map: &MeasureMap) -> Vec<RationalBeat> {
    map.measures
        .iter()
        .skip(1)
        .map(|measure| measure.start)
        .collect()
}

fn match_boundaries(
    gold: &[RationalBeat],
    prediction: &[RationalBeat],
    tolerance: RationalBeat,
) -> Result<(MatchCounts, Vec<RationalBeat>), EvaluationError> {
    let mut gold_index = 0;
    let mut prediction_index = 0;
    let mut counts = MatchCounts {
        matched: 0,
        false_positives: 0,
        false_negatives: 0,
    };
    let mut displacements = Vec::new();

    while gold_index < gold.len() && prediction_index < prediction.len() {
        let expected = gold[gold_index];
        let observed = prediction[prediction_index];
        if observed.within(expected, tolerance)? {
            counts.matched += 1;
            displacements.push(observed.absolute_difference(expected)?);
            gold_index += 1;
            prediction_index += 1;
        } else if observed.cmp_exact(expected) == Ordering::Less {
            counts.false_positives += 1;
            prediction_index += 1;
        } else {
            counts.false_negatives += 1;
            gold_index += 1;
        }
    }

    counts.false_negatives += gold.len() - gold_index;
    counts.false_positives += prediction.len() - prediction_index;
    displacements.sort_by(|left, right| left.cmp_exact(*right));
    Ok((counts, displacements))
}

fn match_measures(
    gold: &[MeasureSpan],
    prediction: &[MeasureSpan],
    tolerance: RationalBeat,
) -> Result<(MatchCounts, usize, usize), EvaluationError> {
    let mut gold_index = 0;
    let mut prediction_index = 0;
    let mut counts = MatchCounts {
        matched: 0,
        false_positives: 0,
        false_negatives: 0,
    };
    let mut class_agreements = 0;
    let mut class_disagreements = 0;

    while gold_index < gold.len() && prediction_index < prediction.len() {
        let expected = &gold[gold_index];
        let observed = &prediction[prediction_index];
        let starts_match = observed.start.within(expected.start, tolerance)?;
        let ends_match = observed.end.within(expected.end, tolerance)?;
        if starts_match && ends_match {
            counts.matched += 1;
            if observed.kind == expected.kind {
                class_agreements += 1;
            } else {
                class_disagreements += 1;
            }
            gold_index += 1;
            prediction_index += 1;
        } else if observed.start.cmp_exact(expected.start) == Ordering::Less {
            counts.false_positives += 1;
            prediction_index += 1;
        } else {
            counts.false_negatives += 1;
            gold_index += 1;
        }
    }

    counts.false_negatives += gold.len() - gold_index;
    counts.false_positives += prediction.len() - prediction_index;
    Ok((counts, class_agreements, class_disagreements))
}

fn metrics(counts: &MatchCounts) -> PrecisionRecallF1 {
    let precision = ratio(counts.matched, counts.matched + counts.false_positives);
    let recall = ratio(counts.matched, counts.matched + counts.false_negatives);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };
    PrecisionRecallF1 {
        precision,
        recall,
        f1,
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn beat(num: i64, den: i64) -> RationalBeat {
        RationalBeat::new(num, den).unwrap()
    }

    fn span(start: (i64, i64), end: (i64, i64), kind: MeasureKind) -> MeasureSpan {
        MeasureSpan {
            start: beat(start.0, start.1),
            end: beat(end.0, end.1),
            kind,
        }
    }

    fn map(source_id: &str, measures: Vec<MeasureSpan>) -> MeasureMap {
        MeasureMap {
            source_id: source_id.into(),
            status: MeasureMapStatus::Supported,
            measures,
        }
    }

    fn report(outcome: EvaluationOutcome) -> BarlineEvaluationReport {
        match outcome {
            EvaluationOutcome::Evaluated(report) => report,
            other => panic!("expected evaluated report, got {other:?}"),
        }
    }

    #[test]
    fn exact_regular_four_four_bar_lines_match() {
        let gold = map(
            "gold-4-4",
            vec![
                span((0, 1), (4, 1), MeasureKind::Regular),
                span((4, 1), (8, 1), MeasureKind::Regular),
                span((8, 1), (12, 1), MeasureKind::Regular),
            ],
        );
        let prediction = gold.clone();
        let result = report(
            evaluate_measure_maps(&gold, &prediction, beat(1, 8)).unwrap(),
        );
        assert_eq!(result.barline_counts.matched, 2);
        assert_eq!(result.barline_counts.false_positives, 0);
        assert_eq!(result.barline_counts.false_negatives, 0);
        assert_eq!(result.barline_metrics.f1, Some(1.0));
        assert_eq!(result.measure_metrics.f1, Some(1.0));
    }

    #[test]
    fn pickup_is_a_real_partial_measure_not_a_shifted_full_bar() {
        let gold = map(
            "gold-pickup",
            vec![
                span((0, 1), (1, 1), MeasureKind::Pickup),
                span((1, 1), (5, 1), MeasureKind::Regular),
                span((5, 1), (9, 1), MeasureKind::Regular),
            ],
        );
        let prediction = gold.clone();
        let result = report(
            evaluate_measure_maps(&gold, &prediction, beat(1, 8)).unwrap(),
        );
        assert_eq!(result.barline_counts.matched, 2);
        assert_eq!(result.measure_counts.matched, 3);
        assert_eq!(result.matched_measure_class_agreements, 3);
    }

    #[test]
    fn additive_meter_and_mixed_bar_lengths_use_exact_quarter_beats() {
        let gold = map(
            "gold-mixed",
            vec![
                span((0, 1), (7, 2), MeasureKind::Regular), // 7/8
                span((7, 2), (6, 1), MeasureKind::Irregular), // 5/8
                span((6, 1), (10, 1), MeasureKind::Regular), // 4/4
            ],
        );
        let prediction = map(
            "pred-ppq-independent",
            vec![
                span((0, 2), (14, 4), MeasureKind::Regular),
                span((14, 4), (12, 2), MeasureKind::Irregular),
                span((12, 2), (20, 2), MeasureKind::Regular),
            ],
        );
        let result = report(
            evaluate_measure_maps(&gold, &prediction, beat(1, 16)).unwrap(),
        );
        assert_eq!(result.barline_counts.matched, 2);
        assert_eq!(result.barline_metrics.f1, Some(1.0));
    }

    #[test]
    fn tolerance_edges_are_inclusive_and_exact() {
        let gold = map(
            "gold",
            vec![
                span((0, 1), (4, 1), MeasureKind::Regular),
                span((4, 1), (8, 1), MeasureKind::Regular),
            ],
        );
        let prediction = map(
            "prediction",
            vec![
                span((0, 1), (33, 8), MeasureKind::Regular),
                span((33, 8), (8, 1), MeasureKind::Regular),
            ],
        );
        let result = report(
            evaluate_measure_maps(&gold, &prediction, beat(1, 8)).unwrap(),
        );
        assert_eq!(result.barline_counts.matched, 1);
        assert_eq!(result.matched_boundary_displacements, vec![beat(1, 8)]);
    }

    #[test]
    fn missing_structure_is_unavailable_not_a_zero_score() {
        let gold = map(
            "gold",
            vec![span((0, 1), (4, 1), MeasureKind::Regular)],
        );
        let prediction = MeasureMap {
            source_id: "midi-without-barline-sidecar".into(),
            status: MeasureMapStatus::Unsupported {
                reason: "MIDI time-signature events do not encode the annotated bar sequence".into(),
            },
            measures: Vec::new(),
        };
        let outcome = evaluate_measure_maps(&gold, &prediction, beat(1, 8)).unwrap();
        assert!(matches!(
            outcome,
            EvaluationOutcome::Unavailable(
                UnavailableReason::PredictionStructureUnsupported { .. }
            )
        ));
    }

    #[test]
    fn non_contiguous_measure_spans_are_rejected() {
        let malformed = map(
            "broken",
            vec![
                span((0, 1), (4, 1), MeasureKind::Regular),
                span((5, 1), (9, 1), MeasureKind::Regular),
            ],
        );
        assert_eq!(
            evaluate_measure_maps(&malformed, &malformed, beat(1, 8)),
            Err(EvaluationError::NonContiguousMeasures { left_index: 0 })
        );
    }

    #[test]
    fn a_single_measure_has_no_internal_barline_metric() {
        let one_measure = map(
            "one-measure",
            vec![span((0, 1), (4, 1), MeasureKind::Regular)],
        );
        let result = report(
            evaluate_measure_maps(&one_measure, &one_measure, beat(1, 8)).unwrap(),
        );
        assert_eq!(result.gold_internal_barline_count, 0);
        assert_eq!(result.barline_metrics.precision, None);
        assert_eq!(result.barline_metrics.recall, None);
        assert_eq!(result.barline_metrics.f1, None);
        assert_eq!(result.measure_metrics.f1, Some(1.0));
    }

    #[test]
    fn rational_coordinates_are_reduced() {
        assert_eq!(beat(2, 4), beat(1, 2));
    }
}
