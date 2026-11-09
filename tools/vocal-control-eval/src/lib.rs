// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dependency-free evaluator for controlled vocal interventions.
//!
//! A control is qualified only when its declared target metric changes in the
//! requested direction by a pre-registered minimum amount, while unrelated
//! guard metrics remain within their allowed ranges. This is an evaluator for
//! measured feature vectors; it does not extract acoustic features from audio.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Increase,
    Decrease,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricRange {
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollateralGuard {
    pub metric: String,
    /// Maximum absolute change as a fraction of that metric's declared range.
    pub maximum_range_fraction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterventionSpec {
    pub id: String,
    pub target_metric: String,
    pub direction: Direction,
    /// Minimum target movement as a fraction of the target metric's range.
    pub minimum_effect_range_fraction: f64,
    pub collateral_guards: Vec<CollateralGuard>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricDelta {
    pub metric: String,
    /// Signed (treatment - baseline) / (declared maximum - minimum).
    pub range_fraction: f64,
    pub absolute_range_fraction: f64,
    pub allowed_absolute_range_fraction: Option<f64>,
    pub within_guard: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterventionScore {
    pub id: String,
    pub target: MetricDelta,
    pub direction_correct: bool,
    pub target_effect_sufficient: bool,
    pub collateral: Vec<MetricDelta>,
    pub collateral_guards_pass: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchSummary {
    pub case_count: usize,
    pub passed_cases: usize,
    pub pass_fraction: f64,
    pub mean_absolute_target_effect_range_fraction: f64,
    pub maximum_collateral_range_fraction: f64,
}

pub type FeatureVector = BTreeMap<String, f64>;
pub type RangeMap = BTreeMap<String, MetricRange>;

/// Score one baseline/treatment pair. All thresholds and metric ranges must be
/// fixed before looking at treatment results; this function validates them.
pub fn score_intervention(
    spec: &InterventionSpec,
    ranges: &RangeMap,
    baseline: &FeatureVector,
    treatment: &FeatureVector,
) -> Result<InterventionScore, String> {
    validate_spec(spec)?;
    validate_range_map(ranges)?;
    let target_range = ranges
        .get(&spec.target_metric)
        .ok_or_else(|| format!("missing range for target metric {:?}", spec.target_metric))?;
    let target = delta(&spec.target_metric, target_range, baseline, treatment, None)?;

    let direction_correct = match spec.direction {
        Direction::Increase => target.range_fraction > 0.0,
        Direction::Decrease => target.range_fraction < 0.0,
    };
    let target_effect_sufficient =
        target.absolute_range_fraction >= spec.minimum_effect_range_fraction;

    let mut collateral = Vec::with_capacity(spec.collateral_guards.len());
    let mut collateral_guards_pass = true;
    for guard in &spec.collateral_guards {
        let range = ranges
            .get(&guard.metric)
            .ok_or_else(|| format!("missing range for collateral metric {:?}", guard.metric))?;
        let mut change = delta(
            &guard.metric,
            range,
            baseline,
            treatment,
            Some(guard.maximum_range_fraction),
        )?;
        let within_guard = change.absolute_range_fraction <= guard.maximum_range_fraction;
        change.within_guard = Some(within_guard);
        collateral_guards_pass &= within_guard;
        collateral.push(change);
    }

    let passed = direction_correct && target_effect_sufficient && collateral_guards_pass;
    Ok(InterventionScore {
        id: spec.id.clone(),
        target,
        direction_correct,
        target_effect_sufficient,
        collateral,
        collateral_guards_pass,
        passed,
    })
}

/// Summarize a pre-registered batch. It reports both efficacy and spillover;
/// it does not convert either into a hidden weighted quality score.
pub fn summarize_batch(scores: &[InterventionScore]) -> Result<BatchSummary, String> {
    if scores.is_empty() {
        return Err("cannot summarize an empty intervention batch".into());
    }
    let passed_cases = scores.iter().filter(|score| score.passed).count();
    let mean_absolute_target_effect_range_fraction = scores
        .iter()
        .map(|score| score.target.absolute_range_fraction)
        .sum::<f64>()
        / scores.len() as f64;
    let maximum_collateral_range_fraction = scores
        .iter()
        .flat_map(|score| score.collateral.iter())
        .map(|metric| metric.absolute_range_fraction)
        .fold(0.0, f64::max);
    Ok(BatchSummary {
        case_count: scores.len(),
        passed_cases,
        pass_fraction: passed_cases as f64 / scores.len() as f64,
        mean_absolute_target_effect_range_fraction,
        maximum_collateral_range_fraction,
    })
}

fn validate_spec(spec: &InterventionSpec) -> Result<(), String> {
    if spec.id.trim().is_empty() || spec.target_metric.trim().is_empty() {
        return Err("intervention id and target metric must be non-empty".into());
    }
    if !spec.minimum_effect_range_fraction.is_finite()
        || !(0.0..=1.0).contains(&spec.minimum_effect_range_fraction)
    {
        return Err("minimum target effect must be finite and within [0, 1]".into());
    }
    let mut seen = BTreeSet::new();
    seen.insert(spec.target_metric.as_str());
    for guard in &spec.collateral_guards {
        if guard.metric.trim().is_empty() {
            return Err("collateral metric must be non-empty".into());
        }
        if !guard.maximum_range_fraction.is_finite()
            || !(0.0..=1.0).contains(&guard.maximum_range_fraction)
        {
            return Err(format!(
                "collateral threshold for {:?} must be finite and within [0, 1]",
                guard.metric
            ));
        }
        if !seen.insert(guard.metric.as_str()) {
            return Err(format!(
                "duplicate or target-overlapping metric guard {:?}",
                guard.metric
            ));
        }
    }
    Ok(())
}

fn validate_range_map(ranges: &RangeMap) -> Result<(), String> {
    for (metric, range) in ranges {
        if metric.trim().is_empty()
            || !range.minimum.is_finite()
            || !range.maximum.is_finite()
            || range.minimum >= range.maximum
        {
            return Err(format!("invalid declared range for metric {metric:?}"));
        }
    }
    Ok(())
}

fn delta(
    metric: &str,
    range: &MetricRange,
    baseline: &FeatureVector,
    treatment: &FeatureVector,
    allowed: Option<f64>,
) -> Result<MetricDelta, String> {
    let baseline_value = baseline
        .get(metric)
        .copied()
        .ok_or_else(|| format!("baseline is missing metric {metric:?}"))?;
    let treatment_value = treatment
        .get(metric)
        .copied()
        .ok_or_else(|| format!("treatment is missing metric {metric:?}"))?;
    if !baseline_value.is_finite() || !treatment_value.is_finite() {
        return Err(format!(
            "metric {metric:?} contains a non-finite observation"
        ));
    }
    if !(range.minimum..=range.maximum).contains(&baseline_value)
        || !(range.minimum..=range.maximum).contains(&treatment_value)
    {
        return Err(format!(
            "metric {metric:?} observation lies outside its frozen range"
        ));
    }
    let range_fraction = (treatment_value - baseline_value) / (range.maximum - range.minimum);
    Ok(MetricDelta {
        metric: metric.into(),
        range_fraction,
        absolute_range_fraction: range_fraction.abs(),
        allowed_absolute_range_fraction: allowed,
        within_guard: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges() -> RangeMap {
        BTreeMap::from([
            (
                "breathiness".into(),
                MetricRange {
                    minimum: 0.0,
                    maximum: 1.0,
                },
            ),
            (
                "vibrato_rate_hz".into(),
                MetricRange {
                    minimum: 3.0,
                    maximum: 8.0,
                },
            ),
            (
                "pitch_median_cents".into(),
                MetricRange {
                    minimum: -100.0,
                    maximum: 100.0,
                },
            ),
            (
                "loudness_lufs".into(),
                MetricRange {
                    minimum: -30.0,
                    maximum: 0.0,
                },
            ),
        ])
    }

    fn breathiness_spec() -> InterventionSpec {
        InterventionSpec {
            id: "breathiness-up".into(),
            target_metric: "breathiness".into(),
            direction: Direction::Increase,
            minimum_effect_range_fraction: 0.15,
            collateral_guards: vec![
                CollateralGuard {
                    metric: "vibrato_rate_hz".into(),
                    maximum_range_fraction: 0.05,
                },
                CollateralGuard {
                    metric: "pitch_median_cents".into(),
                    maximum_range_fraction: 0.05,
                },
                CollateralGuard {
                    metric: "loudness_lufs".into(),
                    maximum_range_fraction: 0.08,
                },
            ],
        }
    }

    fn baseline() -> FeatureVector {
        BTreeMap::from([
            ("breathiness".into(), 0.2),
            ("vibrato_rate_hz".into(), 5.2),
            ("pitch_median_cents".into(), 4.0),
            ("loudness_lufs".into(), -18.0),
        ])
    }

    #[test]
    fn intended_control_change_passes_when_spillover_is_bounded() {
        let treatment = BTreeMap::from([
            ("breathiness".into(), 0.5),
            ("vibrato_rate_hz".into(), 5.25),
            ("pitch_median_cents".into(), 5.0),
            ("loudness_lufs".into(), -17.0),
        ]);
        let result =
            score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment).unwrap();
        assert!(result.direction_correct);
        assert!(result.target_effect_sufficient);
        assert!(result.collateral_guards_pass);
        assert!(result.passed);
    }

    #[test]
    fn wrong_direction_fails_even_when_effect_is_large() {
        let mut treatment = baseline();
        treatment.insert("breathiness".into(), 0.8);
        let mut spec = breathiness_spec();
        spec.direction = Direction::Decrease;
        let result = score_intervention(&spec, &ranges(), &baseline(), &treatment).unwrap();
        assert!(!result.direction_correct);
        assert!(!result.passed);
    }

    #[test]
    fn insufficient_target_change_fails() {
        let mut treatment = baseline();
        treatment.insert("breathiness".into(), 0.25);
        let result =
            score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment).unwrap();
        assert!(!result.target_effect_sufficient);
        assert!(!result.passed);
    }

    #[test]
    fn collateral_drift_fails_even_when_target_is_correct() {
        let mut treatment = baseline();
        treatment.insert("breathiness".into(), 0.6);
        treatment.insert("pitch_median_cents".into(), 40.0);
        let result =
            score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment).unwrap();
        assert!(result.direction_correct);
        assert!(result.target_effect_sufficient);
        assert!(!result.collateral_guards_pass);
        assert!(!result.passed);
    }

    #[test]
    fn missing_metric_fails_closed() {
        let mut treatment = baseline();
        treatment.remove("vibrato_rate_hz");
        let error = score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment)
            .unwrap_err();
        assert!(error.contains("missing metric"));
    }

    #[test]
    fn non_finite_and_out_of_range_observations_are_rejected() {
        let mut treatment = baseline();
        treatment.insert("breathiness".into(), f64::NAN);
        assert!(
            score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment).is_err()
        );

        treatment.insert("breathiness".into(), 1.1);
        assert!(score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment).is_err());
    }

    #[test]
    fn malformed_specs_and_overlapping_guardrails_are_rejected() {
        let mut spec = breathiness_spec();
        spec.collateral_guards.push(CollateralGuard {
            metric: "breathiness".into(),
            maximum_range_fraction: 0.1,
        });
        assert!(score_intervention(&spec, &ranges(), &baseline(), &baseline()).is_err());

        spec = breathiness_spec();
        spec.minimum_effect_range_fraction = f64::INFINITY;
        assert!(score_intervention(&spec, &ranges(), &baseline(), &baseline()).is_err());
    }

    #[test]
    fn batch_summary_reports_control_success_and_worst_spillover_separately() {
        let mut treatment = baseline();
        treatment.insert("breathiness".into(), 0.5);
        let pass =
            score_intervention(&breathiness_spec(), &ranges(), &baseline(), &treatment).unwrap();

        let mut bad_treatment = treatment.clone();
        bad_treatment.insert("pitch_median_cents".into(), 50.0);
        let fail = score_intervention(&breathiness_spec(), &ranges(), &baseline(), &bad_treatment)
            .unwrap();

        let summary = summarize_batch(&[pass, fail]).unwrap();
        assert_eq!(summary.case_count, 2);
        assert_eq!(summary.passed_cases, 1);
        assert_eq!(summary.pass_fraction, 0.5);
        assert!(summary.maximum_collateral_range_fraction >= 0.23);
    }

    #[test]
    fn empty_batches_are_not_vacuous_successes() {
        assert!(summarize_batch(&[]).is_err());
    }
}
