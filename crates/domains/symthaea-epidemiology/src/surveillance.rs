// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative screening of aggregate surveillance time series.
//!
//! This module deliberately produces **statistical change candidates**, not
//! diagnoses, outbreak declarations, forecasts, or response instructions. It is
//! intended to sit downstream of privacy-preserving aggregate evidence and
//! upstream of richer corroboration/hypothesis reasoning.
//!
//! The v1 screen uses a robust historical baseline (median + median absolute
//! deviation) and treats source-supplied uncertainty as a reason to abstain when
//! a nominal point-estimate excursion is not clearly separated from the baseline
//! envelope.

use std::{error::Error, fmt};

use symthaea_statistics::median;

/// Conventional consistency factor that makes MAD comparable to standard
/// deviation for a normal distribution. This does not make the baseline normal
/// or turn the resulting score into a probability.
const MAD_NORMAL_SCALE: f64 = 1.4826;

/// One aggregate estimate with source-supplied uncertainty bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalEstimate {
    estimate: f64,
    lower: f64,
    upper: f64,
}

impl IntervalEstimate {
    pub fn new(estimate: f64, lower: f64, upper: f64) -> Result<Self, SurveillanceScreenError> {
        if !estimate.is_finite() || !lower.is_finite() || !upper.is_finite() {
            return Err(SurveillanceScreenError::NonFiniteMeasurement);
        }
        if lower > upper || estimate < lower || estimate > upper {
            return Err(SurveillanceScreenError::InvalidUncertaintyInterval);
        }
        Ok(Self {
            estimate,
            lower,
            upper,
        })
    }

    pub fn estimate(self) -> f64 {
        self.estimate
    }

    pub fn lower(self) -> f64 {
        self.lower
    }

    pub fn upper(self) -> f64 {
        self.upper
    }

    pub fn contains(self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// A timestamped aggregate surveillance point.
///
/// Missingness is explicit rather than represented by NaN or a sentinel value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurveillancePoint {
    observed_at_unix_s: i64,
    measurement: Option<IntervalEstimate>,
}

impl SurveillancePoint {
    pub fn observed(
        observed_at_unix_s: i64,
        estimate: f64,
        lower: f64,
        upper: f64,
    ) -> Result<Self, SurveillanceScreenError> {
        Ok(Self {
            observed_at_unix_s,
            measurement: Some(IntervalEstimate::new(estimate, lower, upper)?),
        })
    }

    pub const fn missing(observed_at_unix_s: i64) -> Self {
        Self {
            observed_at_unix_s,
            measurement: None,
        }
    }

    pub const fn observed_at_unix_s(self) -> i64 {
        self.observed_at_unix_s
    }

    pub const fn measurement(self) -> Option<IntervalEstimate> {
        self.measurement
    }
}

/// Configuration for a statistical screen.
///
/// Thresholds are screening parameters, not clinical/public-health policy. A
/// deployment should preregister and validate them for its data source instead
/// of interpreting the defaults as universal epidemiological cutoffs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurveillanceScreenConfig {
    min_baseline_observations: usize,
    robust_z_threshold: f64,
    scale_epsilon: f64,
}

impl SurveillanceScreenConfig {
    pub fn new(
        min_baseline_observations: usize,
        robust_z_threshold: f64,
    ) -> Result<Self, SurveillanceScreenError> {
        Self::with_scale_epsilon(min_baseline_observations, robust_z_threshold, 1e-12)
    }

    pub fn with_scale_epsilon(
        min_baseline_observations: usize,
        robust_z_threshold: f64,
        scale_epsilon: f64,
    ) -> Result<Self, SurveillanceScreenError> {
        if min_baseline_observations < 3
            || !robust_z_threshold.is_finite()
            || robust_z_threshold <= 0.0
            || !scale_epsilon.is_finite()
            || scale_epsilon <= 0.0
        {
            return Err(SurveillanceScreenError::InvalidConfiguration);
        }
        Ok(Self {
            min_baseline_observations,
            robust_z_threshold,
            scale_epsilon,
        })
    }

    pub const fn min_baseline_observations(self) -> usize {
        self.min_baseline_observations
    }

    pub const fn robust_z_threshold(self) -> f64 {
        self.robust_z_threshold
    }
}

impl Default for SurveillanceScreenConfig {
    fn default() -> Self {
        // A convenience screening profile only; not a public-health threshold.
        Self {
            min_baseline_observations: 5,
            robust_z_threshold: 3.5,
            scale_epsilon: 1e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustBaseline {
    pub center: f64,
    pub mad: f64,
    pub robust_scale: f64,
    pub lower_threshold_envelope: f64,
    pub upper_threshold_envelope: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChangeDirection {
    Upward,
    Downward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstentionReason {
    /// The current/latest aggregate itself is unavailable.
    LatestMeasurementMissing,
    /// Historical point estimates have effectively zero robust spread, so a
    /// finite standardized deviation cannot be supported by this method.
    DegenerateBaselineSpread,
    /// The point estimate crosses the configured screen, but its source-supplied
    /// uncertainty interval still overlaps the baseline threshold envelope.
    UncertaintyOverlapsThresholdEnvelope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreeningDisposition {
    /// Too few observed historical values for the configured baseline.
    InsufficientBaseline,
    /// No threshold-crossing change candidate from this univariate screen.
    WithinBaseline,
    /// A statistical change candidate, not an outbreak/diagnosis declaration.
    ChangeCandidate(ChangeDirection),
    /// This method refuses to promote the evidence to a candidate.
    Abstain(AbstentionReason),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurveillanceAssessment {
    pub disposition: ScreeningDisposition,
    pub baseline: Option<RobustBaseline>,
    pub baseline_observed: usize,
    pub baseline_missing: usize,
    /// Signed robust standardized deviation of the latest point estimate when
    /// the baseline has non-degenerate spread.
    pub robust_z: Option<f64>,
    pub latest: Option<IntervalEstimate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurveillanceScreenError {
    NonFiniteMeasurement,
    InvalidUncertaintyInterval,
    InvalidConfiguration,
    NonIncreasingTimestamp,
    LatestNotAfterBaseline,
    NonFiniteDerivedStatistic,
}

impl fmt::Display for SurveillanceScreenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteMeasurement => "surveillance measurements and bounds must be finite",
            Self::InvalidUncertaintyInterval => {
                "uncertainty must satisfy lower <= estimate <= upper"
            }
            Self::InvalidConfiguration => {
                "screen config requires >=3 baseline observations and finite positive thresholds"
            }
            Self::NonIncreasingTimestamp => {
                "baseline surveillance timestamps must be strictly increasing"
            }
            Self::LatestNotAfterBaseline => {
                "latest surveillance timestamp must be after the baseline series"
            }
            Self::NonFiniteDerivedStatistic => {
                "surveillance screen produced a non-finite derived statistic"
            }
        };
        f.write_str(message)
    }
}

impl Error for SurveillanceScreenError {}

/// Screen the latest aggregate observation against a robust historical baseline.
///
/// The historical baseline uses observed point estimates only; their uncertainty
/// intervals are retained as source evidence but are not collapsed into a
/// synthetic baseline probability distribution. The latest interval is used as
/// a conservative guardrail: a nominal robust-z excursion is not promoted when
/// its uncertainty overlaps the configured baseline threshold envelope.
///
/// This function does not assess persistence, source independence, causality,
/// disease identity, or operational severity.
pub fn assess_latest_change(
    baseline_points: &[SurveillancePoint],
    latest: SurveillancePoint,
    config: SurveillanceScreenConfig,
) -> Result<SurveillanceAssessment, SurveillanceScreenError> {
    validate_config(config)?;
    validate_time_order(baseline_points, latest)?;

    let observed_values: Vec<f64> = baseline_points
        .iter()
        .filter_map(|point| point.measurement.map(|m| m.estimate))
        .collect();
    let baseline_observed = observed_values.len();
    let baseline_missing = baseline_points.len().saturating_sub(baseline_observed);

    if baseline_observed < config.min_baseline_observations {
        return Ok(SurveillanceAssessment {
            disposition: ScreeningDisposition::InsufficientBaseline,
            baseline: None,
            baseline_observed,
            baseline_missing,
            robust_z: None,
            latest: latest.measurement,
        });
    }

    let center = median(&observed_values).ok_or(SurveillanceScreenError::NonFiniteDerivedStatistic)?;
    let absolute_deviations: Vec<f64> = observed_values
        .iter()
        .map(|value| (value - center).abs())
        .collect();
    let mad = median(&absolute_deviations)
        .ok_or(SurveillanceScreenError::NonFiniteDerivedStatistic)?;
    let robust_scale = mad * MAD_NORMAL_SCALE;

    if !center.is_finite() || !mad.is_finite() || !robust_scale.is_finite() {
        return Err(SurveillanceScreenError::NonFiniteDerivedStatistic);
    }

    let threshold_width = config.robust_z_threshold * robust_scale;
    let lower_threshold_envelope = center - threshold_width;
    let upper_threshold_envelope = center + threshold_width;
    if !threshold_width.is_finite()
        || !lower_threshold_envelope.is_finite()
        || !upper_threshold_envelope.is_finite()
    {
        return Err(SurveillanceScreenError::NonFiniteDerivedStatistic);
    }

    let baseline = RobustBaseline {
        center,
        mad,
        robust_scale,
        lower_threshold_envelope,
        upper_threshold_envelope,
    };

    let Some(latest_measurement) = latest.measurement else {
        return Ok(SurveillanceAssessment {
            disposition: ScreeningDisposition::Abstain(AbstentionReason::LatestMeasurementMissing),
            baseline: Some(baseline),
            baseline_observed,
            baseline_missing,
            robust_z: None,
            latest: None,
        });
    };

    if robust_scale <= config.scale_epsilon {
        let disposition = if latest_measurement.contains(center) {
            ScreeningDisposition::WithinBaseline
        } else {
            ScreeningDisposition::Abstain(AbstentionReason::DegenerateBaselineSpread)
        };
        return Ok(SurveillanceAssessment {
            disposition,
            baseline: Some(baseline),
            baseline_observed,
            baseline_missing,
            robust_z: None,
            latest: Some(latest_measurement),
        });
    }

    let robust_z = (latest_measurement.estimate - center) / robust_scale;
    if !robust_z.is_finite() {
        return Err(SurveillanceScreenError::NonFiniteDerivedStatistic);
    }

    if robust_z.abs() < config.robust_z_threshold {
        return Ok(SurveillanceAssessment {
            disposition: ScreeningDisposition::WithinBaseline,
            baseline: Some(baseline),
            baseline_observed,
            baseline_missing,
            robust_z: Some(robust_z),
            latest: Some(latest_measurement),
        });
    }

    let direction = if robust_z.is_sign_positive() {
        ChangeDirection::Upward
    } else {
        ChangeDirection::Downward
    };
    let interval_clearly_separated = match direction {
        ChangeDirection::Upward => latest_measurement.lower > upper_threshold_envelope,
        ChangeDirection::Downward => latest_measurement.upper < lower_threshold_envelope,
    };
    let disposition = if interval_clearly_separated {
        ScreeningDisposition::ChangeCandidate(direction)
    } else {
        ScreeningDisposition::Abstain(AbstentionReason::UncertaintyOverlapsThresholdEnvelope)
    };

    Ok(SurveillanceAssessment {
        disposition,
        baseline: Some(baseline),
        baseline_observed,
        baseline_missing,
        robust_z: Some(robust_z),
        latest: Some(latest_measurement),
    })
}

fn validate_config(config: SurveillanceScreenConfig) -> Result<(), SurveillanceScreenError> {
    if config.min_baseline_observations < 3
        || !config.robust_z_threshold.is_finite()
        || config.robust_z_threshold <= 0.0
        || !config.scale_epsilon.is_finite()
        || config.scale_epsilon <= 0.0
    {
        return Err(SurveillanceScreenError::InvalidConfiguration);
    }
    Ok(())
}

fn validate_time_order(
    baseline_points: &[SurveillancePoint],
    latest: SurveillancePoint,
) -> Result<(), SurveillanceScreenError> {
    for pair in baseline_points.windows(2) {
        if pair[1].observed_at_unix_s <= pair[0].observed_at_unix_s {
            return Err(SurveillanceScreenError::NonIncreasingTimestamp);
        }
    }
    if let Some(last) = baseline_points.last()
        && latest.observed_at_unix_s <= last.observed_at_unix_s
    {
        return Err(SurveillanceScreenError::LatestNotAfterBaseline);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(t: i64, value: f64) -> SurveillancePoint {
        SurveillancePoint::observed(t, value, value - 0.1, value + 0.1).unwrap()
    }

    fn baseline() -> Vec<SurveillancePoint> {
        vec![
            observed(1, 8.0),
            observed(2, 9.0),
            observed(3, 10.0),
            observed(4, 11.0),
            observed(5, 12.0),
        ]
    }

    fn config() -> SurveillanceScreenConfig {
        SurveillanceScreenConfig::new(5, 3.0).unwrap()
    }

    #[test]
    fn insufficient_observed_history_is_not_upgraded_by_missing_rows() {
        let history = vec![
            observed(1, 9.0),
            SurveillancePoint::missing(2),
            observed(3, 10.0),
            SurveillancePoint::missing(4),
            observed(5, 11.0),
        ];
        let assessment = assess_latest_change(&history, observed(6, 20.0), config()).unwrap();
        assert_eq!(
            assessment.disposition,
            ScreeningDisposition::InsufficientBaseline
        );
        assert_eq!(assessment.baseline_observed, 3);
        assert_eq!(assessment.baseline_missing, 2);
        assert!(assessment.baseline.is_none());
    }

    #[test]
    fn stable_latest_point_remains_within_baseline() {
        let latest = SurveillancePoint::observed(6, 11.0, 10.7, 11.3).unwrap();
        let assessment = assess_latest_change(&baseline(), latest, config()).unwrap();
        assert_eq!(assessment.disposition, ScreeningDisposition::WithinBaseline);
        assert!(assessment.robust_z.unwrap().abs() < 3.0);
    }

    #[test]
    fn clearly_separated_upward_excursion_is_only_a_change_candidate() {
        let latest = SurveillancePoint::observed(6, 20.0, 19.0, 21.0).unwrap();
        let assessment = assess_latest_change(&baseline(), latest, config()).unwrap();
        assert_eq!(
            assessment.disposition,
            ScreeningDisposition::ChangeCandidate(ChangeDirection::Upward)
        );
        assert!(assessment.robust_z.unwrap() > 3.0);
    }

    #[test]
    fn clearly_separated_downward_excursion_is_only_a_change_candidate() {
        let latest = SurveillancePoint::observed(6, 0.0, 0.0, 1.0).unwrap();
        let assessment = assess_latest_change(&baseline(), latest, config()).unwrap();
        assert_eq!(
            assessment.disposition,
            ScreeningDisposition::ChangeCandidate(ChangeDirection::Downward)
        );
        assert!(assessment.robust_z.unwrap() < -3.0);
    }

    #[test]
    fn uncertainty_overlap_forces_abstention_despite_point_estimate_excursion() {
        let latest = SurveillancePoint::observed(6, 16.0, 13.0, 19.0).unwrap();
        let assessment = assess_latest_change(&baseline(), latest, config()).unwrap();
        assert!(assessment.robust_z.unwrap() > 3.0);
        assert_eq!(
            assessment.disposition,
            ScreeningDisposition::Abstain(
                AbstentionReason::UncertaintyOverlapsThresholdEnvelope
            )
        );
    }

    #[test]
    fn zero_mad_never_divides_by_zero_or_invents_an_infinite_score() {
        let history = vec![
            observed(1, 10.0),
            observed(2, 10.0),
            observed(3, 10.0),
            observed(4, 10.0),
            observed(5, 10.0),
        ];
        let latest = SurveillancePoint::observed(6, 12.0, 11.5, 12.5).unwrap();
        let assessment = assess_latest_change(&history, latest, config()).unwrap();
        assert_eq!(
            assessment.disposition,
            ScreeningDisposition::Abstain(AbstentionReason::DegenerateBaselineSpread)
        );
        assert_eq!(assessment.robust_z, None);
        assert_eq!(assessment.baseline.unwrap().robust_scale, 0.0);
    }

    #[test]
    fn zero_mad_with_latest_interval_containing_baseline_is_stable() {
        let history = vec![
            observed(1, 10.0),
            observed(2, 10.0),
            observed(3, 10.0),
            observed(4, 10.0),
            observed(5, 10.0),
        ];
        let latest = SurveillancePoint::observed(6, 10.1, 9.9, 10.2).unwrap();
        let assessment = assess_latest_change(&history, latest, config()).unwrap();
        assert_eq!(assessment.disposition, ScreeningDisposition::WithinBaseline);
        assert_eq!(assessment.robust_z, None);
    }

    #[test]
    fn missing_latest_measurement_is_an_explicit_abstention() {
        let assessment =
            assess_latest_change(&baseline(), SurveillancePoint::missing(6), config()).unwrap();
        assert_eq!(
            assessment.disposition,
            ScreeningDisposition::Abstain(AbstentionReason::LatestMeasurementMissing)
        );
    }

    #[test]
    fn timestamps_must_be_strictly_ordered() {
        let mut history = baseline();
        history[3] = observed(3, 11.0);
        assert_eq!(
            assess_latest_change(&history, observed(6, 20.0), config()),
            Err(SurveillanceScreenError::NonIncreasingTimestamp)
        );

        assert_eq!(
            assess_latest_change(&baseline(), observed(5, 20.0), config()),
            Err(SurveillanceScreenError::LatestNotAfterBaseline)
        );
    }

    #[test]
    fn nonfinite_and_invalid_intervals_fail_at_construction() {
        assert_eq!(
            SurveillancePoint::observed(1, f64::NAN, 0.0, 1.0),
            Err(SurveillanceScreenError::NonFiniteMeasurement)
        );
        assert_eq!(
            SurveillancePoint::observed(1, 2.0, 3.0, 4.0),
            Err(SurveillanceScreenError::InvalidUncertaintyInterval)
        );
    }

    #[test]
    fn screen_configuration_is_not_allowed_to_be_degenerate() {
        assert_eq!(
            SurveillanceScreenConfig::new(2, 3.0),
            Err(SurveillanceScreenError::InvalidConfiguration)
        );
        assert_eq!(
            SurveillanceScreenConfig::new(5, 0.0),
            Err(SurveillanceScreenError::InvalidConfiguration)
        );
    }
}
