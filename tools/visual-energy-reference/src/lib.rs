#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent arithmetic for task-conditioned visual-energy benchmarks.
//!
//! The crate integrates timestamped power samples with a trapezoidal rule,
//! rejects invalid or undersampled traces, and reports energy per successful
//! task without concealing a zero denominator. It does not qualify instruments,
//! task outcomes, optical quality, or a physical display.

use core::fmt;

/// One calibrated power reading at a monotonic time offset from the run start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerSample {
    /// Timestamp in nanoseconds from the run's monotonic-clock origin.
    pub timestamp_ns: u64,
    /// Instantaneous power in watts.
    pub power_watts: f64,
}

/// Closed interval for an energy measurement, expressed in joules.
///
/// Endpoints must conservatively cover the measurand under an explicitly
/// documented uncertainty method. This type does not claim that the interval
/// is a confidence interval or probability distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyInterval {
    /// Lowest allowed energy value in joules.
    pub lower_joules: f64,
    /// Highest allowed energy value in joules.
    pub upper_joules: f64,
}

/// Conservative interval for a relative energy saving, in percentage points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavingIntervalPercent {
    /// Lowest possible percentage saving, potentially negative.
    pub lower_percent: f64,
    /// Highest possible percentage saving, potentially negative or positive.
    pub upper_percent: f64,
}

/// Result of a decision-capability check for one declared measurand.
///
/// Uncertainty and minimum decision effect must use the same units and
/// compatible definitions. The factor of three is a project-level planning
/// criterion, not a universal metrology standard and not a calibration claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionCapability {
    /// Expanded uncertainty for the declared measurement, in its native units.
    pub expanded_uncertainty: f64,
    /// Smallest engineering difference that would change the decision.
    pub minimum_decision_effect: f64,
    /// Maximum uncertainty permitted by the configured one-third rule.
    pub maximum_allowed_uncertainty: f64,
    /// Whether the instrument/setup can resolve the declared decision effect.
    pub capable: bool,
}

/// Invalid trace, denominator, or comparison inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyError {
    /// At least two samples are required to integrate power over time.
    TooFewSamples,
    /// The declared maximum sample interval must be positive.
    ZeroMaximumGap,
    /// Timestamps must be strictly increasing.
    NonMonotonicTimestamp,
    /// Every power reading must be finite.
    NonFinitePower,
    /// Negative power readings are unsupported by this reference contract.
    NegativePower,
    /// A sample interval exceeds the declared maximum interval.
    SamplingGapExceeded,
    /// Integrated energy or a computed result is not finite.
    NonFiniteResult,
    /// Energy must be nonnegative.
    NegativeEnergy,
    /// A ratio requires at least one successful task.
    ZeroSuccessfulTasks,
    /// A relative saving is undefined for a zero-energy baseline.
    ZeroBaselineEnergy,
    /// A task-success rate requires at least one trial.
    ZeroTrials,
    /// Successful tasks cannot exceed the total number of trials.
    SuccessesExceedTrials,
    /// An uncertainty interval has invalid, reversed, or negative endpoints.
    InvalidEnergyInterval,
    /// The lower baseline energy bound must be strictly positive.
    BaselineIntervalIncludesZero,
    /// Measurement uncertainty must be finite and nonnegative.
    InvalidMeasurementUncertainty,
    /// Minimum decision effect must be finite and strictly positive.
    InvalidMinimumDecisionEffect,
}

impl fmt::Display for EnergyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSamples => formatter.write_str("at least two power samples are required"),
            Self::ZeroMaximumGap => formatter.write_str("maximum sample gap must be positive"),
            Self::NonMonotonicTimestamp => {
                formatter.write_str("power sample timestamps must be strictly increasing")
            }
            Self::NonFinitePower => formatter.write_str("power readings must be finite"),
            Self::NegativePower => formatter.write_str("power readings must be nonnegative"),
            Self::SamplingGapExceeded => {
                formatter.write_str("power sample gap exceeds the declared maximum")
            }
            Self::NonFiniteResult => formatter.write_str("energy result must be finite"),
            Self::NegativeEnergy => formatter.write_str("energy must be nonnegative"),
            Self::ZeroSuccessfulTasks => {
                formatter.write_str("energy per successful task needs at least one success")
            }
            Self::ZeroBaselineEnergy => {
                formatter.write_str("relative energy saving needs a positive baseline")
            }
            Self::ZeroTrials => formatter.write_str("task-success rate needs at least one trial"),
            Self::SuccessesExceedTrials => {
                formatter.write_str("successful tasks cannot exceed the number of trials")
            }
            Self::InvalidEnergyInterval => {
                formatter.write_str("energy interval must be finite, nonnegative, and ordered")
            }
            Self::BaselineIntervalIncludesZero => {
                formatter.write_str("baseline energy interval lower bound must be positive")
            }
            Self::InvalidMeasurementUncertainty => {
                formatter.write_str("measurement uncertainty must be finite and nonnegative")
            }
            Self::InvalidMinimumDecisionEffect => {
                formatter.write_str("minimum decision effect must be finite and positive")
            }
        }
    }
}

impl std::error::Error for EnergyError {}

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

fn validate_power_sample(sample: PowerSample) -> Result<(), EnergyError> {
    if !sample.power_watts.is_finite() {
        return Err(EnergyError::NonFinitePower);
    }
    if sample.power_watts < 0.0 {
        return Err(EnergyError::NegativePower);
    }
    Ok(())
}

/// Integrates power in watts over monotonic timestamps using the trapezoidal rule.
///
/// The caller must declare the largest acceptable gap between adjacent samples.
/// Gaps above that bound fail closed rather than being silently interpolated.
/// This function computes only the numeric integral; calibration and trace
/// completeness remain the caller's responsibility.
pub fn integrate_energy_joules(
    samples: &[PowerSample],
    max_gap_ns: u64,
) -> Result<f64, EnergyError> {
    if max_gap_ns == 0 {
        return Err(EnergyError::ZeroMaximumGap);
    }
    if samples.len() < 2 {
        return Err(EnergyError::TooFewSamples);
    }

    let mut total_joules = 0.0;

    for pair in samples.windows(2) {
        let first = pair[0];
        let second = pair[1];
        validate_power_sample(first)?;
        validate_power_sample(second)?;

        if second.timestamp_ns <= first.timestamp_ns {
            return Err(EnergyError::NonMonotonicTimestamp);
        }

        let gap_ns = second.timestamp_ns - first.timestamp_ns;
        if gap_ns > max_gap_ns {
            return Err(EnergyError::SamplingGapExceeded);
        }

        let average_power_watts = (first.power_watts + second.power_watts) / 2.0;
        let duration_seconds = gap_ns as f64 / NANOS_PER_SECOND;
        let segment_joules = average_power_watts * duration_seconds;
        total_joules += segment_joules;

        if !total_joules.is_finite() {
            return Err(EnergyError::NonFiniteResult);
        }
    }

    Ok(total_joules)
}

/// Computes energy per successful task, rejecting rather than smoothing zero successes.
///
/// Energy is derived from the supplied trace; a caller must independently bind
/// that trace to the intended device, configuration, workload and task window.
pub fn joules_per_successful_task(
    samples: &[PowerSample],
    max_gap_ns: u64,
    successful_tasks: u64,
) -> Result<f64, EnergyError> {
    let energy_joules = integrate_energy_joules(samples, max_gap_ns)?;
    if successful_tasks == 0 {
        return Err(EnergyError::ZeroSuccessfulTasks);
    }

    let result = energy_joules / successful_tasks as f64;
    if !result.is_finite() {
        return Err(EnergyError::NonFiniteResult);
    }
    Ok(result)
}

/// Percentage energy saving of a treatment relative to a positive baseline.
///
/// A negative result means the treatment consumed more energy. The result is a
/// point estimate; it says nothing about measurement uncertainty or task quality.
pub fn energy_saving_percent(
    baseline_joules: f64,
    treatment_joules: f64,
) -> Result<f64, EnergyError> {
    if !baseline_joules.is_finite() || !treatment_joules.is_finite() {
        return Err(EnergyError::NonFiniteResult);
    }
    if baseline_joules < 0.0 || treatment_joules < 0.0 {
        return Err(EnergyError::NegativeEnergy);
    }
    if baseline_joules == 0.0 {
        return Err(EnergyError::ZeroBaselineEnergy);
    }

    let result = 100.0 * (baseline_joules - treatment_joules) / baseline_joules;
    if !result.is_finite() {
        return Err(EnergyError::NonFiniteResult);
    }
    Ok(result)
}

/// Computes a conservative interval for percentage energy saving.
///
/// For baseline energy B in [B_low, B_high] and treatment energy T in
/// [T_low, T_high], returns [100 * (1 - T_high / B_low),
/// 100 * (1 - T_low / B_high)]. The result is interval arithmetic, not a
/// confidence interval. It makes no independence assumption and may be
/// conservative when measurements share correlated uncertainty sources.
/// A positive lower baseline bound is required for a finite meaningful ratio.
pub fn conservative_energy_saving_interval_percent(
    baseline: EnergyInterval,
    treatment: EnergyInterval,
) -> Result<SavingIntervalPercent, EnergyError> {
    for interval in [baseline, treatment] {
        if !interval.lower_joules.is_finite() || !interval.upper_joules.is_finite() {
            return Err(EnergyError::InvalidEnergyInterval);
        }
        if interval.lower_joules < 0.0 || interval.upper_joules < interval.lower_joules {
            return Err(EnergyError::InvalidEnergyInterval);
        }
    }
    if baseline.lower_joules == 0.0 {
        return Err(EnergyError::BaselineIntervalIncludesZero);
    }

    let lower = 100.0 * (1.0 - treatment.upper_joules / baseline.lower_joules);
    let upper = 100.0 * (1.0 - treatment.lower_joules / baseline.upper_joules);

    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(EnergyError::NonFiniteResult);
    }

    Ok(SavingIntervalPercent {
        lower_percent: lower,
        upper_percent: upper,
    })
}

/// Returns a task-success point estimate. Statistical uncertainty is out of scope.
pub fn task_success_rate(successes: u64, trials: u64) -> Result<f64, EnergyError> {
    if trials == 0 {
        return Err(EnergyError::ZeroTrials);
    }
    if successes > trials {
        return Err(EnergyError::SuccessesExceedTrials);
    }
    Ok(successes as f64 / trials as f64)
}

/// Returns treatment-minus-baseline task success in percentage points.
///
/// This is a point estimate only. A pass/fail non-inferiority claim requires a
/// preregistered statistical method, confidence interval, and adequate sample size.
pub fn task_success_delta_percentage_points(
    baseline_successes: u64,
    baseline_trials: u64,
    treatment_successes: u64,
    treatment_trials: u64,
) -> Result<f64, EnergyError> {
    let baseline = task_success_rate(baseline_successes, baseline_trials)?;
    let treatment = task_success_rate(treatment_successes, treatment_trials)?;
    let delta = 100.0 * (treatment - baseline);

    if !delta.is_finite() {
        return Err(EnergyError::NonFiniteResult);
    }
    Ok(delta)
}

/// Evaluates whether measurement uncertainty is small enough for a decision.
///
/// Returns capable = true only when expanded uncertainty is no more than one
/// third of the minimum decision effect. Both inputs must be finite; uncertainty
/// may be zero but cannot be negative, and the minimum decision effect must be
/// strictly positive. The units are caller-defined but must match. This check
/// does not establish that the uncertainty estimate itself is valid: the caller
/// must supply a documented uncertainty budget and valid instrument evidence.
pub fn decision_capability_gate(
    expanded_uncertainty: f64,
    minimum_decision_effect: f64,
) -> Result<DecisionCapability, EnergyError> {
    if !expanded_uncertainty.is_finite() || expanded_uncertainty < 0.0 {
        return Err(EnergyError::InvalidMeasurementUncertainty);
    }
    if !minimum_decision_effect.is_finite() || minimum_decision_effect <= 0.0 {
        return Err(EnergyError::InvalidMinimumDecisionEffect);
    }

    let maximum_allowed_uncertainty = minimum_decision_effect / 3.0;
    Ok(DecisionCapability {
        expanded_uncertainty,
        minimum_decision_effect,
        maximum_allowed_uncertainty,
        capable: expanded_uncertainty <= maximum_allowed_uncertainty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND_NS: u64 = 1_000_000_000;

    fn sample(seconds: u64, power_watts: f64) -> PowerSample {
        PowerSample {
            timestamp_ns: seconds * SECOND_NS,
            power_watts,
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn trapezoidal_integral_matches_constant_and_linear_known_answers() {
        let constant = [sample(0, 4.0), sample(3, 4.0)];
        close(
            integrate_energy_joules(&constant, 3 * SECOND_NS).expect("valid constant trace"),
            12.0,
            1e-12,
        );

        let linear = [sample(0, 4.0), sample(1, 6.0)];
        close(
            integrate_energy_joules(&linear, SECOND_NS).expect("valid linear trace"),
            5.0,
            1e-12,
        );
    }

    #[test]
    fn irregular_sample_intervals_are_integrated_segment_by_segment() {
        let trace = [sample(0, 4.0), sample(1, 5.0), sample(3, 7.0)];
        close(
            integrate_energy_joules(&trace, 2 * SECOND_NS).expect("valid irregular trace"),
            16.5,
            1e-12,
        );
    }

    #[test]
    fn rejects_too_few_samples_and_zero_maximum_gap() {
        assert_eq!(
            integrate_energy_joules(&[], SECOND_NS),
            Err(EnergyError::TooFewSamples)
        );
        assert_eq!(
            integrate_energy_joules(&[sample(0, 4.0)], SECOND_NS),
            Err(EnergyError::TooFewSamples)
        );
        assert_eq!(
            integrate_energy_joules(&[sample(0, 4.0), sample(1, 4.0)], 0),
            Err(EnergyError::ZeroMaximumGap)
        );
    }

    #[test]
    fn rejects_duplicate_or_decreasing_timestamps() {
        assert_eq!(
            integrate_energy_joules(&[sample(1, 4.0), sample(1, 5.0)], SECOND_NS),
            Err(EnergyError::NonMonotonicTimestamp)
        );
        assert_eq!(
            integrate_energy_joules(&[sample(2, 4.0), sample(1, 5.0)], SECOND_NS),
            Err(EnergyError::NonMonotonicTimestamp)
        );
    }

    #[test]
    fn rejects_nonfinite_and_negative_power() {
        assert_eq!(
            integrate_energy_joules(&[sample(0, 4.0), sample(1, f64::NAN)], SECOND_NS),
            Err(EnergyError::NonFinitePower)
        );
        assert_eq!(
            integrate_energy_joules(&[sample(0, 4.0), sample(1, f64::INFINITY)], SECOND_NS),
            Err(EnergyError::NonFinitePower)
        );
        assert_eq!(
            integrate_energy_joules(&[sample(0, 4.0), sample(1, -0.1)], SECOND_NS),
            Err(EnergyError::NegativePower)
        );
    }

    #[test]
    fn rejects_sampling_gaps_above_the_declared_bound() {
        assert_eq!(
            integrate_energy_joules(&[sample(0, 4.0), sample(2, 4.0)], SECOND_NS),
            Err(EnergyError::SamplingGapExceeded)
        );
    }

    #[test]
    fn energy_per_successful_task_rejects_zero_successes() {
        let trace = [sample(0, 4.0), sample(1, 6.0)];
        close(
            joules_per_successful_task(&trace, SECOND_NS, 1).expect("one successful task"),
            5.0,
            1e-12,
        );
        assert_eq!(
            joules_per_successful_task(&trace, SECOND_NS, 0),
            Err(EnergyError::ZeroSuccessfulTasks)
        );
    }

    #[test]
    fn energy_saving_reports_savings_and_regressions_without_clamping() {
        close(
            energy_saving_percent(100.0, 90.0).expect("positive baseline"),
            10.0,
            1e-12,
        );
        close(
            energy_saving_percent(100.0, 110.0).expect("treatment regression"),
            -10.0,
            1e-12,
        );
        assert_eq!(
            energy_saving_percent(0.0, 10.0),
            Err(EnergyError::ZeroBaselineEnergy)
        );
        assert_eq!(
            energy_saving_percent(100.0, -1.0),
            Err(EnergyError::NegativeEnergy)
        );
        assert_eq!(
            energy_saving_percent(f64::NAN, 10.0),
            Err(EnergyError::NonFiniteResult)
        );
    }

    #[test]
    fn conservative_saving_interval_refuses_to_promote_uncertain_threshold_crossing() {
        let baseline = EnergyInterval {
            lower_joules: 98.0,
            upper_joules: 102.0,
        };
        let treatment = EnergyInterval {
            lower_joules: 88.0,
            upper_joules: 92.0,
        };
        let interval = conservative_energy_saving_interval_percent(baseline, treatment)
            .expect("valid energy intervals");

        close(interval.lower_percent, 100.0 * (1.0 - 92.0 / 98.0), 1e-12);
        close(interval.upper_percent, 100.0 * (1.0 - 88.0 / 102.0), 1e-12);
        assert!(interval.lower_percent < 10.0);
        assert!(interval.upper_percent > 10.0);
    }

    #[test]
    fn conservative_saving_interval_rejects_invalid_intervals_and_zero_baseline_bound() {
        let valid = EnergyInterval {
            lower_joules: 90.0,
            upper_joules: 100.0,
        };
        assert_eq!(
            conservative_energy_saving_interval_percent(
                EnergyInterval {
                    lower_joules: 0.0,
                    upper_joules: 1.0,
                },
                valid,
            ),
            Err(EnergyError::BaselineIntervalIncludesZero)
        );
        assert_eq!(
            conservative_energy_saving_interval_percent(
                EnergyInterval {
                    lower_joules: 10.0,
                    upper_joules: 9.0,
                },
                valid,
            ),
            Err(EnergyError::InvalidEnergyInterval)
        );
        assert_eq!(
            conservative_energy_saving_interval_percent(
                EnergyInterval {
                    lower_joules: f64::NAN,
                    upper_joules: 11.0,
                },
                valid,
            ),
            Err(EnergyError::InvalidEnergyInterval)
        );
        assert_eq!(
            conservative_energy_saving_interval_percent(
                valid,
                EnergyInterval {
                    lower_joules: -1.0,
                    upper_joules: 2.0,
                },
            ),
            Err(EnergyError::InvalidEnergyInterval)
        );
    }

    #[test]
    fn conservative_saving_interval_supports_regression_values() {
        let baseline = EnergyInterval {
            lower_joules: 99.0,
            upper_joules: 101.0,
        };
        let treatment = EnergyInterval {
            lower_joules: 109.0,
            upper_joules: 111.0,
        };
        let interval = conservative_energy_saving_interval_percent(baseline, treatment)
            .expect("valid energy intervals");
        assert!(interval.lower_percent < 0.0);
        assert!(interval.upper_percent < 0.0);
    }

    #[test]
    fn decision_capability_gate_accepts_boundary_and_rejects_uncertainty_too_large() {
        let boundary = decision_capability_gate(1.0, 3.0).expect("valid boundary");
        assert_eq!(boundary.maximum_allowed_uncertainty, 1.0);
        assert!(boundary.capable);

        let inadequate = decision_capability_gate(1.0001, 3.0).expect("valid inputs");
        assert!(!inadequate.capable);
        assert_eq!(inadequate.minimum_decision_effect, 3.0);

        let zero_uncertainty = decision_capability_gate(0.0, 0.01).expect("valid inputs");
        assert!(zero_uncertainty.capable);
    }

    #[test]
    fn decision_capability_gate_rejects_invalid_uncertainty_and_effect() {
        assert_eq!(
            decision_capability_gate(f64::NAN, 3.0),
            Err(EnergyError::InvalidMeasurementUncertainty)
        );
        assert_eq!(
            decision_capability_gate(f64::INFINITY, 3.0),
            Err(EnergyError::InvalidMeasurementUncertainty)
        );
        assert_eq!(
            decision_capability_gate(-0.1, 3.0),
            Err(EnergyError::InvalidMeasurementUncertainty)
        );
        assert_eq!(
            decision_capability_gate(0.1, 0.0),
            Err(EnergyError::InvalidMinimumDecisionEffect)
        );
        assert_eq!(
            decision_capability_gate(0.1, f64::NEG_INFINITY),
            Err(EnergyError::InvalidMinimumDecisionEffect)
        );
    }

    #[test]
    fn task_success_metrics_validate_counts_and_report_percentage_points() {
        close(task_success_rate(9, 10).expect("valid counts"), 0.9, 1e-12);
        close(
            task_success_delta_percentage_points(9, 10, 8, 10)
                .expect("valid treatment and baseline"),
            -10.0,
            1e-12,
        );
        assert_eq!(task_success_rate(0, 0), Err(EnergyError::ZeroTrials));
        assert_eq!(
            task_success_rate(11, 10),
            Err(EnergyError::SuccessesExceedTrials)
        );
    }
}
