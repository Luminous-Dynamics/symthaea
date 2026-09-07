// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive baseline-drift diagnostics for repeated-shock experiments.
//!
//! Repeated-shock recovery metrics intentionally normalize each shock to its own pre-shock
//! population baseline. That is appropriate for proportional resilience, but it means an improved
//! normalized response can coexist with a substantially smaller (or larger) population entering
//! shock 2. This module makes that context explicit without changing the recovery metric or
//! deciding whether a seed should be included in inference.
//!
//! These diagnostics are descriptive only. They are not eligibility gates and carry no p-values.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeatedShockBaselineShiftV1 {
    pub first_baseline_mean_observed_population: f64,
    pub second_baseline_mean_observed_population: f64,
    /// `second / first`. One means no proportional baseline shift.
    pub second_to_first_ratio: f64,
    /// `second / first - 1`. Positive means a larger population baseline before shock 2.
    pub fractional_change: f64,
    /// `ln(second / first)`. Symmetric in log space: equal proportional increases/decreases have
    /// equal magnitude and opposite sign.
    pub log_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeBaselineShiftV1 {
    /// Candidate ratio divided by reference ratio. Values above one mean the candidate's baseline
    /// grew more (or shrank less) between shocks than the reference's baseline.
    pub candidate_to_reference_ratio_of_ratios: f64,
    /// Candidate log baseline shift minus reference log baseline shift.
    pub log_ratio_advantage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedShockBaselineErrorV1 {
    NonFiniteBaseline,
    NonPositiveBaseline,
    NonFiniteDerivedMetric,
}

pub fn repeated_shock_baseline_shift(
    first_baseline_mean_observed_population: f64,
    second_baseline_mean_observed_population: f64,
) -> Result<RepeatedShockBaselineShiftV1, RepeatedShockBaselineErrorV1> {
    if !first_baseline_mean_observed_population.is_finite()
        || !second_baseline_mean_observed_population.is_finite()
    {
        return Err(RepeatedShockBaselineErrorV1::NonFiniteBaseline);
    }
    if first_baseline_mean_observed_population <= 0.0
        || second_baseline_mean_observed_population <= 0.0
    {
        return Err(RepeatedShockBaselineErrorV1::NonPositiveBaseline);
    }

    let second_to_first_ratio =
        second_baseline_mean_observed_population / first_baseline_mean_observed_population;
    let fractional_change = second_to_first_ratio - 1.0;
    let log_ratio = second_to_first_ratio.ln();
    if !second_to_first_ratio.is_finite()
        || !fractional_change.is_finite()
        || !log_ratio.is_finite()
    {
        return Err(RepeatedShockBaselineErrorV1::NonFiniteDerivedMetric);
    }

    Ok(RepeatedShockBaselineShiftV1 {
        first_baseline_mean_observed_population,
        second_baseline_mean_observed_population,
        second_to_first_ratio,
        fractional_change,
        log_ratio,
    })
}

pub fn compare_relative_baseline_shift(
    reference: &RepeatedShockBaselineShiftV1,
    candidate: &RepeatedShockBaselineShiftV1,
) -> Result<RelativeBaselineShiftV1, RepeatedShockBaselineErrorV1> {
    let candidate_to_reference_ratio_of_ratios =
        candidate.second_to_first_ratio / reference.second_to_first_ratio;
    let log_ratio_advantage = candidate.log_ratio - reference.log_ratio;
    if !candidate_to_reference_ratio_of_ratios.is_finite() || !log_ratio_advantage.is_finite() {
        return Err(RepeatedShockBaselineErrorV1::NonFiniteDerivedMetric);
    }
    Ok(RelativeBaselineShiftV1 {
        candidate_to_reference_ratio_of_ratios,
        log_ratio_advantage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_baseline_is_neutral() {
        let shift = repeated_shock_baseline_shift(20.0, 20.0).expect("positive baselines");
        assert_eq!(shift.second_to_first_ratio, 1.0);
        assert_eq!(shift.fractional_change, 0.0);
        assert_eq!(shift.log_ratio, 0.0);
    }

    #[test]
    fn proportional_baseline_change_is_scale_invariant() {
        let small = repeated_shock_baseline_shift(10.0, 5.0).expect("positive baselines");
        let large = repeated_shock_baseline_shift(1000.0, 500.0).expect("positive baselines");
        assert_eq!(small.second_to_first_ratio, large.second_to_first_ratio);
        assert_eq!(small.fractional_change, large.fractional_change);
        assert_eq!(small.log_ratio, large.log_ratio);
    }

    #[test]
    fn equal_proportional_shifts_have_zero_relative_advantage() {
        let reference = repeated_shock_baseline_shift(10.0, 5.0).expect("reference");
        let candidate = repeated_shock_baseline_shift(40.0, 20.0).expect("candidate");
        let relative = compare_relative_baseline_shift(&reference, &candidate).expect("finite");
        assert_eq!(relative.candidate_to_reference_ratio_of_ratios, 1.0);
        assert_eq!(relative.log_ratio_advantage, 0.0);
    }

    #[test]
    fn positive_relative_log_shift_means_candidate_declined_less() {
        let reference = repeated_shock_baseline_shift(20.0, 10.0).expect("reference");
        let candidate = repeated_shock_baseline_shift(20.0, 15.0).expect("candidate");
        let relative = compare_relative_baseline_shift(&reference, &candidate).expect("finite");
        assert!(relative.candidate_to_reference_ratio_of_ratios > 1.0);
        assert!(relative.log_ratio_advantage > 0.0);
    }

    #[test]
    fn invalid_baselines_fail_closed() {
        assert_eq!(
            repeated_shock_baseline_shift(f64::NAN, 1.0),
            Err(RepeatedShockBaselineErrorV1::NonFiniteBaseline)
        );
        assert_eq!(
            repeated_shock_baseline_shift(0.0, 1.0),
            Err(RepeatedShockBaselineErrorV1::NonPositiveBaseline)
        );
        assert_eq!(
            repeated_shock_baseline_shift(1.0, -1.0),
            Err(RepeatedShockBaselineErrorV1::NonPositiveBaseline)
        );
    }
}
