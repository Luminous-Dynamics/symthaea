// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Checked, deterministic manufacturing KPI calculations.
//!
//! Calculators operate on supplied integer observations only. They do not
//! establish that a standard time, meter reading, inspection result, or source
//! record is valid. Zero denominators and impossible inputs are explicit errors;
//! unknown measurements must remain absent at the caller boundary.

/// Fixed-point scale: 1_000_000 parts per unit (one ppm of a ratio).
pub const PPM_SCALE: u128 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpiError {
    ZeroDenominator,
    NumeratorExceedsDenominator,
    RunTimeExceedsPlannedTime,
    FirstPassGoodExceedsTotal,
    ZeroIdealCycleTime,
    PerformanceAboveOneHundredPercent,
    EmptySample,
    InvalidPercentile,
    ArithmeticOverflow,
}

/// An exact numerator/denominator pair and its truncated parts-per-million value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RatioPpm {
    pub numerator: u128,
    pub denominator: u128,
    pub value_ppm: u128,
}

/// OEE components for one explicitly named work unit or asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OeeReport {
    pub work_unit_ref: String,
    pub availability: RatioPpm,
    pub performance: RatioPpm,
    pub quality: RatioPpm,
    pub oee_ppm: u128,
}

/// Auditable nearest-rank percentile result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PercentileResult {
    pub value: u64,
    pub sample_count: usize,
    /// One-based order-statistic rank selected by nearest-rank.
    pub selected_rank: usize,
    pub percentile_numerator: u32,
    pub percentile_denominator: u32,
}

/// Defect occurrence rate, expressed as milli-defects per 100 inspected units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefectRatePer100 {
    pub defect_occurrences: u64,
    pub inspected_units: u64,
    /// 1_000 means 1.0 defects per 100 units; 1 means 0.001.
    pub milli_defects_per_100_units: u128,
}

/// Calculate a ratio without requiring the numerator to be <= denominator.
///
/// This is appropriate for productivity ratios such as earned standard time /
/// paid direct-labour time, which can exceed 100%. A zero denominator is never
/// interpreted as zero performance.
pub fn ratio_ppm(numerator: u64, denominator: u64) -> Result<RatioPpm, KpiError> {
    ratio_ppm_u128(u128::from(numerator), u128::from(denominator))
}

/// Calculate a bounded proportion. Counts or masses that exceed their
/// corresponding denominator are rejected rather than silently clamped.
pub fn bounded_ratio_ppm(
    numerator: u64,
    denominator: u64,
) -> Result<RatioPpm, KpiError> {
    if numerator > denominator {
        return Err(KpiError::NumeratorExceedsDenominator);
    }
    ratio_ppm(numerator, denominator)
}

fn ratio_ppm_u128(numerator: u128, denominator: u128) -> Result<RatioPpm, KpiError> {
    if denominator == 0 {
        return Err(KpiError::ZeroDenominator);
    }
    let scaled = numerator
        .checked_mul(PPM_SCALE)
        .ok_or(KpiError::ArithmeticOverflow)?;
    Ok(RatioPpm {
        numerator,
        denominator,
        value_ppm: scaled / denominator,
    })
}

/// Quality-adjusted line efficiency:
/// first-pass accepted earned standard minutes / direct paid-labour minutes.
///
/// The ratio is intentionally unbounded: values above 100% are represented,
/// not capped. The caller must bind standard-time and paid-time policies.
pub fn line_efficiency_ppm(
    first_pass_earned_standard_minutes: u64,
    direct_paid_labour_minutes: u64,
) -> Result<RatioPpm, KpiError> {
    ratio_ppm(
        first_pass_earned_standard_minutes,
        direct_paid_labour_minutes,
    )
}

pub fn first_pass_yield_ppm(
    first_pass_good_units: u64,
    units_presented_for_first_inspection: u64,
) -> Result<RatioPpm, KpiError> {
    bounded_ratio_ppm(
        first_pass_good_units,
        units_presented_for_first_inspection,
    )
}

pub fn otif_ppm(
    due_orders_delivered_complete_and_on_time: u64,
    all_due_orders: u64,
) -> Result<RatioPpm, KpiError> {
    bounded_ratio_ppm(
        due_orders_delivered_complete_and_on_time,
        all_due_orders,
    )
}

pub fn material_yield_ppm(
    net_material_mass_in_first_pass_accepted_units: u64,
    gross_material_mass_issued: u64,
) -> Result<RatioPpm, KpiError> {
    bounded_ratio_ppm(
        net_material_mass_in_first_pass_accepted_units,
        gross_material_mass_issued,
    )
}

pub fn traceability_coverage_ppm(
    critical_lots_with_complete_lineage_and_disposition: u64,
    critical_lots_consumed: u64,
) -> Result<RatioPpm, KpiError> {
    bounded_ratio_ppm(
        critical_lots_with_complete_lineage_and_disposition,
        critical_lots_consumed,
    )
}

pub fn defects_per_100_units(
    defect_occurrences: u64,
    inspected_units: u64,
) -> Result<DefectRatePer100, KpiError> {
    if inspected_units == 0 {
        return Err(KpiError::ZeroDenominator);
    }
    let scaled = u128::from(defect_occurrences)
        .checked_mul(100_000)
        .ok_or(KpiError::ArithmeticOverflow)?;
    Ok(DefectRatePer100 {
        defect_occurrences,
        inspected_units,
        milli_defects_per_100_units: scaled / u128::from(inspected_units),
    })
}

/// Calculate OEE for a single named work unit/asset.
///
/// Availability = run_time / planned_production_time.
/// Performance = ideal_cycle_time * total_count / run_time.
/// Quality = first_pass_good_count / total_count.
///
/// Performance above 100% is treated as a likely cycle-time or time-boundary
/// mismatch and rejected; callers should not let this inflate OEE.
pub fn calculate_oee(
    work_unit_ref: impl Into<String>,
    planned_production_time_seconds: u64,
    run_time_seconds: u64,
    ideal_cycle_time_seconds_per_unit: u64,
    total_count: u64,
    first_pass_good_count: u64,
) -> Result<OeeReport, KpiError> {
    if run_time_seconds > planned_production_time_seconds {
        return Err(KpiError::RunTimeExceedsPlannedTime);
    }
    if first_pass_good_count > total_count {
        return Err(KpiError::FirstPassGoodExceedsTotal);
    }
    if ideal_cycle_time_seconds_per_unit == 0 {
        return Err(KpiError::ZeroIdealCycleTime);
    }

    let availability = bounded_ratio_ppm(
        run_time_seconds,
        planned_production_time_seconds,
    )?;
    let ideal_total_seconds = u128::from(ideal_cycle_time_seconds_per_unit)
        .checked_mul(u128::from(total_count))
        .ok_or(KpiError::ArithmeticOverflow)?;
    let performance = ratio_ppm_u128(
        ideal_total_seconds,
        u128::from(run_time_seconds),
    )?;
    if performance.value_ppm > PPM_SCALE {
        return Err(KpiError::PerformanceAboveOneHundredPercent);
    }
    let quality = bounded_ratio_ppm(first_pass_good_count, total_count)?;

    let product = availability
        .value_ppm
        .checked_mul(performance.value_ppm)
        .and_then(|value| value.checked_mul(quality.value_ppm))
        .ok_or(KpiError::ArithmeticOverflow)?;
    let scale_squared = PPM_SCALE
        .checked_mul(PPM_SCALE)
        .ok_or(KpiError::ArithmeticOverflow)?;

    Ok(OeeReport {
        work_unit_ref: work_unit_ref.into(),
        availability,
        performance,
        quality,
        oee_ppm: product / scale_squared,
    })
}

/// Nearest-rank percentile for an integer sample.
///
/// The probability numerator / denominator must be in (0, 1]. The selected
/// one-based rank is ceil(p * n). Input is copied, so caller ordering is not
/// mutated.
pub fn nearest_rank_percentile(
    values: &[u64],
    percentile_numerator: u32,
    percentile_denominator: u32,
) -> Result<PercentileResult, KpiError> {
    if values.is_empty() {
        return Err(KpiError::EmptySample);
    }
    if percentile_denominator == 0
        || percentile_numerator == 0
        || percentile_numerator > percentile_denominator
    {
        return Err(KpiError::InvalidPercentile);
    }

    let n = values.len() as u128;
    let numerator = u128::from(percentile_numerator)
        .checked_mul(n)
        .ok_or(KpiError::ArithmeticOverflow)?;
    let denominator = u128::from(percentile_denominator);
    let rank = numerator
        .checked_add(denominator - 1)
        .ok_or(KpiError::ArithmeticOverflow)?
        / denominator;
    let rank = usize::try_from(rank).map_err(|_| KpiError::ArithmeticOverflow)?;

    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    let selected = *ordered.get(rank.saturating_sub(1)).ok_or(KpiError::InvalidPercentile)?;

    Ok(PercentileResult {
        value: selected,
        sample_count: values.len(),
        selected_rank: rank,
        percentile_numerator,
        percentile_denominator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_ppm_preserves_numerator_and_denominator() {
        let ratio = ratio_ppm(83, 100).unwrap();
        assert_eq!(ratio.numerator, 83);
        assert_eq!(ratio.denominator, 100);
        assert_eq!(ratio.value_ppm, 830_000);
    }

    #[test]
    fn zero_denominator_is_not_zero_performance() {
        assert_eq!(ratio_ppm(0, 0), Err(KpiError::ZeroDenominator));
        assert_eq!(first_pass_yield_ppm(0, 0), Err(KpiError::ZeroDenominator));
    }

    #[test]
    fn bounded_ratio_rejects_numerator_above_denominator() {
        assert_eq!(
            bounded_ratio_ppm(101, 100),
            Err(KpiError::NumeratorExceedsDenominator)
        );
    }

    #[test]
    fn line_efficiency_can_exceed_one_hundred_percent_without_clamping() {
        let efficiency = line_efficiency_ppm(1_200, 1_000).unwrap();
        assert_eq!(efficiency.value_ppm, 1_200_000);
    }

    #[test]
    fn first_pass_yield_known_answer() {
        assert_eq!(first_pass_yield_ppm(936, 1_000).unwrap().value_ppm, 936_000);
    }

    #[test]
    fn otif_known_answer() {
        assert_eq!(otif_ppm(47, 50).unwrap().value_ppm, 940_000);
    }

    #[test]
    fn material_yield_known_answer() {
        assert_eq!(material_yield_ppm(72, 90).unwrap().value_ppm, 800_000);
    }

    #[test]
    fn traceability_coverage_known_answer() {
        assert_eq!(traceability_coverage_ppm(100, 100).unwrap().value_ppm, PPM_SCALE);
    }

    #[test]
    fn defects_per_100_units_known_answer() {
        let rate = defects_per_100_units(13, 200).unwrap();
        assert_eq!(rate.milli_defects_per_100_units, 6_500);
    }

    #[test]
    fn oee_known_answer_for_one_work_unit() {
        let report = calculate_oee("urn:example:asset:sewing-cell-a", 3_600, 3_240, 2, 1_440, 1_368).unwrap();
        assert_eq!(report.availability.value_ppm, 900_000);
        assert_eq!(report.performance.value_ppm, 888_888);
        assert_eq!(report.quality.value_ppm, 950_000);
        assert_eq!(report.oee_ppm, 760_000);
        assert_eq!(report.work_unit_ref, "urn:example:asset:sewing-cell-a");
    }

    #[test]
    fn oee_rejects_runtime_above_planned_time() {
        assert_eq!(
            calculate_oee("asset", 100, 101, 1, 50, 50),
            Err(KpiError::RunTimeExceedsPlannedTime)
        );
    }

    #[test]
    fn oee_rejects_good_count_above_total() {
        assert_eq!(
            calculate_oee("asset", 100, 90, 1, 50, 51),
            Err(KpiError::FirstPassGoodExceedsTotal)
        );
    }

    #[test]
    fn oee_rejects_zero_ideal_cycle_time() {
        assert_eq!(
            calculate_oee("asset", 100, 90, 0, 50, 50),
            Err(KpiError::ZeroIdealCycleTime)
        );
    }

    #[test]
    fn oee_rejects_performance_above_one_hundred_percent() {
        assert_eq!(
            calculate_oee("asset", 100, 90, 2, 50, 50),
            Err(KpiError::PerformanceAboveOneHundredPercent)
        );
    }

    #[test]
    fn oee_rejects_zero_run_time_with_positive_output() {
        assert_eq!(
            calculate_oee("asset", 100, 0, 1, 1, 1),
            Err(KpiError::ZeroDenominator)
        );
    }

    #[test]
    fn nearest_rank_p50_and_p95_are_deterministic() {
        let values = [8, 1, 3, 9, 4];
        let p50 = nearest_rank_percentile(&values, 50, 100).unwrap();
        let p95 = nearest_rank_percentile(&values, 95, 100).unwrap();
        assert_eq!(p50.value, 4);
        assert_eq!(p50.selected_rank, 3);
        assert_eq!(p95.value, 9);
        assert_eq!(p95.selected_rank, 5);
    }

    #[test]
    fn percentile_rejects_empty_sample() {
        assert_eq!(nearest_rank_percentile(&[], 95, 100), Err(KpiError::EmptySample));
    }

    #[test]
    fn percentile_rejects_invalid_probability() {
        assert_eq!(nearest_rank_percentile(&[1, 2], 0, 100), Err(KpiError::InvalidPercentile));
        assert_eq!(nearest_rank_percentile(&[1, 2], 101, 100), Err(KpiError::InvalidPercentile));
        assert_eq!(nearest_rank_percentile(&[1, 2], 1, 0), Err(KpiError::InvalidPercentile));
    }

    #[test]
    fn ratio_is_deterministic_and_supports_large_intermediates() {
        assert_eq!(
            ratio_ppm(u64::MAX, 1),
            Ok(RatioPpm {
                numerator: u128::from(u64::MAX),
                denominator: 1,
                value_ppm: u128::from(u64::MAX) * PPM_SCALE,
            })
        );
    }

    #[test]
    fn oee_detects_performance_scaling_overflow() {
        assert_eq!(
            calculate_oee("asset", u64::MAX, 1, u64::MAX, u64::MAX, 1),
            Err(KpiError::ArithmeticOverflow)
        );
    }

    #[test]
    fn percentile_is_independent_of_input_order() {
        let first = nearest_rank_percentile(&[10, 5, 20, 15], 50, 100).unwrap();
        let second = nearest_rank_percentile(&[20, 10, 15, 5], 50, 100).unwrap();
        assert_eq!(first, second);
    }
}
