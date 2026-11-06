//! Provenance-bound integration of sampled signed grid power into energy.
//!
//! Positive power is export/discharge; negative power is import/charge.
//! Samples use integer milliwatts and Unix UTC seconds. Adjacent samples are
//! trapezoid-integrated only when the elapsed interval is no greater than the
//! caller's declared maximum gap. Larger gaps are reported as uncovered time;
//! no interpolation, forward-fill or extrapolation is performed.
//!
//! The signed energy is retained as an exact numerator in milli-Wh with a
//! denominator of 7,200. This avoids rounding each one-second interval before
//! aggregation. Convert source kW to milliwatts with checked integer scaling.

const ENERGY_DENOMINATOR_MILLI_WH: i128 = 7_200;
const COVERAGE_BPS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerTraceSample<'a> {
    pub sample_id: &'a str,
    pub timestamp_utc_seconds: i64,
    /// Signed integer milliwatts: positive export, negative import.
    pub power_milli_w: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerTraceInputs<'a> {
    pub series_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub horizon_start_utc_seconds: i64,
    pub horizon_end_utc_seconds: i64,
    /// Intervals longer than this are treated as uncovered, not interpolated.
    pub maximum_integrable_gap_seconds: u32,
    pub samples: &'a [PowerTraceSample<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerTraceError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidHorizon,
    InvalidMaximumGap,
    EmptyPowerTrace,
    SampleOutsideHorizon,
    DuplicateSampleId,
    DuplicateTimestamp,
    SamplesOutOfOrder,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerTraceAssessment<'a> {
    pub series_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub horizon_duration_seconds: u64,
    pub sample_count: u64,
    pub integrated_interval_count: u64,
    pub gap_interval_count: u64,
    pub observed_coverage_seconds: u64,
    pub uncovered_seconds: u64,
    pub maximum_observed_gap_seconds: u64,
    pub time_coverage_bps: u64,
    /// Exact trapezoid energy numerator in milli-Wh; divide by 7,200.
    pub signed_energy_milli_wh_numerator: i128,
    pub energy_denominator: i128,
    pub peak_export_power_milli_w: u64,
    pub peak_import_power_abs_milli_w: u64,
    pub trace_complete: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Convert a source kW value to integer milliwatts without floating point.
/// One kW is 1,000,000 milliwatts.
pub fn kilowatts_to_milliwatts(value_kw: i64) -> Result<i64, PowerTraceError> {
    value_kw
        .checked_mul(1_000_000)
        .ok_or(PowerTraceError::ArithmeticOverflow)
}

/// Integrate a single ordered power trace. Only adjacent intervals within the
/// declared gap threshold contribute energy. Coverage and endpoint checks are
/// returned independently from the integrated value so incomplete traces
/// cannot be mistaken for full-horizon energy.
pub fn integrate_power_trace<'a>(
    inputs: PowerTraceInputs<'a>,
) -> Result<PowerTraceAssessment<'a>, PowerTraceError> {
    if !nonempty(inputs.series_id) || !nonempty(inputs.source_ref) {
        return Err(PowerTraceError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.source_sha256) {
        return Err(PowerTraceError::InvalidSourceDigest);
    }
    if inputs.horizon_start_utc_seconds >= inputs.horizon_end_utc_seconds {
        return Err(PowerTraceError::InvalidHorizon);
    }
    if inputs.maximum_integrable_gap_seconds == 0 {
        return Err(PowerTraceError::InvalidMaximumGap);
    }
    if inputs.samples.len() < 2 {
        return Err(PowerTraceError::EmptyPowerTrace);
    }

    let horizon_duration = u64::try_from(
        i128::from(inputs.horizon_end_utc_seconds)
            .checked_sub(i128::from(inputs.horizon_start_utc_seconds))
            .ok_or(PowerTraceError::ArithmeticOverflow)?,
    )
    .map_err(|_| PowerTraceError::ArithmeticOverflow)?;

    let mut integrated_interval_count = 0_u64;
    let mut gap_interval_count = 0_u64;
    let mut observed_coverage_seconds = 0_u64;
    let mut maximum_observed_gap_seconds = 0_u64;
    let mut energy_numerator = 0_i128;
    let mut peak_export = 0_u64;
    let mut peak_import = 0_u64;

    for (index, sample) in inputs.samples.iter().copied().enumerate() {
        if !nonempty(sample.sample_id) {
            return Err(PowerTraceError::InvalidEvidenceReference);
        }
        if sample.timestamp_utc_seconds < inputs.horizon_start_utc_seconds
            || sample.timestamp_utc_seconds > inputs.horizon_end_utc_seconds
        {
            return Err(PowerTraceError::SampleOutsideHorizon);
        }
        if inputs.samples[..index]
            .iter()
            .any(|previous| previous.sample_id == sample.sample_id)
        {
            return Err(PowerTraceError::DuplicateSampleId);
        }

        let power_abs = u64::try_from(i128::from(sample.power_milli_w).abs())
            .map_err(|_| PowerTraceError::ArithmeticOverflow)?;
        if sample.power_milli_w > 0 {
            peak_export = peak_export.max(power_abs);
        } else if sample.power_milli_w < 0 {
            peak_import = peak_import.max(power_abs);
        }

        if index == 0 {
            continue;
        }
        let previous = inputs.samples[index - 1];
        if sample.timestamp_utc_seconds == previous.timestamp_utc_seconds {
            return Err(PowerTraceError::DuplicateTimestamp);
        }
        if sample.timestamp_utc_seconds < previous.timestamp_utc_seconds {
            return Err(PowerTraceError::SamplesOutOfOrder);
        }

        let elapsed = u64::try_from(
            i128::from(sample.timestamp_utc_seconds)
                .checked_sub(i128::from(previous.timestamp_utc_seconds))
                .ok_or(PowerTraceError::ArithmeticOverflow)?,
        )
        .map_err(|_| PowerTraceError::ArithmeticOverflow)?;
        maximum_observed_gap_seconds = maximum_observed_gap_seconds.max(elapsed);
        if elapsed > u64::from(inputs.maximum_integrable_gap_seconds) {
            gap_interval_count = gap_interval_count
                .checked_add(1)
                .ok_or(PowerTraceError::ArithmeticOverflow)?;
            continue;
        }

        let power_sum = i128::from(previous.power_milli_w)
            .checked_add(i128::from(sample.power_milli_w))
            .ok_or(PowerTraceError::ArithmeticOverflow)?;
        let interval_numerator = power_sum
            .checked_mul(i128::from(elapsed))
            .ok_or(PowerTraceError::ArithmeticOverflow)?;
        energy_numerator = energy_numerator
            .checked_add(interval_numerator)
            .ok_or(PowerTraceError::ArithmeticOverflow)?;
        integrated_interval_count = integrated_interval_count
            .checked_add(1)
            .ok_or(PowerTraceError::ArithmeticOverflow)?;
        observed_coverage_seconds = observed_coverage_seconds
            .checked_add(elapsed)
            .ok_or(PowerTraceError::ArithmeticOverflow)?;
    }

    let uncovered_seconds = horizon_duration
        .checked_sub(observed_coverage_seconds)
        .ok_or(PowerTraceError::ArithmeticOverflow)?;
    let coverage_numerator = u128::from(observed_coverage_seconds)
        .checked_mul(COVERAGE_BPS)
        .ok_or(PowerTraceError::ArithmeticOverflow)?;
    let time_coverage_bps = u64::try_from(coverage_numerator / u128::from(horizon_duration))
        .map_err(|_| PowerTraceError::ArithmeticOverflow)?;

    let first_at_start = inputs.samples[0].timestamp_utc_seconds
        == inputs.horizon_start_utc_seconds;
    let last_at_end = inputs.samples[inputs.samples.len() - 1].timestamp_utc_seconds
        == inputs.horizon_end_utc_seconds;
    let trace_complete = first_at_start
        && last_at_end
        && gap_interval_count == 0
        && uncovered_seconds == 0;

    Ok(PowerTraceAssessment {
        series_id: inputs.series_id,
        source_ref: inputs.source_ref,
        source_sha256: inputs.source_sha256,
        horizon_duration_seconds: horizon_duration,
        sample_count: u64::try_from(inputs.samples.len())
            .map_err(|_| PowerTraceError::ArithmeticOverflow)?,
        integrated_interval_count,
        gap_interval_count,
        observed_coverage_seconds,
        uncovered_seconds,
        maximum_observed_gap_seconds,
        time_coverage_bps,
        signed_energy_milli_wh_numerator: energy_numerator,
        energy_denominator: ENERGY_DENOMINATOR_MILLI_WH,
        peak_export_power_milli_w: peak_export,
        peak_import_power_abs_milli_w: peak_import,
        trace_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn sample(id: &'static str, timestamp: i64, power_milli_w: i64) -> PowerTraceSample<'static> {
        PowerTraceSample {
            sample_id: id,
            timestamp_utc_seconds: timestamp,
            power_milli_w,
        }
    }

    fn inputs<'a>(samples: &'a [PowerTraceSample<'a>]) -> PowerTraceInputs<'a> {
        PowerTraceInputs {
            series_id: "M5BAT/POC/2023-04-01",
            source_ref: "m5bat/BESS.csv",
            source_sha256: SOURCE_SHA,
            horizon_start_utc_seconds: 0,
            horizon_end_utc_seconds: 3_600,
            maximum_integrable_gap_seconds: 1,
            samples,
        }
    }

    #[test]
    fn constant_one_kw_export_integrates_to_exactly_one_kwh() {
        let samples = [
            sample("start", 0, 1_000_000),
            sample("middle", 1_800, 1_000_000),
            sample("end", 3_600, 1_000_000),
        ];
        let result = integrate_power_trace(inputs(&samples)).unwrap();
        assert!(result.trace_complete);
        assert_eq!(result.signed_energy_milli_wh_numerator, 7_200_000_000);
        assert_eq!(result.energy_denominator, 7_200);
        assert_eq!(
            result.signed_energy_milli_wh_numerator / result.energy_denominator,
            1_000_000
        );
        assert_eq!(result.peak_export_power_milli_w, 1_000_000);
        assert_eq!(result.peak_import_power_abs_milli_w, 0);
    }

    #[test]
    fn constant_one_kw_import_retains_negative_energy_sign() {
        let samples = [
            sample("start", 0, -1_000_000),
            sample("end", 3_600, -1_000_000),
        ];
        let result = integrate_power_trace(inputs(&samples)).unwrap();
        assert!(result.trace_complete);
        assert_eq!(result.signed_energy_milli_wh_numerator, -7_200_000_000);
        assert_eq!(result.peak_import_power_abs_milli_w, 1_000_000);
        assert_eq!(result.peak_export_power_milli_w, 0);
    }

    #[test]
    fn gaps_are_reported_and_never_forward_filled_or_interpolated() {
        let samples = [
            sample("start", 0, 1_000_000),
            sample("near", 1, 1_000_000),
            sample("after-gap", 5, 1_000_000),
            sample("end", 3_600, 1_000_000),
        ];
        let result = integrate_power_trace(inputs(&samples)).unwrap();
        assert!(!result.trace_complete);
        assert_eq!(result.integrated_interval_count, 1);
        assert_eq!(result.gap_interval_count, 2);
        assert_eq!(result.observed_coverage_seconds, 1);
        assert_eq!(result.uncovered_seconds, 3_599);
        assert_eq!(result.maximum_observed_gap_seconds, 3_595);
        assert_eq!(result.time_coverage_bps, 2);
        assert_eq!(result.signed_energy_milli_wh_numerator, 2_000_000);
    }

    #[test]
    fn incomplete_trace_endpoints_are_not_extrapolated() {
        let samples = [
            sample("middle-a", 10, 1_000_000),
            sample("middle-b", 11, 1_000_000),
        ];
        let result = integrate_power_trace(inputs(&samples)).unwrap();
        assert!(!result.trace_complete);
        assert_eq!(result.observed_coverage_seconds, 1);
        assert_eq!(result.uncovered_seconds, 3_599);
    }

    #[test]
    fn duplicate_ids_timestamps_and_out_of_order_samples_are_rejected() {
        let duplicate_id = [
            sample("same", 0, 1_000_000),
            sample("same", 1, 1_000_000),
        ];
        assert_eq!(
            integrate_power_trace(inputs(&duplicate_id)),
            Err(PowerTraceError::DuplicateSampleId)
        );

        let duplicate_time = [
            sample("a", 0, 1_000_000),
            sample("b", 0, 1_000_000),
        ];
        assert_eq!(
            integrate_power_trace(inputs(&duplicate_time)),
            Err(PowerTraceError::DuplicateTimestamp)
        );

        let out_of_order = [
            sample("a", 1, 1_000_000),
            sample("b", 0, 1_000_000),
        ];
        assert_eq!(
            integrate_power_trace(inputs(&out_of_order)),
            Err(PowerTraceError::SamplesOutOfOrder)
        );
    }

    #[test]
    fn invalid_digest_horizon_and_power_conversion_overflow_fail_closed() {
        assert_eq!(
            kilowatts_to_milliwatts(i64::MAX),
            Err(PowerTraceError::ArithmeticOverflow)
        );
        let samples = [
            sample("start", 0, 1_000_000),
            sample("end", 3_600, 1_000_000),
        ];
        let bad_hash = PowerTraceInputs {
            source_sha256: "not-a-digest",
            ..inputs(&samples)
        };
        assert_eq!(
            integrate_power_trace(bad_hash),
            Err(PowerTraceError::InvalidSourceDigest)
        );
        let bad_horizon = PowerTraceInputs {
            horizon_end_utc_seconds: 0,
            ..inputs(&samples)
        };
        assert_eq!(
            integrate_power_trace(bad_horizon),
            Err(PowerTraceError::InvalidHorizon)
        );
    }
}
