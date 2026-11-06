//! Exact one-interval bridge from sampled signed power to an optimized schedule.
//!
//! The raw power trace retains an exact milli-Wh numerator over denominator
//! 7,200. This module compares that rational energy with the schedule without
//! rounding the trace to integer milli-Wh. Fulfilment is withheld unless the
//! trace covers both interval endpoints and every adjacent sample gap passes
//! the declared integrability limit. Multi-interval booked-settlement
//! reconciliation remains in execution_fidelity.rs.

use crate::power_trace::{
    PowerTraceAssessment, PowerTraceError, PowerTraceInputs, integrate_power_trace,
};

const BASIS_POINTS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledPowerTraceInputs<'a> {
    pub schedule_record_id: &'a str,
    pub schedule_source_ref: &'a str,
    pub schedule_source_sha256: &'a str,
    pub schedule_issued_at_utc: i64,
    pub schedule_available_at_utc: i64,
    pub interval_start_utc_seconds: i64,
    pub interval_end_utc_seconds: i64,
    /// Signed scheduled grid energy in integer milli-Wh.
    pub scheduled_energy_milli_wh: i64,
    /// When the complete meter trace became available to the evaluation process.
    pub trace_available_at_utc: i64,
    pub evaluation_as_of_utc: i64,
    pub trace: PowerTraceInputs<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledPowerTraceError {
    InvalidEvidenceReference,
    InvalidScheduleDigest,
    InvalidScheduleChronology,
    EvaluationBeforeDeliveryEnd,
    TraceHorizonDoesNotMatchSchedule,
    TraceNotAvailableAtEvaluation,
    UnderlyingPowerTrace(PowerTraceError),
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledPowerTraceAssessment<'a> {
    pub schedule_record_id: &'a str,
    pub schedule_source_ref: &'a str,
    pub schedule_source_sha256: &'a str,
    pub interval_start_utc_seconds: i64,
    pub interval_end_utc_seconds: i64,
    pub scheduled_energy_milli_wh: i64,
    pub scheduled_energy_numerator_milli_wh: i128,
    pub energy_denominator: i128,
    pub trace: PowerTraceAssessment<'a>,
    /// Exact absolute deviation numerator over energy_denominator.
    /// None means the trace does not cover the complete schedule interval.
    pub absolute_deviation_numerator_milli_wh: Option<u128>,
    /// Exact directional under-delivery numerator over energy_denominator.
    pub unfulfilled_scheduled_energy_numerator_milli_wh: Option<u128>,
    /// Exact same-direction over-delivery numerator over energy_denominator.
    pub overdelivery_energy_numerator_milli_wh: Option<u128>,
    /// Only populated for a complete trace; over-delivery does not reduce
    /// fulfilment below 100% but remains separately reported.
    pub schedule_fulfilment_bps: Option<u64>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Compare one complete sampled trace with its schedule using exact rational
/// arithmetic. Positive energy is export; negative energy is import.
pub fn evaluate_scheduled_power_trace<'a>(
    inputs: ScheduledPowerTraceInputs<'a>,
) -> Result<ScheduledPowerTraceAssessment<'a>, ScheduledPowerTraceError> {
    if !nonempty(inputs.schedule_record_id) || !nonempty(inputs.schedule_source_ref) {
        return Err(ScheduledPowerTraceError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.schedule_source_sha256) {
        return Err(ScheduledPowerTraceError::InvalidScheduleDigest);
    }
    if inputs.interval_start_utc_seconds >= inputs.interval_end_utc_seconds
        || inputs.schedule_issued_at_utc > inputs.schedule_available_at_utc
        || inputs.schedule_available_at_utc > inputs.interval_start_utc_seconds
    {
        return Err(ScheduledPowerTraceError::InvalidScheduleChronology);
    }
    if inputs.evaluation_as_of_utc < inputs.interval_end_utc_seconds {
        return Err(ScheduledPowerTraceError::EvaluationBeforeDeliveryEnd);
    }
    if inputs.trace.horizon_start_utc_seconds != inputs.interval_start_utc_seconds
        || inputs.trace.horizon_end_utc_seconds != inputs.interval_end_utc_seconds
    {
        return Err(ScheduledPowerTraceError::TraceHorizonDoesNotMatchSchedule);
    }
    if inputs.trace_available_at_utc < inputs.interval_end_utc_seconds
        || inputs.trace_available_at_utc > inputs.evaluation_as_of_utc
    {
        return Err(ScheduledPowerTraceError::TraceNotAvailableAtEvaluation);
    }

    let trace = integrate_power_trace(inputs.trace)
        .map_err(ScheduledPowerTraceError::UnderlyingPowerTrace)?;
    let denominator = trace.energy_denominator;
    let scheduled_numerator = i128::from(inputs.scheduled_energy_milli_wh)
        .checked_mul(denominator)
        .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?;

    let (
        absolute_deviation,
        unfulfilled,
        overdelivery,
        fulfilment,
    ) = if trace.trace_complete {
        let measured = trace.signed_energy_milli_wh_numerator;
        let absolute_deviation = u128::try_from(
            measured
                .checked_sub(scheduled_numerator)
                .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?
                .checked_abs()
                .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?,
        )
        .map_err(|_| ScheduledPowerTraceError::ArithmeticOverflow)?;

        let scheduled_directional = scheduled_numerator
            .checked_abs()
            .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?;
        let metered_directional = if scheduled_numerator > 0 {
            measured.max(0)
        } else if scheduled_numerator < 0 {
            measured
                .checked_neg()
                .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?
                .max(0)
        } else {
            0
        };
        let unfulfilled_signed = scheduled_directional
            .checked_sub(metered_directional)
            .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?;
        let overdelivery_signed = metered_directional
            .checked_sub(scheduled_directional)
            .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?;
        let unfulfilled = u128::try_from(unfulfilled_signed.max(0))
            .map_err(|_| ScheduledPowerTraceError::ArithmeticOverflow)?;
        let overdelivery = u128::try_from(overdelivery_signed.max(0))
            .map_err(|_| ScheduledPowerTraceError::ArithmeticOverflow)?;

        let fulfilment = if scheduled_directional == 0 {
            Some(10_000)
        } else {
            let delivered = scheduled_directional
                .checked_sub(i128::try_from(unfulfilled).map_err(|_| {
                    ScheduledPowerTraceError::ArithmeticOverflow
                })?)
                .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?;
            let numerator = u128::try_from(delivered)
                .map_err(|_| ScheduledPowerTraceError::ArithmeticOverflow)?
                .checked_mul(BASIS_POINTS)
                .ok_or(ScheduledPowerTraceError::ArithmeticOverflow)?;
            Some(
                u64::try_from(
                    numerator
                        / u128::try_from(scheduled_directional)
                            .map_err(|_| ScheduledPowerTraceError::ArithmeticOverflow)?,
                )
                .map_err(|_| ScheduledPowerTraceError::ArithmeticOverflow)?,
            )
        };
        (
            Some(absolute_deviation),
            Some(unfulfilled),
            Some(overdelivery),
            fulfilment,
        )
    } else {
        (None, None, None, None)
    };

    Ok(ScheduledPowerTraceAssessment {
        schedule_record_id: inputs.schedule_record_id,
        schedule_source_ref: inputs.schedule_source_ref,
        schedule_source_sha256: inputs.schedule_source_sha256,
        interval_start_utc_seconds: inputs.interval_start_utc_seconds,
        interval_end_utc_seconds: inputs.interval_end_utc_seconds,
        scheduled_energy_milli_wh: inputs.scheduled_energy_milli_wh,
        scheduled_energy_numerator_milli_wh: scheduled_numerator,
        energy_denominator: denominator,
        trace,
        absolute_deviation_numerator_milli_wh: absolute_deviation,
        unfulfilled_scheduled_energy_numerator_milli_wh: unfulfilled,
        overdelivery_energy_numerator_milli_wh: overdelivery,
        schedule_fulfilment_bps: fulfilment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::power_trace::PowerTraceSample;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SCHEDULE_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn sample(id: &'static str, timestamp: i64, power_milli_w: i64) -> PowerTraceSample<'static> {
        PowerTraceSample {
            sample_id: id,
            timestamp_utc_seconds: timestamp,
            power_milli_w,
        }
    }

    fn inputs<'a>(
        samples: &'a [PowerTraceSample<'a>],
        scheduled_mwh: i64,
    ) -> ScheduledPowerTraceInputs<'a> {
        ScheduledPowerTraceInputs {
            schedule_record_id: "schedule/001",
            schedule_source_ref: "market/schedule/001",
            schedule_source_sha256: SCHEDULE_SHA,
            schedule_issued_at_utc: 90,
            schedule_available_at_utc: 95,
            interval_start_utc_seconds: 100,
            interval_end_utc_seconds: 3_700,
            scheduled_energy_milli_wh: scheduled_mwh,
            trace_available_at_utc: 3_710,
            evaluation_as_of_utc: 4_000,
            trace: PowerTraceInputs {
                series_id: "SAWEM/POC/TP-001",
                source_ref: "meter/trace/001",
                source_sha256: SOURCE_SHA,
                horizon_start_utc_seconds: 100,
                horizon_end_utc_seconds: 3_700,
                maximum_integrable_gap_seconds: 1_800,
                samples,
            },
        }
    }

    #[test]
    fn exact_power_trace_reconciles_export_schedule_without_rounding_energy() {
        let samples = [
            sample("start", 100, 1_000_000),
            sample("end", 3_700, 1_000_000),
        ];
        let result = evaluate_scheduled_power_trace(inputs(&samples, 900_000)).unwrap();
        assert_eq!(result.trace.signed_energy_milli_wh_numerator, 7_200_000_000);
        assert_eq!(result.energy_denominator, 7_200);
        assert_eq!(
            result.scheduled_energy_numerator_milli_wh,
            6_480_000_000
        );
        assert_eq!(
            result.absolute_deviation_numerator_milli_wh,
            Some(720_000_000)
        );
        assert_eq!(
            result.unfulfilled_scheduled_energy_numerator_milli_wh,
            Some(0)
        );
        assert_eq!(
            result.overdelivery_energy_numerator_milli_wh,
            Some(720_000_000)
        );
        assert_eq!(result.schedule_fulfilment_bps, Some(10_000));
    }

    #[test]
    fn underdelivery_is_reported_in_exact_rational_units() {
        let samples = [
            sample("start", 100, 800_000),
            sample("end", 3_700, 800_000),
        ];
        let result = evaluate_scheduled_power_trace(inputs(&samples, 1_000_000)).unwrap();
        assert_eq!(
            result.unfulfilled_scheduled_energy_numerator_milli_wh,
            Some(1_440_000_000)
        );
        assert_eq!(result.overdelivery_energy_numerator_milli_wh, Some(0));
        assert_eq!(result.schedule_fulfilment_bps, Some(8_000));
    }

    #[test]
    fn import_schedule_uses_the_same_signed_directional_reconciliation() {
        let samples = [
            sample("start", 100, -800_000),
            sample("end", 3_700, -800_000),
        ];
        let result = evaluate_scheduled_power_trace(inputs(&samples, -1_000_000)).unwrap();
        assert_eq!(
            result.unfulfilled_scheduled_energy_numerator_milli_wh,
            Some(1_440_000_000)
        );
        assert_eq!(result.schedule_fulfilment_bps, Some(8_000));
    }

    #[test]
    fn opposite_direction_energy_cannot_satisfy_the_schedule() {
        let samples = [
            sample("start", 100, -500_000),
            sample("end", 3_700, -500_000),
        ];
        let result = evaluate_scheduled_power_trace(inputs(&samples, 1_000_000)).unwrap();
        assert_eq!(result.unfulfilled_scheduled_energy_numerator_milli_wh, Some(7_200_000_000));
        assert_eq!(result.overdelivery_energy_numerator_milli_wh, Some(0));
        assert_eq!(result.schedule_fulfilment_bps, Some(0));
    }

    #[test]
    fn incomplete_trace_returns_partial_energy_but_withholds_fulfilment() {
        let samples = [
            sample("interior-a", 1_000, 1_000_000),
            sample("interior-b", 1_001, 1_000_000),
        ];
        let result = evaluate_scheduled_power_trace(inputs(&samples, 1_000_000)).unwrap();
        assert!(!result.trace.trace_complete);
        assert!(result.trace.signed_energy_milli_wh_numerator > 0);
        assert_eq!(result.absolute_deviation_numerator_milli_wh, None);
        assert_eq!(result.unfulfilled_scheduled_energy_numerator_milli_wh, None);
        assert_eq!(result.overdelivery_energy_numerator_milli_wh, None);
        assert_eq!(result.schedule_fulfilment_bps, None);
    }

    #[test]
    fn late_schedule_wrong_trace_horizon_and_future_trace_are_rejected() {
        let samples = [
            sample("start", 100, 1_000_000),
            sample("end", 3_700, 1_000_000),
        ];
        assert_eq!(
            evaluate_scheduled_power_trace(ScheduledPowerTraceInputs {
                schedule_available_at_utc: 101,
                ..inputs(&samples, 1_000_000)
            }),
            Err(ScheduledPowerTraceError::InvalidScheduleChronology)
        );
        assert_eq!(
            evaluate_scheduled_power_trace(ScheduledPowerTraceInputs {
                interval_start_utc_seconds: 101,
                ..inputs(&samples, 1_000_000)
            }),
            Err(ScheduledPowerTraceError::TraceHorizonDoesNotMatchSchedule)
        );
        assert_eq!(
            evaluate_scheduled_power_trace(ScheduledPowerTraceInputs {
                trace_available_at_utc: 4_001,
                ..inputs(&samples, 1_000_000)
            }),
            Err(ScheduledPowerTraceError::TraceNotAvailableAtEvaluation)
        );
    }
}
