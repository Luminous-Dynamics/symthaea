//! Multi-interval trace-backed schedule execution and settlement reconciliation.
//!
//! Every interval is integrated from raw signed power samples through the exact
//! rational power-trace bridge. Per-interval energy is never rounded before
//! aggregation. Schedule, meter and settlement coverage are independently
//! reported. Missing/gapped trace evidence withholds complete fulfilment and
//! balancing totals. This is a synthetic reference evaluator, not a meter
//! authentication or Market Operator settlement service.

use std::collections::{BTreeMap, BTreeSet};

use crate::execution_fidelity::BalancingSettlementEvidence;
use crate::power_trace_execution::{
    ScheduledPowerTraceAssessment, ScheduledPowerTraceError, ScheduledPowerTraceInputs,
    evaluate_scheduled_power_trace,
};

const BASIS_POINTS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceBackedExecutionInterval<'a> {
    pub interval_id: &'a str,
    pub power_trace: ScheduledPowerTraceInputs<'a>,
    pub settlement: Option<BalancingSettlementEvidence<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceBackedExecutionInputs<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub horizon_start_utc: i64,
    pub horizon_end_utc: i64,
    pub intervals: &'a [TraceBackedExecutionInterval<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceBackedExecutionError {
    InvalidEvidenceReference,
    InvalidDigest,
    InvalidHorizon,
    EmptyTrace,
    InvalidInterval,
    IntervalsOutOfOrder,
    OverlappingIntervals,
    DuplicateIntervalId,
    DuplicateScheduleRecordId,
    DuplicatePowerSampleId,
    DuplicateSettlementRecordId,
    DuplicateSettlementSourceReference,
    SourceDigestConflict,
    SettlementWithoutCompleteTrace,
    UnderlyingTraceError(ScheduledPowerTraceError),
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBackedIntervalReceipt<'a> {
    pub interval_id: &'a str,
    pub schedule_record_id: &'a str,
    pub schedule_source_ref: &'a str,
    pub schedule_source_sha256: &'a str,
    pub meter_source_ref: &'a str,
    pub meter_source_sha256: &'a str,
    pub interval_start_utc_seconds: i64,
    pub interval_end_utc_seconds: i64,
    pub sample_count: u64,
    pub observed_coverage_seconds: u64,
    pub uncovered_seconds: u64,
    pub trace_complete: bool,
    /// Exact signed milli-Wh numerator; divide by energy_denominator.
    pub signed_meter_energy_numerator_milli_wh: i128,
    pub energy_denominator: i128,
    pub settlement_record_id: Option<&'a str>,
    pub settlement_source_ref: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBackedExecutionAssessment<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub interval_count: u64,
    pub complete_trace_interval_count: u64,
    pub incomplete_trace_interval_count: u64,
    pub settlement_interval_count: u64,
    pub missing_settlement_interval_count: u64,
    pub schedule_time_coverage_bps: u64,
    pub meter_time_coverage_bps: u64,
    pub settlement_time_coverage_bps: u64,
    pub schedule_evidence_complete: bool,
    pub meter_evidence_complete: bool,
    pub settlement_evidence_complete: bool,
    /// Exact signed energy sum over a common denominator; partial data may be
    /// included here, so consumers must inspect meter_evidence_complete.
    pub known_meter_energy_numerator_milli_wh: i128,
    pub scheduled_energy_abs_numerator_milli_wh: u128,
    /// Complete-horizon metrics are None unless every required interval is
    /// covered by a complete raw trace.
    pub complete_absolute_deviation_numerator_milli_wh: Option<u128>,
    pub complete_unfulfilled_energy_numerator_milli_wh: Option<u128>,
    pub complete_overdelivery_energy_numerator_milli_wh: Option<u128>,
    pub schedule_fulfilment_bps: Option<u64>,
    /// Known booked settlement amounts; may be partial.
    pub known_balancing_charge_sum_minor_zar: i128,
    /// Complete total is withheld if any interval lacks a booked settlement line.
    pub complete_balancing_charge_minor_zar: Option<i128>,
    pub interval_receipts: Vec<TraceBackedIntervalReceipt<'a>>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn coverage_bps(covered_seconds: u128, horizon_seconds: u128) -> Result<u64, TraceBackedExecutionError> {
    let numerator = covered_seconds
        .checked_mul(BASIS_POINTS)
        .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
    u64::try_from(numerator / horizon_seconds)
        .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)
}

fn add_u128(total: u128, value: u128) -> Result<u128, TraceBackedExecutionError> {
    total.checked_add(value).ok_or(TraceBackedExecutionError::ArithmeticOverflow)
}

fn add_i128(total: i128, value: i128) -> Result<i128, TraceBackedExecutionError> {
    total.checked_add(value).ok_or(TraceBackedExecutionError::ArithmeticOverflow)
}

/// Reconcile a full ordered schedule from raw meter traces. Shared source files
/// may support multiple intervals, but a repeated source ref must retain one
/// digest and sample record IDs must be unique across the entire replay.
pub fn evaluate_trace_backed_execution<'a>(
    inputs: TraceBackedExecutionInputs<'a>,
) -> Result<TraceBackedExecutionAssessment<'a>, TraceBackedExecutionError> {
    if !nonempty(inputs.scenario_id) {
        return Err(TraceBackedExecutionError::InvalidEvidenceReference);
    }
    if inputs.horizon_start_utc >= inputs.horizon_end_utc
        || inputs.evaluation_as_of_utc < inputs.horizon_end_utc
    {
        return Err(TraceBackedExecutionError::InvalidHorizon);
    }
    if inputs.intervals.is_empty() {
        return Err(TraceBackedExecutionError::EmptyTrace);
    }

    let horizon_seconds = u128::try_from(
        i128::from(inputs.horizon_end_utc) - i128::from(inputs.horizon_start_utc),
    )
    .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?;

    let mut interval_ids = BTreeSet::new();
    let mut schedule_record_ids = BTreeSet::new();
    let mut sample_ids = BTreeSet::new();
    let mut settlement_record_ids = BTreeSet::new();
    let mut settlement_source_refs = BTreeSet::new();
    let mut schedule_source_digests = BTreeMap::new();
    let mut meter_source_digests = BTreeMap::new();

    let mut previous_start: Option<i64> = None;
    let mut previous_end = inputs.horizon_start_utc;
    let mut scheduled_seconds = 0_u128;
    let mut meter_seconds = 0_u128;
    let mut settlement_seconds = 0_u128;
    let mut scheduled_energy_abs_numerator = 0_u128;
    let mut known_meter_energy_numerator = 0_i128;
    let mut absolute_deviation_numerator = 0_u128;
    let mut unfulfilled_energy_numerator = 0_u128;
    let mut overdelivery_numerator = 0_u128;
    let mut known_balancing_charge = 0_i128;
    let mut complete_trace_interval_count = 0_u64;
    let mut incomplete_trace_interval_count = 0_u64;
    let mut settlement_interval_count = 0_u64;
    let mut interval_receipts = Vec::with_capacity(inputs.intervals.len());

    for item in inputs.intervals.iter().copied() {
        let trace_inputs = item.power_trace;
        if !nonempty(item.interval_id)
            || !nonempty(trace_inputs.schedule_record_id)
            || !nonempty(trace_inputs.schedule_source_ref)
            || !nonempty(trace_inputs.trace.source_ref)
        {
            return Err(TraceBackedExecutionError::InvalidEvidenceReference);
        }
        if !valid_sha256(trace_inputs.schedule_source_sha256)
            || !valid_sha256(trace_inputs.trace.source_sha256)
        {
            return Err(TraceBackedExecutionError::InvalidDigest);
        }
        if !interval_ids.insert(item.interval_id) {
            return Err(TraceBackedExecutionError::DuplicateIntervalId);
        }
        if !schedule_record_ids.insert(trace_inputs.schedule_record_id) {
            return Err(TraceBackedExecutionError::DuplicateScheduleRecordId);
        }

        if let Some(previous) = schedule_source_digests.insert(
            trace_inputs.schedule_source_ref,
            trace_inputs.schedule_source_sha256,
        ) {
            if previous != trace_inputs.schedule_source_sha256 {
                return Err(TraceBackedExecutionError::SourceDigestConflict);
            }
        }
        if let Some(previous) = meter_source_digests.insert(
            trace_inputs.trace.source_ref,
            trace_inputs.trace.source_sha256,
        ) {
            if previous != trace_inputs.trace.source_sha256 {
                return Err(TraceBackedExecutionError::SourceDigestConflict);
            }
        }

        let start = trace_inputs.interval_start_utc_seconds;
        let end = trace_inputs.interval_end_utc_seconds;
        if start < inputs.horizon_start_utc
            || end > inputs.horizon_end_utc
            || start >= end
        {
            return Err(TraceBackedExecutionError::InvalidInterval);
        }
        if previous_start.is_some_and(|previous| start < previous) {
            return Err(TraceBackedExecutionError::IntervalsOutOfOrder);
        }
        if start < previous_end {
            return Err(TraceBackedExecutionError::OverlappingIntervals);
        }
        previous_start = Some(start);
        previous_end = end;

        for sample in trace_inputs.trace.samples {
            if !sample_ids.insert(sample.sample_id) {
                return Err(TraceBackedExecutionError::DuplicatePowerSampleId);
            }
        }

        let interval_duration = u128::try_from(i128::from(end) - i128::from(start))
            .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?;
        scheduled_seconds = add_u128(scheduled_seconds, interval_duration)?;
        let scheduled_abs = u128::try_from(i128::from(trace_inputs.scheduled_energy_milli_wh).abs())
            .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?;
        let denominator = 7_200_i128;
        scheduled_energy_abs_numerator = add_u128(
            scheduled_energy_abs_numerator,
            scheduled_abs
                .checked_mul(u128::try_from(denominator).unwrap())
                .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
        )?;

        let result = evaluate_scheduled_power_trace(trace_inputs)
            .map_err(TraceBackedExecutionError::UnderlyingTraceError)?;
        if result.energy_denominator != denominator {
            return Err(TraceBackedExecutionError::ArithmeticOverflow);
        }

        meter_seconds = add_u128(
            meter_seconds,
            u128::from(result.trace.observed_coverage_seconds),
        )?;
        known_meter_energy_numerator = add_i128(
            known_meter_energy_numerator,
            result.trace.signed_energy_milli_wh_numerator,
        )?;

        if result.trace.trace_complete {
            complete_trace_interval_count = complete_trace_interval_count
                .checked_add(1)
                .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
            absolute_deviation_numerator = add_u128(
                absolute_deviation_numerator,
                result.absolute_deviation_numerator_milli_wh
                    .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
            )?;
            unfulfilled_energy_numerator = add_u128(
                unfulfilled_energy_numerator,
                result.unfulfilled_scheduled_energy_numerator_milli_wh
                    .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
            )?;
            overdelivery_numerator = add_u128(
                overdelivery_numerator,
                result.overdelivery_energy_numerator_milli_wh
                    .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
            )?;
        } else {
            incomplete_trace_interval_count = incomplete_trace_interval_count
                .checked_add(1)
                .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
        }

        let mut settlement_id = None;
        let mut settlement_ref = None;
        if let Some(settlement) = item.settlement {
            if !result.trace.trace_complete {
                return Err(TraceBackedExecutionError::SettlementWithoutCompleteTrace);
            }
            if !nonempty(settlement.record_id) || !nonempty(settlement.source_ref) {
                return Err(TraceBackedExecutionError::InvalidEvidenceReference);
            }
            if !valid_sha256(settlement.source_sha256) {
                return Err(TraceBackedExecutionError::InvalidDigest);
            }
            if settlement.available_at_utc < trace_inputs.trace_available_at_utc
                || settlement.available_at_utc > inputs.evaluation_as_of_utc
            {
                return Err(TraceBackedExecutionError::UnderlyingTraceError(
                    ScheduledPowerTraceError::TraceNotAvailableAtEvaluation,
                ));
            }
            if !settlement_record_ids.insert(settlement.record_id) {
                return Err(TraceBackedExecutionError::DuplicateSettlementRecordId);
            }
            if !settlement_source_refs.insert(settlement.source_ref) {
                return Err(TraceBackedExecutionError::DuplicateSettlementSourceReference);
            }
            settlement_interval_count = settlement_interval_count
                .checked_add(1)
                .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
            settlement_seconds = add_u128(settlement_seconds, interval_duration)?;
            known_balancing_charge =
                add_i128(known_balancing_charge, i128::from(settlement.balancing_charge_minor_zar))?;
            settlement_id = Some(settlement.record_id);
            settlement_ref = Some(settlement.source_ref);
        }

        interval_receipts.push(TraceBackedIntervalReceipt {
            interval_id: item.interval_id,
            schedule_record_id: trace_inputs.schedule_record_id,
            schedule_source_ref: trace_inputs.schedule_source_ref,
            schedule_source_sha256: trace_inputs.schedule_source_sha256,
            meter_source_ref: trace_inputs.trace.source_ref,
            meter_source_sha256: trace_inputs.trace.source_sha256,
            interval_start_utc_seconds: start,
            interval_end_utc_seconds: end,
            sample_count: result.trace.sample_count,
            observed_coverage_seconds: result.trace.observed_coverage_seconds,
            uncovered_seconds: result.trace.uncovered_seconds,
            trace_complete: result.trace.trace_complete,
            signed_meter_energy_numerator_milli_wh: result.trace.signed_energy_milli_wh_numerator,
            energy_denominator: result.energy_denominator,
            settlement_record_id: settlement_id,
            settlement_source_ref: settlement_ref,
        });
    }

    let schedule_coverage = coverage_bps(scheduled_seconds, horizon_seconds)?;
    let meter_coverage = coverage_bps(meter_seconds, horizon_seconds)?;
    let settlement_coverage = coverage_bps(settlement_seconds, horizon_seconds)?;
    let interval_count = u64::try_from(inputs.intervals.len())
        .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?;
    let missing_settlement_count = interval_count.saturating_sub(settlement_interval_count);
    let schedule_complete = schedule_coverage == 10_000;
    let meter_complete = schedule_complete
        && incomplete_trace_interval_count == 0
        && meter_coverage == 10_000;
    let settlement_complete = meter_complete
        && missing_settlement_count == 0
        && settlement_coverage == 10_000;
    let schedule_fulfilment_bps = if meter_complete && scheduled_energy_abs_numerator > 0 {
        let fulfilled = scheduled_energy_abs_numerator
            .saturating_sub(unfulfilled_energy_numerator);
        let numerator = fulfilled
            .checked_mul(BASIS_POINTS)
            .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
        Some(
            u64::try_from(numerator / scheduled_energy_abs_numerator)
                .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?,
        )
    } else {
        None
    };

    Ok(TraceBackedExecutionAssessment {
        scenario_id: inputs.scenario_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        interval_count,
        complete_trace_interval_count,
        incomplete_trace_interval_count,
        settlement_interval_count,
        missing_settlement_interval_count: missing_settlement_count,
        schedule_time_coverage_bps: schedule_coverage,
        meter_time_coverage_bps: meter_coverage,
        settlement_time_coverage_bps: settlement_coverage,
        schedule_evidence_complete: schedule_complete,
        meter_evidence_complete: meter_complete,
        settlement_evidence_complete: settlement_complete,
        known_meter_energy_numerator_milli_wh: known_meter_energy_numerator,
        scheduled_energy_abs_numerator_milli_wh: scheduled_energy_abs_numerator,
        complete_absolute_deviation_numerator_milli_wh: meter_complete.then_some(absolute_deviation_numerator),
        complete_unfulfilled_energy_numerator_milli_wh: meter_complete.then_some(unfulfilled_energy_numerator),
        complete_overdelivery_energy_numerator_milli_wh: meter_complete.then_some(overdelivery_numerator),
        schedule_fulfilment_bps,
        known_balancing_charge_sum_minor_zar: known_balancing_charge,
        complete_balancing_charge_minor_zar: settlement_complete.then_some(known_balancing_charge),
        interval_receipts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_fidelity::BalancingSettlementEvidence;
    use crate::power_trace::PowerTraceSample;
    use crate::power_trace::PowerTraceInputs;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SCHEDULE_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn sample(id: &'static str, t: i64, power: i64) -> PowerTraceSample<'static> {
        PowerTraceSample { sample_id: id, timestamp_utc_seconds: t, power_milli_w: power }
    }

    fn item<'a>(
        interval_id: &'a str,
        schedule_id: &'a str,
        trace_ref: &'a str,
        samples: &'a [PowerTraceSample<'a>],
        start: i64,
        end: i64,
        power_milli_w_schedule: i64,
        settlement: Option<BalancingSettlementEvidence<'a>>,
    ) -> TraceBackedExecutionInterval<'a> {
        TraceBackedExecutionInterval {
            interval_id,
            power_trace: ScheduledPowerTraceInputs {
                schedule_record_id: schedule_id,
                schedule_source_ref: "market/schedules/day-ahead/v1",
                schedule_source_sha256: SCHEDULE_SHA,
                schedule_issued_at_utc: 10,
                schedule_available_at_utc: 20,
                interval_start_utc_seconds: start,
                interval_end_utc_seconds: end,
                scheduled_energy_milli_wh: power_milli_w_schedule,
                trace_available_at_utc: end + 10,
                evaluation_as_of_utc: 8_000,
                trace: PowerTraceInputs {
                    series_id: "SAWEM/POC/TP",
                    source_ref: trace_ref,
                    source_sha256: SOURCE_SHA,
                    horizon_start_utc_seconds: start,
                    horizon_end_utc_seconds: end,
                    maximum_integrable_gap_seconds: end - start,
                    samples,
                },
            },
            settlement,
        }
    }

    fn settlement(id: &'static str, source: &'static str, charge: i64, available: i64) -> BalancingSettlementEvidence<'static> {
        BalancingSettlementEvidence {
            record_id: id,
            source_ref: source,
            source_sha256: SOURCE_SHA,
            available_at_utc: available,
            balancing_charge_minor_zar: charge,
        }
    }

    #[test]
    fn two_interval_raw_trace_replay_aggregates_exact_energy_and_booked_settlement() {
        let samples_a = [sample("a-start", 100, 1_000_000), sample("a-end", 3_700, 1_000_000)];
        let samples_b = [sample("b-start", 3_700, 1_000_000), sample("b-end", 7_300, 1_000_000)];
        let intervals = [
            item("tp-a", "schedule-a", "meter/day-file", &samples_a, 100, 3_700, 900_000,
                Some(settlement("settle-a", "settlement/a", 10, 3_720))),
            item("tp-b", "schedule-b", "meter/day-file", &samples_b, 3_700, 7_300, 1_100_000,
                Some(settlement("settle-b", "settlement/b", 20, 7_320))),
        ];
        let result = evaluate_trace_backed_execution(TraceBackedExecutionInputs {
            scenario_id: "golden-pv-bess/trace-backed",
            evaluation_as_of_utc: 8_000,
            horizon_start_utc: 100,
            horizon_end_utc: 7_300,
            intervals: &intervals,
        })
        .unwrap();

        assert!(result.schedule_evidence_complete);
        assert!(result.meter_evidence_complete);
        assert!(result.settlement_evidence_complete);
        assert_eq!(result.interval_count, 2);
        assert_eq!(result.schedule_time_coverage_bps, 10_000);
        assert_eq!(result.meter_time_coverage_bps, 10_000);
        assert_eq!(result.settlement_time_coverage_bps, 10_000);
        assert_eq!(result.known_meter_energy_numerator_milli_wh, 14_400_000_000);
        assert_eq!(result.scheduled_energy_abs_numerator_milli_wh, 14_400_000_000);
        assert_eq!(result.complete_unfulfilled_energy_numerator_milli_wh, Some(720_000_000));
        assert_eq!(result.complete_overdelivery_energy_numerator_milli_wh, Some(720_000_000));
        assert_eq!(result.schedule_fulfilment_bps, Some(9_500));
        assert_eq!(result.complete_balancing_charge_minor_zar, Some(30));
    }

    #[test]
    fn incomplete_raw_trace_withholds_complete_fulfilment_and_settlement_total() {
        let samples_a = [sample("a-start", 100, 1_000_000), sample("a-end", 3_700, 1_000_000)];
        let samples_b = [sample("b-middle", 4_000, 1_000_000), sample("b-end", 7_300, 1_000_000)];
        let intervals = [
            item("tp-a", "schedule-a", "meter/day-file", &samples_a, 100, 3_700, 900_000,
                Some(settlement("settle-a", "settlement/a", 10, 3_720))),
            item("tp-b", "schedule-b", "meter/day-file", &samples_b, 3_700, 7_300, 1_100_000, None),
        ];
        let result = evaluate_trace_backed_execution(TraceBackedExecutionInputs {
            scenario_id: "golden/incomplete",
            evaluation_as_of_utc: 8_000,
            horizon_start_utc: 100,
            horizon_end_utc: 7_300,
            intervals: &intervals,
        })
        .unwrap();
        assert!(!result.meter_evidence_complete);
        assert!(!result.settlement_evidence_complete);
        assert_eq!(result.incomplete_trace_interval_count, 1);
        assert_eq!(result.schedule_fulfilment_bps, None);
        assert_eq!(result.complete_absolute_deviation_numerator_milli_wh, None);
        assert_eq!(result.complete_balancing_charge_minor_zar, None);
        assert_eq!(result.known_balancing_charge_sum_minor_zar, 10);
    }

    #[test]
    fn schedule_gap_reduces_coverage_and_withholds_complete_result() {
        let samples_a = [sample("a-start", 100, 1_000_000), sample("a-end", 3_700, 1_000_000)];
        let samples_b = [sample("b-start", 4_000, 1_000_000), sample("b-end", 7_300, 1_000_000)];
        let intervals = [
            item("tp-a", "schedule-a", "meter/day-file", &samples_a, 100, 3_700, 900_000, None),
            item("tp-b", "schedule-b", "meter/day-file", &samples_b, 4_000, 7_300, 1_100_000, None),
        ];
        let result = evaluate_trace_backed_execution(TraceBackedExecutionInputs {
            scenario_id: "golden/schedule-gap",
            evaluation_as_of_utc: 8_000,
            horizon_start_utc: 100,
            horizon_end_utc: 7_300,
            intervals: &intervals,
        })
        .unwrap();
        assert_eq!(result.schedule_time_coverage_bps, 9_166);
        assert!(!result.schedule_evidence_complete);
        assert_eq!(result.schedule_fulfilment_bps, None);
    }

    #[test]
    fn reused_power_sample_ids_and_conflicting_source_digests_fail_closed() {
        let samples_a = [sample("shared-start", 100, 1_000_000), sample("a-end", 3_700, 1_000_000)];
        let samples_b = [sample("shared-start", 3_700, 1_000_000), sample("b-end", 7_300, 1_000_000)];
        let intervals = [
            item("tp-a", "schedule-a", "meter/day-file", &samples_a, 100, 3_700, 900_000, None),
            item("tp-b", "schedule-b", "meter/day-file", &samples_b, 3_700, 7_300, 1_100_000, None),
        ];
        assert_eq!(
            evaluate_trace_backed_execution(TraceBackedExecutionInputs {
                scenario_id: "golden/duplicate-sample",
                evaluation_as_of_utc: 8_000,
                horizon_start_utc: 100,
                horizon_end_utc: 7_300,
                intervals: &intervals,
            }),
            Err(TraceBackedExecutionError::DuplicatePowerSampleId)
        );

        let conflicting = [
            intervals[0],
            TraceBackedExecutionInterval {
                power_trace: ScheduledPowerTraceInputs {
                    trace: PowerTraceInputs {
                        source_sha256: "bad",
                        ..intervals[1].power_trace.trace
                    },
                    ..intervals[1].power_trace
                },
                ..intervals[1]
            },
        ];
        assert_eq!(
            evaluate_trace_backed_execution(TraceBackedExecutionInputs {
                scenario_id: "golden/digest-conflict",
                evaluation_as_of_utc: 8_000,
                horizon_start_utc: 100,
                horizon_end_utc: 7_300,
                intervals: &conflicting,
            }),
            Err(TraceBackedExecutionError::InvalidDigest)
        );
    }

    #[test]
    fn duplicate_settlement_receipts_and_settlement_without_complete_trace_are_rejected() {
        let samples_a = [sample("a-start", 100, 1_000_000), sample("a-end", 3_700, 1_000_000)];
        let samples_b = [sample("b-start", 3_700, 1_000_000), sample("b-end", 7_300, 1_000_000)];
        let duplicate = [
            item("tp-a", "schedule-a", "meter/day-file", &samples_a, 100, 3_700, 900_000,
                Some(settlement("settle-a", "settlement/same", 10, 3_720))),
            item("tp-b", "schedule-b", "meter/day-file", &samples_b, 3_700, 7_300, 1_100_000,
                Some(settlement("settle-b", "settlement/same", 20, 7_320))),
        ];
        assert_eq!(
            evaluate_trace_backed_execution(TraceBackedExecutionInputs {
                scenario_id: "golden/duplicate-settlement",
                evaluation_as_of_utc: 8_000,
                horizon_start_utc: 100,
                horizon_end_utc: 7_300,
                intervals: &duplicate,
            }),
            Err(TraceBackedExecutionError::DuplicateSettlementSourceReference)
        );

        let incomplete = [sample("b-mid", 4_000, 1_000_000), sample("b-end", 7_300, 1_000_000)];
        let intervals = [
            item("tp-b", "schedule-b", "meter/day-file", &incomplete, 3_700, 7_300, 1_100_000,
                Some(settlement("settle-b", "settlement/b", 20, 7_320))),
        ];
        assert_eq!(
            evaluate_trace_backed_execution(TraceBackedExecutionInputs {
                scenario_id: "golden/settle-incomplete",
                evaluation_as_of_utc: 8_000,
                horizon_start_utc: 3_700,
                horizon_end_utc: 7_300,
                intervals: &intervals,
            }),
            Err(TraceBackedExecutionError::SettlementWithoutCompleteTrace)
        );
    }
}
