//! Evidence-bound comparison of scheduled grid energy with metered execution.
//!
//! Positive signed energy means net export; negative means net import.
//! Schedule and meter quantities are milli-Wh. A complete fulfilment ratio is
//! returned only when the schedule covers the entire horizon and every
//! scheduled interval has meter evidence. Booked balancing charges are summed
//! only when a settlement receipt exists for every scheduled interval.
//!
//! This is an execution-reconciliation kernel, not a dispatch optimizer,
//! meter-authentication service, or Market Operator settlement calculator.

use crate::power_trace_execution::{
    ScheduledPowerTraceError, ScheduledPowerTraceInputs, evaluate_scheduled_power_trace,
};
use std::collections::{BTreeMap, BTreeSet};

const BASIS_POINTS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeteredEnergyEvidence<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub observed_at_utc: i64,
    pub available_at_utc: i64,
    /// Signed grid energy: positive export, negative import.
    pub metered_energy_milli_wh: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BalancingSettlementEvidence<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub available_at_utc: i64,
    /// Positive is a booked charge; negative is a booked credit.
    pub balancing_charge_minor_zar: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionInterval<'a> {
    pub interval_id: &'a str,
    pub schedule_source_ref: &'a str,
    pub schedule_source_sha256: &'a str,
    pub schedule_issued_at_utc: i64,
    pub schedule_available_at_utc: i64,
    pub interval_start_utc: i64,
    pub interval_end_utc: i64,
    /// Signed grid energy: positive export, negative import.
    pub scheduled_energy_milli_wh: i64,
    pub meter: Option<MeteredEnergyEvidence<'a>>,
    pub settlement: Option<BalancingSettlementEvidence<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionFidelityInputs<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub horizon_start_utc: i64,
    pub horizon_end_utc: i64,
    pub intervals: &'a [ExecutionInterval<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionFidelityError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidHorizon,
    InvalidInterval,
    ScheduleWasNotAvailableAtDelivery,
    MeterObservedBeforeDeliveryEnded,
    MeterNotAvailableAtEvaluation,
    SettlementNotAvailableAtEvaluation,
    SettlementWithoutMeterEvidence,
    IntervalsOutOfOrder,
    OverlappingIntervals,
    DuplicateIntervalId,
    DuplicateMeterRecordId,
    DuplicateMeterSourceReference,
    DuplicateSettlementRecordId,
    DuplicateSettlementSourceReference,
    ArithmeticOverflow,
    EmptyExecutionTrace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFidelityAssessment<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub interval_count: u64,
    pub metered_interval_count: u64,
    pub missing_meter_interval_count: u64,
    pub settlement_interval_count: u64,
    pub missing_settlement_interval_count: u64,
    pub schedule_time_coverage_bps: u64,
    pub meter_time_coverage_bps: u64,
    pub settlement_time_coverage_bps: u64,
    pub trace_complete: bool,
    pub scheduled_energy_abs_milli_wh: u128,
    pub metered_energy_signed_sum_milli_wh: i128,
    pub absolute_energy_deviation_milli_wh: u128,
    pub unfulfilled_scheduled_energy_milli_wh: u128,
    pub overdelivery_energy_milli_wh: u128,
    /// Only populated when scheduled intervals fully cover the horizon and
    /// meter evidence exists for every interval.
    pub schedule_fulfilment_bps: Option<u64>,
    /// Known booked amounts only; this is not promoted to a complete total
    /// when one or more settlement intervals are missing.
    pub known_balancing_charge_sum_minor_zar: i128,
    pub complete_balancing_charge_minor_zar: Option<i128>,
    /// Source refs from the selected meter and settlement records are kept in
    /// the source records; these counters expose how much is still missing.
    pub meter_evidence_complete: bool,
    pub settlement_evidence_complete: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn abs_i64(value: i64) -> Result<u128, ExecutionFidelityError> {
    u128::try_from(i128::from(value).abs()).map_err(|_| ExecutionFidelityError::ArithmeticOverflow)
}

fn checked_sum_u128(current: u128, add: u128) -> Result<u128, ExecutionFidelityError> {
    current
        .checked_add(add)
        .ok_or(ExecutionFidelityError::ArithmeticOverflow)
}

fn checked_sum_i128(current: i128, add: i128) -> Result<i128, ExecutionFidelityError> {
    current
        .checked_add(add)
        .ok_or(ExecutionFidelityError::ArithmeticOverflow)
}

fn signed_directional_energy(
    scheduled: i64,
    metered: i64,
) -> Result<(u128, u128, u128), ExecutionFidelityError> {
    let scheduled_signed = i128::from(scheduled);
    let metered_signed = i128::from(metered);
    let deviation = u128::try_from((metered_signed - scheduled_signed).abs())
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?;
    let scheduled_abs = abs_i64(scheduled)?;
    let directional_metered = if scheduled > 0 {
        metered_signed.max(0)
    } else if scheduled < 0 {
        (-metered_signed).max(0)
    } else {
        0
    };
    let unfulfilled = (i128::try_from(scheduled_abs)
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?
        - directional_metered)
        .max(0);
    let overdelivery = (directional_metered
        - i128::try_from(scheduled_abs).map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?)
    .max(0);
    Ok((
        deviation,
        u128::try_from(unfulfilled).map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?,
        u128::try_from(overdelivery).map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?,
    ))
}

fn coverage_bps(
    covered_seconds: u128,
    horizon_seconds: u128,
) -> Result<u64, ExecutionFidelityError> {
    let numerator = covered_seconds
        .checked_mul(BASIS_POINTS)
        .ok_or(ExecutionFidelityError::ArithmeticOverflow)?;
    u64::try_from(numerator / horizon_seconds)
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)
}

/// Reconcile the schedule, meter and booked settlement evidence over one
/// frozen horizon. Gaps are reported as incomplete time coverage; overlaps,
/// duplicated records and future-known evidence are rejected.
pub fn evaluate_execution_fidelity<'a>(
    inputs: ExecutionFidelityInputs<'a>,
) -> Result<ExecutionFidelityAssessment<'a>, ExecutionFidelityError> {
    if !nonempty(inputs.scenario_id) {
        return Err(ExecutionFidelityError::InvalidEvidenceReference);
    }
    if inputs.horizon_start_utc >= inputs.horizon_end_utc
        || inputs.evaluation_as_of_utc < inputs.horizon_end_utc
    {
        return Err(ExecutionFidelityError::InvalidHorizon);
    }
    if inputs.intervals.is_empty() {
        return Err(ExecutionFidelityError::EmptyExecutionTrace);
    }

    let horizon_seconds =
        u128::try_from(i128::from(inputs.horizon_end_utc) - i128::from(inputs.horizon_start_utc))
            .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?;
    let mut previous_interval_start: Option<i64> = None;
    let mut previous_interval_end = inputs.horizon_start_utc;
    let mut scheduled_seconds = 0_u128;
    let mut meter_seconds = 0_u128;
    let mut settlement_seconds = 0_u128;
    let mut scheduled_energy_abs = 0_u128;
    let mut metered_energy_signed_sum = 0_i128;
    let mut absolute_deviation = 0_u128;
    let mut unfulfilled_energy = 0_u128;
    let mut overdelivery_energy = 0_u128;
    let mut known_balancing_charge = 0_i128;
    let mut interval_ids = BTreeSet::new();
    let mut meter_record_ids = BTreeSet::new();
    let mut meter_source_refs = BTreeSet::new();
    let mut settlement_record_ids = BTreeSet::new();
    let mut settlement_source_refs = BTreeSet::new();
    let mut metered_interval_count = 0_u64;
    let mut settlement_interval_count = 0_u64;

    for interval in inputs.intervals.iter().copied() {
        if !nonempty(interval.interval_id) || !nonempty(interval.schedule_source_ref) {
            return Err(ExecutionFidelityError::InvalidEvidenceReference);
        }
        if !valid_sha256(interval.schedule_source_sha256) {
            return Err(ExecutionFidelityError::InvalidSourceDigest);
        }
        if !interval_ids.insert(interval.interval_id) {
            return Err(ExecutionFidelityError::DuplicateIntervalId);
        }
        if interval.interval_start_utc < inputs.horizon_start_utc
            || interval.interval_end_utc > inputs.horizon_end_utc
            || interval.interval_start_utc >= interval.interval_end_utc
        {
            return Err(ExecutionFidelityError::InvalidInterval);
        }
        if interval.schedule_issued_at_utc > interval.schedule_available_at_utc
            || interval.schedule_available_at_utc > interval.interval_start_utc
        {
            return Err(ExecutionFidelityError::ScheduleWasNotAvailableAtDelivery);
        }
        if previous_interval_start.is_some_and(|previous| interval.interval_start_utc < previous) {
            return Err(ExecutionFidelityError::IntervalsOutOfOrder);
        }
        if interval.interval_start_utc < previous_interval_end {
            return Err(ExecutionFidelityError::OverlappingIntervals);
        }

        let duration = u128::try_from(
            i128::from(interval.interval_end_utc) - i128::from(interval.interval_start_utc),
        )
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?;
        previous_interval_start = Some(interval.interval_start_utc);
        scheduled_seconds = checked_sum_u128(scheduled_seconds, duration)?;
        scheduled_energy_abs = checked_sum_u128(
            scheduled_energy_abs,
            abs_i64(interval.scheduled_energy_milli_wh)?,
        )?;
        previous_interval_end = interval.interval_end_utc;

        match (interval.meter, interval.settlement) {
            (None, Some(_)) => {
                return Err(ExecutionFidelityError::SettlementWithoutMeterEvidence);
            }
            (None, None) => {}
            (Some(meter), settlement) => {
                if !nonempty(meter.record_id) || !nonempty(meter.source_ref) {
                    return Err(ExecutionFidelityError::InvalidEvidenceReference);
                }
                if !valid_sha256(meter.source_sha256) {
                    return Err(ExecutionFidelityError::InvalidSourceDigest);
                }
                if meter.observed_at_utc < interval.interval_end_utc {
                    return Err(ExecutionFidelityError::MeterObservedBeforeDeliveryEnded);
                }
                if meter.available_at_utc < meter.observed_at_utc
                    || meter.available_at_utc > inputs.evaluation_as_of_utc
                {
                    return Err(ExecutionFidelityError::MeterNotAvailableAtEvaluation);
                }
                if !meter_record_ids.insert(meter.record_id) {
                    return Err(ExecutionFidelityError::DuplicateMeterRecordId);
                }
                if !meter_source_refs.insert(meter.source_ref) {
                    return Err(ExecutionFidelityError::DuplicateMeterSourceReference);
                }

                metered_interval_count = metered_interval_count
                    .checked_add(1)
                    .ok_or(ExecutionFidelityError::ArithmeticOverflow)?;
                meter_seconds = checked_sum_u128(meter_seconds, duration)?;
                metered_energy_signed_sum = checked_sum_i128(
                    metered_energy_signed_sum,
                    i128::from(meter.metered_energy_milli_wh),
                )?;
                let (deviation, unfulfilled, overdelivery) = signed_directional_energy(
                    interval.scheduled_energy_milli_wh,
                    meter.metered_energy_milli_wh,
                )?;
                absolute_deviation = checked_sum_u128(absolute_deviation, deviation)?;
                unfulfilled_energy = checked_sum_u128(unfulfilled_energy, unfulfilled)?;
                overdelivery_energy = checked_sum_u128(overdelivery_energy, overdelivery)?;

                if let Some(settlement) = settlement {
                    if !nonempty(settlement.record_id) || !nonempty(settlement.source_ref) {
                        return Err(ExecutionFidelityError::InvalidEvidenceReference);
                    }
                    if !valid_sha256(settlement.source_sha256) {
                        return Err(ExecutionFidelityError::InvalidSourceDigest);
                    }
                    if settlement.available_at_utc < meter.available_at_utc
                        || settlement.available_at_utc > inputs.evaluation_as_of_utc
                    {
                        return Err(ExecutionFidelityError::SettlementNotAvailableAtEvaluation);
                    }
                    if !settlement_record_ids.insert(settlement.record_id) {
                        return Err(ExecutionFidelityError::DuplicateSettlementRecordId);
                    }
                    if !settlement_source_refs.insert(settlement.source_ref) {
                        return Err(ExecutionFidelityError::DuplicateSettlementSourceReference);
                    }
                    settlement_interval_count = settlement_interval_count
                        .checked_add(1)
                        .ok_or(ExecutionFidelityError::ArithmeticOverflow)?;
                    settlement_seconds = checked_sum_u128(settlement_seconds, duration)?;
                    known_balancing_charge = checked_sum_i128(
                        known_balancing_charge,
                        i128::from(settlement.balancing_charge_minor_zar),
                    )?;
                }
            }
        }
    }

    let schedule_coverage = coverage_bps(scheduled_seconds, horizon_seconds)?;
    let meter_coverage = coverage_bps(meter_seconds, horizon_seconds)?;
    let settlement_coverage = coverage_bps(settlement_seconds, horizon_seconds)?;
    let interval_count = u64::try_from(inputs.intervals.len())
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?;
    let missing_meter_count = interval_count - metered_interval_count;
    let missing_settlement_count = interval_count - settlement_interval_count;
    let schedule_coverage_complete = schedule_coverage == 10_000;
    let meter_evidence_complete =
        schedule_coverage_complete && missing_meter_count == 0 && meter_coverage == 10_000;
    let settlement_evidence_complete =
        meter_evidence_complete && missing_settlement_count == 0 && settlement_coverage == 10_000;

    let schedule_fulfilment_bps = if meter_evidence_complete && scheduled_energy_abs > 0 {
        let fulfilled_energy = scheduled_energy_abs.saturating_sub(unfulfilled_energy);
        let numerator = fulfilled_energy
            .checked_mul(BASIS_POINTS)
            .ok_or(ExecutionFidelityError::ArithmeticOverflow)?;
        Some(
            u64::try_from(numerator / scheduled_energy_abs)
                .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?,
        )
    } else {
        None
    };

    Ok(ExecutionFidelityAssessment {
        scenario_id: inputs.scenario_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        interval_count,
        metered_interval_count,
        missing_meter_interval_count: missing_meter_count,
        settlement_interval_count,
        missing_settlement_interval_count: missing_settlement_count,
        schedule_time_coverage_bps: schedule_coverage,
        meter_time_coverage_bps: meter_coverage,
        settlement_time_coverage_bps: settlement_coverage,
        trace_complete: meter_evidence_complete,
        scheduled_energy_abs_milli_wh: scheduled_energy_abs,
        metered_energy_signed_sum_milli_wh: metered_energy_signed_sum,
        absolute_energy_deviation_milli_wh: absolute_deviation,
        unfulfilled_scheduled_energy_milli_wh: unfulfilled_energy,
        overdelivery_energy_milli_wh: overdelivery_energy,
        schedule_fulfilment_bps,
        known_balancing_charge_sum_minor_zar: known_balancing_charge,
        complete_balancing_charge_minor_zar: if settlement_evidence_complete {
            Some(known_balancing_charge)
        } else {
            None
        },
        meter_evidence_complete,
        settlement_evidence_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn meter(id: &'static str, energy: i64) -> MeteredEnergyEvidence<'static> {
        MeteredEnergyEvidence {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            observed_at_utc: 200,
            available_at_utc: 210,
            metered_energy_milli_wh: energy,
        }
    }

    fn settlement(id: &'static str, charge: i64) -> BalancingSettlementEvidence<'static> {
        BalancingSettlementEvidence {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            available_at_utc: 220,
            balancing_charge_minor_zar: charge,
        }
    }

    fn interval(
        id: &'static str,
        start: i64,
        end: i64,
        scheduled: i64,
        metered: Option<MeteredEnergyEvidence<'static>>,
        settlement: Option<BalancingSettlementEvidence<'static>>,
    ) -> ExecutionInterval<'static> {
        let meter = metered.map(|mut evidence| {
            evidence.observed_at_utc = end;
            evidence.available_at_utc = end + 10;
            evidence
        });
        let settlement = settlement.map(|mut evidence| {
            evidence.available_at_utc = end + 20;
            evidence
        });
        ExecutionInterval {
            interval_id: id,
            schedule_source_ref: "schedule/source",
            schedule_source_sha256: SOURCE_SHA,
            schedule_issued_at_utc: 50,
            schedule_available_at_utc: 90,
            interval_start_utc: start,
            interval_end_utc: end,
            scheduled_energy_milli_wh: scheduled,
            meter,
            settlement,
        }
    }

    fn inputs<'a>(intervals: &'a [ExecutionInterval<'a>]) -> ExecutionFidelityInputs<'a> {
        ExecutionFidelityInputs {
            scenario_id: "battery-001/execution",
            evaluation_as_of_utc: 500,
            horizon_start_utc: 100,
            horizon_end_utc: 300,
            intervals,
        }
    }

    #[test]
    fn exact_signed_energy_reconciliation_reports_fulfilment_and_booked_cost() {
        let intervals = [
            interval(
                "export-half-hour",
                100,
                200,
                10_000,
                Some(meter("meter-export", 9_000)),
                Some(settlement("settlement-export", 30)),
            ),
            interval(
                "import-half-hour",
                200,
                300,
                -5_000,
                Some(meter("meter-import", -5_000)),
                Some(settlement("settlement-import", 20)),
            ),
        ];
        let result = evaluate_execution_fidelity(inputs(&intervals)).unwrap();
        assert_eq!(result.interval_count, 2);
        assert_eq!(result.metered_interval_count, 2);
        assert_eq!(result.schedule_time_coverage_bps, 10_000);
        assert_eq!(result.meter_time_coverage_bps, 10_000);
        assert_eq!(result.scheduled_energy_abs_milli_wh, 15_000);
        assert_eq!(result.metered_energy_signed_sum_milli_wh, 4_000);
        assert_eq!(result.absolute_energy_deviation_milli_wh, 1_000);
        assert_eq!(result.unfulfilled_scheduled_energy_milli_wh, 1_000);
        assert_eq!(result.overdelivery_energy_milli_wh, 0);
        assert_eq!(result.schedule_fulfilment_bps, Some(9_333));
        assert_eq!(result.known_balancing_charge_sum_minor_zar, 50);
        assert_eq!(result.complete_balancing_charge_minor_zar, Some(50));
        assert!(result.trace_complete);
    }

    #[test]
    fn overdelivery_is_not_counted_as_unfulfilled_energy() {
        let intervals = [interval(
            "export-over",
            100,
            300,
            10_000,
            Some(meter("meter-over", 12_000)),
            Some(settlement("settlement-over", 0)),
        )];
        let result = evaluate_execution_fidelity(inputs(&intervals)).unwrap();
        assert_eq!(result.absolute_energy_deviation_milli_wh, 2_000);
        assert_eq!(result.unfulfilled_scheduled_energy_milli_wh, 0);
        assert_eq!(result.overdelivery_energy_milli_wh, 2_000);
        assert_eq!(result.schedule_fulfilment_bps, Some(10_000));
    }

    #[test]
    fn opposite_direction_delivery_is_not_credit_for_the_schedule() {
        let intervals = [interval(
            "export-opposite",
            100,
            300,
            10_000,
            Some(meter("meter-opposite", -500)),
            Some(settlement("settlement-opposite", 0)),
        )];
        let result = evaluate_execution_fidelity(inputs(&intervals)).unwrap();
        assert_eq!(result.absolute_energy_deviation_milli_wh, 10_500);
        assert_eq!(result.unfulfilled_scheduled_energy_milli_wh, 10_000);
        assert_eq!(result.schedule_fulfilment_bps, Some(0));
    }

    #[test]
    fn missing_meter_or_settlement_is_not_silently_zero() {
        let missing_meter = [
            interval(
                "observed",
                100,
                200,
                10_000,
                Some(meter("meter-1", 9_000)),
                None,
            ),
            interval("missing", 200, 300, 10_000, None, None),
        ];
        let result = evaluate_execution_fidelity(inputs(&missing_meter)).unwrap();
        assert_eq!(result.missing_meter_interval_count, 1);
        assert_eq!(result.meter_time_coverage_bps, 5_000);
        assert_eq!(result.schedule_fulfilment_bps, None);
        assert_eq!(result.complete_balancing_charge_minor_zar, None);

        let missing_settlement = [interval(
            "no-settlement",
            100,
            300,
            10_000,
            Some(meter("meter-2", 10_000)),
            None,
        )];
        let result = evaluate_execution_fidelity(inputs(&missing_settlement)).unwrap();
        assert_eq!(result.missing_settlement_interval_count, 1);
        assert_eq!(result.known_balancing_charge_sum_minor_zar, 0);
        assert_eq!(result.complete_balancing_charge_minor_zar, None);
    }

    #[test]
    fn schedule_gaps_reduce_time_coverage_and_withhold_complete_fulfilment() {
        let intervals = [
            interval(
                "first",
                100,
                180,
                5_000,
                Some(meter("meter-first", 5_000)),
                Some(settlement("settle-first", 0)),
            ),
            interval(
                "second",
                220,
                300,
                5_000,
                Some(meter("meter-second", 4_000)),
                Some(settlement("settle-second", 5)),
            ),
        ];
        let result = evaluate_execution_fidelity(inputs(&intervals)).unwrap();
        assert_eq!(result.schedule_time_coverage_bps, 8_000);
        assert_eq!(result.meter_time_coverage_bps, 8_000);
        assert_eq!(result.schedule_fulfilment_bps, None);
        assert!(!result.trace_complete);
    }

    #[test]
    fn schedule_revision_published_after_delivery_start_is_rejected() {
        let mut late_schedule = interval(
            "late-schedule",
            100,
            300,
            10_000,
            Some(meter("meter-late-schedule", 10_000)),
            None,
        );
        late_schedule.schedule_available_at_utc = 101;
        assert_eq!(
            evaluate_execution_fidelity(inputs(&[late_schedule])),
            Err(ExecutionFidelityError::ScheduleWasNotAvailableAtDelivery)
        );
    }

    #[test]
    fn settlement_without_meter_and_meter_before_interval_end_are_rejected() {
        let settlement_without_meter = [interval(
            "settlement-only",
            100,
            300,
            10_000,
            None,
            Some(settlement("settlement-only", 10)),
        )];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&settlement_without_meter)),
            Err(ExecutionFidelityError::SettlementWithoutMeterEvidence)
        );

        let early_meter = [ExecutionInterval {
            meter: Some(MeteredEnergyEvidence {
                observed_at_utc: 299,
                ..meter("early-meter", 10_000)
            }),
            ..interval("early-observation", 100, 300, 10_000, None, None)
        }];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&early_meter)),
            Err(ExecutionFidelityError::MeterObservedBeforeDeliveryEnded)
        );
    }

    #[test]
    fn out_of_order_intervals_are_distinguished_from_overlapping_intervals() {
        let out_of_order = [
            interval("later", 200, 300, 5_000, None, None),
            interval("earlier", 100, 200, 5_000, None, None),
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&out_of_order)),
            Err(ExecutionFidelityError::IntervalsOutOfOrder)
        );
    }

    #[test]
    fn duplicate_meter_source_cannot_be_recounted_under_a_second_record_id() {
        let duplicate_source = [
            ExecutionInterval {
                meter: Some(MeteredEnergyEvidence {
                    source_ref: "meter/source/same",
                    ..meter("meter-id-one", 5_000)
                }),
                ..interval("interval-one", 100, 200, 5_000, None, None)
            },
            ExecutionInterval {
                meter: Some(MeteredEnergyEvidence {
                    source_ref: "meter/source/same",
                    observed_at_utc: 300,
                    available_at_utc: 310,
                    ..meter("meter-id-two", 5_000)
                }),
                ..interval("interval-two", 200, 300, 5_000, None, None)
            },
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&duplicate_source)),
            Err(ExecutionFidelityError::DuplicateMeterSourceReference)
        );
    }

    #[test]
    fn duplicate_settlement_source_cannot_be_recounted_under_a_second_line_id() {
        let duplicate_source = [
            ExecutionInterval {
                settlement: Some(BalancingSettlementEvidence {
                    source_ref: "settlement/source/same",
                    ..settlement("settlement-id-one", 10)
                }),
                ..interval(
                    "settlement-interval-one",
                    100,
                    200,
                    5_000,
                    Some(meter("meter-one", 5_000)),
                    None,
                )
            },
            ExecutionInterval {
                settlement: Some(BalancingSettlementEvidence {
                    source_ref: "settlement/source/same",
                    available_at_utc: 320,
                    ..settlement("settlement-id-two", 10)
                }),
                ..interval(
                    "settlement-interval-two",
                    200,
                    300,
                    5_000,
                    Some(MeteredEnergyEvidence {
                        observed_at_utc: 300,
                        available_at_utc: 310,
                        ..meter("meter-two", 5_000)
                    }),
                    None,
                )
            },
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&duplicate_source)),
            Err(ExecutionFidelityError::DuplicateSettlementSourceReference)
        );
    }

    #[test]
    fn duplicate_records_overlapping_intervals_and_future_meter_data_fail_closed() {
        let duplicate_meter = [
            interval(
                "first",
                100,
                200,
                5_000,
                Some(meter("same-meter", 5_000)),
                None,
            ),
            interval(
                "second",
                200,
                300,
                5_000,
                Some(meter("same-meter", 5_000)),
                None,
            ),
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&duplicate_meter)),
            Err(ExecutionFidelityError::DuplicateMeterRecordId)
        );

        let overlapping = [
            interval(
                "first",
                100,
                250,
                5_000,
                Some(meter("meter-a", 5_000)),
                None,
            ),
            interval(
                "second",
                200,
                300,
                5_000,
                Some(meter("meter-b", 5_000)),
                None,
            ),
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&overlapping)),
            Err(ExecutionFidelityError::OverlappingIntervals)
        );

        let mut future_interval = interval(
            "future",
            100,
            300,
            10_000,
            Some(meter("future-meter", 10_000)),
            None,
        );
        future_interval.meter = Some(MeteredEnergyEvidence {
            available_at_utc: 501,
            ..future_interval.meter.unwrap()
        });
        let future_meter = [future_interval];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&future_meter)),
            Err(ExecutionFidelityError::MeterNotAvailableAtEvaluation)
        );
    }
}

// Raw sample-backed aggregation is kept beside the legacy pre-aggregated
// execution evaluator so the same public types feed a higher-assurance path.
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

fn trace_coverage_bps(
    covered_seconds: u128,
    horizon_seconds: u128,
) -> Result<u64, TraceBackedExecutionError> {
    let numerator = covered_seconds
        .checked_mul(BASIS_POINTS)
        .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
    u64::try_from(numerator / horizon_seconds)
        .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)
}

fn trace_add_u128(total: u128, value: u128) -> Result<u128, TraceBackedExecutionError> {
    total
        .checked_add(value)
        .ok_or(TraceBackedExecutionError::ArithmeticOverflow)
}

fn trace_add_i128(total: i128, value: i128) -> Result<i128, TraceBackedExecutionError> {
    total
        .checked_add(value)
        .ok_or(TraceBackedExecutionError::ArithmeticOverflow)
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

    let horizon_seconds =
        u128::try_from(i128::from(inputs.horizon_end_utc) - i128::from(inputs.horizon_start_utc))
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
        if start < inputs.horizon_start_utc || end > inputs.horizon_end_utc || start >= end {
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
        scheduled_seconds = trace_add_u128(scheduled_seconds, interval_duration)?;
        let scheduled_abs =
            u128::try_from(i128::from(trace_inputs.scheduled_energy_milli_wh).abs())
                .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?;
        let denominator = 7_200_i128;
        scheduled_energy_abs_numerator = trace_add_u128(
            scheduled_energy_abs_numerator,
            scheduled_abs
                .checked_mul(
                    u128::try_from(denominator)
                        .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?,
                )
                .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
        )?;

        let result = evaluate_scheduled_power_trace(trace_inputs)
            .map_err(TraceBackedExecutionError::UnderlyingTraceError)?;
        if result.energy_denominator != denominator {
            return Err(TraceBackedExecutionError::ArithmeticOverflow);
        }

        meter_seconds = trace_add_u128(
            meter_seconds,
            u128::from(result.trace.observed_coverage_seconds),
        )?;
        known_meter_energy_numerator = trace_add_i128(
            known_meter_energy_numerator,
            result.trace.signed_energy_milli_wh_numerator,
        )?;

        if result.trace.trace_complete {
            complete_trace_interval_count = complete_trace_interval_count
                .checked_add(1)
                .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?;
            absolute_deviation_numerator = trace_add_u128(
                absolute_deviation_numerator,
                result
                    .absolute_deviation_numerator_milli_wh
                    .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
            )?;
            unfulfilled_energy_numerator = trace_add_u128(
                unfulfilled_energy_numerator,
                result
                    .unfulfilled_scheduled_energy_numerator_milli_wh
                    .ok_or(TraceBackedExecutionError::ArithmeticOverflow)?,
            )?;
            overdelivery_numerator = trace_add_u128(
                overdelivery_numerator,
                result
                    .overdelivery_energy_numerator_milli_wh
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
            settlement_seconds = trace_add_u128(settlement_seconds, interval_duration)?;
            known_balancing_charge = trace_add_i128(
                known_balancing_charge,
                i128::from(settlement.balancing_charge_minor_zar),
            )?;
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

    let schedule_coverage = trace_coverage_bps(scheduled_seconds, horizon_seconds)?;
    let meter_coverage = trace_coverage_bps(meter_seconds, horizon_seconds)?;
    let settlement_coverage = trace_coverage_bps(settlement_seconds, horizon_seconds)?;
    let interval_count = u64::try_from(inputs.intervals.len())
        .map_err(|_| TraceBackedExecutionError::ArithmeticOverflow)?;
    let missing_settlement_count = interval_count.saturating_sub(settlement_interval_count);
    let schedule_complete = schedule_coverage == 10_000;
    let meter_complete =
        schedule_complete && incomplete_trace_interval_count == 0 && meter_coverage == 10_000;
    let settlement_complete =
        meter_complete && missing_settlement_count == 0 && settlement_coverage == 10_000;
    let schedule_fulfilment_bps = if meter_complete && scheduled_energy_abs_numerator > 0 {
        let fulfilled = scheduled_energy_abs_numerator.saturating_sub(unfulfilled_energy_numerator);
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
        complete_absolute_deviation_numerator_milli_wh: meter_complete
            .then_some(absolute_deviation_numerator),
        complete_unfulfilled_energy_numerator_milli_wh: meter_complete
            .then_some(unfulfilled_energy_numerator),
        complete_overdelivery_energy_numerator_milli_wh: meter_complete
            .then_some(overdelivery_numerator),
        schedule_fulfilment_bps,
        known_balancing_charge_sum_minor_zar: known_balancing_charge,
        complete_balancing_charge_minor_zar: settlement_complete.then_some(known_balancing_charge),
        interval_receipts,
    })
}

#[cfg(test)]
mod trace_backed_tests {
    use super::*;
    use crate::power_trace::PowerTraceInputs;
    use crate::power_trace::PowerTraceSample;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SCHEDULE_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn sample(id: &'static str, t: i64, power: i64) -> PowerTraceSample<'static> {
        PowerTraceSample {
            sample_id: id,
            timestamp_utc_seconds: t,
            power_milli_w: power,
        }
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
                    maximum_integrable_gap_seconds: u32::try_from(end - start)
                        .expect("synthetic interval gap fits u32"),
                    samples,
                },
            },
            settlement,
        }
    }

    fn settlement(
        id: &'static str,
        source: &'static str,
        charge: i64,
        available: i64,
    ) -> BalancingSettlementEvidence<'static> {
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
        let samples_a = [
            sample("a-start", 100, 1_000_000),
            sample("a-end", 3_700, 1_000_000),
        ];
        let samples_b = [
            sample("b-start", 3_700, 1_000_000),
            sample("b-end", 7_300, 1_000_000),
        ];
        let intervals = [
            item(
                "tp-a",
                "schedule-a",
                "meter/day-file",
                &samples_a,
                100,
                3_700,
                900_000,
                Some(settlement("settle-a", "settlement/a", 10, 3_720)),
            ),
            item(
                "tp-b",
                "schedule-b",
                "meter/day-file",
                &samples_b,
                3_700,
                7_300,
                1_100_000,
                Some(settlement("settle-b", "settlement/b", 20, 7_320)),
            ),
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
        assert_eq!(
            result.scheduled_energy_abs_numerator_milli_wh,
            14_400_000_000
        );
        assert_eq!(
            result.complete_unfulfilled_energy_numerator_milli_wh,
            Some(720_000_000)
        );
        assert_eq!(
            result.complete_overdelivery_energy_numerator_milli_wh,
            Some(720_000_000)
        );
        assert_eq!(result.schedule_fulfilment_bps, Some(9_500));
        assert_eq!(result.complete_balancing_charge_minor_zar, Some(30));
    }

    #[test]
    fn incomplete_raw_trace_withholds_complete_fulfilment_and_settlement_total() {
        let samples_a = [
            sample("a-start", 100, 1_000_000),
            sample("a-end", 3_700, 1_000_000),
        ];
        let samples_b = [
            sample("b-middle", 4_000, 1_000_000),
            sample("b-end", 7_300, 1_000_000),
        ];
        let intervals = [
            item(
                "tp-a",
                "schedule-a",
                "meter/day-file",
                &samples_a,
                100,
                3_700,
                900_000,
                Some(settlement("settle-a", "settlement/a", 10, 3_720)),
            ),
            item(
                "tp-b",
                "schedule-b",
                "meter/day-file",
                &samples_b,
                3_700,
                7_300,
                1_100_000,
                None,
            ),
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
        let samples_a = [
            sample("a-start", 100, 1_000_000),
            sample("a-end", 3_700, 1_000_000),
        ];
        let samples_b = [
            sample("b-start", 4_000, 1_000_000),
            sample("b-end", 7_300, 1_000_000),
        ];
        let intervals = [
            item(
                "tp-a",
                "schedule-a",
                "meter/day-file",
                &samples_a,
                100,
                3_700,
                900_000,
                None,
            ),
            item(
                "tp-b",
                "schedule-b",
                "meter/day-file",
                &samples_b,
                4_000,
                7_300,
                1_100_000,
                None,
            ),
        ];
        let result = evaluate_trace_backed_execution(TraceBackedExecutionInputs {
            scenario_id: "golden/schedule-gap",
            evaluation_as_of_utc: 8_000,
            horizon_start_utc: 100,
            horizon_end_utc: 7_300,
            intervals: &intervals,
        })
        .unwrap();
        assert_eq!(result.schedule_time_coverage_bps, 9_583);
        assert!(!result.schedule_evidence_complete);
        assert_eq!(result.schedule_fulfilment_bps, None);
    }

    #[test]
    fn reused_power_sample_ids_and_conflicting_source_digests_fail_closed() {
        let samples_a = [
            sample("shared-start", 100, 1_000_000),
            sample("a-end", 3_700, 1_000_000),
        ];
        let samples_b = [
            sample("shared-start", 3_700, 1_000_000),
            sample("b-end", 7_300, 1_000_000),
        ];
        let intervals = [
            item(
                "tp-a",
                "schedule-a",
                "meter/day-file",
                &samples_a,
                100,
                3_700,
                900_000,
                None,
            ),
            item(
                "tp-b",
                "schedule-b",
                "meter/day-file",
                &samples_b,
                3_700,
                7_300,
                1_100_000,
                None,
            ),
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
        let samples_a = [
            sample("a-start", 100, 1_000_000),
            sample("a-end", 3_700, 1_000_000),
        ];
        let samples_b = [
            sample("b-start", 3_700, 1_000_000),
            sample("b-end", 7_300, 1_000_000),
        ];
        let duplicate = [
            item(
                "tp-a",
                "schedule-a",
                "meter/day-file",
                &samples_a,
                100,
                3_700,
                900_000,
                Some(settlement("settle-a", "settlement/same", 10, 3_720)),
            ),
            item(
                "tp-b",
                "schedule-b",
                "meter/day-file",
                &samples_b,
                3_700,
                7_300,
                1_100_000,
                Some(settlement("settle-b", "settlement/same", 20, 7_320)),
            ),
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

        let incomplete = [
            sample("b-mid", 4_000, 1_000_000),
            sample("b-end", 7_300, 1_000_000),
        ];
        let intervals = [item(
            "tp-b",
            "schedule-b",
            "meter/day-file",
            &incomplete,
            3_700,
            7_300,
            1_100_000,
            Some(settlement("settle-b", "settlement/b", 20, 7_320)),
        )];
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
