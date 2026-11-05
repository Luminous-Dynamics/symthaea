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

use std::collections::BTreeSet;

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
    DuplicateSettlementRecordId,
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
    u128::try_from(i128::from(value).abs())
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)
}

fn checked_sum_u128(
    current: u128,
    add: u128,
) -> Result<u128, ExecutionFidelityError> {
    current
        .checked_add(add)
        .ok_or(ExecutionFidelityError::ArithmeticOverflow)
}

fn checked_sum_i128(
    current: i128,
    add: i128,
) -> Result<i128, ExecutionFidelityError> {
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
        - i128::try_from(scheduled_abs)
            .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?)
        .max(0);
    Ok((
        deviation,
        u128::try_from(unfulfilled)
            .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?,
        u128::try_from(overdelivery)
            .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?,
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

    let horizon_seconds = u128::try_from(inputs.horizon_end_utc - inputs.horizon_start_utc)
        .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?;
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
    let mut settlement_record_ids = BTreeSet::new();
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
        if interval.interval_start_utc < previous_interval_end {
            return Err(ExecutionFidelityError::OverlappingIntervals);
        }

        let duration = u128::try_from(interval.interval_end_utc - interval.interval_start_utc)
            .map_err(|_| ExecutionFidelityError::ArithmeticOverflow)?;
        scheduled_seconds = checked_sum_u128(scheduled_seconds, duration)?;
        scheduled_energy_abs =
            checked_sum_u128(scheduled_energy_abs, abs_i64(interval.scheduled_energy_milli_wh)?)?;
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
    let meter_evidence_complete = schedule_coverage_complete
        && missing_meter_count == 0
        && meter_coverage == 10_000;
    let settlement_evidence_complete = meter_evidence_complete
        && missing_settlement_count == 0
        && settlement_coverage == 10_000;

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
        ExecutionInterval {
            interval_id: id,
            schedule_source_ref: "schedule/source",
            schedule_source_sha256: SOURCE_SHA,
            schedule_issued_at_utc: 50,
            schedule_available_at_utc: 90,
            interval_start_utc: start,
            interval_end_utc: end,
            scheduled_energy_milli_wh: scheduled,
            meter: metered,
            settlement,
        }
    }

    fn inputs<'a>(intervals: &'a [ExecutionInterval<'a>]) -> ExecutionFidelityInputs<'a> {
        ExecutionFidelityInputs {
            scenario_id: "battery-001/execution",
            evaluation_as_of_utc: 300,
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
            interval("observed", 100, 200, 10_000, Some(meter("meter-1", 9_000)), None),
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
            interval("first", 100, 180, 5_000, Some(meter("meter-first", 5_000)), Some(settlement("settle-first", 0))),
            interval("second", 220, 300, 5_000, Some(meter("meter-second", 4_000)), Some(settlement("settle-second", 5))),
        ];
        let result = evaluate_execution_fidelity(inputs(&intervals)).unwrap();
        assert_eq!(result.schedule_time_coverage_bps, 8_000);
        assert_eq!(result.meter_time_coverage_bps, 8_000);
        assert_eq!(result.schedule_fulfilment_bps, None);
        assert!(!result.trace_complete);
    }

    #[test]
    fn duplicate_records_overlapping_intervals_and_future_meter_data_fail_closed() {
        let duplicate_meter = [
            interval("first", 100, 200, 5_000, Some(meter("same-meter", 5_000)), None),
            interval("second", 200, 300, 5_000, Some(meter("same-meter", 5_000)), None),
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&duplicate_meter)),
            Err(ExecutionFidelityError::DuplicateMeterRecordId)
        );

        let overlapping = [
            interval("first", 100, 250, 5_000, Some(meter("meter-a", 5_000)), None),
            interval("second", 200, 300, 5_000, Some(meter("meter-b", 5_000)), None),
        ];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&overlapping)),
            Err(ExecutionFidelityError::OverlappingIntervals)
        );

        let future_meter = [interval(
            "future",
            100,
            300,
            10_000,
            Some(MeteredEnergyEvidence {
                available_at_utc: 301,
                ..meter("future-meter", 10_000)
            }),
            None,
        )];
        assert_eq!(
            evaluate_execution_fidelity(inputs(&future_meter)),
            Err(ExecutionFidelityError::MeterNotAvailableAtEvaluation)
        );
    }
}
