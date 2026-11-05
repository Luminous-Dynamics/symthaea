//! Event-ordered treasury liquidity waterfall for SAWEM credit-cover planning.
//!
//! Three paths are evaluated separately:
//! - committed: obligations/receipts supported by committed source evidence;
//! - forecast: committed plus forecast cash movements;
//! - stress: committed plus forecast plus adverse stress adjustments.
//!
//! Planned collateral deposits reduce projected unrestricted cash but never
//! increase Posted Credit Cover in this module. A separate receipt-backed
//! credit-cover ledger remains authoritative for posted cover. The Market
//! Operator Required Credit Cover is an external input; this module does not
//! calculate the §15.10.3 historical-activity / ASE / UEF formula.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityEventClass {
    Committed,
    Forecast,
    Stress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityCashEventKind {
    OperatingReceipt,
    MandatoryOutflow,
    CollateralCashLockup,
    EarlyInvoicePayment,
    CollateralRelease,
    ForecastAdjustment,
    RequiredCoverReassessment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityCashEvent<'a> {
    pub event_id: &'a str,
    pub source_ref: &'a str,
    pub day_ordinal: i32,
    pub minute_of_day: u16,
    /// Stable order for multiple events at the same local minute.
    pub sequence: u32,
    pub class: LiquidityEventClass,
    pub kind: LiquidityCashEventKind,
    /// Positive magnitude for every kind. RequiredCoverReassessment carries
    /// an absolute new MO-notified / scenario Required Credit Cover amount.
    pub amount_minor_zar: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityWaterfallInputs<'a> {
    pub scenario_id: &'a str,
    pub start_day_ordinal: i32,
    pub start_minute_of_day: u16,
    pub horizon_end_day_ordinal: i32,
    pub credit_cover_deadline_day_ordinal: i32,
    pub credit_cover_deadline_minute_of_day: u16,
    /// Civil date coverage and timezone interpretation are supplied outside
    /// this kernel; the ordinal list must cover the entire planning horizon.
    pub calendar_source_ref: &'a str,
    pub calendar_source_sha256: &'a str,
    pub calendar_day_ordinals: &'a [i32],
    pub opening_unrestricted_cash_minor_zar: u64,
    pub opening_cash_source_ref: &'a str,
    pub minimum_operating_cash_floor_minor_zar: u64,
    pub operating_cash_floor_source_ref: &'a str,
    pub market_required_credit_cover_minor_zar: Option<u64>,
    pub required_cover_source_ref: Option<&'a str>,
    pub posted_credit_cover_minor_zar: u64,
    pub posted_cover_source_ref: &'a str,
    pub events: &'a [LiquidityCashEvent<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityWaterfallError {
    InvalidEvidenceReference,
    InvalidCalendarDigest,
    InvalidCalendarCoverage,
    InvalidCalendarOrdering,
    InvalidTimestamp,
    InvalidHorizon,
    InvalidEvent,
    EventsNotChronological,
    DuplicateEventId,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityBreachPoint<'a> {
    pub day_ordinal: i32,
    pub minute_of_day: u16,
    pub event_id: &'a str,
    pub cash_balance_minor_zar: i128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityPathAssessment<'a> {
    pub included_event_count: u32,
    pub end_cash_minor_zar: i128,
    pub minimum_cash_minor_zar: i128,
    pub minimum_cash_before_cover_deadline_minor_zar: i128,
    pub cash_at_cover_deadline_minor_zar: i128,
    pub cash_headroom_at_cover_deadline_minor_zar: u64,
    pub first_operating_floor_breach: Option<LiquidityBreachPoint<'a>>,
    pub maximum_cover_shortfall_minor_zar: Option<u64>,
    pub cover_shortfall_at_deadline_minor_zar: Option<u64>,
    /// This is cash that could fit above the floor in the scenario only. It
    /// is not Posted Credit Cover and is not proof that cash can be transferred.
    pub potentially_fundable_cover_at_deadline_minor_zar: Option<u64>,
    pub external_funding_gap_at_deadline_minor_zar: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityWaterfallAssessment<'a> {
    pub scenario_id: &'a str,
    pub committed: LiquidityPathAssessment<'a>,
    pub forecast: LiquidityPathAssessment<'a>,
    pub stress: LiquidityPathAssessment<'a>,
    pub forecast_event_count: u32,
    pub stress_event_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityWaterfallResult<'a> {
    Determined(LiquidityWaterfallAssessment<'a>),
    Indeterminate { reason: &'static str },
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn event_in_path(class: LiquidityEventClass, path: u8) -> bool {
    match path {
        0 => class == LiquidityEventClass::Committed,
        1 => class != LiquidityEventClass::Stress,
        _ => true,
    }
}

fn cash_delta(event: LiquidityCashEvent<'_>) -> Result<i128, LiquidityWaterfallError> {
    let amount = i128::from(event.amount_minor_zar);
    match event.kind {
        LiquidityCashEventKind::OperatingReceipt
        | LiquidityCashEventKind::CollateralRelease => Ok(amount),
        LiquidityCashEventKind::MandatoryOutflow
        | LiquidityCashEventKind::CollateralCashLockup
        | LiquidityCashEventKind::EarlyInvoicePayment => Ok(-amount),
        LiquidityCashEventKind::ForecastAdjustment => {
            // Positive magnitude is defined as an adverse cash reduction.
            Ok(-amount)
        }
        LiquidityCashEventKind::RequiredCoverReassessment => Ok(0),
    }
}

fn assess_path<'a>(
    inputs: &LiquidityWaterfallInputs<'a>,
    path: u8,
) -> Result<LiquidityPathAssessment<'a>, LiquidityWaterfallError> {
    let mut cash = i128::from(inputs.opening_unrestricted_cash_minor_zar);
    let floor = i128::from(inputs.minimum_operating_cash_floor_minor_zar);
    let mut minimum_cash = cash;
    let mut minimum_before_deadline = cash;
    let mut cash_at_deadline = cash;
    let mut deadline_cover = inputs.market_required_credit_cover_minor_zar;
    let mut maximum_cover_shortfall: Option<u64> = deadline_cover
        .map(|required| required.saturating_sub(inputs.posted_credit_cover_minor_zar));
    let mut first_breach = None;
    let mut included = 0_u32;
    let mut deadline_reached = inputs.start_day_ordinal > inputs.credit_cover_deadline_day_ordinal
        || (inputs.start_day_ordinal == inputs.credit_cover_deadline_day_ordinal
            && inputs.start_minute_of_day > inputs.credit_cover_deadline_minute_of_day);

    for event in inputs.events {
        if !event_in_path(event.class, path) {
            continue;
        }
        included = included.saturating_add(1);
        let is_before_deadline = event.day_ordinal < inputs.credit_cover_deadline_day_ordinal
            || (event.day_ordinal == inputs.credit_cover_deadline_day_ordinal
                && event.minute_of_day <= inputs.credit_cover_deadline_minute_of_day);
        if is_before_deadline {
            match event.kind {
                LiquidityCashEventKind::RequiredCoverReassessment => {
                    deadline_cover = Some(event.amount_minor_zar);
                    minimum_cover_shortfall = deadline_cover
                        .map(|required| required.saturating_sub(inputs.posted_credit_cover_minor_zar));
                }
                _ => {
                    cash = cash
                        .checked_add(cash_delta(*event)?)
                        .ok_or(LiquidityWaterfallError::ArithmeticOverflow)?;
                }
            }
            minimum_before_deadline = minimum_before_deadline.min(cash);
            if first_breach.is_none() && cash < floor {
                first_breach = Some(LiquidityBreachPoint {
                    day_ordinal: event.day_ordinal,
                    minute_of_day: event.minute_of_day,
                    event_id: event.event_id,
                    cash_balance_minor_zar: cash,
                });
            }
            minimum_cash = minimum_cash.min(cash);
            maximum_cover_shortfall = match (
                maximum_cover_shortfall,
                deadline_cover.map(|required| {
                    required.saturating_sub(inputs.posted_credit_cover_minor_zar)
                }),
            ) {
                (Some(previous), Some(current)) => Some(previous.max(current)),
                (Some(previous), None) => Some(previous),
                (None, Some(current)) => Some(current),
                (None, None) => None,
            };
            cash_at_deadline = cash;
            deadline_reached = true;
        } else {
            match event.kind {
                LiquidityCashEventKind::RequiredCoverReassessment => {}
                _ => {
                    cash = cash
                        .checked_add(cash_delta(*event)?)
                        .ok_or(LiquidityWaterfallError::ArithmeticOverflow)?;
                }
            }
            if first_breach.is_none() && cash < floor {
                first_breach = Some(LiquidityBreachPoint {
                    day_ordinal: event.day_ordinal,
                    minute_of_day: event.minute_of_day,
                    event_id: event.event_id,
                    cash_balance_minor_zar: cash,
                });
            }
            minimum_cash = minimum_cash.min(cash);
        }
    }

    if !deadline_reached {
        cash_at_deadline = cash;
    }
    let headroom = (cash_at_deadline - floor).max(0);
    let headroom_u64 =
        u64::try_from(headroom).map_err(|_| LiquidityWaterfallError::ArithmeticOverflow)?;
    let shortfall_at_deadline = deadline_cover
        .map(|required| required.saturating_sub(inputs.posted_credit_cover_minor_zar));
    let potentially_fundable = shortfall_at_deadline.map(|shortfall| shortfall.min(headroom_u64));
    let external_gap = shortfall_at_deadline
        .zip(potentially_fundable)
        .map(|(shortfall, fundable)| shortfall.saturating_sub(fundable));

    Ok(LiquidityPathAssessment {
        included_event_count: included,
        end_cash_minor_zar: cash,
        minimum_cash_minor_zar: minimum_cash,
        minimum_cash_before_cover_deadline_minor_zar: minimum_before_deadline,
        cash_at_cover_deadline_minor_zar: cash_at_deadline,
        cash_headroom_at_cover_deadline_minor_zar: headroom_u64,
        first_operating_floor_breach: first_breach,
        maximum_cover_shortfall_minor_zar: minimum_cover_shortfall,
        cover_shortfall_at_deadline_minor_zar: shortfall_at_deadline,
        potentially_fundable_cover_at_deadline_minor_zar: potentially_fundable,
        external_funding_gap_at_deadline_minor_zar: external_gap,
    })
}

/// Evaluate a chronologically ordered treasury scenario using explicit civil
/// dates, source refs and separate committed / forecast / adverse-stress paths.
/// The MO Required Credit Cover is optional: if absent, cash paths are still
/// returned but cover shortfall/funding gap fields remain None.
pub fn evaluate_liquidity_waterfall<'a>(
    inputs: LiquidityWaterfallInputs<'a>,
) -> Result<LiquidityWaterfallResult<'a>, LiquidityWaterfallError> {
    if !nonempty(inputs.scenario_id)
        || !nonempty(inputs.calendar_source_ref)
        || !nonempty(inputs.opening_cash_source_ref)
        || !nonempty(inputs.operating_cash_floor_source_ref)
        || !nonempty(inputs.posted_cover_source_ref)
    {
        return Err(LiquidityWaterfallError::InvalidEvidenceReference);
    }
    if inputs.market_required_credit_cover_minor_zar.is_some()
        && inputs
            .required_cover_source_ref
            .map(nonempty)
            != Some(true)
    {
        return Ok(LiquidityWaterfallResult::Indeterminate {
            reason: "Required Credit Cover source reference is missing",
        });
    }
    if inputs.market_required_credit_cover_minor_zar.is_none()
        && inputs.required_cover_source_ref.is_some()
    {
        return Err(LiquidityWaterfallError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.calendar_source_sha256) {
        return Err(LiquidityWaterfallError::InvalidCalendarDigest);
    }
    if inputs.start_minute_of_day >= 24 * 60
        || inputs.credit_cover_deadline_minute_of_day >= 24 * 60
        || inputs.horizon_end_day_ordinal < inputs.start_day_ordinal
        || inputs.credit_cover_deadline_day_ordinal < inputs.start_day_ordinal
        || inputs.credit_cover_deadline_day_ordinal > inputs.horizon_end_day_ordinal
        || (inputs.start_day_ordinal == inputs.credit_cover_deadline_day_ordinal
            && inputs.start_minute_of_day > inputs.credit_cover_deadline_minute_of_day)
    {
        return Err(LiquidityWaterfallError::InvalidHorizon);
    }
    if inputs.calendar_day_ordinals.is_empty()
        || inputs.calendar_day_ordinals.first() != Some(&inputs.start_day_ordinal)
        || inputs.calendar_day_ordinals.last() != Some(&inputs.horizon_end_day_ordinal)
    {
        return Err(LiquidityWaterfallError::InvalidCalendarCoverage);
    }
    if inputs
        .calendar_day_ordinals
        .windows(2)
        .any(|pair| pair[0].checked_add(1) != Some(pair[1]))
    {
        return Err(LiquidityWaterfallError::InvalidCalendarOrdering);
    }

    let mut previous: Option<(i32, u16, u32)> = None;
    for (index, event) in inputs.events.iter().enumerate() {
        if !nonempty(event.event_id)
            || !nonempty(event.source_ref)
            || (event.amount_minor_zar == 0
                && event.kind != LiquidityCashEventKind::RequiredCoverReassessment)
            || event.minute_of_day >= 24 * 60
            || event.day_ordinal < inputs.start_day_ordinal
            || event.day_ordinal > inputs.horizon_end_day_ordinal
            || !inputs.calendar_day_ordinals.contains(&event.day_ordinal)
        {
            return Err(LiquidityWaterfallError::InvalidEvent);
        }
        if event.kind == LiquidityCashEventKind::ForecastAdjustment
            && event.class != LiquidityEventClass::Stress
        {
            return Err(LiquidityWaterfallError::InvalidEvent);
        }
        if event.kind == LiquidityCashEventKind::RequiredCoverReassessment
            && inputs.market_required_credit_cover_minor_zar.is_none()
        {
            return Err(LiquidityWaterfallError::InvalidEvent);
        }
        if inputs.events[..index]
            .iter()
            .any(|previous_event| previous_event.event_id == event.event_id)
        {
            return Err(LiquidityWaterfallError::DuplicateEventId);
        }
        let current = (event.day_ordinal, event.minute_of_day, event.sequence);
        if previous.is_some_and(|last| current <= last) {
            return Err(LiquidityWaterfallError::EventsNotChronological);
        }
        previous = Some(current);
    }

    let forecast_event_count = u32::try_from(
        inputs.events.iter().filter(|event| event.class == LiquidityEventClass::Forecast).count(),
    )
    .map_err(|_| LiquidityWaterfallError::ArithmeticOverflow)?;
    let stress_event_count = u32::try_from(
        inputs.events.iter().filter(|event| event.class == LiquidityEventClass::Stress).count(),
    )
    .map_err(|_| LiquidityWaterfallError::ArithmeticOverflow)?;

    Ok(LiquidityWaterfallResult::Determined(
        LiquidityWaterfallAssessment {
            scenario_id: inputs.scenario_id,
            committed: assess_path(&inputs, 0)?,
            forecast: assess_path(&inputs, 1)?,
            stress: assess_path(&inputs, 2)?,
            forecast_event_count,
            stress_event_count,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALENDAR_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn event(
        id: &'static str,
        day: i32,
        minute: u16,
        sequence: u32,
        class: LiquidityEventClass,
        kind: LiquidityCashEventKind,
        amount: u64,
    ) -> LiquidityCashEvent<'static> {
        LiquidityCashEvent {
            event_id: id,
            source_ref: id,
            day_ordinal: day,
            minute_of_day: minute,
            sequence,
            class,
            kind,
            amount_minor_zar: amount,
        }
    }

    fn base_inputs<'a>(
        events: &'a [LiquidityCashEvent<'a>],
    ) -> LiquidityWaterfallInputs<'a> {
        LiquidityWaterfallInputs {
            scenario_id: "battery/project-001/base",
            start_day_ordinal: 1,
            start_minute_of_day: 0,
            horizon_end_day_ordinal: 5,
            credit_cover_deadline_day_ordinal: 3,
            credit_cover_deadline_minute_of_day: 17 * 60,
            calendar_source_ref: "calendar/sa/2026-v1",
            calendar_source_sha256: CALENDAR_SHA,
            calendar_day_ordinals: &[1, 2, 3, 4, 5],
            opening_unrestricted_cash_minor_zar: 50_000_000,
            opening_cash_source_ref: "treasury/opening-cash/001",
            minimum_operating_cash_floor_minor_zar: 10_000_000,
            operating_cash_floor_source_ref: "treasury/operating-floor/001",
            market_required_credit_cover_minor_zar: Some(30_000_000),
            required_cover_source_ref: Some("mo/required-cover/001"),
            posted_credit_cover_minor_zar: 20_000_000,
            posted_cover_source_ref: "mo/posted-cover/001",
            events,
        }
    }

    #[test]
    fn committed_forecast_and_stress_paths_show_different_minimum_cash() {
        let events = [
            event("contracted-receipt", 2, 9 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::OperatingReceipt, 15_000_000),
            event("mandatory-bill", 2, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 20_000_000),
            event("forecast-sales", 3, 9 * 60, 0, LiquidityEventClass::Forecast, LiquidityCashEventKind::OperatingReceipt, 18_000_000),
            event("cash-lock", 3, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::CollateralCashLockup, 12_000_000),
            event("adverse-cost", 4, 9 * 60, 0, LiquidityEventClass::Stress, LiquidityCashEventKind::ForecastAdjustment, 8_000_000),
        ];
        let inputs = base_inputs(&events);
        let result = evaluate_liquidity_waterfall(inputs).unwrap();
        let LiquidityWaterfallResult::Determined(result) = result else {
            panic!("all path inputs are supplied");
        };
        assert_eq!(result.committed.minimum_cash_minor_zar, 33_000_000);
        assert_eq!(result.forecast.minimum_cash_minor_zar, 51_000_000);
        assert_eq!(result.stress.minimum_cash_minor_zar, 43_000_000);
        assert_eq!(result.forecast_event_count, 1);
        assert_eq!(result.stress_event_count, 1);
    }

    #[test]
    fn projected_collateral_and_invoice_payments_reduce_cash_but_not_posted_cover() {
        let events = [
            event("collateral-plan", 2, 9 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::CollateralCashLockup, 8_000_000),
            event("invoice-early-pay", 2, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::EarlyInvoicePayment, 5_000_000),
        ];
        let result = evaluate_liquidity_waterfall(base_inputs(&events)).unwrap();
        let LiquidityWaterfallResult::Determined(result) = result else {
            panic!("base scenario is complete");
        };
        assert_eq!(result.committed.end_cash_minor_zar, 37_000_000);
        assert_eq!(result.committed.cover_shortfall_at_deadline_minor_zar, Some(10_000_000));
        assert_eq!(
            result.committed.potentially_fundable_cover_at_deadline_minor_zar,
            Some(10_000_000)
        );
    }

    #[test]
    fn required_cover_reassessment_changes_cover_gap_only_after_its_event() {
        let events = [
            event("new-mo-cover", 3, 16 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::RequiredCoverReassessment, 40_000_000),
        ];
        let result = evaluate_liquidity_waterfall(base_inputs(&events)).unwrap();
        let LiquidityWaterfallResult::Determined(result) = result else {
            panic!("cover source exists");
        };
        assert_eq!(result.committed.cover_shortfall_at_deadline_minor_zar, Some(20_000_000));
        assert_eq!(result.committed.external_funding_gap_at_deadline_minor_zar, Some(0));
    }

    #[test]
    fn cash_floor_breach_records_first_timestamp_and_scenario() {
        let events = [
            event("outflow-1", 2, 9 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 35_000_000),
            event("outflow-stress", 3, 11 * 60, 0, LiquidityEventClass::Stress, LiquidityCashEventKind::ForecastAdjustment, 20_000_000),
        ];
        let result = evaluate_liquidity_waterfall(base_inputs(&events)).unwrap();
        let LiquidityWaterfallResult::Determined(result) = result else {
            panic!("cash flows are well formed");
        };
        assert_eq!(result.committed.first_operating_floor_breach.unwrap().event_id, "outflow-1");
        assert_eq!(result.forecast.first_operating_floor_breach, result.committed.first_operating_floor_breach);
        assert_eq!(result.stress.first_operating_floor_breach.unwrap().event_id, "outflow-1");
    }

    #[test]
    fn events_after_credit_cover_deadline_do_not_change_deadline_cash() {
        let events = [
            event("after-deadline", 4, 9 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::OperatingReceipt, 25_000_000),
        ];
        let result = evaluate_liquidity_waterfall(base_inputs(&events)).unwrap();
        let LiquidityWaterfallResult::Determined(result) = result else {
            panic!("deadline lies within horizon");
        };
        assert_eq!(result.committed.cash_at_cover_deadline_minor_zar, 50_000_000);
        assert_eq!(result.committed.end_cash_minor_zar, 75_000_000);
    }

    #[test]
    fn missing_required_cover_keeps_cash_paths_but_no_cover_conclusion() {
        let events = [event("bill", 2, 9 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 5_000_000)];
        let inputs = LiquidityWaterfallInputs {
            market_required_credit_cover_minor_zar: None,
            required_cover_source_ref: None,
            ..base_inputs(&events)
        };
        let result = evaluate_liquidity_waterfall(inputs).unwrap();
        let LiquidityWaterfallResult::Determined(result) = result else {
            panic!("cash path remains independently useful");
        };
        assert_eq!(result.committed.minimum_cash_minor_zar, 45_000_000);
        assert_eq!(result.committed.cover_shortfall_at_deadline_minor_zar, None);
        assert_eq!(result.committed.external_funding_gap_at_deadline_minor_zar, None);
    }

    #[test]
    fn unordered_events_duplicate_ids_and_incomplete_calendars_fail_closed() {
        let out_of_order = [
            event("late", 3, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 1_000_000),
            event("early", 2, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 1_000_000),
        ];
        assert_eq!(
            evaluate_liquidity_waterfall(base_inputs(&out_of_order)),
            Err(LiquidityWaterfallError::EventsNotChronological)
        );

        let duplicate = [
            event("same", 2, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 1_000_000),
            event("same", 3, 10 * 60, 0, LiquidityEventClass::Committed, LiquidityCashEventKind::MandatoryOutflow, 1_000_000),
        ];
        assert_eq!(
            evaluate_liquidity_waterfall(base_inputs(&duplicate)),
            Err(LiquidityWaterfallError::DuplicateEventId)
        );

        let incomplete = LiquidityWaterfallInputs {
            calendar_day_ordinals: &[1, 2, 3, 4],
            ..base_inputs(&[])
        };
        assert_eq!(
            evaluate_liquidity_waterfall(incomplete),
            Err(LiquidityWaterfallError::InvalidCalendarCoverage)
        );
    }
}
