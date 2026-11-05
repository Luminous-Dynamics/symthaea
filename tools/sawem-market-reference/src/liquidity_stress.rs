//! Source-bound treasury liquidity headroom assessment for a SAWEM
//! Credit Cover snapshot.
//!
//! The Market Operator's Required Credit Cover is consumed as an external input.
//! This module does not compute the UEF / §15.10.3 exposure formula, post cash,
//! or turn potentially available treasury headroom into Posted Credit Cover.
//! It answers the narrower question: how much of a known cover shortfall could
//! be funded without breaching a separately specified operating cash floor?

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityStressSnapshot<'a> {
    pub scenario_id: &'a str,
    /// Day ordinal mapped by a source-backed civil calendar in the parent profile.
    pub as_of_day_ordinal: i32,
    pub civil_calendar_source_ref: &'a str,
    /// Exact MO-notified amount or a separately source-bound approved test case.
    pub market_required_credit_cover_minor_zar: Option<u64>,
    pub required_cover_source_ref: Option<&'a str>,
    pub posted_credit_cover_minor_zar: u64,
    pub posted_cover_source_ref: &'a str,
    /// Unrestricted treasury cash before the mandatory outflows below.
    pub unrestricted_cash_minor_zar: u64,
    pub unrestricted_cash_source_ref: &'a str,
    /// Outflows due in the same liquidity horizon, excluding amounts already
    /// included in Required Credit Cover or Posted Credit Cover.
    pub mandatory_noncover_outflows_minor_zar: u64,
    pub mandatory_outflows_source_ref: &'a str,
    /// Board / treasury minimum operating cash buffer; an input, not a default.
    pub minimum_operating_cash_floor_minor_zar: u64,
    pub operating_cash_floor_source_ref: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityStressError {
    InvalidEvidenceReference,
    DuplicateEvidenceReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityCoverPosition {
    PostedCoverMeetsMarketRequirement,
    CoverShortfall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityStressAssessment {
    pub as_of_day_ordinal: i32,
    pub market_required_credit_cover_minor_zar: u64,
    pub posted_credit_cover_minor_zar: u64,
    pub cover_position: LiquidityCoverPosition,
    /// A shortfall in Posted Credit Cover, not yet cured by a posting receipt.
    pub credit_cover_shortfall_minor_zar: u64,
    /// Treasury cash after the mandatory non-cover outflows in this snapshot.
    pub cash_after_mandatory_outflows_minor_zar: u64,
    pub operating_cash_floor_minor_zar: u64,
    /// Deficit below the operating cash floor before any cover top-up.
    pub operating_cash_floor_deficit_minor_zar: u64,
    /// Cash headroom above the operating floor. This is not Posted Credit Cover.
    pub cash_available_above_floor_minor_zar: u64,
    /// Maximum part of the current cover shortfall that could be funded from
    /// this headroom, assuming it is transferable by the required deadline.
    /// It remains only potential until a separate receipt-backed posting occurs.
    pub potentially_fundable_cover_minor_zar: u64,
    /// Part of the cover shortfall not fundable from current above-floor cash.
    pub external_funding_gap_minor_zar: u64,
    pub cash_floor_breached: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityStressResult {
    Determined(LiquidityStressAssessment),
    Indeterminate { reason: &'static str },
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

/// Assess a point-in-time cash/cover position from six distinct evidence references.
/// All monetary inputs are minor ZAR. This deliberately does not infer that
/// cash above the floor can be transferred instantly or is accepted by the MO.
pub fn assess_liquidity_stress_snapshot(
    snapshot: LiquidityStressSnapshot<'_>,
) -> Result<LiquidityStressResult, LiquidityStressError> {
    let Some(required_cover) = snapshot.market_required_credit_cover_minor_zar else {
        return Ok(LiquidityStressResult::Indeterminate {
            reason: "Market Operator Required Credit Cover amount is missing",
        });
    };
    let Some(required_ref) = snapshot.required_cover_source_ref else {
        return Ok(LiquidityStressResult::Indeterminate {
            reason: "Required Credit Cover source reference is missing",
        });
    };

    let references = [
        snapshot.civil_calendar_source_ref,
        required_ref,
        snapshot.posted_cover_source_ref,
        snapshot.unrestricted_cash_source_ref,
        snapshot.mandatory_outflows_source_ref,
        snapshot.operating_cash_floor_source_ref,
    ];
    if references.iter().any(|reference| !nonempty(reference)) {
        return Err(LiquidityStressError::InvalidEvidenceReference);
    }
    for (index, reference) in references.iter().enumerate() {
        if references[..index].contains(reference) {
            return Err(LiquidityStressError::DuplicateEvidenceReference);
        }
    }
    if !nonempty(snapshot.scenario_id) {
        return Err(LiquidityStressError::InvalidEvidenceReference);
    }

    let credit_cover_shortfall =
        required_cover.saturating_sub(snapshot.posted_credit_cover_minor_zar);
    let cash_after_outflows = snapshot
        .unrestricted_cash_minor_zar
        .saturating_sub(snapshot.mandatory_noncover_outflows_minor_zar);
    let floor_deficit = snapshot
        .minimum_operating_cash_floor_minor_zar
        .saturating_sub(cash_after_outflows);
    let cash_headroom =
        cash_after_outflows.saturating_sub(snapshot.minimum_operating_cash_floor_minor_zar);
    let potentially_fundable = credit_cover_shortfall.min(cash_headroom);
    let external_gap = credit_cover_shortfall.saturating_sub(potentially_fundable);

    Ok(LiquidityStressResult::Determined(
        LiquidityStressAssessment {
            as_of_day_ordinal: snapshot.as_of_day_ordinal,
            market_required_credit_cover_minor_zar: required_cover,
            posted_credit_cover_minor_zar: snapshot.posted_credit_cover_minor_zar,
            cover_position: if credit_cover_shortfall == 0 {
                LiquidityCoverPosition::PostedCoverMeetsMarketRequirement
            } else {
                LiquidityCoverPosition::CoverShortfall
            },
            credit_cover_shortfall_minor_zar: credit_cover_shortfall,
            cash_after_mandatory_outflows_minor_zar: cash_after_outflows,
            operating_cash_floor_minor_zar: snapshot.minimum_operating_cash_floor_minor_zar,
            operating_cash_floor_deficit_minor_zar: floor_deficit,
            cash_available_above_floor_minor_zar: cash_headroom,
            potentially_fundable_cover_minor_zar: potentially_fundable,
            external_funding_gap_minor_zar: external_gap,
            cash_floor_breached: floor_deficit > 0,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> LiquidityStressSnapshot<'static> {
        LiquidityStressSnapshot {
            scenario_id: "base/2026-11-05",
            as_of_day_ordinal: 5,
            civil_calendar_source_ref: "calendar/sa/2026-v1",
            market_required_credit_cover_minor_zar: Some(20_000_000),
            required_cover_source_ref: Some("mo/required-cover/001"),
            posted_credit_cover_minor_zar: 15_000_000,
            posted_cover_source_ref: "mo/posted-cover/001",
            unrestricted_cash_minor_zar: 20_000_000,
            unrestricted_cash_source_ref: "treasury/unrestricted-cash/001",
            mandatory_noncover_outflows_minor_zar: 5_000_000,
            mandatory_outflows_source_ref: "treasury/due-outflows/001",
            minimum_operating_cash_floor_minor_zar: 10_000_000,
            operating_cash_floor_source_ref: "treasury/board-cash-floor/001",
        }
    }

    #[test]
    fn liquidity_snapshot_separates_cover_shortfall_from_fundable_cash_headroom() {
        let result = assess_liquidity_stress_snapshot(snapshot()).unwrap();
        assert_eq!(
            result,
            LiquidityStressResult::Determined(LiquidityStressAssessment {
                as_of_day_ordinal: 5,
                market_required_credit_cover_minor_zar: 20_000_000,
                posted_credit_cover_minor_zar: 15_000_000,
                cover_position: LiquidityCoverPosition::CoverShortfall,
                credit_cover_shortfall_minor_zar: 5_000_000,
                cash_after_mandatory_outflows_minor_zar: 15_000_000,
                operating_cash_floor_minor_zar: 10_000_000,
                operating_cash_floor_deficit_minor_zar: 0,
                cash_available_above_floor_minor_zar: 5_000_000,
                potentially_fundable_cover_minor_zar: 5_000_000,
                external_funding_gap_minor_zar: 0,
                cash_floor_breached: false,
            })
        );
    }

    #[test]
    fn potentially_fundable_cash_never_promotes_itself_to_posted_cover() {
        let assessment = assess_liquidity_stress_snapshot(snapshot()).unwrap();
        let LiquidityStressResult::Determined(assessment) = assessment else {
            panic!("all required snapshot inputs are present");
        };
        assert_eq!(assessment.credit_cover_shortfall_minor_zar, 5_000_000);
        assert_eq!(assessment.potentially_fundable_cover_minor_zar, 5_000_000);
        assert_eq!(assessment.posted_credit_cover_minor_zar, 15_000_000);
        assert_eq!(
            assessment.cover_position,
            LiquidityCoverPosition::CoverShortfall
        );
    }

    #[test]
    fn severe_scenario_reports_both_operating_floor_deficit_and_external_cover_gap() {
        let stressed = LiquidityStressSnapshot {
            unrestricted_cash_minor_zar: 12_000_000,
            mandatory_noncover_outflows_minor_zar: 5_000_000,
            ..snapshot()
        };
        let result = assess_liquidity_stress_snapshot(stressed).unwrap();
        let LiquidityStressResult::Determined(assessment) = result else {
            panic!("required cover is source-supplied");
        };
        assert_eq!(
            assessment.cash_after_mandatory_outflows_minor_zar,
            7_000_000
        );
        assert_eq!(assessment.operating_cash_floor_deficit_minor_zar, 3_000_000);
        assert!(assessment.cash_floor_breached);
        assert_eq!(assessment.cash_available_above_floor_minor_zar, 0);
        assert_eq!(assessment.potentially_fundable_cover_minor_zar, 0);
        assert_eq!(assessment.external_funding_gap_minor_zar, 5_000_000);
    }

    #[test]
    fn already_posted_cover_and_excess_cash_are_reported_without_false_shortfall() {
        let covered = LiquidityStressSnapshot {
            market_required_credit_cover_minor_zar: Some(20_000_000),
            posted_credit_cover_minor_zar: 20_000_000,
            unrestricted_cash_minor_zar: 30_000_000,
            ..snapshot()
        };
        let result = assess_liquidity_stress_snapshot(covered).unwrap();
        let LiquidityStressResult::Determined(assessment) = result else {
            panic!("cover state is source-supplied");
        };
        assert_eq!(
            assessment.cover_position,
            LiquidityCoverPosition::PostedCoverMeetsMarketRequirement
        );
        assert_eq!(assessment.credit_cover_shortfall_minor_zar, 0);
        assert_eq!(assessment.potentially_fundable_cover_minor_zar, 0);
        assert_eq!(assessment.cash_available_above_floor_minor_zar, 15_000_000);
    }

    #[test]
    fn missing_required_cover_is_indeterminate_not_zero() {
        let missing = LiquidityStressSnapshot {
            market_required_credit_cover_minor_zar: None,
            required_cover_source_ref: None,
            ..snapshot()
        };
        assert_eq!(
            assess_liquidity_stress_snapshot(missing),
            Ok(LiquidityStressResult::Indeterminate {
                reason: "Market Operator Required Credit Cover amount is missing"
            })
        );
    }

    #[test]
    fn duplicate_evidence_references_are_rejected() {
        let duplicate = LiquidityStressSnapshot {
            posted_cover_source_ref: "treasury/unrestricted-cash/001",
            ..snapshot()
        };
        assert_eq!(
            assess_liquidity_stress_snapshot(duplicate),
            Err(LiquidityStressError::DuplicateEvidenceReference)
        );
    }
}
