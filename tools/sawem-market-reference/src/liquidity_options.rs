//! Economic comparison of the two Credit Cover Increase Notice satisfaction
//! paths expressly listed in SAWEM Market Code Rev 2.3 §15.10.2(3):
//! additional cash held in a Collateral Reserve Account or early payment of an
//! issued outstanding invoice.
//!
//! This module is a deterministic, simple-interest scenario calculator. It is
//! not a bank product eligibility verifier, Market Operator authorization,
//! interest-rate forecast, tax calculation, or proof that cash is available.

const COST_DENOMINATOR: u128 = 365 * 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EligibilityEvidence {
    Verified,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityOptionMethod {
    CashCollateralDeposit,
    EarlyInvoicePayment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactLiquidityCost {
    /// Signed minor currency units. Negative means modeled net carry benefit.
    pub numerator_minor_units: i128,
    pub denominator: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiquidityOptionAssessment {
    pub method: LiquidityOptionMethod,
    /// Required-credit-cover reduction supplied by the option.
    pub cover_relief_minor_units: u64,
    /// Immediate cash transferred/committed by the option.
    pub immediate_cash_use_minor_units: u64,
    /// Cash that remains restricted in the Collateral Reserve Account.
    /// This is zero for early invoice payment.
    pub restricted_cash_minor_units: u64,
    /// Outstanding invoice balance removed by early payment.
    /// This is zero for a collateral deposit.
    pub invoice_liability_reduced_minor_units: u64,
    /// Nominal simple carry/financing cost including supplied fees.
    pub net_carry_cost: ExactLiquidityCost,
    /// Explicit holding/advance horizon used in the scenario.
    pub horizon_days: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityOptionResult {
    Determined(LiquidityOptionAssessment),
    Indeterminate { reason: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityOptionError {
    ZeroCoverRelief,
    InvalidEvidenceReference,
    IneligibleCreditCoverProvider,
    CreditCoverReserveAccountNotEstablished,
    MissingInvoice,
    InvoicePaymentExceedsOutstanding,
    MissingAnnualRateSource,
    ArithmeticOverflow,
    OptionsCoverDifferentAmounts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CashCollateralScenario<'a> {
    pub cover_relief_minor_units: u64,
    pub horizon_days: u32,
    /// Participant's marginal cash/funding opportunity cost, annual basis points.
    pub annual_funding_cost_bps: Option<u32>,
    pub funding_cost_source_ref: Option<&'a str>,
    /// Actual Collateral Reserve Account yield, annual basis points.
    /// It is a supplied scenario input, not a Code-guaranteed rate.
    pub annual_account_interest_bps: Option<u32>,
    pub account_interest_source_ref: Option<&'a str>,
    pub provider_eligibility: EligibilityEvidence,
    pub reserve_account_established: EligibilityEvidence,
    pub setup_and_transaction_fees_minor_units: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlyInvoicePaymentScenario<'a> {
    pub invoice_id: &'a str,
    pub invoice_outstanding_minor_units: u64,
    pub cover_relief_minor_units: u64,
    /// Number of days by which payment is advanced relative to the invoice due date.
    pub days_early: u32,
    pub annual_funding_cost_bps: Option<u32>,
    pub funding_cost_source_ref: Option<&'a str>,
    pub transaction_fees_minor_units: u64,
}

fn nonempty_reference(value: &str) -> bool {
    !value.trim().is_empty()
}

fn checked_cost(
    relief_minor_units: u64,
    annual_spread_bps: i64,
    horizon_days: u32,
    fees_minor_units: u64,
) -> Result<ExactLiquidityCost, LiquidityOptionError> {
    let carry_numerator = i128::from(relief_minor_units)
        .checked_mul(i128::from(annual_spread_bps))
        .and_then(|value| value.checked_mul(i128::from(horizon_days)))
        .ok_or(LiquidityOptionError::ArithmeticOverflow)?;
    let fees_numerator = i128::from(fees_minor_units)
        .checked_mul(i128::try_from(COST_DENOMINATOR).map_err(|_| {
            LiquidityOptionError::ArithmeticOverflow
        })?)
        .ok_or(LiquidityOptionError::ArithmeticOverflow)?;
    let numerator = carry_numerator
        .checked_add(fees_numerator)
        .ok_or(LiquidityOptionError::ArithmeticOverflow)?;
    Ok(ExactLiquidityCost {
        numerator_minor_units: numerator,
        denominator: COST_DENOMINATOR,
    })
}

/// Model cash collateral in the Market Bank's Collateral Reserve Account.
/// A qualifying bank and established account are required in the input evidence;
/// missing evidence stays indeterminate, rejected evidence cannot be selected.
pub fn assess_cash_collateral_option(
    scenario: CashCollateralScenario<'_>,
) -> Result<LiquidityOptionResult, LiquidityOptionError> {
    if scenario.cover_relief_minor_units == 0 {
        return Err(LiquidityOptionError::ZeroCoverRelief);
    }
    match scenario.provider_eligibility {
        EligibilityEvidence::Rejected => {
            return Err(LiquidityOptionError::IneligibleCreditCoverProvider);
        }
        EligibilityEvidence::Unknown => {
            return Ok(LiquidityOptionResult::Indeterminate {
                reason: "Credit Cover Provider eligibility is unverified",
            });
        }
        EligibilityEvidence::Verified => {}
    }
    match scenario.reserve_account_established {
        EligibilityEvidence::Rejected => {
            return Err(LiquidityOptionError::CreditCoverReserveAccountNotEstablished);
        }
        EligibilityEvidence::Unknown => {
            return Ok(LiquidityOptionResult::Indeterminate {
                reason: "Collateral Reserve Account establishment is unverified",
            });
        }
        EligibilityEvidence::Verified => {}
    }

    let Some(funding_rate) = scenario.annual_funding_cost_bps else {
        return Ok(LiquidityOptionResult::Indeterminate {
            reason: "participant annual funding/opportunity-cost rate is missing",
        });
    };
    let Some(funding_source) = scenario.funding_cost_source_ref else {
        return Err(LiquidityOptionError::MissingAnnualRateSource);
    };
    if !nonempty_reference(funding_source) {
        return Err(LiquidityOptionError::InvalidEvidenceReference);
    }
    let Some(account_rate) = scenario.annual_account_interest_bps else {
        return Ok(LiquidityOptionResult::Indeterminate {
            reason: "Collateral Reserve Account interest rate is missing",
        });
    };
    let Some(account_source) = scenario.account_interest_source_ref else {
        return Err(LiquidityOptionError::MissingAnnualRateSource);
    };
    if !nonempty_reference(account_source) {
        return Err(LiquidityOptionError::InvalidEvidenceReference);
    }

    let spread = i64::from(funding_rate) - i64::from(account_rate);
    let cost = checked_cost(
        scenario.cover_relief_minor_units,
        spread,
        scenario.horizon_days,
        scenario.setup_and_transaction_fees_minor_units,
    )?;

    Ok(LiquidityOptionResult::Determined(
        LiquidityOptionAssessment {
            method: LiquidityOptionMethod::CashCollateralDeposit,
            cover_relief_minor_units: scenario.cover_relief_minor_units,
            immediate_cash_use_minor_units: scenario.cover_relief_minor_units,
            restricted_cash_minor_units: scenario.cover_relief_minor_units,
            invoice_liability_reduced_minor_units: 0,
            net_carry_cost: cost,
            horizon_days: scenario.horizon_days,
        },
    ))
}

/// Model paying a specific issued, outstanding invoice before its due date.
/// A statement for which no invoice has been issued is not eligible for this
/// path. This function only models the financing cost of advancing payment.
pub fn assess_early_invoice_payment_option(
    scenario: EarlyInvoicePaymentScenario<'_>,
) -> Result<LiquidityOptionResult, LiquidityOptionError> {
    if !nonempty_reference(scenario.invoice_id) {
        return Err(LiquidityOptionError::MissingInvoice);
    }
    if scenario.cover_relief_minor_units == 0 {
        return Err(LiquidityOptionError::ZeroCoverRelief);
    }
    if scenario.cover_relief_minor_units > scenario.invoice_outstanding_minor_units {
        return Err(LiquidityOptionError::InvoicePaymentExceedsOutstanding);
    }
    let Some(funding_rate) = scenario.annual_funding_cost_bps else {
        return Ok(LiquidityOptionResult::Indeterminate {
            reason: "participant annual funding/opportunity-cost rate is missing",
        });
    };
    let Some(funding_source) = scenario.funding_cost_source_ref else {
        return Err(LiquidityOptionError::MissingAnnualRateSource);
    };
    if !nonempty_reference(funding_source) {
        return Err(LiquidityOptionError::InvalidEvidenceReference);
    }

    let cost = checked_cost(
        scenario.cover_relief_minor_units,
        i64::from(funding_rate),
        scenario.days_early,
        scenario.transaction_fees_minor_units,
    )?;

    Ok(LiquidityOptionResult::Determined(
        LiquidityOptionAssessment {
            method: LiquidityOptionMethod::EarlyInvoicePayment,
            cover_relief_minor_units: scenario.cover_relief_minor_units,
            immediate_cash_use_minor_units: scenario.cover_relief_minor_units,
            restricted_cash_minor_units: 0,
            invoice_liability_reduced_minor_units: scenario.cover_relief_minor_units,
            net_carry_cost: cost,
            horizon_days: scenario.days_early,
        },
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerCostOption {
    CashCollateralDeposit,
    EarlyInvoicePayment,
    Equal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityOptionComparison {
    Determined {
        lower_cost_option: LowerCostOption,
        compared_cover_relief_minor_units: u64,
    },
    Indeterminate { reason: &'static str },
}

/// Compare two determined scenarios only when they provide the same amount of
/// Required Credit Cover relief. Exact cross multiplication avoids floating point.
pub fn compare_liquidity_options(
    cash: LiquidityOptionResult,
    early_payment: LiquidityOptionResult,
) -> Result<LiquidityOptionComparison, LiquidityOptionError> {
    let (LiquidityOptionResult::Determined(cash), LiquidityOptionResult::Determined(early)) =
        (cash, early_payment)
    else {
        return Ok(LiquidityOptionComparison::Indeterminate {
            reason: "one or both funding options remain indeterminate",
        });
    };
    if cash.cover_relief_minor_units != early.cover_relief_minor_units {
        return Err(LiquidityOptionError::OptionsCoverDifferentAmounts);
    }

    let cash_left = cash
        .net_carry_cost
        .numerator_minor_units
        .checked_mul(i128::try_from(early.net_carry_cost.denominator).map_err(|_| {
            LiquidityOptionError::ArithmeticOverflow
        })?)
        .ok_or(LiquidityOptionError::ArithmeticOverflow)?;
    let early_left = early
        .net_carry_cost
        .numerator_minor_units
        .checked_mul(i128::try_from(cash.net_carry_cost.denominator).map_err(|_| {
            LiquidityOptionError::ArithmeticOverflow
        })?)
        .ok_or(LiquidityOptionError::ArithmeticOverflow)?;

    let lower_cost_option = match cash_left.cmp(&early_left) {
        std::cmp::Ordering::Less => LowerCostOption::CashCollateralDeposit,
        std::cmp::Ordering::Greater => LowerCostOption::EarlyInvoicePayment,
        std::cmp::Ordering::Equal => LowerCostOption::Equal,
    };
    Ok(LiquidityOptionComparison::Determined {
        lower_cost_option,
        compared_cover_relief_minor_units: cash.cover_relief_minor_units,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cash_scenario(horizon_days: u32) -> CashCollateralScenario<'static> {
        CashCollateralScenario {
            cover_relief_minor_units: 50_000_000, // R500,000
            horizon_days,
            annual_funding_cost_bps: Some(1_200), // 12%
            funding_cost_source_ref: Some("participant-treasury-rate/001"),
            annual_account_interest_bps: Some(800), // 8%
            account_interest_source_ref: Some("market-bank-rate-sheet/001"),
            provider_eligibility: EligibilityEvidence::Verified,
            reserve_account_established: EligibilityEvidence::Verified,
            setup_and_transaction_fees_minor_units: 0,
        }
    }

    fn early_scenario(days_early: u32) -> EarlyInvoicePaymentScenario<'static> {
        EarlyInvoicePaymentScenario {
            invoice_id: "invoice/INV-002",
            invoice_outstanding_minor_units: 50_000_000,
            cover_relief_minor_units: 50_000_000,
            days_early,
            annual_funding_cost_bps: Some(1_200),
            funding_cost_source_ref: Some("participant-treasury-rate/001"),
            transaction_fees_minor_units: 0,
        }
    }

    #[test]
    fn cash_collateral_cost_nets_funding_cost_against_account_interest() {
        let result = assess_cash_collateral_option(cash_scenario(90))
            .expect("complete synthetic cash collateral scenario");
        let LiquidityOptionResult::Determined(assessment) = result else {
            panic!("the synthetic account and rate evidence is complete");
        };
        assert_eq!(assessment.restricted_cash_minor_units, 50_000_000);
        assert_eq!(assessment.net_carry_cost.numerator_minor_units, 180_000_000_000_000);
        assert_eq!(assessment.net_carry_cost.denominator, COST_DENOMINATOR);
    }

    #[test]
    fn early_invoice_payment_has_no_collateral_lockup_but_uses_advance_cost() {
        let result = assess_early_invoice_payment_option(early_scenario(20))
            .expect("issued invoice and funding rate are supplied");
        let LiquidityOptionResult::Determined(assessment) = result else {
            panic!("the synthetic invoice is outstanding");
        };
        assert_eq!(assessment.restricted_cash_minor_units, 0);
        assert_eq!(assessment.invoice_liability_reduced_minor_units, 50_000_000);
        assert_eq!(assessment.net_carry_cost.numerator_minor_units, 1_200_000_000_000);
    }

    #[test]
    fn cash_collateral_cost_increases_with_lockup_horizon() {
        let short = assess_cash_collateral_option(cash_scenario(90)).unwrap();
        let long = assess_cash_collateral_option(cash_scenario(365)).unwrap();
        let LiquidityOptionResult::Determined(short) = short else {
            panic!("short scenario is determined");
        };
        let LiquidityOptionResult::Determined(long) = long else {
            panic!("long scenario is determined");
        };
        assert!(long.net_carry_cost.numerator_minor_units > short.net_carry_cost.numerator_minor_units);
    }

    #[test]
    fn funding_option_comparison_requires_equal_relief_and_ranks_exactly() {
        let cash_90 = assess_cash_collateral_option(cash_scenario(90)).unwrap();
        let early_20 = assess_early_invoice_payment_option(early_scenario(20)).unwrap();
        assert_eq!(
            compare_liquidity_options(cash_90, early_20),
            Ok(LiquidityOptionComparison::Determined {
                lower_cost_option: LowerCostOption::CashCollateralDeposit,
                compared_cover_relief_minor_units: 50_000_000,
            })
        );

        let cash_365 = assess_cash_collateral_option(cash_scenario(365)).unwrap();
        assert_eq!(
            compare_liquidity_options(cash_365, early_20),
            Ok(LiquidityOptionComparison::Determined {
                lower_cost_option: LowerCostOption::EarlyInvoicePayment,
                compared_cover_relief_minor_units: 50_000_000,
            })
        );
    }

    #[test]
    fn unknown_bank_account_or_interest_inputs_remain_indeterminate() {
        let mut scenario = cash_scenario(90);
        scenario.provider_eligibility = EligibilityEvidence::Unknown;
        assert_eq!(
            assess_cash_collateral_option(scenario),
            Ok(LiquidityOptionResult::Indeterminate {
                reason: "Credit Cover Provider eligibility is unverified"
            })
        );
        scenario = cash_scenario(90);
        scenario.annual_account_interest_bps = None;
        assert_eq!(
            assess_cash_collateral_option(scenario),
            Ok(LiquidityOptionResult::Indeterminate {
                reason: "Collateral Reserve Account interest rate is missing"
            })
        );
    }

    #[test]
    fn guarantees_are_not_silently_admitted_as_a_credit_cover_instrument() {
        // The public Rev 2.3 profile has no generic bank-guarantee/LC option:
        // only a cash-held Collateral Reserve Account deposit is modelled here.
        assert_eq!(
            assess_cash_collateral_option(CashCollateralScenario {
                provider_eligibility: EligibilityEvidence::Rejected,
                ..cash_scenario(90)
            }),
            Err(LiquidityOptionError::IneligibleCreditCoverProvider)
        );
    }

    #[test]
    fn early_invoice_payment_rejects_unissued_or_overpaid_invoice_inputs() {
        assert_eq!(
            assess_early_invoice_payment_option(EarlyInvoicePaymentScenario {
                invoice_id: "",
                ..early_scenario(20)
            }),
            Err(LiquidityOptionError::MissingInvoice)
        );
        assert_eq!(
            assess_early_invoice_payment_option(EarlyInvoicePaymentScenario {
                cover_relief_minor_units: 50_000_001,
                ..early_scenario(20)
            }),
            Err(LiquidityOptionError::InvoicePaymentExceedsOutstanding)
        );
    }
}
