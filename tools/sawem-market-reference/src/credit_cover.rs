//! Evidence-qualified credit-cover exposure and remediation arithmetic.
//!
//! This module covers the narrow §15.10 boundary between Actual Exposure,
//! Undefined Potential Exposure, posted cover, and receipt-backed remediation.
//! It does not calculate the Market Operator's Undefined Exposure Factor (UEF),
//! authenticate upstream invoices/statements, model bank-provider credit, or
//! establish that a real collateral transfer has settled.

use super::CreditCoverDeadline;

/// An issued invoice or a Settlement Statement amount for which no invoice
/// has been issued. These are mutually exclusive representations of exposure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualExposureKind<'a> {
    UnpaidInvoice {
        invoice_id: &'a str,
        settlement_statement_ref: Option<&'a str>,
    },
    SettlementStatementWithoutInvoice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActualExposureEntry<'a> {
    /// Stable, namespaced reference, e.g. "invoice/..." or "statement/...".
    pub source_ref: &'a str,
    /// Outstanding balance, not the original gross invoice/statement amount.
    pub outstanding_minor_units: u64,
    pub kind: ActualExposureKind<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreditCoverExposureInputs<'a> {
    pub actual_exposure_entries: &'a [ActualExposureEntry<'a>],
    /// Supplied by a separately versioned/qualified exposure calculation.
    /// Missing UPE is not interpreted as zero.
    pub undefined_potential_exposure_minor_units: Option<u64>,
    pub undefined_potential_source_ref: Option<&'a str>,
    pub posted_credit_cover_minor_units: u64,
    /// Participant/MGC-approved warning threshold; None means not supplied.
    pub warning_limit_minor_units: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreditCoverPosition {
    pub actual_exposure_minor_units: u64,
    pub undefined_potential_exposure_minor_units: u64,
    pub required_credit_cover_minor_units: u64,
    pub posted_credit_cover_minor_units: u64,
    pub additional_cover_required_minor_units: u64,
    pub surplus_minor_units: u64,
    pub warning_notice_due: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreditCoverExposureResult {
    Determined(CreditCoverPosition),
    Indeterminate { reason: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreditCoverExposureError {
    InvalidEvidenceReference,
    DuplicateExposureReference,
    DuplicateInvoiceReference,
    InvoiceAndUninvoicedStatementDoubleCount,
    InvoiceNotOutstanding,
    InvoicePaymentExceedsOutstanding,
    DuplicateActionReference,
    InvalidFundingAction,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreditCoverFundingActionKind<'a> {
    AdditionalCollateralPosted,
    OutstandingInvoicePaidEarly { invoice_id: &'a str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreditCoverFundingActionState {
    Confirmed,
    Expected,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreditCoverFundingAction<'a> {
    pub action_ref: &'a str,
    pub kind: CreditCoverFundingActionKind<'a>,
    pub amount_minor_units: u64,
    pub day_ordinal: i32,
    pub minute_of_day: u16,
    pub state: CreditCoverFundingActionState,
    /// Confirmed actions must have a payment/posting receipt reference.
    pub receipt_ref: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreditCoverRemediationStatus {
    Satisfied,
    Shortfall,
    UnconfirmedFunding,
    DeadlineMissed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreditCoverRemediationResult {
    pub status: CreditCoverRemediationStatus,
    pub actual_exposure_after_early_payments_minor_units: u64,
    pub required_credit_cover_after_actions_minor_units: u64,
    pub posted_credit_cover_after_actions_minor_units: u64,
    pub confirmed_early_invoice_payments_minor_units: u64,
    pub confirmed_additional_collateral_minor_units: u64,
    pub remaining_shortfall_minor_units: u64,
    pub expected_action_count: u32,
    pub late_action_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreditCoverRemediationEvaluation {
    Determined(CreditCoverRemediationResult),
    Indeterminate { reason: &'static str },
}

fn nonempty_reference(value: &str) -> bool {
    !value.trim().is_empty()
}

/// Reconcile §15.10 Actual Exposure and supplied Undefined Potential Exposure.
/// An unpaid invoice and the same statement's no-invoice balance may never be
/// counted simultaneously.
pub fn calculate_credit_cover_exposure(
    inputs: CreditCoverExposureInputs<'_>,
) -> Result<CreditCoverExposureResult, CreditCoverExposureError> {
    let Some(undefined_potential) = inputs.undefined_potential_exposure_minor_units else {
        return Ok(CreditCoverExposureResult::Indeterminate {
            reason: "undefined-potential exposure amount is missing",
        });
    };
    let Some(undefined_potential_ref) = inputs.undefined_potential_source_ref else {
        return Ok(CreditCoverExposureResult::Indeterminate {
            reason: "undefined-potential exposure source reference is missing",
        });
    };
    if !nonempty_reference(undefined_potential_ref) {
        return Err(CreditCoverExposureError::InvalidEvidenceReference);
    }

    let mut actual_exposure = 0_u64;
    for (index, entry) in inputs.actual_exposure_entries.iter().enumerate() {
        if !nonempty_reference(entry.source_ref) {
            return Err(CreditCoverExposureError::InvalidEvidenceReference);
        }
        for previous in &inputs.actual_exposure_entries[..index] {
            if previous.source_ref == entry.source_ref {
                return Err(CreditCoverExposureError::DuplicateExposureReference);
            }
            if let (
                ActualExposureKind::UnpaidInvoice {
                    invoice_id: left, ..
                },
                ActualExposureKind::UnpaidInvoice {
                    invoice_id: right, ..
                },
            ) = (previous.kind, entry.kind)
            {
                if left == right {
                    return Err(CreditCoverExposureError::DuplicateInvoiceReference);
                }
            }
        }

        match entry.kind {
            ActualExposureKind::UnpaidInvoice {
                invoice_id,
                settlement_statement_ref,
            } => {
                if !nonempty_reference(invoice_id) {
                    return Err(CreditCoverExposureError::InvalidEvidenceReference);
                }
                if let Some(statement_ref) = settlement_statement_ref {
                    if !nonempty_reference(statement_ref) {
                        return Err(CreditCoverExposureError::InvalidEvidenceReference);
                    }
                    if inputs.actual_exposure_entries.iter().any(|other| {
                        matches!(
                            other.kind,
                            ActualExposureKind::SettlementStatementWithoutInvoice
                        ) && other.source_ref == statement_ref
                    }) {
                        return Err(
                            CreditCoverExposureError::InvoiceAndUninvoicedStatementDoubleCount,
                        );
                    }
                }
            }
            ActualExposureKind::SettlementStatementWithoutInvoice => {}
        }

        actual_exposure = actual_exposure
            .checked_add(entry.outstanding_minor_units)
            .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
    }

    let required = actual_exposure
        .checked_add(undefined_potential)
        .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
    let additional = required.saturating_sub(inputs.posted_credit_cover_minor_units);
    let surplus = inputs
        .posted_credit_cover_minor_units
        .saturating_sub(required);
    // A warning is due when the remaining cover buffer is at/below the
    // participant/MGC-authorised Warning Limit, including an actual shortfall.
    let warning_notice_due = inputs
        .warning_limit_minor_units
        .map(|warning_limit| surplus <= warning_limit);

    Ok(CreditCoverExposureResult::Determined(CreditCoverPosition {
        actual_exposure_minor_units: actual_exposure,
        undefined_potential_exposure_minor_units: undefined_potential,
        required_credit_cover_minor_units: required,
        posted_credit_cover_minor_units: inputs.posted_credit_cover_minor_units,
        additional_cover_required_minor_units: additional,
        surplus_minor_units: surplus,
        warning_notice_due,
    }))
}

/// Apply receipt-backed actions through the §15.10.2 deadline. Expected funding
/// does not count as Posted Credit Cover; early payment reduces only an issued,
/// unpaid invoice and never removes an un-invoiced Settlement Statement amount.
pub fn evaluate_credit_cover_remediation(
    inputs: CreditCoverExposureInputs<'_>,
    deadline: CreditCoverDeadline,
    actions: &[CreditCoverFundingAction<'_>],
) -> Result<CreditCoverRemediationEvaluation, CreditCoverExposureError> {
    let position = match calculate_credit_cover_exposure(inputs)? {
        CreditCoverExposureResult::Determined(position) => position,
        CreditCoverExposureResult::Indeterminate { reason } => {
            return Ok(CreditCoverRemediationEvaluation::Indeterminate { reason });
        }
    };

    let mut confirmed_invoice_payments = 0_u64;
    let mut confirmed_collateral = 0_u64;
    let mut expected_action_count = 0_u32;
    let mut late_action_count = 0_u32;

    for (index, action) in actions.iter().enumerate() {
        if !nonempty_reference(action.action_ref)
            || action.amount_minor_units == 0
            || action.minute_of_day >= 24 * 60
        {
            return Err(CreditCoverExposureError::InvalidFundingAction);
        }
        for previous in &actions[..index] {
            if previous.action_ref == action.action_ref {
                return Err(CreditCoverExposureError::DuplicateActionReference);
            }
        }
        if action.state == CreditCoverFundingActionState::Confirmed
            && action.receipt_ref.map(nonempty_reference) != Some(true)
        {
            return Err(CreditCoverExposureError::InvalidEvidenceReference);
        }

        if let CreditCoverFundingActionKind::OutstandingInvoicePaidEarly { invoice_id } =
            action.kind
        {
            if !nonempty_reference(invoice_id) {
                return Err(CreditCoverExposureError::InvalidEvidenceReference);
            }
            let invoice = inputs.actual_exposure_entries.iter().find(|entry| {
                matches!(
                    entry.kind,
                    ActualExposureKind::UnpaidInvoice {
                        invoice_id: existing,
                        ..
                    } if existing == invoice_id
                )
            });
            let Some(invoice) = invoice else {
                return Err(CreditCoverExposureError::InvoiceNotOutstanding);
            };

            let mut planned_payments = 0_u64;
            for candidate in actions {
                if candidate.state != CreditCoverFundingActionState::Failed {
                    if let CreditCoverFundingActionKind::OutstandingInvoicePaidEarly {
                        invoice_id: candidate_invoice,
                    } = candidate.kind
                    {
                        if candidate_invoice == invoice_id {
                            planned_payments = planned_payments
                                .checked_add(candidate.amount_minor_units)
                                .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
                        }
                    }
                }
            }
            if planned_payments > invoice.outstanding_minor_units {
                return Err(CreditCoverExposureError::InvoicePaymentExceedsOutstanding);
            }
        }

        let timely = action.day_ordinal < deadline.day_ordinal
            || (action.day_ordinal == deadline.day_ordinal
                && action.minute_of_day <= deadline.minute_of_day);
        match action.state {
            CreditCoverFundingActionState::Confirmed if timely => match action.kind {
                CreditCoverFundingActionKind::AdditionalCollateralPosted => {
                    confirmed_collateral = confirmed_collateral
                        .checked_add(action.amount_minor_units)
                        .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
                }
                CreditCoverFundingActionKind::OutstandingInvoicePaidEarly { .. } => {
                    confirmed_invoice_payments = confirmed_invoice_payments
                        .checked_add(action.amount_minor_units)
                        .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
                }
            },
            CreditCoverFundingActionState::Confirmed if !timely => {
                late_action_count = late_action_count.saturating_add(1);
            }
            CreditCoverFundingActionState::Expected if timely => {
                expected_action_count = expected_action_count.saturating_add(1);
            }
            CreditCoverFundingActionState::Expected if !timely => {
                late_action_count = late_action_count.saturating_add(1);
            }
            CreditCoverFundingActionState::Failed => {}
            _ => {}
        }
    }

    if confirmed_invoice_payments > position.actual_exposure_minor_units {
        return Err(CreditCoverExposureError::InvoicePaymentExceedsOutstanding);
    }
    let actual_after = position
        .actual_exposure_minor_units
        .checked_sub(confirmed_invoice_payments)
        .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
    let required_after = actual_after
        .checked_add(position.undefined_potential_exposure_minor_units)
        .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
    let posted_after = position
        .posted_credit_cover_minor_units
        .checked_add(confirmed_collateral)
        .ok_or(CreditCoverExposureError::ArithmeticOverflow)?;
    let remaining_shortfall = required_after.saturating_sub(posted_after);

    let status = if remaining_shortfall == 0 {
        CreditCoverRemediationStatus::Satisfied
    } else if late_action_count > 0 {
        CreditCoverRemediationStatus::DeadlineMissed
    } else if expected_action_count > 0 {
        CreditCoverRemediationStatus::UnconfirmedFunding
    } else {
        CreditCoverRemediationStatus::Shortfall
    };

    Ok(CreditCoverRemediationEvaluation::Determined(
        CreditCoverRemediationResult {
            status,
            actual_exposure_after_early_payments_minor_units: actual_after,
            required_credit_cover_after_actions_minor_units: required_after,
            posted_credit_cover_after_actions_minor_units: posted_after,
            confirmed_early_invoice_payments_minor_units: confirmed_invoice_payments,
            confirmed_additional_collateral_minor_units: confirmed_collateral,
            remaining_shortfall_minor_units: remaining_shortfall,
            expected_action_count,
            late_action_count,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exposure_entries() -> [ActualExposureEntry<'static>; 2] {
        [
            ActualExposureEntry {
                source_ref: "invoice/INV-001",
                outstanding_minor_units: 10_000,
                kind: ActualExposureKind::UnpaidInvoice {
                    invoice_id: "INV-001",
                    settlement_statement_ref: Some("statement/SET-001"),
                },
            },
            ActualExposureEntry {
                source_ref: "statement/SET-002",
                outstanding_minor_units: 5_000,
                kind: ActualExposureKind::SettlementStatementWithoutInvoice,
            },
        ]
    }

    fn exposure_inputs<'a>(
        entries: &'a [ActualExposureEntry<'a>],
    ) -> CreditCoverExposureInputs<'a> {
        CreditCoverExposureInputs {
            actual_exposure_entries: entries,
            undefined_potential_exposure_minor_units: Some(5_000),
            undefined_potential_source_ref: Some("upe/model-receipt-001"),
            posted_credit_cover_minor_units: 10_000,
            warning_limit_minor_units: Some(2_000),
        }
    }

    #[test]
    fn exposure_reconciles_unpaid_invoices_and_uninvoiced_statements() {
        let entries = exposure_entries();
        let result = calculate_credit_cover_exposure(exposure_inputs(&entries))
            .expect("source-backed synthetic exposure entries are valid");
        assert_eq!(
            result,
            CreditCoverExposureResult::Determined(CreditCoverPosition {
                actual_exposure_minor_units: 15_000,
                undefined_potential_exposure_minor_units: 5_000,
                required_credit_cover_minor_units: 20_000,
                posted_credit_cover_minor_units: 10_000,
                additional_cover_required_minor_units: 10_000,
                surplus_minor_units: 0,
                warning_notice_due: Some(true),
            })
        );
    }

    #[test]
    fn exposure_keeps_missing_upe_indeterminate_and_rejects_statement_double_count() {
        let entries = exposure_entries();
        let missing = CreditCoverExposureInputs {
            undefined_potential_exposure_minor_units: None,
            undefined_potential_source_ref: None,
            ..exposure_inputs(&entries)
        };
        assert_eq!(
            calculate_credit_cover_exposure(missing),
            Ok(CreditCoverExposureResult::Indeterminate {
                reason: "undefined-potential exposure amount is missing",
            })
        );

        let double_counted = [
            entries[0],
            entries[1],
            ActualExposureEntry {
                source_ref: "invoice/INV-002",
                outstanding_minor_units: 2_000,
                kind: ActualExposureKind::UnpaidInvoice {
                    invoice_id: "INV-002",
                    settlement_statement_ref: Some("statement/SET-002"),
                },
            },
        ];
        assert_eq!(
            calculate_credit_cover_exposure(exposure_inputs(&double_counted)),
            Err(CreditCoverExposureError::InvoiceAndUninvoicedStatementDoubleCount)
        );
    }

    #[test]
    fn early_invoice_payment_and_new_collateral_can_satisfy_the_same_increase_notice() {
        let entries = exposure_entries();
        let inputs = exposure_inputs(&entries);
        let actions = [
            CreditCoverFundingAction {
                action_ref: "payment-receipt/001",
                kind: CreditCoverFundingActionKind::OutstandingInvoicePaidEarly {
                    invoice_id: "INV-001",
                },
                amount_minor_units: 5_000,
                day_ordinal: 11,
                minute_of_day: 900,
                state: CreditCoverFundingActionState::Confirmed,
                receipt_ref: Some("bank-receipt/001"),
            },
            CreditCoverFundingAction {
                action_ref: "collateral-receipt/001",
                kind: CreditCoverFundingActionKind::AdditionalCollateralPosted,
                amount_minor_units: 5_000,
                day_ordinal: 12,
                minute_of_day: 1_020,
                state: CreditCoverFundingActionState::Confirmed,
                receipt_ref: Some("market-operator-receipt/001"),
            },
        ];
        let result = evaluate_credit_cover_remediation(
            inputs,
            CreditCoverDeadline {
                day_ordinal: 12,
                minute_of_day: 1_020,
            },
            &actions,
        )
        .expect("both remediation actions have receipts by the deadline");
        assert_eq!(
            result,
            CreditCoverRemediationEvaluation::Determined(CreditCoverRemediationResult {
                status: CreditCoverRemediationStatus::Satisfied,
                actual_exposure_after_early_payments_minor_units: 10_000,
                required_credit_cover_after_actions_minor_units: 15_000,
                posted_credit_cover_after_actions_minor_units: 15_000,
                confirmed_early_invoice_payments_minor_units: 5_000,
                confirmed_additional_collateral_minor_units: 5_000,
                remaining_shortfall_minor_units: 0,
                expected_action_count: 0,
                late_action_count: 0,
            })
        );
    }

    #[test]
    fn expected_cover_does_not_count_and_late_cover_is_excluded() {
        let entries = exposure_entries();
        let inputs = exposure_inputs(&entries);
        let expected = [CreditCoverFundingAction {
            action_ref: "planned-guarantee/001",
            kind: CreditCoverFundingActionKind::AdditionalCollateralPosted,
            amount_minor_units: 10_000,
            day_ordinal: 12,
            minute_of_day: 900,
            state: CreditCoverFundingActionState::Expected,
            receipt_ref: None,
        }];
        let pending = evaluate_credit_cover_remediation(
            inputs,
            CreditCoverDeadline {
                day_ordinal: 12,
                minute_of_day: 1_020,
            },
            &expected,
        )
        .expect("expected cover remains a pending action");
        let CreditCoverRemediationEvaluation::Determined(pending) = pending else {
            panic!("base exposure was fully specified");
        };
        assert_eq!(
            pending.status,
            CreditCoverRemediationStatus::UnconfirmedFunding
        );
        assert_eq!(
            pending.posted_credit_cover_after_actions_minor_units,
            10_000
        );
        assert_eq!(pending.remaining_shortfall_minor_units, 10_000);

        let late = [CreditCoverFundingAction {
            action_ref: "late-collateral/001",
            kind: CreditCoverFundingActionKind::AdditionalCollateralPosted,
            amount_minor_units: 10_000,
            day_ordinal: 13,
            minute_of_day: 600,
            state: CreditCoverFundingActionState::Confirmed,
            receipt_ref: Some("late-receipt/001"),
        }];
        let missed = evaluate_credit_cover_remediation(
            inputs,
            CreditCoverDeadline {
                day_ordinal: 12,
                minute_of_day: 1_020,
            },
            &late,
        )
        .expect("late collateral does not satisfy the earlier deadline");
        let CreditCoverRemediationEvaluation::Determined(missed) = missed else {
            panic!("base exposure was fully specified");
        };
        assert_eq!(missed.status, CreditCoverRemediationStatus::DeadlineMissed);
        assert_eq!(missed.posted_credit_cover_after_actions_minor_units, 10_000);
    }

    #[test]
    fn remediation_rejects_overpayment_and_confirmed_actions_without_receipts() {
        let entries = exposure_entries();
        let inputs = exposure_inputs(&entries);
        let overpayment = [CreditCoverFundingAction {
            action_ref: "overpayment/001",
            kind: CreditCoverFundingActionKind::OutstandingInvoicePaidEarly {
                invoice_id: "INV-001",
            },
            amount_minor_units: 10_001,
            day_ordinal: 11,
            minute_of_day: 900,
            state: CreditCoverFundingActionState::Confirmed,
            receipt_ref: Some("receipt/overpayment"),
        }];
        assert_eq!(
            evaluate_credit_cover_remediation(
                inputs,
                CreditCoverDeadline {
                    day_ordinal: 12,
                    minute_of_day: 1_020,
                },
                &overpayment,
            ),
            Err(CreditCoverExposureError::InvoicePaymentExceedsOutstanding)
        );

        let missing_receipt = [CreditCoverFundingAction {
            action_ref: "no-receipt/001",
            kind: CreditCoverFundingActionKind::AdditionalCollateralPosted,
            amount_minor_units: 10_000,
            day_ordinal: 11,
            minute_of_day: 900,
            state: CreditCoverFundingActionState::Confirmed,
            receipt_ref: None,
        }];
        assert_eq!(
            evaluate_credit_cover_remediation(
                inputs,
                CreditCoverDeadline {
                    day_ordinal: 12,
                    minute_of_day: 1_020,
                },
                &missing_receipt,
            ),
            Err(CreditCoverExposureError::InvalidEvidenceReference)
        );
    }
}
