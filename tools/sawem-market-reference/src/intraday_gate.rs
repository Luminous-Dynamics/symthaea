//! Source-version-bound SAWEM Intraday Market gate-closure evaluation.
//!
//! The caller supplies UTC gate times from the selected Market Code / Trading
//! Rule revision after explicit source-clock conversion. A submission is tested
//! using the Market Operator's receipt time, not a participant workstation
//! clock. Missing MO receipt evidence remains indeterminate.
//!
//! This module checks only whether a re-declaration can enter the named auction
//! or a later remaining-hours auction. It does not construct revised schedules,
//! clear the Intraday Market, authenticate receipts or determine legal meaning.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedeclarationKind {
    DeclaredAvailableCapacity,
    DeclaredConsumption,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntradayGateEvidence<'a> {
    pub gate_id: &'a str,
    pub rule_revision_id: &'a str,
    pub rule_source_ref: &'a str,
    pub rule_source_sha256: &'a str,
    pub trading_unit_id: &'a str,
    pub trading_day_id: &'a str,
    pub target_trading_period_id: &'a str,
    pub target_period_start_utc: i64,
    pub target_period_end_utc: i64,
    pub gate_close_utc: i64,
    pub next_gate_id: Option<&'a str>,
    pub next_gate_close_utc: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RedeclarationSubmission<'a> {
    pub submission_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub rule_revision_id: &'a str,
    pub trading_unit_id: &'a str,
    pub trading_day_id: &'a str,
    pub target_trading_period_id: &'a str,
    pub kind: RedeclarationKind,
    /// Participant-side submission timestamp; not the authoritative gate test.
    pub participant_submitted_at_utc: i64,
    /// Timestamp from the Market Operator receipt/acknowledgement.
    pub market_operator_received_at_utc: Option<i64>,
    pub market_operator_receipt_ref: Option<&'a str>,
    pub market_operator_receipt_sha256: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntradayGateDisposition<'a> {
    IncludedInNamedAuction {
        gate_id: &'a str,
        received_at_utc: i64,
    },
    MissedNamedAuctionEligibleForNext {
        missed_gate_id: &'a str,
        next_gate_id: &'a str,
        received_at_utc: i64,
    },
    MissedAllRemainingGates {
        missed_gate_id: &'a str,
        received_at_utc: i64,
    },
    Indeterminate {
        reason: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntradayGateError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidGateWindow,
    InconsistentNextGate,
    InvalidSubmissionChronology,
    RuleRevisionMismatch,
    TradingUnitMismatch,
    TradingDayMismatch,
    TradingPeriodMismatch,
    IncompleteOperatorReceipt,
    ArithmeticOverflow,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_gate(gate: IntradayGateEvidence<'_>) -> Result<(), IntradayGateError> {
    if !nonempty(gate.gate_id)
        || !nonempty(gate.rule_revision_id)
        || !nonempty(gate.rule_source_ref)
        || !nonempty(gate.trading_unit_id)
        || !nonempty(gate.trading_day_id)
        || !nonempty(gate.target_trading_period_id)
    {
        return Err(IntradayGateError::InvalidEvidenceReference);
    }
    if !valid_sha256(gate.rule_source_sha256) {
        return Err(IntradayGateError::InvalidSourceDigest);
    }
    if gate.target_period_start_utc >= gate.target_period_end_utc
        || gate.gate_close_utc > gate.target_period_start_utc
    {
        return Err(IntradayGateError::InvalidGateWindow);
    }
    match (gate.next_gate_id, gate.next_gate_close_utc) {
        (None, None) => {}
        (Some(next_id), Some(next_close)) => {
            if !nonempty(next_id) || next_id == gate.gate_id {
                return Err(IntradayGateError::InconsistentNextGate);
            }
            if next_close <= gate.gate_close_utc || next_close > gate.target_period_start_utc {
                return Err(IntradayGateError::InconsistentNextGate);
            }
        }
        _ => return Err(IntradayGateError::InconsistentNextGate),
    }
    Ok(())
}

fn validate_submission(
    submission: RedeclarationSubmission<'_>,
) -> Result<Option<i64>, IntradayGateError> {
    if !nonempty(submission.submission_id)
        || !nonempty(submission.source_ref)
        || !nonempty(submission.rule_revision_id)
        || !nonempty(submission.trading_unit_id)
        || !nonempty(submission.trading_day_id)
        || !nonempty(submission.target_trading_period_id)
    {
        return Err(IntradayGateError::InvalidEvidenceReference);
    }
    if !valid_sha256(submission.source_sha256) {
        return Err(IntradayGateError::InvalidSourceDigest);
    }
    let Some(received_at) = submission.market_operator_received_at_utc else {
        if submission.market_operator_receipt_ref.is_some()
            || submission.market_operator_receipt_sha256.is_some()
        {
            return Err(IntradayGateError::IncompleteOperatorReceipt);
        }
        return Ok(None);
    };
    let Some(receipt_ref) = submission.market_operator_receipt_ref else {
        return Err(IntradayGateError::IncompleteOperatorReceipt);
    };
    let Some(receipt_sha256) = submission.market_operator_receipt_sha256 else {
        return Err(IntradayGateError::IncompleteOperatorReceipt);
    };
    if !nonempty(receipt_ref) || !valid_sha256(receipt_sha256) {
        return Err(IntradayGateError::IncompleteOperatorReceipt);
    }
    if received_at < submission.participant_submitted_at_utc {
        return Err(IntradayGateError::InvalidSubmissionChronology);
    }
    Ok(Some(received_at))
}

/// Determine whether the Operator-received re-declaration was eligible for the
/// named six-hour auction, a later auction before the target Trading Period, or
/// no remaining auction for that period. Gate equality is inclusive because
/// the Code says submissions prior to/up to gate closure enter the auction;
/// this evaluator treats a timestamp equal to closure as on time only when the
/// Market Operator receipt itself is exactly timestamped at that boundary.
pub fn assess_intraday_redeclaration<'a>(
    gate: IntradayGateEvidence<'a>,
    submission: RedeclarationSubmission<'a>,
) -> Result<IntradayGateDisposition<'a>, IntradayGateError> {
    validate_gate(gate)?;
    if submission.rule_revision_id != gate.rule_revision_id {
        return Err(IntradayGateError::RuleRevisionMismatch);
    }
    if submission.trading_unit_id != gate.trading_unit_id {
        return Err(IntradayGateError::TradingUnitMismatch);
    }
    if submission.trading_day_id != gate.trading_day_id {
        return Err(IntradayGateError::TradingDayMismatch);
    }
    if submission.target_trading_period_id != gate.target_trading_period_id {
        return Err(IntradayGateError::TradingPeriodMismatch);
    }
    let Some(received_at) = validate_submission(submission)? else {
        return Ok(IntradayGateDisposition::Indeterminate {
            reason: "Market Operator receipt timestamp is missing",
        });
    };
    if received_at <= gate.gate_close_utc {
        return Ok(IntradayGateDisposition::IncludedInNamedAuction {
            gate_id: gate.gate_id,
            received_at_utc: received_at,
        });
    }
    match (gate.next_gate_id, gate.next_gate_close_utc) {
        (Some(next_gate_id), Some(next_gate_close)) if received_at <= next_gate_close => {
            Ok(IntradayGateDisposition::MissedNamedAuctionEligibleForNext {
                missed_gate_id: gate.gate_id,
                next_gate_id,
                received_at_utc: received_at,
            })
        }
        _ => Ok(IntradayGateDisposition::MissedAllRemainingGates {
            missed_gate_id: gate.gate_id,
            received_at_utc: received_at,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SUBMISSION_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const RECEIPT_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn gate() -> IntradayGateEvidence<'static> {
        IntradayGateEvidence {
            gate_id: "idm/2026-11-06/06:00",
            rule_revision_id: "SAWEM-MC-2.3",
            rule_source_ref: "ntcsa/market-code/rev2.3",
            rule_source_sha256: RULE_SHA,
            trading_unit_id: "TU/IPP-001",
            trading_day_id: "2026-11-06",
            target_trading_period_id: "2026-11-06/12:00-13:00",
            target_period_start_utc: 12 * 3_600,
            target_period_end_utc: 13 * 3_600,
            gate_close_utc: 6 * 3_600,
            next_gate_id: Some("idm/2026-11-06/12:00"),
            next_gate_close_utc: Some(12 * 3_600),
        }
    }

    fn submission(received: Option<i64>) -> RedeclarationSubmission<'static> {
        RedeclarationSubmission {
            submission_id: "redeclaration/001",
            source_ref: "participant/submission/001",
            source_sha256: SUBMISSION_SHA,
            rule_revision_id: "SAWEM-MC-2.3",
            trading_unit_id: "TU/IPP-001",
            trading_day_id: "2026-11-06",
            target_trading_period_id: "2026-11-06/12:00-13:00",
            kind: RedeclarationKind::DeclaredAvailableCapacity,
            participant_submitted_at_utc: 5 * 3_600,
            market_operator_received_at_utc: received,
            market_operator_receipt_ref: received.map(|_| "mo/receipt/001"),
            market_operator_receipt_sha256: received.map(|_| RECEIPT_SHA),
        }
    }

    #[test]
    fn receipt_before_named_gate_enters_that_intraday_auction() {
        assert_eq!(
            assess_intraday_redeclaration(gate(), submission(Some(5 * 3_600))),
            Ok(IntradayGateDisposition::IncludedInNamedAuction {
                gate_id: "idm/2026-11-06/06:00",
                received_at_utc: 5 * 3_600,
            })
        );
    }

    #[test]
    fn receipt_exactly_at_gate_boundary_is_included() {
        assert_eq!(
            assess_intraday_redeclaration(gate(), submission(Some(6 * 3_600))),
            Ok(IntradayGateDisposition::IncludedInNamedAuction {
                gate_id: "idm/2026-11-06/06:00",
                received_at_utc: 6 * 3_600,
            })
        );
    }

    #[test]
    fn missing_operator_receipt_timestamp_is_indeterminate() {
        assert_eq!(
            assess_intraday_redeclaration(gate(), submission(None)),
            Ok(IntradayGateDisposition::Indeterminate {
                reason: "Market Operator receipt timestamp is missing",
            })
        );
    }

    #[test]
    fn after_named_gate_but_before_next_gate_uses_next_auction_path() {
        assert_eq!(
            assess_intraday_redeclaration(gate(), submission(Some(8 * 3_600))),
            Ok(IntradayGateDisposition::MissedNamedAuctionEligibleForNext {
                missed_gate_id: "idm/2026-11-06/06:00",
                next_gate_id: "idm/2026-11-06/12:00",
                received_at_utc: 8 * 3_600,
            })
        );
    }

    #[test]
    fn after_final_remaining_gate_is_not_eligible_for_that_trading_period() {
        let final_gate = IntradayGateEvidence {
            gate_id: "idm/2026-11-06/12:00",
            gate_close_utc: 12 * 3_600,
            next_gate_id: None,
            next_gate_close_utc: None,
            ..gate()
        };
        assert_eq!(
            assess_intraday_redeclaration(final_gate, submission(Some(12 * 3_600 + 1))),
            Ok(IntradayGateDisposition::MissedAllRemainingGates {
                missed_gate_id: "idm/2026-11-06/12:00",
                received_at_utc: 12 * 3_600 + 1,
            })
        );
    }

    #[test]
    fn participant_submission_time_cannot_override_late_operator_receipt() {
        let late_receipt = RedeclarationSubmission {
            market_operator_received_at_utc: Some(6 * 3_600 + 1),
            market_operator_receipt_ref: Some("mo/receipt/late"),
            ..submission(Some(5 * 3_600))
        };
        assert_eq!(
            assess_intraday_redeclaration(gate(), late_receipt),
            Ok(IntradayGateDisposition::MissedNamedAuctionEligibleForNext {
                missed_gate_id: "idm/2026-11-06/06:00",
                next_gate_id: "idm/2026-11-06/12:00",
                received_at_utc: 6 * 3_600 + 1,
            })
        );
    }

    #[test]
    fn wrong_rule_revision_or_trading_unit_does_not_match_gate() {
        assert_eq!(
            assess_intraday_redeclaration(
                gate(),
                RedeclarationSubmission {
                    rule_revision_id: "SAWEM-MC-2.2",
                    ..submission(Some(5 * 3_600))
                }
            ),
            Err(IntradayGateError::RuleRevisionMismatch)
        );
        assert_eq!(
            assess_intraday_redeclaration(
                gate(),
                RedeclarationSubmission {
                    trading_unit_id: "TU/OTHER",
                    ..submission(Some(5 * 3_600))
                }
            ),
            Err(IntradayGateError::TradingUnitMismatch)
        );
    }

    #[test]
    fn missing_receipt_identity_bad_digest_and_clock_inversion_fail_closed() {
        assert_eq!(
            assess_intraday_redeclaration(
                gate(),
                RedeclarationSubmission {
                    market_operator_receipt_ref: None,
                    ..submission(Some(5 * 3_600))
                }
            ),
            Err(IntradayGateError::IncompleteOperatorReceipt)
        );
        assert_eq!(
            assess_intraday_redeclaration(
                gate(),
                RedeclarationSubmission {
                    market_operator_receipt_sha256: Some("bad"),
                    ..submission(Some(5 * 3_600))
                }
            ),
            Err(IntradayGateError::IncompleteOperatorReceipt)
        );
        assert_eq!(
            assess_intraday_redeclaration(
                gate(),
                RedeclarationSubmission {
                    participant_submitted_at_utc: 7 * 3_600,
                    market_operator_received_at_utc: Some(6 * 3_600),
                    ..submission(Some(5 * 3_600))
                }
            ),
            Err(IntradayGateError::InvalidSubmissionChronology)
        );
    }

    #[test]
    fn invalid_gate_boundaries_and_missing_next_gate_pair_fail_closed() {
        assert_eq!(
            assess_intraday_redeclaration(
                IntradayGateEvidence {
                    gate_close_utc: 13 * 3_600,
                    ..gate()
                },
                submission(Some(5 * 3_600))
            ),
            Err(IntradayGateError::InvalidGateWindow)
        );
        assert_eq!(
            assess_intraday_redeclaration(
                IntradayGateEvidence {
                    next_gate_id: Some("next"),
                    next_gate_close_utc: None,
                    ..gate()
                },
                submission(Some(5 * 3_600))
            ),
            Err(IntradayGateError::InconsistentNextGate)
        );
    }
}
