#![forbid(unsafe_code)]

//! Independent linear-scan verifier for compound donation ledger receipts.
//!
//! This verifier consumes the ledger's wire-level types but does not call
//! DonationLedger, its transition helpers, or its invariant checker. It uses
//! an independently implemented map-based replay and per-prefix scan.

use std::collections::BTreeMap;

use symthaea_compound_donation_ledger::{
    AuditEvent, CandidateSpec, DonationReservePolicy, LedgerSnapshot, ThresholdMass,
    WEIGHT_SCALE, EVIDENCE_ONE, PRIOR_MASS_SCALE, OvershootRefundReceipt,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    InvalidSpecs,
    UnknownCandidate,
    DuplicateActivation,
    EvidenceBeforeActivation,
    EvidenceDecreased,
    InvalidRecipient,
    RecipientDormant,
    ZeroWeightRecipient,
    MissingOrRepeatedToken,
    InvalidAmount,
    RoundOrder,
    Overflow,
    PrefixBudget,
    EscrowConservation,
    ReceiptMismatch,
    SnapshotMismatch,
    InvalidStaircase,
    StaircaseBudgetExceeded,
    StaircaseValueMismatch,
    InvalidRefundPolicy,
    RefundReplayMismatch,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for VerifyError {}

#[derive(Clone, Debug, Default)]
struct Candidate {
    active: bool,
    x: u128,
    incoming: u128,
    outgoing: u128,
    remainder: u128,
}

#[derive(Clone, Debug)]
struct Token {
    donor: u64,
    destination: u64,
    round: u64,
    initial: u128,
    remaining: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCandidate {
    pub id: u64,
    pub active: bool,
    pub evidence_value: u128,
    pub incoming: u128,
    pub outgoing: u128,
    pub weighted_score: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedToken {
    pub token_id: u64,
    pub donor_id: u64,
    pub destination_id: u64,
    pub created_round: u64,
    pub original_amount: u128,
    pub remaining_amount: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSnapshot {
    pub candidates: Vec<VerifiedCandidate>,
    pub escrow: Vec<VerifiedToken>,
}



/// Independently verifies one cumulative overshoot-refund transition.
///
/// The verifier recomputes the exact rational SCORE-style refund from
/// alpha, previous evidence, previous cumulative refund, and the new
/// time-monotone e-process value. The receipt is data only; it is not trusted.
fn verify_overshoot_refund_transition_with_cap(
    alpha_numerator: u64,
    alpha_denominator: u64,
    refund_cap_numerator: Option<u128>,
    previous_round: Option<u64>,
    previous_evidence_value: u128,
    previous_refund_numerator: u128,
    receipt: &OvershootRefundReceipt,
) -> Result<(), VerifyError> {
    if alpha_numerator == 0 || alpha_denominator == 0 || alpha_numerator >= alpha_denominator {
        return Err(VerifyError::InvalidRefundPolicy);
    }
    if previous_round.is_some_and(|round| receipt.round < round)
        || receipt.evidence_value < previous_evidence_value
    {
        return Err(VerifyError::RefundReplayMismatch);
    }

    let denominator = (alpha_denominator as u128)
        .checked_mul(EVIDENCE_ONE)
        .ok_or(VerifyError::Overflow)?;
    let previous_scaled_evidence = (alpha_numerator as u128)
        .checked_mul(previous_evidence_value)
        .ok_or(VerifyError::Overflow)?;
    let previous_raw_refund = previous_scaled_evidence.saturating_sub(denominator);
    let expected_previous_refund = refund_cap_numerator
        .map_or(previous_raw_refund, |cap| previous_raw_refund.min(cap));
    if expected_previous_refund != previous_refund_numerator {
        return Err(VerifyError::RefundReplayMismatch);
    }
    let alpha_scaled_evidence = (alpha_numerator as u128)
        .checked_mul(receipt.evidence_value)
        .ok_or(VerifyError::Overflow)?;
    let raw_overshoot = alpha_scaled_evidence.saturating_sub(denominator);
    let expected_cumulative = refund_cap_numerator
        .map_or(raw_overshoot, |cap| raw_overshoot.min(cap));
    if expected_cumulative < previous_refund_numerator {
        return Err(VerifyError::RefundReplayMismatch);
    }
    let expected_increment = expected_cumulative - previous_refund_numerator;
    let expected_cost = alpha_scaled_evidence
        .checked_sub(expected_cumulative)
        .ok_or(VerifyError::RefundReplayMismatch)?;
    let expected_rejected = alpha_scaled_evidence >= denominator;

    if receipt.denominator != denominator
        || receipt.cumulative_refund_numerator != expected_cumulative
        || receipt.refund_increment_numerator != expected_increment
        || receipt.effective_cost_numerator != expected_cost
        || receipt.rejected != expected_rejected
        || (expected_rejected && expected_cost < denominator)
    {
        return Err(VerifyError::RefundReplayMismatch);
    }
    Ok(())
}

/// Independently verifies one bounded-credit persistent refund transition.
pub fn verify_overshoot_refund_transition(
    alpha_numerator: u64,
    alpha_denominator: u64,
    previous_round: Option<u64>,
    previous_evidence_value: u128,
    previous_refund_numerator: u128,
    receipt: &OvershootRefundReceipt,
) -> Result<(), VerifyError> {
    let cap = (alpha_numerator as u128)
        .checked_mul(EVIDENCE_ONE)
        .ok_or(VerifyError::Overflow)?;
    verify_overshoot_refund_transition_with_cap(
        alpha_numerator,
        alpha_denominator,
        Some(cap),
        previous_round,
        previous_evidence_value,
        previous_refund_numerator,
        receipt,
    )
}

/// Independently verifies a full, uncapped SCORE-style overshoot transition.
pub fn verify_full_overshoot_refund_transition(
    alpha_numerator: u64,
    alpha_denominator: u64,
    previous_round: Option<u64>,
    previous_evidence_value: u128,
    previous_refund_numerator: u128,
    receipt: &OvershootRefundReceipt,
) -> Result<(), VerifyError> {
    verify_overshoot_refund_transition_with_cap(
        alpha_numerator,
        alpha_denominator,
        None,
        previous_round,
        previous_evidence_value,
        previous_refund_numerator,
        receipt,
    )
}

/// Replays a complete refund receipt sequence from the normalized V_0 = 1
/// state and returns the final cumulative refund numerator.

/// Independently verifies one *uncapped* SCORE overshoot-refund transition.
///
/// The refund is the full positive overshoot, with no artificial cap:
/// O_t = (alpha * V_t - 1)_+. The residual cost is min(alpha * V_t, 1).
pub fn verify_uncapped_overshoot_refund_transition(
    alpha_numerator: u64,
    alpha_denominator: u64,
    previous_round: Option<u64>,
    previous_evidence_value: u128,
    previous_refund_numerator: u128,
    receipt: &OvershootRefundReceipt,
) -> Result<(), VerifyError> {
    if alpha_numerator == 0
        || alpha_denominator == 0
        || alpha_numerator >= alpha_denominator
    {
        return Err(VerifyError::InvalidRefundPolicy);
    }
    if previous_round.is_some_and(|round| receipt.round < round)
        || receipt.evidence_value < previous_evidence_value
    {
        return Err(VerifyError::RefundReplayMismatch);
    }

    let denominator = (alpha_denominator as u128)
        .checked_mul(EVIDENCE_ONE)
        .ok_or(VerifyError::Overflow)?;
    let previous_scaled = (alpha_numerator as u128)
        .checked_mul(previous_evidence_value)
        .ok_or(VerifyError::Overflow)?;
    let previous_expected_refund = previous_scaled.saturating_sub(denominator);
    if previous_expected_refund != previous_refund_numerator {
        return Err(VerifyError::RefundReplayMismatch);
    }

    let scaled = (alpha_numerator as u128)
        .checked_mul(receipt.evidence_value)
        .ok_or(VerifyError::Overflow)?;
    let expected_refund = scaled.saturating_sub(denominator);
    let expected_increment = expected_refund
        .checked_sub(previous_refund_numerator)
        .ok_or(VerifyError::RefundReplayMismatch)?;
    let expected_cost = scaled.min(denominator);
    let expected_rejected = scaled >= denominator;

    if receipt.denominator != denominator
        || receipt.cumulative_refund_numerator != expected_refund
        || receipt.refund_increment_numerator != expected_increment
        || receipt.effective_cost_numerator != expected_cost
        || receipt.rejected != expected_rejected
    {
        return Err(VerifyError::RefundReplayMismatch);
    }
    Ok(())
}

/// Replays uncapped refund receipts from an explicit initial process value.
pub fn verify_uncapped_overshoot_refund_sequence_from(
    alpha_numerator: u64,
    alpha_denominator: u64,
    initial_evidence_value: u128,
    receipts: &[OvershootRefundReceipt],
) -> Result<u128, VerifyError> {
    let mut previous_round = None;
    let mut previous_evidence = initial_evidence_value;
    let mut previous_refund = 0_u128;

    for receipt in receipts {
        verify_uncapped_overshoot_refund_transition(
            alpha_numerator,
            alpha_denominator,
            previous_round,
            previous_evidence,
            previous_refund,
            receipt,
        )?;
        previous_round = Some(receipt.round);
        previous_evidence = receipt.evidence_value;
        previous_refund = receipt.cumulative_refund_numerator;
    }
    Ok(previous_refund)
}

pub fn verify_overshoot_refund_sequence(
    alpha_numerator: u64,
    alpha_denominator: u64,
    receipts: &[OvershootRefundReceipt],
) -> Result<u128, VerifyError> {
    verify_overshoot_refund_sequence_from(
        alpha_numerator,
        alpha_denominator,
        EVIDENCE_ONE,
        receipts,
    )
}

/// Replays a full receipt sequence from an explicit initial process value.
pub fn verify_overshoot_refund_sequence_from(
    alpha_numerator: u64,
    alpha_denominator: u64,
    initial_evidence_value: u128,
    receipts: &[OvershootRefundReceipt],
) -> Result<u128, VerifyError> {
    let mut previous_round = None;
    let mut previous_evidence = initial_evidence_value;
    let mut previous_refund = 0_u128;

    for receipt in receipts {
        verify_overshoot_refund_transition(
            alpha_numerator,
            alpha_denominator,
            previous_round,
            previous_evidence,
            previous_refund,
            receipt,
        )?;
        previous_round = Some(receipt.round);
        previous_evidence = receipt.evidence_value;
        previous_refund = receipt.cumulative_refund_numerator;
    }
    Ok(previous_refund)
}

/// Replays a complete full-overshoot sequence from normalized V_0 = 1.
pub fn verify_full_overshoot_refund_sequence(
    alpha_numerator: u64,
    alpha_denominator: u64,
    receipts: &[OvershootRefundReceipt],
) -> Result<u128, VerifyError> {
    let mut previous_round = None;
    let mut previous_evidence = EVIDENCE_ONE;
    let mut previous_refund = 0_u128;

    for receipt in receipts {
        verify_full_overshoot_refund_transition(
            alpha_numerator,
            alpha_denominator,
            previous_round,
            previous_evidence,
            previous_refund,
            receipt,
        )?;
        previous_round = Some(receipt.round);
        previous_evidence = receipt.evidence_value;
        previous_refund = receipt.cumulative_refund_numerator;
    }
    Ok(previous_refund)
}

/// Independently verifies a finite staircase calibrator using a linear scan.
///
/// This verifier deliberately does not call StaircaseAdjuster::new or
/// StaircaseAdjuster::value_at. The caller supplies the precommitted sorted
/// threshold/prior-mass rows, the running supremum, and the claimed fixed-point
/// output. It checks strict threshold ordering, K_j >= 1, total prior mass <= 1,
/// exact checked arithmetic, and exact downward-rounded output.
pub fn verify_staircase_value(
    points: &[ThresholdMass],
    running_supremum: u128,
    claimed_value: u128,
) -> Result<u128, VerifyError> {
    let mut total_mass = 0_u128;
    let mut numerator = 0_u128;
    let mut previous_threshold = None;

    for point in points {
        if point.threshold < EVIDENCE_ONE
            || previous_threshold.is_some_and(|previous| point.threshold <= previous)
        {
            return Err(VerifyError::InvalidStaircase);
        }
        total_mass = total_mass
            .checked_add(point.prior_mass)
            .ok_or(VerifyError::Overflow)?;
        if total_mass > PRIOR_MASS_SCALE {
            return Err(VerifyError::StaircaseBudgetExceeded);
        }

        if point.threshold <= running_supremum {
            let contribution = point.prior_mass
                .checked_mul(point.threshold)
                .ok_or(VerifyError::Overflow)?;
            numerator = numerator
                .checked_add(contribution)
                .ok_or(VerifyError::Overflow)?;
        }
        previous_threshold = Some(point.threshold);
    }

    let exact_downward_value = numerator / PRIOR_MASS_SCALE;
    if exact_downward_value != claimed_value {
        return Err(VerifyError::StaircaseValueMismatch);
    }
    Ok(exact_downward_value)
}

pub fn verify_receipt(
    specs: &[CandidateSpec],
    receipt: &[AuditEvent],
    expected_snapshot: Option<&LedgerSnapshot>,
) -> Result<VerifiedSnapshot, VerifyError> {
    verify_receipt_with_policy(
        specs,
        DonationReservePolicy::FixedRate,
        receipt,
        expected_snapshot,
    )
}

pub fn verify_receipt_with_policy(
    specs: &[CandidateSpec],
    reserve_policy: DonationReservePolicy,
    receipt: &[AuditEvent],
    expected_snapshot: Option<&LedgerSnapshot>,
) -> Result<VerifiedSnapshot, VerifyError> {
    let mut spec_map = BTreeMap::<u64, CandidateSpec>::new();
    let mut state = BTreeMap::<u64, Candidate>::new();
    let mut weight_sum = 0_u64;
    let mut previous_id = None;

    for spec in specs {
        let invalid_reserve_policy = match reserve_policy {
            DonationReservePolicy::FixedRate => {
                spec.reserve_rate_numerator > spec.reserve_rate_denominator
                    || (spec.destination_id.is_none() && spec.reserve_rate_numerator > 0)
            }
            DonationReservePolicy::CappedOvershoot => {
                spec.reserve_rate_numerator != 0
            }
        };
        if previous_id.is_some_and(|previous| spec.id <= previous)
            || spec.reserve_rate_denominator == 0
            || invalid_reserve_policy
            || spec.destination_id.is_some_and(|destination| destination <= spec.id)
        {
            return Err(VerifyError::InvalidSpecs);
        }
        weight_sum = weight_sum
            .checked_add(spec.weight)
            .ok_or(VerifyError::Overflow)?;
        if weight_sum > WEIGHT_SCALE {
            return Err(VerifyError::InvalidSpecs);
        }
        previous_id = Some(spec.id);
        spec_map.insert(spec.id, spec.clone());
        state.insert(spec.id, Candidate::default());
    }

    let mut tokens = BTreeMap::<u64, Token>::new();
    let mut regenerated = Vec::<AuditEvent>::new();
    let mut next_token = 1_u64;
    let mut previous_round = None;
    let mut total_incoming = 0_u128;
    let mut total_outgoing = 0_u128;

    for event in receipt {
        let round = match event {
            AuditEvent::Activated { round, .. }
            | AuditEvent::EvidenceUpdated { round, .. }
            | AuditEvent::Reserved { round, .. }
            | AuditEvent::Credited { round, .. } => *round,
        };
        if previous_round.is_some_and(|previous| round < previous) {
            return Err(VerifyError::RoundOrder);
        }
        previous_round = Some(round);

        match event {
            AuditEvent::Activated { round, candidate_id } => {
                let candidate = state.get_mut(candidate_id).ok_or(VerifyError::UnknownCandidate)?;
                if candidate.active {
                    return Err(VerifyError::DuplicateActivation);
                }
                candidate.active = true;
                regenerated.push(AuditEvent::Activated {
                    round: *round,
                    candidate_id: *candidate_id,
                });
            }
            AuditEvent::EvidenceUpdated { round, candidate_id, new_value } => {
                let spec = spec_map.get(candidate_id).ok_or(VerifyError::UnknownCandidate)?;
                let candidate = state.get_mut(candidate_id).ok_or(VerifyError::UnknownCandidate)?;
                if !candidate.active {
                    return Err(VerifyError::EvidenceBeforeActivation);
                }
                if *new_value < candidate.x {
                    return Err(VerifyError::EvidenceDecreased);
                }

                let delta = *new_value - candidate.x;
                let (reserve, next_remainder) = match reserve_policy {
                    DonationReservePolicy::FixedRate => {
                        let weighted_delta = (spec.weight as u128)
                            .checked_mul(delta)
                            .ok_or(VerifyError::Overflow)?;
                        let numerator = weighted_delta
                            .checked_mul(spec.reserve_rate_numerator as u128)
                            .and_then(|value| value.checked_add(candidate.remainder))
                            .ok_or(VerifyError::Overflow)?;
                        let denominator = spec.reserve_rate_denominator as u128;
                        let proposed = numerator / denominator;
                        let remainder = numerator % denominator;
                        let donor_limit = (spec.weight as u128)
                            .checked_mul((*new_value).min(EVIDENCE_ONE))
                            .ok_or(VerifyError::Overflow)?;
                        let room = donor_limit
                            .checked_sub(candidate.outgoing)
                            .ok_or(VerifyError::InvalidAmount)?;
                        (proposed.min(room), remainder)
                    }
                    DonationReservePolicy::CappedOvershoot => {
                        let weighted_value = (spec.weight as u128)
                            .checked_mul(*new_value)
                            .ok_or(VerifyError::Overflow)?;
                        let threshold = (WEIGHT_SCALE as u128)
                            .checked_mul(EVIDENCE_ONE)
                            .ok_or(VerifyError::Overflow)?;
                        let raw = weighted_value.saturating_sub(threshold);
                        let cap = (spec.weight as u128)
                            .checked_mul(EVIDENCE_ONE)
                            .ok_or(VerifyError::Overflow)?;
                        let target = if spec.destination_id.is_some() {
                            raw.min(cap)
                        } else {
                            0
                        };
                        let reserve = target
                            .checked_sub(candidate.outgoing)
                            .ok_or(VerifyError::InvalidAmount)?;
                        (reserve, 0)
                    }
                };
                let donor_limit = (spec.weight as u128)
                    .checked_mul((*new_value).min(EVIDENCE_ONE))
                    .ok_or(VerifyError::Overflow)?;
                if candidate.outgoing.checked_add(reserve).ok_or(VerifyError::Overflow)? > donor_limit {
                    return Err(VerifyError::PrefixBudget);
                }

                candidate.x = *new_value;
                candidate.outgoing = candidate.outgoing
                    .checked_add(reserve)
                    .ok_or(VerifyError::Overflow)?;
                candidate.remainder = next_remainder;
                total_outgoing = total_outgoing
                    .checked_add(reserve)
                    .ok_or(VerifyError::Overflow)?;

                regenerated.push(AuditEvent::EvidenceUpdated {
                    round: *round,
                    candidate_id: *candidate_id,
                    new_value: *new_value,
                });

                if reserve > 0 {
                    let token_id = next_token;
                    next_token = next_token.checked_add(1).ok_or(VerifyError::Overflow)?;
                    let destination = spec.destination_id.unwrap_or(0);
                    tokens.insert(token_id, Token {
                        donor: *candidate_id,
                        destination,
                        round: *round,
                        initial: reserve,
                        remaining: reserve,
                    });
                    regenerated.push(AuditEvent::Reserved {
                        round: *round,
                        token_id,
                        donor_id: *candidate_id,
                        destination_id: destination,
                        amount: reserve,
                    });
                }
            }
            AuditEvent::Reserved { .. } => {
                // Derived from the immediately preceding evidence update.
                // It is regenerated, never trusted as a command.
            }
            AuditEvent::Credited { round, token_id, recipient_id, amount } => {
                if *amount == 0 {
                    return Err(VerifyError::InvalidAmount);
                }
                let token = tokens.get_mut(token_id).ok_or(VerifyError::MissingOrRepeatedToken)?;
                if *recipient_id <= token.donor
                    || *recipient_id != token.destination
                    || *round < token.round
                    || *amount > token.remaining
                {
                    return Err(VerifyError::InvalidRecipient);
                }

                let recipient_spec = spec_map.get(recipient_id).ok_or(VerifyError::UnknownCandidate)?;
                if recipient_spec.weight == 0 {
                    return Err(VerifyError::ZeroWeightRecipient);
                }
                let recipient = state.get_mut(recipient_id).ok_or(VerifyError::UnknownCandidate)?;
                if !recipient.active {
                    return Err(VerifyError::RecipientDormant);
                }

                token.remaining -= *amount;
                recipient.incoming = recipient.incoming
                    .checked_add(*amount)
                    .ok_or(VerifyError::Overflow)?;
                total_incoming = total_incoming
                    .checked_add(*amount)
                    .ok_or(VerifyError::Overflow)?;

                regenerated.push(AuditEvent::Credited {
                    round: *round,
                    token_id: *token_id,
                    recipient_id: *recipient_id,
                    amount: *amount,
                });
            }
        }

        // Independently scan the sorted candidate order after each event.
        let mut prefix_in = 0_u128;
        let mut prefix_out = 0_u128;
        for candidate in state.values() {
            prefix_in = prefix_in.checked_add(candidate.incoming).ok_or(VerifyError::Overflow)?;
            prefix_out = prefix_out.checked_add(candidate.outgoing).ok_or(VerifyError::Overflow)?;
            if prefix_in > prefix_out {
                return Err(VerifyError::PrefixBudget);
            }
        }
        if total_incoming > total_outgoing {
            return Err(VerifyError::PrefixBudget);
        }
    }

    if regenerated != receipt {
        return Err(VerifyError::ReceiptMismatch);
    }

    let state_incoming = state.values().try_fold(0_u128, |sum, candidate| {
        sum.checked_add(candidate.incoming).ok_or(VerifyError::Overflow)
    })?;
    let state_outgoing = state.values().try_fold(0_u128, |sum, candidate| {
        sum.checked_add(candidate.outgoing).ok_or(VerifyError::Overflow)
    })?;
    if state_incoming != total_incoming || state_outgoing != total_outgoing {
        return Err(VerifyError::PrefixBudget);
    }

    let mut result_candidates = Vec::with_capacity(state.len());
    for (id, candidate) in &state {
        let spec = spec_map.get(id).ok_or(VerifyError::UnknownCandidate)?;
        if !candidate.active
            && (candidate.x != 0 || candidate.incoming != 0 || candidate.outgoing != 0)
        {
            return Err(VerifyError::PrefixBudget);
        }
        let donor_cap = (spec.weight as u128)
            .checked_mul(candidate.x.min(EVIDENCE_ONE))
            .ok_or(VerifyError::Overflow)?;
        if candidate.outgoing > donor_cap {
            return Err(VerifyError::PrefixBudget);
        }
        let score = (spec.weight as u128)
            .checked_mul(candidate.x)
            .and_then(|value| value.checked_add(candidate.incoming))
            .and_then(|value| value.checked_sub(candidate.outgoing))
            .ok_or(VerifyError::PrefixBudget)?;
        result_candidates.push(VerifiedCandidate {
            id: *id,
            active: candidate.active,
            evidence_value: candidate.x,
            incoming: candidate.incoming,
            outgoing: candidate.outgoing,
            weighted_score: score,
        });
    }

    let mut result_tokens = Vec::with_capacity(tokens.len());
    let mut unspent = 0_u128;
    for (token_id, token) in &tokens {
        if token.remaining > token.initial {
            return Err(VerifyError::EscrowConservation);
        }
        unspent = unspent.checked_add(token.remaining).ok_or(VerifyError::Overflow)?;
        result_tokens.push(VerifiedToken {
            token_id: *token_id,
            donor_id: token.donor,
            destination_id: token.destination,
            created_round: token.round,
            original_amount: token.initial,
            remaining_amount: token.remaining,
        });
    }
    if total_outgoing.checked_sub(total_incoming).ok_or(VerifyError::EscrowConservation)? != unspent {
        return Err(VerifyError::EscrowConservation);
    }

    let result = VerifiedSnapshot {
        candidates: result_candidates,
        escrow: result_tokens,
    };
    if let Some(expected) = expected_snapshot {
        let projected = VerifiedSnapshot {
            candidates: expected.candidates.iter().map(|candidate| VerifiedCandidate {
                id: candidate.id,
                active: candidate.active,
                evidence_value: candidate.evidence_value,
                incoming: candidate.incoming,
                outgoing: candidate.outgoing,
                weighted_score: candidate.weighted_score,
            }).collect(),
            escrow: {
                // Destinations are part of the reference snapshot and are compared
                // directly with the independent replay result.
                expected.escrow.iter().map(|token| {
                    VerifiedToken {
                        token_id: token.token_id,
                        donor_id: token.donor_id,
                        destination_id: token.destination_id,
                        created_round: token.created_round,
                        original_amount: token.original_amount,
                        remaining_amount: token.remaining_amount,
                    }
                }).collect()
            },
        };
        if projected != result {
            return Err(VerifyError::SnapshotMismatch);
        }
    }
    Ok(result)
}


#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_compound_donation_ledger::{
        CandidateSpec, DonationLedger, InputEvent,
    };

    fn specs() -> Vec<CandidateSpec> {
        vec![
            CandidateSpec {
                id: 10,
                weight: 300_000_000,
                reserve_rate_numerator: 1,
                reserve_rate_denominator: 1,
                destination_id: Some(20),
            },
            CandidateSpec {
                id: 20,
                weight: 300_000_000,
                reserve_rate_numerator: 1,
                reserve_rate_denominator: 1,
                destination_id: Some(30),
            },
            CandidateSpec {
                id: 30,
                weight: 200_000_000,
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
                destination_id: None,
            },
        ]
    }





    #[test]
    fn independently_verifies_uncapped_score_refund_sequence() {
        let receipts = vec![
            OvershootRefundReceipt {
                round: 1,
                evidence_value: 19 * EVIDENCE_ONE,
                refund_increment_numerator: 0,
                cumulative_refund_numerator: 0,
                effective_cost_numerator: 19 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: false,
            },
            OvershootRefundReceipt {
                round: 2,
                evidence_value: 21 * EVIDENCE_ONE,
                refund_increment_numerator: EVIDENCE_ONE,
                cumulative_refund_numerator: EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
            OvershootRefundReceipt {
                round: 3,
                evidence_value: 100 * EVIDENCE_ONE,
                refund_increment_numerator: 79 * EVIDENCE_ONE,
                cumulative_refund_numerator: 80 * EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
        ];
        assert_eq!(
            verify_uncapped_overshoot_refund_sequence_from(
                1, 20, EVIDENCE_ONE, &receipts
            ),
            Ok(80 * EVIDENCE_ONE)
        );

        let mut tampered = receipts.clone();
        tampered[2].refund_increment_numerator = 0;
        assert_eq!(
            verify_uncapped_overshoot_refund_sequence_from(
                1, 20, EVIDENCE_ONE, &tampered
            ),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independent_verifier_replays_zero_initialized_refund_sequence() {
        let receipts = vec![
            OvershootRefundReceipt {
                round: 0,
                evidence_value: 0,
                refund_increment_numerator: 0,
                cumulative_refund_numerator: 0,
                effective_cost_numerator: 0,
                denominator: 20 * EVIDENCE_ONE,
                rejected: false,
            },
            OvershootRefundReceipt {
                round: 1,
                evidence_value: 21 * EVIDENCE_ONE,
                refund_increment_numerator: EVIDENCE_ONE,
                cumulative_refund_numerator: EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
        ];
        assert_eq!(
            verify_overshoot_refund_sequence_from(1, 20, 0, &receipts),
            Ok(EVIDENCE_ONE)
        );
        assert_eq!(
            verify_overshoot_refund_sequence(1, 20, &receipts),
            Err(VerifyError::RefundReplayMismatch)
        );
    }


    #[test]
    fn generic_refund_verifier_rejects_forged_previous_balance() {
        let receipt = OvershootRefundReceipt {
            round: 2,
            evidence_value: 100 * EVIDENCE_ONE,
            refund_increment_numerator: 80 * EVIDENCE_ONE,
            cumulative_refund_numerator: 80 * EVIDENCE_ONE,
            effective_cost_numerator: 20 * EVIDENCE_ONE,
            denominator: 20 * EVIDENCE_ONE,
            rejected: true,
        };

        // At the previous evidence value 21, the cumulative refund is exactly
        // 1 evidence tick in the common numerator scale, not zero.
        assert_eq!(
            verify_full_overshoot_refund_transition(
                1, 20, Some(1), 21 * EVIDENCE_ONE, 0, &receipt
            ),
            Err(VerifyError::RefundReplayMismatch)
        );

        // Likewise, the capped account's previous balance must equal its
        // policy-implied cumulative capped overshoot.
        assert_eq!(
            verify_overshoot_refund_transition(
                1, 20, Some(1), 21 * EVIDENCE_ONE, 0, &receipt
            ),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independent_verifier_accepts_full_score_overshoot_receipt() {
        let receipt = OvershootRefundReceipt {
            round: 1,
            evidence_value: 100 * EVIDENCE_ONE,
            refund_increment_numerator: 80 * EVIDENCE_ONE,
            cumulative_refund_numerator: 80 * EVIDENCE_ONE,
            effective_cost_numerator: 20 * EVIDENCE_ONE,
            denominator: 20 * EVIDENCE_ONE,
            rejected: true,
        };
        assert_eq!(
            verify_full_overshoot_refund_transition(
                1, 20, Some(0), EVIDENCE_ONE, 0, &receipt
            ),
            Ok(())
        );
        assert_eq!(
            verify_overshoot_refund_transition(
                1, 20, Some(0), EVIDENCE_ONE, 0, &receipt
            ),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independent_verifier_replays_full_overshoot_sequence_without_double_count() {
        let receipts = vec![
            OvershootRefundReceipt {
                round: 1,
                evidence_value: 21 * EVIDENCE_ONE,
                refund_increment_numerator: EVIDENCE_ONE,
                cumulative_refund_numerator: EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
            OvershootRefundReceipt {
                round: 2,
                evidence_value: 100 * EVIDENCE_ONE,
                refund_increment_numerator: 79 * EVIDENCE_ONE,
                cumulative_refund_numerator: 80 * EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
            OvershootRefundReceipt {
                round: 3,
                evidence_value: 1000 * EVIDENCE_ONE,
                refund_increment_numerator: 900 * EVIDENCE_ONE,
                cumulative_refund_numerator: 980 * EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
        ];
        assert_eq!(
            verify_full_overshoot_refund_sequence(1, 20, &receipts),
            Ok(980 * EVIDENCE_ONE)
        );

        let mut tampered = receipts.clone();
        tampered[2].refund_increment_numerator = 1000 * EVIDENCE_ONE;
        assert_eq!(
            verify_full_overshoot_refund_sequence(1, 20, &tampered),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independent_verifier_replays_complete_refund_sequence_exactly_once() {
        let receipts = vec![
            OvershootRefundReceipt {
                round: 1,
                evidence_value: 19 * EVIDENCE_ONE,
                refund_increment_numerator: 0,
                cumulative_refund_numerator: 0,
                effective_cost_numerator: 19 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: false,
            },
            OvershootRefundReceipt {
                round: 2,
                evidence_value: 21 * EVIDENCE_ONE,
                refund_increment_numerator: EVIDENCE_ONE,
                cumulative_refund_numerator: EVIDENCE_ONE,
                effective_cost_numerator: 20 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
            OvershootRefundReceipt {
                round: 3,
                evidence_value: 30 * EVIDENCE_ONE,
                refund_increment_numerator: 0,
                cumulative_refund_numerator: EVIDENCE_ONE,
                effective_cost_numerator: 29 * EVIDENCE_ONE,
                denominator: 20 * EVIDENCE_ONE,
                rejected: true,
            },
        ];
        assert_eq!(
            verify_overshoot_refund_sequence(1, 20, &receipts),
            Ok(EVIDENCE_ONE)
        );

        let mut tampered = receipts.clone();
        tampered[2].refund_increment_numerator = EVIDENCE_ONE;
        tampered[2].cumulative_refund_numerator = 2 * EVIDENCE_ONE;
        assert_eq!(
            verify_overshoot_refund_sequence(1, 20, &tampered),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independent_verifier_accepts_hand_authored_process_refund() {
        let receipt = OvershootRefundReceipt {
            round: 1,
            evidence_value: 21 * EVIDENCE_ONE,
            refund_increment_numerator: EVIDENCE_ONE,
            cumulative_refund_numerator: EVIDENCE_ONE,
            effective_cost_numerator: 20 * EVIDENCE_ONE,
            denominator: 20 * EVIDENCE_ONE,
            rejected: true,
        };
        assert_eq!(
            verify_overshoot_refund_transition(
                1, 20, Some(0), EVIDENCE_ONE, 0, &receipt
            ),
            Ok(())
        );
    }

    #[test]
    fn independent_verifier_rejects_forged_previous_refund_state() {
        let next = OvershootRefundReceipt {
            round: 3,
            evidence_value: 40 * EVIDENCE_ONE,
            refund_increment_numerator: 0,
            cumulative_refund_numerator: EVIDENCE_ONE,
            effective_cost_numerator: 39 * EVIDENCE_ONE,
            denominator: 20 * EVIDENCE_ONE,
            rejected: true,
        };
        assert_eq!(
            verify_overshoot_refund_transition(
                1, 20, Some(2), 30 * EVIDENCE_ONE, 0, &next
            ),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independent_verifier_rejects_repeated_or_forged_refund() {
        let receipt = OvershootRefundReceipt {
            round: 2,
            evidence_value: 40 * EVIDENCE_ONE,
            refund_increment_numerator: EVIDENCE_ONE,
            cumulative_refund_numerator: 2 * EVIDENCE_ONE,
            effective_cost_numerator: 38 * EVIDENCE_ONE,
            denominator: 20 * EVIDENCE_ONE,
            rejected: true,
        };
        assert_eq!(
            verify_overshoot_refund_transition(
                1, 20, Some(1), 21 * EVIDENCE_ONE, EVIDENCE_ONE, &receipt
            ),
            Err(VerifyError::RefundReplayMismatch)
        );

        let tampered = OvershootRefundReceipt {
            refund_increment_numerator: 0,
            ..receipt.clone()
        };
        assert_eq!(
            verify_overshoot_refund_transition(
                1, 20, Some(1), 21 * EVIDENCE_ONE, EVIDENCE_ONE, &tampered
            ),
            Err(VerifyError::RefundReplayMismatch)
        );
    }

    #[test]
    fn independently_verifies_staircase_at_exact_inclusive_boundaries() {
        let points = vec![
            ThresholdMass { threshold: 2 * EVIDENCE_ONE, prior_mass: 100_000_000 },
            ThresholdMass { threshold: 8 * EVIDENCE_ONE, prior_mass: 200_000_000 },
            ThresholdMass { threshold: 32 * EVIDENCE_ONE, prior_mass: 300_000_000 },
        ];
        let before_second_threshold = 200_000;
        assert_eq!(
            verify_staircase_value(
                &points,
                8 * EVIDENCE_ONE - 1,
                before_second_threshold
            ),
            Ok(before_second_threshold)
        );
        let at_second_threshold = 8 * EVIDENCE_ONE;
        let exact_at_boundary = 1_800_000;
        assert_eq!(
            verify_staircase_value(
                &points,
                at_second_threshold,
                exact_at_boundary
            ),
            Ok(exact_at_boundary)
        );
        assert_eq!(
            verify_staircase_value(
                &points,
                at_second_threshold,
                before_second_threshold
            ),
            Err(VerifyError::StaircaseValueMismatch)
        );
    }

    #[test]
    fn independently_rejects_unbudgeted_or_unsorted_staircase() {
        let over_budget = vec![
            ThresholdMass { threshold: EVIDENCE_ONE, prior_mass: 600_000_000 },
            ThresholdMass { threshold: 2 * EVIDENCE_ONE, prior_mass: 500_000_000 },
        ];
        assert_eq!(
            verify_staircase_value(&over_budget, 2 * EVIDENCE_ONE, 0),
            Err(VerifyError::StaircaseBudgetExceeded)
        );

        let unsorted = vec![
            ThresholdMass { threshold: 2 * EVIDENCE_ONE, prior_mass: 100_000_000 },
            ThresholdMass { threshold: EVIDENCE_ONE, prior_mass: 100_000_000 },
        ];
        assert_eq!(
            verify_staircase_value(&unsorted, 2 * EVIDENCE_ONE, 0),
            Err(VerifyError::InvalidStaircase)
        );
    }


    #[test]
    fn independent_verifier_confirms_capped_overshoot_horizon_extension() {
        let make_spec = |id, weight, destination_id| CandidateSpec {
            id,
            weight,
            destination_id,
            reserve_rate_numerator: 0,
            reserve_rate_denominator: 1,
        };
        let small = vec![
            make_spec(10, 300_000_000, Some(20)),
            make_spec(20, 300_000_000, Some(30)),
        ];
        let extended = vec![
            make_spec(10, 300_000_000, Some(20)),
            make_spec(20, 300_000_000, Some(30)),
            make_spec(30, 200_000_000, Some(40)),
        ];
        let common_prefix = vec![
            AuditEvent::Activated { round: 0, candidate_id: 10 },
            AuditEvent::EvidenceUpdated {
                round: 1,
                candidate_id: 10,
                new_value: 4 * EVIDENCE_ONE,
            },
            AuditEvent::Reserved {
                round: 1,
                token_id: 1,
                donor_id: 10,
                destination_id: 20,
                amount: 200_000_000_000_000,
            },
            AuditEvent::Activated { round: 2, candidate_id: 20 },
            AuditEvent::Credited {
                round: 2,
                token_id: 1,
                recipient_id: 20,
                amount: 200_000_000_000_000,
            },
            AuditEvent::EvidenceUpdated {
                round: 3,
                candidate_id: 20,
                new_value: 4 * EVIDENCE_ONE,
            },
            AuditEvent::Reserved {
                round: 3,
                token_id: 2,
                donor_id: 20,
                destination_id: 30,
                amount: 200_000_000_000_000,
            },
        ];

        let small_result = verify_receipt_with_policy(
            &small,
            DonationReservePolicy::CappedOvershoot,
            &common_prefix,
            None,
        ).unwrap();
        let extended_result = verify_receipt_with_policy(
            &extended,
            DonationReservePolicy::CappedOvershoot,
            &common_prefix,
            None,
        ).unwrap();

        assert_eq!(
            small_result.candidates,
            extended_result.candidates[..2]
        );
        assert_eq!(
            small_result.escrow,
            extended_result.escrow
        );
        assert_eq!(
            small_result.candidates[0].weighted_score,
            1_000_000_000_000_000
        );
        assert_eq!(
            small_result.candidates[1].incoming,
            200_000_000_000_000
        );
        assert_eq!(
            small_result.escrow[0].destination_id,
            30
        );
    }

    #[test]
    fn independent_verifier_accepts_overshoot_policy_with_disabled_destination() {
        let policy = vec![
            CandidateSpec {
                id: 10,
                weight: 500_000_000,
                destination_id: None,
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
            },
            CandidateSpec {
                id: 20,
                weight: 500_000_000,
                destination_id: None,
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
            },
        ];
        let receipt = vec![
            AuditEvent::Activated { round: 0, candidate_id: 10 },
            AuditEvent::EvidenceUpdated {
                round: 1,
                candidate_id: 10,
                new_value: 100 * EVIDENCE_ONE,
            },
        ];
        let verified = verify_receipt_with_policy(
            &policy,
            DonationReservePolicy::CappedOvershoot,
            &receipt,
            None,
        ).unwrap();
        assert_eq!(verified.candidates[0].outgoing, 0);
        assert!(verified.escrow.is_empty());
    }

    #[test]
    fn independent_verifier_replays_capped_overshoot_donation_policy() {
        let policy = vec![
            CandidateSpec {
                id: 10,
                weight: 300_000_000,
                destination_id: Some(20),
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
            },
            CandidateSpec {
                id: 20,
                weight: 300_000_000,
                destination_id: Some(30),
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
            },
            CandidateSpec {
                id: 30,
                weight: 200_000_000,
                destination_id: Some(40),
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
            },
        ];
        let mut writer = DonationLedger::new_capped_overshoot(policy.clone()).unwrap();
        writer.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        writer.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: 4 * EVIDENCE_ONE,
        }).unwrap();
        writer.apply(InputEvent::EvidenceUpdate {
            round: 2,
            candidate_id: 10,
            new_value: 5 * EVIDENCE_ONE,
        }).unwrap();

        let snapshot = writer.snapshot().unwrap();
        let verified = verify_receipt_with_policy(
            &policy,
            DonationReservePolicy::CappedOvershoot,
            writer.audit_log(),
            Some(&snapshot),
        ).unwrap();
        assert_eq!(verified.escrow[0].original_amount, 200_000_000_000_000);
        assert_eq!(verified.escrow[1].original_amount, 100_000_000_000_000);
        assert_eq!(verified.candidates[0].outgoing, 300_000_000_000_000);

        let mut tampered = writer.audit_log().to_vec();
        let reserve = tampered.iter_mut()
            .find(|event| matches!(event, AuditEvent::Reserved { .. }))
            .unwrap();
        if let AuditEvent::Reserved { amount, .. } = reserve {
            *amount += 1;
        }
        assert_eq!(
            verify_receipt_with_policy(
                &policy,
                DonationReservePolicy::CappedOvershoot,
                &tampered,
                None,
            ),
            Err(VerifyError::ReceiptMismatch)
        );
    }

    #[test]
    fn independently_verifies_hand_authored_receipt_without_writer_generation() {
        let policy = vec![
            CandidateSpec {
                id: 10,
                weight: 300_000_000,
                reserve_rate_numerator: 1,
                reserve_rate_denominator: 1,
                destination_id: Some(20),
            },
            CandidateSpec {
                id: 20,
                weight: 300_000_000,
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
                destination_id: Some(30),
            },
            CandidateSpec {
                id: 30,
                weight: 200_000_000,
                reserve_rate_numerator: 0,
                reserve_rate_denominator: 1,
                destination_id: None,
            },
        ];
        let receipt = vec![
            AuditEvent::Activated { round: 0, candidate_id: 10 },
            AuditEvent::Activated { round: 0, candidate_id: 20 },
            AuditEvent::EvidenceUpdated {
                round: 1,
                candidate_id: 10,
                new_value: EVIDENCE_ONE,
            },
            AuditEvent::Reserved {
                round: 1,
                token_id: 1,
                donor_id: 10,
                destination_id: 20,
                amount: 300_000_000_000,
            },
            AuditEvent::Credited {
                round: 2,
                token_id: 1,
                recipient_id: 20,
                amount: 150_000_000_000,
            },
        ];

        let verified = verify_receipt(&policy, &receipt, None).unwrap();
        assert_eq!(verified.candidates[0].weighted_score, 0);
        assert_eq!(verified.candidates[1].incoming, 150_000_000_000);
        assert_eq!(verified.candidates[2].evidence_value, 0);
        assert_eq!(verified.escrow[0].destination_id, 20);
        assert_eq!(verified.escrow[0].remaining_amount, 150_000_000_000);
    }

    #[test]
    fn independently_replays_reference_receipt() {
        let policy = specs();
        let mut writer = DonationLedger::new(policy.clone()).unwrap();
        writer.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        writer.apply(InputEvent::Activate { round: 0, candidate_id: 20 }).unwrap();
        writer.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        let token_amount = writer.snapshot().unwrap().escrow[0].original_amount;
        writer.apply(InputEvent::Credit {
            round: 2,
            token_id: 1,
            recipient_id: 20,
            amount: token_amount,
        }).unwrap();

        let verified = verify_receipt(
            &policy,
            writer.audit_log(),
            Some(&writer.snapshot().unwrap()),
        ).unwrap();
        assert_eq!(verified.candidates.len(), 3);
        assert_eq!(verified.escrow[0].remaining_amount, 0);
        assert_eq!(verified.candidates[1].incoming, token_amount);
    }

    #[test]
    fn independent_verifier_rejects_destination_tampering() {
        let policy = specs();
        let mut writer = DonationLedger::new(policy.clone()).unwrap();
        writer.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        writer.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();

        let mut tampered = writer.audit_log().to_vec();
        let reservation = tampered.iter_mut()
            .find(|event| matches!(event, AuditEvent::Reserved { .. }))
            .unwrap();
        if let AuditEvent::Reserved { destination_id, .. } = reservation {
            *destination_id = 30;
        }
        assert_eq!(
            verify_receipt(&policy, &tampered, None),
            Err(VerifyError::ReceiptMismatch)
        );
    }

    #[test]
    fn independent_verifier_rejects_modified_reservation_and_credit() {
        let policy = specs();
        let mut writer = DonationLedger::new(policy.clone()).unwrap();
        writer.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        writer.apply(InputEvent::Activate { round: 0, candidate_id: 20 }).unwrap();
        writer.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        let token_amount = writer.snapshot().unwrap().escrow[0].original_amount;
        writer.apply(InputEvent::Credit {
            round: 2,
            token_id: 1,
            recipient_id: 20,
            amount: token_amount,
        }).unwrap();

        let mut tampered = writer.audit_log().to_vec();
        let reservation = tampered.iter_mut()
            .find(|event| matches!(event, AuditEvent::Reserved { .. }))
            .unwrap();
        if let AuditEvent::Reserved { amount, .. } = reservation {
            *amount += 1;
        }
        assert_eq!(
            verify_receipt(&policy, &tampered, None),
            Err(VerifyError::ReceiptMismatch)
        );

        let mut tampered = writer.audit_log().to_vec();
        let credit = tampered.iter_mut()
            .find(|event| matches!(event, AuditEvent::Credited { .. }))
            .unwrap();
        if let AuditEvent::Credited { amount, .. } = credit {
            *amount += 1;
        }
        assert_eq!(
            verify_receipt(&policy, &tampered, None),
            Err(VerifyError::InvalidRecipient)
        );
    }
}
