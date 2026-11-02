#![forbid(unsafe_code)]

//! Independent linear-scan verifier for compound donation ledger receipts.
//!
//! This verifier consumes the ledger's wire-level types but does not call
//! DonationLedger, its transition helpers, or its invariant checker. It uses
//! an independently implemented map-based replay and per-prefix scan.

use std::collections::BTreeMap;

use symthaea_compound_donation_ledger::{
    AuditEvent, CandidateSpec, LedgerSnapshot, WEIGHT_SCALE, EVIDENCE_ONE,
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

pub fn verify_receipt(
    specs: &[CandidateSpec],
    receipt: &[AuditEvent],
    expected_snapshot: Option<&LedgerSnapshot>,
) -> Result<VerifiedSnapshot, VerifyError> {
    let mut spec_map = BTreeMap::<u64, CandidateSpec>::new();
    let mut state = BTreeMap::<u64, Candidate>::new();
    let mut weight_sum = 0_u64;
    let mut previous_id = None;

    for spec in specs {
        if previous_id.is_some_and(|previous| spec.id <= previous)
            || spec.reserve_rate_denominator == 0
            || spec.reserve_rate_numerator > spec.reserve_rate_denominator
            || spec.destination_id.is_some_and(|destination| destination <= spec.id)
            || (spec.destination_id.is_none() && spec.reserve_rate_numerator > 0)
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
                let weighted_delta = (spec.weight as u128)
                    .checked_mul(delta)
                    .ok_or(VerifyError::Overflow)?;
                let numerator = weighted_delta
                    .checked_mul(spec.reserve_rate_numerator as u128)
                    .and_then(|value| value.checked_add(candidate.remainder))
                    .ok_or(VerifyError::Overflow)?;
                let denominator = spec.reserve_rate_denominator as u128;
                let proposed = numerator / denominator;
                let next_remainder = numerator % denominator;
                let donor_limit = (spec.weight as u128)
                    .checked_mul((*new_value).min(EVIDENCE_ONE))
                    .ok_or(VerifyError::Overflow)?;
                let room = donor_limit
                    .checked_sub(candidate.outgoing)
                    .ok_or(VerifyError::InvalidAmount)?;
                let reserve = proposed.min(room);

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
                // The reference snapshot omits the fixed destination field. Reconstruct
                // it from the precommitted candidate specification.
                expected.escrow.iter().map(|token| {
                    let destination = spec_map
                        .get(&token.donor_id)
                        .and_then(|spec| spec.destination_id)
                        .unwrap_or(0);
                    VerifiedToken {
                        token_id: token.token_id,
                        donor_id: token.donor_id,
                        destination_id: destination,
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
