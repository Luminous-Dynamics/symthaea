#![forbid(unsafe_code)]

//! Exact, replayable reference core for incremental compound-evidence donations.
//!
//! Evidence values are fixed-point integers. WEIGHT_SCALE represents weight
//! one and EVIDENCE_ONE represents evidence value one. Ledger accounting and
//! staircase threshold comparisons use integer arithmetic only.

use std::collections::BTreeMap;

pub const WEIGHT_SCALE: u64 = 1_000_000_000;
pub const EVIDENCE_ONE: u128 = 1_000_000;
pub const PRIOR_MASS_SCALE: u128 = 1_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSpec {
    pub id: u64,
    pub weight: u64,
    pub reserve_rate_numerator: u64,
    pub reserve_rate_denominator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputEvent {
    Activate { round: u64, candidate_id: u64 },
    EvidenceUpdate { round: u64, candidate_id: u64, new_value: u128 },
    Credit { round: u64, token_id: u64, recipient_id: u64, amount: u128 },
}

impl InputEvent {
    fn round(&self) -> u64 {
        match self {
            Self::Activate { round, .. }
            | Self::EvidenceUpdate { round, .. }
            | Self::Credit { round, .. } => *round,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuditEvent {
    Activated { round: u64, candidate_id: u64 },
    EvidenceUpdated { round: u64, candidate_id: u64, new_value: u128 },
    Reserved {
        round: u64,
        token_id: u64,
        donor_id: u64,
        amount: u128,
    },
    Credited {
        round: u64,
        token_id: u64,
        recipient_id: u64,
        amount: u128,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSnapshot {
    pub id: u64,
    pub active: bool,
    pub evidence_value: u128,
    pub incoming: u128,
    pub outgoing: u128,
    /// Weighted compound evidence: weight * evidence + incoming - outgoing.
    /// This is gamma_i * C_i and avoids division by the candidate weight.
    pub weighted_score: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenSnapshot {
    pub token_id: u64,
    pub donor_id: u64,
    pub created_round: u64,
    pub original_amount: u128,
    pub remaining_amount: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerSnapshot {
    pub candidates: Vec<CandidateSnapshot>,
    pub escrow: Vec<TokenSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    InvalidWeightBudget,
    InvalidCandidateOrder,
    InvalidReserveRate,
    InvalidThresholdGrid,
    UnknownCandidate,
    CandidateAlreadyActive,
    CandidateDormant,
    EvidenceDecreased,
    RoundDecreased,
    InvalidTransferDirection,
    RecipientDormant,
    RecipientHasZeroWeight,
    UnknownToken,
    InvalidTransferAmount,
    ArithmeticOverflow,
    InvariantViolation,
    ReceiptMismatch,
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LedgerError {}

#[derive(Clone, Debug)]
struct CandidateState {
    active: bool,
    evidence_value: u128,
    incoming: u128,
    outgoing: u128,
    rate_remainder: u128,
}

#[derive(Clone, Debug)]
struct TokenState {
    donor_id: u64,
    created_round: u64,
    original_amount: u128,
    remaining_amount: u128,
}

#[derive(Clone, Debug)]
pub struct DonationLedger {
    specs: Vec<CandidateSpec>,
    index: BTreeMap<u64, usize>,
    states: Vec<CandidateState>,
    tokens: BTreeMap<u64, TokenState>,
    audit_log: Vec<AuditEvent>,
    next_token_id: u64,
    last_round: Option<u64>,
    total_incoming: u128,
    total_outgoing: u128,
}

impl DonationLedger {
    pub fn new(specs: Vec<CandidateSpec>) -> Result<Self, LedgerError> {
        let mut index = BTreeMap::new();
        let mut weight_sum = 0_u64;
        let mut previous_id = None;

        for (position, spec) in specs.iter().enumerate() {
            if previous_id.is_some_and(|previous| spec.id <= previous) {
                return Err(LedgerError::InvalidCandidateOrder);
            }
            if spec.reserve_rate_denominator == 0
                || spec.reserve_rate_numerator > spec.reserve_rate_denominator
            {
                return Err(LedgerError::InvalidReserveRate);
            }
            weight_sum = weight_sum
                .checked_add(spec.weight)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            if weight_sum > WEIGHT_SCALE {
                return Err(LedgerError::InvalidWeightBudget);
            }
            index.insert(spec.id, position);
            previous_id = Some(spec.id);
        }

        let states = specs
            .iter()
            .map(|_| CandidateState {
                active: false,
                evidence_value: 0,
                incoming: 0,
                outgoing: 0,
                rate_remainder: 0,
            })
            .collect();

        Ok(Self {
            specs,
            index,
            states,
            tokens: BTreeMap::new(),
            audit_log: Vec::new(),
            next_token_id: 1,
            last_round: None,
            total_incoming: 0,
            total_outgoing: 0,
        })
    }

    pub fn apply(&mut self, event: InputEvent) -> Result<(), LedgerError> {
        let round = event.round();
        if self.last_round.is_some_and(|last| round < last) {
            return Err(LedgerError::RoundDecreased);
        }

        match event {
            InputEvent::Activate { candidate_id, .. } => {
                let position = self.position(candidate_id)?;
                let state = &mut self.states[position];
                if state.active {
                    return Err(LedgerError::CandidateAlreadyActive);
                }
                state.active = true;
                self.audit_log.push(AuditEvent::Activated { round, candidate_id });
            }
            InputEvent::EvidenceUpdate { candidate_id, new_value, .. } => {
                self.apply_evidence_update(round, candidate_id, new_value)?;
            }
            InputEvent::Credit { token_id, recipient_id, amount, .. } => {
                self.apply_credit(round, token_id, recipient_id, amount)?;
            }
        }

        self.last_round = Some(round);
        self.validate_invariants()
    }

    fn apply_evidence_update(
        &mut self,
        round: u64,
        candidate_id: u64,
        new_value: u128,
    ) -> Result<(), LedgerError> {
        let position = self.position(candidate_id)?;
        let spec = &self.specs[position];
        let state = &self.states[position];

        if !state.active {
            return Err(LedgerError::CandidateDormant);
        }
        if new_value < state.evidence_value {
            return Err(LedgerError::EvidenceDecreased);
        }

        let delta = new_value - state.evidence_value;
        let weighted_delta = (spec.weight as u128)
            .checked_mul(delta)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let rate_product = weighted_delta
            .checked_mul(spec.reserve_rate_numerator as u128)
            .and_then(|value| value.checked_add(state.rate_remainder))
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let requested = rate_product / (spec.reserve_rate_denominator as u128);
        let next_remainder = rate_product % (spec.reserve_rate_denominator as u128);

        let donor_cap = (spec.weight as u128)
            .checked_mul(new_value.min(EVIDENCE_ONE))
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let remaining_cap = donor_cap
            .checked_sub(state.outgoing)
            .ok_or(LedgerError::InvariantViolation)?;
        let reserve = requested.min(remaining_cap);
        let next_outgoing = state
            .outgoing
            .checked_add(reserve)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let next_total_outgoing = self
            .total_outgoing
            .checked_add(reserve)
            .ok_or(LedgerError::ArithmeticOverflow)?;

        let weighted_value = (spec.weight as u128)
            .checked_mul(new_value)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let score_before = Self::weighted_score(spec, state)?;
        let score_after = weighted_value
            .checked_add(state.incoming)
            .and_then(|value| value.checked_sub(next_outgoing))
            .ok_or(LedgerError::InvariantViolation)?;
        if score_after < score_before {
            return Err(LedgerError::InvariantViolation);
        }

        let token_id = if reserve > 0 {
            Some(self.next_token_id)
        } else {
            None
        };
        let next_token_id = if token_id.is_some() {
            self.next_token_id
                .checked_add(1)
                .ok_or(LedgerError::ArithmeticOverflow)?
        } else {
            self.next_token_id
        };

        let state = &mut self.states[position];
        state.evidence_value = new_value;
        state.outgoing = next_outgoing;
        state.rate_remainder = next_remainder;
        self.total_outgoing = next_total_outgoing;

        self.audit_log.push(AuditEvent::EvidenceUpdated {
            round,
            candidate_id,
            new_value,
        });

        if let Some(token_id) = token_id {
            self.tokens.insert(
                token_id,
                TokenState {
                    donor_id: candidate_id,
                    created_round: round,
                    original_amount: reserve,
                    remaining_amount: reserve,
                },
            );
            self.next_token_id = next_token_id;
            self.audit_log.push(AuditEvent::Reserved {
                round,
                token_id,
                donor_id: candidate_id,
                amount: reserve,
            });
        }

        Ok(())
    }

    fn apply_credit(
        &mut self,
        round: u64,
        token_id: u64,
        recipient_id: u64,
        amount: u128,
    ) -> Result<(), LedgerError> {
        if amount == 0 {
            return Err(LedgerError::InvalidTransferAmount);
        }

        let recipient_position = self.position(recipient_id)?;
        let recipient_spec = &self.specs[recipient_position];
        if recipient_spec.weight == 0 {
            return Err(LedgerError::RecipientHasZeroWeight);
        }
        if !self.states[recipient_position].active {
            return Err(LedgerError::RecipientDormant);
        }

        let token = self.tokens.get(&token_id).ok_or(LedgerError::UnknownToken)?;
        if recipient_id <= token.donor_id {
            return Err(LedgerError::InvalidTransferDirection);
        }
        if round < token.created_round || amount > token.remaining_amount {
            return Err(LedgerError::InvalidTransferAmount);
        }

        let next_incoming = self.states[recipient_position]
            .incoming
            .checked_add(amount)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let next_total_incoming = self
            .total_incoming
            .checked_add(amount)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let score_before = Self::weighted_score(
            recipient_spec,
            &self.states[recipient_position],
        )?;
        let score_after = (recipient_spec.weight as u128)
            .checked_mul(self.states[recipient_position].evidence_value)
            .and_then(|value| value.checked_add(next_incoming))
            .and_then(|value| value.checked_sub(self.states[recipient_position].outgoing))
            .ok_or(LedgerError::InvariantViolation)?;
        if score_after < score_before {
            return Err(LedgerError::InvariantViolation);
        }

        let token = self.tokens.get_mut(&token_id).ok_or(LedgerError::UnknownToken)?;
        token.remaining_amount -= amount;
        self.states[recipient_position].incoming = next_incoming;
        self.total_incoming = next_total_incoming;
        self.audit_log.push(AuditEvent::Credited {
            round,
            token_id,
            recipient_id,
            amount,
        });
        Ok(())
    }

    fn position(&self, candidate_id: u64) -> Result<usize, LedgerError> {
        self.index
            .get(&candidate_id)
            .copied()
            .ok_or(LedgerError::UnknownCandidate)
    }

    fn weighted_score(
        spec: &CandidateSpec,
        state: &CandidateState,
    ) -> Result<u128, LedgerError> {
        (spec.weight as u128)
            .checked_mul(state.evidence_value)
            .and_then(|value| value.checked_add(state.incoming))
            .and_then(|value| value.checked_sub(state.outgoing))
            .ok_or(LedgerError::InvariantViolation)
    }

    pub fn weighted_score_for(&self, candidate_id: u64) -> Result<u128, LedgerError> {
        let position = self.position(candidate_id)?;
        Self::weighted_score(&self.specs[position], &self.states[position])
    }

    pub fn audit_log(&self) -> &[AuditEvent] {
        &self.audit_log
    }

    pub fn snapshot(&self) -> Result<LedgerSnapshot, LedgerError> {
        let mut candidates = Vec::with_capacity(self.specs.len());
        for (spec, state) in self.specs.iter().zip(self.states.iter()) {
            candidates.push(CandidateSnapshot {
                id: spec.id,
                active: state.active,
                evidence_value: state.evidence_value,
                incoming: state.incoming,
                outgoing: state.outgoing,
                weighted_score: Self::weighted_score(spec, state)?,
            });
        }

        let escrow = self
            .tokens
            .iter()
            .map(|(token_id, token)| TokenSnapshot {
                token_id: *token_id,
                donor_id: token.donor_id,
                created_round: token.created_round,
                original_amount: token.original_amount,
                remaining_amount: token.remaining_amount,
            })
            .collect();

        Ok(LedgerSnapshot { candidates, escrow })
    }

    pub fn validate_invariants(&self) -> Result<(), LedgerError> {
        if self.total_incoming > self.total_outgoing {
            return Err(LedgerError::InvariantViolation);
        }

        let mut prefix_incoming = 0_u128;
        let mut prefix_outgoing = 0_u128;

        for (spec, state) in self.specs.iter().zip(self.states.iter()) {
            let cap = (spec.weight as u128)
                .checked_mul(state.evidence_value.min(EVIDENCE_ONE))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            if state.outgoing > cap {
                return Err(LedgerError::InvariantViolation);
            }
            if !state.active
                && (state.evidence_value != 0 || state.incoming != 0 || state.outgoing != 0)
            {
                return Err(LedgerError::InvariantViolation);
            }
            Self::weighted_score(spec, state)?;
            prefix_incoming = prefix_incoming
                .checked_add(state.incoming)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            prefix_outgoing = prefix_outgoing
                .checked_add(state.outgoing)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            if prefix_incoming > prefix_outgoing {
                return Err(LedgerError::InvariantViolation);
            }
        }

        let mut token_remaining = 0_u128;
        for token in self.tokens.values() {
            if token.remaining_amount > token.original_amount {
                return Err(LedgerError::InvariantViolation);
            }
            token_remaining = token_remaining
                .checked_add(token.remaining_amount)
                .ok_or(LedgerError::ArithmeticOverflow)?;
        }
        if token_remaining > self.total_outgoing - self.total_incoming {
            return Err(LedgerError::InvariantViolation);
        }
        Ok(())
    }

    /// Replays a receipt independently and optionally matches its final snapshot.
    ///
    /// Reserved rows are derived events: replay recomputes them from evidence
    /// updates and compares the complete regenerated log with the supplied log.
    /// They are never accepted as caller-issued transfers.
    pub fn verify_receipt(
        specs: Vec<CandidateSpec>,
        receipt: &[AuditEvent],
        expected_snapshot: Option<&LedgerSnapshot>,
    ) -> Result<LedgerSnapshot, LedgerError> {
        let mut ledger = Self::new(specs)?;
        for event in receipt {
            match event {
                AuditEvent::Activated { round, candidate_id } => {
                    ledger.apply(InputEvent::Activate {
                        round: *round,
                        candidate_id: *candidate_id,
                    })?;
                }
                AuditEvent::EvidenceUpdated { round, candidate_id, new_value } => {
                    ledger.apply(InputEvent::EvidenceUpdate {
                        round: *round,
                        candidate_id: *candidate_id,
                        new_value: *new_value,
                    })?;
                }
                AuditEvent::Credited {
                    round,
                    token_id,
                    recipient_id,
                    amount,
                } => {
                    ledger.apply(InputEvent::Credit {
                        round: *round,
                        token_id: *token_id,
                        recipient_id: *recipient_id,
                        amount: *amount,
                    })?;
                }
                AuditEvent::Reserved { .. } => continue,
            }
        }

        if ledger.audit_log != receipt {
            return Err(LedgerError::ReceiptMismatch);
        }
        let snapshot = ledger.snapshot()?;
        if expected_snapshot.is_some_and(|expected| expected != &snapshot) {
            return Err(LedgerError::ReceiptMismatch);
        }
        Ok(snapshot)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThresholdMass {
    /// Evidence threshold in fixed-point units; one is EVIDENCE_ONE.
    pub threshold: u128,
    /// Prior mass in units of PRIOR_MASS_SCALE.
    pub prior_mass: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaircaseAdjuster {
    points: Vec<ThresholdMass>,
}

impl StaircaseAdjuster {
    pub fn new(mut points: Vec<ThresholdMass>) -> Result<Self, LedgerError> {
        points.sort_by_key(|point| point.threshold);
        let mut total_mass = 0_u128;
        let mut previous_threshold = None;
        for point in &points {
            if point.threshold < EVIDENCE_ONE
                || previous_threshold.is_some_and(|previous| point.threshold == previous)
            {
                return Err(LedgerError::InvalidThresholdGrid);
            }
            total_mass = total_mass
                .checked_add(point.prior_mass)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            previous_threshold = Some(point.threshold);
        }
        if total_mass > PRIOR_MASS_SCALE {
            return Err(LedgerError::InvalidWeightBudget);
        }
        Ok(Self { points })
    }

    /// Evaluates the staircase calibrator at a running-supremum value.
    ///
    /// The exact result is rounded down to evidence ticks. This cannot increase
    /// calibrated evidence and avoids floating-point threshold ambiguity.
    pub fn value_at(&self, running_supremum: u128) -> Result<u128, LedgerError> {
        let mut numerator = 0_u128;
        for point in self.points.iter().take_while(|point| point.threshold <= running_supremum) {
            let contribution = point
                .prior_mass
                .checked_mul(point.threshold)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            numerator = numerator
                .checked_add(contribution)
                .ok_or(LedgerError::ArithmeticOverflow)?;
        }
        Ok(numerator / PRIOR_MASS_SCALE)
    }

    pub fn points(&self) -> &[ThresholdMass] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: u64, weight: u64, rate_num: u64, rate_den: u64) -> CandidateSpec {
        CandidateSpec {
            id,
            weight,
            reserve_rate_numerator: rate_num,
            reserve_rate_denominator: rate_den,
        }
    }

    fn basic_specs() -> Vec<CandidateSpec> {
        vec![
            spec(10, 300_000_000, 1, 1),
            spec(20, 300_000_000, 1, 1),
            spec(30, 200_000_000, 1, 1),
        ]
    }

    #[test]
    fn reserves_only_from_own_new_evidence_and_respects_cap() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: 500_000,
        }).unwrap();

        let first = ledger.snapshot().unwrap();
        let donor = &first.candidates[0];
        assert_eq!(donor.outgoing, 300_000_000_u128 * 500_000_u128);
        assert_eq!(donor.weighted_score, 0);
        assert_eq!(first.escrow.len(), 1);

        ledger.apply(InputEvent::EvidenceUpdate {
            round: 2,
            candidate_id: 10,
            new_value: 2 * EVIDENCE_ONE,
        }).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        let donor = &snapshot.candidates[0];
        assert_eq!(donor.outgoing, 300_000_000_u128 * EVIDENCE_ONE);
        assert_eq!(donor.weighted_score, 300_000_000_u128 * EVIDENCE_ONE);
        ledger.validate_invariants().unwrap();
    }

    #[test]
    fn incoming_donations_cannot_be_redonated() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 20 }).unwrap();
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        let token_amount = ledger.snapshot().unwrap().escrow[0].original_amount;
        ledger.apply(InputEvent::Credit {
            round: 2,
            token_id: 1,
            recipient_id: 20,
            amount: token_amount,
        }).unwrap();
        let before = ledger.weighted_score_for(20).unwrap();

        ledger.apply(InputEvent::EvidenceUpdate {
            round: 3,
            candidate_id: 20,
            new_value: 500_000,
        }).unwrap();

        let after = ledger.weighted_score_for(20).unwrap();
        assert!(after >= before);
        let outgoing = ledger.snapshot().unwrap().candidates[1].outgoing;
        assert_eq!(outgoing, 300_000_000_u128 * 500_000_u128);
        ledger.validate_invariants().unwrap();
    }

    #[test]
    fn dormant_recipient_and_backward_transfers_are_rejected() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        for id in [10, 20, 30] {
            ledger.apply(InputEvent::Activate { round: 0, candidate_id: id }).unwrap();
        }
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 20,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        assert_eq!(
            ledger.apply(InputEvent::Credit {
                round: 2,
                token_id: 1,
                recipient_id: 10,
                amount: 1,
            }),
            Err(LedgerError::InvalidTransferDirection)
        );

        let mut second = DonationLedger::new(basic_specs()).unwrap();
        second.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        second.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        assert_eq!(
            second.apply(InputEvent::Credit {
                round: 2,
                token_id: 1,
                recipient_id: 20,
                amount: 1,
            }),
            Err(LedgerError::RecipientDormant)
        );
    }

    #[test]
    fn credits_are_single_spend_and_prefix_budget_holds() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        for id in [10, 20, 30] {
            ledger.apply(InputEvent::Activate { round: 0, candidate_id: id }).unwrap();
        }
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        let amount = ledger.snapshot().unwrap().escrow[0].original_amount;
        ledger.apply(InputEvent::Credit {
            round: 2,
            token_id: 1,
            recipient_id: 20,
            amount,
        }).unwrap();

        assert_eq!(
            ledger.apply(InputEvent::Credit {
                round: 3,
                token_id: 1,
                recipient_id: 30,
                amount: 1,
            }),
            Err(LedgerError::InvalidTransferAmount)
        );
        ledger.validate_invariants().unwrap();
    }

    #[test]
    fn exact_receipt_replay_rejects_tampering() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 20 }).unwrap();
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        let amount = ledger.snapshot().unwrap().escrow[0].original_amount;
        ledger.apply(InputEvent::Credit {
            round: 2,
            token_id: 1,
            recipient_id: 20,
            amount,
        }).unwrap();

        let snapshot = ledger.snapshot().unwrap();
        DonationLedger::verify_receipt(
            basic_specs(),
            ledger.audit_log(),
            Some(&snapshot),
        ).unwrap();

        let mut tampered = ledger.audit_log().to_vec();
        let row = tampered.iter_mut().find(|event| matches!(event, AuditEvent::Reserved { .. })).unwrap();
        if let AuditEvent::Reserved { amount, .. } = row {
            *amount += 1;
        }
        assert_eq!(
            DonationLedger::verify_receipt(basic_specs(), &tampered, None),
            Err(LedgerError::ReceiptMismatch)
        );
    }

    #[test]
    fn future_horizon_extension_preserves_existing_scores() {
        let small_specs = vec![spec(10, 300_000_000, 1, 1), spec(20, 300_000_000, 1, 1)];
        let large_specs = vec![
            spec(10, 300_000_000, 1, 1),
            spec(20, 300_000_000, 1, 1),
            spec(30, 200_000_000, 1, 1),
        ];
        let events = vec![
            InputEvent::Activate { round: 0, candidate_id: 10 },
            InputEvent::EvidenceUpdate {
                round: 1,
                candidate_id: 10,
                new_value: EVIDENCE_ONE,
            },
            InputEvent::Activate { round: 2, candidate_id: 20 },
            InputEvent::EvidenceUpdate {
                round: 3,
                candidate_id: 20,
                new_value: 500_000,
            },
        ];

        let mut small = DonationLedger::new(small_specs).unwrap();
        let mut large = DonationLedger::new(large_specs).unwrap();
        for event in events {
            small.apply(event.clone()).unwrap();
            large.apply(event).unwrap();
        }
        for id in [10, 20] {
            assert_eq!(small.weighted_score_for(id), large.weighted_score_for(id));
        }

        large.apply(InputEvent::Activate { round: 4, candidate_id: 30 }).unwrap();
        let token_amount = large.snapshot().unwrap().escrow[0].remaining_amount;
        large.apply(InputEvent::Credit {
            round: 5,
            token_id: 1,
            recipient_id: 30,
            amount: token_amount,
        }).unwrap();
        for id in [10, 20] {
            assert_eq!(small.weighted_score_for(id), large.weighted_score_for(id));
        }
        large.validate_invariants().unwrap();
    }

    #[test]
    fn staircase_adjuster_uses_exact_inclusive_thresholds() {
        let adjuster = StaircaseAdjuster::new(vec![
            ThresholdMass { threshold: 10 * EVIDENCE_ONE, prior_mass: 200_000_000 },
            ThresholdMass { threshold: 100 * EVIDENCE_ONE, prior_mass: 300_000_000 },
        ]).unwrap();

        assert_eq!(adjuster.value_at(10 * EVIDENCE_ONE).unwrap(), 2 * EVIDENCE_ONE);
        assert_eq!(
            adjuster.value_at(100 * EVIDENCE_ONE).unwrap(),
            32 * EVIDENCE_ONE
        );
        assert_eq!(adjuster.value_at(100 * EVIDENCE_ONE - 1).unwrap(), 2 * EVIDENCE_ONE);
        assert_eq!(adjuster.value_at(EVIDENCE_ONE).unwrap(), 0);
    }

    #[test]
    fn staircase_adjuster_rejects_unbudgeted_or_invalid_grids() {
        assert_eq!(
            StaircaseAdjuster::new(vec![
                ThresholdMass { threshold: EVIDENCE_ONE, prior_mass: 700_000_000 },
                ThresholdMass { threshold: 2 * EVIDENCE_ONE, prior_mass: 400_000_000 },
            ]),
            Err(LedgerError::InvalidWeightBudget)
        );
        assert_eq!(
            StaircaseAdjuster::new(vec![
                ThresholdMass { threshold: EVIDENCE_ONE - 1, prior_mass: 1 },
            ]),
            Err(LedgerError::InvalidThresholdGrid)
        );
    }

    #[test]
    fn rejects_bad_weights_rates_and_round_order() {
        assert_eq!(
            DonationLedger::new(vec![spec(1, WEIGHT_SCALE, 2, 1)]),
            Err(LedgerError::InvalidReserveRate)
        );
        assert_eq!(
            DonationLedger::new(vec![
                spec(1, WEIGHT_SCALE, 1, 1),
                spec(2, 1, 1, 1),
            ]),
            Err(LedgerError::InvalidWeightBudget)
        );

        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        ledger.apply(InputEvent::Activate { round: 2, candidate_id: 10 }).unwrap();
        assert_eq!(
            ledger.apply(InputEvent::EvidenceUpdate {
                round: 1,
                candidate_id: 10,
                new_value: 1,
            }),
            Err(LedgerError::RoundDecreased)
        );
    }
}
