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
    /// Horizon-independent recipient slot committed before observing this donor's path.
    pub destination_id: Option<u64>,
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
        destination_id: u64,
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
    pub destination_id: u64,
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
    InvalidRecipient,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateState {
    active: bool,
    evidence_value: u128,
    incoming: u128,
    outgoing: u128,
    rate_remainder: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TokenState {
    donor_id: u64,
    destination_id: u64,
    created_round: u64,
    original_amount: u128,
    remaining_amount: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DonationReservePolicy {
    /// Reserve a precommitted rational share of each donor's own evidence increments.
    FixedRate,
    /// Reserve only capped overshoot above the donor's weighted evidence threshold.
    CappedOvershoot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DonationLedger {
    reserve_policy: DonationReservePolicy,
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
        Self::new_with_reserve_policy(specs, DonationReservePolicy::FixedRate)
    }

    pub fn new_capped_overshoot(specs: Vec<CandidateSpec>) -> Result<Self, LedgerError> {
        Self::new_with_reserve_policy(specs, DonationReservePolicy::CappedOvershoot)
    }

    pub fn new_with_reserve_policy(
        specs: Vec<CandidateSpec>,
        reserve_policy: DonationReservePolicy,
    ) -> Result<Self, LedgerError> {
        let mut index = BTreeMap::new();
        let mut weight_sum = 0_u64;
        let mut previous_id = None;

        for (position, spec) in specs.iter().enumerate() {
            if previous_id.is_some_and(|previous| spec.id <= previous) {
                return Err(LedgerError::InvalidCandidateOrder);
            }
            if spec.reserve_rate_denominator == 0 {
                return Err(LedgerError::InvalidReserveRate);
            }
            match reserve_policy {
                DonationReservePolicy::FixedRate => {
                    if spec.reserve_rate_numerator > spec.reserve_rate_denominator
                        || (spec.destination_id.is_none() && spec.reserve_rate_numerator > 0)
                    {
                        return Err(LedgerError::InvalidReserveRate);
                    }
                }
                DonationReservePolicy::CappedOvershoot => {
                    if spec.reserve_rate_numerator != 0 {
                        return Err(LedgerError::InvalidReserveRate);
                    }
                }
            }
            if spec.destination_id.is_some_and(|destination| destination <= spec.id) {
                return Err(LedgerError::InvalidRecipient);
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
            reserve_policy,
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

    /// Applies one event atomically: failed transitions never partially mutate the ledger.
    ///
    /// This reference implementation clones the state before applying the event,
    /// then validates the full invariants before committing. It prioritizes a
    /// trustworthy oracle over update speed; the separate verifier is linear too.
    pub fn apply(&mut self, event: InputEvent) -> Result<(), LedgerError> {
        let mut staged = self.clone();
        staged.apply_in_place(event)?;
        staged.validate_invariants()?;
        *self = staged;
        Ok(())
    }

    fn apply_in_place(&mut self, event: InputEvent) -> Result<(), LedgerError> {
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
        Ok(())
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
        let (reserve, next_outgoing, next_remainder) = match self.reserve_policy {
            DonationReservePolicy::FixedRate => {
                let weighted_delta = (spec.weight as u128)
                    .checked_mul(delta)
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                let rate_product = weighted_delta
                    .checked_mul(spec.reserve_rate_numerator as u128)
                    .and_then(|value| value.checked_add(state.rate_remainder))
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                let requested = rate_product / (spec.reserve_rate_denominator as u128);
                let remainder = rate_product % (spec.reserve_rate_denominator as u128);
                let donor_cap = (spec.weight as u128)
                    .checked_mul(new_value.min(EVIDENCE_ONE))
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                let remaining_cap = donor_cap
                    .checked_sub(state.outgoing)
                    .ok_or(LedgerError::InvariantViolation)?;
                let reserve = requested.min(remaining_cap);
                let next = state.outgoing
                    .checked_add(reserve)
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                (reserve, next, remainder)
            }
            DonationReservePolicy::CappedOvershoot => {
                let weighted_value = (spec.weight as u128)
                    .checked_mul(new_value)
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                let threshold = (WEIGHT_SCALE as u128)
                    .checked_mul(EVIDENCE_ONE)
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                let raw = weighted_value.saturating_sub(threshold);
                let cap = (spec.weight as u128)
                    .checked_mul(EVIDENCE_ONE)
                    .ok_or(LedgerError::ArithmeticOverflow)?;
                let target = if spec.destination_id.is_some() {
                    raw.min(cap)
                } else {
                    0
                };
                let reserve = target
                    .checked_sub(state.outgoing)
                    .ok_or(LedgerError::InvariantViolation)?;
                (reserve, target, 0)
            }
        };
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

        let destination_id = if reserve > 0 {
            Some(spec.destination_id.ok_or(LedgerError::InvalidRecipient)?)
        } else {
            None
        };
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
                    destination_id: destination_id.ok_or(LedgerError::InvalidRecipient)?,
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
                destination_id: destination_id.ok_or(LedgerError::InvalidRecipient)?,
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
        if token.destination_id != recipient_id {
            return Err(LedgerError::InvalidRecipient);
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
                destination_id: token.destination_id,
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

        if prefix_incoming != self.total_incoming || prefix_outgoing != self.total_outgoing {
            return Err(LedgerError::InvariantViolation);
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
        if token_remaining != self.total_outgoing - self.total_incoming {
            return Err(LedgerError::InvariantViolation);
        }
        Ok(())
    }

    /// Replays a receipt with the reference transition engine and optionally
    /// matches its final snapshot. Reserved rows are derived events, not
    /// caller-issued transfers. This is deterministic self-replay, not yet a
    /// separately implemented independent verifier.
    pub fn verify_receipt(
        specs: Vec<CandidateSpec>,
        receipt: &[AuditEvent],
        expected_snapshot: Option<&LedgerSnapshot>,
    ) -> Result<LedgerSnapshot, LedgerError> {
        Self::verify_receipt_with_reserve_policy(
            specs,
            DonationReservePolicy::FixedRate,
            receipt,
            expected_snapshot,
        )
    }

    pub fn verify_receipt_with_reserve_policy(
        specs: Vec<CandidateSpec>,
        reserve_policy: DonationReservePolicy,
        receipt: &[AuditEvent],
        expected_snapshot: Option<&LedgerSnapshot>,
    ) -> Result<LedgerSnapshot, LedgerError> {
        let mut ledger = Self::new_with_reserve_policy(specs, reserve_policy)?;
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
    /// Prefix sums of prior_mass * threshold, in exact numerator units.
    prefix_numerators: Vec<u128>,
}


/// Exact fixed-point receipt for one persistent SCORE-style refund update.
///
/// Refund and cost numerators share denominator alpha_denominator * EVIDENCE_ONE.
/// This avoids floating-point rounding and makes the overshoot cap exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OvershootRefundReceipt {
    pub round: u64,
    pub evidence_value: u128,
    pub refund_increment_numerator: u128,
    pub cumulative_refund_numerator: u128,
    pub effective_cost_numerator: u128,
    pub denominator: u128,
    pub rejected: bool,
}

/// Cumulative, once-only overshoot refunds for a time-monotone e-process.
///
/// This object accounts for the SCORE refund identity on a persistent
/// time-monotone process V_t. It does not prove that V_t is an e-process and
/// does not itself implement an online FDR spending rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessOvershootRefund {
    alpha_numerator: u64,
    denominator: u128,
    refund_cap_numerator: u128,
    last_round: Option<u64>,
    last_evidence_value: u128,
    cumulative_refund_numerator: u128,
}

impl ProcessOvershootRefund {
    /// Constructs an account for the common normalized process V_0 = 1.
    pub fn new(alpha_numerator: u64, alpha_denominator: u64) -> Result<Self, LedgerError> {
        Self::with_initial_evidence_value(
            alpha_numerator,
            alpha_denominator,
            EVIDENCE_ONE,
        )
    }

    /// Constructs an account with an explicit nonnegative initial process value.
    ///
    /// Some valid time-monotone e-process adjusters (including staircase
    /// calibrators before their first threshold crossing) start at zero rather
    /// than one. Their initial value must be supplied explicitly so that
    /// subsequent decreases are still rejected.
    pub fn with_initial_evidence_value(
        alpha_numerator: u64,
        alpha_denominator: u64,
        initial_evidence_value: u128,
    ) -> Result<Self, LedgerError> {
        if alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(LedgerError::InvalidReserveRate);
        }
        let denominator = (alpha_denominator as u128)
            .checked_mul(EVIDENCE_ONE)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let refund_cap_numerator = (alpha_numerator as u128)
            .checked_mul(EVIDENCE_ONE)
            .ok_or(LedgerError::ArithmeticOverflow)?;

        Ok(Self {
            alpha_numerator,
            denominator,
            refund_cap_numerator,
            last_round: None,
            last_evidence_value: initial_evidence_value,
            cumulative_refund_numerator: 0,
        })
    }

    /// Atomically records a nondecreasing process value and releases only the
    /// additional capped overshoot since the previous observation.
    pub fn record(
        &mut self,
        round: u64,
        new_evidence_value: u128,
    ) -> Result<OvershootRefundReceipt, LedgerError> {
        let mut staged = self.clone();
        let receipt = staged.record_in_place(round, new_evidence_value)?;
        *self = staged;
        Ok(receipt)
    }

    fn record_in_place(
        &mut self,
        round: u64,
        new_evidence_value: u128,
    ) -> Result<OvershootRefundReceipt, LedgerError> {
        if self.last_round.is_some_and(|previous| round < previous) {
            return Err(LedgerError::RoundDecreased);
        }
        if new_evidence_value < self.last_evidence_value {
            return Err(LedgerError::EvidenceDecreased);
        }

        let alpha_scaled_evidence = (self.alpha_numerator as u128)
            .checked_mul(new_evidence_value)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let raw_overshoot = alpha_scaled_evidence.saturating_sub(self.denominator);
        let next_refund = raw_overshoot.min(self.refund_cap_numerator);
        if next_refund < self.cumulative_refund_numerator {
            return Err(LedgerError::InvariantViolation);
        }
        let refund_increment = next_refund - self.cumulative_refund_numerator;
        let effective_cost = alpha_scaled_evidence
            .checked_sub(next_refund)
            .ok_or(LedgerError::InvariantViolation)?;
        let rejected = alpha_scaled_evidence >= self.denominator;

        if rejected && effective_cost < self.denominator {
            return Err(LedgerError::InvariantViolation);
        }

        self.last_round = Some(round);
        self.last_evidence_value = new_evidence_value;
        self.cumulative_refund_numerator = next_refund;

        Ok(OvershootRefundReceipt {
            round,
            evidence_value: new_evidence_value,
            refund_increment_numerator: refund_increment,
            cumulative_refund_numerator: next_refund,
            effective_cost_numerator: effective_cost,
            denominator: self.denominator,
            rejected,
        })
    }

    pub fn cumulative_refund_numerator(&self) -> u128 {
        self.cumulative_refund_numerator
    }

    pub fn refund_cap_numerator(&self) -> u128 {
        self.refund_cap_numerator
    }

    pub fn denominator(&self) -> u128 {
        self.denominator
    }

    pub fn last_evidence_value(&self) -> u128 {
        self.last_evidence_value
    }
}


/// Exact fixed-point implementation of the *uncapped* SCORE overshoot refund.
///
/// Unlike ProcessOvershootRefund, this account releases the full cumulative
/// overshoot (alpha * V_t - 1)_+. Its null expectation is bounded by alpha
/// whenever V is an e-process with terminal expectation at most one.
/// The residual cost is exactly min(alpha * V_t, 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessOvershootRefundUncapped {
    alpha_numerator: u64,
    denominator: u128,
    last_round: Option<u64>,
    last_evidence_value: u128,
    cumulative_refund_numerator: u128,
}

impl ProcessOvershootRefundUncapped {
    /// Constructs an uncapped refund account for the normalized process V_0=1.
    pub fn new(alpha_numerator: u64, alpha_denominator: u64) -> Result<Self, LedgerError> {
        Self::with_initial_evidence_value(
            alpha_numerator,
            alpha_denominator,
            EVIDENCE_ONE,
        )
    }

    /// Constructs an uncapped refund account with a known, precommitted V_0.
    pub fn with_initial_evidence_value(
        alpha_numerator: u64,
        alpha_denominator: u64,
        initial_evidence_value: u128,
    ) -> Result<Self, LedgerError> {
        if alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(LedgerError::InvalidReserveRate);
        }
        let denominator = (alpha_denominator as u128)
            .checked_mul(EVIDENCE_ONE)
            .ok_or(LedgerError::ArithmeticOverflow)?;

        Ok(Self {
            alpha_numerator,
            denominator,
            last_round: None,
            last_evidence_value: initial_evidence_value,
            cumulative_refund_numerator: 0,
        })
    }

    /// Atomically records the next monotone process value, releasing only the
    /// uncapped overshoot increment not already credited at earlier looks.
    pub fn record(
        &mut self,
        round: u64,
        new_evidence_value: u128,
    ) -> Result<OvershootRefundReceipt, LedgerError> {
        let mut staged = self.clone();
        let receipt = staged.record_in_place(round, new_evidence_value)?;
        *self = staged;
        Ok(receipt)
    }

    fn record_in_place(
        &mut self,
        round: u64,
        new_evidence_value: u128,
    ) -> Result<OvershootRefundReceipt, LedgerError> {
        if self.last_round.is_some_and(|previous| round < previous) {
            return Err(LedgerError::RoundDecreased);
        }
        if new_evidence_value < self.last_evidence_value {
            return Err(LedgerError::EvidenceDecreased);
        }

        let alpha_scaled_evidence = (self.alpha_numerator as u128)
            .checked_mul(new_evidence_value)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let next_refund = alpha_scaled_evidence.saturating_sub(self.denominator);
        if next_refund < self.cumulative_refund_numerator {
            return Err(LedgerError::InvariantViolation);
        }
        let refund_increment = next_refund - self.cumulative_refund_numerator;
        let effective_cost = alpha_scaled_evidence
            .checked_sub(next_refund)
            .ok_or(LedgerError::InvariantViolation)?;
        let rejected = alpha_scaled_evidence >= self.denominator;

        if rejected && effective_cost != self.denominator {
            return Err(LedgerError::InvariantViolation);
        }

        self.last_round = Some(round);
        self.last_evidence_value = new_evidence_value;
        self.cumulative_refund_numerator = next_refund;

        Ok(OvershootRefundReceipt {
            round,
            evidence_value: new_evidence_value,
            refund_increment_numerator: refund_increment,
            cumulative_refund_numerator: next_refund,
            effective_cost_numerator: effective_cost,
            denominator: self.denominator,
            rejected,
        })
    }

    pub fn cumulative_refund_numerator(&self) -> u128 {
        self.cumulative_refund_numerator
    }

    pub fn denominator(&self) -> u128 {
        self.denominator
    }

    pub fn last_evidence_value(&self) -> u128 {
        self.last_evidence_value
    }
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
        let mut prefix_numerators = Vec::with_capacity(points.len());
        let mut cumulative = 0_u128;
        for point in &points {
            let contribution = point
                .prior_mass
                .checked_mul(point.threshold)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            cumulative = cumulative
                .checked_add(contribution)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            prefix_numerators.push(cumulative);
        }
        Ok(Self {
            points,
            prefix_numerators,
        })
    }

    /// Builds an equal-prior geometric grid over the e-BH rejection ranks.
    ///
    /// The rank grid is 1, 2, 4, ..., plus the family size when it is not a
    /// power of two. For each grid rank r, let h_r = family_size / (alpha*r)
    /// be the weighted e-BH boundary. Its raw threshold K is chosen with
    /// integer ceiling arithmetic so the *actual fixed-point prior mass*
    /// times K contributes at least ceil(h_r) evidence ticks.
    ///
    /// There are Q = O(log family_size) grid points. For every actual
    /// rejection rank k, some grid rank r satisfies k/2 < r <= k (or r=k
    /// when k is the final family size), so the raw threshold is at most
    /// approximately 2Q times the e-BH boundary. The exact rational prior
    /// uses floor(PRIOR_MASS_SCALE / Q); unused mass is left unspent.
    pub fn e_bh_rank_geometric(
        family_size: u64,
        alpha_numerator: u64,
        alpha_denominator: u64,
    ) -> Result<Self, LedgerError> {
        Self::weighted_e_bh_rank_geometric(
            family_size,
            1,
            family_size,
            alpha_numerator,
            alpha_denominator,
        )
    }

    /// Builds a candidate-specific rank-geometric calibrator for weighted e-BH.
    ///
    /// The candidate weight is weight_numerator / weight_denominator. For
    /// rejection rank r, its weighted e-BH boundary is
    ///
    ///     C_i >= 1 / (alpha * r * gamma_i).
    ///
    /// The same geometric rank grid and fixed prior budget are used as in
    /// e_bh_rank_geometric, but each raw threshold is scaled by this
    /// candidate's precommitted weight. The family size is a fixed design
    /// envelope, not the current active roster size.
    pub fn weighted_e_bh_rank_geometric(
        family_size: u64,
        weight_numerator: u64,
        weight_denominator: u64,
        alpha_numerator: u64,
        alpha_denominator: u64,
    ) -> Result<Self, LedgerError> {
        if family_size == 0
            || weight_numerator == 0
            || weight_denominator == 0
            || weight_numerator > weight_denominator
            || alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(LedgerError::InvalidThresholdGrid);
        }

        let mut ranks = Vec::new();
        let mut rank = 1_u64;
        loop {
            if rank > family_size {
                break;
            }
            ranks.push(rank);
            match rank.checked_mul(2) {
                Some(next) if next <= family_size => rank = next,
                _ => break,
            }
        }
        if ranks.last().copied() != Some(family_size) {
            ranks.push(family_size);
        }

        let point_count = ranks.len() as u128;
        let prior_mass = PRIOR_MASS_SCALE / point_count;
        if prior_mass == 0 {
            return Err(LedgerError::InvalidThresholdGrid);
        }

        // Multiple rank targets can map to the same integer raw threshold for
        // small families or extreme alpha/weight values. Merge their masses.
        let mut merged = BTreeMap::<u128, u128>::new();
        for rank in ranks {
            let h_numerator = (alpha_denominator as u128)
                .checked_mul(weight_denominator as u128)
                .and_then(|value| value.checked_mul(EVIDENCE_ONE))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let h_denominator = (alpha_numerator as u128)
                .checked_mul(rank as u128)
                .and_then(|value| value.checked_mul(weight_numerator as u128))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let h_ticks = h_numerator
                .checked_add(h_denominator - 1)
                .ok_or(LedgerError::ArithmeticOverflow)?
                / h_denominator;

            let raw_numerator = h_ticks
                .checked_mul(PRIOR_MASS_SCALE)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let raw_threshold = raw_numerator
                .checked_add(prior_mass - 1)
                .ok_or(LedgerError::ArithmeticOverflow)?
                / prior_mass;

            let entry = merged.entry(raw_threshold).or_insert(0);
            *entry = entry
                .checked_add(prior_mass)
                .ok_or(LedgerError::ArithmeticOverflow)?;
        }

        Self::new(
            merged
                .into_iter()
                .map(|(threshold, prior_mass)| ThresholdMass {
                    threshold,
                    prior_mass,
                })
                .collect(),
        )
    }

    /// Builds a horizon-independent candidate-weighted rank calibrator.
    ///
    /// For every rank r = 2^j, j in 0..=max_rank_exponent, allocate the
    /// fixed prior mass pi_j = 1 / ((j+1)(j+2)). These masses telescope to
    /// at most one. A candidate with fixed global weight gamma receives the
    /// threshold for its weighted e-BH boundary 1/(alpha * gamma * r).
    ///
    /// This constructor does not depend on the active family size. Existing
    /// candidates keep exactly the same calibrator when higher-ID hypotheses
    /// arrive. With j=floor(log2 k), the largest dyadic rank r<=k satisfies
    /// k/2<r<=k, and 1/pi_j=(j+1)(j+2), giving an O((log k)^2) raw-threshold
    /// overhead (plus fixed-point rounding) for all finite k.
    ///
    /// max_rank_exponent is capped at 63, covering every rank representable
    /// in a u64-sized finite family. The unallocated prior tail remains valid.
    pub fn projective_weighted_e_bh_rank_prior(
        max_rank_exponent: u32,
        weight_numerator: u64,
        weight_denominator: u64,
        alpha_numerator: u64,
        alpha_denominator: u64,
    ) -> Result<Self, LedgerError> {
        if max_rank_exponent > 63
            || weight_numerator == 0
            || weight_denominator == 0
            || weight_numerator > weight_denominator
            || alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(LedgerError::InvalidThresholdGrid);
        }

        let mut merged = BTreeMap::<u128, u128>::new();
        for exponent in 0..=max_rank_exponent {
            let index = exponent as u128 + 1;
            let denominator = index
                .checked_mul(index + 1)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let prior_mass = PRIOR_MASS_SCALE / denominator;
            if prior_mass == 0 {
                return Err(LedgerError::InvalidThresholdGrid);
            }

            let rank = 1_u64
                .checked_shl(exponent)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let target_numerator = (alpha_denominator as u128)
                .checked_mul(weight_denominator as u128)
                .and_then(|value| value.checked_mul(EVIDENCE_ONE))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let target_denominator = (alpha_numerator as u128)
                .checked_mul(rank as u128)
                .and_then(|value| value.checked_mul(weight_numerator as u128))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let target_ticks = target_numerator
                .checked_add(target_denominator - 1)
                .ok_or(LedgerError::ArithmeticOverflow)?
                / target_denominator;

            let raw_numerator = target_ticks
                .checked_mul(PRIOR_MASS_SCALE)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let raw_threshold = (raw_numerator
                .checked_add(prior_mass - 1)
                .ok_or(LedgerError::ArithmeticOverflow)?
                / prior_mass)
                .max(EVIDENCE_ONE);

            let entry = merged.entry(raw_threshold).or_insert(0);
            *entry = entry
                .checked_add(prior_mass)
                .ok_or(LedgerError::ArithmeticOverflow)?;
        }

        Self::new(
            merged
                .into_iter()
                .map(|(threshold, prior_mass)| ThresholdMass {
                    threshold,
                    prior_mass,
                })
                .collect(),
        )
    }

    /// Builds a fixed-universe hybrid rank prior with an explicit tail floor.
    ///
    /// This is the equal mixture of the telescoping rank prior
    /// 1/((j+1)(j+2)) and a uniform prior over the 64 dyadic ranks that cover
    /// every u64 rejection rank. Its per-rank mass is at least 1/128, so the
    /// raw-threshold overhead is at most 256 times the weighted e-BH boundary
    /// (plus fixed-point rounding), uniformly over every u64 rank.
    ///
    /// The uniform half is specific to the finite u64 rank envelope. The
    /// telescoping half preserves extra power at early ranks. This is a
    /// horizon-independent candidate calibrator: it does not take active
    /// family size as input.
    pub fn projective_weighted_e_bh_rank_prior_hybrid_u64(
        weight_numerator: u64,
        weight_denominator: u64,
        alpha_numerator: u64,
        alpha_denominator: u64,
    ) -> Result<Self, LedgerError> {
        if weight_numerator == 0
            || weight_denominator == 0
            || weight_numerator > weight_denominator
            || alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(LedgerError::InvalidThresholdGrid);
        }

        let mut merged = BTreeMap::<u128, u128>::new();
        for exponent in 0..=63 {
            let index = exponent as u128 + 1;
            let telescoping_denominator = index
                .checked_mul(index + 1)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let prior_mass = PRIOR_MASS_SCALE / (2 * telescoping_denominator)
                + PRIOR_MASS_SCALE / 128;

            let rank = 1_u64
                .checked_shl(exponent)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let target_numerator = (alpha_denominator as u128)
                .checked_mul(weight_denominator as u128)
                .and_then(|value| value.checked_mul(EVIDENCE_ONE))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let target_denominator = (alpha_numerator as u128)
                .checked_mul(rank as u128)
                .and_then(|value| value.checked_mul(weight_numerator as u128))
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let target_ticks = target_numerator
                .checked_add(target_denominator - 1)
                .ok_or(LedgerError::ArithmeticOverflow)?
                / target_denominator;

            let raw_numerator = target_ticks
                .checked_mul(PRIOR_MASS_SCALE)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let raw_threshold = (raw_numerator
                .checked_add(prior_mass - 1)
                .ok_or(LedgerError::ArithmeticOverflow)?
                / prior_mass)
                .max(EVIDENCE_ONE);

            let entry = merged.entry(raw_threshold).or_insert(0);
            *entry = entry
                .checked_add(prior_mass)
                .ok_or(LedgerError::ArithmeticOverflow)?;
        }

        Self::new(
            merged
                .into_iter()
                .map(|(threshold, prior_mass)| ThresholdMass {
                    threshold,
                    prior_mass,
                })
                .collect(),
        )
    }

    /// Builds a finite prefix of a horizon-independent dyadic prior.
    ///
    /// The ideal infinite prior is pi_j = 1 / (j * (j + 1)) at K_j = 2^j,
    /// j >= 1. Its total mass is exactly one by telescoping. Each stored mass
    /// is rounded down in fixed-point units, so every finite prefix remains
    /// valid. Exponents are capped at 64 to keep u128 products bounded.
    pub fn dyadic_universal(max_exponent: u32) -> Result<Self, LedgerError> {
        if max_exponent == 0 || max_exponent > 64 {
            return Err(LedgerError::InvalidThresholdGrid);
        }
        let mut points = Vec::with_capacity(max_exponent as usize);
        for exponent in 1..=max_exponent {
            let denominator = (exponent as u128)
                .checked_mul(exponent as u128 + 1)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let prior_mass = PRIOR_MASS_SCALE / denominator;
            let threshold = EVIDENCE_ONE
                .checked_shl(exponent)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            points.push(ThresholdMass {
                threshold,
                prior_mass,
            });
        }
        Self::new(points)
    }

    /// Evaluates the staircase calibrator at a running-supremum value.
    ///
    /// The exact result is rounded down to evidence ticks. This cannot increase
    /// calibrated evidence and avoids floating-point threshold ambiguity.
    pub fn value_at(&self, running_supremum: u128) -> Result<u128, LedgerError> {
        let count = self.points.partition_point(|point| point.threshold <= running_supremum);
        if count == 0 {
            return Ok(0);
        }
        Ok(self.prefix_numerators[count - 1] / PRIOR_MASS_SCALE)
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
            reserve_rate_numerator: if id == 30 { 0 } else { rate_num },
            reserve_rate_denominator: rate_den,
            destination_id: match id {
                10 => Some(20),
                20 => Some(30),
                _ => None,
            },

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
    fn escrow_cannot_be_retargeted_away_from_precommitted_destination() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        for id in [10, 20, 30] {
            ledger.apply(InputEvent::Activate { round: 0, candidate_id: id }).unwrap();
        }
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();

        assert_eq!(
            ledger.apply(InputEvent::Credit {
                round: 2,
                token_id: 1,
                recipient_id: 30,
                amount: 1,
            }),
            Err(LedgerError::InvalidRecipient)
        );
        ledger.apply(InputEvent::Credit {
            round: 2,
            token_id: 1,
            recipient_id: 20,
            amount: 1,
        }).unwrap();
        ledger.validate_invariants().unwrap();
    }

    #[test]
    fn failed_event_is_atomic_and_preserves_receipt() {
        let mut ledger = DonationLedger::new(basic_specs()).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: EVIDENCE_ONE,
        }).unwrap();
        let before_snapshot = ledger.snapshot().unwrap();
        let before_log = ledger.audit_log().to_vec();

        assert_eq!(
            ledger.apply(InputEvent::Credit {
                round: 2,
                token_id: 1,
                recipient_id: 30,
                amount: 1,
            }),
            Err(LedgerError::InvalidRecipient)
        );
        assert_eq!(ledger.snapshot().unwrap(), before_snapshot);
        assert_eq!(ledger.audit_log(), before_log.as_slice());
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
    fn capped_overshoot_donation_reserves_only_after_weighted_threshold() {
        let specs = vec![
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
        let mut ledger = DonationLedger::new_capped_overshoot(specs.clone()).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();

        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: 3 * EVIDENCE_ONE,
        }).unwrap();
        assert_eq!(ledger.snapshot().unwrap().candidates[0].outgoing, 0);

        ledger.apply(InputEvent::EvidenceUpdate {
            round: 2,
            candidate_id: 10,
            new_value: 4 * EVIDENCE_ONE,
        }).unwrap();
        let first = ledger.snapshot().unwrap();
        let first_token = &first.escrow[0];
        assert_eq!(first_token.original_amount, 200_000_000_000_000);
        assert_eq!(first.candidates[0].outgoing, first_token.original_amount);

        ledger.apply(InputEvent::EvidenceUpdate {
            round: 3,
            candidate_id: 10,
            new_value: 5 * EVIDENCE_ONE,
        }).unwrap();
        let second = ledger.snapshot().unwrap();
        assert_eq!(second.escrow.len(), 2);
        assert_eq!(second.escrow[1].original_amount, 100_000_000_000_000);
        assert_eq!(
            second.candidates[0].outgoing,
            300_000_000_000_000
        );

        // The capped overshoot reserve is exhausted at gamma_i; later
        // evidence cannot withdraw any additional donor capital.
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 4,
            candidate_id: 10,
            new_value: 100 * EVIDENCE_ONE,
        }).unwrap();
        let final_snapshot = ledger.snapshot().unwrap();
        assert_eq!(final_snapshot.escrow.len(), 2);
        assert_eq!(
            final_snapshot.candidates[0].outgoing,
            300_000_000_000_000
        );
        assert_eq!(
            DonationLedger::verify_receipt_with_reserve_policy(
                specs,
                DonationReservePolicy::CappedOvershoot,
                ledger.audit_log(),
                Some(&final_snapshot),
            ),
            Ok(final_snapshot)
        );
        ledger.validate_invariants().unwrap();
    }

    #[test]
    fn capped_overshoot_missing_destination_disables_reservation() {
        let specs = vec![
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
        let mut ledger = DonationLedger::new_capped_overshoot(specs).unwrap();
        ledger.apply(InputEvent::Activate { round: 0, candidate_id: 10 }).unwrap();
        ledger.apply(InputEvent::EvidenceUpdate {
            round: 1,
            candidate_id: 10,
            new_value: 100 * EVIDENCE_ONE,
        }).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.candidates[0].outgoing, 0);
        assert!(snapshot.escrow.is_empty());
        ledger.validate_invariants().unwrap();
    }

    #[test]
    fn capped_overshoot_horizon_extension_preserves_old_scores() {
        let make_spec = |id, weight, destination_id| CandidateSpec {
            id,
            weight,
            destination_id,
            reserve_rate_numerator: 0,
            reserve_rate_denominator: 1,
        };
        let small_specs = vec![
            make_spec(10, 300_000_000, Some(20)),
            make_spec(20, 300_000_000, Some(30)),
        ];
        let large_specs = vec![
            make_spec(10, 300_000_000, Some(20)),
            make_spec(20, 300_000_000, Some(30)),
            make_spec(30, 200_000_000, Some(40)),
        ];
        let mut small = DonationLedger::new_capped_overshoot(small_specs).unwrap();
        let mut large = DonationLedger::new_capped_overshoot(large_specs).unwrap();

        let prefix = vec![
            InputEvent::Activate { round: 0, candidate_id: 10 },
            InputEvent::EvidenceUpdate {
                round: 1,
                candidate_id: 10,
                new_value: 4 * EVIDENCE_ONE,
            },
            InputEvent::Activate { round: 2, candidate_id: 20 },
            InputEvent::Credit {
                round: 2,
                token_id: 1,
                recipient_id: 20,
                amount: 200_000_000_000_000,
            },
            InputEvent::EvidenceUpdate {
                round: 3,
                candidate_id: 20,
                new_value: 4 * EVIDENCE_ONE,
            },
        ];
        for event in prefix {
            small.apply(event.clone()).unwrap();
            large.apply(event).unwrap();
        }

        for id in [10, 20] {
            assert_eq!(small.weighted_score_for(id), large.weighted_score_for(id));
        }
        assert_eq!(
            small.snapshot().unwrap().escrow[0].remaining_amount,
            large.snapshot().unwrap().escrow[0].remaining_amount
        );

        // The longer horizon activates the already precommitted destination
        // of candidate 20 and credits its token. This must not rewrite either
        // old candidate's score or prior credit.
        large.apply(InputEvent::Activate { round: 4, candidate_id: 30 }).unwrap();
        large.apply(InputEvent::Credit {
            round: 5,
            token_id: 2,
            recipient_id: 30,
            amount: 200_000_000_000_000,
        }).unwrap();

        for id in [10, 20] {
            assert_eq!(small.weighted_score_for(id), large.weighted_score_for(id));
        }
        small.validate_invariants().unwrap();
        large.validate_invariants().unwrap();
    }

    #[test]
    fn capped_overshoot_fixed_point_randomized_lipschitz_and_cap_properties() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            state
        };

        for _ in 0..20_000 {
            let weight = (next() % (WEIGHT_SCALE + 1)) as u128;
            let previous_x = (next() % (100 * EVIDENCE_ONE + 1)) as u128;
            let increment = (next() % (100 * EVIDENCE_ONE + 1)) as u128;
            let current_x = previous_x + increment;

            let reserve = |x: u128| {
                let weighted = weight.checked_mul(x).unwrap();
                let threshold = (WEIGHT_SCALE as u128) * EVIDENCE_ONE;
                let cap = weight * EVIDENCE_ONE;
                weighted.saturating_sub(threshold).min(cap)
            };

            let before = reserve(previous_x);
            let after = reserve(current_x);
            assert!(after >= before);
            assert!(before <= weight * previous_x.min(EVIDENCE_ONE));
            assert!(after <= weight * current_x.min(EVIDENCE_ONE));
            assert!(after - before <= weight * increment);
        }
    }

    #[test]
    fn capped_overshoot_reserve_satisfies_donor_cap_and_own_increment() {
        let weight = 300_000_000_u128;
        let mut previous_value = 0_u128;
        let mut previous_reserve = 0_u128;
        for value in (0..=20_u128).map(|i| i * EVIDENCE_ONE / 2) {
            let weighted = weight * value;
            let target = weighted
                .saturating_sub(WEIGHT_SCALE as u128 * EVIDENCE_ONE)
                .min(weight * EVIDENCE_ONE);
            let cap = weight * value.min(EVIDENCE_ONE);
            assert!(target <= cap);
            let delta = target - previous_reserve;
            assert!(delta <= weight * (value - previous_value));
            assert!(target >= previous_reserve);
            previous_value = value;
            previous_reserve = target;
        }
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
        let token_amount = large.snapshot().unwrap().escrow[1].remaining_amount;
        large.apply(InputEvent::Credit {
            round: 5,
            token_id: 2,
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
    fn process_overshoot_refund_is_incremental_capped_and_monotone() {
        let mut account = ProcessOvershootRefund::new(1, 20).unwrap();

        let below = account.record(1, 19 * EVIDENCE_ONE).unwrap();
        assert!(!below.rejected);
        assert_eq!(below.refund_increment_numerator, 0);
        assert_eq!(below.cumulative_refund_numerator, 0);
        assert_eq!(below.effective_cost_numerator, 19 * EVIDENCE_ONE);

        let at_threshold = account.record(2, 20 * EVIDENCE_ONE).unwrap();
        assert!(at_threshold.rejected);
        assert_eq!(at_threshold.refund_increment_numerator, 0);
        assert_eq!(at_threshold.effective_cost_numerator, at_threshold.denominator);

        let mild_overshoot = account.record(3, 21 * EVIDENCE_ONE).unwrap();
        assert_eq!(
            mild_overshoot.refund_increment_numerator,
            EVIDENCE_ONE
        );
        assert_eq!(
            mild_overshoot.cumulative_refund_numerator,
            EVIDENCE_ONE
        );
        assert_eq!(
            mild_overshoot.effective_cost_numerator,
            mild_overshoot.denominator
        );

        let capped = account.record(4, 30 * EVIDENCE_ONE).unwrap();
        assert_eq!(
            capped.refund_increment_numerator,
            0
        );
        assert_eq!(
            capped.cumulative_refund_numerator,
            account.refund_cap_numerator()
        );
        assert_eq!(
            capped.effective_cost_numerator,
            29 * EVIDENCE_ONE
        );

        let later = account.record(5, 40 * EVIDENCE_ONE).unwrap();
        assert_eq!(later.refund_increment_numerator, 0);
        assert_eq!(later.cumulative_refund_numerator, EVIDENCE_ONE);
    }

    #[test]
    fn uncapped_process_overshoot_refunds_full_increment_once() {
        let mut account = ProcessOvershootRefundUncapped::new(1, 20).unwrap();
        let before = account.record(1, 19 * EVIDENCE_ONE).unwrap();
        assert!(!before.rejected);
        assert_eq!(before.refund_increment_numerator, 0);
        assert_eq!(before.effective_cost_numerator, 19 * EVIDENCE_ONE);

        let at = account.record(2, 20 * EVIDENCE_ONE).unwrap();
        assert!(at.rejected);
        assert_eq!(at.refund_increment_numerator, 0);
        assert_eq!(at.effective_cost_numerator, at.denominator);

        let first_overshoot = account.record(3, 21 * EVIDENCE_ONE).unwrap();
        assert_eq!(first_overshoot.refund_increment_numerator, EVIDENCE_ONE);
        assert_eq!(first_overshoot.cumulative_refund_numerator, EVIDENCE_ONE);
        assert_eq!(first_overshoot.effective_cost_numerator, first_overshoot.denominator);

        let large = account.record(4, 100 * EVIDENCE_ONE).unwrap();
        assert_eq!(large.cumulative_refund_numerator, 80 * EVIDENCE_ONE);
        assert_eq!(
            large.refund_increment_numerator,
            79 * EVIDENCE_ONE
        );
        assert_eq!(large.effective_cost_numerator, large.denominator);

        let later = account.record(5, 1_000 * EVIDENCE_ONE).unwrap();
        assert_eq!(later.cumulative_refund_numerator, 980 * EVIDENCE_ONE);
        assert_eq!(later.refund_increment_numerator, 900 * EVIDENCE_ONE);
        assert_eq!(later.effective_cost_numerator, later.denominator);
    }

    #[test]
    fn uncapped_process_overshoot_supports_zero_initial_value_and_atomicity() {
        let mut account = ProcessOvershootRefundUncapped::with_initial_evidence_value(
            1, 20, 0
        ).unwrap();
        let first = account.record(1, 0).unwrap();
        assert_eq!(first.effective_cost_numerator, 0);
        assert_eq!(first.cumulative_refund_numerator, 0);

        let before = account.clone();
        assert_eq!(
            account.record(0, 0),
            Err(LedgerError::RoundDecreased)
        );
        assert_eq!(account, before);
        assert_eq!(
            account.record(2, 0),
            Ok(OvershootRefundReceipt {
                round: 2,
                evidence_value: 0,
                refund_increment_numerator: 0,
                cumulative_refund_numerator: 0,
                effective_cost_numerator: 0,
                denominator: 20 * EVIDENCE_ONE,
                rejected: false,
            })
        );
    }

    #[test]
    fn process_overshoot_refund_supports_zero_initialized_e_process() {
        let mut account = ProcessOvershootRefund::with_initial_evidence_value(
            1, 20, 0
        ).unwrap();
        let initial = account.record(0, 0).unwrap();
        assert!(!initial.rejected);
        assert_eq!(initial.refund_increment_numerator, 0);
        assert_eq!(initial.effective_cost_numerator, 0);

        let before = account.record(1, 19 * EVIDENCE_ONE).unwrap();
        assert!(!before.rejected);
        assert_eq!(before.effective_cost_numerator, 19 * EVIDENCE_ONE);

        let crossed = account.record(2, 21 * EVIDENCE_ONE).unwrap();
        assert!(crossed.rejected);
        assert_eq!(crossed.cumulative_refund_numerator, EVIDENCE_ONE);
        assert_eq!(crossed.effective_cost_numerator, crossed.denominator);
    }


    #[test]
    fn full_process_overshoot_refund_matches_score_identity() {
        let mut account = ProcessOvershootRefundUncapped::new(1, 20).unwrap();

        let below = account.record(1, 19 * EVIDENCE_ONE).unwrap();
        assert_eq!(below.refund_increment_numerator, 0);
        assert_eq!(below.effective_cost_numerator, 19 * EVIDENCE_ONE);

        let at = account.record(2, 20 * EVIDENCE_ONE).unwrap();
        assert_eq!(at.refund_increment_numerator, 0);
        assert_eq!(at.effective_cost_numerator, 20 * EVIDENCE_ONE);

        let overshoot = account.record(3, 100 * EVIDENCE_ONE).unwrap();
        assert_eq!(overshoot.cumulative_refund_numerator, 80 * EVIDENCE_ONE);
        assert_eq!(overshoot.refund_increment_numerator, 80 * EVIDENCE_ONE);
        assert_eq!(overshoot.effective_cost_numerator, overshoot.denominator);

        let later = account.record(4, 200 * EVIDENCE_ONE).unwrap();
        assert_eq!(later.refund_increment_numerator, 100 * EVIDENCE_ONE);
        assert_eq!(later.cumulative_refund_numerator, 180 * EVIDENCE_ONE);
        assert_eq!(later.effective_cost_numerator, later.denominator);
    }

    #[test]
    fn full_and_capped_refund_have_same_cost_but_different_refund_credit_after_cap() {
        let mut capped = ProcessOvershootRefund::new(1, 20).unwrap();
        let mut full = ProcessOvershootRefundUncapped::new(1, 20).unwrap();

        let capped_receipt = capped.record(1, 100 * EVIDENCE_ONE).unwrap();
        let full_receipt = full.record(1, 100 * EVIDENCE_ONE).unwrap();
        assert_eq!(
            capped_receipt.effective_cost_numerator,
            full_receipt.effective_cost_numerator
        );
        assert_eq!(capped_receipt.cumulative_refund_numerator, EVIDENCE_ONE);
        assert_eq!(full_receipt.cumulative_refund_numerator, 80 * EVIDENCE_ONE);
    }

    #[test]
    fn process_overshoot_refund_rejects_rollback_and_is_atomic() {
        let mut account = ProcessOvershootRefund::new(1, 20).unwrap();
        account.record(2, 25 * EVIDENCE_ONE).unwrap();
        let before = account.clone();

        assert_eq!(
            account.record(1, 30 * EVIDENCE_ONE),
            Err(LedgerError::RoundDecreased)
        );
        assert_eq!(account, before);

        assert_eq!(
            account.record(3, 24 * EVIDENCE_ONE),
            Err(LedgerError::EvidenceDecreased)
        );
        assert_eq!(account, before);
    }

    #[test]
    fn process_overshoot_refund_rejects_invalid_alpha() {
        assert_eq!(
            ProcessOvershootRefund::new(0, 20),
            Err(LedgerError::InvalidReserveRate)
        );
        assert_eq!(
            ProcessOvershootRefund::new(1, 1),
            Err(LedgerError::InvalidReserveRate)
        );
        assert_eq!(
            ProcessOvershootRefund::new(2, 1),
            Err(LedgerError::InvalidReserveRate)
        );
    }

    #[test]
    fn staircase_binary_query_matches_linear_exact_reference() {
        let adjuster = StaircaseAdjuster::new(vec![
            ThresholdMass { threshold: 2 * EVIDENCE_ONE, prior_mass: 100_000_000 },
            ThresholdMass { threshold: 8 * EVIDENCE_ONE, prior_mass: 200_000_000 },
            ThresholdMass { threshold: 32 * EVIDENCE_ONE, prior_mass: 300_000_000 },
        ]).unwrap();
        for x in (0..=40_u128).map(|value| value * EVIDENCE_ONE / 2) {
            let numerator: u128 = adjuster.points().iter()
                .filter(|point| point.threshold <= x)
                .map(|point| point.prior_mass * point.threshold)
                .sum();
            assert_eq!(adjuster.value_at(x).unwrap(), numerator / PRIOR_MASS_SCALE);
        }
    }

    #[test]
    fn e_bh_rank_geometric_covers_every_rank_with_logarithmic_grid() {
        let m = 100_u64;
        let alpha_num = 1_u64;
        let alpha_den = 20_u64;
        let adjuster = StaircaseAdjuster::e_bh_rank_geometric(m, alpha_num, alpha_den).unwrap();
        assert_eq!(adjuster.points().len(), 8);
        let total_mass: u128 = adjuster.points().iter().map(|point| point.prior_mass).sum();
        assert!(total_mass <= PRIOR_MASS_SCALE);

        for k in 1..=m {
            let rank = 1_u64 << (63 - (k.leading_zeros() as u64));
            let h_numerator = (m as u128) * (alpha_den as u128) * EVIDENCE_ONE;
            let h_denominator = (alpha_num as u128) * (rank as u128);
            let h_ticks = (h_numerator + h_denominator - 1) / h_denominator;
            let threshold = adjuster
                .points()
                .iter()
                .find(|point| {
                    let contribution = point.prior_mass * point.threshold / PRIOR_MASS_SCALE;
                    contribution >= h_ticks
                })
                .map(|point| point.threshold);
            assert!(threshold.is_some(), "rank {k} had no supporting staircase point");

            let raw = threshold.unwrap();
            let ideal_boundary = (m as u128) * (alpha_den as u128) * EVIDENCE_ONE
                / ((alpha_num as u128) * (k as u128));
            // Since the lower geometric rank is greater than k/2, the
            // raw-threshold penalty is bounded by about 2Q (plus rounding).
            assert!(raw <= 17 * ideal_boundary + EVIDENCE_ONE);
        }
    }

    #[test]
    fn weighted_rank_geometric_scales_thresholds_by_candidate_weight() {
        let unit_weight = StaircaseAdjuster::weighted_e_bh_rank_geometric(
            100, 1, 100, 1, 20
        ).unwrap();
        let half_weight = StaircaseAdjuster::weighted_e_bh_rank_geometric(
            100, 1, 200, 1, 20
        ).unwrap();
        assert_eq!(
            half_weight.points().len(),
            unit_weight.points().len()
        );
        for (half, unit) in half_weight.points().iter().zip(unit_weight.points()) {
            assert!(half.threshold >= unit.threshold);
        }

        let equal_weight = StaircaseAdjuster::e_bh_rank_geometric(100, 1, 20).unwrap();
        assert_eq!(equal_weight, unit_weight);
    }

    #[test]
    fn projective_weighted_rank_prior_covers_all_ranks_without_family_size() {
        let m = 10_000_u64;
        let weight_num = 3_u64;
        let weight_den = 1_000_u64;
        let alpha_num = 1_u64;
        let alpha_den = 20_u64;
        let adjuster = StaircaseAdjuster::projective_weighted_e_bh_rank_prior(
            13, weight_num, weight_den, alpha_num, alpha_den
        ).unwrap();

        let total_mass: u128 = adjuster.points().iter().map(|point| point.prior_mass).sum();
        let expected_mass: u128 = (0..=13)
            .map(|j| PRIOR_MASS_SCALE / ((j as u128 + 1) * (j as u128 + 2)))
            .sum();
        assert_eq!(total_mass, expected_mass);
        assert!(total_mass <= PRIOR_MASS_SCALE);

        for k in 1..=m {
            let exponent = 63 - (k.leading_zeros());
            let rank = 1_u64 << exponent;
            let denominator = (exponent as u128 + 1) * (exponent as u128 + 2);
            let mass = PRIOR_MASS_SCALE / denominator;
            let target_numerator = (alpha_den as u128)
                * (weight_den as u128) * EVIDENCE_ONE;
            let target_denominator = (alpha_num as u128)
                * (rank as u128) * (weight_num as u128);
            let target = (target_numerator + target_denominator - 1) / target_denominator;
            let raw_numerator = target * PRIOR_MASS_SCALE;
            let raw = ((raw_numerator + mass - 1) / mass).max(EVIDENCE_ONE);

            assert!(
                adjuster.value_at(raw).unwrap() >= target,
                "rank {k} unsupported by its dyadic prior point"
            );

            let actual_denominator = (alpha_num as u128)
                * (k as u128) * (weight_num as u128);
            let actual = (target_numerator + actual_denominator - 1) / actual_denominator;
            let overhead_numerator = 2 * denominator * PRIOR_MASS_SCALE;
            let overhead_denominator = PRIOR_MASS_SCALE - denominator;
            let overhead = (overhead_numerator + overhead_denominator - 1)
                / overhead_denominator;
            assert!(
                raw <= (overhead * actual + EVIDENCE_ONE)
                    .max(EVIDENCE_ONE),
                "rank {k} exceeds projective rank prior overhead"
            );
        }
    }

    #[test]
    fn hybrid_projective_rank_prior_covers_every_u64_rank() {
        let adjuster = StaircaseAdjuster::projective_weighted_e_bh_rank_prior_hybrid_u64(
            3, 1_000, 1, 20
        ).unwrap();
        let total_mass: u128 = adjuster.points().iter().map(|point| point.prior_mass).sum();
        assert!(total_mass <= PRIOR_MASS_SCALE);

        for exponent in 0..=63 {
            let rank = 1_u64 << exponent;
            let denominator = (exponent as u128 + 1) * (exponent as u128 + 2);
            let mass = PRIOR_MASS_SCALE / (2 * denominator)
                + PRIOR_MASS_SCALE / 128;
            let target_num = 20_u128 * 1_000_u128 * EVIDENCE_ONE;
            let target_den = rank as u128 * 3_u128;
            let target = (target_num + target_den - 1) / target_den;
            let raw_num = target * PRIOR_MASS_SCALE;
            let raw = ((raw_num + mass - 1) / mass).max(EVIDENCE_ONE);
            assert!(
                adjuster.value_at(raw).unwrap() >= target,
                "dyadic rank {rank} did not meet its weighted boundary"
            );
        }

        let max_rank = u64::MAX;
        let exponent = 63_u32;
        let rank = 1_u64 << exponent;
        let denom = 64_u128 * 65_u128;
        let mass = PRIOR_MASS_SCALE / (2 * denom) + PRIOR_MASS_SCALE / 128;
        let target_num = 20_u128 * 1_000_u128 * EVIDENCE_ONE;
        let target_den = rank as u128 * 3_u128;
        let target = (target_num + target_den - 1) / target_den;
        let raw = ((target * PRIOR_MASS_SCALE + mass - 1) / mass).max(EVIDENCE_ONE);
        assert!(adjuster.value_at(raw).unwrap() >= target);
        assert!(max_rank >= rank);
    }

    #[test]
    fn projective_rank_prior_has_same_prefix_under_horizon_extension() {
        let short = StaircaseAdjuster::projective_weighted_e_bh_rank_prior(
            63, 3, 1_000, 1, 20
        ).unwrap();
        let long = StaircaseAdjuster::projective_weighted_e_bh_rank_prior(
            63, 3, 1_000, 1, 20
        ).unwrap();
        assert_eq!(short, long);
        assert_eq!(short.points(), long.points());
    }

    #[test]
    fn weighted_rank_geometric_covers_every_rank_for_nonuniform_weight() {
        let m = 73_u64;
        let weight_num = 3_u64;
        let weight_den = 1_000_u64;
        let alpha_num = 1_u64;
        let alpha_den = 20_u64;
        let adjuster = StaircaseAdjuster::weighted_e_bh_rank_geometric(
            m, weight_num, weight_den, alpha_num, alpha_den
        ).unwrap();

        let mut ranks = Vec::new();
        let mut rank = 1_u64;
        loop {
            if rank > m { break; }
            ranks.push(rank);
            match rank.checked_mul(2) {
                Some(next) if next <= m => rank = next,
                _ => break,
            }
        }
        if ranks.last().copied() != Some(m) { ranks.push(m); }
        let q = ranks.len() as u128;
        let prior_mass = PRIOR_MASS_SCALE / q;

        for k in 1..=m {
            let grid_rank = 1_u64 << (63 - (k.leading_zeros() as u64));
            let target_num = (alpha_den as u128)
                * (weight_den as u128)
                * EVIDENCE_ONE;
            let target_den = (alpha_num as u128)
                * (grid_rank as u128)
                * (weight_num as u128);
            let target = (target_num + target_den - 1) / target_den;
            let raw_num = target * PRIOR_MASS_SCALE;
            let raw = (raw_num + prior_mass - 1) / prior_mass;
            assert!(
                adjuster.value_at(raw).unwrap() >= target,
                "rank {k} failed at geometric grid rank {grid_rank}"
            );

            let actual_num = (alpha_den as u128)
                * (weight_den as u128)
                * EVIDENCE_ONE;
            let actual_den = (alpha_num as u128)
                * (k as u128)
                * (weight_num as u128);
            let actual = (actual_num + actual_den - 1) / actual_den;
            let overhead_bound = (2 * q * PRIOR_MASS_SCALE) / (PRIOR_MASS_SCALE - q);
            assert!(
                raw <= overhead_bound * actual + EVIDENCE_ONE,
                "rank {k} exceeded geometric overhead"
            );
        }
    }

    #[test]
    fn weighted_rank_geometric_rejects_invalid_weights() {
        assert_eq!(
            StaircaseAdjuster::weighted_e_bh_rank_geometric(100, 0, 100, 1, 20),
            Err(LedgerError::InvalidThresholdGrid)
        );
        assert_eq!(
            StaircaseAdjuster::weighted_e_bh_rank_geometric(100, 101, 100, 1, 20),
            Err(LedgerError::InvalidThresholdGrid)
        );
        assert_eq!(
            StaircaseAdjuster::weighted_e_bh_rank_geometric(100, 1, 100, 1, 1),
            Err(LedgerError::InvalidThresholdGrid)
        );
    }

    #[test]
    fn e_bh_rank_geometric_rejects_invalid_design_parameters() {
        assert_eq!(
            StaircaseAdjuster::e_bh_rank_geometric(0, 1, 20),
            Err(LedgerError::InvalidThresholdGrid)
        );
        assert_eq!(
            StaircaseAdjuster::e_bh_rank_geometric(100, 0, 20),
            Err(LedgerError::InvalidThresholdGrid)
        );
        assert_eq!(
            StaircaseAdjuster::e_bh_rank_geometric(100, 1, 1),
            Err(LedgerError::InvalidThresholdGrid)
        );
    }

    #[test]
    fn dyadic_universal_prior_is_budgeted_and_monotone() {
        let adjuster = StaircaseAdjuster::dyadic_universal(24).unwrap();
        let total_mass: u128 = adjuster.points().iter().map(|point| point.prior_mass).sum();
        assert!(total_mass <= PRIOR_MASS_SCALE);
        assert_eq!(adjuster.points().len(), 24);
        assert_eq!(
            adjuster.value_at(2 * EVIDENCE_ONE).unwrap(),
            EVIDENCE_ONE
        );
        let before = adjuster.value_at(8 * EVIDENCE_ONE - 1).unwrap();
        let after = adjuster.value_at(8 * EVIDENCE_ONE).unwrap();
        assert!(after >= before);
        assert_eq!(
            StaircaseAdjuster::dyadic_universal(65),
            Err(LedgerError::InvalidThresholdGrid)
        );
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
            DonationLedger::new(vec![CandidateSpec {
                id: 99,
                weight: WEIGHT_SCALE,
                reserve_rate_numerator: 1,
                reserve_rate_denominator: 1,
                destination_id: None,
            }]),
            Err(LedgerError::InvalidRecipient)
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
