// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic-only prospective reveal protocol for EUREKA-002 V2 HeldOut.
//!
//! This module qualifies the evaluator state machine before any real HeldOut
//! target/comparator execution exists. It owns the canonical HeldOut plan, but
//! tickets expose only pre-outcome information. Post-state can be obtained only
//! by consuming a paired, role-bound, frozen prediction object.

#![allow(dead_code)]

use super::consequence::{
    ConsequencePrediction, ConsequenceScore, ConsequenceScoringError, PredictionOutcome,
    score_consequence,
};
use super::hidden_world::{PublicAction, PublicObservation, PublicValue};
use super::v2_corpus_schedule::{V2ScheduleMaterializationError, V2ScheduledRow};
use super::v2_heldout_plan::{V2HeldOutPlan, materialize_heldout_plan};
use super::v2_public_schema::{V2PublicFamily, V2PublicState};

pub(super) const V2_HELDOUT_REVEAL_PROTOCOL_REVISION: &str =
    "EUREKA.002.V2.HELDOUT_REVEAL_PROTOCOL.v1";
pub(super) const V2_HELDOUT_TICKET_REVISION: &str =
    "EUREKA.002.V2.HELDOUT_TICKET.v1";
pub(super) const V2_HELDOUT_PREDICTION_FREEZE_REVISION: &str =
    "EUREKA.002.V2.HELDOUT_PREDICTION_FREEZE.v1";
pub(super) const V2_HELDOUT_PAIRED_FREEZE_REVISION: &str =
    "EUREKA.002.V2.HELDOUT_PAIRED_FREEZE.v1";
pub(super) const V2_HELDOUT_REVEAL_RECEIPT_REVISION: &str =
    "EUREKA.002.V2.HELDOUT_REVEAL_RECEIPT.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2HeldOutRevealError {
    Materialization(V2ScheduleMaterializationError),
    ZeroCampaignManifestCommitment,
    ZeroTargetSubjectCommitment,
    ZeroComparatorSubjectCommitment,
    RowIndexOutOfRange,
    TicketManifestMismatch,
    TicketRowMismatch,
    TicketCommitmentMismatch,
    TargetActionMismatch,
    ComparatorActionMismatch,
    TargetTicketMismatch,
    ComparatorTicketMismatch,
    TargetSubjectMismatch,
    ComparatorSubjectMismatch,
    PairedCommitmentMismatch,
    Scoring(ConsequenceScoringError),
}

impl From<V2ScheduleMaterializationError> for V2HeldOutRevealError {
    fn from(value: V2ScheduleMaterializationError) -> Self {
        Self::Materialization(value)
    }
}

impl From<ConsequenceScoringError> for V2HeldOutRevealError {
    fn from(value: ConsequenceScoringError) -> Self {
        Self::Scoring(value)
    }
}

/// Public pre-outcome capability for one exact canonical HeldOut row.
///
/// Deliberately contains no post-state and no method that returns one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2HeldOutTicket {
    campaign_manifest_commitment: [u8; 32],
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
    commitment: [u8; 32],
}

impl V2HeldOutTicket {
    pub(super) const fn campaign_manifest_commitment(self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn row_index(self) -> u16 {
        self.row_index
    }

    pub(super) const fn row_identity(self) -> [u8; 32] {
        self.row_identity
    }

    pub(super) const fn family(self) -> V2PublicFamily {
        self.family
    }

    pub(super) const fn pre(self) -> V2PublicState {
        self.pre
    }

    pub(super) const fn action(self) -> PublicAction {
        self.action
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

/// Move-only target prediction frozen against one ticket and one target identity.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2FrozenTargetPrediction {
    ticket_commitment: [u8; 32],
    target_subject_commitment: [u8; 32],
    prediction: ConsequencePrediction,
    commitment: [u8; 32],
}

/// Move-only comparator prediction frozen against one ticket and one comparator identity.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2FrozenComparatorPrediction {
    ticket_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    prediction: ConsequencePrediction,
    commitment: [u8; 32],
}

/// Move-only proof that both role-specific predictions were frozen for the same
/// exact pre-outcome ticket before evaluator reveal.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2PairedFrozenPredictions {
    ticket: V2HeldOutTicket,
    target: V2FrozenTargetPrediction,
    comparator: V2FrozenComparatorPrediction,
    commitment: [u8; 32],
}

impl V2PairedFrozenPredictions {
    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Move-only reveal receipt. It exists only after a paired frozen object has
/// been consumed, so evaluator post-state cannot precede paired commitment.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2HeldOutRevealReceipt {
    ticket: V2HeldOutTicket,
    target_prediction: ConsequencePrediction,
    comparator_prediction: ConsequencePrediction,
    paired_prediction_commitment: [u8; 32],
    post: V2PublicState,
    commitment: [u8; 32],
}

impl V2HeldOutRevealReceipt {
    pub(super) const fn post(&self) -> V2PublicState {
        self.post
    }

    pub(super) const fn paired_prediction_commitment(&self) -> [u8; 32] {
        self.paired_prediction_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Consume the one reveal receipt and score both subjects on the exact same
    /// pre/action/post tuple. No aggregate scientific disposition is minted.
    pub(super) fn score(self) -> Result<V2HeldOutPairScore, V2HeldOutRevealError> {
        let pre = observation(self.ticket.pre, 0);
        let post = observation(self.post, 1);
        let target_score = score_consequence(&pre, &self.target_prediction, &post)?;
        let comparator_score = score_consequence(&pre, &self.comparator_prediction, &post)?;
        Ok(V2HeldOutPairScore {
            campaign_manifest_commitment: self.ticket.campaign_manifest_commitment,
            row_index: self.ticket.row_index,
            row_identity: self.ticket.row_identity,
            action: self.ticket.action,
            paired_prediction_commitment: self.paired_prediction_commitment,
            reveal_receipt_commitment: self.commitment,
            target_score,
            comparator_score,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct V2HeldOutPairScore {
    pub campaign_manifest_commitment: [u8; 32],
    pub row_index: u16,
    pub row_identity: [u8; 32],
    pub action: PublicAction,
    pub paired_prediction_commitment: [u8; 32],
    pub reveal_receipt_commitment: [u8; 32],
    pub target_score: ConsequenceScore,
    pub comparator_score: ConsequenceScore,
}

/// Evaluator-owned synthetic qualification protocol. This type never invokes a
/// real target or comparator. Callers supply synthetic predictions explicitly.
#[derive(Debug)]
pub(super) struct V2HeldOutRevealProtocol {
    plan: V2HeldOutPlan,
    campaign_manifest_commitment: [u8; 32],
    target_subject_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
}

impl V2HeldOutRevealProtocol {
    pub(super) fn synthetic_canonical(
        campaign_manifest_commitment: [u8; 32],
        target_subject_commitment: [u8; 32],
        comparator_subject_commitment: [u8; 32],
    ) -> Result<Self, V2HeldOutRevealError> {
        if campaign_manifest_commitment == [0_u8; 32] {
            return Err(V2HeldOutRevealError::ZeroCampaignManifestCommitment);
        }
        if target_subject_commitment == [0_u8; 32] {
            return Err(V2HeldOutRevealError::ZeroTargetSubjectCommitment);
        }
        if comparator_subject_commitment == [0_u8; 32] {
            return Err(V2HeldOutRevealError::ZeroComparatorSubjectCommitment);
        }
        Ok(Self {
            plan: materialize_heldout_plan()?,
            campaign_manifest_commitment,
            target_subject_commitment,
            comparator_subject_commitment,
        })
    }

    pub(super) fn ticket(&self, row_index: usize) -> Result<V2HeldOutTicket, V2HeldOutRevealError> {
        let row = self
            .plan
            .ordered_rows()
            .get(row_index)
            .copied()
            .ok_or(V2HeldOutRevealError::RowIndexOutOfRange)?;
        let row_index = u16::try_from(row_index).map_err(|_| V2HeldOutRevealError::RowIndexOutOfRange)?;
        Ok(ticket_from_row(
            self.campaign_manifest_commitment,
            row_index,
            row,
        ))
    }

    pub(super) fn freeze_target(
        &self,
        ticket: V2HeldOutTicket,
        prediction: &ConsequencePrediction,
    ) -> Result<V2FrozenTargetPrediction, V2HeldOutRevealError> {
        self.validate_ticket(ticket)?;
        if prediction.action != ticket.action {
            return Err(V2HeldOutRevealError::TargetActionMismatch);
        }
        let prediction = prediction.clone();
        let commitment = role_prediction_commitment(
            b"target",
            ticket.commitment,
            self.target_subject_commitment,
            &prediction,
        );
        Ok(V2FrozenTargetPrediction {
            ticket_commitment: ticket.commitment,
            target_subject_commitment: self.target_subject_commitment,
            prediction,
            commitment,
        })
    }

    pub(super) fn freeze_comparator(
        &self,
        ticket: V2HeldOutTicket,
        prediction: &ConsequencePrediction,
    ) -> Result<V2FrozenComparatorPrediction, V2HeldOutRevealError> {
        self.validate_ticket(ticket)?;
        if prediction.action != ticket.action {
            return Err(V2HeldOutRevealError::ComparatorActionMismatch);
        }
        let prediction = prediction.clone();
        let commitment = role_prediction_commitment(
            b"comparator",
            ticket.commitment,
            self.comparator_subject_commitment,
            &prediction,
        );
        Ok(V2FrozenComparatorPrediction {
            ticket_commitment: ticket.commitment,
            comparator_subject_commitment: self.comparator_subject_commitment,
            prediction,
            commitment,
        })
    }

    pub(super) fn pair(
        &self,
        ticket: V2HeldOutTicket,
        target: V2FrozenTargetPrediction,
        comparator: V2FrozenComparatorPrediction,
    ) -> Result<V2PairedFrozenPredictions, V2HeldOutRevealError> {
        self.validate_ticket(ticket)?;
        if target.ticket_commitment != ticket.commitment {
            return Err(V2HeldOutRevealError::TargetTicketMismatch);
        }
        if comparator.ticket_commitment != ticket.commitment {
            return Err(V2HeldOutRevealError::ComparatorTicketMismatch);
        }
        if target.target_subject_commitment != self.target_subject_commitment {
            return Err(V2HeldOutRevealError::TargetSubjectMismatch);
        }
        if comparator.comparator_subject_commitment != self.comparator_subject_commitment {
            return Err(V2HeldOutRevealError::ComparatorSubjectMismatch);
        }
        if target.prediction.action != ticket.action {
            return Err(V2HeldOutRevealError::TargetActionMismatch);
        }
        if comparator.prediction.action != ticket.action {
            return Err(V2HeldOutRevealError::ComparatorActionMismatch);
        }
        let commitment = paired_prediction_commitment(
            ticket,
            self.target_subject_commitment,
            self.comparator_subject_commitment,
            &target.prediction,
            target.commitment,
            &comparator.prediction,
            comparator.commitment,
        );
        Ok(V2PairedFrozenPredictions {
            ticket,
            target,
            comparator,
            commitment,
        })
    }

    pub(super) fn reveal(
        &self,
        paired: V2PairedFrozenPredictions,
    ) -> Result<V2HeldOutRevealReceipt, V2HeldOutRevealError> {
        self.validate_ticket(paired.ticket)?;
        if paired.target.ticket_commitment != paired.ticket.commitment {
            return Err(V2HeldOutRevealError::TargetTicketMismatch);
        }
        if paired.comparator.ticket_commitment != paired.ticket.commitment {
            return Err(V2HeldOutRevealError::ComparatorTicketMismatch);
        }
        if paired.target.target_subject_commitment != self.target_subject_commitment {
            return Err(V2HeldOutRevealError::TargetSubjectMismatch);
        }
        if paired.comparator.comparator_subject_commitment != self.comparator_subject_commitment {
            return Err(V2HeldOutRevealError::ComparatorSubjectMismatch);
        }
        let expected_pair = paired_prediction_commitment(
            paired.ticket,
            self.target_subject_commitment,
            self.comparator_subject_commitment,
            &paired.target.prediction,
            paired.target.commitment,
            &paired.comparator.prediction,
            paired.comparator.commitment,
        );
        if paired.commitment != expected_pair {
            return Err(V2HeldOutRevealError::PairedCommitmentMismatch);
        }
        let row = self.row_for_ticket(paired.ticket)?;
        let commitment = reveal_receipt_commitment(
            paired.ticket,
            paired.commitment,
            row.post(),
        );
        Ok(V2HeldOutRevealReceipt {
            ticket: paired.ticket,
            target_prediction: paired.target.prediction,
            comparator_prediction: paired.comparator.prediction,
            paired_prediction_commitment: paired.commitment,
            post: row.post(),
            commitment,
        })
    }

    fn validate_ticket(&self, ticket: V2HeldOutTicket) -> Result<(), V2HeldOutRevealError> {
        if ticket.campaign_manifest_commitment != self.campaign_manifest_commitment {
            return Err(V2HeldOutRevealError::TicketManifestMismatch);
        }
        let row = self.row_for_ticket(ticket)?;
        let expected = ticket_from_row(
            self.campaign_manifest_commitment,
            ticket.row_index,
            row,
        );
        if expected.row_identity != ticket.row_identity
            || expected.family != ticket.family
            || expected.pre != ticket.pre
            || expected.action != ticket.action
        {
            return Err(V2HeldOutRevealError::TicketRowMismatch);
        }
        if expected.commitment != ticket.commitment {
            return Err(V2HeldOutRevealError::TicketCommitmentMismatch);
        }
        Ok(())
    }

    fn row_for_ticket(&self, ticket: V2HeldOutTicket) -> Result<V2ScheduledRow, V2HeldOutRevealError> {
        self.plan
            .ordered_rows()
            .get(usize::from(ticket.row_index))
            .copied()
            .ok_or(V2HeldOutRevealError::RowIndexOutOfRange)
    }
}

fn ticket_from_row(
    campaign_manifest_commitment: [u8; 32],
    row_index: u16,
    row: V2ScheduledRow,
) -> V2HeldOutTicket {
    let commitment = ticket_commitment(
        campaign_manifest_commitment,
        row_index,
        row.row_identity(),
        row.family(),
        row.pre(),
        row.action(),
    );
    V2HeldOutTicket {
        campaign_manifest_commitment,
        row_index,
        row_identity: row.row_identity(),
        family: row.family(),
        pre: row.pre(),
        action: row.action(),
        commitment,
    }
}

fn ticket_commitment(
    campaign_manifest_commitment: [u8; 32],
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_HELDOUT_TICKET_REVISION.as_bytes());
    bytes.extend_from_slice(&campaign_manifest_commitment);
    bytes.extend_from_slice(&row_index.to_le_bytes());
    bytes.extend_from_slice(&row_identity);
    encode_family(&mut bytes, family);
    encode_state(&mut bytes, pre);
    encode_action(&mut bytes, action);
    *blake3::hash(&bytes).as_bytes()
}

fn role_prediction_commitment(
    role: &[u8],
    ticket_commitment: [u8; 32],
    subject_commitment: [u8; 32],
    prediction: &ConsequencePrediction,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_HELDOUT_PREDICTION_FREEZE_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, role);
    bytes.extend_from_slice(&ticket_commitment);
    bytes.extend_from_slice(&subject_commitment);
    encode_prediction(&mut bytes, prediction);
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn paired_prediction_commitment(
    ticket: V2HeldOutTicket,
    target_subject_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    target_prediction: &ConsequencePrediction,
    target_prediction_commitment: [u8; 32],
    comparator_prediction: &ConsequencePrediction,
    comparator_prediction_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_HELDOUT_PAIRED_FREEZE_REVISION.as_bytes());
    bytes.extend_from_slice(&ticket.commitment);
    bytes.extend_from_slice(&target_subject_commitment);
    bytes.extend_from_slice(&comparator_subject_commitment);
    bytes.extend_from_slice(&target_prediction_commitment);
    bytes.extend_from_slice(&comparator_prediction_commitment);
    encode_prediction(&mut bytes, target_prediction);
    encode_prediction(&mut bytes, comparator_prediction);
    *blake3::hash(&bytes).as_bytes()
}

fn reveal_receipt_commitment(
    ticket: V2HeldOutTicket,
    paired_prediction_commitment: [u8; 32],
    post: V2PublicState,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_HELDOUT_REVEAL_RECEIPT_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&ticket.commitment);
    bytes.extend_from_slice(&paired_prediction_commitment);
    encode_state(&mut bytes, post);
    *blake3::hash(&bytes).as_bytes()
}

fn observation(state: V2PublicState, step: u64) -> PublicObservation {
    PublicObservation {
        step,
        fields: state
            .fields()
            .into_iter()
            .map(PublicValue::Count)
            .collect(),
    }
}

fn encode_prediction(bytes: &mut Vec<u8>, prediction: &ConsequencePrediction) {
    encode_action(bytes, prediction.action);
    match &prediction.outcome {
        PredictionOutcome::Predicted { fields } => {
            bytes.push(1);
            bytes.extend_from_slice(&(fields.len() as u64).to_le_bytes());
            for value in fields {
                encode_value(bytes, *value);
            }
        }
        PredictionOutcome::AbstainInsufficientEvidence => bytes.push(2),
        PredictionOutcome::OutOfQualifiedDomain => bytes.push(3),
    }
}

fn encode_family(bytes: &mut Vec<u8>, family: V2PublicFamily) {
    bytes.push(match family {
        V2PublicFamily::PublicFlowV2 => 1,
        V2PublicFamily::PublicRelayV2 => 2,
    });
}

fn encode_state(bytes: &mut Vec<u8>, state: V2PublicState) {
    for value in state.fields() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn encode_action(bytes: &mut Vec<u8>, action: PublicAction) {
    match action {
        PublicAction::NoOp => bytes.push(1),
        PublicAction::Pulse { slot } => {
            bytes.push(2);
            bytes.push(slot);
        }
        PublicAction::Transfer { from, to, amount } => {
            bytes.push(3);
            bytes.push(from);
            bytes.push(to);
            bytes.extend_from_slice(&amount.to_le_bytes());
        }
    }
}

fn encode_value(bytes: &mut Vec<u8>, value: PublicValue) {
    match value {
        PublicValue::Bit(value) => {
            bytes.push(1);
            bytes.push(u8::from(value));
        }
        PublicValue::Count(value) => {
            bytes.push(2);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST_A: [u8; 32] = [0x11; 32];
    const MANIFEST_B: [u8; 32] = [0x12; 32];
    const TARGET_A: [u8; 32] = [0x21; 32];
    const TARGET_B: [u8; 32] = [0x22; 32];
    const COMPARATOR_A: [u8; 32] = [0x31; 32];

    fn protocol() -> V2HeldOutRevealProtocol {
        V2HeldOutRevealProtocol::synthetic_canonical(MANIFEST_A, TARGET_A, COMPARATOR_A).unwrap()
    }

    fn prediction(ticket: V2HeldOutTicket, delta: i32) -> ConsequencePrediction {
        let mut fields = ticket.pre().fields();
        fields[0] = fields[0].saturating_add(delta);
        ConsequencePrediction {
            action: ticket.action(),
            outcome: PredictionOutcome::Predicted {
                fields: fields.into_iter().map(PublicValue::Count).collect(),
            },
        }
    }

    #[test]
    fn ticket_exposes_only_pre_outcome_information() {
        let protocol = protocol();
        let ticket = protocol.ticket(0).unwrap();
        assert_eq!(ticket.campaign_manifest_commitment(), MANIFEST_A);
        assert_eq!(ticket.row_index(), 0);
        assert_ne!(ticket.row_identity(), [0_u8; 32]);
        assert_ne!(ticket.commitment(), [0_u8; 32]);

        let source = include_str!("v2_heldout_reveal_protocol.rs");
        let start = source.find("struct V2HeldOutTicket").unwrap();
        let tail = &source[start..];
        let end = tail.find("}\n\nimpl V2HeldOutTicket").unwrap();
        let body = &tail[..end];
        assert!(!body.contains("post"));
    }

    #[test]
    fn paired_freeze_precedes_reveal_and_binds_both_predictions() {
        let protocol = protocol();
        let ticket = protocol.ticket(7).unwrap();
        let target_prediction = prediction(ticket, 1);
        let comparator_prediction = prediction(ticket, 0);
        let target = protocol.freeze_target(ticket, &target_prediction).unwrap();
        let comparator = protocol
            .freeze_comparator(ticket, &comparator_prediction)
            .unwrap();
        let paired = protocol.pair(ticket, target, comparator).unwrap();
        let paired_commitment = paired.commitment();
        let reveal = protocol.reveal(paired).unwrap();
        assert_eq!(reveal.paired_prediction_commitment(), paired_commitment);
        assert_ne!(reveal.commitment(), [0_u8; 32]);
    }

    #[test]
    fn wrong_actions_fail_before_pair_or_reveal() {
        let protocol = protocol();
        let ticket = protocol.ticket(0).unwrap();
        let wrong_action = match ticket.action() {
            PublicAction::NoOp => PublicAction::Pulse { slot: 0 },
            _ => PublicAction::NoOp,
        };
        let wrong = ConsequencePrediction {
            action: wrong_action,
            outcome: PredictionOutcome::OutOfQualifiedDomain,
        };
        assert_eq!(
            protocol.freeze_target(ticket, &wrong),
            Err(V2HeldOutRevealError::TargetActionMismatch)
        );
        assert_eq!(
            protocol.freeze_comparator(ticket, &wrong),
            Err(V2HeldOutRevealError::ComparatorActionMismatch)
        );
    }

    #[test]
    fn caller_mutation_after_role_freeze_cannot_change_paired_identity() {
        let protocol = protocol();
        let ticket = protocol.ticket(3).unwrap();
        let mut target_prediction = prediction(ticket, 1);
        let comparator_prediction = prediction(ticket, 0);
        let target = protocol.freeze_target(ticket, &target_prediction).unwrap();
        let target_commitment = target.commitment;
        target_prediction.outcome = PredictionOutcome::OutOfQualifiedDomain;
        assert_eq!(target.commitment, target_commitment);
        let comparator = protocol
            .freeze_comparator(ticket, &comparator_prediction)
            .unwrap();
        let paired = protocol.pair(ticket, target, comparator).unwrap();
        assert_ne!(paired.commitment(), [0_u8; 32]);
    }

    #[test]
    fn changing_either_prediction_changes_pair_commitment() {
        let protocol = protocol();
        let ticket = protocol.ticket(11).unwrap();

        let target_a = protocol.freeze_target(ticket, &prediction(ticket, 0)).unwrap();
        let comparator_a = protocol
            .freeze_comparator(ticket, &prediction(ticket, 0))
            .unwrap();
        let a = protocol.pair(ticket, target_a, comparator_a).unwrap().commitment();

        let target_b = protocol.freeze_target(ticket, &prediction(ticket, 1)).unwrap();
        let comparator_b = protocol
            .freeze_comparator(ticket, &prediction(ticket, 0))
            .unwrap();
        let b = protocol.pair(ticket, target_b, comparator_b).unwrap().commitment();
        assert_ne!(a, b);

        let target_c = protocol.freeze_target(ticket, &prediction(ticket, 0)).unwrap();
        let comparator_c = protocol
            .freeze_comparator(ticket, &prediction(ticket, 1))
            .unwrap();
        let c = protocol.pair(ticket, target_c, comparator_c).unwrap().commitment();
        assert_ne!(a, c);
    }

    #[test]
    fn changing_row_manifest_or_subject_identity_changes_freeze_identity() {
        let protocol_a = protocol();
        let ticket_a0 = protocol_a.ticket(0).unwrap();
        let ticket_a1 = protocol_a.ticket(1).unwrap();
        assert_ne!(ticket_a0.commitment(), ticket_a1.commitment());

        let protocol_b = V2HeldOutRevealProtocol::synthetic_canonical(
            MANIFEST_B,
            TARGET_A,
            COMPARATOR_A,
        )
        .unwrap();
        let ticket_b0 = protocol_b.ticket(0).unwrap();
        assert_ne!(ticket_a0.commitment(), ticket_b0.commitment());

        let protocol_c = V2HeldOutRevealProtocol::synthetic_canonical(
            MANIFEST_A,
            TARGET_B,
            COMPARATOR_A,
        )
        .unwrap();
        let target_a = protocol_a
            .freeze_target(ticket_a0, &prediction(ticket_a0, 0))
            .unwrap();
        let target_c = protocol_c
            .freeze_target(ticket_a0, &prediction(ticket_a0, 0))
            .unwrap();
        assert_ne!(target_a.commitment, target_c.commitment);
    }

    #[test]
    fn cross_row_and_cross_campaign_reveal_fail_closed() {
        let protocol_a = protocol();
        let ticket = protocol_a.ticket(5).unwrap();
        let target = protocol_a.freeze_target(ticket, &prediction(ticket, 0)).unwrap();
        let comparator = protocol_a
            .freeze_comparator(ticket, &prediction(ticket, 0))
            .unwrap();
        let mut paired = protocol_a.pair(ticket, target, comparator).unwrap();
        paired.ticket.row_identity[0] ^= 0x01;
        assert!(matches!(
            protocol_a.reveal(paired),
            Err(V2HeldOutRevealError::TicketRowMismatch)
                | Err(V2HeldOutRevealError::TicketCommitmentMismatch)
        ));

        let ticket = protocol_a.ticket(5).unwrap();
        let target = protocol_a.freeze_target(ticket, &prediction(ticket, 0)).unwrap();
        let comparator = protocol_a
            .freeze_comparator(ticket, &prediction(ticket, 0))
            .unwrap();
        let paired = protocol_a.pair(ticket, target, comparator).unwrap();
        let protocol_b = V2HeldOutRevealProtocol::synthetic_canonical(
            MANIFEST_B,
            TARGET_A,
            COMPARATOR_A,
        )
        .unwrap();
        assert_eq!(
            protocol_b.reveal(paired),
            Err(V2HeldOutRevealError::TicketManifestMismatch)
        );
    }

    #[test]
    fn reveal_receipt_scores_both_subjects_on_identical_transition() {
        let protocol = protocol();
        let ticket = protocol.ticket(21).unwrap();
        let target_prediction = prediction(ticket, 0);
        let comparator_prediction = prediction(ticket, 1);
        let target = protocol.freeze_target(ticket, &target_prediction).unwrap();
        let comparator = protocol
            .freeze_comparator(ticket, &comparator_prediction)
            .unwrap();
        let paired = protocol.pair(ticket, target, comparator).unwrap();
        let reveal = protocol.reveal(paired).unwrap();
        let revealed_post = reveal.post();
        let score = reveal.score().unwrap();

        assert_eq!(score.row_identity, ticket.row_identity());
        assert_eq!(score.action, ticket.action());
        assert_ne!(revealed_post, ticket.pre());
        assert!(matches!(score.target_score, ConsequenceScore::Scored(_)));
        assert!(matches!(score.comparator_score, ConsequenceScore::Scored(_)));
    }

    #[test]
    fn protocol_source_contains_no_real_heldout_subject_execution() {
        let source = include_str!("v2_heldout_reveal_protocol.rs");
        let production_source = source.split("#[cfg(test)]").next().unwrap();
        for forbidden in [
            "FepHeldOutSubject",
            "FepEvaluationTrial",
            "predict_once(",
            ".trial(",
            "V2SelectedComparatorSubject",
            "run_canonical_v2_fep_development",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "synthetic reveal qualification must not execute a real subject: {forbidden}"
            );
        }
    }
}
