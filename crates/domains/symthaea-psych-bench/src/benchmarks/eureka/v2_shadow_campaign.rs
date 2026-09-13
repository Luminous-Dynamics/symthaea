// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full synthetic-only rehearsal of the EUREKA-002 V2 HeldOut campaign.
//!
//! This exercises all 128 canonical HeldOut rows through the qualified
//! freeze/pair/reveal/score state machine without invoking the real FEP target
//! or Calibration-selected comparator. The report preserves raw row evidence
//! and aggregate counts but mints no scientific disposition.

#![allow(dead_code)]

use std::collections::BTreeSet;

use super::consequence::{
    ConsequenceMetrics, ConsequencePrediction, ConsequenceScore, PredictionOutcome,
};
use super::hidden_world::{PublicAction, PublicValue};
use super::v2_heldout_plan::V2_HELDOUT_ROWS_TOTAL;
use super::v2_heldout_reveal_protocol::{
    V2HeldOutPairScore, V2HeldOutRevealError, V2HeldOutRevealProtocol,
};
use super::v2_public_schema::{
    V2PublicFamily, action_from_index, action_index,
};

pub(super) const V2_SHADOW_CAMPAIGN_REVISION: &str =
    "EUREKA.002.V2.SHADOW_CAMPAIGN.v1";
const SHADOW_CAMPAIGN_MANIFEST: [u8; 32] = [0xD1; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ShadowPolicy {
    PublicMechanismOracle,
    CopyPreState,
    Abstain,
    WrongActionAt(u16),
}

impl V2ShadowPolicy {
    fn stable_id(self) -> String {
        match self {
            Self::PublicMechanismOracle => "public-mechanism-oracle-v1".to_string(),
            Self::CopyPreState => "copy-pre-state-v1".to_string(),
            Self::Abstain => "abstain-v1".to_string(),
            Self::WrongActionAt(index) => format!("wrong-action-at-{index}-v1"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ShadowCampaignError {
    RevealProtocol(V2HeldOutRevealError),
    UnsupportedPublicAction,
    WrongRowCount,
    DuplicateRowIndex,
    DuplicateRowIdentity,
    NonCanonicalRowOrder,
}

impl From<V2HeldOutRevealError> for V2ShadowCampaignError {
    fn from(value: V2HeldOutRevealError) -> Self {
        Self::RevealProtocol(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct V2ShadowRowReceipt {
    row_index: u16,
    row_identity: [u8; 32],
    action: PublicAction,
    paired_prediction_commitment: [u8; 32],
    reveal_receipt_commitment: [u8; 32],
    target_score: ConsequenceScore,
    comparator_score: ConsequenceScore,
    commitment: [u8; 32],
}

impl V2ShadowRowReceipt {
    pub(super) const fn row_index(&self) -> u16 {
        self.row_index
    }

    pub(super) const fn row_identity(&self) -> [u8; 32] {
        self.row_identity
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct V2ShadowAggregate {
    pub scored_rows: u32,
    pub abstained_rows: u32,
    pub out_of_domain_rows: u32,
    pub actual_changed: u64,
    pub true_positive_changes: u64,
    pub false_positive_changes: u64,
    pub missed_changes: u64,
    pub correct_changed_values: u64,
}

impl V2ShadowAggregate {
    fn accumulate(&mut self, score: ConsequenceScore) {
        match score {
            ConsequenceScore::Scored(metrics) => {
                self.scored_rows = self.scored_rows.saturating_add(1);
                self.actual_changed = self
                    .actual_changed
                    .saturating_add(metrics.actual_changed as u64);
                self.true_positive_changes = self
                    .true_positive_changes
                    .saturating_add(metrics.true_positive_changes as u64);
                self.false_positive_changes = self
                    .false_positive_changes
                    .saturating_add(metrics.false_positive_changes as u64);
                self.missed_changes = self
                    .missed_changes
                    .saturating_add(metrics.missed_changes as u64);
                self.correct_changed_values = self
                    .correct_changed_values
                    .saturating_add(metrics.correct_changed_values as u64);
            }
            ConsequenceScore::AbstainedInsufficientEvidence => {
                self.abstained_rows = self.abstained_rows.saturating_add(1);
            }
            ConsequenceScore::OutOfQualifiedDomain => {
                self.out_of_domain_rows = self.out_of_domain_rows.saturating_add(1);
            }
        }
    }

    pub(super) fn changed_field_f1_bps(self) -> Option<u16> {
        let numerator = self.true_positive_changes.saturating_mul(2);
        let denominator = numerator
            .saturating_add(self.false_positive_changes)
            .saturating_add(self.missed_changes);
        ratio_bps(numerator, denominator)
    }

    pub(super) fn changed_value_accuracy_bps(self) -> Option<u16> {
        ratio_bps(self.correct_changed_values, self.actual_changed)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct V2ShadowCampaignReport {
    target_policy: V2ShadowPolicy,
    comparator_policy: V2ShadowPolicy,
    target_subject_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    rows: Vec<V2ShadowRowReceipt>,
    row_receipt_root: [u8; 32],
    target_aggregate: V2ShadowAggregate,
    comparator_aggregate: V2ShadowAggregate,
    commitment: [u8; 32],
}

impl V2ShadowCampaignReport {
    pub(super) fn rows(&self) -> &[V2ShadowRowReceipt] {
        &self.rows
    }

    pub(super) const fn row_receipt_root(&self) -> [u8; 32] {
        self.row_receipt_root
    }

    pub(super) const fn target_aggregate(&self) -> V2ShadowAggregate {
        self.target_aggregate
    }

    pub(super) const fn comparator_aggregate(&self) -> V2ShadowAggregate {
        self.comparator_aggregate
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

pub(super) fn execute_shadow_campaign(
    target_policy: V2ShadowPolicy,
    comparator_policy: V2ShadowPolicy,
) -> Result<V2ShadowCampaignReport, V2ShadowCampaignError> {
    let target_subject_commitment = synthetic_subject_commitment(b"target", target_policy);
    let comparator_subject_commitment =
        synthetic_subject_commitment(b"comparator", comparator_policy);
    let protocol = V2HeldOutRevealProtocol::synthetic_canonical(
        SHADOW_CAMPAIGN_MANIFEST,
        target_subject_commitment,
        comparator_subject_commitment,
    )?;

    let mut rows = Vec::with_capacity(V2_HELDOUT_ROWS_TOTAL);
    let mut target_aggregate = V2ShadowAggregate::default();
    let mut comparator_aggregate = V2ShadowAggregate::default();

    for row_index in 0..V2_HELDOUT_ROWS_TOTAL {
        let ticket = protocol.ticket(row_index)?;
        let target_prediction = prediction_for_policy(target_policy, ticket)?;
        let comparator_prediction = prediction_for_policy(comparator_policy, ticket)?;
        let target = protocol.freeze_target(ticket, &target_prediction)?;
        let comparator = protocol.freeze_comparator(ticket, &comparator_prediction)?;
        let paired = protocol.pair(ticket, target, comparator)?;
        let paired_prediction_commitment = paired.commitment();
        let reveal = protocol.reveal(paired)?;
        let reveal_receipt_commitment = reveal.commitment();
        let pair_score = reveal.score()?;

        target_aggregate.accumulate(pair_score.target_score);
        comparator_aggregate.accumulate(pair_score.comparator_score);
        rows.push(row_receipt(
            pair_score,
            paired_prediction_commitment,
            reveal_receipt_commitment,
        ));
    }

    validate_complete_rows(&rows)?;
    let row_receipt_root = ordered_row_receipt_root(&rows);
    let commitment = shadow_campaign_commitment(
        target_policy,
        comparator_policy,
        target_subject_commitment,
        comparator_subject_commitment,
        row_receipt_root,
        target_aggregate,
        comparator_aggregate,
        rows.len(),
    );
    Ok(V2ShadowCampaignReport {
        target_policy,
        comparator_policy,
        target_subject_commitment,
        comparator_subject_commitment,
        rows,
        row_receipt_root,
        target_aggregate,
        comparator_aggregate,
        commitment,
    })
}

fn prediction_for_policy(
    policy: V2ShadowPolicy,
    ticket: super::v2_heldout_reveal_protocol::V2HeldOutTicket,
) -> Result<ConsequencePrediction, V2ShadowCampaignError> {
    let outcome = match policy {
        V2ShadowPolicy::PublicMechanismOracle => PredictionOutcome::Predicted {
            fields: public_mechanism_prediction(ticket)?
                .into_iter()
                .map(PublicValue::Count)
                .collect(),
        },
        V2ShadowPolicy::CopyPreState => PredictionOutcome::Predicted {
            fields: ticket
                .pre()
                .fields()
                .into_iter()
                .map(PublicValue::Count)
                .collect(),
        },
        V2ShadowPolicy::Abstain => PredictionOutcome::AbstainInsufficientEvidence,
        V2ShadowPolicy::WrongActionAt(index) => {
            if ticket.row_index() == index {
                return Ok(ConsequencePrediction {
                    action: different_action(ticket.action())?,
                    outcome: PredictionOutcome::AbstainInsufficientEvidence,
                });
            }
            PredictionOutcome::Predicted {
                fields: ticket
                    .pre()
                    .fields()
                    .into_iter()
                    .map(PublicValue::Count)
                    .collect(),
            }
        }
    };
    Ok(ConsequencePrediction {
        action: ticket.action(),
        outcome,
    })
}

fn public_mechanism_prediction(
    ticket: super::v2_heldout_reveal_protocol::V2HeldOutTicket,
) -> Result<[i32; 4], V2ShadowCampaignError> {
    let action = u8::try_from(action_index(ticket.action()).map_err(|_| V2ShadowCampaignError::UnsupportedPublicAction)?)
        .map_err(|_| V2ShadowCampaignError::UnsupportedPublicAction)?;
    let [mut x, mut y, mut z, context] = ticket.pre().fields();
    match ticket.family() {
        V2PublicFamily::PublicFlowV2 => {
            let mut values = [x, y, z];
            match action {
                0 => {}
                1 => transfer_one(&mut values, 0, 1),
                2 => transfer_one(&mut values, 1, 2),
                3 => transfer_one(&mut values, 2, 0),
                _ => return Err(V2ShadowCampaignError::UnsupportedPublicAction),
            }
            match context {
                0 => {}
                1 => values[0] = values[0].saturating_add(1),
                2 => transfer_one(&mut values, 0, 1),
                3 => transfer_one(&mut values, 1, 2),
                _ => return Err(V2ShadowCampaignError::UnsupportedPublicAction),
            }
            Ok([values[0], values[1], values[2], context])
        }
        V2PublicFamily::PublicRelayV2 => {
            match action {
                0 => {}
                1 => x = x.saturating_add(1),
                2 => y = y.saturating_add(1),
                3 => z = z.saturating_add(1),
                _ => return Err(V2ShadowCampaignError::UnsupportedPublicAction),
            }
            match context {
                4 => {}
                5 => y = x,
                6 => z = y,
                7 => x = z,
                _ => return Err(V2ShadowCampaignError::UnsupportedPublicAction),
            }
            Ok([x, y, z, context])
        }
    }
}

fn transfer_one(values: &mut [i32; 3], from: usize, to: usize) {
    if values[from] > 0 {
        values[from] -= 1;
        values[to] += 1;
    }
}

fn different_action(action: PublicAction) -> Result<PublicAction, V2ShadowCampaignError> {
    let index = action_index(action).map_err(|_| V2ShadowCampaignError::UnsupportedPublicAction)?;
    action_from_index((index + 1) % 4).map_err(|_| V2ShadowCampaignError::UnsupportedPublicAction)
}

fn row_receipt(
    score: V2HeldOutPairScore,
    paired_prediction_commitment: [u8; 32],
    reveal_receipt_commitment: [u8; 32],
) -> V2ShadowRowReceipt {
    let commitment = shadow_row_receipt_commitment(
        score.row_index,
        score.row_identity,
        score.action,
        paired_prediction_commitment,
        reveal_receipt_commitment,
        score.target_score,
        score.comparator_score,
    );
    V2ShadowRowReceipt {
        row_index: score.row_index,
        row_identity: score.row_identity,
        action: score.action,
        paired_prediction_commitment,
        reveal_receipt_commitment,
        target_score: score.target_score,
        comparator_score: score.comparator_score,
        commitment,
    }
}

fn validate_complete_rows(rows: &[V2ShadowRowReceipt]) -> Result<(), V2ShadowCampaignError> {
    if rows.len() != V2_HELDOUT_ROWS_TOTAL {
        return Err(V2ShadowCampaignError::WrongRowCount);
    }
    let mut indexes = BTreeSet::new();
    let mut identities = BTreeSet::new();
    for (expected, row) in rows.iter().enumerate() {
        if usize::from(row.row_index) != expected {
            return Err(V2ShadowCampaignError::NonCanonicalRowOrder);
        }
        if !indexes.insert(row.row_index) {
            return Err(V2ShadowCampaignError::DuplicateRowIndex);
        }
        if !identities.insert(row.row_identity) {
            return Err(V2ShadowCampaignError::DuplicateRowIdentity);
        }
    }
    Ok(())
}

fn ordered_row_receipt_root(rows: &[V2ShadowRowReceipt]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.SHADOW_ROW_ROOT.v1");
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        bytes.extend_from_slice(&row.commitment);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn synthetic_subject_commitment(role: &[u8], policy: V2ShadowPolicy) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.SHADOW_SUBJECT.v1");
    encode_bytes(&mut bytes, role);
    encode_bytes(&mut bytes, policy.stable_id().as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn shadow_row_receipt_commitment(
    row_index: u16,
    row_identity: [u8; 32],
    action: PublicAction,
    paired_prediction_commitment: [u8; 32],
    reveal_receipt_commitment: [u8; 32],
    target_score: ConsequenceScore,
    comparator_score: ConsequenceScore,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.SHADOW_ROW_RECEIPT.v1");
    bytes.extend_from_slice(&SHADOW_CAMPAIGN_MANIFEST);
    bytes.extend_from_slice(&row_index.to_le_bytes());
    bytes.extend_from_slice(&row_identity);
    encode_action(&mut bytes, action);
    bytes.extend_from_slice(&paired_prediction_commitment);
    bytes.extend_from_slice(&reveal_receipt_commitment);
    encode_score(&mut bytes, target_score);
    encode_score(&mut bytes, comparator_score);
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn shadow_campaign_commitment(
    target_policy: V2ShadowPolicy,
    comparator_policy: V2ShadowPolicy,
    target_subject_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    row_receipt_root: [u8; 32],
    target_aggregate: V2ShadowAggregate,
    comparator_aggregate: V2ShadowAggregate,
    row_count: usize,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_SHADOW_CAMPAIGN_REVISION.as_bytes());
    bytes.extend_from_slice(&SHADOW_CAMPAIGN_MANIFEST);
    encode_bytes(&mut bytes, target_policy.stable_id().as_bytes());
    encode_bytes(&mut bytes, comparator_policy.stable_id().as_bytes());
    bytes.extend_from_slice(&target_subject_commitment);
    bytes.extend_from_slice(&comparator_subject_commitment);
    bytes.extend_from_slice(&row_receipt_root);
    encode_aggregate(&mut bytes, target_aggregate);
    encode_aggregate(&mut bytes, comparator_aggregate);
    bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_score(bytes: &mut Vec<u8>, score: ConsequenceScore) {
    match score {
        ConsequenceScore::Scored(metrics) => {
            bytes.push(1);
            encode_metrics(bytes, metrics);
        }
        ConsequenceScore::AbstainedInsufficientEvidence => bytes.push(2),
        ConsequenceScore::OutOfQualifiedDomain => bytes.push(3),
    }
}

fn encode_metrics(bytes: &mut Vec<u8>, metrics: ConsequenceMetrics) {
    for value in [
        metrics.field_count,
        metrics.actual_changed,
        metrics.predicted_changed,
        metrics.true_positive_changes,
        metrics.false_positive_changes,
        metrics.missed_changes,
        metrics.correct_changed_values,
        metrics.correct_unchanged_values,
        metrics.correct_full_state_values,
    ] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
}

fn encode_aggregate(bytes: &mut Vec<u8>, aggregate: V2ShadowAggregate) {
    for value in [
        u64::from(aggregate.scored_rows),
        u64::from(aggregate.abstained_rows),
        u64::from(aggregate.out_of_domain_rows),
        aggregate.actual_changed,
        aggregate.true_positive_changes,
        aggregate.false_positive_changes,
        aggregate.missed_changes,
        aggregate.correct_changed_values,
    ] {
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

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn ratio_bps(numerator: u64, denominator: u64) -> Option<u16> {
    if denominator == 0 {
        return None;
    }
    let value = (u128::from(numerator) * 10_000_u128) / u128::from(denominator);
    Some(u16::try_from(value).expect("basis-point ratio is <= 10_000"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_vs_copy_rehearses_all_rows_and_preserves_raw_counts() {
        let report = execute_shadow_campaign(
            V2ShadowPolicy::PublicMechanismOracle,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        assert_eq!(report.rows().len(), V2_HELDOUT_ROWS_TOTAL);
        assert_eq!(report.target_aggregate().scored_rows, 128);
        assert_eq!(report.comparator_aggregate().scored_rows, 128);
        assert_eq!(report.target_aggregate().abstained_rows, 0);
        assert_eq!(report.target_aggregate().missed_changes, 0);
        assert_eq!(report.target_aggregate().false_positive_changes, 0);
        assert_eq!(
            report.target_aggregate().changed_field_f1_bps(),
            Some(10_000)
        );
        assert_eq!(
            report.target_aggregate().changed_value_accuracy_bps(),
            Some(10_000)
        );
        assert_eq!(report.comparator_aggregate().true_positive_changes, 0);
        assert_eq!(report.comparator_aggregate().changed_field_f1_bps(), Some(0));
        assert_ne!(report.row_receipt_root(), [0_u8; 32]);
        assert_ne!(report.commitment(), [0_u8; 32]);
    }

    #[test]
    fn every_canonical_row_appears_exactly_once_in_order() {
        let report = execute_shadow_campaign(
            V2ShadowPolicy::CopyPreState,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        let mut identities = BTreeSet::new();
        for (index, row) in report.rows().iter().enumerate() {
            assert_eq!(usize::from(row.row_index()), index);
            assert!(identities.insert(row.row_identity()));
        }
        assert_eq!(identities.len(), V2_HELDOUT_ROWS_TOTAL);
    }

    #[test]
    fn deterministic_replay_has_identical_root_and_campaign_identity() {
        let first = execute_shadow_campaign(
            V2ShadowPolicy::PublicMechanismOracle,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        let second = execute_shadow_campaign(
            V2ShadowPolicy::PublicMechanismOracle,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        assert_eq!(first.row_receipt_root(), second.row_receipt_root());
        assert_eq!(first.commitment(), second.commitment());
        assert_eq!(first.target_aggregate(), second.target_aggregate());
        assert_eq!(first.comparator_aggregate(), second.comparator_aggregate());
    }

    #[test]
    fn copy_vs_copy_is_a_null_control() {
        let report = execute_shadow_campaign(
            V2ShadowPolicy::CopyPreState,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        assert_eq!(report.target_aggregate(), report.comparator_aggregate());
    }

    #[test]
    fn abstention_remains_explicit_and_is_not_converted_to_no_change() {
        let report = execute_shadow_campaign(
            V2ShadowPolicy::Abstain,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        assert_eq!(report.target_aggregate().scored_rows, 0);
        assert_eq!(report.target_aggregate().abstained_rows, 128);
        assert_eq!(report.target_aggregate().actual_changed, 0);
        assert_eq!(report.comparator_aggregate().scored_rows, 128);
    }

    #[test]
    fn one_wrong_action_aborts_whole_campaign_without_success_report() {
        let result = execute_shadow_campaign(
            V2ShadowPolicy::WrongActionAt(17),
            V2ShadowPolicy::CopyPreState,
        );
        assert!(matches!(
            result,
            Err(V2ShadowCampaignError::RevealProtocol(
                V2HeldOutRevealError::TargetActionMismatch
            ))
        ));
    }

    #[test]
    fn policy_and_row_receipt_changes_change_campaign_identity() {
        let oracle = execute_shadow_campaign(
            V2ShadowPolicy::PublicMechanismOracle,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        let copy = execute_shadow_campaign(
            V2ShadowPolicy::CopyPreState,
            V2ShadowPolicy::CopyPreState,
        )
        .unwrap();
        assert_ne!(oracle.commitment(), copy.commitment());

        let mut mutated = oracle.rows().to_vec();
        let original_root = ordered_row_receipt_root(&mutated);
        mutated[9].commitment[0] ^= 0x01;
        assert_ne!(original_root, ordered_row_receipt_root(&mutated));
    }

    #[test]
    fn shadow_runner_has_no_real_subject_execution_reachability() {
        let source = include_str!("v2_shadow_campaign.rs");
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
                "shadow campaign must not execute a real subject: {forbidden}"
            );
        }
    }
}
