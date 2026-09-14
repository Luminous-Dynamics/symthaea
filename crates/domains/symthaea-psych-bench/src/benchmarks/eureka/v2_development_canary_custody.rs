// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Development-only reveal custody for the EUREKA-002 V2 plumbing canary.
//!
//! This module is intentionally isolated from HeldOut reveal machinery. It
//! validates both real-subject prediction receipts against the exact authorized
//! Development row before prediction bytes are consumed, then requires one
//! move-only paired freeze before the already-consumed Development post-state
//! can be revealed. Every result remains PlumbingOnlyNonConfirmatory.

#![allow(dead_code)]

use super::consequence::{
    ConsequenceMetrics, ConsequencePrediction, ConsequenceScore, ConsequenceScoringError,
    score_consequence,
};
use super::hidden_world::{PublicObservation, PublicValue};
use super::v2_canary_authorization::{V2CanaryAuthorization, V2CanaryEvidenceClass};
use super::v2_corpus_schedule::{
    V2ScheduleMaterializationError, V2SchedulePartition, V2ScheduledRow,
};
use super::v2_development_order::{V2DevelopmentPlan, materialize_development_plan};
use super::v2_preheldout_custody::{
    V2PreHeldOutCampaignCapability, V2PreHeldOutManifest,
};
use super::v2_prospective_ticket::{
    V2ProspectiveTicket, V2ProspectiveTicketDomain, V2ProspectiveTicketError,
};
use super::v2_real_subject_adapters::{
    V2RealComparatorPredictionReceipt, V2RealSubjectAdapterError, V2RealSubjectAdapterPair,
    V2RealTargetPredictionReceipt,
};

pub(super) const V2_DEVELOPMENT_CANARY_PAIRED_FREEZE_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_PAIRED_FREEZE.v1";
pub(super) const V2_DEVELOPMENT_CANARY_REVEAL_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_REVEAL.v1";
pub(super) const V2_DEVELOPMENT_CANARY_SCORE_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_SCORE.v1";

#[derive(Debug)]
pub(super) enum V2DevelopmentCanaryCustodyError {
    Materialization(V2ScheduleMaterializationError),
    ProspectiveTicket(V2ProspectiveTicketError),
    Adapter(V2RealSubjectAdapterError),
    Scoring(ConsequenceScoringError),
    EvidenceClassMismatch,
    CampaignMismatch,
    AdapterPairMismatch,
    AdapterSourceMismatch,
    DevelopmentPlanMismatch,
    ScheduleRootMismatch,
    DevelopmentCorpusMismatch,
    DevelopmentOrderMismatch,
    RowIndexOutOfRange,
    RowNotDevelopment,
    AuthorizedRowMismatch,
    ProspectiveTicketMismatch,
    ProspectiveDomainMismatch,
    TargetReceiptTicketMismatch,
    ComparatorReceiptTicketMismatch,
    TargetReceiptDomainMismatch,
    ComparatorReceiptDomainMismatch,
    TargetReceiptCampaignMismatch,
    ComparatorReceiptCampaignMismatch,
    TargetSubjectMismatch,
    ComparatorSubjectMismatch,
    ReceiptAdapterSourceMismatch,
    TargetActionMismatch,
    ComparatorActionMismatch,
    FrozenAuthorizationMismatch,
    FrozenEvidenceClassMismatch,
    FrozenTicketMismatch,
    FrozenAdapterSourceMismatch,
    FrozenPairCommitmentMismatch,
    RevealAuthorizationMismatch,
    RevealEvidenceClassMismatch,
    RevealTicketMismatch,
    RevealCommitmentMismatch,
}

impl From<V2ScheduleMaterializationError> for V2DevelopmentCanaryCustodyError {
    fn from(value: V2ScheduleMaterializationError) -> Self {
        Self::Materialization(value)
    }
}

impl From<V2ProspectiveTicketError> for V2DevelopmentCanaryCustodyError {
    fn from(value: V2ProspectiveTicketError) -> Self {
        Self::ProspectiveTicket(value)
    }
}

impl From<V2RealSubjectAdapterError> for V2DevelopmentCanaryCustodyError {
    fn from(value: V2RealSubjectAdapterError) -> Self {
        Self::Adapter(value)
    }
}

impl From<ConsequenceScoringError> for V2DevelopmentCanaryCustodyError {
    fn from(value: ConsequenceScoringError) -> Self {
        Self::Scoring(value)
    }
}

/// Evaluator-owned custody for exactly one authorized Development canary row.
///
/// This type owns no HeldOut plan or HeldOut reveal capability.
#[derive(Debug)]
pub(super) struct V2DevelopmentCanaryCustodian {
    authorization: V2CanaryAuthorization,
    manifest: V2PreHeldOutManifest,
    plan: V2DevelopmentPlan,
}

/// Move-only proof that both real-subject receipts were validated against the
/// same exact Development canary ticket before either prediction was admitted.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2DevelopmentCanaryFrozenPair {
    evidence_class: V2CanaryEvidenceClass,
    authorization_commitment: [u8; 32],
    ticket: V2ProspectiveTicket,
    target_receipt_commitment: [u8; 32],
    comparator_receipt_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
    target_prediction: ConsequencePrediction,
    comparator_prediction: ConsequencePrediction,
    commitment: [u8; 32],
}

impl V2DevelopmentCanaryFrozenPair {
    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Move-only Development reveal receipt. The post-state can exist here only
/// after a paired freeze has already been produced and consumed by reveal().
#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2DevelopmentCanaryRevealReceipt {
    evidence_class: V2CanaryEvidenceClass,
    authorization_commitment: [u8; 32],
    ticket: V2ProspectiveTicket,
    paired_freeze_commitment: [u8; 32],
    target_prediction: ConsequencePrediction,
    comparator_prediction: ConsequencePrediction,
    post: super::v2_public_schema::V2PublicState,
    commitment: [u8; 32],
}

impl V2DevelopmentCanaryRevealReceipt {
    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Final canary-only score. There is deliberately no conversion from this type
/// into HeldOut evidence, campaign reports, scientific dispositions, or claim
/// maturity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct V2DevelopmentCanaryScore {
    evidence_class: V2CanaryEvidenceClass,
    authorization_commitment: [u8; 32],
    prospective_ticket_commitment: [u8; 32],
    paired_freeze_commitment: [u8; 32],
    reveal_receipt_commitment: [u8; 32],
    scorer_source_commitment: [u8; 32],
    target_score: ConsequenceScore,
    comparator_score: ConsequenceScore,
    commitment: [u8; 32],
}

impl V2DevelopmentCanaryScore {
    pub(super) const fn evidence_class(self) -> V2CanaryEvidenceClass {
        self.evidence_class
    }

    pub(super) const fn target_score(self) -> ConsequenceScore {
        self.target_score
    }

    pub(super) const fn comparator_score(self) -> ConsequenceScore {
        self.comparator_score
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

impl V2DevelopmentCanaryCustodian {
    pub(super) fn bind(
        campaign: &V2PreHeldOutCampaignCapability,
        authorization: V2CanaryAuthorization,
    ) -> Result<Self, V2DevelopmentCanaryCustodyError> {
        if authorization.evidence_class() != V2CanaryEvidenceClass::PlumbingOnlyNonConfirmatory {
            return Err(V2DevelopmentCanaryCustodyError::EvidenceClassMismatch);
        }

        let manifest = campaign.manifest();
        if authorization.campaign_manifest_commitment() != manifest.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::CampaignMismatch);
        }

        let adapter_pair = V2RealSubjectAdapterPair::bind_campaign(campaign)?;
        if authorization.adapter_pair_commitment() != adapter_pair.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::AdapterPairMismatch);
        }
        if authorization.adapter_source_commitment() != adapter_pair.adapter_source_commitment() {
            return Err(V2DevelopmentCanaryCustodyError::AdapterSourceMismatch);
        }

        let plan = materialize_development_plan()?;
        let development_receipt = campaign.development_receipt();
        if authorization.development_plan_commitment() != plan.commitment()
            || plan.commitment() != development_receipt.development_plan_commitment()
        {
            return Err(V2DevelopmentCanaryCustodyError::DevelopmentPlanMismatch);
        }
        if authorization.full_schedule_root() != plan.full_schedule_root()
            || plan.full_schedule_root() != development_receipt.full_schedule_root()
            || plan.full_schedule_root() != manifest.full_schedule_root()
        {
            return Err(V2DevelopmentCanaryCustodyError::ScheduleRootMismatch);
        }
        if authorization.development_corpus_commitment()
            != plan.development_corpus().commitment()
            || plan.development_corpus().commitment()
                != development_receipt.development_corpus_commitment()
        {
            return Err(V2DevelopmentCanaryCustodyError::DevelopmentCorpusMismatch);
        }
        if authorization.development_order_root() != plan.development_order_root()
            || plan.development_order_root() != development_receipt.development_order_root()
        {
            return Err(V2DevelopmentCanaryCustodyError::DevelopmentOrderMismatch);
        }

        let row = canonical_row(&plan, authorization.row_index())?;
        validate_authorized_row(&authorization, row)?;

        Ok(Self {
            authorization,
            manifest,
            plan,
        })
    }

    pub(super) fn prospective_ticket(
        &self,
    ) -> Result<V2ProspectiveTicket, V2DevelopmentCanaryCustodyError> {
        Ok(V2ProspectiveTicket::new(
            V2ProspectiveTicketDomain::DevelopmentPlumbingCanary,
            self.authorization.campaign_manifest_commitment(),
            self.authorization.row_index(),
            self.authorization.row_identity(),
            self.authorization.family(),
            self.authorization.pre(),
            self.authorization.action(),
        )?)
    }

    pub(super) fn freeze_receipts(
        &self,
        ticket: V2ProspectiveTicket,
        target: V2RealTargetPredictionReceipt,
        comparator: V2RealComparatorPredictionReceipt,
    ) -> Result<V2DevelopmentCanaryFrozenPair, V2DevelopmentCanaryCustodyError> {
        let canonical_ticket = self.prospective_ticket()?;
        if ticket != canonical_ticket {
            return Err(V2DevelopmentCanaryCustodyError::ProspectiveTicketMismatch);
        }
        if ticket.domain() != V2ProspectiveTicketDomain::DevelopmentPlumbingCanary {
            return Err(V2DevelopmentCanaryCustodyError::ProspectiveDomainMismatch);
        }

        self.validate_receipt_metadata(ticket, &target, &comparator)?;

        // Receipt commitments are captured before ownership-consuming extraction.
        // Every metadata check above has already succeeded at this point.
        let target_receipt_commitment = target.commitment();
        let comparator_receipt_commitment = comparator.commitment();
        let adapter_source_commitment = target.adapter_source_commitment();
        let target_prediction = target.into_prediction();
        let comparator_prediction = comparator.into_prediction();

        if target_prediction.action != self.authorization.action() {
            return Err(V2DevelopmentCanaryCustodyError::TargetActionMismatch);
        }
        if comparator_prediction.action != self.authorization.action() {
            return Err(V2DevelopmentCanaryCustodyError::ComparatorActionMismatch);
        }

        let commitment = paired_freeze_commitment(
            self.authorization.commitment(),
            ticket.commitment(),
            target_receipt_commitment,
            comparator_receipt_commitment,
            adapter_source_commitment,
        );

        Ok(V2DevelopmentCanaryFrozenPair {
            evidence_class: V2CanaryEvidenceClass::PlumbingOnlyNonConfirmatory,
            authorization_commitment: self.authorization.commitment(),
            ticket,
            target_receipt_commitment,
            comparator_receipt_commitment,
            adapter_source_commitment,
            target_prediction,
            comparator_prediction,
            commitment,
        })
    }

    pub(super) fn reveal(
        &self,
        frozen: V2DevelopmentCanaryFrozenPair,
    ) -> Result<V2DevelopmentCanaryRevealReceipt, V2DevelopmentCanaryCustodyError> {
        let canonical_ticket = self.prospective_ticket()?;
        if frozen.evidence_class != V2CanaryEvidenceClass::PlumbingOnlyNonConfirmatory {
            return Err(V2DevelopmentCanaryCustodyError::FrozenEvidenceClassMismatch);
        }
        if frozen.authorization_commitment != self.authorization.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::FrozenAuthorizationMismatch);
        }
        if frozen.ticket != canonical_ticket {
            return Err(V2DevelopmentCanaryCustodyError::FrozenTicketMismatch);
        }
        if frozen.adapter_source_commitment != self.authorization.adapter_source_commitment() {
            return Err(V2DevelopmentCanaryCustodyError::FrozenAdapterSourceMismatch);
        }
        let expected_pair = paired_freeze_commitment(
            frozen.authorization_commitment,
            frozen.ticket.commitment(),
            frozen.target_receipt_commitment,
            frozen.comparator_receipt_commitment,
            frozen.adapter_source_commitment,
        );
        if frozen.commitment != expected_pair {
            return Err(V2DevelopmentCanaryCustodyError::FrozenPairCommitmentMismatch);
        }

        let row = canonical_row(&self.plan, self.authorization.row_index())?;
        validate_authorized_row(&self.authorization, row)?;

        // This is the only canonical Development-plan post-state lookup in the
        // production custodian, and it occurs only after paired-freeze integrity
        // has been revalidated above.
        let post = row.post();
        let commitment = reveal_receipt_commitment(
            self.authorization.commitment(),
            frozen.ticket.commitment(),
            frozen.commitment,
            post,
        );

        Ok(V2DevelopmentCanaryRevealReceipt {
            evidence_class: frozen.evidence_class,
            authorization_commitment: frozen.authorization_commitment,
            ticket: frozen.ticket,
            paired_freeze_commitment: frozen.commitment,
            target_prediction: frozen.target_prediction,
            comparator_prediction: frozen.comparator_prediction,
            post,
            commitment,
        })
    }

    pub(super) fn score(
        &self,
        reveal: V2DevelopmentCanaryRevealReceipt,
    ) -> Result<V2DevelopmentCanaryScore, V2DevelopmentCanaryCustodyError> {
        let canonical_ticket = self.prospective_ticket()?;
        if reveal.evidence_class != V2CanaryEvidenceClass::PlumbingOnlyNonConfirmatory {
            return Err(V2DevelopmentCanaryCustodyError::RevealEvidenceClassMismatch);
        }
        if reveal.authorization_commitment != self.authorization.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::RevealAuthorizationMismatch);
        }
        if reveal.ticket != canonical_ticket {
            return Err(V2DevelopmentCanaryCustodyError::RevealTicketMismatch);
        }
        let expected_reveal = reveal_receipt_commitment(
            reveal.authorization_commitment,
            reveal.ticket.commitment(),
            reveal.paired_freeze_commitment,
            reveal.post,
        );
        if reveal.commitment != expected_reveal {
            return Err(V2DevelopmentCanaryCustodyError::RevealCommitmentMismatch);
        }

        let pre = observation(reveal.ticket.pre(), 0);
        let post = observation(reveal.post, 1);
        let target_score = score_consequence(&pre, &reveal.target_prediction, &post)?;
        let comparator_score = score_consequence(&pre, &reveal.comparator_prediction, &post)?;
        let scorer_source_commitment = self.manifest.scorer_source_commitment();
        let commitment = canary_score_commitment(
            reveal.authorization_commitment,
            reveal.ticket.commitment(),
            reveal.paired_freeze_commitment,
            reveal.commitment,
            scorer_source_commitment,
            target_score,
            comparator_score,
        );

        Ok(V2DevelopmentCanaryScore {
            evidence_class: V2CanaryEvidenceClass::PlumbingOnlyNonConfirmatory,
            authorization_commitment: reveal.authorization_commitment,
            prospective_ticket_commitment: reveal.ticket.commitment(),
            paired_freeze_commitment: reveal.paired_freeze_commitment,
            reveal_receipt_commitment: reveal.commitment,
            scorer_source_commitment,
            target_score,
            comparator_score,
            commitment,
        })
    }

    fn validate_receipt_metadata(
        &self,
        ticket: V2ProspectiveTicket,
        target: &V2RealTargetPredictionReceipt,
        comparator: &V2RealComparatorPredictionReceipt,
    ) -> Result<(), V2DevelopmentCanaryCustodyError> {
        if target.prospective_ticket_commitment() != ticket.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::TargetReceiptTicketMismatch);
        }
        if comparator.prospective_ticket_commitment() != ticket.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::ComparatorReceiptTicketMismatch);
        }
        if target.domain() != V2ProspectiveTicketDomain::DevelopmentPlumbingCanary {
            return Err(V2DevelopmentCanaryCustodyError::TargetReceiptDomainMismatch);
        }
        if comparator.domain() != V2ProspectiveTicketDomain::DevelopmentPlumbingCanary {
            return Err(V2DevelopmentCanaryCustodyError::ComparatorReceiptDomainMismatch);
        }
        if target.campaign_manifest_commitment() != self.manifest.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::TargetReceiptCampaignMismatch);
        }
        if comparator.campaign_manifest_commitment() != self.manifest.commitment() {
            return Err(V2DevelopmentCanaryCustodyError::ComparatorReceiptCampaignMismatch);
        }
        if target.subject_commitment().as_bytes() != &self.manifest.learned_subject_commitment() {
            return Err(V2DevelopmentCanaryCustodyError::TargetSubjectMismatch);
        }
        if comparator.subject_commitment() != self.manifest.selected_comparator_commitment() {
            return Err(V2DevelopmentCanaryCustodyError::ComparatorSubjectMismatch);
        }
        if target.adapter_source_commitment() != comparator.adapter_source_commitment()
            || target.adapter_source_commitment() != self.authorization.adapter_source_commitment()
        {
            return Err(V2DevelopmentCanaryCustodyError::ReceiptAdapterSourceMismatch);
        }
        Ok(())
    }
}

fn canonical_row(
    plan: &V2DevelopmentPlan,
    row_index: u16,
) -> Result<V2ScheduledRow, V2DevelopmentCanaryCustodyError> {
    plan.ordered_rows()
        .get(usize::from(row_index))
        .copied()
        .ok_or(V2DevelopmentCanaryCustodyError::RowIndexOutOfRange)
}

fn validate_authorized_row(
    authorization: &V2CanaryAuthorization,
    row: V2ScheduledRow,
) -> Result<(), V2DevelopmentCanaryCustodyError> {
    if row.partition() != V2SchedulePartition::Development {
        return Err(V2DevelopmentCanaryCustodyError::RowNotDevelopment);
    }
    if row.row_identity() != authorization.row_identity()
        || row.family() != authorization.family()
        || row.pre() != authorization.pre()
        || row.action() != authorization.action()
    {
        return Err(V2DevelopmentCanaryCustodyError::AuthorizedRowMismatch);
    }
    Ok(())
}

fn paired_freeze_commitment(
    authorization_commitment: [u8; 32],
    prospective_ticket_commitment: [u8; 32],
    target_receipt_commitment: [u8; 32],
    comparator_receipt_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_DEVELOPMENT_CANARY_PAIRED_FREEZE_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&authorization_commitment);
    bytes.extend_from_slice(&prospective_ticket_commitment);
    bytes.extend_from_slice(&target_receipt_commitment);
    bytes.extend_from_slice(&comparator_receipt_commitment);
    bytes.extend_from_slice(&adapter_source_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn reveal_receipt_commitment(
    authorization_commitment: [u8; 32],
    prospective_ticket_commitment: [u8; 32],
    paired_freeze_commitment: [u8; 32],
    post: super::v2_public_schema::V2PublicState,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_DEVELOPMENT_CANARY_REVEAL_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&authorization_commitment);
    bytes.extend_from_slice(&prospective_ticket_commitment);
    bytes.extend_from_slice(&paired_freeze_commitment);
    for field in post.fields() {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn canary_score_commitment(
    authorization_commitment: [u8; 32],
    prospective_ticket_commitment: [u8; 32],
    paired_freeze_commitment: [u8; 32],
    reveal_receipt_commitment: [u8; 32],
    scorer_source_commitment: [u8; 32],
    target_score: ConsequenceScore,
    comparator_score: ConsequenceScore,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_DEVELOPMENT_CANARY_SCORE_REVISION.as_bytes());
    bytes.extend_from_slice(&authorization_commitment);
    bytes.extend_from_slice(&prospective_ticket_commitment);
    bytes.extend_from_slice(&paired_freeze_commitment);
    bytes.extend_from_slice(&reveal_receipt_commitment);
    bytes.extend_from_slice(&scorer_source_commitment);
    encode_score(&mut bytes, target_score);
    encode_score(&mut bytes, comparator_score);
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

fn observation(state: super::v2_public_schema::V2PublicState, step: u64) -> PublicObservation {
    PublicObservation {
        step,
        fields: state
            .fields()
            .into_iter()
            .map(PublicValue::Count)
            .collect(),
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_source_is_development_only_and_non_promotable() {
        let source = include_str!("v2_development_canary_custody.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "v2_heldout_plan",
            "materialize_heldout_plan",
            "V2HeldOutRevealProtocol",
            "V2HeldOutTicket",
            "V2ShadowCampaignReport",
            "ScientificDisposition",
            "UnderstandingMaturity",
            "learn_from_actual(",
            "freeze_for_evaluation(",
            "ActiveInferenceAgent",
        ] {
            assert!(
                !source.contains(forbidden),
                "Development canary custody gained forbidden authority: {forbidden}"
            );
        }
        assert!(source.contains("PlumbingOnlyNonConfirmatory"));
        assert!(source.contains("V2ProspectiveTicketDomain::DevelopmentPlumbingCanary"));
    }

    #[test]
    fn metadata_validation_precedes_prediction_extraction() {
        let source = include_str!("v2_development_canary_custody.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("pub(super) fn freeze_receipts(").unwrap();
        let tail = &source[start..];
        let end = tail.find("\n    pub(super) fn reveal(").unwrap();
        let body = &tail[..end];
        let validate = body.find("self.validate_receipt_metadata").unwrap();
        let target_extract = body.find("target.into_prediction()").unwrap();
        let comparator_extract = body.find("comparator.into_prediction()").unwrap();
        assert!(validate < target_extract);
        assert!(validate < comparator_extract);
    }

    #[test]
    fn canonical_development_post_lookup_occurs_only_after_pair_validation() {
        let source = include_str!("v2_development_canary_custody.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert_eq!(source.matches("row.post()").count(), 1);
        let reveal_start = source.find("pub(super) fn reveal(").unwrap();
        let post_lookup = source.find("let post = row.post();").unwrap();
        let pair_check = source.find("FrozenPairCommitmentMismatch").unwrap();
        assert!(reveal_start < pair_check);
        assert!(pair_check < post_lookup);
    }

    #[test]
    fn frozen_pair_and_reveal_receipt_are_move_only_by_source_contract() {
        let source = include_str!("v2_development_canary_custody.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let frozen = source.find("struct V2DevelopmentCanaryFrozenPair").unwrap();
        let reveal = source.find("struct V2DevelopmentCanaryRevealReceipt").unwrap();
        let frozen_prefix = &source[frozen.saturating_sub(64)..frozen];
        let reveal_prefix = &source[reveal.saturating_sub(64)..reveal];
        assert!(!frozen_prefix.contains("Clone"));
        assert!(!frozen_prefix.contains("Copy"));
        assert!(!reveal_prefix.contains("Clone"));
        assert!(!reveal_prefix.contains("Copy"));
    }

    #[test]
    fn final_score_identity_binds_raw_scores_and_scorer_source() {
        let metrics = ConsequenceMetrics {
            field_count: 4,
            actual_changed: 2,
            predicted_changed: 2,
            true_positive_changes: 2,
            false_positive_changes: 0,
            missed_changes: 0,
            correct_changed_values: 2,
            correct_unchanged_values: 2,
            correct_full_state_values: 4,
            changed_field_precision: Some(1.0),
            changed_field_recall: Some(1.0),
            changed_field_f1: Some(1.0),
            changed_value_accuracy: Some(1.0),
            unchanged_state_preservation: Some(1.0),
            full_state_accuracy: 1.0,
        };
        let exact = canary_score_commitment(
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            [5; 32],
            ConsequenceScore::Scored(metrics),
            ConsequenceScore::OutOfQualifiedDomain,
        );
        let changed_source = canary_score_commitment(
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            [6; 32],
            ConsequenceScore::Scored(metrics),
            ConsequenceScore::OutOfQualifiedDomain,
        );
        let changed_score = canary_score_commitment(
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            [5; 32],
            ConsequenceScore::AbstainedInsufficientEvidence,
            ConsequenceScore::OutOfQualifiedDomain,
        );
        assert_ne!(exact, changed_source);
        assert_ne!(exact, changed_score);
    }
}
