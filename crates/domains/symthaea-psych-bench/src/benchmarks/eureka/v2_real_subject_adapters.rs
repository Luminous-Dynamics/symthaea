// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-only adapters from sealed real subjects into the qualified V2
//! prospective prediction protocol.
//!
//! A real-subject prediction is not returned as loose bytes. Each adapter mints
//! a role-specific, move-only receipt binding the exact prospective ticket,
//! sealed subject, adapter implementation, and canonical prediction content.
//! This module defines no row loop, evaluator reveal, scoring, or learning path.

#![allow(dead_code)]

use symthaea_fep::{FepHeldOutSubject, FepPredictionSessionError, FrozenPredictionCommitment};

use super::consequence::{ConsequencePrediction, PredictionOutcome};
use super::hidden_world::PublicValue;
use super::v2_preheldout_custody::{
    V2PreHeldOutCampaignCapability, V2PreHeldOutManifest,
};
use super::v2_prospective_ticket::{V2ProspectiveTicket, V2ProspectiveTicketDomain};
use super::v2_public_schema::action_index;
use super::v2_selected_comparator::V2SelectedComparatorSubject;
use super::v2_target_contract::{V2FepAdapter, V2TargetContractError};

pub(super) const V2_REAL_TARGET_ADAPTER_REVISION: &str =
    "EUREKA.002.V2.REAL_TARGET_ADAPTER.v3";
pub(super) const V2_REAL_COMPARATOR_ADAPTER_REVISION: &str =
    "EUREKA.002.V2.REAL_COMPARATOR_ADAPTER.v3";
pub(super) const V2_REAL_ADAPTER_SOURCE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.REAL_ADAPTER_SOURCE.v3";
pub(super) const V2_REAL_ADAPTER_PAIR_REVISION: &str =
    "EUREKA.002.V2.REAL_ADAPTER_PAIR.v3";
pub(super) const V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION: &str =
    "EUREKA.002.V2.REAL_TARGET_PREDICTION_RECEIPT.v1";
pub(super) const V2_REAL_COMPARATOR_PREDICTION_RECEIPT_REVISION: &str =
    "EUREKA.002.V2.REAL_COMPARATOR_PREDICTION_RECEIPT.v1";

#[derive(Debug)]
pub(super) enum V2RealSubjectAdapterError {
    Fep(FepPredictionSessionError),
    Target(V2TargetContractError),
    TicketCampaignMismatch,
    TargetSubjectCommitmentMismatch,
    ComparatorSubjectCommitmentMismatch,
    TargetActionMismatch,
    ComparatorActionMismatch,
}

impl From<FepPredictionSessionError> for V2RealSubjectAdapterError {
    fn from(value: FepPredictionSessionError) -> Self {
        Self::Fep(value)
    }
}

impl From<V2TargetContractError> for V2RealSubjectAdapterError {
    fn from(value: V2TargetContractError) -> Self {
        Self::Target(value)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2RealTargetPredictionReceipt {
    prospective_ticket_commitment: [u8; 32],
    domain: V2ProspectiveTicketDomain,
    campaign_manifest_commitment: [u8; 32],
    subject_commitment: FrozenPredictionCommitment,
    adapter_source_commitment: [u8; 32],
    prediction: ConsequencePrediction,
    commitment: [u8; 32],
}

impl V2RealTargetPredictionReceipt {
    pub(super) const fn prospective_ticket_commitment(&self) -> [u8; 32] {
        self.prospective_ticket_commitment
    }

    pub(super) const fn domain(&self) -> V2ProspectiveTicketDomain {
        self.domain
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn subject_commitment(&self) -> FrozenPredictionCommitment {
        self.subject_commitment
    }

    pub(super) const fn adapter_source_commitment(&self) -> [u8; 32] {
        self.adapter_source_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn into_prediction(self) -> ConsequencePrediction {
        self.prediction
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct V2RealComparatorPredictionReceipt {
    prospective_ticket_commitment: [u8; 32],
    domain: V2ProspectiveTicketDomain,
    campaign_manifest_commitment: [u8; 32],
    subject_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
    prediction: ConsequencePrediction,
    commitment: [u8; 32],
}

impl V2RealComparatorPredictionReceipt {
    pub(super) const fn prospective_ticket_commitment(&self) -> [u8; 32] {
        self.prospective_ticket_commitment
    }

    pub(super) const fn domain(&self) -> V2ProspectiveTicketDomain {
        self.domain
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn subject_commitment(&self) -> [u8; 32] {
        self.subject_commitment
    }

    pub(super) const fn adapter_source_commitment(&self) -> [u8; 32] {
        self.adapter_source_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn into_prediction(self) -> ConsequencePrediction {
        self.prediction
    }
}

pub(super) struct V2RealTargetAdapter<'a> {
    subject: &'a FepHeldOutSubject,
    campaign_manifest_commitment: [u8; 32],
    expected_subject_commitment: FrozenPredictionCommitment,
}

impl<'a> V2RealTargetAdapter<'a> {
    fn bind(
        subject: &'a FepHeldOutSubject,
        manifest: V2PreHeldOutManifest,
    ) -> Result<Self, V2RealSubjectAdapterError> {
        if subject.commitment().as_bytes() != &manifest.learned_subject_commitment() {
            return Err(V2RealSubjectAdapterError::TargetSubjectCommitmentMismatch);
        }
        Ok(Self {
            subject,
            campaign_manifest_commitment: manifest.commitment(),
            expected_subject_commitment: subject.commitment(),
        })
    }

    pub(super) fn predict_ticket(
        &self,
        ticket: V2ProspectiveTicket,
    ) -> Result<V2RealTargetPredictionReceipt, V2RealSubjectAdapterError> {
        if ticket.campaign_manifest_commitment() != self.campaign_manifest_commitment {
            return Err(V2RealSubjectAdapterError::TicketCampaignMismatch);
        }
        if self.subject.commitment() != self.expected_subject_commitment {
            return Err(V2RealSubjectAdapterError::TargetSubjectCommitmentMismatch);
        }
        let observation = V2FepAdapter::encode_state(ticket.pre());
        let action = V2FepAdapter::encode_action(ticket.action())?;
        let outcome = self.subject.trial().predict_once(
            &observation,
            1.0,
            ticket.domain().fep_modality(),
            action,
        )?;
        let prediction = V2FepAdapter::prediction_from_outcome(ticket.action(), &outcome)?;
        if prediction.action != ticket.action() {
            return Err(V2RealSubjectAdapterError::TargetActionMismatch);
        }
        let adapter_source_commitment = real_subject_adapter_source_commitment();
        let commitment = prediction_receipt_commitment(
            V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
            ticket,
            self.expected_subject_commitment.as_bytes(),
            adapter_source_commitment,
            &prediction,
        );
        Ok(V2RealTargetPredictionReceipt {
            prospective_ticket_commitment: ticket.commitment(),
            domain: ticket.domain(),
            campaign_manifest_commitment: ticket.campaign_manifest_commitment(),
            subject_commitment: self.expected_subject_commitment,
            adapter_source_commitment,
            prediction,
            commitment,
        })
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn subject_commitment(&self) -> FrozenPredictionCommitment {
        self.expected_subject_commitment
    }
}

pub(super) struct V2RealComparatorAdapter<'a> {
    subject: &'a V2SelectedComparatorSubject,
    campaign_manifest_commitment: [u8; 32],
    expected_subject_commitment: [u8; 32],
}

impl<'a> V2RealComparatorAdapter<'a> {
    fn bind(
        subject: &'a V2SelectedComparatorSubject,
        manifest: V2PreHeldOutManifest,
    ) -> Result<Self, V2RealSubjectAdapterError> {
        if subject.commitment() != manifest.selected_comparator_commitment() {
            return Err(V2RealSubjectAdapterError::ComparatorSubjectCommitmentMismatch);
        }
        Ok(Self {
            subject,
            campaign_manifest_commitment: manifest.commitment(),
            expected_subject_commitment: subject.commitment(),
        })
    }

    pub(super) fn predict_ticket(
        &self,
        ticket: V2ProspectiveTicket,
    ) -> Result<V2RealComparatorPredictionReceipt, V2RealSubjectAdapterError> {
        if ticket.campaign_manifest_commitment() != self.campaign_manifest_commitment {
            return Err(V2RealSubjectAdapterError::TicketCampaignMismatch);
        }
        if self.subject.commitment() != self.expected_subject_commitment {
            return Err(V2RealSubjectAdapterError::ComparatorSubjectCommitmentMismatch);
        }
        let prediction = self
            .subject
            .predict(ticket.family(), ticket.pre(), ticket.action());
        if prediction.action != ticket.action() {
            return Err(V2RealSubjectAdapterError::ComparatorActionMismatch);
        }
        let adapter_source_commitment = real_subject_adapter_source_commitment();
        let commitment = prediction_receipt_commitment(
            V2_REAL_COMPARATOR_PREDICTION_RECEIPT_REVISION,
            ticket,
            &self.expected_subject_commitment,
            adapter_source_commitment,
            &prediction,
        );
        Ok(V2RealComparatorPredictionReceipt {
            prospective_ticket_commitment: ticket.commitment(),
            domain: ticket.domain(),
            campaign_manifest_commitment: ticket.campaign_manifest_commitment(),
            subject_commitment: self.expected_subject_commitment,
            adapter_source_commitment,
            prediction,
            commitment,
        })
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn subject_commitment(&self) -> [u8; 32] {
        self.expected_subject_commitment
    }
}

pub(super) struct V2RealSubjectAdapterPair<'a> {
    target: V2RealTargetAdapter<'a>,
    comparator: V2RealComparatorAdapter<'a>,
    adapter_source_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl<'a> V2RealSubjectAdapterPair<'a> {
    pub(super) fn bind_campaign(
        campaign: &'a V2PreHeldOutCampaignCapability,
    ) -> Result<Self, V2RealSubjectAdapterError> {
        Self::bind(
            campaign.target_subject(),
            campaign.selected_comparator(),
            campaign.manifest(),
        )
    }

    fn bind(
        target_subject: &'a FepHeldOutSubject,
        comparator_subject: &'a V2SelectedComparatorSubject,
        manifest: V2PreHeldOutManifest,
    ) -> Result<Self, V2RealSubjectAdapterError> {
        let target = V2RealTargetAdapter::bind(target_subject, manifest)?;
        let comparator = V2RealComparatorAdapter::bind(comparator_subject, manifest)?;
        let adapter_source_commitment = real_subject_adapter_source_commitment();
        let commitment = real_subject_adapter_pair_commitment(
            manifest.commitment(),
            target.subject_commitment(),
            comparator.subject_commitment(),
            adapter_source_commitment,
        );
        Ok(Self {
            target,
            comparator,
            adapter_source_commitment,
            commitment,
        })
    }

    pub(super) fn target(&self) -> &V2RealTargetAdapter<'a> {
        &self.target
    }

    pub(super) fn comparator(&self) -> &V2RealComparatorAdapter<'a> {
        &self.comparator
    }

    pub(super) const fn adapter_source_commitment(&self) -> [u8; 32] {
        self.adapter_source_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

pub(super) fn real_subject_adapter_source_commitment() -> [u8; 32] {
    let whole_source = include_str!("v2_real_subject_adapters.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("real-subject adapter source has a production section");
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_REAL_ADAPTER_SOURCE_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, production_source.as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn real_subject_adapter_pair_commitment(
    campaign_manifest_commitment: [u8; 32],
    target_subject_commitment: FrozenPredictionCommitment,
    comparator_subject_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_REAL_ADAPTER_PAIR_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_REAL_TARGET_ADAPTER_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_REAL_COMPARATOR_ADAPTER_REVISION.as_bytes());
    bytes.extend_from_slice(&campaign_manifest_commitment);
    bytes.extend_from_slice(target_subject_commitment.as_bytes());
    bytes.extend_from_slice(&comparator_subject_commitment);
    bytes.extend_from_slice(&adapter_source_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn prediction_receipt_commitment(
    revision: &str,
    ticket: V2ProspectiveTicket,
    subject_commitment: &[u8; 32],
    adapter_source_commitment: [u8; 32],
    prediction: &ConsequencePrediction,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, revision.as_bytes());
    bytes.extend_from_slice(&ticket.commitment());
    encode_bytes(&mut bytes, ticket.domain().fep_modality().as_bytes());
    bytes.extend_from_slice(&ticket.campaign_manifest_commitment());
    bytes.extend_from_slice(subject_commitment);
    bytes.extend_from_slice(&adapter_source_commitment);
    encode_prediction(&mut bytes, prediction);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_prediction(bytes: &mut Vec<u8>, prediction: &ConsequencePrediction) {
    let action = action_index(prediction.action)
        .expect("real V2 prediction action must belong to the canonical action vocabulary");
    bytes.extend_from_slice(&(action as u64).to_le_bytes());
    match &prediction.outcome {
        PredictionOutcome::Predicted { fields } => {
            bytes.push(1);
            bytes.extend_from_slice(&(fields.len() as u64).to_le_bytes());
            for field in fields {
                encode_public_value(bytes, *field);
            }
        }
        PredictionOutcome::AbstainInsufficientEvidence => bytes.push(2),
        PredictionOutcome::OutOfQualifiedDomain => bytes.push(3),
    }
}

fn encode_public_value(bytes: &mut Vec<u8>, value: PublicValue) {
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
    use super::super::v2_development_order::materialize_development_plan;

    const MANIFEST: [u8; 32] = [0x81; 32];
    const SUBJECT_A: [u8; 32] = [0x91; 32];
    const SUBJECT_B: [u8; 32] = [0x92; 32];
    const ADAPTER_A: [u8; 32] = [0xA1; 32];
    const ADAPTER_B: [u8; 32] = [0xA2; 32];

    fn sample_ticket(domain: V2ProspectiveTicketDomain) -> V2ProspectiveTicket {
        let plan = materialize_development_plan().unwrap();
        let row = plan.ordered_rows()[0];
        V2ProspectiveTicket::new(
            domain,
            MANIFEST,
            0,
            row.row_identity(),
            row.family(),
            row.pre(),
            row.action(),
        )
        .unwrap()
    }

    fn sample_prediction(ticket: V2ProspectiveTicket) -> ConsequencePrediction {
        ConsequencePrediction {
            action: ticket.action(),
            outcome: PredictionOutcome::Predicted {
                fields: ticket
                    .pre()
                    .fields()
                    .into_iter()
                    .map(PublicValue::Count)
                    .collect(),
            },
        }
    }

    #[test]
    fn receipt_identity_binds_ticket_domain_subject_adapter_and_prediction() {
        let heldout = sample_ticket(V2ProspectiveTicketDomain::HeldOutEvaluation);
        let canary = sample_ticket(V2ProspectiveTicketDomain::DevelopmentPlumbingCanary);
        let prediction = sample_prediction(heldout);
        let canonical = prediction_receipt_commitment(
            V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
            heldout,
            &SUBJECT_A,
            ADAPTER_A,
            &prediction,
        );
        assert_ne!(
            canonical,
            prediction_receipt_commitment(
                V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
                canary,
                &SUBJECT_A,
                ADAPTER_A,
                &prediction,
            )
        );
        assert_ne!(
            canonical,
            prediction_receipt_commitment(
                V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
                heldout,
                &SUBJECT_B,
                ADAPTER_A,
                &prediction,
            )
        );
        assert_ne!(
            canonical,
            prediction_receipt_commitment(
                V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
                heldout,
                &SUBJECT_A,
                ADAPTER_B,
                &prediction,
            )
        );
        let changed_prediction = ConsequencePrediction {
            action: heldout.action(),
            outcome: PredictionOutcome::AbstainInsufficientEvidence,
        };
        assert_ne!(
            canonical,
            prediction_receipt_commitment(
                V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
                heldout,
                &SUBJECT_A,
                ADAPTER_A,
                &changed_prediction,
            )
        );
    }

    #[test]
    fn target_and_comparator_receipt_roles_are_commitment_separated() {
        let ticket = sample_ticket(V2ProspectiveTicketDomain::HeldOutEvaluation);
        let prediction = sample_prediction(ticket);
        assert_ne!(
            prediction_receipt_commitment(
                V2_REAL_TARGET_PREDICTION_RECEIPT_REVISION,
                ticket,
                &SUBJECT_A,
                ADAPTER_A,
                &prediction,
            ),
            prediction_receipt_commitment(
                V2_REAL_COMPARATOR_PREDICTION_RECEIPT_REVISION,
                ticket,
                &SUBJECT_A,
                ADAPTER_A,
                &prediction,
            )
        );
    }

    #[test]
    fn public_adapter_api_returns_role_receipts_not_raw_predictions() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("Result<V2RealTargetPredictionReceipt, V2RealSubjectAdapterError>"));
        assert!(source.contains(
            "Result<V2RealComparatorPredictionReceipt, V2RealSubjectAdapterError>"
        ));
        assert!(!source.contains("Result<ConsequencePrediction, V2RealSubjectAdapterError>"));
        assert!(source.contains("pub(super) fn into_prediction(self)"));
    }

    #[test]
    fn role_receipts_are_move_only_by_source_contract() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let target_start = source.find("struct V2RealTargetPredictionReceipt").unwrap();
        let comparator_start = source
            .find("struct V2RealComparatorPredictionReceipt")
            .unwrap();
        let target_prefix = &source[target_start.saturating_sub(80)..target_start];
        let comparator_prefix = &source[comparator_start.saturating_sub(80)..comparator_start];
        assert!(!target_prefix.contains("Clone"));
        assert!(!target_prefix.contains("Copy"));
        assert!(!comparator_prefix.contains("Clone"));
        assert!(!comparator_prefix.contains("Copy"));
    }

    #[test]
    fn receipt_encoding_uses_no_debug_or_display_strings() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(!source.contains("format!("));
        assert!(!source.contains("{:?}"));
        assert!(source.contains("action_index(prediction.action)"));
        assert!(source.contains("encode_public_value(bytes, *field)"));
        assert!(source.contains("ticket.domain().fep_modality().as_bytes()"));
    }

    #[test]
    fn source_defines_no_campaign_loop_reveal_scoring_or_learning_authority() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "for row",
            "for ticket",
            "0..128",
            ".reveal(",
            "score_consequence(",
            "materialize_heldout_plan",
            "learn_from_actual(",
            "freeze_for_evaluation(",
            "FepEvaluationSnapshot",
            "ActiveInferenceAgent",
        ] {
            assert!(
                !source.contains(forbidden),
                "adapter receipt layer must not gain execution authority: {forbidden}"
            );
        }
    }

    #[test]
    fn prediction_surface_is_partition_neutral_and_one_shot() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("V2ProspectiveTicket"));
        assert!(!source.contains("V2HeldOutTicket"));
        assert!(source.contains("ticket.domain().fep_modality()"));
        assert!(source.contains(".trial().predict_once("));
    }

    #[test]
    fn comparator_adapter_has_no_runtime_kind_selector() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(!source.contains("ShortcutBaselineKind"));
        assert!(!source.contains("kind:"));
        assert!(source.contains(".predict(ticket.family(), ticket.pre(), ticket.action())"));
    }

    #[test]
    fn public_binding_requires_whole_campaign_capability() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("pub(super) fn bind_campaign("));
        assert!(source.contains("campaign: &'a V2PreHeldOutCampaignCapability"));
        assert!(source.contains("campaign.target_subject()"));
        assert!(source.contains("campaign.selected_comparator()"));
        assert!(source.contains("campaign.manifest()"));
        assert_eq!(source.matches("pub(super) fn bind(").count(), 0);
    }

    #[test]
    fn exact_production_adapter_source_is_cryptographically_bound() {
        let canonical = real_subject_adapter_source_commitment();
        let whole_source = include_str!("v2_real_subject_adapters.rs");
        let production_source = whole_source
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let mut changed = production_source.as_bytes().to_vec();
        changed.push(b'\n');
        let mut bytes = Vec::new();
        encode_bytes(
            &mut bytes,
            V2_REAL_ADAPTER_SOURCE_COMMITMENT_REVISION.as_bytes(),
        );
        encode_bytes(&mut bytes, &changed);
        let changed_commitment = *blake3::hash(&bytes).as_bytes();
        assert_ne!(canonical, [0_u8; 32]);
        assert_ne!(canonical, changed_commitment);
    }
}
