// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-only adapters from sealed real subjects into the qualified V2
//! prospective prediction protocol.
//!
//! This module deliberately defines no row loop and no evaluator reveal. It
//! proves only that already-sealed production subjects can produce canonical
//! `ConsequencePrediction` values from pre-outcome `V2ProspectiveTicket` values
//! while preserving exact campaign, subject, action, domain, and adapter-source
//! lineage.

#![allow(dead_code)]

use symthaea_fep::{FepHeldOutSubject, FepPredictionSessionError, FrozenPredictionCommitment};

use super::consequence::ConsequencePrediction;
use super::v2_preheldout_custody::{
    V2PreHeldOutCampaignCapability, V2PreHeldOutManifest,
};
use super::v2_prospective_ticket::V2ProspectiveTicket;
use super::v2_selected_comparator::V2SelectedComparatorSubject;
use super::v2_target_contract::{V2FepAdapter, V2TargetContractError};

pub(super) const V2_REAL_TARGET_ADAPTER_REVISION: &str =
    "EUREKA.002.V2.REAL_TARGET_ADAPTER.v2";
pub(super) const V2_REAL_COMPARATOR_ADAPTER_REVISION: &str =
    "EUREKA.002.V2.REAL_COMPARATOR_ADAPTER.v2";
pub(super) const V2_REAL_ADAPTER_SOURCE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.REAL_ADAPTER_SOURCE.v2";
pub(super) const V2_REAL_ADAPTER_PAIR_REVISION: &str =
    "EUREKA.002.V2.REAL_ADAPTER_PAIR.v2";

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

/// Narrow prediction adapter over one sealed FEP subject.
///
/// The adapter borrows the subject, so it cannot outlive campaign custody and
/// cannot recover any trainable snapshot/session from it. Construction is
/// private to the paired binding below.
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
    ) -> Result<ConsequencePrediction, V2RealSubjectAdapterError> {
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
        Ok(prediction)
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn subject_commitment(&self) -> FrozenPredictionCommitment {
        self.expected_subject_commitment
    }
}

/// Narrow prediction adapter over the single comparator selected on Calibration.
/// No comparator-kind selector is exposed.
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
    ) -> Result<ConsequencePrediction, V2RealSubjectAdapterError> {
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
        Ok(prediction)
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn subject_commitment(&self) -> [u8; 32] {
        self.expected_subject_commitment
    }
}

/// One typed binding of both real prediction-only subjects to one exact
/// non-Clone pre-HeldOut campaign capability and exact adapter semantics.
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

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_defines_no_campaign_loop_or_reveal_authority() {
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
            "V2HeldOutRevealProtocol::synthetic_canonical",
            "V2ShadowCampaignReport",
            "materialize_heldout_plan",
        ] {
            assert!(
                !source.contains(forbidden),
                "adapter layer must not become a campaign runner: {forbidden}"
            );
        }
    }

    #[test]
    fn prediction_surface_is_partition_neutral() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("V2ProspectiveTicket"));
        assert!(!source.contains("V2HeldOutTicket"));
        assert!(source.contains("ticket.domain().fep_modality()"));
    }

    #[test]
    fn target_adapter_has_only_one_shot_prediction_reachability() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains(".trial()"));
        assert!(source.contains(".predict_once("));
        for forbidden in [
            "FepPredictionSession::",
            "FepEvaluationSnapshot",
            "learn_from_actual(",
            "freeze_for_evaluation(",
            "ActiveInferenceAgent",
        ] {
            assert!(
                !source.contains(forbidden),
                "real target adapter must not reach trainable FEP authority: {forbidden}"
            );
        }
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
    fn both_adapters_bind_typed_manifest_not_raw_expected_ids() {
        let source = include_str!("v2_real_subject_adapters.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("manifest.learned_subject_commitment()"));
        assert!(source.contains("manifest.selected_comparator_commitment()"));
        assert!(source.matches("TicketCampaignMismatch").count() >= 3);
        assert!(source.contains("V2RealSubjectAdapterPair"));
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
