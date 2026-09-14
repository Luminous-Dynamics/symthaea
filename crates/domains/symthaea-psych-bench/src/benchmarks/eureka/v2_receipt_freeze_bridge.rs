// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provenance-preserving bridge from real adapter prediction receipts into the
//! qualified EUREKA-002 V2 HeldOut freeze state machine.
//!
//! This bridge owns no reveal or scoring authority. It constructs the existing
//! HeldOut protocol from the exact typed manifest, mints/proves one canonical
//! pre-outcome ticket, validates both role-specific real-prediction receipts,
//! consumes those receipts, and returns only paired frozen predictions.

#![allow(dead_code)]

use super::v2_heldout_reveal_protocol::{
    V2HeldOutRevealError, V2HeldOutRevealProtocol, V2HeldOutTicket,
    V2PairedFrozenPredictions,
};
use super::v2_preheldout_custody::V2PreHeldOutManifest;
use super::v2_prospective_ticket::{
    V2ProspectiveTicket, V2ProspectiveTicketDomain, V2ProspectiveTicketError,
};
use super::v2_real_subject_adapters::{
    V2RealComparatorPredictionReceipt, V2RealTargetPredictionReceipt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ReceiptFreezeBridgeError {
    RevealProtocol(V2HeldOutRevealError),
    ProspectiveTicket(V2ProspectiveTicketError),
    HeldOutTicketMismatch,
    ProspectiveTicketMismatch,
    ProspectiveDomainMismatch,
    TargetTicketMismatch,
    ComparatorTicketMismatch,
    TargetCampaignMismatch,
    ComparatorCampaignMismatch,
    TargetSubjectMismatch,
    ComparatorSubjectMismatch,
    AdapterSourceMismatch,
}

impl From<V2HeldOutRevealError> for V2ReceiptFreezeBridgeError {
    fn from(value: V2HeldOutRevealError) -> Self {
        Self::RevealProtocol(value)
    }
}

impl From<V2ProspectiveTicketError> for V2ReceiptFreezeBridgeError {
    fn from(value: V2ProspectiveTicketError) -> Self {
        Self::ProspectiveTicket(value)
    }
}

/// Manifest-bound HeldOut freeze authority. The embedded protocol is private;
/// this type deliberately exposes no reveal or scoring method.
#[derive(Debug)]
pub(super) struct V2ReceiptFreezeBridge {
    protocol: V2HeldOutRevealProtocol,
    manifest: V2PreHeldOutManifest,
}

impl V2ReceiptFreezeBridge {
    pub(super) fn canonical(
        manifest: V2PreHeldOutManifest,
    ) -> Result<Self, V2ReceiptFreezeBridgeError> {
        let protocol = V2HeldOutRevealProtocol::synthetic_canonical(
            manifest.commitment(),
            manifest.learned_subject_commitment(),
            manifest.selected_comparator_commitment(),
        )?;
        Ok(Self { protocol, manifest })
    }

    /// Mint the evaluator-owned HeldOut ticket and its subject-facing projection
    /// from the same canonical row before any prediction is produced.
    pub(super) fn ticket_pair(
        &self,
        row_index: usize,
    ) -> Result<(V2HeldOutTicket, V2ProspectiveTicket), V2ReceiptFreezeBridgeError> {
        let heldout = self.protocol.ticket(row_index)?;
        let prospective = self.project_heldout_ticket(heldout)?;
        Ok((heldout, prospective))
    }

    /// Admit two real-subject receipts into the existing V2J role-freeze state
    /// machine. All metadata checks happen before either receipt is consumed.
    pub(super) fn freeze_receipts(
        &self,
        heldout: V2HeldOutTicket,
        prospective: V2ProspectiveTicket,
        target: V2RealTargetPredictionReceipt,
        comparator: V2RealComparatorPredictionReceipt,
    ) -> Result<V2PairedFrozenPredictions, V2ReceiptFreezeBridgeError> {
        let canonical_heldout = self.protocol.ticket(usize::from(heldout.row_index()))?;
        if heldout != canonical_heldout {
            return Err(V2ReceiptFreezeBridgeError::HeldOutTicketMismatch);
        }

        let canonical_prospective = self.project_heldout_ticket(canonical_heldout)?;
        if prospective != canonical_prospective {
            return Err(V2ReceiptFreezeBridgeError::ProspectiveTicketMismatch);
        }
        if prospective.domain() != V2ProspectiveTicketDomain::HeldOutEvaluation {
            return Err(V2ReceiptFreezeBridgeError::ProspectiveDomainMismatch);
        }

        if target.prospective_ticket_commitment() != prospective.commitment() {
            return Err(V2ReceiptFreezeBridgeError::TargetTicketMismatch);
        }
        if comparator.prospective_ticket_commitment() != prospective.commitment() {
            return Err(V2ReceiptFreezeBridgeError::ComparatorTicketMismatch);
        }
        if target.domain() != V2ProspectiveTicketDomain::HeldOutEvaluation {
            return Err(V2ReceiptFreezeBridgeError::ProspectiveDomainMismatch);
        }
        if comparator.domain() != V2ProspectiveTicketDomain::HeldOutEvaluation {
            return Err(V2ReceiptFreezeBridgeError::ProspectiveDomainMismatch);
        }
        if target.campaign_manifest_commitment() != self.manifest.commitment() {
            return Err(V2ReceiptFreezeBridgeError::TargetCampaignMismatch);
        }
        if comparator.campaign_manifest_commitment() != self.manifest.commitment() {
            return Err(V2ReceiptFreezeBridgeError::ComparatorCampaignMismatch);
        }
        if target.subject_commitment().as_bytes() != &self.manifest.learned_subject_commitment() {
            return Err(V2ReceiptFreezeBridgeError::TargetSubjectMismatch);
        }
        if comparator.subject_commitment() != self.manifest.selected_comparator_commitment() {
            return Err(V2ReceiptFreezeBridgeError::ComparatorSubjectMismatch);
        }
        if target.adapter_source_commitment() != comparator.adapter_source_commitment() {
            return Err(V2ReceiptFreezeBridgeError::AdapterSourceMismatch);
        }

        // Prediction bytes become accessible only after every provenance check
        // above has succeeded. Both role receipts are consumed exactly once.
        let target_prediction = target.into_prediction();
        let comparator_prediction = comparator.into_prediction();
        let frozen_target = self
            .protocol
            .freeze_target(canonical_heldout, &target_prediction)?;
        let frozen_comparator = self
            .protocol
            .freeze_comparator(canonical_heldout, &comparator_prediction)?;
        Ok(self
            .protocol
            .pair(canonical_heldout, frozen_target, frozen_comparator)?)
    }

    fn project_heldout_ticket(
        &self,
        heldout: V2HeldOutTicket,
    ) -> Result<V2ProspectiveTicket, V2ReceiptFreezeBridgeError> {
        if heldout.campaign_manifest_commitment() != self.manifest.commitment() {
            return Err(V2ReceiptFreezeBridgeError::HeldOutTicketMismatch);
        }
        Ok(V2ProspectiveTicket::new(
            V2ProspectiveTicketDomain::HeldOutEvaluation,
            heldout.campaign_manifest_commitment(),
            heldout.row_index(),
            heldout.row_identity(),
            heldout.family(),
            heldout.pre(),
            heldout.action(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_source_has_no_reveal_scoring_or_learning_authority() {
        let source = include_str!("v2_receipt_freeze_bridge.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            ".reveal(",
            ".score(",
            "score_consequence(",
            "learn_from_actual(",
            "freeze_for_evaluation(",
            "FepPredictionSession",
            "ActiveInferenceAgent",
        ] {
            assert!(
                !source.contains(forbidden),
                "receipt-freeze bridge must not gain outcome/training authority: {forbidden}"
            );
        }
    }

    #[test]
    fn projection_is_explicitly_heldout_and_pre_outcome_only() {
        let source = include_str!("v2_receipt_freeze_bridge.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("fn project_heldout_ticket(").unwrap();
        let body = &source[start..];
        assert!(body.contains("V2ProspectiveTicketDomain::HeldOutEvaluation"));
        assert!(body.contains("heldout.pre()"));
        assert!(body.contains("heldout.action()"));
        assert!(!body.contains("heldout.post()"));
    }

    #[test]
    fn receipt_consumption_occurs_after_all_metadata_checks() {
        let source = include_str!("v2_receipt_freeze_bridge.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let target_consume = source.find("target.into_prediction()").unwrap();
        let comparator_consume = source.find("comparator.into_prediction()").unwrap();
        for required in [
            "ProspectiveTicketMismatch",
            "TargetTicketMismatch",
            "ComparatorTicketMismatch",
            "TargetCampaignMismatch",
            "ComparatorCampaignMismatch",
            "TargetSubjectMismatch",
            "ComparatorSubjectMismatch",
            "AdapterSourceMismatch",
        ] {
            let check = source.find(required).unwrap();
            assert!(check < target_consume, "metadata check must precede target receipt consumption: {required}");
            assert!(check < comparator_consume, "metadata check must precede comparator receipt consumption: {required}");
        }
    }

    #[test]
    fn protocol_identity_is_constructed_from_typed_manifest() {
        let source = include_str!("v2_receipt_freeze_bridge.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("manifest.commitment()"));
        assert!(source.contains("manifest.learned_subject_commitment()"));
        assert!(source.contains("manifest.selected_comparator_commitment()"));
        assert!(source.contains("V2HeldOutRevealProtocol::synthetic_canonical("));
    }
}
