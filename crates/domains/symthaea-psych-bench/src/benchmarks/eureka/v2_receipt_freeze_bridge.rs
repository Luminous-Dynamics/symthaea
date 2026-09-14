// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provenance-preserving bridge from real adapter prediction receipts into the
//! qualified EUREKA-002 V2 HeldOut freeze state machine.
//!
//! This bridge owns no reveal or scoring authority. It constructs the existing
//! HeldOut protocol from the exact typed manifest, mints/proves one canonical
//! pre-outcome ticket, validates both role-specific real-prediction receipts,
//! consumes those receipts, and returns only a move-only real-subject
//! provenance seal around the generic paired freeze.

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

pub(super) const V2_REAL_HELDOUT_PAIRED_FREEZE_REVISION: &str =
    "EUREKA.002.V2.REAL_HELDOUT_PAIRED_FREEZE.v1";
pub(super) const V2_RECEIPT_FREEZE_BRIDGE_SOURCE_REVISION: &str =
    "EUREKA.002.V2.RECEIPT_FREEZE_BRIDGE_SOURCE.v1";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct V2RealHeldOutPairLineage {
    campaign_manifest_commitment: [u8; 32],
    heldout_ticket_commitment: [u8; 32],
    prospective_ticket_commitment: [u8; 32],
    target_subject_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    target_receipt_commitment: [u8; 32],
    comparator_receipt_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
    generic_paired_freeze_commitment: [u8; 32],
    bridge_source_commitment: [u8; 32],
}

/// Move-only proof that the generic HeldOut paired freeze was constructed from
/// the two exact real-subject prediction receipts admitted by the typed V2
/// manifest. The generic pair is deliberately private and this tranche exposes
/// no unwrap, reveal, or score path.
#[derive(Debug)]
pub(super) struct V2RealHeldOutPairedFreeze {
    paired: V2PairedFrozenPredictions,
    lineage: V2RealHeldOutPairLineage,
    commitment: [u8; 32],
}

impl V2RealHeldOutPairedFreeze {
    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.lineage.campaign_manifest_commitment
    }

    pub(super) const fn prospective_ticket_commitment(&self) -> [u8; 32] {
        self.lineage.prospective_ticket_commitment
    }

    pub(super) const fn target_receipt_commitment(&self) -> [u8; 32] {
        self.lineage.target_receipt_commitment
    }

    pub(super) const fn comparator_receipt_commitment(&self) -> [u8; 32] {
        self.lineage.comparator_receipt_commitment
    }

    pub(super) const fn adapter_source_commitment(&self) -> [u8; 32] {
        self.lineage.adapter_source_commitment
    }

    pub(super) const fn bridge_source_commitment(&self) -> [u8; 32] {
        self.lineage.bridge_source_commitment
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
    /// The generic pair never escapes this bridge: callers receive only a
    /// provenance-sealed real-subject pair with no reveal authority.
    pub(super) fn freeze_receipts(
        &self,
        heldout: V2HeldOutTicket,
        prospective: V2ProspectiveTicket,
        target: V2RealTargetPredictionReceipt,
        comparator: V2RealComparatorPredictionReceipt,
    ) -> Result<V2RealHeldOutPairedFreeze, V2ReceiptFreezeBridgeError> {
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

        // Capture every real-receipt lineage value before either move-only
        // receipt is consumed into loose prediction bytes.
        let target_receipt_commitment = target.commitment();
        let comparator_receipt_commitment = comparator.commitment();
        let target_subject_commitment = *target.subject_commitment().as_bytes();
        let comparator_subject_commitment = comparator.subject_commitment();
        let adapter_source_commitment = target.adapter_source_commitment();

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
        let paired = self
            .protocol
            .pair(canonical_heldout, frozen_target, frozen_comparator)?;

        let lineage = V2RealHeldOutPairLineage {
            campaign_manifest_commitment: self.manifest.commitment(),
            heldout_ticket_commitment: canonical_heldout.commitment(),
            prospective_ticket_commitment: prospective.commitment(),
            target_subject_commitment,
            comparator_subject_commitment,
            target_receipt_commitment,
            comparator_receipt_commitment,
            adapter_source_commitment,
            generic_paired_freeze_commitment: paired.commitment(),
            bridge_source_commitment: receipt_freeze_bridge_source_commitment(),
        };
        let commitment = real_heldout_paired_freeze_commitment(lineage);
        Ok(V2RealHeldOutPairedFreeze {
            paired,
            lineage,
            commitment,
        })
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

pub(super) fn receipt_freeze_bridge_source_commitment() -> [u8; 32] {
    let whole_source = include_str!("v2_receipt_freeze_bridge.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("receipt-freeze bridge source has production section");
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_RECEIPT_FREEZE_BRIDGE_SOURCE_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, production_source.as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn real_heldout_paired_freeze_commitment(lineage: V2RealHeldOutPairLineage) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_REAL_HELDOUT_PAIRED_FREEZE_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&lineage.campaign_manifest_commitment);
    bytes.extend_from_slice(&lineage.heldout_ticket_commitment);
    bytes.extend_from_slice(&lineage.prospective_ticket_commitment);
    bytes.extend_from_slice(&lineage.target_subject_commitment);
    bytes.extend_from_slice(&lineage.comparator_subject_commitment);
    bytes.extend_from_slice(&lineage.target_receipt_commitment);
    bytes.extend_from_slice(&lineage.comparator_receipt_commitment);
    bytes.extend_from_slice(&lineage.adapter_source_commitment);
    bytes.extend_from_slice(&lineage.generic_paired_freeze_commitment);
    bytes.extend_from_slice(&lineage.bridge_source_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lineage() -> V2RealHeldOutPairLineage {
        V2RealHeldOutPairLineage {
            campaign_manifest_commitment: [1; 32],
            heldout_ticket_commitment: [2; 32],
            prospective_ticket_commitment: [3; 32],
            target_subject_commitment: [4; 32],
            comparator_subject_commitment: [5; 32],
            target_receipt_commitment: [6; 32],
            comparator_receipt_commitment: [7; 32],
            adapter_source_commitment: [8; 32],
            generic_paired_freeze_commitment: [9; 32],
            bridge_source_commitment: [10; 32],
        }
    }

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
    fn receipt_consumption_occurs_after_all_metadata_checks_and_lineage_capture() {
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
            "let target_receipt_commitment = target.commitment()",
            "let comparator_receipt_commitment = comparator.commitment()",
        ] {
            let check = source.find(required).unwrap();
            assert!(check < target_consume, "metadata/lineage capture must precede target receipt consumption: {required}");
            assert!(check < comparator_consume, "metadata/lineage capture must precede comparator receipt consumption: {required}");
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

    #[test]
    fn real_bridge_returns_sealed_real_pair_not_generic_pair() {
        let source = include_str!("v2_receipt_freeze_bridge.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("pub(super) fn freeze_receipts(").unwrap();
        let body = &source[start..];
        assert!(body.contains("Result<V2RealHeldOutPairedFreeze"));
    }

    #[test]
    fn real_pair_seal_is_move_only_and_has_no_escape_hatch() {
        let source = include_str!("v2_receipt_freeze_bridge.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("struct V2RealHeldOutPairedFreeze").unwrap();
        let prefix = &source[start.saturating_sub(96)..start];
        assert!(!prefix.contains("Clone"));
        assert!(!prefix.contains("Copy"));

        let impl_start = source
            .find("impl V2RealHeldOutPairedFreeze")
            .unwrap();
        let impl_tail = &source[impl_start..];
        let impl_end = impl_tail
            .find("\n}\n\n/// Manifest-bound HeldOut freeze authority")
            .unwrap();
        let api = &impl_tail[..impl_end];
        for forbidden in ["reveal", "score", "into_inner", "unwrap", "paired("] {
            assert!(
                !api.contains(forbidden),
                "real pair seal must not expose generic/reveal authority: {forbidden}"
            );
        }
    }

    #[test]
    fn every_real_pair_lineage_dimension_changes_seal_identity() {
        let canonical = sample_lineage();
        let expected = real_heldout_paired_freeze_commitment(canonical);
        assert_ne!(expected, [0_u8; 32]);

        for slot in 0..10 {
            let mut changed = canonical;
            match slot {
                0 => changed.campaign_manifest_commitment[0] ^= 1,
                1 => changed.heldout_ticket_commitment[0] ^= 1,
                2 => changed.prospective_ticket_commitment[0] ^= 1,
                3 => changed.target_subject_commitment[0] ^= 1,
                4 => changed.comparator_subject_commitment[0] ^= 1,
                5 => changed.target_receipt_commitment[0] ^= 1,
                6 => changed.comparator_receipt_commitment[0] ^= 1,
                7 => changed.adapter_source_commitment[0] ^= 1,
                8 => changed.generic_paired_freeze_commitment[0] ^= 1,
                9 => changed.bridge_source_commitment[0] ^= 1,
                _ => unreachable!(),
            }
            assert_ne!(expected, real_heldout_paired_freeze_commitment(changed));
        }
    }

    #[test]
    fn bridge_source_commitment_is_nonzero_and_source_sensitive() {
        let canonical = receipt_freeze_bridge_source_commitment();
        assert_ne!(canonical, [0_u8; 32]);

        let whole_source = include_str!("v2_receipt_freeze_bridge.rs");
        let production_source = whole_source
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let mut changed = production_source.as_bytes().to_vec();
        changed.push(b'\n');
        let mut bytes = Vec::new();
        encode_bytes(
            &mut bytes,
            V2_RECEIPT_FREEZE_BRIDGE_SOURCE_REVISION.as_bytes(),
        );
        encode_bytes(&mut bytes, &changed);
        assert_ne!(canonical, *blake3::hash(&bytes).as_bytes());
    }
}
