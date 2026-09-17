// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Immutable audit receipt for a fully validated EKM restart-v2 wire snapshot.
//!
//! This module deliberately stops before wire-to-quarantine construction. A
//! receipt proves only that the current read-only validators accepted one exact
//! decoded snapshot and binds their reports to the snapshot's integrity anchors.
//! It is not a signature, authorization, trusted timestamp, quarantine token, or
//! activation capability.

use super::belief_revision_schema_wire_validation::{
    BeliefRevisionSchemaWireValidationError, BeliefRevisionSchemaWireValidationReport,
    BeliefRevisionSchemaWireValidator,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use super::epistemic_restart_wire_v2_validation::{
    EpistemicRestartWireV2ValidationError, EpistemicRestartWireV2ValidationReport,
    EpistemicRestartWireV2Validator,
};
use super::epistemic_restart_wire_validation::{
    EpistemicRestartWireValidationError, EpistemicRestartWireValidationReport,
    EpistemicRestartWireValidator,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartValidationReceiptVersion {
    V1,
}

/// Explicitly describes what EKM-044 can and cannot establish about the legacy
/// V1 manifest digest carried inside the V2 bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyManifestAssuranceV1 {
    /// The V2 typed-schema digest is independently canonical and is correctly
    /// bound to the embedded V1 manifest digest, but that historical V1 digest is
    /// not independently re-derived from untrusted wire bytes in this tranche.
    EmbeddedDigestAnchorOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartValidationAuthorityV1 {
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
}

impl RestartValidationAuthorityV1 {
    fn none() -> Self {
        Self {
            quarantine_construction_authorized: false,
            activation_authorized: false,
        }
    }

    pub fn quarantine_construction_authorized(self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(self) -> bool {
        self.activation_authorized
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicRestartValidationReceiptDigest([u8; 32]);

impl EpistemicRestartValidationReceiptDigest {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

/// Read-only audit artifact proving that all currently available restart-v2
/// validators accepted one exact decoded snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicRestartValidationReceiptV1 {
    version: EpistemicRestartValidationReceiptVersion,
    captured_at_cycle: u64,
    outer_checksum: [u8; 32],
    base_wire_checksum: [u8; 32],
    schema_wire_checksum: [u8; 32],
    embedded_v1_manifest_digest: [u8; 32],
    claimed_v2_digest: [u8; 32],
    base_report: EpistemicRestartWireValidationReport,
    schema_report: BeliefRevisionSchemaWireValidationReport,
    bundle_report: EpistemicRestartWireV2ValidationReport,
    legacy_manifest_assurance: LegacyManifestAssuranceV1,
    authority: RestartValidationAuthorityV1,
    receipt_digest: EpistemicRestartValidationReceiptDigest,
}

impl EpistemicRestartValidationReceiptV1 {
    pub fn validate_and_capture(
        snapshot: &EpistemicRestartWireSnapshotV2,
    ) -> Result<Self, EpistemicRestartValidationReceiptError> {
        // Intentionally re-run all three validators independently. EKM-043 also
        // composes EKM-035/041, but storing their direct reports makes the receipt
        // self-describing and catches future accidental report-shape drift.
        let base_report = EpistemicRestartWireValidator::validate(&snapshot.base)
            .map_err(EpistemicRestartValidationReceiptError::BaseSemantic)?;
        let schema_report = BeliefRevisionSchemaWireValidator::validate(&snapshot.revision_schemas)
            .map_err(EpistemicRestartValidationReceiptError::SchemaSemantic)?;
        let bundle_report = EpistemicRestartWireV2Validator::validate(snapshot)
            .map_err(EpistemicRestartValidationReceiptError::BundleSemantic)?;

        if base_report.captured_at_cycle != schema_report.captured_at_cycle
            || base_report.captured_at_cycle != bundle_report.captured_at_cycle
        {
            return Err(EpistemicRestartValidationReceiptError::ReportEpochMismatch);
        }
        if base_report.revision_count != schema_report.record_count
            || base_report.revision_count != bundle_report.revision_count
        {
            return Err(EpistemicRestartValidationReceiptError::ReportRevisionCountMismatch);
        }
        if schema_report.eligible_count != bundle_report.eligible_revision_count
            || schema_report.rejected_count != bundle_report.rejected_revision_count
        {
            return Err(EpistemicRestartValidationReceiptError::ReportDecisionCountMismatch);
        }
        if !bundle_report.claimed_digest_binding_valid {
            return Err(EpistemicRestartValidationReceiptError::DigestBindingNotValidated);
        }

        let mut receipt = Self {
            version: EpistemicRestartValidationReceiptVersion::V1,
            captured_at_cycle: base_report.captured_at_cycle,
            outer_checksum: snapshot.outer_checksum,
            base_wire_checksum: snapshot.base.wire_checksum,
            schema_wire_checksum: snapshot.revision_schemas.wire_checksum,
            embedded_v1_manifest_digest: snapshot.base.manifest.manifest_digest,
            claimed_v2_digest: snapshot.claimed_v2_digest,
            base_report,
            schema_report,
            bundle_report,
            legacy_manifest_assurance: LegacyManifestAssuranceV1::EmbeddedDigestAnchorOnly,
            authority: RestartValidationAuthorityV1::none(),
            receipt_digest: EpistemicRestartValidationReceiptDigest([0; 32]),
        };
        receipt.receipt_digest = digest_receipt(&receipt)?;
        Ok(receipt)
    }

    /// Re-runs all validators against the supplied snapshot and requires an exact
    /// receipt match. This is an integrity/reproducibility check only.
    pub fn verify_against(
        &self,
        snapshot: &EpistemicRestartWireSnapshotV2,
    ) -> Result<(), EpistemicRestartValidationReceiptError> {
        let recomputed = Self::validate_and_capture(snapshot)?;
        if self != &recomputed {
            return Err(EpistemicRestartValidationReceiptError::ReceiptMismatch);
        }
        let digest = digest_receipt(self)?;
        if digest != self.receipt_digest {
            return Err(EpistemicRestartValidationReceiptError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn version(&self) -> EpistemicRestartValidationReceiptVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn outer_checksum(&self) -> [u8; 32] {
        self.outer_checksum
    }

    pub fn base_wire_checksum(&self) -> [u8; 32] {
        self.base_wire_checksum
    }

    pub fn schema_wire_checksum(&self) -> [u8; 32] {
        self.schema_wire_checksum
    }

    pub fn embedded_v1_manifest_digest(&self) -> [u8; 32] {
        self.embedded_v1_manifest_digest
    }

    pub fn claimed_v2_digest(&self) -> [u8; 32] {
        self.claimed_v2_digest
    }

    pub fn base_report(&self) -> &EpistemicRestartWireValidationReport {
        &self.base_report
    }

    pub fn schema_report(&self) -> &BeliefRevisionSchemaWireValidationReport {
        &self.schema_report
    }

    pub fn bundle_report(&self) -> &EpistemicRestartWireV2ValidationReport {
        &self.bundle_report
    }

    pub fn legacy_manifest_assurance(&self) -> LegacyManifestAssuranceV1 {
        self.legacy_manifest_assurance
    }

    pub fn authority(&self) -> RestartValidationAuthorityV1 {
        self.authority
    }

    pub fn receipt_digest(&self) -> EpistemicRestartValidationReceiptDigest {
        self.receipt_digest
    }
}

fn digest_receipt(
    receipt: &EpistemicRestartValidationReceiptV1,
) -> Result<EpistemicRestartValidationReceiptDigest, EpistemicRestartValidationReceiptError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-validation-receipt-v1");
    hasher.update(&[1]); // receipt version tag
    hasher.update(&receipt.captured_at_cycle.to_le_bytes());
    hasher.update(&receipt.outer_checksum);
    hasher.update(&receipt.base_wire_checksum);
    hasher.update(&receipt.schema_wire_checksum);
    hasher.update(&receipt.embedded_v1_manifest_digest);
    hasher.update(&receipt.claimed_v2_digest);

    hash_base_report(&mut hasher, &receipt.base_report)?;
    hash_schema_report(&mut hasher, &receipt.schema_report)?;
    hash_bundle_report(&mut hasher, &receipt.bundle_report)?;

    hasher.update(&[1]); // EmbeddedDigestAnchorOnly
    hasher.update(&[u8::from(receipt.authority.quarantine_construction_authorized)]);
    hasher.update(&[u8::from(receipt.authority.activation_authorized)]);
    Ok(EpistemicRestartValidationReceiptDigest(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_base_report(
    hasher: &mut blake3::Hasher,
    report: &EpistemicRestartWireValidationReport,
) -> Result<(), EpistemicRestartValidationReceiptError> {
    hasher.update(&report.captured_at_cycle.to_le_bytes());
    hash_usize(hasher, report.provenance_count)?;
    hash_usize(hasher, report.claim_count)?;
    hash_usize(hasher, report.evidence_count)?;
    hash_usize(hasher, report.support_state_count)?;
    hash_usize(hasher, report.mutation_count)?;
    hash_usize(hasher, report.revision_count)?;
    hash_usize(hasher, report.rejected_revision_count)?;
    Ok(())
}

fn hash_schema_report(
    hasher: &mut blake3::Hasher,
    report: &BeliefRevisionSchemaWireValidationReport,
) -> Result<(), EpistemicRestartValidationReceiptError> {
    hasher.update(&report.captured_at_cycle.to_le_bytes());
    hash_usize(hasher, report.record_count)?;
    hash_usize(hasher, report.eligible_count)?;
    hash_usize(hasher, report.rejected_count)?;
    hash_usize(hasher, report.failure_count)?;
    Ok(())
}

fn hash_bundle_report(
    hasher: &mut blake3::Hasher,
    report: &EpistemicRestartWireV2ValidationReport,
) -> Result<(), EpistemicRestartValidationReceiptError> {
    hasher.update(&report.captured_at_cycle.to_le_bytes());
    hash_usize(hasher, report.revision_count)?;
    hash_usize(hasher, report.eligible_revision_count)?;
    hash_usize(hasher, report.rejected_revision_count)?;
    hasher.update(&[u8::from(report.claimed_digest_binding_valid)]);
    Ok(())
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), EpistemicRestartValidationReceiptError> {
    let value = u64::try_from(value)
        .map_err(|_| EpistemicRestartValidationReceiptError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartValidationReceiptError {
    BaseSemantic(EpistemicRestartWireValidationError),
    SchemaSemantic(BeliefRevisionSchemaWireValidationError),
    BundleSemantic(EpistemicRestartWireV2ValidationError),
    ReportEpochMismatch,
    ReportRevisionCountMismatch,
    ReportDecisionCountMismatch,
    DigestBindingNotValidated,
    LengthOverflow,
    ReceiptMismatch,
    ReceiptDigestMismatch,
}

impl fmt::Display for EpistemicRestartValidationReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart validation receipt invalid: {self:?}")
    }
}

impl Error for EpistemicRestartValidationReceiptError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
        EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
        EpistemicRestartCapsuleV2, EpistemicRestartWireV2, EpistemicRevisionProposal,
        EpistemicSupportStore, EvidenceKind, EvidencePolarity,
    };

    fn snapshot() -> EpistemicRestartWireSnapshotV2 {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let inventory = EpistemicLedgerInventoryV1::new(
            vec![claim],
            vec![evidence],
            vec![provenance],
        )
        .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schema_history = BeliefRevisionSchemaHistoryV1::new();
        schema_history
            .evaluate_and_record(
                &mut receipts,
                &ledger,
                &proposal,
                &schema,
                None,
                None,
                3,
            )
            .unwrap();
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 4).unwrap();
        let revisions = BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, 4).unwrap();
        let schemas = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revisions,
            4,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(&ledger, &inventory, &mutations, &revisions, 4)
            .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        let bytes = EpistemicRestartWireV2::encode(&v2).unwrap();
        EpistemicRestartWireV2::decode(&bytes).unwrap()
    }

    #[test]
    fn valid_snapshot_produces_reproducible_non_authorizing_receipt() {
        let snapshot = snapshot();
        let receipt = EpistemicRestartValidationReceiptV1::validate_and_capture(&snapshot).unwrap();
        receipt.verify_against(&snapshot).unwrap();
        assert_eq!(
            receipt.legacy_manifest_assurance(),
            LegacyManifestAssuranceV1::EmbeddedDigestAnchorOnly
        );
        assert!(!receipt.authority().quarantine_construction_authorized());
        assert!(!receipt.authority().activation_authorized());
        assert_eq!(receipt.receipt_digest().to_hex().len(), 64);
    }

    #[test]
    fn receipt_binds_outer_checksum_even_when_semantics_are_unchanged() {
        let snapshot = snapshot();
        let receipt = EpistemicRestartValidationReceiptV1::validate_and_capture(&snapshot).unwrap();
        let mut changed = snapshot.clone();
        changed.outer_checksum[0] ^= 1;
        assert_eq!(
            receipt.verify_against(&changed).unwrap_err(),
            EpistemicRestartValidationReceiptError::ReceiptMismatch
        );
    }

    #[test]
    fn invalid_claimed_digest_never_gets_a_receipt() {
        let mut snapshot = snapshot();
        snapshot.claimed_v2_digest[0] ^= 1;
        assert!(matches!(
            EpistemicRestartValidationReceiptV1::validate_and_capture(&snapshot),
            Err(EpistemicRestartValidationReceiptError::BundleSemantic(_))
        ));
    }
}
