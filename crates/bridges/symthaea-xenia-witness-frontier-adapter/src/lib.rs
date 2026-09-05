// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-only bridge from verified Xenia witness currentness into the
//! transport-neutral qualification-witness recovery model.
//!
//! This crate deliberately performs no cryptographic verification, freshness
//! verification, durable storage, anchoring, publication, or authority
//! admission. Those semantics remain in `symthaea-xenia-authority` and
//! `symthaea-qualification-witness-frontier` respectively.
//!
//! The only accepted source object is `VerifiedXeniaWitnessFrontierV1`. Its
//! already-verified fields are projected into `ExternalWitnessFrontierClaimV1`,
//! then the generic recovery crate independently re-validates the structural
//! frontier commitment before producing its own opaque verified frontier.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_qualification_witness_frontier::{
    EXTERNAL_ANCHOR_SCHEMA_VERSION, ExternalAnchorVerificationError,
    ExternalWitnessFrontierClaimV1, ExternalWitnessFrontierVerifier, FrontierRecoveryError,
    VerifiedExternalWitnessFrontierV1, verify_external_witness_frontier_v1,
};
use symthaea_xenia_authority::VerifiedXeniaWitnessFrontierV1;
use thiserror::Error;

/// Both representations of one already-authenticated Xenia witness frontier.
///
/// The generic proof is the only member intended for #452 recovery/classification.
/// The retained Xenia proof preserves source-specific forensic commitments such
/// as the anchor fingerprint, operation id and observation timestamp.
#[derive(Debug)]
pub struct XeniaExternalWitnessFrontierV1 {
    xenia: VerifiedXeniaWitnessFrontierV1,
    external: VerifiedExternalWitnessFrontierV1,
}

impl XeniaExternalWitnessFrontierV1 {
    /// Generic transport-neutral proof consumed by #452 recovery semantics.
    pub fn external(&self) -> &VerifiedExternalWitnessFrontierV1 {
        &self.external
    }

    /// Original independently verified Xenia chronology evidence.
    pub fn xenia(&self) -> &VerifiedXeniaWitnessFrontierV1 {
        &self.xenia
    }

    /// Consume the wrapper while retaining both opaque proofs.
    pub fn into_parts(
        self,
    ) -> (
        VerifiedXeniaWitnessFrontierV1,
        VerifiedExternalWitnessFrontierV1,
    ) {
        (self.xenia, self.external)
    }
}

/// Convert one already-verified Xenia frontier into #452's transport-neutral
/// external-frontier proof.
///
/// No raw Xenia anchor, observation, time value, challenge or caller-built
/// external claim is accepted here. The generic claim is derived exclusively
/// from the opaque Xenia proof and checked again by #452's structural verifier.
pub fn adapt_verified_xenia_witness_frontier_v1(
    xenia: VerifiedXeniaWitnessFrontierV1,
) -> Result<XeniaExternalWitnessFrontierV1, XeniaFrontierAdapterError> {
    let snapshot = XeniaFrontierSnapshotV1::from_verified(&xenia);
    let claim = snapshot.claim();
    let verifier = ExactVerifiedXeniaSource { expected: snapshot };
    let external = verify_external_witness_frontier_v1(claim, &verifier)?;
    Ok(XeniaExternalWitnessFrontierV1 { xenia, external })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct XeniaFrontierSnapshotV1 {
    source_id: [u8; 16],
    source_epoch: u64,
    source_sequence: u64,
    witness_id: [u8; 16],
    high_watermark: u64,
    reservation_head: Digest32,
    frontier_statement_digest: Digest32,
    freshness_evidence_digest: Digest32,
}

impl XeniaFrontierSnapshotV1 {
    fn from_verified(verified: &VerifiedXeniaWitnessFrontierV1) -> Self {
        Self {
            source_id: verified.source_id(),
            source_epoch: verified.source_epoch(),
            source_sequence: verified.source_sequence(),
            witness_id: verified.witness_id(),
            high_watermark: verified.high_watermark(),
            reservation_head: verified.reservation_head(),
            frontier_statement_digest: verified.frontier_statement_digest(),
            freshness_evidence_digest: verified.freshness_evidence_digest(),
        }
    }

    fn claim(self) -> ExternalWitnessFrontierClaimV1 {
        ExternalWitnessFrontierClaimV1 {
            schema_version: EXTERNAL_ANCHOR_SCHEMA_VERSION,
            source_id: self.source_id,
            source_epoch: self.source_epoch,
            source_sequence: self.source_sequence,
            witness_id: self.witness_id,
            high_watermark: self.high_watermark,
            reservation_head: self.reservation_head,
            frontier_statement_digest: self.frontier_statement_digest,
            freshness_evidence_digest: self.freshness_evidence_digest,
        }
    }
}

struct ExactVerifiedXeniaSource {
    expected: XeniaFrontierSnapshotV1,
}

impl ExternalWitnessFrontierVerifier for ExactVerifiedXeniaSource {
    fn verify_current(
        &self,
        claim: &ExternalWitnessFrontierClaimV1,
    ) -> Result<(), ExternalAnchorVerificationError> {
        let actual = XeniaFrontierSnapshotV1 {
            source_id: claim.source_id,
            source_epoch: claim.source_epoch,
            source_sequence: claim.source_sequence,
            witness_id: claim.witness_id,
            high_watermark: claim.high_watermark,
            reservation_head: claim.reservation_head,
            frontier_statement_digest: claim.frontier_statement_digest,
            freshness_evidence_digest: claim.freshness_evidence_digest,
        };
        if actual != self.expected {
            return Err(ExternalAnchorVerificationError {
                reason: "external frontier differs from the already-verified Xenia chronology"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Fail-closed bridge errors.
#[derive(Debug, Error)]
pub enum XeniaFrontierAdapterError {
    /// #452 rejected the derived claim structurally or rejected exact-currentness
    /// equality against the source-specific verified evidence.
    #[error("generic witness-frontier recovery rejected verified Xenia evidence: {0}")]
    Recovery(#[from] FrontierRecoveryError),
}

#[cfg(test)]
mod tests {
    use symthaea_qualification_witness_frontier::{
        FrontierRecoveryError, verify_external_witness_frontier_v1,
    };
    use symthaea_xenia_authority::witness_frontier_statement_digest;

    use super::*;

    fn snapshot() -> XeniaFrontierSnapshotV1 {
        let witness_id = [0x22; 16];
        let high_watermark = 9;
        let reservation_head = Digest32([0x33; 32]);
        XeniaFrontierSnapshotV1 {
            source_id: [0x11; 16],
            source_epoch: 3,
            source_sequence: 7,
            witness_id,
            high_watermark,
            reservation_head,
            frontier_statement_digest: Digest32(witness_frontier_statement_digest(
                witness_id,
                high_watermark,
                reservation_head.0,
            )),
            freshness_evidence_digest: Digest32([0x55; 32]),
        }
    }

    #[test]
    fn exact_projection_passes_generic_structural_and_currentness_checks() {
        let expected = snapshot();
        let verifier = ExactVerifiedXeniaSource { expected };
        let external = verify_external_witness_frontier_v1(expected.claim(), &verifier).unwrap();
        assert_eq!(external.source_id(), expected.source_id);
        assert_eq!(external.source_epoch(), expected.source_epoch);
        assert_eq!(external.source_sequence(), expected.source_sequence);
        assert_eq!(external.point().witness_id, expected.witness_id);
        assert_eq!(external.point().high_watermark, expected.high_watermark);
        assert_eq!(external.point().reservation_head, expected.reservation_head);
        assert_eq!(
            external.freshness_evidence_digest(),
            expected.freshness_evidence_digest
        );
    }

    #[test]
    fn exact_source_verifier_rejects_field_substitution() {
        let expected = snapshot();
        let verifier = ExactVerifiedXeniaSource { expected };
        let mut claim = expected.claim();
        claim.source_sequence += 1;
        assert!(matches!(
            verify_external_witness_frontier_v1(claim, &verifier),
            Err(FrontierRecoveryError::ExternalVerification(_))
        ));
    }
}
