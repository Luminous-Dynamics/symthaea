// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Externally verified trust evidence for EKM restart continuity anchors.
//!
//! EKM-045 can classify rollback and same-cycle equivocation relative to a trusted
//! anchor, but deliberately does not decide why that anchor should be trusted.
//! This module adds a bounded verification-provider boundary and a monotonic anchor
//! chain without choosing a concrete signature, TPM, transparency-log, or quorum
//! implementation.
//!
//! A successful verification result can be converted into the existing EKM-045
//! continuity anchor. It still grants no quarantine-construction or activation
//! authority.

use super::epistemic_restart_continuity::TrustedRestartValidationAnchorV1;
use super::epistemic_restart_validation_receipt::{
    EpistemicRestartValidationReceiptDigest, EpistemicRestartValidationReceiptV1,
};
use std::error::Error;
use std::fmt;

pub const MAX_RESTART_ANCHOR_AUTHORITY_ID_BYTES: usize = 256;
pub const MAX_RESTART_ANCHOR_PROOF_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartAnchorEvidenceKindV1 {
    /// Deployment-specific protected storage or monotonic state verified upstream.
    ProtectedCheckpoint,
    /// A signature-bearing checkpoint verified by an external provider.
    SignedCheckpoint,
    /// A checkpoint observed by an externally verified witness/quorum mechanism.
    WitnessedCheckpoint,
    /// Hardware-backed monotonic/attested state verified by an external provider.
    HardwareAttestedCheckpoint,
    /// An externally verified append-only/transparency-log checkpoint.
    TransparencyCheckpoint,
}

impl RestartAnchorEvidenceKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::ProtectedCheckpoint => 1,
            Self::SignedCheckpoint => 2,
            Self::WitnessedCheckpoint => 3,
            Self::HardwareAttestedCheckpoint => 4,
            Self::TransparencyCheckpoint => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartAnchorDigestV1([u8; 32]);

impl RestartAnchorDigestV1 {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartAnchorStatementV1 {
    sequence: u64,
    captured_at_cycle: u64,
    receipt_digest: EpistemicRestartValidationReceiptDigest,
    previous_anchor_digest: Option<RestartAnchorDigestV1>,
    issued_at_cycle: u64,
    expires_at_cycle: u64,
    authority_id: String,
    evidence_kind: RestartAnchorEvidenceKindV1,
}

impl RestartAnchorStatementV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: u64,
        receipt: &EpistemicRestartValidationReceiptV1,
        previous_anchor_digest: Option<RestartAnchorDigestV1>,
        issued_at_cycle: u64,
        expires_at_cycle: u64,
        authority_id: impl Into<String>,
        evidence_kind: RestartAnchorEvidenceKindV1,
    ) -> Result<Self, RestartAnchorEvidenceError> {
        let statement = Self {
            sequence,
            captured_at_cycle: receipt.captured_at_cycle(),
            receipt_digest: receipt.receipt_digest(),
            previous_anchor_digest,
            issued_at_cycle,
            expires_at_cycle,
            authority_id: authority_id.into(),
            evidence_kind,
        };
        statement.validate_shape()?;
        Ok(statement)
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn receipt_digest(&self) -> EpistemicRestartValidationReceiptDigest {
        self.receipt_digest
    }

    pub fn previous_anchor_digest(&self) -> Option<RestartAnchorDigestV1> {
        self.previous_anchor_digest
    }

    pub fn issued_at_cycle(&self) -> u64 {
        self.issued_at_cycle
    }

    pub fn expires_at_cycle(&self) -> u64 {
        self.expires_at_cycle
    }

    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub fn evidence_kind(&self) -> RestartAnchorEvidenceKindV1 {
        self.evidence_kind
    }

    fn validate_shape(&self) -> Result<(), RestartAnchorEvidenceError> {
        if self.sequence == 0 {
            return Err(RestartAnchorEvidenceError::InvalidSequence);
        }
        if (self.sequence == 1) != self.previous_anchor_digest.is_none() {
            return Err(RestartAnchorEvidenceError::InvalidPredecessorShape);
        }
        if self.captured_at_cycle > self.issued_at_cycle {
            return Err(RestartAnchorEvidenceError::IssuedBeforeCapture {
                captured_at_cycle: self.captured_at_cycle,
                issued_at_cycle: self.issued_at_cycle,
            });
        }
        if self.issued_at_cycle >= self.expires_at_cycle {
            return Err(RestartAnchorEvidenceError::InvalidValidityWindow);
        }
        if self.authority_id.trim().is_empty()
            || self.authority_id != self.authority_id.trim()
            || self.authority_id.len() > MAX_RESTART_ANCHOR_AUTHORITY_ID_BYTES
        {
            return Err(RestartAnchorEvidenceError::InvalidAuthorityId);
        }
        Ok(())
    }
}

/// Opaque verification evidence supplied by the deployment-specific trust layer.
///
/// This module bounds the proof bytes but does not interpret their cryptographic
/// format. The provider is responsible for signature/TPM/transparency/quorum
/// verification appropriate to the selected evidence kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartAnchorEvidenceV1 {
    statement: RestartAnchorStatementV1,
    proof: Vec<u8>,
}

impl RestartAnchorEvidenceV1 {
    pub fn new(
        statement: RestartAnchorStatementV1,
        proof: Vec<u8>,
    ) -> Result<Self, RestartAnchorEvidenceError> {
        statement.validate_shape()?;
        if proof.is_empty() {
            return Err(RestartAnchorEvidenceError::EmptyProof);
        }
        if proof.len() > MAX_RESTART_ANCHOR_PROOF_BYTES {
            return Err(RestartAnchorEvidenceError::ProofTooLarge {
                actual: proof.len(),
                maximum: MAX_RESTART_ANCHOR_PROOF_BYTES,
            });
        }
        Ok(Self { statement, proof })
    }

    pub fn statement(&self) -> &RestartAnchorStatementV1 {
        &self.statement
    }

    pub fn proof(&self) -> &[u8] {
        &self.proof
    }
}

/// Deployment/provider boundary for cryptographic or protected-state verification.
///
/// Returning true means only that this provider accepts the proof for the exact
/// statement digest under its own external trust configuration.
pub trait RestartAnchorEvidenceVerifierV1 {
    fn verify_restart_anchor_evidence(
        &self,
        evidence_kind: RestartAnchorEvidenceKindV1,
        authority_id: &str,
        statement_digest: [u8; 32],
        proof: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRestartAnchorEvidenceV1 {
    statement: RestartAnchorStatementV1,
    statement_digest: RestartAnchorDigestV1,
    proof_digest: [u8; 32],
    verified_at_cycle: u64,
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
}

impl VerifiedRestartAnchorEvidenceV1 {
    pub fn statement(&self) -> &RestartAnchorStatementV1 {
        &self.statement
    }

    pub fn statement_digest(&self) -> RestartAnchorDigestV1 {
        self.statement_digest
    }

    pub fn proof_digest(&self) -> [u8; 32] {
        self.proof_digest
    }

    pub fn verified_at_cycle(&self) -> u64 {
        self.verified_at_cycle
    }

    /// Converts externally verified anchor evidence into EKM-045's narrow
    /// continuity comparison input. This does not grant restore authority.
    pub fn continuity_anchor(&self) -> TrustedRestartValidationAnchorV1 {
        TrustedRestartValidationAnchorV1::from_verified_parts(
            self.statement.captured_at_cycle,
            self.statement.receipt_digest,
        )
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
}

pub fn verify_restart_anchor_evidence(
    evidence: &RestartAnchorEvidenceV1,
    receipt: &EpistemicRestartValidationReceiptV1,
    observed_at_cycle: u64,
    verifier: &dyn RestartAnchorEvidenceVerifierV1,
) -> Result<VerifiedRestartAnchorEvidenceV1, RestartAnchorEvidenceError> {
    evidence.statement.validate_shape()?;
    if evidence.statement.captured_at_cycle != receipt.captured_at_cycle()
        || evidence.statement.receipt_digest != receipt.receipt_digest()
    {
        return Err(RestartAnchorEvidenceError::ReceiptBindingMismatch);
    }
    if observed_at_cycle < evidence.statement.issued_at_cycle {
        return Err(RestartAnchorEvidenceError::NotYetValid {
            observed_at_cycle,
            issued_at_cycle: evidence.statement.issued_at_cycle,
        });
    }
    if observed_at_cycle >= evidence.statement.expires_at_cycle {
        return Err(RestartAnchorEvidenceError::Expired {
            observed_at_cycle,
            expires_at_cycle: evidence.statement.expires_at_cycle,
        });
    }

    let statement_digest = digest_restart_anchor_statement(&evidence.statement)?;
    let accepted = verifier
        .verify_restart_anchor_evidence(
            evidence.statement.evidence_kind,
            &evidence.statement.authority_id,
            statement_digest.as_bytes(),
            &evidence.proof,
        )
        .map_err(RestartAnchorEvidenceError::VerificationProvider)?;
    if !accepted {
        return Err(RestartAnchorEvidenceError::ProofRejected);
    }

    let mut proof_hasher = blake3::Hasher::new();
    proof_hasher.update(b"symthaea-ekm-restart-anchor-proof-v1");
    proof_hasher.update(&evidence.proof);
    Ok(VerifiedRestartAnchorEvidenceV1 {
        statement: evidence.statement.clone(),
        statement_digest,
        proof_digest: *proof_hasher.finalize().as_bytes(),
        verified_at_cycle: observed_at_cycle,
        quarantine_construction_authorized: false,
        activation_authorized: false,
    })
}

pub fn digest_restart_anchor_statement(
    statement: &RestartAnchorStatementV1,
) -> Result<RestartAnchorDigestV1, RestartAnchorEvidenceError> {
    statement.validate_shape()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-anchor-statement-v1");
    hasher.update(&statement.sequence.to_le_bytes());
    hasher.update(&statement.captured_at_cycle.to_le_bytes());
    hasher.update(&statement.receipt_digest.as_bytes());
    match statement.previous_anchor_digest {
        Some(previous) => {
            hasher.update(&[1]);
            hasher.update(&previous.as_bytes());
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&statement.issued_at_cycle.to_le_bytes());
    hasher.update(&statement.expires_at_cycle.to_le_bytes());
    let authority_len = u64::try_from(statement.authority_id.len())
        .map_err(|_| RestartAnchorEvidenceError::LengthOverflow)?;
    hasher.update(&authority_len.to_le_bytes());
    hasher.update(statement.authority_id.as_bytes());
    hasher.update(&[statement.evidence_kind.tag()]);
    Ok(RestartAnchorDigestV1(*hasher.finalize().as_bytes()))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestartAnchorTrackerV1 {
    latest_sequence: Option<u64>,
    latest_anchor_digest: Option<RestartAnchorDigestV1>,
    latest_receipt_digest: Option<EpistemicRestartValidationReceiptDigest>,
}

impl RestartAnchorTrackerV1 {
    pub fn latest_sequence(&self) -> Option<u64> {
        self.latest_sequence
    }

    pub fn latest_anchor_digest(&self) -> Option<RestartAnchorDigestV1> {
        self.latest_anchor_digest
    }

    pub fn latest_receipt_digest(&self) -> Option<EpistemicRestartValidationReceiptDigest> {
        self.latest_receipt_digest
    }

    /// Accept a verified anchor into a strict contiguous predecessor chain.
    ///
    /// The tracker is in-memory only. Persisting it in rollback-resistant storage
    /// remains an external deployment responsibility.
    pub fn accept(
        &mut self,
        verified: &VerifiedRestartAnchorEvidenceV1,
        observed_at_cycle: u64,
    ) -> Result<(), RestartAnchorTrackingError> {
        if observed_at_cycle < verified.verified_at_cycle
            || observed_at_cycle < verified.statement.issued_at_cycle
        {
            return Err(RestartAnchorTrackingError::ObservationPredatesVerification {
                observed_at_cycle,
                verified_at_cycle: verified.verified_at_cycle,
            });
        }
        if observed_at_cycle >= verified.statement.expires_at_cycle {
            return Err(RestartAnchorTrackingError::Expired {
                observed_at_cycle,
                expires_at_cycle: verified.statement.expires_at_cycle,
            });
        }

        let sequence = verified.statement.sequence;
        let digest = verified.statement_digest;

        match (self.latest_sequence, self.latest_anchor_digest) {
            (None, None) => {
                if sequence != 1 || verified.statement.previous_anchor_digest.is_some() {
                    return Err(RestartAnchorTrackingError::InvalidGenesis);
                }
            }
            (Some(latest_sequence), Some(latest_digest)) => {
                if sequence < latest_sequence {
                    return Err(RestartAnchorTrackingError::SequenceRollback {
                        latest: latest_sequence,
                        proposed: sequence,
                    });
                }
                if sequence == latest_sequence {
                    if digest == latest_digest {
                        return Err(RestartAnchorTrackingError::Replay);
                    }
                    return Err(RestartAnchorTrackingError::SameSequenceSubstitution);
                }
                let expected = latest_sequence
                    .checked_add(1)
                    .ok_or(RestartAnchorTrackingError::SequenceOverflow)?;
                if sequence != expected {
                    return Err(RestartAnchorTrackingError::SequenceGap {
                        expected,
                        actual: sequence,
                    });
                }
                if verified.statement.previous_anchor_digest != Some(latest_digest) {
                    return Err(RestartAnchorTrackingError::PreviousAnchorMismatch);
                }
            }
            _ => return Err(RestartAnchorTrackingError::InvalidTrackerState),
        }

        self.latest_sequence = Some(sequence);
        self.latest_anchor_digest = Some(digest);
        self.latest_receipt_digest = Some(verified.statement.receipt_digest);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAnchorEvidenceError {
    InvalidSequence,
    InvalidPredecessorShape,
    InvalidValidityWindow,
    IssuedBeforeCapture {
        captured_at_cycle: u64,
        issued_at_cycle: u64,
    },
    InvalidAuthorityId,
    EmptyProof,
    ProofTooLarge {
        actual: usize,
        maximum: usize,
    },
    ReceiptBindingMismatch,
    NotYetValid {
        observed_at_cycle: u64,
        issued_at_cycle: u64,
    },
    Expired {
        observed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    VerificationProvider(String),
    ProofRejected,
    LengthOverflow,
}

impl fmt::Display for RestartAnchorEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart anchor evidence invalid: {self:?}")
    }
}

impl Error for RestartAnchorEvidenceError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAnchorTrackingError {
    InvalidTrackerState,
    InvalidGenesis,
    SequenceRollback { latest: u64, proposed: u64 },
    SameSequenceSubstitution,
    Replay,
    SequenceGap { expected: u64, actual: u64 },
    PreviousAnchorMismatch,
    SequenceOverflow,
    ObservationPredatesVerification {
        observed_at_cycle: u64,
        verified_at_cycle: u64,
    },
    Expired {
        observed_at_cycle: u64,
        expires_at_cycle: u64,
    },
}

impl fmt::Display for RestartAnchorTrackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart anchor tracking rejected: {self:?}")
    }
}

impl Error for RestartAnchorTrackingError {}

#[cfg(test)]
mod tests {
    use super::*;

    struct AcceptOnly(&'static [u8]);

    impl RestartAnchorEvidenceVerifierV1 for AcceptOnly {
        fn verify_restart_anchor_evidence(
            &self,
            _evidence_kind: RestartAnchorEvidenceKindV1,
            _authority_id: &str,
            _statement_digest: [u8; 32],
            proof: &[u8],
        ) -> Result<bool, String> {
            Ok(proof == self.0)
        }
    }

    // These tests use the EKM-045 module's own deterministic receipt fixture
    // indirectly through a helper kept here to avoid introducing test-only
    // constructors for receipt digests.
    fn receipt(capture_cycle: u64, statement: &str) -> EpistemicRestartValidationReceiptV1 {
        use crate::knowledge::{
            BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
            BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
            BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
            EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
            EpistemicRestartCapsuleV2, EpistemicRestartWireV2, EpistemicRevisionProposal,
            EpistemicSupportStore, EvidenceKind, EvidencePolarity,
        };

        let mut ledger = EpistemicLedger::new();
        let provenance = ledger.add_provenance("lab", None, None, 1, vec![]).unwrap();
        let claim = ledger.add_claim(statement, ClaimKind::Predictive, None, None, 1);
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
        let inventory =
            EpistemicLedgerInventoryV1::new(vec![claim], vec![evidence], vec![provenance]).unwrap();
        let proposal =
            EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement").unwrap();
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
        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[], capture_cycle).unwrap();
        let revisions =
            BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, capture_cycle).unwrap();
        let schemas = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revisions,
            capture_cycle,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(
            &ledger,
            &inventory,
            &mutations,
            &revisions,
            capture_cycle,
        )
        .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        let bytes = EpistemicRestartWireV2::encode(&v2).unwrap();
        let snapshot = EpistemicRestartWireV2::decode(&bytes).unwrap();
        EpistemicRestartValidationReceiptV1::validate_and_capture(&snapshot).unwrap()
    }

    fn verified(
        receipt: &EpistemicRestartValidationReceiptV1,
        sequence: u64,
        previous: Option<RestartAnchorDigestV1>,
        proof: &[u8],
    ) -> Result<VerifiedRestartAnchorEvidenceV1, RestartAnchorEvidenceError> {
        let statement = RestartAnchorStatementV1::new(
            sequence,
            receipt,
            previous,
            receipt.captured_at_cycle(),
            receipt.captured_at_cycle() + 10,
            "test-authority",
            RestartAnchorEvidenceKindV1::ProtectedCheckpoint,
        )?;
        let evidence = RestartAnchorEvidenceV1::new(statement, proof.to_vec())?;
        verify_restart_anchor_evidence(
            &evidence,
            receipt,
            receipt.captured_at_cycle(),
            &AcceptOnly(b"valid"),
        )
    }

    #[test]
    fn verified_anchor_binds_exact_receipt_and_never_authorizes_restore() {
        let receipt = receipt(4, "X predicts Y");
        let verified = verified(&receipt, 1, None, b"valid").unwrap();
        assert_eq!(
            verified.statement().receipt_digest(),
            receipt.receipt_digest()
        );
        assert!(!verified.quarantine_construction_authorized());
        assert!(!verified.activation_authorized());
        let anchor = verified.continuity_anchor();
        assert_eq!(anchor.receipt_digest(), receipt.receipt_digest());
    }

    #[test]
    fn provider_rejection_fails_closed() {
        let receipt = receipt(4, "X predicts Y");
        assert_eq!(
            verified(&receipt, 1, None, b"wrong").unwrap_err(),
            RestartAnchorEvidenceError::ProofRejected
        );
    }

    #[test]
    fn tracker_requires_exact_predecessor_chain() {
        let first_receipt = receipt(4, "X predicts Y");
        let first = verified(&first_receipt, 1, None, b"valid").unwrap();

        let second_receipt = receipt(5, "X predicts Z");
        let second = verified(
            &second_receipt,
            2,
            Some(first.statement_digest()),
            b"valid",
        )
        .unwrap();

        let mut tracker = RestartAnchorTrackerV1::default();
        tracker.accept(&first, 4).unwrap();
        tracker.accept(&second, 5).unwrap();
        assert_eq!(tracker.latest_sequence(), Some(2));
    }

    #[test]
    fn tracker_rejects_gap_substitution_and_replay() {
        let first_receipt = receipt(4, "X predicts Y");
        let first = verified(&first_receipt, 1, None, b"valid").unwrap();
        let next_receipt = receipt(5, "X predicts Z");

        let mut tracker = RestartAnchorTrackerV1::default();
        tracker.accept(&first, 4).unwrap();
        assert_eq!(
            tracker.accept(&first, 4).unwrap_err(),
            RestartAnchorTrackingError::Replay
        );

        let gap = verified(
            &next_receipt,
            3,
            Some(first.statement_digest()),
            b"valid",
        )
        .unwrap();
        assert!(matches!(
            tracker.accept(&gap, 5),
            Err(RestartAnchorTrackingError::SequenceGap { .. })
        ));
    }

    #[test]
    fn expired_or_mismatched_anchor_evidence_is_rejected() {
        let receipt = receipt(4, "X predicts Y");
        let statement = RestartAnchorStatementV1::new(
            1,
            &receipt,
            None,
            4,
            5,
            "test-authority",
            RestartAnchorEvidenceKindV1::SignedCheckpoint,
        )
        .unwrap();
        let evidence = RestartAnchorEvidenceV1::new(statement, b"valid".to_vec()).unwrap();
        assert!(matches!(
            verify_restart_anchor_evidence(&evidence, &receipt, 5, &AcceptOnly(b"valid")),
            Err(RestartAnchorEvidenceError::Expired { .. })
        ));

        let other = receipt(4, "X predicts Z");
        assert_eq!(
            verify_restart_anchor_evidence(&evidence, &other, 4, &AcceptOnly(b"valid"))
                .unwrap_err(),
            RestartAnchorEvidenceError::ReceiptBindingMismatch
        );
    }
}
