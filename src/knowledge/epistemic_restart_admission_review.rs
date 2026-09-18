// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Staged, non-mutating admission review for validated EKM restart candidates.
//!
//! A candidate may pass semantic validation, local trust policy, external proof
//! verification, and continuity checks while still not being authorized to build
//! or activate quarantine state. This module binds those facts into one immutable
//! review receipt without advancing the real trusted-anchor tracker.
//!
//! That ordering matters: advancing the trust anchor before a later quarantine or
//! activation succeeds could make the still-running state appear to be a rollback
//! after a failed restore attempt. Anchor advancement therefore remains a separate
//! future commit boundary.

use super::epistemic_restart_anchor::{
    RestartAnchorDigestV1, RestartAnchorEvidenceV1, RestartAnchorEvidenceVerifierV1,
    RestartAnchorTrackerV1, RestartAnchorTrackingError,
};
use super::epistemic_restart_anchor_policy::{
    verify_restart_anchor_evidence_under_policy, RestartAnchorPolicyVerificationError,
    RestartAnchorTrustPolicyV1,
};
use super::epistemic_restart_continuity::{
    RestartContinuityDispositionV1, RestartContinuityGateV1, TrustedRestartValidationAnchorV1,
};
use super::epistemic_restart_validation_receipt::{
    EpistemicRestartValidationReceiptDigest, EpistemicRestartValidationReceiptError,
    EpistemicRestartValidationReceiptV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartAdmissionReviewVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartAnchorTrustPolicyDigestV1([u8; 32]);

impl RestartAnchorTrustPolicyDigestV1 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicRestartAdmissionReviewDigest([u8; 32]);

impl EpistemicRestartAdmissionReviewDigest {
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
pub struct EpistemicRestartAdmissionReviewReceiptV1 {
    version: EpistemicRestartAdmissionReviewVersion,
    candidate_capture_cycle: u64,
    candidate_validation_receipt_digest: EpistemicRestartValidationReceiptDigest,
    policy_digest: RestartAnchorTrustPolicyDigestV1,
    prior_anchor_sequence: u64,
    prior_anchor_digest: RestartAnchorDigestV1,
    prior_receipt_digest: EpistemicRestartValidationReceiptDigest,
    prior_capture_cycle: u64,
    candidate_anchor_sequence: u64,
    candidate_anchor_statement_digest: RestartAnchorDigestV1,
    candidate_anchor_proof_digest: [u8; 32],
    candidate_anchor_verified_at_cycle: u64,
    continuity_disposition: RestartContinuityDispositionV1,
    observed_at_cycle: u64,
    anchor_tracker_mutated: bool,
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
    review_digest: EpistemicRestartAdmissionReviewDigest,
}

impl EpistemicRestartAdmissionReviewReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_review(
        snapshot: &EpistemicRestartWireSnapshotV2,
        policy: &RestartAnchorTrustPolicyV1,
        candidate_anchor_evidence: &RestartAnchorEvidenceV1,
        verifier: &dyn RestartAnchorEvidenceVerifierV1,
        trusted_tracker: &RestartAnchorTrackerV1,
        observed_at_cycle: u64,
    ) -> Result<Self, EpistemicRestartAdmissionReviewError> {
        let validation_receipt = EpistemicRestartValidationReceiptV1::validate_and_capture(snapshot)
            .map_err(EpistemicRestartAdmissionReviewError::Validation)?;
        let verified_anchor = verify_restart_anchor_evidence_under_policy(
            policy,
            candidate_anchor_evidence,
            &validation_receipt,
            observed_at_cycle,
            verifier,
        )
        .map_err(EpistemicRestartAdmissionReviewError::PolicyVerification)?;

        let prior_sequence = trusted_tracker
            .latest_sequence()
            .ok_or(EpistemicRestartAdmissionReviewError::UninitializedTrustedTracker)?;
        let prior_anchor_digest = trusted_tracker
            .latest_anchor_digest()
            .ok_or(EpistemicRestartAdmissionReviewError::UninitializedTrustedTracker)?;
        let prior_receipt_digest = trusted_tracker
            .latest_receipt_digest()
            .ok_or(EpistemicRestartAdmissionReviewError::UninitializedTrustedTracker)?;
        let prior_capture_cycle = trusted_tracker
            .latest_captured_at_cycle()
            .ok_or(EpistemicRestartAdmissionReviewError::UninitializedTrustedTracker)?;

        // Preview against a clone. The real tracker is intentionally not advanced.
        let mut preview = trusted_tracker.clone();
        preview
            .accept(&verified_anchor, observed_at_cycle)
            .map_err(EpistemicRestartAdmissionReviewError::TrackerPreview)?;

        let continuity_anchor = TrustedRestartValidationAnchorV1::from_verified_parts(
            prior_capture_cycle,
            prior_receipt_digest,
        );
        let continuity = RestartContinuityGateV1::evaluate(continuity_anchor, &validation_receipt);
        if continuity.disposition() != RestartContinuityDispositionV1::ForwardProgress
            || !continuity.further_review_eligible()
        {
            return Err(EpistemicRestartAdmissionReviewError::ContinuityNotForwardProgress(
                continuity.disposition(),
            ));
        }

        let policy_digest = digest_restart_anchor_trust_policy(policy)?;
        let mut receipt = Self {
            version: EpistemicRestartAdmissionReviewVersion::V1,
            candidate_capture_cycle: validation_receipt.captured_at_cycle(),
            candidate_validation_receipt_digest: validation_receipt.receipt_digest(),
            policy_digest,
            prior_anchor_sequence: prior_sequence,
            prior_anchor_digest,
            prior_receipt_digest,
            prior_capture_cycle,
            candidate_anchor_sequence: verified_anchor.statement().sequence(),
            candidate_anchor_statement_digest: verified_anchor.statement_digest(),
            candidate_anchor_proof_digest: verified_anchor.proof_digest(),
            candidate_anchor_verified_at_cycle: verified_anchor.verified_at_cycle(),
            continuity_disposition: continuity.disposition(),
            observed_at_cycle,
            anchor_tracker_mutated: false,
            quarantine_construction_authorized: false,
            activation_authorized: false,
            review_digest: EpistemicRestartAdmissionReviewDigest([0; 32]),
        };
        receipt.review_digest = digest_admission_review(&receipt)?;
        Ok(receipt)
    }

    pub fn version(&self) -> EpistemicRestartAdmissionReviewVersion {
        self.version
    }

    pub fn candidate_capture_cycle(&self) -> u64 {
        self.candidate_capture_cycle
    }

    pub fn candidate_validation_receipt_digest(&self) -> EpistemicRestartValidationReceiptDigest {
        self.candidate_validation_receipt_digest
    }

    pub fn policy_digest(&self) -> RestartAnchorTrustPolicyDigestV1 {
        self.policy_digest
    }

    pub fn prior_anchor_sequence(&self) -> u64 {
        self.prior_anchor_sequence
    }

    pub fn prior_anchor_digest(&self) -> RestartAnchorDigestV1 {
        self.prior_anchor_digest
    }

    pub fn prior_receipt_digest(&self) -> EpistemicRestartValidationReceiptDigest {
        self.prior_receipt_digest
    }

    pub fn prior_capture_cycle(&self) -> u64 {
        self.prior_capture_cycle
    }

    pub fn candidate_anchor_sequence(&self) -> u64 {
        self.candidate_anchor_sequence
    }

    pub fn candidate_anchor_statement_digest(&self) -> RestartAnchorDigestV1 {
        self.candidate_anchor_statement_digest
    }

    pub fn candidate_anchor_proof_digest(&self) -> [u8; 32] {
        self.candidate_anchor_proof_digest
    }

    pub fn candidate_anchor_verified_at_cycle(&self) -> u64 {
        self.candidate_anchor_verified_at_cycle
    }

    pub fn continuity_disposition(&self) -> RestartContinuityDispositionV1 {
        self.continuity_disposition
    }

    pub fn observed_at_cycle(&self) -> u64 {
        self.observed_at_cycle
    }

    pub fn anchor_tracker_mutated(&self) -> bool {
        self.anchor_tracker_mutated
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn review_digest(&self) -> EpistemicRestartAdmissionReviewDigest {
        self.review_digest
    }
}

pub fn digest_restart_anchor_trust_policy(
    policy: &RestartAnchorTrustPolicyV1,
) -> Result<RestartAnchorTrustPolicyDigestV1, EpistemicRestartAdmissionReviewError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-anchor-trust-policy-v1");
    hash_usize(&mut hasher, policy.allowed_authority_ids().len())?;
    for authority in policy.allowed_authority_ids() {
        hash_bytes(&mut hasher, authority.as_bytes())?;
    }
    hash_usize(&mut hasher, policy.allowed_evidence_kinds().len())?;
    for kind in policy.allowed_evidence_kinds() {
        hasher.update(&[evidence_kind_tag(*kind)]);
    }
    hasher.update(&policy.maximum_validity_window_cycles().to_le_bytes());
    hasher.update(&policy.maximum_verification_delay_cycles().to_le_bytes());
    Ok(RestartAnchorTrustPolicyDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn digest_admission_review(
    receipt: &EpistemicRestartAdmissionReviewReceiptV1,
) -> Result<EpistemicRestartAdmissionReviewDigest, EpistemicRestartAdmissionReviewError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-admission-review-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.candidate_capture_cycle.to_le_bytes());
    hasher.update(&receipt.candidate_validation_receipt_digest.as_bytes());
    hasher.update(&receipt.policy_digest.as_bytes());
    hasher.update(&receipt.prior_anchor_sequence.to_le_bytes());
    hasher.update(&receipt.prior_anchor_digest.as_bytes());
    hasher.update(&receipt.prior_receipt_digest.as_bytes());
    hasher.update(&receipt.prior_capture_cycle.to_le_bytes());
    hasher.update(&receipt.candidate_anchor_sequence.to_le_bytes());
    hasher.update(&receipt.candidate_anchor_statement_digest.as_bytes());
    hasher.update(&receipt.candidate_anchor_proof_digest);
    hasher.update(&receipt.candidate_anchor_verified_at_cycle.to_le_bytes());
    hasher.update(&[continuity_tag(receipt.continuity_disposition)]);
    hasher.update(&receipt.observed_at_cycle.to_le_bytes());
    hasher.update(&[u8::from(receipt.anchor_tracker_mutated)]);
    hasher.update(&[u8::from(receipt.quarantine_construction_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    Ok(EpistemicRestartAdmissionReviewDigest(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), EpistemicRestartAdmissionReviewError> {
    let value = u64::try_from(value)
        .map_err(|_| EpistemicRestartAdmissionReviewError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), EpistemicRestartAdmissionReviewError> {
    hash_usize(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn evidence_kind_tag(kind: super::epistemic_restart_anchor::RestartAnchorEvidenceKindV1) -> u8 {
    use super::epistemic_restart_anchor::RestartAnchorEvidenceKindV1;
    match kind {
        RestartAnchorEvidenceKindV1::ProtectedCheckpoint => 1,
        RestartAnchorEvidenceKindV1::SignedCheckpoint => 2,
        RestartAnchorEvidenceKindV1::WitnessedCheckpoint => 3,
        RestartAnchorEvidenceKindV1::HardwareAttestedCheckpoint => 4,
        RestartAnchorEvidenceKindV1::TransparencyCheckpoint => 5,
    }
}

fn continuity_tag(disposition: RestartContinuityDispositionV1) -> u8 {
    match disposition {
        RestartContinuityDispositionV1::IdempotentReplay => 1,
        RestartContinuityDispositionV1::ForwardProgress => 2,
        RestartContinuityDispositionV1::Rollback => 3,
        RestartContinuityDispositionV1::SameCycleEquivocation => 4,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartAdmissionReviewError {
    Validation(EpistemicRestartValidationReceiptError),
    PolicyVerification(RestartAnchorPolicyVerificationError),
    UninitializedTrustedTracker,
    TrackerPreview(RestartAnchorTrackingError),
    ContinuityNotForwardProgress(RestartContinuityDispositionV1),
    LengthOverflow,
}

impl fmt::Display for EpistemicRestartAdmissionReviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart admission review rejected: {self:?}")
    }
}

impl Error for EpistemicRestartAdmissionReviewError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
        EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
        EpistemicRestartCapsuleV2, EpistemicRestartWireV2, EpistemicRevisionProposal,
        EpistemicSupportStore, EvidenceKind, EvidencePolarity, RestartAnchorEvidenceKindV1,
        RestartAnchorStatementV1,
    };
    use crate::knowledge::epistemic_restart_anchor_policy::RestartAnchorTrustPolicyV1;

    struct Accept;

    impl RestartAnchorEvidenceVerifierV1 for Accept {
        fn verify_restart_anchor_evidence(
            &self,
            _evidence_kind: RestartAnchorEvidenceKindV1,
            _authority_id: &str,
            _statement_digest: [u8; 32],
            _proof: &[u8],
        ) -> Result<bool, String> {
            Ok(true)
        }
    }

    fn snapshot(capture_cycle: u64, statement: &str) -> EpistemicRestartWireSnapshotV2 {
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
        EpistemicRestartWireV2::decode(&bytes).unwrap()
    }

    fn validation_receipt(
        snapshot: &EpistemicRestartWireSnapshotV2,
    ) -> EpistemicRestartValidationReceiptV1 {
        EpistemicRestartValidationReceiptV1::validate_and_capture(snapshot).unwrap()
    }

    fn evidence(
        receipt: &EpistemicRestartValidationReceiptV1,
        sequence: u64,
        previous: Option<RestartAnchorDigestV1>,
    ) -> RestartAnchorEvidenceV1 {
        let statement = RestartAnchorStatementV1::new(
            sequence,
            receipt,
            previous,
            receipt.captured_at_cycle(),
            receipt.captured_at_cycle() + 10,
            "authority-a",
            RestartAnchorEvidenceKindV1::SignedCheckpoint,
        )
        .unwrap();
        RestartAnchorEvidenceV1::new(statement, b"proof".to_vec()).unwrap()
    }

    fn policy() -> RestartAnchorTrustPolicyV1 {
        RestartAnchorTrustPolicyV1::new(
            vec!["authority-a".into()],
            vec![RestartAnchorEvidenceKindV1::SignedCheckpoint],
            10,
            3,
        )
        .unwrap()
    }

    #[test]
    fn review_is_staged_and_does_not_advance_real_tracker() {
        let prior_snapshot = snapshot(4, "X predicts Y");
        let prior_receipt = validation_receipt(&prior_snapshot);
        let prior_evidence = evidence(&prior_receipt, 1, None);
        let prior_verified = verify_restart_anchor_evidence_under_policy(
            &policy(),
            &prior_evidence,
            &prior_receipt,
            4,
            &Accept,
        )
        .unwrap();
        let mut tracker = RestartAnchorTrackerV1::default();
        tracker.accept(&prior_verified, 4).unwrap();
        let before = tracker.clone();

        let candidate_snapshot = snapshot(5, "X predicts Z");
        let candidate_receipt = validation_receipt(&candidate_snapshot);
        let candidate_evidence = evidence(
            &candidate_receipt,
            2,
            Some(prior_verified.statement_digest()),
        );
        let review = EpistemicRestartAdmissionReviewReceiptV1::validate_and_review(
            &candidate_snapshot,
            &policy(),
            &candidate_evidence,
            &Accept,
            &tracker,
            5,
        )
        .unwrap();

        assert_eq!(tracker, before);
        assert!(!review.anchor_tracker_mutated());
        assert_eq!(
            review.continuity_disposition(),
            RestartContinuityDispositionV1::ForwardProgress
        );
        assert_eq!(review.prior_anchor_sequence(), 1);
        assert_eq!(review.candidate_anchor_sequence(), 2);
        assert!(!review.quarantine_construction_authorized());
        assert!(!review.activation_authorized());
    }

    #[test]
    fn stale_or_non_next_candidate_cannot_produce_review_receipt() {
        let current_snapshot = snapshot(5, "current");
        let current_receipt = validation_receipt(&current_snapshot);
        let current_evidence = evidence(&current_receipt, 1, None);
        let current_verified = verify_restart_anchor_evidence_under_policy(
            &policy(),
            &current_evidence,
            &current_receipt,
            5,
            &Accept,
        )
        .unwrap();
        let mut tracker = RestartAnchorTrackerV1::default();
        tracker.accept(&current_verified, 5).unwrap();

        let old_snapshot = snapshot(4, "old");
        let old_receipt = validation_receipt(&old_snapshot);
        let old_evidence = evidence(
            &old_receipt,
            2,
            Some(current_verified.statement_digest()),
        );
        assert!(matches!(
            EpistemicRestartAdmissionReviewReceiptV1::validate_and_review(
                &old_snapshot,
                &policy(),
                &old_evidence,
                &Accept,
                &tracker,
                5,
            ),
            Err(EpistemicRestartAdmissionReviewError::TrackerPreview(
                RestartAnchorTrackingError::CaptureCycleRollback { .. }
            ))
        ));
    }

    #[test]
    fn policy_digest_is_canonical_for_constructor_order() {
        let a = RestartAnchorTrustPolicyV1::new(
            vec!["z".into(), "a".into()],
            vec![
                RestartAnchorEvidenceKindV1::TransparencyCheckpoint,
                RestartAnchorEvidenceKindV1::SignedCheckpoint,
            ],
            10,
            3,
        )
        .unwrap();
        let b = RestartAnchorTrustPolicyV1::new(
            vec!["a".into(), "z".into()],
            vec![
                RestartAnchorEvidenceKindV1::SignedCheckpoint,
                RestartAnchorEvidenceKindV1::TransparencyCheckpoint,
            ],
            10,
            3,
        )
        .unwrap();
        assert_eq!(
            digest_restart_anchor_trust_policy(&a).unwrap(),
            digest_restart_anchor_trust_policy(&b).unwrap()
        );
    }
}
