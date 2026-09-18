// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifier provenance binding for staged epistemic restart admission.
//!
//! EKM-048 binds the candidate, trust policy, anchor proof, continuity decision,
//! and pre-activation tracker state, but the external verifier itself is still an
//! abstract provider. This module records which verifier profile and trust snapshot
//! were in force across the verification call.
//!
//! The profile is sampled before and after EKM-048 review and must remain exactly
//! stable. This detects provider/profile substitution during verification, but it
//! does not make a malicious provider truthful. No quarantine or activation
//! authority is granted here.

use super::epistemic_restart_admission_review::{
    EpistemicRestartAdmissionReviewDigest, EpistemicRestartAdmissionReviewError,
    EpistemicRestartAdmissionReviewReceiptV1,
};
use super::epistemic_restart_anchor::{
    RestartAnchorEvidenceV1, RestartAnchorEvidenceVerifierV1, RestartAnchorTrackerV1,
};
use super::epistemic_restart_anchor_policy::RestartAnchorTrustPolicyV1;
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

pub const MAX_RESTART_VERIFIER_ID_BYTES: usize = 256;
pub const MAX_RESTART_VERIFIER_IMPLEMENTATION_BYTES: usize = 256;
pub const MAX_RESTART_VERIFIER_VERSION_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartVerifierProvenanceVersion {
    V1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartVerifierProfileV1 {
    verifier_id: String,
    implementation_id: String,
    implementation_version: String,
    trust_snapshot_digest: [u8; 32],
    trust_snapshot_sequence: u64,
    trust_valid_from_cycle: u64,
    trust_valid_until_cycle: u64,
    configuration_digest: [u8; 32],
}

impl RestartVerifierProfileV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verifier_id: impl Into<String>,
        implementation_id: impl Into<String>,
        implementation_version: impl Into<String>,
        trust_snapshot_digest: [u8; 32],
        trust_snapshot_sequence: u64,
        trust_valid_from_cycle: u64,
        trust_valid_until_cycle: u64,
        configuration_digest: [u8; 32],
    ) -> Result<Self, RestartVerifierProvenanceError> {
        let profile = Self {
            verifier_id: verifier_id.into(),
            implementation_id: implementation_id.into(),
            implementation_version: implementation_version.into(),
            trust_snapshot_digest,
            trust_snapshot_sequence,
            trust_valid_from_cycle,
            trust_valid_until_cycle,
            configuration_digest,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn verifier_id(&self) -> &str {
        &self.verifier_id
    }

    pub fn implementation_id(&self) -> &str {
        &self.implementation_id
    }

    pub fn implementation_version(&self) -> &str {
        &self.implementation_version
    }

    pub fn trust_snapshot_digest(&self) -> [u8; 32] {
        self.trust_snapshot_digest
    }

    pub fn trust_snapshot_sequence(&self) -> u64 {
        self.trust_snapshot_sequence
    }

    pub fn trust_valid_from_cycle(&self) -> u64 {
        self.trust_valid_from_cycle
    }

    pub fn trust_valid_until_cycle(&self) -> u64 {
        self.trust_valid_until_cycle
    }

    pub fn configuration_digest(&self) -> [u8; 32] {
        self.configuration_digest
    }

    pub fn is_fresh_at(&self, cycle: u64) -> bool {
        cycle >= self.trust_valid_from_cycle && cycle < self.trust_valid_until_cycle
    }

    fn validate(&self) -> Result<(), RestartVerifierProvenanceError> {
        validate_identifier(
            &self.verifier_id,
            MAX_RESTART_VERIFIER_ID_BYTES,
            RestartVerifierProfileField::VerifierId,
        )?;
        validate_identifier(
            &self.implementation_id,
            MAX_RESTART_VERIFIER_IMPLEMENTATION_BYTES,
            RestartVerifierProfileField::ImplementationId,
        )?;
        validate_identifier(
            &self.implementation_version,
            MAX_RESTART_VERIFIER_VERSION_BYTES,
            RestartVerifierProfileField::ImplementationVersion,
        )?;
        if self.trust_snapshot_sequence == 0 {
            return Err(RestartVerifierProvenanceError::InvalidTrustSnapshotSequence);
        }
        if self.trust_valid_from_cycle >= self.trust_valid_until_cycle {
            return Err(RestartVerifierProvenanceError::InvalidTrustValidityWindow);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartVerifierProfileField {
    VerifierId,
    ImplementationId,
    ImplementationVersion,
}

/// Extension of the EKM-046 verifier boundary with an auditable verifier profile.
///
/// Implementations are responsible for ensuring that `restart_verifier_profile()`
/// describes the trust/configuration state actually used by
/// `verify_restart_anchor_evidence()`. EKM-049 checks that the profile remains
/// stable across the call, but cannot detect a provider that lies about itself.
pub trait ProfiledRestartAnchorEvidenceVerifierV1: RestartAnchorEvidenceVerifierV1 {
    fn restart_verifier_profile(&self) -> Result<RestartVerifierProfileV1, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartVerifierProfileDigestV1([u8; 32]);

impl RestartVerifierProfileDigestV1 {
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
pub struct RestartVerifierProvenanceDigestV1([u8; 32]);

impl RestartVerifierProvenanceDigestV1 {
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
pub struct RestartVerifierProvenanceReceiptV1 {
    version: RestartVerifierProvenanceVersion,
    admission_review_digest: EpistemicRestartAdmissionReviewDigest,
    verifier_profile: RestartVerifierProfileV1,
    verifier_profile_digest: RestartVerifierProfileDigestV1,
    observed_at_cycle: u64,
    profile_stable_across_verification: bool,
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
    provenance_digest: RestartVerifierProvenanceDigestV1,
}

impl RestartVerifierProvenanceReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_review(
        snapshot: &EpistemicRestartWireSnapshotV2,
        policy: &RestartAnchorTrustPolicyV1,
        candidate_anchor_evidence: &RestartAnchorEvidenceV1,
        verifier: &dyn ProfiledRestartAnchorEvidenceVerifierV1,
        trusted_tracker: &RestartAnchorTrackerV1,
        observed_at_cycle: u64,
    ) -> Result<(EpistemicRestartAdmissionReviewReceiptV1, Self), RestartVerifierProvenanceError>
    {
        let profile_before = verifier
            .restart_verifier_profile()
            .map_err(RestartVerifierProvenanceError::ProfileProvider)?;
        profile_before.validate()?;
        if !profile_before.is_fresh_at(observed_at_cycle) {
            return Err(RestartVerifierProvenanceError::TrustSnapshotNotFresh {
                observed_at_cycle,
                valid_from_cycle: profile_before.trust_valid_from_cycle,
                valid_until_cycle: profile_before.trust_valid_until_cycle,
            });
        }

        let review = EpistemicRestartAdmissionReviewReceiptV1::validate_and_review(
            snapshot,
            policy,
            candidate_anchor_evidence,
            verifier,
            trusted_tracker,
            observed_at_cycle,
        )
        .map_err(RestartVerifierProvenanceError::AdmissionReview)?;

        let profile_after = verifier
            .restart_verifier_profile()
            .map_err(RestartVerifierProvenanceError::ProfileProvider)?;
        profile_after.validate()?;
        if profile_before != profile_after {
            return Err(RestartVerifierProvenanceError::ProfileChangedDuringVerification);
        }
        if !profile_after.is_fresh_at(observed_at_cycle) {
            return Err(RestartVerifierProvenanceError::TrustSnapshotNotFresh {
                observed_at_cycle,
                valid_from_cycle: profile_after.trust_valid_from_cycle,
                valid_until_cycle: profile_after.trust_valid_until_cycle,
            });
        }

        let profile_digest = digest_restart_verifier_profile(&profile_after)?;
        let mut receipt = Self {
            version: RestartVerifierProvenanceVersion::V1,
            admission_review_digest: review.review_digest(),
            verifier_profile: profile_after,
            verifier_profile_digest: profile_digest,
            observed_at_cycle,
            profile_stable_across_verification: true,
            quarantine_construction_authorized: false,
            activation_authorized: false,
            provenance_digest: RestartVerifierProvenanceDigestV1([0; 32]),
        };
        receipt.provenance_digest = digest_provenance_receipt(&receipt)?;
        Ok((review, receipt))
    }

    pub fn version(&self) -> RestartVerifierProvenanceVersion {
        self.version
    }

    pub fn admission_review_digest(&self) -> EpistemicRestartAdmissionReviewDigest {
        self.admission_review_digest
    }

    pub fn verifier_profile(&self) -> &RestartVerifierProfileV1 {
        &self.verifier_profile
    }

    pub fn verifier_profile_digest(&self) -> RestartVerifierProfileDigestV1 {
        self.verifier_profile_digest
    }

    pub fn observed_at_cycle(&self) -> u64 {
        self.observed_at_cycle
    }

    pub fn profile_stable_across_verification(&self) -> bool {
        self.profile_stable_across_verification
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn provenance_digest(&self) -> RestartVerifierProvenanceDigestV1 {
        self.provenance_digest
    }
}

pub fn digest_restart_verifier_profile(
    profile: &RestartVerifierProfileV1,
) -> Result<RestartVerifierProfileDigestV1, RestartVerifierProvenanceError> {
    profile.validate()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-verifier-profile-v1");
    hash_bytes(&mut hasher, profile.verifier_id.as_bytes())?;
    hash_bytes(&mut hasher, profile.implementation_id.as_bytes())?;
    hash_bytes(&mut hasher, profile.implementation_version.as_bytes())?;
    hasher.update(&profile.trust_snapshot_digest);
    hasher.update(&profile.trust_snapshot_sequence.to_le_bytes());
    hasher.update(&profile.trust_valid_from_cycle.to_le_bytes());
    hasher.update(&profile.trust_valid_until_cycle.to_le_bytes());
    hasher.update(&profile.configuration_digest);
    Ok(RestartVerifierProfileDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_provenance_receipt(
    receipt: &RestartVerifierProvenanceReceiptV1,
) -> Result<RestartVerifierProvenanceDigestV1, RestartVerifierProvenanceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-verifier-provenance-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.admission_review_digest.as_bytes());
    hasher.update(&receipt.verifier_profile_digest.as_bytes());
    hasher.update(&receipt.observed_at_cycle.to_le_bytes());
    hasher.update(&[u8::from(receipt.profile_stable_across_verification)]);
    hasher.update(&[u8::from(receipt.quarantine_construction_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    Ok(RestartVerifierProvenanceDigestV1(*hasher.finalize().as_bytes()))
}

fn validate_identifier(
    value: &str,
    maximum: usize,
    field: RestartVerifierProfileField,
) -> Result<(), RestartVerifierProvenanceError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > maximum {
        return Err(RestartVerifierProvenanceError::InvalidProfileIdentifier {
            field,
            actual_length: value.len(),
            maximum,
        });
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), RestartVerifierProvenanceError> {
    let length = u64::try_from(bytes.len()).map_err(|_| RestartVerifierProvenanceError::LengthOverflow)?;
    hasher.update(&length.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum RestartVerifierProvenanceError {
    InvalidProfileIdentifier {
        field: RestartVerifierProfileField,
        actual_length: usize,
        maximum: usize,
    },
    InvalidTrustSnapshotSequence,
    InvalidTrustValidityWindow,
    TrustSnapshotNotFresh {
        observed_at_cycle: u64,
        valid_from_cycle: u64,
        valid_until_cycle: u64,
    },
    ProfileProvider(String),
    ProfileChangedDuringVerification,
    AdmissionReview(EpistemicRestartAdmissionReviewError),
    LengthOverflow,
}

impl fmt::Display for RestartVerifierProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart verifier provenance rejected: {self:?}")
    }
}

impl Error for RestartVerifierProvenanceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
        EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
        EpistemicRestartCapsuleV2, EpistemicRestartValidationReceiptV1, EpistemicRestartWireV2,
        EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind, EvidencePolarity,
        RestartAnchorEvidenceKindV1, RestartAnchorEvidenceV1, RestartAnchorStatementV1,
        RestartAnchorTrackerV1,
    };
    use crate::knowledge::epistemic_restart_anchor_policy::RestartAnchorTrustPolicyV1;
    use std::cell::Cell;

    struct StableVerifier;

    impl RestartAnchorEvidenceVerifierV1 for StableVerifier {
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

    impl ProfiledRestartAnchorEvidenceVerifierV1 for StableVerifier {
        fn restart_verifier_profile(&self) -> Result<RestartVerifierProfileV1, String> {
            RestartVerifierProfileV1::new(
                "verifier-a",
                "restart-verifier",
                "1.0.0",
                [7; 32],
                9,
                0,
                100,
                [8; 32],
            )
            .map_err(|error| format!("{error:?}"))
        }
    }

    struct SwappingVerifier {
        reads: Cell<u64>,
    }

    impl RestartAnchorEvidenceVerifierV1 for SwappingVerifier {
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

    impl ProfiledRestartAnchorEvidenceVerifierV1 for SwappingVerifier {
        fn restart_verifier_profile(&self) -> Result<RestartVerifierProfileV1, String> {
            let read = self.reads.get();
            self.reads.set(read + 1);
            RestartVerifierProfileV1::new(
                "verifier-a",
                "restart-verifier",
                if read == 0 { "1.0.0" } else { "1.0.1" },
                [7; 32],
                9,
                0,
                100,
                [8; 32],
            )
            .map_err(|error| format!("{error:?}"))
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

    fn validation_receipt(snapshot: &EpistemicRestartWireSnapshotV2) -> EpistemicRestartValidationReceiptV1 {
        EpistemicRestartValidationReceiptV1::validate_and_capture(snapshot).unwrap()
    }

    fn policy() -> RestartAnchorTrustPolicyV1 {
        RestartAnchorTrustPolicyV1::new(
            vec!["authority-a".into()],
            vec![RestartAnchorEvidenceKindV1::SignedCheckpoint],
            20,
            10,
        )
        .unwrap()
    }

    fn evidence(
        receipt: &EpistemicRestartValidationReceiptV1,
        sequence: u64,
        previous: Option<super::super::epistemic_restart_anchor::RestartAnchorDigestV1>,
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

    fn trusted_tracker() -> RestartAnchorTrackerV1 {
        let snapshot = snapshot(4, "old");
        let receipt = validation_receipt(&snapshot);
        let evidence = evidence(&receipt, 1, None);
        let verifier = StableVerifier;
        let verified = super::super::epistemic_restart_anchor_policy::verify_restart_anchor_evidence_under_policy(
            &policy(),
            &evidence,
            &receipt,
            4,
            &verifier,
        )
        .unwrap();
        let mut tracker = RestartAnchorTrackerV1::default();
        tracker.accept(&verified, 4).unwrap();
        tracker
    }

    #[test]
    fn stable_profile_is_bound_to_exact_admission_review() {
        let tracker = trusted_tracker();
        let snapshot = snapshot(5, "new");
        let receipt = validation_receipt(&snapshot);
        let evidence = evidence(&receipt, 2, tracker.latest_anchor_digest());
        let verifier = StableVerifier;
        let (review, provenance) = RestartVerifierProvenanceReceiptV1::validate_and_review(
            &snapshot,
            &policy(),
            &evidence,
            &verifier,
            &tracker,
            5,
        )
        .unwrap();
        assert_eq!(provenance.admission_review_digest(), review.review_digest());
        assert!(provenance.profile_stable_across_verification());
        assert_eq!(provenance.verifier_profile().trust_snapshot_sequence(), 9);
        assert!(!provenance.quarantine_construction_authorized());
        assert!(!provenance.activation_authorized());
    }

    #[test]
    fn profile_swap_during_verification_fails_closed() {
        let tracker = trusted_tracker();
        let snapshot = snapshot(5, "new");
        let receipt = validation_receipt(&snapshot);
        let evidence = evidence(&receipt, 2, tracker.latest_anchor_digest());
        let verifier = SwappingVerifier { reads: Cell::new(0) };
        assert_eq!(
            RestartVerifierProvenanceReceiptV1::validate_and_review(
                &snapshot,
                &policy(),
                &evidence,
                &verifier,
                &tracker,
                5,
            )
            .unwrap_err(),
            RestartVerifierProvenanceError::ProfileChangedDuringVerification
        );
    }

    #[test]
    fn stale_trust_snapshot_is_rejected_before_review() {
        struct Stale;
        impl RestartAnchorEvidenceVerifierV1 for Stale {
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
        impl ProfiledRestartAnchorEvidenceVerifierV1 for Stale {
            fn restart_verifier_profile(&self) -> Result<RestartVerifierProfileV1, String> {
                RestartVerifierProfileV1::new(
                    "verifier-a",
                    "restart-verifier",
                    "1.0.0",
                    [7; 32],
                    9,
                    0,
                    5,
                    [8; 32],
                )
                .map_err(|error| format!("{error:?}"))
            }
        }

        let tracker = trusted_tracker();
        let snapshot = snapshot(5, "new");
        let receipt = validation_receipt(&snapshot);
        let evidence = evidence(&receipt, 2, tracker.latest_anchor_digest());
        assert!(matches!(
            RestartVerifierProvenanceReceiptV1::validate_and_review(
                &snapshot,
                &policy(),
                &evidence,
                &Stale,
                &tracker,
                5,
            ),
            Err(RestartVerifierProvenanceError::TrustSnapshotNotFresh { .. })
        ));
    }
}
