//! Deterministic RFC 9942 receipt-selection audit.
//!
//! RFC 9942 receipt arrays are priority ordered.  This module turns the
//! selection process into a first-class, content-addressed decision record.
//! It deliberately separates:
//!
//! * the exact ordered collection bytes,
//! * candidate receipt wire identities,
//! * the explicit selection policy,
//! * candidate verification outcomes, and
//! * the selected candidate.
//!
//! A candidate after the selected receipt is intentionally marked
//! `NotEvaluatedAfterSelection`; the decision must not imply that lower-priority
//! receipts were tested when the policy short-circuited at the first success.

use blake3::Hasher;
use sha2::{Digest, Sha256};

use crate::semantic_evidence_vds::{
    Rfc9942ReceiptCollection, Rfc9942ReceiptEnvelope, Rfc9942SignaturePayload,
    Rfc9942SignatureWithReceipts, Rfc9942VerifiedSignatureWithReceipt, Rfc9942VdpError,
    MAX_RFC9942_RECEIPTS,
};

pub const MAX_SELECTION_CANDIDATES: usize = MAX_RFC9942_RECEIPTS;
pub const POLICY_ID: &str = "rfc9942/priority-first-valid-v1";
pub const POLICY_VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/rfc9942-receipt-selection-decision-v1";

/// Stable reason code for an unsuccessful candidate evaluation.
///
/// The error payload itself is intentionally not serialized into the decision
/// identity; the policy version fixes this classification vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ReceiptSelectionRejection {
    InvalidStructure = 1,
    InvalidEncoding = 2,
    InvalidProof = 3,
    WrongProofKind = 4,
    NoMatchingProof = 5,
    InvalidSignature = 6,
    InvalidPublicKey = 7,
    UnsupportedAlgorithm = 8,
    DetachedPayloadRequired = 9,
    VdsMismatch = 10,
    ResourceLimitExceeded = 11,
    Other = 255,
}

impl ReceiptSelectionRejection {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub fn from_error(error: &Rfc9942VdpError) -> Self {
        match error {
            Rfc9942VdpError::InvalidStructure => Self::InvalidStructure,
            Rfc9942VdpError::InvalidEncoding | Rfc9942VdpError::TrailingBytes => {
                Self::InvalidEncoding
            }
            Rfc9942VdpError::InvalidProof(_) => Self::InvalidProof,
            Rfc9942VdpError::WrongProofKind => Self::WrongProofKind,
            Rfc9942VdpError::NoMatchingProof => Self::NoMatchingProof,
            Rfc9942VdpError::InvalidEs256Signature
            | Rfc9942VdpError::InvalidEd25519Signature => Self::InvalidSignature,
            Rfc9942VdpError::InvalidEs256PublicKey
            | Rfc9942VdpError::InvalidEd25519PublicKey
            | Rfc9942VdpError::InvalidEs256CoseKey => Self::InvalidPublicKey,
            Rfc9942VdpError::UnsupportedSignatureAlgorithm(_)
            | Rfc9942VdpError::Es256CoseKeyAlgorithmMismatch
            | Rfc9942VdpError::Es256CoseKeyOperationNotPermitted
            | Rfc9942VdpError::Es256PrivateKeyMaterial => Self::UnsupportedAlgorithm,
            Rfc9942VdpError::DetachedPayloadRequired => Self::DetachedPayloadRequired,
            Rfc9942VdpError::VdsMismatch(_) => Self::VdsMismatch,
            Rfc9942VdpError::ResourceLimitExceeded
            | Rfc9942VdpError::ReceiptCollectionResourceLimitExceeded
            | Rfc9942VdpError::SignaturePayloadResourceLimitExceeded => {
                Self::ResourceLimitExceeded
            }
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReceiptSelectionCandidateStatus {
    Selected,
    Rejected(ReceiptSelectionRejection),
    NotEvaluatedAfterSelection,
}

impl ReceiptSelectionCandidateStatus {
    fn tag(self) -> u8 {
        match self {
            Self::Selected => 1,
            Self::Rejected(_) => 2,
            Self::NotEvaluatedAfterSelection => 3,
        }
    }

    fn rejection_code(self) -> u8 {
        match self {
            Self::Rejected(reason) => reason.code(),
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReceiptSelectionCandidate {
    pub index: u32,
    pub receipt_sha256: [u8; 32],
    pub status: ReceiptSelectionCandidateStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptSelectionDecision {
    pub collection_sha256: [u8; 32],
    pub collection_len: u32,
    pub policy_id: &'static str,
    pub policy_version: u16,
    pub selected_index: Option<u32>,
    pub selected_receipt_sha256: Option<[u8; 32]>,
    pub candidates: Vec<ReceiptSelectionCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptSelectionDecisionError {
    EmptyCollection,
    CollectionTooLarge,
    CandidateCountMismatch,
    CandidateIndexMismatch,
    PolicyMismatch,
    CollectionDigestZero,
    CandidateDigestZero,
    SelectionDigestZero,
    SelectedCandidateMismatch,
    RejectedAfterSelection,
    UnselectedCandidateMarkedNotEvaluated,
}

impl ReceiptSelectionDecision {
    /// Validate the decision's internal referential invariants before it crosses
    /// into a durable evidence system.
    pub fn validate(&self) -> Result<(), ReceiptSelectionDecisionError> {
        if self.collection_len == 0 {
            return Err(ReceiptSelectionDecisionError::EmptyCollection);
        }
        if self.collection_len as usize > MAX_SELECTION_CANDIDATES {
            return Err(ReceiptSelectionDecisionError::CollectionTooLarge);
        }
        if self.candidates.len() != self.collection_len as usize {
            return Err(ReceiptSelectionDecisionError::CandidateCountMismatch);
        }
        if self.policy_id != POLICY_ID || self.policy_version != POLICY_VERSION {
            return Err(ReceiptSelectionDecisionError::PolicyMismatch);
        }
        if self.collection_sha256 == [0; 32] {
            return Err(ReceiptSelectionDecisionError::CollectionDigestZero);
        }

        let mut selected_count = 0usize;
        for (expected_index, candidate) in self.candidates.iter().enumerate() {
            if candidate.index as usize != expected_index {
                return Err(ReceiptSelectionDecisionError::CandidateIndexMismatch);
            }
            if candidate.receipt_sha256 == [0; 32] {
                return Err(ReceiptSelectionDecisionError::CandidateDigestZero);
            }
            match candidate.status {
                ReceiptSelectionCandidateStatus::Selected => {
                    selected_count += 1;
                    if Some(candidate.index) != self.selected_index
                        || Some(candidate.receipt_sha256) != self.selected_receipt_sha256
                    {
                        return Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch);
                    }
                }
                ReceiptSelectionCandidateStatus::Rejected(_) => {
                    if self.selected_index.is_some_and(|index| candidate.index > index) {
                        return Err(ReceiptSelectionDecisionError::RejectedAfterSelection);
                    }
                }
                ReceiptSelectionCandidateStatus::NotEvaluatedAfterSelection => {
                    if self.selected_index.is_none()
                        || candidate.index <= self.selected_index.unwrap()
                    {
                        return Err(ReceiptSelectionDecisionError::UnselectedCandidateMarkedNotEvaluated);
                    }
                }
            }
        }

        match self.selected_index {
            Some(index) => {
                if index >= self.collection_len
                    || selected_count != 1
                    || self.selected_receipt_sha256.is_none()
                    || self.selected_receipt_sha256 == Some([0; 32])
                {
                    return Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch);
                }
            }
            None => {
                if selected_count != 0 || self.selected_receipt_sha256.is_some() {
                    return Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch);
                }
            }
        }

        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            DOMAIN.len() + self.candidates.len() * 37 + 128,
        );
        put_bytes(&mut out, DOMAIN);
        put_u16(&mut out, 1);
        out.extend_from_slice(&self.collection_sha256);
        put_u32(&mut out, self.collection_len);
        put_string(&mut out, self.policy_id);
        put_u16(&mut out, self.policy_version);

        match (self.selected_index, self.selected_receipt_sha256) {
            (Some(index), Some(digest)) => {
                out.push(1);
                put_u32(&mut out, index);
                out.extend_from_slice(&digest);
            }
            (None, None) => out.push(0),
            _ => {
                // This state is not constructible through the evaluator, but a
                // stable malformed marker keeps the encoder total.
                out.push(2);
            }
        }

        put_u32(&mut out, self.candidates.len() as u32);
        for candidate in &self.candidates {
            put_u32(&mut out, candidate.index);
            out.extend_from_slice(&candidate.receipt_sha256);
            out.push(candidate.status.tag());
            out.push(candidate.status.rejection_code());
        }
        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let canonical = self.canonical_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&(DOMAIN.len() as u64).to_be_bytes());
        hasher.update(DOMAIN);
        hasher.update(&POLICY_VERSION.to_be_bytes());
        hasher.update(&canonical);
        *hasher.finalize().as_bytes()
    }

    /// Digest an internally consistent selection decision.
    ///
    /// This is the boundary for durable publication: malformed
    /// candidate/selection relationships are rejected first.
    pub fn validated_digest(&self) -> Result<[u8; 32], ReceiptSelectionDecisionError> {
        self.validate()?;
        Ok(self.digest())
    }
}

/// Evaluate receipts strictly in RFC 9942 priority order.
///
/// The callback performs cryptographic/semantic verification for exactly one
/// candidate.  The first success is selected; lower-priority candidates are
/// recorded as not evaluated rather than falsely labeled rejected.

#[cfg(feature = "semantic-receipts")]
impl Rfc9942SignatureWithReceipts {
    /// Verify the outer signature and choose the first inclusion Receipt that
    /// passes the supplied cryptographic verification context.
    ///
    /// The returned decision records every higher-priority rejection and marks
    /// lower-priority candidates as not evaluated once a valid receipt wins.
    /// The final verified capability is rechecked through the existing
    /// composition verifier, keeping its construction boundary intact.
    pub fn verify_es256_inclusion_priority_first_valid_receipt_state(
        &self,
        receipt_public_key: &[u8],
        outer_public_key: &[u8],
        receipt_external_aad: &[u8],
        outer_external_aad: &[u8],
        detached_outer_payload: Option<&[u8]>,
    ) -> Result<
        (Rfc9942VerifiedSignatureWithReceipt, ReceiptSelectionDecision),
        Rfc9942VdpError,
    > {
        let payload = match (self.payload(), detached_outer_payload) {
            (
                Rfc9942SignaturePayload::Attached(bytes),
                None,
            ) => bytes.as_slice(),
            (
                Rfc9942SignaturePayload::Attached(_),
                Some(_),
            ) => return Err(Rfc9942VdpError::InvalidStructure),
            (
                Rfc9942SignaturePayload::Detached,
                Some(bytes),
            ) => bytes,
            (
                Rfc9942SignaturePayload::Detached,
                None,
            ) => return Err(Rfc9942VdpError::DetachedPayloadRequired),
        };

        self.verify_es256(outer_public_key, outer_external_aad, detached_outer_payload)?;

        let collection = self.receipts().ok_or(Rfc9942VdpError::ReceiptsMissing)?;
        let decision = evaluate_priority_first_valid(
            collection,
            |_index, receipt| {
                receipt
                    .verify_es256_inclusion_state(
                        payload,
                        receipt_public_key,
                        receipt_external_aad,
                        None,
                    )
                    .map(|_| ())
            },
        );

        let selected_index = decision
            .selected_index
            .ok_or(Rfc9942VdpError::NoMatchingProof)? as usize;

        let verified = self.verify_es256_inclusion_receipt_state(
            selected_index,
            receipt_public_key,
            outer_public_key,
            receipt_external_aad,
            outer_external_aad,
            detached_outer_payload,
        )?;

        Ok((verified, decision))
    }
}

pub fn evaluate_priority_first_valid<F>(
    collection: &Rfc9942ReceiptCollection,
    mut verify: F,
) -> ReceiptSelectionDecision
where
    F: FnMut(usize, &Rfc9942ReceiptEnvelope) -> Result<(), Rfc9942VdpError>,
{
    // Hash the exact wire representation owned by the collection. Callers
    // cannot pair one collection with unrelated bytes.
    let collection_bytes = collection
        .serialized_bytes()
        .map_or_else(|| collection.to_cbor(), ToOwned::to_owned);

    let mut candidates = Vec::with_capacity(collection.len());
    let mut selected_index = None;
    let mut selected_receipt_sha256 = None;

    for (index, receipt) in collection.iter().enumerate() {
        let receipt_bytes = collection
            .serialized_receipt_bytes(index)
            .unwrap_or_else(|| receipt.to_cbor());
        let receipt_sha256 = sha256(receipt_bytes);

        if selected_index.is_some() {
            candidates.push(ReceiptSelectionCandidate {
                index: index as u32,
                receipt_sha256,
                status: ReceiptSelectionCandidateStatus::NotEvaluatedAfterSelection,
            });
            continue;
        }

        match verify(index, receipt) {
            Ok(()) => {
                selected_index = Some(index as u32);
                selected_receipt_sha256 = Some(receipt_sha256);
                candidates.push(ReceiptSelectionCandidate {
                    index: index as u32,
                    receipt_sha256,
                    status: ReceiptSelectionCandidateStatus::Selected,
                });
            }
            Err(error) => candidates.push(ReceiptSelectionCandidate {
                index: index as u32,
                receipt_sha256,
                status: ReceiptSelectionCandidateStatus::Rejected(
                    ReceiptSelectionRejection::from_error(&error),
                ),
            }),
        }
    }

    let decision = ReceiptSelectionDecision {
        collection_sha256: sha256(collection_bytes),
        collection_len: collection.len() as u32,
        policy_id: POLICY_ID,
        policy_version: POLICY_VERSION,
        selected_index,
        selected_receipt_sha256,
        candidates,
    };
    debug_assert!(decision.validate().is_ok());
    decision
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
    out.extend_from_slice(value);
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_evidence_vds::{
        Rfc9942ProofKind, Rfc9942ReceiptPayload, Rfc9942Vdp, Rfc9162InclusionProof,
        COSE_EDDSA_ALGORITHM_ID, COSE_ES256_ALGORITHM_ID,
    };

    fn collection() -> Rfc9942ReceiptCollection {
        let proof = Rfc9162InclusionProof::new(2, 0, vec![[0x11; 32]]).to_cbor();
        let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
        Rfc9942ReceiptCollection::new(vec![
            Rfc9942ReceiptEnvelope::new(
                COSE_ES256_ALGORITHM_ID,
                vdp.clone(),
                Rfc9942ReceiptPayload::Attached([1; 32]),
                vec![1; 64],
            )
            .unwrap(),
            Rfc9942ReceiptEnvelope::new(
                COSE_EDDSA_ALGORITHM_ID,
                vdp,
                Rfc9942ReceiptPayload::Detached,
                vec![2; 64],
            )
            .unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn first_valid_selects_in_priority_order() {
        let collection = collection();
        let decision = evaluate_priority_first_valid(
            &collection,
            |index, _| {
                if index == 1 {
                    Ok(())
                } else {
                    Err(Rfc9942VdpError::InvalidEs256Signature)
                }
            },
        );
        assert_eq!(decision.selected_index, Some(1));
        assert_eq!(
            decision.candidates[0].status,
            ReceiptSelectionCandidateStatus::Rejected(
                ReceiptSelectionRejection::InvalidSignature
            )
        );
        assert_eq!(
            decision.candidates[1].status,
            ReceiptSelectionCandidateStatus::Selected
        );
    }

    #[test]
    fn decision_validation_accepts_evaluator_output() {
        let collection = collection();
        let decision = evaluate_priority_first_valid(&collection, |_index, _| Ok(()));
        assert_eq!(decision.validate(), Ok(()));
    }

    #[test]
    fn validated_digest_refuses_malformed_decision() {
        let collection = collection();
        let mut decision = evaluate_priority_first_valid(&collection, |_index, _| Ok(()));
        decision.selected_receipt_sha256 = Some([0xAA; 32]);
        assert_eq!(
            decision.validated_digest(),
            Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch)
        );
    }

    #[test]
    fn decision_validation_rejects_oversized_collection() {
        let mut decision = ReceiptSelectionDecision {
            collection_sha256: [1; 32],
            collection_len: (MAX_SELECTION_CANDIDATES as u32) + 1,
            policy_id: POLICY_ID,
            policy_version: POLICY_VERSION,
            selected_index: None,
            selected_receipt_sha256: None,
            candidates: Vec::new(),
        };
        decision.candidates = (0..decision.collection_len)
            .map(|index| ReceiptSelectionCandidate {
                index,
                receipt_sha256: [1; 32],
                status: ReceiptSelectionCandidateStatus::Rejected(
                    ReceiptSelectionRejection::NoMatchingProof,
                ),
            })
            .collect();
        assert_eq!(
            decision.validate(),
            Err(ReceiptSelectionDecisionError::CollectionTooLarge)
        );
    }

    #[test]
    fn decision_validation_rejects_zero_candidate_digest() {
        let collection = collection();
        let mut decision = evaluate_priority_first_valid(&collection, |_index, _| Ok(()));
        decision.candidates[1].receipt_sha256 = [0; 32];
        assert_eq!(
            decision.validate(),
            Err(ReceiptSelectionDecisionError::CandidateDigestZero)
        );
    }

    #[test]
    fn decision_validation_rejects_mismatched_selected_digest() {
        let collection = collection();
        let mut decision = evaluate_priority_first_valid(&collection, |_index, _| Ok(()));
        decision.selected_receipt_sha256 = Some([0xAA; 32]);
        assert_eq!(
            decision.validate(),
            Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch)
        );
    }

    #[test]
    fn decision_validation_rejects_unjustified_short_circuit_tail() {
        let collection = collection();
        let mut decision = evaluate_priority_first_valid(&collection, |_index, _| Ok(()));
        decision.candidates[1].status = ReceiptSelectionCandidateStatus::Rejected(
            ReceiptSelectionRejection::NoMatchingProof,
        );
        assert_eq!(
            decision.validate(),
            Err(ReceiptSelectionDecisionError::RejectedAfterSelection)
        );
    }

    #[test]
    fn identity_mutations_change_or_fail_closed() {
        let collection = collection();
        let baseline = evaluate_priority_first_valid(
            &collection,
            |index, _| if index == 1 {
                Ok(())
            } else {
                Err(Rfc9942VdpError::InvalidEs256Signature)
            },
        );
        let baseline_digest = baseline.validated_digest().unwrap();

        let mut rejection_reason = baseline.clone();
        rejection_reason.candidates[0].status =
            ReceiptSelectionCandidateStatus::Rejected(ReceiptSelectionRejection::NoMatchingProof);
        assert_eq!(rejection_reason.validate(), Ok(()));
        assert_ne!(rejection_reason.digest(), baseline_digest);

        let mut policy = baseline.clone();
        policy.policy_id = "rfc9942/priority-first-valid-v2";
        assert_eq!(
            policy.validated_digest(),
            Err(ReceiptSelectionDecisionError::PolicyMismatch)
        );
        assert_ne!(policy.digest(), baseline_digest);

        let mut policy_version = baseline.clone();
        policy_version.policy_version = POLICY_VERSION + 1;
        assert_eq!(
            policy_version.validated_digest(),
            Err(ReceiptSelectionDecisionError::PolicyMismatch)
        );
        assert_ne!(policy_version.digest(), baseline_digest);

        let mut collection_digest = baseline.clone();
        collection_digest.collection_sha256 = [0; 32];
        assert_eq!(
            collection_digest.validated_digest(),
            Err(ReceiptSelectionDecisionError::CollectionDigestZero)
        );
        assert_ne!(collection_digest.digest(), baseline_digest);

        let mut candidate_digest = baseline.clone();
        candidate_digest.candidates[0].receipt_sha256 = [0; 32];
        assert_eq!(
            candidate_digest.validated_digest(),
            Err(ReceiptSelectionDecisionError::CandidateDigestZero)
        );
        assert_ne!(candidate_digest.digest(), baseline_digest);

        let mut selected_digest = baseline.clone();
        selected_digest.selected_receipt_sha256 = Some([0xAB; 32]);
        assert_eq!(
            selected_digest.validated_digest(),
            Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch)
        );
        assert_ne!(selected_digest.digest(), baseline_digest);

        let mut selected_index = baseline.clone();
        selected_index.selected_index = Some(0);
        assert_eq!(
            selected_index.validated_digest(),
            Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch)
        );
        assert_ne!(selected_index.digest(), baseline_digest);

        let mut candidate_result = baseline.clone();
        candidate_result.candidates[0].status = ReceiptSelectionCandidateStatus::Selected;
        assert_eq!(
            candidate_result.validated_digest(),
            Err(ReceiptSelectionDecisionError::SelectedCandidateMismatch)
        );
        assert_ne!(candidate_result.digest(), baseline_digest);
    }

    #[test]
    fn candidates_after_selection_are_not_reported_as_rejected() {
        let collection = collection();
        let decision = evaluate_priority_first_valid(&collection, |_index, _| {
            Ok(())
        });
        assert_eq!(
            decision.candidates[1].status,
            ReceiptSelectionCandidateStatus::NotEvaluatedAfterSelection
        );
    }

    #[test]
    fn no_valid_receipt_is_an_explicit_selection_failure() {
        let collection = collection();
        let decision = evaluate_priority_first_valid(
            &collection,
            |_index, _| Err(Rfc9942VdpError::NoMatchingProof),
        );
        assert_eq!(decision.selected_index, None);
        assert!(decision.selected_receipt_sha256.is_none());
        assert!(decision
            .candidates
            .iter()
            .all(|candidate| matches!(
                candidate.status,
                ReceiptSelectionCandidateStatus::Rejected(
                    ReceiptSelectionRejection::NoMatchingProof
                )
            )));
    }

    #[test]
    fn changing_rejection_reason_changes_decision_digest() {
        let collection = collection();
        let signature_failure = evaluate_priority_first_valid(
            &collection,
            |_index, _| Err(Rfc9942VdpError::InvalidEs256Signature),
        );
        let proof_failure = evaluate_priority_first_valid(
            &collection,
            |_index, _| Err(Rfc9942VdpError::NoMatchingProof),
        );
        assert_ne!(signature_failure.digest(), proof_failure.digest());
    }

    #[test]
    fn receipt_order_changes_decision_digest() {
        let collection = collection();
        let reversed = Rfc9942ReceiptCollection::new(
            collection.receipts().iter().cloned().rev().collect(),
        )
        .unwrap();
        let first = evaluate_priority_first_valid(
            &collection,
            |index, _| if index == 0 { Ok(()) } else { Err(Rfc9942VdpError::NoMatchingProof) },
        );
        let second = evaluate_priority_first_valid(
            &reversed,
            |index, _| if index == 0 { Ok(()) } else { Err(Rfc9942VdpError::NoMatchingProof) },
        );
        assert_ne!(first.collection_sha256, second.collection_sha256);
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn exact_selected_receipt_wire_bytes_are_fingerprinted() {
        let collection = collection();
        let first_bytes = collection.serialized_receipt_bytes(0).unwrap();
        assert_eq!(collection.serialized_receipt_bytes(0).unwrap(), first_bytes);
        let decision = evaluate_priority_first_valid(
            &collection,
            |index, _| if index == 0 { Ok(()) } else { Err(Rfc9942VdpError::NoMatchingProof) },
        );
        assert_eq!(decision.selected_receipt_sha256, Some(sha256(first_bytes)));
    }
}
