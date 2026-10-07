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
    Rfc9942ReceiptCollection, Rfc9942ReceiptEnvelope, Rfc9942VdpError,
};

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

impl ReceiptSelectionDecision {
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
}

/// Evaluate receipts strictly in RFC 9942 priority order.
///
/// The callback performs cryptographic/semantic verification for exactly one
/// candidate.  The first success is selected; lower-priority candidates are
/// recorded as not evaluated rather than falsely labeled rejected.
pub fn evaluate_priority_first_valid<F>(
    collection: &Rfc9942ReceiptCollection,
    collection_bytes: &[u8],
    mut verify: F,
) -> ReceiptSelectionDecision
where
    F: FnMut(usize, &Rfc9942ReceiptEnvelope) -> Result<(), Rfc9942VdpError>,
{
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

    ReceiptSelectionDecision {
        collection_sha256: sha256(collection_bytes),
        collection_len: collection.len() as u32,
        policy_id: POLICY_ID,
        policy_version: POLICY_VERSION,
        selected_index,
        selected_receipt_sha256,
        candidates,
    }
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
        let bytes = collection.to_cbor();
        let decision = evaluate_priority_first_valid(
            &collection,
            &bytes,
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
    fn candidates_after_selection_are_not_reported_as_rejected() {
        let collection = collection();
        let decision = evaluate_priority_first_valid(&collection, &collection.to_cbor(), |_index, _| {
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
            &collection.to_cbor(),
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
        let bytes = collection.to_cbor();
        let signature_failure = evaluate_priority_first_valid(
            &collection,
            &bytes,
            |_index, _| Err(Rfc9942VdpError::InvalidEs256Signature),
        );
        let proof_failure = evaluate_priority_first_valid(
            &collection,
            &bytes,
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
            &collection.to_cbor(),
            |index, _| if index == 0 { Ok(()) } else { Err(Rfc9942VdpError::NoMatchingProof) },
        );
        let second = evaluate_priority_first_valid(
            &reversed,
            &reversed.to_cbor(),
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
            &collection.to_cbor(),
            |index, _| if index == 0 { Ok(()) } else { Err(Rfc9942VdpError::NoMatchingProof) },
        );
        assert_eq!(decision.selected_receipt_sha256, Some(sha256(first_bytes)));
    }
}
