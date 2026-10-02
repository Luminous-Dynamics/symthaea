// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local append-only history witness for semantic transition evidence.
//!
//! This module deliberately stops short of being a transparency service or
//! Verifiable Data Structure (VDS). It provides deterministic sequencing and
//! tamper-evident linkage for locally retained evidence. Global
//! non-equivocation, inclusion/consistency proofs, receipts, signatures, and
//! consensus remain separate layers.

use crate::semantic_evidence_digest::{evidence_digest, EvidenceDigest, EvidenceDigestError};
use crate::semantic_transition::TransitionEvidence;

pub const ALGORITHM: &str = "BLAKE3-256";
pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-evidence-history";
pub const ENTRY_DOMAIN: &[u8] = b"symthaea-swarm/semantic-evidence-history-entry";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoryEntryCommitment([u8; 32]);

impl HistoryEntryCommitment {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceHistoryEntry {
    sequence: u64,
    evidence: TransitionEvidence,
    evidence_digest: EvidenceDigest,
    previous_entry_commitment: Option<HistoryEntryCommitment>,
    commitment: HistoryEntryCommitment,
}

impl EvidenceHistoryEntry {
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn evidence(&self) -> &TransitionEvidence { &self.evidence }
    pub fn evidence_digest(&self) -> EvidenceDigest { self.evidence_digest }
    pub fn previous_entry_commitment(&self) -> Option<HistoryEntryCommitment> { self.previous_entry_commitment }
    pub fn commitment(&self) -> HistoryEntryCommitment { self.commitment }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum EvidenceHistoryError {
    #[error("history sequence overflow")]
    SequenceOverflow,
    #[error("history entry sequence must be {expected}, got {actual}")]
    SequenceMismatch { expected: u64, actual: u64 },
    #[error("history entry evidence digest is inconsistent")]
    EvidenceDigestMismatch,
    #[error("history entry previous commitment is inconsistent")]
    PreviousCommitmentMismatch,
    #[error("history entry commitment is inconsistent")]
    CommitmentMismatch,
    #[error("evidence digest failed: {0}")]
    EvidenceDigest(#[from] EvidenceDigestError),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvidenceHistory {
    entries: Vec<EvidenceHistoryEntry>,
}

impl EvidenceHistory {
    pub fn new() -> Self { Self::default() }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn entries(&self) -> &[EvidenceHistoryEntry] { &self.entries }
    pub fn head_commitment(&self) -> Option<HistoryEntryCommitment> {
        self.entries.last().map(EvidenceHistoryEntry::commitment)
    }

    /// Append one complete evidence envelope to the local history.
    ///
    /// Sequence numbers begin at zero. A new entry commits to the prior entry
    /// commitment, making deletion, reordering, or mutation detectable when
    /// the retained history is subsequently verified.
    pub fn append(
        &mut self,
        evidence: TransitionEvidence,
    ) -> Result<&EvidenceHistoryEntry, EvidenceHistoryError> {
        let sequence = u64::try_from(self.entries.len())
            .map_err(|_| EvidenceHistoryError::SequenceOverflow)?;
        let evidence_digest = evidence_digest(&evidence)?;
        let previous_entry_commitment = self.head_commitment();
        let commitment = entry_commitment(sequence, &evidence_digest, previous_entry_commitment);

        self.entries.push(EvidenceHistoryEntry {
            sequence,
            evidence,
            evidence_digest,
            previous_entry_commitment,
            commitment,
        });

        Ok(self.entries.last().expect("entry was just pushed"))
    }

    /// Verify the complete local history from its first entry through its current head.
    pub fn verify(&self) -> Result<(), EvidenceHistoryError> {
        let mut previous = None;

        for (index, entry) in self.entries.iter().enumerate() {
            let expected_sequence =
                u64::try_from(index).map_err(|_| EvidenceHistoryError::SequenceOverflow)?;

            if entry.sequence != expected_sequence {
                return Err(EvidenceHistoryError::SequenceMismatch {
                    expected: expected_sequence,
                    actual: entry.sequence,
                });
            }

            let expected_evidence_digest = evidence_digest(&entry.evidence)?;
            if entry.evidence_digest != expected_evidence_digest {
                return Err(EvidenceHistoryError::EvidenceDigestMismatch);
            }

            if entry.previous_entry_commitment != previous {
                return Err(EvidenceHistoryError::PreviousCommitmentMismatch);
            }

            let expected_commitment = entry_commitment(
                entry.sequence,
                &entry.evidence_digest,
                entry.previous_entry_commitment,
            );
            if entry.commitment != expected_commitment {
                return Err(EvidenceHistoryError::CommitmentMismatch);
            }

            previous = Some(entry.commitment);
        }

        Ok(())
    }

    /// Verify only the first prefix_len entries.
    ///
    /// This establishes local prefix consistency. It is intentionally not an
    /// inclusion or consistency proof for an independently held history; those
    /// require a VDS-specific proof system.
    pub fn verify_prefix(&self, prefix_len: usize) -> Result<(), EvidenceHistoryError> {
        if prefix_len > self.entries.len() {
            return Err(EvidenceHistoryError::SequenceMismatch {
                expected: self.entries.len() as u64,
                actual: prefix_len as u64,
            });
        }

        let prefix = Self { entries: self.entries[..prefix_len].to_vec() };
        prefix.verify()
    }
}

fn entry_commitment(
    sequence: u64,
    evidence_digest: &EvidenceDigest,
    previous: Option<HistoryEntryCommitment>,
) -> HistoryEntryCommitment {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(ENTRY_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(ENTRY_DOMAIN);
    hasher.update(&VERSION.to_be_bytes());
    hasher.update(&sequence.to_be_bytes());

    match previous {
        Some(previous) => {
            hasher.update(&[1]);
            hasher.update(previous.as_bytes());
        }
        None => hasher.update(&[0]),
    }

    hasher.update(evidence_digest.as_bytes());
    HistoryEntryCommitment(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        decide, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationKey,
        ObservationRecord, SemanticAdmissionState,
    };
    use crate::semantic_transition::{build_transition_evidence, TransitionClaim};
    use uuid::Uuid;

    fn evidence(seed: u128) -> TransitionEvidence {
        let delivery = DeliveryContract {
            logical_delivery_id: Uuid::from_u128(seed),
            schema_version: 1,
            expires_at_ms: 1_000,
            payload: format!("delivery-{seed}").into_bytes(),
        };
        let observation = ObservationRecord {
            key: ObservationKey {
                namespace: "source".into(),
                observation_id: Uuid::from_u128(seed + 100),
            },
            source_id: Uuid::from_u128(seed + 200),
            observed_at_ms: 10,
            payload: format!("observation-{seed}").into_bytes(),
        };
        let before = SemanticAdmissionState::default();
        let policy = AdmissionPolicy {
            allow_new_observation: true,
            ..AdmissionPolicy::default()
        };
        let outcome = decide(&before, &delivery, &observation, policy, 10);
        let AdmissionOutcome::Admitted { next_state, result } = outcome else {
            panic!("fixture admission should succeed");
        };
        build_transition_evidence(
            &before,
            &next_state,
            TransitionClaim::Admission {
                delivery,
                observation,
                policy,
                now_ms: 10,
                result,
            },
        )
        .unwrap()
    }

    #[test]
    fn append_is_deterministic() {
        let mut first = EvidenceHistory::new();
        first.append(evidence(1)).unwrap();
        first.append(evidence(2)).unwrap();

        let mut second = EvidenceHistory::new();
        second.append(evidence(1)).unwrap();
        second.append(evidence(2)).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert!(first.head_commitment().is_some());
        first.verify().unwrap();
    }

    #[test]
    fn sequence_and_previous_link_are_committed() {
        let mut history = EvidenceHistory::new();
        let first = history.append(evidence(1)).unwrap().clone();
        let second = history.append(evidence(2)).unwrap().clone();

        assert_eq!(first.sequence(), 0);
        assert_eq!(first.previous_entry_commitment(), None);
        assert_eq!(second.sequence(), 1);
        assert_eq!(second.previous_entry_commitment(), Some(first.commitment()));
    }

    #[test]
    fn evidence_mutation_is_detected() {
        let mut history = EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        history.entries[0].evidence.version += 1;
        assert_eq!(history.verify(), Err(EvidenceHistoryError::EvidenceDigestMismatch));
    }

    #[test]
    fn evidence_digest_mutation_is_detected() {
        let mut history = EvidenceHistory::new();
        history.append(evidence(1)).unwrap();

        let mut digest = *history.entries[0].evidence_digest.as_bytes();
        digest[0] ^= 1;
        history.entries[0].evidence_digest = EvidenceDigest(digest);

        assert_eq!(history.verify(), Err(EvidenceHistoryError::EvidenceDigestMismatch));
    }

    #[test]
    fn previous_link_mutation_is_detected() {
        let mut history = EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        history.append(evidence(2)).unwrap();
        history.entries[1].previous_entry_commitment = None;
        assert_eq!(history.verify(), Err(EvidenceHistoryError::PreviousCommitmentMismatch));
    }

    #[test]
    fn commitment_mutation_is_detected() {
        let mut history = EvidenceHistory::new();
        history.append(evidence(1)).unwrap();

        let mut commitment = *history.entries[0].commitment.as_bytes();
        commitment[0] ^= 1;
        history.entries[0].commitment = HistoryEntryCommitment(commitment);

        assert_eq!(history.verify(), Err(EvidenceHistoryError::CommitmentMismatch));
    }

    #[test]
    fn prefix_verification_checks_only_retained_prefix() {
        let mut history = EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        history.append(evidence(2)).unwrap();

        history.verify_prefix(0).unwrap();
        history.verify_prefix(1).unwrap();
        history.verify_prefix(2).unwrap();
        assert!(history.verify_prefix(3).is_err());
    }

    #[test]
    fn changing_an_earlier_entry_breaks_the_history() {
        let mut history = EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        history.append(evidence(2)).unwrap();
        let original_head = history.head_commitment();

        history.entries[0].evidence_digest = evidence_digest(&evidence(99)).unwrap();

        assert!(history.verify().is_err());
        assert_eq!(history.head_commitment(), original_head);
    }
}
