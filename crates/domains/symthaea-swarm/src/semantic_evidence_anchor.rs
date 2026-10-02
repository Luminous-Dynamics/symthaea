// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic local observation of history checkpoints.
//!
//! This module separates local checkpoint observation from any future
//! transparency/VDS layer. It can detect conflicting observations at the same
//! history position among anchors it has observed, but does not establish
//! global non-equivocation, signatures, authorization, or consensus.

use std::collections::BTreeMap;

use blake3::Hasher;

use crate::semantic_evidence_history::{HistoryCheckpoint, VERSION as HISTORY_VERSION};

pub const ALGORITHM: &str = "BLAKE3-256";
pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-evidence-history-anchor";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoryAnchorCommitment([u8; 32]);

impl HistoryAnchorCommitment {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoryAnchor {
    history_id: [u8; 32],
    checkpoint: HistoryCheckpoint,
    commitment: HistoryAnchorCommitment,
}

impl HistoryAnchor {
    pub fn new(history_id: [u8; 32], checkpoint: HistoryCheckpoint) -> Self {
        let commitment = anchor_commitment(history_id, checkpoint);
        Self { history_id, checkpoint, commitment }
    }

    pub fn history_id(&self) -> &[u8; 32] { &self.history_id }

    pub fn checkpoint(&self) -> HistoryCheckpoint { self.checkpoint }

    pub fn commitment(&self) -> HistoryAnchorCommitment { self.commitment }

    pub fn verify(&self) -> bool {
        self.checkpoint.verify()
            && self.commitment == anchor_commitment(self.history_id, self.checkpoint)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HistoryAnchorError {
    #[error("history anchor commitment is inconsistent")]
    CommitmentMismatch,
    #[error(
        "conflicting checkpoints observed for history position {history_id:?} at length {length}"
    )]
    ConflictingCheckpoint { history_id: [u8; 32], length: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorObservation {
    New,
    AlreadyKnown,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HistoryAnchorRegistry {
    observations: BTreeMap<([u8; 32], u64), HistoryAnchor>,
}

impl HistoryAnchorRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn len(&self) -> usize { self.observations.len() }
    pub fn is_empty(&self) -> bool { self.observations.is_empty() }

    pub fn get(&self, history_id: &[u8; 32], length: u64) -> Option<HistoryAnchor> {
        self.observations.get(&(*history_id, length)).copied()
    }

    pub fn observe(
        &mut self,
        anchor: HistoryAnchor,
    ) -> Result<AnchorObservation, HistoryAnchorError> {
        if !anchor.verify() {
            return Err(HistoryAnchorError::CommitmentMismatch);
        }

        let key = (anchor.history_id, anchor.checkpoint.length());
        match self.observations.get(&key) {
            Some(existing) if *existing == anchor => Ok(AnchorObservation::AlreadyKnown),
            Some(_) => Err(HistoryAnchorError::ConflictingCheckpoint {
                history_id: anchor.history_id,
                length: anchor.checkpoint.length(),
            }),
            None => {
                self.observations.insert(key, anchor);
                Ok(AnchorObservation::New)
            }
        }
    }
}

fn anchor_commitment(
    history_id: [u8; 32],
    checkpoint: HistoryCheckpoint,
) -> HistoryAnchorCommitment {
    let mut hasher = Hasher::new();
    hasher.update(&(DOMAIN.len() as u64).to_be_bytes());
    hasher.update(DOMAIN);
    hasher.update(&VERSION.to_be_bytes());
    hasher.update(&HISTORY_VERSION.to_be_bytes());
    hasher.update(&history_id);
    hasher.update(&checkpoint.length().to_be_bytes());

    match checkpoint.head() {
        Some(head) => {
            hasher.update(&[1]);
            hasher.update(head.as_bytes());
        }
        None => hasher.update(&[0]),
    }

    hasher.update(checkpoint.snapshot().as_bytes());
    HistoryAnchorCommitment(*hasher.finalize().as_bytes())
}

// The VDS boundary lives in a separate file so its proof format can evolve
// independently. This path keeps the boundary compiled without pretending
// that the chained history itself implements a Merkle consistency proof.
#[cfg(feature = "semantic-digest")]
#[path = "../semantic_evidence_vds.rs"]
pub mod vds;

#[cfg(feature = "semantic-digest")]
pub use vds::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_admission::{
        decide, AdmissionOutcome, AdmissionPolicy, DeliveryContract, ObservationKey,
        ObservationRecord, SemanticAdmissionState,
    };
    use crate::semantic_evidence_digest::evidence_digest;
    use crate::semantic_transition::{build_transition_evidence, TransitionClaim};
    use uuid::Uuid;

    fn evidence(seed: u128) -> crate::semantic_transition::TransitionEvidence {
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

    fn history_id(seed: u8) -> [u8; 32] { [seed; 32] }

    fn checkpoint(seed: u128) -> HistoryCheckpoint {
        let mut history = crate::semantic_evidence_history::EvidenceHistory::new();
        history.append(evidence(seed)).unwrap();
        history.checkpoint()
    }

    #[test]
    fn anchor_commitment_is_deterministic() {
        let cp = checkpoint(1);
        let first = HistoryAnchor::new(history_id(7), cp);
        let second = HistoryAnchor::new(history_id(7), cp);
        assert_eq!(first, second);
        assert!(first.verify());
    }

    #[test]
    fn tampered_checkpoint_is_rejected() {
        let mut cp = checkpoint(1);
        cp.snapshot = crate::semantic_evidence_history::HistoryEntryCommitment([0; 32]);
        let mut anchor = HistoryAnchor::new(history_id(1), cp);
        anchor.commitment = anchor_commitment(anchor.history_id, anchor.checkpoint);
        let mut registry = HistoryAnchorRegistry::new();
        assert_eq!(
            registry.observe(anchor),
            Err(HistoryAnchorError::CommitmentMismatch)
        );
    }

    #[test]
    fn identical_observation_is_idempotent() {
        let anchor = HistoryAnchor::new(history_id(1), checkpoint(1));
        let mut registry = HistoryAnchorRegistry::new();
        assert_eq!(registry.observe(anchor), Ok(AnchorObservation::New));
        assert_eq!(registry.observe(anchor), Ok(AnchorObservation::AlreadyKnown));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn conflicting_same_position_is_rejected() {
        let first = HistoryAnchor::new(history_id(1), checkpoint(1));
        let second = HistoryAnchor::new(history_id(1), checkpoint(2));
        assert_eq!(first.checkpoint().length(), second.checkpoint().length());
        assert_ne!(first.checkpoint(), second.checkpoint());

        let mut registry = HistoryAnchorRegistry::new();
        registry.observe(first).unwrap();
        assert_eq!(
            registry.observe(second),
            Err(HistoryAnchorError::ConflictingCheckpoint {
                history_id: history_id(1),
                length: 1,
            })
        );
    }

    #[test]
    fn different_histories_are_independent() {
        let cp = checkpoint(1);
        let first = HistoryAnchor::new(history_id(1), cp);
        let second = HistoryAnchor::new(history_id(2), cp);
        let mut registry = HistoryAnchorRegistry::new();
        assert_eq!(registry.observe(first), Ok(AnchorObservation::New));
        assert_eq!(registry.observe(second), Ok(AnchorObservation::New));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn longer_checkpoint_does_not_imply_consistency() {
        let mut history = crate::semantic_evidence_history::EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        let short = history.checkpoint();
        history.append(evidence(2)).unwrap();
        let long = history.checkpoint();

        let mut registry = HistoryAnchorRegistry::new();
        assert_eq!(
            registry.observe(HistoryAnchor::new(history_id(3), short)),
            Ok(AnchorObservation::New)
        );
        assert_eq!(
            registry.observe(HistoryAnchor::new(history_id(3), long)),
            Ok(AnchorObservation::New)
        );
    }

    #[test]
    fn tampered_anchor_is_rejected() {
        let mut anchor = HistoryAnchor::new(history_id(4), checkpoint(1));
        anchor.commitment = HistoryAnchorCommitment([0; 32]);
        let mut registry = HistoryAnchorRegistry::new();
        assert_eq!(
            registry.observe(anchor),
            Err(HistoryAnchorError::CommitmentMismatch)
        );
    }

    #[test]
    fn anchor_commitment_changes_with_identity_and_position() {
        let cp = checkpoint(1);
        assert_ne!(
            HistoryAnchor::new(history_id(1), cp).commitment(),
            HistoryAnchor::new(history_id(2), cp).commitment()
        );

        let mut history = crate::semantic_evidence_history::EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        let first = history.checkpoint();
        history.append(evidence(2)).unwrap();
        let second = history.checkpoint();

        assert_ne!(
            HistoryAnchor::new(history_id(1), first).commitment(),
            HistoryAnchor::new(history_id(1), second).commitment()
        );
    }

    #[test]
    fn evidence_digest_remains_the_underlying_history_witness() {
        let ev = evidence(9);
        assert_eq!(evidence_digest(&ev).unwrap(), evidence_digest(&ev).unwrap());
    }
}
