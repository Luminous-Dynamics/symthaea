// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit boundary between local history witnesses and VDS consistency proofs.
//!
//! This module deliberately does not reinterpret the chained local
//! EvidenceHistory as a Merkle VDS. RFC 9942 makes proof formats VDS-specific,
//! and RFC 9162's consistency proofs are defined for its own Merkle tree
//! construction.
//!
//! The boundary here gives callers typed semantics now:
//! * Valid — a concrete VDS verifier established append-only consistency.
//! * Invalid — a concrete verifier rejected the proof.
//! * Unsupported — no VDS implementation is available for the requested proof.
//!
//! The current chained local history returns Unsupported. A future Merkle VDS
//! adapter can implement the trait without changing semantic admission,
//! evidence history, or anchor semantics.

use crate::semantic_evidence_history::HistoryCheckpoint;

pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-evidence-vds";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyStatus {
    Valid,
    Invalid,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsistencyRequest {
    older: HistoryCheckpoint,
    newer: HistoryCheckpoint,
}

impl ConsistencyRequest {
    pub fn new(older: HistoryCheckpoint, newer: HistoryCheckpoint) -> Self {
        Self { older, newer }
    }

    pub fn older(&self) -> HistoryCheckpoint {
        self.older
    }

    pub fn newer(&self) -> HistoryCheckpoint {
        self.newer
    }

    pub fn is_strict_extension_request(&self) -> bool {
        self.older.length() < self.newer.length()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistencyProof {
    vds: &'static str,
    version: u16,
    bytes: Vec<u8>,
}

impl ConsistencyProof {
    pub fn new(vds: &'static str, version: u16, bytes: Vec<u8>) -> Self {
        Self {
            vds,
            version,
            bytes,
        }
    }

    pub fn vds(&self) -> &'static str {
        self.vds
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// VDS adapter boundary for append-only consistency proofs.
///
/// Implementations must not infer consistency merely because a newer
/// checkpoint has a greater length. The proof must be verified according to
/// the exact VDS algorithm named by the adapter.
pub trait HistoryVds {
    fn vds_name(&self) -> &'static str;

    fn prove_consistency(
        &self,
        _request: &ConsistencyRequest,
    ) -> Result<ConsistencyProof, ConsistencyError> {
        Err(ConsistencyError::Unsupported)
    }

    fn verify_consistency(
        &self,
        request: &ConsistencyRequest,
        proof: &ConsistencyProof,
    ) -> ConsistencyStatus;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConsistencyError {
    #[error("this VDS adapter does not support consistency proofs")]
    Unsupported,
    #[error("a consistency proof cannot be generated for the supplied request")]
    CannotGenerate,
}

/// Adapter representing the current chained local witness.
///
/// This is intentionally an explicit Unsupported implementation. It prevents
/// callers from accidentally treating the previous-entry commitment chain as
/// an RFC 9162 Merkle consistency proof.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainedHistoryVds;

impl HistoryVds for ChainedHistoryVds {
    fn vds_name(&self) -> &'static str {
        "symthaea-chained-history-v1"
    }

    fn verify_consistency(
        &self,
        _request: &ConsistencyRequest,
        _proof: &ConsistencyProof,
    ) -> ConsistencyStatus {
        ConsistencyStatus::Unsupported
    }
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

    fn checkpoint(seed: u128) -> HistoryCheckpoint {
        let mut history = crate::semantic_evidence_history::EvidenceHistory::new();
        history.append(evidence(seed)).unwrap();
        history.checkpoint()
    }

    #[test]
    fn request_distinguishes_strict_extension() {
        let older = checkpoint(1);
        let mut history = crate::semantic_evidence_history::EvidenceHistory::new();
        history.append(evidence(1)).unwrap();
        history.append(evidence(2)).unwrap();
        let newer = history.checkpoint();

        let request = ConsistencyRequest::new(older, newer);
        assert!(request.is_strict_extension_request());
        assert_eq!(request.older().length(), 1);
        assert_eq!(request.newer().length(), 2);
    }

    #[test]
    fn chained_history_is_explicitly_not_a_vds_consistency_proof() {
        let request = ConsistencyRequest::new(checkpoint(1), checkpoint(2));
        let adapter = ChainedHistoryVds;
        let proof = ConsistencyProof::new(adapter.vds_name(), VERSION, Vec::new());

        assert_eq!(
            adapter.verify_consistency(&request, &proof),
            ConsistencyStatus::Unsupported
        );
        assert_eq!(
            adapter.prove_consistency(&request),
            Err(ConsistencyError::Unsupported)
        );
    }

    #[test]
    fn proof_is_opaque_at_the_boundary() {
        let proof = ConsistencyProof::new("example-vds", 7, vec![1, 2, 3]);
        assert_eq!(proof.vds(), "example-vds");
        assert_eq!(proof.version(), 7);
        assert_eq!(proof.bytes(), &[1, 2, 3]);
    }

    #[test]
    fn unsupported_is_not_invalid() {
        assert_ne!(ConsistencyStatus::Unsupported, ConsistencyStatus::Invalid);
        assert_ne!(ConsistencyStatus::Unsupported, ConsistencyStatus::Valid);
    }
}
