// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-bound receipts for the symbolic edit-locality verifier.
//!
//! A plain EditLocalityEvidence describes the operation and residuals, but does
//! not identify the exact source and target sequences it was computed from.
//! This envelope adds role-separated SHA-256 commitments and deterministic
//! replay. It is an integrity binding, not a digital signature or provenance
//! proof: a party able to replace the inputs and receipt can produce new hashes.

use crate::edit_locality::{
    EditLocalityConfig, EditLocalityError, EditLocalityEvidence,
    EditRequest, LocalityEvent, verify_edit_locality,
    verify_edit_locality_with_config,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// Versioned byte-level encoding for event-sequence commitments.
pub const EDIT_LOCALITY_BINDING_VERSION: &str = "edit-locality-sha256-v1.0";

/// Receipt whose verdict is bound to exact symbolic source and target events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundEditLocalityEvidence {
    pub binding_version: String,
    /// SHA-256 over the canonical binary event encoding with role "source".
    pub source_sha256: String,
    /// SHA-256 over the canonical binary event encoding with role "target".
    pub target_sha256: String,
    pub evidence: EditLocalityEvidence,
}

/// Verify with default tolerances and bind the result to the exact event data.
pub fn verify_edit_locality_bound(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    request: &EditRequest,
) -> Result<BoundEditLocalityEvidence, EditLocalityError> {
    let evidence = verify_edit_locality(source, target, request)?;
    Ok(bind_evidence(source, target, evidence))
}

/// Verify with explicit tolerances and bind the result to the exact event data.
pub fn verify_edit_locality_bound_with_config(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    request: &EditRequest,
    config: &EditLocalityConfig,
) -> Result<BoundEditLocalityEvidence, EditLocalityError> {
    let evidence = verify_edit_locality_with_config(source, target, request, config)?;
    Ok(bind_evidence(source, target, evidence))
}

/// Replay a bound receipt against supplied source/target values.
///
/// Returns false if either sequence commitment, the binding version, or any
/// recomputed evidence field differs. Invalid inputs still return the verifier's
/// explicit error. The event budget is checked before the hashing traversal.
pub fn replay_edit_locality_bound(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    request: &EditRequest,
    config: &EditLocalityConfig,
    receipt: &BoundEditLocalityEvidence,
) -> Result<bool, EditLocalityError> {
    // Re-run the verifier first so malformed input is surfaced as an explicit
    // error, and so the base verifier enforces its event budget before hashing.
    let recomputed = verify_edit_locality_bound_with_config(source, target, request, config)?;
    if receipt.binding_version != EDIT_LOCALITY_BINDING_VERSION {
        return Ok(false);
    }
    Ok(&recomputed == receipt)
}

fn bind_evidence(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    evidence: EditLocalityEvidence,
) -> BoundEditLocalityEvidence {
    BoundEditLocalityEvidence {
        binding_version: EDIT_LOCALITY_BINDING_VERSION.to_owned(),
        source_sha256: event_sequence_sha256(source, "source"),
        target_sha256: event_sequence_sha256(target, "target"),
        evidence,
    }
}

/// Hash the exact f64 bit patterns, integer pitches, event order, count and role.
/// Encoding is fixed little-endian and domain-separated to avoid ambiguous
/// concatenation or treating source and target commitments as interchangeable.
fn event_sequence_sha256(events: &[LocalityEvent], role: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.edit-locality.event-sequence\0");
    hasher.update(EDIT_LOCALITY_BINDING_VERSION.as_bytes());
    hasher.update([0]);
    hasher.update(role.as_bytes());
    hasher.update([0]);
    hasher.update((events.len() as u64).to_le_bytes());
    for event in events {
        hasher.update(event.pitch_midi.to_le_bytes());
        hasher.update(event.onset_beats.to_bits().to_le_bytes());
        hasher.update(event.duration_beats.to_bits().to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Vec<LocalityEvent> {
        vec![
            LocalityEvent { pitch_midi: 60, onset_beats: 0.0, duration_beats: 0.5 },
            LocalityEvent { pitch_midi: 62, onset_beats: 1.0, duration_beats: 0.75 },
            LocalityEvent { pitch_midi: 65, onset_beats: 2.0, duration_beats: 0.5 },
            LocalityEvent { pitch_midi: 64, onset_beats: 3.0, duration_beats: 1.0 },
        ]
    }

    fn request() -> EditRequest {
        EditRequest { operation: crate::edit_locality::EditOperation::Transpose { semitones: 5 } }
    }

    fn target(source: &[LocalityEvent]) -> Vec<LocalityEvent> {
        source.iter().map(|event| LocalityEvent {
            pitch_midi: event.pitch_midi + 5,
            onset_beats: event.onset_beats,
            duration_beats: event.duration_beats,
        }).collect()
    }

    #[test]
    fn binds_exact_source_and_target_sequences_and_replays() {
        let source = source();
        let target = target(&source);
        let receipt = verify_edit_locality_bound(&source, &target, &request()).unwrap();
        assert_eq!(receipt.source_sha256.len(), 64);
        assert_eq!(receipt.target_sha256.len(), 64);
        assert_ne!(receipt.source_sha256, receipt.target_sha256);
        assert!(replay_edit_locality_bound(
            &source, &target, &request(), &EditLocalityConfig::default(), &receipt
        ).unwrap());
    }

    #[test]
    fn rejects_replay_if_source_or_target_is_mutated() {
        let source_events = source();
        let target_events = target(&source_events);
        let receipt = verify_edit_locality_bound(&source_events, &target_events, &request()).unwrap();

        let mut changed_source = source_events.clone();
        changed_source[1].duration_beats = 0.8;
        assert!(!replay_edit_locality_bound(
            &changed_source, &target_events, &request(), &EditLocalityConfig::default(), &receipt
        ).unwrap());

        let mut changed_target = target_events.clone();
        changed_target[2].onset_beats = 2.01;
        assert!(!replay_edit_locality_bound(
            &source_events, &changed_target, &request(), &EditLocalityConfig::default(), &receipt
        ).unwrap());
    }

    #[test]
    fn rejects_receipt_mutation_even_when_inputs_match() {
        let source_events = source();
        let target_events = target(&source_events);
        let mut receipt = verify_edit_locality_bound(&source_events, &target_events, &request()).unwrap();
        receipt.evidence.passed = false;
        assert!(!replay_edit_locality_bound(
            &source_events, &target_events, &request(), &EditLocalityConfig::default(), &receipt
        ).unwrap());
    }

    #[test]
    fn binding_is_deterministic_and_role_separated() {
        let events = source();
        let first = event_sequence_sha256(&events, "source");
        let second = event_sequence_sha256(&events, "source");
        let other_role = event_sequence_sha256(&events, "target");
        assert_eq!(first, second);
        assert_ne!(first, other_role);
    }

    #[test]
    fn receipt_json_round_trip_preserves_binding() {
        let source_events = source();
        let target_events = target(&source_events);
        let receipt = verify_edit_locality_bound(&source_events, &target_events, &request()).unwrap();
        let encoded = serde_json::to_vec(&receipt).unwrap();
        let decoded: BoundEditLocalityEvidence = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, receipt);
        assert!(replay_edit_locality_bound(
            &source_events, &target_events, &request(), &EditLocalityConfig::default(), &decoded
        ).unwrap());
    }

    #[test]
    fn rejects_unsupported_binding_version() {
        let source_events = source();
        let target_events = target(&source_events);
        let mut receipt = verify_edit_locality_bound(&source_events, &target_events, &request()).unwrap();
        receipt.binding_version = "unknown-version".to_owned();
        assert!(!replay_edit_locality_bound(
            &source_events, &target_events, &request(), &EditLocalityConfig::default(), &receipt
        ).unwrap());
    }

    #[test]
    fn records_contract_identity_inside_nested_evidence() {
        let source_events = source();
        let target_events = target(&source_events);
        let receipt = verify_edit_locality_bound(&source_events, &target_events, &request()).unwrap();
        assert_eq!(receipt.evidence.contract_version, crate::edit_locality::EDIT_LOCALITY_CONTRACT_VERSION);
    }
}
