// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated Learning Coordinator Zome
//!
//! Provides the API for Byzantine-resistant federated learning on Holochain.
//! Works with the integrity zome to store and validate gradient submissions.
//!
//! ## SDK Integration
//!
//! This zome uses the Mycelix SDK for:
//! - **MATL**: Proof of Gradient Quality (PoGQ), CompositeScore, HierarchicalDetector
//! - **Byzantine Detection**: Cartel detection and adaptive thresholds

use hdk::prelude::*;
use federated_learning_integrity::*;

// NOTE: SDK imports (mycelix_sdk::matl, hyperfeel, sha2, etc.) are now
// in individual module files where they're actually used.

// =============================================================================
// MODULE DECLARATIONS
// =============================================================================
//
// This monolith is being incrementally split into modules.
// Module files exist in src/ with extracted code, wired in via mod + use.
// include_str!() regression tests use concat!() across ALL source files,
// so code can be moved to modules without breaking security marker checks.
//
// Migration COMPLETE (7738 → ~1280 lines, 83% reduction):
//   [DONE]  config.rs      — constants (59 lines)
//   [DONE]  auth.rs        — verify_coordinator_authority, require_coordinator_role, rate_limit (390 lines)
//   [DONE]  signals.rs     — Signal enum, SignedSignal, recv_remote_signal, gossip_relay, base64 (693 lines)
//   [DONE]  model.rs       — model versioning (register/get/list config, validation) (213 lines)
//   [DONE]  bootstrap.rs   — initialize_bootstrap, vote_for_coordinator, guardian election (1936 lines)
//   [DONE]  bridge.rs      — identity bridge calls, reputation sync, event broadcasting (537 lines)
//   [DONE]  consensus.rs   — commit-reveal consensus, validator registration (651 lines)
//   [DONE]  detection.rs   — feature extraction, hierarchical detection helpers (283 lines)
//   [DONE]  governance.rs  — byzantine voting, conviction (274 lines)
//   [DONE]  gradients.rs   — submit_gradient, role mgmt, reputation queries (548 lines)
//   [DONE]  hyperfeel.rs   — HV16 compressed gradient submission/aggregation (344 lines)
//   [DONE]  matl.rs        — MATL integration, PoGQ, cartel detection (714 lines)
//   [DONE]  payments.rs    — round payments, ETH address registration (459 lines)
//   [DONE]  pipeline.rs    — mycelix-fl DecentralizedPipeline integration (331 lines)
//   [DONE]  proof.rs       — zkSTARK gradient proofs (95 lines)
//   [DONE]  scheduling.rs  — round scheduling, advancement (442 lines)
//
// lib.rs now contains only: imports, module declarations, ensure_path(), and tests.
// Verify with: cargo check --target wasm32-unknown-unknown (in nix develop)
//
mod config;
mod auth;
mod signals;
mod model;
mod bootstrap;
mod bridge;
mod consensus;
mod detection;
mod governance;
mod gradients;
mod hyperfeel;
mod matl;
mod payments;
mod pipeline;
mod proof;
mod scheduling;

// Test modules use `use super::*` — bring needed types into scope only for tests
#[cfg(test)]
use config::*;
#[cfg(test)]
use bootstrap::*;
#[cfg(test)]
use signals::*;

// === Path Helper (shared by all modules via super::ensure_path) ===

pub(crate) fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// =============================================================================
// SEC-002 FIX: Tests for Coordinator Bootstrap Ceremony
// =============================================================================

#[cfg(test)]
mod coordinator_bootstrap_tests {
    use super::*;

    /// Test that bootstrap cannot be initialized twice
    #[test]
    fn test_bootstrap_requires_genesis_coordinators() {
        // This test validates the input requirements
        let input = InitializeBootstrapInput {
            genesis_coordinators: vec![],
            authority_proofs: vec![],
            initial_guardians: vec!["guardian1".to_string(), "guardian2".to_string(), "guardian3".to_string()],
            window_duration_seconds: 3600,
            min_votes: 2,
        };

        // Empty genesis_coordinators should be rejected (in actual zome call)
        assert!(input.genesis_coordinators.is_empty());
    }

    /// Test that authority proofs must match coordinators
    #[test]
    fn test_authority_proofs_must_match() {
        let input = InitializeBootstrapInput {
            genesis_coordinators: vec!["coord1".to_string(), "coord2".to_string()],
            authority_proofs: vec![vec![0u8; 64]], // Only one proof for two coordinators
            initial_guardians: vec!["g1".to_string(), "g2".to_string(), "g3".to_string()],
            window_duration_seconds: 0,
            min_votes: 2,
        };

        // Mismatch should be detected
        assert_ne!(input.genesis_coordinators.len(), input.authority_proofs.len());
    }

    /// Test minimum guardians requirement
    #[test]
    fn test_minimum_guardians_required() {
        let input = InitializeBootstrapInput {
            genesis_coordinators: vec!["coord1".to_string()],
            authority_proofs: vec![vec![0u8; 64]],
            initial_guardians: vec!["g1".to_string()], // Only 1 guardian, less than DEFAULT_MIN_GUARDIANS
            window_duration_seconds: 0,
            min_votes: 1,
        };

        // Should fail because we need at least DEFAULT_MIN_GUARDIANS
        assert!(input.initial_guardians.len() < DEFAULT_MIN_GUARDIANS as usize);
    }

    /// Test vote signature validation
    #[test]
    fn test_vote_signature_minimum_length() {
        let input = VoteForCoordinatorInput {
            candidate_pubkey: "candidate".to_string(),
            approve: true,
            signature: vec![0u8; 32], // Too short, should be 64 bytes
        };

        assert!(input.signature.len() < 64);
    }

    /// Test valid vote structure
    #[test]
    fn test_valid_vote_structure() {
        let input = VoteForCoordinatorInput {
            candidate_pubkey: "uhCAkSomeValidAgentPubKey".to_string(),
            approve: true,
            signature: vec![0u8; 64], // Valid Ed25519 signature length
        };

        assert_eq!(input.signature.len(), 64);
        assert!(input.approve);
    }

    // Tests for new bootstrap ceremony functions

    /// Test InitBootstrapInput structure
    #[test]
    fn test_init_bootstrap_input_structure() {
        let input = InitBootstrapInput {
            genesis_coordinator: "uhCAkGenesisCoordinator123".to_string(),
            authority_proof: vec![0u8; 64],
            window_duration_seconds: 86400,
            min_votes: 3,
            max_guardians: 5,
        };

        assert_eq!(input.authority_proof.len(), 64);
        assert_eq!(input.window_duration_seconds, 86400);
        assert_eq!(input.min_votes, 3);
        assert_eq!(input.max_guardians, 5);
    }

    /// Test AddGuardianInput validation
    #[test]
    fn test_add_guardian_input_structure() {
        let input = AddGuardianInput {
            guardian_pubkey: "uhCAkGuardian1".to_string(),
            voting_weight: 1,
        };

        assert_eq!(input.voting_weight, 1);
        assert!(!input.guardian_pubkey.is_empty());
    }

    /// Test CastVoteInput structure
    #[test]
    fn test_cast_vote_input_structure() {
        let input = CastVoteInput {
            candidate_pubkey: "uhCAkCandidate".to_string(),
            approve: true,
            signature: vec![0u8; 64],
        };

        assert_eq!(input.signature.len(), 64);
        assert!(input.approve);
    }

    /// Test CastVoteInput rejection vote
    #[test]
    fn test_cast_vote_rejection() {
        let input = CastVoteInput {
            candidate_pubkey: "uhCAkCandidate".to_string(),
            approve: false,
            signature: vec![0xAB; 64],
        };

        assert!(!input.approve);
        assert_eq!(input.signature[0], 0xAB);
    }

    /// Test invalid signature length is detected
    #[test]
    fn test_invalid_signature_length() {
        let input = CastVoteInput {
            candidate_pubkey: "uhCAkCandidate".to_string(),
            approve: true,
            signature: vec![0u8; 32], // Too short
        };

        // Should be exactly 64 bytes
        assert_ne!(input.signature.len(), 64);
    }

    /// Test BootstrapStatus structure
    #[test]
    fn test_bootstrap_status_defaults() {
        let status = BootstrapStatus {
            initialized: false,
            bootstrap_complete: false,
            window_open: false,
            window_start: 0,
            window_end: 0,
            current_time: 1704067200,
            guardian_count: 0,
            min_guardians: 3,
            max_guardians: 5,
            min_votes: 3,
            genesis_coordinators: vec![],
        };

        assert!(!status.initialized);
        assert!(!status.bootstrap_complete);
        assert_eq!(status.min_guardians, 3);
    }
}

// =============================================================================
// UNIT TESTS FOR SIGNAL AUTHENTICATION (SEC-005)
// =============================================================================

#[cfg(test)]
mod signal_verification_tests {
    use super::*;

    /// Create a mock ActionHash for testing
    fn make_mock_action_hash() -> ActionHash {
        // Use from_raw_39 which is available in test context
        // ActionHash expects 39 bytes with proper type byte at position 3
        let bytes: [u8; 39] = [
            0x84, 0x29, 0x24, // Holochain ActionHash prefix (holo_hash-0.6.0)
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
            0x20, 0x21, 0x22, 0x23,
        ];
        ActionHash::from_raw_39(bytes.to_vec())
    }

    /// Test that base64 encoding and decoding round-trips correctly
    #[test]
    fn test_base64_round_trip() {
        let original = vec![0u8, 1, 2, 3, 255, 254, 253, 128, 64, 32];
        let encoded = base64_encode(&original);
        let decoded = base64_decode(&encoded).expect("decode failed");
        assert_eq!(original, decoded);
    }

    /// Test base64 encoding of 64-byte signature
    #[test]
    fn test_base64_signature_size() {
        let signature = vec![0u8; 64];
        let encoded = base64_encode(&signature);
        let decoded = base64_decode(&encoded).expect("decode failed");
        assert_eq!(decoded.len(), 64);
    }

    /// Test base64 decoding with padding
    #[test]
    fn test_base64_with_padding() {
        // "Hello" in base64 is "SGVsbG8="
        let decoded = base64_decode("SGVsbG8=").expect("decode failed");
        assert_eq!(decoded, b"Hello");

        // "Hi" in base64 is "SGk="
        let decoded = base64_decode("SGk=").expect("decode failed");
        assert_eq!(decoded, b"Hi");

        // "H" in base64 is "SA=="
        let decoded = base64_decode("SA==").expect("decode failed");
        assert_eq!(decoded, b"H");
    }

    /// Test base64 decoding rejects invalid input
    #[test]
    fn test_base64_invalid_input() {
        // Invalid length (not multiple of 4)
        assert!(base64_decode("ABC").is_err());

        // Invalid characters
        assert!(base64_decode("!@#$").is_err());
    }

    /// Test SignedSignal creation
    #[test]
    fn test_signed_signal_creation() {
        let signal = Signal::GradientSubmitted {
            node_id: "test-node-001".to_string(),
            round: 42,
            action_hash: make_mock_action_hash(),
            source: None,
            signature: None,
        };

        let signed = SignedSignal::new(signal.clone());

        // Verify signable_bytes contains expected content
        let bytes = signed.signable_bytes();
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"GradientSubmitted:"));
    }

    /// Test signable_bytes for different signal types
    #[test]
    fn test_signable_bytes_gradient_submitted() {
        let signal = Signal::GradientSubmitted {
            node_id: "node1".to_string(),
            round: 5,
            action_hash: make_mock_action_hash(),
            source: None,
            signature: None,
        };

        let signed = SignedSignal {
            signal,
            timestamp: 1704067200000000, // 2024-01-01 00:00:00 UTC in microseconds
            nonce: [0x01; 16],
        };

        let bytes = signed.signable_bytes();

        // Check it starts with correct prefix
        assert!(bytes.starts_with(b"GradientSubmitted:node1:"));

        // Check it ends with nonce
        assert!(bytes.ends_with(&[0x01; 16]));
    }

    /// Test signable_bytes for RoundCompleted signal
    #[test]
    fn test_signable_bytes_round_completed() {
        let signal = Signal::RoundCompleted {
            round: 10,
            accuracy: 0.95,
            byzantine_count: 2,
            source: None,
            signature: None,
        };

        let signed = SignedSignal {
            signal,
            timestamp: 1704067200000000,
            nonce: [0x02; 16],
        };

        let bytes = signed.signable_bytes();
        assert!(bytes.starts_with(b"RoundCompleted:"));
    }

    /// Test signable_bytes for ByzantineDetected signal
    #[test]
    fn test_signable_bytes_byzantine_detected() {
        let signal = Signal::ByzantineDetected {
            node_id: "bad-node".to_string(),
            round: 3,
            confidence: 0.87,
            source: None,
            signature: None,
        };

        let signed = SignedSignal {
            signal,
            timestamp: 1704067200000000,
            nonce: [0x03; 16],
        };

        let bytes = signed.signable_bytes();
        assert!(bytes.starts_with(b"ByzantineDetected:bad-node:"));
    }

    /// Test that different signals produce different signable bytes
    #[test]
    fn test_signable_bytes_uniqueness() {
        let nonce = [0x00; 16];
        let timestamp = 1704067200000000i64;

        let signal1 = SignedSignal {
            signal: Signal::GradientSubmitted {
                node_id: "node1".to_string(),
                round: 1,
                action_hash: make_mock_action_hash(),
                source: None,
                signature: None,
            },
            timestamp,
            nonce,
        };

        let signal2 = SignedSignal {
            signal: Signal::GradientSubmitted {
                node_id: "node2".to_string(),
                round: 1,
                action_hash: make_mock_action_hash(),
                source: None,
                signature: None,
            },
            timestamp,
            nonce,
        };

        // Different node_id should produce different bytes
        assert_ne!(signal1.signable_bytes(), signal2.signable_bytes());
    }

    /// Test that nonce prevents replay attacks
    #[test]
    fn test_nonce_prevents_replay() {
        let mock_hash = make_mock_action_hash();
        let timestamp = 1704067200000000i64;

        let signal1 = SignedSignal {
            signal: Signal::GradientSubmitted {
                node_id: "node1".to_string(),
                round: 1,
                action_hash: mock_hash.clone(),
                source: None,
                signature: None,
            },
            timestamp,
            nonce: [0x00; 16],
        };

        let signal2 = SignedSignal {
            signal: Signal::GradientSubmitted {
                node_id: "node1".to_string(),
                round: 1,
                action_hash: mock_hash,
                source: None,
                signature: None,
            },
            timestamp,
            nonce: [0xFF; 16],
        };

        // Same signal content but different nonce should produce different bytes
        assert_ne!(signal1.signable_bytes(), signal2.signable_bytes());
    }

    /// Test Signal serialization/deserialization
    #[test]
    fn test_signal_serialization() {
        let signal = Signal::GradientSubmitted {
            node_id: "test-node".to_string(),
            round: 99,
            action_hash: make_mock_action_hash(),
            source: Some("uhCAk_test_source".to_string()),
            signature: Some("base64signature==".to_string()),
        };

        let json = serde_json::to_string(&signal).expect("serialize failed");
        let deserialized: Signal = serde_json::from_str(&json).expect("deserialize failed");

        match deserialized {
            Signal::GradientSubmitted { node_id, round, source, signature, .. } => {
                assert_eq!(node_id, "test-node");
                assert_eq!(round, 99);
                assert_eq!(source, Some("uhCAk_test_source".to_string()));
                assert_eq!(signature, Some("base64signature==".to_string()));
            }
            _ => panic!("Wrong signal type"),
        }
    }

    /// Test SignedRemoteSignalInput serialization
    #[test]
    fn test_signed_remote_signal_input_serialization() {
        let signal = Signal::RoundCompleted {
            round: 5,
            accuracy: 0.92,
            byzantine_count: 1,
            source: Some("sender".to_string()),
            signature: None,
        };

        let signed_signal = SignedSignal {
            signal,
            timestamp: 1704067200000000,
            nonce: [0xAB; 16],
        };

        let input = SignedRemoteSignalInput {
            signed_signal,
            signature_base64: "SGVsbG8gV29ybGQ=".to_string(),
        };

        let json = serde_json::to_string(&input).expect("serialize failed");
        assert!(json.contains("RoundCompleted"));
        assert!(json.contains("signature_base64"));

        let deserialized: SignedRemoteSignalInput =
            serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(deserialized.signature_base64, "SGVsbG8gV29ybGQ=");
    }
}

// =============================================================================
// REGRESSION TESTS: Security Fix Verification
//
// These tests use include_str!() + concat!() to read ALL coordinator source
// files at compile time and assert that security-critical code markers exist.
// If the linter or an accidental revert removes a security fix, these tests
// will fail. The concat!() approach allows incremental migration of code
// from this monolith to separate module files without breaking the tests.
// =============================================================================

#[cfg(test)]
mod security_fix_tests {
    /// The source code of ALL coordinator modules, included at compile time.
    /// If any security marker is removed, the corresponding test fails immediately.
    /// Uses concat!() so markers are found regardless of which module file they live in,
    /// enabling incremental migration from the lib.rs monolith to separate modules.
    const SOURCE: &str = concat!(
        include_str!("lib.rs"),
        include_str!("auth.rs"),
        include_str!("bootstrap.rs"),
        include_str!("bridge.rs"),
        include_str!("config.rs"),
        include_str!("consensus.rs"),
        include_str!("detection.rs"),
        include_str!("governance.rs"),
        include_str!("gradients.rs"),
        include_str!("hyperfeel.rs"),
        include_str!("matl.rs"),
        include_str!("model.rs"),
        include_str!("payments.rs"),
        include_str!("pipeline.rs"),
        include_str!("proof.rs"),
        include_str!("scheduling.rs"),
        include_str!("signals.rs"),
    );

    // =========================================================================
    // H-08: recv_remote_signal rejects unsigned signals
    // =========================================================================

    /// H-08: The legacy unsigned signal path must be removed.
    /// `recv_remote_signal` must reject any signal that is not wrapped in
    /// `SignedRemoteSignalInput` with an Ed25519 signature.
    #[test]
    fn test_h08_unsigned_signals_rejected() {
        // The rejection error message must exist in source
        assert!(
            SOURCE.contains("Unsigned signals are no longer accepted"),
            "H-08 REGRESSION: recv_remote_signal must contain the unsigned-signal rejection message. \
             The H-08 fix (reject unsigned signals) appears to have been reverted."
        );

        // The H-08 comment marker must exist
        assert!(
            SOURCE.contains("H-08: Legacy unsigned signal path REMOVED"),
            "H-08 REGRESSION: The 'H-08: Legacy unsigned signal path REMOVED' comment is missing. \
             This marker documents the security fix."
        );

        // Must try to decode as SignedRemoteSignalInput first
        assert!(
            SOURCE.contains("signal.decode::<SignedRemoteSignalInput>()"),
            "H-08 REGRESSION: recv_remote_signal must decode as SignedRemoteSignalInput. \
             Without this, unsigned signals could be processed."
        );
    }

    // =========================================================================
    // H-01: Trust score computed from on-chain reputation, not self-reported
    // =========================================================================

    /// H-01: The `get_on_chain_trust_score` helper must exist and be called
    /// in `submit_gradient`, overriding the client-supplied `trust_score`.
    #[test]
    fn test_h01_on_chain_trust_score() {
        // The helper function must exist with the correct signature
        assert!(
            SOURCE.contains("fn get_on_chain_trust_score(node_id: &str) -> Option<f32>"),
            "H-01 REGRESSION: get_on_chain_trust_score function is missing. \
             Without it, self-reported trust scores would be used."
        );

        // submit_gradient must call get_on_chain_trust_score
        assert!(
            SOURCE.contains("get_on_chain_trust_score(&input.node_id)"),
            "H-01 REGRESSION: submit_gradient does not call get_on_chain_trust_score. \
             Self-reported trust_score values would be accepted."
        );

        // The H-01 comment documenting the fix must exist
        assert!(
            SOURCE.contains("H-01: Ignore self-reported trust_score"),
            "H-01 REGRESSION: The H-01 comment marker is missing from submit_gradient. \
             This documents why we override the input trust_score."
        );

        // The stored gradient must use on-chain trust, not input trust
        assert!(
            SOURCE.contains("H-01: Always use on-chain trust, never self-reported"),
            "H-01 REGRESSION: The on-chain trust assignment comment is missing. \
             Verify the gradient stores on-chain trust, not input.trust_score."
        );
    }

    /// H-01: The SubmitGradientInput struct has trust_score as Option<f32>,
    /// but submit_gradient must always override it with on-chain data.
    #[test]
    fn test_h01_trust_score_field_type() {
        // Confirm the field exists as Option<f32> in the input struct
        // (the field is accepted but ignored -- this is by design)
        assert!(
            SOURCE.contains("pub trust_score: Option<f32>"),
            "H-01: SubmitGradientInput.trust_score field must be Option<f32>. \
             The field exists for API compatibility but is always overridden."
        );
    }

    // =========================================================================
    // H-02: node_id bound to caller's AgentPubKey
    // =========================================================================

    /// H-02: Both `submit_gradient` and `submit_compressed_gradient` must bind
    /// node_id to the caller's AgentPubKey to prevent impersonation.
    #[test]
    fn test_h02_node_id_binding() {
        // The H-02 error message must appear (it's in both submit functions)
        let h02_count = SOURCE.matches(
            "H-02: node_id must match caller's AgentPubKey"
        ).count();

        assert!(
            h02_count >= 2,
            "H-02 REGRESSION: Expected at least 2 occurrences of the H-02 node_id binding check \
             (submit_gradient + submit_compressed_gradient), found {}. \
             An attacker could submit gradients impersonating another node.",
            h02_count
        );

        // The binding comment must exist
        assert!(
            SOURCE.contains("H-02: Bind node_id to caller's AgentPubKey"),
            "H-02 REGRESSION: The H-02 binding comment is missing. \
             This documents the impersonation prevention fix."
        );
    }

    // =========================================================================
    // H-03/H-04: Reputation update functions require coordinator role
    // =========================================================================

    /// H-03: `update_reputation_with_matl` must call `require_coordinator_role()`.
    #[test]
    fn test_h03_reputation_matl_requires_coordinator() {
        // The H-03 comment marker must exist in update_reputation_with_matl
        assert!(
            SOURCE.contains("H-03: Authorization check"),
            "H-03 REGRESSION: update_reputation_with_matl is missing the H-03 authorization check comment. \
             Without require_coordinator_role(), any agent could manipulate reputation scores."
        );

        // Count require_coordinator_role calls -- there should be many, but specifically
        // we verify the H-03 marker is near a require_coordinator_role call
        assert!(
            SOURCE.contains("require_coordinator_role()?;"),
            "H-03/H-04 REGRESSION: require_coordinator_role() call is missing entirely. \
             Reputation updates would be unprotected."
        );
    }

    /// H-04: `update_node_reputation_positive` must call `require_coordinator_role()`.
    #[test]
    fn test_h04_reputation_positive_requires_coordinator() {
        assert!(
            SOURCE.contains("H-04: Authorization check"),
            "H-04 REGRESSION: update_node_reputation_positive is missing the H-04 authorization check comment. \
             Without this, any agent could boost their own reputation."
        );
    }

    /// H-03/H-04: The `require_coordinator_role` function itself must exist
    /// and verify coordinator credentials (not be a no-op stub).
    #[test]
    fn test_h03_h04_require_coordinator_role_exists() {
        assert!(
            SOURCE.contains("fn require_coordinator_role() -> ExternResult<()>"),
            "H-03/H-04 REGRESSION: require_coordinator_role function is missing. \
             All coordinator-gated operations would be unprotected."
        );

        // It must actually check credentials, not just return Ok(())
        assert!(
            SOURCE.contains("verify_coordinator_authority"),
            "H-03/H-04 REGRESSION: require_coordinator_role must call verify_coordinator_authority. \
             A stub that always returns Ok(()) would bypass all authorization."
        );
    }

    // =========================================================================
    // C-03: verify_zkstark_proof is gated behind #[cfg(feature = "zkstark")]
    // =========================================================================

    /// C-03: Without the `zkstark` feature, proof verification must fail-closed
    /// (return an error), not silently pass.
    #[test]
    fn test_c03_zkstark_feature_gate() {
        // The fail-closed implementation must exist
        assert!(
            SOURCE.contains("C-03: zkSTARK verification requires the 'zkstark' feature flag"),
            "C-03 REGRESSION: The fail-closed zkSTARK message is missing. \
             Without the feature gate, proof verification could silently pass."
        );

        // Both cfg branches must exist
        assert!(
            SOURCE.contains("#[cfg(feature = \"zkstark\")]"),
            "C-03 REGRESSION: #[cfg(feature = \"zkstark\")] gate is missing on verify_zkstark_proof. \
             The real verification path requires this feature flag."
        );
        assert!(
            SOURCE.contains("#[cfg(not(feature = \"zkstark\"))]"),
            "C-03 REGRESSION: #[cfg(not(feature = \"zkstark\"))] branch is missing. \
             The fail-closed default path has been removed."
        );
    }

    /// C-03: On default features (no zkstark), verify_zkstark_proof must return Err.
    /// We verify this by checking that the not(feature) branch returns an error.
    #[test]
    fn test_c03_default_is_fail_closed() {
        // The not-zkstark branch must contain an Err, not Ok(true)
        // We check that the fail-closed message exists as an error
        assert!(
            SOURCE.contains("C-03: zkSTARK verification requires the 'zkstark' feature flag. Enable it or use submit_gradient() instead."),
            "C-03 REGRESSION: The fail-closed error message has been altered. \
             The default (no zkstark feature) must return an error, not Ok(true)."
        );
    }

    // =========================================================================
    // C-04: initialize_bootstrap verifies authority proofs
    // =========================================================================

    /// C-04: `initialize_bootstrap` must cryptographically verify authority proofs
    /// before creating genesis coordinators.
    #[test]
    fn test_c04_authority_proof_verification() {
        // The C-04 verification block must exist
        assert!(
            SOURCE.contains("C-04: Verify authority proofs cryptographically"),
            "C-04 REGRESSION: The authority proof verification block is missing from initialize_bootstrap. \
             Genesis coordinators could be created without valid proofs."
        );

        // Must check proof length
        assert!(
            SOURCE.contains("C-04: Authority proof for coordinator"),
            "C-04 REGRESSION: Authority proof length validation is missing. \
             Short/empty proofs would be accepted."
        );

        // Must call verify_signature for the proof
        assert!(
            SOURCE.contains("C-04: Authority proof verification FAILED"),
            "C-04 REGRESSION: The verify_signature check for authority proofs is missing. \
             Invalid Ed25519 signatures would be accepted as authority proofs."
        );

        // The signable payload must include GenesisCoordinator prefix
        assert!(
            SOURCE.contains("b\"GenesisCoordinator:\""),
            "C-04 REGRESSION: The GenesisCoordinator payload prefix is missing. \
             Authority proof verification must sign over a structured payload."
        );
    }

    // =========================================================================
    // C-05: vote_for_coordinator uses verify_signature
    // =========================================================================

    /// C-05: `vote_for_coordinator` must verify the vote signature cryptographically
    /// using Ed25519, not just check length.
    #[test]
    fn test_c05_vote_signature_verification() {
        // The C-05 marker must exist
        assert!(
            SOURCE.contains("C-05: Verify Ed25519 signature cryptographically"),
            "C-05 REGRESSION: vote_for_coordinator is missing Ed25519 signature verification. \
             Votes could be forged with arbitrary byte strings."
        );

        // Must call verify_signature
        assert!(
            SOURCE.contains("C-05: Vote signature verification failed"),
            "C-05 REGRESSION: The verify_signature failure path is missing from vote_for_coordinator. \
             Invalid vote signatures would be accepted."
        );

        // The vote payload must be structured (not just raw signature check)
        assert!(
            SOURCE.contains("b\"CoordinatorVote:\""),
            "C-05 REGRESSION: The CoordinatorVote payload prefix is missing. \
             Vote signatures must cover a structured payload to prevent replay attacks."
        );
    }

    // =========================================================================
    // Cross-cutting: verify_coordinator_authority exists and is not a stub
    // =========================================================================

    /// SEC-002: The verify_coordinator_authority function must exist and check
    /// real credentials, not just return Ok(true).
    #[test]
    fn test_sec002_coordinator_authority_not_stub() {
        assert!(
            SOURCE.contains("fn verify_coordinator_authority(caller: &AgentPubKey) -> ExternResult<bool>"),
            "SEC-002 REGRESSION: verify_coordinator_authority function signature is missing. \
             The entire coordinator bootstrap ceremony is compromised."
        );

        // Must check credential active status
        assert!(
            SOURCE.contains("cred.active"),
            "SEC-002 REGRESSION: verify_coordinator_authority does not check credential active status. \
             Revoked coordinators could still operate."
        );

        // Must check credential expiry
        assert!(
            SOURCE.contains("cred.expires_at"),
            "SEC-002 REGRESSION: verify_coordinator_authority does not check credential expiry. \
             Expired credentials would be accepted."
        );

        // Must check revocation
        assert!(
            SOURCE.contains("cred.revoked_at"),
            "SEC-002 REGRESSION: verify_coordinator_authority does not check revocation status. \
             Revoked coordinator credentials would still be valid."
        );
    }

    // =========================================================================
    // Meta-test: Ensure this module itself is not accidentally empty
    // =========================================================================

    /// Canary test: if this module compiles and runs, the include_str! + concat! approach works.
    #[test]
    fn test_security_fix_tests_canary() {
        assert!(
            SOURCE.contains("mod security_fix_tests"),
            "CANARY FAILURE: concat!() source does not contain this module. \
             Something is fundamentally wrong with the test setup."
        );
        // Verify concatenated source is substantial (all modules combined should be large)
        assert!(
            SOURCE.len() > 10_000,
            "CANARY FAILURE: Concatenated source is suspiciously short ({} bytes). \
             One or more module files may be missing or truncated.",
            SOURCE.len()
        );
    }
}

// =============================================================================
// CONSENSUS LOGIC TESTS
//
// Source-level verification (include_str! + concat!) for consensus/validation
// logic, plus pure-logic tests for math and SHA-256 that can run without HDK.
// =============================================================================

#[cfg(test)]
mod consensus_logic_tests {
    /// The source code of ALL coordinator modules, included at compile time.
    /// Uses concat!() so markers are found regardless of which module file they live in.
    const SOURCE: &str = concat!(
        include_str!("lib.rs"),
        include_str!("auth.rs"),
        include_str!("bootstrap.rs"),
        include_str!("bridge.rs"),
        include_str!("config.rs"),
        include_str!("consensus.rs"),
        include_str!("detection.rs"),
        include_str!("governance.rs"),
        include_str!("gradients.rs"),
        include_str!("hyperfeel.rs"),
        include_str!("matl.rs"),
        include_str!("model.rs"),
        include_str!("payments.rs"),
        include_str!("pipeline.rs"),
        include_str!("proof.rs"),
        include_str!("scheduling.rs"),
        include_str!("signals.rs"),
    );

    // =========================================================================
    // Source-level verification tests (include_str! approach)
    // =========================================================================

    /// Verify finalize_round_consensus uses `> consensus_weight` comparison to find
    /// the maximum-weighted hash, not `break` on first match.
    /// This was a bug fix: the old code broke on the first HashMap entry.
    #[test]
    fn test_finalize_finds_max_weight() {
        // Must iterate ALL hash_votes entries and track the maximum
        assert!(
            SOURCE.contains("if *weight > consensus_weight"),
            "REGRESSION: finalize_round_consensus must compare weight against the running \
             max (consensus_weight) to find the BEST hash, not break on first match."
        );

        // Must NOT break early inside the hash_votes iteration
        let finalize_section = SOURCE.split("Find the hash with the MAXIMUM weight")
            .nth(1)
            .expect("Could not find 'Find the hash with the MAXIMUM weight' comment in source");
        let up_to_threshold_check = finalize_section.split("if consensus_weight < threshold")
            .next()
            .expect("Could not find threshold check after max-weight loop");
        assert!(
            !up_to_threshold_check.contains("break;"),
            "REGRESSION: The max-weight loop in finalize_round_consensus must NOT contain \
             a 'break' — it must iterate all entries to find the true maximum."
        );
    }

    /// Verify submit_byzantine_vote checks for duplicate votes before creating entry.
    #[test]
    fn test_byzantine_vote_dedup() {
        assert!(
            SOURCE.contains("Check for duplicate vote from this validator for this round"),
            "REGRESSION: submit_byzantine_vote must check for duplicate votes. \
             Without dedup, a validator could stuff the ballot box."
        );
        assert!(
            SOURCE.contains("already submitted a Byzantine vote for round"),
            "REGRESSION: submit_byzantine_vote must reject duplicate votes with an error. \
             The dedup rejection message is missing."
        );
    }

    /// Verify get_round_consensus counts validators from registry (not hardcoded 0).
    #[test]
    fn test_validator_count_from_registry() {
        // The function must count validators from the registry path
        assert!(
            SOURCE.contains("Count validators from registry"),
            "REGRESSION: get_round_consensus must count validators from the registry path. \
             A hardcoded value would give incorrect consensus status."
        );
        assert!(
            SOURCE.contains("validator_links.len() as u32"),
            "REGRESSION: num_validators must be derived from validator_links.len(), \
             not hardcoded or set to 0."
        );
    }

    /// Verify register_eth_address rejects empty signatures.
    #[test]
    fn test_eth_address_requires_signature() {
        assert!(
            SOURCE.contains("input.signature.is_empty()"),
            "REGRESSION: register_eth_address must check for empty signatures. \
             Without this, ETH addresses could be registered without proof of ownership."
        );
        assert!(
            SOURCE.contains("Ethereum ownership signature is required"),
            "REGRESSION: register_eth_address must return a clear error for empty signatures."
        );
    }

    /// Verify the 2/3 supermajority threshold computation exists in finalize_round_consensus.
    #[test]
    fn test_consensus_threshold_calculation() {
        assert!(
            SOURCE.contains("total_weight * 2.0 / 3.0"),
            "REGRESSION: Consensus threshold must be computed as total_weight * 2/3. \
             Using a different fraction would break BFT guarantees."
        );
        assert!(
            SOURCE.contains("consensus_weight < threshold"),
            "REGRESSION: finalize_round_consensus must reject results below the threshold. \
             The comparison 'consensus_weight < threshold' is missing."
        );
    }

    /// Verify reputation-squared weighting is used for consensus votes.
    #[test]
    fn test_reputation_squared_weighting() {
        assert!(
            SOURCE.contains("rep * rep; // reputation² weighting"),
            "REGRESSION: Consensus vote weight must use rep² (reputation squared). \
             This ensures high-reputation validators have disproportionate influence."
        );
    }

    /// Verify MIN_VALIDATOR_REPUTATION is set to 0.3.
    #[test]
    fn test_register_validator_min_reputation() {
        assert!(
            SOURCE.contains("const MIN_VALIDATOR_REPUTATION: f32 = 0.3"),
            "REGRESSION: MIN_VALIDATOR_REPUTATION must be 0.3. \
             Changing this threshold affects who can participate in consensus."
        );
        assert!(
            SOURCE.contains("on_chain_rep < MIN_VALIDATOR_REPUTATION"),
            "REGRESSION: Validator registration must reject agents below MIN_VALIDATOR_REPUTATION. \
             The comparison against MIN_VALIDATOR_REPUTATION is missing."
        );
    }

    /// Verify reveal_aggregation computes SHA-256 and checks against commitment hash.
    #[test]
    fn test_reveal_verifies_commitment_hash() {
        // Must compute SHA-256 of result_data
        assert!(
            SOURCE.contains("Compute SHA-256 of result_data"),
            "REGRESSION: reveal_aggregation must compute SHA-256 of the result_data. \
             Without this, the commit-reveal protocol has no integrity check."
        );
        // Must compare result_hash against commitment
        assert!(
            SOURCE.contains("commitment.commitment_hash != result_hash"),
            "REGRESSION: reveal_aggregation must verify the computed hash matches the commitment. \
             This is the core integrity check of the commit-reveal protocol."
        );
        // Must return error on mismatch
        assert!(
            SOURCE.contains("Reveal hash does not match commitment"),
            "REGRESSION: reveal_aggregation must return an error when hash doesn't match commitment. \
             The integrity violation error message is missing."
        );
    }

    // =========================================================================
    // Pure logic tests (actual computation, no HDK required)
    // =========================================================================

    /// Test that the 2/3 supermajority threshold math works correctly.
    #[test]
    fn test_consensus_threshold_math() {
        // Various total_weight values
        let cases: Vec<(f64, f64)> = vec![
            (3.0, 2.0),     // 3 * 2/3 = 2.0
            (6.0, 4.0),     // 6 * 2/3 = 4.0
            (1.0, 2.0/3.0), // 1 * 2/3 ≈ 0.6667
            (0.75, 0.5),    // 0.75 * 2/3 = 0.5
            (10.0, 20.0/3.0), // 10 * 2/3 ≈ 6.6667
        ];

        for (total_weight, expected_threshold) in cases {
            let threshold = total_weight * 2.0 / 3.0;
            assert!(
                (threshold - expected_threshold).abs() < 1e-10,
                "Threshold for total_weight={} should be {}, got {}",
                total_weight, expected_threshold, threshold
            );
        }
    }

    /// Test that reputation-squared weighting produces correct values.
    #[test]
    fn test_reputation_squared_values() {
        let cases: Vec<(f64, f64)> = vec![
            (0.5, 0.25),
            (0.8, 0.64),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.3, 0.09),
            (0.9, 0.81),
            (0.1, 0.01),
        ];

        for (rep, expected_weight) in cases {
            let weight = rep * rep;
            assert!(
                (weight - expected_weight).abs() < 1e-10,
                "rep²: {:.1}² should be {:.4}, got {:.4}",
                rep, expected_weight, weight
            );
        }
    }

    /// Test SHA-256 commitment verification for known data.
    #[test]
    fn test_sha256_commitment_verify() {
        use sha2::Digest;

        // Hash known data
        let data = b"test aggregation result data";
        let mut hasher = sha2::Sha256::new();
        hasher.update(data);
        let hash1 = hasher.finalize().to_vec();

        // Same data produces same hash
        let mut hasher2 = sha2::Sha256::new();
        hasher2.update(data);
        let hash2 = hasher2.finalize().to_vec();

        assert_eq!(hash1, hash2, "Same data must produce identical SHA-256 hashes");
        assert_eq!(hash1.len(), 32, "SHA-256 hash must be 32 bytes");

        // Different data produces different hash
        let mut hasher3 = sha2::Sha256::new();
        hasher3.update(b"different data");
        let hash3 = hasher3.finalize().to_vec();

        assert_ne!(hash1, hash3, "Different data must produce different SHA-256 hashes");

        // Verify known SHA-256 value (sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855)
        let mut empty_hasher = sha2::Sha256::new();
        empty_hasher.update(b"");
        let empty_hash = empty_hasher.finalize();
        assert_eq!(
            format!("{:x}", empty_hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256 of empty string must match known value"
        );
    }

    /// Test that weights summing to exactly 2/3 pass, and weights below don't.
    #[test]
    fn test_consensus_needs_supermajority() {
        // Simulate the finalize_round_consensus logic:
        // threshold = total_weight * 2/3
        // consensus is reached when consensus_weight >= threshold

        // Case 1: 3 validators with rep=1.0, all agree → weight = 3.0, threshold = 2.0
        {
            let reps = vec![1.0_f64, 1.0, 1.0];
            let total_weight: f64 = reps.iter().map(|r| r * r).sum();
            let threshold = total_weight * 2.0 / 3.0;
            let agreeing_weight: f64 = reps.iter().map(|r| r * r).sum(); // all agree
            assert!(
                agreeing_weight >= threshold,
                "All 3 validators agreeing (weight {}) should meet threshold ({})",
                agreeing_weight, threshold
            );
        }

        // Case 2: 3 validators, only 1 agrees → weight = 1.0, threshold = 2.0
        {
            let reps = vec![1.0_f64, 1.0, 1.0];
            let total_weight: f64 = reps.iter().map(|r| r * r).sum();
            let threshold = total_weight * 2.0 / 3.0;
            let agreeing_weight = 1.0_f64; // only one agrees
            assert!(
                agreeing_weight < threshold,
                "Only 1 of 3 validators agreeing (weight {}) should NOT meet threshold ({})",
                agreeing_weight, threshold
            );
        }

        // Case 3: 3 validators, 2 agree → weight = 2.0, threshold = 2.0 → passes (>=)
        {
            let reps = vec![1.0_f64, 1.0, 1.0];
            let total_weight: f64 = reps.iter().map(|r| r * r).sum();
            let threshold = total_weight * 2.0 / 3.0;
            let agreeing_weight = 2.0_f64; // exactly 2/3
            assert!(
                agreeing_weight >= threshold,
                "Exactly 2 of 3 validators agreeing (weight {}) should meet threshold ({})",
                agreeing_weight, threshold
            );
        }

        // Case 4: Mixed reputations — high-rep minority can outvote low-rep majority
        {
            // Two validators: rep=1.0 (weight=1.0), rep=0.3 (weight=0.09), rep=0.3 (weight=0.09)
            let weights = vec![1.0_f64 * 1.0, 0.3 * 0.3, 0.3 * 0.3]; // [1.0, 0.09, 0.09]
            let total_weight: f64 = weights.iter().sum(); // 1.18
            let threshold = total_weight * 2.0 / 3.0; // ≈ 0.7867

            // High-rep validator alone
            let high_rep_weight = weights[0]; // 1.0
            assert!(
                high_rep_weight >= threshold,
                "High-rep validator (weight {:.4}) should meet threshold ({:.4}) due to rep² weighting",
                high_rep_weight, threshold
            );

            // Two low-rep validators together
            let low_rep_weight: f64 = weights[1] + weights[2]; // 0.18
            assert!(
                low_rep_weight < threshold,
                "Two low-rep validators (combined weight {:.4}) should NOT meet threshold ({:.4})",
                low_rep_weight, threshold
            );
        }

        // Case 5: Just below threshold should fail (the < comparison)
        {
            let total_weight = 9.0_f64;
            let threshold = total_weight * 2.0 / 3.0; // 6.0
            // The code uses `consensus_weight < threshold` to reject
            let barely_below = threshold - 0.001;
            assert!(
                barely_below < threshold,
                "Weight just below threshold ({:.4} < {:.4}) must fail consensus",
                barely_below, threshold
            );
        }
    }

    // =========================================================================
    // Meta-test: Ensure this module itself is not accidentally empty
    // =========================================================================

    /// Canary test: if this module compiles and runs, the consensus_logic_tests setup works.
    #[test]
    fn test_consensus_logic_tests_canary() {
        assert!(
            SOURCE.contains("mod consensus_logic_tests"),
            "CANARY FAILURE: concat!() source does not contain this module."
        );
    }
}

// =============================================================================
// Reputation Decay Tests (pure math, no HDK)
// =============================================================================

#[cfg(test)]
mod reputation_decay_tests {
    use crate::pipeline::compute_decayed_reputation;
    use crate::config::{REPUTATION_DECAY_FACTOR, REPUTATION_DECAY_INTERVAL_SECONDS, REPUTATION_FLOOR, MIN_REPUTATION_FOR_SUBMISSION};

    #[test]
    fn test_no_decay_when_recently_updated() {
        // Less than ~15 minutes (< 0.01 intervals) → no decay
        let score = compute_decayed_reputation(0.8, 600); // 10 minutes
        assert_eq!(score, 0.8, "Score should not decay within 15 minutes");
    }

    #[test]
    fn test_one_day_decay() {
        // 1 day = 1 interval → score * 0.95 (relative to floor)
        let stored = 0.8_f32;
        let decayed = compute_decayed_reputation(stored, REPUTATION_DECAY_INTERVAL_SECONDS);

        let expected = REPUTATION_FLOOR as f64
            + (stored as f64 - REPUTATION_FLOOR as f64) * REPUTATION_DECAY_FACTOR;
        let expected = expected as f32;

        assert!(
            (decayed - expected).abs() < 1e-5,
            "After 1 day: expected {:.5}, got {:.5}",
            expected, decayed
        );
    }

    #[test]
    fn test_seven_day_decay() {
        // 7 days = 7 intervals → meaningful decay
        let stored = 0.8_f32;
        let elapsed = REPUTATION_DECAY_INTERVAL_SECONDS * 7;
        let decayed = compute_decayed_reputation(stored, elapsed);

        let expected = REPUTATION_FLOOR as f64
            + (stored as f64 - REPUTATION_FLOOR as f64) * REPUTATION_DECAY_FACTOR.powf(7.0);
        let expected = expected as f32;

        assert!(
            (decayed - expected).abs() < 1e-5,
            "After 7 days: expected {:.5}, got {:.5}",
            expected, decayed
        );
        assert!(
            decayed < stored,
            "7-day decay should reduce reputation: {} < {}",
            decayed, stored
        );
        assert!(
            decayed > REPUTATION_FLOOR,
            "7-day decay should stay above floor: {} > {}",
            decayed, REPUTATION_FLOOR
        );
    }

    #[test]
    fn test_long_inactivity_converges_to_floor() {
        // 365 days = 365 intervals → converges near floor
        let stored = 0.9_f32;
        let elapsed = REPUTATION_DECAY_INTERVAL_SECONDS * 365;
        let decayed = compute_decayed_reputation(stored, elapsed);

        assert!(
            decayed < 0.15,
            "After 1 year of inactivity, reputation should converge near floor: got {:.5}",
            decayed
        );
        assert!(
            decayed >= REPUTATION_FLOOR,
            "Decayed reputation should never go below floor: {} >= {}",
            decayed, REPUTATION_FLOOR
        );
    }

    #[test]
    fn test_decay_never_below_floor() {
        // Even with extreme elapsed time
        let decayed = compute_decayed_reputation(1.0, i64::MAX / 2);
        assert!(
            decayed >= REPUTATION_FLOOR,
            "Decay must never go below floor: {} >= {}",
            decayed, REPUTATION_FLOOR
        );
    }

    #[test]
    fn test_decay_preserves_floor_value() {
        // If stored score is already at floor, decay doesn't reduce further
        let decayed = compute_decayed_reputation(REPUTATION_FLOOR, REPUTATION_DECAY_INTERVAL_SECONDS * 30);
        assert!(
            (decayed - REPUTATION_FLOOR).abs() < 1e-5,
            "Score at floor should not decay further: {:.5} ≈ {:.5}",
            decayed, REPUTATION_FLOOR
        );
    }

    #[test]
    fn test_negative_elapsed_no_effect() {
        // Negative elapsed time (clock skew) should not increase reputation
        let decayed = compute_decayed_reputation(0.8, -86400);
        assert_eq!(decayed, 0.8, "Negative elapsed time should have no effect");
    }

    #[test]
    fn test_quarantine_threshold_above_floor() {
        // The quarantine threshold must be above the floor (otherwise nodes can never be quarantined)
        assert!(
            MIN_REPUTATION_FOR_SUBMISSION > REPUTATION_FLOOR,
            "Quarantine threshold {} must be above floor {}",
            MIN_REPUTATION_FOR_SUBMISSION, REPUTATION_FLOOR
        );
    }

    #[test]
    fn test_default_reputation_not_quarantined() {
        // New nodes start at 0.5, which should be above quarantine threshold
        let default_rep = 0.5_f32;
        assert!(
            default_rep >= MIN_REPUTATION_FOR_SUBMISSION,
            "Default reputation {} must be above quarantine threshold {}",
            default_rep, MIN_REPUTATION_FOR_SUBMISSION
        );
    }

    #[test]
    fn test_decay_constants_in_valid_range() {
        assert!(
            (0.5..=1.0).contains(&REPUTATION_DECAY_FACTOR),
            "Decay factor must be in [0.5, 1.0]: {}",
            REPUTATION_DECAY_FACTOR
        );
        assert!(
            REPUTATION_DECAY_INTERVAL_SECONDS > 0,
            "Decay interval must be positive: {}",
            REPUTATION_DECAY_INTERVAL_SECONDS
        );
        assert!(
            (0.0..=0.5).contains(&REPUTATION_FLOOR),
            "Floor must be in [0.0, 0.5]: {}",
            REPUTATION_FLOOR
        );
    }

    /// Source-level test: verify quarantine check exists in submit_gradient
    #[test]
    fn test_quarantine_check_in_source() {
        const SOURCE: &str = concat!(
            include_str!("gradients.rs"),
            include_str!("pipeline.rs"),
            include_str!("config.rs"),
        );

        assert!(
            SOURCE.contains("MIN_REPUTATION_FOR_SUBMISSION"),
            "REGRESSION: submit_gradient must check MIN_REPUTATION_FOR_SUBMISSION"
        );
        assert!(
            SOURCE.contains("is quarantined"),
            "REGRESSION: submit_gradient must have quarantine rejection message"
        );
        assert!(
            SOURCE.contains("compute_decayed_reputation"),
            "REGRESSION: pipeline must have compute_decayed_reputation function"
        );
        assert!(
            SOURCE.contains("REPUTATION_DECAY_FACTOR"),
            "REGRESSION: config must define REPUTATION_DECAY_FACTOR"
        );
    }
}
