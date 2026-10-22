// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # PQC-Assisted KEM Handshake — Claim-Integrity Review
//!
//! This module combines an Ed25519 challenge-response API with ML-KEM-768
//! encapsulation. The current code is a research integration, **not a reviewed
//! hybrid key-exchange protocol**.
//!
//! ## Current data flow
//!
//! ```text
//! Ed25519 challenge-response (separate API)
//!   → challenge nonce is public protocol input, not a shared secret
//! ML-KEM-768 encapsulation
//!   → shared secret comes from the KEM
//! Session derivation
//!   → BLAKE3 derive_key(context, public_challenge || kem_shared_secret)
//! ```
//!
//! ## Security claim boundaries
//!
//! - Mixing a public challenge nonce with the KEM secret does not add an
//!   independent classical secret contribution to confidentiality.
//! - This API does not itself enforce that the Ed25519 exchange completed, nor
//!   bind both peer identities, the KEM public key and KEM ciphertext to one
//!   authenticated transcript. Callers must not infer authentication simply
//!   because they invoked the session-key API.
//! - The manager holds one ML-KEM keypair across sessions. Recorded KEM
//!   ciphertexts can be decapsulated if that private key is later compromised;
//!   the current design therefore does not provide KEM-based forward secrecy.
//! - Do not claim the session remains secure if either the classical or KEM
//!   component fails without a specified combiner proof, authenticated key
//!   binding, and independent protocol review.
//!
//! A production protocol needs an authenticated, transcript-bound handshake,
//! explicit downgrade rules, per-session key lifecycle/erasure, and independent
//! tests against active attacks. See
//! `docs/security/MESH_KEY_AGREEMENT_CLAIM_REVIEW.md`.
//!
//! ## Feature Gate
//!
//! Requires `pqc-handshake` feature (implies `identity` + `mesh-key-exchange`
//! + `mycelix-crypto` with `native` feature).
//!
//! ## References
//!
//! - NIST FIPS 203 (2024) — ML-KEM (Module-Lattice Key Encapsulation Mechanism)
//! - Bindel et al. (2019) — Hybrid key exchange in TLS 1.3
//! - Schwabe et al. (2020) — CRYSTALS-Kyber algorithm specification

use crate::swarm::{SwarmConfig, SwarmError, SwarmResult};
use mycelix_crypto::pqc::ml_kem::MlKem768KeyPair;
use mycelix_crypto::traits::KeyEncapsulator;
use mycelix_crypto::{AlgorithmId, TaggedPublicKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zeroize::Zeroize;

// ML-KEM-768 encodings from FIPS 203. Validate lengths before storing or decapsulating.
const ML_KEM_768_PUBLIC_KEY_BYTES: usize = 1_184;
const ML_KEM_768_CIPHERTEXT_BYTES: usize = 1_088;
/// Bound caller-controlled pending exchange state (public keys plus candidate keys).
const MAX_PENDING_KEM_EXCHANGES: usize = 256;
/// Bound peer labels before copying them into pending-state maps.
const MAX_KEM_PEER_ID_BYTES: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// PQC SESSION KEY
// ═══════════════════════════════════════════════════════════════════════════════

/// Unconfirmed key material derived from KEM output and public context.
///
/// This is not an established session key. It remains in the pending map until
/// a reviewed protocol implements transcript binding and explicit key confirmation.
struct PqcCandidateKey {
    candidate_key: [u8; 32],
    created_at: std::time::SystemTime,
}

impl PqcCandidateKey {
    fn key(&self) -> &[u8; 32] {
        &self.candidate_key
    }

    fn created_at(&self) -> std::time::SystemTime {
        self.created_at
    }
}

impl Drop for PqcCandidateKey {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.candidate_key.zeroize();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KEM PUBLIC KEY EXCHANGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Data sent during PQC handshake phase 2: KEM public key + encapsulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KemExchange {
    /// ML-KEM-768 public key bytes (1184 bytes).
    pub kem_public_key: Vec<u8>,
    /// KEM ciphertext (1088 bytes) — only present in the initiator's response.
    pub kem_ciphertext: Option<Vec<u8>>,
    /// Peer node ID for routing.
    pub peer_node_id: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PQC HANDSHAKE MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages research-stage ML-KEM exchange state.
///
/// This type does not currently establish a completed hybrid session: KEM
/// results remain pending until a reviewed, transcript-bound key-confirmation
/// protocol exists. Do not treat candidate derivation as session establishment.
pub struct PqcHandshakeManager {
    /// Classical handshake manager (Ed25519 or BLAKE3 fallback).
    classical: crate::swarm::handshake::HybridHandshake,
    /// This node's ML-KEM-768 keypair.
    kem_keypair: MlKem768KeyPair,
    /// Candidate keys derived from KEM but not yet transcript-confirmed.
    pending_session_keys: HashMap<String, PqcCandidateKey>,
    /// Pending KEM exchanges (peer_id → their KEM public key).
    pending_kem: HashMap<String, Vec<u8>>,
}

impl PqcHandshakeManager {
    /// Create a new PQC handshake manager.
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            classical: crate::swarm::handshake::HybridHandshake::new(config),
            kem_keypair: MlKem768KeyPair::generate(),
            pending_session_keys: HashMap::new(),
            pending_kem: HashMap::new(),
        }
    }

    /// Export this node's KEM public key only when a real cryptographic backend is linked.
    ///
    /// The workspace-local `mycelix-crypto` dependency is a compilation stub
    /// with deterministic all-zero outputs. It is explicitly refused here.
    pub fn kem_public_key_bytes(&self) -> SwarmResult<Vec<u8>> {
        Self::ensure_real_kem_backend()?;
        let tagged = self.kem_keypair.public_key();
        Ok(tagged.key_bytes.clone())
    }

    fn ensure_real_kem_backend() -> SwarmResult<()> {
        if mycelix_crypto::CRYPTO_BACKEND_IS_STUB {
            return Err(SwarmError::FeatureNotEnabled {
                feature: "real ML-KEM-768 backend (workspace crypto stub is active)".to_string(),
            });
        }
        Ok(())
    }

    /// Get the classical handshake manager (for Ed25519 operations).
    pub fn classical(&self) -> &crate::swarm::handshake::HybridHandshake {
        &self.classical
    }

    /// Get the classical handshake manager mutably.
    pub fn classical_mut(&mut self) -> &mut crate::swarm::handshake::HybridHandshake {
        &mut self.classical
    }

    /// Get an established session key for a peer.
    ///
    /// KEM output alone is not an established session. Until authenticated
    /// transcript binding and explicit key confirmation are implemented, this
    /// returns `None` for candidate keys.
    pub fn session_key(&self, _peer_id: &str) -> Option<&[u8; 32]> {
        // No session-activation API exists yet. Candidate material is never an active key.
        None
    }

    /// Number of established sessions.
    pub fn session_count(&self) -> usize {
        0
    }

    /// Number of derived candidate keys awaiting transcript/key confirmation.
    pub fn pending_session_count(&self) -> usize {
        self.pending_session_keys.len()
    }

    /// Remove a peer's active and pending key material (e.g., on disconnect).
    pub fn remove_session(&mut self, peer_id: &str) {
        self.pending_session_keys.remove(peer_id);
        self.pending_kem.remove(peer_id);
    }

    // ════════════════════════════════════════════════════════════════════
    // INITIATOR SIDE
    // ════════════════════════════════════════════════════════════════════

    /// Store a candidate responder KEM public key after the caller has
    /// authenticated its identity/transcript.
    ///
    /// This method validates the ML-KEM-768 encoding size and bounds peer labels
    /// and pending entries. It does not authenticate the supplied public key.
    pub fn receive_kem_public_key(
        &mut self,
        peer_id: &str,
        kem_public_key: &[u8],
    ) -> SwarmResult<()> {
        if peer_id.is_empty() || peer_id.len() > MAX_KEM_PEER_ID_BYTES {
            return Err(SwarmError::TrustVerificationError {
                reason: "PQC peer identifier is empty or exceeds the 256-byte limit".to_string(),
            });
        }
        if kem_public_key.len() != ML_KEM_768_PUBLIC_KEY_BYTES {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid ML-KEM-768 public key length: expected {ML_KEM_768_PUBLIC_KEY_BYTES}, got {}",
                    kem_public_key.len()
                ),
            });
        }
        if !self.pending_kem.contains_key(peer_id)
            && self.pending_kem.len() >= MAX_PENDING_KEM_EXCHANGES
        {
            return Err(SwarmError::TrustVerificationError {
                reason: "Maximum pending PQC key exchanges reached".to_string(),
            });
        }
        Self::ensure_real_kem_backend()?;
        self.pending_kem
            .insert(peer_id.to_string(), kem_public_key.to_vec());
        Ok(())
    }

    /// Encapsulate a shared secret to the responder's supplied KEM public key.
    ///
    /// The `classical_nonce` is public challenge/context input; it is not a
    /// classical shared secret. This method does not verify that the classical
    /// handshake succeeded or that this KEM public key is authenticated for
    /// `peer_id`; the caller must enforce those conditions before use.
    ///
    /// Returns the KEM ciphertext and stores only a pending candidate key.
    /// The candidate is not returned by `session_key` until a future reviewed
    /// protocol supplies authenticated transcript binding and key confirmation.
    /// The current KEM private key is reused across sessions, so this API does
    /// not provide forward secrecy against later compromise of that key.
    pub fn encapsulate_for_peer(
        &mut self,
        peer_id: &str,
        classical_nonce: &[u8],
    ) -> SwarmResult<Vec<u8>> {
        if peer_id.is_empty() || peer_id.len() > MAX_KEM_PEER_ID_BYTES {
            return Err(SwarmError::TrustVerificationError {
                reason: "PQC peer identifier is empty or exceeds the 256-byte limit".to_string(),
            });
        }
        if !self.pending_session_keys.contains_key(peer_id)
            && self.pending_session_keys.len() >= MAX_PENDING_KEM_EXCHANGES
        {
            return Err(SwarmError::TrustVerificationError {
                reason: "Maximum pending PQC candidate keys reached".to_string(),
            });
        }
        let kem_pk_bytes = self
            .pending_kem
            .get(peer_id)
            .cloned()
            .ok_or_else(|| SwarmError::TrustVerificationError {
                reason: format!("No KEM public key stored for peer '{peer_id}'"),
            })?;
        Self::ensure_real_kem_backend()?;
        self.pending_kem.remove(peer_id);

        // Build TaggedPublicKey for the responder
        let tagged_pk = TaggedPublicKey::new(AlgorithmId::MlKem768, kem_pk_bytes).map_err(|e| {
            SwarmError::TrustVerificationError {
                reason: format!("Invalid KEM public key: {e}"),
            }
        })?;

        // Encapsulate shared secret
        let (ciphertext, mut kem_shared_secret) =
            self.kem_keypair.encapsulate(&tagged_pk).map_err(|e| {
                SwarmError::TrustVerificationError {
                    reason: format!("ML-KEM-768 encapsulation failed: {e}"),
                }
            })?;

        // Derive using BLAKE3 context-based derivation; the nonce is public context.
        let mut session_key = Self::blake3_derive_session_key(classical_nonce, &kem_shared_secret);
        kem_shared_secret.zeroize();

        self.pending_session_keys.insert(
            peer_id.to_string(),
            PqcCandidateKey {
                candidate_key: session_key,
                created_at: std::time::SystemTime::now(),
            },
        );
        // Arrays are Copy; clear the temporary stack copy after insertion.
        session_key.zeroize();

        Ok(ciphertext)
    }

    // ════════════════════════════════════════════════════════════════════
    // RESPONDER SIDE
    // ════════════════════════════════════════════════════════════════════

    /// Decapsulate a KEM ciphertext and derive a pending candidate key.
    ///
    /// The candidate is not exposed as an established session and this method
    /// returns no key bytes. A future reviewed protocol must bind the ciphertext,
    /// KEM key identity, peer identities, protocol version and classical
    /// transcript, then complete explicit key confirmation before activation.
    pub fn decapsulate_and_derive(
        &mut self,
        peer_id: &str,
        kem_ciphertext: &[u8],
        classical_nonce: &[u8],
    ) -> SwarmResult<()> {
        if peer_id.is_empty() || peer_id.len() > MAX_KEM_PEER_ID_BYTES {
            return Err(SwarmError::TrustVerificationError {
                reason: "PQC peer identifier is empty or exceeds the 256-byte limit".to_string(),
            });
        }
        if kem_ciphertext.len() != ML_KEM_768_CIPHERTEXT_BYTES {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid ML-KEM-768 ciphertext length: expected {ML_KEM_768_CIPHERTEXT_BYTES}, got {}",
                    kem_ciphertext.len()
                ),
            });
        }
        if !self.pending_session_keys.contains_key(peer_id)
            && self.pending_session_keys.len() >= MAX_PENDING_KEM_EXCHANGES
        {
            return Err(SwarmError::TrustVerificationError {
                reason: "Maximum pending PQC candidate keys reached".to_string(),
            });
        }
        Self::ensure_real_kem_backend()?;

        // Decapsulate shared secret
        let mut kem_shared_secret = self.kem_keypair.decapsulate(kem_ciphertext).map_err(|e| {
            SwarmError::TrustVerificationError {
                reason: format!("ML-KEM-768 decapsulation failed: {e}"),
            }
        })?;

        // Derive using BLAKE3 context-based derivation; the nonce is public context.
        let mut session_key = Self::blake3_derive_session_key(classical_nonce, &kem_shared_secret);
        kem_shared_secret.zeroize();

        self.pending_session_keys.insert(
            peer_id.to_string(),
            PqcCandidateKey {
                candidate_key: session_key,
                created_at: std::time::SystemTime::now(),
            },
        );
        // Arrays are Copy; clear the temporary stack copy after insertion.
        session_key.zeroize();

        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    // KEY DERIVATION
    // ════════════════════════════════════════════════════════════════════

    /// Derive a 32-byte key using BLAKE3's context-based derive-key API.
    ///
    /// The public challenge bytes are concatenated with the secret ML-KEM
    /// output as input material. The challenge contributes context binding only;
    /// it is not secret entropy and this function does not implement HKDF or a
    /// proven hybrid KDF combiner.
    fn blake3_derive_session_key(
        classical_nonce: &[u8],
        kem_shared_secret: &[u8],
    ) -> [u8; 32] {
        let mut ikm = zeroize::Zeroizing::new(
            Vec::with_capacity(classical_nonce.len() + kem_shared_secret.len()),
        );
        ikm.extend_from_slice(classical_nonce);
        ikm.extend_from_slice(kem_shared_secret);
        blake3::derive_key("symthaea-pqc-session-v1", ikm.as_slice())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SwarmConfig {
        SwarmConfig::default()
    }

    #[test]
    fn test_workspace_crypto_stub_is_never_accepted_as_ml_kem() {
        assert!(
            mycelix_crypto::CRYPTO_BACKEND_IS_STUB,
            "this workspace dependency must identify itself as a test/compile-only stub"
        );
        let manager = PqcHandshakeManager::new(test_config());
        assert!(manager.kem_public_key_bytes().is_err());
        assert_eq!(manager.session_count(), 0);
        assert_eq!(manager.pending_session_count(), 0);
        assert!(manager.session_key("any-peer").is_none());
    }

    #[test]
    fn test_kem_public_key_length_and_peer_id_are_validated_before_storage() {
        let mut manager = PqcHandshakeManager::new(test_config());
        let valid_length = vec![0u8; ML_KEM_768_PUBLIC_KEY_BYTES];

        assert!(manager.receive_kem_public_key("", &valid_length).is_err());
        assert!(
            manager
                .receive_kem_public_key(&"p".repeat(MAX_KEM_PEER_ID_BYTES + 1), &valid_length)
                .is_err()
        );
        assert!(manager.pending_kem.is_empty());

        for length in [0, ML_KEM_768_PUBLIC_KEY_BYTES - 1, ML_KEM_768_PUBLIC_KEY_BYTES + 1] {
            let malformed = vec![0u8; length];
            assert!(
                manager.receive_kem_public_key("peer-invalid", &malformed).is_err(),
                "malformed public-key length {length} must be rejected"
            );
            assert!(manager.pending_kem.is_empty());
        }

        // Correct length is not sufficient: the current backend is an explicit
        // all-zero stub, so it must fail closed without storing the key.
        assert!(manager.receive_kem_public_key("peer-stub", &valid_length).is_err());
        assert!(manager.pending_kem.is_empty());
    }

    #[test]
    fn test_pending_kem_exchange_count_is_bounded() {
        let mut manager = PqcHandshakeManager::new(test_config());
        let encoded_pk = vec![0u8; ML_KEM_768_PUBLIC_KEY_BYTES];

        // Seed private test state directly to exercise the capacity guard
        // independently of the stub-backend guard.
        for index in 0..MAX_PENDING_KEM_EXCHANGES {
            manager.pending_kem.insert(format!("peer-{index}"), encoded_pk.clone());
        }
        assert_eq!(manager.pending_kem.len(), MAX_PENDING_KEM_EXCHANGES);

        assert!(
            manager.receive_kem_public_key("peer-over-limit", &encoded_pk).is_err(),
            "a new peer beyond the fixed pending limit must be rejected"
        );
        assert_eq!(manager.pending_kem.len(), MAX_PENDING_KEM_EXCHANGES);
        assert!(!manager.pending_kem.contains_key("peer-over-limit"));
    }

    #[test]
    fn test_pending_candidate_count_is_bounded_before_kem_work() {
        let mut manager = PqcHandshakeManager::new(test_config());
        let now = std::time::SystemTime::now();

        for index in 0..MAX_PENDING_KEM_EXCHANGES {
            manager.pending_session_keys.insert(
                format!("candidate-{index}"),
                PqcCandidateKey {
                    candidate_key: [index as u8; 32],
                    created_at: now,
                },
            );
        }
        manager.pending_kem.insert(
            "peer-over-limit".to_string(),
            vec![0u8; ML_KEM_768_PUBLIC_KEY_BYTES],
        );

        // Capacity checks must happen before KEM calls and before pending
        // public-key state is consumed.
        assert!(
            manager
                .encapsulate_for_peer("peer-over-limit", &[0x42u8; 32])
                .is_err()
        );
        assert!(
            manager
                .decapsulate_and_derive(
                    "peer-over-limit",
                    &vec![0u8; ML_KEM_768_CIPHERTEXT_BYTES],
                    &[0x42u8; 32],
                )
                .is_err()
        );
        assert_eq!(manager.pending_session_count(), MAX_PENDING_KEM_EXCHANGES);
        assert_eq!(manager.pending_kem.len(), 1);
        assert!(manager.pending_kem.contains_key("peer-over-limit"));
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_invalid_kem_ciphertext_lengths_do_not_create_candidates() {
        let mut manager = PqcHandshakeManager::new(test_config());
        let public_context = [0x5Au8; 32];

        for length in [0, ML_KEM_768_CIPHERTEXT_BYTES - 1, ML_KEM_768_CIPHERTEXT_BYTES + 1] {
            let ciphertext = vec![0u8; length];
            assert!(
                manager
                    .decapsulate_and_derive("peer-invalid", &ciphertext, &public_context)
                    .is_err(),
                "malformed ciphertext length {length} must be rejected"
            );
            assert_eq!(manager.pending_session_count(), 0);
        }

        // Correct ciphertext length still cannot use the compile-only stub.
        assert!(
            manager
                .decapsulate_and_derive(
                    "peer-stub",
                    &vec![0xFFu8; ML_KEM_768_CIPHERTEXT_BYTES],
                    &public_context,
                )
                .is_err()
        );
        assert_eq!(manager.pending_session_count(), 0);
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_stub_backend_cannot_create_candidate_or_active_session() {
        let mut manager = PqcHandshakeManager::new(test_config());
        let public_key = vec![0u8; ML_KEM_768_PUBLIC_KEY_BYTES];

        // The backend guard rejects a correctly sized public key. A caller
        // cannot obtain stub outputs through the supported public API.
        assert!(manager.receive_kem_public_key("peer-B", &public_key).is_err());
        assert!(manager.encapsulate_for_peer("peer-B", &[0x42u8; 32]).is_err());
        assert!(manager
            .decapsulate_and_derive(
                "peer-B",
                &vec![0u8; ML_KEM_768_CIPHERTEXT_BYTES],
                &[0x42u8; 32],
            )
            .is_err());

        assert!(manager.pending_kem.is_empty());
        assert_eq!(manager.pending_session_count(), 0);
        assert_eq!(manager.session_count(), 0);
        assert!(manager.session_key("peer-B").is_none());
    }

    #[test]
    fn test_remove_session_clears_pending_candidate_and_public_key() {
        let mut manager = PqcHandshakeManager::new(test_config());
        manager.pending_kem.insert(
            "peer-A".to_string(),
            vec![0u8; ML_KEM_768_PUBLIC_KEY_BYTES],
        );
        manager.pending_session_keys.insert(
            "peer-A".to_string(),
            PqcCandidateKey {
                candidate_key: [0x42; 32],
                created_at: std::time::SystemTime::now(),
            },
        );

        assert_eq!(manager.pending_session_count(), 1);
        assert!(manager.pending_kem.contains_key("peer-A"));
        manager.remove_session("peer-A");
        assert_eq!(manager.session_count(), 0);
        assert_eq!(manager.pending_session_count(), 0);
        assert!(!manager.pending_kem.contains_key("peer-A"));
        assert!(manager.session_key("peer-A").is_none());
    }

    #[test]
    fn test_missing_kem_public_key_fails() {
        let mut manager = PqcHandshakeManager::new(test_config());
        assert!(manager.encapsulate_for_peer("peer-X", &[0x42u8; 32]).is_err());
    }

    #[test]
    fn test_candidate_key_is_not_exposed_as_established_session() {
        let mut manager = PqcHandshakeManager::new(test_config());
        manager.pending_session_keys.insert(
            "peer-B".to_string(),
            PqcCandidateKey {
                candidate_key: [0xAB; 32],
                created_at: std::time::SystemTime::now(),
            },
        );

        let candidate = manager.pending_session_keys.get("peer-B").unwrap();
        assert_eq!(candidate.key().len(), 32);
        assert!(candidate.created_at() <= std::time::SystemTime::now());
        assert_eq!(manager.pending_session_count(), 1);
        assert_eq!(manager.session_count(), 0);
        assert!(manager.session_key("peer-B").is_none());
    }

    #[test]
    fn test_blake3_derive_session_key_deterministic() {
        let public_context = vec![0x42u8; 32];
        let test_secret = vec![0xABu8; 32];

        let key1 = PqcHandshakeManager::blake3_derive_session_key(&public_context, &test_secret);
        let key2 = PqcHandshakeManager::blake3_derive_session_key(&public_context, &test_secret);
        assert_eq!(key1, key2, "BLAKE3 derive-key derivation must be deterministic");
    }

    #[test]
    fn test_blake3_derive_session_key_different_inputs() {
        let context_a = vec![0x01u8; 32];
        let context_b = vec![0x02u8; 32];
        let test_secret = vec![0xABu8; 32];

        let key1 = PqcHandshakeManager::blake3_derive_session_key(&context_a, &test_secret);
        let key2 = PqcHandshakeManager::blake3_derive_session_key(&context_b, &test_secret);
        assert_ne!(key1, key2, "Different inputs must produce different keys");
    }
}
