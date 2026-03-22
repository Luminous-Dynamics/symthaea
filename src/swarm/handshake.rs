// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hybrid Handshake Protocol - Trust Bridge between Holochain and Iroh
//!
//! This module implements the "Trust Challenge" protocol that ensures
//! only trusted peers (verified via Ed25519 signatures) can establish
//! real-time tensor streams (via Iroh).
//!
//! ## Protocol Flow
//!
//! ```text
//!     Node A                                    Node B
//!        │                                         │
//!        │──────── 1. Iroh Connect ───────────────>│
//!        │                                         │
//!        │<─────── 2. Trust Challenge ─────────────│
//!        │         (nonce)                         │
//!        │                                         │
//!        │──────── 3. Trust Response ─────────────>│
//!        │         (Ed25519 signature + pubkey)    │
//!        │                                         │
//!        │         4. Verify Ed25519 signature     │
//!        │         5. Check Holochain DHT Trust    │
//!        │                                         │
//!        │<─────── 6. Trust Ack / Reject ──────────│
//!        │                                         │
//!        │<═══════ 7. Tensor Streaming ═══════════>│
//!        │         (if trusted)                    │
//! ```
//!
//! ## Security Model
//!
//! - The nonce prevents replay attacks
//! - The Ed25519 signature proves possession of the private key (asymmetric)
//! - The verifier only needs the public key — no shared secrets
//! - The DHT lookup verifies the agent's reputation
//! - Only after trust verification can tensor streaming begin
//!
//! ## Dual-Mode Operation
//!
//! - **With `identity` feature**: Real Ed25519 signatures via `ed25519-dalek`
//! - **Without `identity` feature**: BLAKE3 keyed MAC fallback (symmetric, legacy)

use crate::cognitive_loop::thresholds;
use crate::swarm::{SwarmConfig, SwarmError, SwarmMessage, SwarmResult, TrustLevel};
use rand::Rng;
use std::fmt;
use std::time::{Duration, SystemTime};
use tracing::warn;

// ============================================================================
// HANDSHAKE ERRORS
// ============================================================================

/// Errors specific to the handshake protocol
#[derive(Debug, Clone)]
pub enum HandshakeError {
    /// Expected a TrustChallenge message but received a different variant
    UnexpectedMessageType {
        expected: &'static str,
        actual: String,
    },
    /// Challenge nonce extraction failed
    ChallengeExtractionFailed { reason: String },
    /// Response extraction failed
    ResponseExtractionFailed { reason: String },
    /// Invalid handshake state
    InvalidState { expected: String, actual: String },
    /// Protocol violation
    ProtocolViolation { message: String },
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedMessageType { expected, actual } => {
                write!(f, "Expected {expected} message, got {actual}")
            }
            Self::ChallengeExtractionFailed { reason } => {
                write!(f, "Failed to extract challenge: {reason}")
            }
            Self::ResponseExtractionFailed { reason } => {
                write!(f, "Failed to extract response: {reason}")
            }
            Self::InvalidState { expected, actual } => {
                write!(
                    f,
                    "Invalid handshake state: expected {expected}, got {actual}"
                )
            }
            Self::ProtocolViolation { message } => {
                write!(f, "Handshake protocol violation: {message}")
            }
        }
    }
}

impl std::error::Error for HandshakeError {}

// ============================================================================
// SWARM MESSAGE HELPERS
// ============================================================================

/// Extension trait for extracting specific message types from SwarmMessage
pub trait SwarmMessageExt {
    /// Try to extract the nonce from a TrustChallenge message
    fn try_into_challenge_nonce(self) -> Result<Vec<u8>, HandshakeError>;

    /// Try to extract (signed_nonce, agent_key) from a TrustResponse message
    fn try_into_response(self) -> Result<(Vec<u8>, String), HandshakeError>;
}

impl SwarmMessageExt for SwarmMessage {
    fn try_into_challenge_nonce(self) -> Result<Vec<u8>, HandshakeError> {
        match self {
            SwarmMessage::TrustChallenge { nonce } => Ok(nonce),
            other => {
                let actual = other.message_type().to_string();
                warn!(
                    expected = "TrustChallenge",
                    actual = %actual,
                    "Unexpected message type during handshake"
                );
                Err(HandshakeError::UnexpectedMessageType {
                    expected: "TrustChallenge",
                    actual,
                })
            }
        }
    }

    fn try_into_response(self) -> Result<(Vec<u8>, String), HandshakeError> {
        match self {
            SwarmMessage::TrustResponse {
                signed_nonce,
                agent_key,
            } => Ok((signed_nonce, agent_key)),
            other => {
                let actual = other.message_type().to_string();
                warn!(
                    expected = "TrustResponse",
                    actual = %actual,
                    "Unexpected message type during handshake"
                );
                Err(HandshakeError::UnexpectedMessageType {
                    expected: "TrustResponse",
                    actual,
                })
            }
        }
    }
}

/// Ed25519 signature length in bytes
#[cfg(feature = "identity")]
const SIGNATURE_LEN: usize = 64;

/// BLAKE3 MAC length (legacy fallback without identity feature)
#[cfg(not(feature = "identity"))]
const MAC_LEN: usize = 32;

/// The hybrid handshake manager
pub struct HybridHandshake {
    /// Configuration
    config: SwarmConfig,

    /// Pending challenges we've issued
    pending_challenges: std::collections::HashMap<String, PendingChallenge>,

    /// Challenge timeout
    challenge_timeout: Duration,

    /// Optional genesis-seeded RNG for deterministic nonce generation
    seeded_rng: Option<symthaea_core::genesis::ShakeRng>,

    /// Peer trust levels established by completed handshakes.
    /// Maps peer_node_id → (trust_level, agent_public_key_hex).
    peer_trust: std::collections::HashMap<String, (TrustLevel, String)>,
}

/// A pending trust challenge
struct PendingChallenge {
    /// The nonce we sent
    nonce: Vec<u8>,

    /// When the challenge was issued
    issued_at: SystemTime,

    /// The peer we challenged
    #[allow(dead_code)] // RESERVED(mesh): handshake protocol state
    peer_node_id: String,
}

impl HybridHandshake {
    /// Create a new handshake manager
    pub fn new(config: SwarmConfig) -> Self {
        let timeout = Duration::from_secs(config.challenge_timeout_secs);
        Self {
            config,
            pending_challenges: std::collections::HashMap::new(),
            challenge_timeout: timeout,
            seeded_rng: None,
            peer_trust: std::collections::HashMap::new(),
        }
    }

    /// Create a handshake manager with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: SwarmConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let timeout = Duration::from_secs(config.challenge_timeout_secs);
        Self {
            config,
            pending_challenges: std::collections::HashMap::new(),
            challenge_timeout: timeout,
            seeded_rng: Some(genesis.domain(&format!("{label}::handshake"))),
            peer_trust: std::collections::HashMap::new(),
        }
    }

    /// Generate a trust challenge for a new peer.
    ///
    /// Returns `Err` if rate limiting is triggered:
    /// - Too many total pending challenges (cap: `HANDSHAKE_MAX_PENDING_CHALLENGES`)
    /// - Challenge already pending for this specific peer
    pub fn create_challenge(&mut self, peer_node_id: &str) -> Result<SwarmMessage, HandshakeError> {
        // Rate limit: cap total pending challenges
        if self.pending_challenges.len() >= thresholds::HANDSHAKE_MAX_PENDING_CHALLENGES {
            return Err(HandshakeError::ProtocolViolation {
                message: format!(
                    "Too many pending challenges ({} >= {})",
                    self.pending_challenges.len(),
                    thresholds::HANDSHAKE_MAX_PENDING_CHALLENGES,
                ),
            });
        }

        // Rate limit: one challenge per peer (duplicates indicate replay)
        if self.pending_challenges.contains_key(peer_node_id) {
            return Err(HandshakeError::ProtocolViolation {
                message: format!("Challenge already pending for peer '{}'", peer_node_id,),
            });
        }

        // Generate cryptographic nonce (seeded or random)
        let nonce: Vec<u8> = if let Some(ref mut rng) = self.seeded_rng {
            use rand::RngCore;
            let mut buf = vec![0u8; 32];
            rng.fill_bytes(&mut buf);
            buf
        } else {
            rand::thread_rng()
                .sample_iter(rand::distributions::Standard)
                .take(32)
                .collect()
        };

        // Store pending challenge
        self.pending_challenges.insert(
            peer_node_id.to_string(),
            PendingChallenge {
                nonce: nonce.clone(),
                issued_at: SystemTime::now(),
                peer_node_id: peer_node_id.to_string(),
            },
        );

        Ok(SwarmMessage::TrustChallenge { nonce })
    }

    // ========================================================================
    // Ed25519 path (with `identity` feature)
    // ========================================================================

    /// Create a response to a trust challenge using Ed25519 signature.
    ///
    /// Signs the nonce with the agent's Ed25519 private key. The verifier
    /// only needs the public key to verify — no shared secrets required.
    #[cfg(feature = "identity")]
    pub fn create_response(
        &self,
        nonce: &[u8],
        agent_key: &str,
        signing_key: &ed25519_dalek::SigningKey,
    ) -> SwarmMessage {
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(nonce);
        SwarmMessage::TrustResponse {
            signed_nonce: signature.to_bytes().to_vec(),
            agent_key: agent_key.to_string(),
        }
    }

    /// Verify a trust response using Ed25519 signature verification.
    ///
    /// The `agent_key` is hex-encoded Ed25519 public key (64 hex chars = 32 bytes).
    #[cfg(feature = "identity")]
    pub fn verify_response(
        &mut self,
        peer_node_id: &str,
        signed_nonce: &[u8],
        agent_key: &str,
    ) -> SwarmResult<TrustLevel> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        // Get pending challenge
        let challenge = self
            .pending_challenges
            .remove(peer_node_id)
            .ok_or_else(|| SwarmError::TrustVerificationError {
                reason: "No pending challenge for this peer".to_string(),
            })?;

        // Check timeout
        if challenge.issued_at.elapsed().unwrap_or(Duration::MAX) > self.challenge_timeout {
            return Err(SwarmError::TrustVerificationError {
                reason: "Challenge timed out".to_string(),
            });
        }

        // Verify signature length
        if signed_nonce.len() != SIGNATURE_LEN {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid signature length: expected {} bytes, got {}",
                    SIGNATURE_LEN,
                    signed_nonce.len()
                ),
            });
        }

        // Decode public key from hex
        let pubkey_bytes =
            hex::decode(agent_key).map_err(|e| SwarmError::TrustVerificationError {
                reason: format!("Invalid hex public key: {}", e),
            })?;

        if pubkey_bytes.len() != 32 {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid public key length: expected 32 bytes, got {}",
                    pubkey_bytes.len()
                ),
            });
        }

        let pubkey_array: [u8; 32] =
            pubkey_bytes
                .try_into()
                .map_err(|_| SwarmError::TrustVerificationError {
                    reason: "Failed to convert public key bytes".to_string(),
                })?;

        let verifying_key = VerifyingKey::from_bytes(&pubkey_array).map_err(|e| {
            SwarmError::TrustVerificationError {
                reason: format!("Invalid Ed25519 public key: {}", e),
            }
        })?;

        // Parse signature
        let sig_bytes: [u8; 64] =
            signed_nonce
                .try_into()
                .map_err(|_| SwarmError::TrustVerificationError {
                    reason: "Failed to convert signature bytes".to_string(),
                })?;
        let signature = Signature::from_bytes(&sig_bytes);

        // Verify Ed25519 signature over the nonce
        verifying_key
            .verify(&challenge.nonce, &signature)
            .map_err(|_| SwarmError::TrustVerificationError {
                reason: "Ed25519 signature verification failed".to_string(),
            })?;

        // Signature valid — peer proved possession of private key
        let trust = TrustLevel::Verified(self.config.initial_trust_score);
        self.peer_trust.insert(
            peer_node_id.to_string(),
            (trust.clone(), agent_key.to_string()),
        );

        Ok(trust)
    }

    // ========================================================================
    // BLAKE3 MAC fallback path (without `identity` feature)
    // ========================================================================

    /// Create a response using BLAKE3 keyed MAC (legacy fallback).
    ///
    /// Requires the verifier to know the agent's key material (symmetric).
    #[cfg(not(feature = "identity"))]
    pub fn create_response(
        &self,
        nonce: &[u8],
        agent_key: &str,
        agent_private_key: &[u8],
    ) -> SwarmMessage {
        let mac = Self::compute_mac(nonce, agent_private_key);
        SwarmMessage::TrustResponse {
            signed_nonce: mac,
            agent_key: agent_key.to_string(),
        }
    }

    /// Verify a trust response using BLAKE3 MAC (legacy fallback).
    #[cfg(not(feature = "identity"))]
    pub fn verify_response(
        &mut self,
        peer_node_id: &str,
        signed_nonce: &[u8],
        agent_key: &str,
    ) -> SwarmResult<TrustLevel> {
        // Get pending challenge
        let challenge = self
            .pending_challenges
            .remove(peer_node_id)
            .ok_or_else(|| SwarmError::TrustVerificationError {
                reason: "No pending challenge for this peer".to_string(),
            })?;

        // Check timeout
        if challenge.issued_at.elapsed().unwrap_or(Duration::MAX) > self.challenge_timeout {
            return Err(SwarmError::TrustVerificationError {
                reason: "Challenge timed out".to_string(),
            });
        }

        // Verify MAC length
        if signed_nonce.len() != MAC_LEN {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid MAC length: expected {} bytes, got {}",
                    MAC_LEN,
                    signed_nonce.len()
                ),
            });
        }

        // Recompute MAC with the agent's key material
        let expected_mac = Self::compute_mac(&challenge.nonce, agent_key.as_bytes());

        // Constant-time comparison to prevent timing attacks
        if !constant_time_eq(signed_nonce, &expected_mac) {
            return Err(SwarmError::TrustVerificationError {
                reason: "MAC verification failed".to_string(),
            });
        }

        let trust = TrustLevel::Verified(self.config.initial_trust_score);
        self.peer_trust.insert(
            peer_node_id.to_string(),
            (trust.clone(), agent_key.to_string()),
        );

        Ok(trust)
    }

    /// Compute a BLAKE3 keyed MAC over the given data (legacy, used without identity feature).
    #[cfg(not(feature = "identity"))]
    fn compute_mac(data: &[u8], key_material: &[u8]) -> Vec<u8> {
        let derived_key = blake3::hash(key_material);
        let key_bytes: [u8; 32] = *derived_key.as_bytes();
        let mac = blake3::keyed_hash(&key_bytes, data);
        mac.as_bytes().to_vec()
    }

    // ========================================================================
    // Common methods
    // ========================================================================

    /// Check if a trust level meets minimum requirements
    pub fn meets_trust_requirement(&self, trust: &TrustLevel) -> bool {
        trust.value() >= self.config.min_trust_level
    }

    /// Check if a peer has been verified by this handshake manager.
    pub fn is_peer_trusted(&self, peer_node_id: &str) -> bool {
        self.peer_trust
            .get(peer_node_id)
            .map(|(t, _)| self.meets_trust_requirement(t))
            .unwrap_or(false)
    }

    /// Get the trust level of a verified peer.
    pub fn peer_trust_level(&self, peer_node_id: &str) -> TrustLevel {
        self.peer_trust
            .get(peer_node_id)
            .map(|(t, _)| t.clone())
            .unwrap_or(TrustLevel::Unknown)
    }

    /// Get the number of verified peers.
    pub fn verified_peer_count(&self) -> usize {
        self.peer_trust.len()
    }

    /// Remove a peer from the trust map (e.g., on disconnect).
    pub fn remove_peer(&mut self, peer_node_id: &str) {
        self.peer_trust.remove(peer_node_id);
    }

    /// Clean up expired challenges
    pub fn cleanup_expired(&mut self) {
        self.pending_challenges.retain(|_, challenge| {
            challenge
                .issued_at
                .elapsed()
                .map(|d| d < self.challenge_timeout)
                .unwrap_or(false)
        });
    }

    /// Get the number of pending challenges
    pub fn pending_count(&self) -> usize {
        self.pending_challenges.len()
    }

    // ========================================================================
    // Mutual handshake (both sides prove identity)
    // ========================================================================

    /// Create a mutual challenge: the initiator generates a nonce for the
    /// responder to sign AND prepares to be challenged back.
    ///
    /// Returns the outbound challenge message. The responder should call
    /// `create_counter_challenge()` to challenge the initiator back.
    pub fn create_mutual_challenge(
        &mut self,
        peer_node_id: &str,
    ) -> Result<SwarmMessage, HandshakeError> {
        self.create_challenge(peer_node_id)
    }

    /// Create a counter-challenge: the responder generates a nonce for the
    /// initiator to sign back, proving mutual identity.
    ///
    /// This should be called by the responder after receiving a challenge
    /// from the initiator and sending a response.
    pub fn create_counter_challenge(
        &mut self,
        peer_node_id: &str,
    ) -> Result<SwarmMessage, HandshakeError> {
        // Remove any existing challenge for this peer (we're re-challenging)
        self.pending_challenges.remove(peer_node_id);
        self.create_challenge(peer_node_id)
    }

    /// Verify a mutual handshake response.
    ///
    /// This is the same as `verify_response()` but documents the intent
    /// that both sides have now been verified.
    #[cfg(feature = "identity")]
    pub fn verify_mutual_response(
        &mut self,
        peer_node_id: &str,
        signed_nonce: &[u8],
        agent_key: &str,
    ) -> SwarmResult<TrustLevel> {
        self.verify_response(peer_node_id, signed_nonce, agent_key)
    }

    /// Verify a mutual handshake response (BLAKE3 fallback).
    #[cfg(not(feature = "identity"))]
    pub fn verify_mutual_response(
        &mut self,
        peer_node_id: &str,
        signed_nonce: &[u8],
        agent_key: &str,
    ) -> SwarmResult<TrustLevel> {
        self.verify_response(peer_node_id, signed_nonce, agent_key)
    }
}

/// Constant-time byte comparison to prevent timing attacks.
/// Public within crate for reuse in mesh encryption and radio dispatcher.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Result of a complete handshake
#[derive(Debug, Clone)]
pub struct HandshakeResult {
    /// The peer's node ID
    pub peer_node_id: String,

    /// The peer's agent key (if verified)
    pub agent_key: Option<String>,

    /// Trust level after verification
    pub trust_level: TrustLevel,

    /// Whether streaming is allowed
    pub streaming_allowed: bool,

    /// Time taken for handshake (milliseconds)
    pub handshake_time_ms: u64,
}

impl HandshakeResult {
    /// Create a successful handshake result
    pub fn success(
        peer_node_id: impl Into<String>,
        agent_key: impl Into<String>,
        trust_level: TrustLevel,
        handshake_time_ms: u64,
    ) -> Self {
        Self {
            peer_node_id: peer_node_id.into(),
            agent_key: Some(agent_key.into()),
            trust_level,
            streaming_allowed: true,
            handshake_time_ms,
        }
    }

    /// Create a failed handshake result
    pub fn failed(peer_node_id: impl Into<String>) -> Self {
        Self {
            peer_node_id: peer_node_id.into(),
            agent_key: None,
            trust_level: TrustLevel::Untrusted,
            streaming_allowed: false,
            handshake_time_ms: 0,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// TESTS — Ed25519 path (with identity feature)
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "identity")]
#[allow(clippy::field_reassign_with_default)]
mod tests_ed25519 {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn generate_keypair() -> (SigningKey, String) {
        let mut seed = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut seed);
        let signing_key = SigningKey::from_bytes(&seed);
        let pubkey_hex = hex::encode(signing_key.verifying_key().as_bytes());
        (signing_key, pubkey_hex)
    }

    #[test]
    fn test_create_challenge() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge
            .try_into_challenge_nonce()
            .expect("create_challenge should return TrustChallenge");
        assert_eq!(nonce.len(), 32);
        assert_eq!(handshake.pending_count(), 1);
    }

    #[test]
    fn test_ed25519_sign_verify_roundtrip() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);
        let (signing_key, pubkey_hex) = generate_keypair();

        // Create challenge
        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge
            .try_into_challenge_nonce()
            .expect("should be TrustChallenge");

        // Create response with Ed25519 signature
        let response = handshake.create_response(&nonce, &pubkey_hex, &signing_key);
        let (signed_nonce, agent_key) = response
            .try_into_response()
            .expect("should be TrustResponse");

        // Signature should be exactly 64 bytes
        assert_eq!(signed_nonce.len(), 64);
        assert_eq!(agent_key, pubkey_hex);

        // Verify
        let trust = handshake
            .verify_response("peer-123", &signed_nonce, &agent_key)
            .unwrap();
        assert!(matches!(trust, TrustLevel::Verified(_)));
    }

    #[test]
    fn test_ed25519_wrong_key_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);
        let (signing_key, _) = generate_keypair();
        let (_, wrong_pubkey_hex) = generate_keypair();

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();

        // Sign with one key, claim another public key
        let response = handshake.create_response(&nonce, &wrong_pubkey_hex, &signing_key);
        let (signed_nonce, agent_key) = response.try_into_response().unwrap();

        let result = handshake.verify_response("peer-123", &signed_nonce, &agent_key);
        assert!(result.is_err(), "Wrong public key should fail verification");
    }

    #[test]
    fn test_ed25519_tampered_signature_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);
        let (signing_key, pubkey_hex) = generate_keypair();

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();

        let response = handshake.create_response(&nonce, &pubkey_hex, &signing_key);
        let (mut signed_nonce, agent_key) = response.try_into_response().unwrap();

        // Tamper with signature
        signed_nonce[0] ^= 0xFF;

        let result = handshake.verify_response("peer-123", &signed_nonce, &agent_key);
        assert!(result.is_err(), "Tampered signature should fail");
    }

    #[test]
    fn test_no_pending_challenge_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let result = handshake.verify_response("peer-123", &[0u8; 64], "deadbeef");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_pubkey_hex_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let _ = handshake.create_challenge("peer-123").unwrap();
        let result = handshake.verify_response("peer-123", &[0u8; 64], "not-hex!");
        assert!(result.is_err());
    }

    #[test]
    fn test_short_pubkey_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let _ = handshake.create_challenge("peer-123").unwrap();
        // Valid hex but too short (4 bytes instead of 32)
        let result = handshake.verify_response("peer-123", &[0u8; 64], "aabbccdd");
        assert!(result.is_err());
    }

    #[test]
    fn test_trust_requirement() {
        let mut config = SwarmConfig::default();
        config.min_trust_level = 0.5;
        let handshake = HybridHandshake::new(config);

        assert!(handshake.meets_trust_requirement(&TrustLevel::Verified(0.7)));
        assert!(!handshake.meets_trust_requirement(&TrustLevel::Verified(0.3)));
        assert!(!handshake.meets_trust_requirement(&TrustLevel::Unknown));
    }

    #[test]
    fn test_peer_trust_tracking() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);
        let (signing_key, pubkey_hex) = generate_keypair();

        // Complete a handshake
        let challenge = handshake.create_challenge("peer-A").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();
        let response = handshake.create_response(&nonce, &pubkey_hex, &signing_key);
        let (signed_nonce, agent_key) = response.try_into_response().unwrap();
        handshake
            .verify_response("peer-A", &signed_nonce, &agent_key)
            .unwrap();

        // Peer should now be tracked
        assert!(handshake.is_peer_trusted("peer-A"));
        assert!(!handshake.is_peer_trusted("peer-B"));
        assert_eq!(handshake.verified_peer_count(), 1);

        // Remove peer
        handshake.remove_peer("peer-A");
        assert!(!handshake.is_peer_trusted("peer-A"));
        assert_eq!(handshake.verified_peer_count(), 0);
    }

    #[test]
    fn test_handshake_result() {
        let success = HandshakeResult::success("peer-1", "agent-1", TrustLevel::Verified(0.8), 150);
        assert!(success.streaming_allowed);

        let failed = HandshakeResult::failed("peer-2");
        assert!(!failed.streaming_allowed);
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    #[test]
    fn test_rate_limited_duplicate_challenge() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        // First challenge succeeds
        assert!(handshake.create_challenge("peer-123").is_ok());

        // Duplicate challenge for same peer is rejected
        let result = handshake.create_challenge("peer-123");
        assert!(result.is_err());
        match result {
            Err(HandshakeError::ProtocolViolation { message }) => {
                assert!(message.contains("already pending"));
            }
            _ => panic!("Expected ProtocolViolation"),
        }
    }

    #[test]
    fn test_configurable_trust_score() {
        let mut config = SwarmConfig::default();
        config.initial_trust_score = 0.9;
        let mut handshake = HybridHandshake::new(config);
        let (signing_key, pubkey_hex) = generate_keypair();

        let challenge = handshake.create_challenge("peer-A").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();
        let response = handshake.create_response(&nonce, &pubkey_hex, &signing_key);
        let (signed_nonce, agent_key) = response.try_into_response().unwrap();
        let trust = handshake
            .verify_response("peer-A", &signed_nonce, &agent_key)
            .unwrap();

        match trust {
            TrustLevel::Verified(v) => assert!((v - 0.9).abs() < f64::EPSILON),
            _ => panic!("Expected Verified trust"),
        }
    }

    #[test]
    fn test_configurable_timeout() {
        let mut config = SwarmConfig::default();
        config.challenge_timeout_secs = 5;
        let handshake = HybridHandshake::new(config);
        assert_eq!(
            handshake.challenge_timeout,
            std::time::Duration::from_secs(5)
        );
    }

    #[test]
    fn test_mutual_handshake_roundtrip() {
        let config_a = SwarmConfig::default();
        let config_b = SwarmConfig::default();
        let mut hs_a = HybridHandshake::new(config_a);
        let mut hs_b = HybridHandshake::new(config_b);
        let (key_a, pub_a) = generate_keypair();
        let (key_b, pub_b) = generate_keypair();

        // Step 1: A challenges B
        let challenge_ab = hs_a.create_mutual_challenge("peer-B").unwrap();
        let nonce_ab = challenge_ab.try_into_challenge_nonce().unwrap();

        // Step 2: B responds to A's challenge and counter-challenges A
        let response_ba = hs_b.create_response(&nonce_ab, &pub_b, &key_b);
        let counter_challenge = hs_b.create_counter_challenge("peer-A").unwrap();
        let nonce_ba = counter_challenge.try_into_challenge_nonce().unwrap();

        // Step 3: A verifies B's response
        let (sig_b, key_b_hex) = response_ba.try_into_response().unwrap();
        let trust_b = hs_a.verify_response("peer-B", &sig_b, &key_b_hex).unwrap();
        assert!(matches!(trust_b, TrustLevel::Verified(_)));

        // Step 4: A responds to B's counter-challenge
        let response_ab = hs_a.create_response(&nonce_ba, &pub_a, &key_a);
        let (sig_a, key_a_hex) = response_ab.try_into_response().unwrap();

        // Step 5: B verifies A's response
        let trust_a = hs_b
            .verify_mutual_response("peer-A", &sig_a, &key_a_hex)
            .unwrap();
        assert!(matches!(trust_a, TrustLevel::Verified(_)));

        // Both sides are now mutually verified
        assert!(hs_a.is_peer_trusted("peer-B"));
        assert!(hs_b.is_peer_trusted("peer-A"));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// TESTS — BLAKE3 MAC fallback path (without identity feature)
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(not(feature = "identity"))]
#[allow(clippy::field_reassign_with_default)]
mod tests_blake3 {
    use super::*;

    #[test]
    fn test_create_challenge() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge
            .try_into_challenge_nonce()
            .expect("create_challenge should return TrustChallenge");
        assert_eq!(nonce.len(), 32);
        assert_eq!(handshake.pending_count(), 1);
    }

    #[test]
    fn test_create_response_produces_mac() {
        let config = SwarmConfig::default();
        let handshake = HybridHandshake::new(config);

        let nonce = vec![1, 2, 3, 4];
        let response = handshake.create_response(&nonce, "agent-key", b"private-key");
        let (signed_nonce, agent_key) = response
            .try_into_response()
            .expect("create_response should return TrustResponse");
        assert_eq!(signed_nonce.len(), MAC_LEN);
        assert_eq!(agent_key, "agent-key");
    }

    #[test]
    fn test_verify_valid_response() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();

        let response = handshake.create_response(&nonce, "agent-key", b"agent-key");
        let (signed_nonce, agent_key) = response.try_into_response().unwrap();

        let trust = handshake
            .verify_response("peer-123", &signed_nonce, &agent_key)
            .unwrap();
        assert!(matches!(trust, TrustLevel::Verified(_)));
    }

    #[test]
    fn test_verify_wrong_key_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();

        let response = handshake.create_response(&nonce, "agent-key", b"wrong-key");
        let (signed_nonce, agent_key) = response.try_into_response().unwrap();

        let result = handshake.verify_response("peer-123", &signed_nonce, &agent_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_tampered_mac_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123").unwrap();
        let nonce = challenge.try_into_challenge_nonce().unwrap();

        let response = handshake.create_response(&nonce, "agent-key", b"agent-key");
        let (mut signed_nonce, agent_key) = response.try_into_response().unwrap();
        signed_nonce[0] ^= 0xFF;

        let result = handshake.verify_response("peer-123", &signed_nonce, &agent_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_no_pending_challenge_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let result = handshake.verify_response("peer-123", &[0u8; 32], "agent-key");
        assert!(result.is_err());
    }

    #[test]
    fn test_mac_is_deterministic() {
        let mac1 = HybridHandshake::compute_mac(b"test-nonce", b"test-key");
        let mac2 = HybridHandshake::compute_mac(b"test-nonce", b"test-key");
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_mac_differs_for_different_inputs() {
        let mac1 = HybridHandshake::compute_mac(b"nonce-1", b"key");
        let mac2 = HybridHandshake::compute_mac(b"nonce-2", b"key");
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    #[test]
    fn test_trust_requirement() {
        let mut config = SwarmConfig::default();
        config.min_trust_level = 0.5;
        let handshake = HybridHandshake::new(config);

        assert!(handshake.meets_trust_requirement(&TrustLevel::Verified(0.7)));
        assert!(!handshake.meets_trust_requirement(&TrustLevel::Verified(0.3)));
        assert!(!handshake.meets_trust_requirement(&TrustLevel::Unknown));
    }

    #[test]
    fn test_handshake_result() {
        let success = HandshakeResult::success("peer-1", "agent-1", TrustLevel::Verified(0.8), 150);
        assert!(success.streaming_allowed);

        let failed = HandshakeResult::failed("peer-2");
        assert!(!failed.streaming_allowed);
    }
}
