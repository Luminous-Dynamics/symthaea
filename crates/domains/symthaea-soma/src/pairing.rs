// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BLE Pairing Manager — Ed25519 trust establishment for Spore-to-Spore pairing.
//!
//! Follows the BleMesh pattern: mode-gated, tick-driven, queue-based, WASM-safe.
//! Uses cycle-based timeouts (not SystemTime) for deterministic behavior.
//!
//! ## Dual-Mode Crypto
//!
//! - **With `pairing` feature**: Ed25519 signatures via `ed25519-dalek`.
//! - **Without `pairing` feature**: X25519 Diffie-Hellman derives a shared
//!   secret for a BLAKE3 MAC.
//! - Both modes authenticate a v2, domain-separated transcript binding the
//!   fresh nonce and both role-labelled public keys.
//!
//! This application pairing protocol is **not** Bluetooth LE Secure Connections
//! and does not by itself prevent an active man-in-the-middle: public keys are
//! exchanged in-band without an authenticated out-of-band or user-confirmation
//! step. Do not represent successful transcript verification as MITM-proof
//! device identity.

use serde::{Deserialize, Serialize};
use symthaea_spore::config::PairingMode;

/// Maximum number of paired devices (matches BleMesh peer cap).
const MAX_PAIRED_DEVICES: usize = 16;

/// Pairing challenge timeout in cycles (~30s at 50Hz).
const CHALLENGE_TIMEOUT_CYCLES: u64 = 1500;

/// Maximum outbound queue capacity (matches HolonBridge).
const OUTBOUND_CAP: usize = 100;

/// Maximum pending challenges.
const MAX_PENDING: usize = 16;

// ============================================================================
// MESSAGES
// ============================================================================

/// Outbound pairing messages to send over BLE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PairingOutbound {
    /// Challenge nonce sent to a peer requesting pairing.
    ///
    /// `initiator_pubkey` is the challenger's own public key (Ed25519
    /// verifying key or X25519 DH public key, depending on build).
    /// In the non-`pairing` (X25519) build, the responder needs this to
    /// compute the shared secret used to MAC the nonce.
    Challenge {
        peer_id: u64,
        nonce: Vec<u8>,
        initiator_pubkey: Vec<u8>,
    },
    /// Signed response to a received challenge.
    Response {
        peer_id: u64,
        signature: Vec<u8>,
        pubkey: Vec<u8>,
    },
    /// Pairing accepted.
    Ack { peer_id: u64 },
    /// Pairing rejected.
    Reject { peer_id: u64, reason: String },
}

/// Inbound pairing messages received from BLE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PairingInbound {
    /// Peer requests pairing — contains their nonce for us to sign, and
    /// their public key (see [`PairingOutbound::Challenge`]).
    Challenge {
        peer_id: u64,
        nonce: Vec<u8>,
        initiator_pubkey: Vec<u8>,
    },
    /// Peer's signed response to our challenge.
    Response {
        peer_id: u64,
        signature: Vec<u8>,
        pubkey: Vec<u8>,
    },
    /// Peer acknowledged our pairing.
    Ack { peer_id: u64 },
    /// Peer rejected pairing.
    Reject { peer_id: u64, reason: String },
}

// ============================================================================
// PAIRED DEVICE
// ============================================================================

/// A verified paired device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedDevice {
    /// Peer identifier (BLE device ID).
    pub peer_id: u64,
    /// Peer's public key (32 bytes: Ed25519 verifying key or X25519 public key).
    pub pubkey: [u8; 32],
    /// Trust level (0.0–1.0), initially 0.7 on first pairing.
    pub trust_level: f32,
    /// Last cycle the device was seen.
    pub last_seen: u64,
}

// ============================================================================
// PENDING CHALLENGE
// ============================================================================

struct PendingChallenge {
    nonce: Vec<u8>,
    /// Exact initiator key sent with the challenge; retained to verify the
    /// response against the same context even if local key state later changes.
    initiator_pubkey: Vec<u8>,
    issued_cycle: u64,
}

// ============================================================================
// PAIRING MANAGER
// ============================================================================

/// Manages BLE device pairing with Ed25519 (or BLAKE3 fallback) trust verification.
pub struct PairingManager {
    mode: PairingMode,
    /// Our keypair material (32 bytes seed for Ed25519, or BLAKE3 key).
    keypair_seed: Option<[u8; 32]>,
    /// Known paired devices.
    paired_devices: Vec<PairedDevice>,
    /// Pending outbound challenges indexed by peer_id.
    pending_challenges: Vec<(u64, PendingChallenge)>,
    /// Outbound message queue.
    outbound: std::collections::VecDeque<PairingOutbound>,
    /// Current cycle counter (set by tick).
    cycle: u64,
}

impl PairingManager {
    /// Create a new pairing manager.
    pub fn new(mode: PairingMode) -> Self {
        Self {
            mode,
            keypair_seed: None,
            paired_devices: Vec::new(),
            pending_challenges: Vec::new(),
            outbound: std::collections::VecDeque::new(),
            cycle: 0,
        }
    }

    /// Generate a new keypair and store the seed.
    ///
    /// The seed is drawn from the OS entropy source (`getrandom`), not from
    /// any predictable state like the cycle counter. A predictable seed
    /// would let an observer who can estimate device uptime reconstruct the
    /// signing/DH key and impersonate the device.
    pub fn generate_keypair(&mut self) {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).expect("OS entropy source unavailable");
        self.keypair_seed = Some(seed);
    }

    /// Get the public key as bytes (32 bytes).
    ///
    /// Returns `None` if no keypair has been generated.
    ///
    /// In the `pairing` (Ed25519) build this is a genuine verifying key. In
    /// the fallback build this is a genuine X25519 Diffie-Hellman public
    /// key -- **not** a hash of the secret -- so it is safe to transmit in
    /// the clear; it never functions as the MAC key itself (see
    /// [`Self::derive_shared_mac_key`]).
    pub fn get_pubkey(&self) -> Option<[u8; 32]> {
        let seed = self.keypair_seed.as_ref()?;

        #[cfg(feature = "pairing")]
        {
            use ed25519_dalek::SigningKey;
            let sk = SigningKey::from_bytes(seed);
            Some(sk.verifying_key().to_bytes())
        }

        #[cfg(not(feature = "pairing"))]
        {
            use x25519_dalek::{PublicKey, StaticSecret};
            let secret = StaticSecret::from(*seed);
            Some(PublicKey::from(&secret).to_bytes())
        }
    }

    /// Derive the shared MAC key for the fallback (non-`pairing`) build via
    /// X25519 Diffie-Hellman: `DH(my_secret, their_pubkey)`.
    ///
    /// By ECDH commutativity, both peers independently compute the same
    /// 32-byte shared secret from their own private key and the other
    /// party's public key -- the secret itself is never transmitted, so an
    /// eavesdropper who observes both public keys and the resulting MAC
    /// cannot reconstruct it (discrete-log hardness).
    #[cfg(not(feature = "pairing"))]
    fn derive_shared_mac_key(&self, their_pubkey: &[u8]) -> Option<[u8; 32]> {
        use x25519_dalek::{PublicKey, StaticSecret};
        let seed = self.keypair_seed?;
        let their_pubkey: [u8; 32] = their_pubkey.try_into().ok()?;
        let secret = StaticSecret::from(seed);
        let shared = secret.diffie_hellman(&PublicKey::from(their_pubkey));
        let shared_bytes = *shared.as_bytes();
        // X25519's all-zero output signals a non-contributory peer key. Never
        // turn that public, degenerate value into a MAC key.
        if shared_bytes.iter().all(|byte| *byte == 0) {
            return None;
        }
        Some(shared_bytes)
    }

    /// Initiate pairing with a remote peer.
    ///
    /// Creates a challenge nonce and queues it for sending.
    /// Returns `false` if mode is Off, no keypair, device cap reached, or
    /// already pending.
    pub fn initiate_pairing(&mut self, peer_id: u64) -> bool {
        if self.mode == PairingMode::Off {
            return false;
        }
        if self.paired_devices.len() >= MAX_PAIRED_DEVICES {
            return false;
        }
        if self.pending_challenges.len() >= MAX_PENDING {
            return false;
        }
        if self.pending_challenges.iter().any(|(id, _)| *id == peer_id) {
            return false;
        }
        let initiator_pubkey = match self.get_pubkey() {
            Some(pk) => pk.to_vec(),
            None => return false,
        };

        // A challenge is a one-use random value, never derived from cycle or
        // long-lived key material. Entropy failure aborts pairing.
        let nonce = match Self::generate_nonce() {
            Some(nonce) => nonce,
            None => return false,
        };

        self.pending_challenges.push((
            peer_id,
            PendingChallenge {
                nonce: nonce.clone(),
                initiator_pubkey: initiator_pubkey.clone(),
                issued_cycle: self.cycle,
            },
        ));

        self.enqueue(PairingOutbound::Challenge {
            peer_id,
            nonce,
            initiator_pubkey,
        });
        true
    }

    /// Handle an inbound challenge from a peer.
    ///
    /// Signs (or, in the fallback build, MACs via an X25519-derived shared
    /// key) the versioned transcript binding the nonce and both public keys,
    /// then queues the response.
    /// Returns `false` if mode is Off, no keypair, or (fallback build only)
    /// `initiator_pubkey` is not a valid X25519 public key.
    pub fn receive_challenge(
        &mut self,
        peer_id: u64,
        nonce: &[u8],
        initiator_pubkey: &[u8],
    ) -> bool {
        if self.mode == PairingMode::Off {
            return false;
        }
        let seed = match self.keypair_seed {
            Some(s) => s,
            None => return false,
        };

        let pubkey = match self.get_pubkey() {
            Some(pk) => pk.to_vec(),
            None => return false,
        };
        let transcript = match pairing_transcript(nonce, initiator_pubkey, &pubkey) {
            Some(transcript) => transcript,
            None => return false,
        };
        let signature = match self.authenticate_transcript(&transcript, &seed, initiator_pubkey) {
            Some(sig) => sig,
            None => return false,
        };

        self.enqueue(PairingOutbound::Response {
            peer_id,
            signature,
            pubkey,
        });
        true
    }

    /// Verify a peer's response to our challenge.
    ///
    /// On success, adds the peer to paired devices and queues an Ack.
    /// Returns `false` on verification failure.
    pub fn verify_response(&mut self, peer_id: u64, signature: &[u8], pubkey: &[u8]) -> bool {
        if self.mode == PairingMode::Off {
            return false;
        }

        // Find and remove pending challenge
        let idx = self
            .pending_challenges
            .iter()
            .position(|(id, _)| *id == peer_id);
        let challenge = match idx {
            Some(i) => self.pending_challenges.remove(i).1,
            None => return false,
        };

        // Check timeout
        if self.cycle.saturating_sub(challenge.issued_cycle) > CHALLENGE_TIMEOUT_CYCLES {
            self.enqueue(PairingOutbound::Reject {
                peer_id,
                reason: "challenge expired".into(),
            });
            return false;
        }

        // Verify the response against the exact key context we challenged.
        // The peer key is also in the transcript, so replacing it in transit
        // invalidates either the Ed25519 signature or the DH-derived MAC.
        let valid = match pairing_transcript(
            &challenge.nonce,
            &challenge.initiator_pubkey,
            pubkey,
        ) {
            Some(transcript) => self.verify_transcript(&transcript, signature, pubkey),
            None => false,
        };
        if !valid {
            self.enqueue(PairingOutbound::Reject {
                peer_id,
                reason: "signature verification failed".into(),
            });
            return false;
        }

        // Convert pubkey to [u8; 32]
        let mut pk = [0u8; 32];
        let len = pubkey.len().min(32);
        pk[..len].copy_from_slice(&pubkey[..len]);

        // Add to paired devices (evict oldest if at cap)
        if self.paired_devices.len() >= MAX_PAIRED_DEVICES {
            // Remove least-recently-seen
            if let Some(oldest_idx) = self
                .paired_devices
                .iter()
                .enumerate()
                .min_by_key(|(_, d)| d.last_seen)
                .map(|(i, _)| i)
            {
                self.paired_devices.remove(oldest_idx);
            }
        }

        self.paired_devices.push(PairedDevice {
            peer_id,
            pubkey: pk,
            trust_level: 0.7,
            last_seen: self.cycle,
        });

        self.enqueue(PairingOutbound::Ack { peer_id });
        true
    }

    /// Check if a peer is currently paired.
    pub fn is_paired(&self, peer_id: u64) -> bool {
        self.paired_devices.iter().any(|d| d.peer_id == peer_id)
    }

    /// Get current pairing mode.
    pub fn mode(&self) -> PairingMode {
        self.mode
    }

    /// Set pairing mode. Switching to Off clears pending challenges.
    pub fn set_mode(&mut self, mode: PairingMode) {
        if mode == PairingMode::Off {
            self.pending_challenges.clear();
        }
        self.mode = mode;
    }

    /// Advance the cycle counter and expire stale challenges.
    pub fn tick(&mut self, cycle: u64) {
        self.cycle = cycle;
        self.pending_challenges
            .retain(|(_, c)| cycle.saturating_sub(c.issued_cycle) <= CHALLENGE_TIMEOUT_CYCLES);
    }

    /// Drain all outbound messages (for BLE transmission).
    pub fn drain_outbound(&mut self) -> Vec<PairingOutbound> {
        self.outbound.drain(..).collect()
    }

    /// Process an inbound pairing message.
    pub fn receive_inbound(&mut self, msg: PairingInbound) {
        match msg {
            PairingInbound::Challenge {
                peer_id,
                nonce,
                initiator_pubkey,
            } => {
                self.receive_challenge(peer_id, &nonce, &initiator_pubkey);
            }
            PairingInbound::Response {
                peer_id,
                signature,
                pubkey,
            } => {
                self.verify_response(peer_id, &signature, &pubkey);
            }
            PairingInbound::Ack { peer_id } => {
                // Update last_seen for the peer
                if let Some(dev) = self
                    .paired_devices
                    .iter_mut()
                    .find(|d| d.peer_id == peer_id)
                {
                    dev.last_seen = self.cycle;
                }
            }
            PairingInbound::Reject { .. } => {
                // No action needed — the initiator can retry
            }
        }
    }

    /// Get the number of paired devices.
    pub fn paired_count(&self) -> usize {
        self.paired_devices.len()
    }

    /// Get a read-only view of paired devices.
    pub fn paired_devices(&self) -> &[PairedDevice] {
        &self.paired_devices
    }

    /// Get the number of pending challenges.
    pub fn pending_count(&self) -> usize {
        self.pending_challenges.len()
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn enqueue(&mut self, msg: PairingOutbound) {
        if self.outbound.len() >= OUTBOUND_CAP {
            self.outbound.pop_front();
        }
        self.outbound.push_back(msg);
    }

    fn generate_nonce() -> Option<Vec<u8>> {
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).ok()?;
        Some(nonce.to_vec())
    }

    #[cfg(feature = "pairing")]
    fn authenticate_transcript(
        &self,
        transcript: &[u8],
        seed: &[u8; 32],
        _initiator_pubkey: &[u8],
    ) -> Option<Vec<u8>> {
        use ed25519_dalek::{Signer, SigningKey};
        let sk = SigningKey::from_bytes(seed);
        let sig = sk.sign(transcript);
        Some(sig.to_bytes().to_vec())
    }

    /// MAC the nonce using the X25519-derived shared secret between us and
    /// `initiator_pubkey`. Returns `None` if `initiator_pubkey` is not a
    /// well-formed 32-byte public key.
    #[cfg(not(feature = "pairing"))]
    fn sign_nonce(
        &self,
        transcript: &[u8],
        _seed: &[u8; 32],
        initiator_pubkey: &[u8],
    ) -> Option<Vec<u8>> {
        let shared = self.derive_shared_mac_key(initiator_pubkey)?;
        let mac = blake3::keyed_hash(&shared, transcript);
        Some(mac.as_bytes().to_vec())
    }

    #[cfg(feature = "pairing")]
    fn verify_transcript(&self, transcript: &[u8], signature: &[u8], pubkey: &[u8]) -> bool {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        if signature.len() != 64 || pubkey.len() != 32 {
            return false;
        }
        let pk_array: [u8; 32] = match pubkey.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let vk = match VerifyingKey::from_bytes(&pk_array) {
            Ok(k) => k,
            Err(_) => return false,
        };
        let sig_bytes: [u8; 64] = match signature.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let sig = Signature::from_bytes(&sig_bytes);
        vk.verify(transcript, &sig).is_ok()
    }

    /// Verify the responder's MAC by independently deriving the same
    /// X25519 shared secret (`DH(our_secret, responder_pubkey)`) and
    /// recomputing the expected MAC -- the shared secret itself is never
    /// transmitted, so this cannot be forged from `pubkey`/`signature` alone.
    #[cfg(not(feature = "pairing"))]
    fn verify_signature(&self, transcript: &[u8], signature: &[u8], pubkey: &[u8]) -> bool {
        if signature.len() != 32 || pubkey.len() != 32 {
            return false;
        }
        let shared = match self.derive_shared_mac_key(pubkey) {
            Some(s) => s,
            None => return false,
        };
        let mac = blake3::keyed_hash(&shared, transcript);
        // Constant-time comparison
        let mut diff = 0u8;
        for (a, b) in signature.iter().zip(mac.as_bytes().iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

/// Canonical pairing transcript. Every field is role-labelled and
/// length-prefixed under a versioned domain, preventing field-splicing and
/// cross-protocol reuse. This does not authenticate freshly exchanged keys
/// against an active man-in-the-middle; the application needs a separate
/// authenticated association step for that property.
fn pairing_transcript(
    nonce: &[u8],
    initiator_pubkey: &[u8],
    responder_pubkey: &[u8],
) -> Option<Vec<u8>> {
    if nonce.len() != 32 || initiator_pubkey.len() != 32 || responder_pubkey.len() != 32 {
        return None;
    }

    let mut transcript = b"SYMTHAEA-SPORE-PAIRING-TRANSCRIPT/v2\0".to_vec();
    append_pairing_field(&mut transcript, b"nonce", nonce);
    append_pairing_field(&mut transcript, b"initiator_pubkey", initiator_pubkey);
    append_pairing_field(&mut transcript, b"responder_pubkey", responder_pubkey);
    Some(transcript)
}

fn append_pairing_field(transcript: &mut Vec<u8>, tag: &[u8], value: &[u8]) {
    // Tags are fixed compile-time strings and every payload is capped at 32
    // bytes by pairing_transcript, so these length encodings cannot truncate.
    transcript.extend_from_slice(&(tag.len() as u16).to_be_bytes());
    transcript.extend_from_slice(tag);
    transcript.extend_from_slice(&(value.len() as u32).to_be_bytes());
    transcript.extend_from_slice(value);
}

impl Default for PairingManager {
    fn default() -> Self {
        Self::new(PairingMode::Off)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_off_mode_ignores_everything() {
        let mut pm = PairingManager::new(PairingMode::Off);
        assert!(!pm.initiate_pairing(42));
        assert!(!pm.receive_challenge(42, &[1, 2, 3], &[0u8; 32]));
        assert!(pm.drain_outbound().is_empty());
    }

    #[test]
    fn test_generate_keypair() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        assert!(pm.get_pubkey().is_none());
        pm.generate_keypair();
        let pk = pm.get_pubkey();
        assert!(pk.is_some());
        assert_eq!(pk.unwrap().len(), 32);
    }

    #[test]
    fn test_challenges_have_fresh_32_byte_nonces() {
        let mut alice = PairingManager::new(PairingMode::Discoverable);
        alice.generate_keypair();
        alice.tick(7);

        assert!(alice.initiate_pairing(1));
        assert!(alice.initiate_pairing(2));
        let messages = alice.drain_outbound();
        assert_eq!(messages.len(), 2);

        let nonces: Vec<Vec<u8>> = messages
            .iter()
            .map(|message| match message {
                PairingOutbound::Challenge { nonce, .. } => nonce.clone(),
                other => panic!("Expected Challenge, got {:?}", other),
            })
            .collect();
        assert_eq!(nonces[0].len(), 32);
        assert_eq!(nonces[1].len(), 32);
        assert_ne!(nonces[0], nonces[1], "same-cycle challenges must not repeat");
    }

    #[cfg(not(feature = "pairing"))]
    #[test]
    fn test_x25519_rejects_non_contributory_all_zero_peer_key() {
        let mut alice = PairingManager::new(PairingMode::Discoverable);
        alice.generate_keypair();
        assert!(alice.derive_shared_mac_key(&[0u8; 32]).is_none());
    }

    #[test]
    fn test_response_rejected_when_challenge_initiator_key_is_rewritten() {
        let mut alice = PairingManager::new(PairingMode::Discoverable);
        let mut bob = PairingManager::new(PairingMode::Discoverable);
        alice.generate_keypair();
        bob.generate_keypair();
        alice.tick(100);
        bob.tick(100);

        assert!(alice.initiate_pairing(2));
        let messages = alice.drain_outbound();
        let (nonce, mut initiator_pubkey) = match &messages[0] {
            PairingOutbound::Challenge { nonce, initiator_pubkey, .. } => {
                (nonce.clone(), initiator_pubkey.clone())
            }
            other => panic!("Expected Challenge, got {:?}", other),
        };

        // Model an intermediary replacing Alice's key but leaving the
        // challenge nonce unchanged.
        initiator_pubkey[0] ^= 0x40;
        assert!(bob.receive_challenge(1, &nonce, &initiator_pubkey));
        let response = bob.drain_outbound();
        let (signature, responder_pubkey) = match &response[0] {
            PairingOutbound::Response { signature, pubkey, .. } => {
                (signature.clone(), pubkey.clone())
            }
            other => panic!("Expected Response, got {:?}", other),
        };

        assert!(
            !alice.verify_response(2, &signature, &responder_pubkey),
            "a response to a rewritten initiator key must not verify"
        );
        assert!(!alice.is_paired(2));
    }

    #[test]
    fn test_full_roundtrip() {
        let mut alice = PairingManager::new(PairingMode::Discoverable);
        let mut bob = PairingManager::new(PairingMode::Discoverable);
        alice.generate_keypair();
        bob.generate_keypair();
        alice.tick(100);
        bob.tick(100);

        // Alice initiates pairing with Bob (peer_id=2)
        assert!(alice.initiate_pairing(2));
        let msgs = alice.drain_outbound();
        assert_eq!(msgs.len(), 1);

        // Extract challenge nonce + Alice's pubkey
        let (nonce, alice_pubkey) = match &msgs[0] {
            PairingOutbound::Challenge {
                nonce,
                initiator_pubkey,
                ..
            } => (nonce.clone(), initiator_pubkey.clone()),
            other => panic!("Expected Challenge, got {:?}", other),
        };

        // Bob receives challenge and responds
        assert!(bob.receive_challenge(1, &nonce, &alice_pubkey));
        let bob_msgs = bob.drain_outbound();
        assert_eq!(bob_msgs.len(), 1);

        // Extract Bob's signature and pubkey
        let (sig, pubkey) = match &bob_msgs[0] {
            PairingOutbound::Response {
                signature, pubkey, ..
            } => (signature.clone(), pubkey.clone()),
            other => panic!("Expected Response, got {:?}", other),
        };

        // Alice verifies Bob's response
        assert!(alice.verify_response(2, &sig, &pubkey));
        assert!(alice.is_paired(2));

        // Should have queued an Ack
        let ack_msgs = alice.drain_outbound();
        assert_eq!(ack_msgs.len(), 1);
        assert!(matches!(ack_msgs[0], PairingOutbound::Ack { peer_id: 2 }));
    }

    #[test]
    fn test_wrong_signature_rejected() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        pm.generate_keypair();
        pm.tick(10);

        assert!(pm.initiate_pairing(99));
        let _ = pm.drain_outbound();

        // Provide garbage signature
        let fake_sig = vec![0xAA; 32];
        let fake_pk = vec![0xBB; 32];
        assert!(!pm.verify_response(99, &fake_sig, &fake_pk));
        assert!(!pm.is_paired(99));
    }

    #[test]
    fn test_timeout_expiry() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        pm.generate_keypair();
        pm.tick(0);

        assert!(pm.initiate_pairing(50));
        let _ = pm.drain_outbound();

        // Advance past timeout
        pm.tick(CHALLENGE_TIMEOUT_CYCLES + 10);
        // Pending should be expired
        assert_eq!(pm.pending_count(), 0);
    }

    #[test]
    fn test_device_cap() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        pm.generate_keypair();
        pm.tick(1);

        // Fill paired devices manually
        for i in 0..MAX_PAIRED_DEVICES {
            pm.paired_devices.push(PairedDevice {
                peer_id: i as u64,
                pubkey: [i as u8; 32],
                trust_level: 0.7,
                last_seen: 1,
            });
        }

        // initiate_pairing should fail at cap
        assert!(!pm.initiate_pairing(999));
    }

    #[test]
    fn test_mode_transition_clears_pending() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        pm.generate_keypair();
        pm.tick(1);

        assert!(pm.initiate_pairing(10));
        assert_eq!(pm.pending_count(), 1);

        pm.set_mode(PairingMode::Off);
        assert_eq!(pm.pending_count(), 0);
    }

    #[test]
    fn test_drain_outbound() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        pm.generate_keypair();
        pm.tick(1);

        pm.initiate_pairing(1);
        pm.initiate_pairing(2);
        let drained = pm.drain_outbound();
        assert_eq!(drained.len(), 2);
        // Second drain should be empty
        assert!(pm.drain_outbound().is_empty());
    }

    #[test]
    fn test_receive_inbound_ack() {
        let mut pm = PairingManager::new(PairingMode::Paired);
        pm.paired_devices.push(PairedDevice {
            peer_id: 7,
            pubkey: [0; 32],
            trust_level: 0.7,
            last_seen: 0,
        });
        pm.tick(100);

        pm.receive_inbound(PairingInbound::Ack { peer_id: 7 });
        assert_eq!(pm.paired_devices[0].last_seen, 100);
    }

    #[test]
    fn test_duplicate_initiation_blocked() {
        let mut pm = PairingManager::new(PairingMode::Discoverable);
        pm.generate_keypair();
        pm.tick(1);

        assert!(pm.initiate_pairing(42));
        assert!(!pm.initiate_pairing(42)); // duplicate
    }

    #[test]
    fn test_generate_keypair_is_not_deterministic() {
        // Two managers "born" at the same cycle must not derive the same
        // key material -- a predictable (e.g. cycle-seeded) generator would
        // let an attacker who estimates uptime reconstruct the key.
        let mut a = PairingManager::new(PairingMode::Discoverable);
        let mut b = PairingManager::new(PairingMode::Discoverable);
        a.tick(0);
        b.tick(0);
        a.generate_keypair();
        b.generate_keypair();
        assert_ne!(
            a.get_pubkey(),
            b.get_pubkey(),
            "two independently generated keypairs at the same cycle must differ"
        );
    }

    #[test]
    fn test_eavesdropper_cannot_forge_mac_from_public_values() {
        // Regression test for the pubkey-doubles-as-MAC-key flaw: an
        // observer who only ever sees the two public keys, the nonce, and
        // the resulting MAC must not be able to recompute that MAC using
        // only those public values (i.e. the "pubkey" must not itself be
        // usable as the keyed-hash key).
        let mut alice = PairingManager::new(PairingMode::Discoverable);
        let mut bob = PairingManager::new(PairingMode::Discoverable);
        alice.generate_keypair();
        bob.generate_keypair();
        alice.tick(1);
        bob.tick(1);

        assert!(alice.initiate_pairing(2));
        let msgs = alice.drain_outbound();
        let (nonce, alice_pubkey) = match &msgs[0] {
            PairingOutbound::Challenge {
                nonce,
                initiator_pubkey,
                ..
            } => (nonce.clone(), initiator_pubkey.clone()),
            other => panic!("Expected Challenge, got {:?}", other),
        };

        assert!(bob.receive_challenge(1, &nonce, &alice_pubkey));
        let bob_msgs = bob.drain_outbound();
        let (mac, bob_pubkey) = match &bob_msgs[0] {
            PairingOutbound::Response {
                signature, pubkey, ..
            } => (signature.clone(), pubkey.clone()),
            other => panic!("Expected Response, got {:?}", other),
        };

        // The v2 transcript is public; the MAC key must still come from
        // private DH output, not either transmitted public key.
        let transcript = pairing_transcript(&nonce, &alice_pubkey, &bob_pubkey).unwrap();
        let forged = blake3::keyed_hash(bob_pubkey[..32].try_into().unwrap(), &transcript);
        assert_ne!(
            forged.as_bytes().to_vec(),
            mac,
            "MAC must not be derivable from the responder public key alone"
        );
        // Likewise using alice's transmitted pubkey as a naive MAC key.
        let forged_alice =
            blake3::keyed_hash(alice_pubkey[..32].try_into().unwrap(), &transcript);
        assert_ne!(
            forged_alice.as_bytes().to_vec(),
            mac,
            "MAC must not be derivable from the initiator public key alone"
        );

        // But Alice, who holds her own private key, CAN verify it (sanity).
        assert!(alice.verify_response(2, &mac, &bob_pubkey));
    }
}
