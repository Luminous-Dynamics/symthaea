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
//! - **With `pairing` feature**: Real Ed25519 signatures via `ed25519-dalek`
//! - **Without `pairing` feature**: BLAKE3 keyed MAC fallback (symmetric)

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
    Challenge { peer_id: u64, nonce: Vec<u8> },
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
    /// Peer requests pairing — contains their nonce for us to sign.
    Challenge { peer_id: u64, nonce: Vec<u8> },
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
    /// Peer's public key (32 bytes Ed25519 or BLAKE3 key hash).
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
    pub fn generate_keypair(&mut self) {
        let mut seed = [0u8; 32];
        use symthaea_core::hdc::ContinuousHV;
        // Use random HV to generate entropy without pulling in rand directly
        let hv = ContinuousHV::random(8, self.cycle);
        for (i, chunk) in hv.values.chunks(1).enumerate().take(32) {
            seed[i] = (chunk[0].to_bits() & 0xFF) as u8;
        }
        // Mix with cycle count for additional entropy
        let cycle_bytes = self.cycle.to_le_bytes();
        for i in 0..8 {
            seed[i] ^= cycle_bytes[i];
        }
        self.keypair_seed = Some(seed);
    }

    /// Get the public key as bytes (32 bytes).
    ///
    /// Returns `None` if no keypair has been generated.
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
            // BLAKE3 hash of seed as "public key" (symmetric fallback)
            let hash = blake3::hash(seed);
            Some(*hash.as_bytes())
        }
    }

    /// Initiate pairing with a remote peer.
    ///
    /// Creates a challenge nonce and queues it for sending.
    /// Returns `false` if mode is Off, device cap reached, or already pending.
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

        // Generate nonce
        let nonce = self.generate_nonce();

        self.pending_challenges.push((
            peer_id,
            PendingChallenge {
                nonce: nonce.clone(),
                issued_cycle: self.cycle,
            },
        ));

        self.enqueue(PairingOutbound::Challenge { peer_id, nonce });
        true
    }

    /// Handle an inbound challenge from a peer.
    ///
    /// Signs the nonce and queues the response.
    /// Returns `false` if mode is Off or no keypair.
    pub fn receive_challenge(&mut self, peer_id: u64, nonce: &[u8]) -> bool {
        if self.mode == PairingMode::Off {
            return false;
        }
        let seed = match self.keypair_seed {
            Some(s) => s,
            None => return false,
        };

        let signature = self.sign_nonce(nonce, &seed);
        let pubkey = match self.get_pubkey() {
            Some(pk) => pk.to_vec(),
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

        // Verify
        let valid = self.verify_signature(&challenge.nonce, signature, pubkey);
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
            PairingInbound::Challenge { peer_id, nonce } => {
                self.receive_challenge(peer_id, &nonce);
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

    fn generate_nonce(&self) -> Vec<u8> {
        // Deterministic nonce from cycle + constant for reproducibility
        let mut data = Vec::with_capacity(40);
        data.extend_from_slice(&self.cycle.to_le_bytes());
        data.extend_from_slice(b"spore-pairing-nonce");
        if let Some(ref seed) = self.keypair_seed {
            data.extend_from_slice(seed);
        }
        let hash = blake3::hash(&data);
        hash.as_bytes().to_vec()
    }

    #[cfg(feature = "pairing")]
    fn sign_nonce(&self, nonce: &[u8], seed: &[u8; 32]) -> Vec<u8> {
        use ed25519_dalek::{Signer, SigningKey};
        let sk = SigningKey::from_bytes(seed);
        let sig = sk.sign(nonce);
        sig.to_bytes().to_vec()
    }

    #[cfg(not(feature = "pairing"))]
    fn sign_nonce(&self, nonce: &[u8], seed: &[u8; 32]) -> Vec<u8> {
        // BLAKE3 keyed MAC
        let key = blake3::hash(seed);
        let mac = blake3::keyed_hash(key.as_bytes(), nonce);
        mac.as_bytes().to_vec()
    }

    #[cfg(feature = "pairing")]
    fn verify_signature(&self, nonce: &[u8], signature: &[u8], pubkey: &[u8]) -> bool {
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
        vk.verify(nonce, &sig).is_ok()
    }

    #[cfg(not(feature = "pairing"))]
    fn verify_signature(&self, nonce: &[u8], signature: &[u8], pubkey: &[u8]) -> bool {
        if signature.len() != 32 || pubkey.len() != 32 {
            return false;
        }
        // Reconstruct MAC from pubkey (which is hash of seed in BLAKE3 mode)
        // In BLAKE3 fallback, the "pubkey" IS the key material — symmetric.
        let pk_array: &[u8; 32] = pubkey.try_into().unwrap();
        let mac = blake3::keyed_hash(pk_array, nonce);
        // Constant-time comparison
        let mut diff = 0u8;
        for (a, b) in signature.iter().zip(mac.as_bytes().iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
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
        assert!(!pm.receive_challenge(42, &[1, 2, 3]));
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

        // Extract challenge nonce
        let nonce = match &msgs[0] {
            PairingOutbound::Challenge { nonce, .. } => nonce.clone(),
            other => panic!("Expected Challenge, got {:?}", other),
        };

        // Bob receives challenge and responds
        assert!(bob.receive_challenge(1, &nonce));
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
}
