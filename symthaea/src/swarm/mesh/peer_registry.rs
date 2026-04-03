// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh peer tracking and key management.

use super::mesh_telemetry::{MeshStats, MeshUrgency, PayloadType};
use super::wisdom_packet::WisdomPacket;

// ============================================================================
// MESH PEER REGISTRY
// ============================================================================

/// Maximum packets allowed per peer within the rate limit window.
const MESH_RATE_LIMIT_MAX: u64 = 100;
/// Duration of the per-peer rate limiting window.
const MESH_RATE_LIMIT_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// Tracked state for a single mesh peer (distinct from [`MeshPeer`](super::MeshPeer) which
/// tracks LoRa fragment reassembly statistics).
#[derive(Debug, Clone)]
pub struct MeshPeerEntry {
    /// First 8 bytes of the peer's node identity.
    pub source_id: [u8; 8],
    /// Last time we received a packet from this peer.
    pub last_seen: std::time::Instant,
    /// Total packets received from this peer.
    pub packets_received: u64,
    /// Most recent Phi value reported by this peer.
    pub last_phi: f32,
    /// Most recent urgency level from this peer.
    pub last_urgency: MeshUrgency,
    /// Most recent payload type from this peer.
    pub last_payload_type: PayloadType,
    /// Start of the current rate-limiting window.
    pub window_start: std::time::Instant,
    /// Packet count within the current rate-limiting window.
    pub window_count: u64,
}

/// Registry of active mesh peers, updated each tick from incoming packets.
///
/// Provides swarm awareness: "which peers are alive?", "what's the swarm's
/// average phi?", etc. Stale peers are expired after `stale_timeout`.
pub struct MeshPeerRegistry {
    pub(super) peers: std::collections::HashMap<[u8; 8], MeshPeerEntry>,
    stale_timeout: std::time::Duration,
}

impl MeshPeerRegistry {
    /// Create a new registry with the default 60-second stale timeout.
    pub fn new() -> Self {
        Self {
            peers: std::collections::HashMap::new(),
            stale_timeout: std::time::Duration::from_secs(60),
        }
    }

    /// Create a new registry with a custom stale timeout (useful for tests).
    pub fn with_timeout(timeout: std::time::Duration) -> Self {
        Self {
            peers: std::collections::HashMap::new(),
            stale_timeout: timeout,
        }
    }

    /// Update the registry with a received packet.
    pub fn update(&mut self, packet: &WisdomPacket) {
        let entry = self
            .peers
            .entry(packet.source_id)
            .or_insert_with(|| MeshPeerEntry {
                source_id: packet.source_id,
                last_seen: std::time::Instant::now(),
                packets_received: 0,
                last_phi: packet.phi,
                last_urgency: packet.urgency,
                last_payload_type: packet.payload_type,
                window_start: std::time::Instant::now(),
                window_count: 0,
            });
        entry.last_seen = std::time::Instant::now();
        entry.packets_received += 1;
        entry.last_phi = packet.phi;
        entry.last_urgency = packet.urgency;
        entry.last_payload_type = packet.payload_type;
    }

    /// Remove peers not seen within the stale timeout.
    ///
    /// Returns the `source_id`s of expired peers (empty if none expired).
    pub fn expire_stale(&mut self) -> Vec<[u8; 8]> {
        let cutoff = std::time::Instant::now() - self.stale_timeout;
        let expired: Vec<[u8; 8]> = self
            .peers
            .iter()
            .filter(|(_, entry)| entry.last_seen <= cutoff)
            .map(|(id, _)| *id)
            .collect();
        for id in &expired {
            self.peers.remove(id);
        }
        expired
    }

    /// Get all active (non-expired) peers.
    pub fn active_peers(&self) -> Vec<&MeshPeerEntry> {
        self.peers.values().collect()
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Check if a peer has exceeded the rate limit.
    ///
    /// Returns `true` if the peer's packet count within the current window
    /// exceeds [`MESH_RATE_LIMIT_MAX`]. Resets the window if the previous
    /// one has elapsed.
    pub fn is_rate_limited(&mut self, source_id: &[u8; 8]) -> bool {
        let entry = match self.peers.get_mut(source_id) {
            Some(e) => e,
            None => return false, // Unknown peer — not rate limited
        };
        let now = std::time::Instant::now();
        if now.duration_since(entry.window_start) >= MESH_RATE_LIMIT_WINDOW {
            entry.window_start = now;
            entry.window_count = 0;
        }
        entry.window_count += 1;
        entry.window_count > MESH_RATE_LIMIT_MAX
    }

    /// Average Phi across all tracked peers. Returns 0.0 if no peers.
    pub fn average_phi(&self) -> f32 {
        if self.peers.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.peers.values().map(|p| p.last_phi).sum();
        sum / self.peers.len() as f32
    }

    /// Check if a peer with the given source_id is currently tracked.
    pub fn has_peer(&self, source_id: &[u8; 8]) -> bool {
        self.peers.contains_key(source_id)
    }

    /// Detect if the mesh appears partitioned.
    ///
    /// Returns `true` if we previously had peers but now have zero
    /// (all expired), suggesting a network partition rather than normal
    /// peer departure. Returns `false` if we never had peers or still have some.
    pub fn is_partitioned(&self, stats: &MeshStats) -> bool {
        self.peer_count() == 0 && stats.peers_expired > 0 && stats.wisdom_received > 0
    }

    /// Count peers that haven't been seen within the given duration.
    pub fn stale_peer_count(&self, timeout: std::time::Duration) -> usize {
        let now = std::time::Instant::now();
        self.peers
            .values()
            .filter(|p| now.duration_since(p.last_seen) > timeout)
            .count()
    }
}

impl Default for MeshPeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// KEY ROTATION
// ============================================================================

/// Manages encryption key rotation with a grace period.
///
/// During rotation, both the old and new keys are tried for decryption.
/// After the grace period expires, only the new key is accepted.
///
/// ```text
/// rotate_key(new_key)
///   |-- grace period (accepts old OR new) --> grace expires
///   |                                          |-- old_key = None
///   |                                          +-- only new_key accepted
/// ```
#[cfg(feature = "mesh-encryption")]
#[derive(Clone, zeroize::ZeroizeOnDrop)]
pub struct RotatingKeyPair {
    /// Current (primary) encryption key — used for all outbound packets.
    current: [u8; 32],
    /// Previous key — accepted for inbound during grace period, then discarded.
    /// NOT skipped by zeroize — key material must be wiped on drop.
    previous: Option<[u8; 32]>,
    /// Tick at which the previous key expires (absolute tick count).
    #[zeroize(skip)]
    grace_expires_at: u64,
    /// Key version — incremented on every rotation, embedded in ciphertext header.
    /// Wraps at u8::MAX. Prevents nonce reuse across key rotations.
    #[zeroize(skip)]
    key_version: u8,
    /// Version of the previous key (for versioned decrypt hints).
    #[zeroize(skip)]
    previous_version: u8,
    /// Random epoch byte generated once at construction. Prevents nonce reuse
    /// across node restarts under the same key material (P0 security fix).
    #[zeroize(skip)]
    epoch: u8,
    /// Optional sensor context key overlay. When set, the effective encryption
    /// key is `current XOR context_overlay` — binding mesh encryption to the
    /// physical environment (location, light, motion, altitude).
    /// Both peers must independently derive the same context key.
    context_overlay: Option<[u8; 32]>,
}

#[cfg(feature = "mesh-encryption")]
impl RotatingKeyPair {
    /// Create a new key pair with a single key (no rotation in progress).
    ///
    /// Generates a random epoch byte to prevent nonce reuse across restarts.
    pub fn new(key: [u8; 32]) -> Self {
        Self {
            current: key,
            previous: None,
            grace_expires_at: 0,
            key_version: 0,
            previous_version: 0,
            epoch: rand::Rng::gen(&mut rand::thread_rng()),
            context_overlay: None,
        }
    }

    /// Rotate to a new key. The old key remains valid for `grace_ticks` cycles.
    ///
    /// Outbound packets immediately use the new key. Inbound packets accept
    /// either key until the grace period expires. The key version is incremented
    /// to prevent nonce reuse across rotations.
    pub fn rotate(&mut self, new_key: [u8; 32], current_tick: u64, grace_ticks: u64) {
        self.previous = Some(self.current);
        self.previous_version = self.key_version;
        self.key_version = self.key_version.wrapping_add(1);
        self.current = new_key;
        self.grace_expires_at = current_tick.saturating_add(grace_ticks);
    }

    /// The current key version (embedded in ciphertext header).
    pub fn key_version(&self) -> u8 {
        self.key_version
    }

    /// Expire the old key if the grace period has elapsed.
    ///
    /// Call this once per tick to garbage-collect the old key.
    pub fn tick(&mut self, current_tick: u64) {
        if self.previous.is_some() && current_tick >= self.grace_expires_at {
            self.previous = None;
        }
    }

    /// The current key (used for encryption).
    pub fn current_key(&self) -> &[u8; 32] {
        &self.current
    }

    /// Whether a rotation is in progress (grace period active).
    pub fn is_rotating(&self) -> bool {
        self.previous.is_some()
    }

    /// Encrypt with the current key using the stored random epoch.
    ///
    /// If a sensor context overlay is set, the effective key is
    /// `current XOR context_overlay` — physically binding the encryption.
    pub fn encrypt(&self, envelope: &[u8], source_id: &[u8; 8], sequence: u32) -> Vec<u8> {
        let key = self.effective_key();
        super::packet_crypto::encrypt_packet(envelope, &key, source_id, self.epoch, sequence)
    }

    /// Get the epoch byte (useful for logging/telemetry).
    pub fn epoch(&self) -> u8 {
        self.epoch
    }

    /// Set the sensor context key overlay.
    ///
    /// When set, all encryption/decryption uses `base_key XOR context_overlay`
    /// as the effective key. Both peers must independently derive the same
    /// context key from their sensor readings (same location, similar conditions).
    ///
    /// Call with `None` to disable context binding.
    pub fn set_context_overlay(&mut self, overlay: Option<[u8; 32]>) {
        self.context_overlay = overlay;
    }

    /// Whether a sensor context overlay is active.
    pub fn has_context_overlay(&self) -> bool {
        self.context_overlay.is_some()
    }

    /// Compute the effective encryption key (base XOR context overlay).
    pub(super) fn effective_key(&self) -> [u8; 32] {
        match &self.context_overlay {
            Some(overlay) => {
                let mut key = self.current;
                for (k, o) in key.iter_mut().zip(overlay.iter()) {
                    *k ^= o;
                }
                key
            }
            None => self.current,
        }
    }

    /// Encrypt with the current key, using typed nonce for cross-type safety.
    ///
    /// Prepends a 1-byte key version header for versioned decrypt hints.
    pub fn encrypt_typed(
        &self,
        envelope: &[u8],
        source_id: &[u8; 8],
        payload_type: u8,
        epoch: u8,
        sequence: u32,
    ) -> Vec<u8> {
        super::packet_crypto::encrypt_packet_versioned(
            envelope,
            &self.current,
            self.key_version,
            source_id,
            payload_type,
            epoch,
            sequence,
        )
    }

    /// Decrypt trying versioned format first, then unversioned for backward compat.
    ///
    /// Versioned format: `[key_version (1) | nonce (12) | ciphertext+tag]`
    /// - If version matches current -> try current key only
    /// - If version matches previous -> try previous key only
    /// - Otherwise -> try both (backward compat with pre-versioned packets)
    pub fn decrypt(&self, data: &[u8]) -> Option<Vec<u8>> {
        use super::packet_crypto::{decrypt_packet, AEAD_NONCE_SIZE, AEAD_TAG_SIZE};
        // Need at least 1 (version) + 12 (nonce) + 16 (tag) = 29 bytes for versioned
        if data.len() >= 1 + AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
            let version_hint = data[0];
            let inner = &data[1..];

            if version_hint == self.key_version {
                // Matches current — try only current key
                if let Some(pt) = decrypt_packet(inner, &self.current) {
                    return Some(pt);
                }
            } else if version_hint == self.previous_version {
                // Matches previous — try only previous key
                if let Some(ref prev) = self.previous {
                    if let Some(pt) = decrypt_packet(inner, prev) {
                        return Some(pt);
                    }
                }
            }
        }

        // Fall back to unversioned (backward compat with pre-versioned packets)
        if let Some(plaintext) = super::packet_crypto::decrypt_packet(data, &self.current) {
            return Some(plaintext);
        }
        if let Some(ref prev) = self.previous {
            return super::packet_crypto::decrypt_packet(data, prev);
        }
        None
    }
}

// ============================================================================
// X25519 PER-PEER KEY AGREEMENT
// ============================================================================

/// Per-peer key store: X25519 Diffie-Hellman -> BLAKE3 KDF -> ChaCha20 key.
///
/// Each peer pair derives a unique symmetric key from their DH shared secret.
/// This provides forward secrecy (compromising one peer's key doesn't reveal
/// other pairs' traffic) and eliminates the single shared secret.
///
/// ```text
/// Node A (secret_a)                    Node B (secret_b)
///   |                                    |
///   |-- public_a = X25519(secret_a) ---->|
///   |<---- public_b = X25519(secret_b) --|
///   |                                    |
///   |-- shared = DH(secret_a, public_b)  |-- shared = DH(secret_b, public_a)
///   |-- key = BLAKE3(shared | ctx)       |-- key = BLAKE3(shared | ctx)
///   +-- (same key on both sides)         +-- (same key on both sides)
/// ```
#[cfg(feature = "mesh-key-exchange")]
pub struct PeerKeyStore {
    /// Our long-term X25519 secret key.
    secret: x25519_dalek::StaticSecret,
    /// Our public key (derived from secret).
    public: x25519_dalek::PublicKey,
    /// Derived symmetric keys per peer, keyed by source_id.
    peer_keys: std::collections::HashMap<[u8; 8], [u8; 32]>,
    /// Derived authentication keys per peer (separate from encryption keys).
    peer_auth_keys: std::collections::HashMap<[u8; 8], [u8; 32]>,
}

#[cfg(feature = "mesh-key-exchange")]
impl PeerKeyStore {
    /// Create a new key store with a random X25519 keypair.
    pub fn new(secret_bytes: [u8; 32]) -> Self {
        let secret = x25519_dalek::StaticSecret::from(secret_bytes);
        let public = x25519_dalek::PublicKey::from(&secret);
        Self {
            secret,
            public,
            peer_keys: std::collections::HashMap::new(),
            peer_auth_keys: std::collections::HashMap::new(),
        }
    }

    /// Our X25519 public key (32 bytes, send to peers).
    pub fn public_key(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    /// Perform DH key agreement with a peer's public key.
    ///
    /// Derives two symmetric keys via HKDF-SHA256:
    /// - Encryption key (info: `symthaea-mesh-chacha20-v1`)
    /// - Authentication key (info: `symthaea-mesh-auth-v1`)
    ///
    /// Returns the encryption key.
    pub fn agree(&mut self, peer_source_id: [u8; 8], peer_public: &[u8; 32]) -> [u8; 32] {
        let peer_pk = x25519_dalek::PublicKey::from(*peer_public);
        let shared_secret = self.secret.diffie_hellman(&peer_pk);

        // KDF: HKDF-SHA256 with domain-separated info strings
        let hk = hkdf::Hkdf::<sha2::Sha256>::new(None, shared_secret.as_bytes());

        let mut chacha_key = [0u8; 32];
        hk.expand(b"symthaea-mesh-chacha20-v1", &mut chacha_key)
            .expect("32-byte output is valid for HKDF-SHA256");

        let mut auth_key = [0u8; 32];
        hk.expand(b"symthaea-mesh-auth-v1", &mut auth_key)
            .expect("32-byte output is valid for HKDF-SHA256");

        self.peer_keys.insert(peer_source_id, chacha_key);
        self.peer_auth_keys.insert(peer_source_id, auth_key);
        chacha_key
    }

    /// Get the derived symmetric key for a peer (if agreement has been done).
    pub fn peer_key(&self, source_id: &[u8; 8]) -> Option<&[u8; 32]> {
        self.peer_keys.get(source_id)
    }

    /// Get the derived authentication key for a peer.
    pub fn peer_auth_key(&self, source_id: &[u8; 8]) -> Option<&[u8; 32]> {
        self.peer_auth_keys.get(source_id)
    }

    /// Number of peers with established keys.
    pub fn peer_count(&self) -> usize {
        self.peer_keys.len()
    }

    /// Remove a peer's keys (both encryption and auth).
    pub fn remove_peer(&mut self, source_id: &[u8; 8]) {
        self.peer_keys.remove(source_id);
        self.peer_auth_keys.remove(source_id);
    }
}

#[cfg(feature = "mesh-key-exchange")]
impl Drop for PeerKeyStore {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        for key in self.peer_keys.values_mut() {
            key.zeroize();
        }
        for key in self.peer_auth_keys.values_mut() {
            key.zeroize();
        }
    }
}
